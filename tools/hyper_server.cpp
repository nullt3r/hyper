// hyper OpenAI-compatible server: /v1/chat/completions (streaming and not), /v1/models, /health.
// Chat templates, reasoning and tool-call parsing come from mainline llama.cpp's libcommon (same code as
// llama-server); the vocabulary is loaded with libllama (vocab only). Generation samples (greedy at temperature 0) with MTP speculative
// decoding (exact: drafts are accepted when they equal the token sampled at their row), prompt cache with
// recurrent-state snapshots; requests are served one at a time.
//
// usage: hyper-server <model.gguf> [--host 0.0.0.0] [--port 8080] [--ctx 262144] [--draft 3] [--alias name]
//                     [--temp 0.6] [--top-p 0.95] [--top-k 20] [--min-p 0] [--snapshots 48]   (request fields override)
#include "engine.h"

#include "chat.h"
#include "llama.h"

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using ojson = nlohmann::ordered_json;
using namespace hyper;

namespace {

struct Ctx {
    SamplingParams defaults;
    Engine * eng = nullptr;
    llama_model * vm = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr tmpls;
    std::string alias;
    std::mutex mu;
    std::atomic<long> next_id{1};
};

std::vector<int> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> t(n);
    n = llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), n, true, true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    t.resize(n);
    return std::vector<int>(t.begin(), t.end());
}

std::string piece(const llama_vocab * vocab, int tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof buf, 0, true);
    if (n < 0) {
        std::string s(-n, '\0');
        llama_token_to_piece(vocab, tok, s.data(), (int) s.size(), 0, true);
        return s;
    }
    return std::string(buf, n);
}

// length of the longest prefix that does not end inside a UTF-8 sequence
size_t utf8_complete(const std::string & s) {
    size_t i = s.size();
    for (int back = 1; back <= 4 && back <= (int) s.size(); ++back) {
        const unsigned char c = s[s.size() - back];
        if ((c & 0xC0) == 0x80) continue;                   // continuation byte
        const int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (len > back) i = s.size() - back;
        break;
    }
    return i;
}

ojson to_ojson(const common_json & j) { return ojson::parse(j.dump()); }
common_json to_cjson(const ojson & j) { return common_json::parse(j.dump()); }

ojson diff_to_delta(const common_chat_msg_diff & d) {
    ojson delta = ojson::object();
    if (!d.reasoning_content_delta.empty()) delta["reasoning_content"] = d.reasoning_content_delta;
    if (!d.content_delta.empty()) delta["content"] = d.content_delta;
    if (d.tool_call_index != std::string::npos) {
        ojson tc = {{"index", d.tool_call_index}};
        if (!d.tool_call_delta.id.empty()) { tc["id"] = d.tool_call_delta.id; tc["type"] = "function"; }
        ojson fn = ojson::object();
        if (!d.tool_call_delta.name.empty()) fn["name"] = d.tool_call_delta.name;
        fn["arguments"] = d.tool_call_delta.arguments;
        tc["function"] = fn;
        delta["tool_calls"] = ojson::array({tc});
    }
    return delta;
}

std::string random_id(const char * prefix, long n) {
    char buf[64];
    snprintf(buf, sizeof buf, "%s%08lx%06lx", prefix, (long) std::chrono::system_clock::now().time_since_epoch().count() & 0xffffffffL, n);
    return buf;
}

struct Request {
    std::vector<int> prompt;
    common_chat_parser_params pp;
    std::vector<std::string> stops;
    int max_tokens = 0;
    bool stream = false, include_usage = true;
    SamplingParams sp;
};

Request prepare(Ctx & c, const ojson & body) {
    Request r;
    common_chat_templates_inputs in;
    in.messages = common_chat_msgs_parse_oaicompat(to_cjson(body.at("messages")));
    if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty())
        in.tools = common_chat_tools_parse_oaicompat(to_cjson(body["tools"]));
    if (body.contains("tool_choice") && body["tool_choice"].is_string())
        in.tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
    in.parallel_tool_calls = body.value("parallel_tool_calls", true);
    in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    in.enable_thinking = true;
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        for (auto & [k, v] : body["chat_template_kwargs"].items()) in.chat_template_kwargs[k] = v.dump();
        if (body["chat_template_kwargs"].contains("enable_thinking"))
            in.enable_thinking = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
    }
    if (body.value("reasoning_effort", std::string()) == "none") in.enable_thinking = false;
    const common_chat_params cp = common_chat_templates_apply(c.tmpls.get(), in);
    r.prompt = tokenize(c.vocab, cp.prompt);
    r.pp = common_chat_parser_params(cp);
    r.pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    r.pp.parse_tool_calls = !in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    if (!cp.parser.empty()) r.pp.parser.load(cp.parser);
    r.stops = cp.additional_stops;
    if (body.contains("stop")) {
        if (body["stop"].is_string()) r.stops.push_back(body["stop"]);
        else if (body["stop"].is_array()) for (auto & s : body["stop"]) r.stops.push_back(s);
    }
    const int room = c.eng->max_pos() - (int) r.prompt.size() - 8;
    if (room <= 0) throw std::invalid_argument("prompt (" + std::to_string(r.prompt.size()) + " tokens) exceeds the context");
    r.max_tokens = room;
    for (const char * k : {"max_tokens", "max_completion_tokens"})
        if (body.contains(k) && body[k].is_number_integer() && body[k].get<int>() > 0) r.max_tokens = std::min(room, body[k].get<int>());
    r.sp = c.defaults;
    auto num = [&](const char * k, float & dst) { if (body.contains(k) && body[k].is_number()) dst = body[k].get<float>(); };
    num("temperature", r.sp.temp);
    num("top_p", r.sp.top_p);
    num("min_p", r.sp.min_p);
    if (body.contains("top_k") && body["top_k"].is_number_integer()) r.sp.top_k = body["top_k"].get<int>();
    if (body.contains("seed") && body["seed"].is_number_integer()) r.sp.seed = body["seed"].get<uint64_t>();
    r.stream = body.value("stream", false);
    if (body.contains("stream_options") && body["stream_options"].is_object())
        r.include_usage = body["stream_options"].value("include_usage", true);
    return r;
}

// runs generation, calling on_msg(new_msg) whenever the parsed message may have changed; returns finish reason
struct GenOut { std::string text, finish; int n_gen = 0, reused = 0; double t_prompt = 0, t_gen = 0; };

GenOut run(Ctx & c, const Request & r, const std::function<bool(const std::string &)> & on_text) {
    GenOut o;
    o.finish = "length";
    GenStats st;
    auto t0 = std::chrono::steady_clock::now();
    bool first = true;
    std::chrono::steady_clock::time_point t1 = t0;
    c.eng->generate(r.prompt, r.max_tokens, true, &st, [&](int tok) {
        if (first) { t1 = std::chrono::steady_clock::now(); first = false; }
        if (llama_vocab_is_eog(c.vocab, tok)) { o.finish = "stop"; return false; }
        ++o.n_gen;
        o.text += piece(c.vocab, tok);
        for (const auto & s : r.stops) {
            const size_t at = o.text.find(s, o.text.size() > s.size() + 64 ? o.text.size() - s.size() - 64 : 0);
            if (at != std::string::npos) { o.text.resize(at); o.finish = "stop"; return false; }
        }
        if (on_text && !on_text(o.text)) { o.finish = "cancelled"; return false; }
        return true;
    }, r.sp);
    auto t2 = std::chrono::steady_clock::now();
    o.reused = st.prompt_reused;
    o.t_prompt = std::chrono::duration<double>(t1 - t0).count();
    o.t_gen = std::chrono::duration<double>(t2 - t1).count();
    return o;
}

ojson timings(const Request & r, const GenOut & o) {
    const int n_new = (int) r.prompt.size() - o.reused;
    return {{"cache_n", o.reused}, {"prompt_n", n_new}, {"prompt_ms", 1e3 * o.t_prompt},
            {"prompt_per_second", n_new / std::max(1e-9, o.t_prompt)}, {"predicted_n", o.n_gen},
            {"predicted_ms", 1e3 * o.t_gen}, {"predicted_per_second", o.n_gen / std::max(1e-9, o.t_gen)}};
}

ojson usage(const Request & r, const GenOut & o) {
    return {{"prompt_tokens", (int) r.prompt.size()}, {"completion_tokens", o.n_gen},
            {"prompt_tokens_details", {{"cached_tokens", o.reused}}},
            {"total_tokens", (int) r.prompt.size() + o.n_gen}};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--host H] [--port P] [--ctx N] [--draft K] [--alias name]\n", argv[0]); return 1; }
    std::string host = "0.0.0.0", alias;
    int port = 8080, ctx = 262144, draft = 3, snaps = 48;
    SamplingParams defaults;   // Qwen's recommendation for thinking mode
    defaults.temp = 0.6f; defaults.top_p = 0.95f; defaults.top_k = 20; defaults.min_p = 0.0f;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--host") host = v; else if (k == "--port") port = std::stoi(v); else if (k == "--ctx") ctx = std::stoi(v);
        else if (k == "--draft") draft = std::stoi(v); else if (k == "--alias") alias = v;
        else if (k == "--temp") defaults.temp = std::stof(v); else if (k == "--top-p") defaults.top_p = std::stof(v);
        else if (k == "--top-k") defaults.top_k = std::stoi(v); else if (k == "--min-p") defaults.min_p = std::stof(v);
        else if (k == "--snapshots") snaps = std::stoi(v);
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    const std::string path = argv[1];
    if (alias.empty()) { alias = path.substr(path.find_last_of('/') + 1); if (alias.size() > 5) alias.resize(alias.size() - 5); }

    Ctx c;
    c.alias = alias;
    c.defaults = defaults;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    c.vm = llama_model_load_from_file(path.c_str(), mp);
    if (!c.vm) { fprintf(stderr, "cannot load vocabulary\n"); return 1; }
    c.vocab = llama_model_get_vocab(c.vm);
    c.tmpls = common_chat_templates_init(c.vm, "");

    EngineOptions opt;
    opt.max_pos = ctx;
    opt.n_draft = draft;
    opt.prompt_cache = true;
    opt.max_snapshots = snaps;
    Engine eng(path, opt);
    c.eng = &eng;
    {
        const std::vector<int> im = tokenize(c.vocab, "<|im_start|>");
        if (im.size() == 1) eng.set_snapshot_token(im[0]);
        else fprintf(stderr, "hyper-server: no single <|im_start|> token, prompt-cache snapshots only every 4096 tokens\n");
    }
    {   // warm up: builds the CUDA graphs
        GenStats st;
        eng.generate(tokenize(c.vocab, "Hello"), 4, true, &st);
    }

    httplib::Server srv;
    srv.set_read_timeout(3600);
    srv.set_write_timeout(3600);
    auto cors = [](httplib::Response & res) { res.set_header("Access-Control-Allow-Origin", "*"); };
    srv.Options(R"(.*)", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    });
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) { cors(res); res.set_content(R"({"status":"ok"})", "application/json"); });
    auto models = [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        ojson m = {{"id", c.alias}, {"object", "model"}, {"created", 0}, {"owned_by", "hyper"}, {"meta", {{"n_ctx_train", ctx}}}};
        res.set_content(ojson({{"object", "list"}, {"data", ojson::array({m})}}).dump(), "application/json");
    };
    srv.Get("/v1/models", models);
    srv.Get("/models", models);

    auto chat = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        auto fail = [&](int code, const std::string & msg) {
            res.status = code;
            res.set_content(ojson({{"error", {{"message", msg}, {"type", code == 400 ? "invalid_request_error" : "server_error"}}}}).dump(), "application/json");
        };
        std::shared_ptr<Request> r;
        try {
            const ojson body = ojson::parse(req.body);
            r = std::make_shared<Request>(prepare(c, body));
        } catch (const std::exception & e) { fail(400, e.what()); return; }
        const std::string id = random_id("chatcmpl-", c.next_id++);
        const long created = (long) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        fprintf(stderr, "hyper-server: request %s: %zu prompt tokens, stream=%d\n", id.c_str(), r->prompt.size(), (int) r->stream);

        if (!r->stream) {
            std::lock_guard<std::mutex> lk(c.mu);
            GenOut o;
            try { o = run(c, *r, {}); } catch (const std::exception & e) { fail(500, e.what()); return; }
            common_chat_msg msg = common_chat_parse(o.text, false, r->pp);
            long tc = 0;
            for (auto & t : msg.tool_calls) if (t.id.empty()) t.id = random_id("call_", tc++);
            ojson m = to_ojson(msg.to_json_oaicompat());
            m["role"] = "assistant";
            const std::string finish = !msg.tool_calls.empty() ? "tool_calls" : o.finish;
            ojson out = {{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", c.alias},
                         {"choices", ojson::array({ojson({{"index", 0}, {"message", m}, {"finish_reason", finish}})})},
                         {"usage", usage(*r, o)}, {"timings", timings(*r, o)}};
            fprintf(stderr, "hyper-server: %s done: %d tokens, prompt %zu (cached %d) %.0f t/s, gen %.1f t/s\n", id.c_str(), o.n_gen,
                    r->prompt.size(), o.reused, (r->prompt.size() - o.reused) / std::max(1e-9, o.t_prompt), o.n_gen / std::max(1e-9, o.t_gen));
            res.set_content(out.dump(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&c, r, id, created](size_t, httplib::DataSink & sink) {
            auto send = [&](const ojson & j) {
                const std::string s = "data: " + j.dump(-1, ' ', false, ojson::error_handler_t::replace) + "\n\n";
                return sink.write(s.data(), s.size());
            };
            auto chunk = [&](const ojson & delta, const ojson & finish) {
                return ojson({{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", c.alias},
                              {"choices", ojson::array({ojson({{"index", 0}, {"delta", delta}, {"finish_reason", finish}})})}});
            };
            std::lock_guard<std::mutex> lk(c.mu);
            bool ok = send(chunk({{"role", "assistant"}, {"content", nullptr}}, nullptr));
            common_chat_msg prev = common_chat_parse("", true, r->pp);
            std::vector<std::string> ids;
            long tc = 0;
            auto gen_id = [&] { return random_id("call_", tc++); };
            auto push = [&](const std::string & text, bool partial) {
                common_chat_msg msg;
                try { msg = common_chat_parse(text, partial, r->pp); } catch (const std::exception &) { return true; }
                msg.set_tool_call_ids(ids, gen_id);
                for (const auto & d : common_chat_msg_diff::compute_diffs(prev, msg))
                    if (!send(chunk(diff_to_delta(d), nullptr))) return false;
                prev = msg;
                return true;
            };
            GenOut o;
            try {
                o = run(c, *r, [&](const std::string & text) {
                    if (!ok) return false;
                    ok = push(text.substr(0, utf8_complete(text)), true);
                    return ok;
                });
            } catch (const std::exception & e) {
                send(ojson({{"error", {{"message", e.what()}}}}));
                sink.done();
                return false;
            }
            if (ok) {
                push(o.text, false);
                const std::string finish = !prev.tool_calls.empty() ? "tool_calls" : o.finish == "cancelled" ? "stop" : o.finish;
                ojson last = chunk(ojson::object(), finish);
                if (r->include_usage) last["usage"] = usage(*r, o);
                last["timings"] = timings(*r, o);
                send(last);
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
            }
            fprintf(stderr, "hyper-server: %s done (%s): %d tokens, prompt %zu (cached %d) %.0f t/s, gen %.1f t/s\n", id.c_str(),
                    o.finish.c_str(), o.n_gen, r->prompt.size(), o.reused, (r->prompt.size() - o.reused) / std::max(1e-9, o.t_prompt),
                    o.n_gen / std::max(1e-9, o.t_gen));
            sink.done();
            return true;
        });
    };
    srv.Post("/v1/chat/completions", chat);
    srv.Post("/chat/completions", chat);

    fprintf(stderr, "hyper-server: listening on http://%s:%d (model \"%s\", context %d, %d drafts)\n", host.c_str(), port,
            c.alias.c_str(), ctx, draft);
    if (!srv.listen(host, port)) { fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
