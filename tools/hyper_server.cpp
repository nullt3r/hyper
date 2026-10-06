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

using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// live state of the running request, for the log and the /stats page
struct Live {
    std::mutex mu;
    std::string phase = "idle", id;
    int prompt_total = 0, prompt_done = 0, cached = 0, gen = 0, ctx_max = 0;
    double prompt_tps = 0, gen_tps_now = 0, gen_tps_avg = 0;
    clk::time_point t_start, t_gen, t_win;
    int win_n = 0;
    long requests = 0;
    ojson last = nullptr;
    ojson json() {
        std::lock_guard<std::mutex> lk(mu);
        return {{"phase", phase}, {"id", id}, {"prompt_total", prompt_total}, {"prompt_done", prompt_done}, {"cached", cached},
                {"prompt_per_second", prompt_tps}, {"generated", gen}, {"gen_per_second_now", gen_tps_now},
                {"gen_per_second_avg", gen_tps_avg}, {"ctx_used", prompt_total + gen}, {"ctx_max", ctx_max},
                {"requests", requests}, {"last", last}};
    }
};

struct Ctx {
    Live live;
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
    bool stream = false, include_usage = true, timings_per_token = false;
    std::string id;
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
    r.timings_per_token = body.value("timings_per_token", false);
    if (body.contains("stream_options") && body["stream_options"].is_object())
        r.include_usage = body["stream_options"].value("include_usage", true);
    return r;
}

// runs generation, calling on_msg(new_msg) whenever the parsed message may have changed; returns finish reason
struct GenOut { std::string text, finish; int n_gen = 0, reused = 0, steps = 0, accepted = 0; double t_prompt = 0, t_gen = 0; };

GenOut run(Ctx & c, const Request & r, const std::function<bool(const std::string &)> & on_text) {
    GenOut o;
    o.finish = "length";
    GenStats st;
    Live & lv = c.live;
    const int P = (int) r.prompt.size();
    auto t0 = clk::now();
    {
        std::lock_guard<std::mutex> lk(lv.mu);
        lv.phase = "prompt"; lv.id = r.id; lv.prompt_total = P; lv.prompt_done = 0; lv.cached = 0; lv.gen = 0;
        lv.prompt_tps = lv.gen_tps_now = lv.gen_tps_avg = 0; lv.t_start = t0; lv.requests++;
    }
    fprintf(stderr, "[%s] prompt: %d tokens, max %d new\n", r.id.c_str(), P, r.max_tokens);
    auto t_log = t0;
    c.eng->set_prefill_progress([&](int done, int total, int reused) {
        const auto now = clk::now();
        const double tps = (done - reused) / std::max(1e-9, secs(t0, now));
        {
            std::lock_guard<std::mutex> lk(lv.mu);
            lv.prompt_done = done; lv.cached = reused; lv.prompt_tps = tps;
        }
        if (secs(t_log, now) >= 2.0 || done == total) {
            fprintf(stderr, "[%s] prompt eval: %6d / %d (cached %d)  %.0f t/s\n", r.id.c_str(), done, total, reused, tps);
            t_log = now;
        }
    });
    bool first = true;
    clk::time_point t1 = t0;
    c.eng->generate(r.prompt, r.max_tokens, true, &st, [&](int tok) {
        const auto now = clk::now();
        if (first) {
            t1 = now; first = false; t_log = now;
            std::lock_guard<std::mutex> lk(lv.mu);
            lv.phase = "generating"; lv.t_gen = now; lv.t_win = now; lv.win_n = 0;
        }
        if (llama_vocab_is_eog(c.vocab, tok)) { o.finish = "stop"; return false; }
        ++o.n_gen;
        {
            std::lock_guard<std::mutex> lk(lv.mu);
            lv.gen = o.n_gen;
            const double w = secs(lv.t_win, now);
            if (w >= 0.5) { lv.gen_tps_now = (o.n_gen - lv.win_n) / w; lv.t_win = now; lv.win_n = o.n_gen; }
            lv.gen_tps_avg = o.n_gen / std::max(1e-9, secs(t1, now));
        }
        if (secs(t_log, now) >= 2.0) {
            std::lock_guard<std::mutex> lk(lv.mu);
            fprintf(stderr, "[%s] generating: %6d tokens  now %.1f t/s  avg %.1f t/s  ctx %d / %d\n", r.id.c_str(), o.n_gen,
                    lv.gen_tps_now, lv.gen_tps_avg, P + o.n_gen, lv.ctx_max);
            t_log = now;
        }
        o.text += piece(c.vocab, tok);
        for (const auto & s : r.stops) {
            const size_t at = o.text.find(s, o.text.size() > s.size() + 64 ? o.text.size() - s.size() - 64 : 0);
            if (at != std::string::npos) { o.text.resize(at); o.finish = "stop"; return false; }
        }
        if (on_text && !on_text(o.text)) { o.finish = "cancelled"; return false; }
        return true;
    }, r.sp);
    c.eng->set_prefill_progress({});
    auto t2 = clk::now();
    if (first) t1 = t2;
    o.reused = st.prompt_reused;
    o.steps = st.steps; o.accepted = st.accepted;
    o.t_prompt = secs(t0, t1);
    o.t_gen = secs(t1, t2);
    const int n_new = P - o.reused;
    fprintf(stderr, "[%s] prompt eval time = %10.2f ms / %6d tokens (%8.2f ms per token, %8.2f tokens per second), cached %d\n",
            r.id.c_str(), 1e3 * o.t_prompt, n_new, 1e3 * o.t_prompt / std::max(1, n_new), n_new / std::max(1e-9, o.t_prompt), o.reused);
    fprintf(stderr, "[%s]        eval time = %10.2f ms / %6d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            r.id.c_str(), 1e3 * o.t_gen, o.n_gen, 1e3 * o.t_gen / std::max(1, o.n_gen), o.n_gen / std::max(1e-9, o.t_gen));
    fprintf(stderr, "[%s]       total time = %10.2f ms / %6d tokens, finish: %s, draft acceptance %.2f / %d per step\n",
            r.id.c_str(), 1e3 * secs(t0, t2), n_new + o.n_gen, o.finish.c_str(), o.steps ? (double) o.accepted / o.steps : 0.0,
            c.eng->n_draft());
    {
        std::lock_guard<std::mutex> lk(lv.mu);
        lv.phase = "idle";
        lv.last = {{"id", r.id}, {"prompt_n", n_new}, {"cached", o.reused}, {"prompt_ms", 1e3 * o.t_prompt},
                   {"prompt_per_second", n_new / std::max(1e-9, o.t_prompt)}, {"predicted_n", o.n_gen}, {"predicted_ms", 1e3 * o.t_gen},
                   {"predicted_per_second", o.n_gen / std::max(1e-9, o.t_gen)}, {"finish", o.finish},
                   {"draft_acceptance", o.steps ? (double) o.accepted / o.steps : 0.0}};
    }
    return o;
}

const char * STATS_PAGE = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>hyper live</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dd;--acc:#2f6fdf}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#9a998f;--card:#1e1e1c;--line:#33332f;--acc:#6ea2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:28px;font-variant-numeric:tabular-nums}
.bar{height:6px;background:var(--line);border-radius:3px;margin-top:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
table{width:100%;border-collapse:collapse;margin-top:16px;font-variant-numeric:tabular-nums}td{padding:4px 0;border-bottom:1px solid var(--line)}td:last-child{text-align:right}
</style></head><body><main>
<h1>hyper live</h1><div class="sub" id="ph">…</div>
<div class="grid">
<div class="c"><div class="l">generation now</div><div class="v" id="g1">–</div></div>
<div class="c"><div class="l">generation avg</div><div class="v" id="g2">–</div></div>
<div class="c"><div class="l">prompt eval</div><div class="v" id="p1">–</div><div class="bar"><i id="pb" style="width:0"></i></div></div>
<div class="c"><div class="l">context</div><div class="v" id="cx">–</div><div class="bar"><i id="cb" style="width:0"></i></div></div>
</div>
<table id="last"></table>
</main><script>
const f=(x,d=1)=>Number(x).toFixed(d);
async function tick(){try{const s=await (await fetch('/stats')).json();
document.getElementById('ph').textContent=s.phase+(s.id?' · '+s.id:'')+' · requests '+s.requests;
const busy=s.phase!=='idle';
document.getElementById('g1').textContent=busy&&s.phase==='generating'?f(s.gen_per_second_now)+' t/s':'–';
document.getElementById('g2').textContent=s.generated?f(s.gen_per_second_avg)+' t/s':'–';
document.getElementById('p1').textContent=s.prompt_total?f(s.prompt_per_second,0)+' t/s':'–';
document.getElementById('pb').style.width=(s.prompt_total?100*s.prompt_done/s.prompt_total:0)+'%';
document.getElementById('cx').textContent=s.ctx_used.toLocaleString()+' / '+s.ctx_max.toLocaleString();
document.getElementById('cb').style.width=(100*s.ctx_used/s.ctx_max)+'%';
const L=s.last;document.getElementById('last').innerHTML=L?`<tr><td colspan=2><b>last request ${L.id}</b></td></tr>
<tr><td>prompt eval</td><td>${L.prompt_n} tokens (cached ${L.cached}) · ${f(L.prompt_ms,0)} ms · ${f(L.prompt_per_second,0)} t/s</td></tr>
<tr><td>generation</td><td>${L.predicted_n} tokens · ${f(L.predicted_ms,0)} ms · ${f(L.predicted_per_second)} t/s</td></tr>
<tr><td>draft acceptance</td><td>${f(L.draft_acceptance,2)} per step</td></tr><tr><td>finish</td><td>${L.finish}</td></tr>`:'';
}catch(e){}}
setInterval(tick,500);tick();
</script></body></html>)HTML";

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
    c.live.ctx_max = ctx;
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
    srv.Get("/stats", [&](const httplib::Request &, httplib::Response & res) { cors(res); res.set_content(c.live.json().dump(), "application/json"); });
    srv.Get("/", [&](const httplib::Request &, httplib::Response & res) { res.set_content(STATS_PAGE, "text/html; charset=utf-8"); });
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
        r->id = id.substr(9, 6) + id.substr(id.size() - 4);
        const long created = (long) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

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
                for (const auto & d : common_chat_msg_diff::compute_diffs(prev, msg)) {
                    ojson ch = chunk(diff_to_delta(d), nullptr);
                    if (r->timings_per_token) {
                        const ojson lj = c.live.json();
                        ch["timings"] = {{"cache_n", lj["cached"]}, {"prompt_n", lj["prompt_total"].get<int>() - lj["cached"].get<int>()},
                                         {"prompt_per_second", lj["prompt_per_second"]}, {"predicted_n", lj["generated"]},
                                         {"predicted_per_second", lj["gen_per_second_avg"]}};
                    }
                    if (!send(ch)) return false;
                }
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
