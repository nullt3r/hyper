// hyper OpenAI-compatible server: /v1/chat/completions (streaming and not), /v1/models, /health; web chat at /, live stats at /live.
// Chat templates, reasoning and tool-call parsing come from mainline llama.cpp's libcommon (same code as
// llama-server); the vocabulary is loaded with libllama (vocab only). Generation samples (greedy at temperature 0) with MTP speculative
// decoding (exact: drafts are accepted when they equal the token sampled at their row), prompt cache with
// recurrent-state snapshots; requests are served one at a time.
//
// usage: hyper-server <model.gguf> [--host 0.0.0.0] [--port 8080] [--ctx 262144] [--draft 3] [--alias name]
//                     [--temp 0.6] [--top-p 0.95] [--top-k 20] [--min-p 0] [--snapshots 48]   (request fields override)
//                     qwen4exp: [--gpu-frac 1.0] [--cpu-threads 30] [--expert-stats file] [--mtp nextn.gguf]   (context default 131072)
//                     glm5-next: [--gpu-frac 1.0] [--cpu-threads 30] [--expert-stats file]   (context default 65536)
//                     [--reasoning-effort low|high|max|none]   (template default when absent; requests may override)
//                     [--chat-template-file t.jinja]   (instead of the GGUF's template)
#include "engine.h"
#include "engine4.h"
#include "engine5.h"
#include "chat_page.h"

#include "chat.h"
#include "llama.h"

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>
#include <set>
#include <string>
#include <vector>

using ojson = nlohmann::ordered_json;
static std::string g_reasoning_effort;   // --reasoning-effort: default for requests without one
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
    std::function<void()> after_request;
    SamplingParams defaults;
    LLM * eng = nullptr;
    llama_model * vm = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr tmpls;
    std::string alias;
    bool glm = false;   // glm5-next: its templates know reasoning effort low / high (/ max)
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
    {   // reasoning effort: the request's OpenAI field, else --reasoning-effort (GLM templates know low / high, anything else = max;
        // Qwen's know xhigh / medium / low, unsloth's also high)
        std::string eff = body.contains("reasoning_effort") && body["reasoning_effort"].is_string() ? body["reasoning_effort"].get<std::string>()
                                                                                                   : g_reasoning_effort;
        if (c.glm && eff == "medium") eff = "high";
        if (eff == "none") in.enable_thinking = false;
        else if (!eff.empty() && !in.chat_template_kwargs.count("reasoning_effort")) in.chat_template_kwargs["reasoning_effort"] = "\"" + eff + "\"";
    }
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

// Streaming: the parse + send of each snapshot of the text runs on its own thread (parsing the whole text is O(n) per
// token; on the generation thread it would sit between two tokens while the GPUs wait). Snapshots are coalesced: the
// thread always takes the latest text; the generation thread only copies the string.
struct Streamer {
    std::mutex mu;
    std::condition_variable cv;
    std::string latest;
    bool has = false, stop = false, cancelled = false;
    std::thread th;
    double busy_ms = 0;
    long n_parsed = 0;
    void start(std::function<bool(const std::string &)> fn) {
        th = std::thread([this, fn = std::move(fn)] {
            for (;;) {
                std::string text;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return has || stop; });
                    if (!has && stop) return;
                    text.swap(latest); has = false;
                }
                const auto t0 = clk::now();
                bool ok = true;
                try { ok = fn(text); }
                catch (const std::exception & e) { fprintf(stderr, "stream: snapshot skipped (%s)\n", e.what()); }
                busy_ms += 1e3 * secs(t0, clk::now());
                ++n_parsed;
                if (!ok) { std::lock_guard<std::mutex> lk(mu); cancelled = true; }
            }
        });
    }
    bool offer(const std::string & text) {   // false once the consumer gave up
        std::lock_guard<std::mutex> lk(mu);
        if (cancelled) return false;
        latest = text; has = true;
        cv.notify_one();
        return true;
    }
    void finish() {
        { std::lock_guard<std::mutex> lk(mu); stop = true; cv.notify_one(); }
        if (th.joinable()) th.join();
    }
};

GenOut run(Ctx & c, const Request & r, const std::function<bool(const std::string &)> & on_text) {
    GenOut o;
    o.finish = "length";
    Streamer streamer;
    if (on_text) streamer.start(on_text);
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
    c.eng->generate(r.prompt, r.max_tokens, c.eng->n_draft() > 0, &st, [&](int tok) {
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
        if (on_text && !streamer.offer(o.text)) { o.finish = "cancelled"; return false; }
        return true;
    }, r.sp);
    streamer.finish();
    if (on_text && streamer.cancelled && o.finish != "cancelled") o.finish = "cancelled";
    c.eng->set_prefill_progress({});
    if (c.after_request) c.after_request();
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
    if (on_text) fprintf(stderr, "[%s]  stream parse+send: %.0f ms on the stream thread (%ld snapshots of %d tokens)\n", r.id.c_str(),
                         streamer.busy_ms, streamer.n_parsed, o.n_gen);
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
<h1>hyper live <a href="/" style="font-size:14px;font-weight:400;color:var(--acc)">chat</a></h1><div class="sub" id="ph">…</div>
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


// ---------------- OpenAI Responses API (/v1/responses) on top of the chat path ----------------
std::string part_text(const ojson & c) {   // string, or an array of content parts (input_text / output_text / text / ...)
    if (c.is_string()) return c.get<std::string>();
    std::string t;
    if (c.is_array())
        for (auto & p : c) {
            if (p.is_string()) t += p.get<std::string>();
            else if (p.is_object() && p.contains("text") && p["text"].is_string()) t += p["text"].get<std::string>();
        }
    return t;
}

// Responses request -> chat completions body; custom (freeform) tools become functions with one string argument "input"
ojson responses_to_chat(const ojson & body, std::set<std::string> & custom_tools) {
    ojson chat = ojson::object();
    std::string sys = body.contains("instructions") && body["instructions"].is_string() ? body["instructions"].get<std::string>() : "";
    ojson msgs = ojson::array();
    std::string pending_reasoning;
    auto assistant = [&]() -> ojson & {   // the open assistant message of this turn (created on demand)
        if (msgs.empty() || msgs.back()["role"] != "assistant" || msgs.back().contains("closed")) {
            ojson m = {{"role", "assistant"}, {"content", ""}};
            msgs.push_back(m);
        }
        ojson & m = msgs.back();
        if (!pending_reasoning.empty()) { m["reasoning_content"] = pending_reasoning; pending_reasoning.clear(); }
        return m;
    };
    auto add_call = [&](const std::string & id, const std::string & name, const std::string & args) {
        ojson & m = assistant();
        if (!m.contains("tool_calls")) m["tool_calls"] = ojson::array();
        m["tool_calls"].push_back({{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", args}}}});
    };
    const ojson & input = body.contains("input") ? body["input"] : ojson();
    if (input.is_string()) msgs.push_back({{"role", "user"}, {"content", input.get<std::string>()}});
    else if (input.is_array())
        for (const auto & it : input) {
            if (!it.is_object()) continue;
            const std::string type = it.value("type", it.contains("role") ? "message" : "");
            if (type == "message") {
                std::string role = it.value("role", "user");
                const std::string text = part_text(it.contains("content") ? it["content"] : ojson(""));
                if (role == "system" || role == "developer") { sys += (sys.empty() ? "" : "\n\n") + text; continue; }
                if (role == "assistant") {
                    ojson & m = assistant();
                    std::string cur = m["content"].is_string() ? m["content"].get<std::string>() : "";
                    m["content"] = cur + text;
                    continue;
                }
                if (!msgs.empty() && msgs.back()["role"] == "assistant") msgs.back()["closed"] = true;
                msgs.push_back({{"role", role}, {"content", text}});
            } else if (type == "reasoning") {
                std::string t = it.contains("content") ? part_text(it["content"]) : "";
                if (t.empty() && it.contains("summary")) t = part_text(it["summary"]);
                pending_reasoning += t;
            } else if (type == "function_call") {
                add_call(it.value("call_id", ""), it.value("name", ""), it.value("arguments", "{}"));
            } else if (type == "custom_tool_call") {
                add_call(it.value("call_id", ""), it.value("name", ""), ojson({{"input", it.value("input", "")}}).dump());
            } else if (type == "function_call_output" || type == "custom_tool_call_output") {
                if (!msgs.empty() && msgs.back()["role"] == "assistant") msgs.back()["closed"] = true;
                const std::string out = it.contains("output") ? (it["output"].is_string() ? it["output"].get<std::string>()
                                                                 : it["output"].is_object() && it["output"].contains("content")
                                                                       ? part_text(it["output"]["content"]) : part_text(it["output"]))
                                                              : "";
                msgs.push_back({{"role", "tool"}, {"tool_call_id", it.value("call_id", "")}, {"content", out}});
            }
        }
    for (auto & m : msgs) m.erase("closed");
    if (!sys.empty()) msgs.insert(msgs.begin(), ojson({{"role", "system"}, {"content", sys}}));
    chat["messages"] = msgs;
    ojson tools = ojson::array();
    if (body.contains("tools") && body["tools"].is_array())
        for (const auto & t : body["tools"]) {
            const std::string type = t.value("type", "");
            if (type == "function") {
                ojson fn = {{"name", t.value("name", "")}, {"description", t.value("description", "")}};
                fn["parameters"] = t.contains("parameters") && t["parameters"].is_object() ? t["parameters"] : ojson({{"type", "object"}, {"properties", ojson::object()}});
                tools.push_back({{"type", "function"}, {"function", fn}});
            } else if (type == "custom") {
                const std::string name = t.value("name", "");
                custom_tools.insert(name);
                std::string desc = t.value("description", "");
                if (t.contains("format") && t["format"].is_object() && t["format"].contains("definition"))
                    desc += "\n\nThe input must follow this grammar:\n" + t["format"].value("definition", "");
                tools.push_back({{"type", "function"}, {"function", {{"name", name}, {"description", desc},
                    {"parameters", {{"type", "object"}, {"properties", {{"input", {{"type", "string"}, {"description", "the raw tool input"}}}}},
                                    {"required", ojson::array({"input"})}}}}}});
            }
        }
    if (!tools.empty()) chat["tools"] = tools;
    if (body.contains("tool_choice") && body["tool_choice"].is_string()) chat["tool_choice"] = body["tool_choice"];
    if (body.contains("parallel_tool_calls")) chat["parallel_tool_calls"] = body["parallel_tool_calls"];
    if (body.contains("max_output_tokens") && body["max_output_tokens"].is_number_integer()) chat["max_tokens"] = body["max_output_tokens"];
    for (const char * k : {"temperature", "top_p", "top_k", "min_p", "seed", "chat_template_kwargs"}) if (body.contains(k) && !body[k].is_null()) chat[k] = body[k];
    if (body.contains("reasoning") && body["reasoning"].is_object() && body["reasoning"].contains("effort") && body["reasoning"]["effort"].is_string())
        chat["reasoning_effort"] = body["reasoning"]["effort"];
    chat["stream"] = body.value("stream", false);
    return chat;
}

ojson resp_usage(const Request & r, const GenOut & o) {
    return {{"input_tokens", (int) r.prompt.size()}, {"input_tokens_details", {{"cached_tokens", o.reused}}},
            {"output_tokens", o.n_gen}, {"output_tokens_details", {{"reasoning_tokens", 0}}},
            {"total_tokens", (int) r.prompt.size() + o.n_gen}};
}
ojson tool_call_item(const common_chat_tool_call & t, const std::string & item_id, const std::set<std::string> & custom, bool done) {
    if (custom.count(t.name)) {
        std::string input = t.arguments;
        try { const ojson a = ojson::parse(t.arguments); if (a.contains("input") && a["input"].is_string()) input = a["input"].get<std::string>(); } catch (...) {}
        return {{"type", "custom_tool_call"}, {"id", item_id}, {"call_id", t.id}, {"name", t.name}, {"input", done ? input : ""},
                {"status", done ? "completed" : "in_progress"}};
    }
    return {{"type", "function_call"}, {"id", item_id}, {"call_id", t.id}, {"name", t.name}, {"arguments", done ? t.arguments : ""},
            {"status", done ? "completed" : "in_progress"}};
}
ojson reasoning_item(const std::string & id, const std::string & text) {
    return {{"type", "reasoning"}, {"id", id}, {"summary", ojson::array({ojson({{"type", "summary_text"}, {"text", text}})})},
            {"content", ojson::array({ojson({{"type", "reasoning_text"}, {"text", text}})})}};
}
ojson message_item(const std::string & id, const std::string & text, bool done) {
    return {{"type", "message"}, {"id", id}, {"status", done ? "completed" : "in_progress"}, {"role", "assistant"},
            {"content", done ? ojson::array({ojson({{"type", "output_text"}, {"text", text}, {"annotations", ojson::array()}})}) : ojson::array()}};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--host H] [--port P] [--ctx N] [--draft K] [--alias name]\n", argv[0]); return 1; }
    std::string host = "0.0.0.0", alias;
    int port = 8080, ctx = 262144, draft = 3, snaps = 48, cpu_threads = 30;
    float gpu_frac = 1.0f;
    std::string stats_path;   // qwen4exp: routing statistics, read at start and updated after every request
    std::string mtp_path;     // qwen4exp: separate NextN (MTP) GGUF for speculative decoding
    std::string template_file;   // chat template (jinja) instead of the GGUF's
    bool ctx_given = false;
    SamplingParams defaults;   // Qwen's recommendation for thinking mode
    defaults.temp = 0.6f; defaults.top_p = 0.95f; defaults.top_k = 20; defaults.min_p = 0.0f;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--host") host = v; else if (k == "--port") port = std::stoi(v); else if (k == "--ctx") { ctx = std::stoi(v); ctx_given = true; }
        else if (k == "--gpu-frac") gpu_frac = std::stof(v); else if (k == "--cpu-threads") cpu_threads = std::stoi(v);
        else if (k == "--expert-stats") { setenv("HYPER4_STATS", v.c_str(), 1); stats_path = v; }
        else if (k == "--mtp") mtp_path = v;
        else if (k == "--draft") draft = std::stoi(v); else if (k == "--alias") alias = v;
        else if (k == "--temp") defaults.temp = std::stof(v); else if (k == "--top-p") defaults.top_p = std::stof(v);
        else if (k == "--top-k") defaults.top_k = std::stoi(v); else if (k == "--min-p") defaults.min_p = std::stof(v);
        else if (k == "--snapshots") snaps = std::stoi(v);
        else if (k == "--reasoning-effort") g_reasoning_effort = v;
        else if (k == "--chat-template-file") template_file = v;
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
    std::string tmpl_override;
    if (!template_file.empty()) {
        std::ifstream tf(template_file);
        if (!tf) { fprintf(stderr, "cannot read %s\n", template_file.c_str()); return 1; }
        tmpl_override.assign(std::istreambuf_iterator<char>(tf), std::istreambuf_iterator<char>());
        fprintf(stderr, "hyper-server: chat template from %s (%zu chars)\n", template_file.c_str(), tmpl_override.size());
    }
    c.tmpls = common_chat_templates_init(c.vm, tmpl_override);

    std::string arch;
    { GGUF g(path); arch = g.arch(); }
    c.glm = arch == "glm5-next";
    std::unique_ptr<LLM> engine;
    if (arch == "qwen4exp") {
        if (!ctx_given) ctx = 131072;
        Engine4Options o4;
        o4.max_pos = ctx;
        o4.gpu_expert_frac = gpu_frac;
        o4.cpu_threads = cpu_threads;
        o4.prompt_cache = true;
        o4.max_snapshots = snaps;
        o4.mtp_path = mtp_path;
        o4.n_draft = draft;
        auto e4 = std::make_unique<Engine4>(path, o4);
        if (!stats_path.empty()) {
            Engine4 * pe = e4.get();
            c.after_request = [pe, stats_path] { pe->save_expert_stats(stats_path); };
        }
        draft = engine ? draft : e4->n_draft();
        engine = std::move(e4);
    } else if (arch == "glm5-next") {
        if (!ctx_given) ctx = 65536;
        Engine5Options o5;
        o5.max_pos = ctx;
        o5.gpu_expert_frac = gpu_frac;
        o5.cpu_threads = cpu_threads;
        o5.prompt_cache = true;
        o5.max_snapshots = snaps;
        o5.n_draft = draft;   // prompt-lookup speculation
        auto e5 = std::make_unique<Engine5>(path, o5);
        if (!stats_path.empty()) {
            Engine5 * pe = e5.get();
            c.after_request = [pe, stats_path] { pe->save_expert_stats(stats_path); };
        }
        draft = e5->n_draft();
        engine = std::move(e5);
    } else {
        EngineOptions opt;
        opt.max_pos = ctx;
        opt.n_draft = draft;
        opt.prompt_cache = true;
        opt.max_snapshots = snaps;
        engine = std::make_unique<Engine>(path, opt);
    }
    c.live.ctx_max = ctx;
    LLM & eng = *engine;
    c.eng = &eng;
    {
        int n_snap = 0;   // message starts: ChatML, or GLM's role tokens (an engine keeps every token it is given)
        for (const char * m : {"<|im_start|>", "<|assistant|>", "<|user|>", "<|observation|>"}) {
            const std::vector<int> im = tokenize(c.vocab, m);
            if (im.size() == 1 && (n_snap == 0 || arch == "glm5-next")) { eng.set_snapshot_token(im[0]); ++n_snap; }
        }
        if (!n_snap) fprintf(stderr, "hyper-server: no message-start token, prompt-cache snapshots only every 4096 tokens\n");
    }
    {   // warm up: builds the CUDA graphs
        GenStats st;
        eng.generate(tokenize(c.vocab, "Hello"), 4, eng.n_draft() > 0, &st);
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
    srv.Get("/props", [&](const httplib::Request &, httplib::Response & res) {   // (web UI: the server's sampling defaults)
        cors(res);
        res.set_content(ojson({{"model", c.alias}, {"sampling", {{"temp", defaults.temp}, {"top_p", defaults.top_p}, {"top_k", defaults.top_k},
                                                                  {"min_p", defaults.min_p}}}}).dump(), "application/json");
    });
    srv.Get("/stats", [&](const httplib::Request &, httplib::Response & res) { cors(res); res.set_content(c.live.json().dump(), "application/json"); });
    srv.Get("/", [&](const httplib::Request &, httplib::Response & res) { res.set_content(CHAT_PAGE, "text/html; charset=utf-8"); });
    srv.Get("/live", [&](const httplib::Request &, httplib::Response & res) { res.set_content(STATS_PAGE, "text/html; charset=utf-8"); });
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
            if (const char * dump = getenv("HYPER_DUMP_REQUEST")) {   // request bodies <dir>/<time>.json (replay for benchmarks)
                const std::string fn = std::string(dump) + "/" + std::to_string((long long) time(nullptr)) + ".json";
                FILE * f = fopen(fn.c_str(), "wb");
                if (f) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
            }
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
                // a partial parse can briefly see fewer tool calls than the previous one (a call whose arguments are
                // still just "{"): compute_diffs throws then; skip this snapshot, a later one (or the final parse) catches up
                std::vector<common_chat_msg_diff> diffs;
                try { diffs = common_chat_msg_diff::compute_diffs(prev, msg); } catch (const std::exception &) { return true; }
                for (const auto & d : diffs) {
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

    // OpenAI Responses API (Codex & co.): translated to the chat path, output as response items / response.* events
    auto responses = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        auto fail = [&](int code, const std::string & msg) {
            res.status = code;
            res.set_content(ojson({{"error", {{"message", msg}, {"type", code == 400 ? "invalid_request_error" : "server_error"}}}}).dump(), "application/json");
        };
        std::shared_ptr<Request> r;
        auto custom = std::make_shared<std::set<std::string>>();
        ojson rbody;
        try {
            rbody = ojson::parse(req.body);
            if (const char * dump = getenv("HYPER_DUMP_REQUEST")) {
                const std::string fn = std::string(dump) + "/" + std::to_string((long long) time(nullptr)) + "-responses.json";
                FILE * f = fopen(fn.c_str(), "wb");
                if (f) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
            }
            r = std::make_shared<Request>(prepare(c, responses_to_chat(rbody, *custom)));
        } catch (const std::exception & e) { fail(400, e.what()); return; }
        const std::string rid = random_id("resp_", c.next_id++);
        r->id = rid.substr(5, 6) + rid.substr(rid.size() - 4);
        const long created = (long) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto base = std::make_shared<ojson>(ojson({{"id", rid}, {"object", "response"}, {"created_at", created}, {"model", c.alias},
                                                   {"parallel_tool_calls", rbody.value("parallel_tool_calls", true)},
                                                   {"tool_choice", rbody.contains("tool_choice") ? rbody["tool_choice"] : ojson("auto")},
                                                   {"tools", rbody.contains("tools") ? rbody["tools"] : ojson::array()}}));

        auto final_output = [r, custom](const common_chat_msg & msg, long & n) {
            ojson out = ojson::array();
            if (!msg.reasoning_content.empty()) out.push_back(reasoning_item(random_id("rs_", n++), msg.reasoning_content));
            if (!msg.content.empty()) out.push_back(message_item(random_id("msg_", n++), msg.content, true));
            for (const auto & t : msg.tool_calls) out.push_back(tool_call_item(t, random_id("fc_", n++), *custom, true));
            return out;
        };
        auto finish_obj = [base, r](ojson out, const GenOut & o) {
            ojson resp = *base;
            resp["status"] = o.finish == "length" ? "incomplete" : "completed";
            if (o.finish == "length") resp["incomplete_details"] = {{"reason", "max_output_tokens"}};
            resp["output"] = std::move(out);
            resp["usage"] = resp_usage(*r, o);
            return resp;
        };

        if (!r->stream) {
            std::lock_guard<std::mutex> lk(c.mu);
            GenOut o;
            try { o = run(c, *r, {}); } catch (const std::exception & e) { fail(500, e.what()); return; }
            common_chat_msg msg = common_chat_parse(o.text, false, r->pp);
            long n = 0;
            for (auto & t : msg.tool_calls) if (t.id.empty()) t.id = random_id("call_", n++);
            res.set_content(finish_obj(final_output(msg, n), o).dump(-1, ' ', false, ojson::error_handler_t::replace), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&c, r, custom, base, finish_obj](size_t, httplib::DataSink & sink) {
            long seq = 0, nid = 0;
            auto send = [&](const char * type, ojson j) {
                j["type"] = type;
                j["sequence_number"] = seq++;
                const std::string s = std::string("event: ") + type + "\ndata: " + j.dump(-1, ' ', false, ojson::error_handler_t::replace) + "\n\n";
                return sink.write(s.data(), s.size());
            };
            std::lock_guard<std::mutex> lk(c.mu);
            {
                ojson resp = *base;
                resp["status"] = "in_progress";
                resp["output"] = ojson::array();
                send("response.created", {{"response", resp}});
                send("response.in_progress", {{"response", resp}});
            }
            // open item state: 0 none, 1 reasoning, 2 message, 3 tool call
            int open = 0, out_index = -1;
            std::string item_id, acc;
            size_t open_tool = std::string::npos;
            ojson done_items = ojson::array();
            common_chat_msg prev = common_chat_parse("", true, r->pp);
            std::vector<std::string> ids;
            long tc = 0;
            auto gen_id = [&] { return random_id("call_", tc++); };
            bool ok = true;
            auto close_item = [&](const common_chat_msg & msg) {
                if (open == 1) {
                    send("response.reasoning_summary_text.done", {{"item_id", item_id}, {"output_index", out_index}, {"summary_index", 0}, {"text", acc}});
                    send("response.reasoning_summary_part.done", {{"item_id", item_id}, {"output_index", out_index}, {"summary_index", 0},
                                                                   {"part", {{"type", "summary_text"}, {"text", acc}}}});
                    const ojson it = reasoning_item(item_id, acc);
                    send("response.output_item.done", {{"output_index", out_index}, {"item", it}});
                    done_items.push_back(it);
                } else if (open == 2) {
                    send("response.output_text.done", {{"item_id", item_id}, {"output_index", out_index}, {"content_index", 0}, {"text", acc}});
                    send("response.content_part.done", {{"item_id", item_id}, {"output_index", out_index}, {"content_index", 0},
                                                        {"part", {{"type", "output_text"}, {"text", acc}, {"annotations", ojson::array()}}}});
                    const ojson it = message_item(item_id, acc, true);
                    send("response.output_item.done", {{"output_index", out_index}, {"item", it}});
                    done_items.push_back(it);
                } else if (open == 3 && open_tool < msg.tool_calls.size()) {
                    const ojson it = tool_call_item(msg.tool_calls[open_tool], item_id, *custom, true);
                    if (it["type"] == "function_call")
                        send("response.function_call_arguments.done", {{"item_id", item_id}, {"output_index", out_index}, {"arguments", it["arguments"]}});
                    else send("response.custom_tool_call_input.done", {{"item_id", item_id}, {"output_index", out_index}, {"input", it["input"]}});
                    send("response.output_item.done", {{"output_index", out_index}, {"item", it}});
                    done_items.push_back(it);
                }
                open = 0; acc.clear(); open_tool = std::string::npos;
            };
            auto push = [&](const std::string & text, bool partial) {
                common_chat_msg msg;
                try { msg = common_chat_parse(text, partial, r->pp); } catch (const std::exception &) { return true; }
                msg.set_tool_call_ids(ids, gen_id);
                // a partial parse can briefly see fewer tool calls than the previous one (a call whose arguments are
                // still just "{"): compute_diffs throws then; skip this snapshot, a later one (or the final parse) catches up
                std::vector<common_chat_msg_diff> diffs;
                try { diffs = common_chat_msg_diff::compute_diffs(prev, msg); } catch (const std::exception &) { return true; }
                for (const auto & d : diffs) {
                    if (!d.reasoning_content_delta.empty()) {
                        if (open != 1) {
                            close_item(msg);
                            open = 1; ++out_index; item_id = random_id("rs_", nid++);
                            fprintf(stderr, "[%s] responses: reasoning (item %d)\n", r->id.c_str(), out_index);
                            send("response.output_item.added", {{"output_index", out_index}, {"item", {{"type", "reasoning"}, {"id", item_id}, {"summary", ojson::array()}}}});
                            send("response.reasoning_summary_part.added", {{"item_id", item_id}, {"output_index", out_index}, {"summary_index", 0},
                                                                            {"part", {{"type", "summary_text"}, {"text", ""}}}});
                        }
                        acc += d.reasoning_content_delta;
                        if (!send("response.reasoning_summary_text.delta", {{"item_id", item_id}, {"output_index", out_index}, {"summary_index", 0},
                                                                            {"delta", d.reasoning_content_delta}})) return false;
                        // the same text as raw reasoning (clients that show raw reasoning, e.g. Codex show_raw_agent_reasoning)
                        if (!send("response.reasoning_text.delta", {{"item_id", item_id}, {"output_index", out_index}, {"content_index", 0},
                                                                     {"delta", d.reasoning_content_delta}})) return false;
                    }
                    if (!d.content_delta.empty()) {
                        if (open != 2) {
                            close_item(msg);
                            open = 2; ++out_index; item_id = random_id("msg_", nid++);
                            fprintf(stderr, "[%s] responses: message text (item %d)\n", r->id.c_str(), out_index);
                            send("response.output_item.added", {{"output_index", out_index}, {"item", message_item(item_id, "", false)}});
                            send("response.content_part.added", {{"item_id", item_id}, {"output_index", out_index}, {"content_index", 0},
                                                                 {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", ojson::array()}}}});
                        }
                        acc += d.content_delta;
                        if (!send("response.output_text.delta", {{"item_id", item_id}, {"output_index", out_index}, {"content_index", 0},
                                                                 {"delta", d.content_delta}})) return false;
                    }
                    if (d.tool_call_index != std::string::npos && d.tool_call_index < msg.tool_calls.size()) {
                        if (open != 3 || open_tool != d.tool_call_index) {
                            close_item(msg);
                            open = 3; open_tool = d.tool_call_index; ++out_index; item_id = random_id("fc_", nid++);
                            fprintf(stderr, "[%s] responses: tool call %s (item %d; clients show it when complete)\n", r->id.c_str(),
                                    msg.tool_calls[open_tool].name.c_str(), out_index);
                            send("response.output_item.added", {{"output_index", out_index},
                                                                {"item", tool_call_item(msg.tool_calls[open_tool], item_id, *custom, false)}});
                        }
                        if (!d.tool_call_delta.arguments.empty() && !custom->count(msg.tool_calls[open_tool].name))
                            if (!send("response.function_call_arguments.delta", {{"item_id", item_id}, {"output_index", out_index},
                                                                                  {"delta", d.tool_call_delta.arguments}})) return false;
                    }
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
                ojson resp = *base;
                resp["status"] = "failed";
                resp["error"] = {{"code", "server_error"}, {"message", e.what()}};
                send("response.failed", {{"response", resp}});
                sink.done();
                return false;
            }
            if (ok) {
                push(o.text, false);
                close_item(prev);
                send("response.completed", {{"response", finish_obj(done_items, o)}});
            }
            sink.done();
            return true;
        });
    };
    srv.Post("/v1/responses", responses);
    srv.Post("/responses", responses);

    fprintf(stderr, "hyper-server: listening on http://%s:%d (model \"%s\", context %d, %d drafts)\n", host.c_str(), port,
            c.alias.c_str(), ctx, draft);
    if (!srv.listen(host, port)) { fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
