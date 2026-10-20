// hyper OpenAI-compatible server: /v1/chat/completions (streaming and not), /v1/models, /health; web chat at /, live stats at /live.
// Chat templates, reasoning and tool-call parsing come from mainline llama.cpp's libcommon (same code as
// llama-server); the vocabulary is loaded with libllama (vocab only). Generation samples (greedy at temperature 0) with MTP speculative
// decoding (exact: drafts are accepted when they equal the token sampled at their row), prompt cache with
// recurrent-state snapshots; requests are served one at a time.
//
// usage: hyper-server <model.gguf> [--host 0.0.0.0] [--port 8080] [--ctx 262144] [--draft 3] [--alias name] [--mcp name=url ...]
//                     [--temp 0.6] [--top-p 0.95] [--top-k 20] [--min-p 0] [--snapshots 48]   (request fields override)
//                     qwen4exp: [--gpu-frac 1.0] [--cpu-threads 30] [--expert-stats file] [--mtp nextn.gguf]   (context default 131072)
//                     glm5-next: [--gpu-frac 1.0] [--cpu-threads 30] [--expert-stats file]   (context default 65536)
//                     [--reasoning-effort low|high|max|none]   (template default when absent; requests may override)
//                     [--chat-template-file t.jinja]   (instead of the GGUF's template)
#include "engine.h"
#include "engine4.h"
#include "engine5.h"
#include "engine6.h"
#include "engine7.h"
#include "chat_page.h"

#include "chat.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <list>
#include <memory>
#include <map>
#include <mutex>
#include <csignal>
#include <pthread.h>
#include <thread>
#include <set>
#include <string>
#include <vector>

using ojson = nlohmann::ordered_json;
static std::string g_reasoning_effort;   // --reasoning-effort: default for requests without one
static bool g_vision = false;            // --mmproj loaded: image parts reach the model
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
    // --mmproj: the vision encoder (libmtmd); images of a prompt become its rows at the image token's positions
    mtmd_context * mtmd = nullptr;
    int image_tok = -1, n_embd_inp = 0;
    std::mutex mtmd_mu;
    // encoded images by content hash (a conversation sends its images again with every request), least recently used out
    std::list<std::pair<std::string, std::shared_ptr<std::vector<float>>>> img_cache;
    size_t img_cache_bytes = 0;
    std::mutex mu;
    std::atomic<long> next_id{1};
    // prompt tokenization reused across requests (see tokenize_prompt): the previous prompt's text and tokens, and the special
    // tokens that start a message (llama.cpp tokenizes the text between special tokens piece by piece)
    std::mutex tok_mu;
    struct TokEntry { std::string text; std::vector<int> ids; uint64_t used = 0; };
    std::vector<TokEntry> tok_cache;   // a few recent prompts (an agent's side requests -- a title, a summary -- keep the main one)
    uint64_t tok_clock = 0;
    std::vector<std::pair<std::string, int>> tok_marks;
};

std::vector<int> tokenize(const llama_vocab * vocab, const std::string & text) {
    // one pass into a buffer of the upper bound (a token covers at least one byte; + BOS): a counting pass first doubled the
    // time for a 150k-token prompt
    std::vector<llama_token> t(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), (int) t.size(), true, true);
    if (n < 0) {   // (never with the bound above; the counting pass as before)
        t.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), (int) t.size(), true, true);
    }
    if (n < 0) throw std::runtime_error("tokenize failed");
    t.resize(n);
    return std::vector<int>(t.begin(), t.end());
}

// tokens of text[from, end) without the BOS (the continuation of a prompt from a special token on)
static std::vector<int> tokenize_tail(const llama_vocab * vocab, const std::string & text, size_t from) {
    std::vector<llama_token> t(text.size() - from + 8);
    int n = llama_tokenize(vocab, text.c_str() + from, (int) (text.size() - from), t.data(), (int) t.size(), false, true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    t.resize(n);
    return std::vector<int>(t.begin(), t.end());
}

// A prompt's tokens, reusing the previous prompt's up to the last message-start special token inside the text both share
// (llama.cpp splits the text at special tokens and tokenizes the pieces independently, so the tokens before that point
// are the same): an agent's next request repeats ~600 KB of history, whose full tokenization took ~230 ms.
// HYPER_TOKCACHE=0: always the whole text; HYPER_TOKCACHE_CHECK=1: also the whole text, compared (log on a mismatch)
std::vector<int> tokenize_prompt(Ctx & c, const std::string & text) {
    static const bool off = getenv("HYPER_TOKCACHE") && atoi(getenv("HYPER_TOKCACHE")) == 0;
    static const bool check = getenv("HYPER_TOKCACHE_CHECK") != nullptr;
    if (off) return tokenize(c.vocab, text);
    std::lock_guard<std::mutex> lk(c.tok_mu);
    if (c.tok_marks.empty())
        for (const char * m : {"<|im_start|>", "<|user|>", "<|assistant|>", "<|system|>", "<|observation|>"}) {
            const std::vector<int> t = tokenize_tail(c.vocab, m, 0);
            if (t.size() == 1) c.tok_marks.push_back({m, t[0]});
        }
    // occurrences of the marks in a text (byte offsets) and in a token list (indices), in order
    auto text_marks = [&](const std::string & s, size_t end) {
        std::vector<size_t> o;
        for (size_t q = s.find("<|", 0); q != std::string::npos && q < end; q = s.find("<|", q + 1))
            for (auto & [m, id] : c.tok_marks) if (s.compare(q, m.size(), m) == 0) { o.push_back(q); break; }
        return o;
    };
    std::vector<int> out;
    size_t from = 0;
    // the cached prompt sharing the longest text prefix
    int best = -1;
    size_t lcp = 0;
    for (int i = 0; i < (int) c.tok_cache.size(); ++i) {
        const std::string & t = c.tok_cache[i].text;
        size_t l = 0;
        const size_t lim = std::min(text.size(), t.size());
        while (l < lim && text[l] == t[l]) ++l;
        if (best < 0 || l > lcp) { best = i; lcp = l; }
    }
    if (best >= 0 && !c.tok_marks.empty()) {
        const Ctx::TokEntry & ce = c.tok_cache[best];
        const std::vector<size_t> om = text_marks(ce.text, ce.text.size());
        std::vector<size_t> ti;
        for (size_t i = 0; i < ce.ids.size(); ++i)
            for (auto & [m, id] : c.tok_marks) if (ce.ids[i] == id) { ti.push_back(i); break; }
        if (om.size() == ti.size()) {   // (else the marks do not map one to one: the whole text)
            int k = -1;   // the last mark starting inside the common prefix
            for (size_t i = 0; i < om.size() && om[i] < lcp; ++i) k = (int) i;
            if (k > 0) {
                out.assign(ce.ids.begin(), ce.ids.begin() + ti[k]);
                from = om[k];
            }
        }
    }
    if (from == 0) out = tokenize(c.vocab, text);
    else {
        const std::vector<int> tail = tokenize_tail(c.vocab, text, from);
        out.insert(out.end(), tail.begin(), tail.end());
    }
    if (check && from > 0) {
        const std::vector<int> full = tokenize(c.vocab, text);
        if (full != out) fprintf(stderr, "tokenize_prompt: MISMATCH (reused %zu bytes; %zu vs %zu tokens)\n", from, out.size(), full.size());
        else fprintf(stderr, "tokenize_prompt: reused %zu of %zu bytes, same tokens\n", from, text.size());
    }
    // the entry this prompt continues is replaced, else the least recently used one (at most 4)
    int slot = from > 0 ? best : -1;
    if (slot < 0 && c.tok_cache.size() < 4) { c.tok_cache.emplace_back(); slot = (int) c.tok_cache.size() - 1; }
    if (slot < 0)
        slot = (int) (std::min_element(c.tok_cache.begin(), c.tok_cache.end(), [](const Ctx::TokEntry & a, const Ctx::TokEntry & b) {
                          return a.used < b.used; }) - c.tok_cache.begin());
    c.tok_cache[slot].text = text;
    c.tok_cache[slot].ids = out;
    c.tok_cache[slot].used = ++c.tok_clock;
    return out;
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

ojson to_ojson(const common_json & j) { return ojson::parse(j.dump_safe()); }   // (a text
// cut inside a UTF-8 sequence at max_tokens must not throw)
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
    std::shared_ptr<PromptMedia> media;                        // images (vision): rows, positions, cache keys
    std::vector<std::shared_ptr<std::vector<float>>> media_keep;   // (their rows)
    common_chat_parser_params pp;
    std::vector<std::string> stops;
    int max_tokens = 0;
    bool stream = false, include_usage = true, timings_per_token = false;
    std::string id;
    SamplingParams sp;
    std::map<std::string, std::map<std::string, std::string>> arg_types;   // tool -> argument -> JSON schema type
    double t_parse = 0, t_convert = 0, t_template = 0, t_tokenize = 0;    // request preprocessing (ms)
};

// Tool calls the template's parser misses or rejects, in either XML syntax:
//   Qwen:  <tool_call><function=name><parameter=k>v</parameter>...</function></tool_call>
//   GLM:   <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
// llama.cpp's parsers want the arguments in schema order (and GLM's only string values); the models write them in any
// order (often alphabetical) and numbers / booleans bare. Values of non-string arguments are read as JSON. Only complete
// calls (closed blocks after the reasoning) are taken.
static bool g_dsml_v41 = false;   // the template writes DeepSeek-V4.1 DSML tool calls ("<｜DSML｜ calls>")

bool xml_tool_fallback(const std::string & text, const Request & r, common_chat_msg & out) {
    const size_t think_end = text.rfind("</think>");
    const size_t t0 = text.find("<tool_call>", think_end == std::string::npos ? 0 : think_end);
    if (t0 == std::string::npos || text.find("</tool_call>", t0) == std::string::npos) return false;
    try {
        out = common_chat_parse(text.substr(0, t0), false, r.pp);   // reasoning and content before the calls
    } catch (const std::exception &) {
        out = common_chat_msg();
        out.role = "assistant";
        out.content = text.substr(0, t0);
    }
    out.tool_calls.clear();
    auto trim = [](const std::string & v) {
        const size_t x = v.find_first_not_of(" \t\r\n"), y = v.find_last_not_of(" \t\r\n");
        return x == std::string::npos ? std::string() : v.substr(x, y - x + 1);
    };
    auto value = [&](const std::string & name, const std::string & key, const std::string & raw, ojson & args) {
        std::string type;
        const auto ti = r.arg_types.find(name);
        if (ti != r.arg_types.end()) { const auto a = ti->second.find(key); if (a != ti->second.end()) type = a->second; }
        if (type.empty() || type == "string") { args[key] = raw; return; }
        try { args[key] = ojson::parse(trim(raw)); } catch (const std::exception &) { args[key] = raw; }
    };
    for (size_t p = t0; (p = text.find("<tool_call>", p)) != std::string::npos;) {
        const size_t e = text.find("</tool_call>", p);
        if (e == std::string::npos) break;   // (an unfinished call at the end: left out)
        const std::string body = trim(text.substr(p + 11, e - p - 11));
        common_chat_tool_call tc;
        ojson args = ojson::object();
        if (body.rfind("<function=", 0) == 0) {   // Qwen
            const size_t ne = body.find('>');
            if (ne == std::string::npos) return false;
            tc.name = trim(body.substr(10, ne - 10));
            for (size_t k = body.find("<parameter=", ne); k != std::string::npos; k = body.find("<parameter=", k)) {
                const size_t ke = body.find('>', k), ve = ke == std::string::npos ? ke : body.find("</parameter>", ke);
                if (ke == std::string::npos || ve == std::string::npos) return false;
                std::string raw = body.substr(ke + 1, ve - ke - 1);
                if (!raw.empty() && raw.front() == '\n') raw.erase(0, 1);   // (the template puts the value on its own line)
                if (!raw.empty() && raw.back() == '\n') raw.pop_back();
                value(tc.name, trim(body.substr(k + 11, ke - k - 11)), raw, args);
                k = ve + 12;
            }
        } else {   // GLM
            size_t k = body.find("<arg_key>");
            tc.name = trim(body.substr(0, k));
            while (k != std::string::npos) {
                const size_t ke = body.find("</arg_key>", k), v0 = ke == std::string::npos ? ke : body.find("<arg_value>", ke);
                const size_t ve = v0 == std::string::npos ? v0 : body.find("</arg_value>", v0);
                if (ke == std::string::npos || v0 == std::string::npos || ve == std::string::npos) return false;
                value(tc.name, trim(body.substr(k + 9, ke - k - 9)), body.substr(v0 + 11, ve - v0 - 11), args);
                k = body.find("<arg_key>", ve);
            }
        }
        if (tc.name.empty()) return false;
        tc.arguments = args.dump();
        out.tool_calls.push_back(tc);
        p = e + 12;
    }
    return !out.tool_calls.empty();
}


// DeepSeek-V4.1's DSML tool calls: "<｜DSML｜ calls>" blocks of "<｜DSML｜ invoke name=..>" / "<｜DSML｜ parameter name=.. string=..>"
// (leading-space tag names, encoding/encoding.py), which libcommon's DeepSeek parser (V3.2 "function_calls", V4 "tool_calls",
// selected by a dsml_token template variable) does not know. The text before the block goes through the template's parser
// (reasoning, content); the calls are read here (partial: the complete invokes so far). false: no block in the text
bool dsml_v41_parse(const std::string & text, bool partial, const Request & r, common_chat_msg & out) {
    static const std::string D = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c", open = "<" + D + " calls>";
    const size_t think_end = text.rfind("</think>");
    const size_t from = think_end == std::string::npos ? 0 : think_end;
    size_t t0 = text.find(open, from);
    if (t0 == std::string::npos) {
        if (!partial) return false;
        // (streaming: a tail that may still become the block's opening tag is held back, not shown as content)
        const size_t lt = text.rfind('<');
        if (lt == std::string::npos || lt < from || text.size() - lt >= open.size() || open.compare(0, text.size() - lt, text, lt) != 0) return false;
        t0 = lt;
    }
    try {
        out = common_chat_parse(text.substr(0, t0), partial, r.pp);
    } catch (const std::exception &) {
        out = common_chat_msg();
        out.role = "assistant";
        out.content = text.substr(0, t0);
    }
    out.tool_calls.clear();
    const std::string inv = "<" + D + " invoke name=\"", inv_end = "</" + D + " invoke>", par = "<" + D + " parameter name=\"",
                      par_end = "</" + D + " parameter>";
    for (size_t p = text.find(inv, t0); p != std::string::npos; p = text.find(inv, p)) {
        const size_t ne = text.find("\">", p + inv.size()), e = text.find(inv_end, p);
        if (ne == std::string::npos || e == std::string::npos || ne > e) break;   // (an unfinished invoke: left out)
        common_chat_tool_call tc;
        tc.name = text.substr(p + inv.size(), ne - p - inv.size());
        ojson args = ojson::object();
        for (size_t k = text.find(par, ne); k != std::string::npos && k < e; k = text.find(par, k)) {
            const size_t ke = text.find('"', k + par.size()), te = text.find("\">", ke == std::string::npos ? k : ke);
            const size_t ve = te == std::string::npos ? te : text.find(par_end, te);
            if (ke == std::string::npos || te == std::string::npos || ve == std::string::npos || ve > e) break;
            const std::string key = text.substr(k + par.size(), ke - k - par.size());
            const bool is_str = text.compare(ke, te - ke, "\" string=\"true") == 0;
            const std::string raw = text.substr(te + 2, ve - te - 2);
            if (is_str) args[key] = raw;
            else { try { args[key] = ojson::parse(raw); } catch (const std::exception &) { args[key] = raw; } }
            k = ve + par_end.size();
        }
        tc.arguments = args.dump();
        out.tool_calls.push_back(tc);
        p = e + inv_end.size();
    }
    return true;
}

// the template's parser; its result is replaced by the fallback when it rejects the text, or (final parse) when it found
// fewer calls than the text has complete <tool_call> blocks after the reasoning
common_chat_msg parse_out(const std::string & text, bool partial, const Request & r) {
    common_chat_msg m;
    if (g_dsml_v41 && r.pp.parse_tool_calls && dsml_v41_parse(text, partial, r, m)) return m;
    try {
        m = common_chat_parse(text, partial, r.pp);
    } catch (const std::exception &) {
        common_chat_msg f;
        if (r.pp.parse_tool_calls && xml_tool_fallback(text, r, f)) return f;
        throw;
    }
    if (!partial && r.pp.parse_tool_calls) {
        const size_t think_end = text.rfind("</think>");
        size_t blocks = 0;
        for (size_t q = think_end == std::string::npos ? 0 : think_end; (q = text.find("</tool_call>", q)) != std::string::npos; ++q) ++blocks;
        common_chat_msg f;
        if (blocks > m.tool_calls.size() && xml_tool_fallback(text, r, f) && f.tool_calls.size() > m.tool_calls.size()) {
            fprintf(stderr, "request: %zu of %zu tool calls recovered by the fallback parser\n", f.tool_calls.size() - m.tool_calls.size(), blocks);
            return f;
        }
    }
    return m;
}

bool is_image_part(const ojson & p);
std::string part_text(const ojson & c);
extern const char * kImageNote;

// ---------------- images (vision, --mmproj) ----------------
std::string image_url_of(const ojson & p) {   // chat {"image_url": {"url"}}, Responses {"image_url": "..."}
    if (p.contains("image_url")) {
        const ojson & u = p["image_url"];
        if (u.is_string()) return u.get<std::string>();
        if (u.is_object() && u.contains("url") && u["url"].is_string()) return u["url"].get<std::string>();
    }
    if (p.contains("url") && p["url"].is_string()) return p["url"].get<std::string>();
    return "";
}
std::string b64_decode(const std::string & in) {
    static int8_t T[256];
    static bool init = false;
    if (!init) {
        memset(T, -1, sizeof T);
        const char * A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[(uint8_t) A[i]] = (int8_t) i;
        T[(uint8_t) '-'] = 62; T[(uint8_t) '_'] = 63;   // (url-safe alphabet too)
        init = true;
    }
    std::string out;
    out.reserve(in.size() * 3 / 4);
    unsigned v = 0; int bits = 0;
    for (unsigned char ch : in) {
        if (T[ch] < 0) continue;   // (padding, whitespace)
        v = v << 6 | (unsigned) T[ch]; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((char) (v >> bits & 0xff)); }
    }
    return out;
}
// an image's bytes: a data: URL, or downloaded (http / https)
std::string load_image(const std::string & url) {
    if (url.rfind("data:", 0) == 0) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos) throw std::invalid_argument("image: malformed data URL");
        if (url.substr(0, comma).find(";base64") == std::string::npos) throw std::invalid_argument("image: data URL is not base64");
        return b64_decode(url.substr(comma + 1));
    }
    const size_t se = url.find("://");
    if (se == std::string::npos || (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0))
        throw std::invalid_argument("image: only data: and http(s) URLs are supported");
    const size_t ps = url.find('/', se + 3);
    httplib::Client cli(url.substr(0, ps));
    cli.set_follow_location(true);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);
    auto res = cli.Get(ps == std::string::npos ? "/" : url.substr(ps));
    if (!res || res->status != 200) throw std::invalid_argument("image: cannot download " + url.substr(0, 120));
    return res->body;
}
// the prompt text with one media marker per image -> tokens, and for the engine the images' rows (encoded once per image
// content), M-RoPE positions and cache keys
void build_media(Ctx & c, const std::string & text, const std::vector<std::string> & urls, Request & r, const std::string & rid) {
    std::vector<mtmd_bitmap *> bms;
    struct Free { std::vector<mtmd_bitmap *> & v; ~Free() { for (auto * b : v) mtmd_bitmap_free(b); } } fr{bms};
    for (size_t i = 0; i < urls.size(); ++i) {
        const std::string data = load_image(urls[i]);
        const mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(c.mtmd, (const unsigned char *) data.data(), data.size(), false,
                                                                             mtmd_helper_init_opt_default());
        if (!w.bitmap) throw std::invalid_argument("image " + std::to_string(i + 1) + ": not a readable image (" + std::to_string(data.size()) + " bytes)");
        bms.push_back(w.bitmap);
    }
    std::lock_guard<std::mutex> lk(c.mtmd_mu);
    mtmd_input_text in{};
    in.text = text.c_str();
    in.text_len = text.size();
    in.add_special = true;
    in.parse_special = true;
    std::unique_ptr<mtmd_input_chunks, void (*)(mtmd_input_chunks *)> ch(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const int rc = mtmd_tokenize(c.mtmd, ch.get(), &in, (const mtmd_bitmap **) bms.data(), bms.size());
    if (rc != 0) throw std::invalid_argument("images: tokenizing the prompt failed (" + std::to_string(rc) + ")");
    auto md = std::make_shared<PromptMedia>();
    int n_past = 0, n_img = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(ch.get()); ++i) {
        const mtmd_input_chunk * k = mtmd_input_chunks_get(ch.get(), i);
        const auto type = mtmd_input_chunk_get_type(k);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(k, &n);
            for (size_t j = 0; j < n; ++j, ++n_past) {
                r.prompt.push_back(t[j]);
                md->keys.push_back(t[j]);
                md->pos3.insert(md->pos3.end(), {n_past, n_past, n_past});
            }
            continue;
        }
        if (type != MTMD_INPUT_CHUNK_TYPE_IMAGE) throw std::invalid_argument("only images are supported");
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(k);
        const size_t nt = mtmd_image_tokens_get_n_tokens(it);
        const std::string id = mtmd_input_chunk_get_id(k) ? mtmd_input_chunk_get_id(k) : std::to_string(i);
        std::shared_ptr<std::vector<float>> e;
        for (auto ci = c.img_cache.begin(); ci != c.img_cache.end(); ++ci)
            if (ci->first == id) { e = ci->second; c.img_cache.splice(c.img_cache.begin(), c.img_cache, ci); break; }
        double enc_s = 0;
        if (!e) {
            const auto t0 = clk::now();
            if (mtmd_encode_chunk(c.mtmd, k) != 0) throw std::runtime_error("image " + std::to_string(n_img + 1) + ": encoding failed");
            const float * out = mtmd_get_output_embd(c.mtmd);
            e = std::make_shared<std::vector<float>>(out, out + nt * (size_t) c.n_embd_inp);
            enc_s = secs(t0, clk::now());
            c.img_cache.push_front({id, e});
            c.img_cache_bytes += e->size() * sizeof(float);
            while (c.img_cache_bytes > ((size_t) 2 << 30) && c.img_cache.size() > 1) {
                c.img_cache_bytes -= c.img_cache.back().second->size() * sizeof(float);
                c.img_cache.pop_back();
            }
        }
        // (an image's positions all carry one key from its hash: another image in the same place is not a cache hit)
        const int key = -1 - (int) (std::strtoul(id.substr(0, 7).c_str(), nullptr, 16) & 0x0fffffff);
        for (size_t j = 0; j < nt; ++j) {
            const mtmd_decoder_pos dp = mtmd_image_tokens_get_decoder_pos(it, n_past, j);
            md->rows.push_back({(int) r.prompt.size(), e->data() + j * (size_t) c.n_embd_inp});
            r.prompt.push_back(c.image_tok);
            md->keys.push_back(key);
            md->pos3.insert(md->pos3.end(), {(int) dp.t, (int) dp.y, (int) dp.x});
        }
        n_past += mtmd_image_tokens_get_n_pos(it);
        r.media_keep.push_back(e);
        ++n_img;
        if (enc_s > 0) fprintf(stderr, "[%s] image %d: %zu tokens, encoded in %.2f s\n", rid.c_str(), n_img, nt, enc_s);
        else fprintf(stderr, "[%s] image %d: %zu tokens (encoded before)\n", rid.c_str(), n_img, nt);
    }
    md->next = n_past;
    r.media = md;
}

// tool-call arguments must be a JSON object: a call cut off mid-arguments ("{", a truncated command) becomes "{}" (not
// completed: a truncated shell command must not run as something else); the client reports the error to the model
std::string valid_args(const std::string & a, const char * where = nullptr) {
    if (ojson::accept(a)) { const ojson j = ojson::parse(a); if (j.is_object()) return a; }
    if (where) fprintf(stderr, "%s: invalid tool-call arguments replaced by {} (%zu bytes: %.40s)\n", where, a.size(), a.c_str());
    return "{}";
}

// the final parse of a non-streamed answer: a text the parser rejects (e.g. cut inside a tool call at max_tokens) is parsed
// as partial, and failing that returned as plain content -- the client gets the output instead of an error
common_chat_msg parse_final(const std::string & text, const Request & r) {
    try { return parse_out(text, false, r); }
    catch (const std::exception & e) {
        fprintf(stderr, "[%s] final parse failed (%s), answer returned as parsed so far\n", r.id.c_str(), e.what());
    }
    try {
        common_chat_msg m = parse_out(text, true, r);
        for (auto & tc : m.tool_calls) tc.arguments = valid_args(tc.arguments, "final parse");   // (a call cut off mid-arguments:
        // {}, never a truncated command)
        return m;
    } catch (const std::exception &) {}
    common_chat_msg m;
    m.role = "assistant";
    m.content = text;
    return m;
}

Request prepare(Ctx & c, const ojson & body) {
    Request r;
    common_chat_templates_inputs in;
    ojson msgs_in = body.at("messages");   // (broken tool-call arguments in the history: repaired, not rejected)
    // image parts: with the vision encoder the content becomes text with a media marker per image (in prompt order);
    // text-only, a note per image
    std::vector<std::string> images;
    const std::string marker = mtmd_default_marker();
    if (msgs_in.is_array())
        for (auto & m : msgs_in)
            if (m.is_object() && m.contains("content") && m["content"].is_array()) {
                bool img = false;
                for (auto & p : m["content"]) img |= is_image_part(p);
                if (!img) continue;
                if (!c.mtmd) { m["content"] = part_text(m["content"]); fprintf(stderr, "request: image(s) replaced by a text note (text-only)\n"); continue; }
                std::string t;
                for (auto & p : m["content"]) {
                    std::string x;
                    if (is_image_part(p)) {
                        const std::string u = image_url_of(p);
                        if (u.empty()) { x = kImageNote; }
                        else { t += marker; images.push_back(u); continue; }
                    } else if (p.is_string()) x = p.get<std::string>();
                    else if (p.is_object() && p.contains("text") && p["text"].is_string()) x = p["text"].get<std::string>();
                    for (size_t q; (q = x.find(marker)) != std::string::npos;) x.replace(q, marker.size(), "<_media_>");   // (not a marker)
                    t += x;
                }
                m["content"] = t;
            }
    if (msgs_in.is_array())
        for (auto & m : msgs_in)
            if (m.is_object() && m.contains("tool_calls") && m["tool_calls"].is_array())
                for (auto & tc : m["tool_calls"])
                    if (tc.is_object() && tc.contains("function") && tc["function"].is_object() && tc["function"].contains("arguments") &&
                        tc["function"]["arguments"].is_string())
                        tc["function"]["arguments"] = valid_args(tc["function"]["arguments"].get<std::string>(), "request history");
    in.messages = common_chat_msgs_parse_oaicompat(to_cjson(msgs_in));
    if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
        in.tools = common_chat_tools_parse_oaicompat(to_cjson(body["tools"]));
        for (const auto & t : body["tools"]) {   // argument types (the GLM tool-call fallback parser)
            if (!t.is_object() || !t.contains("function") || !t["function"].is_object()) continue;
            const auto & fn = t["function"];
            if (!fn.contains("parameters") || !fn["parameters"].is_object() || !fn["parameters"].contains("properties")) continue;
            auto & m = r.arg_types[fn.value("name", "")];
            for (const auto & [k, v] : fn["parameters"]["properties"].items()) {
                std::string ty;
                if (v.is_object() && v.contains("type")) ty = v["type"].is_string() ? v["type"].get<std::string>() : v["type"].is_array() && !v["type"].empty() ? v["type"][0].get<std::string>() : "";
                m[k] = ty;
            }
        }
    }
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
    const auto tt0 = clk::now();
    const common_chat_params cp = common_chat_templates_apply(c.tmpls.get(), in);
    const auto tt1 = clk::now();
    if (getenv("HYPER_SHOW_PROMPT")) fprintf(stderr, "request prompt (%zu images):\n%s\n---\n", images.size(), cp.prompt.c_str());
    if (images.empty()) r.prompt = tokenize_prompt(c, cp.prompt);
    else build_media(c, cp.prompt, images, r, "request");
    r.t_template = 1e3 * secs(tt0, tt1);
    r.t_tokenize = 1e3 * secs(tt1, clk::now());
    r.pp = common_chat_parser_params(cp);
    r.pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    r.pp.parse_tool_calls = !in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    if (!cp.parser.empty()) r.pp.parser.load(cp.parser);
    r.stops = cp.additional_stops;
    if (!r.stops.empty()) { std::string l; for (auto & x : r.stops) l += ojson(x).dump() + " "; fprintf(stderr, "request: template stop strings %s\n", l.c_str()); }
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
    ~Streamer() { finish(); }   // (an engine exception unwinds through run(): a joinable thread here would std::terminate)
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
    fprintf(stderr, "[%s] prompt: %d tokens, max %d new (request: parse %.0f, convert %.0f, template %.0f, tokenize %.0f ms)\n", r.id.c_str(), P,
            r.max_tokens, r.t_parse, r.t_convert, r.t_template, r.t_tokenize);
    if (const char * dt = getenv("HYPER_DUMP_TOKENS")) {   // the prompt as a token-only reference file (v2, no logits) for the CLI tools
        const std::string fn = std::string(dt) + "-" + r.id + ".bin";
        if (FILE * f = fopen(fn.c_str(), "wb")) {
            const int hdr[4] = {0x32464552, P, llama_vocab_n_tokens(c.vocab), P};
            fwrite(hdr, 4, 4, f);
            fwrite(r.prompt.data(), 4, r.prompt.size(), f);
            fclose(f);
        }
    }
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
    if (!c.eng->set_media(r.media.get())) throw std::invalid_argument("this model takes no images");
    c.eng->generate(r.prompt, r.max_tokens, c.eng->n_draft() > 0, &st, [&](int tok) {
        const auto now = clk::now();
        if (first) {
            t1 = now; first = false; t_log = now;
            std::lock_guard<std::mutex> lk(lv.mu);
            lv.phase = "generating"; lv.t_gen = now; lv.t_win = now; lv.win_n = 0;
        }
        if (llama_vocab_is_eog(c.vocab, tok)) {
            size_t open = 0, closed = 0;   // (an end-of-generation token inside a tool call: logged)
            for (size_t q = 0; (q = o.text.find("<tool_call>", q)) != std::string::npos; ++q) ++open;
            for (size_t q = 0; (q = o.text.find("</tool_call>", q)) != std::string::npos; ++q) ++closed;
            if (open > closed) fprintf(stderr, "[%s] end-of-generation token %d inside a tool call\n", r.id.c_str(), tok);
            o.finish = "stop"; return false;
        }
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
            if (at != std::string::npos) {
                fprintf(stderr, "[%s] stop string %s at %zu of %zu\n", r.id.c_str(), ojson(s).dump().c_str(), at, o.text.size());
                o.text.resize(at); o.finish = "stop"; return false;
            }
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
    fprintf(stderr, "[%s]       total time = %10.2f ms / %6d tokens, finish: %s, draft acceptance %.2f / %d per step (%.2f drafted)\n",
            r.id.c_str(), 1e3 * secs(t0, t2), n_new + o.n_gen, o.finish.c_str(), o.steps ? (double) o.accepted / o.steps : 0.0,
            c.eng->n_draft(), o.steps ? (double) st.drafted / o.steps : 0.0);
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
// the engines are text-only: an image becomes this note (instead of vanishing, after which models describe what they
// "saw")
const char * kImageNote = "[image not shown: this model server cannot see images]";
bool is_image_part(const ojson & p) {
    if (!p.is_object()) return false;
    const std::string ty = p.value("type", "");
    return ty == "input_image" || ty == "image_url" || ty == "image" || (ty == "input_file" && p.contains("image_url"));
}
std::string part_text(const ojson & c) {   // string, or an array of content parts (input_text / output_text / text / ...)
    if (c.is_string()) return c.get<std::string>();
    std::string t;
    if (c.is_array())
        for (auto & p : c) {
            if (p.is_string()) t += p.get<std::string>();
            else if (is_image_part(p)) { if (!t.empty() && t.back() != '\n') t += "\n"; t += kImageNote; t += "\n"; }
            else if (p.is_object() && p.contains("text") && p["text"].is_string()) t += p["text"].get<std::string>();
        }
    return t;
}

// Responses request -> chat completions body; custom (freeform) tools become functions with one string argument "input"
// Responses tools as chat functions. A namespace (Codex: one per MCP server, e.g. mcp__windows_vm) becomes its functions under
// "<namespace>__<name>", its description goes into the system message; a call of such a function goes back with the
// namespace and the plain name (Codex resolves calls by both)
struct RespTools {
    std::set<std::string> custom;                                       // freeform tools (flat names): one string argument "input"
    std::map<std::string, std::pair<std::string, std::string>> ns;     // flat name -> (namespace, name)
};
std::string jstr(const ojson & j, const char * k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); }
std::string flat_name(const ojson & it) { const std::string ns = jstr(it, "namespace"), n = jstr(it, "name"); return ns.empty() ? n : ns + "__" + n; }

ojson responses_to_chat(const ojson & body, RespTools & rt) {
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
                bool img = false;   // (images of a user / tool message: kept as parts, the chat path sees them)
                if (g_vision && it.contains("content") && it["content"].is_array()) for (auto & p : it["content"]) img |= is_image_part(p);
                if (img && role != "system" && role != "developer" && role != "assistant") {
                    if (!msgs.empty() && msgs.back()["role"] == "assistant") msgs.back()["closed"] = true;
                    ojson parts = ojson::array();
                    for (auto & p : it["content"]) {
                        if (is_image_part(p)) parts.push_back({{"type", "image_url"}, {"image_url", {{"url", image_url_of(p)}}}});
                        else if (p.is_object() && p.contains("text") && p["text"].is_string()) parts.push_back({{"type", "text"}, {"text", p["text"]}});
                    }
                    msgs.push_back({{"role", role}, {"content", parts}});
                    continue;
                }
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
                add_call(it.value("call_id", ""), flat_name(it), valid_args(it.value("arguments", "{}"), "responses history"));
            } else if (type == "custom_tool_call") {
                add_call(it.value("call_id", ""), flat_name(it), ojson({{"input", it.value("input", "")}}).dump());
            } else if (type == "function_call_output" || type == "custom_tool_call_output") {
                if (!msgs.empty() && msgs.back()["role"] == "assistant") msgs.back()["closed"] = true;
                if (g_vision && it.contains("output") && it["output"].is_array()) {   // (a tool's output with images, e.g. a screenshot)
                    bool img = false;
                    for (auto & p : it["output"]) img |= is_image_part(p);
                    if (img) {
                        ojson parts = ojson::array();
                        for (auto & p : it["output"]) {
                            if (is_image_part(p)) parts.push_back({{"type", "image_url"}, {"image_url", {{"url", image_url_of(p)}}}});
                            else if (p.is_object() && p.contains("text") && p["text"].is_string()) parts.push_back({{"type", "text"}, {"text", p["text"]}});
                        }
                        msgs.push_back({{"role", "tool"}, {"tool_call_id", it.value("call_id", "")}, {"content", parts}});
                        continue;
                    }
                }
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
    std::string groups;   // (the namespaces' descriptions)
    auto add_tool = [&](const ojson & t, const std::string & ns) {
        const std::string type = jstr(t, "type"), name = jstr(t, "name"), flat = ns.empty() ? name : ns + "__" + name;
        if (!ns.empty() && (type == "function" || type == "custom")) rt.ns[flat] = {ns, name};
        if (type == "function") {
            ojson fn = {{"name", flat}, {"description", jstr(t, "description")}};
            fn["parameters"] = t.contains("parameters") && t["parameters"].is_object() ? t["parameters"] : ojson({{"type", "object"}, {"properties", ojson::object()}});
            tools.push_back({{"type", "function"}, {"function", fn}});
        } else if (type == "custom") {
            rt.custom.insert(flat);
            std::string desc = jstr(t, "description");
            if (t.contains("format") && t["format"].is_object() && t["format"].contains("definition"))
                desc += "\n\nThe input must follow this grammar:\n" + t["format"].value("definition", "");
            tools.push_back({{"type", "function"}, {"function", {{"name", flat}, {"description", desc},
                {"parameters", {{"type", "object"}, {"properties", {{"input", {{"type", "string"}, {"description", "the raw tool input"}}}}},
                                {"required", ojson::array({"input"})}}}}}});
        }
    };
    if (body.contains("tools") && body["tools"].is_array())
        for (const auto & t : body["tools"]) {
            if (jstr(t, "type") == "namespace" && t.contains("tools") && t["tools"].is_array()) {
                const std::string ns = jstr(t, "name"), desc = jstr(t, "description");
                if (!desc.empty()) groups += "\n\n## Tools " + ns + "__*\n" + desc;
                for (const auto & x : t["tools"]) add_tool(x, ns);
            } else add_tool(t, "");
        }
    if (!groups.empty()) {   // (what the namespaces say about their tools: how to use them)
        ojson & m = chat["messages"];
        const std::string g = "# Tool groups" + groups;
        if (!m.empty() && m[0]["role"] == "system") m[0]["content"] = m[0]["content"].get<std::string>() + "\n\n" + g;
        else m.insert(m.begin(), ojson({{"role", "system"}, {"content", g}}));
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
ojson tool_call_item(const common_chat_tool_call & t, const std::string & item_id, const RespTools & rt, bool done) {
    const auto ns = rt.ns.find(t.name);   // (a namespaced tool: its namespace and plain name)
    ojson it;
    if (rt.custom.count(t.name)) {
        std::string input = t.arguments;
        try { const ojson a = ojson::parse(t.arguments); if (a.contains("input") && a["input"].is_string()) input = a["input"].get<std::string>(); } catch (...) {}
        it = {{"type", "custom_tool_call"}, {"id", item_id}, {"call_id", t.id}, {"name", t.name}, {"input", done ? input : ""},
              {"status", done ? "completed" : "in_progress"}};
    } else {
        it = {{"type", "function_call"}, {"id", item_id}, {"call_id", t.id}, {"name", t.name},
              {"arguments", done ? valid_args(t.arguments, "responses output") : ""}, {"status", done ? "completed" : "in_progress"}};
    }
    if (ns != rt.ns.end()) { it["name"] = ns->second.second; it["namespace"] = ns->second.first; }
    return it;
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
    // SIGINT / SIGTERM: a thread waits for them (blocked in every thread, also when the shell started us as a background job with
    // SIGINT ignored) and stops the HTTP server, so main returns and the engine is torn down in order (GPU memory, pinned
    // buffers, the GGUF mapping) instead of only by the kernel at exit; a second signal exits at once
    sigset_t stop_sigs;
    sigemptyset(&stop_sigs);
    sigaddset(&stop_sigs, SIGINT);
    sigaddset(&stop_sigs, SIGTERM);
    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    pthread_sigmask(SIG_BLOCK, &stop_sigs, nullptr);
    std::string host = "0.0.0.0", alias;
    int port = 8080, ctx = 262144, draft = 3, snaps = 48, cpu_threads = 30;
    float gpu_frac = 1.0f;
    std::string stats_path;   // qwen4exp: routing statistics, read at start and updated after every request
    std::string mtp_path;     // qwen4exp: separate NextN (MTP) GGUF; glm5-next: a GGUF with blk.<n_layer> (the original split files)
    std::string template_file;   // chat template (jinja) instead of the GGUF's
    std::string mmproj_path;     // vision encoder (mmproj GGUF): images in prompts
    std::string vocab_path;      // vocabulary / chat template from another GGUF (a model libllama does not know: deepseek41)
    std::vector<std::pair<std::string, std::string>> mcps;   // --mcp name=url: tool servers for the chat page
    bool ctx_given = false, temp_given = false, top_p_given = false, top_k_given = false;
    SamplingParams defaults;   // Qwen's recommendation for thinking mode
    defaults.temp = 0.6f; defaults.top_p = 0.95f; defaults.top_k = 20; defaults.min_p = 0.0f;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--host") host = v; else if (k == "--port") port = std::stoi(v); else if (k == "--ctx") { ctx = std::stoi(v); ctx_given = true; }
        else if (k == "--gpu-frac") gpu_frac = std::stof(v); else if (k == "--cpu-threads") cpu_threads = std::stoi(v);
        else if (k == "--expert-stats") { setenv("HYPER4_STATS", v.c_str(), 1); stats_path = v; }
        else if (k == "--mtp") mtp_path = v;
        else if (k == "--draft") draft = std::stoi(v); else if (k == "--alias") alias = v;
        else if (k == "--temp") { defaults.temp = std::stof(v); temp_given = true; } else if (k == "--top-p") { defaults.top_p = std::stof(v); top_p_given = true; }
        else if (k == "--top-k") { defaults.top_k = std::stoi(v); top_k_given = true; } else if (k == "--min-p") defaults.min_p = std::stof(v);
        else if (k == "--snapshots") snaps = std::stoi(v);
        else if (k == "--mcp") {
            const size_t eq = v.find('=');
            if (eq == std::string::npos || eq == 0) { fprintf(stderr, "--mcp expects name=url\n"); return 1; }
            mcps.push_back({v.substr(0, eq), v.substr(eq + 1)});
        }
        else if (k == "--reasoning-effort") g_reasoning_effort = v;
        else if (k == "--chat-template-file") template_file = v;
        else if (k == "--mmproj") mmproj_path = v;
        else if (k == "--vocab") vocab_path = v;
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
    c.vm = llama_model_load_from_file((vocab_path.empty() ? path : vocab_path).c_str(), mp);
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
    {   // DeepSeek-V4.1 DSML (leading-space tag names): parsed here, see dsml_v41_parse
        const std::string src = common_chat_templates_source(c.tmpls.get());
        g_dsml_v41 = src.find("\xef\xbd\x9c" "DSML" "\xef\xbd\x9c") != std::string::npos && src.find(" calls>") != std::string::npos;
        if (g_dsml_v41) fprintf(stderr, "hyper-server: DeepSeek-V4.1 DSML tool calls\n");
    }

    if (mmproj_path.empty() && getenv("HYPER_MMPROJ")) mmproj_path = getenv("HYPER_MMPROJ");   // (tests)
    if (!mmproj_path.empty()) {   // before the engine: its expert placement sees the encoder's VRAM as taken
        mtmd_context_params mpar = mtmd_context_params_default();
        mpar.use_gpu = !getenv("HYPER_MMPROJ_CPU");
        mpar.n_threads = 8;
        mpar.print_timings = false;
        mpar.warmup = true;
        c.mtmd = mtmd_init_from_file(mmproj_path.c_str(), c.vm, mpar);
        if (!c.mtmd || !mtmd_support_vision(c.mtmd)) { fprintf(stderr, "cannot load the vision encoder %s\n", mmproj_path.c_str()); return 1; }
        const std::vector<int> it = tokenize(c.vocab, "<|image_pad|>");
        c.image_tok = it.size() == 1 ? it[0] : -1;
        {   // the encoder's rows are the model's input rows (the vocabulary-only model knows no embedding size)
            GGUF gp(mmproj_path), gm(path);
            c.n_embd_inp = (int) gp.get_int("clip.vision.projection_dim", 0);
            const int n_model = (int) gm.get_int(gm.arch() + ".embedding_length", 0);
            if (c.n_embd_inp <= 0 || c.n_embd_inp != n_model) {
                fprintf(stderr, "the vision encoder's output (%d) does not fit the model's input rows (%d)\n", c.n_embd_inp, n_model);
                return 1;
            }
        }
        if (c.image_tok < 0) { fprintf(stderr, "the vocabulary has no <|image_pad|> token\n"); return 1; }
        g_vision = true;
        fprintf(stderr, "hyper-server: vision encoder %s (%s, rows of %d, image token %d, M-RoPE %s)\n", mmproj_path.c_str(), mpar.use_gpu ? "GPU" : "CPU",
                c.n_embd_inp, c.image_tok, mtmd_decode_use_mrope(c.mtmd) ? "yes" : "no");
    }
    std::string arch;
    { GGUF g(path); arch = g.arch(); }
    // HYPER_ANSWER_REUSE=0/1: a request that repeats the previous answer resumes after it (its K/V and state as decoding computed
    // them) instead of re-prefilling it from the message start. Default on for the 27B; off for the MoE models, whose decode
    // runs the CPU-held experts on 8-bit activations (prefill streams them to the GPUs: the more precise K/V)
    auto answer_reuse = [](bool def) { return getenv("HYPER_ANSWER_REUSE") ? atoi(getenv("HYPER_ANSWER_REUSE")) != 0 : def; };
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
        o4.answer_reuse = answer_reuse(false);
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
        // GLM's recommended sampling: temperature 1.0, top_p 0.95, no top_k (the defaults above are Qwen's)
        if (!temp_given) c.defaults.temp = 1.0f;
        if (!top_p_given) c.defaults.top_p = 0.95f;
        if (!top_k_given) c.defaults.top_k = 0;
        Engine5Options o5;
        o5.max_pos = ctx;
        o5.gpu_expert_frac = gpu_frac;
        o5.cpu_threads = cpu_threads;
        o5.prompt_cache = true;
        o5.max_snapshots = snaps;
        o5.answer_reuse = answer_reuse(false);
        o5.n_draft = draft;   // MTP drafts with --mtp, else prompt-lookup speculation
        o5.mtp_path = mtp_path;
        auto e5 = std::make_unique<Engine5>(path, o5);
        if (!stats_path.empty()) {
            Engine5 * pe = e5.get();
            c.after_request = [pe, stats_path] { pe->save_expert_stats(stats_path); };
        }
        draft = e5->n_draft();
        engine = std::move(e5);
    } else if (arch == "deepseek41") {
        if (!ctx_given) ctx = 65536;
        // DeepSeek's recommended sampling: temperature 1.0, top_p 1.0
        if (!temp_given) c.defaults.temp = 1.0f;
        if (!top_p_given) c.defaults.top_p = 1.0f;
        if (!top_k_given) c.defaults.top_k = 0;
        Engine7Options o7;
        o7.max_pos = ctx;
        o7.gpu_expert_frac = gpu_frac;
        o7.cpu_threads = cpu_threads;
        o7.prompt_cache = true;
        o7.n_draft = draft;   // prompt-lookup speculation
        auto e7 = std::make_unique<Engine7>(path, o7);
        if (!stats_path.empty()) {
            Engine7 * pe = e7.get();
            c.after_request = [pe, stats_path] { pe->save_expert_stats(stats_path); };
        }
        draft = e7->n_draft();
        engine = std::move(e7);
    } else if (arch == "mimo2") {
        if (!ctx_given) ctx = 65536;
        // MiMo's recommended sampling: temperature 1.0, top_p 0.95
        if (!temp_given) c.defaults.temp = 1.0f;
        if (!top_p_given) c.defaults.top_p = 0.95f;
        if (!top_k_given) c.defaults.top_k = 0;
        Engine6Options o6;
        o6.max_pos = ctx;
        o6.gpu_expert_frac = gpu_frac;
        o6.cpu_threads = cpu_threads;
        o6.prompt_cache = true;
        o6.max_snapshots = snaps;
        o6.answer_reuse = answer_reuse(false);
        o6.n_draft = draft;   // MTP drafts with --mtp (the mtp-*.gguf sidecar), else prompt-lookup speculation
        o6.mtp_path = mtp_path;
        auto e6 = std::make_unique<Engine6>(path, o6);
        if (!stats_path.empty()) {
            Engine6 * pe = e6.get();
            c.after_request = [pe, stats_path] { pe->save_expert_stats(stats_path); };
        }
        draft = e6->n_draft();
        engine = std::move(e6);
    } else {
        EngineOptions opt;
        opt.max_pos = ctx;
        opt.n_draft = draft;
        opt.prompt_cache = true;
        opt.max_snapshots = snaps;
        opt.answer_reuse = answer_reuse(true);
        engine = std::make_unique<Engine>(path, opt);
    }
    c.live.ctx_max = ctx;
    LLM & eng = *engine;
    c.eng = &eng;
    if (c.mtmd) {   // (an engine that takes no images: the encoder is of no use)
        PromptMedia probe;
        if (!eng.set_media(&probe)) { fprintf(stderr, "hyper-server: this model takes no images, --mmproj ignored\n"); mtmd_free(c.mtmd); c.mtmd = nullptr; g_vision = false; }
        else eng.set_media(nullptr);
    }
    {
        int n_snap = 0;   // message starts: ChatML, or GLM's role tokens (an engine keeps every token it is given)
        for (const char * m : {"<|im_start|>", "<|assistant|>", "<|user|>", "<|observation|>"}) {
            const std::vector<int> im = tokenize(c.vocab, m);
            if (im.size() == 1 && (n_snap == 0 || arch == "glm5-next")) { eng.set_snapshot_token(im[0]); ++n_snap; }
        }
        if (!n_snap) fprintf(stderr, "hyper-server: no message-start token, prompt-cache snapshots only every 4096 tokens\n");
    }
    {   // warm up: builds the CUDA graphs; then (unless HYPER_NO_WARMUP) a prompt of a few prefill chunks with message starts,
        // so that the first real request does not pay the one-time costs (pinned snapshot buffers allocated under memory
        // pressure, first cuBLAS calls, the CPU experts' work buffers): the first cold prompt after a start ran 2-4x slower
        GenStats st;
        eng.generate(tokenize(c.vocab, "Hello"), 4, eng.n_draft() > 0, &st);
        if (!getenv("HYPER_NO_WARMUP")) {
            const auto t0 = clk::now();
            std::string text;
            for (int i = 0; text.size() < 24000; ++i)
                text += "<|im_start|>user\nParagraph " + std::to_string(i) + ": the quick brown fox jumps over the lazy dog, again and again.<|im_end|>\n";
            std::vector<int> wp = tokenize(c.vocab, text);
            wp.resize(std::min<size_t>(wp.size(), (size_t) std::max(64, std::min(eng.max_pos() / 2, 5000))));
            eng.generate(wp, 16, eng.n_draft() > 0, &st);
            eng.reset_cache();
            fprintf(stderr, "hyper-server: warm-up (%zu tokens) in %.1f s\n", wp.size(), secs(t0, clk::now()));
        }
    }

    httplib::Server srv;
    // (an agent's request carries every screenshot of its session as base64: 85 images were over httplib's 100 MB default)
    srv.set_payload_max_length((size_t) 1 << 30);
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
        ojson m = {{"id", c.alias}, {"object", "model"}, {"created", 0}, {"owned_by", "hyper"}, {"meta", {{"n_ctx_train", ctx}, {"vision", c.mtmd != nullptr}}}};
        res.set_content(ojson({{"object", "list"}, {"data", ojson::array({m})}}).dump(), "application/json");
    };
    srv.Get("/v1/models", models);
    srv.Get("/props", [&](const httplib::Request &, httplib::Response & res) {   // (web UI: the server's sampling defaults)
        cors(res);
        res.set_content(ojson({{"model", c.alias}, {"sampling", {{"temp", c.defaults.temp}, {"top_p", c.defaults.top_p}, {"top_k", c.defaults.top_k},
                                                                  {"min_p", c.defaults.min_p}}}}).dump(), "application/json");
    });
    srv.Get("/stats", [&](const httplib::Request &, httplib::Response & res) { cors(res); res.set_content(c.live.json().dump(), "application/json"); });
    srv.Get("/", [&](const httplib::Request &, httplib::Response & res) { res.set_content(CHAT_PAGE, "text/html; charset=utf-8"); });
    srv.Get("/live", [&](const httplib::Request &, httplib::Response & res) { res.set_content(STATS_PAGE, "text/html; charset=utf-8"); });
    // MCP tool servers for the chat page (--mcp name=url, streamable HTTP): the page runs the tool loop itself; hyper only
    // lists the configured servers and forwards the page's JSON-RPC requests (same origin: the tool servers need no CORS
    // and can listen on localhost)
    srv.Get("/v1/mcp", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        ojson a = ojson::array();
        for (const auto & m : mcps) a.push_back({{"name", m.first}});
        res.set_content(ojson({{"servers", a}}).dump(), "application/json");
    });
    srv.Post(R"(/mcp/([A-Za-z0-9_.-]+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string name = req.matches[1];
        const auto it = std::find_if(mcps.begin(), mcps.end(), [&](const auto & m) { return m.first == name; });
        if (it == mcps.end()) { res.status = 404; res.set_content(R"({"error":"unknown MCP server"})", "application/json"); return; }
        const std::string & url = it->second;
        const size_t sch = url.find("://"), slash = url.find('/', sch == std::string::npos ? 0 : sch + 3);
        const std::string base = slash == std::string::npos ? url : url.substr(0, slash), path = slash == std::string::npos ? "/" : url.substr(slash);
        httplib::Client cli(base);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(240);   // (a page behind bot protection opens in a browser: tens of seconds)
        httplib::Headers h;
        for (const char * k : {"Mcp-Session-Id", "MCP-Protocol-Version", "Last-Event-ID"})
            if (req.has_header(k)) h.emplace(k, req.get_header_value(k));
        h.emplace("Accept", req.has_header("Accept") ? req.get_header_value("Accept") : "application/json, text/event-stream");
        const std::string ct = req.get_header_value("Content-Type");
        auto r = cli.Post(path, h, req.body, ct.empty() ? "application/json" : ct);
        if (!r) {
            res.status = 502;
            res.set_content(ojson({{"error", "MCP server " + name + " unreachable (" + httplib::to_string(r.error()) + ")"}}).dump(), "application/json");
            return;
        }
        res.status = r->status;
        if (r->has_header("Mcp-Session-Id")) res.set_header("Mcp-Session-Id", r->get_header_value("Mcp-Session-Id"));
        const std::string rct = r->get_header_value("Content-Type");
        res.set_content(r->body, rct.empty() ? "application/json" : rct);
    });
    srv.Get("/models", models);

    auto chat = [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        auto fail = [&](int code, const std::string & msg) {
            res.status = code;
            res.set_content(ojson({{"error", {{"message", msg}, {"type", code == 400 ? "invalid_request_error" : "server_error"}}}}).dump(), "application/json");
        };
        std::shared_ptr<Request> r;
        try {
            const auto tp0 = clk::now();
            const ojson body = ojson::parse(req.body);
            const auto tp1 = clk::now();
            if (const char * dump = getenv("HYPER_DUMP_REQUEST")) {   // request bodies <dir>/<time>.json (replay for benchmarks)
                const std::string fn = std::string(dump) + "/" + std::to_string((long long) time(nullptr)) + ".json";
                FILE * f = fopen(fn.c_str(), "wb");
                if (f) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
            }
            r = std::make_shared<Request>(prepare(c, body));
            r->t_parse = 1e3 * secs(tp0, tp1);
            r->t_convert = 1e3 * secs(tp1, clk::now()) - r->t_template - r->t_tokenize;
        } catch (const std::exception & e) { fail(400, e.what()); return; }
        const std::string id = random_id("chatcmpl-", c.next_id++);
        r->id = id.substr(9, 6) + id.substr(id.size() - 4);
        const long created = (long) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        if (!r->stream) {
            std::lock_guard<std::mutex> lk(c.mu);
            GenOut o;
            try { o = run(c, *r, {}); } catch (const std::exception & e) { fail(500, e.what()); return; }
            common_chat_msg msg = parse_final(o.text, *r);
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
                auto final_fail = [&](const char * what, const std::exception & e) {
                    if (!partial)
                        fprintf(stderr, "[%s] stream: final %s failed (%s); text tail: %s\n", r->id.c_str(), what, e.what(),
                                text.substr(text.size() > 300 ? text.size() - 300 : 0).c_str());
                    return true;
                };
                try { msg = parse_out(text, partial, *r); } catch (const std::exception & e) { return final_fail("parse", e); }
                msg.set_tool_call_ids(ids, gen_id);
                // the arguments of the call still being generated (the last one) go out once it is complete: a partial
                // parse can render them in a way the finished call does not extend (no later diff applies, and the client
                // keeps a fragment such as "{"). Earlier calls are finished (compute_diffs only extends the last call)
                if (partial && !msg.tool_calls.empty()) msg.tool_calls.back().arguments.clear();
                // a partial parse can briefly see fewer tool calls than the previous one (a call whose arguments are
                // still just "{"): compute_diffs throws then; skip this snapshot, a later one (or the final parse) catches up
                std::vector<common_chat_msg_diff> diffs;
                try { diffs = common_chat_msg_diff::compute_diffs(prev, msg); } catch (const std::exception & e) { return final_fail("diff", e); }
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
        auto custom = std::make_shared<RespTools>();
        ojson rbody;
        try {
            const auto tp0 = clk::now();
            rbody = ojson::parse(req.body);
            const auto tp1 = clk::now();
            if (const char * dump = getenv("HYPER_DUMP_REQUEST")) {
                const std::string fn = std::string(dump) + "/" + std::to_string((long long) time(nullptr)) + "-responses.json";
                FILE * f = fopen(fn.c_str(), "wb");
                if (f) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
            }
            ojson chat_body = responses_to_chat(rbody, *custom);
            const auto tp2 = clk::now();
            r = std::make_shared<Request>(prepare(c, chat_body));
            r->t_parse = 1e3 * secs(tp0, tp1);
            r->t_convert = 1e3 * secs(tp1, tp2) + 1e3 * secs(tp2, clk::now()) - r->t_template - r->t_tokenize;
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
            common_chat_msg msg = parse_final(o.text, *r);
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
                auto final_fail = [&](const char * what, const std::exception & e) {
                    if (!partial)
                        fprintf(stderr, "[%s] stream: final %s failed (%s); text tail: %s\n", r->id.c_str(), what, e.what(),
                                text.substr(text.size() > 300 ? text.size() - 300 : 0).c_str());
                    return true;
                };
                try { msg = parse_out(text, partial, *r); } catch (const std::exception & e) { return final_fail("parse", e); }
                msg.set_tool_call_ids(ids, gen_id);
                // the arguments of the call still being generated (the last one) go out once it is complete: a partial
                // parse can render them in a way the finished call does not extend (no later diff applies, and the client
                // keeps a fragment such as "{"). Earlier calls are finished (compute_diffs only extends the last call)
                if (partial && !msg.tool_calls.empty()) msg.tool_calls.back().arguments.clear();
                // a partial parse can briefly see fewer tool calls than the previous one (a call whose arguments are
                // still just "{"): compute_diffs throws then; skip this snapshot, a later one (or the final parse) catches up
                std::vector<common_chat_msg_diff> diffs;
                try { diffs = common_chat_msg_diff::compute_diffs(prev, msg); } catch (const std::exception & e) { return final_fail("diff", e); }
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
                        if (!d.tool_call_delta.arguments.empty() && !custom->custom.count(msg.tool_calls[open_tool].name))
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

    std::thread([&srv, stop_sigs] {
        int sig = 0;
        sigwait(&stop_sigs, &sig);
        fprintf(stderr, "hyper-server: signal %d, stopping (again: exit at once)\n", sig);
        srv.stop();
        sigwait(&stop_sigs, &sig);
        _exit(128 + sig);
    }).detach();
    fprintf(stderr, "hyper-server: listening on http://%s:%d (model \"%s\", context %d, %d drafts)\n", host.c_str(), port,
            c.alias.c_str(), ctx, draft);
    if (!srv.listen(host, port)) { fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 1; }
    fprintf(stderr, "hyper-server: stopped\n");
    return 0;
}
