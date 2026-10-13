# MCP server (streamable HTTP, stateless, JSON responses) with web tools for hyper's chat:
#   web_search(query)  -- SearXNG metasearch (Google, Bing, DuckDuckGo, Brave, Wikipedia, ...)
#   fetch_url(url)     -- a page's main text as markdown (product pages: schema.org price / availability / parameters
#                         first). Plain HTTP first; a real browser (Google Chrome over CDP,
#                         SeleniumBase UC mode, headful in Xvfb) for bot challenges (Cloudflare, Turnstile, ...) and
#                         JavaScript-only pages, or when asked (browser=True)
#   browser_open / click / type / select / scroll / back / read / find
#                      -- an interactive browser kept open between calls: the page as text with numbered elements [n]
import asyncio
import base64
import html as htmllib
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx
import trafilatura
from mcp.server.fastmcp import FastMCP

SEARX = os.environ.get("SEARXNG_URL", "http://searxng:8080")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "Accept-Language": "en-US,en;q=0.7,cs;q=0.5", "Upgrade-Insecure-Requests": "1"}
# markers of a challenge / block page itself (not of pages that merely embed a Turnstile widget or a bot-check script)
CHALLENGE = re.compile(r"<title>\s*(Just a moment|Attention Required|Access denied|DDoS-Guard)|Performing security verification|"
                       r"Checking your browser before accessing|cf_chl_opt|cf-chl-widget|Verify you are human by completing|"
                       r"captcha-delivery\.com|px-captcha|_Incapsula_Resource|Please enable JS and disable any ad blocker|"
                       r"Access to this page has been denied", re.I)
mcp = FastMCP("web", host="0.0.0.0", port=8890, stateless_http=True, json_response=True)

# every browser call on one thread: SeleniumBase's CDP mode, and the Turnstile click goes through the shared virtual display
_ON = ThreadPoolExecutor(max_workers=1)


async def on_browser(fn, *args, timeout=90):
    return await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(_ON, fn, *args), timeout=timeout)


def _pass_challenge(sb) -> bool:
    """Bot challenges clear themselves or get their Turnstile box clicked like a person would."""
    for _ in range(4):
        if not CHALLENGE.search(sb.get_page_source()[:200000]):
            return True
        try:
            sb.uc_gui_click_captcha()
        except Exception:
            pass
        sb.sleep(3)
    return not CHALLENGE.search(sb.get_page_source()[:200000])


def _browser_fetch(url: str) -> tuple[str, str]:
    """Google Chrome over CDP (SeleniumBase UC / CDP mode, headful in Xvfb): bot challenges clear themselves or get their
    Turnstile box clicked like a person would."""
    from seleniumbase import SB
    with SB(uc=True, xvfb=True, locale="en") as sb:
        sb.activate_cdp_mode(url)
        sb.sleep(3)
        _pass_challenge(sb)
        sb.sleep(1)
        return sb.get_current_url(), sb.get_page_source()


async def fetch_browser(url: str) -> tuple[str, str]:
    return await on_browser(_browser_fetch, url)


LD_JSON = re.compile(r"<script[^>]*application/ld\+json[^>]*>(.*?)</script>", re.S | re.I)


def _ld_nodes(node):
    """schema.org objects of a JSON-LD block (lists and @graph flattened)"""
    if isinstance(node, list):
        for x in node:
            yield from _ld_nodes(x)
    elif isinstance(node, dict):
        yield node
        yield from _ld_nodes(node.get("@graph", []))


def _ld_str(v) -> str:
    if isinstance(v, dict):
        v = v.get("name", "")
    return htmllib.unescape(str(v)).replace("\xa0", " ").strip()


def _offer(o: dict) -> str:
    price, cur = o.get("price"), o.get("priceCurrency", "")
    spec = o.get("priceSpecification")
    for s in (spec if isinstance(spec, list) else [spec] if isinstance(spec, dict) else []):
        price, cur = price if price is not None else s.get("price"), cur or s.get("priceCurrency", "")
    if price is None and o.get("lowPrice") is not None:   # AggregateOffer (marketplaces, price comparison sites)
        price = f"{o['lowPrice']}–{o.get('highPrice', '?')}"
        if o.get("offerCount"):
            cur = f"{cur} ({o['offerCount']} offers)"
    parts = [f"{price} {cur}".strip() if price is not None else "price not given"]
    for k in ("availability", "itemCondition"):
        if o.get(k):
            parts.append(str(o[k]).rsplit("/", 1)[-1])
    if o.get("seller"):
        parts.append("seller " + _ld_str(o["seller"]))
    return ", ".join(parts)


def product_data(html: str) -> str:
    """schema.org Product data from the page's JSON-LD (name, price, availability, rating, parameters): e-shops show
    prices outside the main text that trafilatura keeps."""
    out = []
    for m in LD_JSON.finditer(html):
        try:
            data = json.loads(m.group(1), strict=False)
        except ValueError:
            continue
        for d in _ld_nodes(data):
            t = d.get("@type", "")
            if "Product" not in (t if isinstance(t, list) else [t]) or len(out) >= 3:
                continue
            ids = ", ".join(f"{k} {_ld_str(d[k])}" for k in ("brand", "sku", "mpn", "gtin13") if d.get(k))
            lines = [f"Product: {_ld_str(d.get('name', '?'))}" + (f" ({ids})" if ids else "")]
            offers = d.get("offers")
            for o in (offers if isinstance(offers, list) else [offers] if isinstance(offers, dict) else [])[:5]:
                lines.append("Offer: " + _offer(o))
            r = d.get("aggregateRating")
            if isinstance(r, dict) and r.get("ratingValue") is not None:
                lines.append(f"Rating: {r['ratingValue']} ({r.get('reviewCount') or r.get('ratingCount') or '?'} reviews)")
            props = [p for p in d.get("additionalProperty", []) if isinstance(p, dict) and p.get("name")][:40]
            if props:
                lines.append("Parameters: " + "; ".join(f"{_ld_str(p['name'])}: {_ld_str(p.get('value', ''))}" for p in props))
            out.append("\n".join(lines))
    return "\n\n".join(out)


def extract(html: str, url: str) -> str:
    text = trafilatura.extract(html, output_format="markdown", include_links=True, include_tables=True, url=url)
    if not text:
        text = trafilatura.extract(html, favor_recall=True, url=url) or ""
    data = product_data(html)
    return f"[structured product data]\n{data}\n\n[page text]\n{text}" if data else text


def _bing_url(href: str) -> str:
    """Bing's click-tracking links (bing.com/ck/a?...&u=a1<base64url>) to the target URL."""
    if "bing.com/ck/a" not in href:
        return href
    u = parse_qs(urlparse(href).query).get("u", [""])[0]
    if u.startswith("a1"):
        try:
            return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
        except Exception:
            pass
    return href


def _browser_search(query: str, n: int) -> list[tuple[str, str, str]]:
    """Bing in the real browser: when every SearXNG engine is rate-limited or captcha-blocked."""
    from lxml import html as lh
    from seleniumbase import SB
    with SB(uc=True, xvfb=True, locale="en") as sb:
        sb.activate_cdp_mode("https://www.bing.com/search?q=" + quote_plus(query))
        sb.sleep(2.5)
        if CHALLENGE.search(sb.get_page_source()[:200000]):
            try:
                sb.uc_gui_click_captcha()
            except Exception:
                pass
            sb.sleep(3)
        src = sb.get_page_source()
    out = []
    for li in lh.fromstring(src).xpath("//li[contains(concat(' ', normalize-space(@class), ' '), ' b_algo ')]"):
        a = li.xpath(".//h2//a")
        if not a:
            continue
        snippet = " ".join(" ".join(li.xpath(".//p//text()")).split())
        out.append((a[0].text_content().strip(), _bing_url(a[0].get("href", "")), snippet))
    return out[:n]


@mcp.tool()
async def web_search(query: str, max_results: int = 8, language: str = "auto") -> str:
    """Search the web. Returns numbered results: title, URL and a snippet. Read promising pages with fetch_url.
    language: e.g. "en", "cs" or "auto"."""
    async with httpx.AsyncClient(timeout=25) as c:
        r = await c.get(f"{SEARX}/search", params={"q": query, "format": "json", "language": language, "safesearch": 0})
        r.raise_for_status()
        data = r.json()
    n = max(1, min(max_results, 20))
    out = []
    for a in data.get("answers", [])[:2]:
        out.append("Answer: " + (a if isinstance(a, str) else str(a.get("answer", a))))
    for i, it in enumerate(data.get("results", [])[:n], 1):
        snippet = " ".join((it.get("content") or "").split())
        out.append(f"[{i}] {it.get('title', '').strip()}\n{it.get('url', '')}\n{snippet}")
    if not data.get("results"):   # every engine blocked or empty: Bing in the browser
        try:
            res = await on_browser(_browser_search, query, n, timeout=60)
            out += [f"[{i}] {t}\n{u}\n{sn}" for i, (t, u, sn) in enumerate(res, 1)]
        except Exception:
            pass
    return "\n\n".join(out) if out else f"No results for: {query}"


@mcp.tool()
async def fetch_url(url: str, max_chars: int = 20000, browser: bool = False) -> str:
    """Fetch a web page and return its main text as markdown (articles, documentation, READMEs, forum posts).
    Product pages of e-shops also give the product's structured data (price, availability, parameters).
    Pages behind bot protection or built with JavaScript are opened in a real (stealth) browser automatically;
    browser=True forces that. Long pages are cut at max_chars."""
    final, text, how = url, "", "http"
    if not browser:
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=HEADERS) as c:
                r = await c.get(url)
            ct = r.headers.get("content-type", "")
            final = str(r.url)
            if r.status_code < 400 and not CHALLENGE.search(r.text[:200000]):
                if "html" in ct or not ct:
                    text = extract(r.text, final)
                elif ct.startswith("text/") or "json" in ct or "xml" in ct:
                    text = r.text
                else:
                    return f"{final}: unsupported content type {ct}"
        except Exception:
            text = ""
    if len(text.strip()) < 40:   # blocked, a challenge, an empty JavaScript shell, or forced: the browser
        try:
            final, html = await fetch_browser(url)
            how = "browser"
            if CHALLENGE.search(html[:200000]):
                return f"{final}: blocked by bot protection (the browser could not get through)"
            text = extract(html, final) or "(no readable text on the page)"
        except Exception as e:
            if not text:
                return f"{url}: could not load ({type(e).__name__}: {e})"
    max_chars = max(1000, min(max_chars, 100000))
    cut = f"\n\n[... cut at {max_chars} of {len(text)} characters]" if len(text) > max_chars else ""
    return f"{final} ({how})\n\n{text[:max_chars]}{cut}"


# ---------- the interactive browser: one Chrome kept open between calls (cookies, logins and forms stay), closed after
# IDLE seconds unused. Every action returns the page as text in which interactive elements carry numbers: [12 link -> /x].
IDLE = 600
PAGE_CHARS = 12000
_S = {"ctx": None, "sb": None, "last": 0.0, "text": "", "els": {}}

# the page in document order: visible text, headings and list items marked, interactive elements numbered (data-hid)
SNAP_JS = r"""(() => {
 const INTER = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],[role=tab],[role=menuitem],' +
  '[role=checkbox],[role=radio],[role=option],[role=switch],[role=combobox],[role=textbox],[contenteditable=""],[contenteditable=true],[onclick]';
 const BLOCK = /^(P|DIV|SECTION|ARTICLE|MAIN|HEADER|FOOTER|NAV|ASIDE|LI|UL|OL|DL|DT|DD|TR|TABLE|THEAD|TBODY|H[1-6]|FORM|FIELDSET|LEGEND|BR|HR|BLOCKQUOTE|PRE|FIGURE|FIGCAPTION|DETAILS|DIALOG)$/;
 const MAX = 1500, out = [], els = []; let n = 0;
 document.querySelectorAll('[data-hid]').forEach(e => e.removeAttribute('data-hid'));
 const sq = t => (t || '').replace(/\s+/g, ' ').trim();
 const vis = e => { const s = getComputedStyle(e); if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse') return false;
  if (s.display === 'contents') return true; const r = e.getBoundingClientRect(); return r.width > 0 || r.height > 0 || s.overflow !== 'hidden' && e.children.length > 0 };
 const label = e => sq(e.getAttribute('aria-label') || e.innerText || e.getAttribute('title') || e.getAttribute('alt') || (e.querySelector('img[alt]') || {}).alt || '').slice(0, 120);
 const path = h => { try { const u = new URL(h, location.href); return (u.origin === location.origin ? '' : u.host) + u.pathname + u.search } catch (x) { return '' } };
 function mark(e) {
  const id = ++n; e.setAttribute('data-hid', id);
  const tag = e.tagName.toLowerCase(), role = e.getAttribute('role');
  els.push({n: id, tag, type: e.type || '', name: e.getAttribute('name') || '', idattr: e.id || '', ph: e.getAttribute('placeholder') || '',
   aria: e.getAttribute('aria-label') || '', href: e.getAttribute('href') || '', label: label(e)});   // (to find it again if the page replaces it)
  if (tag === 'input') { const t = (e.type || 'text').toLowerCase();
   if (t === 'checkbox' || t === 'radio') return `[${id} ${t}${e.checked ? ' checked' : ''}] ${sq(e.labels && e.labels[0] ? e.labels[0].innerText : label(e))}`;
   if (t === 'submit' || t === 'button' || t === 'reset' || t === 'image') return `[${id} button] ${sq(e.value) || label(e)}`;
   return `[${id} input ${t}${e.name ? ' name=' + e.name : ''}${e.placeholder ? ' "' + sq(e.placeholder).slice(0, 60) + '"' : ''}${e.value ? ' value="' + e.value.slice(0, 80) + '"' : ''}${e.disabled ? ' disabled' : ''}]` }
  if (tag === 'textarea') return `[${id} textarea${e.placeholder ? ' "' + sq(e.placeholder).slice(0, 60) + '"' : ''}${e.value ? ' value="' + e.value.slice(0, 120) + '"' : ''}]`;
  if (tag === 'select') return `[${id} select: ${[...e.options].slice(0, 40).map(o => (o.selected ? '*' : '') + sq(o.text)).join(' | ')}]`;
  if (tag === 'a') return `[${id} link -> ${path(e.getAttribute('href')).slice(0, 80)}] ${label(e)}`;
  return `[${id} ${tag === 'button' ? 'button' : role || 'clickable'}${e.getAttribute('aria-expanded') === 'true' ? ' expanded' : ''}${e.disabled ? ' disabled' : ''}] ${label(e)}`;
 }
 function walk(node) {
  if (node.nodeType === 3) { const t = node.textContent.replace(/\s+/g, ' '); if (t.trim()) out.push(t); return }
  if (node.nodeType !== 1) return;
  const e = node, tag = e.tagName;
  if (/^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|SVG|CANVAS|IFRAME|HEAD|OBJECT)$/.test(tag) || !vis(e)) return;
  if (n < MAX && e.matches(INTER)) {
   const m = mark(e);
   // (a small control is one line; a big clickable block keeps its content, whose own controls get numbers too)
   if (/^(INPUT|TEXTAREA|SELECT)$/.test(tag) || (!e.querySelector(INTER) && sq(e.innerText).length <= 200)) { out.push(' ' + m + ' '); return }
   out.push('\n' + m + '\n');
  }
  const block = BLOCK.test(tag);
  if (block) out.push('\n');
  if (/^H[1-6]$/.test(tag)) out.push('#'.repeat(+tag[1]) + ' ');
  else if (tag === 'LI') out.push('- ');
  else if (tag === 'IMG' && e.alt) out.push(` [image: ${sq(e.alt).slice(0, 100)}] `);
  for (const c of e.childNodes) walk(c);
  if (e.shadowRoot) for (const c of e.shadowRoot.childNodes) walk(c);
  if (block) out.push('\n');
 }
 walk(document.body);
 const t = out.join('').replace(/[ \t]+/g, ' ').replace(/ *\n */g, '\n').replace(/\n{3,}/g, '\n\n').trim();
 return JSON.stringify({text: t + (n >= MAX ? `\n\n(only the first ${MAX} interactive elements are numbered)` : ''), els});
})()"""


def _close():
    if _S["ctx"] is not None:
        try:
            _S["ctx"].__exit__(None, None, None)
        except Exception:
            pass
    _S.update(ctx=None, sb=None, text="")


def _session():
    sb = _S["sb"]
    if sb is not None:
        try:
            sb.cdp.get_current_url()   # (still alive?)
        except Exception:
            _close()
            sb = None
    if sb is None:
        from seleniumbase import SB
        ctx = SB(uc=True, xvfb=True, locale="en")
        sb = ctx.__enter__()
        sb.activate_cdp_mode("about:blank")
        _S.update(ctx=ctx, sb=sb)
    _S["last"] = time.time()
    return sb


def _idle_loop():
    while True:
        time.sleep(60)
        if _S["sb"] is not None and time.time() - _S["last"] > IDLE:
            _ON.submit(lambda: time.time() - _S["last"] > IDLE and _close())


threading.Thread(target=_idle_loop, daemon=True).start()


def _settle(sb, wait=1.0):
    sb.sleep(wait)
    for _ in range(20):
        try:
            if sb.cdp.evaluate("document.readyState") == "complete":
                break
        except Exception:
            pass
        sb.sleep(0.5)
    _pass_challenge(sb)
    # then until the page stops changing (scripts that build or replace parts of it after the load), at most 3 s
    last, t0 = None, time.time()
    while time.time() - t0 < 3:
        try:
            v = sb.cdp.evaluate("(() => { if (window.__hq == null) { window.__hq = 0; new MutationObserver(() => window.__hq++)"
                                ".observe(document, {subtree: true, childList: true}) } return window.__hq })()")
        except Exception:
            break
        if v == last:
            break
        last = v
        sb.sleep(0.35)


def _cut(text: str, start: int) -> str:
    start = max(0, start)
    part, rest = text[start:start + PAGE_CHARS], len(text) - start - PAGE_CHARS
    return part + (f"\n\n[... {rest} more characters: browser_read(start={start + PAGE_CHARS}) continues]" if rest > 0 else "")


def _page(sb, note: str = "") -> str:
    if CHALLENGE.search(sb.get_page_source()[:200000]):
        return f"{sb.cdp.get_current_url()}: blocked by bot protection (the browser could not get through)"
    r = json.loads(sb.cdp.evaluate(SNAP_JS) or "{}")
    body, _S["els"] = r.get("text") or "(no text on the page)", {e["n"]: e for e in r.get("els", [])}
    _S["text"] = f"Title: {sb.cdp.get_title() or ''}\nURL: {sb.cdp.get_current_url()}\n\n{body}"
    return (note + "\n\n" if note else "") + _cut(_S["text"], 0)


def _el(sb, element):
    sel = f'[data-hid="{int(element)}"]'
    if sb.cdp.is_element_present(sel):
        return sel
    d = _S.get("els", {}).get(int(element))   # (the page replaced it, e.g. a framework mounting its own search box)
    return sel if d and sb.cdp.evaluate(_REFIND % json.dumps(d)) else None


# the element most like a remembered one (same id / name / href / label ...; visible ones first) gets its number back
_REFIND = """((d) => { const sq = t => (t || '').replace(/\\s+/g, ' ').trim(); let best = null, bs = 0;
 for (const e of document.querySelectorAll(d.tag)) { let s = 0;
  if (d.idattr && e.id === d.idattr) s += 5; if (d.name && e.getAttribute('name') === d.name) s += 3; if (d.href && e.getAttribute('href') === d.href) s += 3;
  if (d.ph && e.getAttribute('placeholder') === d.ph) s += 2; if (d.aria && e.getAttribute('aria-label') === d.aria) s += 2;
  if (d.label && sq(e.getAttribute('aria-label') || e.innerText) === d.label) s += 2; if (d.type && e.type === d.type) s += 1;
  const r = e.getBoundingClientRect(); if (r.width > 0 && r.height > 0) s += 1;
  if (s > bs) { bs = s; best = e } }
 if (best && bs >= 3) { best.setAttribute('data-hid', d.n); return true } return false })(%s)"""


def _what(element) -> str:   # (the element as the last page showed it: what an action says it acted on)
    d = _S.get("els", {}).get(int(element)) or {}
    t = d.get("label") or d.get("ph") or d.get("aria") or d.get("name") or d.get("href") or d.get("tag") or "?"
    return " ".join(str(t).split())[:100]


def _missing(sb, element) -> str:
    return _page(sb, f"There is no element [{element}] on the page now (numbers change with every page). The current page:")


def _b_open(url):
    sb = _session()
    sb.cdp.open(url)
    _settle(sb, 1.5)
    return _page(sb)


def _b_click(element):
    sb = _session()
    sel = _el(sb, element)
    if not sel:
        return _missing(sb, element)
    what = f"Clicked [{element}] {_what(element)}"
    sb.cdp.evaluate(f"(() => {{ const a = document.querySelector('{sel}').closest('a'); if (a) a.removeAttribute('target') }})()")   # (stay in this tab)
    tabs = len(sb.cdp.get_tabs())
    try:
        sb.cdp.click(sel, timeout=3)
    except Exception:   # (not clickable like a person would: hidden, covered or replaced) its click event then
        sel = _el(sb, element)
        if not sel:
            return _missing(sb, element)
        sb.cdp.evaluate(f"document.querySelector('{sel}').click()")
    _settle(sb)
    if len(sb.cdp.get_tabs()) > tabs:   # (a new window after all: continue there)
        sb.cdp.switch_to_newest_tab()
        _settle(sb, 0.5)
    return _page(sb, what)


def _b_type(element, text, submit):
    sb = _session()
    sel = _el(sb, element)
    if not sel:
        return _missing(sb, element)
    what = f"Typed into [{element}] {_what(element)}"
    try:
        sb.cdp.type(sel, text, timeout=3)   # (replaces the old content)
        if submit:
            sb.cdp.press_keys(sel, "\n", timeout=3)
    except Exception:   # (a field that is hidden, or that the page replaced when it got the focus): value and events directly
        sel = _el(sb, element)
        if not sel:
            return _missing(sb, element)
        sb.cdp.evaluate(_SET_VALUE % (json.dumps(sel), json.dumps(text), "true" if submit else "false"))
    if submit:
        _settle(sb)
    else:
        sb.sleep(0.4)
    return _page(sb, what)


# (the native value setter, so that frameworks such as React see the change; submitting: the form, or Enter on the field)
_SET_VALUE = """(() => { const e = document.querySelector(%s), v = %s; e.focus();
 const d = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(e), 'value'); if (d && d.set) d.set.call(e, v); else e.value = v;
 e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true}));
 if (%s) { if (e.form) { e.form.requestSubmit ? e.form.requestSubmit() : e.form.submit() }
  else for (const t of ['keydown', 'keypress', 'keyup']) e.dispatchEvent(new KeyboardEvent(t, {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true})) } })()"""


def _b_select(element, option):
    sb = _session()
    sel = _el(sb, element)
    if not sel:
        return _missing(sb, element)
    what = f"Selected in [{element}] {_what(element)}"
    sb.cdp.select_option_by_text(sel, option)
    _settle(sb, 0.6)
    return _page(sb, what)


def _b_scroll(direction):
    sb = _session()
    sign = -1 if str(direction).lower().startswith("u") else 1
    sb.cdp.evaluate(f"window.scrollBy(0, {sign} * window.innerHeight * 0.9)")
    _settle(sb, 0.8)
    y = sb.cdp.evaluate("Math.round(100 * (window.scrollY + window.innerHeight) / Math.max(1, document.documentElement.scrollHeight))")
    return _page(sb, f"Scrolled {'up' if sign < 0 else 'down'}: the window now ends at {y}% of the page.")


def _b_back():
    sb = _session()
    sb.cdp.go_back()
    _settle(sb)
    return _page(sb)


def _fold(t: str) -> str:   # (lower case without diacritics: "graficke" finds "Grafické")
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))


def _b_find(text):
    if _S["sb"] is None or not _S["text"]:
        return "No page is open in the browser: use browser_open first."
    _S["last"] = time.time()
    q = _fold(str(text)).strip()
    hits = [ln.strip() for ln in _S["text"].split("\n") if q and q in _fold(ln)]
    if not hits:
        return f"Nothing on the page contains \u201c{text}\u201d."
    return (f"{len(hits)} lines of the page contain \u201c{text}\u201d" + (" (the first 60)" if len(hits) > 60 else "") + ":\n\n" +
            "\n".join(h[:500] for h in hits[:60]))


def _b_read(start):
    if _S["sb"] is None or not _S["text"]:
        return "No page is open in the browser: use browser_open first."
    _S["last"] = time.time()
    return _cut(_S["text"], int(start))


async def _act(fn, *args) -> str:
    try:
        return await on_browser(fn, *args, timeout=120)
    except asyncio.TimeoutError:
        return "The browser did not finish within 120 s (the page may still be loading): try browser_read."
    except Exception as e:
        return f"Browser error: {type(e).__name__}: {str(e)[:300]}"


@mcp.tool()
async def browser_open(url: str) -> str:
    """Open a page in a real web browser that stays open between calls (cookies, logins and form contents are kept).
    Returns the page as text in which interactive elements are numbered, like [12 link -> /pricing] or [7 input text "Search"].
    Use it to click, fill in forms, use a site's own search or move through pages; to just read a page, fetch_url is faster.
    A long page comes in parts (the cut says how browser_read continues); browser_find lists the lines with a phrase."""
    return await _act(_b_open, url)


@mcp.tool()
async def browser_click(element: int) -> str:
    """Click element [n] (link, button, tab, checkbox, ...) of the page open in the browser; returns the page after the click."""
    return await _act(_b_click, element)


@mcp.tool()
async def browser_type(element: int, text: str, submit: bool = False) -> str:
    """Type text into input [n] of the page open in the browser (its old content is replaced); submit=true then presses Enter."""
    return await _act(_b_type, element, text, submit)


@mcp.tool()
async def browser_select(element: int, option: str) -> str:
    """Choose the option with this visible text in select [n] of the page open in the browser."""
    return await _act(_b_select, element, option)


@mcp.tool()
async def browser_scroll(direction: str = "down") -> str:
    """Scroll the page open in the browser by one screen, "down" or "up" (loads content that appears on scrolling)."""
    return await _act(_b_scroll, direction)


@mcp.tool()
async def browser_back() -> str:
    """Go back to the previous page in the browser."""
    return await _act(_b_back)


@mcp.tool()
async def browser_find(text: str) -> str:
    """The lines of the page open in the browser that contain this text (any case, with or without diacritics), with their
    element numbers: to find a product, link or button on a long page without reading all of it."""
    return await _act(_b_find, text)


@mcp.tool()
async def browser_read(start: int = 0) -> str:
    """The text of the page open in the browser again, from character `start` (long pages are cut; the cut says where to go on)."""
    return await _act(_b_read, start)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
