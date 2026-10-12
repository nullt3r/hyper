# MCP server (streamable HTTP, stateless, JSON responses) with two tools for hyper's chat:
#   web_search(query)  -- SearXNG metasearch (Google, Bing, DuckDuckGo, Brave, Wikipedia, ...)
#   fetch_url(url)     -- a page's main text as markdown (product pages: schema.org price / availability / parameters
#                         first). Plain HTTP first; a real browser (Google Chrome over CDP,
#                         SeleniumBase UC mode, headful in Xvfb) for bot challenges (Cloudflare, Turnstile, ...) and
#                         JavaScript-only pages, or when asked (browser=True)
import asyncio
import base64
import html as htmllib
import json
import os
import re
import threading
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

_lock = threading.Lock()   # one browser at a time (the Turnstile click goes through the shared virtual display)


def _browser_fetch(url: str) -> tuple[str, str]:
    """Google Chrome over CDP (SeleniumBase UC / CDP mode, headful in Xvfb): bot challenges clear themselves or get their
    Turnstile box clicked like a person would."""
    from seleniumbase import SB
    with _lock, SB(uc=True, xvfb=True, locale="en") as sb:
        sb.activate_cdp_mode(url)
        sb.sleep(3)
        for _ in range(4):
            if not CHALLENGE.search(sb.get_page_source()[:200000]):
                break
            try:
                sb.uc_gui_click_captcha()
            except Exception:
                pass
            sb.sleep(3)
        sb.sleep(1)
        return sb.get_current_url(), sb.get_page_source()


async def fetch_browser(url: str) -> tuple[str, str]:
    return await asyncio.wait_for(asyncio.to_thread(_browser_fetch, url), timeout=90)


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
    with _lock, SB(uc=True, xvfb=True, locale="en") as sb:
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
            res = await asyncio.wait_for(asyncio.to_thread(_browser_search, query, n), timeout=60)
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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
