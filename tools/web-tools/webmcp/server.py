# MCP server (streamable HTTP, stateless, JSON responses) with two tools for hyper's chat:
#   web_search(query)  -- SearXNG metasearch (Google, Bing, DuckDuckGo, Brave, Wikipedia, ...)
#   fetch_url(url)     -- a page's main text as markdown. Plain HTTP first; a real browser (Google Chrome over CDP,
#                         SeleniumBase UC mode, headful in Xvfb) for bot challenges (Cloudflare, Turnstile, ...) and
#                         JavaScript-only pages, or when asked (browser=True)
import asyncio
import os
import re
import threading

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


def extract(html: str, url: str) -> str:
    text = trafilatura.extract(html, output_format="markdown", include_links=True, include_tables=True, url=url)
    if not text:
        text = trafilatura.extract(html, favor_recall=True, url=url) or ""
    return text


@mcp.tool()
async def web_search(query: str, max_results: int = 8, language: str = "auto") -> str:
    """Search the web. Returns numbered results: title, URL and a snippet. Read promising pages with fetch_url.
    language: e.g. "en", "cs" or "auto"."""
    async with httpx.AsyncClient(timeout=25) as c:
        r = await c.get(f"{SEARX}/search", params={"q": query, "format": "json", "language": language, "safesearch": 0})
        r.raise_for_status()
        data = r.json()
    out = []
    for a in data.get("answers", [])[:2]:
        out.append("Answer: " + (a if isinstance(a, str) else str(a.get("answer", a))))
    for i, it in enumerate(data.get("results", [])[: max(1, min(max_results, 20))], 1):
        snippet = " ".join((it.get("content") or "").split())
        out.append(f"[{i}] {it.get('title', '').strip()}\n{it.get('url', '')}\n{snippet}")
    return "\n\n".join(out) if out else f"No results for: {query}"


@mcp.tool()
async def fetch_url(url: str, max_chars: int = 20000, browser: bool = False) -> str:
    """Fetch a web page and return its main text as markdown (articles, documentation, READMEs, forum posts).
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
