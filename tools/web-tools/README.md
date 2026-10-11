# Web tools for hyper's chat (outside hyper)

`docker compose up -d --build` starts two services on localhost:

- **searxng** (127.0.0.1:8888): metasearch (Google, Bing, DuckDuckGo, Brave, Wikipedia, ...), JSON API enabled.
- **webmcp** (127.0.0.1:8890/mcp): an MCP server (streamable HTTP, stateless) with `web_search(query)` and
  `fetch_url(url)`. A page comes over plain HTTP first; bot challenges (Cloudflare / Turnstile, DataDome, PerimeterX, ...)
  and JavaScript-only pages open in Google Chrome over CDP (SeleniumBase UC mode, headful in Xvfb).

hyper-server only forwards: `--mcp web=http://127.0.0.1:8890/mcp` lists the server at `/v1/mcp` and passes the chat
page's JSON-RPC requests on at `/mcp/web`; the page runs the tool loop (the "Web" switch next to "Thinking").
Both containers are capped at 2 CPUs; a tool runs while the model waits for its result, so inference is not slowed.
