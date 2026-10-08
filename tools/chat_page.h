// hyper-server web UI: chat at "/" (conversations in the browser's localStorage, streaming, reasoning, markdown with
// syntax highlighting and math when the CDN libraries load, a built-in renderer otherwise); live statistics at "/live"
#pragma once

static const char * CHAT_PAGE = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>hyper</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.10.0/highlight.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
:root{
 --bg:#ffffff;--bg2:#f7f7f8;--bg3:#efeff1;--fg:#16161a;--fg2:#4a4a52;--mut:#8b8b94;--line:#e6e6ea;--line2:#d9d9de;
 --acc:#5b5bd6;--acc2:#4a4ac4;--accbg:#ededfc;--ubub:#f2f2f4;--code:#0f1117;--codefg:#e6e6ea;--codehd:#1a1d26;
 --ok:#22a06b;--danger:#d64545;--shadow:0 1px 2px rgba(16,16,24,.04),0 4px 16px rgba(16,16,24,.06);--r:16px}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#141417;--bg2:#1b1b1f;--bg3:#232328;--fg:#ececf0;--fg2:#b8b8c2;--mut:#7d7d88;--line:#2a2a30;--line2:#35353c;
 --acc:#8b8bf5;--acc2:#a3a3ff;--accbg:#25254a;--ubub:#26262c;--code:#0d0f14;--codefg:#e6e6ea;--codehd:#171a22;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25)}}
:root[data-theme=dark]{--bg:#141417;--bg2:#1b1b1f;--bg3:#232328;--fg:#ececf0;--fg2:#b8b8c2;--mut:#7d7d88;--line:#2a2a30;--line2:#35353c;
 --acc:#8b8bf5;--acc2:#a3a3ff;--accbg:#25254a;--ubub:#26262c;--code:#0d0f14;--codefg:#e6e6ea;--codehd:#171a22;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25)}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
 -webkit-font-smoothing:antialiased;display:flex;overflow:hidden}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer}
svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex-shrink:0}
.ib{width:34px;height:34px;display:inline-grid;place-items:center;border-radius:10px;color:var(--fg2);transition:background .15s,color .15s}
.ib:hover{background:var(--bg3);color:var(--fg)}
/* sidebar */
aside{width:268px;background:var(--bg2);border-right:1px solid var(--line);display:flex;flex-direction:column;flex-shrink:0;transition:margin .2s}
body.collapsed aside{margin-left:-268px}
.sb-top{display:flex;align-items:center;gap:6px;padding:12px 10px 8px}
.brand{flex:1;font-weight:600;font-size:15px;letter-spacing:-.01em;display:flex;align-items:center;gap:8px;padding-left:6px}
.brand i{width:22px;height:22px;border-radius:7px;background:linear-gradient(135deg,var(--acc),#c66df0);display:inline-block}
.newbtn{margin:4px 10px 8px;display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:12px;border:1px solid var(--line2);background:var(--bg);font-weight:500;box-shadow:var(--shadow)}
.newbtn:hover{border-color:var(--acc)}
.search{margin:0 10px 6px;display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:10px;background:var(--bg3);color:var(--mut)}
.search input{flex:1;border:0;background:none;color:var(--fg);font:inherit;outline:none;font-size:14px}
#convs{flex:1;overflow:auto;padding:4px 8px 12px}
.grp{font-size:12px;color:var(--mut);font-weight:500;padding:12px 8px 4px}
.cv{display:flex;align-items:center;gap:4px;padding:7px 8px 7px 10px;border-radius:10px;cursor:pointer;font-size:14px;color:var(--fg2)}
.cv:hover{background:var(--bg3);color:var(--fg)}.cv.on{background:var(--bg3);color:var(--fg);font-weight:500}
.cv span{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.cv .ib{width:26px;height:26px;opacity:0}.cv:hover .ib{opacity:1}
.sb-foot{border-top:1px solid var(--line);padding:10px;display:flex;align-items:center;gap:8px;font-size:13px;color:var(--mut)}
.sb-foot a{color:var(--fg2);text-decoration:none;display:flex;align-items:center;gap:6px;padding:6px 8px;border-radius:8px}.sb-foot a:hover{background:var(--bg3)}
/* main */
main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
header{height:56px;display:flex;align-items:center;gap:8px;padding:0 12px;flex-shrink:0}
.model{display:flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;font-weight:500;font-size:14px}
.model .dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 20%,transparent)}
.model .dot.busy{background:#e5a100;box-shadow:0 0 0 3px color-mix(in srgb,#e5a100 22%,transparent);animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.45}}
.sp{flex:1}
.ctx{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--mut);padding:0 6px}
.ring{width:18px;height:18px}
#scroll{flex:1;overflow:auto;scroll-behavior:auto}
.col{max-width:780px;margin:0 auto;padding:8px 24px 40px}
.msg{margin:22px 0}
.msg.user{display:flex;justify-content:flex-end}
.ububble{background:var(--ubub);border-radius:20px;padding:10px 16px;max-width:85%;white-space:pre-wrap;word-wrap:break-word}
.msg.user:hover .uact{opacity:1}
.uact{opacity:0;align-self:center;margin-right:6px;transition:opacity .15s}
.amsg{position:relative}
.md{word-wrap:break-word}
.md>*:first-child{margin-top:0}.md>*:last-child{margin-bottom:0}
.md p{margin:0 0 .85em}.md ul,.md ol{margin:0 0 .85em;padding-left:1.4em}.md li{margin:.2em 0}
.md h1,.md h2,.md h3,.md h4{margin:1.3em 0 .5em;line-height:1.3;letter-spacing:-.01em}.md h1{font-size:1.45em}.md h2{font-size:1.25em}.md h3{font-size:1.08em}
.md a{color:var(--acc);text-decoration:none}.md a:hover{text-decoration:underline}
.md blockquote{margin:0 0 .85em;padding:2px 14px;border-left:3px solid var(--line2);color:var(--fg2)}
.md hr{border:0;border-top:1px solid var(--line);margin:1.4em 0}
.md table{border-collapse:collapse;margin:0 0 1em;font-size:14px;display:block;overflow:auto}
.md th,.md td{border:1px solid var(--line);padding:6px 10px;text-align:left}.md th{background:var(--bg2);font-weight:600}
.md :not(pre)>code{font:13.5px "JetBrains Mono",ui-monospace,monospace;font-variant-ligatures:none;background:var(--bg3);padding:.12em .38em;border-radius:6px}
.cb{margin:0 0 1em;border-radius:12px;overflow:hidden;background:var(--code);border:1px solid var(--line)}
.cb .hd{display:flex;align-items:center;justify-content:space-between;padding:6px 8px 6px 14px;background:var(--codehd);color:#9a9aa8;font-size:12px}
.cb .hd button{display:flex;align-items:center;gap:5px;color:#b8b8c6;font-size:12px;padding:4px 8px;border-radius:7px}.cb .hd button:hover{background:#2a2e3a;color:#fff}
.cb .hd svg{width:14px;height:14px}
.cb pre{margin:0;padding:12px 14px;overflow:auto;color:var(--codefg);font:13px/1.55 "JetBrains Mono",ui-monospace,monospace;font-variant-ligatures:none}
.cb pre code{font:inherit;background:none;padding:0}
.think{margin:0 0 12px}
.think .th{display:inline-flex;align-items:center;gap:6px;color:var(--mut);font-size:14px;padding:4px 10px 4px 8px;border-radius:999px;cursor:pointer;user-select:none}
.think .th:hover{background:var(--bg2);color:var(--fg2)}
.think .th svg{width:15px;height:15px;transition:transform .2s}.think.open .th .chev{transform:rotate(90deg)}
.think .tb{display:none;margin:6px 0 0 6px;padding:2px 0 2px 14px;border-left:2px solid var(--line2);color:var(--fg2);font-size:14px;white-space:pre-wrap;max-height:420px;overflow:auto}
.think.open .tb{display:block}
.shimmer{background:linear-gradient(90deg,var(--mut) 25%,var(--fg) 50%,var(--mut) 75%);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 1.6s linear infinite}
@keyframes sh{to{background-position:-200% 0}}
.cursor:after{content:"";display:inline-block;width:8px;height:16px;margin-left:2px;vertical-align:-2px;background:var(--fg);border-radius:2px;animation:bl 1s steps(1) infinite}
@keyframes bl{50%{opacity:0}}
.tool{margin:0 0 12px;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tool .hd{padding:8px 12px;font-size:13px;color:var(--fg2);background:var(--bg2);display:flex;gap:6px;align-items:center;font-weight:500}
.tool pre{margin:0;padding:10px 12px;font:12.5px "JetBrains Mono",monospace;white-space:pre-wrap;color:var(--fg2)}
.foot{display:flex;align-items:center;gap:2px;margin:6px 0 0 -8px;color:var(--mut);font-size:12.5px;min-height:30px}
.foot .ib{width:30px;height:30px;opacity:.85}
.foot .stat{display:flex;align-items:center;gap:10px;margin-left:6px}
.foot .stat span{display:flex;align-items:center;gap:4px}.foot .stat svg{width:13px;height:13px}
.err{color:var(--danger);background:color-mix(in srgb,var(--danger) 8%,transparent);border-radius:10px;padding:8px 12px;font-size:14px;margin-top:8px}
/* composer */
.comp-wrap{padding:0 24px 18px;flex-shrink:0}
.comp{max-width:780px;margin:0 auto;background:var(--bg);border:1px solid var(--line2);border-radius:24px;box-shadow:var(--shadow);padding:10px 10px 8px 16px;transition:border-color .15s}
.comp:focus-within{border-color:color-mix(in srgb,var(--acc) 55%,var(--line2))}
.comp textarea{width:100%;border:0;outline:none;resize:none;background:none;color:var(--fg);font:inherit;font-size:15.5px;line-height:1.5;max-height:240px;padding:4px 0}
.comp textarea::placeholder{color:var(--mut)}
.crow{display:flex;align-items:center;gap:6px;margin-top:4px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 11px;border-radius:999px;border:1px solid var(--line2);font-size:13px;color:var(--fg2)}
.pill:hover{background:var(--bg2)}.pill.on{background:var(--accbg);border-color:transparent;color:var(--acc)}
.pill svg{width:15px;height:15px}
.send{width:36px;height:36px;border-radius:50%;background:var(--fg);color:var(--bg);display:grid;place-items:center;transition:opacity .15s,transform .1s}
.send:disabled{opacity:.25;cursor:default}.send:not(:disabled):active{transform:scale(.94)}
.send svg{width:18px;height:18px;stroke-width:2.2}
.hint{max-width:780px;margin:8px auto 0;text-align:center;font-size:12px;color:var(--mut)}
/* empty state */
.hero{flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 24px 12vh}
.hero h1{font-size:30px;font-weight:600;letter-spacing:-.02em;margin:0 0 6px;text-align:center}
.hero p{color:var(--mut);margin:0 0 26px;text-align:center}
.hero .comp-wrap{width:100%;padding:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:16px;max-width:780px}
.chip{padding:8px 14px;border:1px solid var(--line2);border-radius:999px;font-size:13.5px;color:var(--fg2);background:var(--bg)}
.chip:hover{background:var(--bg2);color:var(--fg)}
.jump{position:absolute;left:50%;transform:translateX(-50%);bottom:140px;width:36px;height:36px;border-radius:50%;background:var(--bg);border:1px solid var(--line2);box-shadow:var(--shadow);display:none;place-items:center;color:var(--fg2)}
.jump.on{display:grid}
/* settings drawer */
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.25);opacity:0;pointer-events:none;transition:opacity .2s;z-index:9}
.scrim.on{opacity:1;pointer-events:auto}
.drawer{position:fixed;top:0;right:0;bottom:0;width:360px;max-width:92vw;background:var(--bg);border-left:1px solid var(--line);box-shadow:-8px 0 32px rgba(0,0,0,.12);transform:translateX(100%);transition:transform .22s ease;z-index:10;display:flex;flex-direction:column}
.drawer.on{transform:none}
.drawer .dh{display:flex;align-items:center;padding:14px 14px 10px 20px;font-weight:600}
.drawer .db{padding:4px 20px 20px;overflow:auto;display:flex;flex-direction:column;gap:18px}
.fld label{display:flex;justify-content:space-between;font-size:13px;font-weight:500;color:var(--fg2);margin-bottom:6px}
.fld label span{color:var(--mut);font-weight:400;font-variant-numeric:tabular-nums}
.fld textarea,.fld input[type=number]{width:100%;font:inherit;font-size:14px;color:var(--fg);background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:8px 10px;outline:none}
.fld textarea{min-height:110px;resize:vertical}.fld textarea:focus,.fld input:focus{border-color:var(--acc)}
input[type=range]{width:100%;accent-color:var(--acc)}
.seg{display:flex;background:var(--bg2);border-radius:10px;padding:3px;gap:3px}.seg button{flex:1;padding:6px;border-radius:8px;font-size:13px;color:var(--fg2)}.seg button.on{background:var(--bg);color:var(--fg);box-shadow:var(--shadow)}
.note{font-size:12px;color:var(--mut)}
@media (max-width:760px){aside{position:fixed;z-index:8;height:100%;box-shadow:var(--shadow)}body.collapsed aside{margin-left:-268px}.col{padding:8px 14px 30px}.comp-wrap{padding:0 10px 10px}}
.hljs{color:#e6e6ea}.hljs-keyword,.hljs-selector-tag,.hljs-built_in{color:#c792ea}.hljs-string,.hljs-attr{color:#c3e88d}.hljs-number,.hljs-literal{color:#f78c6c}
.hljs-comment{color:#6b7089;font-style:italic}.hljs-title,.hljs-function .hljs-title,.hljs-title.function_{color:#82aaff}.hljs-type,.hljs-class .hljs-title{color:#ffcb6b}
.hljs-variable,.hljs-params{color:#e6e6ea}.hljs-meta{color:#89ddff}.hljs-symbol,.hljs-property{color:#f07178}
</style></head><body>
<aside>
 <div class="sb-top"><div class="brand"><i></i>hyper</div><button class="ib" title="Hide sidebar" onclick="toggleSide()"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/></svg></button></div>
 <button class="newbtn" onclick="newConv()"><svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>New chat</button>
 <div class="search"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><input id="q" placeholder="Search chats" oninput="renderSide()"></div>
 <div id="convs"></div>
 <div class="sb-foot"><a href="/live"><svg viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>Live stats</a><span class="sp"></span>
  <button class="ib" title="Theme" onclick="cycleTheme()"><svg viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg></button></div>
</aside>
<main>
 <header>
  <button class="ib" id="sbopen" title="Show sidebar" onclick="toggleSide()" style="display:none"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/></svg></button>
  <button class="ib" id="sbnew" title="New chat" onclick="newConv()" style="display:none"><svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4z"/></svg></button>
  <div class="model"><span class="dot" id="dot"></span><span id="mdl">hyper</span></div>
  <span class="sp"></span>
  <div class="ctx" id="ctx" title="Context in use"></div>
  <button class="ib" title="Settings" onclick="openSet(true)"><svg viewBox="0 0 24 24"><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/></svg></button>
 </header>
 <div id="view" style="flex:1;display:flex;flex-direction:column;min-height:0"></div>
 <button class="jump" id="jump" onclick="toBottom(true)"><svg viewBox="0 0 24 24"><path d="M12 5v14M5 12l7 7 7-7"/></svg></button>
</main>
<div class="scrim" id="scrim" onclick="openSet(false)"></div>
<div class="drawer" id="drawer">
 <div class="dh"><span style="flex:1">Settings</span><button class="ib" onclick="openSet(false)"><svg viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg></button></div>
 <div class="db">
  <div class="fld"><label>System prompt</label><textarea id="sys" placeholder="You are a helpful assistant."></textarea></div>
  <div class="fld"><label>Temperature <span id="tempv"></span></label><input type="range" id="temp" min="0" max="2" step="0.05"></div>
  <div class="fld"><label>Top-p <span id="toppv"></span></label><input type="range" id="topp" min="0.05" max="1" step="0.05"></div>
  <div class="fld"><label>Max new tokens</label><input type="number" id="maxt" min="16" step="256"></div>
  <div class="fld"><label>Theme</label><div class="seg" id="theme"><button data-v="system">System</button><button data-v="light">Light</button><button data-v="dark">Dark</button></div></div>
  <div class="note">Empty temperature / top-p use the server defaults. Conversations are stored in this browser only.</div>
  <button class="pill" style="align-self:flex-start" onclick="resetSampling()">Use server defaults</button>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const S={get(k,d){try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v)}catch(e){return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const IC={copy:'<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2.5"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
 check:'<svg viewBox="0 0 24 24"><path d="m5 12 5 5L20 7"/></svg>',regen:'<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16M3 21v-5h5"/></svg>',
 edit:'<svg viewBox="0 0 24 24"><path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4z"/></svg>',del:'<svg viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>',
 chev:'<svg class="chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>',spark:'<svg viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/></svg>',
 bolt:'<svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>',inbox:'<svg viewBox="0 0 24 24"><path d="M4 13h4l2 3h4l2-3h4"/><path d="M4 13l2-8h12l2 8v6H4z"/></svg>',
 db:'<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
 send:'<svg viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',stop:'<svg viewBox="0 0 24 24" style="fill:currentColor;stroke:none"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>',
 brain:'<svg viewBox="0 0 24 24"><path d="M9 3a3 3 0 0 0-3 3v.2A3 3 0 0 0 4 9a3 3 0 0 0 .8 2A3 3 0 0 0 4 13a3 3 0 0 0 2 2.8V16a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM15 3a3 3 0 0 1 3 3v.2A3 3 0 0 1 20 9a3 3 0 0 1-.8 2 3 3 0 0 1 .8 2 3 3 0 0 1-2 2.8V16a3 3 0 0 1-6 0"/></svg>',
 tool:'<svg viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/></svg>',trash:'<svg viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>'};
let convs=S.get('hyper.convs',[]),cur=S.get('hyper.cur',null),busy=null,live=null,defaults={};
const cfg=Object.assign({sys:'',temp:'',topp:'',maxt:16384,think:true,theme:'system'},S.get('hyper.cfg',{}));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function saveCfg(){S.set('hyper.cfg',cfg)}
function save(){S.set('hyper.convs',convs);S.set('hyper.cur',cur)}
function conv(){return convs.find(c=>c.id===cur)}
/* ---------- theme / layout ---------- */
function applyTheme(){if(cfg.theme==='system')document.documentElement.removeAttribute('data-theme');else document.documentElement.setAttribute('data-theme',cfg.theme);
 document.querySelectorAll('#theme button').forEach(b=>b.classList.toggle('on',b.dataset.v===cfg.theme))}
function cycleTheme(){const o=['system','light','dark'];cfg.theme=o[(o.indexOf(cfg.theme)+1)%3];saveCfg();applyTheme()}
document.querySelectorAll('#theme button').forEach(b=>b.onclick=()=>{cfg.theme=b.dataset.v;saveCfg();applyTheme()});
function toggleSide(){document.body.classList.toggle('collapsed');const c=document.body.classList.contains('collapsed');S.set('hyper.side',c);
 $('sbopen').style.display=c?'':'none';$('sbnew').style.display=c?'':'none'}
if(S.get('hyper.side',window.innerWidth<760)){document.body.classList.add('collapsed');$('sbopen').style.display='';$('sbnew').style.display=''}
function openSet(on){$('drawer').classList.toggle('on',on);$('scrim').classList.toggle('on',on)}
/* ---------- settings ---------- */
function syncSet(){$('sys').value=cfg.sys;$('maxt').value=cfg.maxt;
 $('temp').value=cfg.temp===''?(defaults.temp??0.6):cfg.temp;$('tempv').textContent=cfg.temp===''?'server default':(+cfg.temp).toFixed(2);
 $('topp').value=cfg.topp===''?(defaults.top_p??0.95):cfg.topp;$('toppv').textContent=cfg.topp===''?'server default':(+cfg.topp).toFixed(2)}
$('sys').oninput=()=>{cfg.sys=$('sys').value;saveCfg()};$('maxt').onchange=()=>{cfg.maxt=+$('maxt').value||16384;saveCfg()};
$('temp').oninput=()=>{cfg.temp=$('temp').value;saveCfg();syncSet()};$('topp').oninput=()=>{cfg.topp=$('topp').value;saveCfg();syncSet()};
function resetSampling(){cfg.temp='';cfg.topp='';saveCfg();syncSet()}
/* ---------- sidebar ---------- */
function newConv(){if(busy)return;const c=conv();if(c&&!c.msgs.length){render();focusIn();return}
 const n={id:Date.now().toString(36)+Math.random().toString(36).slice(2,6),title:'New chat',msgs:[],t:Date.now()};convs.unshift(n);cur=n.id;save();render();focusIn()}
function delConv(id,e){e.stopPropagation();if(busy&&id===cur)return;convs=convs.filter(c=>c.id!==id);if(cur===id)cur=convs[0]?convs[0].id:null;save();render()}
function pick(id){if(busy)return;cur=id;save();render();toBottom(false);if(window.innerWidth<760)toggleSide()}
function renderSide(){const q=$('q').value.trim().toLowerCase(),now=new Date(),d0=new Date(now.getFullYear(),now.getMonth(),now.getDate()).getTime();
 const groups=[['Today',x=>x>=d0],['Yesterday',x=>x>=d0-864e5],['Previous 7 days',x=>x>=d0-7*864e5],['Older',()=>true]];let h='',used=new Set();
 const list=convs.filter(c=>!q||c.title.toLowerCase().includes(q)||c.msgs.some(m=>(m.content||'').toLowerCase().includes(q)));
 for(const [g,f] of groups){const items=list.filter(c=>!used.has(c.id)&&f(c.t||0));if(!items.length)continue;h+=`<div class="grp">${g}</div>`;
  for(const c of items){used.add(c.id);h+=`<div class="cv${c.id===cur?' on':''}" onclick="pick('${c.id}')"><span>${esc(c.title)}</span><button class="ib" title="Delete" onclick="delConv('${c.id}',event)">${IC.trash}</button></div>`}}
 $('convs').innerHTML=h||`<div class="grp" style="text-align:center;padding-top:30px">${q?'No matches':'No conversations yet'}</div>`}
/* ---------- markdown ---------- */
function codeBlock(code,lang){let hl=esc(code);
 if(window.hljs){try{hl=lang&&hljs.getLanguage(lang)?hljs.highlight(code,{language:lang,ignoreIllegals:true}).value:hljs.highlightAuto(code).value}catch(e){}}
 return `<div class="cb"><div class="hd"><span>${esc(lang||'code')}</span><button class="cpy">${IC.copy}<span>Copy</span></button></div><pre><code class="hljs">${hl}</code></pre></div>`}
function mathify(s){if(!window.katex)return s;const out=[];let i=0;
 s=s.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g,(m,a,b)=>{try{out.push(katex.renderToString(a||b,{displayMode:true,throwOnError:false}))}catch(e){return m}return `@@M${out.length-1}@@`})
  .replace(/\\\(([\s\S]+?)\\\)|(^|[^\\$\w])\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?![\w$])/g,(m,a,p,b)=>{const t=a||b;try{out.push(katex.renderToString(t,{throwOnError:false}))}catch(e){return m}return (p||'')+`@@M${out.length-1}@@`});
 return {s,out}}
let mdR=null;
function md(src){if(window.marked&&window.DOMPurify){
  if(!mdR){mdR=new marked.Renderer();mdR.code=(c,l)=>{if(typeof c==='object'){l=c.lang;c=c.text}return codeBlock(c,(l||'').split(/\s/)[0])};
   mdR.link=(h,t,x)=>{if(typeof h==='object'){x=h.text;h=h.href}return `<a href="${esc(h||'')}" target="_blank" rel="noopener">${x}</a>`};}
  // protect code from math: math only outside fenced / inline code
  const parts=src.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/);let M=[];
  const txt=parts.map((p,k)=>{if(k%2)return p;const r=mathify(p);if(typeof r==='string')return r;const base=M.length;M=M.concat(r.out);
   return r.s.replace(/@@M(\d+)@@/g,(_,n)=>`@@M${+n+base}@@`)}).join('');
  let html=marked.parse(txt,{renderer:mdR,gfm:true,breaks:false});
  html=DOMPurify.sanitize(html,{ADD_ATTR:['target']});
  return html.replace(/@@M(\d+)@@/g,(_,n)=>M[+n]||'')}
 return mdLite(src)}
function inl(s){return esc(s).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(^|[^*\w])\*([^*\n]+)\*/g,'$1<i>$2</i>')
 .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>')}
function mdLite(src){const L=src.split('\n');let o='',i=0;
 while(i<L.length){const l=L[i];
  if(/^```/.test(l)){const lang=l.slice(3).trim();const c=[];i++;while(i<L.length&&!/^```/.test(L[i]))c.push(L[i++]);i++;o+=codeBlock(c.join('\n'),lang);continue}
  if(/^#{1,4} /.test(l)){const n=l.match(/^#+/)[0].length;o+=`<h${n}>${inl(l.slice(n+1))}</h${n}>`;i++;continue}
  if(/^\s*([-*+]|\d+\.) /.test(l)){const ol=/^\s*\d+\./.test(l);o+=ol?'<ol>':'<ul>';while(i<L.length&&/^\s*([-*+]|\d+\.) /.test(L[i]))o+='<li>'+inl(L[i++].replace(/^\s*([-*+]|\d+\.) /,''))+'</li>';o+=ol?'</ol>':'</ul>';continue}
  if(/^> ?/.test(l)){const q=[];while(i<L.length&&/^> ?/.test(L[i]))q.push(L[i++].replace(/^> ?/,''));o+='<blockquote>'+mdLite(q.join('\n'))+'</blockquote>';continue}
  if(!l.trim()){i++;continue}
  const p=[];while(i<L.length&&L[i].trim()&&!/^(```|#{1,4} |> ?|\s*([-*+]|\d+\.) )/.test(L[i]))p.push(L[i++]);
  if(!p.length){o+='<p>'+inl(L[i++])+'</p>';continue}o+='<p>'+inl(p.join('\n')).replace(/\n/g,'<br>')+'</p>'}
 return o}
function cpCode(b){const t=b.closest('.cb').querySelector('code').innerText;navigator.clipboard.writeText(t);const s=b.innerHTML;b.innerHTML=IC.check+'<span>Copied</span>';setTimeout(()=>b.innerHTML=s,1400)}
function cpMsg(k,b){const m=conv().msgs[k];navigator.clipboard.writeText(m.content||'');b.innerHTML=IC.check;setTimeout(()=>b.innerHTML=IC.copy,1400)}
/* ---------- messages ---------- */
const fmtS=s=>s<60?Math.round(s)+'s':Math.floor(s/60)+'m '+Math.round(s%60)+'s';
function thinkHTML(m,k){if(!m.reasoning)return'';const live=m.thinking;
 const label=live?`<span class="shimmer">Thinking${m.thinkStart?' · '+fmtS((Date.now()-m.thinkStart)/1000):''}</span>`:`Thought for ${fmtS(m.thinkSecs||0)}`;
 return `<div class="think${m.open?' open':''}" data-k="${k}"><div class="th" onclick="toggleThink(${k})">${IC.chev}${label}</div><div class="tb">${esc(m.reasoning)}</div></div>`}
function toggleThink(k){const m=conv().msgs[k];m.open=!m.open;const el=document.querySelector(`.think[data-k="${k}"]`);if(el)el.classList.toggle('open',m.open)}
function statHTML(m){if(!m.t)return'';const t=m.t;let h=`<span title="generation">${IC.bolt}${t.predicted_per_second.toFixed(1)} t/s · ${t.predicted_n} tok</span>`;
 if(t.prompt_n!=null)h+=`<span title="prompt">${IC.inbox}${t.prompt_n.toLocaleString()} @ ${Math.round(t.prompt_per_second)} t/s</span>`;
 if(t.cache_n)h+=`<span title="reused from the prompt cache">${IC.db}${t.cache_n.toLocaleString()} cached</span>`;return `<div class="stat">${h}</div>`}
function msgHTML(m,k,n){if(m.role==='user')return `<div class="msg user" data-k="${k}"><div class="uact">${busy?'':`<button class="ib" title="Edit" onclick="editMsg(${k})">${IC.edit}</button>`}</div><div class="ububble">${esc(m.content)}</div></div>`;
 let h=`<div class="msg amsg" data-k="${k}">`+thinkHTML(m,k);
 if(m.tool)h+=`<div class="tool"><div class="hd">${IC.tool}Tool call</div><pre>${esc(m.tool)}</pre></div>`;
 const streaming=busy&&k===n-1;
 h+=`<div class="md${streaming&&!m.thinking?' cursor':''}">${md(m.content||'')}</div>`;
 if(streaming&&m.thinking&&!m.reasoning)h+=`<div class="shimmer" style="font-size:14px">Thinking…</div>`;
 if(m.err)h+=`<div class="err">${esc(m.err)}</div>`;
 if(!streaming)h+=`<div class="foot"><button class="ib" title="Copy" onclick="cpMsg(${k},this)">${IC.copy}</button>${k===n-1?`<button class="ib" title="Regenerate" onclick="regen()">${IC.regen}</button>`:''}${statHTML(m)}</div>`;
 return h+'</div>'}
function composer(hero){return `<div class="comp-wrap"><div class="comp"><textarea id="in" rows="1" placeholder="${hero?'Ask anything':'Message hyper'}"></textarea>
 <div class="crow"><button class="pill${cfg.think?' on':''}" id="thinkb" onclick="toggleThinkMode()">${IC.brain}Thinking</button><span class="sp"></span>
 <button class="send" id="send" onclick="sendMsg()" ${busy?'':'disabled'} title="${busy?'Stop':'Send'}">${busy?IC.stop:IC.send}</button></div></div>
 ${hero?'':'<div class="hint">Enter to send · Shift+Enter for a new line</div>'}</div>`}
function toggleThinkMode(){cfg.think=!cfg.think;saveCfg();$('thinkb').classList.toggle('on',cfg.think)}
function bindIn(){const t=$('in');if(!t)return;t.value=draft;autosz();
 t.addEventListener('input',()=>{draft=t.value;autosz();updSend()});
 t.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendMsg()}})}
let draft='';
function updSend(){const b=$('send');if(!b)return;b.disabled=!busy&&!draft.trim();b.innerHTML=busy?IC.stop:IC.send;b.title=busy?'Stop':'Send'}
function autosz(){const t=$('in');if(!t)return;t.style.height='auto';t.style.height=Math.min(t.scrollHeight,240)+'px'}
function focusIn(){const t=$('in');if(t)t.focus()}
const SUG=['Explain how a mixture-of-experts model routes tokens','Write a Python script that watches a folder and resizes new images','Compare Rust and Go for a CLI tool, with a short example in each','Help me plan a 3-day trip to Prague'];
function render(){renderSide();const c=conv();
 if(!c||!c.msgs.length){$('view').innerHTML=`<div class="hero"><h1>What can I help with?</h1><p id="hsub">Running locally on hyper</p>${composer(true)}
  <div class="chips">${SUG.map((s,i)=>`<button class="chip" onclick="useSug(${i})">${esc(s)}</button>`).join('')}</div></div>`;
  bindIn();updSend();$('jump').classList.remove('on');document.title='hyper';if(live&&live.model)$('hsub').textContent='Running locally on hyper · '+live.model;return}
 document.title=c.title+' · hyper';
 $('view').innerHTML=`<div id="scroll"><div class="col" id="col">${c.msgs.map((m,k)=>msgHTML(m,k,c.msgs.length)).join('')}</div></div>${composer(false)}`;
 bindIn();updSend();const sc=$('scroll');sc.onscroll=()=>$('jump').classList.toggle('on',sc.scrollHeight-sc.scrollTop-sc.clientHeight>200)}
function useSug(i){draft=SUG[i];sendMsg()}
function toBottom(smooth){const sc=$('scroll');if(sc)sc.scrollTo({top:sc.scrollHeight,behavior:smooth?'smooth':'auto'})}
function updLast(){const c=conv(),col=$('col'),sc=$('scroll');if(!col)return;const k=c.msgs.length-1;
 const stick=sc.scrollHeight-sc.scrollTop-sc.clientHeight<120;const el=col.lastElementChild;const tb=el&&el.querySelector('.tb');const tst=tb?tb.scrollTop:0,tbot=tb?tb.scrollHeight-tb.scrollTop-tb.clientHeight<40:true;
 const t=document.createElement('div');t.innerHTML=msgHTML(c.msgs[k],k,c.msgs.length);el.replaceWith(t.firstElementChild);
 const nb=col.lastElementChild.querySelector('.tb');if(nb)nb.scrollTop=tbot?nb.scrollHeight:tst;if(stick)toBottom(false)}
function editMsg(k){if(busy)return;const c=conv();draft=c.msgs[k].content;c.msgs.splice(k);save();render();focusIn()}
/* ---------- generation ---------- */
async function generate(){const c=conv();const a={role:'assistant',content:'',reasoning:'',thinking:true,open:true,thinkStart:Date.now()};c.msgs.push(a);
 const msgs=[];if(cfg.sys.trim())msgs.push({role:'system',content:cfg.sys});
 for(const m of c.msgs.slice(0,-1)){if(m.role==='user')msgs.push({role:'user',content:m.content});
  else if(!m.err||m.content)msgs.push(Object.assign({role:'assistant',content:m.content||''},m.reasoning?{reasoning_content:m.reasoning}:{}))}
 const body={messages:msgs,stream:true,max_tokens:+cfg.maxt||16384,chat_template_kwargs:{enable_thinking:!!cfg.think}};
 if(cfg.temp!=='')body.temperature=+cfg.temp;if(cfg.topp!=='')body.top_p=+cfg.topp;
 busy=new AbortController();render();toBottom(false);$('dot').classList.add('busy');let tool='',tick=0,timer=setInterval(()=>{if(a.thinking&&a.reasoning)updLast()},1000);
 try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:busy.signal});
  if(!r.ok)throw new Error(`Server error ${r.status}: ${await r.text()}`);
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
  for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
   while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+2);if(!line.startsWith('data:'))continue;
    const d=line.slice(5).trim();if(d==='[DONE]')continue;let j;try{j=JSON.parse(d)}catch(e){continue}
    if(j.error){a.err=j.error.message||String(j.error);continue}
    const ch=j.choices&&j.choices[0];if(ch&&ch.delta){const dl=ch.delta;
     if(dl.reasoning_content)a.reasoning+=dl.reasoning_content;
     if(dl.content){if(a.thinking){a.thinking=false;a.open=false;a.thinkSecs=(Date.now()-a.thinkStart)/1000}a.content+=dl.content}
     if(dl.tool_calls)for(const tc of dl.tool_calls){const f=tc.function||{};if(f.name)tool+=(tool?'\n':'')+f.name+' ';if(f.arguments)tool+=f.arguments}
     if(tool)a.tool=tool}
    if(j.timings)a.t=j.timings}
   const now=Date.now();if(now-tick>70){tick=now;updLast()}}
 }catch(e){if(e.name!=='AbortError')a.err=String(e.message||e)}
 clearInterval(timer);if(a.thinking){a.thinking=false;a.thinkSecs=(Date.now()-a.thinkStart)/1000;a.open=false}
 busy=null;$('dot').classList.remove('busy');c.t=Date.now();convs.sort((x,y)=>(y.t||0)-(x.t||0));save();render();toBottom(false);refreshLive();focusIn()}
function sendMsg(){if(busy){busy.abort();return}const t=draft.trim();if(!t)return;if(!conv()||busy)newConv();const c=conv();
 c.msgs.push({role:'user',content:t});if(c.title==='New chat')c.title=t.replace(/\s+/g,' ').slice(0,60);c.t=Date.now();draft='';save();generate()}
function regen(){const c=conv();if(busy||!c)return;if(c.msgs.length&&c.msgs[c.msgs.length-1].role==='assistant')c.msgs.pop();generate()}
/* ---------- server status ---------- */
async function refreshLive(){try{const s=await (await fetch('/stats')).json();live=Object.assign(live||{},s);
 const u=s.ctx_used/s.ctx_max,R=7,C=2*Math.PI*R;
 $('ctx').innerHTML=`<svg class="ring" viewBox="0 0 18 18"><circle cx="9" cy="9" r="${R}" stroke="var(--line2)" stroke-width="2.5" fill="none"/><circle cx="9" cy="9" r="${R}" stroke="var(--acc)" stroke-width="2.5" fill="none" stroke-dasharray="${C*u} ${C}" transform="rotate(-90 9 9)"/></svg>${(s.ctx_used/1000).toFixed(1)}k / ${Math.round(s.ctx_max/1000)}k`;
 $('dot').classList.toggle('busy',!!busy||s.phase!=='idle')}catch(e){}}
fetch('/v1/models').then(r=>r.json()).then(j=>{const id=j.data[0].id;$('mdl').textContent=id;live=Object.assign(live||{},{model:id});const h=$('hsub');if(h)h.textContent='Running locally on hyper · '+id}).catch(()=>{});
fetch('/props').then(r=>r.ok?r.json():{}).then(j=>{defaults=j.sampling||{};syncSet()}).catch(()=>syncSet());
document.addEventListener('click',e=>{const b=e.target.closest('.cb .cpy');if(b)cpCode(b)});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key.toLowerCase()==='o'){e.preventDefault();newConv()}if(e.key==='Escape')openSet(false)});
window.addEventListener('load',()=>{mdR=null;if(conv()&&conv().msgs.length){render();toBottom(false)}});
applyTheme();syncSet();render();toBottom(false);refreshLive();setInterval(refreshLive,4000);focusIn();
</script></body></html>)HTML";
