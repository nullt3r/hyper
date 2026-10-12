// hyper-server web UI: chat at "/" (conversations in the browser's localStorage, streaming with live speed and prompt
// progress, reasoning, answer versions, markdown with syntax highlighting and math when the CDN libraries load, a built-in
// renderer otherwise). Mobile first: safe areas, the on-screen keyboard (visual viewport), touch actions, bottom sheet.
// Tools run by the page: the MCP servers the server forwards (web), and Python (Pyodide in a sandboxed frame in this browser)
// with a per-conversation /mnt/data for attached files and files to download (IndexedDB). Finished html / svg / mermaid code
// blocks show a sandboxed preview. Live statistics at "/live".
#pragma once

static const char * CHAT_PAGE = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,interactive-widget=resizes-content">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#141414" media="(prefers-color-scheme: dark)">
<meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default"><meta name="apple-mobile-web-app-title" content="hyper">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0' stop-color='%233a3a3a'/%3E%3Cstop offset='1' stop-color='%238a8a8a'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='32' height='32' rx='9' fill='url(%23g)'/%3E%3Cpath d='M18 4 7 18h8l-1.3 10L25 14h-8z' fill='white'/%3E%3C/svg%3E">
<title>hyper</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400..700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.10.0/highlight.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
:root{
 --bg:#ffffff;--bg2:#f7f7f7;--bg3:#eeeeee;--card:#ffffff;--fg:#141414;--fg2:#474747;--mut:#878787;--line:#e8e8e8;--line2:#d8d8d8;
 --acc:#2a2a2a;--acc2:#8a8a8a;--accfg:#2a2a2a;--accbg:#ececec;--ubub:#f1f1f1;--code:#171717;--codehd:#212121;--codefg:#e6e6e6;
 --ok:#a3a3a3;--warn:#2a2a2a;--danger:#d94141;--markfg:#ffffff;--swk:#ffffff;--sh1:0 1px 2px rgba(17,17,26,.05);--sh2:0 2px 6px rgba(17,17,26,.04),0 12px 32px rgba(17,17,26,.09);
 --hh:calc(54px + env(safe-area-inset-top));--sbw:272px;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#141414;--bg2:#1a1a1a;--bg3:#262626;--card:#1e1e1e;--fg:#ececec;--fg2:#b9b9b9;--mut:#828282;--line:#2a2a2a;--line2:#3a3a3a;
 --acc:#e2e2e2;--acc2:#8f8f8f;--accfg:#e8e8e8;--accbg:#303030;--ubub:#262626;--code:#0e0e0e;--codehd:#1b1b1b;--codefg:#e6e6e6;
 --ok:#8f8f8f;--warn:#e2e2e2;--markfg:#141414;--swk:#141414;
 --sh1:0 1px 2px rgba(0,0,0,.4);--sh2:0 2px 8px rgba(0,0,0,.3),0 14px 36px rgba(0,0,0,.4);color-scheme:dark}}
:root[data-theme=dark]{
 --bg:#141414;--bg2:#1a1a1a;--bg3:#262626;--card:#1e1e1e;--fg:#ececec;--fg2:#b9b9b9;--mut:#828282;--line:#2a2a2a;--line2:#3a3a3a;
 --acc:#e2e2e2;--acc2:#8f8f8f;--accfg:#e8e8e8;--accbg:#303030;--ubub:#262626;--code:#0e0e0e;--codehd:#1b1b1b;--codefg:#e6e6e6;
 --ok:#8f8f8f;--warn:#e2e2e2;--markfg:#141414;--swk:#141414;
 --sh1:0 1px 2px rgba(0,0,0,.4);--sh2:0 2px 8px rgba(0,0,0,.3),0 14px 36px rgba(0,0,0,.4);color-scheme:dark}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{height:100%;margin:0;overflow:hidden;overscroll-behavior:none}
body{background:var(--bg);color:var(--fg);font:15.5px/1.65 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;-webkit-text-size-adjust:100%}
#app{position:fixed;left:0;right:0;top:var(--vvt,0px);height:var(--vvh,100dvh);display:flex;overflow:hidden;background:var(--bg)}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer;padding:0}
input,textarea{font:inherit;color:inherit}
svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex-shrink:0}
.ib{width:36px;height:36px;display:inline-grid;place-items:center;border-radius:10px;color:var(--fg2);transition:background .15s,color .15s,transform .1s}
.ib:active{transform:scale(.93)}
.ib.sm{width:30px;height:30px;border-radius:8px}.ib.sm svg{width:16px;height:16px}
.sp{flex:1}
[data-i]{display:inline-flex;align-items:center;line-height:0}   /* (icon wrappers: no text line box under the svg) */
.mark{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:inline-grid;place-items:center;flex-shrink:0}
.mark svg{width:15px;height:15px;fill:var(--markfg);stroke:none}
@media (hover:hover){.ib:hover{background:var(--bg3);color:var(--fg)}
 ::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--line2);border-radius:6px;border:3px solid transparent;background-clip:padding-box}::-webkit-scrollbar-track{background:transparent}}
/* sidebar */
aside{width:var(--sbw);background:var(--bg2);border-right:1px solid var(--line);display:flex;flex-direction:column;flex-shrink:0;
 padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom);transition:margin-left .24s ease,transform .26s cubic-bezier(.2,.8,.2,1)}
.sb-top{display:flex;align-items:center;gap:8px;padding:10px 10px 6px 14px;height:54px}
.brand{flex:1;display:flex;align-items:center;gap:10px;font-weight:650;font-size:16.5px;letter-spacing:-.02em}
.newbtn{margin:6px 10px 10px;display:flex;align-items:center;gap:10px;height:42px;padding:0 12px;border-radius:12px;background:var(--card);border:1px solid var(--line);box-shadow:var(--sh1);font-weight:550;font-size:14.5px;transition:border-color .15s}
.newbtn kbd{margin-left:auto;font:500 11px Inter,sans-serif;color:var(--mut);border:1px solid var(--line);border-radius:6px;padding:1px 6px}
.search{margin:0 10px 4px;display:flex;align-items:center;gap:8px;padding:0 10px;border-radius:10px;background:var(--bg3);color:var(--mut);height:38px}
.search svg{width:16px;height:16px}
.search input{flex:1;min-width:0;border:0;background:none;outline:none;font-size:14px;color:var(--fg)}
#convs{flex:1;overflow-y:auto;padding:2px 8px 12px;overscroll-behavior:contain}
.grp{font-size:11.5px;color:var(--mut);font-weight:600;letter-spacing:.03em;padding:16px 8px 4px}
.cv{display:flex;align-items:center;gap:6px;height:40px;padding:0 4px 0 10px;border-radius:10px;cursor:pointer;font-size:14px;color:var(--fg2)}
.cv.on{background:var(--bg3);color:var(--fg);font-weight:550}
.cv span{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.cv input{flex:1;min-width:0;border:1px solid var(--acc);border-radius:8px;background:var(--bg);padding:4px 8px;font-size:16px;outline:none}
.cv .ib{width:30px;height:30px;opacity:0}.cv.on .ib{opacity:1}
.gdot{width:7px;height:7px;border-radius:50%;background:var(--acc);flex-shrink:0;animation:pulse 1.1s infinite}
.sb-foot{border-top:1px solid var(--line);padding:8px 10px;display:flex;align-items:center;gap:2px}
.sb-foot a{color:var(--fg2);text-decoration:none;display:flex;align-items:center;gap:8px;height:36px;padding:0 10px;border-radius:10px;font-size:13.5px}
@media (hover:hover){.newbtn:hover{border-color:var(--line2)}.cv:hover{background:var(--bg3);color:var(--fg)}.cv:hover .ib{opacity:1}.sb-foot a:hover{background:var(--bg3)}}
@media (hover:none){.cv .ib{opacity:.6}}
@media (min-width:769px){body.side-off aside{margin-left:calc(-1 * var(--sbw))}}
@media (max-width:768px){aside{position:absolute;left:0;top:0;bottom:0;z-index:30;width:min(86vw,320px);transform:translateX(-102%);border-right:0}
 body.side-on aside{transform:none;box-shadow:var(--sh2)}.newbtn kbd{display:none}.search input{font-size:16px}}
.scrim{position:absolute;inset:0;background:rgba(8,8,14,.42);opacity:0;pointer-events:none;transition:opacity .22s;z-index:25}
.scrim.on{opacity:1;pointer-events:auto}#scrim{z-index:39}
/* main */
main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
header{position:absolute;z-index:5;top:0;left:0;right:0;height:var(--hh);padding:env(safe-area-inset-top) 8px 0;display:flex;align-items:center;gap:2px;
 background:color-mix(in srgb,var(--bg) 82%,transparent);-webkit-backdrop-filter:saturate(1.8) blur(16px);backdrop-filter:saturate(1.8) blur(16px);border-bottom:1px solid transparent;transition:border-color .2s}
header.sc{border-bottom-color:var(--line)}
.hside{display:none}body.side-off .hside{display:inline-grid}
.model{display:flex;align-items:center;gap:8px;height:36px;padding:0 10px;border-radius:12px;font-weight:600;font-size:14.5px;min-width:0;max-width:min(56vw,380px);letter-spacing:-.01em}
.model .nm{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.model svg{width:14px;height:14px;color:var(--mut)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);flex-shrink:0;box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 18%,transparent)}
.dot.busy{background:var(--warn);box-shadow:0 0 0 3px color-mix(in srgb,var(--warn) 22%,transparent);animation:pulse 1.2s infinite}
.dot.off{background:var(--danger);box-shadow:0 0 0 3px color-mix(in srgb,var(--danger) 20%,transparent);animation:none}
@keyframes pulse{50%{opacity:.35}}
.ctx{display:flex;align-items:center;gap:6px;height:32px;padding:0 8px;border-radius:10px;font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums;white-space:nowrap}
@media (hover:hover){.model:hover,.ctx:hover{background:var(--bg3)}}
@media (max-width:768px){.hside{display:inline-grid}.hdesk{display:none}}
@media (max-width:420px){.ctx .mx{display:none}}
#scroll{flex:1;overflow-y:auto;overflow-x:hidden;overscroll-behavior-y:contain;display:flex;flex-direction:column}
.hsp{flex:0 0 var(--hh)}   /* (room under the header: a spacer, not padding -- sticky offsets count from the content edge) */
.col{flex:1 0 auto;width:100%;max-width:768px;margin:0 auto;padding:16px 20px 26px}
@media (max-width:768px){.col{padding:12px 16px 20px}}
/* empty state */
main.empty .col{display:flex;flex-direction:column;justify-content:center}
.hero{text-align:center;padding:0 8px 4vh}
.hero .mark{width:52px;height:52px;border-radius:16px;margin-bottom:18px;box-shadow:var(--sh2)}
.hero .mark svg{width:28px;height:28px}
.hero h1{font-size:28px;line-height:1.2;font-weight:650;letter-spacing:-.03em;margin:0 0 8px}
.hero p{margin:0;color:var(--mut);font-size:14.5px}
main.empty .dock::before,main.empty .jump{display:none}
@media (min-width:769px){main.empty{justify-content:center}main.empty #scroll{flex:0 0 auto;overflow:visible}main.empty .col{padding-bottom:30px}
 main.empty .sugs{order:3;margin:14px auto 0}}
/* dock and composer */
.dock{position:relative;padding:0 16px max(env(safe-area-inset-bottom),12px);display:flex;flex-direction:column}
.dock::before{content:"";position:absolute;left:0;right:0;top:-30px;height:30px;background:linear-gradient(to bottom,transparent,var(--bg));pointer-events:none}
.dock>*{width:100%;max-width:744px;margin-left:auto;margin-right:auto}
.comp{background:var(--card);border:1px solid var(--line2);border-radius:26px;box-shadow:var(--sh2);padding:12px 10px 10px 18px;transition:border-color .15s}
.comp:focus-within{border-color:color-mix(in srgb,var(--acc) 50%,var(--line2))}
.comp textarea{display:block;width:100%;border:0;outline:none;resize:none;background:none;font-size:16px;line-height:1.5;max-height:min(220px,30dvh);overflow-y:auto;padding:2px 8px 4px 0}
.comp textarea::placeholder{color:var(--mut)}
.crow{display:flex;align-items:center;gap:6px;margin:8px 0 0 -8px}
.pill{display:inline-flex;align-items:center;gap:6px;height:34px;padding:0 12px;border-radius:999px;border:1px solid var(--line2);font-size:13.5px;color:var(--fg2);transition:background .15s,color .15s,border-color .15s}
.pill.on{background:var(--accbg);border-color:transparent;color:var(--accfg)}
.pill svg{width:16px;height:16px}
.send{position:relative;width:38px;height:38px;border-radius:50%;background:var(--fg);color:var(--bg);display:grid;place-items:center;transition:opacity .15s,transform .1s}
.send:disabled{opacity:.18;cursor:default}.send:not(:disabled):active{transform:scale(.9)}
.send svg{stroke-width:2.4}
.send.stop svg{width:14px;height:14px;fill:currentColor;stroke:none}
.send.stop::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:2px solid transparent;border-top-color:var(--acc);border-right-color:var(--acc2);animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.hint{text-align:center;font-size:11.5px;color:var(--mut);margin-top:8px}
@media (hover:hover){.pill:hover{background:var(--bg2)}.pill.on:hover{background:var(--accbg)}}
@media (hover:none),(max-width:768px){.hint{display:none}}
.sugs{display:flex;gap:8px;overflow-x:auto;scrollbar-width:none;margin-bottom:10px;padding:2px 0;scroll-snap-type:x mandatory}
.sugs::-webkit-scrollbar{display:none}
.sug{flex:0 0 auto;max-width:250px;text-align:left;padding:10px 14px;border-radius:16px;border:1px solid var(--line);background:var(--card);font-size:13.5px;line-height:1.4;color:var(--mut);scroll-snap-align:start;transition:background .15s,border-color .15s}
.sug b{display:block;color:var(--fg);font-weight:600}
@media (hover:hover){.sug:hover{background:var(--bg2);border-color:var(--line2)}}
@media (min-width:769px){.sugs{display:grid;grid-template-columns:1fr 1fr;overflow:visible}.sug{max-width:none}}
@media (max-width:768px){.dock>.sugs{width:auto;max-width:none;margin:0 -16px 10px;padding:2px 16px;scroll-padding:0 16px}}
.dock>.jump{position:absolute;left:50%;top:-52px;width:38px;height:38px;margin:0;border-radius:50%;background:var(--card);border:1px solid var(--line2);box-shadow:var(--sh2);display:grid;place-items:center;color:var(--fg2);
 opacity:0;pointer-events:none;transform:translate(-50%,8px);transition:opacity .2s,transform .2s;z-index:2}
.dock>.jump.on{opacity:1;pointer-events:auto;transform:translate(-50%,0)}
/* messages */
.msg{margin:0 0 30px}
.msg.new{animation:rise .3s ease-out both}
@keyframes rise{from{opacity:0;transform:translateY(8px)}}
.msg.user{display:flex;flex-direction:column;align-items:flex-end}
.ububble{background:var(--ubub);border-radius:22px 22px 8px 22px;padding:10px 16px;max-width:min(86%,600px);white-space:pre-wrap;overflow-wrap:anywhere}
.ububble.clamp{max-height:13.5em;overflow:hidden;-webkit-mask-image:linear-gradient(#000 60%,transparent);mask-image:linear-gradient(#000 60%,transparent)}
.acts{display:flex;align-items:center;color:var(--mut);opacity:0;transition:opacity .15s}
.msg.user .acts{margin:4px -4px 0 0}
.msg:hover .acts,.msg.ai:last-child .acts{opacity:1}
@media (hover:none){.acts{opacity:1}}
.edit{width:100%;background:var(--card);border:1px solid var(--line2);border-radius:20px;padding:12px 14px 10px 16px;box-shadow:var(--sh1)}
.edit textarea{display:block;width:100%;border:0;outline:none;resize:none;background:none;font-size:16px;line-height:1.5;max-height:40dvh;overflow-y:auto}
.erow{display:flex;justify-content:flex-end;gap:8px;margin-top:8px}
.btn{height:34px;padding:0 14px;border-radius:999px;font-size:13.5px;font-weight:550;border:1px solid var(--line2);background:var(--card);white-space:nowrap}
.btn.pri{background:var(--fg);color:var(--bg);border-color:transparent}
.btn.sm{height:30px;padding:0 12px;font-size:13px}
.btn.danger{color:var(--danger)}
.btn.ghost{border:0;background:none;color:var(--mut);padding:0 8px}
@media (hover:hover){.btn:hover{background:var(--bg2)}.btn.pri:hover{background:var(--fg);opacity:.88}}
.md{overflow-wrap:break-word}
.md>*:first-child{margin-top:0}.md>*:last-child{margin-bottom:0}
.md p{margin:0 0 .9em}
.md ul,.md ol{margin:0 0 .9em;padding-left:1.45em}.md li{margin:.3em 0}.md li>p{margin:0 0 .4em}.md li::marker{color:var(--mut)}
.md h1,.md h2,.md h3,.md h4,.md h5{margin:1.45em 0 .55em;line-height:1.3;letter-spacing:-.02em;font-weight:650}
.md h1{font-size:1.5em}.md h2{font-size:1.28em}.md h3{font-size:1.1em}.md h4,.md h5{font-size:1em}
.md strong{font-weight:650}
.md a{color:var(--accfg);text-decoration:underline;text-decoration-color:color-mix(in srgb,var(--accfg) 35%,transparent);text-underline-offset:2px}
.md blockquote{margin:0 0 .9em;padding:2px 0 2px 14px;border-left:3px solid var(--line2);color:var(--fg2)}
.md hr{border:0;border-top:1px solid var(--line);margin:1.6em 0}
.md table{border-collapse:separate;border-spacing:0;margin:0 0 1em;font-size:14px;display:block;overflow-x:auto;max-width:100%;width:max-content;border:1px solid var(--line);border-radius:12px}
.md th,.md td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
.md th+th,.md td+td{border-left:1px solid var(--line)}
.md tr:last-child td{border-bottom:0}
.md th{background:var(--bg2);font-weight:600;font-size:13px}
.md :not(pre)>code{font:.86em "JetBrains Mono",ui-monospace,monospace;font-variant-ligatures:none;background:var(--bg3);padding:.14em .4em;border-radius:6px;overflow-wrap:anywhere}
.md img{max-width:100%;border-radius:12px}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:4px 0}
.cb{margin:0 0 1em;border-radius:14px;background:var(--code);border:1px solid var(--line);overflow:hidden;overflow:clip}
.cb .hd{position:sticky;top:var(--hh);z-index:1;display:flex;align-items:center;gap:2px;height:38px;padding:0 6px 0 14px;background:var(--codehd);color:#9a9a9a;font-size:12px;font-weight:500}
.cb .hd .ln{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.cb .hd button{display:flex;align-items:center;gap:6px;height:28px;color:#bdbdbd;font-size:12px;padding:0 8px;border-radius:8px}
.cb .hd svg{width:14px;height:14px}
.cb pre{margin:0;padding:14px 16px;overflow-x:auto;color:var(--codefg);font:13px/1.6 "JetBrains Mono",ui-monospace,monospace;font-variant-ligatures:none;tab-size:4}
body.wrap .cb pre{white-space:pre-wrap;overflow-wrap:anywhere}
body.wrap .cb .wrp{color:#fff;background:#333333}
.cb pre code{font:inherit;background:none;padding:0}
@media (hover:hover){.cb .hd button:hover{background:#2e2e2e;color:#fff}}
@media (max-width:768px){.cb pre{font-size:12.5px;padding:12px 14px}.cb .hd .tx{display:none}}
.think{margin:0 0 14px}
.think .th{display:inline-flex;align-items:center;gap:7px;height:32px;padding:0 10px 0 8px;margin-left:-8px;border-radius:10px;color:var(--mut);font-size:14px;user-select:none;-webkit-user-select:none}
.think .th svg{width:16px;height:16px}.think .th .chev{width:14px;height:14px;transition:transform .2s}
.think.open .th .chev{transform:rotate(90deg)}
.tsec{font-variant-numeric:tabular-nums}
.think .tb{display:none;margin-top:4px;padding:2px 0 2px 16px;border-left:2px solid var(--line2);color:var(--fg2);font-size:13.5px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
.think.open .tb{display:block;max-height:min(50dvh,480px);overflow-y:auto;overscroll-behavior:contain}
.think.live:not(.open) .tb{display:block;max-height:5em;overflow:hidden;color:var(--mut);-webkit-mask-image:linear-gradient(to bottom,transparent,#000 75%);mask-image:linear-gradient(to bottom,transparent,#000 75%)}
@media (hover:hover){.think .th:hover{background:var(--bg2);color:var(--fg2)}}
.shimmer{background:linear-gradient(90deg,var(--mut) 30%,var(--fg) 50%,var(--mut) 70%);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 1.8s linear infinite}
@keyframes sh{from{background-position:100% 0}to{background-position:-100% 0}}
.tool{margin:0 0 14px;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tool .hd{padding:8px 12px;font-size:13px;color:var(--fg2);background:var(--bg2);display:flex;gap:6px;align-items:center;font-weight:550}
.tool pre{margin:0;padding:10px 12px;font:12.5px/1.5 "JetBrains Mono",monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--fg2)}
.tcs{margin:0 0 12px}
.tc{margin:0 0 8px;border:1px solid var(--line);border-radius:12px;background:var(--bg2);overflow:hidden}
.tch{display:flex;align-items:center;gap:8px;width:100%;min-height:38px;padding:6px 12px;font-size:13.5px;color:var(--fg2);text-align:left}
.tch .tl2{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tch b{font-weight:600;color:var(--fg)}.tch svg{width:15px;height:15px}
.tch .chev{width:14px;height:14px;transition:transform .2s}.tc.open .tch .chev{transform:rotate(90deg)}
.tcb{display:none;margin:0;padding:8px 12px 12px;font:12px/1.5 "JetBrains Mono",ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--fg2);max-height:320px;overflow-y:auto;border-top:1px solid var(--line)}
.tsr{display:none;padding:4px 12px 8px;border-top:1px solid var(--line);max-height:340px;overflow-y:auto}
.tc.open .tcb,.tc.open .tsr{display:block}
.tsr a{display:block;padding:6px 0;font-size:13.5px;line-height:1.35;color:var(--fg);text-decoration:none;border-bottom:1px solid var(--line)}.tsr a:last-child{border-bottom:0}
.tsr a small{display:block;color:var(--mut);font-size:12px}
.tc.err .tch{color:var(--danger)}
.tcode{display:none}.tc.open .tcode{display:block}
.tcode .cb{margin:0;border:0;border-top:1px solid var(--line);border-radius:0}.tcode .cb .hd{position:static}
.tfig img{display:block;max-width:100%;border-radius:12px;border:1px solid var(--line);background:#fff;margin:0 0 8px}
.fchips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.fchip{display:inline-flex;align-items:center;gap:7px;max-width:100%;height:34px;padding:0 10px;border-radius:10px;border:1px solid var(--line2);background:var(--card);font-size:13px;color:var(--fg)}
.fchip svg{width:15px;height:15px;color:var(--fg2)}.fchip span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}.fchip small{color:var(--mut);white-space:nowrap}
.fchip .x{margin:0 -5px 0 0;width:24px;height:24px;display:grid;place-items:center;border-radius:6px;color:var(--mut)}
.msg.user .fchips{justify-content:flex-end;margin:6px 0 0}
.comp .fchips{margin:0 0 8px -6px}
@media (hover:hover){.fchip:hover{border-color:var(--acc2)}.fchip .x:hover{background:var(--bg3);color:var(--fg)}}
#app.drop::after{content:"Drop files to attach";position:absolute;inset:12px;z-index:80;display:grid;place-items:center;border:2px dashed var(--acc2);border-radius:20px;
 background:color-mix(in srgb,var(--bg) 88%,transparent);font-weight:600;color:var(--fg2);pointer-events:none}
/* previews of html / svg / mermaid code blocks (sandboxed frames) */
.cb.art .seg2{display:flex;background:#2a2a2a;border-radius:8px;padding:2px;margin-right:2px}
.cb.art .hd .seg2 button{height:24px;padding:0 10px;border-radius:6px;color:#9a9a9a}
.cb.art.pv .hd .art-p,.cb.art:not(.pv) .hd .art-c{background:#454545;color:#fff}
.cb.art.pv pre,.cb.art:not(.pv) .artv{display:none}
.artv{background:#fff}.artv iframe{display:block;width:100%;height:360px;border:0;background:#fff}
.cb.art.full{position:fixed;inset:0;z-index:70;margin:0;border:0;border-radius:0;display:flex;flex-direction:column}
.cb.art.full .hd{position:static;height:auto;min-height:44px;padding-top:env(safe-area-inset-top)}
.cb.art.full .artv,.cb.art.full pre{flex:1;min-height:0;overflow:auto}.cb.art.full .artv iframe{height:100%!important}
@media (max-width:420px){#webb .pt,#codeb .pt{display:none}#webb,#codeb{padding:0 9px}}
.msg.ai.round{margin-bottom:6px}
.livebar{display:flex;align-items:center;flex-wrap:wrap;gap:6px 10px;min-height:24px;margin-top:8px;font-size:12.5px;color:var(--mut);font-variant-numeric:tabular-nums}
.livebar:empty{display:none}
.pd{width:7px;height:7px;border-radius:50%;background:var(--acc);animation:pulse 1s infinite}
.prog{width:110px;height:4px;border-radius:2px;background:var(--bg3);overflow:hidden}
.prog i{display:block;height:100%;width:0;border-radius:2px;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .45s ease}
.prog.ind i{width:35%;transition:none;animation:ind 1.3s ease-in-out infinite}
@keyframes ind{from{transform:translateX(-100%)}to{transform:translateX(290%)}}
.foot{display:flex;align-items:center;flex-wrap:wrap;gap:2px 6px;margin:8px 0 0 -6px;min-height:30px}
.pager{display:inline-flex;align-items:center;font-size:12.5px;color:var(--mut);font-variant-numeric:tabular-nums;margin-left:2px}
.pager button{width:24px;height:28px;display:grid;place-items:center;border-radius:7px}.pager button:disabled{opacity:.3;cursor:default}
.pager svg{width:14px;height:14px}
.tag{font-size:12px;color:var(--mut);padding:1px 9px;border:1px solid var(--line);border-radius:999px}
.stat{display:flex;align-items:center;flex-wrap:wrap;gap:2px 12px;font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums;padding:4px 6px;border-radius:8px;text-align:left}
.stat span{display:inline-flex;align-items:center;gap:4px}.stat svg{width:13px;height:13px}
.stat .x{display:none}.stat.more .x{display:inline-flex}
@media (min-width:769px){.stat .x{display:inline-flex}}
@media (hover:hover){.pager button:not(:disabled):hover{background:var(--bg3)}.stat:hover{color:var(--fg2)}}
.err{display:flex;align-items:center;gap:10px;color:var(--danger);background:color-mix(in srgb,var(--danger) 8%,transparent);border:1px solid color-mix(in srgb,var(--danger) 22%,transparent);border-radius:12px;padding:8px 8px 8px 12px;font-size:14px;margin-top:10px}
.err span{flex:1;overflow-wrap:anywhere}
/* settings sheet */
.sheet{position:absolute;z-index:40;background:var(--bg);display:flex;flex-direction:column;transition:transform .3s cubic-bezier(.2,.8,.2,1)}
@media (min-width:769px){.sheet{top:0;right:0;bottom:0;width:400px;border-left:1px solid var(--line);box-shadow:var(--sh2);transform:translateX(105%)}}
@media (max-width:768px){.sheet{left:0;right:0;bottom:0;max-height:90%;border-radius:22px 22px 0 0;box-shadow:0 -10px 40px rgba(0,0,0,.2);transform:translateY(105%);padding-bottom:env(safe-area-inset-bottom)}.grab{display:block!important}}
.sheet.on{transform:none}
.grab{display:none;width:40px;height:5px;border-radius:3px;background:var(--line2);margin:8px auto 0;flex-shrink:0}
.sh-hd{display:flex;align-items:center;padding:8px 10px 4px 20px;font-weight:650;font-size:17px;letter-spacing:-.01em;flex-shrink:0;touch-action:none}
.sh-bd{overflow-y:auto;padding:4px 16px 24px;overscroll-behavior:contain}
.sec{margin:18px 4px 6px;font-size:11.5px;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.07em}
.card{background:var(--bg2);border:1px solid var(--line);border-radius:16px;padding:2px 14px}
.row{display:flex;align-items:center;gap:12px;min-height:50px;padding:8px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row .lb{flex:1;font-size:14.5px;min-width:0}
.row .lb small{display:block;color:var(--mut);font-size:12.5px;line-height:1.4}
.row .val{font-size:13px;color:var(--fg2);font-variant-numeric:tabular-nums}
.row.v{flex-direction:column;align-items:stretch;gap:4px}
.row.v .top{display:flex;align-items:center}
.card textarea{width:100%;min-height:96px;resize:vertical;border:1px solid var(--line);background:var(--bg);border-radius:12px;padding:9px 11px;font-size:16px;line-height:1.45;outline:none}
.card input[type=number]{width:104px;border:1px solid var(--line);background:var(--bg);border-radius:10px;padding:6px 10px;font-size:16px;text-align:right;outline:none}
.card textarea:focus,.card input:focus{border-color:var(--acc)}
input[type=range]{width:100%;accent-color:var(--acc);height:28px;margin:0}
.sw{position:relative;width:46px;height:28px;flex-shrink:0}
.sw input{position:absolute;opacity:0;width:0;height:0}
.sw i{position:absolute;inset:0;border-radius:14px;background:var(--line2);transition:background .2s;cursor:pointer}
.sw i::after{content:"";position:absolute;top:3px;left:3px;width:22px;height:22px;border-radius:50%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.3);transition:transform .2s}
.sw input:checked+i{background:var(--acc)}.sw input:checked+i::after{transform:translateX(18px);background:var(--swk)}
.seg{display:flex;background:var(--bg3);border-radius:10px;padding:3px;gap:2px;width:210px}
.seg button{flex:1;height:30px;border-radius:8px;font-size:13px;color:var(--fg2)}
.seg button.on{background:var(--card);color:var(--fg);box-shadow:var(--sh1);font-weight:550}
.srv{display:flex;gap:12px;align-items:center;padding:12px 0}
.srv .mark{width:38px;height:38px;border-radius:12px}.srv .mark svg{width:20px;height:20px}
.srv .nm{font-weight:600;font-size:14.5px;overflow-wrap:anywhere;line-height:1.35}
.srv small{display:block;color:var(--mut);font-size:12.5px;line-height:1.45}
.lnk{color:var(--accfg);text-decoration:none;font-size:13.5px;font-weight:550;white-space:nowrap}
/* menu, toast */
.menu{position:absolute;z-index:50;min-width:176px;background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--sh2);padding:5px;opacity:0;transform:scale(.96);transform-origin:top right;pointer-events:none;transition:opacity .12s,transform .12s}
.menu.on{opacity:1;transform:none;pointer-events:auto}
.menu button{display:flex;align-items:center;gap:10px;width:100%;height:40px;padding:0 10px;border-radius:9px;font-size:14px;text-align:left}
.menu button.danger{color:var(--danger)}
.menu button small{margin-left:auto;padding-left:12px;color:var(--mut);font-size:12px}.menu button.sel{font-weight:650}
#effmenu{transform-origin:bottom left}.menu svg{width:16px;height:16px}
@media (hover:hover){.menu button:hover{background:var(--bg3)}}
#toast{position:absolute;z-index:60;left:50%;bottom:calc(var(--dockh,110px) + 14px);transform:translate(-50%,10px);opacity:0;pointer-events:none;display:flex;align-items:center;gap:14px;
 background:var(--fg);color:var(--bg);padding:9px 16px;border-radius:12px;font-size:13.5px;box-shadow:var(--sh2);transition:opacity .2s,transform .2s;max-width:calc(100% - 32px);white-space:nowrap}
#toast span{overflow:hidden;text-overflow:ellipsis}
#toast.on{opacity:1;transform:translate(-50%,0);pointer-events:auto}
#toast button{color:var(--accbg);font-weight:650;margin-right:-4px;padding:2px 4px}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
.hljs{color:#e4e6ee}.hljs-keyword,.hljs-selector-tag,.hljs-built_in{color:#c792ea}.hljs-string,.hljs-attr{color:#c3e88d}.hljs-number,.hljs-literal{color:#f78c6c}
.hljs-comment{color:#6b7089;font-style:italic}.hljs-title,.hljs-function .hljs-title,.hljs-title.function_{color:#82aaff}.hljs-type,.hljs-class .hljs-title{color:#ffcb6b}
.hljs-variable,.hljs-params{color:#e4e6ee}.hljs-meta{color:#89ddff}.hljs-symbol,.hljs-property{color:#f07178}
</style></head><body>
<div id="app">
<aside id="side">
 <div class="sb-top"><div class="brand"><span class="mark" data-i="boltf"></span>hyper</div><button class="ib" aria-label="Close sidebar" onclick="side(false)" data-i="side"></button></div>
 <button class="newbtn" onclick="newConv()"><span data-i="newc"></span>New chat<kbd>Ctrl ⇧ O</kbd></button>
 <label class="search"><span data-i="search"></span><input id="q" placeholder="Search chats" autocomplete="off" oninput="renderSide()"></label>
 <div id="convs"></div>
 <div class="sb-foot"><a href="/live" target="_blank" rel="noopener"><span data-i="pulse"></span>Live stats</a><span class="sp"></span>
  <button class="ib" id="themeb" aria-label="Theme" onclick="cycleTheme()"></button><button class="ib" aria-label="Settings" onclick="openSet(true)" data-i="sliders"></button></div>
</aside>
<div class="scrim" id="sscrim" onclick="side(false)"></div>
<main id="main">
 <header id="hdr">
  <button class="ib hside" aria-label="Open sidebar" onclick="side(true)" data-i="menu"></button>
  <button class="model" onclick="openSet(true)" aria-label="Model and settings"><span class="dot" id="dot"></span><span class="nm" id="mdl">hyper</span><span data-i="chevd"></span></button>
  <span class="sp"></span>
  <button class="ctx" id="ctx" onclick="ctxInfo()" aria-label="Context in use"></button>
  <button class="ib hside" aria-label="New chat" onclick="newConv()" data-i="newc"></button>
  <button class="ib hdesk" aria-label="Settings" onclick="openSet(true)" data-i="sliders"></button>
 </header>
 <div id="scroll"><div class="hsp"></div><div class="col" id="col"></div></div>
 <div class="dock" id="dock">
  <button class="jump" id="jump" aria-label="Scroll to the latest message" onclick="toBottom(true)" data-i="down"></button>
  <div class="sugs" id="sugs"></div>
  <div class="comp"><div class="fchips" id="atts" style="display:none"></div><textarea id="in" rows="1" placeholder="Message hyper" aria-label="Message"></textarea>
   <div class="crow"><button class="ib sm" id="attb" aria-label="Attach files" title="Attach files" onclick="$('fpick').click()" data-i="clip"></button><button class="pill" id="thinkb" onclick="effortMenu(event)" aria-label="Thinking effort"><span data-i="brain"></span><span id="effl">High</span></button><button class="pill" id="webb" onclick="toggleWeb()" style="display:none" aria-label="Web search"><span data-i="globe"></span><span class="pt">Web</span></button><button class="pill" id="codeb" onclick="toggleCode()" aria-label="Python"><span data-i="term"></span><span class="pt">Python</span></button><span class="sp"></span>
    <button class="send" id="send" onclick="sendMsg()" aria-label="Send"></button></div></div>
  <div class="hint" id="hint"></div>
  <input type="file" id="fpick" multiple hidden>
 </div>
</main>
<div class="scrim" id="scrim" onclick="openSet(false)"></div>
<div class="sheet" id="sheet" role="dialog" aria-label="Settings">
 <div class="grab"></div>
 <div class="sh-hd" id="shhd"><span class="sp">Settings</span><button class="ib" aria-label="Close" onclick="openSet(false)" data-i="x"></button></div>
 <div class="sh-bd">
  <div class="card"><div class="srv"><span class="mark" data-i="boltf"></span><div style="flex:1;min-width:0"><div class="nm" id="s-model">hyper</div><small id="s-state">connecting…</small></div>
   <a class="lnk" href="/live" target="_blank" rel="noopener">Live stats</a></div></div>
  <div class="sec">Response</div>
  <div class="card">
   <div class="row v"><div class="top"><div class="lb">Thinking effort<small>Changing it in a long conversation re-reads the conversation once (the setting is at the start of the prompt)</small></div></div>
    <div class="seg" id="s-effort" style="width:100%"><button data-e="off">Off</button><button data-e="low">Low</button><button data-e="medium">Medium</button><button data-e="high">High</button></div></div>
   <div class="row" id="s-webrow" style="display:none"><div class="lb">Web search &amp; browsing<small id="s-webtools"></small></div><label class="sw"><input type="checkbox" id="s-web"><i></i></label></div>
   <div class="row"><div class="lb">Python &amp; files<small>Code runs in a sandbox in this browser (Pyodide): calculations, charts, documents. Attached files and the files it makes are kept in this browser</small></div><label class="sw"><input type="checkbox" id="s-code"><i></i></label></div>
   <div class="row v"><div class="lb">System prompt</div><textarea id="sys" placeholder="You are a helpful assistant."></textarea></div>
  </div>
  <div class="sec">Sampling</div>
  <div class="card">
   <div class="row v"><div class="top"><div class="lb">Temperature</div><span class="val" id="tempv"></span></div><input type="range" id="temp" min="0" max="2" step="0.05"></div>
   <div class="row v"><div class="top"><div class="lb">Top-p</div><span class="val" id="toppv"></span></div><input type="range" id="topp" min="0.05" max="1" step="0.05"></div>
   <div class="row"><div class="lb">Top-k<small>0 = off, empty = server default</small></div><input type="number" id="topk" min="0" step="1" inputmode="numeric"></div>
   <div class="row"><div class="lb">Min-p<small>Empty = server default</small></div><input type="number" id="minp" min="0" max="1" step="0.01" inputmode="decimal"></div>
   <div class="row"><div class="lb">Max new tokens<small>Empty = up to the context size</small></div><input type="number" id="maxt" min="16" step="256" inputmode="numeric" placeholder="no limit"></div>
   <div class="row"><div class="lb"><small id="s-def"></small></div><button class="btn sm" onclick="resetSampling()">Use defaults</button></div>
  </div>
  <div class="sec">Interface</div>
  <div class="card">
   <div class="row"><div class="lb">Theme</div><div class="seg" id="theme"><button data-v="system">Auto</button><button data-v="light">Light</button><button data-v="dark">Dark</button></div></div>
   <div class="row"><div class="lb">Enter sends<small>Off: Enter adds a new line, Ctrl+Enter or the button sends</small></div><label class="sw"><input type="checkbox" id="s-enter"><i></i></label></div>
   <div class="row"><div class="lb">Wrap long code lines</div><label class="sw"><input type="checkbox" id="s-wrap"><i></i></label></div>
  </div>
  <div class="sec">Data</div>
  <div class="card">
   <div class="row"><div class="lb">Export chats<small>All conversations as a JSON file</small></div><button class="btn sm" onclick="exportAll()">Export</button></div>
   <div class="row"><div class="lb">Delete all chats<small>Chats are stored in this browser only</small></div><button class="btn sm danger" onclick="clearAll()">Delete</button></div>
  </div>
 </div>
</div>
<div class="menu" id="effmenu"><button data-e="off" onclick="setEffort('off')">Off<small>answer right away</small></button><button data-e="low" onclick="setEffort('low')">Low<small>brief thinking</small></button><button data-e="medium" onclick="setEffort('medium')">Medium</button><button data-e="high" onclick="setEffort('high')">High<small>the model's default, most careful</small></button></div>
<div class="menu" id="menu"><button onclick="menuAct('ren')"><span data-i="pencil"></span>Rename</button><button class="danger" onclick="menuAct('del')"><span data-i="trash"></span>Delete</button></div>
<div id="toast" role="status"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const S={get(k,d){try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v)}catch(e){return d}},
 set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){toast('Browser storage is full: export and delete old chats')}}};
const sv=(p,a)=>`<svg viewBox="0 0 24 24"${a||''}>${p}</svg>`;
const IC={menu:sv('<path d="M4 8h16M4 16h11"/>'),side:sv('<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/>'),
 newc:sv('<path d="M12 4H7a3 3 0 0 0-3 3v10a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-5"/><path d="M18.4 2.6a2 2 0 0 1 3 3L12.5 14.5 9 15.5l1-3.5z"/>'),
 search:sv('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>'),pulse:sv('<path d="M3 12h4l3-8 4 16 3-8h4"/>'),
 sliders:sv('<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>'),x:sv('<path d="M18 6 6 18M6 6l12 12"/>'),
 copy:sv('<rect x="9" y="9" width="12" height="12" rx="2.5"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>'),check:sv('<path d="m5 12 5 5L20 7"/>'),
 regen:sv('<path d="M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16M3 21v-5h5"/>'),
 pencil:sv('<path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4z"/>'),trash:sv('<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>'),
 chev:sv('<path d="m9 6 6 6-6 6"/>',' class="chev"'),chl:sv('<path d="m15 6-6 6 6 6"/>'),chr:sv('<path d="m9 6 6 6-6 6"/>'),chevd:sv('<path d="m6 9 6 6 6-6"/>'),
 brain:sv('<path d="M9 3a3 3 0 0 0-3 3v.2A3 3 0 0 0 4 9a3 3 0 0 0 .8 2A3 3 0 0 0 4 13a3 3 0 0 0 2 2.8V16a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM15 3a3 3 0 0 1 3 3v.2A3 3 0 0 1 20 9a3 3 0 0 1-.8 2 3 3 0 0 1 .8 2 3 3 0 0 1-2 2.8V16a3 3 0 0 1-6 0"/>'),
 tool:sv('<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/>'),
 bolt:sv('<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>'),boltf:sv('<path d="M13.6 2 4.5 13.6H11L10 22l9.5-12H13z"/>'),
 inbox:sv('<path d="M4 13h4l2 3h4l2-3h4"/><path d="M4 13l2-8h12l2 8v6H4z"/>'),clock:sv('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
 db:sv('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'),
 send:sv('<path d="M12 19V5M5 12l7-7 7 7"/>'),stop:sv('<rect x="5" y="5" width="14" height="14" rx="3"/>'),down:sv('<path d="M12 5v14M5 12l7 7 7-7"/>'),
 sun:sv('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
 moon:sv('<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>'),auto:sv('<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" style="fill:currentColor"/>'),
 wrap:sv('<path d="M3 6h18M3 12h14a3 3 0 0 1 0 6h-4"/><path d="m15 16-2 2 2 2M3 18h6"/>'),
 more:sv('<circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/>',' style="fill:currentColor;stroke:none"'),
 alert:sv('<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5v.01"/>'),
 globe:sv('<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>'),
 term:sv('<rect x="3" y="4" width="18" height="16" rx="3"/><path d="m7.5 9.5 3 2.5-3 2.5M13 15h4"/>'),
 clip:sv('<path d="m20.5 11.5-8.4 8.4a5.2 5.2 0 0 1-7.4-7.4l8.4-8.4a3.5 3.5 0 0 1 4.9 4.9l-8.4 8.4a1.7 1.7 0 0 1-2.5-2.5l7.8-7.8"/>'),
 file:sv('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'),
 dl:sv('<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>'),expand:sv('<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>')};
document.querySelectorAll('[data-i]').forEach(e=>e.insertAdjacentHTML('beforeend',IC[e.dataset.i]));
const touch=matchMedia('(hover:none)').matches,narrow=()=>innerWidth<=768;
let convs=S.get('hyper.convs',[]).filter(c=>c.msgs&&c.msgs.length),cur=S.get('hyper.cur',null),busy=null,gen=null,live={},defaults={},online=true,editing=-1,renaming=null,menuId=null;
const cfg=Object.assign({sys:'',temp:'',topp:'',topk:'',minp:'',maxt:'',think:true,web:true,code:true,theme:'system',enter:!touch,wrap:false},S.get('hyper.cfg',{}));
if(cfg.maxt===16384)cfg.maxt='';
if(!cfg.effort)cfg.effort=cfg.think===false?'off':'high';   // (thinking effort: off / low / medium / high = the model's default)
const MCP={tools:[],map:{},servers:[]};   // tools of the MCP servers the server forwards to (--mcp)
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtS=s=>{if(s<10)return s.toFixed(1)+'s';s=Math.round(s);return s<60?s+'s':Math.floor(s/60)+'m '+s%60+'s'};
const fmtK=n=>n<1000?String(n):n<1e5?(n/1000).toFixed(1)+'k':Math.round(n/1000)+'k';
const num=v=>v==null||v===''?'–':String(+(+v).toFixed(3));
const IN=$('in');
function saveCfg(){S.set('hyper.cfg',cfg)}
function clean(m){const o=Object.assign({},m);for(const f of ['thinking','phase','pp','thinkStart','open'])delete o[f];return o}
function save(){S.set('hyper.convs',convs.map(c=>Object.assign({},c,{msgs:c.msgs.map(clean)})));S.set('hyper.cur',cur)}
function conv(){return convs.find(c=>c.id===cur)}
/* ---------- viewport (on-screen keyboard), theme, layout ---------- */
const VV=window.visualViewport;let atBottom=true;
function fitVV(){if(VV&&Math.abs(VV.scale-1)>.01)return;const r=document.documentElement.style;
 r.setProperty('--vvh',(VV?VV.height:innerHeight)+'px');r.setProperty('--vvt',(VV?VV.offsetTop:0)+'px');if(atBottom)requestAnimationFrame(()=>toBottom())}
if(VV){VV.addEventListener('resize',fitVV);VV.addEventListener('scroll',fitVV)}
addEventListener('resize',()=>{fitVV();if(!narrow()){document.body.classList.remove('side-on');$('sscrim').classList.remove('on')}});
function applyTheme(){const r=document.documentElement;if(cfg.theme==='system')r.removeAttribute('data-theme');else r.setAttribute('data-theme',cfg.theme);
 document.querySelectorAll('#theme button').forEach(b=>b.classList.toggle('on',b.dataset.v===cfg.theme));
 $('themeb').innerHTML=IC[{system:'auto',light:'sun',dark:'moon'}[cfg.theme]];$('themeb').title='Theme: '+cfg.theme;
 const bg=getComputedStyle(r).getPropertyValue('--bg').trim();document.querySelectorAll('meta[name=theme-color]').forEach(m=>m.content=bg)}
function cycleTheme(){const o=['system','light','dark'];cfg.theme=o[(o.indexOf(cfg.theme)+1)%3];saveCfg();applyTheme();toast('Theme: '+{system:'automatic',light:'light',dark:'dark'}[cfg.theme])}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',applyTheme);
document.querySelectorAll('#theme button').forEach(b=>b.onclick=()=>{cfg.theme=b.dataset.v;saveCfg();applyTheme()});
function applyWrap(){document.body.classList.toggle('wrap',!!cfg.wrap)}
function side(on){if(narrow()){document.body.classList.toggle('side-on',on);$('sscrim').classList.toggle('on',on);if(on)IN.blur()}
 else{document.body.classList.toggle('side-off',!on);S.set('hyper.side',!on)}}
if(S.get('hyper.side',false))document.body.classList.add('side-off');
function openSet(on){if(on){if(narrow())side(false);IN.blur();syncSet();updState()}$('sheet').classList.toggle('on',on);$('scrim').classList.toggle('on',on)}
/* ---------- settings ---------- */
function syncSet(){const dT=defaults.temp??1,dP=defaults.top_p??.95;
 $('sys').value=cfg.sys;$('maxt').value=cfg.maxt;$('topk').value=cfg.topk;$('minp').value=cfg.minp;
 $('topk').placeholder=defaults.top_k??'';$('minp').placeholder=defaults.min_p!=null?num(defaults.min_p):'';
 $('temp').value=cfg.temp===''?dT:cfg.temp;$('tempv').textContent=cfg.temp===''?(+dT).toFixed(2)+' · default':(+cfg.temp).toFixed(2);
 $('topp').value=cfg.topp===''?dP:cfg.topp;$('toppv').textContent=cfg.topp===''?(+dP).toFixed(2)+' · default':(+cfg.topp).toFixed(2);
 $('effl').textContent={off:'Off',low:'Low',medium:'Medium',high:'High'}[cfg.effort]||'High';
 document.querySelectorAll('#s-effort button,#effmenu button').forEach(b=>{b.classList.toggle('on',b.dataset.e===cfg.effort);b.classList.toggle('sel',b.dataset.e===cfg.effort)});$('s-web').checked=!!cfg.web;$('webb').classList.toggle('on',!!cfg.web);$('s-code').checked=!!cfg.code;$('codeb').classList.toggle('on',!!cfg.code);$('s-enter').checked=!!cfg.enter;$('s-wrap').checked=!!cfg.wrap;$('thinkb').classList.toggle('on',!!cfg.think);
 $('s-def').textContent=`Server defaults: temperature ${num(defaults.temp)} · top-p ${num(defaults.top_p)} · top-k ${num(defaults.top_k)} · min-p ${num(defaults.min_p)}`}
$('sys').oninput=()=>{cfg.sys=$('sys').value;saveCfg()};
$('maxt').onchange=()=>{const v=$('maxt').value.trim();cfg.maxt=v===''?'':Math.max(16,parseInt(v)||16);saveCfg();syncSet()};
$('s-web').onchange=()=>{cfg.web=$('s-web').checked;saveCfg();syncSet()};
$('s-code').onchange=()=>{cfg.code=$('s-code').checked;saveCfg();syncSet()};
$('temp').oninput=()=>{cfg.temp=$('temp').value;saveCfg();syncSet()};$('topp').oninput=()=>{cfg.topp=$('topp').value;saveCfg();syncSet()};
$('topk').onchange=()=>{const v=$('topk').value.trim();cfg.topk=v===''?'':String(Math.max(0,parseInt(v)||0));saveCfg();syncSet()};
$('minp').onchange=()=>{const v=$('minp').value.trim();cfg.minp=v===''?'':String(Math.min(1,Math.max(0,+v||0)));saveCfg();syncSet()};
document.querySelectorAll('#s-effort button').forEach(b=>b.onclick=()=>setEffort(b.dataset.e));
$('s-enter').onchange=()=>{cfg.enter=$('s-enter').checked;saveCfg();updSend()};
$('s-wrap').onchange=()=>{cfg.wrap=$('s-wrap').checked;saveCfg();applyWrap()};
function resetSampling(){cfg.temp=cfg.topp=cfg.topk=cfg.minp='';saveCfg();syncSet();toast('Using the server defaults')}
function effortMenu(e){e.stopPropagation();const r=e.currentTarget.getBoundingClientRect(),a=$('app').getBoundingClientRect(),m=$('effmenu');
 m.style.left=Math.max(8,r.left-a.left)+'px';m.style.top='auto';m.style.bottom=(a.bottom-r.top+6)+'px';m.classList.add('on')}
function setEffort(v){$('effmenu').classList.remove('on');if(cfg.effort===v)return;cfg.effort=v;saveCfg();syncSet();
 const c=conv(),long=c&&c.msgs.reduce((n,m)=>n+(m.content||'').length+(m.reasoning||'').length,0)>20000;
 toast('Thinking: '+{off:'off',low:'low',medium:'medium',high:'high'}[v]+(long?' · this chat is re-read once':''),null,null,long?3000:1600)}
function updState(){$('s-model').textContent=live.model||'hyper';
 $('s-state').textContent=!online?'Server not reachable':(busy||(live.phase&&live.phase!=='idle')?(live.phase==='prompt'?'Reading a prompt':'Generating'):'Idle')+
  (live.ctx_max?` · context ${fmtK(live.ctx_used||0)} / ${fmtK(live.ctx_max)}`:'')+(live.requests!=null?` · ${live.requests} requests`:'')}
function exportAll(){const b=new Blob([JSON.stringify(convs.map(c=>Object.assign({},c,{msgs:c.msgs.map(clean)})),null,1)],{type:'application/json'});
 const u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=`hyper-chats-${new Date().toISOString().slice(0,10)}.json`;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),3000)}
function clearAll(){if(busy)return toast('Stop the response first');if(!convs.length)return toast('No chats to delete');if(!confirm(`Delete all ${convs.length} chats?`))return;
 const old=convs;convs=[];cur=null;save();render();openSet(false);toast('All chats deleted','Undo',()=>{convs=old;save();render()},7000)}
/* ---------- toast, clipboard ---------- */
let tT=0;
function toast(msg,act,fn,ms){const t=$('toast');t.innerHTML=`<span>${esc(msg)}</span>`+(act?`<button>${esc(act)}</button>`:'');
 if(act)t.querySelector('button').onclick=()=>{t.classList.remove('on');fn()};t.classList.add('on');clearTimeout(tT);tT=setTimeout(()=>t.classList.remove('on'),ms||(act?5000:1600))}
function legacyCopy(t){const a=document.createElement('textarea');a.value=t;a.setAttribute('readonly','');a.style.cssText='position:fixed;top:0;left:0;opacity:0;font-size:16px';
 document.body.appendChild(a);a.select();a.setSelectionRange(0,t.length);let ok=false;try{ok=document.execCommand('copy')}catch(e){}a.remove();return ok}
function copyText(t){const p=navigator.clipboard&&window.isSecureContext?navigator.clipboard.writeText(t).then(()=>true,()=>legacyCopy(t)):Promise.resolve(legacyCopy(t));
 return p.then(ok=>{toast(ok?'Copied':'Copy failed');return ok})}
function flash(b,html){b.innerHTML=IC.check+(html||'');setTimeout(()=>b.innerHTML=IC.copy+(html||''),1400)}
function cpMsg(k,b){const m=conv().msgs[k];copyText(m.content||'').then(ok=>ok&&flash(b))}
/* ---------- sidebar ---------- */
function newConv(){editing=-1;cur=null;save();render();if(narrow())side(false);if(!touch)IN.focus()}
function pick(id){if(renaming)return;editing=-1;cur=id;save();render();toBottom();if(narrow())side(false)}
function delConv(id){if(gen&&gen.c.id===id)return toast('Stop the response first');const i=convs.findIndex(c=>c.id===id);if(i<0)return;
 const [c]=convs.splice(i,1),was=cur===id;if(was)cur=null;save();render();toast('Chat deleted','Undo',()=>{convs.splice(i,0,c);if(was)cur=c.id;save();render()})}
function convMenu(id,e){e.stopPropagation();menuId=id;const r=e.currentTarget.getBoundingClientRect(),a=$('app').getBoundingClientRect(),m=$('menu');
 m.style.top=Math.min(r.bottom+4-a.top,a.height-100)+'px';m.style.left=Math.max(8,Math.min(r.right-176-a.left,a.width-184))+'px';m.classList.add('on')}
function menuAct(a){$('menu').classList.remove('on');if(a==='ren'){renaming=menuId;renderSide()}else delConv(menuId)}
function renderSide(){const q=$('q').value.trim().toLowerCase(),now=new Date(),d0=new Date(now.getFullYear(),now.getMonth(),now.getDate()).getTime();
 const groups=[['Today',x=>x>=d0],['Yesterday',x=>x>=d0-864e5],['Previous 7 days',x=>x>=d0-7*864e5],['Previous 30 days',x=>x>=d0-30*864e5],['Older',()=>true]];
 const list=convs.filter(c=>!q||c.title.toLowerCase().includes(q)||c.msgs.some(m=>(m.content||'').toLowerCase().includes(q)));let h='';const used=new Set();
 for(const [g,f] of groups){const items=list.filter(c=>!used.has(c.id)&&f(c.t||0));if(!items.length)continue;h+=`<div class="grp">${g}</div>`;
  for(const c of items){used.add(c.id);h+=`<div class="cv${c.id===cur?' on':''}" onclick="pick('${c.id}')">`+(renaming===c.id?`<input id="ren" value="${esc(c.title)}" onclick="event.stopPropagation()">`:
   `<span>${esc(c.title)}</span>`)+(gen&&gen.c.id===c.id?'<i class="gdot"></i>':'')+`<button class="ib" aria-label="Chat options" onclick="convMenu('${c.id}',event)">${IC.more}</button></div>`}}
 $('convs').innerHTML=h||`<div class="grp" style="text-align:center;padding-top:34px;font-weight:500">${q?'No matches':'No chats yet'}</div>`;
 const i=$('ren');if(i){i.focus();i.select();let done=false;
  const fin=ok=>{if(done)return;done=true;const c=convs.find(x=>x.id===renaming);if(ok&&c&&i.value.trim()){c.title=i.value.trim();save();if(c.id===cur)document.title=c.title+' · hyper'}renaming=null;renderSide()};
  i.onkeydown=e=>{if(e.key==='Enter')fin(true);if(e.key==='Escape')fin(false)};i.onblur=()=>fin(true)}}
/* ---------- markdown ---------- */
const HL=new Map();let mdLive=false;
function codeBlock(code,lang){const key=lang+'\u0001'+code;let hl=HL.get(key);
 if(hl==null){hl=esc(code);if(window.hljs&&code.length<300000){try{if(lang&&hljs.getLanguage(lang))hl=hljs.highlight(code,{language:lang,ignoreIllegals:true}).value;
   else if(!mdLive)hl=hljs.highlightAuto(code).value}catch(e){}}
  if(!mdLive||lang){if(HL.size>256)HL.clear();HL.set(key,hl)}}
 const ak=!mdLive&&artKind(lang,code);
 if(ak){let a=ART.get(code);if(!a){if(ART.size>200){ART.clear();ARTI.clear()}a={id:'a'+(++artN),kind:ak,code};ART.set(code,a);ARTI.set(a.id,a)}
  const pv=ak!=='html'||/<!doctype|<html|<body|<script|<svg|<canvas|<style/i.test(code);   // (a whole page opens as a preview, a snippet as code)
  return `<div class="cb art${pv?' pv':''}" data-art="${a.id}"><div class="hd"><span class="ln">${esc(lang)}</span><span class="seg2"><button class="art-p">Preview</button><button class="art-c">Code</button></span>`+
   `<button class="art-f" aria-label="Full screen" title="Full screen">${IC.expand}</button><button class="cpy" aria-label="Copy code">${IC.copy}<span class="tx">Copy</span></button></div>`+
   `<div class="artv"></div><pre><code class="hljs">${hl}</code></pre></div>`}
 return `<div class="cb"><div class="hd"><span class="ln">${esc(lang||'code')}</span><button class="wrp" aria-label="Wrap lines" title="Wrap lines">${IC.wrap}</button>`+
  `<button class="cpy" aria-label="Copy code">${IC.copy}<span class="tx">Copy</span></button></div><pre><code class="hljs">${hl}</code></pre></div>`}
// previews: finished html / svg / mermaid blocks render in a sandboxed frame (opaque origin: no access to this page, its storage or
// the server; the CSP allows only libraries from the CDN and no other network). The frame reports its height.
const ART=new Map(),ARTI=new Map();let artN=0;
function artKind(lang,code){lang=(lang||'').toLowerCase();
 if(lang==='html'||lang==='htm')return /<[a-z!]/i.test(code)?'html':'';
 if(lang==='svg'||(lang==='xml'&&/^\s*<svg[\s>]/i.test(code)))return 'svg';
 return lang==='mermaid'?'mermaid':''}
const ART_CSP="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "+
 "font-src https://fonts.gstatic.com https://cdn.jsdelivr.net data:; img-src data: blob:; media-src data: blob:; connect-src https://cdn.jsdelivr.net; worker-src blob:";
function artDoc(kind,code){
 const h=`<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${ART_CSP}"><meta name="viewport" content="width=device-width,initial-scale=1">`+
  `<script>(()=>{const p=()=>{const b=document.body;if(!b)return;const s=getComputedStyle(b);parent.postMessage({hyperArt:Math.ceil(Math.max(b.scrollHeight,b.offsetHeight)+parseFloat(s.marginTop)+parseFloat(s.marginBottom))},'*')};`+
  `addEventListener('load',()=>{p();new ResizeObserver(p).observe(document.body)});setTimeout(p,300);setTimeout(p,1500)})()<\/script>`;
 if(kind==='svg')return h+`<style>body{margin:0;padding:12px;display:flex;justify-content:center;background:#fff}svg{max-width:100%;height:auto}</style></head><body>${code}</body></html>`;
 if(kind==='mermaid')return h+`<style>body{margin:0;padding:16px;display:flex;justify-content:center;background:#fff}</style></head><body><pre class="mermaid">${esc(code)}</pre>`+
  `<script type="module">import m from 'https://cdn.jsdelivr.net/npm/mermaid@12/dist/mermaid.esm.min.mjs';m.initialize({startOnLoad:false});await m.run()<\/script></body></html>`;
 return h+code}
function mountArts(root){(root||$('col')).querySelectorAll('.cb.art.pv .artv:empty').forEach(v=>{const a=ARTI.get(v.parentElement.dataset.art);if(!a)return;
 const f=document.createElement('iframe');f.setAttribute('sandbox','allow-scripts');f.title='Preview';f.srcdoc=artDoc(a.kind,a.code);v.appendChild(f)})}
function mathify(s){if(!window.katex)return s;const out=[];
 s=s.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g,(m,a,b)=>{try{out.push(katex.renderToString(a||b,{displayMode:true,throwOnError:false}))}catch(e){return m}return `@@M${out.length-1}@@`})
  .replace(/\\\(([\s\S]+?)\\\)|(^|[^\\$\w])\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?![\w$])/g,(m,a,p,b)=>{const t=a||b;try{out.push(katex.renderToString(t,{throwOnError:false}))}catch(e){return m}return (p||'')+`@@M${out.length-1}@@`});
 return {s,out}}
let mdR=null;
function md(src,streaming){mdLive=!!streaming;if(window.marked&&window.DOMPurify){
  if(!mdR){mdR=new marked.Renderer();mdR.code=(c,l)=>{if(typeof c==='object'){l=c.lang;c=c.text}return codeBlock(c,(l||'').split(/\s/)[0])};
   mdR.link=(h,t,x)=>{if(typeof h==='object'){x=h.text;h=h.href}
    const fp=/^(?:sandbox:)?\/?mnt\/data\/(.+)$/.exec(h||'');   // (a file of the Python sandbox: downloaded from this browser's storage)
    if(fp){let p=fp[1];try{p=decodeURIComponent(p)}catch(e){}return `<a href="#" data-dl="${esc(p)}">${x}</a>`}
    return `<a href="${esc(h||'')}" target="_blank" rel="noopener">${x}</a>`}}
  // math only outside fenced / inline code
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
/* ---------- messages ---------- */
const VF=['content','reasoning','thinkSecs','t','err','tool','stopped','model'];
function snap(m){const o={};for(const f of VF)if(m[f]!=null&&m[f]!=='')o[f]=m[f];return o}
const isGen=(c,k)=>!!gen&&gen.c===c&&k===c.msgs.length-1;
const shp=(m,st)=>`${+st}${+!!m.reasoning}${+!!m.tool}${+!!m.err}${+!!m.thinking}${m.phase||''}${m.calls?m.calls.length:0}`;
function thinkLabel(m,st){const on=st&&m.thinking;const s=on?(Date.now()-m.thinkStart)/1000:(m.thinkSecs||0);
 return on?`<span class="shimmer">Thinking</span> <span class="tsec">${fmtS(s)}</span>`:`Thought for ${fmtS(s)}`}
function thinkHTML(m,k,st){if(!m.reasoning)return'';
 return `<div class="think${m.open?' open':''}${st&&m.thinking?' live':''}"><button class="th" onclick="toggleThink(${k})">${IC.brain}<span class="tl">${thinkLabel(m,st)}</span>${IC.chev}</button><div class="tb">${esc(st&&gen?m.reasoning.slice(0,gen.sr):m.reasoning)}</div></div>`}
function toggleThink(k){const c=conv(),m=c&&c.msgs[k];if(!m)return;m.open=!m.open;const el=document.querySelector(`#col>.msg[data-k="${k}"] .think`);
 if(el){el.classList.toggle('open',m.open);const tb=el.querySelector('.tb');if(m.open&&el.classList.contains('live'))tb.scrollTop=tb.scrollHeight;else peek(el)}}
function fillLive(lb,m){if(!lb||!gen)return;if(lb.dataset.ph!==m.phase){lb.dataset.ph=m.phase;
  lb.innerHTML=m.phase==='prompt'?'<span class="shimmer">Reading the prompt</span><span class="prog ind"><i></i></span><span class="lt"></span>':'<span class="pd"></span><span class="lt"></span>'}
 const lt=lb.querySelector('.lt');
 if(m.phase==='prompt'){const p=m.pp,f=p&&p.total?Math.min(1,p.done/p.total):0,pr=lb.querySelector('.prog');pr.classList.toggle('ind',!f);pr.firstChild.style.width=f?(100*f).toFixed(1)+'%':'';
  lt.textContent=(p&&p.total?(f?`${Math.round(100*f)}% · ${fmtK(p.done)} / ${fmtK(p.total)}`:`${fmtK(p.total)} tokens`)+(p.cached?` · ${fmtK(p.cached)} cached`:'')+' · ':'')+fmtS((Date.now()-gen.t0)/1000)}
 else{const t=m.t,on=!!(t&&t.predicted_n);lb.style.display=on?'':'none';if(on)lt.textContent=`${(+t.predicted_per_second||0).toFixed(1)} t/s · ${t.predicted_n.toLocaleString()} tokens`}}
function statHTML(m){const t=m.t;if(!t||t.predicted_n==null)return'';
 let h=`<span>${IC.bolt}${(+t.predicted_per_second||0).toFixed(1)} t/s</span><span>${t.predicted_n.toLocaleString()} tokens</span>`;
 if(t.predicted_ms!=null)h+=`<span class="x">${IC.clock}${fmtS(t.predicted_ms/1000)}</span>`;
 if(t.prompt_ms!=null&&t.prompt_n!=null)h+=`<span class="x">${IC.inbox}${t.prompt_n.toLocaleString()} prompt @ ${Math.round(t.prompt_per_second).toLocaleString()} t/s</span>`;
 if(t.cache_n)h+=`<span class="x">${IC.db}${t.cache_n.toLocaleString()} cached</span>`;
 return `<button class="stat" onclick="this.classList.toggle('more')" title="${esc(m.model||'')}">${h}</button>`}
function footHTML(c,m,k){const last=k===c.msgs.length-1;
 if(m.calls&&m.calls.length&&!last&&c.msgs[k+1].role==='assistant')return '';   /* (a tool round: the turn's last message has the actions) */let h=`<div class="foot"><div class="acts"><button class="ib sm" aria-label="Copy" title="Copy" onclick="cpMsg(${k},this)">${IC.copy}</button>`;
 if(last)h+=`<button class="ib sm" aria-label="Regenerate" title="Regenerate" onclick="regen()">${IC.regen}</button>`;
 if(m.vers&&m.vers.length>1){const i=m.vi||0;h+=`<span class="pager"><button onclick="ver(${k},-1)" aria-label="Previous answer"${i?'':' disabled'}>${IC.chl}</button>${i+1} / ${m.vers.length}`+
  `<button onclick="ver(${k},1)" aria-label="Next answer"${i<m.vers.length-1?'':' disabled'}>${IC.chr}</button></span>`}
 h+='</div>'+(m.stopped?'<span class="tag">Stopped</span>':'')+statHTML(m);return h+'</div>'}
function aiInner(c,m,k,st){let h=thinkHTML(m,k,st)+toolHTML(m);
 if(m.tool)h+=`<div class="tool"><div class="hd">${IC.tool}Tool call</div><pre>${esc(m.tool)}</pre></div>`;
 h+=`<div class="md${st&&m.phase==='write'?' streaming':''}">${md(st&&gen?m.content.slice(0,gen.sc):(m.content||''),st)}</div>`;
 if(st)h+='<div class="livebar"></div>';
 if(m.err)h+=`<div class="err">${IC.alert}<span>${esc(m.err)}</span>${!st&&k===c.msgs.length-1?'<button class="btn sm" onclick="regen()">Retry</button>':''}</div>`;
 if(!st)h+=footHTML(c,m,k);
 return h}
function msgHTML(c,m,k){
 if(m.role==='user'){
  if(editing===k)return `<div class="msg user" data-k="${k}"><div class="edit"><textarea id="edt" rows="1" aria-label="Edit message">${esc(m.content)}</textarea>`+
   `<div class="erow"><button class="btn" onclick="editCancel()">Cancel</button><button class="btn pri" onclick="editSave(${k})">Send</button></div></div></div>`;
  const long=m.content.length>1200||m.content.split('\n').length>18;
  return `<div class="msg user" data-k="${k}">`+(m.content?`<div class="ububble${long?' clamp':''}">${esc(m.content)}</div>`:'')+
   (m.files&&m.files.length?`<div class="fchips">${m.files.map(dlChip).join('')}</div>`:'')+`<div class="acts">`+(long?'<button class="btn sm ghost" onclick="expandU(this)">Show more</button>':'')+
   `<button class="ib sm" aria-label="Copy" title="Copy" onclick="cpMsg(${k},this)">${IC.copy}</button>`+
   (busy?'':`<button class="ib sm" aria-label="Edit" title="Edit" onclick="editMsg(${k})">${IC.pencil}</button>`)+'</div></div>'}
 const st=isGen(c,k),rd=m.calls&&m.calls.length&&k<c.msgs.length-1&&c.msgs[k+1].role==='assistant';
 return `<div class="msg ai${rd?' round':''}" data-k="${k}" data-sh="${shp(m,st)}">${aiInner(c,m,k,st)}</div>`}
const SUG=[['Explain','how mixture-of-experts models route tokens'],['Write','a Python script that resizes new images in a folder'],['Compare','Rust and Go for a small CLI tool'],['Plan','a 3-day trip to Prague']];
function render(){renderSide();const c=conv(),empty=!c||!c.msgs.length,col=$('col');$('main').classList.toggle('empty',empty);
 if(empty){col.innerHTML=`<div class="hero"><span class="mark">${IC.boltf}</span><h1>What can I help with?</h1><p id="hsub">${esc(live.model||'hyper')} · running locally</p></div>`;document.title='hyper'}
 else{col.innerHTML=c.msgs.map((m,k)=>msgHTML(c,m,k)).join('');document.title=c.title+' · hyper'}
 $('sugs').innerHTML=empty?SUG.map((s,i)=>`<button class="sug" onclick="useSug(${i})"><b>${esc(s[0])}</b>${esc(s[1])}</button>`).join(''):'';
 $('sugs').style.display=empty?'':'none';if(gen&&c===gen.c){const lm=col.lastElementChild,d=lm&&lm.querySelector('.md'),tb=lm&&lm.querySelector('.think .tb');if(d){fadeTail(d);d._n=gen.sc}if(tb)tb._n=gen.sr;
  fillLive(col.querySelector('.livebar'),gen.a)}mountArts(col);mountMedia(col);updSend();peek();onScroll()}
function expandU(b){const c=b.closest('.msg').querySelector('.ububble').classList.toggle('clamp');b.textContent=c?'Show more':'Show less'}
function rerender(k){const c=conv(),el=document.querySelector(`#col>.msg[data-k="${k}"]`);if(!c||!el||!c.msgs[k])return render();
 const t=document.createElement('div');t.innerHTML=msgHTML(c,c.msgs[k],k);const n=t.firstElementChild;el.replaceWith(n);mountArts(n);mountMedia(n)}
function markNew(n){const l=[...$('col').children].slice(-n);l.forEach(e=>e.classList.add('new'))}
function peek(root){const p=(root||$('col')).querySelector('.think.live:not(.open) .tb');if(p)p.scrollTop=p.scrollHeight}
function useSug(i){IN.value=SUG[i][0]+' '+SUG[i][1];sendMsg()}
function nearBottom(){const s=$('scroll');return s.scrollHeight-s.scrollTop-s.clientHeight<100}
function toBottom(smooth){const s=$('scroll');s.scrollTo({top:s.scrollHeight,behavior:smooth?'smooth':'auto'})}
function onScroll(){const s=$('scroll'),d=s.scrollHeight-s.scrollTop-s.clientHeight;atBottom=d<100;
 $('jump').classList.toggle('on',d>240&&!$('main').classList.contains('empty'));$('hdr').classList.toggle('sc',s.scrollTop>4)}
$('scroll').addEventListener('scroll',onScroll,{passive:true});
function editMsg(k){if(busy)return;editing=k;rerender(k);const t=$('edt');if(!t)return;autosz(t);t.focus();t.setSelectionRange(t.value.length,t.value.length);
 t.oninput=()=>autosz(t);t.onkeydown=e=>{if(e.key==='Escape')editCancel();else if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing&&(cfg.enter||e.ctrlKey||e.metaKey)){e.preventDefault();editSave(k)}}}
function editCancel(){const k=editing;editing=-1;rerender(k)}
function editSave(k){const t=$('edt'),v=t?t.value.trim():'';if(!v||busy)return;const c=conv();c.msgs[k].content=v;c.msgs.splice(k+1);editing=-1;c.t=Date.now();save();generate()}
function ver(k,d){if(busy)return;const m=conv().msgs[k],i=(m.vi||0)+d;if(!m.vers||i<0||i>=m.vers.length)return;
 m.vers[m.vi||0]=snap(m);for(const f of VF)delete m[f];Object.assign(m,m.vers[i]);m.vi=i;save();rerender(k)}
/* ---------- composer ---------- */
function autosz(t){t.style.height='auto';t.style.height=t.scrollHeight+'px'}
function updSend(){const b=$('send'),stop=!!busy;b.disabled=!stop&&!IN.value.trim()&&!PEND.length;b.classList.toggle('stop',stop);b.innerHTML=stop?IC.stop:IC.send;
 b.setAttribute('aria-label',stop?'Stop':'Send');b.title=stop?'Stop (Esc)':'Send';IN.enterKeyHint=cfg.enter?'send':'enter';
 $('hint').textContent=cfg.enter?'Enter to send · Shift+Enter for a new line':'Ctrl+Enter to send · Enter for a new line'}
IN.value=S.get('hyper.draft','');
IN.addEventListener('input',()=>{autosz(IN);updSend();S.set('hyper.draft',IN.value)});
// attachments: kept for the next message, then stored with the conversation (its /mnt/data in the Python sandbox)
let PEND=[];
function renderAtts(){const a=$('atts');a.style.display=PEND.length?'':'none';
 a.innerHTML=PEND.map((f,i)=>`<span class="fchip">${IC.file}<span>${esc(f.name)}</span><small>${fmtB(f.size)}</small><button class="x" aria-label="Remove" onclick="unattach(${i})">${IC.x}</button></span>`).join('');updSend()}
function attach(list){for(const f of list){if(f.size>64*1048576){toast(f.name+' is larger than 64 MB');continue}PEND.push(f)}
 if(PEND.length&&!cfg.code){cfg.code=true;saveCfg();syncSet();toast('Python is on: it works with the files')}renderAtts()}
function unattach(i){PEND.splice(i,1);renderAtts()}
$('fpick').onchange=()=>{attach([...$('fpick').files]);$('fpick').value=''};
IN.addEventListener('paste',e=>{const f=e.clipboardData?[...e.clipboardData.files]:[];if(f.length){e.preventDefault();attach(f)}});
(()=>{let n=0;const has=e=>e.dataTransfer&&[...e.dataTransfer.types].includes('Files');
 addEventListener('dragenter',e=>{if(has(e)){n++;$('app').classList.add('drop')}});
 addEventListener('dragleave',e=>{if(has(e)&&--n<=0){n=0;$('app').classList.remove('drop')}});
 addEventListener('dragover',e=>{if(has(e))e.preventDefault()});
 addEventListener('drop',e=>{n=0;$('app').classList.remove('drop');if(e.dataTransfer&&e.dataTransfer.files.length){e.preventDefault();attach([...e.dataTransfer.files])}})})();
IN.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing&&(cfg.enter||e.ctrlKey||e.metaKey)){e.preventDefault();if(!busy)sendMsg()}});
/* ---------- files (IndexedDB): each conversation's /mnt/data (attachments, files the Python code writes) and figures ---------- */
let fdbP=null;
function fdb(){return fdbP||(fdbP=new Promise((res,rej)=>{const r=indexedDB.open('hyper-files',1);r.onupgradeneeded=()=>r.result.createObjectStore('f');
 r.onsuccess=()=>res(r.result);r.onerror=()=>{fdbP=null;rej(r.error)}}))}
function fop(mode,fn){return fdb().then(db=>new Promise((res,rej)=>{const t=db.transaction('f',mode),r=fn(t.objectStore('f'));
 t.oncomplete=()=>res(r&&r.result);t.onerror=t.onabort=()=>rej(t.error||new Error('browser storage'))}))}
const FK=(cid,p)=>'f:'+cid+'/'+p;
function fput(cid,p,data){const t=Date.now();return fop('readwrite',s=>s.put({name:p,data,size:data.byteLength,t},FK(cid,p))).then(()=>t)}
const fget=(cid,p)=>fop('readonly',s=>s.get(FK(cid,p)));
const fdel=(cid,p)=>fop('readwrite',s=>s.delete(FK(cid,p)));
const flist=cid=>fop('readonly',s=>s.getAll(IDBKeyRange.bound('f:'+cid+'/','f:'+cid+'/￿')));
const fmtB=n=>n<1024?n+' B':n<1048576?Math.round(n/1024)+' KB':(n/1048576).toFixed(1)+' MB';
const dlChip=f=>`<button class="fchip" data-dl="${esc(f.name)}" title="Download">${IC.dl}<span>${esc(f.name)}</span><small>${fmtB(f.size)}</small></button>`;
async function dl(name){const c=conv();let f=null;try{f=c&&await fget(c.id,name)}catch(e){}
 if(!f)return toast('The file is not in this browser');const u=URL.createObjectURL(new Blob([f.data],{type:'application/octet-stream'})),a=document.createElement('a');
 a.href=u;a.download=name.split('/').pop();document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),5000)}
const GU=new Map();   // figure id -> object URL
function mountMedia(root){(root||$('col')).querySelectorAll('img[data-g]:not([src])').forEach(im=>{const id=im.dataset.g;if(GU.has(id)){im.src=GU.get(id);return}
 fop('readonly',s=>s.get('g:'+id)).then(v=>{if(!v)return;const u=URL.createObjectURL(new Blob([v.data],{type:'image/png'}));GU.set(id,u);im.src=u}).catch(()=>{})})}
// (once after start: the files of deleted conversations and figures no message shows any more)
setTimeout(()=>{const ids=new Set(convs.map(c=>c.id)),figs=new Set();for(const c of convs)for(const m of c.msgs)for(const x of m.calls||[])for(const g of x.figs||[])figs.add(g);
 fop('readwrite',s=>{const r=s.openKeyCursor();r.onsuccess=()=>{const cu=r.result;if(!cu)return;const k=String(cu.key);
  const g=k.slice(2);   // (a figure id starts with its time: one being saved right now is not in a message yet)
  if(k.startsWith('f:')?!ids.has(k.slice(2,k.indexOf('/'))):!figs.has(g)&&Date.now()-parseInt(g.slice(0,8),36)>36e5)s.delete(cu.key);cu.continue()}}).catch(()=>{})},10000);
/* ---------- Python: Pyodide in a Worker inside a sandboxed iframe (opaque origin: no access to this page, its storage or the
   server; its CSP allows only the package CDNs). /mnt/data holds the conversation's files; variables persist per conversation ---------- */
const PY_TOOL={type:'function',function:{name:'run_python',description:"Run Python code (Pyodide) in a sandbox in the user's browser. Returns what it prints, the value of the last expression and errors. Variables persist between calls in this conversation; files are in /mnt/data.",
 parameters:{type:'object',properties:{code:{type:'string',description:'The Python code to run'}},required:['code']}}};
function codePrompt(){return 'You can run Python with the run_python tool: Python 3.14 (Pyodide) in a sandbox in the user\'s browser, with numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels, sympy, networkx, Pillow, lxml, opencv and polars. '+
 'Pure-Python packages from PyPI install automatically when imported (python-docx, openpyxl, python-pptx, pypdf, xlsxwriter, seaborn, ...); there is no other internet access. '+
 'Use it for calculations, data analysis, charts and files rather than working them out by hand. The working directory /mnt/data holds the files the user attached; save files for the user there '+
 'and link each one as [name](sandbox:/mnt/data/name) so the user can download it. Matplotlib figures are shown to the user as images (you do not see them: describe them from the data). '+
 'For an interactive chart, diagram or small app, write one self-contained HTML document in a ```html code block (or a ```mermaid diagram) instead: the user sees it rendered; it may load libraries only from https://cdn.jsdelivr.net.'}
const PY_CSP="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' https://cdn.jsdelivr.net blob:; worker-src blob:; connect-src https://cdn.jsdelivr.net https://pypi.org https://files.pythonhosted.org";
const PY={f:null,up:null,ok:null,seq:0,wait:new Map(),sent:new Map()};   // sent: file key -> version the sandbox has
function pyFrame(){if(PY.up)return PY.up;
 PY.up=new Promise((res,rej)=>{const f=document.createElement('iframe');f.setAttribute('sandbox','allow-scripts');f.style.display='none';
  f.srcdoc=`<!doctype html><meta http-equiv="Content-Security-Policy" content="${PY_CSP}"><script>(${pyFrameMain})()<\/script>`;PY.f=f;PY.ok=res;document.body.appendChild(f);
  setTimeout(()=>rej(new Error('the Python sandbox did not start')),20000)});
 PY.up.catch(()=>{if(PY.f)PY.f.remove();PY.f=PY.up=null});return PY.up}
// (runs in the iframe: starts the worker, times cells out and restarts it, passes messages and files through)
function pyFrameMain(){let src='',w=null,cur=null,tm=0;const up=(o,tr)=>parent.postMessage(o,'*',tr||[]);
 function start(){w=new Worker(URL.createObjectURL(new Blob([src],{type:'text/javascript'})));
  w.onmessage=e=>{const d=e.data||{};if(!cur||d.id!==cur.id)return;
   if(d.type==='begin'){const L=cur.limit;clearTimeout(tm);tm=setTimeout(()=>reset('Stopped after '+L+' s'),L*1000)}
   if(d.type==='done'){clearTimeout(tm);cur=null;up(d,(d.figs||[]).concat((d.written||[]).map(x=>x.data)))}else up(d)};
  w.onerror=e=>{e.preventDefault();if(cur)reset('Python crashed ('+(e.message||'worker error')+')')}}
 function reset(why){if(w)w.terminate();const id=cur&&cur.id;cur=null;clearTimeout(tm);start();up({type:'reset'});
  if(id)up({type:'done',id,error:why+': the Python session was restarted (variables are gone, the files in /mnt/data are kept)'})}
 addEventListener('message',e=>{if(e.source!==parent)return;const d=e.data||{};
  if(d.type==='init'&&!w){src=d.src;start()}
  else if(d.type==='run'&&w){cur=d;clearTimeout(tm);tm=setTimeout(()=>reset('Python did not start within 180 s'),180000);w.postMessage(d,(d.files||[]).map(f=>f.data))}
  else if(d.type==='kill'&&cur)reset('Stopped by the user')});
 up({type:'ready'})}
// (runs in the worker)
function pyWorker(){const BASE='https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
 const SETUP=String.raw`
import os, sys, io, shutil, linecache, traceback, warnings, importlib.util
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings('ignore', message='.*(non-interactive|non-GUI|cannot be shown).*')
warnings.filterwarnings('ignore', message='The [xy] parameter as float was deprecated')   # (every savefig in this matplotlib build)
from pyodide.code import eval_code_async, find_imports
DATA = '/mnt/data'
os.makedirs(DATA, exist_ok=True)
os.makedirs('/mnt/sessions', exist_ok=True)
os.chdir(DATA)
_spaces, _cur = {}, [None]
_PIP = {'docx': 'python-docx', 'pptx': 'python-pptx', 'yaml': 'pyyaml', 'bs4': 'beautifulsoup4', 'dateutil': 'python-dateutil',
        'sklearn': 'scikit-learn', 'PIL': 'pillow', 'cv2': 'opencv-python', 'Crypto': 'pycryptodome', 'dotenv': 'python-dotenv',
        'jwt': 'pyjwt', 'attr': 'attrs', 'odf': 'odfpy', 'fitz': 'pymupdf'}

def _hyper_enter(cid):
    # /mnt/data is the conversation's folder; the other conversations' folders wait in /mnt/sessions
    if _cur[0] != cid:
        os.chdir('/')   # (the working directory cannot be renamed)
        if _cur[0] is not None:
            q = '/mnt/sessions/' + _cur[0]
            if os.path.isdir(q):
                shutil.rmtree(q)
            os.rename(DATA, q)
        p = '/mnt/sessions/' + cid
        if os.path.isdir(p):
            os.rename(p, DATA)
        else:
            os.makedirs(DATA, exist_ok=True)
        _cur[0] = cid
    os.chdir(DATA)

async def _hyper_pip(code):
    # imported modules that are not Pyodide packages: pure-Python wheels from PyPI
    try:
        mods = find_imports(code)
    except Exception:
        return
    need = [m for m in mods if importlib.util.find_spec(m) is None]
    if need:
        import micropip
        for m in need:
            try:
                await micropip.install(_PIP.get(m, m))
            except Exception:
                pass   # (the import in the cell reports it)

def _hyper_files():
    out = {}
    for root, _, files in os.walk(DATA):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, DATA)] = (st.st_size, st.st_mtime_ns)
    return out

async def _hyper_cell(code, cid):
    ns = _spaces.setdefault(cid, {'__name__': '__main__'})
    linecache.cache['<cell>'] = (len(code), None, code.splitlines(True), '<cell>')
    before = _hyper_files()
    res = err = None
    try:
        val = await eval_code_async(code, ns, filename='<cell>')
        if val is not None:
            res = repr(val)
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        i = next((k for k, f in enumerate(tb) if f.filename == '<cell>'), len(tb))
        err = ('Traceback (most recent call last):\n' + ''.join(traceback.format_list(tb[i:])) if i < len(tb) else '') + \
              ''.join(traceback.format_exception_only(type(e), e))
    figs = []
    if 'matplotlib.pyplot' in sys.modules:
        import matplotlib.pyplot as plt
        for n in plt.get_fignums()[:8]:
            b = io.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.figure(n).savefig(b, format='png', dpi=110, bbox_inches='tight')
            figs.append(b.getvalue())
        plt.close('all')
    after = _hyper_files()
    written = []
    for p, v in after.items():
        if before.get(p) != v and v[0] <= 64 << 20:
            with open(os.path.join(DATA, p), 'rb') as f:
                written.append((p, f.read()))
    removed = [p for p in before if p not in after]
    return [res, err, figs, written, removed]
`;
 let py=null,out=[],chain=Promise.resolve();
 const boot=(async()=>{importScripts(BASE+'pyodide.js');py=await loadPyodide({indexURL:BASE,stdout:s=>out.push(s),stderr:s=>out.push(s)});
  await py.loadPackage('micropip',{messageCallback:()=>{}});await py.runPythonAsync(SETUP)})();
 boot.catch(()=>{});
 onmessage=e=>{const d=e.data;chain=chain.then(()=>run(d))};
 const call=async(n,...a)=>{const f=py.globals.get(n);try{return await f(...a)}finally{f.destroy()}};
 async function run(d){const post=(o,tr)=>postMessage(Object.assign({id:d.id},o),tr||[]);out=[];
  try{if(!py)post({type:'status',text:'Starting Python'});
   try{await boot}catch(x){throw new Error('Python could not be loaded ('+(x&&x.message||x)+')')}
   await call('_hyper_enter',d.cid);
   for(const f of d.files||[]){const p='/mnt/data/'+f.path;py.FS.mkdirTree(p.slice(0,p.lastIndexOf('/')));py.FS.writeFile(p,new Uint8Array(f.data))}
   await py.loadPackagesFromImports(d.code,{messageCallback:m=>post({type:'status',text:m}),errorCallback:()=>{}});
   await call('_hyper_pip',d.code);
   out=[];post({type:'begin'});
   const r=await call('_hyper_cell',d.code,d.cid),v=r.toJs();r.destroy();
   const figs=v[2].map(b=>b.slice().buffer),written=v[3].map(x=>({path:x[0],data:x[1].slice().buffer}));
   post({type:'done',out:out.join('\n'),result:v[0],error:v[1],figs,written,removed:v[4]},figs.concat(written.map(x=>x.data)))}
  catch(x){post({type:'done',out:out.join('\n'),error:String(x&&x.message||x)})}}}
addEventListener('message',e=>{const d=e.data;if(!d||typeof d!=='object')return;
 if(PY.f&&e.source===PY.f.contentWindow){
  if(d.type==='ready'){PY.f.contentWindow.postMessage({type:'init',src:`(${pyWorker})()`},'*');PY.ok()}
  else if(d.type==='reset')PY.sent.clear();
  else{const w=PY.wait.get(d.id);if(!w)return;if(d.type==='status')w.status(String(d.text||'').slice(0,100));else if(d.type==='done'){PY.wait.delete(d.id);w.done(d)}}
  return}
 if(typeof d.hyperArt==='number')for(const f of document.querySelectorAll('.artv iframe'))if(f.contentWindow===e.source){
  f.style.height=Math.round(Math.min(Math.max(d.hyperArt,120),Math.max(320,innerHeight*.75)))+'px';break}});
const safePath=p=>{p=String(p||'').replace(/\\/g,'/').replace(/^\/+/,'').replace(/^mnt\/data\//,'');
 return p&&p.length<=200&&!p.split('/').some(s=>!s||s==='.'||s==='..')?p:null};
async function runPython(c,x,signal,k){let a={};try{a=JSON.parse(x.args||'{}')}catch(e){throw new Error('arguments are not valid JSON: '+x.args)}
 const code=typeof a.code==='string'?a.code:'';if(!code.trim())throw new Error('no code given');
 await pyFrame();
 const files=[];   // (the conversation's files the sandbox does not have yet)
 for(const f of await flist(c.id)){const key=c.id+'/'+f.name;if(PY.sent.get(key)!==f.t){files.push({path:f.name,data:f.data.slice(0)});PY.sent.set(key,f.t)}}
 const d=await new Promise(res=>{const id=++PY.seq,kill=()=>PY.f&&PY.f.contentWindow.postMessage({type:'kill'},'*');signal.addEventListener('abort',kill);
  PY.wait.set(id,{status:t=>{x.status=t;if(conv()===c)rerender(k)},done:v=>{signal.removeEventListener('abort',kill);res(v)}});
  PY.f.contentWindow.postMessage({type:'run',id,cid:c.id,code,files,limit:120},'*',files.map(f=>f.data))});
 delete x.status;x.figs=[];x.files=[];
 for(const b of d.figs||[])if(b instanceof ArrayBuffer&&x.figs.length<8){const id=Date.now().toString(36)+Math.random().toString(36).slice(2,8);
  await fop('readwrite',s=>s.put({data:b},'g:'+id));x.figs.push(id)}
 for(const w of d.written||[]){const p=safePath(w&&w.path);if(!p||!(w.data instanceof ArrayBuffer))continue;
  PY.sent.set(c.id+'/'+p,await fput(c.id,p,w.data));x.files.push({name:p,size:w.data.byteLength})}
 for(const r of d.removed||[]){const p=safePath(r);if(p){await fdel(c.id,p);PY.sent.delete(c.id+'/'+p)}}
 const cut=(s,n)=>s.length>n?s.slice(0,n/2)+`\n… (${s.length-n} characters left out) …\n`+s.slice(-n/2):s,r=[];
 if(d.out)r.push(cut(String(d.out),16000));if(d.result!=null)r.push('Out: '+cut(String(d.result),8000));if(d.error)r.push(cut(String(d.error),8000));
 if(x.files.length)r.push('Saved in /mnt/data (offered to the user for download): '+x.files.map(f=>`${f.name} (${fmtB(f.size)})`).join(', '));
 if(x.figs.length)r.push(x.figs.length+(x.figs.length>1?' figures':' figure')+' shown to the user');
 return {text:r.join('\n\n')||'(no output)',isError:!!d.error}}
function toggleCode(){cfg.code=!cfg.code;saveCfg();syncSet();toast(cfg.code?'Python on':'Python off')}
/* ---------- web tools (MCP servers behind the server's /mcp/<name>) ---------- */
let rpcId=0;
async function mcpRpc(server,method,params){const id=++rpcId;
 const r=await fetch('/mcp/'+encodeURIComponent(server),{method:'POST',headers:{'Content-Type':'application/json','Accept':'application/json, text/event-stream'},
  body:JSON.stringify({jsonrpc:'2.0',id,method,params:params||{}})});
 const t=await r.text();let j=null;
 if((r.headers.get('content-type')||'').includes('event-stream')){for(const l of t.split('\n'))if(l.startsWith('data:')){try{const x=JSON.parse(l.slice(5));if(x.id===id)j=x}catch(e){}}}
 else{try{j=JSON.parse(t)}catch(e){}}
 if(!j)throw new Error(`MCP ${server}: HTTP ${r.status} ${t.slice(0,200)}`);
 if(j.error)throw new Error(j.error.message||JSON.stringify(j.error));
 return j.result}
async function loadTools(){try{const j=await (await fetch('/v1/mcp')).json();const tools=[],map={},names=[];
  for(const sv of j.servers||[]){try{
   await mcpRpc(sv.name,'initialize',{protocolVersion:'2025-03-26',capabilities:{},clientInfo:{name:'hyper-chat',version:'1'}});
   const r=await mcpRpc(sv.name,'tools/list');
   for(const t of r.tools||[]){let n=t.name;if(map[n])n=sv.name+'__'+t.name;map[n]={server:sv.name,name:t.name};names.push(t.name);
    tools.push({type:'function',function:{name:n,description:t.description||'',parameters:t.inputSchema||{type:'object',properties:{}}}})}
  }catch(e){console.warn('MCP server',sv.name,e)}}
  MCP.tools=tools;MCP.map=map;MCP.servers=j.servers||[];
  $('webb').style.display=tools.length?'':'none';$('s-webrow').style.display=tools.length?'':'none';$('s-webtools').textContent=names.join(', ')}catch(e){}}
function toggleWeb(){cfg.web=!cfg.web;saveCfg();syncSet();toast(cfg.web?'Web search on':'Web search off')}
function toolPrompt(day){const n=MCP.tools.map(t=>t.function.name);
 return `You have web access through tools (${n.join(', ')}): search the web for anything recent, for facts you are not sure about and when the user asks about a site or a page; then read the most relevant results. Open only URLs that appeared in search results or on pages you read: never construct or guess links. Cite the pages you used as markdown links. Today is ${day}.`}
function hostOf(u){try{return new URL(u).hostname.replace(/^www\./,'')}catch(e){return u||''}}
function toolHTML(m){if(!m.calls||!m.calls.length)return'';
 return '<div class="tcs">'+m.calls.map(x=>{let a={};try{a=JSON.parse(x.args||'{}')}catch(e){}
  const t=(MCP.map[x.name]||{}).name||x.name,done=x.result!=null,run=x.state==='run',py=t==='run_python'&&!MCP.map[x.name];
  const w=t==='web_search'?['Searching','Searched',a.query?'“'+a.query+'”':'']:t==='fetch_url'?['Reading','Read',hostOf(a.url)]:py?[x.status||'Running Python','Ran Python','']:['Running','Ran',t||'tool'];
  const lbl=done?`${w[1]}${w[2]?` <b>${esc(w[2])}</b>`:''}`:`<span class="shimmer">${esc(w[0])}${w[2]?' '+esc(w[2]):''}${run?'…':''}</span>`;
  let body='',more='';
  if(py){const code=typeof a.code==='string'?a.code.replace(/^\s*\n|\s+$/g,''):'';
   body=`<div class="tcode">${code?codeBlock(code,'python'):''}${done?`<pre class="tcb">${esc(x.result.slice(0,6000))}${x.result.length>6000?'\n…':''}</pre>`:''}</div>`;
   if(x.figs&&x.figs.length)more+=`<div class="tfig">${x.figs.map(g=>`<img data-g="${esc(g)}" alt="Figure"${GU.has(g)?` src="${GU.get(g)}"`:''}>`).join('')}</div>`;
   if(x.files&&x.files.length)more+=`<div class="fchips">${x.files.map(dlChip).join('')}</div>`}
  else if(done){const items=t==='web_search'&&!x.isError?x.result.split(/\n\n(?=\[\d+\])/).map(b=>{const L=b.split('\n'),mt=/^\[\d+\]\s*(.*)$/.exec(L[0]);return mt&&L[1]?{title:mt[1],url:L[1].trim()}:null}).filter(Boolean):[];
   body=items.length?`<div class="tsr">${items.map(it=>`<a href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.title||it.url)}<small>${esc(hostOf(it.url))}</small></a>`).join('')}</div>`
    :`<pre class="tcb">${esc(x.result.slice(0,6000))}${x.result.length>6000?'\n…':''}</pre>`}
  return `<div class="tc${x.isError?' err':''}"><button class="tch" onclick="this.parentElement.classList.toggle('open')">${t==='web_search'?IC.search:t==='fetch_url'?IC.globe:py?IC.term:IC.tool}<span class="tl2">${lbl}</span>${done||py?IC.chev:''}</button>${body}</div>`+more}).join('')+'</div>'}
// the request history: an assistant turn with tool calls is followed by one tool message per call
function pushHist(msgs,m){if(m.role==='user'){const f=m.files&&m.files.length?'[Attached: '+m.files.map(x=>`/mnt/data/${x.name} (${fmtB(x.size)})`).join(', ')+']':'';
  msgs.push({role:'user',content:m.content&&f?m.content+'\n\n'+f:m.content||f});return}
 const calls=(m.calls||[]).filter(x=>x.name);
 if(m.err&&!m.content&&!calls.length)return;
 const o={role:'assistant',content:m.content||''};if(m.reasoning)o.reasoning_content=m.reasoning;
 if(!calls.length){msgs.push(o);return}
 o.tool_calls=calls.map((x,i)=>({id:x.id||'call_'+i,type:'function',function:{name:x.name,arguments:x.args||'{}'}}));msgs.push(o);
 calls.forEach((x,i)=>msgs.push({role:'tool',tool_call_id:x.id||'call_'+i,content:x.result!=null?x.result:'(no result: interrupted)'}))}
async function runTools(c,a,ctl){const k=c.msgs.indexOf(a);
 for(const x of a.calls){if(ctl.signal.aborted)return false;if(!x.name)continue;
  x.state='run';if(conv()===c)rerender(k);
  try{const m=MCP.map[x.name];
   if(!m&&x.name==='run_python'){const r=await runPython(c,x,ctl.signal,k);x.result=r.text;x.isError=r.isError}
   else{if(!m)throw new Error('unknown tool '+x.name);let args={};
    try{args=x.args?JSON.parse(x.args):{}}catch(e){throw new Error('arguments are not valid JSON: '+x.args)}
    const r=await mcpRpc(m.server,'tools/call',{name:m.name,arguments:args});
    x.result=(r.content||[]).map(p=>p.type==='text'?p.text:'['+p.type+']').join('\n')||'(empty result)';x.isError=!!r.isError}}
  catch(e){delete x.status;x.result='Error: '+(e.message||e);x.isError=true}
  x.result=x.result.slice(0,40000);delete x.state;save();if(conv()===c){const st=nearBottom();rerender(k);if(st)toBottom()}}
 return !ctl.signal.aborted}
const MAX_TOOL_ROUNDS=25;   // (then one more answer without tools, from what was found)
/* ---------- generation ---------- */
// smooth streaming: received text is revealed at a steady pace that follows its backlog (token bursts are spread
// out), the newest characters fade in, and the markdown is re-rendered at most once per frame (less often when slow)
function reveal(s,n,dt,fast){const back=n-s;if(back<=0)return n;return Math.min(n,s+Math.max(1,Math.round(Math.max(fast?800:80,back/(fast?.1:.22))*dt)))}
function tick(g,now){if(gen!==g)return;const m=g.a,dt=Math.min(.1,(now-(g.tp||now))/1000);g.tp=now;
 g.sr=reveal(g.sr,m.reasoning.length,dt,g.done);g.sc=reveal(g.sc,m.content.length,dt,g.done);
 if(now-g.lr>=g.iv){g.lr=now;const t0=performance.now();updLast();g.iv=Math.min(100,(performance.now()-t0)*2.5)}
 if(g.done&&g.sr>=m.reasoning.length&&g.sc>=m.content.length){if(g.fin)g.fin();return}
 g.raf=requestAnimationFrame(t=>tick(g,t))}
function fadeTail(root,N){N=N||48;const w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT),todo=[];let n=w.lastChild(),left=N;
 while(n&&left>0){const p=n.parentElement;if(p&&p.closest('.katex,.hd'))break;if(n.data.trim()){const k=Math.min(left,n.data.length);todo.push([n,k,N-left]);left-=k}n=w.previousNode()}
 for(const [n,k,off] of todo){const t=n.data,f=document.createDocumentFragment();if(t.length>k)f.appendChild(document.createTextNode(t.slice(0,t.length-k)));
  for(let i=t.length-k;i<t.length;i+=3){const sp=document.createElement('span');sp.textContent=t.slice(i,i+3);sp.style.opacity=(.06+.94*Math.min(1,(off+t.length-1-i)/N)).toFixed(2);f.appendChild(sp)}
  n.parentNode.replaceChild(f,n)}}
function updLast(){if(!gen||conv()!==gen.c)return;const g=gen,c=g.c,k=c.msgs.length-1,m=c.msgs[k],el=$('col').lastElementChild;if(!el||+el.dataset.k!==k)return;
 const stick=nearBottom(),sh=shp(m,true);
 if(el.dataset.sh!==sh){el.dataset.sh=sh;el.innerHTML=aiInner(c,m,k,true);const tb=el.querySelector('.think .tb');if(tb){tb._n=g.sr;if(m.open)tb.scrollTop=tb.scrollHeight}
  const d=el.querySelector('.md');if(d){fadeTail(d);d._n=g.sc}mountMedia(el)}
 else{const tl=el.querySelector('.think .tl');if(tl){const l=thinkLabel(m,true);if(tl._s!==l){tl.innerHTML=l;tl._s=l}}
  const tb=el.querySelector('.think .tb');if(tb&&tb._n!==g.sr){const keep=tb.scrollHeight-tb.scrollTop-tb.clientHeight<40;
   if(tb.firstChild&&tb._n!=null&&tb._n<g.sr)tb.firstChild.appendData(m.reasoning.slice(tb._n,g.sr));else tb.textContent=m.reasoning.slice(0,g.sr);tb._n=g.sr;if(keep)tb.scrollTop=tb.scrollHeight}
  const d=el.querySelector('.md');if(d&&d._n!==g.sc){d.innerHTML=md(m.content.slice(0,g.sc),true);fadeTail(d);d._n=g.sc}
  const tp=el.querySelector('.tool pre');if(tp&&tp.textContent!==m.tool)tp.textContent=m.tool;
  const tw=el.querySelector('.tcs');if(tw){const h=toolHTML(m);if(el._th!==h){tw.outerHTML=h;el._th=h;mountMedia(el)}}}
 fillLive(el.querySelector('.livebar'),m);peek(el);if(stick)toBottom()}
async function pollPrompt(a){try{const s=await (await fetch('/stats',{cache:'no-store'})).json();if(a.phase!=='prompt')return;
 if(s.phase!=='idle')a.pp={done:s.prompt_done,total:s.prompt_total,cached:s.cached}}catch(e){}}
async function generate(vers,round,final){round=round||0;const c=conv();
 const a={role:'assistant',content:'',reasoning:'',phase:'prompt',thinking:false,model:live.model};if(vers&&vers.length)a.vers=vers;
 const web=!final&&cfg.web&&MCP.tools.length>0,py=!final&&!!cfg.code,tl=(web?MCP.tools:[]).concat(py?[PY_TOOL]:[]),tools=tl.length?tl:null;
 // (the date is the conversation's first day: a prompt that changed at midnight would lose the server's prompt cache)
 if(!c.day)c.day=new Date().toISOString().slice(0,10);
 const msgs=[],sys=[cfg.sys.trim(),web?toolPrompt(c.day):'',py?codePrompt():''].filter(Boolean).join('\n\n');if(sys)msgs.push({role:'system',content:sys});
 for(const m of c.msgs)pushHist(msgs,m);
 c.msgs.push(a);
 const body={messages:msgs,stream:true,timings_per_token:true,chat_template_kwargs:{enable_thinking:cfg.effort!=='off'}};
 if(cfg.effort==='low'||cfg.effort==='medium')body.reasoning_effort=cfg.effort;   // (high: the template's default -- Qwen xhigh, GLM max)
 if(cfg.maxt!==''&&+cfg.maxt>0)body.max_tokens=+cfg.maxt;
 if(tools){body.tools=tools;body.tool_choice='auto'}
 if(cfg.temp!=='')body.temperature=+cfg.temp;if(cfg.topp!=='')body.top_p=+cfg.topp;if(cfg.topk!=='')body.top_k=parseInt(cfg.topk);if(cfg.minp!=='')body.min_p=+cfg.minp;
 const ctl=busy=new AbortController();const g=gen={c,a,t0:Date.now(),sc:0,sr:0,lr:0,iv:0};render();markNew(vers||round?1:2);toBottom();setDot();g.raf=requestAnimationFrame(t=>tick(g,t));
 const timer=setInterval(()=>{if(a.phase==='prompt')pollPrompt(a)},500);let aborted=false;
 try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:busy.signal});
  if(!r.ok){let t=await r.text();try{t=JSON.parse(t).error.message||t}catch(e){}throw new Error(`Server error ${r.status}: ${t}`)}
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
  for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
   while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+2);if(!line.startsWith('data:'))continue;
    const d=line.slice(5).trim();if(d==='[DONE]')continue;let j;try{j=JSON.parse(d)}catch(e){continue}
    if(j.error){a.err=j.error.message||String(j.error);continue}
    const ch=j.choices&&j.choices[0];if(ch&&ch.delta){const dl=ch.delta;
     if(dl.reasoning_content){if(!a.thinking&&!a.content){a.thinking=true;a.thinkStart=a.thinkStart||Date.now()}a.phase='think';a.reasoning+=dl.reasoning_content}
     if(dl.content){if(a.thinking){a.thinking=false;a.thinkSecs=(Date.now()-a.thinkStart)/1000}a.phase='write';a.content+=dl.content}
     if(dl.tool_calls){a.calls=a.calls||[];for(const tc of dl.tool_calls){const i=tc.index||0,x=a.calls[i]||(a.calls[i]={id:'',name:'',args:''});
      if(tc.id)x.id=tc.id;const f=tc.function||{};if(f.name&&!x.name)x.name=f.name;if(f.arguments)x.args+=f.arguments}}}
    if(j.timings)a.t=Object.assign(a.t||{},j.timings)}}
 }catch(e){if(e.name==='AbortError')aborted=true;else a.err=String(e.message||e)}
 clearInterval(timer);g.done=true;   // let the reveal catch up (fast) before the final render
 await Promise.race([new Promise(r=>g.fin=r),new Promise(r=>setTimeout(r,3000))]);cancelAnimationFrame(g.raf);
 if(a.thinking){a.thinking=false;a.thinkSecs=(Date.now()-a.thinkStart)/1000}
 delete a.phase;delete a.pp;a.open=false;if(aborted)a.stopped=true;if(a.vers){a.vi=a.vers.length;a.vers.push(snap(a))}
 if(a.calls)a.calls=a.calls.filter(Boolean);
 if(!aborted&&!a.err&&a.calls&&a.calls.length&&tools){   // the model called tools: run them, then continue
  gen=null;save();if(conv()===c){const st=nearBottom();render();if(st)toBottom()}
  if(round>=MAX_TOOL_ROUNDS){   // the tool budget is used up: the calls are answered with a note, then a final answer without tools
   for(const x of a.calls)x.result=`Not run: the limit of ${MAX_TOOL_ROUNDS} tool rounds was reached. Answer now from what you have found so far.`;
   save();if(busy===ctl){c.t=Date.now();return generate(undefined,round+1,true)}}
  else if(await runTools(c,a,ctl)&&busy===ctl){c.t=Date.now();save();return generate(undefined,round+1)}}
 busy=null;gen=null;c.t=Date.now();convs.sort((x,y)=>(y.t||0)-(x.t||0));save();
 const stick=conv()===c&&nearBottom();render();if(stick)toBottom();setDot();refreshLive();if(!touch&&editing<0)IN.focus()}
let sending=false;
async function sendMsg(){if(busy){busy.abort();return}const t=IN.value.trim(),fl=PEND.slice();if((!t&&!fl.length)||sending)return;let c=conv();
 if(!c){c={id:Date.now().toString(36)+Math.random().toString(36).slice(2,6),title:'New chat',msgs:[],t:Date.now()};convs.unshift(c);cur=c.id}
 const m={role:'user',content:t};
 if(fl.length){sending=true;PEND=[];renderAtts();m.files=[];   // (into the conversation's /mnt/data; a file of the same name is replaced)
  for(const f of fl){const name=f.name.replace(/[\\/:*?"<>|\u0000-\u001f]+/g,'_').replace(/^\.+/,'').slice(-120)||'file';
   try{await fput(c.id,name,await f.arrayBuffer());m.files=m.files.filter(x=>x.name!==name).concat({name,size:f.size})}catch(e){toast('Could not store '+name)}}
  sending=false}
 c.msgs.push(m);if(c.title==='New chat')c.title=(t||m.files.map(f=>f.name).join(', ')).replace(/\s+/g,' ').slice(0,60);c.t=Date.now();
 IN.value='';S.set('hyper.draft','');autosz(IN);if(touch)IN.blur();editing=-1;save();generate()}
function regen(){const c=conv();if(busy||!c)return;let t=0;for(let i=c.msgs.length-1;i>=0&&c.msgs[i].role==='assistant';--i)++t;
 if(t>1||(t===1&&c.msgs[c.msgs.length-1].calls)){c.msgs.splice(c.msgs.length-t);return generate()}
 let vers;const l=c.msgs[c.msgs.length-1];
 if(l&&l.role==='assistant'){c.msgs.pop();vers=(l.vers||[]).slice();
  if(l.err&&!l.content)vers.splice(l.vi||0,1);else vers[l.vi||0]=snap(l)}generate(vers)}
/* ---------- server status ---------- */
function setDot(){const d=$('dot');d.classList.toggle('off',!online);d.classList.toggle('busy',online&&(!!busy||(!!live.phase&&live.phase!=='idle')))}
async function refreshLive(){try{const s=await (await fetch('/stats',{cache:'no-store'})).json();online=true;Object.assign(live,s);
  const u=Math.min(1,(s.ctx_used||0)/Math.max(1,s.ctx_max)),R=7,C=2*Math.PI*R;
  $('ctx').innerHTML=`<svg viewBox="0 0 18 18"><circle cx="9" cy="9" r="${R}" stroke="var(--line2)" stroke-width="2.4" fill="none"/><circle cx="9" cy="9" r="${R}" stroke="var(--acc)" stroke-width="2.4" fill="none" stroke-linecap="butt" stroke-dasharray="${(C*u).toFixed(2)} ${C.toFixed(2)}" transform="rotate(-90 9 9)"/></svg>`+
   `<span>${fmtK(s.ctx_used||0)}<span class="mx"> / ${fmtK(s.ctx_max)}</span></span>`}
 catch(e){online=false}setDot();updState()}
function ctxInfo(){if(!live.ctx_max)return;toast(`Context: ${(live.ctx_used||0).toLocaleString()} / ${live.ctx_max.toLocaleString()} tokens (${(100*(live.ctx_used||0)/live.ctx_max).toFixed(1)} %)`,null,null,3200)}
fetch('/v1/models').then(r=>r.json()).then(j=>{live.model=j.data[0].id;$('mdl').textContent=live.model;const h=$('hsub');if(h)h.textContent=live.model+' · running locally';updState()}).catch(()=>{});
fetch('/props').then(r=>r.ok?r.json():{}).then(j=>{defaults=j.sampling||{};syncSet()}).catch(()=>syncSet());
loadTools();
/* ---------- global events ---------- */
document.addEventListener('click',e=>{const b=e.target.closest('.cb .cpy');if(b)copyText(b.closest('.cb').querySelector('code').innerText).then(ok=>ok&&flash(b,'<span class="tx">Copy</span>'));
 if(e.target.closest('.cb .wrp')){cfg.wrap=!cfg.wrap;saveCfg();applyWrap();syncSet()}
 const fl=e.target.closest('[data-dl]');if(fl){e.preventDefault();dl(fl.dataset.dl)}
 const ab=e.target.closest('.cb.art .hd button');if(ab){const cb=ab.closest('.cb');
  if(ab.classList.contains('art-p')){cb.classList.add('pv');mountArts(cb.parentElement)}else if(ab.classList.contains('art-c'))cb.classList.remove('pv');
  else if(ab.classList.contains('art-f'))cb.classList.toggle('full')}
 if(!e.target.closest('#menu'))$('menu').classList.remove('on');if(!e.target.closest('#effmenu'))$('effmenu').classList.remove('on')});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key.toLowerCase()==='o'){e.preventDefault();newConv()}
 else if(e.key==='Escape'){const fs=document.querySelector('.cb.art.full');if(fs)fs.classList.remove('full');else if($('menu').classList.contains('on'))$('menu').classList.remove('on');else if($('sheet').classList.contains('on'))openSet(false);
  else if(document.body.classList.contains('side-on'))side(false);else if(busy&&editing<0)busy.abort()}
 else if(e.key==='/'&&!e.ctrlKey&&!e.metaKey&&!e.target.closest('input,textarea')){e.preventDefault();IN.focus()}});
// touch: swipe from the left edge opens the sidebar, swipe left closes it; drag the settings sheet's header down to close it
(()=>{let x0=null,y0=0,open=false;
 document.addEventListener('touchstart',e=>{if(!narrow()||$('sheet').classList.contains('on'))return;const t=e.touches[0];open=document.body.classList.contains('side-on');
  if(!open&&t.clientX>24)return;x0=t.clientX;y0=t.clientY},{passive:true});
 document.addEventListener('touchend',e=>{if(x0==null)return;const t=e.changedTouches[0],dx=t.clientX-x0,dy=t.clientY-y0;x0=null;
  if(Math.abs(dx)<60||Math.abs(dx)<1.5*Math.abs(dy))return;if(!open&&dx>0)side(true);else if(open&&dx<0)side(false)},{passive:true});
 const sh=$('sheet');let s0=null,sd=0;
 sh.addEventListener('touchstart',e=>{if(!narrow()||!e.target.closest('.grab,.sh-hd')||e.target.closest('button'))return;s0=e.touches[0].clientY;sd=0;sh.style.transition='none'},{passive:true});
 sh.addEventListener('touchmove',e=>{if(s0==null)return;sd=Math.max(0,e.touches[0].clientY-s0);sh.style.transform=`translateY(${sd}px)`},{passive:true});
 sh.addEventListener('touchend',()=>{if(s0==null)return;s0=null;sh.style.transition='';sh.style.transform='';if(sd>80)openSet(false)})})();
new ResizeObserver(()=>document.documentElement.style.setProperty('--dockh',$('dock').offsetHeight+'px')).observe($('dock'));
addEventListener('load',()=>{mdR=null;if(conv()&&conv().msgs.length){render();toBottom()}});
fitVV();applyTheme();applyWrap();syncSet();render();toBottom();autosz(IN);refreshLive();setInterval(refreshLive,3000);if(!touch)IN.focus();
</script></body></html>)HTML";
