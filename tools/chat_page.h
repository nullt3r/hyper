// hyper-server web UI: chat at "/" (conversations in the browser's localStorage, streaming, reasoning, markdown);
// the live statistics page is served at "/live"
#pragma once

static const char * CHAT_PAGE = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hyper chat</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dd;--acc:#2f6fdf;--user:#e8eefb;--code:#f0f0ec;--side:#efefeb}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#9a998f;--card:#1e1e1c;--line:#33332f;--acc:#6ea2ff;--user:#22304a;--code:#262624;--side:#1a1a18}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;height:100vh;display:flex}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
aside{width:250px;background:var(--side);border-right:1px solid var(--line);display:flex;flex-direction:column;flex-shrink:0}
aside .top{padding:10px;display:flex;gap:6px}aside .top button{flex:1}
#convs{flex:1;overflow:auto;padding:0 6px}
.cv{padding:8px 10px;border-radius:8px;cursor:pointer;display:flex;gap:6px;align-items:center;margin-bottom:2px}
.cv:hover{background:var(--card)}.cv.on{background:var(--card);border:1px solid var(--line)}
.cv span{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.cv b{opacity:0;font-weight:400;color:var(--mut)}.cv:hover b{opacity:1}
aside .foot{padding:10px;border-top:1px solid var(--line);font-size:13px;color:var(--mut)}aside .foot a{color:var(--acc)}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
header .t{flex:1;font-weight:600}header .m{color:var(--mut);font-size:13px}
#msgs{flex:1;overflow:auto;padding:16px}
.msg{max-width:820px;margin:0 auto 14px}
.msg.user .b{background:var(--user);border-radius:12px;padding:10px 14px;white-space:pre-wrap;margin-left:15%}
.msg.assistant .b{padding:2px 2px}
.think{border-left:3px solid var(--line);margin:4px 0 8px;padding:2px 10px;color:var(--mut);font-size:14px}
.think summary{cursor:pointer;user-select:none}.think div{white-space:pre-wrap;max-height:400px;overflow:auto}
.st{color:var(--mut);font-size:12px;margin-top:4px;display:flex;gap:10px;flex-wrap:wrap}.st a{cursor:pointer;color:var(--acc)}
.b pre{background:var(--code);border-radius:8px;padding:10px;overflow:auto;position:relative;font-size:13px}
.b pre button{position:absolute;top:6px;right:6px;font-size:12px;padding:2px 8px;opacity:.7}
.b code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.b :not(pre)>code{background:var(--code);padding:1px 4px;border-radius:4px}
.b table{border-collapse:collapse}.b td,.b th{border:1px solid var(--line);padding:4px 8px}
.b p{margin:6px 0}.b h1,.b h2,.b h3,.b h4{margin:12px 0 6px}.b ul,.b ol{margin:6px 0;padding-left:24px}
.b blockquote{border-left:3px solid var(--line);margin:6px 0;padding-left:10px;color:var(--mut)}
.tc{background:var(--code);border-radius:8px;padding:8px 10px;font-size:13px;font-family:ui-monospace,monospace;white-space:pre-wrap}
footer{padding:10px 16px;border-top:1px solid var(--line)}
.inp{max-width:820px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;resize:none;max-height:240px}
textarea:focus{outline:none;border-color:var(--acc)}
#set{display:none;max-width:820px;margin:0 auto 10px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;gap:10px;flex-wrap:wrap}
#set.on{display:flex}#set label{font-size:13px;color:var(--mut);display:flex;flex-direction:column;gap:2px}
#set input{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 6px;width:110px}
#set textarea{width:100%;min-height:60px;resize:vertical}
.empty{color:var(--mut);text-align:center;margin-top:20vh}
@media (max-width:700px){aside{display:none}.msg.user .b{margin-left:0}}
</style></head><body>
<aside><div class="top"><button class="pri" onclick="newConv()">+ New chat</button></div><div id="convs"></div>
<div class="foot"><a href="/live">Live stats</a> · <span id="mdl"></span></div></aside>
<main><header><span class="t" id="title">hyper</span><span class="m" id="ctx"></span><button onclick="toggleSet()">Settings</button></header>
<div id="msgs"></div>
<footer><div id="set">
<label style="width:100%">System prompt<textarea id="sys" rows="2"></textarea></label>
<label>Temperature<input id="temp" type="number" step="0.05" min="0" max="2"></label>
<label>Top-p<input id="topp" type="number" step="0.05" min="0" max="1"></label>
<label>Max tokens<input id="maxt" type="number" step="256" min="16"></label>
<label>Thinking<input id="think" type="checkbox" style="width:auto"></label>
</div>
<div class="inp"><textarea id="in" rows="1" placeholder="Message (Enter to send, Shift+Enter for a new line)"></textarea>
<button class="pri" id="send" onclick="sendMsg()">Send</button></div></footer></main>
<script>
const $=id=>document.getElementById(id);
const S={get(k,d){try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v)}catch(e){return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
let convs=S.get('hyper.convs',[]),cur=S.get('hyper.cur',null),busy=null;
const cfg=Object.assign({sys:'',temp:'',topp:'',maxt:8192,think:true},S.get('hyper.cfg',{}));
for(const k of['sys','temp','topp','maxt'])$(k).value=cfg[k];$('think').checked=cfg.think;
for(const k of['sys','temp','topp','maxt','think'])$(k).onchange=()=>{cfg[k]=k==='think'?$(k).checked:$(k).value;S.set('hyper.cfg',cfg)};
function toggleSet(){$('set').classList.toggle('on')}
function save(){S.set('hyper.convs',convs);S.set('hyper.cur',cur)}
function conv(){return convs.find(c=>c.id===cur)}
function newConv(){const c={id:Date.now().toString(36),title:'New chat',msgs:[]};convs.unshift(c);cur=c.id;save();render();$('in').focus()}
function delConv(id,e){e.stopPropagation();convs=convs.filter(c=>c.id!==id);if(cur===id)cur=convs[0]?convs[0].id:null;save();render()}
function pick(id){if(busy)return;cur=id;save();render()}
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function inline(s){return esc(s).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<i>$2</i>')
 .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>')}
function md(src){const L=src.split('\n');let o='',i=0;
 while(i<L.length){let l=L[i];
  if(/^```/.test(l)){const lang=l.slice(3).trim();let c=[];i++;while(i<L.length&&!/^```/.test(L[i]))c.push(L[i++]);i++;
   o+='<pre><button onclick="cp(this)">copy</button><code'+(lang?' data-l="'+esc(lang)+'"':'')+'>'+esc(c.join('\n'))+'</code></pre>';continue}
  if(/^#{1,4} /.test(l)){const n=l.match(/^#+/)[0].length;o+=`<h${n}>${inline(l.slice(n+1))}</h${n}>`;i++;continue}
  if(/^\s*([-*+]|\d+\.) /.test(l)){const ol=/^\s*\d+\./.test(l);o+=ol?'<ol>':'<ul>';
   while(i<L.length&&/^\s*([-*+]|\d+\.) /.test(L[i])){o+='<li>'+inline(L[i].replace(/^\s*([-*+]|\d+\.) /,''))+'</li>';i++}o+=ol?'</ol>':'</ul>';continue}
  if(/^> ?/.test(l)){let q=[];while(i<L.length&&/^> ?/.test(L[i]))q.push(L[i++].replace(/^> ?/,''));o+='<blockquote>'+md(q.join('\n'))+'</blockquote>';continue}
  if(/^\|.*\|\s*$/.test(l)&&i+1<L.length&&/^\|[\s:|-]+\|\s*$/.test(L[i+1])){const row=r=>r.trim().replace(/^\||\|$/g,'').split('|');
   o+='<table><tr>'+row(l).map(c=>'<th>'+inline(c.trim())+'</th>').join('')+'</tr>';i+=2;
   while(i<L.length&&/^\|.*\|\s*$/.test(L[i]))o+='<tr>'+row(L[i++]).map(c=>'<td>'+inline(c.trim())+'</td>').join('')+'</tr>';o+='</table>';continue}
  if(!l.trim()){i++;continue}
  let p=[];while(i<L.length&&L[i].trim()&&!/^(```|#{1,4} |> ?|\s*([-*+]|\d+\.) )/.test(L[i]))p.push(L[i++]);
  if(!p.length){o+='<p>'+inline(L[i++])+'</p>';continue}
  o+='<p>'+inline(p.join('\n')).replace(/\n/g,'<br>')+'</p>'}
 return o}
function cp(b){navigator.clipboard.writeText(b.nextSibling.textContent);b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}
function stats(m){if(!m.t)return'';const t=m.t;let s=`${t.predicted_n} tokens · ${t.predicted_per_second.toFixed(1)} t/s`;
 if(t.prompt_n!=null)s+=` · prompt ${t.prompt_n} @ ${t.prompt_per_second.toFixed(0)} t/s`+(t.cache_n?` (cached ${t.cache_n})`:'');return s}
function msgHTML(m,k,last){if(m.role==='user')return `<div class="msg user"><div class="b">${esc(m.content)}</div></div>`;
 let h='<div class="msg assistant"><div class="b">';
 if(m.reasoning)h+=`<details class="think"${m.open?' open':''}><summary>Thinking${m.thinking?' …':''} (${m.reasoning.length} chars)</summary><div>${esc(m.reasoning)}</div></details>`;
 h+=md(m.content||'');if(m.err)h+=`<p style="color:#d33">${esc(m.err)}</p>`;h+='</div><div class="st">';
 if(m.t)h+=`<span>${stats(m)}</span>`;if(last&&!busy)h+=`<a onclick="regen()">regenerate</a>`;return h+'</div></div>'}
function render(){const c=conv();$('convs').innerHTML=convs.map(x=>`<div class="cv${x.id===cur?' on':''}" onclick="pick('${x.id}')"><span>${esc(x.title)}</span><b onclick="delConv('${x.id}',event)">✕</b></div>`).join('');
 $('title').textContent=c?c.title:'hyper';
 if(!c||!c.msgs.length){$('msgs').innerHTML='<div class="empty">Start a conversation.</div>';return}
 $('msgs').innerHTML=c.msgs.map((m,k)=>msgHTML(m,k,k===c.msgs.length-1)).join('')}
function updLast(){const c=conv(),box=$('msgs'),el=box.lastElementChild,m=c.msgs[c.msgs.length-1];
 const stick=box.scrollHeight-box.scrollTop-box.clientHeight<80;const t=document.createElement('div');t.innerHTML=msgHTML(m,c.msgs.length-1,true);
 const od=el.querySelector('details');if(od){m.open=od.open}el.replaceWith(t.firstElementChild);
 const nd=box.lastElementChild.querySelector('details div');if(nd&&m.thinking)nd.scrollTop=nd.scrollHeight;if(stick)box.scrollTop=box.scrollHeight}
async function generate(){const c=conv();const a={role:'assistant',content:'',reasoning:'',thinking:true,open:true};c.msgs.push(a);render();
 $('msgs').scrollTop=$('msgs').scrollHeight;
 const msgs=[];if(cfg.sys.trim())msgs.push({role:'system',content:cfg.sys});
 for(const m of c.msgs.slice(0,-1))msgs.push(m.role==='user'?{role:'user',content:m.content}:Object.assign({role:'assistant',content:m.content},m.reasoning?{reasoning_content:m.reasoning}:{}));
 const body={messages:msgs,stream:true,max_tokens:+cfg.maxt||8192,chat_template_kwargs:{enable_thinking:!!cfg.think}};
 if(cfg.temp!=='')body.temperature=+cfg.temp;if(cfg.topp!=='')body.top_p=+cfg.topp;
 busy=new AbortController();$('send').textContent='Stop';let tool='';
 try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:busy.signal});
  if(!r.ok)throw new Error('HTTP '+r.status+' '+await r.text());
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='',tick=0;
  for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
   while((k=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,k).trim();buf=buf.slice(k+2);if(!line.startsWith('data:'))continue;
    const d=line.slice(5).trim();if(d==='[DONE]')continue;let j;try{j=JSON.parse(d)}catch(e){continue}
    if(j.error){a.err=j.error.message;continue}
    const ch=j.choices&&j.choices[0];if(ch&&ch.delta){const dl=ch.delta;
     if(dl.reasoning_content)a.reasoning+=dl.reasoning_content;
     if(dl.content){if(a.thinking){a.thinking=false;a.open=false}a.content+=dl.content}
     if(dl.tool_calls)for(const tc of dl.tool_calls)tool+=(tc.function&&tc.function.name?'\n'+tc.function.name+'(':'')+(tc.function&&tc.function.arguments||'')}
    if(j.timings)a.t=j.timings}
   const now=Date.now();if(now-tick>80){tick=now;updLast()}}
 }catch(e){if(e.name!=='AbortError')a.err=String(e.message||e)}
 if(tool)a.content+='\n```\n'+tool.trim()+'\n```';a.thinking=false;busy=null;$('send').textContent='Send';save();render();refreshCtx()}
function sendMsg(){if(busy){busy.abort();return}const t=$('in').value;if(!t.trim())return;if(!conv())newConv();const c=conv();
 c.msgs.push({role:'user',content:t});if(c.title==='New chat')c.title=t.slice(0,48);$('in').value='';autosz();save();render();generate()}
function regen(){const c=conv();if(busy||!c)return;if(c.msgs.length&&c.msgs[c.msgs.length-1].role==='assistant')c.msgs.pop();generate()}
function autosz(){const t=$('in');t.style.height='auto';t.style.height=Math.min(t.scrollHeight,240)+'px'}
$('in').addEventListener('input',autosz);
$('in').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendMsg()}});
async function refreshCtx(){try{const s=await (await fetch('/stats')).json();$('ctx').textContent='ctx '+s.ctx_used.toLocaleString()+' / '+s.ctx_max.toLocaleString()}catch(e){}}
fetch('/v1/models').then(r=>r.json()).then(j=>{$('mdl').textContent=j.data[0].id}).catch(()=>{});
render();refreshCtx();$('in').focus();
</script></body></html>)HTML";
