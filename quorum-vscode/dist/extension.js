"use strict";var K=Object.create;var f=Object.defineProperty;var B=Object.getOwnPropertyDescriptor;var L=Object.getOwnPropertyNames;var M=Object.getPrototypeOf,D=Object.prototype.hasOwnProperty;var U=(t,e)=>{for(var o in e)f(t,o,{get:e[o],enumerable:!0})},Q=(t,e,o,n)=>{if(e&&typeof e=="object"||typeof e=="function")for(let s of L(e))!D.call(t,s)&&s!==o&&f(t,s,{get:()=>e[s],enumerable:!(n=B(e,s))||n.enumerable});return t};var v=(t,e,o)=>(o=t!=null?K(M(t)):{},Q(e||!t||!t.__esModule?f(o,"default",{value:t,enumerable:!0}):o,t)),W=t=>Q(f({},"__esModule",{value:!0}),t);var ue={};U(ue,{activate:()=>re,deactivate:()=>ie});module.exports=W(ue);var d=v(require("vscode"));var r=v(require("vscode")),l;function R(t,e,o){return r.commands.registerCommand("quorum.ask",async()=>{if(!await G())return;let n=await r.window.showInputBox({prompt:"Ask Quorum...",placeHolder:'e.g. "What is the safest way to store an API key in a VS Code extension?"',ignoreFocusOut:!0});n&&await p(t,e,n,"Quorum: Ask",o)})}function E(t,e,o){return r.commands.registerCommand("quorum.askAboutSelection",async()=>{let n=y("Quorum: highlight some code first.");if(!n)return;let s=await r.window.showInputBox({prompt:"What about this code?",ignoreFocusOut:!0});if(!s)return;let u=`\`\`\`${n.language}
${n.code}
\`\`\`

${s}`;await p(t,e,u,"Quorum: Ask about selection",o)})}function A(t,e,o){return r.commands.registerCommand("quorum.explainSelection",async()=>{let n=y("Quorum: highlight code to explain.");if(!n)return;let s=`Explain the following ${n.language} code in 3-5 sentences. What does it do? What are the edge cases?

\`\`\`${n.language}
${n.code}
\`\`\``;await p(t,e,s,"Quorum: Explain selection",o)})}function O(t,e,o){return r.commands.registerCommand("quorum.reviewSelection",async()=>{let n=y("Quorum: highlight code to review.");if(!n)return;let s=`Review the following ${n.language} code for: bugs, security issues, performance problems, missing edge cases. Be terse. List findings with severity (critical/high/medium/low) and a one-line fix recommendation each.

\`\`\`${n.language}
${n.code}
\`\`\``;await p(t,e,s,"Quorum: Review selection",o)})}function q(t,e,o){return r.commands.registerCommand("quorum.compareImplementations",async()=>{let n=y("Quorum: highlight implementation to compare.");if(!n)return;if(!l){l={code:n.code,language:n.language},await r.window.showInformationMessage(`Quorum: implementation A captured (${n.code.length} chars). Now highlight implementation B and run "Quorum: Compare two implementations" again.`,"Cancel")==="Cancel"&&(l=void 0);return}let s=l;l=void 0;let u=n,c=`Compare implementations A and B for correctness, readability, performance. Which is better and why?

A:
\`\`\`
${s.code}
\`\`\`

B:
\`\`\`
${u.code}
\`\`\``;await p(t,e,c,"Quorum: Compare implementations",o)})}function F(){return r.commands.registerCommand("quorum.openSettings",async()=>{await r.commands.executeCommand("workbench.action.openSettings","@ext:sovereignchain.quorum-vscode")})}function I(){return r.commands.registerCommand("quorum.getFreeKey",async()=>{await r.env.openExternal(r.Uri.parse("https://quorum-ai.dev/signup")),r.window.showInformationMessage("Quorum signup opened in your browser. Enter your email to get a free API key (100 queries/month), then paste it into Settings \u2192 Quorum \u2192 API Key.")})}async function G(){let t=r.workspace.getConfiguration("quorum").get("apiKey","");if(t&&t.trim().length>0)return!0;let e=await r.window.showWarningMessage("No Quorum API key set. Get a free key (100 queries/month, no card) at quorum-ai.dev/signup, then paste it into Settings.","Get Free Key","Open Settings","Dismiss");return e==="Get Free Key"?await r.commands.executeCommand("quorum.getFreeKey"):e==="Open Settings"&&await r.commands.executeCommand("quorum.openSettings"),!1}function y(t){let e=r.window.activeTextEditor;if(!e){r.window.showErrorMessage(t);return}let o=e.document.getText(e.selection);if(o.length===0){r.window.showErrorMessage(t);return}return{code:o,language:e.document.languageId||"text"}}async function p(t,e,o,n,s){let u=r.workspace.getConfiguration("quorum").get("providers",[]).length,c=u>0?`Querying ${u} providers...`:"Querying Quorum...";e.appendLine(`[${new Date().toISOString()}] ${n} \u2014 ${o.length} chars`),s?.setBusy();try{let i=await r.window.withProgress({location:r.ProgressLocation.Notification,title:c,cancellable:!1},async()=>t.ask(o));s?.setSuccess(i),H(n,o,i)}catch(i){let a=i.message??String(i);e.appendLine(`  ERROR: ${a}`),s?.setError(i),r.window.showErrorMessage(`Quorum failed: ${a}`)}}function H(t,e,o){let n=r.window.createWebviewPanel("quorum.result",t,r.ViewColumn.Beside,{enableScripts:!1,retainContextWhenHidden:!0});n.webview.html=V(e,o)}function w(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;")}function V(t,e){let o=r.workspace.getConfiguration("quorum").get("showCostInline",!0),n=Math.round(e.confidence*100),s=e.models.map(i=>{let a=i.error?'<span class="badge error">error</span>':"",S=o?`<td class="num">$${i.cost_usd.toFixed(4)}</td>`:"";return`
        <tr>
          <td><strong>${w(i.name)}</strong> ${a}</td>
          <td class="num">${i.weight.toFixed(2)}</td>
          <td class="num">${i.latency_ms} ms</td>
          ${S}
        </tr>
        <tr class="response-row">
          <td colspan="${o?4:3}">
            <pre>${w(i.error??i.response)}</pre>
          </td>
        </tr>`}).join(""),u=o?"<th>Cost</th>":"",c=o?`<p class="muted">Total cost: $${e.totalCostUsd.toFixed(4)}</p>`:"";return`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline';">
<title>Quorum Result</title>
<style>
  body { font-family: var(--vscode-font-family); padding: 1rem; line-height: 1.5; }
  h1 { font-size: 1.2rem; margin-top: 0; }
  h2 { font-size: 1rem; margin-top: 1.5rem; }
  pre {
    background: var(--vscode-textCodeBlock-background);
    padding: 0.75rem;
    border-radius: 4px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: var(--vscode-editor-font-family);
    font-size: var(--vscode-editor-font-size);
  }
  .prompt { opacity: 0.85; }
  .answer { border-left: 3px solid var(--vscode-textLink-foreground); padding-left: 0.75rem; }
  .confidence {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
    font-weight: bold;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--vscode-panel-border); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .response-row td { padding-top: 0; padding-bottom: 0.75rem; }
  .muted { opacity: 0.7; font-size: 0.9rem; }
  .badge { font-size: 0.75rem; padding: 0.1rem 0.4rem; border-radius: 3px; margin-left: 0.4rem; }
  .badge.error { background: var(--vscode-errorForeground); color: var(--vscode-editor-background); }
</style>
</head>
<body>
  <h1>Consensus answer
    <span class="confidence">${n}% confidence</span>
  </h1>
  <div class="answer"><pre>${w(e.answer)}</pre></div>

  <h2>Prompt</h2>
  <div class="prompt"><pre>${w(t)}</pre></div>

  <h2>Per-model responses (${e.models.length})</h2>
  ${c}
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th class="num">Weight</th>
        <th class="num">Latency</th>
        ${u}
      </tr>
    </thead>
    <tbody>
      ${s||'<tr><td colspan="4" class="muted">No per-model data returned.</td></tr>'}
    </tbody>
  </table>
</body>
</html>`}var P=v(require("vscode")),b=class extends Error{status;body;constructor(e,o,n){super(e),this.name="QuorumError",this.status=o,this.body=n}},z="https://quorum-api-86770458722.europe-west1.run.app",j="quorum",J="quorum.apiKey",Y=3e4,X=(t,e)=>globalThis.fetch(t,e);var C=class{constructor(e){this.secrets=e}async ask(e,o={}){let n=P.workspace.getConfiguration(j),s=n.get("endpoint","https://api.quorum-ai.dev"),u=await this.resolveApiKey(n),c=o.providers??n.get("providers",[]),i=o.maxLatencyMs??n.get("maxLatencyMs",Y),a="/v1/consensus",S=T(s)+a,N=JSON.stringify({prompt:e,providers:c,max_latency_ms:i}),$={"Content-Type":"application/json"};u&&($["X-Quorum-API-Key"]=u);let h=await this.fetchWithFallback(S,a,{method:"POST",headers:$,body:N},i),x=await h.text();if(!h.ok)throw new b(ee(h.status,x),h.status,x);return oe(x)}async resolveApiKey(e){if(this.secrets){let o=await this.secrets.get(J);if(o&&o.length>0)return o}return e.get("apiKey","")}async fetchWithFallback(e,o,n,s){try{return await this.timedFetch(e,n,s)}catch(u){if(!Z(u))throw u;let c=T(z)+o;return await this.timedFetch(c,n,s)}}async timedFetch(e,o,n){let s=new AbortController,u=setTimeout(()=>s.abort(),n);try{return await X(e,{...o,signal:s.signal})}finally{clearTimeout(u)}}};function T(t){return t.replace(/\/+$/,"")}function Z(t){if(!t||typeof t!="object")return!1;let e=t,o=e.cause?.code??e.code??"",n=e.message??"";return o==="ENOTFOUND"||o==="ECONNREFUSED"||o==="EAI_AGAIN"||/ENOTFOUND|ECONNREFUSED|EAI_AGAIN|getaddrinfo/i.test(n)}function ee(t,e){return t===401||t===403?"Invalid API key. Set quorum.apiKey in settings.":t===429?"Rate limited. Try again in a minute.":t>=500?"Quorum server error.":`Quorum request failed (HTTP ${t}): ${te(e,200)}`}function te(t,e){return t.length<=e?t:t.slice(0,e)+"..."}function oe(t){let e;try{e=JSON.parse(t)}catch(i){throw new b(`Quorum returned non-JSON response: ${i.message}`,200,t)}let o=e??{},s=(o.models??[]).map(i=>{let a=i??{};return{name:String(a.name??a.provider??"unknown"),response:String(a.response??a.text??a.answer??""),weight:g(a.weight,0),latency_ms:g(a.latency_ms??a.latencyMs,0),cost_usd:g(a.cost_usd??a.costUsd,0),status:a.status!=null?String(a.status):void 0,error:a.error!=null?String(a.error):void 0}}),u=g(o.totalCostUsd??o.total_cost_usd,NaN),c=Number.isFinite(u)?u:s.reduce((i,a)=>i+a.cost_usd,0);return{answer:String(o.answer??""),confidence:g(o.confidence,0),models:s,totalCostUsd:c}}function g(t,e){return typeof t=="number"&&Number.isFinite(t)?t:e}var m=v(require("vscode")),ne="quorum",k=class{item;state={kind:"idle"};lastResult;constructor(){this.item=m.window.createStatusBarItem(m.StatusBarAlignment.Right,100),this.item.command="quorum.ask",this.render(),this.item.show()}setBusy(){this.state={kind:"busy"},this.render()}setSuccess(e){this.lastResult=e,this.state={kind:"idle"},this.render()}setError(e){let o=e instanceof Error?e.message:String(e);this.state={kind:"error",message:o},this.render()}refresh(){this.render()}dispose(){this.item.dispose()}render(){let e=m.workspace.getConfiguration(ne).get("endpoint","https://api.quorum-ai.dev");switch(this.state.kind){case"busy":this.item.text="$(sync~spin) Quorum\u2026",this.item.tooltip=this.buildTooltip(e);break;case"error":this.item.text="$(error) Quorum",this.item.tooltip=`Quorum: last query failed \u2014 ${this.state.message}

${this.buildTooltip(e)}`;break;case"idle":default:this.item.text="$(symbol-class) Quorum",this.item.tooltip=this.buildTooltip(e);break}}buildTooltip(e){let o=this.lastResult;return o?`Click to ask Quorum. Endpoint: ${e}. Last query cost: $${o.totalCostUsd.toFixed(6)} | confidence ${(o.confidence*100).toFixed(0)}%`:`Click to ask Quorum. Endpoint: ${e}.`}};var _="quorum",se="quorum.apiKey";async function re(t){await ae(t);let e=d.window.createOutputChannel("Quorum");t.subscriptions.push(e);let o=new C(t.secrets),n=new k;t.subscriptions.push(n),t.subscriptions.push(d.workspace.onDidChangeConfiguration(s=>{s.affectsConfiguration(`${_}.endpoint`)&&n.refresh()})),t.subscriptions.push(R(o,e,n),E(o,e,n),A(o,e,n),O(o,e,n),q(o,e,n),F(),I()),e.appendLine(`[${new Date().toISOString()}] Quorum activated (v${t.extension.packageJSON.version??"unknown"}).`)}function ie(){}async function ae(t){let e=d.workspace.getConfiguration(_),o=e.inspect("apiKey"),n=o?.globalValue??o?.workspaceValue??"";if(!(!n||n.length===0))try{await t.secrets.store(se,n),o?.globalValue!==void 0&&await e.update("apiKey",void 0,d.ConfigurationTarget.Global),o?.workspaceValue!==void 0&&await e.update("apiKey",void 0,d.ConfigurationTarget.Workspace),d.window.showWarningMessage("Quorum: Moved your API key from settings to VS Code secret storage.")}catch(s){console.error("Quorum: failed to migrate apiKey to secret storage",s)}}0&&(module.exports={activate,deactivate});
