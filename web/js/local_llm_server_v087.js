import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT = "LocalLLM.PersistentServer";
const SIDEBAR_ID = "local-llm-server";
let status = {
  state: "stopped",
  queue_count: 0,
  current_tokens_per_second: 0,
  last_average_tokens_per_second: null,
  last_tokens: null,
  last_request_seconds: null,
  capabilities: null,
  matching_vision: null,
};
let config = null;
let catalog = null;
let modal = null;
let logsTimer = null;
let statusTimer = null;
let vramEstimate = null;
let vramEstimateTimer = null;
let vramEstimateSeq = 0;
let lastVramEstimateAt = 0;
let tunerState = {state:"idle",running:false,results:[],recommendation:null};
let tunerRefreshPending = false;
let dirty = false;
let saving = false;
let currentModelInfo = null;
let floatingStatus = null;
let floatingDrag = null;
let allowNativeSidebarToggle = false;
let sidebarCollapsePending = false;
const FLOAT_POS_KEY = "local-llm-floating-status-position-v1";
const serviceGenerateNodes = new Set();
let serviceGenerateGraphConfigureDepth = 0;
const REQUEST_PRESET_FIELDS = ["temperature","top_p","top_k","min_p","repeat_penalty","presence_penalty","frequency_penalty","max_tokens"];

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (m) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
}

function injectCSS() {
  if (document.getElementById("local-llm-server-css")) return;
  const style = document.createElement("style");
  style.id = "local-llm-server-css";
  style.textContent = `
    /* Sidebar robot is a pseudo-element on ComfyUI's own launcher button.
       Do not depend on ComfyUI creating a particular <i>/<svg> icon child: that
       DOM has changed between frontend releases and caused the robot to vanish. */
    [data-local-llm-launcher="1"]{--local-llm-robot:#8b8b8b;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;gap:4px!important}
    [data-local-llm-launcher="1"]::before{content:"";display:block!important;width:20px!important;height:20px!important;min-width:20px!important;min-height:20px!important;flex:0 0 20px!important;background-color:var(--local-llm-robot)!important;
      -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cg fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3v2'/%3E%3Ccircle cx='12' cy='2.5' r='.7' fill='black'/%3E%3Crect x='5' y='6' width='14' height='12' rx='3'/%3E%3Cpath d='M5 10H3v4h2M19 10h2v4h-2M9 18v2M15 18v2M8.5 11.2h.01M15.5 11.2h.01M9 14.5h6'/%3E%3C/g%3E%3C/svg%3E") center/20px 20px no-repeat;
      mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cg fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3v2'/%3E%3Ccircle cx='12' cy='2.5' r='.7' fill='black'/%3E%3Crect x='5' y='6' width='14' height='12' rx='3'/%3E%3Cpath d='M5 10H3v4h2M19 10h2v4h-2M9 18v2M15 18v2M8.5 11.2h.01M15.5 11.2h.01M9 14.5h6'/%3E%3C/g%3E%3C/svg%3E") center/20px 20px no-repeat;
      transition:background-color .15s ease,filter .15s ease,transform .15s ease}
    .local-llm-sidebar-ready{--local-llm-robot:#38d26f}.local-llm-sidebar-ready::before{filter:drop-shadow(0 0 5px rgba(56,210,111,.55))}
    .local-llm-sidebar-processing{--local-llm-robot:#55ef8b}.local-llm-sidebar-processing::before{filter:drop-shadow(0 0 8px rgba(85,239,139,.85));animation:localLLMPulse .8s ease-in-out infinite alternate}
    .local-llm-sidebar-generating{--local-llm-robot:#55ef8b}.local-llm-sidebar-generating::before{filter:drop-shadow(0 0 8px rgba(85,239,139,.85));animation:localLLMPulse .9s ease-in-out infinite alternate}
    .local-llm-sidebar-loading{--local-llm-robot:#f3b83f}.local-llm-sidebar-loading::before{filter:drop-shadow(0 0 6px rgba(243,184,63,.6));animation:localLLMPulse .8s ease-in-out infinite alternate}
    .local-llm-sidebar-yielded{--local-llm-robot:#65a9ff}.local-llm-sidebar-yielded::before{filter:drop-shadow(0 0 5px rgba(101,169,255,.5))}
    .local-llm-sidebar-error{--local-llm-robot:#ff5b5b}.local-llm-sidebar-error::before{filter:drop-shadow(0 0 5px rgba(255,91,91,.55))}
    @keyframes localLLMPulse{from{opacity:.55;transform:scale(.96)}to{opacity:1;transform:scale(1.08)}}
    #local-llm-modal-root{position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.56);display:flex;align-items:center;justify-content:center;padding:24px;font-family:Inter,system-ui,sans-serif}
    #local-llm-modal{width:min(980px,96vw);height:min(820px,92vh);background:var(--comfy-menu-bg,#202020);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#4c4c4c);border-radius:12px;box-shadow:0 18px 70px rgba(0,0,0,.55);display:flex;flex-direction:column;overflow:hidden}
    .llm-head{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid #444;background:#252525}.llm-head h2{font-size:18px;margin:0;flex:1}.llm-close{font-size:22px;background:none;border:0;color:#bbb;cursor:pointer}
    .llm-status-dot{width:11px;height:11px;border-radius:50%;background:#777;display:inline-block}.llm-status-dot.yielded{background:#d5a600;box-shadow:0 0 8px rgba(213,166,0,.45)}
.llm-status-dot.processing{background:#55ef8b;box-shadow:0 0 8px rgba(85,239,139,.7);animation:localLLMPulse .8s infinite alternate}.llm-status-dot.ready{background:#38d26f;box-shadow:0 0 8px rgba(56,210,111,.7)}.llm-status-dot.generating{background:#55ef8b;animation:localLLMPulse .8s infinite alternate}.llm-status-dot.loading,.llm-status-dot.reloading{background:#f3b83f;animation:localLLMPulse .8s infinite alternate}.llm-status-dot.error{background:#ff5b5b}.llm-status-dot.stopped{background:#777}
    .llm-tabs{display:flex;gap:3px;padding:8px 12px 0;border-bottom:1px solid #444;background:#1d1d1d}.llm-tab{padding:9px 14px;border:0;border-radius:7px 7px 0 0;background:transparent;color:#aaa;cursor:pointer}.llm-tab.active{background:#303030;color:#fff}
    .llm-body{padding:16px 18px;overflow:auto;flex:1}.llm-pane{display:none}.llm-pane.active{display:block}.llm-grid{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:10px 14px;align-items:center}.llm-grid label{font-size:13px;color:#bbb}.llm-grid input,.llm-grid select,.llm-grid textarea{width:100%;box-sizing:border-box;background:#151515;color:#eee;border:1px solid #555;border-radius:6px;padding:8px}.llm-grid input[type=checkbox]{width:auto;justify-self:start}.llm-grid textarea{min-height:88px;resize:vertical}
    .llm-card{border:1px solid #444;background:#272727;border-radius:9px;padding:13px;margin-bottom:14px}.llm-card h3{font-size:14px;margin:0 0 10px;color:#eee}.llm-card-row{display:flex;gap:16px;flex-wrap:wrap}.llm-stat{min-width:130px}.llm-stat b{display:block;font-size:17px;color:#fff}.llm-stat span{font-size:11px;color:#999}
    .llm-vram-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:12px}.llm-vram-box{border:1px solid #444;background:#202020;border-radius:7px;padding:9px 10px;min-width:0}.llm-vram-box b{display:block;font-size:15px;font-variant-numeric:tabular-nums;color:#eee;white-space:nowrap}.llm-vram-box span{display:block;margin-top:2px;font-size:10.5px;color:#999}.llm-vram-breakdown{display:grid;grid-template-columns:minmax(135px,1fr) auto;gap:5px 14px;font-size:12px;align-items:center}.llm-vram-breakdown .value{font-variant-numeric:tabular-nums;text-align:right;color:#ddd}.llm-vram-meter{height:7px;border-radius:999px;background:#171717;border:1px solid #3d3d3d;overflow:hidden;margin:10px 0 7px}.llm-vram-meter>span{display:block;height:100%;width:0;background:#4ea96b;transition:width .18s ease}.llm-vram-meter.warn>span{background:#d0a13a}.llm-vram-meter.danger>span{background:#c95c5c}.llm-vram-warning{margin-top:9px;font-size:11.5px;color:#f2c25b}.llm-vram-caption{margin-top:7px;font-size:10.5px;color:#999;line-height:1.35}
    .llm-actions{display:flex;gap:8px;flex-wrap:wrap}.llm-btn{border:1px solid #555;background:#343434;color:#eee;border-radius:6px;padding:8px 13px;cursor:pointer}.llm-btn:hover{background:#404040}.llm-btn.primary{background:#246b3e;border-color:#328954}.llm-btn.danger{background:#6b2b2b;border-color:#8e3b3b}.llm-btn.warn{background:#6b5524;border-color:#8e7132}.llm-btn:disabled{opacity:.45;cursor:not-allowed}.llm-footer{display:flex;gap:8px;justify-content:flex-end;padding:11px 16px;border-top:1px solid #444;background:#1d1d1d}.llm-muted{color:#999;font-size:12px}.llm-error{color:#ff8585;white-space:pre-wrap}.llm-success{color:#58dc87}.llm-warn{color:#f2c25b}.llm-log{background:#111;border:1px solid #444;border-radius:7px;padding:10px;font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:pre-wrap;min-height:360px;max-height:520px;overflow:auto}.llm-api-row{display:flex;gap:8px}.llm-api-row input{flex:1}.llm-sidebar-launch{padding:12px}.llm-sidebar-launch .llm-btn{width:100%}.llm-note{padding:9px 11px;border-radius:6px;background:#202b24;border:1px solid #36523d;font-size:12px;margin:10px 0}
    .local-llm-node-preset-actions{display:flex;width:100%;height:28px;gap:6px;box-sizing:border-box;align-items:stretch;padding:0 1px}.local-llm-node-preset-actions button{min-width:0;height:100%;border:1px solid var(--border-color,#555);border-radius:4px;background:var(--comfy-input-bg,#2b2b2b);color:var(--input-text,#ddd);font:inherit;cursor:pointer;line-height:1;padding:0 8px}.local-llm-node-preset-actions button:first-child{flex:1 1 auto}.local-llm-node-preset-actions button:last-child{flex:0 0 auto;min-width:72px}.local-llm-node-preset-actions button:hover{background:var(--comfy-menu-secondary-bg,#3a3a3a)}.local-llm-node-preset-actions button.delete:hover{border-color:#985555;color:#ffb0b0}
    .llm-tuner-progress{height:9px;border:1px solid #454545;background:#171717;border-radius:999px;overflow:hidden;margin:10px 0}.llm-tuner-progress>span{display:block;height:100%;width:0;background:#4ea96b;transition:width .2s ease}.llm-tuner-table-wrap{overflow:auto;max-height:360px;border:1px solid #444;border-radius:7px}.llm-tuner-table{width:100%;border-collapse:collapse;font-size:11.5px;white-space:nowrap}.llm-tuner-table th,.llm-tuner-table td{padding:7px 9px;border-bottom:1px solid #383838;text-align:right;font-variant-numeric:tabular-nums}.llm-tuner-table th:first-child,.llm-tuner-table td:first-child{text-align:left;white-space:normal;min-width:180px}.llm-tuner-table th{position:sticky;top:0;background:#202020;color:#bbb;z-index:1}.llm-tuner-table tr.best td{background:#203126}.llm-tuner-table tr.skipped td{opacity:.55}.llm-tuner-table tr.error td{color:#e99797}.llm-tuner-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px;margin:10px 0}.llm-tuner-summary>div{border:1px solid #444;background:#202020;border-radius:7px;padding:9px}.llm-tuner-summary b{display:block;font-size:15px}.llm-tuner-summary span{font-size:10.5px;color:#999}.llm-tuner-options{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:10px 14px;align-items:center}.llm-tuner-options input,.llm-tuner-options select{width:100%;box-sizing:border-box;background:#151515;color:#eee;border:1px solid #555;border-radius:6px;padding:8px}.llm-tuner-options input[type=checkbox]{width:auto;justify-self:start}
    /* Keep the floating status on the workflow/canvas layer. ComfyUI sidebars,
       menus, dialogs, and popovers use higher UI layers and therefore naturally
       render above it. */
    #local-llm-floating-status{position:fixed;z-index:2;width:max-content;min-width:300px;max-width:min(460px,calc(100vw - 16px));box-sizing:border-box;padding:8px 10px;border:1px solid var(--border-color,#454545);border-radius:8px;background:color-mix(in srgb,var(--comfy-menu-bg,#202020) 94%,transparent);color:var(--fg-color,#ddd);box-shadow:0 5px 18px rgba(0,0,0,.34);font-family:Inter,system-ui,sans-serif;user-select:none;cursor:grab;backdrop-filter:blur(5px);-webkit-backdrop-filter:blur(5px)}
    #local-llm-floating-status.dragging{cursor:grabbing}.llm-float-top{display:grid;grid-template-columns:21px minmax(0,1fr) auto;align-items:center;column-gap:7px;min-width:0}.llm-float-robot{display:flex;width:21px;height:21px;flex:0 0 21px;color:#888;grid-column:1;grid-row:1}.llm-float-robot svg{width:21px;height:21px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.llm-float-name{font-size:12px;font-weight:650;white-space:nowrap;grid-column:2;grid-row:1;min-width:0}.llm-float-speed{margin-left:auto;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--fg-color,#ddd);grid-column:3;grid-row:1}.llm-float-sub,.llm-float-last{margin:3px 0 0 0;font-size:10.5px;line-height:1.3;color:var(--descrip-text,#999);white-space:normal;overflow:visible;text-overflow:clip;max-width:none;overflow-wrap:break-word}.llm-float-sub{margin-top:4px}.llm-float-last{font-variant-numeric:tabular-nums;opacity:.92}
    #local-llm-floating-status.processing .llm-float-robot{color:#55ef8b;filter:drop-shadow(0 0 6px rgba(85,239,139,.75));animation:localLLMPulse .8s ease-in-out infinite alternate}#local-llm-floating-status.ready .llm-float-robot{color:#38d26f;filter:drop-shadow(0 0 4px rgba(56,210,111,.5))}#local-llm-floating-status.generating .llm-float-robot{color:#55ef8b;filter:drop-shadow(0 0 6px rgba(85,239,139,.75));animation:localLLMPulse .9s ease-in-out infinite alternate}#local-llm-floating-status.loading .llm-float-robot,#local-llm-floating-status.reloading .llm-float-robot,#local-llm-floating-status.tuning .llm-float-robot{color:#f3b83f;animation:localLLMPulse .8s ease-in-out infinite alternate}#local-llm-floating-status.yielded .llm-float-robot{color:#65a9ff;filter:drop-shadow(0 0 4px rgba(101,169,255,.45))}#local-llm-floating-status.error .llm-float-robot{color:#ff5b5b;filter:drop-shadow(0 0 4px rgba(255,91,91,.5))}
  `;
  document.head.appendChild(style);
}

function stateClass(s) {
  if (s === "ready") return "ready";
  if (s === "yielded") return "yielded";
  if (s === "generating") return "generating";
  if (s === "processing") return "processing";
  if (s === "loading" || s === "reloading" || s === "stopping" || s === "tuning") return "loading";
  if (s === "error") return "error";
  return "stopped";
}

// Sidebar/menu robot should pulse only while real LLM work is active.
// In particular, do not map "stopping" to the animated loading state.
function sidebarStateClass(s) {
  if (s === "loading" || s === "reloading" || s === "tuning") return "loading";
  if (s === "processing") return "processing";
  if (s === "generating") return "generating";
  if (s === "ready") return "ready";
  if (s === "yielded") return "yielded";
  if (s === "error") return "error";
  return "stopped";
}

function displayState() {
  if (status?.state === "ready" && status?.vram_yielded) return "yielded";
  return status?.state || "stopped";
}

function robotHeadSVG() {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v2"/><circle cx="12" cy="2.5" r=".7"/><rect x="5" y="6" width="14" height="12" rx="3"/><path d="M5 10H3v4h2M19 10h2v4h-2M9 18v2M15 18v2M8.5 11.2h.01M15.5 11.2h.01M9 14.5h6"/></svg>`;
}

function loadFloatingPosition() {
  try {
    const p = JSON.parse(localStorage.getItem(FLOAT_POS_KEY) || "null");
    if (p && Number.isFinite(p.left) && Number.isFinite(p.top)) return p;
  } catch (_) {}
  return null;
}

function saveFloatingPosition(el) {
  try {
    const r = el.getBoundingClientRect();
    localStorage.setItem(FLOAT_POS_KEY, JSON.stringify({left: Math.round(r.left), top: Math.round(r.top)}));
  } catch (_) {}
}

function clampFloating(el) {
  if (!el?.isConnected) return;
  const r = el.getBoundingClientRect();
  const left = Math.max(8, Math.min(r.left, Math.max(8, window.innerWidth - r.width - 8)));
  const top = Math.max(8, Math.min(r.top, Math.max(8, window.innerHeight - r.height - 8)));
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
  el.style.right = "auto";
}

function installFloatingDrag(el) {
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const r = el.getBoundingClientRect();
    floatingDrag = {id:e.pointerId, startX:e.clientX, startY:e.clientY, left:r.left, top:r.top, moved:false};
    el.setPointerCapture?.(e.pointerId);
    el.classList.add("dragging");
  });
  el.addEventListener("pointermove", (e) => {
    if (!floatingDrag || floatingDrag.id !== e.pointerId) return;
    const dx=e.clientX-floatingDrag.startX, dy=e.clientY-floatingDrag.startY;
    if (Math.abs(dx)+Math.abs(dy) > 4) floatingDrag.moved=true;
    el.style.left=`${floatingDrag.left+dx}px`;
    el.style.top=`${floatingDrag.top+dy}px`;
    el.style.right="auto";
  });
  const finish=(e)=>{
    if (!floatingDrag || floatingDrag.id !== e.pointerId) return;
    const moved=floatingDrag.moved;
    floatingDrag=null;
    el.classList.remove("dragging");
    clampFloating(el);
    saveFloatingPosition(el);
    if (!moved) openModal();
  };
  el.addEventListener("pointerup", finish);
  el.addEventListener("pointercancel", (e)=>{ if(floatingDrag?.id===e.pointerId){floatingDrag=null;el.classList.remove("dragging");} });
}

function ensureFloatingStatus(forceVisible=null) {
  const visible = forceVisible == null ? !!config?.show_status_indicator : !!forceVisible;
  if (!visible) {
    floatingStatus?.remove();
    floatingStatus=null;
    return;
  }
  if (!floatingStatus?.isConnected) {
    floatingStatus=document.createElement("div");
    floatingStatus.id="local-llm-floating-status";
    floatingStatus.title="Drag to move • Click to open Local LLM Server";
    floatingStatus.innerHTML=`<div class="llm-float-top"><span class="llm-float-robot">${robotHeadSVG()}</span><span class="llm-float-name">LLM</span><span class="llm-float-speed">0.0 t/s</span></div><div class="llm-float-sub">Stopped</div><div class="llm-float-last">Last: —</div>`;
    document.body.appendChild(floatingStatus);
    const pos=loadFloatingPosition();
    if(pos){floatingStatus.style.left=`${pos.left}px`;floatingStatus.style.top=`${pos.top}px`;}
    else {floatingStatus.style.right="18px";floatingStatus.style.top="72px";}
    installFloatingDrag(floatingStatus);
    requestAnimationFrame(()=>clampFloating(floatingStatus));
  }
  updateFloatingStatus();
}

function updateFloatingStatus() {
  if (!floatingStatus?.isConnected) return;
  const ds=displayState();
  floatingStatus.className=stateClass(ds);
  const speed=floatingStatus.querySelector(".llm-float-speed");
  const sub=floatingStatus.querySelector(".llm-float-sub");
  const last=floatingStatus.querySelector(".llm-float-last");
  if(speed){
    const live=status.current_tokens_per_second;
    if(status.state==="generating") speed.textContent=`${Number(live || 0).toFixed(1)} t/s`;
    else if(status.state==="processing") speed.textContent="Processing";
    else if(status.state==="waiting_comfy") speed.textContent="Waiting";
    else speed.textContent="0.0 t/s";
  }
  let text="Stopped";
  if(ds==="ready") text="Ready";
  else if(ds==="yielded") text="Yielded to ComfyUI";
  else if(ds==="processing") text=`Processing…${status.queue_count ? ` • Q ${status.queue_count}` : ""}`;
  else if(ds==="waiting_comfy") text="Waiting for ComfyUI…";
  else if(ds==="generating") text=`Generating${status.current_completion_tokens ? ` • ${status.current_completion_tokens} tok` : ""}${status.current_prompt_tokens_per_second ? ` • proc ${Number(status.current_prompt_tokens_per_second).toFixed(0)} t/s` : ""}${status.queue_count ? ` • Q ${status.queue_count}` : ""}`;
  else if(ds==="loading") text="Loading…";
  else if(ds==="reloading") text="Reloading…";
  else if(ds==="error") text="Error";
  if(sub) sub.textContent=text;
  if(last){
    const gen = status.last_average_tokens_per_second ?? status.last_tokens_per_second;
    const proc = status.last_prompt_tokens_per_second;
    const toks = status.last_tokens ?? status.last_completion_tokens;
    const secs = status.last_total_seconds ?? status.last_request_seconds ?? status.last_generation_seconds;
    const loadSecs = Number(status.last_load_seconds || 0);
    const hasLast = gen != null || proc != null || toks != null || secs != null;
    const procText = proc != null && Number(proc) > 0 ? ` • ${Number(proc).toFixed(0)} proc t/s` : "";
    const loadText = loadSecs >= 0.05 ? ` • load ${loadSecs.toFixed(1)}s` : "";
    last.textContent = hasLast
      ? `Last: ${Number(toks || 0)} tok • ${Number(gen || 0).toFixed(1)} gen t/s${procText} • ${Number(secs || 0).toFixed(1)}s total${loadText}`
      : "Last: —";
  }
  // The box width is content-driven. Re-clamp after text changes so a longer
  // status/Last line can never grow beyond the viewport edge.
  requestAnimationFrame(() => clampFloating(floatingStatus));
}

function decorateSidebar() {
  const buttons = [...document.querySelectorAll("button,[role='button']")];
  let b = buttons.find((el) => {
    const title = `${el.getAttribute("title") || ""} ${el.getAttribute("aria-label") || ""}`.toLowerCase();
    const txt = (el.textContent || "").trim().toLowerCase();
    return title.includes("local llm server") || txt === "llm";
  });
  if (!b) return;
  b.dataset.localLlmLauncher = "1";
  try { b.disabled = false; b.removeAttribute("disabled"); b.setAttribute("aria-disabled", "false"); } catch (_) {}

  // Clean up robot elements injected by older extension versions.  The current
  // robot is CSS ::before on the launcher itself, so it cannot disappear merely
  // because ComfyUI changes its internal icon child markup.
  try { b.querySelectorAll(".local-llm-robot-slot").forEach((el)=>el.remove()); } catch (_) {}

  // Remove every state class first. v0.10.2 omitted the processing class here,
  // which could leave the pulse animation stuck on after processing finished.
  b.classList.remove("local-llm-sidebar-ready","local-llm-sidebar-processing","local-llm-sidebar-generating","local-llm-sidebar-loading","local-llm-sidebar-yielded","local-llm-sidebar-error");
  const sc = sidebarStateClass(displayState());
  if (sc !== "stopped") b.classList.add(`local-llm-sidebar-${sc}`);
  const ds=displayState();
  const speed=(status.state==="generating" && status.current_tokens_per_second!=null) ? ` • ${Number(status.current_tokens_per_second).toFixed(1)} t/s` : "";
  b.title=`Local LLM Server • ${ds}${speed}`;
  b.setAttribute("aria-label", `Local LLM Server • ${ds}${speed}`);
  b.dataset.localLlmState = ds;
}

async function fetchJSON(path, options={}) {
  const r = await api.fetchApi(path, options);
  let data = null;
  try { data = await r.json(); } catch (_) { data = {}; }
  if (!r.ok) throw new Error(data?.error?.message || data?.message || `${r.status} ${r.statusText}`);
  return data;
}

async function refreshTunerStatus(force=false) {
  if (tunerRefreshPending) return;
  const tunerTabOpen = modal?.querySelector('.llm-tab.active')?.dataset.tab === "tuner";
  if (!force && !tunerState?.running && !tunerTabOpen) return;
  tunerRefreshPending = true;
  try {
    tunerState = await fetchJSON("/local_llm_server/tuner/status");
    renderTuner();
  } catch (_) {} finally { tunerRefreshPending = false; }
}

async function refreshStatus() {
  try {
    status = await fetchJSON("/local_llm_server/status");
    updateLiveUI();
    refreshTunerStatus(false);
  } catch (_) {}
}

async function loadData() {
  const [c, cat, st, tuner] = await Promise.all([
    fetchJSON("/local_llm_server/config"),
    fetchJSON("/local_llm_server/catalog"),
    fetchJSON("/local_llm_server/status"),
    fetchJSON("/local_llm_server/tuner/status").catch(()=>({state:"idle",running:false,results:[],recommendation:null})),
  ]);
  config = c; catalog = cat; status = st; tunerState = tuner;
}

function badgeText() {
  if ((status.queue_count || 0) > 0) return String(status.queue_count);
  if (status.state === "error") return "!";
  return null;
}

function selectOptions(values, selected) {
  return (values || []).map((x) => `<option value="${esc(x)}" ${String(x)===String(selected)?"selected":""}>${esc(x)}</option>`).join("");
}

function field(key, label, type="text", extra="", tooltip="") {
  const v = config?.[key];
  const tip = tooltip ? ` title="${esc(tooltip)}"` : "";
  if (type === "checkbox") return `<label${tip}>${esc(label)}</label><input class="llm-config" data-key="${key}" type="checkbox" ${v?"checked":""} ${extra}${tip}>`;
  if (type === "select") return `<label${tip}>${esc(label)}</label><select class="llm-config" data-key="${key}" ${extra}${tip}></select>`;
  return `<label${tip}>${esc(label)}</label><input class="llm-config" data-key="${key}" type="${type}" value="${esc(v)}" ${type==="number"?'data-type="number"':""} ${extra}${tip}>`;
}

function formatTunerVRAM(bytes) {
  const n=Number(bytes||0);
  return n>0 ? `${(n/(1024**3)).toFixed(2)} GiB` : "—";
}

function renderTuner() {
  if (!modal?.isConnected) return;
  const t=tunerState || {};
  const running=!!t.running;
  const progress=Math.max(0,Math.min(1,Number(t.progress||0)));
  const bar=modal.querySelector("#llm-tuner-progress-bar"); if(bar)bar.style.width=`${(progress*100).toFixed(1)}%`;
  const phase=modal.querySelector("#llm-tuner-phase");
  if(phase){
    const counter=Number(t.candidate_total||0)>0 ? ` • candidate ${Number(t.candidate_index||0)}/${Number(t.candidate_total||0)}` : "";
    const current=t.current_candidate ? ` • ${t.current_candidate}` : "";
    phase.textContent=`${t.phase||t.state||"Idle"}${counter}${current}`;
  }
  const msg=modal.querySelector("#llm-tuner-message"); if(msg)msg.textContent=t.error ? t.error : (t.message||"Ready to benchmark the saved server configuration.");

  const results=Array.isArray(t.results)?t.results:[];
  const tbody=modal.querySelector("#llm-tuner-results");
  const rec=t.recommendation||null;
  const recLabel=rec?.label||null;
  if(tbody){
    if(!results.length) tbody.innerHTML='<tr><td colspan="10" class="llm-muted">No benchmark results yet.</td></tr>';
    else tbody.innerHTML=results.map(r=>{
      const ok=r.status==="ok";
      const cls=[r.status||"", recLabel && r.label===recLabel ? "best" : ""].filter(Boolean).join(" ");
      const load=ok&&Number(r.load_seconds||0)>=0?`${Number(r.load_seconds||0).toFixed(3)}s`:"—";
      const prompt=ok&&Number(r.prompt_tps||0)>0?Number(r.prompt_tps).toFixed(0):"—";
      const decode=ok&&Number(r.decode_tps||0)>0?Number(r.decode_tps).toFixed(2):"—";
      const unload=ok&&Number(r.unload_seconds||0)>=0?`${Number(r.unload_seconds||0).toFixed(3)}s`:"—";
      const infer=ok&&Number(r.fixed_work_seconds||0)>0?`${Number(r.fixed_work_seconds).toFixed(3)}s`:"—";
      const cycle=ok&&Number(r.cycle_fixed_seconds||0)>0?`${Number(r.cycle_fixed_seconds).toFixed(3)}s`:"—";
      const vram=ok?formatTunerVRAM(r.measured_vram_bytes||r.estimated_vram_bytes):formatTunerVRAM(r.estimated_vram_bytes);
      const spec=ok?(r.speculative_effective||"Off"):"—";
      const flags=[];
      if(r.acceptance_rate!=null) flags.push(`${(Number(r.acceptance_rate)*100).toFixed(1)}% accept`);
      if(r.quality_tradeoff) flags.push("KV quality tradeoff");
      const result=ok?(flags.join(" • ")||"OK"):(r.reason||r.status||"—");
      const title=[r.reason||r.label||"",...(Array.isArray(r.tradeoffs)?r.tradeoffs:[])].filter(Boolean).join("\n");
      return `<tr class="${esc(cls)}"><td title="${esc(title)}">${esc(r.label||"Candidate")}</td><td>${load}</td><td>${prompt}</td><td>${decode}</td><td>${unload}</td><td>${infer}</td><td>${cycle}</td><td>${vram}</td><td>${esc(spec)}</td><td>${esc(result)}</td></tr>`;
    }).join("");
  }

  const set=(id,value)=>{const el=modal.querySelector(`#${id}`);if(el)el.textContent=value;};
  if(rec){
    const mode=rec.score_mode||"ComfyUI Cycle";
    set("llm-tuner-gain",`${Number(rec.improvement_percent||0).toFixed(1)}%`);
    set("llm-tuner-gain-label",mode==="Inference Only"?"Inference improvement":"ComfyUI-cycle improvement");
    set("llm-tuner-load",`${Number(rec.load_seconds||0).toFixed(3)}s`);
    set("llm-tuner-prompt",Number(rec.prompt_tps||0)>0?`${Number(rec.prompt_tps).toFixed(0)} t/s`:"—");
    set("llm-tuner-decode",Number(rec.decode_tps||0)>0?`${Number(rec.decode_tps).toFixed(2)} t/s`:"—");
    set("llm-tuner-unload",`${Number(rec.unload_seconds||0).toFixed(3)}s`);
    set("llm-tuner-cycle",`${Number(rec.best_cycle_seconds||0).toFixed(3)}s`);
    set("llm-tuner-vram",formatTunerVRAM(rec.measured_vram_bytes));
    const patch=rec.patch||{};
    const names={
      prompt_batch_size:"n_batch",memory_batch_size:"n_ubatch",flash_attention:"Flash Attention",
      speculative_mode:"Speculative",ngram_pred_tokens:"N-gram draft",mtp_draft_tokens:"MTP draft",
      use_mmap:"mmap",use_mlock:"mlock",kv_cache_location:"KV location",gpu_layers:"GPU layers",
      kv_cache_k:"KV K",kv_cache_v:"KV V"
    };
    const parts=Object.entries(patch).map(([k,v])=>`${names[k]||k}: ${typeof v==="boolean"?(v?"On":"Off"):v}`);
    const settings=modal.querySelector("#llm-tuner-settings");
    let text=parts.length?parts.join(" • "):"Current saved settings were fastest within the tuner's noise threshold; no changes are recommended.";
    const tradeoffs=Array.isArray(rec.tradeoffs)?rec.tradeoffs:[];
    if(tradeoffs.length) text+=`  Considerations: ${tradeoffs.join(" ")}`;
    if(settings){settings.textContent=text;settings.classList.toggle("llm-warn",tradeoffs.length>0);}
  } else {
    set("llm-tuner-gain","—");set("llm-tuner-gain-label","ComfyUI-cycle improvement");
    set("llm-tuner-load","—");set("llm-tuner-prompt","—");set("llm-tuner-decode","—");set("llm-tuner-unload","—");set("llm-tuner-cycle","—");set("llm-tuner-vram","—");
    const settings=modal.querySelector("#llm-tuner-settings"); if(settings){settings.textContent="Run the tuner to produce a recommendation.";settings.classList.remove("llm-warn");}
  }

  const run=modal.querySelector('[data-action="tuner-start"]');
  const cancel=modal.querySelector('[data-action="tuner-cancel"]');
  const use=modal.querySelector('[data-action="tuner-use"]');
  const save=modal.querySelector('[data-action="tuner-save-preset"]');
  const ready=status?.state==="ready" && !!status?.active;
  if(run){run.disabled=running||dirty||saving||!ready;run.title=dirty?"Save the server configuration before benchmarking":(!ready?"Start the Local LLM service before benchmarking":"Benchmark the saved configuration");}
  if(cancel)cancel.disabled=!running;
  if(use)use.disabled=running||!rec;
  if(save)save.disabled=running||!rec;
  for(const id of ["llm-tuner-profile","llm-tuner-score","llm-tuner-headroom","llm-tuner-batches","llm-tuner-flash","llm-tuner-spec","llm-tuner-memory","llm-tuner-kv-precision"]){const el=modal.querySelector(`#${id}`);if(el)el.disabled=running;}
}

async function startTuner() {
  if(dirty){const n=modal?.querySelector("#llm-tuner-message");if(n)n.textContent="Save the server configuration before benchmarking.";return;}
  const body={
    profile:modal.querySelector("#llm-tuner-profile")?.value||"Quick",
    score_mode:modal.querySelector("#llm-tuner-score")?.value||"ComfyUI Cycle",
    safety_headroom_mib:Number(modal.querySelector("#llm-tuner-headroom")?.value||1024),
    tune_batches:!!modal.querySelector("#llm-tuner-batches")?.checked,
    tune_flash_attention:!!modal.querySelector("#llm-tuner-flash")?.checked,
    tune_speculative:!!modal.querySelector("#llm-tuner-spec")?.checked,
    tune_memory:!!modal.querySelector("#llm-tuner-memory")?.checked,
    tune_kv_precision:!!modal.querySelector("#llm-tuner-kv-precision")?.checked,
  };
  try{
    tunerState=await fetchJSON("/local_llm_server/tuner/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    renderTuner();
    refreshStatus();
  }catch(e){tunerState={...(tunerState||{}),state:"error",running:false,error:e.message,message:e.message};renderTuner();}
}

async function cancelTuner() {
  try{tunerState=await fetchJSON("/local_llm_server/tuner/cancel",{method:"POST"});renderTuner();}catch(e){const n=modal?.querySelector("#llm-tuner-message");if(n)n.textContent=e.message;}
}

function useTunerRecommendation() {
  const rec=tunerState?.recommendation;
  if(!rec)return;
  for(const [k,v] of Object.entries(rec.patch||{})) setConfigField(k,v);
  const mem=modal.querySelector("#llm-memory-preset"); if(mem)mem.value="Custom";
  applySpeculativeUI(currentModelInfo || {capabilities:status?.capabilities,speculative_support:status?.speculative_support,speculative:status?.speculative});
  setDirty(true);scheduleVRAMEstimate(60);renderTuner();
}

async function saveTunerPreset() {
  const rec=tunerState?.recommendation;
  if(!rec)return;
  const modelName=String(config?.model||"Local LLM").split(/[\\/]/).pop().replace(/\.gguf$/i,"");
  const name=window.prompt("Name this memory/performance preset:",`${modelName} Tuned`);
  if(!name)return;
  try{
    const d=await fetchJSON("/local_llm_server/memory_presets",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name,settings:rec.settings||{}})});
    catalog=d.catalog||catalog;
    const sel=modal.querySelector("#llm-memory-preset");
    if(sel)sel.innerHTML=selectOptions(catalog.memory_presets||["Custom"],sel.value||"Custom");
    const n=modal.querySelector("#llm-tuner-message");if(n)n.textContent=`Saved memory preset: ${d.saved?.name||name}`;
  }catch(e){const n=modal.querySelector("#llm-tuner-message");if(n)n.textContent=`Preset save failed: ${e.message}`;}
}

function renderModal() {
  if (!config || !catalog) return;
  if (modal) modal.remove();
  modal = document.createElement("div");
  modal.id = "local-llm-modal-root";
  const apiBase = `${location.origin}/local-llm/v1`;
  modal.innerHTML = `
    <div id="local-llm-modal">
      <div class="llm-head"><span class="llm-status-dot ${stateClass(displayState())}" id="llm-live-dot"></span><h2>Local LLM Server</h2><span id="llm-head-state" class="llm-muted">${esc(status.state)}</span><button class="llm-close" title="Close">×</button></div>
      <div class="llm-tabs">
        <button class="llm-tab active" data-tab="server">Server</button><button class="llm-tab" data-tab="model">Model</button><button class="llm-tab" data-tab="memory">Memory</button><button class="llm-tab" data-tab="tuner">Tuner</button><button class="llm-tab" data-tab="api">API</button><button class="llm-tab" data-tab="logs">Logs</button>
      </div>
      <div class="llm-body">
        <div class="llm-pane active" data-pane="server">
          <div class="llm-card"><h3>Runtime</h3><div class="llm-card-row">
            <div class="llm-stat"><b id="llm-stat-model">${esc(status.model || "—")}</b><span>Model</span></div>
            <div class="llm-stat"><b id="llm-stat-state">${esc(status.state)}</b><span>Status</span></div>
            <div class="llm-stat"><b id="llm-stat-speed">${status.state === "generating" ? Number(status.current_tokens_per_second || 0).toFixed(2) : "0.00"}</b><span>tok/s</span></div>
            <div class="llm-stat"><b id="llm-stat-generated">${status.current_completion_tokens || status.last_completion_tokens || 0}</b><span>Generated tokens</span></div>
            <div class="llm-stat"><b id="llm-stat-queue">${status.queue_count || 0}</b><span>Queued</span></div>
            <div class="llm-stat"><b id="llm-stat-requests">${status.requests_total || 0}</b><span>Requests</span></div>
          </div><div id="llm-runtime-note" class="llm-note">${runtimeNote()}</div></div>
          <div class="llm-card"><h3>Controls</h3><div class="llm-actions">
            <button class="llm-btn primary" data-action="start">Start</button><button class="llm-btn warn" data-action="suspend">Suspend</button><button class="llm-btn danger" data-action="stop">Stop / Unload</button>
          </div></div>
          <div class="llm-card"><h3>Startup</h3><div class="llm-grid">
            <label>Startup mode</label><select class="llm-config" data-key="startup_mode">${selectOptions(["Off","On Demand","Auto Start"],config.startup_mode)}</select>
            <label>Current client</label><div id="llm-current-client">${esc(status.current_client || "Idle")}</div>
          </div></div>
          <div class="llm-card"><h3>Interface</h3><div class="llm-grid">
            ${field("show_status_indicator","Show floating status box","checkbox")}
          </div><div class="llm-note">The floating status box is draggable, survives workflow/tab changes, and opens this LLM panel when clicked. Its robot icon uses the same live status colors as the sidebar icon.</div></div>
          <div id="llm-global-error" class="llm-error">${esc(status.error || "")}</div>
        </div>
        <div class="llm-pane" data-pane="model">
          <div class="llm-card"><h3>GGUF model</h3><div class="llm-grid">
            <label>Model</label><select class="llm-config" data-key="model" id="llm-model-select">${selectOptions(catalog.models,config.model)}</select>
            <label id="llm-vision-model-label" title="Vision projector used by multimodal GGUF models. Auto selects a compatible mmproj when one can be identified safely.">Vision / mmproj</label><select class="llm-config" data-key="vision_model" id="llm-vision-model" title="Vision projector used by multimodal GGUF models. Auto selects a compatible mmproj when one can be identified safely.">${selectOptions(catalog.vision,config.vision_model)}</select>
            <label>Detected family</label><div id="llm-detected-family">${esc(status.family || "unknown")}</div>
            <label>Capabilities</label><div id="llm-model-capabilities" class="llm-muted">Detecting…</div>
            <label class="llm-vision-setting">Max still images</label><input class="llm-config llm-vision-setting" data-key="vision_max_images" data-type="number" type="number" min="1" max="32" value="${esc(config.vision_max_images ?? 4)}">
            <label class="llm-vision-setting">Max video frames</label><input class="llm-config llm-vision-setting" data-key="vision_max_frames" data-type="number" type="number" min="1" max="1024" value="${esc(config.vision_max_frames ?? 24)}">
            <label class="llm-vision-setting">Vision max edge</label><input class="llm-config llm-vision-setting" data-key="vision_max_edge" data-type="number" type="number" min="256" max="4096" step="64" value="${esc(config.vision_max_edge ?? 1536)}">
            <label>Model preset</label><select class="llm-config" data-key="model_preset" id="llm-model-preset"></select>
            <label>Thinking mode</label><select class="llm-config model-owned" data-key="thinking_mode">${selectOptions(["Auto","Enabled","Disabled"],config.thinking_mode)}</select>
            <label>Reasoning effort</label><select class="llm-config model-owned" data-key="reasoning_effort">${selectOptions(["Auto","Low","Medium","High","XHigh"],config.reasoning_effort)}</select>
            ${field("preserve_thinking","Preserve thinking history","checkbox")}
            ${field("temperature","Temperature","number",'step="0.01"')}
            ${field("top_p","Top P","number",'step="0.01"')}
            ${field("top_k","Top K","number")}
            ${field("min_p","Min P","number",'step="0.01"')}
            ${field("repeat_penalty","Repeat penalty","number",'step="0.01"')}
            ${field("presence_penalty","Presence penalty","number",'step="0.01"')}
            ${field("frequency_penalty","Frequency penalty","number",'step="0.01"')}
            ${field("max_tokens","Default max tokens","number")}
          </div></div>
        </div>
        <div class="llm-pane" data-pane="memory">
          <div class="llm-card"><h3>Live VRAM Estimate</h3>
            <div class="llm-vram-summary">
              <div class="llm-vram-box"><b id="llm-vram-est-total">Calculating…</b><span>Estimated total</span></div>
              <div class="llm-vram-box"><b id="llm-vram-measured">—</b><span>Measured native</span></div>
              <div class="llm-vram-box"><b id="llm-vram-free">—</b><span>Current free</span></div>
              <div class="llm-vram-box"><b id="llm-vram-headroom">—</b><span>Projected headroom</span></div>
            </div>
            <div class="llm-vram-meter" id="llm-vram-meter"><span></span></div>
            <div class="llm-vram-breakdown">
              <span>Model weights</span><span class="value" id="llm-vram-weights">—</span>
              <span>KV cache</span><span class="value" id="llm-vram-kv">—</span>
              <span>Compute / batch</span><span class="value" id="llm-vram-compute">—</span>
              <span>Speculative runtime</span><span class="value" id="llm-vram-spec">—</span>
              <span>Vision / mmproj <small>(if used)</small></span><span class="value" id="llm-vram-vision">—</span>
              <span>Base text total</span><span class="value" id="llm-vram-base">—</span>
              <span>Auto-Yield reload target</span><span class="value" id="llm-vram-target">—</span>
            </div>
            <div class="llm-vram-warning" id="llm-vram-warning" style="display:none"></div>
            <div class="llm-vram-caption" id="llm-vram-caption">Estimate is conservative until this exact configuration has completed a verified load.</div>
          </div>
          <div class="llm-card"><h3>Speculative Decoding</h3><div class="llm-grid">
            <label title="Lossless acceleration: draft tokens are always verified by the target model. Auto prefers embedded MTP when genuinely supported, otherwise N-gram.">Mode</label><select class="llm-config" data-key="speculative_mode" id="llm-spec-mode" title="Lossless acceleration: draft tokens are always verified by the target model. Auto prefers embedded MTP when genuinely supported, otherwise N-gram.">${selectOptions(["Off","Auto","N-gram","MTP"],config.speculative_mode || "Off")}</select>
            <label>Runtime support</label><div id="llm-spec-support" class="llm-muted">Detecting…</div>
            <label class="llm-spec-ngram-setting" title="Maximum tokens proposed per N-gram draft step. Larger values can help repetitive output but may waste verification work when acceptance is low.">N-gram draft tokens</label><input class="llm-config llm-spec-ngram-setting" data-key="ngram_pred_tokens" data-type="number" type="number" min="1" max="64" value="${esc(config.ngram_pred_tokens ?? 10)}" title="Maximum tokens proposed per N-gram draft step. Larger values can help repetitive output but may waste verification work when acceptance is low.">
            <label class="llm-spec-ngram-setting" title="N-gram match length used to find draft continuations in prior context.">N-gram size</label><input class="llm-config llm-spec-ngram-setting" data-key="ngram_size" data-type="number" type="number" min="1" max="16" value="${esc(config.ngram_size ?? 3)}" title="N-gram match length used to find draft continuations in prior context.">
            <label class="llm-spec-ngram-setting" title="k is the normal map lookup. k4v uses the alternate four-value map mode when supported by the installed binding.">N-gram mode</label><select class="llm-config llm-spec-ngram-setting" data-key="ngram_mode" title="k is the normal map lookup. k4v uses the alternate four-value map mode when supported by the installed binding.">${selectOptions(["k","k4v"],config.ngram_mode || "k")}</select>
            <label class="llm-spec-ngram-setting" title="Minimum number of matching observations before an N-gram continuation is considered.">Minimum hits</label><input class="llm-config llm-spec-ngram-setting" data-key="ngram_min_hits" data-type="number" type="number" min="1" max="64" value="${esc(config.ngram_min_hits ?? 2)}" title="Minimum number of matching observations before an N-gram continuation is considered.">
            <label class="llm-spec-ngram-setting" title="Maximum stored continuations per N-gram key. 0 lets the implementation use an unlimited/default value.">Entries per key</label><input class="llm-config llm-spec-ngram-setting" data-key="ngram_max_entries_per_key" data-type="number" type="number" min="0" max="1024" value="${esc(config.ngram_max_entries_per_key ?? 8)}" title="Maximum stored continuations per N-gram key. 0 lets the implementation use an unlimited/default value.">
            <label class="llm-spec-ngram-setting" title="How often the map implementation checks/synchronizes its rolling context state.">Sync check tokens</label><input class="llm-config llm-spec-ngram-setting" data-key="ngram_sync_check_tokens" data-type="number" type="number" min="1" max="1024" value="${esc(config.ngram_sync_check_tokens ?? 16)}" title="How often the map implementation checks/synchronizes its rolling context state.">
            <label class="llm-spec-mtp-setting" title="Maximum embedded NextN/MTP tokens proposed per step. Start small; acceptance rate determines whether larger values help.">MTP draft tokens</label><input class="llm-config llm-spec-mtp-setting" data-key="mtp_draft_tokens" data-type="number" type="number" min="1" max="8" value="${esc(config.mtp_draft_tokens ?? 2)}" title="Maximum embedded NextN/MTP tokens proposed per step. Start small; acceptance rate determines whether larger values help.">
            <label class="llm-spec-mtp-setting" title="Minimum probability threshold used by the native MTP draft verifier. Higher values are more selective.">MTP p-min</label><input class="llm-config llm-spec-mtp-setting" data-key="mtp_p_min" data-type="number" type="number" min="0" max="1" step="0.01" value="${esc(config.mtp_p_min ?? 0.5)}" title="Minimum probability threshold used by the native MTP draft verifier. Higher values are more selective.">
          </div><div class="llm-note" id="llm-spec-note">Speculative decoding is capability-gated. Unsupported providers are disabled rather than silently emulated.</div></div>
          <div class="llm-card"><h3>Prompt Prefix Cache</h3><div class="llm-grid">
            <label title="Auto lets llama.cpp reuse an exact token prefix already present in the current resident KV context. Off resets the context before every request.">Mode</label><select class="llm-config" data-key="prompt_cache_mode" id="llm-prompt-cache-mode" title="Auto lets llama.cpp reuse an exact token prefix already present in the current resident KV context. Off resets the context before every request.">${selectOptions(["Auto","Off"],config.prompt_cache_mode || "Auto")}</select>
            <label>Scope</label><div class="llm-muted">Current resident context</div>
            <label>Last request</label><div id="llm-prompt-cache-last" class="llm-muted">No request measured yet</div>
            <label>Prefill saved</label><div id="llm-prompt-cache-saved" class="llm-muted">—</div>
          </div><div class="llm-note" id="llm-prompt-cache-note">Auto reuses only an exact resident token prefix. It adds no separate RAM cache; Suspend, Stop / Unload, model reload, and vision requests clear or bypass the reusable state.</div></div>
          <div class="llm-card"><h3>Memory / VRAM</h3><div class="llm-grid">
            <label title="Auto Yield releases the native LLM context when ComfyUI needs VRAM and restores it on the next request. Keep Resident never yields automatically.">VRAM policy</label><select class="llm-config" data-key="vram_policy" title="Auto Yield releases the native LLM context when ComfyUI needs VRAM and restores it on the next request. Keep Resident never yields automatically.">${selectOptions(["Auto Yield to ComfyUI","Keep Resident"],config.vram_policy || "Auto Yield to ComfyUI")}</select>
            <label>Memory preset</label><select class="llm-config" data-key="memory_preset" id="llm-memory-preset">${selectOptions(catalog.memory_presets,config.memory_preset)}</select>
            <label title="Maximum token context allocated by llama.cpp. Larger contexts increase KV-cache memory use.">Context size</label><select class="llm-config memory-owned" data-key="context_size" data-type="number" id="llm-context-size" title="Maximum token context allocated by llama.cpp. Larger contexts increase KV-cache memory use.">${selectOptions(catalog.context_sizes || [],config.context_size)}</select>
            <label title="Key-cache precision. Lower-bit formats reduce KV memory, sometimes with a small quality or performance tradeoff.">KV cache K</label><select class="llm-config memory-owned" data-key="kv_cache_k" title="Key-cache precision. Lower-bit formats reduce KV memory, sometimes with a small quality or performance tradeoff.">${selectOptions(["Auto","f32","f16","bf16","q8_0","q5_1","q5_0","q4_1","q4_0","iq4_nl"],config.kv_cache_k)}</select>
            <label title="Value-cache precision. Lower-bit formats reduce KV memory, sometimes with a small quality or performance tradeoff.">KV cache V</label><select class="llm-config memory-owned" data-key="kv_cache_v" title="Value-cache precision. Lower-bit formats reduce KV memory, sometimes with a small quality or performance tradeoff.">${selectOptions(["Auto","f32","f16","bf16","q8_0","q5_1","q5_0","q4_1","q4_0","iq4_nl"],config.kv_cache_v)}</select>
            <label title="GPU is normally fastest. CPU saves VRAM but can reduce generation performance.">KV location</label><select class="llm-config memory-owned" data-key="kv_cache_location" title="GPU is normally fastest. CPU saves VRAM but can reduce generation performance.">${selectOptions(["GPU","CPU"],config.kv_cache_location)}</select>
            ${field("gpu_layers","GPU layers (-1 = all)","number","","Number of model layers offloaded to GPU. -1 asks llama.cpp to offload all supported layers.")}
            <label title="Primary CUDA device used by llama.cpp. Relevant mainly with multiple GPUs or split modes.">Main GPU</label><select class="llm-config memory-owned" data-key="main_gpu" title="Primary CUDA device used by llama.cpp. Relevant mainly with multiple GPUs or split modes.">${selectOptions(catalog.gpus,config.main_gpu)}</select>
            ${field("flash_attention","Flash Attention","checkbox","","Uses llama.cpp Flash Attention when supported. Usually faster and can reduce attention memory use.")}
            ${field("prompt_batch_size","Prompt batch (n_batch)","number","","Logical maximum prompt-evaluation batch size. Primarily affects prompt processing, not decode tok/s.")}
            ${field("memory_batch_size","Micro batch (n_ubatch)","number","","Physical prompt-evaluation micro-batch size. Larger values may improve prompt throughput but use more temporary VRAM.")}
            ${field("use_mmap","Use mmap","checkbox","","Memory-map the GGUF file. Usually improves load and hot-reload behavior by allowing the OS to reuse cached file pages in RAM.")}
            ${field("use_mlock","Use mlock","checkbox","","Locks mapped model pages in system RAM so the OS cannot reclaim them. Can increase RAM pressure; normally leave off.")}
            <label title="Controls how model tensors are distributed across multiple GPUs. Leave at single GPU unless intentionally using more than one GPU.">Split mode</label><select class="llm-config memory-owned" data-key="split_mode" title="Controls how model tensors are distributed across multiple GPUs. Leave at single GPU unless intentionally using more than one GPU.">${selectOptions(["None (single GPU)","Layer","Row","Tensor"],config.split_mode)}</select>
            ${field("tensor_split","Tensor split","text","","Optional per-GPU split ratios for multi-GPU use. Leave blank for normal single-GPU operation.")}
          </div><div class="llm-note"><b>Auto Yield to ComfyUI</b> keeps the native llama.cpp context resident while there is room, fully closes it when ComfyUI needs VRAM, and restores it on the next LLM request. <b>Keep Resident</b> never voluntarily yields the native allocation.</div></div>
        </div>
        <div class="llm-pane" data-pane="tuner">
          <div class="llm-card"><h3>Performance Tuner</h3>
            <div class="llm-tuner-options">
              <label title="Quick uses one measured full-cycle trial per candidate. Standard uses two trials, a wider memory search, and tunes the draft-token count of the winning speculative provider.">Profile</label><select id="llm-tuner-profile"><option>Quick</option><option>Standard</option></select>
              <label title="ComfyUI Cycle scores warm reload + fixed prompt/generation work + Suspend-style unload. Inference Only scores prompt/generation work without load/unload time.">Optimize for</label><select id="llm-tuner-score"><option>ComfyUI Cycle</option><option>Inference Only</option></select>
              <label title="Candidates projected to leave less than this much free VRAM are skipped before loading.">Minimum VRAM headroom (MiB)</label><input id="llm-tuner-headroom" type="number" min="256" step="256" value="1024" title="Candidates projected to leave less than this much free VRAM are skipped before loading.">
              <label>Batch sizes</label><input id="llm-tuner-batches" type="checkbox" checked>
              <label>Flash Attention</label><input id="llm-tuner-flash" type="checkbox" checked>
              <label>Speculative decoding</label><input id="llm-tuner-spec" type="checkbox" checked>
              <label title="Tests mmap, mlock, GPU/CPU KV placement, and useful GPU-layer offload points. Context size is never reduced by the tuner.">Memory / offload options</label><input id="llm-tuner-memory" type="checkbox" checked title="Tests mmap, mlock, GPU/CPU KV placement, and useful GPU-layer offload points. Context size is never reduced by the tuner.">
              <label title="Also tests f16, q8_0, and q4_0 KV-cache precision combinations. Lower-bit KV can slightly affect output quality, so this search is opt-in.">KV precision variants</label><input id="llm-tuner-kv-precision" type="checkbox" title="Also tests f16, q8_0, and q4_0 KV-cache precision combinations. Lower-bit KV can slightly affect output quality, so this search is opt-in.">
            </div>
            <div class="llm-actions" style="margin-top:12px"><button class="llm-btn primary" data-action="tuner-start">Run Tuner</button><button class="llm-btn danger" data-action="tuner-cancel" disabled>Cancel</button></div>
            <div class="llm-note"><b>ComfyUI Cycle</b> is the default because short workflow requests can spend as much time reloading and yielding as generating. Every measured trial starts yielded, performs a warm reload, runs the same text workload with prompt-prefix caching disabled, then performs a lightweight Suspend/Auto-Yield close. Context size and sampler values stay fixed. KV precision testing is separate because quantized KV can slightly affect output.</div>
          </div>
          <div class="llm-card"><h3>Progress</h3>
            <div class="llm-tuner-progress"><span id="llm-tuner-progress-bar"></span></div>
            <div id="llm-tuner-phase" class="llm-muted">Idle</div>
            <div id="llm-tuner-message" class="llm-muted" style="margin-top:5px">Ready to benchmark the saved server configuration.</div>
          </div>
          <div class="llm-card" id="llm-tuner-recommendation"><h3>Recommendation</h3>
            <div class="llm-tuner-summary">
              <div><b id="llm-tuner-gain">—</b><span id="llm-tuner-gain-label">ComfyUI-cycle improvement</span></div>
              <div><b id="llm-tuner-load">—</b><span>Warm reload</span></div>
              <div><b id="llm-tuner-prompt">—</b><span>Prompt processing</span></div>
              <div><b id="llm-tuner-decode">—</b><span>Generation</span></div>
              <div><b id="llm-tuner-unload">—</b><span>Suspend / unload</span></div>
              <div><b id="llm-tuner-cycle">—</b><span>Full fixed-work cycle</span></div>
              <div><b id="llm-tuner-vram">—</b><span>Measured resident VRAM</span></div>
            </div>
            <div id="llm-tuner-settings" class="llm-muted">Run the tuner to produce a recommendation.</div>
            <div class="llm-actions" style="margin-top:11px"><button class="llm-btn" data-action="tuner-use" disabled>Use Recommendation</button><button class="llm-btn" data-action="tuner-save-preset" disabled>Save as Memory Preset</button></div>
          </div>
          <div class="llm-card"><h3>Results</h3><div class="llm-tuner-table-wrap"><table class="llm-tuner-table">
            <thead><tr><th>Candidate</th><th>Load</th><th>Prompt t/s</th><th>Gen t/s</th><th>Unload</th><th>Inference</th><th>Cycle</th><th>VRAM</th><th>Spec</th><th>Result</th></tr></thead><tbody id="llm-tuner-results"><tr><td colspan="10" class="llm-muted">No benchmark results yet.</td></tr></tbody>
          </table></div></div>
        </div>
        <div class="llm-pane" data-pane="api">
          <div class="llm-card"><h3>OpenAI-compatible API</h3><div class="llm-grid">
            ${field("external_api_enabled","Enable external API","checkbox")}
            ${field("allow_buffered_streaming","Allow streaming requests","checkbox")}
            <label>API base</label><div class="llm-api-row"><input readonly id="llm-api-base" value="${esc(apiBase)}"><button class="llm-btn" data-action="copy-api">Copy</button></div>
            <label>API key</label><div class="llm-api-row"><input class="llm-config" data-key="api_key" id="llm-api-key" type="password" value="${esc(config.api_key || "")}"><button class="llm-btn" data-action="toggle-key">Show</button><button class="llm-btn" data-action="regen-key">Regenerate</button></div>
          </div><div class="llm-note">SillyTavern: choose a Custom/OpenAI-compatible Chat Completion endpoint and use the base URL above. SSE responses stream generated chunks live from llama.cpp.</div></div>
        </div>
        <div class="llm-pane" data-pane="logs">
          <div class="llm-card"><h3>Privacy</h3><div class="llm-grid">
            ${field("log_prompt_content","Log prompt content","checkbox")}
            ${field("log_response_content","Log response content","checkbox")}
          </div><div class="llm-note">Prompt and response text are <b>not logged by default</b>. Enable these only when you explicitly want content recorded in the Local LLM log/ComfyUI console.</div></div>
          <div class="llm-actions" style="margin-bottom:8px"><button class="llm-btn" data-action="refresh-logs">Refresh</button></div><div class="llm-log" id="llm-log-view">Loading…</div>
        </div>
      </div>
      <div class="llm-footer"><span id="llm-save-note" class="llm-muted" style="margin-right:auto"></span><button class="llm-btn" data-action="save">Save</button><button class="llm-btn warn" data-action="reload">Reload Model</button></div>
    </div>`;
  document.body.appendChild(modal);
  bindModal();
  dirty = false;
  saving = false;
  updateModelPresetChoices();
  updateLiveUI();
  refreshLogs();
  refreshVRAMEstimate(true);
  renderTuner();
}

function formatVRAMBytes(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const b = Number(value);
  const gib = b / (1024 ** 3);
  if (Math.abs(gib) >= 0.1) return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
  return `${(b / (1024 ** 2)).toFixed(0)} MiB`;
}

function renderVRAMEstimate(data=vramEstimate) {
  if (!modal?.isConnected) return;
  const set=(id,val)=>{const el=modal.querySelector(`#${id}`);if(el)el.textContent=val;};
  if (!data?.available) {
    set("llm-vram-est-total", "—"); set("llm-vram-measured", "—"); set("llm-vram-free", "—"); set("llm-vram-headroom", "—");
    for (const id of ["llm-vram-weights","llm-vram-kv","llm-vram-compute","llm-vram-spec","llm-vram-vision","llm-vram-base","llm-vram-target"]) set(id,"—");
    const cap=modal.querySelector("#llm-vram-caption"); if(cap)cap.textContent=data?.reason || "Select a GGUF model to estimate VRAM.";
    return;
  }
  const c=data.components || {};
  const displayTotal = data.vision_optional ? c.total_bytes : c.base_total_bytes;
  set("llm-vram-est-total", formatVRAMBytes(displayTotal));
  set("llm-vram-measured", data.measured_bytes > 0 ? formatVRAMBytes(data.measured_bytes) + (data.measured_applies ? "" : "*") : "—");
  set("llm-vram-free", formatVRAMBytes(data.raw_free_bytes));
  set("llm-vram-headroom", formatVRAMBytes(data.projected_headroom_bytes));
  set("llm-vram-weights", formatVRAMBytes(c.weights_bytes));
  set("llm-vram-kv", formatVRAMBytes(c.kv_cache_bytes));
  set("llm-vram-compute", formatVRAMBytes(c.compute_batch_bytes));
  set("llm-vram-spec", Number(c.speculative_bytes || 0) > 0 ? `+ ${formatVRAMBytes(c.speculative_bytes)}` : (data?.speculative?.effective === "N-gram" ? "~0 MiB (N-gram)" : "—"));
  set("llm-vram-vision", c.vision_bytes > 0 ? `+ ${formatVRAMBytes(c.vision_bytes)}` : "—");
  set("llm-vram-base", formatVRAMBytes(c.base_total_bytes));
  set("llm-vram-target", `${formatVRAMBytes(data.reload_target_bytes)} (${data.reload_target_source || "estimate"})`);

  const meter=modal.querySelector("#llm-vram-meter");
  if(meter){
    const total=Number(data.total_vram_bytes || 0);
    const need=Number(displayTotal || 0);
    const pct=total>0 ? Math.max(0,Math.min(100,(need/total)*100)) : 0;
    const fill=meter.querySelector("span"); if(fill)fill.style.width=`${pct}%`;
    meter.classList.toggle("danger", Number(data.projected_headroom_bytes) < 0);
    meter.classList.toggle("warn", Number(data.projected_headroom_bytes) >= 0 && Number(data.projected_headroom_bytes) < 1024**3);
  }
  const warn=modal.querySelector("#llm-vram-warning");
  if(warn){warn.textContent=data.warning || "";warn.style.display=data.warning?"block":"none";}
  const cap=modal.querySelector("#llm-vram-caption");
  if(cap){
    const parts=[];
    if(data.resident) parts.push("LLM is resident; current free VRAM already reflects its allocation.");
    if(data.measured_bytes>0) parts.push(data.measured_applies ? `Measured value is valid for the saved configuration (${data.measured_source || "prior load"}).` : "* Measured value is from the currently saved/previous configuration, not these unsaved settings.");
    else parts.push("Estimate remains conservative until this configuration completes a verified GPU load.");
    if(data.vision_optional) parts.push(data.measured_includes_vision ? "The measured allocation includes an active vision projector." : "Vision/mmproj is added only when a vision request activates it; the measured value shown may reflect the base text load only.");
    if(data?.speculative?.effective && data.speculative.effective!=="Off") parts.push(`Speculative mode: ${data.speculative.effective}${Number(c.speculative_bytes||0)>0 ? ` (+${formatVRAMBytes(c.speculative_bytes)} runtime allowance)` : ""}.`);
    if(data.split_mode && data.split_mode !== "None (single GPU)") parts.push("Multi-GPU totals are aggregate estimates; per-GPU headroom may differ.");
    cap.textContent=parts.join(" ");
  }
}

async function refreshVRAMEstimate(force=false) {
  if (!modal?.isConnected) return;
  const pane=modal.querySelector('.llm-pane[data-pane="memory"]');
  if(!force && !pane?.classList.contains("active")) return;
  const now=performance.now();
  if(!force && now-lastVramEstimateAt < 800) return;
  lastVramEstimateAt=now;
  const seq=++vramEstimateSeq;
  try{
    const payload=collectConfig();
    const data=await fetchJSON("/local_llm_server/vram_estimate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(seq!==vramEstimateSeq || !modal?.isConnected) return;
    vramEstimate=data;
    renderVRAMEstimate(data);
  }catch(e){
    if(seq!==vramEstimateSeq) return;
    const cap=modal?.querySelector("#llm-vram-caption"); if(cap)cap.textContent=`VRAM estimate unavailable: ${e.message}`;
  }
}

function scheduleVRAMEstimate(delay=120) {
  if(vramEstimateTimer) clearTimeout(vramEstimateTimer);
  vramEstimateTimer=setTimeout(()=>{vramEstimateTimer=null;refreshVRAMEstimate(true);},delay);
}

function runtimeNote() {
  if (status.state === "tuning") return `Performance tuner is running. Open the Tuner tab for candidate progress and results.`;
  if (status.state === "processing") {
    return `Processing prompt for <b>${esc(status.current_client || "client")}</b>${status.current_phase_seconds != null ? ` • ${Number(status.current_phase_seconds).toFixed(1)}s` : ""}`;
  }
  if (status.state === "generating") {
    const speed = status.current_tokens_per_second != null ? ` • ${Number(status.current_tokens_per_second).toFixed(1)} tok/s` : "";
    const toks = status.current_completion_tokens ? ` • ${status.current_completion_tokens} tokens` : "";
    return `Generating for <b>${esc(status.current_client || "client")}</b>${speed}${toks}${status.current_seconds != null ? ` • ${Number(status.current_seconds).toFixed(1)}s` : ""}`;
  }
  if (status.restart_required) return `<span class="llm-warn">Configuration changed. Reload the model to apply load/memory changes.</span>`;
  if (status.state === "ready" && status.vram_yielded) return `<span class="llm-muted">LLM VRAM yielded to ComfyUI; it will reload automatically on the next request.</span>`;
  if (status.state === "ready") return `<span class="llm-success">Model loaded and ready.</span>`;
  if (status.state === "error") return `<span class="llm-error">${esc(status.error || "Service error")}</span>`;
  if (status.state === "waiting_comfy") return `Waiting for the active ComfyUI workflow to finish before using the LLM GPU context.`;
  if (status.state === "loading" || status.state === "reloading") return `Loading model…${status.current_phase_seconds != null ? ` • ${Number(status.current_phase_seconds).toFixed(1)}s` : ""}`;
  return `Service stopped. Start it manually, use On Demand, or enable Auto Start.`;
}

function updateLiveUI() {
  decorateSidebar();
  ensureFloatingStatus();
  updateFloatingStatus();
  syncAllServiceGenerateCapabilities(status.capabilities);
  if (!modal?.isConnected) return;
  const set = (id, v) => { const el=modal.querySelector(`#${id}`); if (el) el.textContent = v ?? "—"; };
  const dot = modal.querySelector("#llm-live-dot"); if (dot) dot.className = `llm-status-dot ${stateClass(displayState())}`;
  set("llm-head-state", (status.vram_yielded && status.state === "ready" ? "yielded" : status.state) + ((status.queue_count||0) ? ` • queue ${status.queue_count}` : ""));
  set("llm-stat-model", status.model || "—"); set("llm-stat-state", status.state || "—");
  set("llm-stat-speed", status.state === "generating" ? Number(status.current_tokens_per_second || 0).toFixed(2) : "0.00");
  set("llm-stat-generated", status.current_completion_tokens || status.last_completion_tokens || 0);
  set("llm-stat-queue", status.queue_count || 0); set("llm-stat-requests", status.requests_total || 0); set("llm-current-client", status.current_client || "Idle");
  const note=modal.querySelector("#llm-runtime-note"); if(note) note.innerHTML=runtimeNote();
  const err=modal.querySelector("#llm-global-error"); if(err) err.textContent=status.error||"";
  const family=modal.querySelector("#llm-detected-family"); if(family) family.textContent=status.family||"unknown";
  applyModalCapabilityUI(currentModelInfo || {capabilities: status.capabilities, matching_vision: status.matching_vision});
  renderPromptCache();
  updateActionButtons();
  refreshVRAMEstimate(false);
  renderTuner();
}

function capabilitySummary(caps, matchingVision=null) {
  const c = caps || {};
  const yesNoMaybe = (v) => v === true ? "yes" : (v === false ? "no" : "unknown");
  const parts = [
    `text ${yesNoMaybe(c.text)}`,
    `vision ${yesNoMaybe(c.vision)}`,
    `audio ${yesNoMaybe(c.audio)}`,
    `embeddings ${yesNoMaybe(c.embeddings)}`,
    `MTP ${yesNoMaybe(c.mtp)}`,
  ];
  const handlers = Array.isArray(c.preferred_chat_handlers) ? c.preferred_chat_handlers.filter(Boolean) : [];
  if (handlers.length) parts.push(`handler ${handlers.join(" / ")}`);
  if (matchingVision) parts.push(`auto mmproj ${matchingVision}`);
  return parts.join(" • ");
}

function renderPromptCache() {
  if (!modal?.isConnected) return;
  const cache = status?.last_prompt_cache || {};
  const last = modal.querySelector("#llm-prompt-cache-last");
  const saved = modal.querySelector("#llm-prompt-cache-saved");
  const note = modal.querySelector("#llm-prompt-cache-note");
  const modeEl = modal.querySelector("#llm-prompt-cache-mode");
  const requested = modeEl?.value || config?.prompt_cache_mode || "Auto";

  if (last) {
    const total = Number(cache.prompt_tokens || 0);
    const reused = Number(cache.reused_tokens || 0);
    const pct = Number(cache.reuse_percent || 0);
    if (total > 0) {
      last.textContent = cache.hit
        ? `${reused.toLocaleString()} / ${total.toLocaleString()} tokens reused • ${pct.toFixed(1)}%`
        : `${total.toLocaleString()} prompt tokens • no resident-prefix hit`;
    } else {
      last.textContent = "No request measured yet";
    }
  }
  if (saved) {
    const evaluated = Number(cache.evaluated_tokens || 0);
    const seconds = Number(cache.estimated_seconds_saved || 0);
    if (Number(cache.prompt_tokens || 0) > 0) {
      const rawRate = Number(cache.uncached_prompt_tokens_per_second || 0);
      const effectiveRate = Number(cache.effective_prompt_tokens_per_second || 0);
      const rates = rawRate > 0 ? ` • ${rawRate.toFixed(0)} raw t/s${effectiveRate > rawRate ? ` • ${effectiveRate.toFixed(0)} effective t/s` : ""}` : "";
      saved.textContent = `${evaluated.toLocaleString()} tokens evaluated${rates}${seconds > 0 ? ` • ~${seconds.toFixed(2)}s avoided` : ""}`;
    } else saved.textContent = "—";
  }
  if (note) {
    if (requested === "Off") {
      note.textContent = "Disabled: the llama.cpp context is reset before each request, so no previous prompt KV prefix is reused.";
    } else if (cache.effective === "Off" && cache.reason) {
      note.textContent = cache.reason;
    } else if (cache.reason && Number(cache.prompt_tokens || 0) > 0) {
      note.textContent = `${cache.reason} Cache is resident-only and does not survive Suspend, Stop / Unload, or model reload.`;
    } else {
      note.textContent = "Auto reuses only an exact resident token prefix. It adds no separate RAM cache; Suspend, Stop / Unload, model reload, and vision requests clear or bypass the reusable state.";
    }
  }
}

function applySpeculativeUI(info) {
  if (!modal?.isConnected) return;
  const caps = info?.capabilities || status?.capabilities || {};
  const support = info?.speculative_support || status?.speculative_support || {};
  const sel = modal.querySelector("#llm-spec-mode");
  const requested = sel?.value || config?.speculative_mode || "Off";
  const mtpModel = caps.mtp === true || Number(caps.mtp_layers || info?.speculative?.mtp_layers || status?.speculative?.mtp_layers || 0) > 0;
  const ngramOK = !!support.ngram;
  const gpuLayersEl=modal.querySelector('[data-key="gpu_layers"]');
  const gpuLayers=Number(gpuLayersEl?.value ?? config?.gpu_layers ?? -1);
  const mtpFullGPU = Number.isFinite(gpuLayers) && gpuLayers < 0;
  const mtpOK = !!support.mtp && mtpModel && mtpFullGPU;
  if (sel) {
    const nOpt=[...sel.options].find(o=>o.value==="N-gram"); if(nOpt)nOpt.disabled=!ngramOK;
    const mOpt=[...sel.options].find(o=>o.value==="MTP"); if(mOpt)mOpt.disabled=!mtpOK;
  }
  let effective=requested;
  if(requested==="Auto") effective=mtpOK ? "MTP" : (ngramOK ? "N-gram" : "Off");
  if(requested==="N-gram" && !ngramOK) effective="Off";
  if(requested==="MTP" && !mtpOK) effective="Off";
  for(const el of modal.querySelectorAll(".llm-spec-ngram-setting")) el.style.display=effective==="N-gram" ? "" : "none";
  for(const el of modal.querySelectorAll(".llm-spec-mtp-setting")) el.style.display=effective==="MTP" ? "" : "none";
  const supportEl=modal.querySelector("#llm-spec-support");
  if(supportEl){
    const pieces=[`N-gram ${ngramOK ? (support.ngram_implementation || "yes") : "unavailable"}`, `MTP bridge ${support.mtp ? "yes" : "unavailable"}`, `model MTP ${mtpModel ? `${Number(caps.mtp_layers || info?.speculative?.mtp_layers || 0)} layer(s)` : "no"}`, `MTP full GPU ${mtpFullGPU ? "yes" : "required"}`];
    supportEl.textContent=pieces.join(" • ");
  }
  const note=modal.querySelector("#llm-spec-note");
  if(note){
    let text=`Effective mode: ${effective}. `;
    if(effective==="N-gram") text+="N-gram uses prior context to draft candidates and adds essentially no model-weight VRAM; the target model still verifies every token.";
    else if(effective==="MTP") text+="Native MTP uses embedded NextN layers and requires the experimental native bridge exposed by the installed llama-cpp-python build.";
    else if(requested==="MTP" && !mtpModel) text+="The selected GGUF does not advertise embedded NextN/MTP layers.";
    else if(requested==="MTP" && !support.mtp) text+="The installed binding does not expose the native MTP bridge/load_mtp combination.";
    else if(requested==="MTP" && !mtpFullGPU) text+="Native MTP currently requires GPU layers = -1 (full GPU offload).";
    else text+="No speculative provider will be attached to the llama context.";
    note.textContent=text;
  }
}

function applyModalCapabilityUI(info) {
  if (!modal?.isConnected) return;
  const caps = info?.capabilities || status?.capabilities || {};
  const visionUnsupported = caps.vision === false;
  const visionSelect = modal.querySelector("#llm-vision-model");
  const visionLabel = modal.querySelector("#llm-vision-model-label");
  if (visionSelect) {
    visionSelect.disabled = visionUnsupported;
    visionSelect.title = visionUnsupported ? "Detected text-only model: vision/mmproj is unavailable." : (info?.matching_vision ? `Auto match: ${info.matching_vision}` : "Vision projector selection");
  }
  if (visionLabel) visionLabel.style.opacity = visionUnsupported ? "0.5" : "";
  for (const el of modal.querySelectorAll(".llm-vision-setting")) {
    el.style.display = visionUnsupported ? "none" : "";
  }
  const out = modal.querySelector("#llm-model-capabilities");
  if (out) out.textContent = capabilitySummary(caps, info?.matching_vision || status?.matching_vision || null);
  applySpeculativeUI(info);
}

async function updateModelPresetChoices() {
  const sel = modal?.querySelector("#llm-model-preset");
  const model = modal?.querySelector("#llm-model-select")?.value || config?.model;
  if (!sel || !model) return;
  try {
    const info = await fetchJSON(`/local_llm_server/model_info?model=${encodeURIComponent(model)}`);
    currentModelInfo = info;
    const choices = ["Auto (Detected)","Custom",...(info.available_presets||[])];
    sel.innerHTML = selectOptions([...new Set(choices)], config.model_preset);
    if (![...sel.options].some(o => o.value === String(config.model_preset))) sel.value = "Auto (Detected)";
    const fam=modal.querySelector("#llm-detected-family"); if(fam) fam.textContent=info.family||"unknown";
    applyModalCapabilityUI(info);
    updateContextChoices(info.context_sizes || catalog.context_sizes || [], info.native_context);
  } catch (_) {
    currentModelInfo = null;
    sel.innerHTML = selectOptions(catalog.model_presets, config.model_preset);
    applyModalCapabilityUI({capabilities: status?.capabilities, matching_vision: status?.matching_vision});
    updateContextChoices(catalog.context_sizes || [], null);
  }
}

function normalizeContextChoice(values, requested) {
  const nums = [...new Set((values || []).map(Number).filter(Number.isFinite))].sort((a,b)=>a-b);
  if (!nums.length) return Number(requested) || 32768;
  const n = Number(requested);
  if (nums.includes(n)) return n;
  const lower = nums.filter(x => x <= n);
  return lower.length ? lower[lower.length - 1] : nums[0];
}

function updateContextChoices(values, nativeContext=null) {
  const sel = modal?.querySelector("#llm-context-size");
  if (!sel) return;
  const before = Number(sel.value || config?.context_size || 32768);
  const chosen = normalizeContextChoice(values, before);
  sel.innerHTML = selectOptions(values, chosen);
  sel.value = String(chosen);
  const label = sel.previousElementSibling;
  if (label && label.tagName === "LABEL") {
    label.textContent = nativeContext ? `Context size (native max ${Number(nativeContext).toLocaleString()})` : "Context size";
  }
}

function setConfigField(key, value) {
  const el = modal?.querySelector(`.llm-config[data-key="${CSS.escape(key)}"]`);
  if (!el) return;
  if (el.type === "checkbox") el.checked = !!value;
  else if (key === "context_size" && el.tagName === "SELECT") {
    const vals = [...el.options].map(o => Number(o.value));
    el.value = String(normalizeContextChoice(vals, value));
  } else el.value = value;
}

async function applyPreset(layer, name) {
  if (!catalog?.presets || name === "Custom") return;
  let resolved = name;
  if (layer === "model" && name === "Auto (Detected)") {
    try {
      const model = modal?.querySelector("#llm-model-select")?.value || config?.model;
      const info = await fetchJSON(`/local_llm_server/model_info?model=${encodeURIComponent(model || "")}`);
      resolved = info.recommended_preset || "Generic Chat";
    } catch (_) {
      resolved = "Generic Chat";
    }
  }
  const values = catalog.presets[layer]?.[resolved];
  if (!values) return;
  for (const [k,v] of Object.entries(values)) setConfigField(k,v);
}

function collectConfig() {
  const out = {};
  for (const el of modal.querySelectorAll(".llm-config[data-key]")) {
    const k=el.dataset.key;
    if (el.type === "checkbox") out[k]=el.checked;
    else if (el.dataset.type === "number") out[k]=Number(el.value);
    else out[k]=el.value;
  }
  return out;
}

function setDirty(value=true) {
  dirty = !!value;
  const note = modal?.querySelector("#llm-save-note");
  if (note && !saving) note.textContent = dirty ? "Unsaved changes" : "";
  updateActionButtons();
}

function updateActionButtons() {
  if (!modal?.isConnected) return;
  const busy = ["loading","reloading","processing","generating","stopping","waiting_comfy","tuning"].includes(status.state);
  const resident = !!status.model_loaded;
  const serviceActive = !!status.active && status.state !== "error";
  const yielded = !!status.vram_yielded;
  const start = modal.querySelector('[data-action="start"]');
  const suspend = modal.querySelector('[data-action="suspend"]');
  const stop = modal.querySelector('[data-action="stop"]');
  const save = modal.querySelector('[data-action="save"]');
  const reload = modal.querySelector('[data-action="reload"]');
  if (start) {
    start.disabled = busy || serviceActive || dirty || saving;
    start.title = dirty ? "Save changes before starting" : (serviceActive ? (yielded ? "Service is suspended and will fast-reload on the next request" : "Model is already running") : "Start the saved server configuration");
  }
  if (suspend) {
    suspend.disabled = busy || !serviceActive || !resident || yielded;
    suspend.title = yielded ? "Model is already suspended / yielded" : (resident ? "Release LLM VRAM while preserving fast-reload state" : "No resident model to suspend");
  }
  if (stop) {
    stop.disabled = busy || !serviceActive;
    stop.title = serviceActive ? "Stop the service and fully clear the model/load state" : "Service is not running";
  }
  if (save) {
    save.disabled = saving || !dirty || busy;
    save.title = busy ? "Wait for the current LLM operation to finish" : (dirty ? "Save configuration" : "No unsaved changes");
  }
  if (reload) {
    reload.disabled = saving || dirty || busy || !serviceActive;
    reload.title = !serviceActive ? "Reload is available only while the service is started" : (dirty ? "Save changes before reloading" : (busy ? "Wait for the current operation to finish" : "Perform a deliberate full reload using saved settings"));
  }
}

async function saveConfig() {
  const note=modal.querySelector("#llm-save-note");
  saving = true;
  if(note) note.textContent="Saving…";
  updateActionButtons();
  try {
    const data = await fetchJSON("/local_llm_server/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body:JSON.stringify(collectConfig())});
    config=data.config; status=data.status;
    ensureFloatingStatus();
    setConfigField("api_key", config.api_key || "");
    setConfigField("context_size", config.context_size);
    dirty = false;
    if(note) note.textContent="Saved";
    updateLiveUI();
    refreshAllDefaultServiceGenerateNodes();
    setTimeout(()=>{ if(modal?.isConnected && !dirty && !saving){ const n=modal.querySelector("#llm-save-note"); if(n?.textContent==="Saved") n.textContent=""; } },1800);
    return true;
  } catch(e) {
    if(note) note.textContent=`Error: ${e.message}`;
    return false;
  } finally {
    saving = false;
    updateActionButtons();
  }
}

async function doAction(action) {
  const note=modal?.querySelector("#llm-save-note");
  if (dirty && (action === "start" || action === "reload")) {
    if (note) note.textContent = "Save changes first";
    return;
  }
  try {
    if (action === "start") status=await fetchJSON("/local_llm_server/start",{method:"POST"});
    else if(action === "suspend") status=await fetchJSON("/local_llm_server/suspend",{method:"POST"});
    else if(action === "stop") status=await fetchJSON("/local_llm_server/stop",{method:"POST"});
    else if(action === "reload") status=await fetchJSON("/local_llm_server/reload",{method:"POST"});
    updateLiveUI();
  } catch(e) { status={...status,state:"error",error:e.message}; updateLiveUI(); }
}

async function refreshLogs() {
  if (!modal?.isConnected) return;
  try {
    const data=await fetchJSON("/local_llm_server/logs");
    const view=modal.querySelector("#llm-log-view");
    if(view) view.textContent=(data.logs||[]).map(x=>`${new Date((x.time||0)*1000).toLocaleTimeString()} [${(x.level||"info").toUpperCase()}] ${x.message}`).join("\n") || "No log entries.";
  } catch(e) { const view=modal.querySelector("#llm-log-view"); if(view)view.textContent=e.message; }
}

function bindModal() {
  modal.querySelector(".llm-close").onclick=closeModal;
  modal.onclick=(e)=>{ if(e.target===modal) closeModal(); };
  for(const tab of modal.querySelectorAll(".llm-tab")) tab.onclick=()=>{
    modal.querySelectorAll(".llm-tab").forEach(x=>x.classList.toggle("active",x===tab));
    modal.querySelectorAll(".llm-pane").forEach(x=>x.classList.toggle("active",x.dataset.pane===tab.dataset.tab));
    if(tab.dataset.tab==="logs") refreshLogs();
    if(tab.dataset.tab==="memory") refreshVRAMEstimate(true);
    if(tab.dataset.tab==="tuner") refreshTunerStatus(true);
  };
  modal.querySelector("#llm-model-select").addEventListener("change", async (e)=>{
    config.model=e.target.value;
    await updateModelPresetChoices();
    const preset = modal.querySelector("#llm-model-preset")?.value || "Auto (Detected)";
    if (preset !== "Custom") await applyPreset("model", preset);
    const memPreset = modal.querySelector("#llm-memory-preset")?.value || "Custom";
    if (memPreset !== "Custom") {
      await applyPreset("memory", memPreset);
      updateContextChoices(currentModelInfo?.context_sizes || catalog.context_sizes || [], currentModelInfo?.native_context);
    }
    setDirty(true);
    scheduleVRAMEstimate();
  });
  modal.querySelector("#llm-model-preset").addEventListener("change",async e=>{await applyPreset("model",e.target.value);setDirty(true);scheduleVRAMEstimate();});
  modal.querySelector("#llm-memory-preset").addEventListener("change",async e=>{
    await applyPreset("memory",e.target.value);
    updateContextChoices(currentModelInfo?.context_sizes || catalog.context_sizes || [], currentModelInfo?.native_context);
    applySpeculativeUI(currentModelInfo || {capabilities:status?.capabilities,speculative_support:status?.speculative_support,speculative:status?.speculative});
    setDirty(true);
    scheduleVRAMEstimate();
  });
  const modelOwned=new Set(["thinking_mode","reasoning_effort","preserve_thinking","temperature","top_p","top_k","min_p","repeat_penalty","presence_penalty","frequency_penalty"]);
  const memoryOwned=new Set(["context_size","kv_cache_k","kv_cache_v","kv_cache_location","gpu_layers","flash_attention","prompt_batch_size","memory_batch_size","use_mmap","use_mlock","main_gpu","split_mode","tensor_split"]);
  const specOwned=new Set(["speculative_mode","ngram_pred_tokens","ngram_size","ngram_mode","ngram_min_hits","ngram_max_entries_per_key","ngram_sync_check_tokens","mtp_draft_tokens","mtp_p_min"]);
  for(const el of modal.querySelectorAll(".llm-config[data-key]")) el.addEventListener("change",()=>{
    const k=el.dataset.key;
    if(modelOwned.has(k)){const s=modal.querySelector("#llm-model-preset");if(s&&s.value!=="Custom")s.value="Custom";}
    if(memoryOwned.has(k)){const s=modal.querySelector("#llm-memory-preset");if(s&&s.value!=="Custom")s.value="Custom";}
    if(k==="show_status_indicator"){config.show_status_indicator=!!el.checked;ensureFloatingStatus(el.checked);}
    if(specOwned.has(k) || k==="gpu_layers" || k==="context_size") applySpeculativeUI(currentModelInfo || {capabilities:status?.capabilities,speculative_support:status?.speculative_support,speculative:status?.speculative});
    if(k==="prompt_cache_mode") renderPromptCache();
    setDirty(true);
    if(memoryOwned.has(k) || specOwned.has(k) || k==="vision_model" || k==="model") scheduleVRAMEstimate();
  });
  for(const el of modal.querySelectorAll('.llm-config[data-key][data-type="number"]')) el.addEventListener("input",()=>{
    const k=el.dataset.key;
    if(memoryOwned.has(k) || specOwned.has(k)) scheduleVRAMEstimate(180);
  });
  for(const b of modal.querySelectorAll("[data-action]")) b.onclick=async()=>{
    const a=b.dataset.action;
    if(["start","suspend","stop","reload"].includes(a)) return doAction(a);
    if(a==="save") return saveConfig();
    if(a==="tuner-start") return startTuner();
    if(a==="tuner-cancel") return cancelTuner();
    if(a==="tuner-use") return useTunerRecommendation();
    if(a==="tuner-save-preset") return saveTunerPreset();
    if(a==="refresh-logs") return refreshLogs();
    if(a==="copy-api"){await navigator.clipboard?.writeText(modal.querySelector("#llm-api-base").value);return;}
    if(a==="toggle-key"){const i=modal.querySelector("#llm-api-key");i.type=i.type==="password"?"text":"password";b.textContent=i.type==="password"?"Show":"Hide";return;}
    if(a==="regen-key"){const wasDirty=dirty;const d=await fetchJSON("/local_llm_server/api_key/regenerate",{method:"POST"});config.api_key=d.api_key;setConfigField("api_key",d.api_key);setDirty(wasDirty);return;}
  };
}

function closeModal(){ if(modal){modal.remove();modal=null;} if(logsTimer){clearInterval(logsTimer);logsTimer=null;} if(vramEstimateTimer){clearTimeout(vramEstimateTimer);vramEstimateTimer=null;} }

async function openModal() {
  injectCSS();
  try { await loadData(); renderModal(); } catch(e) { app.extensionManager?.toast?.add?.({severity:"error",summary:"Local LLM Server",detail:e.message,life:5000}); }
}

function collapseAccidentalLLMSidebar() {
  if (sidebarCollapsePending) return;
  sidebarCollapsePending = true;
  requestAnimationFrame(async () => {
    try {
      // Newer ComfyUI frontends expose a generated toggle command for sidebar
      // tabs. Prefer it when present; it cleanly returns the canvas to its prior
      // width without relying on private store APIs.
      const commandId = `Workspace.ToggleSidebarTab.${SIDEBAR_ID}`;
      const commands = app.extensionManager?.command?.commands || [];
      if (commands.some((cmd) => cmd?.id === commandId)) {
        allowNativeSidebarToggle = true;
        try { await app.extensionManager.command.execute(commandId); } finally { allowNativeSidebarToggle = false; }
        return;
      }

      // Compatibility fallback: if ComfyUI mounted our sidebar tab, clicking the
      // already-active launcher toggles it closed. Temporarily let this synthetic
      // click pass through our modal-only interception.
      const launcher = document.querySelector('[data-local-llm-launcher="1"]');
      if (launcher) {
        allowNativeSidebarToggle = true;
        try { launcher.click(); } finally { allowNativeSidebarToggle = false; }
      }
    } catch (e) {
      console.debug("[Local LLM] Could not auto-collapse accidental sidebar activation", e);
    } finally {
      sidebarCollapsePending = false;
    }
  });
}

function renderSidebarLauncher(el) {
  // This tab exists only to obtain a normal ComfyUI sidebar launcher/icon. The
  // actual UI is modal-only. If the host activates/mounts the tab despite our
  // event interception, immediately toggle it closed instead of opening a second
  // copy of the UI in the side panel.
  el.innerHTML=`<div class="llm-sidebar-launch"><div class="llm-card"><b>Local LLM Server</b><div class="llm-muted" style="margin:6px 0 10px">Opening…</div></div></div>`;
  collapseAccidentalLLMSidebar();
}

function nodeWidget(node, name) { return node?.widgets?.find((w) => w.name === name); }

function setNodeWidgetOption(w, key, value) {
  if (!w) return;
  w.options ||= {};
  w.options[key] = value;
  try { if (w._state?.options) w._state.options[key] = value; } catch (_) {}
}

function setServiceWidgetCapabilityHidden(w, hidden) {
  if (!w) return;
  if (w.__localLLMOriginalHidden === undefined) w.__localLLMOriginalHidden = !!(w.hidden || w.options?.hidden);
  const shouldHide = !!w.__localLLMOriginalHidden || !!hidden;
  w.hidden = shouldHide;
  setNodeWidgetOption(w, "hidden", shouldHide);
  for (const key of ["element", "inputEl"]) {
    const el = w[key];
    if (el?.style) el.style.display = shouldHide ? "none" : "";
  }
}

function nodeInputSlot(node, name) {
  return node?.inputs?.find?.((input) => input?.name === name) || null;
}

function restoreSlotColor(slot, key) {
  const savedKey = key === "color_on" ? "__localLLMOriginalColorOn" : "__localLLMOriginalColorOff";
  const value = slot?.[savedKey];
  if (value === undefined) {
    try { delete slot[key]; } catch (_) { slot[key] = undefined; }
  } else {
    slot[key] = value;
  }
}

function setServiceVisionSlotState(node, name, unsupported) {
  const slot = nodeInputSlot(node, name);
  if (!slot) return;
  if (!("__localLLMOriginalColorOn" in slot)) slot.__localLLMOriginalColorOn = slot.color_on;
  if (!("__localLLMOriginalColorOff" in slot)) slot.__localLLMOriginalColorOff = slot.color_off;

  // `label` is display-only; the backend/input identity remains `image` so old
  // workflows and Python keyword arguments stay fully compatible.
  const baseLabel = name === "image" ? "image(s)" : "video_frames";
  slot.__localLLMCapabilityDisabled = !!unsupported;
  slot.label = unsupported ? `${baseLabel}  ×` : baseLabel;
  if (unsupported) {
    // LiteGraph officially supports per-slot on/off colors. Use the familiar
    // red unavailable marker without changing the IMAGE type or removing links.
    slot.color_on = "#d85b5b";
    slot.color_off = "#d85b5b";
  } else {
    restoreSlotColor(slot, "color_on");
    restoreSlotColor(slot, "color_off");
  }
}

function syncServiceGenerateCapabilities(node, caps=status?.capabilities) {
  if (!node?.widgets) return;
  const visionUnsupported = caps?.vision === false;
  for (const name of ["vision_max_images", "vision_max_frames", "vision_max_edge"]) {
    setServiceWidgetCapabilityHidden(nodeWidget(node, name), visionUnsupported);
  }
  setServiceVisionSlotState(node, "image", visionUnsupported);
  setServiceVisionSlotState(node, "video_frames", visionUnsupported);
  node.__localLLMCapabilities = caps || null;
  redrawNode(node);
}

function syncAllServiceGenerateCapabilities(caps=status?.capabilities) {
  for (const node of [...serviceGenerateNodes]) {
    if (!node?.widgets) { serviceGenerateNodes.delete(node); continue; }
    syncServiceGenerateCapabilities(node, caps);
  }
}

function installServiceVisionConnectionGuard(node) {
  if (!node || node.__localLLMVisionConnectionGuard) return;
  node.__localLLMVisionConnectionGuard = true;
  const old = node.onConnectInput;
  node.onConnectInput = function(slotIndex, ...args) {
    const slot = this.inputs?.[slotIndex];
    if (slot?.__localLLMCapabilityDisabled) {
      console.warn(`[Local LLM] ${slot.name} is unavailable for the selected text-only model.`);
      return false;
    }
    const result = old?.call(this, slotIndex, ...args);
    return result === undefined ? true : result;
  };
}

function redrawNode(node) {
  // Redraw only. Do not clone the widgets array, call computeSize(), or call
  // setSize() here. Those operations let ComfyUI recalculate the node geometry
  // during preset changes, workflow/tab restore, and execution-time UI refreshes.
  // A user's chosen node dimensions must remain authoritative.
  try {
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  } catch (_) {}
}

function copyNodeSize(node) {
  const s = node?.size;
  if (!s || s.length < 2) return null;
  const w = Number(s[0]);
  const h = Number(s[1]);
  return Number.isFinite(w) && Number.isFinite(h) ? [w, h] : null;
}

function restoreNodeSize(node, size) {
  if (!node || !size) return;
  try {
    // Assign directly first so Vue/LiteGraph state observes the exact saved size,
    // then use setSize when available to keep legacy node internals synchronized.
    node.size = [size[0], size[1]];
    node.setSize?.([size[0], size[1]]);
  } catch (_) {
    try { node.size = [size[0], size[1]]; } catch (_) {}
  }
}

function preserveLoadedNodeSizeAcrossFrames(node, size) {
  if (!node || !size) return;
  restoreNodeSize(node, size);
  let remaining = 3;
  const reassert = () => {
    if (!node?.graph || remaining-- <= 0) return;
    restoreNodeSize(node, size);
    redrawNode(node);
    if (remaining > 0) requestAnimationFrame(reassert);
  };
  requestAnimationFrame(reassert);
}

function fitNewServiceGenerateNodeOnce(node) {
  // Only genuinely new nodes may auto-fit. nodeCreated also fires during workflow
  // deserialization, so callers must never invoke this while configureGraph runs.
  if (!node || node.__localLLMInitialFitDone) return;
  node.__localLLMInitialFitDone = true;
  try {
    const before = copyNodeSize(node) || [320, 100];
    const computed = node.computeSize?.();
    if (!computed) return;
    node.setSize?.([
      Math.max(before[0] || 320, computed[0] || 0),
      Math.max(before[1] || 100, computed[1] || 0),
    ]);
  } catch (_) {}
}

function setPresetChoices(widget, names, builtins=[]) {
  if (!widget) return;
  const choices = [...new Set([...(builtins || []), ...(names || []).filter(Boolean)])];
  if (widget.value && !choices.includes(widget.value)) choices.push(widget.value);
  setNodeWidgetOption(widget, "values", choices);
}

function applySamplerSettings(node, values) {
  if (!values || typeof values !== "object") return;
  node.__localLLMApplyingPreset = true;
  try {
    for (const key of REQUEST_PRESET_FIELDS) {
      if (!(key in values)) continue;
      const w = nodeWidget(node, key);
      if (w) w.value = values[key];
    }
  } finally {
    node.__localLLMApplyingPreset = false;
  }
  scheduleServiceNodePanelSync(node);
  redrawNode(node);
}

function applyTextPreset(node, widgetName, value) {
  node.__localLLMApplyingPreset = true;
  try {
    const w = nodeWidget(node, widgetName);
    if (w) w.value = value ?? "";
  } finally {
    node.__localLLMApplyingPreset = false;
  }
  scheduleServiceNodePanelSync(node);
  redrawNode(node);
}

async function fetchNodePresetCatalog() {
  return await fetchJSON("/local_llm_server/node_presets");
}

function installCatalogChoices(node, data) {
  const sampler = nodeWidget(node, "sampling_mode");
  const systemSel = nodeWidget(node, "system_prompt_preset");
  const promptSel = nodeWidget(node, "prompt_preset");
  setPresetChoices(sampler, data?.sampler?.names || [], ["Default", "Custom"]);
  setPresetChoices(systemSel, data?.system_prompts?.names || [], ["Custom"]);
  setPresetChoices(promptSel, data?.prompts?.names || [], ["Custom"]);
  node.__localLLMPresetCatalog = data;
  scheduleServiceNodePanelSync(node);
}

async function applyPresetSelection(node, kind, selection, suppliedCatalog=null) {
  const data = suppliedCatalog || await fetchNodePresetCatalog();
  installCatalogChoices(node, data);
  if (kind === "sampler") {
    const name = selection || "Default";
    if (name === "Default") applySamplerSettings(node, data?.sampler?.default || {});
    else if (name !== "Custom") applySamplerSettings(node, data?.sampler?.presets?.[name] || {});
    return;
  }
  if (kind === "system_prompts") {
    const name = selection || "Custom";
    if (name !== "Custom") applyTextPreset(node, "system_prompt", data?.system_prompts?.presets?.[name] ?? "");
    return;
  }
  if (kind === "prompts") {
    const name = selection || "Custom";
    if (name !== "Custom") applyTextPreset(node, "prompt", data?.prompts?.presets?.[name] ?? "");
  }
}

function currentSamplerSettings(node) {
  const out = {};
  for (const key of REQUEST_PRESET_FIELDS) {
    const w = nodeWidget(node, key);
    if (w) out[key] = w.value;
  }
  return out;
}

function toast(severity, summary, detail) {
  try { app.extensionManager?.toast?.add?.({severity, summary, detail, life:5000}); } catch (_) {}
}

async function saveNodePreset(node, kind) {
  const meta = {
    sampler: {selector:"sampling_mode", label:"Sampler", endpointKind:"sampler"},
    system_prompts: {selector:"system_prompt_preset", label:"System Prompt", endpointKind:"system_prompts", textWidget:"system_prompt"},
    prompts: {selector:"prompt_preset", label:"Prompt", endpointKind:"prompts", textWidget:"prompt"},
  }[kind];
  if (!meta) return;
  const selector = nodeWidget(node, meta.selector);
  if (!selector) return;
  if (selector.value !== "Custom") {
    toast("warn", `Local LLM ${meta.label} Preset`, `Edit the ${meta.label.toLowerCase()} first; edited values automatically switch this selector to Custom.`);
    return;
  }
  const requested = window.prompt(`Name this ${meta.label.toLowerCase()} preset:`, "");
  if (!requested?.trim()) return;
  const body = {kind:meta.endpointKind, name:requested.trim()};
  if (kind === "sampler") body.settings = currentSamplerSettings(node);
  else body.text = nodeWidget(node, meta.textWidget)?.value ?? "";
  try {
    const result = await fetchJSON("/local_llm_server/node_presets", {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body),
    });
    const data = result.catalog || await fetchNodePresetCatalog();
    installCatalogChoices(node, data);
    selector.value = result.saved?.name || requested.trim();
    if (kind === "sampler") applySamplerSettings(node, result.saved?.settings || body.settings);
    else applyTextPreset(node, meta.textWidget, result.saved?.text ?? body.text);
    scheduleServiceNodePanelSync(node);
    redrawNode(node);
    refreshAllDefaultServiceGenerateNodes();
    toast("success", `Local LLM ${meta.label} Preset`, `Saved ${selector.value}`);
  } catch (e) {
    toast("error", `Local LLM ${meta.label} Preset`, e.message);
  }
}

async function deleteNodePreset(node, kind) {
  const meta = {
    sampler: {selector:"sampling_mode", label:"Sampler", endpointKind:"sampler"},
    system_prompts: {selector:"system_prompt_preset", label:"System Prompt", endpointKind:"system_prompts"},
    prompts: {selector:"prompt_preset", label:"Prompt", endpointKind:"prompts"},
  }[kind];
  if (!meta) return;
  const selector = nodeWidget(node, meta.selector);
  if (!selector) return;
  const name = String(selector.value || "").trim();
  if (!name || name === "Default" || name === "Custom") {
    toast("warn", `Local LLM ${meta.label} Preset`, "Select a saved user preset to delete. Built-in selectors cannot be deleted.");
    return;
  }
  let data;
  try { data = await fetchNodePresetCatalog(); }
  catch (e) { toast("error", `Local LLM ${meta.label} Preset`, e.message); return; }
  const section = data?.[meta.endpointKind] || {};
  const deletable = section.deletable_names || Object.keys(section.presets || {});
  if (!deletable.includes(name)) {
    toast("warn", `Local LLM ${meta.label} Preset`, `“${name}” is not a deletable user preset.`);
    return;
  }
  if (!window.confirm(`Delete ${meta.label.toLowerCase()} preset “${name}”?\n\nThe current values in this node will remain.`)) return;
  try {
    const result = await fetchJSON("/local_llm_server/node_presets", {
      method:"DELETE", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({kind:meta.endpointKind, name}),
    });
    const updated = result.catalog || await fetchNodePresetCatalog();
    installCatalogChoices(node, updated);
    selector.value = "Custom";
    scheduleServiceNodePanelSync(node);
    redrawNode(node);
    refreshAllDefaultServiceGenerateNodes();
    toast("success", `Local LLM ${meta.label} Preset`, `Deleted ${name}`);
  } catch (e) {
    toast("error", `Local LLM ${meta.label} Preset`, e.message);
  }
}

function installPresetCallbacks(node) {
  if (node.__localLLMPresetCallbacks) return;
  node.__localLLMPresetCallbacks = true;
  const selectors = [
    ["sampling_mode", "sampler"],
    ["system_prompt_preset", "system_prompts"],
    ["prompt_preset", "prompts"],
  ];
  for (const [widgetName, kind] of selectors) {
    const w = nodeWidget(node, widgetName);
    if (!w) continue;
    if (widgetName === "sampling_mode") w.label = "Sampler Preset";
    else if (widgetName === "system_prompt_preset") w.label = "System Prompt Preset";
    else w.label = "Prompt Preset";
    const old = w.callback;
    w.callback = function(value, ...args) {
      const result = old?.call(this, value, ...args);
      if (!node.__localLLMApplyingPreset) {
        applyPresetSelection(node, kind, value)
          .then(()=>scheduleServiceNodePanelSync(node))
          .catch((e)=>console.warn("[Local LLM] preset load failed", e));
      }
      scheduleServiceNodePanelSync(node);
      return result;
    };
  }

  for (const key of REQUEST_PRESET_FIELDS) {
    const w = nodeWidget(node, key);
    if (!w) continue;
    const old = w.callback;
    w.callback = function(value, ...args) {
      const result = old?.call(this, value, ...args);
      if (!node.__localLLMApplyingPreset) {
        const p = nodeWidget(node, "sampling_mode");
        if (p && p.value !== "Custom") p.value = "Custom";
      }
      persistServiceNodeUIState(node);
      scheduleServiceNodePanelSync(node);
      redrawNode(node);
      return result;
    };
  }
  for (const [textName, selectorName] of [["system_prompt","system_prompt_preset"],["prompt","prompt_preset"]]) {
    const w = nodeWidget(node, textName);
    if (!w) continue;
    const old = w.callback;
    w.callback = function(value, ...args) {
      const result = old?.call(this, value, ...args);
      if (!node.__localLLMApplyingPreset) {
        const p = nodeWidget(node, selectorName);
        if (p && p.value !== "Custom") p.value = "Custom";
      }
      persistServiceNodeUIState(node);
      scheduleServiceNodePanelSync(node);
      redrawNode(node);
      return result;
    };
  }
}

const GENERATE_SNAPSHOT_KEY = "local_llm_service_generate_values_v087";
const SERVICE_UI_STATE_KEY = "local_llm_service_ui_state_v1";
const SERVICE_TEXTAREA_KEYS = ["system_prompt", "prompt"];
const SERVICE_PANEL_WIDGET_NAME = "local_llm_service_panel";
const SERVICE_NUMERIC_FIELDS = [
  ["temperature", "Temperature"],
  ["top_p", "Top P"],
  ["top_k", "Top K"],
  ["min_p", "Min P"],
  ["repeat_penalty", "Repeat Penalty"],
  ["presence_penalty", "Presence Penalty"],
  ["frequency_penalty", "Frequency Penalty"],
  ["max_tokens", "Max Tokens"],
  ["vision_max_images", "Vision Max Images"],
  ["vision_max_frames", "Vision Max Frames"],
  ["vision_max_edge", "Vision Max Edge"],
];

function markServiceNodeChanged(node) {
  try { node?.graph?.change?.(); } catch (_) {}
  redrawNode(node);
}

function collectGenerateWidgetValues(node) {
  const out = {};
  for (const w of node?.widgets || []) {
    if (!w?.name || w.name === SERVICE_PANEL_WIDGET_NAME || w.__localLLMPresetSaveButton || w.__localLLMPresetActionRow) continue;
    if (w.type === "button") continue;
    try { out[w.name] = w.value; } catch (_) {}
  }
  return out;
}

function normalizeServiceTextareaHeight(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 80) return 0;
  return Math.max(80, Math.min(1600, Math.round(n)));
}

function serviceTextareaHeightSnapshot(node) {
  const prior = node?.__localLLMTextareaHeights || {};
  const controls = node?.__localLLMPanelControls || {};
  const result = {};
  for (const key of SERVICE_TEXTAREA_KEYS) {
    const element = controls[key];
    const measured = normalizeServiceTextareaHeight(element?.getBoundingClientRect?.().height || element?.offsetHeight);
    const fallback = normalizeServiceTextareaHeight(prior[key]);
    const value = measured || fallback;
    if (value) result[key] = value;
  }
  return result;
}

function applyServiceTextareaHeights(node, heights, {schedule=true}={}) {
  if (!node) return;
  const normalized = {};
  for (const key of SERVICE_TEXTAREA_KEYS) {
    const h = normalizeServiceTextareaHeight(heights?.[key]);
    if (h) normalized[key] = h;
  }
  node.__localLLMTextareaHeights = normalized;
  const controls = node.__localLLMPanelControls || {};
  node.__localLLMApplyingTextareaHeights = true;
  try {
    for (const key of SERVICE_TEXTAREA_KEYS) {
      const el = controls[key];
      if (!el) continue;
      const h = normalized[key];
      el.style.height = h ? `${h}px` : "";
    }
  } finally { node.__localLLMApplyingTextareaHeights = false; }
  if (schedule) scheduleServiceNodePanelHeight(node);
}

function captureServiceTextareaHeights(node) {
  if (!node || node.__localLLMApplyingTextareaHeights || !node.__localLLMPanelControls) return;
  const next = serviceTextareaHeightSnapshot(node);
  const prev = node.__localLLMTextareaHeights || {};
  const changed = SERVICE_TEXTAREA_KEYS.some((key)=>Number(next[key]||0)!==Number(prev[key]||0));
  if (!changed) return;
  node.__localLLMTextareaHeights = next;
  persistServiceNodeUIState(node);
  markServiceNodeChanged(node);
  scheduleServiceNodePanelHeight(node);
}

function persistServiceNodeUIState(node, serializedData=null) {
  if (!node) return;
  const state = { textareaHeights: serviceTextareaHeightSnapshot(node) };
  node.properties ||= {};
  node.properties[SERVICE_UI_STATE_KEY] = state;
  if (serializedData) {
    serializedData.properties ||= {};
    serializedData.properties[SERVICE_UI_STATE_KEY] = state;
  }
}

function restoreServiceNodeUIState(node) {
  const state = node?.properties?.[SERVICE_UI_STATE_KEY];
  if (!state) return;
  applyServiceTextareaHeights(node, state.textareaHeights, {schedule:false});
}

function restoreGenerateWidgetValues(node) {
  const saved = node?.properties?.[GENERATE_SNAPSHOT_KEY] || node?.properties?.local_llm_service_generate_values_v082;
  if (!saved || typeof saved !== "object") return;
  node.__localLLMApplyingPreset = true;
  try {
    for (const [name, value] of Object.entries(saved)) {
      const w = nodeWidget(node, name);
      if (w) w.value = value;
    }
  } finally { node.__localLLMApplyingPreset = false; }
}

function installGeneratePersistence(node) {
  if (node.__localLLMGeneratePersistence) return;
  node.__localLLMGeneratePersistence = true;
  const old = node.onSerialize;
  node.onSerialize = function(o) {
    this.properties ||= {};
    const values = collectGenerateWidgetValues(this);
    this.properties[GENERATE_SNAPSHOT_KEY] = values;
    persistServiceNodeUIState(this, o);
    const result = old?.call(this, o);
    if (o) {
      o.properties ||= {};
      o.properties[GENERATE_SNAPSHOT_KEY] = values;
      if (Array.isArray(o.inputs)) {
        for (const input of o.inputs) {
          if (input?.name === "image") input.label = "image(s)";
          else if (input?.name === "video_frames" && typeof input.label === "string") input.label = input.label.replace(/\s*×\s*$/, "");
        }
      }
    }
    return result;
  };
}

function serviceSeedControlWidget(node) {
  const seed = nodeWidget(node, "seed");
  return (seed?.linkedWidgets || []).find((w)=>String(w?.name || "").toLowerCase().includes("control")) || null;
}

function servicePanelComboValues(w) {
  if (!w) return [];
  let values = w.options?.values;
  if (typeof values === "function") {
    try { values = values(); } catch (_) { values = []; }
  }
  return Array.isArray(values) ? values.map((v)=>String(v ?? "")) : [];
}

function servicePanelElement(tag, className="") {
  const el = document.createElement(tag);
  if (className) el.className = className;
  return el;
}

function servicePanelButton(label, title, callback) {
  const b = servicePanelElement("button", "llmn-button");
  b.type = "button";
  b.textContent = label;
  b.title = title || label;
  b.addEventListener("click", (e)=>{e.preventDefault();e.stopPropagation();callback?.();});
  return b;
}

function servicePanelIconButton(kind, title, callback) {
  const b = servicePanelButton("", title, callback);
  b.classList.add("llmn-icon-button");
  if (kind === "delete") b.classList.add("llmn-delete");
  b.setAttribute("aria-label", title);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = kind === "save"
    ? '<path d="M5 3h12l2 2v16H5z"/><path d="M8 3v6h8V3"/><path d="M8 21v-7h8v7"/>'
    : '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M7 7l1 14h8l1-14"/><path d="M10 11v6M14 11v6"/>';
  b.appendChild(svg);
  return b;
}

function servicePanelFieldLabel(text) {
  const label = servicePanelElement("div", "llmn-field-label");
  label.textContent = text;
  return label;
}

function setServicePanelSelectOptions(select, values, selected) {
  if (!select) return;
  const normalized = Array.isArray(values) ? values.map(String) : [];
  const signature = JSON.stringify(normalized);
  if (select.__localLLMOptionsSignature !== signature) {
    select.__localLLMOptionsSignature = signature;
    const frag = document.createDocumentFragment();
    for (const value of normalized) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      frag.appendChild(option);
    }
    select.replaceChildren(frag);
  }
  const next = String(selected ?? "");
  if (select.value !== next) select.value = next;
}

function setServicePanelNativeValue(node, name, value) {
  const w = nodeWidget(node, name);
  if (!w) return;
  w.value = value;
  try { w.callback?.(value); } catch (_) {}
  persistServiceNodeUIState(node);
  markServiceNodeChanged(node);
  scheduleServiceNodePanelSync(node);
}

const SERVICE_INTEGER_FIELDS = new Set(["top_k","max_tokens","vision_max_images","vision_max_frames","vision_max_edge","seed"]);

function servicePanelNumericStep(w, name) {
  const exact = Number(w?.options?.step2);
  if (Number.isFinite(exact) && exact > 0) return exact;
  // ComfyUI's legacy numeric widget stores `step` at 10x the input-spec step.
  // Newer frontends expose the exact value as `step2`; divide only in this
  // compatibility fallback so this DOM control follows both generations.
  const legacy = Number(w?.options?.step);
  if (Number.isFinite(legacy) && legacy > 0) return legacy / 10;
  const precision = Number(w?.options?.precision);
  if (Number.isInteger(precision) && precision >= 0 && precision <= 12) return precision === 0 ? 1 : 10 ** (-precision);
  return SERVICE_INTEGER_FIELDS.has(name) ? 1 : 1;
}

function servicePanelNumericPrecision(w, name, step=servicePanelNumericStep(w,name)) {
  const configured = Number(w?.options?.precision);
  if (Number.isInteger(configured) && configured >= 0 && configured <= 12) return configured;
  if (SERVICE_INTEGER_FIELDS.has(name)) return 0;
  if (!Number.isFinite(step) || step <= 0) return undefined;
  const text = String(step).toLowerCase();
  if (text.includes("e-")) {
    const n = Number(text.split("e-")[1]);
    return Number.isFinite(n) ? Math.min(12, Math.max(0, n)) : undefined;
  }
  const dot = text.indexOf(".");
  return dot < 0 ? 0 : Math.min(12, text.length - dot - 1);
}

function normalizeServicePanelNumericValue(w, name, raw) {
  let value = Number(raw);
  if (!Number.isFinite(value)) value = Number(w?.value) || 0;
  const min = Number(w?.options?.min), max = Number(w?.options?.max);
  const step = servicePanelNumericStep(w, name);
  if (Number.isFinite(step) && step > 0) {
    const anchor = Number.isFinite(min) ? min : 0;
    value = anchor + Math.round((value - anchor) / step) * step;
    const precision = servicePanelNumericPrecision(w, name, step);
    if (Number.isInteger(precision) && precision >= 0 && precision <= 12) value = Number(value.toFixed(precision));
  }
  if (Number.isFinite(min)) value = Math.max(min, value);
  if (Number.isFinite(max)) value = Math.min(max, value);
  if (SERVICE_INTEGER_FIELDS.has(name)) value = Math.round(value);
  return value;
}

function formatServicePanelNumericValue(w, name, value=w?.value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value ?? "");
  const precision = servicePanelNumericPrecision(w, name);
  return Number.isInteger(precision) && precision >= 0 ? n.toFixed(precision) : String(n);
}

function servicePanelNumericPercent(w, value=w?.value) {
  const min = Number(w?.options?.min), max = Number(w?.options?.max), n = Number(value);
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(n) || max <= min) return null;
  return Math.max(0, Math.min(100, ((n - min) / (max - min)) * 100));
}

function setServicePanelNativeNumericValue(node, name, raw) {
  const w = nodeWidget(node, name);
  if (!w) return;
  setServicePanelNativeValue(node, name, normalizeServicePanelNumericValue(w, name, raw));
}

function hideNativeServiceWidget(w) {
  if (!w || w.__localLLMDomHidden) return;
  w.__localLLMDomHidden = true;
  w.hidden = true;
  setNodeWidgetOption(w, "hidden", true);
  w.computeSize = () => [0,0];
  w.computeLayoutSize = () => ({minHeight:0,maxHeight:0,minWidth:0});
}

function hideNativeWidgetsForServicePanel(node) {
  if (!node?.__localLLMDomPanelWidget) return;
  const names = [
    "system_prompt_preset","system_prompt","prompt_preset","prompt","sampling_mode",
    ...REQUEST_PRESET_FIELDS,"vision_max_images","vision_max_frames","vision_max_edge","seed",
  ];
  for (const name of names) hideNativeServiceWidget(nodeWidget(node, name));
  hideNativeServiceWidget(serviceSeedControlWidget(node));
}

function removeCopiedServicePanelWidget(node) {
  if (!node?.widgets) return;
  for (let i=node.widgets.length-1;i>=0;i--) {
    const w=node.widgets[i];
    if (w?.name !== SERVICE_PANEL_WIDGET_NAME) continue;
    node.widgets.splice(i,1);
    try { w.onRemove?.(); } catch (_) {}
  }
}

function pinServiceDomWidgetFullWidth(domWidget) {
  if (!domWidget || domWidget.__localLLMFullWidthPinned) return;
  domWidget.__localLLMFullWidthPinned = true;
  try {
    Object.defineProperty(domWidget, "width", {configurable:true,enumerable:true,get(){return undefined;},set(_v){}});
  } catch (_) { try { domWidget.width = undefined; } catch (_) {} }
}

function clampServicePanelToNode(node) {
  for (const el of [node?.__localLLMPanelRoot,node?.__localLLMPanelContent]) {
    if (!el) continue;
    el.style.minWidth="0";
    el.style.maxWidth="100%";
  }
}

function measureServicePanelHeight(node) {
  const root=node?.__localLLMPanelRoot, content=node?.__localLLMPanelContent;
  if (!root || !content) return 420;
  let contentHeight=Number(content.scrollHeight)||Number(content.offsetHeight)||0;
  let padding=10;
  try {
    const s=getComputedStyle(root);
    padding=(parseFloat(s.paddingTop)||0)+(parseFloat(s.paddingBottom)||0);
  } catch (_) {}
  if (contentHeight<=0) return Number(node.__localLLMPanelMeasuredHeight)||420;
  return Math.max(180,Math.ceil(contentHeight+padding+4));
}

function measureServicePanelNodeHeight(node, panelHeight) {
  const w=node?.__localLLMDomPanelWidget;
  const margin=Math.max(0,Number(w?.margin ?? w?.options?.margin ?? 4)||0);
  const y=Number(w?.y);
  let geometryHeight=Number.isFinite(y)&&y>=0 ? Math.ceil(y+panelHeight+margin) : 0;
  let computedHeight=0;
  try { computedHeight=Number(node?.computeSize?.()?.[1])||0; } catch (_) {}
  return Math.max(120,geometryHeight,computedHeight);
}

function scheduleServiceNodePanelHeight(node) {
  if (!node?.__localLLMPanelContent || node.__localLLMPanelHeightPending) return;
  node.__localLLMPanelHeightPending=true;
  requestAnimationFrame(()=>{
    node.__localLLMPanelHeightPending=false;
    const height=measureServicePanelHeight(node);
    const previous=Number(node.__localLLMPanelMeasuredHeight)||0;
    const current=copyNodeSize(node);
    const currentWidth=Number(current?.[0])||0;
    const previousWidth=Number(node.__localLLMPanelMeasuredNodeWidth)||0;
    node.__localLLMPanelMeasuredHeight=height;
    node.__localLLMPanelMeasuredNodeWidth=currentWidth;
    const root=node.__localLLMPanelRoot;
    if(root) root.style.setProperty("--comfy-widget-min-height",`${height}px`);
    const w=node.__localLLMDomPanelWidget;
    if(w?.options){
      w.options.getMinHeight=()=>Number(node.__localLLMPanelMeasuredHeight)||height;
      w.options.getHeight=()=>Number(node.__localLLMPanelMeasuredHeight)||height;
    }
    clampServicePanelToNode(node);
    pinServiceDomWidgetFullWidth(w);
    if((Math.abs(height-previous)>=1 || previous===0 || Math.abs(currentWidth-previousWidth)>=1) && !node.__localLLMAutoSizing){
      try{
        const size=copyNodeSize(node);
        if(size){
          const target=measureServicePanelNodeHeight(node,height);
          if(Math.abs(size[1]-target)>=2){
            node.__localLLMAutoSizing=true;
            node.setSize?.([size[0],target]);
            requestAnimationFrame(()=>{
              node.__localLLMAutoSizing=false;
              scheduleServiceNodePanelHeight(node);
            });
          }
        }
      }catch(_){node.__localLLMAutoSizing=false;}
    }
    redrawNode(node);
  });
}

function restoreLoadedServiceWidthAndAutosize(node, saved) {
  if (!node || !saved) return;
  const current=copyNodeSize(node)||[420,120];
  const width=Math.max(320,Number(saved[0])||current[0]||420);
  restoreNodeSize(node,[width,current[1]]);
  const settle=()=>{
    pinServiceDomWidgetFullWidth(node.__localLLMDomPanelWidget);
    clampServicePanelToNode(node);
    scheduleServiceNodePanelHeight(node);
  };
  requestAnimationFrame(()=>{settle();requestAnimationFrame(settle);});
}

function serviceSettingsConnected(node) {
  const slot=nodeInputSlot(node,"settings");
  if (!slot) return false;
  return slot.link != null || (Array.isArray(slot.links) && slot.links.length>0);
}

function addServicePanelPresetField(node, content, controls, kind, selectorName, labelText) {
  if (!nodeWidget(node, selectorName)) return;
  const field=servicePanelElement("div","llmn-field");
  field.appendChild(servicePanelFieldLabel(labelText));
  const row=servicePanelElement("div","llmn-selector-actions");
  const save=servicePanelIconButton("save",`Save ${labelText}`,()=>void saveNodePreset(node,kind));
  const select=servicePanelElement("select","llmn-select");
  select.addEventListener("change",()=>setServicePanelNativeValue(node,selectorName,select.value));
  const del=servicePanelIconButton("delete",`Delete selected ${labelText.toLowerCase()}`,()=>void deleteNodePreset(node,kind));
  row.append(save,select,del);
  field.appendChild(row);
  content.appendChild(field);
  controls[`${selectorName}_select`]=select;
  controls[`${selectorName}_save`]=save;
  controls[`${selectorName}_delete`]=del;
}

function servicePanelStepIcon(kind) {
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
  svg.setAttribute("viewBox","0 0 24 24");
  svg.setAttribute("aria-hidden","true");
  svg.innerHTML=kind==="minus" ? '<path d="M5 12h14"/>' : '<path d="M12 5v14M5 12h14"/>';
  return svg;
}

function setServicePanelNumberEditing(input, editing) {
  const shell=input?.__localLLMNumberControl?.shell;
  if(!shell) return;
  shell.classList.toggle("llmn-number-editing",!!editing);
  if(editing){
    input.value=String(nodeWidget(input.__localLLMNode,input.__localLLMName)?.value ?? input.value);
    requestAnimationFrame(()=>{try{input.focus({preventScroll:true});input.select();}catch(_){}});
  }
}

function syncServicePanelNumberControl(input, w, name) {
  const control=input?.__localLLMNumberControl;
  if(!input || !control || !w) return;
  const editing=control.shell.classList.contains("llmn-number-editing") && document.activeElement===input;
  if(!editing){
    const text=formatServicePanelNumericValue(w,name,w.value);
    if(input.value!==text) input.value=text;
  }
  const pct=servicePanelNumericPercent(w,w.value);
  if(pct==null){control.fill.style.display="none";control.fill.style.width="0%";}
  else{control.fill.style.display="block";control.fill.style.width=`${pct}%`;}
  const min=Number(w.options?.min), max=Number(w.options?.max), value=Number(w.value);
  const disabled=!!input.disabled;
  control.dec.disabled=disabled || (Number.isFinite(min) && Number.isFinite(value) && value<=min);
  control.inc.disabled=disabled || (Number.isFinite(max) && Number.isFinite(value) && value>=max);
  input.setAttribute("aria-valuenow",String(w.value ?? ""));
  if(Number.isFinite(min)) input.setAttribute("aria-valuemin",String(min)); else input.removeAttribute("aria-valuemin");
  if(Number.isFinite(max)) input.setAttribute("aria-valuemax",String(max)); else input.removeAttribute("aria-valuemax");
}

function setServicePanelNumberDisabled(input, disabled) {
  if(!input) return;
  input.disabled=!!disabled;
  const control=input.__localLLMNumberControl;
  if(!control) return;
  control.shell.classList.toggle("llmn-number-disabled",!!disabled);
  if(disabled && control.shell.classList.contains("llmn-number-editing")){control.shell.classList.remove("llmn-number-editing");try{input.blur();}catch(_){}}
  syncServicePanelNumberControl(input,nodeWidget(input.__localLLMNode,input.__localLLMName),input.__localLLMName);
}

function createServicePanelNumberControl(node, name, labelText) {
  const w=nodeWidget(node,name);
  if(!w) return null;
  const shell=servicePanelElement("div","llmn-number");
  shell.title=w.options?.tooltip || labelText || name;
  const fill=servicePanelElement("div","llmn-number-fill");
  const dec=servicePanelElement("button","llmn-number-step llmn-number-dec");
  dec.type="button"; dec.setAttribute("aria-label",`Decrease ${labelText || name}`); dec.appendChild(servicePanelStepIcon("minus"));
  const center=servicePanelElement("div","llmn-number-center");
  const input=servicePanelElement("input","llmn-number-input");
  input.type="text"; input.inputMode=SERVICE_INTEGER_FIELDS.has(name)?"numeric":"decimal"; input.autocomplete="off"; input.spellcheck=false;
  input.setAttribute("role","spinbutton"); input.setAttribute("aria-label",labelText || name);
  const scrub=servicePanelElement("div","llmn-number-scrub");
  scrub.setAttribute("aria-hidden","true");
  const inc=servicePanelElement("button","llmn-number-step llmn-number-inc");
  inc.type="button"; inc.setAttribute("aria-label",`Increase ${labelText || name}`); inc.appendChild(servicePanelStepIcon("plus"));
  center.append(input,scrub); shell.append(fill,dec,center,inc);
  input.__localLLMNode=node; input.__localLLMName=name;
  input.__localLLMNumberControl={shell,fill,dec,inc,center,scrub};

  const nudge=(direction,multiplier=1)=>{
    if(input.disabled) return;
    const step=servicePanelNumericStep(w,name);
    const current=Number(w.value)||0;
    setServicePanelNativeNumericValue(node,name,current+direction*step*multiplier);
  };
  dec.addEventListener("click",(e)=>{e.preventDefault();e.stopPropagation();nudge(-1);});
  inc.addEventListener("click",(e)=>{e.preventDefault();e.stopPropagation();nudge(1);});

  const commit=()=>{
    if(!shell.classList.contains("llmn-number-editing")) return;
    setServicePanelNativeNumericValue(node,name,input.value);
    shell.classList.remove("llmn-number-editing");
    syncServicePanelNumberControl(input,w,name);
  };
  input.addEventListener("blur",commit);
  input.addEventListener("keydown",(e)=>{
    if(e.key==="Enter"){e.preventDefault();commit();input.blur();}
    else if(e.key==="Escape"){e.preventDefault();shell.classList.remove("llmn-number-editing");syncServicePanelNumberControl(input,w,name);input.blur();}
    else if(e.key==="ArrowUp"){e.preventDefault();nudge(1);}
    else if(e.key==="ArrowDown"){e.preventDefault();nudge(-1);}
    else if(e.key==="PageUp"){e.preventDefault();nudge(1,10);}
    else if(e.key==="PageDown"){e.preventDefault();nudge(-1,10);}
  });

  scrub.addEventListener("pointerdown",(e)=>{
    if(input.disabled || e.button!==0) return;
    e.preventDefault();e.stopPropagation();
    const pointerId=e.pointerId, startX=e.clientX, startValue=Number(w.value)||0;
    const step=servicePanelNumericStep(w,name);
    let appliedSteps=0, moved=false;
    shell.classList.add("llmn-number-scrubbing");
    try{scrub.setPointerCapture(pointerId);}catch(_){}
    const move=(ev)=>{
      if(ev.pointerId!==pointerId) return;
      const dx=ev.clientX-startX;
      const totalSteps=Math.trunc(dx/10);
      if(totalSteps!==0) moved=true;
      if(totalSteps===appliedSteps) return;
      appliedSteps=totalSteps;
      setServicePanelNativeNumericValue(node,name,startValue+totalSteps*step);
      ev.preventDefault();ev.stopPropagation();
    };
    const finish=(ev,cancelled=false)=>{
      if(ev.pointerId!==pointerId) return;
      scrub.removeEventListener("pointermove",move);
      scrub.removeEventListener("pointerup",up);
      scrub.removeEventListener("pointercancel",cancel);
      shell.classList.remove("llmn-number-scrubbing");
      try{scrub.releasePointerCapture(pointerId);}catch(_){}
      if(!cancelled && !moved && Math.abs(ev.clientX-startX)<4) setServicePanelNumberEditing(input,true);
      ev.preventDefault();ev.stopPropagation();
    };
    const up=(ev)=>finish(ev,false), cancel=(ev)=>finish(ev,true);
    scrub.addEventListener("pointermove",move);
    scrub.addEventListener("pointerup",up);
    scrub.addEventListener("pointercancel",cancel);
  });
  syncServicePanelNumberControl(input,w,name);
  return {shell,input};
}

function addServicePanelNumberField(node, content, controls, name, labelText) {
  const created=createServicePanelNumberControl(node,name,labelText);
  if(!created) return;
  const field=servicePanelElement("div","llmn-field");
  field.appendChild(servicePanelFieldLabel(labelText));
  field.appendChild(created.shell);
  content.appendChild(field);
  controls[name]=created.input;
}

function addServicePanelTextarea(node, content, controls, name, placeholder) {
  if(!nodeWidget(node,name)) return;
  const area=servicePanelElement("textarea","llmn-textarea");
  area.placeholder=placeholder;
  area.spellcheck=true;
  area.title=nodeWidget(node,name)?.options?.tooltip || placeholder;
  area.addEventListener("input",()=>setServicePanelNativeValue(node,name,area.value));
  content.appendChild(area);
  controls[name]=area;
}

function createServiceNodeDomPanel(node) {
  if (!node || typeof node.addDOMWidget !== "function") return null;
  if (node.__localLLMDomPanelWidget?.node === node) return node.__localLLMDomPanelWidget;
  try { node.__localLLMPanelResizeObserver?.disconnect?.(); } catch (_) {}
  try { node.__localLLMTextareaResizeObserver?.disconnect?.(); } catch (_) {}
  node.__localLLMPanelResizeObserver=null;
  node.__localLLMTextareaResizeObserver=null;
  node.__localLLMDomPanelWidget=null;
  node.__localLLMPanelControls=null;
  node.__localLLMPanelRoot=null;
  node.__localLLMPanelContent=null;
  node.__localLLMPanelMeasuredHeight=0;
  removeCopiedServicePanelWidget(node);

  const root=servicePanelElement("div","local-llm-service-node-panel");
  root.dataset.nodeId=String(node.id ?? "");
  root.setAttribute("role","group");
  root.setAttribute("aria-label","Local LLM node controls");
  Object.assign(root.style,{width:"100%",minWidth:"0",maxWidth:"100%",boxSizing:"border-box",display:"block",position:"relative",padding:"2px 3px 8px",color:"inherit",fontFamily:"inherit",fontSize:"13px",lineHeight:"1.25",userSelect:"text",touchAction:"manipulation",overflow:"visible"});
  for(const eventName of ["pointerdown","mousedown","touchstart","click","dblclick","contextmenu"]) root.addEventListener(eventName,(e)=>e.stopPropagation());
  root.addEventListener("wheel",(e)=>e.stopPropagation(),{passive:true});

  const css=servicePanelElement("style");
  css.textContent=`
    .local-llm-service-node-panel *{box-sizing:border-box}
    .local-llm-service-node-panel,.local-llm-service-node-panel .llmn-content,.local-llm-service-node-panel .llmn-field,.local-llm-service-node-panel .llmn-selector-actions,.local-llm-service-node-panel .llmn-seed-grid{min-width:0;max-width:100%}
    .local-llm-service-node-panel .llmn-textarea,.local-llm-service-node-panel .llmn-input{width:100%;min-width:0;max-width:100%;color:inherit;font:inherit;background:rgba(127,127,127,.12);border:1px solid rgba(127,127,127,.42);border-radius:6px;outline:none}
    .local-llm-service-node-panel .llmn-select{width:100%;min-width:0;max-width:100%;font:inherit;color:#f3f4f6!important;background-color:#27292d!important;border:1px solid #555a62;border-radius:6px;outline:none;color-scheme:dark}
    .local-llm-service-node-panel .llmn-select option,.local-llm-service-node-panel .llmn-select optgroup{color:#f3f4f6!important;background-color:#27292d!important}
    .local-llm-service-node-panel .llmn-textarea{display:block;min-height:108px;padding:8px 9px;resize:vertical;line-height:1.35}
    .local-llm-service-node-panel .llmn-select:focus,.local-llm-service-node-panel .llmn-input:focus,.local-llm-service-node-panel .llmn-textarea:focus{border-color:rgba(120,170,255,.9);box-shadow:0 0 0 1px rgba(120,170,255,.32)}
    .local-llm-service-node-panel .llmn-select,.local-llm-service-node-panel .llmn-input{min-height:38px;padding:5px 8px}
    .local-llm-service-node-panel .llmn-number{position:relative;display:flex;align-items:stretch;width:100%;min-width:0;height:38px;overflow:hidden;border:1px solid rgba(127,127,127,.38);border-radius:10px;background:rgba(127,127,127,.14);color:inherit;font-variant-numeric:tabular-nums;isolation:isolate;user-select:none}
    .local-llm-service-node-panel .llmn-number-fill{position:absolute;z-index:0;inset:0 auto 0 0;width:0;pointer-events:none;background:rgba(71,133,181,.38);transition:width .06s linear}
    .local-llm-service-node-panel .llmn-number-step{position:relative;z-index:2;flex:0 0 38px;width:38px;height:100%;padding:0;border:0;border-radius:0;background:transparent;color:rgba(235,235,235,.6);display:flex;align-items:center;justify-content:center;cursor:pointer;touch-action:manipulation}
    .local-llm-service-node-panel .llmn-number-step:hover:not(:disabled){background:rgba(255,255,255,.07);color:rgba(255,255,255,.82)}
    .local-llm-service-node-panel .llmn-number-step:active:not(:disabled){background:rgba(255,255,255,.11)}
    .local-llm-service-node-panel .llmn-number-step:disabled{opacity:.28;cursor:default}
    .local-llm-service-node-panel .llmn-number-step svg{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;pointer-events:none}
    .local-llm-service-node-panel .llmn-number-center{position:relative;z-index:2;min-width:4ch;flex:1 1 auto;height:100%}
    .local-llm-service-node-panel .llmn-number-input{position:absolute;inset:0;width:100%;height:100%;min-width:0;border:0;outline:0;background:transparent;color:#f5f5f5;font:inherit;font-size:15px;font-weight:500;text-align:center;padding:0 5px;font-variant-numeric:tabular-nums;user-select:text}
    .local-llm-service-node-panel .llmn-number-input:focus{outline:0;box-shadow:none}
    .local-llm-service-node-panel .llmn-number-scrub{position:absolute;z-index:3;inset:0;cursor:ew-resize;touch-action:pan-y}
    .local-llm-service-node-panel .llmn-number-editing .llmn-number-scrub{display:none}
    .local-llm-service-node-panel .llmn-number-editing{border-color:rgba(120,170,255,.9);box-shadow:0 0 0 1px rgba(120,170,255,.28)}
    .local-llm-service-node-panel .llmn-number-scrubbing{cursor:ew-resize;box-shadow:inset 0 0 0 1px rgba(120,170,255,.22)}
    .local-llm-service-node-panel .llmn-number-scrubbing .llmn-number-fill{transition:none}
    .local-llm-service-node-panel .llmn-number-disabled{opacity:.48}
    .local-llm-service-node-panel .llmn-number-disabled .llmn-number-scrub{cursor:default}
    .local-llm-service-node-panel .llmn-field{display:grid;grid-template-columns:minmax(118px,38%) 1fr;gap:8px;align-items:center;width:100%}
    .local-llm-service-node-panel .llmn-field-label{font-weight:600;opacity:.9;min-width:0}
    .local-llm-service-node-panel .llmn-selector-actions{display:grid;grid-template-columns:42px minmax(0,1fr) 42px;gap:6px;align-items:center;width:100%}
    .local-llm-service-node-panel .llmn-button{min-height:38px;min-width:42px;padding:6px 10px;border:1px solid rgba(127,127,127,.42);border-radius:6px;background:rgba(127,127,127,.18);color:inherit;font:inherit;font-weight:600;cursor:pointer;touch-action:manipulation}
    .local-llm-service-node-panel .llmn-button:hover:not(:disabled){filter:brightness(1.12)}
    .local-llm-service-node-panel .llmn-button:disabled{opacity:.42;cursor:default}
    .local-llm-service-node-panel .llmn-icon-button{width:42px;min-width:42px;padding:0;display:inline-flex;align-items:center;justify-content:center}
    .local-llm-service-node-panel .llmn-icon-button svg{width:19px;height:19px;pointer-events:none;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
    .local-llm-service-node-panel .llmn-icon-button.llmn-delete:not(:disabled){color:#ffb4b4}
    .local-llm-service-node-panel .llmn-section{font-weight:700;opacity:.86;margin:3px 0 0;padding-top:3px;border-top:1px solid rgba(127,127,127,.22)}
    .local-llm-service-node-panel .llmn-seed-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:7px}
    .local-llm-service-node-panel .llmn-mini-label{display:block;font-size:11px;font-weight:600;opacity:.75;margin:0 0 3px 2px}
    .local-llm-service-node-panel .llmn-badge{font-size:11px;opacity:.75;text-align:right}
    .local-llm-service-node-panel .llmn-disabled{opacity:.48}
    @media(max-width:520px){.local-llm-service-node-panel .llmn-field{grid-template-columns:1fr;gap:4px}.local-llm-service-node-panel .llmn-selector-actions{grid-template-columns:44px minmax(0,1fr) 44px}.local-llm-service-node-panel .llmn-icon-button{width:44px;min-width:44px}.local-llm-service-node-panel .llmn-select,.local-llm-service-node-panel .llmn-input{min-height:42px}.local-llm-service-node-panel .llmn-number{height:44px}.local-llm-service-node-panel .llmn-number-step{flex-basis:44px;width:44px}.local-llm-service-node-panel .llmn-textarea{min-height:120px}.local-llm-service-node-panel .llmn-seed-grid{grid-template-columns:1fr}}
  `;
  root.appendChild(css);
  const content=servicePanelElement("div","llmn-content");
  Object.assign(content.style,{width:"100%",minWidth:"0",maxWidth:"100%",boxSizing:"border-box",display:"flex",flexDirection:"column",gap:"7px",height:"auto",minHeight:"0",flex:"0 0 auto"});
  root.appendChild(content);
  const controls={};

  if(nodeWidget(node,"system_prompt_preset")){
    addServicePanelPresetField(node,content,controls,"system_prompts","system_prompt_preset","System Prompt Preset");
    addServicePanelTextarea(node,content,controls,"system_prompt","System Prompt");
    addServicePanelPresetField(node,content,controls,"prompts","prompt_preset","Prompt Preset");
    addServicePanelTextarea(node,content,controls,"prompt","Prompt");
  }

  addServicePanelPresetField(node,content,controls,"sampler","sampling_mode","Sampler Preset");
  const samplerTitle=servicePanelElement("div","llmn-section"); samplerTitle.textContent="Generation"; content.appendChild(samplerTitle);
  for(const [name,label] of SERVICE_NUMERIC_FIELDS.slice(0,8)) addServicePanelNumberField(node,content,controls,name,label);
  const visionTitle=servicePanelElement("div","llmn-section"); visionTitle.textContent="Vision"; content.appendChild(visionTitle);
  for(const [name,label] of SERVICE_NUMERIC_FIELDS.slice(8)) addServicePanelNumberField(node,content,controls,name,label);

  const seedTitle=servicePanelElement("div","llmn-section"); seedTitle.textContent="Seed"; content.appendChild(seedTitle);
  const seedWrap=servicePanelElement("div"); controls.seedWrap=seedWrap;
  controls.settingsBadge=servicePanelElement("div","llmn-badge"); seedWrap.appendChild(controls.settingsBadge);
  const seedGrid=servicePanelElement("div","llmn-seed-grid");
  const seedBox=servicePanelElement("div"); const seedLabel=servicePanelElement("label","llmn-mini-label"); seedLabel.textContent="Seed";
  const seedNumber=createServicePanelNumberControl(node,"seed","Seed");
  if(seedNumber){controls.seed=seedNumber.input;seedBox.append(seedLabel,seedNumber.shell);}else seedBox.appendChild(seedLabel);
  const ctlBox=servicePanelElement("div"); const ctlLabel=servicePanelElement("label","llmn-mini-label"); ctlLabel.textContent="Control After Generate";
  controls.seedControl=servicePanelElement("select","llmn-select"); controls.seedControl.addEventListener("change",()=>{const w=serviceSeedControlWidget(node);if(!w)return;w.value=controls.seedControl.value;try{w.callback?.(w.value);}catch(_){}persistServiceNodeUIState(node);markServiceNodeChanged(node);scheduleServiceNodePanelSync(node);});
  ctlBox.append(ctlLabel,controls.seedControl); seedGrid.append(seedBox,ctlBox); seedWrap.appendChild(seedGrid); content.appendChild(seedWrap);

  node.__localLLMPanelControls=controls;
  node.__localLLMPanelRoot=root;
  node.__localLLMPanelContent=content;
  applyServiceTextareaHeights(node,node.__localLLMTextareaHeights || node?.properties?.[SERVICE_UI_STATE_KEY]?.textareaHeights,{schedule:false});

  const panelHeight=()=>measureServicePanelHeight(node);
  const domWidget=node.addDOMWidget(SERVICE_PANEL_WIDGET_NAME,SERVICE_PANEL_WIDGET_NAME,root,{serialize:false,hideOnZoom:false,margin:4,getValue:()=>"",setValue:()=>{},getMinHeight:panelHeight,getHeight:panelHeight,afterResize:()=>scheduleServiceNodePanelHeight(node)});
  pinServiceDomWidgetFullWidth(domWidget);
  clampServicePanelToNode(node);
  domWidget.serialize=false; domWidget.options ||= {}; domWidget.options.serialize=false;
  node.__localLLMDomPanelWidget=domWidget;
  hideNativeWidgetsForServicePanel(node);
  syncServiceNodeDomPanel(node);

  if(typeof ResizeObserver!=="undefined"){
    const ro=new ResizeObserver(()=>scheduleServiceNodePanelHeight(node)); ro.observe(content); node.__localLLMPanelResizeObserver=ro;
    const tro=new ResizeObserver(()=>captureServiceTextareaHeights(node));
    if(controls.system_prompt) tro.observe(controls.system_prompt);
    if(controls.prompt) tro.observe(controls.prompt);
    node.__localLLMTextareaResizeObserver=tro;
  }
  scheduleServiceNodePanelHeight(node);
  return domWidget;
}

function scheduleServiceNodePanelSync(node) {
  if(!node?.__localLLMPanelControls || node.__localLLMPanelSyncPending) return;
  node.__localLLMPanelSyncPending=true;
  queueMicrotask(()=>{node.__localLLMPanelSyncPending=false;syncServiceNodeDomPanel(node);});
}

function syncServiceNodeDomPanel(node) {
  const c=node?.__localLLMPanelControls;
  if(!c) return;
  const syncText=(name)=>{if(c[name]){const v=String(nodeWidget(node,name)?.value ?? "");if(c[name].value!==v)c[name].value=v;}};
  syncText("system_prompt"); syncText("prompt");

  const presetSpecs=[
    ["system_prompt_preset","system_prompts"],
    ["prompt_preset","prompts"],
    ["sampling_mode","sampler"],
  ];
  for(const [name,kind] of presetSpecs){
    const w=nodeWidget(node,name), select=c[`${name}_select`];
    if(!w || !select) continue;
    setServicePanelSelectOptions(select,servicePanelComboValues(w),w.value);
    const current=String(w.value || "");
    const save=c[`${name}_save`], del=c[`${name}_delete`];
    if(save) save.disabled=current!=="Custom";
    if(del){
      const catalog=node.__localLLMPresetCatalog?.[kind] || {};
      const deletable=catalog.deletable_names || Object.keys(catalog.presets || {});
      del.disabled=!deletable.includes(current);
    }
  }
  for(const [name] of SERVICE_NUMERIC_FIELDS){
    const input=c[name], w=nodeWidget(node,name); if(!input || !w) continue;
    syncServicePanelNumberControl(input,w,name);
  }
  if(c.seed){const w=nodeWidget(node,"seed");if(w)syncServicePanelNumberControl(c.seed,w,"seed");}
  const seedCtl=serviceSeedControlWidget(node);
  if(c.seedControl){
    const vals=servicePanelComboValues(seedCtl); setServicePanelSelectOptions(c.seedControl,vals.length?vals:["fixed","increment","decrement","randomize"],seedCtl?.value ?? "fixed");
  }

  const settingsConnected=serviceSettingsConnected(node);
  if(c.settingsBadge) c.settingsBadge.textContent=settingsConnected?"Local LLM Settings connected — request settings below are overridden":"";
  const requestControlNames=[...SERVICE_NUMERIC_FIELDS.map(([n])=>n),"seed"];
  for(const name of requestControlNames) if(c[name]) setServicePanelNumberDisabled(c[name],settingsConnected);
  if(c.seedControl) c.seedControl.disabled=settingsConnected;
  c.seedWrap?.classList.toggle("llmn-disabled",settingsConnected);
  const samplerSelect=c.sampling_mode_select;
  if(samplerSelect) samplerSelect.disabled=settingsConnected;
  if(c.sampling_mode_save) c.sampling_mode_save.disabled=settingsConnected || String(nodeWidget(node,"sampling_mode")?.value||"")!=="Custom";
  if(c.sampling_mode_delete) {
    const samplerName=String(nodeWidget(node,"sampling_mode")?.value || "");
    const samplerCatalog=node.__localLLMPresetCatalog?.sampler || {};
    const samplerDeletable=samplerCatalog.deletable_names || Object.keys(samplerCatalog.presets || {});
    c.sampling_mode_delete.disabled=settingsConnected || !samplerDeletable.includes(samplerName);
  }

  const visionUnsupported=node.__localLLMCapabilities?.vision===false;
  for(const name of ["vision_max_images","vision_max_frames","vision_max_edge"]){
    const input=c[name]; if(!input) continue;
    const field=input.closest?.(".llmn-field"); if(field) field.style.display=visionUnsupported?"none":"";
  }
  scheduleServiceNodePanelHeight(node);
}

function wrapServiceNodePanelSyncCallbacks(node) {
  if (!node || node.__localLLMDomSyncWrapped) return;
  node.__localLLMDomSyncWrapped=true;
  const names=["system_prompt_preset","system_prompt","prompt_preset","prompt","sampling_mode",...REQUEST_PRESET_FIELDS,"vision_max_images","vision_max_frames","vision_max_edge","seed"];
  const widgets=names.map((n)=>nodeWidget(node,n)).filter(Boolean);
  const control=serviceSeedControlWidget(node); if(control) widgets.push(control);
  for(const w of widgets){
    if(w.__localLLMDomSyncCallbackWrapped) continue;
    w.__localLLMDomSyncCallbackWrapped=true;
    const original=w.callback;
    w.callback=function(...args){const result=original?.apply(this,args);scheduleServiceNodePanelSync(node);return result;};
  }
}

async function initializeServiceGenerateNode(node, loaded=false, serializedSize=null) {
  serviceGenerateNodes.add(node);
  const preservedSize=loaded ? (serializedSize || copyNodeSize(node)) : null;
  installGeneratePersistence(node);
  installServiceVisionConnectionGuard(node);
  if(loaded) restoreGenerateWidgetValues(node);
  restoreServiceNodeUIState(node);
  installPresetCallbacks(node);
  const panel=createServiceNodeDomPanel(node);
  if(panel){hideNativeWidgetsForServicePanel(node);wrapServiceNodePanelSyncCallbacks(node);}
  syncServiceGenerateCapabilities(node,status?.capabilities);
  try{
    const data=await fetchNodePresetCatalog();
    node.__localLLMPresetCatalog=data;
    installCatalogChoices(node,data);
    const sampler=nodeWidget(node,"sampling_mode");
    const systemSel=nodeWidget(node,"system_prompt_preset");
    const promptSel=nodeWidget(node,"prompt_preset");
    if(sampler && sampler.value!=="Custom") await applyPresetSelection(node,"sampler",sampler.value,data);
    if(systemSel && systemSel.value!=="Custom") await applyPresetSelection(node,"system_prompts",systemSel.value,data);
    if(promptSel && promptSel.value!=="Custom") await applyPresetSelection(node,"prompts",promptSel.value,data);
  }catch(e){console.warn("[Local LLM] node preset initialization failed",e);}
  scheduleServiceNodePanelSync(node);
  if(preservedSize) restoreLoadedServiceWidthAndAutosize(node,preservedSize);
  else if(!loaded){
    try{const computed=node.computeSize?.();if(computed?.length>=2)node.setSize?.([Math.max(420,computed[0]),computed[1]]);}catch(_){}
    scheduleServiceNodePanelHeight(node);
  }
}


async function refreshAllDefaultServiceGenerateNodes() {
  let data = null;
  try { data = await fetchNodePresetCatalog(); } catch (_) { return; }
  for (const node of [...serviceGenerateNodes]) {
    if (!node?.widgets) { serviceGenerateNodes.delete(node); continue; }
    installCatalogChoices(node, data);
    const sampler = nodeWidget(node, "sampling_mode");
    if (sampler?.value === "Default") applySamplerSettings(node, data?.sampler?.default || {});
    syncServiceGenerateCapabilities(node, status?.capabilities);
  }
}

app.registerExtension({
  name: EXT,
  beforeConfigureGraph() {
    serviceGenerateGraphConfigureDepth++;
  },
  afterConfigureGraph() {
    serviceGenerateGraphConfigureDepth = Math.max(0, serviceGenerateGraphConfigureDepth - 1);
  },
  nodeCreated(node) {
    if (node?.comfyClass === "LocalLLMServiceGenerate" || node?.comfyClass === "LocalLLMSettings") {
      // ComfyUI also calls nodeCreated while loading/refreshing/switching workflow
      // tabs. Do absolutely no new-node initialization in that path; otherwise the
      // temporary pre-deserialization size can be auto-fitted and overwrite the
      // workflow's saved geometry before loadedGraphNode runs.
      if (serviceGenerateGraphConfigureDepth > 0) return;
      setTimeout(()=>initializeServiceGenerateNode(node, false), 0);
    }
  },
  loadedGraphNode(node) {
    if (node?.comfyClass === "LocalLLMServiceGenerate" || node?.comfyClass === "LocalLLMSettings") {
      // Capture the serialized geometry synchronously, before any async preset
      // catalog work or canvas-only widget insertion can trigger host auto-layout.
      const serializedSize = copyNodeSize(node);
      setTimeout(()=>initializeServiceGenerateNode(node, true, serializedSize), 0);
    }
  },
  async setup() {
    injectCSS();
    try { config = await fetchJSON("/local_llm_server/config"); } catch (_) {}
    try { await refreshStatus(); } catch (_) {}
    ensureFloatingStatus();
    app.extensionManager.registerSidebarTab({
      id: SIDEBAR_ID,
      title: "LLM",
      label: "LLM",
      tooltip: "Local LLM Server",
      iconBadge: () => badgeText(),
      type: "custom",
      render: renderSidebarLauncher,
    });
    // Treat the sidebar icon as a modal-only launcher. ComfyUI versions differ
    // on whether sidebar activation happens on pointerdown or click, so intercept
    // both in the capture phase. stopImmediatePropagation() prevents a native
    // sidebar-toggle handler on the same event from also opening the side panel.
    const interceptLLMLauncher = (event, open=false) => {
      if (allowNativeSidebarToggle) return;
      const launcher = event.target?.closest?.('[data-local-llm-launcher="1"]');
      if (!launcher) return;
      event.preventDefault();
      event.stopImmediatePropagation?.();
      event.stopPropagation();
      if (open) openModal();
    };
    document.addEventListener("pointerdown", (event) => interceptLLMLauncher(event, true), true);
    document.addEventListener("click", (event) => interceptLLMLauncher(event, false), true);
    try {
      api.addEventListener("local_llm_server_status", (event) => {
        status = event?.detail || event || status;
        updateLiveUI();
      });
    } catch (_) {}
    statusTimer=setInterval(refreshStatus,1500);
    const observer=new MutationObserver(()=>decorateSidebar());
    observer.observe(document.body,{childList:true,subtree:true});
    window.addEventListener("resize",()=>{ if(floatingStatus?.isConnected){clampFloating(floatingStatus);saveFloatingPosition(floatingStatus);} });
    decorateSidebar();
    ensureFloatingStatus();
  },
});
