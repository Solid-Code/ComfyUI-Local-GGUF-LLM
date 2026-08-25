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
    .llm-actions{display:flex;gap:8px;flex-wrap:wrap}.llm-btn{border:1px solid #555;background:#343434;color:#eee;border-radius:6px;padding:8px 13px;cursor:pointer}.llm-btn:hover{background:#404040}.llm-btn.primary{background:#246b3e;border-color:#328954}.llm-btn.danger{background:#6b2b2b;border-color:#8e3b3b}.llm-btn.warn{background:#6b5524;border-color:#8e7132}.llm-btn:disabled{opacity:.45;cursor:not-allowed}.llm-footer{display:flex;gap:8px;justify-content:flex-end;padding:11px 16px;border-top:1px solid #444;background:#1d1d1d}.llm-muted{color:#999;font-size:12px}.llm-error{color:#ff8585;white-space:pre-wrap}.llm-success{color:#58dc87}.llm-warn{color:#f2c25b}.llm-log{background:#111;border:1px solid #444;border-radius:7px;padding:10px;font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:pre-wrap;min-height:360px;max-height:520px;overflow:auto}.llm-api-row{display:flex;gap:8px}.llm-api-row input{flex:1}.llm-sidebar-launch{padding:12px}.llm-sidebar-launch .llm-btn{width:100%}.llm-note{padding:9px 11px;border-radius:6px;background:#202b24;border:1px solid #36523d;font-size:12px;margin:10px 0}
    /* Keep the floating status on the workflow/canvas layer. ComfyUI sidebars,
       menus, dialogs, and popovers use higher UI layers and therefore naturally
       render above it. */
    #local-llm-floating-status{position:fixed;z-index:2;width:max-content;min-width:300px;max-width:min(460px,calc(100vw - 16px));box-sizing:border-box;padding:8px 10px;border:1px solid var(--border-color,#454545);border-radius:8px;background:color-mix(in srgb,var(--comfy-menu-bg,#202020) 94%,transparent);color:var(--fg-color,#ddd);box-shadow:0 5px 18px rgba(0,0,0,.34);font-family:Inter,system-ui,sans-serif;user-select:none;cursor:grab;backdrop-filter:blur(5px);-webkit-backdrop-filter:blur(5px)}
    #local-llm-floating-status.dragging{cursor:grabbing}.llm-float-top{display:grid;grid-template-columns:21px minmax(0,1fr) auto;align-items:center;column-gap:7px;min-width:0}.llm-float-robot{display:flex;width:21px;height:21px;flex:0 0 21px;color:#888;grid-column:1;grid-row:1}.llm-float-robot svg{width:21px;height:21px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.llm-float-name{font-size:12px;font-weight:650;white-space:nowrap;grid-column:2;grid-row:1;min-width:0}.llm-float-speed{margin-left:auto;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--fg-color,#ddd);grid-column:3;grid-row:1}.llm-float-sub,.llm-float-last{margin:3px 0 0 0;font-size:10.5px;line-height:1.3;color:var(--descrip-text,#999);white-space:normal;overflow:visible;text-overflow:clip;max-width:none;overflow-wrap:break-word}.llm-float-sub{margin-top:4px}.llm-float-last{font-variant-numeric:tabular-nums;opacity:.92}
    #local-llm-floating-status.processing .llm-float-robot{color:#55ef8b;filter:drop-shadow(0 0 6px rgba(85,239,139,.75));animation:localLLMPulse .8s ease-in-out infinite alternate}#local-llm-floating-status.ready .llm-float-robot{color:#38d26f;filter:drop-shadow(0 0 4px rgba(56,210,111,.5))}#local-llm-floating-status.generating .llm-float-robot{color:#55ef8b;filter:drop-shadow(0 0 6px rgba(85,239,139,.75));animation:localLLMPulse .9s ease-in-out infinite alternate}#local-llm-floating-status.loading .llm-float-robot,#local-llm-floating-status.reloading .llm-float-robot{color:#f3b83f;animation:localLLMPulse .8s ease-in-out infinite alternate}#local-llm-floating-status.yielded .llm-float-robot{color:#65a9ff;filter:drop-shadow(0 0 4px rgba(101,169,255,.45))}#local-llm-floating-status.error .llm-float-robot{color:#ff5b5b;filter:drop-shadow(0 0 4px rgba(255,91,91,.5))}
  `;
  document.head.appendChild(style);
}

function stateClass(s) {
  if (s === "ready") return "ready";
  if (s === "yielded") return "yielded";
  if (s === "generating") return "generating";
  if (s === "processing") return "processing";
  if (s === "loading" || s === "reloading" || s === "stopping") return "loading";
  if (s === "error") return "error";
  return "stopped";
}

// Sidebar/menu robot should pulse only while real LLM work is active.
// In particular, do not map "stopping" to the animated loading state.
function sidebarStateClass(s) {
  if (s === "loading" || s === "reloading") return "loading";
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

async function refreshStatus() {
  try {
    status = await fetchJSON("/local_llm_server/status");
    updateLiveUI();
  } catch (_) {}
}

async function loadData() {
  const [c, cat, st] = await Promise.all([
    fetchJSON("/local_llm_server/config"),
    fetchJSON("/local_llm_server/catalog"),
    fetchJSON("/local_llm_server/status"),
  ]);
  config = c; catalog = cat; status = st;
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
        <button class="llm-tab active" data-tab="server">Server</button><button class="llm-tab" data-tab="model">Model</button><button class="llm-tab" data-tab="memory">Memory</button><button class="llm-tab" data-tab="api">API</button><button class="llm-tab" data-tab="logs">Logs</button>
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
            <button class="llm-btn primary" data-action="start">Start</button><button class="llm-btn danger" data-action="stop">Stop / Unload</button>
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
}

function runtimeNote() {
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
  updateActionButtons();
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
  const busy = ["loading","reloading","processing","generating","stopping"].includes(status.state);
  const loaded = !!status.model_loaded;
  const start = modal.querySelector('[data-action="start"]');
  const stop = modal.querySelector('[data-action="stop"]');
  const save = modal.querySelector('[data-action="save"]');
  const reload = modal.querySelector('[data-action="reload"]');
  if (start) {
    start.disabled = busy || loaded || dirty || saving;
    start.title = dirty ? "Save changes before starting" : (loaded ? "Model is already running" : "Start the saved server configuration");
  }
  if (stop) {
    stop.disabled = busy || !loaded;
    stop.title = loaded ? "Stop and unload the model" : "No model is running";
  }
  if (save) {
    save.disabled = saving || !dirty;
    save.title = dirty ? "Save configuration" : "No unsaved changes";
  }
  if (reload) {
    reload.disabled = saving || dirty || busy || !loaded;
    reload.title = !loaded ? "Reload is available only while a model is running" : (dirty ? "Save changes before reloading" : (busy ? "Wait for the current operation to finish" : "Reload the running model using saved settings"));
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
  });
  modal.querySelector("#llm-model-preset").addEventListener("change",async e=>{await applyPreset("model",e.target.value);setDirty(true);});
  modal.querySelector("#llm-memory-preset").addEventListener("change",async e=>{
    await applyPreset("memory",e.target.value);
    updateContextChoices(currentModelInfo?.context_sizes || catalog.context_sizes || [], currentModelInfo?.native_context);
    setDirty(true);
  });
  const modelOwned=new Set(["thinking_mode","reasoning_effort","preserve_thinking","temperature","top_p","top_k","min_p","repeat_penalty","presence_penalty","frequency_penalty"]);
  const memoryOwned=new Set(["context_size","kv_cache_k","kv_cache_v","kv_cache_location","gpu_layers","flash_attention","prompt_batch_size","memory_batch_size","use_mmap","use_mlock","main_gpu","split_mode","tensor_split"]);
  for(const el of modal.querySelectorAll(".llm-config[data-key]")) el.addEventListener("change",()=>{
    const k=el.dataset.key;
    if(modelOwned.has(k)){const s=modal.querySelector("#llm-model-preset");if(s&&s.value!=="Custom")s.value="Custom";}
    if(memoryOwned.has(k)){const s=modal.querySelector("#llm-memory-preset");if(s&&s.value!=="Custom")s.value="Custom";}
    if(k==="show_status_indicator"){config.show_status_indicator=!!el.checked;ensureFloatingStatus(el.checked);}
    setDirty(true);
  });
  for(const b of modal.querySelectorAll("[data-action]")) b.onclick=async()=>{
    const a=b.dataset.action;
    if(["start","stop","reload"].includes(a)) return doAction(a);
    if(a==="save") return saveConfig();
    if(a==="refresh-logs") return refreshLogs();
    if(a==="copy-api"){await navigator.clipboard?.writeText(modal.querySelector("#llm-api-base").value);return;}
    if(a==="toggle-key"){const i=modal.querySelector("#llm-api-key");i.type=i.type==="password"?"text":"password";b.textContent=i.type==="password"?"Show":"Hide";return;}
    if(a==="regen-key"){const wasDirty=dirty;const d=await fetchJSON("/local_llm_server/api_key/regenerate",{method:"POST"});config.api_key=d.api_key;setConfigField("api_key",d.api_key);setDirty(wasDirty);return;}
  };
}

function closeModal(){ if(modal){modal.remove();modal=null;} if(logsTimer){clearInterval(logsTimer);logsTimer=null;} }

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
    redrawNode(node);
    refreshAllDefaultServiceGenerateNodes();
    toast("success", `Local LLM ${meta.label} Preset`, `Saved ${selector.value}`);
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
    if (widgetName === "sampling_mode") w.label = "sampler preset";
    else if (widgetName === "system_prompt_preset") w.label = "system prompt preset";
    else w.label = "prompt preset";
    const old = w.callback;
    w.callback = function(value, ...args) {
      const result = old?.call(this, value, ...args);
      if (!node.__localLLMApplyingPreset) {
        applyPresetSelection(node, kind, value).catch((e)=>console.warn("[Local LLM] preset load failed", e));
      }
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
        if (p && p.value !== "Custom") { p.value = "Custom"; redrawNode(node); }
      }
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
        if (p && p.value !== "Custom") { p.value = "Custom"; redrawNode(node); }
      }
      return result;
    };
  }
}

const GENERATE_SNAPSHOT_KEY = "local_llm_service_generate_values_v082";

function collectGenerateWidgetValues(node) {
  const out = {};
  for (const w of node?.widgets || []) {
    if (!w?.name || w.__localLLMPresetSaveButton) continue;
    if (w.type === "button") continue;
    try { out[w.name] = w.value; } catch (_) {}
  }
  return out;
}

function restoreGenerateWidgetValues(node) {
  const saved = node?.properties?.[GENERATE_SNAPSHOT_KEY];
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

function moveWidgetDirectlyAfter(node, widget, selectorName) {
  const widgets = node?.widgets;
  const anchor = nodeWidget(node, selectorName);
  if (!widgets || !widget || !anchor) return;
  const current = widgets.indexOf(widget);
  if (current >= 0) widgets.splice(current, 1);
  const anchorIndex = widgets.indexOf(anchor);
  widgets.splice(anchorIndex + 1, 0, widget);
}

function installPresetSaveButtons(node, fitNewNode=false, preservedSize=null) {
  if (node.__localLLMPresetSaveButtons) {
    if (preservedSize) restoreNodeSize(node, preservedSize);
    return;
  }
  node.__localLLMPresetSaveButtons = true;
  const specs = [
    ["system_prompts", "save system prompt preset", "system_prompt_preset"],
    ["prompts", "save prompt preset", "prompt_preset"],
    ["sampler", "save sampler preset", "sampling_mode"],
  ];
  for (const [kind,label,selectorName] of specs) {
    const b = node.addWidget?.("button", label, null, ()=>saveNodePreset(node, kind), {serialize:false});
    if (!b) continue;
    b.__localLLMPresetSaveButton = true;
    b.serialize = false;
    setNodeWidgetOption(b, "serialize", false);
    setNodeWidgetOption(b, "canvasOnly", true);
    b.label = label;
    moveWidgetDirectlyAfter(node, b, selectorName);
  }
  if (preservedSize) restoreNodeSize(node, preservedSize);
  else if (fitNewNode) fitNewServiceGenerateNodeOnce(node);
  redrawNode(node);
}
async function initializeServiceGenerateNode(node, loaded=false, serializedSize=null) {
  serviceGenerateNodes.add(node);
  const preservedSize = loaded ? (serializedSize || copyNodeSize(node)) : null;
  installGeneratePersistence(node);
  installServiceVisionConnectionGuard(node);
  if (loaded) restoreGenerateWidgetValues(node);
  installPresetCallbacks(node);
  installPresetSaveButtons(node, !loaded, preservedSize);
  syncServiceGenerateCapabilities(node, status?.capabilities);
  try {
    const data = await fetchNodePresetCatalog();
    installCatalogChoices(node, data);
    const sampler = nodeWidget(node, "sampling_mode");
    const systemSel = nodeWidget(node, "system_prompt_preset");
    const promptSel = nodeWidget(node, "prompt_preset");
      // Workflow-saved Custom content is authoritative. Default and named presets
    // intentionally resolve to their current live values/files on load.
    if (sampler && sampler.value !== "Custom") await applyPresetSelection(node, "sampler", sampler.value, data);
    if (systemSel && systemSel.value !== "Custom") await applyPresetSelection(node, "system_prompts", systemSel.value, data);
    if (promptSel && promptSel.value !== "Custom") await applyPresetSelection(node, "prompts", promptSel.value, data);
  } catch (e) { console.warn("[Local LLM] node preset initialization failed", e); }
  if (preservedSize) preserveLoadedNodeSizeAcrossFrames(node, preservedSize);
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
    if (node?.comfyClass === "LocalLLMServiceGenerate") {
      // ComfyUI also calls nodeCreated while loading/refreshing/switching workflow
      // tabs. Do absolutely no new-node initialization in that path; otherwise the
      // temporary pre-deserialization size can be auto-fitted and overwrite the
      // workflow's saved geometry before loadedGraphNode runs.
      if (serviceGenerateGraphConfigureDepth > 0) return;
      setTimeout(()=>initializeServiceGenerateNode(node, false), 0);
    }
  },
  loadedGraphNode(node) {
    if (node?.comfyClass === "LocalLLMServiceGenerate") {
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
