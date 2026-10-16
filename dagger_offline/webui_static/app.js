// HG-RLT console front end: one event stream from the server, plain DOM.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "width") node.style.width = v; // CSSOM: the CSP blocks style attributes
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== undefined && c !== null) node.append(c instanceof Node ? c : String(c));
  return node;
};
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const clock = (t) => (t ? new Date(t * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "");

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json", "X-HGRLT": "1" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---- state -----------------------------------------------------------------------
const state = { jobs: new Map(), meta: { failure_types: [], phrase: "我在机械臂旁" }, status: null, models: [] };
const terms = new Map(); // job id -> Term

class Term {
  // Enough of a terminal for line-oriented output: \n, \r (rewrite line), \b.
  constructor() { this.lines = [""]; this.col = 0; }
  write(text) {
    for (const ch of text) {
      if (ch === "\n") { this.lines.push(""); this.col = 0; }
      else if (ch === "\r") this.col = 0;
      else if (ch === "\b") this.col = Math.max(0, this.col - 1);
      else if (ch >= " " || ch === "\t") {
        const line = this.lines[this.lines.length - 1];
        this.lines[this.lines.length - 1] = line.slice(0, this.col) + ch + line.slice(this.col + 1);
        this.col += 1;
      }
    }
    if (this.lines.length > 3000) this.lines.splice(0, this.lines.length - 2500);
  }
  text() { return this.lines.join("\n"); }
}

function term(id) {
  if (!terms.has(id)) terms.set(id, new Term());
  return terms.get(id);
}

const pending = new Set();
function scheduleRender(what) {
  if (!pending.size) requestAnimationFrame(() => { const all = [...pending]; pending.clear(); for (const f of all) f(); });
  pending.add(what);
}

function paintTerm(pre, job) {
  if (!job) { pre.textContent = ""; return; }
  const stick = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 60;
  pre.textContent = term(job.id).text();
  if (stick) pre.scrollTop = pre.scrollHeight;
}

// latest job of a kind: the running one, else the most recent
function jobOf(kind) {
  const all = [...state.jobs.values()].filter((j) => j.kind === kind);
  return all.find((j) => j.running) || all.sort((a, b) => b.started - a.started)[0] || null;
}

// ---- tabs ------------------------------------------------------------------------
for (const button of $$(".tabs button")) button.addEventListener("click", () => showTab(button.dataset.tab));
function activeTab() { return $(".tabs button.active").dataset.tab; }
function showTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  try { localStorage.setItem("hgrlt-tab", name); } catch { /* storage blocked */ }
  if (["models", "hw", "status"].includes(name)) refreshStatus();
  if (name === "results") loadResults();
}

// ---- events ----------------------------------------------------------------------
function onEvent(e) {
  if (e.type === "hello") {
    state.meta = e.meta;
    state.jobs.clear();
    terms.clear();
    for (const job of e.jobs) {
      const { output, ...rest } = job;
      state.jobs.set(job.id, rest);
      term(job.id).write(output || "");
    }
    renderAll();
  } else if (e.type === "job") {
    const known = state.jobs.has(e.job.id);
    state.jobs.set(e.job.id, e.job);
    // the server keeps one ended job per kind
    for (const [id, j] of state.jobs) {
      if (id !== e.job.id && j.kind === e.job.kind && !j.running) { state.jobs.delete(id); terms.delete(id); }
    }
    if (!known) term(e.job.id);
    renderAll();
    if (e.job.kind === "model" || !e.job.running) refreshStatus();
  } else if (e.type === "output") {
    term(e.id).write(e.text);
    const job = state.jobs.get(e.id);
    if (job) scheduleRender(() => paintTerms(job.kind));
  } else if (e.type === "state") {
    const job = state.jobs.get(e.id);
    if (job) { job.state = e.state; scheduleRender(renderSession); }
  }
}

function connect() {
  const source = new EventSource("/api/events");
  source.onopen = () => { $("#conn").textContent = "已连接"; $("#conn").className = "conn ok"; };
  source.onmessage = (m) => onEvent(JSON.parse(m.data));
  source.onerror = () => {
    $("#conn").textContent = "连接断开，重试中"; $("#conn").className = "conn bad";
  };
}

function renderAll() {
  renderSession();
  renderModels();
  renderDaemon();
  paintTerms("session"); paintTerms("model"); paintTerms("daemon");
}

function paintTerms(kind) {
  const id = { session: "#session-term", model: "#model-term", daemon: "#daemon-term" }[kind];
  if (id) paintTerm($(id), jobOf(kind));
}

// ---- session ---------------------------------------------------------------------
const MODE_TEXT = {
  HOLD: ["HOLD", "保持：h 人工接管 / p 交给策略"],
  POLICY_RUNNING: ["POLICY_RUNNING", "策略控制：h 立即接管 / s 保持"],
  INTERVENTION: ["INTERVENTION", "人工接管：p 交还策略 / s 保持"],
  FAULT: ["FAULT", "故障锁定：本条不可用于训练"],
};
const STAGE_TEXT = {
  starting: "准备中：检查配置、连接相机和主臂…",
  await_start: "重置场景后按回车开始本条；q 回车结束本次采集",
  confirm_homing: "确认双臂回零的路径无障碍后按回车（每帧 ≤0.01 rad，30 Hz）",
  homing: "双臂回零中，夹爪张开…",
  recording: "",
  review: "标记本条结果",
  failure_type: "选择失败类型",
  saving: "正在落盘，请勿中断",
  saved: "已保存",
  next: "准备下一条",
  ending: "正在结束采集",
  fault: "故障：查看终端输出，核对机械臂和记录",
  ended: "采集已停止（原因见终端输出）",
  finished: "本次采集结束",
};
const KEY_INFO = {
  p: ["交给策略", "k-policy"], h: ["人工接管", "k-human"], s: ["保持 HOLD", "k-stop"], k: ["结束本条", "k-end"],
  c: ["检查点", ""], x: ["丢弃检查点后", ""], q: ["整条弃用", "k-end"], r: ["交给 RL actor", "k-policy"],
  "?": ["按键帮助", ""], enter: ["回车", ""], "ctrl-c": ["Ctrl-C 中断", "k-end"],
};

// Keys and lines reach the process in the order pressed: one request at a time.
let inputQueue = Promise.resolve();
function queued(path, body) {
  const next = inputQueue.then(() => api(path, body));
  inputQueue = next.catch(() => {});
  return next;
}
function sendKey(key) {
  const job = jobOf("session");
  if (!job || !job.running) return;
  queued("/api/pty/key", { job: job.id, key }).catch((err) => showNotice(err.message, true));
}
function sendLine(kind, text) {
  const job = jobOf(kind);
  if (!job || !job.running) return Promise.resolve();
  return queued("/api/pty/line", { job: job.id, text });
}

function padButtons(job) {
  const s = job.state || {};
  const types = state.meta.failure_types || [];
  const key = (k, label, cls) => ({ kbd: k === "enter" ? "回车" : k, label: label ?? KEY_INFO[k][0], cls: cls ?? KEY_INFO[k][1], run: () => sendKey(k), k });
  switch (s.stage) {
    case "await_start":
      return [
        { kbd: "回车", label: "开始本条", cls: "k-policy", run: () => sendLine("session", ""), k: "enter" },
        { kbd: "q 回车", label: "结束本次采集", cls: "k-end", run: () => sendLine("session", "q"), k: "q" },
      ];
    case "confirm_homing":
      return [
        { kbd: "回车", label: "确认回零", cls: "k-human", run: () => sendLine("session", ""), k: "enter" },
        { kbd: "n 回车", label: "取消（不运动）", cls: "k-stop", run: () => sendLine("session", "n"), k: "n" },
      ];
    case "review":
      return [key("1", "成功", "k-human"), key("2", "失败", "k-end"), key("3", "弃用", "k-stop")];
    case "failure_type":
      return types.map(([, label], i) => key(String(i + 1), label, ""));
    case "recording": {
      const list = [key("p"), key("h"), key("s"), key("k"), key("c"), key("x"), key("q")];
      const rlt = job.argv.includes("--rlt");
      if (rlt) list.splice(2, 0, key("r"));
      list.push(key("?"));
      return list;
    }
    default:
      return [key("s")];
  }
}

function renderSession() {
  const job = jobOf("session");
  const running = Boolean(job && job.running);
  $("#session-form").hidden = running;
  $("#session-view").hidden = !job;
  $("#session-term-card").hidden = !job;
  const chip = $("#mode-chip");
  if (!job) { chip.hidden = true; return; }
  const s = job.state || {};
  $("#session-label").textContent = job.label;
  $("#session-sub").textContent = running
    ? `PID ${job.pid} · 开始于 ${clock(job.started)}${job.stopping ? " · 正在结束…" : ""}`
    : `已结束 ${clock(job.ended)} · 退出码 ${job.code}`;
  $("#btn-session-stop").hidden = !running;
  $("#btn-session-stop").disabled = job.stopping;

  const mode = running && s.stage === "recording" || s.mode === "FAULT" ? s.mode : null;
  const big = $("#mode-big");
  big.className = `mode-big m-${mode || "none"}`;
  big.textContent = "";
  if (mode) {
    const [name, hint] = MODE_TEXT[mode];
    big.append(name + (s.rl && mode === "POLICY_RUNNING" ? " · RL" : ""), el("small", {}, !s.takeover ? hint : s.takeover.phase === "return" ? "接管准备：扳机回位后进入人工接管；s / p 取消" : "接管准备：扳机已断电，两侧各自完全松开再扣到底；s / p 取消"));
  } else {
    big.append(running ? (s.episode || "采集进程") : "已结束", el("small", {}, STAGE_TEXT[s.stage] || s.stage || ""));
  }
  chip.hidden = !running;
  chip.className = `mode-chip m-${mode || "none"}`;
  chip.textContent = mode ? (s.takeover ? "接管准备" : mode) : (STAGE_TEXT[s.stage] ? s.stage : "采集");

  const take = $("#takeover");
  take.hidden = !s.takeover;
  take.textContent = "";
  if (s.takeover) {
    const t = s.takeover;
    take.append(
      el("span", { class: t.left ? "done" : "" }, `左扳机 ${t.left ? "✓ 已确认" : "完全松开→扣到底"}`),
      el("span", { class: t.right ? "done" : "" }, `右扳机 ${t.right ? "✓ 已确认" : "完全松开→扣到底"}`),
      el("span", { class: t.phase === "return" ? "done" : "" }, t.phase === "return" ? "扳机回位中，松开手指" : "等待确认"),
    );
    if (t.warning) take.append(el("span", {}, t.warning));
  }
  $("#stage-text").textContent = mode ? (s.episode || "") : "";
  if (s.notice && running) showNotice(`[${s.notice.label}] ${s.notice.text}`, ["故障", "采集停止"].includes(s.notice.label));
  else if (!running) $("#notice").hidden = true;

  const stats = $("#session-stats");
  stats.textContent = "";
  if (s.stats) {
    for (const [k, label] of [["total", "累计"], ["success", "成功"], ["fail", "失败"], ["discard", "弃用"], ["incomplete", "未完成"]]) {
      stats.append(el("div", {}, el("b", {}, s.stats[k]), el("span", {}, label)));
    }
  }

  const unlocked = $("#unlock").checked;
  const pad = $("#pad");
  pad.textContent = "";
  if (running) {
    for (const b of padButtons(job)) {
      pad.append(el("button", {
        class: b.cls, type: "button", disabled: !unlocked && b.k !== "s",
        onclick: (ev) => { ev.currentTarget.blur(); b.run(); },
      }, el("kbd", {}, b.kbd), el("span", {}, b.label)));
    }
  }
  const all = $("#keys-all");
  all.textContent = "";
  if (running) {
    const keys = ["p", "h", "s", "k", "c", "x", "q", "r", "?", "1", "2", "3", "4", "5", "6", "enter", "ctrl-c"];
    for (const k of keys) {
      const label = KEY_INFO[k]?.[0] ?? (k <= "3" ? `结果 / 类型 ${k}` : `失败类型 ${k}`);
      all.append(el("button", {
        type: "button", disabled: !unlocked && k !== "s",
        onclick: (ev) => {
          ev.currentTarget.blur();
          if (k === "ctrl-c" && !confirm("发送 Ctrl-C：采集按中断处理（本条记为未完成，移入 trash/）。继续？")) return;
          sendKey(k);
        },
      }, el("kbd", {}, k === "enter" ? "⏎" : k === "ctrl-c" ? "^C" : k), label));
    }
  }
}

function showNotice(text, bad = false) {
  const n = $("#notice");
  n.hidden = false;
  n.className = `notice${bad ? " bad" : ""}`;
  n.textContent = text;
}

$("#unlock").addEventListener("change", renderSession);

document.addEventListener("keydown", (e) => {
  if (activeTab() !== "session" || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof Element && e.target.closest("input, select, textarea, dialog")) return;
  const job = jobOf("session");
  if (!job || !job.running) return;
  const k = e.key === "Enter" ? "enter" : e.key.toLowerCase();
  if (!"phscxkqr?123456".includes(k) && k !== "enter") return;
  if (k.length !== 1 && k !== "enter") return;
  e.preventDefault(); // Enter must not also click a focused button
  if (!$("#unlock").checked && k !== "s") {
    showNotice("先勾选“我在机械臂旁”再用键盘控制（s 保持始终可用）");
    return;
  }
  sendKey(k);
});

// session form: plan preview, default directory name, start
const form = $("#session-form");
const fields = form.elements; // not form.name: that is the form's own name attribute
let nameEdited = false;
function defaultName() {
  const model = fields.deployment.value || "baseline";
  return `${state.meta.today || ""}-${model}`;
}
function formParams() {
  const episodes = Number(fields.episodes.value);
  return {
    kind: fields.kind.value, deployment: fields.deployment.value, name: fields.name.value.trim(),
    episodes: Number.isInteger(episodes) ? episodes : -1, command: fields.command.value,
    rlt: fields.rlt.value.trim() || null,
  };
}
let planTimer;
function updatePlan() {
  if (!nameEdited && fields.kind.value === "eval") fields.name.value = defaultName();
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    try {
      const plan = await api("/api/session/plan", formParams());
      $("#session-plan").textContent = plan.argv.map((a) => a.replace(`${state.meta.root}/`, "")).join(" ");
      $("#session-error").textContent = "";
    } catch (err) {
      $("#session-plan").textContent = "";
      $("#session-error").textContent = err.message;
    }
  }, 150);
}
form.addEventListener("input", (e) => {
  if (e.target === fields.name) nameEdited = fields.name.value !== "";
  if (e.target === fields.kind && fields.kind.value === "recordings" && !nameEdited) fields.name.value = "";
  updatePlan();
});
form.addEventListener("change", updatePlan);
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const p = formParams();
  const what = p.kind === "eval" ? "评估" : "采集";
  const phrase = await confirmPhrase(`开始${what}`,
    `${$("#session-plan").textContent}\n\n每条开始前在页面按回车，再确认回零路径；回零和策略执行都会让机械臂运动。确认现场有人能随时断电。`);
  if (!phrase) return;
  try {
    await api("/api/session/start", { ...p, confirm: phrase });
    $("#unlock").checked = true;
    renderSession();
  } catch (err) {
    $("#session-error").textContent = err.message;
  }
});

$("#btn-session-stop").addEventListener("click", async () => {
  if (!confirm("结束采集：按当前阶段发送按键——录制中先 s 保持再 q 弃用本条（文件保留在 trash/），标记结果时记为弃用，等待开始时 q 结束。继续？")) return;
  try { await api("/api/session/stop", {}); } catch (err) { showNotice(err.message, true); }
});

for (const lineForm of $$("form[data-line]")) {
  lineForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("input", lineForm);
    try {
      await sendLine(lineForm.dataset.line, input.value);
      input.value = "";
    } catch (err) {
      alert(err.message);
    }
  });
}

// ---- confirm dialog --------------------------------------------------------------
function confirmPhrase(title, text) {
  const dialog = $("#confirm");
  const phrase = state.meta.phrase;
  $("#confirm-title").textContent = title;
  $("#confirm-text").textContent = text;
  $("#confirm-phrase").textContent = phrase;
  const input = $("#confirm-input");
  input.value = "";
  $("#confirm-ok").disabled = true;
  input.oninput = () => { $("#confirm-ok").disabled = input.value !== phrase; };
  dialog.showModal();
  input.focus();
  return new Promise((resolve) => {
    dialog.onclose = () => resolve(dialog.returnValue === "ok" && input.value === phrase ? input.value : null);
  });
}

// ---- status, models, daemon ------------------------------------------------------
let statusBusy = false;
async function refreshStatus() {
  if (statusBusy) return;
  statusBusy = true;
  try {
    const [status, models] = await Promise.all([api("/api/status"), api("/api/models")]);
    state.status = status;
    state.models = models;
    renderStatus();
    renderModels();
    renderDaemon();
    renderModelOptions();
  } catch (err) {
    $("#models-error").textContent = err.message;
  } finally {
    statusBusy = false;
  }
}
$("#btn-status").addEventListener("click", refreshStatus);
$("#btn-models").addEventListener("click", refreshStatus);
setInterval(() => { if (["models", "hw", "status"].includes(activeTab()) && !document.hidden) refreshStatus(); }, 10000);

function portInfo(port) {
  return state.status?.ports.find((p) => p.port === port);
}
function owner(info) {
  if (!info?.listening) return "";
  return info.pid ? `PID ${info.pid}（${info.user}）` : "其他用户的进程";
}

function renderModelOptions() {
  const select = $("#session-model");
  const current = select.value;
  select.textContent = "";
  for (const m of state.models) {
    if (m.error) continue;
    const info = portInfo(m.port);
    const tag = info?.listening ? "监听中" : "未运行";
    select.append(el("option", { value: m.name }, `${m.name} · ${m.port} · ${tag}${m.rl_token ? " · RL token" : ""}`));
  }
  if ([...select.options].some((o) => o.value === current)) select.value = current;
  updatePlan();
}

function renderModels() {
  const body = $("#models");
  body.textContent = "";
  const job = jobOf("model");
  const mine = job && job.running ? job : null;
  for (const m of state.models) {
    if (m.error) {
      body.append(el("tr", {}, el("td", {}, m.name), el("td", { colspan: 5, class: "error" }, m.error)));
      continue;
    }
    const info = portInfo(m.port);
    let status, action = null;
    if (mine && mine.meta.name === m.name) {
      status = el("span", { class: `chip ${mine.meta.listening ? "ok" : "info"}` },
        mine.stopping ? "正在停止" : mine.meta.listening ? "控制台启动 · 监听中" : "控制台启动 · 加载中");
      action = el("button", { class: "danger small", disabled: mine.stopping, onclick: () => stopModel(mine) }, "停止");
    } else if (info?.listening) {
      status = el("span", { class: "chip warn" }, `其他人在用 · ${owner(info)}`);
    } else {
      status = el("span", { class: "chip" }, "未运行");
      action = el("button", { class: "small", disabled: Boolean(mine), onclick: () => startModel(m.name) }, "启动");
    }
    body.append(el("tr", {},
      el("td", {}, el("b", {}, m.name), m.rl_token ? el("span", { class: "chip info" }, "RL token") : null),
      el("td", { class: "num" }, m.port),
      el("td", { class: "mono" }, m.checkpoint || ""),
      el("td", { class: "mono" }, m.policy_id ? m.policy_id.slice(0, 12) : "–"),
      el("td", {}, status, info?.args && !(mine && mine.meta.name === m.name) ? el("div", { class: "mono muted" }, info.args) : null),
      el("td", {}, action),
    ));
  }
  $("#model-term-card").hidden = !job;
  if (job) $("#model-term-title").textContent = `${job.label} · ${job.running ? `PID ${job.pid}` : `已退出（${job.code}）`}`;
}

async function startModel(name) {
  $("#models-error").textContent = "";
  try { await api("/api/model/start", { name }); } catch (err) { $("#models-error").textContent = err.message; }
  refreshStatus();
}
async function stopModel(job) {
  if (!confirm(`停止 ${job.label}？依次发送 SIGINT / SIGTERM，必要时 SIGKILL。`)) return;
  try { await api("/api/model/stop", { job: job.id }); } catch (err) { $("#models-error").textContent = err.message; }
}

const DAEMON_KIND = {
  hgrlt: ["ok", "HG-RLT 硬件进程"],
  foreign: ["warn", "其他项目的硬件进程（如 RhOSPolicy）；采集前托住双臂执行重启换成本项目的进程"],
  other: ["bad", "端口被其他程序占用"],
  unknown: ["warn", "端口在监听，进程属于其他用户，无法识别"],
  none: ["bad", "未运行"],
};
function renderDaemon() {
  const box = $("#daemon-summary");
  box.textContent = "";
  const d = state.status?.daemon;
  if (d) {
    const [cls, text] = DAEMON_KIND[d.kind] || ["", d.kind];
    box.append(el("div", { class: "k" }, "进程"), el("div", {}, el("span", { class: `chip ${cls}` }, text), d.pid ? ` PID ${d.pid}（${d.user}）` : ""));
    box.append(el("div", { class: "k" }, "端口"), el("div", {}, Object.entries(d.ports).map(([p, on]) => `${p} ${on ? "监听" : "空闲"}`).join(" · ")));
    if (d.args) box.append(el("div", { class: "k" }, "命令"), el("div", { class: "mono" }, d.args));
  }
  const job = jobOf("daemon");
  const session = jobOf("session");
  const busy = Boolean((job && job.running) || (session && session.running));
  $("#btn-daemon-restart").disabled = busy;
  $("#btn-daemon-stop").disabled = busy;
  $("#btn-daemon-check").disabled = Boolean(job && job.running);
  $("#daemon-term-card").hidden = !job;
  if (job) $("#daemon-term-title").textContent = `${job.label} · ${job.running ? `PID ${job.pid}` : `已退出（${job.code}）`}`;
}

$("#btn-daemon-check").addEventListener("click", async (e) => {
  const button = e.currentTarget;
  const out = $("#daemon-check-out");
  button.disabled = true;
  out.hidden = false;
  out.textContent = "检查中…";
  try {
    const r = await api("/api/daemon/check", {});
    const verdict = { ok: "通过：safe_hold、HOLD 确认、CONTRACT OK", foreign: "不是本项目的会话协议（旧进程或其他项目）", not_hold: "不在 HOLD", error: "检查失败" }[r.verdict];
    out.textContent = `[${verdict}] 退出码 ${r.code}\n\n${r.output}`;
  } catch (err) {
    out.textContent = err.message;
  } finally {
    button.disabled = false;
    refreshStatus();
  }
});

async function daemonAction(action) {
  const text = action === "restart"
    ? "先关闭双臂电机（失去保持力），旧进程退出后重新使能，不回零。\n全程用手托住双臂。命令启动后，在下方终端回答它自己的确认（HG-DAEMON-ON-SITE）。"
    : "关闭双臂电机并退出硬件进程，机械臂失去保持力。\n先用手托住双臂。命令启动后，在下方终端输入 off 确认。";
  const phrase = await confirmPhrase(action === "restart" ? "重启硬件进程" : "关闭硬件进程", text);
  if (!phrase) return;
  try {
    await api(`/api/daemon/${action}`, { confirm: phrase });
    setTimeout(() => $("#daemon-term-card input").focus(), 200);
  } catch (err) {
    alert(err.message);
  }
}
$("#btn-daemon-restart").addEventListener("click", () => daemonAction("restart"));
$("#btn-daemon-stop").addEventListener("click", () => daemonAction("stop"));

function renderStatus() {
  const s = state.status;
  if (!s) return;
  $("#status-time").textContent = `更新于 ${clock(s.time)}`;
  const gpu = $("#gpu");
  gpu.textContent = "";
  if (!s.gpu) gpu.append(el("p", { class: "error" }, "读不到 nvidia-smi"));
  else {
    for (const g of s.gpu.gpus) {
      gpu.append(el("div", {}, `GPU ${g.index} · ${g.name} · ${g.used_mib} / ${g.total_mib} MiB`),
        el("div", { class: "bar" }, el("div", { width: `${Math.round((100 * g.used_mib) / g.total_mib)}%` })));
    }
    if (!s.gpu.compute.length) gpu.append(el("p", { class: "hint" }, "没有计算进程"));
    else {
      gpu.append(el("div", { class: "table-wrap" }, el("table", { class: "table" },
        el("thead", {}, el("tr", {}, ["PID", "用户", "MiB", "命令", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, s.gpu.compute.map((a) => el("tr", {},
          el("td", { class: "num" }, a.pid), el("td", {}, a.user || "?"), el("td", { class: "num" }, a.used_mib ?? "?"),
          el("td", { class: "mono" }, a.args || a.name),
          el("td", {}, el("span", { class: `chip ${a.console ? "ok" : "warn"}` }, a.console ? "控制台" : "其他人在用")),
        ))))));
    }
  }
  const ports = $("#ports");
  ports.textContent = "";
  const purpose = (p) => (p.port === 8767 ? "baseline" : p.models.join(", ") || "模型");
  for (const p of s.ports) {
    ports.append(el("tr", {}, el("td", { class: "num" }, p.port), el("td", {}, purpose(p)),
      el("td", {}, p.listening ? el("span", { class: `chip ${p.console_job ? "ok" : "warn"}` }, p.console_job ? "控制台 · 监听" : `监听 · ${owner(p)}`) : el("span", { class: "chip" }, "空闲"))));
  }
  for (const port of Object.keys(s.daemon.ports)) {
    ports.append(el("tr", {}, el("td", { class: "num" }, port), el("td", {}, "硬件进程"),
      el("td", {}, el("span", { class: `chip ${s.daemon.ports[port] ? "ok" : ""}` }, s.daemon.ports[port] ? "监听" : "空闲"))));
  }
  const procs = $("#collect-procs");
  procs.textContent = "";
  const session = jobOf("session");
  if (!s.collect.length) procs.append(el("p", { class: "hint" }, session?.running ? "只有控制台的采集" : "没有采集进程"));
  for (const p of s.collect) procs.append(el("div", {}, el("span", { class: "chip warn" }, "其他终端"), ` PID ${p.pid}（${p.user}） `, el("span", { class: "mono" }, p.args)));
  const v = $("#version");
  v.textContent = "";
  v.append(el("div", { class: "k" }, "安装"), el("div", { class: "mono" }, s.root),
    el("div", { class: "k" }, "版本"), el("div", { class: "mono" }, s.git.describe || "?"),
    el("div", { class: "k" }, "提交"), el("div", { class: "mono" }, s.git.commit || "?"));
}

// ---- results ---------------------------------------------------------------------
const typeLabel = (t) => Object.fromEntries(state.meta.failure_types || [])[t] || (t === "unspecified" ? "未选类型" : t);
const failText = (types) => Object.entries(types).map(([t, n]) => `${typeLabel(t)} ${n}`).join(" · ") || "–";

async function loadResults() {
  let data;
  try { data = await api("/api/results"); } catch (err) { $("#compare").textContent = err.message; return; }
  const byName = new Map(data.dirs.filter((d) => d.kind === "eval").map((d) => [d.name, d]));
  const compare = $("#compare");
  compare.textContent = "";
  if (!Object.keys(data.compare).length) compare.append(el("p", { class: "hint" }, "还没有 data/eval/MMDD-… 目录"));
  for (const [date, names] of Object.entries(data.compare)) {
    compare.append(el("h3", {}, `${date.slice(0, 2)} 月 ${date.slice(2)} 日`));
    compare.append(el("div", { class: "table-wrap" }, el("table", { class: "table" },
      el("thead", {}, el("tr", {}, ["目录", "模型", "已标", "成功率", "自主成功率", "接管 / 条", "失败类型"].map((h) => el("th", {}, h)))),
      el("tbody", {}, names.map((n) => byName.get(n)).map((d) => el("tr", { class: "clickable", onclick: () => loadDetail(d.kind, d.name) },
        el("td", {}, d.name, d.active ? el("span", { class: "chip info" }, "采集中") : null),
        el("td", {}, d.models.join(", ") || "–"),
        el("td", { class: "num" }, d.labelled),
        el("td", { class: "num" }, `${pct(d.success_rate)}（${d.success}）`),
        el("td", { class: "num" }, `${pct(d.autonomous_rate)}（${d.autonomous_success}）`),
        el("td", { class: "num" }, d.takeovers_per_episode === null ? "–" : d.takeovers_per_episode.toFixed(2)),
        el("td", {}, failText(d.failure_types)),
      ))))));
  }
  const body = $("#results");
  body.textContent = "";
  for (const d of data.dirs) {
    body.append(el("tr", { class: "clickable", onclick: () => loadDetail(d.kind, d.name) },
      el("td", {}, `${d.kind}/${d.name}`, d.active ? el("span", { class: "chip info" }, "采集中") : null),
      el("td", {}, d.models.join(", ") || "–"),
      el("td", { class: "num" }, `${d.success} / ${d.labelled}`),
      el("td", { class: "num" }, d.autonomous_success),
      el("td", { class: "num" }, `${d.takeovers}（${d.episodes_with_takeover} 条）`),
      el("td", { class: "num" }, `${d.discarded} / ${d.incomplete}`),
      el("td", {}, failText(d.failure_types)),
    ));
  }
}
$("#btn-results").addEventListener("click", loadResults);

const OUTCOME = { success: ["ok", "成功"], fail: ["bad", "失败"], discard: ["warn", "弃用"], unlabelled: ["", "未标"], in_progress: ["info", "未完成"], unreadable: ["info", "读取中 / 写入中"] };
async function loadDetail(kind, name) {
  let data;
  try { data = await api(`/api/results/detail?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`); } catch (err) { alert(err.message); return; }
  $("#detail-card").hidden = false;
  $("#detail-title").textContent = `${kind}/${name} · ${data.episodes.length} 个文件`;
  const body = $("#detail");
  body.textContent = "";
  for (const r of data.episodes) {
    const [cls, text] = OUTCOME[r.outcome] || ["", r.outcome];
    body.append(el("tr", {},
      el("td", { class: "mono" }, r.name), el("td", {}, r.folder === "trash" ? "trash/" : "episodes/"),
      el("td", {}, el("span", { class: `chip ${cls}` }, text)), el("td", {}, r.failure_type ? typeLabel(r.failure_type) : ""),
      el("td", { class: "num" }, r.takeovers ?? ""), el("td", { class: "num" }, r.rl_segments ?? ""),
      el("td", { class: "num" }, r.duration_s ?? ""), el("td", {}, r.model || ""),
    ));
  }
  $("#detail-card").scrollIntoView({ behavior: "smooth" });
}

// ---- shutdown --------------------------------------------------------------------
$("#btn-shutdown").addEventListener("click", async () => {
  if (!confirm("关闭控制台：先结束采集（录制中的一条记为弃用）、停止控制台启动的模型服务，然后退出。继续？")) return;
  try { await api("/api/shutdown", {}); } catch (err) { alert(err.message); }
});

// ---- start -----------------------------------------------------------------------
let saved = "session";
try { saved = localStorage.getItem("hgrlt-tab") || "session"; } catch { /* storage blocked */ }
if (location.hash.length > 1) saved = location.hash.slice(1); // e.g. /#results
connect();
refreshStatus();
if ($(`#tab-${saved}`)) showTab(saved);
