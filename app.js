const GPT2_URL = "http://127.0.0.1:8000/gpt2";
const BERT_URL = "http://127.0.0.1:8000/bert";

// ---------- GPT-2 ----------

function renderTokens(tokens) {
  gptTokensEl.innerHTML = "";
  tokens.forEach((tok, i) => {
    const el = document.createElement("div");
    el.className = "tok";
    el.textContent = tok;
    el.addEventListener("click", () => setActiveIndex(i));
    gptTokensEl.appendChild(el);
  });
}

function setActiveIndex(i) {
  if (!gptState) return;

  // highlight clicked token
  [...gptTokensEl.children].forEach((el, j) => {
    el.classList.toggle("active", j === i);
  });

  // show top-k distribution that predicted token i
  const dist = gptState.topk[i] || [];
  gptProbsTbody.innerHTML = "";
  dist.forEach(({ token, prob }) => {
    const tr = document.createElement("tr");
    const tdTok = document.createElement("td");
    const tdProb = document.createElement("td");
    tdTok.textContent = token;
    tdProb.textContent = (prob ?? 0).toFixed(6);
    tr.append(tdTok, tdProb);
    gptProbsTbody.appendChild(tr);
  });

  // show actual observed probability at i
  const a = gptState.actual[i];
  if (a) {
    gptActiveNote.textContent = `Valt token: “${a.token}” | P = ${a.prob.toFixed(6)} | log P = ${a.logprob.toFixed(3)}`;
  } else {
    gptActiveNote.textContent = "";
  }
}

function renderSummary(s) {
  if (!s || !isFinite(s.avg_log_prob)) {
    gptSummaryNote.textContent = "";
    return;
  }
  gptSummaryNote.innerHTML = `
    <h3>Sammanfattning över ${s.num_predicted} tokens</h3>
    <ul>
    <li><i>log10 P(text)</i> = ${s.log10_prob.toFixed(3)}</li>
    <li><i>Sannolikhet</i> ≈ ${Math.exp(s.log10_prob).toFixed(30)}</li>
    <!-- <li><i>avg log P/token</i> = ${s.avg_log_prob.toFixed(4)}</li> -->
  `;
}

const $ = (sel) => document.querySelector(sel);

const gptText = $("#gpt-text");
const gptRunBtn = $("#gpt-run");
const gptStatus = $("#gpt-status");
const gptTokensEl = $("#gpt-tokens");
const gptProbsTbody = $("#gpt-probs");
const gptActiveNote = $("#gpt-active-note");
const gptSummaryNote = $("#gpt-summary-note");

let gptState = null;

async function runGpt() {
  // Clean slate
  gptState = null;
  gptStatus && (gptStatus.textContent = "Running…");
  if (gptTokensEl) gptTokensEl.innerHTML = "";
  if (gptProbsTbody) gptProbsTbody.innerHTML = "";
  if (gptActiveNote) gptActiveNote.textContent = "";
  if (gptSummaryNote) gptSummaryNote.textContent = "";

  const text = gptText && gptText.value;

  try {
    const resp = await fetch("/gpt2", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!resp.ok) {
      const msg = await safeReadText(resp);
      throw new Error(`HTTP ${resp.status}: ${msg || "request failed"}`);
    }

    const data = await resp.json();

    // Minimum sanity checks so we don’t propagate garbage
    if (
      !data ||
      !Array.isArray(data.tokens) ||
      !Array.isArray(data.topk) ||
      !Array.isArray(data.actual)
    ) {
      throw new Error("Malformed response from /gpt2");
    }

    // Lock it in
    gptState = data;

    // Optional server note (like truncation)
    if (gptStatus) gptStatus.textContent = data.note || "";

    // Render UI
    renderTokens(data.tokens);
    renderSummary(data.summary);

    // Activate the first token if present
    if (data.tokens.length > 0) {
      setActiveIndex(0);
    } else {
      if (gptStatus) gptStatus.textContent = "Nothing to score (empty input).";
    }
  } catch (err) {
    if (gptStatus) gptStatus.textContent = `Error: ${err.message}`;
    // make sure downstream code won’t try to read null
    gptState = { tokens: [], topk: [], actual: [], summary: null };
  }
}

async function safeReadText(resp) {
  try {
    return await resp.text();
  } catch {
    return "";
  }
}

// Wire up the button
if (gptRunBtn) gptRunBtn.addEventListener("click", runGpt);

// ---------- DistilBERT (attention arcs with labels) ----------
const MIN_WEIGHT = 0.02;
const TOP_LINKS = 5;

const bertText = document.getElementById("bert-text");
const bertRun = document.getElementById("bert-run");
const bertStatus = document.getElementById("bert-status");
const bertTokens = document.getElementById("bert-tokens");
const bertSVG = document.getElementById("bert-arcs");
const bertNote = document.getElementById("bert-note");

let bertState = { tokens: [], attn: null, active: 0 };

function renderBertTokens() {
  bertTokens.innerHTML = "";
  bertState.tokens.forEach((t, i) => {
    const s = document.createElement("span");
    s.className = "tok" + (i === bertState.active ? " active" : "");
    s.textContent = String(t).replace(/\s/g, "␣");
    s.onclick = () => {
      bertState.active = i;
      renderBertTokens();
      drawArcs();
    };
    bertTokens.appendChild(s);
  });
}

function clearArcs() {
  while (bertSVG.firstChild) bertSVG.removeChild(bertSVG.firstChild);
}
function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {}))
    el.setAttribute(k, String(v));
  return el;
}

function tokenAnchors() {
  // start/end at the TOP edge of each token to arc upwards above the boxes
  const svgBB = bertSVG.getBoundingClientRect();
  return [...bertTokens.children].map((sp) => {
    const bb = sp.getBoundingClientRect();
    return {
      x: (bb.left + bb.right) / 2 - svgBB.left,
      yTop: bb.top - svgBB.top, // top edge of token box
      yBottom: bb.bottom - svgBB.top, // bottom if you ever want downward arcs
    };
  });
}

function bezierMid(p0, p1, c) {
  const t = 0.5;
  const x = (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * c.x + t * t * p1.x;
  const y = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * c.y + t * t * p1.y;
  return { x, y };
}

function drawArcs() {
  clearArcs();
  const A = bertState.attn;
  if (!A) {
    bertNote.textContent = "No attention returned.";
    return;
  }
  bertNote.textContent = "";

  // marker definition (arrowheads)
  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "arrow",
    markerWidth: 8,
    markerHeight: 8,
    refX: 8,
    refY: 4,
    orient: "auto",
    markerUnits: "strokeWidth",
  });
  marker.appendChild(
    svgEl("path", { d: "M0,0 L8,4 L0,8 Z", class: "marker-arrow" }),
  );
  defs.appendChild(marker);
  bertSVG.appendChild(defs);

  const pts = tokenAnchors();
  const n = A.length;
  const focus = Math.max(0, Math.min(bertState.active, n - 1));

  // rank outgoing weights
  const edges = A[focus]
    .map((w, j) => ({ j, w }))
    .filter(({ j, w }) => j !== focus && w >= MIN_WEIGHT)
    .sort((a, b) => b.w - a.w)
    .slice(0, TOP_LINKS);

  edges.forEach(({ j, w }) => {
    const p0 = { x: pts[focus].x, y: pts[focus].yTop }; // start at top of focus token
    const p1 = { x: pts[j].x, y: pts[j].yTop }; // end at top of target token

    // Curve height grows with horizontal distance, capped
    const dx = Math.abs(p1.x - p0.x);
    const h = Math.min(160, 20 + dx * 0.25);

    // Control point centered horizontally, lifted ABOVE tokens
    const cx = (p0.x + p1.x) / 2;
    const cy = Math.min(p0.y, p1.y) - h;

    const d = `M ${p0.x},${p0.y} Q ${cx},${cy} ${p1.x},${p1.y}`;
    const path = svgEl("path", {
      d,
      class: "arc-path",
      "marker-end": "url(#arrow)",
    });
    path.style.strokeWidth = String(1.5 + 4.5 * Math.sqrt(w));
    bertSVG.appendChild(path);

    // Label near the midpoint, slightly above the arc
    const mid = bezierMid(p0, p1, { x: cx, y: cy });
    const label = svgEl("text", {
      x: mid.x,
      y: mid.y - 6,
      class: "arc-label",
      "text-anchor": "middle",
    });
    label.textContent = w.toFixed(2);
    bertSVG.appendChild(label);
  });
}

async function runBert() {
  try {
    const text = bertText.value || "";
    bertStatus.textContent = "Analyzing…";
    const res = await fetch(BERT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    const data = await res.json();
    bertState.tokens = data.tokens || [];
    bertState.attn = data.attn || null; // [seq][seq], last layer avg heads
    bertState.active = 0;
    renderBertTokens();
    // ensure tokens are laid out before measuring
    requestAnimationFrame(drawArcs);
    bertStatus.textContent = "";
  } catch (e) {
    bertStatus.textContent = e.message || "Error";
    clearArcs();
    bertTokens.innerHTML = "";
  }
}

// resize handling so arcs stay aligned after window resizes
window.addEventListener("resize", () => {
  if (bertState.tokens.length) drawArcs();
});

// Defaults + wire-up
document.getElementById("gpt-run").addEventListener("click", runGpt);
document.getElementById("bert-run").addEventListener("click", runBert);
gptText.value = "In the room the women come and go, talking of Michelangelo.";
bertText.value = "The dog walked over the street because it was lazy.";

// ===== Minimal Next-token lab =====
(() => {
  const root = document.querySelector("#next-token-lab");
  if (!root) return;

  const $ = (s) => root.querySelector(s);
  const input = $("#ntl-input");
  const count = $("#ntl-count");
  const stepBtn = $("#ntl-step");
  const tbody = $("#ntl-body");
  const statusEl = $("#ntl-status");
  const noteEl = $("#ntl-note");

  const fmt = (x) => Number(x).toPrecision(4);

  function updateCount() {
    const n = (input.value || "").length;
    count.textContent = n;
  }

  function appendToken(displayToken) {
    const t = displayToken === "<new-line>" ? "\n" : displayToken;
    input.value = (input.value || "") + t.replace("\u2423", " ");
    updateCount();
  }

  function renderTopK(rows) {
    tbody.innerHTML = "";
    (rows || []).forEach((r) => {
      const tr = document.createElement("tr");
      const tdTok = document.createElement("td");
      const tdProb = document.createElement("td");
      tdTok.textContent = r.token;
      tdProb.textContent = fmt(r.prob);
      tr.appendChild(tdTok);
      tr.appendChild(tdProb);
      tr.addEventListener("click", async () => {
        appendToken(r.token);
        await fetchNext(); // loop again automatically after click
      });
      tbody.appendChild(tr);
    });
  }

  async function fetchNext() {
    statusEl.textContent = "Väntar...";
    noteEl.textContent = "";
    tbody.innerHTML = "";
    try {
      const res = await fetch("/gpt2/next", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: input.value }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderTopK(data.topk || []);
      statusEl.textContent = "Redo.";
      if (data.note) noteEl.textContent = data.note;
    } catch (e) {
      statusEl.textContent = `Fel: ${e.message}`;
    }
  }

  // events
  input.addEventListener("input", updateCount);
  stepBtn.addEventListener("click", fetchNext);

  // init
  updateCount();
})();

(() => {
  const t = document.querySelector("#gpt-text");
  const c = document.querySelector("#gpt-count");
  if (!t || !c) return;
  const upd = () => (c.textContent = (t.value || "").length);
  t.addEventListener("input", upd);
  upd();
})();
