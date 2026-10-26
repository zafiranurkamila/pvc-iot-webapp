const $ = (id) => document.getElementById(id);
const STAGES = { filtering: "Filtering (0.5–40 Hz, 50 Hz notch)", beat_extraction: "R-peak detection and PVC extraction",
  normalization: "Z-score normalization", psd: "PSD (Welch)", cwt: "CWT scalogram", cnn_inference: "HDR-CNN inference",
  total: "Total pipeline" };

async function init() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("model").textContent = `Model ${s.model} · ${s.features.join(" + ").toUpperCase()} · ${s.parameters.toLocaleString()} parameters`;
    const samples = await (await fetch("/api/samples")).json();
    for (const n of samples) $("sample").add(new Option(n.replace(".csv", ""), n));
  } catch { $("model").textContent = "server not reachable"; }
  $("run").onclick = run;
  refreshHistory();
  setInterval(refreshHistory, 3000);
}

async function run() {
  const file = $("file").files[0], sample = $("sample").value;
  if (!file && !sample) return msg("Choose a sample or upload a CSV file.", true);
  $("run").disabled = true;
  msg("Processing…");
  try {
    let res;
    if (file) { const fd = new FormData(); fd.append("file", file); res = await fetch("/api/predict", { method: "POST", body: fd }); }
    else res = await fetch(`/api/predict?sample=${encodeURIComponent(sample)}`, { method: "POST" });
    const r = await res.json();
    if (!res.ok) throw new Error(r.error || res.statusText);
    show(r);
    msg(r.status === "ok" ? "" : r.message, r.status !== "ok");
    $("file").value = "";
    refreshHistory();
  } catch (e) { msg(`Error: ${e.message}`, true); }
  $("run").disabled = false;
}

function msg(t, err = false) { $("msg").textContent = t; $("msg").className = "hint" + (err ? " err" : ""); }

function show(r) {
  for (const id of ["result", "plotcard", "beatcard", "timecard"]) $(id).hidden = false;
  const ok = r.status === "ok";
  $("origin").textContent = ok ? r.prediction : "No PVC";
  $("origin").className = "origin " + (ok ? r.prediction : "");
  $("conf").textContent = ok ? `confidence ${(r.confidence * 100).toFixed(1)}%` : "";
  $("record").textContent = r.record;
  $("nbeats").textContent = ok ? r.n_pvc_beats : 0;
  $("dur").textContent = `${r.duration_s} s · ${r.r_peaks.length} beats`;
  $("total").textContent = `${(r.timing_ms.total ?? r.request_ms).toFixed(0)} ms`;
  $("pbar").style.left = ok ? `${r.p_lvot * 100}%` : "50%";
  $("plvot").textContent = ok ? `P(LVOT) = ${r.p_lvot.toFixed(3)}` : "";
  $("beats").innerHTML = ok ? r.beats.map((b) => `<tr><td>${b.beat}</td><td class="num">${b.t.toFixed(2)}</td>
      <td class="num">${b.qrs_ms}</td><td class="num">${b.p_lvot.toFixed(3)}</td><td class="tag ${b.vote}">${b.vote}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">No PVC beats</td></tr>`;
  $("timing").innerHTML = Object.entries(r.timing_ms).map(([k, v]) =>
    `<tr><td>${STAGES[k] || k}</td><td class="num">${v.toFixed(1)} ms</td></tr>`).join("");
  plot(r);
}

function plot(r) {
  const c = $("ecg"), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = 220;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
  const v = r.lead_ii.values, fs = r.lead_ii.fs, T = v.length / fs;
  const lo = Math.min(...v), hi = Math.max(...v), pad = 18;
  const x = (t) => (t / T) * W, y = (a) => pad + (1 - (a - lo) / (hi - lo || 1)) * (H - 2 * pad);
  g.clearRect(0, 0, W, H);
  for (let t = 0; t <= T; t += 0.2) {           // ECG-paper grid: 0.2 s major lines, 0.04 s minor
    g.strokeStyle = col("--grid2"); g.lineWidth = Math.round(t * 5) % 5 === 0 ? 1 : 0.5;
    g.beginPath(); g.moveTo(x(t), 0); g.lineTo(x(t), H); g.stroke();
  }
  g.strokeStyle = col("--ink"); g.lineWidth = 1; g.beginPath();
  v.forEach((a, i) => (i ? g.lineTo(x(i / fs), y(a)) : g.moveTo(x(0), y(a))));
  g.stroke();
  const used = new Set((r.beats || []).map((b) => b.beat));
  r.r_peaks.forEach((p, i) => {
    const a = v[Math.min(v.length - 1, Math.round(p.t * fs))], px = x(p.t), py = y(a) - 8;
    g.fillStyle = used.has(i + 1) ? col("--lvot") : col("--rvot");
    g.beginPath();
    if (used.has(i + 1)) { g.moveTo(px - 5, py - 8); g.lineTo(px + 5, py - 8); g.lineTo(px, py); }
    else g.arc(px, py - 3, 3.5, 0, 2 * Math.PI);
    g.fill();
  });
  g.fillStyle = col("--muted"); g.font = "11px Segoe UI, sans-serif";
  for (let t = 0; t <= T; t += 1) g.fillText(`${t}s`, x(t) + 2, H - 4);
}

async function refreshHistory() {
  try {
    const h = await (await fetch("/api/history")).json();
    if (!h.length) return;
    $("history").innerHTML = h.map((e) => `<tr><td>${e.time}</td><td>${e.device}</td><td>${e.record}</td>
      <td class="tag ${e.prediction || ""}">${e.prediction || "no PVC"}</td>
      <td class="num">${e.confidence != null ? (e.confidence * 100).toFixed(1) + "%" : "–"}</td>
      <td class="num">${e.n_pvc_beats}</td><td class="num">${e.request_ms.toFixed(0)} ms</td></tr>`).join("");
  } catch { /* server offline: keep the last table */ }
}

init();
