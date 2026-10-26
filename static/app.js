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
  $("repcard").hidden = !ok;
  if (ok) {
    $("replab").textContent = `lead ${r.representations.lead}, mean of the ${r.n_pvc_beats} PVC beats`;
    plotPSD(r.representations);
    plotCWT(r.representations);
  }
}

function canvas2d(id, H) {
  const c = $(id), dpr = window.devicePixelRatio || 1, W = c.clientWidth;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  return { g, W, H, col: (n) => css.getPropertyValue(n).trim() };
}

function axes(g, col, L, T, R, B, xlab, ylab) {
  g.strokeStyle = col("--line"); g.lineWidth = 1; g.strokeRect(L, T, R - L, B - T);
  g.fillStyle = col("--muted"); g.font = "11px Segoe UI, sans-serif";
  g.fillText(xlab, (L + R) / 2 - g.measureText(xlab).width / 2, B + 28);
  g.save(); g.translate(12, (T + B) / 2); g.rotate(-Math.PI / 2); g.fillText(ylab, -g.measureText(ylab).width / 2, 0); g.restore();
}

function plotPSD(rep) {
  const { g, W, H, col } = canvas2d("psd", 200), L = 44, T = 8, R = W - 8, B = H - 34;
  const f = rep.psd_hz, p = rep.psd_log10, lo = Math.min(...p), hi = Math.max(...p);
  const x = (v) => L + (v / 60) * (R - L), y = (v) => B - ((v - lo) / (hi - lo || 1)) * (B - T);
  axes(g, col, L, T, R, B, "Frequency (Hz)", "log10 power");
  g.strokeStyle = col("--accent"); g.lineWidth = 2; g.beginPath();
  f.forEach((v, i) => (i ? g.lineTo(x(v), y(p[i])) : g.moveTo(x(v), y(p[i]))));
  g.stroke();
  g.fillStyle = col("--muted");
  for (const t of [0, 10, 20, 30, 40, 50, 60]) g.fillText(t, x(t) - 4, B + 13);
  for (const v of [lo, hi]) g.fillText(v.toFixed(1), 18, y(v) + 4);
}

function magma(t) {  // compact perceptual colormap (black -> purple -> orange -> pale yellow)
  const s = [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]];
  const u = Math.min(0.9999, Math.max(0, t)) * (s.length - 1), i = Math.floor(u), k = u - i;
  return s[i].map((a, j) => Math.round(a + (s[i + 1][j] - a) * k));
}

function plotCWT(rep) {
  const { g, W, H, col } = canvas2d("cwt", 200), L = 44, T = 8, R = W - 8, B = H - 34;
  const m = rep.cwt, nf = m.length, nt = m[0].length, flat = m.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
  const img = g.createImageData(nt, nf);
  for (let fi = 0; fi < nf; fi++) for (let ti = 0; ti < nt; ti++) {
    const [r, gg, b] = magma((m[fi][ti] - lo) / (hi - lo || 1)), o = 4 * ((nf - 1 - fi) * nt + ti);  // high freq on top
    img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = nt; off.height = nf;
  off.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true; g.drawImage(off, L, T, R - L, B - T);
  axes(g, col, L, T, R, B, "Time from R peak (ms)", "Frequency (Hz)");
  const [t0, t1] = rep.cwt_ms, xt = (t) => L + ((t - t0) / (t1 - t0)) * (R - L);
  g.strokeStyle = "#ffffff"; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(xt(0), T); g.lineTo(xt(0), B); g.stroke(); g.setLineDash([]);
  g.fillStyle = col("--muted");
  for (const t of [-300, -150, 0, 150, 300, 500]) g.fillText(t, xt(t) - 8, B + 13);
  const fmin = rep.cwt_hz[0], fmax = rep.cwt_hz[nf - 1], yf = (f) => B - (Math.log(f / fmin) / Math.log(fmax / fmin)) * (B - T);
  for (const f of [2, 5, 10, 20, 60]) g.fillText(f, 22, yf(f) + 4);
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
