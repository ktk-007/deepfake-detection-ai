// Unmask page: upload -> analyzing (live steps) -> result or error. No build step, no external requests.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const MAX = 100 * 1024 * 1024;
  const EXT = [".mp4", ".mov", ".avi"];
  const STEPS = ["extract", "detect", "analyze", "result"];
  const ICONS = {
    fake: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17h.01"/></svg>',
    real: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>',
    inconclusive: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01"/></svg>',
    noface: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="8" r="4"/><path d="M3 21a7 7 0 0 1 10-6.3"/><circle cx="18" cy="17" r="4"/><path d="M17 15.6a1.2 1.2 0 1 1 1.6 1.2c-.4.2-.6.5-.6.9M18 19.6h.01"/></svg>',
    bad: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="14" height="14" rx="2"/><path d="m17 10 4-2.5v9L17 14M7 9l6 6M13 9l-6 6"/></svg>',
  };
  const TITLES = { fake: "Likely Fake", real: "Likely Real", inconclusive: "Inconclusive" };

  let file = null, fileUrl = null, meta = {}, pollTimer = null;

  // ---------------------------------------------------------------- theme
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  function effectiveTheme() {
    return document.documentElement.dataset.theme || (media.matches ? "dark" : "light");
  }
  function paintThemeButtons() {
    document.querySelectorAll("[data-set-theme]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.setTheme === effectiveTheme())));
  }
  document.querySelectorAll("[data-set-theme]").forEach((b) => b.addEventListener("click", () => {
    document.documentElement.dataset.theme = b.dataset.setTheme;
    try { localStorage.setItem("unmask-theme", b.dataset.setTheme); } catch (e) { /* private mode */ }
    paintThemeButtons();
  }));
  media.addEventListener?.("change", paintThemeButtons);
  paintThemeButtons();

  // ---------------------------------------------------------------- views
  function show(name) {
    for (const v of ["home", "analyzing", "result", "error"]) $("view-" + v).hidden = v !== name;
    window.scrollTo({ top: 0 });
  }
  function homeError(msg) {
    $("home-error").textContent = msg || "";
    $("home-error").hidden = !msg;
  }
  function fmtSize(b) { return b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB"; }
  function fmtDur(s) {
    if (!isFinite(s) || s <= 0) return "";
    s = Math.round(s);
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }
  function infoLine() { return [meta.size, meta.resolution, meta.duration].filter(Boolean).join(" · "); }

  // ---------------------------------------------------------------- models
  async function loadModels() {
    const sel = $("model");
    try {
      const r = await fetch("/api/models");
      const list = await r.json();
      sel.innerHTML = "";
      for (const m of list) {
        const o = document.createElement("option");
        o.value = m.key; o.textContent = m.label;
        sel.appendChild(o);
      }
      if (!list.length) {
        const o = document.createElement("option");
        o.textContent = "No trained model found"; o.value = "";
        sel.appendChild(o);
        homeError("No trained model was found in weights\\ or runs\\.");
      }
    } catch (e) {
      homeError("Could not reach the Unmask server.");
    }
    updateAnalyze();
  }
  function updateAnalyze() { $("analyze").disabled = !(file && $("model").value); }
  $("model").addEventListener("change", updateAnalyze);

  // ---------------------------------------------------------------- file selection
  function clearFile() {
    file = null; meta = {};
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    fileUrl = null;
    $("thumb").removeAttribute("src");
    $("drop-empty").hidden = false;
    $("drop-file").hidden = true;
    $("file").value = "";
    updateAnalyze();
  }
  function pick(f) {
    homeError("");
    if (!f) return;
    const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    if (!EXT.includes(ext)) { clearFile(); homeError("This file can't be analyzed. Use MP4, MOV or AVI."); return; }
    if (f.size > MAX) { clearFile(); homeError("This file is larger than 100 MB."); return; }
    clearFile();
    file = f;
    fileUrl = URL.createObjectURL(f);
    meta = { size: fmtSize(f.size) };
    $("file-name").textContent = f.name;
    $("file-info").textContent = infoLine();
    $("thumb-dur").textContent = "";
    const v = $("thumb");
    v.onloadedmetadata = () => {
      if (v.videoWidth) meta.resolution = v.videoWidth + " x " + v.videoHeight;
      meta.duration = fmtDur(v.duration);
      $("file-info").textContent = infoLine();
      $("thumb-dur").textContent = meta.duration;
      try { v.currentTime = Math.min(1, (v.duration || 2) / 2); } catch (e) { /* ignore */ }
    };
    v.onerror = () => { $("file-info").textContent = infoLine() + " · preview not available"; };
    v.src = fileUrl;
    $("drop-empty").hidden = true;
    $("drop-file").hidden = false;
    updateAnalyze();
  }
  const drop = $("drop");
  drop.addEventListener("click", (e) => {
    if (e.target.closest("#remove") || e.target.closest("#drop-file")) return;
    $("file").click();
  });
  drop.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !file) { e.preventDefault(); $("file").click(); }
  });
  $("file").addEventListener("change", (e) => pick(e.target.files[0]));
  $("remove").addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

  // ---------------------------------------------------------------- analyzing
  function paintSteps(step, progress) {
    const at = STEPS.indexOf(step);
    document.querySelectorAll("#steps li").forEach((li) => {
      const i = STEPS.indexOf(li.dataset.step);
      li.classList.toggle("done", i < at || (i === at && progress >= 1 && step === "result"));
      li.classList.toggle("active", i === at && !(progress >= 1 && step === "result"));
    });
    const p = document.querySelector('#steps [data-step="analyze"] .progress');
    const pct = step === "analyze" ? Math.round(progress * 100) : at > 2 ? 100 : 0;
    p.hidden = at < 2;
    p.querySelector("i").style.width = pct + "%";
    p.querySelector(".pct").textContent = pct + "%";
  }
  async function analyze() {
    if (!file) return;
    show("analyzing");
    paintSteps("extract", 0);
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("model", $("model").value);
    fd.append("size", meta.size || "");
    fd.append("resolution", meta.resolution || "");
    fd.append("duration", meta.duration || "");
    let job;
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) return showError("unreadable");
      job = (await r.json()).job;
    } catch (e) { return showError("unreadable"); }
    poll(job);
  }
  function poll(job) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      let j;
      try { j = await (await fetch("/api/jobs/" + job)).json(); } catch (e) { return showError("unreadable"); }
      if (j.status === "done") { paintSteps("result", 1); return setTimeout(() => showResult(job, j.result), 250); }
      if (j.status === "error") return showError(j.error && j.error.kind);
      paintSteps(j.step, j.progress || 0);
      poll(job);
    }, 300);
  }
  $("analyze").addEventListener("click", analyze);

  // ---------------------------------------------------------------- result
  const svgNS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, text) {
    const n = document.createElementNS(svgNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function chart(probs, low, high) {
    const w = 900, h = 230, L = 46, R = 112, T = 10, B = 40, pw = w - L - R, ph = h - T - B;
    const y = (p) => T + ph * (1 - p);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img",
      "aria-label": `Fake score for each of ${probs.length} analysed frames` });
    svg.appendChild(el("rect", { x: L, y: y(high), width: pw, height: y(low) - y(high), class: "band" }));
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1])
      svg.appendChild(el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end", class: "ax" }, Math.round(t * 100) + "%"));
    const bw = pw / probs.length;
    probs.forEach((p, i) => {
      const c = p >= high ? "var(--fake)" : p <= low ? "var(--real)" : "var(--unsure)";
      const r = el("rect", { x: L + i * bw + bw * 0.18, y: y(p), width: bw * 0.64, height: Math.max(1, ph * p), rx: 3, fill: c });
      r.appendChild(el("title", {}, `Frame ${i + 1}: ${(p * 100).toFixed(1)}% fake`));
      svg.appendChild(r);
      if (probs.length <= 16 || i % 4 === 0 || i === probs.length - 1)
        svg.appendChild(el("text", { x: L + i * bw + bw / 2, y: T + ph + 18, "text-anchor": "middle", class: "ax" }, i + 1));
    });
    svg.appendChild(el("line", { x1: L, x2: L + pw, y1: y(0), y2: y(0), class: "base" }));
    for (const [t, lab] of [[high, "Fake above"], [low, "Real below"]]) {
      svg.appendChild(el("line", { x1: L, x2: L + pw, y1: y(t), y2: y(t), class: "thr" }));
      svg.appendChild(el("text", { x: L + pw + 8, y: y(t) + 4, class: "ax" }, `${lab} ${Math.round(t * 100)}%`));
    }
    svg.appendChild(el("text", { x: L + pw / 2, y: h - 4, "text-anchor": "middle", class: "ax" }, "Analysed frames"));
    svg.appendChild(el("text", { x: 12, y: T + ph / 2, "text-anchor": "middle", class: "ax",
      transform: `rotate(-90 12 ${T + ph / 2})` }, "Fake score"));
    return svg;
  }
  function showResult(job, r) {
    const lab = r.label;
    const fakeSide = lab !== "real";
    const p = fakeSide ? r.score : 1 - r.score;
    const pill = $("r-pill");
    pill.className = "pill " + lab;
    pill.innerHTML = ICONS[lab];
    pill.appendChild(document.createTextNode(" " + TITLES[lab]));
    $("r-sentence").textContent = r.sentence;
    $("r-prob-label").textContent = fakeSide ? "Fake probability" : "Real probability";
    const color = lab === "fake" ? "var(--fake)" : lab === "real" ? "var(--real)" : "var(--unsure)";
    $("r-prob").textContent = (p * 100).toFixed(1) + "%";
    $("r-prob").style.color = color;
    $("r-bar").style.width = (p * 100).toFixed(1) + "%";
    $("r-bar").style.background = color;
    $("r-bar-pct").textContent = Math.round(p * 100) + "%";
    const conf = $("r-conf");
    conf.className = "conf " + (lab === "fake" ? "fake-c" : lab === "real" ? "real-c" : "unsure-c");
    conf.textContent = r.confidence ? r.confidence + " confidence" : "Inconclusive: inside the uncertain band";

    $("r-chart").replaceChildren(chart(r.frame_probs, r.low, r.high));
    $("r-band").textContent = `Uncertain band ${Math.round(r.low * 100)}–${Math.round(r.high * 100)}%` +
      (r.calibrated ? " (calibrated on validation data)." : " (default band; calibration not fitted yet).");

    const faces = $("r-faces");
    faces.replaceChildren();
    for (const f of r.faces) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = f.image; img.alt = `Face from analysed frame ${f.frame}, ${(f.prob * 100).toFixed(0)}% fake`;
      const cap = document.createElement("figcaption");
      cap.textContent = `Frame ${f.frame} · ${(f.prob * 100).toFixed(0)}%`;
      fig.append(img, cap);
      faces.appendChild(fig);
    }
    $("r-model").textContent = r.model;
    $("r-frames").textContent = r.n_frames;
    $("r-time").textContent = r.seconds.toFixed(1) + " s";
    $("r-name").textContent = file ? file.name : "";
    $("r-info").textContent = infoLine();
    $("player").src = fileUrl || "";
    $("r-report").href = "/api/report/" + job;
    $("r-note").textContent = r.note;
    show("result");
  }

  // ---------------------------------------------------------------- errors
  function showError(kind) {
    const noFace = kind === "no_face";
    $("err-icon").innerHTML = noFace ? ICONS.noface : ICONS.bad;
    $("err-title").textContent = noFace ? "No face found" : "This file can't be analyzed";
    $("err-text").textContent = noFace
      ? "We could not find a clear face in this video. Try a video where one person faces the camera."
      : "The video could not be read. Try another MP4, MOV or AVI file under 100 MB.";
    show("error");
  }

  document.querySelectorAll("[data-new]").forEach((b) => b.addEventListener("click", () => {
    clearTimeout(pollTimer);
    $("player").removeAttribute("src");
    clearFile();
    homeError("");
    show("home");
    loadModels();
  }));

  loadModels();
})();
