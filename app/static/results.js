/* Results page. Every number shown here is read at runtime from /api/metrics (reports/summary.json) and
   /api/health. Nothing is hard-coded: a chart whose data is missing is hidden, and so is its chapter. */
"use strict";

(() => {
  const { html, setHTML, pct, fix, int, isNum, clamp, get, api, interp, onVisible, reduced } = U;
  // count up to a number; the first animation frame can be stamped early, so never show a value below zero
  const countUp = (el, to, fmt, opts) => U.countUp(el, to, (v) => fmt(Math.max(0, v)), opts);
  const setFill = (input, f) => input.style.setProperty("--f", clamp(f).toFixed(4));
  const $ = (id) => document.getElementById(id);
  document.documentElement.classList.add("js");   // scroll reveals only hide content once the scripts are known to run
  let M = null, H = null;
  const S = { budget: 0.3, listeners: [] };

  const pctA = (x, d) => (isNum(x) ? (100 * x).toFixed(d != null ? d : (Math.abs(x) < 0.1 ? 1 : 0)) + "%" : "–");
  const short = (name) => String(name).replace(/\s*\((GPU|CPU)\)/g, "");
  const cleanParams = (p) => String(p || "").split(/,\s*(?=[A-Za-z_]+=)/).filter((kv) => kv && !/^device=/i.test(kv)).join(", ");
  const paramsLabel = (m) => (m >= 1000 ? (m / 1000).toFixed(m >= 10000 ? 0 : 1) + "B" : (m >= 100 ? Math.round(m) : m.toFixed(m < 10 ? 1 : 0)) + "M");
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const hideChapter = (id) => { const s = $(id); if (s) s.hidden = true; };
  function safe(id, fn) {
    try { if (fn() === false) hideChapter(id); } catch (e) { hideChapter(id); }
  }

  // Which router answers on this server? /api/health says. A "light" server loads no transformer,
  // so nothing below may claim the transformer is the one deployed here.
  const hr = () => (H && H.router) || {};
  const isLight = () => {
    const r = hr();
    if (typeof r.mode === "string") return r.mode.toLowerCase() === "light";
    return Array.isArray(r.components) && r.components.length > 0 && !/transformer/i.test(r.components.join(" "));
  };
  const firstNum = (...a) => a.find(isNum);
  /** In light mode, mark a chapter whose numbers were measured with the full router. */
  function fullTag(id) {
    const head = document.querySelector(`#${id} .ch-head`);
    if (!head || head.querySelector(".full-tag")) return;
    const p = document.createElement("span");
    p.className = "full-tag";
    p.textContent = "Measured with the full transformer router, not the light one running on this server";
    head.appendChild(p);
  }

  function setBudget(b, src) {
    S.budget = clamp(Math.round(b * 100) / 100);
    S.listeners.forEach((f) => f(S.budget, src));
  }

  /* =========================================================================================
     Hero headline
     ========================================================================================= */
  function hero() {
    const R = M.router || {}, B = M.baselines || {}, D = M.data || {};
    const light = isLight(), LR = R.light || {}, h = hr();
    // headline numbers describe the router this server runs
    const live = light
      ? { apgr: firstNum(h.test_apgr, LR.test_apgr), auc: firstNum(h.test_auc, LR.test_auc), cpt50: firstNum(h.cpt50, LR.cpt50), cpt80: firstNum(h.cpt80, LR.cpt80) }
      : { apgr: R.apgr, auc: R.auc, cpt50: R.cpt50, cpt80: R.cpt80 };
    if (light) {
      const names = (h.components || LR.components || []).map((c) => String(c).replace(/\s*\(.*?\)/g, "")).join(", ");
      const n = $("modeNote");
      n.hidden = false;
      n.textContent = `This server runs the light router${names ? ` (${names})` : ""}, which fits on a very small machine. The headline numbers below are its own.${isNum(R.apgr) ? ` The full transformer router scores APGR ${fix(R.apgr, 3)}.` : ""} Chapters measured with the full router say so.`;
    }
    if (isNum(D.test_rows)) $("rEyebrow").textContent = `Results · ${int(D.test_rows)} held-out test prompts`;
    if (isNum(D.train_rows) && isNum(D.fail_rate_test)) {
      const p = document.createElement("span");
      p.textContent = ` The router learned from ${int(D.train_rows)} real prompts. On the test prompts, the cheap model gave a poor answer ${pctA(D.fail_rate_test, 1)} of the time.`;
      $("rLede").appendChild(p);
    }
    const tiles = [];
    if (isNum(live.apgr)) {
      const rnd = B.random_apgr, orc = B.oracle_apgr;
      const hasScale = isNum(rnd) && isNum(orc) && orc > rnd;
      const f = hasScale ? clamp((live.apgr - rnd) / (orc - rnd)) : 0;
      tiles.push(html`<div class="tile main"><div class="k">Quality gap recovered, averaged over every budget (APGR)</div>
        <b class="v num" data-to="${live.apgr}" data-d="3">${fix(live.apgr, 3)}</b>
        ${hasScale ? html`<div class="scale" data-f="${f.toFixed(4)}"><div class="bar"></div><div class="fill"></div>
          <span class="mk start" style="left:0">Random ${fix(rnd, 3)}</span><span class="mk end" style="left:100%">Perfect ${fix(orc, 3)}</span><span class="me"></span></div>
          <p class="scale-note">${pctA(f, 0)} of the way from random routing to a perfect router.</p>` : ""}</div>`);
    }
    if (isNum(live.auc)) tiles.push(html`<div class="tile"><div class="k">AUC</div><b class="v num" data-to="${live.auc}" data-d="3">${fix(live.auc, 3)}</b>
      <p class="d">How well it ranks risky prompts above safe ones. 0.5 is guessing, 1.0 is perfect.</p></div>`);
    if (isNum(live.cpt50)) tiles.push(html`<div class="tile"><div class="k">CPT50</div><b class="v num" data-to="${live.cpt50 * 100}" data-d="0" data-s="%">${pctA(live.cpt50, 0)}</b>
      <p class="d">of prompts on the strong model recover half the quality gap. Random routing needs 50%.</p></div>`);
    if (isNum(live.cpt80)) tiles.push(html`<div class="tile"><div class="k">CPT80</div><b class="v num" data-to="${live.cpt80 * 100}" data-d="0" data-s="%">${pctA(live.cpt80, 0)}</b>
      <p class="d">of prompts on the strong model recover 80% of the gap. Random routing needs 80%.</p></div>`);
    setHTML($("headline"), html`${tiles}`);
    U.initReveals($("headline"));
    $("headline").querySelectorAll("[data-to]").forEach((b) => {
      const to = Number(b.dataset.to), d = Number(b.dataset.d), s = b.dataset.s || "";
      onVisible(b, () => countUp(b, to, (v) => v.toFixed(d) + s, { dur: 800 }));
    });
    const sc = $("headline").querySelector(".scale");
    if (sc) onVisible(sc, () => {
      const f = Number(sc.dataset.f);
      setTimeout(() => { sc.querySelector(".fill").style.width = `${f * 100}%`; sc.querySelector(".me").style.left = `${f * 100}%`; }, 50);
    });
  }

  /* =========================================================================================
     01 Routing curve with a draggable scrubber
     ========================================================================================= */
  function curve() {
    const cv = M.curves || {}, R = M.router || {};
    if (!Array.isArray(cv.x) || !Array.isArray(cv.final)) return false;
    const q0 = R.quality_cheap_only, q1 = R.quality_strong_only;
    const rname = isLight() ? "Full router" : "RouteWise";
    const box = $("curveChart");
    let ref = null;
    C.mount(box, (w) => {
      box.textContent = "";
      const Hh = Math.round(clamp(w * 0.52, 260, 430));
      const wide = w >= 560;
      const m = { l: wide ? 68 : 50, r: 18, t: 38, b: 48 };
      const svg = C.el("svg", { viewBox: `0 0 ${w} ${Hh}`, width: w, height: Hh });
      const x = C.lin(0, 1, m.l, w - m.r), y = C.lin(0, 1, Hh - m.b, m.t);
      C.yAxis(svg, y, m.l, w - m.r, [0, 0.25, 0.5, 0.75, 1], (v) => pct(v));
      C.xAxis(svg, x, m.t, Hh - m.b, [0, 0.25, 0.5, 0.75, 1], (v) => pct(v), false);
      C.text(svg, (m.l + w - m.r) / 2, Hh - 6, w < 440 ? "Share sent to the strong model" : "Share of prompts sent to the strong model", { class: "axis-label", "text-anchor": "middle" });
      if (wide) C.text(svg, 14, (m.t + Hh - m.b) / 2, "Quality gap recovered", { class: "axis-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + Hh - m.b) / 2})` });
      const pts = (ys) => cv.x.map((xv, i) => `${x(xv).toFixed(1)},${y(ys[i]).toFixed(1)}`).join(" L");
      C.el("path", { d: `M${pts(cv.final)} L${x(1)},${y(0)} L${x(0)},${y(0)} Z`, class: "area fade" }, svg);
      C.axes(svg, m.l, w - m.r, m.t, Hh - m.b);
      C.el("path", { d: `M${x(0)},${y(0)} L${x(1)},${y(1)}`, class: "ln ln-random fade" }, svg);
      if (Array.isArray(cv.oracle)) C.el("path", { d: `M${pts(cv.oracle)}`, class: "ln ln-oracle fade" }, svg);
      C.el("path", { d: `M${pts(cv.final)}`, class: "ln ln-router-edge draw", pathLength: 1 }, svg);
      C.el("path", { d: `M${pts(cv.final)}`, class: "ln ln-router draw", pathLength: 1 }, svg);
      // direct labels instead of a legend
      C.text(svg, x(0.84), y(0.84) + 24, "Random routing", { class: "ann fade", "text-anchor": "middle" });
      if (Array.isArray(cv.oracle) && w >= 520) C.text(svg, x(0.36), y(1) + 23, "Perfect router (knows the answers)", { class: "ann fade" });
      const lx = w < 520 ? 0.55 : 0.5;
      if (w >= 520) C.text(svg, x(lx) - 10, y(interp(cv.x, cv.final, lx)) - 14, rname, { class: "ann-b fade", "text-anchor": "end" });   // on a phone the readout below names the lines
      if (isNum(R.apgr) && w > 420) {
        const ax = 0.66;
        C.text(svg, x(ax), y(interp(cv.x, cv.final, ax) * 0.42), `Shaded area = APGR ${fix(R.apgr, 3)}`, { class: "ann-b fade", "text-anchor": "middle" });
      }
      const g = C.el("g", { class: "scrub" }, svg);
      const ln = C.el("line", { y1: m.t, y2: Hh - m.b }, g);
      const dO = Array.isArray(cv.oracle) ? C.el("circle", { r: 7.5, class: "dot-o" }, g) : null;
      const dR = C.el("circle", { r: 7.5, class: "dot-r" }, g);
      const knob = C.el("circle", { r: 12, class: "knob" }, g);
      const hb = C.el("rect", { width: 60, height: 28, y: m.t - 32, class: "handle" }, g);
      const ht = C.text(g, 0, m.t - 12, "", { class: "handle-t", "text-anchor": "middle" });
      const hit = C.el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: Hh - m.t - m.b, class: "hit" }, svg);
      const toB = (e) => {
        const r = svg.getBoundingClientRect();
        return clamp(x.inv(((e.clientX - r.left) / r.width) * w));
      };
      let drag = false;
      hit.addEventListener("pointerdown", (e) => { drag = true; hit.setPointerCapture(e.pointerId); setBudget(toB(e), "curve"); });
      hit.addEventListener("pointermove", (e) => { if (drag) setBudget(toB(e), "curve"); });
      hit.addEventListener("pointerup", () => { drag = false; });
      hit.addEventListener("pointercancel", () => { drag = false; });
      box.appendChild(svg);
      ref = { x, y, ln, dO, dR, knob, hb, ht };
      place(S.budget);
    });
    function place(b) {
      if (!ref) return;
      const X = ref.x(b), vr = interp(cv.x, cv.final, b);
      ref.ln.setAttribute("x1", X); ref.ln.setAttribute("x2", X);
      ref.knob.setAttribute("cx", X); ref.knob.setAttribute("cy", ref.y(vr));
      ref.dR.setAttribute("cx", X); ref.dR.setAttribute("cy", ref.y(b));
      if (ref.dO) { ref.dO.setAttribute("cx", X); ref.dO.setAttribute("cy", ref.y(interp(cv.x, cv.oracle, b))); }
      ref.hb.setAttribute("x", X - 30); ref.ht.setAttribute("x", X); ref.ht.textContent = pct(b);
    }
    function read(b) {
      const vr = interp(cv.x, cv.final, b);
      const vo = Array.isArray(cv.oracle) ? interp(cv.x, cv.oracle, b) : NaN;
      const cells = [
        html`<div><span><i style="background:var(--lilac)"></i>${rname} recovers</span><b class="num">${pctA(vr, 1)}</b></div>`,
        html`<div><span><i style="background:var(--coral)"></i>Random would recover</span><b class="num">${pctA(b, 1)}</b></div>`,
        isNum(vo) ? html`<div><span><i style="background:var(--ink)"></i>Perfect would recover</span><b class="num">${pctA(Math.min(vo, 1), 1)}</b></div>` : "",
        isNum(q0) && isNum(q1) ? html`<div><span>Average score (1 to 5)</span><b class="num">${fix(q0 + vr * (q1 - q0), 2)}</b></div>` : "",
      ];
      setHTML($("curveRead"), html`${cells}`);
      $("curveOut").textContent = pct(b);
      const ins = $("curveInsight");
      if (b > 0.005 && b < 0.995) {
        ins.hidden = false;
        setHTML(ins, html`<span>With <b>${pct(b)}</b> of prompts on the strong model, the router recovers <b>${pctA(vr, 1)}</b> of the quality gap. Random routing with the same budget recovers ${pctA(b, 1)}: the router gets <b>${(vr / b).toFixed(1)}×</b> as much for the same spend.</span>`);
      } else {
        ins.hidden = false;
        setHTML(ins, html`<span>${b <= 0.005 ? "With no budget, every prompt stays on the cheap model, so no gap is recovered." : "With the whole budget, every prompt goes to the strong model and the whole gap is recovered, at full cost."}</span>`);
      }
    }
    $("curveRange").addEventListener("input", (e) => setBudget(Number(e.target.value) / 100, "range"));
    S.listeners.push((b) => { place(b); read(b); $("curveRange").value = Math.round(b * 100); setFill($("curveRange"), b); });
    C.reveal(box);
    return true;
  }

  /* =========================================================================================
     02 Budget and cost
     ========================================================================================= */
  function budget() {
    const cv = M.curves || {}, R = M.router || {};
    if (!Array.isArray(cv.x) || !Array.isArray(cv.final)) return false;
    const q0 = R.quality_cheap_only, q1 = R.quality_strong_only;
    const ratio = $("costRatio");
    function upd() {
      const b = S.budget, N = Number(ratio.value);
      const spend = (b * N + (1 - b)) / N;
      const vr = interp(cv.x, cv.final, b);
      $("costBudgetOut").textContent = pct(b);
      $("costBudget").value = Math.round(b * 100);
      setFill($("costBudget"), b);
      setFill(ratio, (N - Number(ratio.min)) / ((Number(ratio.max) - Number(ratio.min)) || 1));
      $("costRatioOut").textContent = `${N}×`;
      $("costSpend").textContent = pctA(spend, 0);
      $("costSpendBar").style.width = `${spend * 100}%`;
      $("costPgr").textContent = pctA(vr, 1);
      $("costPgrBar").style.width = `${vr * 100}%`;
      if (isNum(q0) && isNum(q1)) {
        const q = q0 + vr * (q1 - q0);
        $("costQ").textContent = `${fix(q, 2)} / 5`;
        $("costQBar").style.width = `${clamp((q - q0) / (q1 - q0 || 1)) * 100}%`;
        $("costQCap").textContent = `The bar runs from the cheap model alone (${fix(q0, 2)}) to the strong model alone (${fix(q1, 2)}; its answers count as perfect, see the limits).`;
      }
    }
    $("costBudget").addEventListener("input", (e) => setBudget(Number(e.target.value) / 100, "cost"));
    ratio.addEventListener("input", upd);
    S.listeners.push(upd);

    // at-budget cards
    const cards = Object.keys(R).filter((k) => /^at_\d+pct$/.test(k) && R[k] && isNum(R[k].quality))
      .map((k) => R[k]).sort((a, b) => a.strong_share - b.strong_share)
      .map((a) => html`<div class="at-card"><div class="k">AT A ${pct(a.strong_share)} BUDGET</div><b class="num">${fix(a.quality, 2)} / 5</b><span>${pctA(a.pgr, 1)} of the quality gap recovered</span></div>`);
    setHTML($("atCards"), html`${cards}`);
    $("atCards").hidden = !cards.length;

    // budget check chart
    const bc = R.budget_check || {};
    const pts = Object.keys(bc).map((k) => ({ a: parseFloat(k) / 100, g: bc[k] })).filter((p) => isNum(p.a) && isNum(p.g)).sort((p, q) => p.a - q.a);
    const box = $("budgetChart");
    if (pts.length < 2) { box.parentElement.hidden = true; return true; }
    C.mount(box, (w) => {
      box.textContent = "";
      const Hh = Math.round(clamp(w * 0.78, 220, 320));
      const m = { l: 54, r: 16, t: 14, b: 48 };
      const svg = C.el("svg", { viewBox: `0 0 ${w} ${Hh}`, width: w, height: Hh });
      const x = C.lin(0, 1, m.l, w - m.r), y = C.lin(0, 1, Hh - m.b, m.t);
      C.yAxis(svg, y, m.l, w - m.r, [0, 0.5, 1], (v) => pct(v));
      C.xAxis(svg, x, m.t, Hh - m.b, [0, 0.5, 1], (v) => pct(v), false);
      C.text(svg, (m.l + w - m.r) / 2, Hh - 4, "Asked for", { class: "axis-label", "text-anchor": "middle" });
      C.axes(svg, m.l, w - m.r, m.t, Hh - m.b);
      C.el("path", { d: `M${x(0)},${y(0)} L${x(1)},${y(1)}`, class: "ln ln-ident fade" }, svg);
      pts.forEach((p, i) => {
        const c = C.el("circle", { cx: x(p.a), cy: y(p.g), r: 8.5, class: "fade pt-dot", style: `fill:var(--blue);transition-delay:${0.2 + i * 0.03}s;cursor:pointer`, tabindex: 0, "aria-label": `Asked ${pct(p.a)}, got ${pctA(p.g, 1)}` }, svg);
        C.bindTip(c, () => html`Asked <b>${pct(p.a)}</b><br>Measured <b>${pctA(p.g, 1)}</b>`);
      });
      box.appendChild(svg);
    });
    C.reveal(box);
    const p30 = pts.find((p) => Math.abs(p.a - 0.3) < 1e-6) || pts[Math.floor(pts.length / 2)];
    $("budgetCap").textContent = `Dots sit on the dashed line when the budget holds. Asked for ${pct(p30.a)}, the router sent ${pctA(p30.g, 1)} of test prompts to the strong model.`;
    return true;
  }

  /* =========================================================================================
     03 Leaderboard
     ========================================================================================= */
  function leaderboard() {
    const cls = M.classifiers || [], R = M.router || {}, B = M.baselines || {};
    if (!cls.length) return false;
    const light = isLight();
    const comps = light ? [] : (Array.isArray(hr().components) ? hr().components : R.components || []);
    const deployed = comps.length === 1 ? comps[0] : null;
    const tagOf = (n) => (/fine-tuned/i.test(n) ? "tuned" : /neural|mlp|embedding/i.test(n) ? "neural" : /beyond syllabus/i.test(n) ? "beyond" : "classic");
    const items = cls.map((c, i) => {
      const p = cleanParams(c.params);
      const hl = c.name === deployed;
      const tuned = /fine-tuned/i.test(c.name);
      const head = /deficit/i.test(c.name) ? "score-deficit head" : "pass/fail head";
      const sub = tuned ? head : /beyond syllabus/i.test(c.name) ? "beyond the syllabus" : p.split(/,\s*(?=[A-Za-z_]+=)/).slice(0, 2).join(", ");
      const label = tuned ? "Fine-tuned transformer" : short(c.name).replace(/\s*\(beyond syllabus\)/i, "");
      return {
        id: i, label, sub, tag: tagOf(c.name), hl,
        values: { apgr: c.apgr, auc: c.test_auc, cpt80: c.cpt80, f1: c.f1, speed: c.predict_ms_per_1k },
        tip: () => html`<b>${short(c.name)}</b>${p ? html`<br><span class="muted">${p}</span>` : ""}
          <br>APGR ${fix(c.apgr, 3)} · AUC ${fix(c.test_auc, 3)} · CPT80 ${pctA(c.cpt80, 0)}
          <br>Precision ${pctA(c.precision, 1)} · Recall ${pctA(c.recall, 1)} · Specificity ${pctA(c.specificity, 1)}
          ${isNum(c.fit_seconds) ? html`<br>Training time ${c.fit_seconds < 120 ? c.fit_seconds.toFixed(1) + " s" : (c.fit_seconds / 60).toFixed(1) + " min"}` : ""}`,
      };
    });
    const L = R.light;
    if (L && isNum(L.test_apgr)) {
      items.push({ id: "light", label: "Light router", sub: (L.components || []).map(short).join(", ") || "for small servers", tag: "light", hl: light,
        values: { apgr: L.test_apgr, auc: L.test_auc, cpt80: L.cpt80 },
        tip: () => html`<b>Light router</b><br><span class="muted">${(L.components || []).join(", ")}: no transformer, for servers with little memory.</span><br>APGR ${fix(L.test_apgr, 3)} · AUC ${fix(L.test_auc, 3)}` });
    }
    const rnd = B.random_apgr, orc = B.oracle_apgr;
    const vals = (k) => items.map((it) => it.values[k]).filter(isNum);
    C.rankBars($("lbChart"), {
      items, initial: "apgr",
      metrics: [
        { key: "apgr", label: "APGR", better: "high", fmt: (v) => v.toFixed(3),
          domain: () => [Math.min(isNum(rnd) ? rnd : 0.5, ...vals("apgr")) - 0.05, isNum(orc) ? orc : Math.max(...vals("apgr"))],
          refs: [{ v: rnd, label: "random" }, { v: orc, label: "perfect" }],
          caption: `Quality gap recovered, averaged over every budget. Higher is better.${isNum(rnd) ? ` Random routing gets ${fix(rnd, 3)}` : ""}${isNum(orc) ? `, a perfect router ${fix(orc, 3)}.` : "."}` },
        { key: "auc", label: "AUC", better: "high", fmt: (v) => v.toFixed(3), domain: [0.45, 1], refs: [{ v: 0.5, label: "guessing" }],
          caption: "How well each model ranks risky prompts above safe ones, on the test set. 0.5 is guessing, 1.0 is perfect." },
        { key: "cpt80", label: "CPT80", better: "low", fmt: (v) => pctA(v, 0), domain: [0, 1], refs: [{ v: 0.8, label: "random" }],
          caption: "Share of prompts that must go to the strong model to recover 80% of the gap. Lower is better." },
        { key: "f1", label: "F1", better: "high", fmt: (v) => v.toFixed(3), domain: (v) => [0, C.niceDomain(0, Math.max(...v))[1]],
          caption: "F1 score of the \"cheap model fails\" prediction: a balance of precision and recall." },
        { key: "speed", label: "Speed", better: "low", log: true, fmt: (v) => (v >= 1000 ? (v / 1000).toFixed(1) + " s" : v >= 10 ? v.toFixed(0) + " ms" : v.toFixed(1) + " ms"),
          domain: (v) => [Math.min(...v) / 1.5, Math.max(...v) * 1.5],
          caption: "Time to score 1,000 prompts. Shorter bars are faster. Log scale." },
      ],
    });
    return true;
  }

  /* =========================================================================================
     04 Model-size study
     ========================================================================================= */
  function sizeStudy() {
    const ft = M.finetune || {}, sweep = ft.sweep || [];
    const paramsOf = (base) => (sweep.find((r) => r.base === base && isNum(r.params_millions)) || {}).params_millions;
    const nameOf = (tag) => {
      let n = String(tag).replace(/^study:\s*/, "").replace(/\s*\(.*?\)/g, "").split(",")[0].trim();
      if (/LoRA/i.test(tag)) n += " (LoRA)";
      return n;
    };
    const pts = sweep.filter((r) => r.kind === "study" && !/FAILED/i.test(r.tag) && isNum(r.params_millions) && isNum(r.val_auc_fail))
      .map((r) => ({ name: nameOf(r.tag), p: r.params_millions, auc: r.val_auc_fail, apgr: r.val_apgr_deficit, fam: /^qwen/i.test(r.base.split("/").pop()) ? "dec" : "enc", tag: r.tag }));
    const smallBase = ft.base_model, smallP = paramsOf(smallBase);
    const plain = sweep.filter((r) => r.kind === "sweep" && r.base === smallBase && isNum(r.val_apgr_deficit))
      .sort((a, b) => b.val_apgr_deficit - a.val_apgr_deficit)[0];
    const smallName = smallBase ? smallBase.split("/").pop().replace(/-en-v[\d.]+$/, "") : "small model";
    let noT = null, stu = null;
    if (plain && isNum(smallP)) { noT = { name: `${smallName}, no teacher`, p: smallP, auc: plain.val_auc_fail, apgr: plain.val_apgr_deficit, fam: "enc", tag: plain.tag }; pts.push(noT); }
    const res = ft.results || {};
    const fail = Object.keys(res).find((k) => !/deficit/i.test(k)), def = Object.keys(res).find((k) => /deficit/i.test(k));
    if (isNum(smallP) && fail && isNum(res[fail].val_auc)) {
      stu = { name: isLight() ? `${smallName}, student (full router)` : `${smallName}, student (deployed)`, p: smallP, auc: res[fail].val_auc, apgr: def ? res[def].val_apgr : NaN, fam: "dep" };
      pts.push(stu);
    }
    if (pts.length < 3) return false;
    // file-size limit as a parameter count: the served int8 file holds about one byte per parameter
    const rule = get(ft, "export.rule") || "";
    const lm = rule.match(/<\s*([\d.]+)\s*MB/i);
    const limitMB = lm ? parseFloat(lm[1]) : NaN;
    const mbPerM = isNum(get(ft, "export.served_mb")) && isNum(smallP) ? ft.export.served_mb / smallP : NaN;
    const limitP = isNum(limitMB) && isNum(mbPerM) ? limitMB / mbPerM : NaN;

    let metric = "auc", ref = null;
    const box = $("sizeChart");
    C.mount(box, (w) => {
      box.textContent = "";
      const Hh = Math.round(clamp(w * 0.56, 300, 460));
      const m = { l: 58, r: 24, t: 22, b: 50 };
      const svg = C.el("svg", { viewBox: `0 0 ${w} ${Hh}`, width: w, height: Hh });
      const pMin = Math.min(...pts.map((d) => d.p)), pMax = Math.max(...pts.map((d) => d.p));
      const x = C.logScale(Math.pow(10, Math.floor(Math.log10(pMin / 1.3) * 2) / 2), Math.pow(10, Math.ceil(Math.log10(pMax * 1.3) * 2) / 2), m.l, w - m.r);
      const gGrid = C.el("g", null, svg);
      const xt = [10, 30, 100, 300, 1000, 3000, 10000, 30000].filter((v) => v >= x.domain[0] && v <= x.domain[1] && (w >= 480 || String(v)[0] === "1"));   // decades only on a phone
      if (isNum(limitP)) {
        const xl = x(Math.min(limitP, x.domain[1]));
        C.el("rect", { x: m.l, y: m.t, width: Math.max(0, xl - m.l), height: Hh - m.t - m.b, class: "zone fade" }, svg);
        C.el("line", { x1: xl, x2: xl, y1: m.t, y2: Hh - m.b, class: "zone-line fade" }, svg);
        if (w >= 560) C.text(svg, xl + 8, m.t + 16, `≈${limitMB} MB file limit`, { class: "ann fade" });
        else C.text(svg, xl - 7, m.t + 16, `≈${limitMB} MB`, { class: "ann fade", "text-anchor": "end" });
      }
      C.xAxis(svg, x, m.t, Hh - m.b, xt, (v) => paramsLabel(v));
      C.text(svg, (m.l + w - m.r) / 2, Hh - 6, "Parameters (log scale)", { class: "axis-label", "text-anchor": "middle" });
      C.axes(svg, m.l, w - m.r, m.t, Hh - m.b);
      const gy = C.el("g", null, svg);
      const arrow = C.el("path", { class: "ln draw", pathLength: 1, style: "stroke:var(--ink);stroke-width:4;marker-end:url(#arw)" }, svg);
      const defs = C.el("defs", null, svg);
      const mk = C.el("marker", { id: "arw", viewBox: "0 0 10 10", refX: 7, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
      C.el("path", { d: "M0,0 L10,5 L0,10 z", style: "fill:var(--ink)" }, mk);
      const arrowT = C.text(svg, 0, 0, "", { class: "ann-b fade" });
      const sorted = pts.slice().sort((a, b) => a.p - b.p);
      const nodes = sorted.map((d, i) => {
        const g = C.el("g", { class: "pt", tabindex: 0, style: `transition:transform .8s var(--ease), opacity .5s ${0.3 + i * 0.08}s; opacity:0; cursor:pointer` }, svg);
        const color = d.fam === "dec" ? "var(--ink)" : d.fam === "dep" ? "var(--lilac)" : "var(--white)";
        C.el("circle", { r: d.fam === "dep" ? 13 : 9, class: "pt-dot", style: `fill:${color}` }, g);
        const showLabel = w >= 560 || d.fam === "dep" || i === sorted.length - 1;   // a phone has room for two labels; tap a dot for the rest
        const lbl = w >= 560 ? d.name : d.fam === "dep" ? "Student" : d.name;
        const t = showLabel ? C.text(g, 0, 0, lbl, { class: d.fam === "dep" ? "ann-b" : "ann" }) : null;
        C.bindTip(g, () => html`<b>${d.name}</b><br>${paramsLabel(d.p)} parameters<br>Validation AUC ${fix(d.auc, 4)} · APGR ${fix(d.apgr, 4)}`);
        g.setAttribute("aria-label", `${d.name}: ${paramsLabel(d.p)} parameters, validation AUC ${fix(d.auc, 4)}`);
        return { d, g, t, i };
      });
      box.appendChild(svg);
      ref = { x, gy, nodes, arrow, arrowT, m, w, Hh, svg };
      layout(false);
    });
    function layout() {
      if (!ref) return;
      const { x, gy, nodes, arrow, arrowT, m, w, Hh } = ref;
      const vals = pts.map((d) => d[metric]).filter(isNum);
      const [lo, hi] = C.niceDomain(Math.min(...vals) - 0.004, Math.max(...vals) + 0.004, 5);
      const y = C.lin(lo, hi, Hh - m.b, m.t);
      gy.textContent = "";
      C.yAxis(gy, y, m.l, w - m.r, C.ticks(lo, hi, 5), (v) => v.toFixed(metric === "auc" ? 2 : 3));
      const canGain = noT && stu && isNum(noT[metric]) && isNum(stu[metric]);
      const gain = canGain ? `${stu[metric] >= noT[metric] ? "+" : ""}${(stu[metric] - noT[metric]).toFixed(3)}` : "";
      if (w < 560) nodes.forEach(({ d, t }) => { if (t && d.fam === "dep") t.textContent = gain ? `Student ${gain}` : "Student"; });
      const placed = [];
      const hits = (b) => placed.some((q) => b.x < q.x + q.w && b.x + b.w > q.x && b.y < q.y + q.h && b.y + b.h > q.y);
      // place the important labels first, then the rest, trying a few offsets to avoid overlaps
      const order = nodes.slice().sort((a, b) => (b.d.fam === "dep") - (a.d.fam === "dep") || (b.d === noT) - (a.d === noT) || a.d.p - b.d.p);
      nodes.forEach(({ d }) => { const X = x(d.p), Y = y(d[metric]); placed.push({ x: X - 13, y: Y - 13, w: 26, h: 26 }); });
      order.forEach(({ d, g, t }) => {
        const X = x(d.p), Y = isNum(y(d[metric])) ? y(d[metric]) : Hh;
        g.style.transform = `translate(${X.toFixed(1)}px, ${Y.toFixed(1)}px)`;
        if (!t) return;
        const right = X < w * 0.74;
        const tw = (t.getComputedTextLength ? t.getComputedTextLength() : 60) || 60;
        const tries = d.fam === "dep" ? [-22, 34] : d === noT ? (w < 560 ? [-17, 28, -34] : [28, -17, -34]) : [-14, 26, -32, 44, -50];
        const inside = (o) => Y + o - 14 >= m.t - 6 && Y + o + 4 <= Hh - m.b - 3;
        let dy = tries.find(inside);
        if (dy == null) dy = tries[0];
        for (const o of tries) {
          const b = { x: right ? X + 17 : X - 17 - tw, y: Y + o - 14, w: tw, h: 18 };
          if (!inside(o)) continue;   // keep labels inside the plot
          if (!hits(b)) { dy = o; placed.push(b); break; }
        }
        t.setAttribute("x", right ? 17 : -17);
        t.setAttribute("y", dy);
        t.setAttribute("text-anchor", right ? "start" : "end");
      });
      if (canGain) {
        const X = x(stu.p), y0 = y(noT[metric]) - 14, y1 = y(stu[metric]) + 19;
        arrow.setAttribute("d", `M${X - 22},${y0} C${X - 42},${(y0 + y1) / 2} ${X - 38},${y1 + 6} ${X - 18},${y1}`);
        arrowT.setAttribute("x", X - 46); arrowT.setAttribute("y", (y0 + y1) / 2 + 4); arrowT.setAttribute("text-anchor", "end");
        arrowT.textContent = w >= 560 ? `distillation ${gain}` : "";
      } else { arrow.setAttribute("d", ""); arrowT.textContent = ""; }
    }
    const show = () => ref && ref.nodes.forEach(({ g }) => { g.style.opacity = 1; });
    C.reveal(box, show);
    const mo = new MutationObserver(() => { if (box.classList.contains("in")) show(); });
    mo.observe(box, { childList: true });
    $("sizeSeg").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      metric = b.dataset.k;
      $("sizeSeg").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      layout();
    }));
    // insight
    const best = pts.filter((d) => d.fam !== "dep").sort((a, b) => b.auc - a.auc)[0];
    const bigInt8 = sweep.map((r) => String(r.tag).match(/int8 file ([\d.]+)\s*MB/i)).find(Boolean);
    const bigBase = sweep.find((r) => /int8 file/i.test(r.tag));
    const parts = [];
    if (best) parts.push(html`<b>${best.name}</b> scored best, with validation AUC <b>${fix(best.auc, 3)}</b>.`);
    if (bigInt8 && bigBase && isNum(limitMB)) parts.push(html` Even ${bigBase.base.split("/").pop().replace(/-en-v[\d.]+$/, "")} is too big: its int8 file is ${bigInt8[1]} MB, over the ${limitMB} MB limit.`);
    if (noT && stu) parts.push(html` Distillation moved the deployable small model from <b>${fix(noT.auc, 3)}</b> to <b>${fix(stu.auc, 3)}</b>.`);
    if (parts.length) { $("sizeInsight").hidden = false; setHTML($("sizeInsight"), html`<span>${parts}</span>`); }
    if (isLight() && $("sizeDepLegend")) $("sizeDepLegend").textContent = "Small model after distillation (used by the full router)";
    return true;
  }

  /* =========================================================================================
     05 Distillation
     ========================================================================================= */
  function distill() {
    const ft = M.finetune || {}, sweep = ft.sweep || [], folds = ft.teacher_folds || [];
    const tName = ft.teacher_name;
    if (!tName) return false;
    const study = sweep.find((r) => r.kind === "study" && r.tag.includes(tName + " ") && isNum(r.params_millions));
    const tFolds = folds.filter((f) => f.tag.includes(tName + ",") || f.tag.includes(tName + " ")).map((f) => f.val_auc_fail).filter(isNum);
    const sP = (sweep.find((r) => r.base === ft.base_model && isNum(r.params_millions)) || {}).params_millions;
    const res = ft.results || {};
    const fail = Object.keys(res).find((k) => !/deficit/i.test(k));
    const sAuc = fail ? res[fail].val_auc : NaN;
    const tP = study ? study.params_millions : NaN;
    const ratio = isNum(tP) && isNum(sP) ? tP / sP : NaN;
    const side = 76, sSide = isNum(ratio) ? Math.max(4, side / Math.sqrt(ratio)) : 0;
    const sName = ft.base_model ? ft.base_model.split("/").pop() : "small model";
    // show the card name only ("NVIDIA H100"): the memory suffix adds nothing here
    const gpu = ft.gpu ? String(ft.gpu).replace(/\s+\d+\s*GB\b.*$/i, "").trim() : null;
    setHTML($("flow"), html`
      <div class="agent teacher"><div class="role">Teacher</div><h4>${tName}</h4>
        <div class="size">${isNum(tP) ? html`<i style="width:${side}px;height:${side}px"></i><span>${paramsLabel(tP)} parameters</span>` : ""}</div>
        <dl>${tFolds.length ? html`<dt>Copies trained</dt><dd>${tFolds.length}, one per fold</dd>
          <dt>Validation AUC</dt><dd class="num">${fix(Math.min(...tFolds), 3)}–${fix(Math.max(...tFolds), 3)}</dd>` : ""}
          ${study ? html`<dt>As one model</dt><dd class="num">AUC ${fix(study.val_auc_fail, 3)}</dd>` : ""}</dl></div>
      <div class="wire"><svg viewBox="0 0 120 40" aria-hidden="true"><path d="M4 20 C40 4, 80 36, 116 20"/><circle r="3.5"><animateMotion dur="1.6s" repeatCount="indefinite" path="M4 20 C40 4, 80 36, 116 20"/></circle><circle r="3.5"><animateMotion dur="1.6s" begin=".8s" repeatCount="indefinite" path="M4 20 C40 4, 80 36, 116 20"/></circle></svg>
        <span>soft labels: how likely each training prompt is to fail</span></div>
      <div class="agent student"><div class="role">${isLight() ? "Student · in the full router" : "Student · deployed"}</div><h4>${sName}</h4>
        <div class="size">${sSide ? html`<i style="width:${sSide.toFixed(1)}px;height:${sSide.toFixed(1)}px"></i><span>${paramsLabel(sP)} parameters${isNum(ratio) ? html` · ${Math.round(ratio)}× smaller` : ""}</span>` : ""}</div>
        <dl>${isNum(sAuc) ? html`<dt>Validation AUC</dt><dd class="num">${fix(sAuc, 4)}</dd>` : ""}
          ${isNum(get(ft, "export.served_mb")) ? html`<dt>Served file</dt><dd class="num">${ft.export.served_variant || ""} ${ft.export.served_mb.toFixed(1)} MB</dd>` : ""}
          ${isNum(ft.train_minutes) ? html`<dt>Training</dt><dd class="num">${ft.train_minutes.toFixed(1)} min${gpu ? html` on ${gpu.replace(/^NVIDIA\s*/, "NVIDIA ")}` : ""}</dd>` : ""}</dl></div>`);
    if (reduced()) $("flow").querySelectorAll("animateMotion").forEach((a) => a.remove());
    onVisible($("flowWrap"), () => $("flowWrap").classList.add("in"));
    if (isNum(ft.distill_alpha)) {
      const a = ft.distill_alpha;
      $("formula").hidden = false;
      setHTML($("formula"), html`The student's target: <code>α × real label + (1 − α) × teacher probability</code>, with α = <b class="num">${a}</b>.
        So ${pctA(1 - a, 0)} of what it learns from comes from the teacher.`);
    }

    // ladder: each teacher against its best student run
    const teachers = Array.from(new Set(folds.map((f) => (f.tag.match(/teacher:\s*(Qwen3-[\d.]+B)/) || [])[1]).filter(Boolean)));
    const rows = [];
    const plain = sweep.filter((r) => r.kind === "sweep" && r.base === ft.base_model && isNum(r.val_apgr_deficit)).sort((a, b) => b.val_apgr_deficit - a.val_apgr_deficit)[0];
    if (plain) rows.push({ label: "No teacher", sub: "small model alone", points: [{ v: plain.val_auc_fail, cls: "student", label: "Small model, no teacher" }] });
    const pOf = (t) => (sweep.find((r) => r.kind === "study" && r.tag.includes(t + " ")) || {}).params_millions || 0;
    teachers.sort((a, b) => pOf(a) - pOf(b)).forEach((t) => {
      const fa = folds.filter((f) => new RegExp("teacher:\\s*" + t.replace(/\./g, "\\.") + "\\b").test(f.tag)).map((f) => f.val_auc_fail).filter(isNum);
      const re = new RegExp("student of " + t.replace(/\./g, "\\.") + "(?![\\d.])(?!\\s*\\+)");
      const st = sweep.filter((r) => r.kind === "distill" && re.test(r.tag) && isNum(r.val_auc_fail));
      if (!fa.length || !st.length) return;
      const bestS = Math.max(...st.map((r) => r.val_auc_fail));
      rows.push({ label: t, sub: t === tName ? "teacher (chosen)" : "teacher", join: true, points: [
        { v: bestS, cls: t === tName ? "deployed below" : "student below", label: "Best student", tip: `best of ${st.length} student runs` },
        { v: mean(fa), cls: "teacher", label: `${t} teacher`, tip: `mean of ${fa.length} fold copies` },
      ] });
    });
    if (rows.length > 1) {
      const all = rows.flatMap((r) => r.points.map((p) => p.v));
      const dom = C.niceDomain(Math.min(...all) - 0.003, Math.max(...all) + 0.003, 4);
      C.dotLine($("ladder"), { rows, domain: dom, fmt: (v) => v.toFixed(3), axisLabel: "Validation AUC" });
      const leg = document.createElement("div");
      leg.className = "legend";
      setHTML(leg, html`<span><i class="round sw-tuned"></i>Teacher</span><span><i class="round sw-open"></i>Best student</span><span><i class="round sw-router"></i>Student of the chosen teacher</span>`);
      $("ladder").appendChild(leg);
    } else $("ladder").parentElement.hidden = true;

    // learning curve
    const hist = (ft.history || []).filter((h) => isNum(h.epoch) && isNum(h.val_auc_fail));
    const box = $("learnChart");
    if (hist.length < 2) { box.parentElement.hidden = true; return true; }
    C.mount(box, (w) => {
      box.textContent = "";
      const Hh = Math.round(clamp(w * 0.6, 200, 280));
      const m = { l: 62, r: 18, t: 16, b: 50 };
      const svg = C.el("svg", { viewBox: `0 0 ${w} ${Hh}`, width: w, height: Hh });
      const xs = hist.map((h) => h.epoch), ys = hist.map((h) => h.val_auc_fail);
      const x = C.lin(0, Math.max(...xs), m.l, w - m.r);
      const [lo, hi] = C.niceDomain(Math.min(...ys) - 0.002, Math.max(...ys) + 0.002, 4);
      const y = C.lin(lo, hi, Hh - m.b, m.t);
      C.yAxis(svg, y, m.l, w - m.r, C.ticks(lo, hi, 4), (v) => v.toFixed(3));
      C.xAxis(svg, x, m.t, Hh - m.b, C.ticks(0, Math.max(...xs), 4).filter((v) => v > 0), (v) => `${v}`, false);
      C.text(svg, (m.l + w - m.r) / 2, Hh - 4, "Epochs", { class: "axis-label", "text-anchor": "middle" });
      C.axes(svg, m.l, w - m.r, m.t, Hh - m.b);
      C.el("path", { d: "M" + hist.map((h) => `${x(h.epoch).toFixed(1)},${y(h.val_auc_fail).toFixed(1)}`).join(" L"), class: "ln ln-ink draw", pathLength: 1 }, svg);
      hist.forEach((h, i) => {
        const c = C.el("circle", { cx: x(h.epoch), cy: y(h.val_auc_fail), r: 8.5, class: "fade pt-dot", style: `fill:var(--lilac);transition-delay:${0.2 + i * 0.05}s;cursor:pointer`, tabindex: 0, "aria-label": `Epoch ${h.epoch}: validation AUC ${fix(h.val_auc_fail, 4)}` }, svg);
        C.bindTip(c, () => html`Epoch <b>${h.epoch}</b><br>Validation AUC <b>${fix(h.val_auc_fail, 4)}</b>`);
      });
      box.appendChild(svg);
    });
    C.reveal(box);
    const last = hist[hist.length - 1], maxV = Math.max(...hist.map((h) => h.val_auc_fail));
    const served = isLight() ? "the one the full router uses" : "the one served";
    $("learnCap").textContent = last.val_auc_fail >= maxV
      ? `Still improving at the last checkpoint (epoch ${last.epoch}). The best checkpoint is ${served}.`
      : `The best checkpoint (AUC ${fix(maxV, 4)}) is ${served}.`;
    if (isLight() && $("learnTitle")) $("learnTitle").textContent = "How the student learned";
    return true;
  }

  /* =========================================================================================
     06 Confusion matrix
     ========================================================================================= */
  function matrix() {
    const R = M.router || {}, D = M.data || {};
    const key = Object.keys(R).find((k) => /^classification_at_\d+pct$/.test(k));
    const c = key && R[key], cm = c && c.confusion_matrix;
    if (!cm || ![cm.tn, cm.fp, cm.fn, cm.tp].every(isNum)) return false;
    const share = parseInt(key.match(/(\d+)pct/)[1], 10);
    $("h-matrix").textContent = `At a ${share}% budget, which calls were right?`;
    $("matrixLede").textContent = `Every test prompt lands in one of four boxes, depending on where the router sent it and whether the cheap model really failed.${isNum(c.threshold) ? ` Here the threshold on P(cheap fails) is ${pctA(c.threshold, 1)}.` : ""}`;
    const max = Math.max(cm.tn, cm.fp, cm.fn, cm.tp);
    const cell = (n, cls, lbl, exp) => html`<div class="cell ${cls}" tabindex="0" style="--o:${(0.1 + 0.42 * (n / max)).toFixed(3)}" aria-label="${lbl}: ${int(n)}. ${exp}">
      <span class="lbl">${lbl}</span><b class="num" data-n="${n}">0</b><span class="exp">${exp}</span></div>`;
    setHTML($("cm"), html`<span></span><span class="hd">Sent to the strong model</span><span class="hd">Kept on the cheap model</span>
      <span class="rh">Cheap model failed</span>
      ${cell(cm.tp, "caught", "Caught", "The cheap model would have failed. The router paid for the strong one.")}
      ${cell(cm.fn, "bad", "Missed", "The cheap model failed, and the router let it.")}
      <span class="rh">Cheap model was fine</span>
      ${cell(cm.fp, "meh", "Extra spend", "The cheap model would have been fine. The strong call was not needed.")}
      ${cell(cm.tn, "good", "Saved", "The cheap model was fine, and the router kept it there.")}`);
    onVisible($("cm"), () => {
      $("cm").classList.add("in");
      $("cm").querySelectorAll("[data-n]").forEach((b) => countUp(b, Number(b.dataset.n), (v) => int(v), { dur: 800 }));
    });
    const ms = [
      ["Recall", c.recall_sensitivity, "of real failures were caught"],
      ["Specificity", c.specificity, "of fine prompts stayed on the cheap model"],
      ["Precision", c.precision, "of strong-model calls were really needed"],
      ["Accuracy", c.accuracy, "of all decisions were right"],
    ].filter((r) => isNum(r[1]));
    setHTML($("cmMetrics"), html`${ms.map(([k, v, d]) => html`<div class="meter"><div class="meter-top"><span><b>${k}</b> · ${d}</span><b class="num">${pctA(v, 1)}</b></div><div class="meter-bar"><i class="a" data-w="${v}" style="width:0"></i></div></div>`)}
      ${isNum(D.fail_rate_test) && isNum(c.precision) ? html`<div class="insight" style="margin-top:4px"><span>Only <b>${pctA(D.fail_rate_test, 1)}</b> of test prompts fail on the cheap model. Of the prompts the router sent to the strong model, ${pctA(c.precision, 0)} really needed it: <b>${(c.precision / D.fail_rate_test).toFixed(1)}×</b> the base rate.</span></div>` : ""}`);
    onVisible($("cmMetrics"), () => $("cmMetrics").querySelectorAll("[data-w]").forEach((i) => { i.style.width = `${Number(i.dataset.w) * 100}%`; }));
    return true;
  }

  /* =========================================================================================
     07 Clusters
     ========================================================================================= */
  function clusters() {
    const cl = M.clustering || {}, D = M.data || {};
    const list = cl.clusters || [];
    if (!list.length) return false;
    const items = list.map((c) => ({
      id: c.cluster, label: (c.top_words || []).slice(0, 3).join(" · ") || `cluster ${c.cluster}`, sub: `${int(c.size_train)} prompts`,
      tag: isNum(D.fail_rate_test) && c.fail_rate_test > D.fail_rate_test ? "hot" : "cool",
      values: { test: c.fail_rate_test, train: c.fail_rate_train },
      tip: () => html`<b>Cluster ${c.cluster}</b><br><span class="muted">${(c.top_words || []).join(", ")}</span><br>${int(c.size_train)} training prompts${c.main_source ? html` · mostly from ${c.main_source}` : ""}<br>Fails ${pctA(c.fail_rate_test, 1)} on test, ${pctA(c.fail_rate_train, 1)} on training${isNum(c.mean_score_train) ? html`<br>Average cheap-model score ${fix(c.mean_score_train, 2)} / 5` : ""}`,
    }));
    const maxV = Math.max(...list.flatMap((c) => [c.fail_rate_test, c.fail_rate_train]).filter(isNum));
    const dom = [0, C.niceDomain(0, maxV)[1]];
    C.rankBars($("clChart"), {
      items, initial: "test",
      metrics: [
        { key: "test", label: "Test prompts", better: "high", fmt: (v) => pctA(v, 1), domain: dom, refs: [{ v: D.fail_rate_test, label: "average" }],
          caption: "Share of each group's test prompts that the cheap model failed." },
        { key: "train", label: "Training prompts", better: "high", fmt: (v) => pctA(v, 1), domain: dom, refs: [{ v: D.fail_rate_train, label: "average" }],
          caption: "Share of each group's training prompts that the cheap model failed." },
      ],
    });
    const hi = cl.hierarchical || {}, db = cl.dbscan || {};
    const facts = [
      isNum(cl.k) ? html`<div class="fact"><b class="num">${cl.k}</b><span>K-Means groups, chosen by silhouette score</span></div>` : "",
      isNum(hi.silhouette_hierarchical) ? html`<div class="fact"><b class="num">${fix(hi.silhouette_hierarchical, 3)}</b><span>silhouette of Ward hierarchical clustering${isNum(hi.silhouette_kmeans) ? `, against ${fix(hi.silhouette_kmeans, 3)} for K-Means` : ""}${isNum(hi.sample_size) ? ` (${int(hi.sample_size)} prompt sample)` : ""}</span></div>` : "",
      isNum(hi.adjusted_rand_index) ? html`<div class="fact"><b class="num">${fix(hi.adjusted_rand_index, 2)}</b><span>adjusted Rand index: how much the two clusterings agree (1 = fully)</span></div>` : "",
      isNum(db.n_clusters) ? html`<div class="fact"><b class="num">${db.n_clusters}</b><span>DBSCAN clusters${isNum(db.noise_fraction) ? `; ${pctA(db.noise_fraction, 0)} of prompts were left as noise` : ""}</span></div>` : "",
    ].filter(Boolean);
    setHTML($("clFacts"), html`${facts}`);
    $("clFacts").hidden = !facts.length;
    return true;
  }

  /* =========================================================================================
     08 Association rules
     ========================================================================================= */
  function rules() {
    const A = M.association || {}, cl = (M.clustering || {}).clusters || [];
    const list = A.rules_to_fail || [];
    if (!list.length) return false;
    const words = {};
    cl.forEach((c) => { words[c.cluster] = (c.top_words || []).slice(0, 2).join(", "); });
    const pretty = (s) => {
      let m;
      if ((m = s.match(/^cluster=(\d+)$/))) return `cluster ${m[1]}${words[m[1]] ? ` (${words[m[1]]})` : ""}`;
      if ((m = s.match(/^length=(\w+)$/))) return `${m[1]} prompt`;
      if ((m = s.match(/^source=(.+)$/))) return `from ${m[1]}`;
      if (s === "has_math") return "has maths";
      if (s === "outcome=FAIL") return "cheap model fails";
      return s.replace(/^has_/, "has ").replace(/_/g, " ");
    };
    const maxL = Math.max(...list.map((r) => r.lift).filter(isNum));
    const base = A.base_fail_rate;
    setHTML($("rulesList"), html`${list.map((r, i) => html`<div class="rule" style="--d:${i}">
      <div class="rule-if"><span>IF</span>${(r.if || []).map((a, j) => html`${j ? html`<span>+</span>` : ""}<span class="chip">${pretty(a)}</span>`)}<span>THEN</span><span class="chip then">${(r.then || []).map(pretty).join(", ")}</span></div>
      <div class="lift"><div class="lift-top"><span>lift</span><b class="num">×${fix(r.lift, 2)}</b></div>
        <div class="lift-bar" style="--w:${(r.lift / maxL).toFixed(4)};--b:${(1 / maxL).toFixed(4)}"><i></i><u title="average"></u></div>
        <div class="lift-top"><span>fails ${pctA(r.confidence, 1)}${isNum(base) ? ` (avg ${pctA(base, 1)})` : ""}</span><span>support ${pctA(r.support, 1)}</span></div></div></div>`)}`);
    onVisible($("rulesList"), () => $("rulesList").classList.add("in"));
    const ap = A.apriori, nv = A.naive;
    if (ap && nv && isNum(ap.candidates_counted) && isNum(nv.candidates_counted)) {
      const mx = Math.max(ap.candidates_counted, nv.candidates_counted);
      const row = (k, o, cls) => html`<div class="meter"><div class="meter-top"><span><b>${k}</b> · ${isNum(o.seconds) ? `${o.seconds.toFixed(2)} s` : ""}</span><b class="num">${int(o.candidates_counted)} candidates counted</b></div><div class="meter-bar"><i class="${cls}" data-w="${o.candidates_counted / mx}" style="width:0"></i></div></div>`;
      setHTML($("algo"), html`${row("Apriori", ap, "a")}${row("Naive", nv, "w")}
        <div class="insight"><span>Same result${isNum(ap.frequent_itemsets) ? html` (<b>${int(ap.frequent_itemsets)}</b> frequent itemsets${isNum(A.min_support) ? `, minimum support ${pctA(A.min_support, 1)}` : ""})` : ""}, with <b>${(nv.candidates_counted / ap.candidates_counted).toFixed(1)}×</b> fewer candidates to count.</span></div>`);
      onVisible($("algo"), () => $("algo").querySelectorAll("[data-w]").forEach((i) => { i.style.width = `${Number(i.dataset.w) * 100}%`; }));
    } else $("algo").parentElement.hidden = true;
    return true;
  }

  /* =========================================================================================
     09 Regression
     ========================================================================================= */
  function regression() {
    const list = M.regression || [], bl = M.regression_baseline || {}, B = M.baselines || {};
    if (!list.length) return false;
    const items = list.map((r, i) => ({
      id: i, label: r.name.replace(/\s*\(.*?\)/g, ""), sub: (r.name.match(/\((.*?)\)/) || [])[1] || "",
      values: { r2: r.r2, rmse: r.rmse, mae: r.mae, r: r.pearson_r, apgr: r.apgr },
      tip: () => html`<b>${r.name}</b><br>R² ${fix(r.r2, 3)} · RMSE ${fix(r.rmse, 3)} · MAE ${fix(r.mae, 3)}<br>Correlation ${fix(r.pearson_r, 3)} · APGR ${fix(r.apgr, 3)}${r.note ? html`<br><span class="muted">${r.note}</span>` : ""}`,
    }));
    const v = (k) => list.map((r) => r[k]).filter(isNum);
    C.rankBars($("regChart"), {
      items, initial: "r2",
      metrics: [
        { key: "r2", label: "R²", better: "high", fmt: (x) => x.toFixed(3), domain: () => [0, C.niceDomain(0, Math.max(...v("r2")))[1]], refs: [{ v: bl.r2, label: "predict the average" }],
          caption: "Share of the score's variation each model explains. Predicting the average every time scores about 0." },
        { key: "rmse", label: "RMSE", better: "low", fmt: (x) => x.toFixed(3), domain: () => [0, Math.max(...v("rmse"), isNum(bl.rmse) ? bl.rmse : 0) * 1.05], refs: [{ v: bl.rmse, label: "predict the average" }],
          caption: "Typical size of the error, in score points (1 to 5). Lower is better." },
        { key: "mae", label: "MAE", better: "low", fmt: (x) => x.toFixed(3), domain: () => [0, Math.max(...v("mae"), isNum(bl.mae) ? bl.mae : 0) * 1.05], refs: [{ v: bl.mae, label: "predict the average" }],
          caption: "Average absolute error, in score points. Lower is better." },
        { key: "r", label: "Correlation", better: "high", fmt: (x) => x.toFixed(3), domain: [0, 1],
          caption: "Pearson correlation between the predicted and the real score." },
        { key: "apgr", label: "As a router", better: "high", fmt: (x) => x.toFixed(3), domain: () => [Math.min(isNum(B.random_apgr) ? B.random_apgr : 0.5, ...v("apgr")) - 0.05, isNum(B.oracle_apgr) ? B.oracle_apgr : 1],
          refs: [{ v: B.random_apgr, label: "random" }, { v: B.oracle_apgr, label: "perfect" }],
          caption: "APGR when the lowest predicted scores go to the strong model." },
      ],
    });
    return true;
  }

  /* =========================================================================================
     10 Features and activations
     ========================================================================================= */
  function more() {
    const F = M.features || {}, ab = F.ablation || {}, act = M.activation_functions || {};
    let any = false;
    const abRows = Object.keys(ab).filter((k) => isNum(ab[k].val_auc)).map((k) => ({
      label: k.replace(/_/g, " ").replace(/\+/g, " + "), sub: isNum(ab[k].n_features) ? `${int(ab[k].n_features)} features` : null,
      points: [{ v: ab[k].val_auc, cls: k === F.chosen ? "deployed" : "dim", label: k === F.chosen ? "chosen" : k.replace(/_/g, " ") }],
    }));
    if (abRows.length) {
      const all = abRows.map((r) => r.points[0].v);
      C.dotLine($("ablation"), { rows: abRows, domain: C.niceDomain(Math.min(...all) - 0.002, Math.max(...all) + 0.002, 4), fmt: (x) => x.toFixed(4), tickFmt: (x) => x.toFixed(3), axisLabel: "Validation AUC" });
      any = true;
    } else $("ablation").parentElement.hidden = true;
    const keys = Object.keys(act).filter((k) => isNum(act[k].val_auc));
    if (keys.length) {
      const nets = {};
      keys.forEach((k) => { const [net, fn] = k.split(/,\s*/); (nets[net] = nets[net] || []).push({ fn: fn || k, v: act[k].val_auc, ep: act[k].epochs }); });
      const rows = [];
      Object.keys(nets).forEach((net) => {
        const best = Math.max(...nets[net].map((x) => x.v));
        nets[net].forEach((x) => rows.push({ label: `${x.fn}`, sub: short(net), points: [{ v: x.v, cls: x.v === best ? "deployed" : "student", label: `${short(net)}, ${x.fn}`, tip: isNum(x.ep) ? `${x.ep} epochs` : null }] }));
      });
      const all = rows.map((r) => r.points[0].v);
      C.dotLine($("acts"), { rows, domain: C.niceDomain(Math.min(...all) - 0.003, Math.max(...all) + 0.003, 4), fmt: (x) => x.toFixed(4), tickFmt: (x) => x.toFixed(2), axisLabel: "Validation AUC (best per network in lilac)" });
      any = true;
    } else $("acts").parentElement.hidden = true;
    return any;
  }

  /* =========================================================================================
     11 Limits
     ========================================================================================= */
  function limits() {
    const R = M.router || {}, ft = M.finetune || {}, sweep = ft.sweep || [], hr = (H && H.router) || {};
    const L = [];
    L.push(["Strong answers count as perfect", `The judge graded the cheap model's answers using the strong model's answers as the reference, so strong-only scores ${isNum(R.quality_strong_only) ? fix(R.quality_strong_only, 1) : "full marks"} out of 5. Real strong answers are not perfect, so the savings are a little optimistic.`]);
    if (hr.cheap_model_in_data && hr.strong_model_in_data && H) {
      L.push(["The labels come from an older pair of models", `The training data compares ${hr.cheap_model_in_data} with ${hr.strong_model_in_data}. The live app routes between ${H.cheap_model} and ${H.strong_model}. Prompt difficulty tends to carry over, but we have not re-measured it for the new pair.`]);
    } else {
      L.push(["The labels come from an older pair of models", "The live app routes between newer models than the ones in the training data. Prompt difficulty tends to carry over, but it has not been re-measured."]);
    }
    L.push(["The judge is a language model too", "Each label is one grade from one judge, so some labels are wrong. That noise caps how well any router can do."]);
    if (isNum(ft.max_tokens)) L.push(["Long prompts are only read in part", `${isLight() ? "In the full router, the" : "The"} transformer reads the first ${ft.max_tokens} tokens of a prompt. Anything after that does not change the score.`]);
    // seed noise
    const nc = sweep.find((r) => /noise check/i.test(r.tag) && r.kind === "sweep");
    const twin = nc && sweep.find((r) => r !== nc && r.kind === "sweep" && r.base === nc.base && r.lr === nc.lr && r.batch_size === nc.batch_size && r.max_tokens === nc.max_tokens && r.epochs === nc.epochs && r.seed !== nc.seed && r.teacher == null);
    L.push(["Small differences are noise", twin ? `Changing only the random seed moved validation AUC by ${Math.abs(twin.val_auc_fail - nc.val_auc_fail).toFixed(4)}. Many differences between settings were no bigger, so close results are ties.` : "Changing only the random seed moves the scores a little, so close results are ties."]);
    L.push(["One setup per model in the size study", "Each large model was trained once with a standard setup. A separate search for each model could change the order."]);
    const s8 = sweep.filter((r) => r.kind === "distill" && /student of Qwen3-8B/.test(r.tag)).map((r) => r.val_auc_fail).filter(isNum);
    const sT = ft.teacher_name && sweep.filter((r) => r.kind === "distill" && new RegExp("student of " + ft.teacher_name.replace(/\./g, "\\.") + "(?![\\d.])(?!\\s*\\+)").test(r.tag)).map((r) => r.val_auc_fail).filter(isNum);
    if (s8.length && sT && sT.length && ft.teacher_name !== "Qwen3-8B") {
      L.push(["A stronger teacher stopped helping", `The best student of the Qwen3-8B teacher reached ${fix(Math.max(...s8), 4)} validation AUC, against ${fix(Math.max(...sT), 4)} for the ${ft.teacher_name} teacher. Past a point, the small student is the limit, not the teacher.`]);
    }
    const lt = R.light;
    if (lt && isNum(lt.test_auc) && isNum(R.auc)) L.push(isLight()
      ? ["This server runs the weaker light router", `To fit in very little memory, this server runs a light router without the transformer. Its test AUC is ${fix(lt.test_auc, 3)}, against ${fix(R.auc, 3)} for the full router that most chapters describe.`]
      : ["The light router is weaker", `For servers with little memory there is a light router without the transformer. Its test AUC is ${fix(lt.test_auc, 3)}, against ${fix(R.auc, 3)} for the full router.`]);
    L.push(["The test set was scored more than once", "Each new final setup was picked on validation data only, then scored on the test set once. The final router is the last validation choice, even where an earlier one scored a little higher on test."]);
    setHTML($("limitsList"), html`${L.map(([h, t], i) => html`<div class="limit"><h3>${h}</h3><p>${t}</p></div>`)}`);
    U.initReveals($("limitsList"));
    return true;
  }

  /* =========================================================================================
     TOC, scroll progress
     ========================================================================================= */
  function toc() {
    const secs = Array.from(document.querySelectorAll(".chapter")).filter((s) => !s.hidden);
    const ol = $("toc");
    ol.textContent = "";
    secs.forEach((s, i) => {
      const n = String(i + 1).padStart(2, "0");
      const num = s.querySelector(".ch-num");
      if (num) num.textContent = `Chapter ${n}`;
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `#${s.id}`;
      setHTML(a, html`<span>${n}</span>${s.dataset.title}`);
      li.appendChild(a);
      ol.appendChild(li);
    });
    const links = Array.from(ol.querySelectorAll("a"));
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => {
        es.forEach((e) => {
          if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === `#${e.target.id}`));
        });
      }, { rootMargin: "-35% 0px -60% 0px" });
      secs.forEach((s) => io.observe(s));
    }
    const bar = $("progress");
    const onScroll = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = `scaleX(${h > 0 ? clamp(window.scrollY / h) : 0})`;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* The black band under the header: the same facts the home page shows, for the router this server runs. */
  function ticker() {
    const track = $("tickerTrack");
    if (!track) return;
    const R = M.router || {}, light = isLight(), src = light ? (R.light || {}) : R, perf = src.performance || {}, h = hr();
    const apgr = firstNum(h.test_apgr, light ? src.test_apgr : R.apgr), auc = firstNum(h.test_auc, light ? src.test_auc : R.auc);
    const cpt50 = firstNum(h.cpt50, src.cpt50), mb = light ? perf.artifact_mb : perf.text_model_mb;
    const rows = get(M, "data.train_rows");
    const items = ["Reads the prompt first"];
    if (isNum(perf.single_p50_ms)) items.push(`About ${Math.round(perf.single_p50_ms)} ms per prompt on a CPU`);
    items.push("Cheap when it can");
    if (isNum(mb)) items.push(`${mb.toFixed(1)} MB of router`);
    if (isNum(rows)) items.push(`Trained on ${int(rows)} graded prompts`);
    items.push("Strong when it must");
    if (isNum(apgr)) items.push(`APGR ${apgr.toFixed(3)}`);
    if (isNum(cpt50)) items.push(`${Math.round(cpt50 * 100)}% strong calls wins back half the gap`);
    if (isNum(auc)) items.push(`AUC ${auc.toFixed(3)}`);
    items.push("No prompt left unread");
    track.textContent = "";
    [false, true].forEach((dup) => {
      const set = document.createElement("div");
      set.className = "ticker-set";
      if (dup) set.setAttribute("aria-hidden", "true");
      items.forEach((t) => { const s = document.createElement("span"); s.textContent = t; set.appendChild(s); });
      track.appendChild(set);
    });
  }

  async function boot() {
    const [m, h] = await Promise.allSettled([api("/api/metrics"), api("/api/health")]);
    if (m.status !== "fulfilled") {
      $("loadErr").hidden = false;
      $("loadErr").textContent = `Could not load the results: ${m.reason && m.reason.message ? m.reason.message : "unknown error"}`;
      document.querySelectorAll(".chapter").forEach((s) => { s.hidden = true; });
      return;
    }
    M = m.value;
    H = h.status === "fulfilled" ? h.value : null;
    try { ticker(); } catch (e) { /* the static band stays */ }
    safe("r-hero", hero);
    safe("curve", curve);
    safe("budget", budget);
    safe("leaderboard", leaderboard);
    safe("size", sizeStudy);
    safe("distill", distill);
    safe("matrix", matrix);
    safe("clusters", clusters);
    safe("rules", rules);
    safe("regression", regression);
    safe("more", more);
    safe("limits", limits);
    if (isLight()) ["curve", "budget", "matrix"].forEach(fullTag);
    setBudget(S.budget, "init");
    toc();
    U.initReveals();
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
