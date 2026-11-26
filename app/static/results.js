/* Results page: an editorial data story rendered entirely from GET /api/metrics (schema: scripts/08_summary.py).
   No number on this page is hard-coded: anything missing from the response renders as "-". */
"use strict";

(() => {
  const { html, raw, setHTML, isNum, pct, fix, int, get, clamp, api } = U;
  const arr = (x) => (Array.isArray(x) ? x : []);
  const obj = (x) => (x && typeof x === "object" && !Array.isArray(x) ? x : {});
  const ms = (x, d = 1) => (isNum(x) ? `${x.toFixed(d)} ms` : "-");
  const times = (a, b, d = 1) => (isNum(a) && isNum(b) && b !== 0 ? `${(a / b).toFixed(d)}×` : "-");
  const id = (() => { let n = 0; return (p) => `${p}${++n}`; })();

  /* ---------------- routing curve (hand-built SVG) ---------------- */
  function curveChart(s) {
    const c = obj(s.curves), r = obj(s.router), base = obj(s.baselines);
    const x = arr(c.x);
    const ok = (y) => Array.isArray(y) && y.length === x.length && y.every(isNum);
    if (x.length < 2 || !x.every(isNum)) return null;
    const series = [
      { key: "oracle", cls: "s-oracle", y: c.oracle, label: "Perfect router (knows the answers)", apgr: base.oracle_apgr, color: "var(--strong)", dash: "1.5 4" },
      { key: "random", cls: "s-rand", y: x, label: "Random router", apgr: base.random_apgr, color: "var(--muted)", dash: "6 5" },
      { key: "best", cls: "s-best", y: c.best_single, label: `Best single model${c.best_single_name ? ": " + c.best_single_name : ""}`, apgr: c.best_single_apgr, color: "var(--ink-2)" },
      { key: "final", cls: "s-final", y: c.final, label: "Final router", apgr: r.apgr, color: "var(--signal)", w: 3.2 },
    ].filter((sr) => ok(sr.y));
    if (!series.length) return null;
    const narrow = window.innerWidth < 640;
    const W = narrow ? 400 : 720, H = narrow ? 340 : 440, L = narrow ? 60 : 56, R = narrow ? 10 : 16, T = 16, B = 52;
    const ymax = Math.max(1, ...series.flatMap((sr) => sr.y));
    const sx = (v) => L + v * (W - L - R), sy = (v) => H - B - (Math.max(0, v) / ymax) * (H - B - T);
    const path = (ys) => ys.map((v, i) => `${i ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`).join("");
    let g = "";
    for (let i = 0; i <= 10; i += 2) {
      const v = i / 10;
      g += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${sy(0)}" y2="${sy(ymax)}"/><line x1="${sx(0)}" x2="${sx(1)}" y1="${sy(v)}" y2="${sy(v)}"/>`;
    }
    let labels = "";
    for (let i = 0; i <= 10; i += 2) {
      labels += `<text x="${sx(i / 10)}" y="${H - B + 18}" text-anchor="middle">${i * 10}%</text>`;
      labels += `<text x="${L - 10}" y="${sy(i / 10) + 4}" text-anchor="end">${i * 10}%</text>`;
    }
    labels += `<text class="axis-t" x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">share of prompts sent to the strong model →</text>`;
    labels += `<text class="axis-t" transform="translate(${narrow ? 10 : 14} ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">quality gap recovered →</text>`;
    const fin = series.find((sr) => sr.key === "final");
    const gain = fin ? `<path class="gain" d="${path(fin.y)}L${sx(x[x.length - 1])},${sy(x[x.length - 1])}${x.slice().reverse().map((v) => `L${sx(v).toFixed(1)},${sy(v).toFixed(1)}`).join("")}Z"/>` : "";
    const lines = series.map((sr) => `<path class="${sr.cls} ${sr.key === "final" ? "draw" : ""}" d="${path(sr.y)}"/>`).join("");
    const dots = series.map((sr) => `<circle class="hover-dot" data-k="${sr.key}" r="${sr.key === "final" ? 5 : 3.5}" fill="${sr.color}" stroke="var(--panel)" stroke-width="2"/>`).join("");
    const svgId = id("curve");
    const svg = `<svg class="chart" id="${svgId}" viewBox="0 0 ${W} ${H}" role="img" tabindex="0"
      aria-label="Routing curve: quality gap recovered against the share of prompts sent to the strong model. Use arrow keys to inspect points.">
      <g class="grid">${g}</g>${gain}${lines}${labels}
      <line class="hover-line" x1="0" x2="0" y1="${sy(0)}" y2="${sy(ymax)}"/>${dots}
      <rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" data-hit/></svg>`;
    const legend = series.slice().reverse().map((sr) => html`<span><svg viewBox="0 0 26 10" aria-hidden="true"><line x1="1" x2="25" y1="5" y2="5" stroke="${raw(sr.color)}" stroke-width="${sr.w || 2}" ${raw(sr.dash ? `stroke-dasharray="${sr.dash}"` : "")} stroke-linecap="round"/></svg>${sr.label} <b class="num">${fix(sr.apgr)}</b></span>`);
    const rows = x.map((v, i) => html`<tr><td class="n">${pct(v)}</td>${series.map((sr) => html`<td class="n">${pct(sr.y[i], 1)}</td>`)}</tr>`);
    const table = html`<details class="data-table"><summary>Show the curve as a table</summary><div class="table-scroll" style="max-height:320px;margin-top:10px">
      <table class="data"><thead><tr><th class="n">Strong share</th>${series.map((sr) => html`<th class="n">${sr.key === "best" ? "Best single" : sr.label.split(" (")[0]}</th>`)}</tr></thead><tbody>${rows}</tbody></table></div></details>`;

    const mount = (root) => {
      const el = root.querySelector("#" + svgId);
      if (!el) return;
      const p = el.querySelector(".draw");
      if (p) p.style.setProperty("--len", Math.ceil(p.getTotalLength()));
      const wrap = el.parentElement, tip = wrap.querySelector(".tip");
      const show = (i) => {
        i = Math.max(0, Math.min(x.length - 1, i));
        el.dataset.i = i;
        el.classList.add("hovering");
        const hx = sx(x[i]);
        const hl = el.querySelector(".hover-line");
        hl.setAttribute("x1", hx); hl.setAttribute("x2", hx);
        series.forEach((sr) => { const d = el.querySelector(`[data-k="${sr.key}"]`); d.setAttribute("cx", hx); d.setAttribute("cy", sy(sr.y[i])); });
        setHTML(tip, html`<div><b>${pct(x[i])}</b> of prompts to the strong model</div>${series.slice().reverse().map((sr) => html`<div><i style="background:${raw(sr.color)}"></i>${sr.key === "best" ? "Best single" : sr.label.split(" (")[0]}: <b>${pct(sr.y[i], 1)}</b></div>`)}`);
        const box = el.getBoundingClientRect(), k = box.width / W;
        const left = hx * k, tw = tip.offsetWidth;
        tip.style.left = `${Math.min(Math.max(0, left + 14), box.width - tw)}px`;
        tip.style.top = `${Math.max(0, sy(fin ? fin.y[i] : 0.5) * k - 90)}px`;
        tip.classList.add("on");
      };
      const hide = () => { el.classList.remove("hovering"); tip.classList.remove("on"); };
      const fromEvent = (e) => {
        const box = el.getBoundingClientRect();
        const vx = ((e.clientX - box.left) / box.width) * W;
        const v = (vx - L) / (W - L - R);
        let best = 0;
        x.forEach((xv, i) => { if (Math.abs(xv - v) < Math.abs(x[best] - v)) best = i; });
        show(best);
      };
      el.addEventListener("pointermove", fromEvent);
      el.addEventListener("pointerdown", fromEvent);
      el.addEventListener("pointerleave", hide);
      el.addEventListener("blur", hide);
      el.addEventListener("keydown", (e) => {
        const i = el.dataset.i ? +el.dataset.i : Math.round((x.length - 1) * 0.3);
        if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); show(i + 1); }
        else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); show(i - 1); }
        else if (e.key === "Escape") hide();
      });
      el.addEventListener("focus", () => show(el.dataset.i ? +el.dataset.i : Math.round((x.length - 1) * 0.3)));
    };
    return { tpl: html`<div class="chart-wrap">${raw(svg)}<div class="tip" aria-hidden="true"></div></div><div class="legend">${legend}</div>${table}`, mount };
  }

  /* ---------------- bars ---------------- */
  function barChart(rows, { lo, hi, marks = [], digits = 3, fmt }) {
    const span = hi - lo || 1;
    const posOf = (v) => `${(clamp((v - lo) / span) * 100).toFixed(2)}%`;
    const f = fmt || ((v) => fix(v, digits));
    return html`<div class="bars">${rows.map((r, i) => html`<div class="bar-row ${r.hl ? "hl" : ""}">
        <span class="name" title="${r.name}">${r.name}</span>
        <span class="bar-track">${isNum(r.v) ? html`<span class="bar-fill" style="width:${posOf(r.v)};animation-delay:${i * 35}ms"></span>` : ""}
          ${marks.filter((m) => isNum(m.v)).map((m) => html`<span class="bar-mark ${m.solid ? "solid" : ""}" style="left:${posOf(m.v)}" title="${m.label}: ${f(m.v)}"></span>`)}</span>
        <span class="v">${f(r.v)}</span></div>`)}
      <div class="bar-axis" aria-hidden="true"><span></span><span class="ax"><span>${f(lo)}</span><span>${f(lo + span / 2)}</span><span>${f(hi)}</span></span><span></span></div></div>`;
  }
  const domain = (vals, pad = 0.02, step = 0.05) => {
    const v = vals.filter(isNum);
    if (!v.length) return { lo: 0, hi: 1 };
    return { lo: Math.max(0, Math.floor((Math.min(...v) - pad) / step) * step), hi: Math.min(1, Math.ceil((Math.max(...v) + pad / 2) / step) * step) };
  };

  /* ---------------- sortable table ---------------- */
  function sortableTable(cols, rows, { hl } = {}) {
    const tid = id("tbl");
    const head = html`<thead><tr>${cols.map((c, j) => html`<th class="${c.num ? "n" : ""}" ${raw(c.sort ? `aria-sort="none"` : "")} scope="col">${c.sort ? html`<button type="button" data-col="${j}">${c.label}</button>` : c.label}</th>`)}</tr></thead>`;
    const body = (rs) => html`${rs.map((r) => html`<tr class="${hl && hl(r) ? "hl" : ""}">${cols.map((c) => html`<td class="${c.num ? "n" : ""}">${c.cell(r)}</td>`)}</tr>`)}`;
    const tpl = html`<div class="table-scroll"><table class="data" id="${tid}">${head}<tbody>${body(rows)}</tbody></table></div>`;
    const mount = (root) => {
      const t = root.querySelector("#" + tid);
      if (!t) return;
      let state = { j: -1, dir: -1 };
      t.querySelectorAll("th button").forEach((b) => b.addEventListener("click", () => {
        const j = +b.dataset.col;
        state = { j, dir: state.j === j ? -state.dir : (cols[j].num ? -1 : 1) };
        const key = cols[j].sort;
        const sorted = rows.slice().sort((a, b2) => {
          const va = key(a), vb = key(b2);
          if (va == null || (typeof va === "number" && !isNum(va))) return 1;
          if (vb == null || (typeof vb === "number" && !isNum(vb))) return -1;
          return (va > vb ? 1 : va < vb ? -1 : 0) * state.dir;
        });
        t.querySelectorAll("th[aria-sort]").forEach((th) => th.setAttribute("aria-sort", "none"));
        b.parentElement.setAttribute("aria-sort", state.dir > 0 ? "ascending" : "descending");
        setHTML(t.querySelector("tbody"), body(sorted));
      }));
    };
    return { tpl, mount };
  }

  /* ---------------- sections ---------------- */
  function render(s, h) {
    const mounts = [];
    const use = (part) => { if (part && part.mount) mounts.push(part.mount); return part ? part.tpl : ""; };
    const r = obj(s.router), base = obj(s.baselines), d = obj(s.data), perf = obj(r.performance), feats = obj(s.features);
    const hr = obj(h && h.router);
    const cheapName = hr.cheap_model_in_data || "the cheap model";
    const strongName = hr.strong_model_in_data || "the strong model";
    const at50 = obj(r.at_50pct);
    const clfs = arr(s.classifiers);
    const comps = arr(r.components);

    /* hero */
    const hero = html`<section class="r-hero" aria-labelledby="rTitle">
      <div class="eyebrow">Results · held-out test set${s.built ? html` · built ${s.built}` : ""}</div>
      <h1 id="rTitle">How much of the strong model's quality can a <em>classical</em> router buy back?</h1>
      <p class="standfirst">Trained on <b>${int(d.train_rows)}</b> prompts and tested on <b>${int(d.test_rows)}</b> it never saw,
        the router recovers <b>${pct(at50.pgr)}</b> of the quality gap between ${cheapName} and ${strongName}
        while sending <b>${pct(at50.strong_share)}</b> of prompts to ${strongName}.</p>
      <div class="kpis">
        ${kpi(fix(r.apgr), "APGR of the final router", `random ${fix(base.random_apgr)} · perfect ${fix(base.oracle_apgr)}`, true)}
        ${kpi(pct(r.cpt80), "strong-model calls needed to recover 80% of the gap", `50% of the gap: ${pct(r.cpt50)}`)}
        ${kpi(pct(at50.pgr), `of the gap recovered at a ${pct(at50.strong_share)} budget`, `average score ${fix(at50.quality, 2)} (1-5)`)}
        ${kpi(fix(r.auc), "ROC-AUC for spotting a poor cheap answer", `best single classifier ${fix(c50(clfs, "test_auc"))}`)}
        ${kpi(isNum(perf.single_p50_ms) ? raw(html`${fix(perf.single_p50_ms, 1)}<small>ms</small>`.s) : "-", "median time to route one prompt", `p95 ${ms(perf.single_p95_ms)}`)}
        ${kpi(int(d.train_rows), "training prompts", `${int(d.test_rows)} held-out test prompts`)}
        ${kpi(pct(d.fail_rate_test, 1), "of test prompts get a poor cheap answer", `train: ${pct(d.fail_rate_train, 1)}`)}
        ${kpi(isNum(perf.artifact_mb) ? raw(html`${fix(perf.artifact_mb, 1)}<small>MB</small>`.s) : "-", "deployed router artifact", `${comps.length || "-"} component model${comps.length === 1 ? "" : "s"} + stacker`)}
      </div></section>`;

    const chapters = [];
    const chap = (key, title, lede, body) => chapters.push({ key, title, tpl: html`<section class="chapter" id="${key}" aria-labelledby="${key}-h">
      <header><span class="no" aria-hidden="true">${String(chapters.length + 1).padStart(2, "0")}</span><h2 id="${key}-h">${title}</h2>${lede ? html`<p>${lede}</p>` : ""}</header>${body}</section>` });

    /* 01 problem */
    const okShare = isNum(d.fail_rate_test) ? 1 - d.fail_rate_test : null;
    chap("problem", "Why route at all?",
      html`Strong LLMs cost far more per call than small ones, yet most prompts don't need them. A judge scored ${cheapName}'s answers from 1 to 5; a score of 3 or less counts as a poor answer.`,
      html`<div class="grid-2">
        <div class="card pad">
          <div class="card-title"><h3>Test prompts by ${cheapName}'s answer quality</h3></div>
          <div class="split" style="height:26px;border-radius:8px;margin:6px 0 10px" role="img" aria-label="${pct(okShare, 1)} good, ${pct(d.fail_rate_test, 1)} poor">
            <i class="bg-cheap" style="flex-grow:${isNum(okShare) ? okShare : 0}"></i><i class="bg-strong" style="flex-grow:${isNum(d.fail_rate_test) ? d.fail_rate_test : 0}"></i></div>
          <div class="legend"><span><i class="legend-key"><i class="bg-cheap"></i></i>good (4-5): <b class="num">${pct(okShare, 1)}</b></span>
            <span><i class="legend-key"><i class="bg-strong"></i></i>poor (1-3): <b class="num">${pct(d.fail_rate_test, 1)}</b></span></div>
          <p class="cap">The job is to find that minority before any model answers, from the prompt text alone.</p>
        </div>
        <div class="card pad">
          <div class="card-title"><h3>The two extremes (average score, 1-5)</h3></div>
          <div class="mini-stats">
            <div><b class="c-cheap">${fix(r.quality_cheap_only, 2)}</b><span>always ${cheapName}</span></div>
            <div><b class="c-strong">${fix(r.quality_strong_only, 2)}</b><span>always ${strongName}</span></div>
          </div>
          <p class="cap">The difference between these two is the "quality gap". A router's score is the share of that gap it recovers for a given share of strong-model calls.</p>
        </div></div>`);

    /* 02 routing curve */
    const curve = curveChart(s);
    const budgetRows = [r.at_20pct, r.at_30pct, r.at_50pct].map(obj);
    const bc = obj(r.budget_check);
    const bcKeys = Object.keys(bc).filter((k) => isNum(bc[k])).sort((a, b) => parseFloat(a) - parseFloat(b));
    chap("curve", "The routing curve",
      "Each point shows how much of the quality gap is recovered when a given share of prompts goes to the strong model. Further up and to the left is better. The shaded band is what the router adds over picking prompts at random.",
      html`${isNum(at50.pgr) ? html`<p class="callout">At a <b>${pct(at50.strong_share)}</b> budget the final router recovers <b>${pct(at50.pgr, 1)}</b> of the gap; <b>${pct(r.cpt80)}</b> of strong-model calls are enough for 80%.</p>` : ""}
      <div class="grid-2">
        <div class="card pad">${curve ? use(curve) : html`<p class="muted">Curve data not available.</p>`}</div>
        <div class="stack">
          <div class="card pad"><div class="card-title"><h3>What the budget buys</h3></div>
            <div class="table-scroll"><table class="data compact"><thead><tr><th>Strong share</th><th class="n">Gap recovered</th><th class="n">Avg score</th><th class="n">Calls saved</th></tr></thead><tbody>
              <tr><td>0% (cheap only)</td><td class="n">${pct(0)}</td><td class="n">${fix(r.quality_cheap_only, 2)}</td><td class="n">${pct(1)}</td></tr>
              ${budgetRows.map((p) => html`<tr><td>${pct(p.strong_share)}</td><td class="n">${pct(p.pgr, 1)}</td><td class="n">${fix(p.quality, 2)}</td><td class="n">${isNum(p.strong_share) ? pct(1 - p.strong_share) : "-"}</td></tr>`)}
              <tr><td>100% (strong only)</td><td class="n">${pct(1)}</td><td class="n">${fix(r.quality_strong_only, 2)}</td><td class="n">${pct(0)}</td></tr>
            </tbody></table></div>
            <p class="cap">Gap recovered is 0% and 100% at the two extremes by definition.</p></div>
          <div class="card pad"><div class="card-title"><h3>Does the budget hold on unseen data?</h3></div>
            ${bcKeys.length ? budgetCheck(bc, bcKeys) : html`<p class="muted small">-</p>`}
            <p class="cap">Thresholds were set on validation data. Bars show the share of test prompts actually sent to the strong model for each target.</p></div>
        </div></div>`);

    /* 03 models */
    const apgrs = clfs.map((m) => m.apgr).concat([r.apgr, base.random_apgr]);
    const dm = domain(apgrs);
    const sortedClf = clfs.slice().sort((a, b) => (isNum(b.apgr) ? b.apgr : -1) - (isNum(a.apgr) ? a.apgr : -1));
    const barRows = [{ name: "Final router (stacked)", v: r.apgr, hl: true }].concat(sortedClf.map((m) => ({ name: m.name, v: m.apgr })));
    const cols = [
      { label: "Model", cell: (m) => html`<b>${m.name}</b>${comps.includes(m.name) ? html`<span class="sub c-signal">component of the final router</span>` : ""}`, sort: (m) => String(m.name || "") },
      { label: "Best settings", cell: (m) => html`<code>${m.params || "-"}</code>` },
      { label: "Val AUC", num: true, cell: (m) => fix(m.val_auc), sort: (m) => m.val_auc },
      { label: "Test AUC", num: true, cell: (m) => fix(m.test_auc), sort: (m) => m.test_auc },
      { label: "APGR", num: true, cell: (m) => fix(m.apgr), sort: (m) => m.apgr },
      { label: "CPT80", num: true, cell: (m) => pct(m.cpt80), sort: (m) => m.cpt80 },
      { label: "Precision", num: true, cell: (m) => fix(m.precision), sort: (m) => m.precision },
      { label: "Recall", num: true, cell: (m) => fix(m.recall), sort: (m) => m.recall },
      { label: "Specificity", num: true, cell: (m) => fix(m.specificity), sort: (m) => m.specificity },
      { label: "F1", num: true, cell: (m) => fix(m.f1), sort: (m) => m.f1 },
      { label: "Accuracy", num: true, cell: (m) => fix(m.accuracy), sort: (m) => m.accuracy },
      { label: "Fit (s)", num: true, cell: (m) => fix(m.fit_seconds, 1), sort: (m) => m.fit_seconds },
      { label: "ms / 1k", num: true, cell: (m) => fix(m.predict_ms_per_1k, 1), sort: (m) => m.predict_ms_per_1k },
    ];
    const tbl = sortableTable(cols, sortedClf, { hl: (m) => m.name === get(s, "curves.best_single_name") });
    const cls30 = obj(r.classification_at_30pct), cm = obj(cls30.confusion_matrix);
    const act = Object.entries(obj(s.activation_functions));
    chap("models", "Every classifier, tuned and compared",
      "Hyper-parameters were chosen on a validation split of the training data; each model's best setting was then scored once on the untouched test set. Precision, recall, specificity and F1 use the threshold that sends 30% of validation prompts to the strong model.",
      html`<div class="grid-2">
        <div class="card pad"><div class="card-title"><h3>APGR on the test set</h3><span class="small muted">dashed line: random router</span></div>
          ${barChart(barRows, { lo: dm.lo, hi: dm.hi, marks: [{ v: base.random_apgr, label: "Random" }] })}
          <p class="cap">The axis starts at ${fix(dm.lo, 2)}, not zero, so small differences are visible.</p></div>
        <div class="stack">
          <div class="card pad"><div class="card-title"><h3>The final router</h3></div>
            <p class="small" style="margin:0 0 10px">${comps.length ? html`${comps.map((c, i) => html`${i ? (i === comps.length - 1 ? " and " : ", ") : ""}<b>${c}</b>`)}, combined by a logistic-regression stacker.` : "-"}</p>
            <div class="mini-stats">
              <div><b>${fix(cls30.precision)}</b><span>precision</span></div><div><b>${fix(cls30.recall_sensitivity)}</b><span>recall</span></div>
              <div><b>${fix(cls30.specificity)}</b><span>specificity</span></div><div><b>${fix(cls30.f1)}</b><span>F1</span></div>
            </div></div>
          <div class="card pad"><div class="card-title"><h3>Confusion matrix at a 30% budget</h3></div>
            <div class="cm" role="table" aria-label="Confusion matrix">
              <span class="h" role="columnheader"></span><span class="h" role="columnheader">sent to cheap</span><span class="h" role="columnheader">sent to strong</span>
              <span class="h" role="rowheader">cheap was fine</span>
              <div class="cell ok" role="cell"><b>${int(cm.tn)}</b><span>money saved</span></div>
              <div class="cell bad" role="cell"><b>${int(cm.fp)}</b><span>paid for nothing</span></div>
              <span class="h" role="rowheader">cheap failed</span>
              <div class="cell bad" role="cell"><b>${int(cm.fn)}</b><span>missed failures</span></div>
              <div class="cell hit" role="cell"><b>${int(cm.tp)}</b><span>failures rescued</span></div>
            </div></div>
        </div></div>
      <div class="card pad" style="margin-top:18px"><div class="card-title"><h3>Full comparison</h3><span class="small muted">click a column to sort · highlighted: best single model by APGR</span></div>${use(tbl)}</div>
      ${act.length ? html`<div class="card pad" style="margin-top:18px"><div class="card-title"><h3>Neural network: activation functions</h3><span class="small muted">validation AUC</span></div>
        ${barChart(act.map(([k, v]) => ({ name: `${k} · ${isNum(obj(v).epochs) ? obj(v).epochs + " epochs" : "-"}`, v: obj(v).val_auc })), domain(act.map(([, v]) => obj(v).val_auc), 0.01, 0.01))}</div>` : ""}`);

    /* 04 regression */
    const reg = arr(s.regression), rb = obj(s.regression_baseline);
    const bestR2 = Math.max(...reg.map((m) => m.r2).filter(isNum));
    chap("regression", "Regression: predicting the score itself",
      "Instead of a yes/no label, these models predict the 1-5 quality score. Used as routers, they send the prompts with the lowest predicted score to the strong model, so they can be compared on the same routing metric.",
      html`<div class="card pad"><div class="table-scroll"><table class="data"><thead><tr><th>Model</th><th class="n">RMSE</th><th class="n">MAE</th><th class="n">Pearson r</th><th class="n">R²</th><th class="n">APGR as router</th></tr></thead><tbody>
        ${reg.map((m) => html`<tr class="${m.r2 === bestR2 ? "hl" : ""}"><td>${m.name}${m.note ? html`<span class="sub">${m.note}</span>` : ""}</td><td class="n">${fix(m.rmse)}</td><td class="n">${fix(m.mae)}</td><td class="n">${fix(m.pearson_r)}</td><td class="n">${fix(m.r2)}</td><td class="n">${fix(m.apgr)}</td></tr>`)}
        <tr><td><span class="muted">Baseline: always predict the mean</span></td><td class="n">${fix(rb.rmse)}</td><td class="n">${fix(rb.mae)}</td><td class="n">${fix(rb.pearson_r)}</td><td class="n">${fix(rb.r2)}</td><td class="n">-</td></tr>
      </tbody></table></div>
      <p class="cap">Highlighted: highest R². For comparison, the final classification router's APGR is ${fix(r.apgr)}.</p></div>`);

    /* 05 clustering */
    const cl = obj(s.clustering), clusters = arr(cl.clusters).slice().sort((a, b) => (b.fail_rate_train || 0) - (a.fail_rate_train || 0));
    const hi = obj(cl.hierarchical), db = obj(cl.dbscan);
    const maxFail = Math.max(...clusters.map((c) => c.fail_rate_train).filter(isNum), isNum(d.fail_rate_train) ? d.fail_rate_train : 0, 1e-9);
    const scale = Math.min(1, Math.ceil((maxFail * 1.1) * 10) / 10);
    const clusterRows = clusters.map((c) => {
      const hot = isNum(c.fail_rate_train) && isNum(d.fail_rate_train) && c.fail_rate_train > d.fail_rate_train;
      return html`<div class="cl ${hot ? "hot" : ""}">
        <span class="id">${isNum(c.cluster) ? "#" + c.cluster : "-"}</span>
        <div><div class="words">${arr(c.top_words).slice(0, 7).map((w) => html`<span>${w}</span>`)}</div>
          <div class="meta">${int(c.size_train)} prompts · mostly ${c.main_source || "-"} · mean score ${fix(c.mean_score_train, 2)}</div></div>
        <div class="rate"><div class="top"><span>fails</span><b>${pct(c.fail_rate_train, 1)}</b></div>
          <span class="bar-track">${isNum(c.fail_rate_train) ? html`<span class="bar-fill" style="width:${(clamp(c.fail_rate_train / scale) * 100).toFixed(1)}%"></span>` : ""}
            ${isNum(d.fail_rate_train) ? html`<span class="bar-mark solid" style="left:${(clamp(d.fail_rate_train / scale) * 100).toFixed(1)}%" title="overall failure rate ${pct(d.fail_rate_train, 1)}"></span>` : ""}</span>
          <div class="top"><span>test: ${pct(c.fail_rate_test, 1)}</span><span>0-${pct(scale)}</span></div></div></div>`;
    });
    chap("clustering", "Which kinds of prompt fail?",
      html`K-Means grouped the training prompts into <b>${isNum(cl.k) ? cl.k : "-"}</b> types (k chosen by silhouette score) on the SVD-reduced features. Each type is named by its top words. Violet types fail more often than the overall rate of ${pct(d.fail_rate_train, 1)} (the black tick).`,
      html`<div class="card pad"><div class="card-title"><h3>Prompt types, riskiest first</h3><span class="small muted">failure rate on train</span></div>
          <div class="clusters">${clusterRows.length ? clusterRows : html`<p class="muted">-</p>`}</div></div>
        <div class="grid-2 even" style="margin-top:18px">
          <div class="card pad"><div class="card-title"><h3>Cross-checks</h3></div>
            <div class="mini-stats">
              <div><b>${fix(hi.adjusted_rand_index, 2)}</b><span>adjusted Rand index, Ward hierarchical vs K-Means (${int(hi.sample_size)}-prompt sample)</span></div>
              <div><b>${fix(hi.silhouette_kmeans, 3)}</b><span>silhouette, K-Means</span></div>
              <div><b>${fix(hi.silhouette_hierarchical, 3)}</b><span>silhouette, hierarchical</span></div>
              <div><b>${isNum(db.n_clusters) ? db.n_clusters : "-"}</b><span>DBSCAN dense groups of near-template prompts</span></div>
              <div><b>${pct(db.noise_fraction)}</b><span>marked as noise by DBSCAN (eps ${fix(db.eps, 2)})</span></div>
              <div><b>${pct(db.fail_rate_noise, 1)}</b><span>failure rate among DBSCAN noise</span></div>
            </div></div>
          ${figure("cluster_failure_rates.png", "Cheap-model failure rate per K-Means cluster")}
        </div>`);

    /* 06 association */
    const as = obj(s.association), ap = obj(as.apriori), nv = obj(as.naive), rules = arr(as.rules_to_fail);
    const clusterWords = new Map(arr(cl.clusters).map((c) => [String(c.cluster), arr(c.top_words).slice(0, 3).join(", ")]));
    const item = (t) => {
      const m = String(t).match(/^cluster=(.+)$/);
      if (m) return html`<span class="item">type #${m[1]}${clusterWords.has(m[1]) ? html` <small>(${clusterWords.get(m[1])})</small>` : ""}</span>`;
      return html`<span class="item">${String(t).replace(/_/g, " ").replace("=", ": ")}</span>`;
    };
    const maxLift = Math.max(...rules.map((x) => x.lift).filter(isNum), 1);
    const candMax = Math.max(ap.candidates_counted || 0, nv.candidates_counted || 0, 1);
    chap("association", "Association rules: what travels with failure",
      html`Each prompt became a transaction of properties (length, source, prompt type, code, maths…) plus its outcome. Apriori, written from scratch, found the frequent itemsets (minimum support ${fix(as.min_support, 3)}); rules ending in "cheap model fails" are ranked by lift.`,
      html`<div class="grid-2 even" style="margin-bottom:18px">
        <div class="card pad"><div class="card-title"><h3>Apriori vs naive enumeration</h3><span class="small muted">candidate itemsets counted</span></div>
          ${barChart([{ name: "Apriori", v: ap.candidates_counted, hl: true }, { name: "Naive", v: nv.candidates_counted }], { lo: 0, hi: candMax, fmt: (v) => int(v) })}
          <p class="cap">Both found ${int(ap.frequent_itemsets)}${nv.frequent_itemsets !== ap.frequent_itemsets ? html` / ${int(nv.frequent_itemsets)}` : ""} frequent itemsets; the naive method counted ${times(nv.candidates_counted, ap.candidates_counted)} as many candidates (${fix(ap.seconds, 2)} s vs ${fix(nv.seconds, 2)} s).</p></div>
        <div class="card pad"><div class="card-title"><h3>Reading a rule</h3></div>
          <p class="small" style="margin:0">Lift = confidence ÷ base failure rate (${pct(as.base_fail_rate, 1)}). A lift of 2 means the cheap model fails twice as often as usual when the prompt has those properties. Support is the share of all prompts that match the whole rule.</p></div>
      </div>
      <div class="rules">${rules.length ? rules.map((x) => html`<article class="rule">
          <div class="if"><span class="k">IF</span>${arr(x.if).map(item)}<span class="k">THEN</span><span class="item then">cheap model fails</span></div>
          <div class="lift" title="lift ${fix(x.lift, 2)}"><i style="width:${isNum(x.lift) ? ((x.lift / maxLift) * 100).toFixed(1) : 0}%"></i></div>
          <div class="nums"><span><b>${fix(x.lift, 2)}</b>lift</span><span><b>${pct(x.confidence, 1)}</b>confidence</span><span><b>${pct(x.support, 2)}</b>support</span></div>
        </article>`) : html`<p class="muted">-</p>`}</div>`);

    /* 07 data */
    const abl = Object.entries(obj(feats.ablation));
    const dmA = domain(abl.map(([, v]) => obj(v).val_auc), 0.01, 0.01);
    chap("data", "Data and pre-processing",
      `Prompts from the RouteLLM dataset, cleaned and split before anything was trained. The feature set was picked by an ablation on the training data only; the test set was never used for tuning.`,
      html`<div class="grid-2 even">
        <div class="card pad"><div class="card-title"><h3>The data</h3></div>
          <div class="mini-stats">
            <div><b>${int(d.train_rows)}</b><span>training prompts</span></div>
            <div><b>${int(d.test_rows)}</b><span>test prompts</span></div>
            <div><b>${int(d.removed)}</b><span>rows removed (empty, duplicate or overlapping the test set)</span></div>
            <div><b>${int(feats.n_features)}</b><span>features: word + character TF-IDF + prompt statistics</span></div>
            <div><b>${isNum(feats.svd_components) ? feats.svd_components : "-"}</b><span>SVD dimensions for dense models (${pct(feats.svd_explained_variance, 1)} of variance)</span></div>
            <div><b>${pct(d.fail_rate_train, 1)}</b><span>poor cheap answers in training</span></div>
          </div></div>
        <div class="card pad"><div class="card-title"><h3>Feature-set ablation</h3><span class="small muted">validation AUC</span></div>
          ${abl.length ? barChart(abl.map(([k, v]) => ({ name: `${k.replace(/_/g, " ")}${k === feats.chosen ? " ✓" : ""}`, v: obj(v).val_auc, hl: k === feats.chosen })), dmA) : html`<p class="muted">-</p>`}
          <p class="cap">✓ marks the configuration used. Axis starts at ${fix(dmA.lo, 2)}.</p></div>
      </div>`);

    /* 08 figures */
    chap("figures", "Figures from the pipeline", "Static plots written by the training scripts. Any figure that hasn't been generated yet is simply left out.",
      html`<div class="figs">
        ${figure("routing_curve.png", "Routing curve on the held-out test set")}
        ${figure("roc_curves.png", "ROC curves for every model and the final router")}
        ${figure("model_comparison.png", "APGR for every model")}
        ${figure("kmeans_selection.png", "Choosing k for K-Means")}
        ${figure("cluster_failure_rates.png", "Failure rate per prompt type")}
      </div><p class="cap" id="noFigs" hidden>No figures are available yet.</p>`);

    const toc = html`<nav class="toc" aria-label="Sections"><div class="eyebrow">Contents</div><ol>
      ${chapters.map((c, i) => html`<li><a href="#${c.key}"><span>${String(i + 1).padStart(2, "0")}</span>${c.title}</a></li>`)}</ol></nav>`;

    const root = document.getElementById("root");
    setHTML(root, html`${hero}<div class="r-layout">${toc}<div>${chapters.map((c) => c.tpl)}
      <p class="cap" style="margin-top:36px">Caveat: the strong model's own answers are treated as perfect (score 5), because the judge used them as the reference.
      ${s.built ? html`Results built ${s.built}.` : ""}</p></div></div>`);
    mounts.forEach((m) => m(root));
    wireFigures(root);
    wireToc(root);
    document.getElementById("main").setAttribute("aria-busy", "false");
    if (s.built) document.getElementById("builtNote").textContent = `RouteWise · results built ${s.built}`;
  }

  function c50(clfs, key) { const v = clfs.map((m) => m[key]).filter(isNum); return v.length ? Math.max(...v) : null; }

  function kpi(n, label, sub, hl) {
    return html`<div class="kpi ${hl ? "hl" : ""}"><div class="n">${n}</div><div class="l">${label}</div>${sub ? html`<div class="s">${sub}</div>` : ""}</div>`;
  }

  function budgetCheck(bc, keys) {
    return html`<div class="bars">${keys.map((k) => {
      const target = parseFloat(k) / 100, got = bc[k];
      return html`<div class="bar-row" style="grid-template-columns:44px minmax(0,1fr) 54px"><span class="name num">${k}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${(clamp(got) * 100).toFixed(1)}%;background:var(--strong);opacity:.7"></span>
        ${isNum(target) ? html`<span class="bar-mark solid" style="left:${(clamp(target) * 100).toFixed(1)}%" title="target ${k}"></span>` : ""}</span>
        <span class="v">${pct(got, 1)}</span></div>`;
    })}</div>`;
  }

  function figure(name, caption) {
    return html`<figure class="card fig" data-fig><a href="/figures/${name}" target="_blank" rel="noopener"><img src="/figures/${name}" alt="${caption}" loading="lazy" width="800" height="600"></a><figcaption>${caption}</figcaption></figure>`;
  }

  function wireFigures(root) {
    root.querySelectorAll("figure[data-fig] img").forEach((img) => {
      const gone = () => {
        const f = img.closest("figure");
        const gal = f.parentElement;
        f.remove();
        if (gal.classList.contains("figs") && !gal.querySelector("figure")) { const n = document.getElementById("noFigs"); if (n) n.hidden = false; }
      };
      if (img.complete && img.naturalWidth === 0) gone();
      else { img.addEventListener("error", gone); img.addEventListener("load", () => { img.removeAttribute("height"); }); }
    });
  }

  function wireToc(root) {
    const links = [...root.querySelectorAll(".toc a")];
    if (!("IntersectionObserver" in window) || !links.length) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => { const on = a.getAttribute("href") === "#" + e.target.id; a.classList.toggle("on", on); if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    root.querySelectorAll(".chapter").forEach((c) => io.observe(c));
  }

  function fail(err) {
    const missing = err && err.status === 404;
    setHTML(document.getElementById("root"), html`<section class="state">
      <div class="eyebrow">Results</div>
      <h2>${missing ? "Results haven't been built yet" : "Couldn't load the results"}</h2>
      <p>${err && err.message ? err.message : "Unknown error"}</p>
      ${missing ? html`<p>Run <code>python scripts/run_all.py</code> to train the models and write <code>reports/summary.json</code>.</p>` : ""}
      <p><a class="btn" href="/results">Try again</a></p></section>`);
    document.getElementById("main").setAttribute("aria-busy", "false");
  }

  Promise.all([api("/api/metrics"), api("/api/health").catch(() => null)])
    .then(([m, h]) => {
      try { render(m, h); } catch (e) { console.error(e); fail(new Error("The metrics file has an unexpected shape: " + e.message)); }
    })
    .catch(fail);
})();
