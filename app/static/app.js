/* Home page: chat + live routing visualisation. All user/model text goes through U.html (auto-escaped) or textContent. */
"use strict";

(() => {
  const { html, raw, setHTML, pct, int, isNum, clamp, api, reduced } = U;
  const $ = (id) => document.getElementById(id);

  const EXAMPLES = [
    ["Fact", "What is the capital of Australia?"],
    ["Creative", "Write a haiku about autumn rain."],
    ["Maths proof", "Prove that the square root of 2 is irrational."],
    ["Code", "Write a Python function that merges overlapping intervals and explain its time complexity."],
    ["Translation", "Translate 'good morning, how are you?' into French."],
    ["Reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step."],
  ];
  const PROBE = "budget probe";
  const MAX = 4000;

  const STAT_LABELS = {
    log_chars: "prompt length", log_words: "word count", n_lines: "number of lines", avg_word_len: "average word length",
    frac_digit: "share of digits", frac_upper: "share of capitals", frac_punct: "punctuation density",
    frac_non_ascii: "non-English characters", n_question_marks: "question marks", starts_wh_question: "starts with a question word",
    starts_imperative: "starts with a command", code_markers: "code syntax", code_words: "programming words",
    math_markers: "maths markers", n_numbers: "numbers", has_url: "contains a link", creative_words: "creative-writing words",
    constraint_words: "constraints (must, exactly…)", reasoning_words: "reasoning words", translate: "translation request",
    mcq_options: "multiple-choice options", has_context_block: "context block (Input:, Passage:)", n_sentences: "number of sentences",
    frac_space: "share of whitespace",
  };

  /** Turn "word: translate", "chars: 'def '", "stat: log_chars" into a readable {kind, label, code}. */
  function featureLabel(f) {
    const s = String(f);
    let m;
    if ((m = s.match(/^word:\s?([\s\S]*)$/))) {
      const w = m[1].trim();
      return { kind: /\s/.test(w) ? "phrase" : "word", label: w || m[1], code: false };
    }
    if ((m = s.match(/^chars:\s?'([\s\S]*)'$/)) || (m = s.match(/^chars:\s?([\s\S]*)$/))) {
      const vis = m[1].replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
      return { kind: "chars", label: vis, code: true, title: `character piece "${m[1]}"` };
    }
    if ((m = s.match(/^stat:\s?([\s\S]*)$/))) {
      const k = m[1].trim();
      return { kind: "signal", label: STAT_LABELS[k] || k.replace(/_/g, " "), code: false };
    }
    return { kind: "feature", label: s, code: false };
  }

  const S = {
    health: null, busy: false, share: 0.3, sel: null, curve: null, budgetAbort: null, budgetTimer: 0,
    session: { all: 0, cheap: 0, strong: 0, pSum: 0 },
  };
  const share = () => Number($("share").value) / 100;
  const ROUTE_NAME = { cheap: "Cheap model", strong: "Strong model" };
  const ICON = {
    cheap: html`<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="4" fill="currentColor"/></svg>`,
    strong: html`<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5l5.5 5.5L8 13.5 2.5 8z" fill="currentColor"/></svg>`,
  };
  const badge = (route, model) => html`<span class="route-badge ${route}">${ICON[route]}${ROUTE_NAME[route] || route}${model ? html`<span class="model">${model}</span>` : ""}</span>`;

  /* ---------------- gauge ---------------- */
  (function drawTicks() {
    const g = [];
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const a = Math.PI * (1 - t);
      const x1 = 160 + Math.cos(a) * 146, y1 = 160 - Math.sin(a) * 146;
      const x2 = 160 + Math.cos(a) * 152, y2 = 160 - Math.sin(a) * 152;
      const lx = 160 + Math.cos(a) * 166, ly = 160 - Math.sin(a) * 166 + 3;
      g.push(`<line class="tick" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`);
      g.push(`<text class="tick-l" x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${t < 0.5 ? "end" : t > 0.5 ? "start" : "middle"}">${t * 100}%</text>`);
    }
    $("ticks").innerHTML = g.join(""); // static numbers only
  })();

  function setGauge(p, thr) {
    const g = $("gauge");
    const th = clamp(thr);
    $("zoneCheap").style.strokeDasharray = `${(th * 100).toFixed(2)} 100`;
    $("zoneStrong").style.strokeDasharray = `${((1 - th) * 100).toFixed(2)} 100`;
    $("zoneStrong").style.strokeDashoffset = `${(-th * 100).toFixed(2)}`;
    $("thrG").style.transform = `rotate(${((th - 0.5) * 180).toFixed(2)}deg)`;
    if (isNum(p)) {
      g.classList.remove("idle");
      $("needleG").style.transform = `rotate(${((clamp(p) - 0.5) * 180).toFixed(2)}deg)`;
    }
  }

  let countRaf = 0;
  function countTo(el, to) {
    cancelAnimationFrame(countRaf);
    const from = parseFloat(el.dataset.v || "0");
    el.dataset.v = String(to);
    if (reduced()) { el.textContent = pct(to, 1); return; }
    const t0 = performance.now(), dur = 800;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = pct(from + (to - from) * e, 1);
      if (k < 1) countRaf = requestAnimationFrame(step);
    };
    countRaf = requestAnimationFrame(step);
  }

  /* ---------------- signal box ---------------- */
  function chips(list, cls) {
    if (!list || !list.length) return html`<span class="none">nothing notable</span>`;
    const maxW = Math.max(...list.map((r) => Math.abs(r.weight) || 0), 1e-9);
    return list.map((r, i) => {
      const f = featureLabel(r.feature);
      const w = Math.abs(r.weight);
      return html`<span class="chip ${cls}" style="animation-delay:${i * 40}ms" title="${f.title || r.feature} · weight ${isNum(r.weight) ? r.weight.toFixed(4) : "-"}">
        <span class="kind">${f.kind}</span><span class="lbl ${f.code ? "code" : ""}">${f.label}</span>
        <span class="w" style="width:${Math.round(100 * w / maxW)}%"></span></span>`;
    });
  }

  function showDecision(d, prompt, opts = {}) {
    const route = d.route === "strong" ? "strong" : "cheap";
    setGauge(d.p_cheap_fails, d.threshold);
    const big = $("pBig");
    big.style.color = `var(--${route})`;
    if (isNum(d.p_cheap_fails)) countTo(big, d.p_cheap_fails); else big.textContent = "-";
    $("gaugeDesc").textContent = `Risk ${pct(d.p_cheap_fails, 1)} against a threshold of ${pct(d.threshold, 1)}: routed to the ${route} model.`;
    const gap = isNum(d.p_cheap_fails) && isNum(d.threshold) ? (d.p_cheap_fails - d.threshold) * 100 : null;
    setHTML($("verdict"), html`${badge(route)}
      <span class="small muted num">${gap == null ? "" : `${Math.abs(gap).toFixed(1)} pts ${gap >= 0 ? "above" : "below"} the threshold`}</span>`);
    setHTML($("verdictNote"), opts.rescored
      ? html`<span class="rescored">Re-scored at a ${pct(d.strong_share)} budget</span>: threshold ${pct(d.threshold, 1)}. The chat history keeps the original decision.`
      : html`Threshold <b class="c-signal num">${pct(d.threshold, 1)}</b> at a ${pct(d.strong_share)} budget · decided in ${isNum(d.router_latency_ms) ? d.router_latency_ms.toFixed(1) : "-"} ms`);
    $("sbSub").textContent = prompt.length > 42 ? `“${prompt.slice(0, 40)}…”` : `“${prompt}”`;

    const e = d.explanation || {};
    $("why").hidden = false;
    setHTML($("whyStrong"), html`${chips(e.towards_strong, "s")}`);
    setHTML($("whyCheap"), html`${chips(e.towards_cheap, "c")}`);

    const st = d.prompt_stats || {};
    const cell = (v, l) => html`<div><b>${isNum(v) ? int(v) : "-"}</b><span>${l}</span></div>`;
    $("signals").hidden = false;
    setHTML($("signals"), html`${cell(prompt.length, "characters")}${cell(st.n_question_marks, "question marks")}
      ${cell(st.code_markers, "code markers")}${cell(st.math_markers, "maths markers")}
      ${cell(st.constraint_words, "constraint words")}${cell(st.reasoning_words, "reasoning words")}`);
  }

  function select(item, scroll) {
    if (S.sel && S.sel.el) S.sel.el.classList.remove("sel");
    S.sel = item;
    item.el.classList.add("sel");
    showDecision(item.decision, item.prompt);
    if (Math.abs(item.decision.strong_share - share()) > 1e-9) rescore();
    if (scroll && window.matchMedia("(max-width: 960px)").matches) {
      document.querySelector(".side").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
    }
  }

  /* ---------------- budget ---------------- */
  function budgetLabels() {
    const s = share();
    $("shareLbl").textContent = pct(s);
    $("share").setAttribute("aria-valuetext", `${Math.round(s * 100)} percent`);
    $("share").style.setProperty("--fill", `${((s * 100 - 5) / 75) * 100}%`);
    $("saved").textContent = pct(1 - s);
    $("howShare").textContent = pct(s);
    $("howShare2").textContent = pct(s);
  }

  async function rescore() {
    clearTimeout(S.budgetTimer);
    if (S.budgetAbort) S.budgetAbort.abort();
    const ctl = (S.budgetAbort = new AbortController());
    const sel = S.sel;
    try {
      const d = await api("/api/route", { prompt: sel ? sel.prompt : PROBE, strong_share: share() }, { signal: ctl.signal });
      $("expPgr").textContent = pct(d.expected && d.expected.pgr, 1);
      $("expQ").textContent = d.expected && isNum(d.expected.quality) ? d.expected.quality.toFixed(2) : "-";
      S.lastPgr = d.expected && d.expected.pgr;
      drawSpark(S.lastPgr);
      if (sel && S.sel === sel) showDecision(d, sel.prompt, { rescored: Math.abs(sel.decision.strong_share - d.strong_share) > 1e-9 });
      else if (!sel) setGauge(null, d.threshold);
    } catch (err) {
      if (err.name === "AbortError") return;
      $("expPgr").textContent = "-"; $("expQ").textContent = "-";
    }
  }

  function drawSpark(pgr) {
    const svg = $("spark");
    const c = S.curve;
    if (!c) { svg.hidden = true; return; }
    svg.hidden = false;
    const W = Math.max(160, Math.round(svg.getBoundingClientRect().width) || 300), H = 86, T = 6, B = 14;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const sx = (x) => x * W, sy = (y) => T + (1 - clamp(y)) * (H - T - B);
    const pts = c.x.map((x, i) => `${sx(x).toFixed(1)},${sy(c.y[i]).toFixed(1)}`);
    const s = share();
    const my = isNum(pgr) ? pgr : null;
    svg.innerHTML = [
      `<path class="area" d="M0,${sy(0)} L${pts.join(" L")} L${W},${sy(0)} Z"/>`,
      `<line class="rnd" x1="0" y1="${sy(0)}" x2="${W}" y2="${sy(1)}"/>`,
      `<path class="ln" d="M${pts.join(" L")}"/>`,
      `<line class="vline" x1="${sx(s)}" x2="${sx(s)}" y1="${T}" y2="${H - B}"/>`,
      my == null ? "" : `<circle class="mark" cx="${sx(s).toFixed(1)}" cy="${sy(my).toFixed(1)}" r="4.5"/>`,
      `<text x="2" y="${H - 2}">0%</text><text x="${W - 2}" y="${H - 2}" text-anchor="end">100% strong</text>`,
    ].join(""); // numbers only, no user text
  }

  /* ---------------- thread ---------------- */
  function trackSvg(route) {
    const lit = route === "strong" ? "M2 15 H26 C38 15 44 5 56 5 H72" : "M2 15 H26 C38 15 44 25 56 25 H72";
    const cy = route === "strong" ? 5 : 25;
    return html`<svg class="track" viewBox="0 0 74 30" aria-hidden="true">
      <path class="rail" d="M2 15 H26 C38 15 44 5 56 5 H72"/><path class="rail" d="M26 15 C38 15 44 25 56 25 H72"/>
      <path class="lit ${route}" pathLength="100" d="${lit}"/>
      <circle class="sw" cx="26" cy="15" r="3.2"/><circle class="car ${route}" cx="70" cy="${cy}" r="3.6"/></svg>`;
  }

  function miniMeter(d, route) {
    return html`<span class="mini-meter" title="P(cheap fails) ${pct(d.p_cheap_fails, 1)} · threshold ${pct(d.threshold, 1)}">
      <span class="fill" style="width:${(clamp(d.p_cheap_fails) * 100).toFixed(1)}%;background:var(--${route})"></span>
      <span class="thr" style="left:${(clamp(d.threshold) * 100).toFixed(1)}%"></span></span>
      <span class="small muted num">${pct(d.p_cheap_fails, 1)} vs ${pct(d.threshold, 1)}</span>`;
  }

  function addEl(cls, tpl) {
    const empty = $("empty");
    if (empty) empty.remove();
    $("clearBtn").hidden = false;
    const el = document.createElement("div");
    el.className = "msg " + cls;
    if (tpl !== undefined) setHTML(el, tpl);
    $("thread").appendChild(el);
    requestAnimationFrame(() => { $("thread").scrollTop = $("thread").scrollHeight; });
    return el;
  }

  function renderTicket(el, item, phase) {
    const d = item.decision, route = d.route === "strong" ? "strong" : "cheap";
    const a = item.answer;
    let body;
    if (phase === "asking") {
      body = html`<div class="note"><span class="dots c-${route}"><i></i><i></i><i></i></span> Asking the ${route} model…</div>`;
    } else if (a) {
      body = html`<div class="prose">${raw(Markdown.render(a.answer))}</div>`;
    } else if (item.note) {
      body = html`<div class="note">${item.note}</div>`;
    } else {
      body = html`<div class="note">${item.routeOnly ? "Route-only: no model was called. This prompt would go to the " + route + " model." : "No answer was returned."}</div>`;
    }
    const foot = [];
    if (isNum(d.router_latency_ms)) foot.push(`routed in ${d.router_latency_ms.toFixed(1)} ms`);
    if (a && isNum(a.latency_ms)) foot.push(`answered in ${int(a.latency_ms)} ms`);
    if (a && a.usage && isNum(a.usage.total_tokens)) foot.push(`${int(a.usage.total_tokens)} tokens`);
    if (isNum(d.strong_share)) foot.push(`${pct(d.strong_share)} budget`);
    const c = item.comparison;
    const other = route === "strong" ? "cheap" : "strong";
    setHTML(el, html`
      <div class="ticket-top">${trackSvg(route)}
        <div class="meta">${badge(route, a && a.model)}${miniMeter(d, route)}</div>
        <button type="button" class="btn ghost why" data-why>Why?</button>
      </div>
      <div class="ticket-body">${body}</div>
      ${c ? html`<details class="compare"><summary>Compare: ${other} model's answer${c.model ? html` <span class="mono small muted">${c.model}</span>` : ""}</summary>
        <div class="prose">${raw(Markdown.render(c.answer))}</div>
        <div class="small muted mono" style="margin-top:8px">${[isNum(c.latency_ms) ? `${int(c.latency_ms)} ms` : "", c.usage && isNum(c.usage.total_tokens) ? `${int(c.usage.total_tokens)} tokens` : ""].filter(Boolean).join(" · ")}</div></details>` : ""}
      <div class="ticket-foot">${foot.map((t) => html`<span>${t}</span>`)}</div>`);
    el.querySelector("[data-why]").addEventListener("click", () => select(item, true));
  }

  function errorTicket(el, err, text) {
    const st = err.status;
    const title = st === 429 ? "Rate limit reached" : st === 422 ? "The server rejected this prompt" : st === 502 ? "The LLM provider failed"
      : st === 0 ? "Server unreachable" : st === 404 ? "Router not available" : "Something went wrong";
    el.className = "msg ticket error";
    setHTML(el, html`<div class="ticket-top" style="grid-template-columns:1fr auto"><span>${title}</span>
        <button type="button" class="btn ghost" data-retry>Retry</button></div>
      <div class="ticket-body"><div class="note">${err.message}</div></div>`);
    el.querySelector("[data-retry]").addEventListener("click", () => { if (!S.busy) { el.remove(); send(text, true); } });
  }

  function tally(route, p) {
    const s = S.session;
    s.all++; s[route]++; if (isNum(p)) s.pSum += p;
    $("nAll").textContent = s.all; $("nCheap").textContent = s.cheap; $("nStrong").textContent = s.strong;
    $("splitCheap").style.flexGrow = s.cheap; $("splitStrong").style.flexGrow = s.strong;
    $("sessShare").textContent = s.all ? `${pct(s.strong / s.all)} went strong` : "";
    $("sessAvg").textContent = s.all ? `Average risk this session: ${pct(s.pSum / s.all, 1)}. Current budget: ${pct(share())} strong.` : "No prompts yet.";
  }

  function setBusy(b) {
    S.busy = b;
    $("send").disabled = b;
    $("send").setAttribute("aria-busy", String(b));
  }

  async function send(text, isRetry) {
    if (S.busy) return;
    const prompt = String(text || "");
    if (!prompt.trim()) { $("prompt").focus(); return; }
    if (prompt.length > MAX) return;
    setBusy(true);
    if (!isRetry) { const u = addEl("user"); u.textContent = prompt; }
    const el = addEl("ticket pending", html`<div class="ticket-top" style="grid-template-columns:1fr"><span class="meta"><span class="dots"><i></i><i></i><i></i></span> Routing…</span></div>`);
    const live = !!(S.health && S.health.llm_configured);
    const routeOnly = !live || $("routeOnly").checked;
    const s = share();
    try {
      const d = await api("/api/route", { prompt, strong_share: s });
      const item = { prompt, decision: d, el, routeOnly, note: !live ? "Demo mode: this server has no LLM key, so no model was called." : null };
      el.className = "msg ticket";
      if (routeOnly) {
        renderTicket(el, item, "done");
        tally(d.route, d.p_cheap_fails);
        select(item);
        return;
      }
      renderTicket(el, item, "asking");
      select(item);
      const r = await api("/api/chat", { prompt, strong_share: s, compare: $("compare").checked });
      item.decision = r.decision || d;
      item.answer = r.answer || null;
      item.comparison = r.comparison || null;
      item.note = r.note || null;
      tally(item.decision.route, item.decision.p_cheap_fails);
      renderTicket(el, item, "done");
      if (S.sel === item) showDecision(item.decision, prompt);
    } catch (err) {
      errorTicket(el, err, prompt);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => { $("thread").scrollTop = $("thread").scrollHeight; });
    }
  }

  /* ---------------- composer ---------------- */
  const ta = $("prompt");
  function autosize() {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
    const n = ta.value.length;
    $("count").textContent = `${int(n)} / ${int(MAX)}`;
    $("count").classList.toggle("warn", n > MAX * 0.9);
  }
  ta.addEventListener("input", autosize);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      $("ask").requestSubmit();
    }
  });
  $("ask").addEventListener("submit", (e) => {
    e.preventDefault();
    if (S.busy) return;
    const t = ta.value;
    if (!t.trim()) return;
    ta.value = ""; autosize();
    send(t);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) && !document.activeElement.isContentEditable) {
      e.preventDefault(); ta.focus();
    }
  });
  $("thread").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    const code = b.closest(".code").querySelector("pre").textContent;
    try { await navigator.clipboard.writeText(code); b.textContent = "Copied"; }
    catch { b.textContent = "Press Ctrl+C"; const r = document.createRange(); r.selectNodeContents(b.closest(".code").querySelector("pre")); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); }
    setTimeout(() => { b.textContent = "Copy"; }, 1600);
  });
  $("clearBtn").addEventListener("click", () => {
    $("thread").innerHTML = "";
    const tpl = document.createElement("div");
    tpl.className = "empty"; tpl.id = "empty";
    setHTML(tpl, html`<h3>Fresh start.</h3><p>Pick an example or type your own prompt.</p><div class="examples" id="examples"></div>`);
    $("thread").appendChild(tpl);
    renderExamples();
    S.session = { all: 0, cheap: 0, strong: 0, pSum: 0 };
    ["nAll", "nCheap", "nStrong"].forEach((id) => { $(id).textContent = "0"; });
    $("splitCheap").style.flexGrow = 0; $("splitStrong").style.flexGrow = 0;
    $("sessShare").textContent = ""; $("sessAvg").textContent = "No prompts yet.";
    S.sel = null;
    $("clearBtn").hidden = true;
    ta.focus();
  });

  $("share").addEventListener("input", () => {
    budgetLabels();
    clearTimeout(S.budgetTimer);
    S.budgetTimer = setTimeout(rescore, 220);
  });
  $("share").addEventListener("change", () => { clearTimeout(S.budgetTimer); rescore(); });

  function renderExamples() {
    const box = $("examples");
    if (!box) return;
    box.textContent = "";
    for (const [k, t] of EXAMPLES) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "example";
      setHTML(b, html`<span class="k">${k}</span><span class="t">${t}</span>`);
      b.setAttribute("aria-label", `Send example: ${t}`);
      b.addEventListener("click", () => send(t));
      box.appendChild(b);
    }
  }

  /* ---------------- boot ---------------- */
  function renderStatus(h, err) {
    const pills = [];
    if (err) {
      pills.push(html`<span class="pill down"><span class="led"></span><span>Router offline</span></span>`);
      pills.push(html`<span class="pill">${err.message}</span>`);
    } else if (h.llm_configured) {
      pills.push(html`<span class="pill live"><span class="led"></span><span>Live: models answer</span></span>`);
      pills.push(html`<span class="pill"><i class="legend-key"><i class="bg-cheap"></i></i>${h.cheap_model || "cheap model"}</span>`);
      pills.push(html`<span class="pill"><i class="legend-key"><i class="bg-strong"></i></i>${h.strong_model || "strong model"}</span>`);
    } else {
      pills.push(html`<span class="pill demo"><span class="led"></span><span>Routing-only demo</span></span>`);
    }
    setHTML($("status"), html`${pills}`);
  }

  async function boot() {
    renderExamples();
    budgetLabels();
    setGauge(null, 0.5);
    try {
      const h = (S.health = await api("/api/health"));
      renderStatus(h);
      const r = h.router || {};
      if (isNum(r.trained_rows)) $("heroEyebrow").textContent = `Classical ML router · trained on ${int(r.trained_rows)} prompts`;
      const comps = Array.isArray(r.components) ? r.components : [];
      setHTML($("components"), comps.length ? html`${comps.map((c) => html`<li>${c}</li>`)}` : html`<li class="muted">-</li>`);
      if (!h.llm_configured) {
        $("demoBanner").hidden = false;
        $("routeOnly").checked = true; $("routeOnly").disabled = true; $("compare").disabled = true;
      }
    } catch (err) {
      renderStatus(null, err);
      setHTML($("components"), html`<li class="muted">-</li>`);
    }
    try {
      const m = await api("/api/metrics");
      const c = m && m.curves;
      if (c && Array.isArray(c.x) && Array.isArray(c.final) && c.x.length === c.final.length && c.x.length > 1) S.curve = { x: c.x, y: c.final };
    } catch { /* results not built: the sparkline simply stays hidden */ }
    rescore();
  }

  $("routeOnly").addEventListener("change", () => { $("compare").disabled = $("routeOnly").checked || !(S.health && S.health.llm_configured); });
  let rz = 0;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => drawSpark(S.lastPgr), 150); });
  boot();
})();
