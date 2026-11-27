/* A tiny chart kit, drawn by hand with SVG and HTML: scales, a shared tooltip, responsive mounting,
   draw-in on scroll, ranked bar lists that re-sort with animation, and dot-on-a-line plots.
   Every label goes through textContent or U.html, so data strings are never parsed as HTML. */
"use strict";

const C = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const { html, setHTML, isNum, clamp, onVisible, reduced } = U;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, attrs) {
    const t = el("text", Object.assign({ x, y }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }

  const lin = (d0, d1, r0, r1) => {
    const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.inv = (r) => d0 + ((r - r0) / (r1 - r0 || 1)) * (d1 - d0);
    f.domain = [d0, d1];
    return f;
  };
  const logScale = (d0, d1, r0, r1) => {
    const a = Math.log10(d0), b = Math.log10(d1);
    const f = (v) => r0 + ((Math.log10(v) - a) / (b - a || 1)) * (r1 - r0);
    f.inv = (r) => Math.pow(10, a + ((r - r0) / (r1 - r0 || 1)) * (b - a));
    f.domain = [d0, d1];
    return f;
  };

  /** "Nice" tick values between lo and hi. */
  function ticks(lo, hi, n = 5) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  /** Round a domain outwards to tick steps. */
  function niceDomain(lo, hi, n = 5) {
    const t = ticks(lo, hi, n);
    const step = t.length > 1 ? t[1] - t[0] : (hi - lo) || 1;
    return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
  }

  /* ---- tooltip ---------------------------------------------------------------------------- */
  let tipEl = null;
  function tip(x, y, content) {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "tip";
      tipEl.setAttribute("role", "tooltip");
      document.body.appendChild(tipEl);
    }
    setHTML(tipEl, content);
    tipEl.classList.add("on");
    const r = tipEl.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    let left = x - r.width / 2;
    left = clamp(left, 8, vw - r.width - 8);
    let top = y - r.height - 14;
    if (top < 8) top = y + 18;
    tipEl.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
  function hideTip() { if (tipEl) tipEl.classList.remove("on"); }
  window.addEventListener("scroll", hideTip, { passive: true });

  /** Show a tooltip for an element (mouse hover or keyboard focus). */
  function bindTip(node, contentFn) {
    const show = (e) => {
      const r = node.getBoundingClientRect();
      const x = e && e.clientX != null && e.type.startsWith("pointer") ? e.clientX : r.left + r.width / 2;
      const y = e && e.clientY != null && e.type.startsWith("pointer") ? e.clientY : r.top;
      tip(x, y, contentFn());
    };
    node.addEventListener("pointerenter", show);
    node.addEventListener("pointermove", show);
    node.addEventListener("pointerleave", hideTip);
    node.addEventListener("focus", show);
    node.addEventListener("blur", hideTip);
  }

  /* ---- responsive mount + reveal ---------------------------------------------------------- */
  /** render(width) now and whenever the box changes width. */
  function mount(box, render) {
    let w = 0;
    const go = () => {
      const nw = Math.round(box.clientWidth);
      if (nw && Math.abs(nw - w) > 1) { w = nw; render(nw); }
    };
    if ("ResizeObserver" in window) {
      let raf = 0;
      new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(go); }).observe(box);
    } else window.addEventListener("resize", go);
    go();
  }

  /** Mark a chart box "in" when it scrolls into view; CSS then runs the draw-in transitions. */
  function reveal(box, after) {
    onVisible(box, (instant) => {
      if (instant) box.classList.add("no-anim");
      requestAnimationFrame(() => { box.classList.add("in"); if (after) after(instant); });
    }, 0.25);
  }

  /** Axis with grid lines. */
  function yAxis(g, scale, x0, x1, vals, fmt) {
    vals.forEach((v) => {
      const y = scale(v);
      el("line", { x1: x0, x2: x1, y1: y, y2: y, class: "grid" }, g);
      text(g, x0 - 10, y + 4.5, fmt(v), { class: "tick", "text-anchor": "end" });
    });
  }
  function xAxis(g, scale, y0, y1, vals, fmt, grid = true) {
    vals.forEach((v) => {
      const x = scale(v);
      if (grid) el("line", { x1: x, x2: x, y1: y0, y2: y1, class: "grid" }, g);
      text(g, x, y1 + 22, fmt(v), { class: "tick", "text-anchor": "middle" });
    });
  }
  /** Thick ink axis lines along the left and bottom edges of the plot. */
  function axes(g, x0, x1, y0, y1) {
    return el("path", { d: `M${x0},${y0} L${x0},${y1} L${x1},${y1}`, class: "axis" }, g);
  }

  /* ---- ranked bars (HTML) ----------------------------------------------------------------- */
  /**
   * items:   [{ id, label, sub, tag, hl, values: {key: number}, tip: () => Raw }]
   * metrics: [{ key, label, better: "high"|"low", domain: [lo, hi] | (vals) => [lo, hi], fmt, refs: [{v, label}], log }]
   */
  function rankBars(box, { items, metrics, initial, caption, onChange }) {
    let cur = metrics.find((m) => m.key === initial) || metrics[0];
    box.classList.add("rb");
    const ctl = metrics.length > 1 ? document.createElement("div") : null;
    if (ctl) {
      ctl.className = "seg";
      ctl.setAttribute("role", "group");
      ctl.setAttribute("aria-label", "Metric");
      metrics.forEach((m) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = m.label;
        b.dataset.k = m.key;
        b.setAttribute("aria-pressed", String(m === cur));
        b.addEventListener("click", () => { cur = m; update(); if (onChange) onChange(m); });
        ctl.appendChild(b);
      });
      box.appendChild(ctl);
    }
    const cap = document.createElement("p");
    cap.className = "rb-cap";
    box.appendChild(cap);
    const rows = document.createElement("div");
    rows.className = "rb-rows";
    rows.style.setProperty("--n", items.length);
    box.appendChild(rows);
    const refLayer = document.createElement("div");
    refLayer.className = "rb-refs";
    rows.appendChild(refLayer);

    const rowEls = items.map((it) => {
      const r = document.createElement("div");
      r.className = "rb-row" + (it.hl ? " hl" : "") + (it.tag ? " t-" + it.tag : "");
      r.tabIndex = 0;
      setHTML(r, html`<span class="rb-label"><span class="rb-name">${it.label}</span>${it.sub ? html`<span class="rb-sub">${it.sub}</span>` : ""}</span>
        <span class="rb-track"><i></i></span><span class="rb-val num"></span>`);
      if (it.tip) bindTip(r, it.tip);
      rows.appendChild(r);
      return r;
    });

    function update() {
      if (ctl) ctl.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k === cur.key)));
      const vals = items.map((it) => it.values[cur.key]).filter(isNum);
      let [lo, hi] = typeof cur.domain === "function" ? cur.domain(vals) : cur.domain || [0, Math.max(...vals)];
      const f = cur.log
        ? (v) => (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
        : (v) => (v - lo) / (hi - lo || 1);
      const order = items.map((it, i) => i).sort((a, b) => {
        const va = items[a].values[cur.key], vb = items[b].values[cur.key];
        if (!isNum(va)) return 1;
        if (!isNum(vb)) return -1;
        return cur.better === "low" ? va - vb : vb - va;
      });
      let shown = 0;
      order.forEach((idx, rank) => {
        const r = rowEls[idx], v = items[idx].values[cur.key];
        r.style.setProperty("--i", rank);
        r.hidden = !isNum(v);
        if (isNum(v)) shown++;
        r.style.setProperty("--f", isNum(v) ? clamp(f(v), 0.004, 1).toFixed(4) : 0);
        r.querySelector(".rb-val").textContent = isNum(v) ? cur.fmt(v) : "–";
        r.setAttribute("aria-label", `${rank + 1}. ${items[idx].label}: ${isNum(v) ? cur.fmt(v) : "no value"}`);
      });
      rows.style.setProperty("--n", shown);
      refLayer.textContent = "";
      (cur.refs || []).forEach((ref) => {
        if (!isNum(ref.v)) return;
        const x = clamp(f(ref.v), 0, 1);
        const d = document.createElement("div");
        d.className = "rb-ref" + (x < 0.18 ? " l" : x > 0.82 ? " r" : "");
        d.style.setProperty("--x", x.toFixed(4));
        const s = document.createElement("span");
        s.textContent = ref.label;
        d.appendChild(s);
        refLayer.appendChild(d);
      });
      cap.textContent = typeof cur.caption === "function" ? cur.caption() : cur.caption || caption || "";
      cap.hidden = !cap.textContent;
    }
    update();
    reveal(box);
    return { update, set: (k) => { const m = metrics.find((x) => x.key === k); if (m) { cur = m; update(); } } };
  }

  /* ---- dots on a number line (HTML) ------------------------------------------------------- */
  /**
   * rows: [{ label, sub, points: [{ v, cls, label, tip }], join: bool }]
   */
  function dotLine(box, { rows, domain, fmt, tickVals, axisLabel, tickFmt }) {
    const [lo, hi] = domain;
    const f = (v) => clamp((v - lo) / (hi - lo || 1), 0, 1);
    box.classList.add("dl");
    const wrap = document.createElement("div");
    wrap.className = "dl-rows";
    rows.forEach((row, ri) => {
      const r = document.createElement("div");
      r.className = "dl-row";
      r.style.setProperty("--d", ri);
      setHTML(r, html`<span class="dl-label"><span>${row.label}</span>${row.sub ? html`<small>${row.sub}</small>` : ""}</span><span class="dl-track"></span>`);
      const track = r.querySelector(".dl-track");
      const pts = row.points.filter((p) => isNum(p.v));
      if (row.join && pts.length > 1) {
        const a = Math.min(...pts.map((p) => f(p.v))), b = Math.max(...pts.map((p) => f(p.v)));
        const j = document.createElement("i");
        j.className = "dl-join";
        j.style.setProperty("--a", a.toFixed(4));
        j.style.setProperty("--b", b.toFixed(4));
        track.appendChild(j);
      }
      pts.forEach((p) => {
        const d = document.createElement("button");
        d.type = "button";
        d.className = "dl-dot " + (p.cls || "");
        d.style.setProperty("--x", f(p.v).toFixed(4));
        d.setAttribute("aria-label", `${row.label}, ${p.label || ""}: ${fmt(p.v)}`);
        const s = document.createElement("span");
        s.className = "dl-v num";
        s.textContent = fmt(p.v);
        d.appendChild(s);
        bindTip(d, () => html`<b>${p.label || row.label}</b><br>${fmt(p.v)}${p.tip ? html`<br><span class="muted">${p.tip}</span>` : ""}`);
        track.appendChild(d);
      });
      wrap.appendChild(r);
    });
    box.appendChild(wrap);
    const axis = document.createElement("div");
    axis.className = "dl-axis";
    (tickVals || ticks(lo, hi, 4)).forEach((t) => {
      const s = document.createElement("span");
      s.style.setProperty("--x", f(t).toFixed(4));
      s.textContent = (tickFmt || fmt)(t);
      axis.appendChild(s);
    });
    box.appendChild(axis);
    // drop every second (or third) tick label when they would touch
    mount(box, () => {
      const spans = Array.from(axis.children);
      const step = Math.max(1, Math.ceil((spans.length * 58) / Math.max(axis.clientWidth, 1)));
      spans.forEach((sp, i) => { sp.hidden = i % step !== 0; });
    });
    if (axisLabel) {
      const p = document.createElement("div");
      p.className = "dl-axis-label";
      p.textContent = axisLabel;
      box.appendChild(p);
    }
    reveal(box);
  }

  return { el, text, lin, logScale, ticks, niceDomain, tip, hideTip, bindTip, mount, reveal, yAxis, xAxis, axes, rankBars, dotLine, NS, reducedMotion: reduced };
})();
