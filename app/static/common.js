/* Shared helpers for both pages: auto-escaping HTML templates, number formatting, API calls, theme toggle,
   scroll reveals and count-up numbers. Nothing here inserts untrusted text as HTML. */
"use strict";

const U = (() => {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

  // html`...` escapes every interpolated value unless it is itself a Raw (built by html`` or raw()).
  class Raw { constructor(s) { this.s = s; } toString() { return this.s; } }
  const raw = (s) => new Raw(String(s));
  const val = (v) => (v instanceof Raw ? v.s : Array.isArray(v) ? v.map(val).join("")
    : v == null || v === false ? "" : esc(v));
  const html = (strings, ...vals) => new Raw(strings.reduce((a, s, i) => a + s + (i < vals.length ? val(vals[i]) : ""), ""));
  const setHTML = (el, tpl) => { el.innerHTML = tpl instanceof Raw ? tpl.s : esc(tpl); return el; };

  const isNum = (x) => typeof x === "number" && Number.isFinite(x);
  const pct = (x, d = 0) => (isNum(x) ? (100 * x).toFixed(d) + "%" : "–");
  const fix = (x, d = 3) => (isNum(x) ? x.toFixed(d) : "–");
  const int = (x) => (isNum(x) ? Math.round(x).toLocaleString("en-US") : "–");
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  const clamp = (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /** Linear interpolation of y at x on a sorted x grid. */
  function interp(xs, ys, x) {
    if (!xs || !ys || !xs.length) return NaN;
    if (x <= xs[0]) return ys[0];
    for (let i = 1; i < xs.length; i++) {
      if (x <= xs[i]) { const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1); return ys[i - 1] + t * (ys[i] - ys[i - 1]); }
    }
    return ys[ys.length - 1];
  }

  class ApiError extends Error { constructor(msg, status) { super(msg); this.status = status; } }

  async function api(path, body, opts = {}) {
    let r;
    try {
      r = await fetch(path, body === undefined ? { signal: opts.signal } : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: opts.signal,
      });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      throw new ApiError("Can't reach the server. Is it running?", 0);
    }
    const j = await r.json().catch(() => null);
    if (!r.ok) {
      let d = j && j.detail;
      if (Array.isArray(d)) d = d.map((x) => (x && x.msg) || String(x)).join("; ");
      else if (d && typeof d !== "string") d = JSON.stringify(d);
      throw new ApiError(d || r.statusText || "Request failed", r.status);
    }
    if (j == null) throw new ApiError("The server sent an unreadable response.", r.status);
    return j;
  }

  /* ---- animation helpers ------------------------------------------------------------------ */
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);

  /** Animate a number from 0 (or `from`) to `to`, writing fmt(value) into el.textContent. */
  function countUp(el, to, fmt, { from = 0, dur = 1100 } = {}) {
    if (!el || !isNum(to)) return;
    if (reduced()) { el.textContent = fmt(to); return; }
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      el.textContent = fmt(from + (to - from) * easeOut(t));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /** Call cb once when el scrolls into view (immediately when motion is reduced). */
  function onVisible(el, cb, threshold = 0.2) {
    if (!el) return;
    if (reduced() || !("IntersectionObserver" in window)) { cb(true); return; }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); cb(false); }
    }, { threshold, rootMargin: "0px 0px -8% 0px" });
    io.observe(el);
  }

  /** Fade-up every .reveal element as it enters the viewport. */
  function initReveals(root = document) {
    $$(".reveal:not(.is-in)", root).forEach((el) => onVisible(el, () => el.classList.add("is-in"), 0.12));
  }

  /* ---- theme ------------------------------------------------------------------------------ */
  function effectiveTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    return t === "dark" ? "dark" : "light"; // light is the default theme
  }
  function initTheme() {
    const btn = document.getElementById("themeBtn");
    if (!btn) return;
    const sync = () => {
      const t = effectiveTheme();
      btn.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
      btn.title = btn.getAttribute("aria-label");
      document.dispatchEvent(new CustomEvent("themechange", { detail: t }));
    };
    btn.addEventListener("click", () => {
      const next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("llmr-theme", next); } catch (e) { /* not persisted */ }
      sync();
    });
    sync();
  }

  function initHeader() {
    const head = document.getElementById("siteHead");
    if (!head) return;
    const on = () => head.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", on, { passive: true });
    on();
  }

  /** Copy buttons inside rendered markdown code blocks (event delegation). */
  function initCopy() {
    document.addEventListener("click", async (e) => {
      const b = e.target.closest && e.target.closest("[data-copy]");
      if (!b) return;
      const code = b.closest(".code") && b.closest(".code").querySelector("code");
      if (!code) return;
      try { await navigator.clipboard.writeText(code.textContent); b.textContent = "Copied"; }
      catch (err) { b.textContent = "Press Ctrl+C"; }
      setTimeout(() => { b.textContent = "Copy"; }, 1400);
    });
  }

  return {
    esc, raw, html, setHTML, Raw, isNum, pct, fix, int, get, clamp, reduced, interp, api, ApiError,
    countUp, onVisible, initReveals, initTheme, effectiveTheme, $, $$, easeOut, initHeader, initCopy,
  };
})();

document.addEventListener("DOMContentLoaded", () => { U.initTheme(); U.initHeader(); U.initCopy(); U.initReveals(); });
