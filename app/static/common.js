/* Shared helpers for both pages: auto-escaping HTML templates, number formatting, API calls, theme toggle. */
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
  const pct = (x, d = 0) => (isNum(x) ? (100 * x).toFixed(d) + "%" : "-");
  const fix = (x, d = 3) => (isNum(x) ? x.toFixed(d) : "-");
  const int = (x) => (isNum(x) ? Math.round(x).toLocaleString("en-US") : "-");
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  const clamp = (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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

  function effectiveTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
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
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", sync);
    sync();
  }

  return { esc, raw, html, setHTML, Raw, isNum, pct, fix, int, get, clamp, reduced, api, ApiError, initTheme };
})();

document.addEventListener("DOMContentLoaded", U.initTheme);
