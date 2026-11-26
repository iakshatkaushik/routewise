/* Safe markdown -> HTML for model answers.
   Every piece of source text is HTML-escaped before any markup is added; the only tags emitted are the
   fixed set below, and links are kept only for http(s) URLs. Supports fenced code, inline code, bold,
   italics, headings, ordered/unordered (nested) lists, blockquotes, rules and links. */
"use strict";

const Markdown = (() => {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);
  const PH = "\u0000";

  function inline(text) {
    const slots = [];
    const keep = (h) => PH + (slots.push(h) - 1) + PH;
    let s = text.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, __, code) => keep(`<code>${esc(code)}</code>`));
    s = esc(s);
    // [text](http(s)://url) - text is escaped already; url can't contain raw quotes or angle brackets after escaping
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) =>
      keep(`<a href="${url}" target="_blank" rel="noopener noreferrer nofollow">${emph(label)}</a>`));
    // bare http(s) URLs
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]*[^\s<.,:;!?)\]'"&])/g, (_, pre, url) =>
      pre + keep(`<a href="${url}" target="_blank" rel="noopener noreferrer nofollow">${url}</a>`));
    s = emph(s);
    return s.replace(new RegExp(PH + "(\\d+)" + PH, "g"), (_, i) => slots[+i]);
  }

  function emph(s) {
    return s
      .replace(/\*\*\*(?=\S)([\s\S]*?\S)\*\*\*/g, "<strong><em>$1</em></strong>")
      .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, "$1<strong>$2</strong>")
      .replace(/(^|[^*\w])\*(?=[^\s*])([^*]*?[^\s*])\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/(^|[^\w])_(?=[^\s_])([^_]*?[^\s_])_(?!\w)/g, "$1<em>$2</em>");
  }

  const RE = {
    fence: /^ {0,3}(`{3,}|~{3,})\s*([\w+#.-]*)[^`]*$/,
    heading: /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/,
    hr: /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/,
    quote: /^ {0,3}>\s?(.*)$/,
    item: /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/,
    blank: /^\s*$/,
  };

  function list(lines, i) {
    const stack = [];
    let out = "";
    const open = (type, indent, start) => {
      stack.push({ type, indent });
      out += type === "ol" ? (start && start !== 1 ? `<ol start="${start}">` : "<ol>") : "<ul>";
    };
    const closeTop = () => { const t = stack.pop(); out += `</li></${t.type}>`; };
    let liOpen = false;
    for (; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(RE.item);
      if (!m) {
        if (RE.blank.test(line)) {
          const next = lines[i + 1];
          if (next !== undefined && RE.item.test(next)) continue;
          break;
        }
        if (/^\s{2,}\S/.test(line) && liOpen) { out += "<br>" + inline(line.trim()); continue; }
        break;
      }
      const indent = m[1].replace(/\t/g, "    ").length;
      const type = /\d/.test(m[2]) ? "ol" : "ul";
      const start = type === "ol" ? parseInt(m[2], 10) : 0;
      if (!stack.length) { open(type, indent, start); }
      else {
        let top = stack[stack.length - 1];
        if (indent > top.indent + 1) { open(type, indent, start); liOpen = false; }
        else {
          while (stack.length > 1 && indent < stack[stack.length - 1].indent - 1) closeTop();
          top = stack[stack.length - 1];
          if (top.type !== type && stack.length === 1) { closeTop(); open(type, indent, start); liOpen = false; }
          else if (liOpen) out += "</li>";
        }
      }
      out += "<li>" + inline(m[3]);
      liOpen = true;
    }
    while (stack.length) closeTop();
    return [out, i];
  }

  function blocks(src, depth) {
    const lines = src.split("\n");
    let out = "";
    let para = [];
    const flush = () => { if (para.length) { out += "<p>" + para.map(inline).join("<br>") + "</p>"; para = []; } };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      let m;
      if ((m = line.match(RE.fence))) {
        flush();
        const fence = m[1], lang = m[2];
        const body = [];
        for (i++; i < lines.length && !(lines[i].trim().startsWith(fence[0].repeat(fence.length)) && lines[i].trim().replace(/[`~]/g, "") === ""); i++) body.push(lines[i]);
        out += `<div class="code"><div class="code-head"><span>${esc(lang || "code")}</span><button type="button" data-copy>Copy</button></div><pre><code>${esc(body.join("\n"))}</code></pre></div>`;
        continue;
      }
      if (RE.blank.test(line)) { flush(); continue; }
      if ((m = line.match(RE.heading))) {
        flush();
        const lvl = Math.min(6, m[1].length + 2);
        out += `<h${lvl}>${inline(m[2])}</h${lvl}>`;
        continue;
      }
      if (RE.hr.test(line)) { flush(); out += "<hr>"; continue; }
      if (RE.quote.test(line) && depth < 3) {
        flush();
        const q = [];
        while (i < lines.length && (m = lines[i].match(RE.quote))) { q.push(m[1]); i++; }
        i--;
        out += "<blockquote>" + blocks(q.join("\n"), depth + 1) + "</blockquote>";
        continue;
      }
      if (RE.item.test(line) && (para.length === 0 || /^\s*([-*+]|\d{1,9}[.)])\s/.test(line))) {
        flush();
        const [h, next] = list(lines, i);
        out += h;
        i = next - 1;
        continue;
      }
      para.push(line.trim());
    }
    flush();
    return out;
  }

  function render(src) {
    const text = String(src == null ? "" : src).replace(/\r\n?/g, "\n").replace(/\u0000/g, "");
    return blocks(text, 0);
  }

  return { render, inline };
})();
