/* Runs in <head> before paint: apply a saved manual theme choice, if any. */
(function () {
  try {
    var t = localStorage.getItem("llmr-theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage blocked: follow the system theme */ }
})();
