/* Static dashboard client.
 *
 * Every view was pre-rendered at build time, so there is no computation here:
 * pick the payload for (tab, theme, horizon, model), inject its HTML, and hand
 * each figure to Plotly. Payloads are fetched once and cached.
 *
 * Figures are pre-rendered per theme rather than recoloured in the browser,
 * because the palette's light and dark steps are separately validated for
 * colour-vision separation against their own surface -- flipping one to the
 * other is not a transformation, it is a different set of colours.
 */
(function () {
  "use strict";

  var BOOT = window.__BOOT__ || {};
  var cache = Object.create(null);
  Object.keys(BOOT.inline || {}).forEach(function (k) { cache[k] = BOOT.inline[k]; });

  var state = {
    tab: "forecast",
    theme: localStorage.getItem("btc-theme") || "dark",
    horizon: localStorage.getItem("btc-horizon") || "1w",
    model: localStorage.getItem("btc-model") || (BOOT.models || [])[0],
    optionModel: localStorage.getItem("btc-option-model") || (BOOT.optionModels || [])[0],
    greek: localStorage.getItem("btc-greek") || "theta"
  };
  if ((BOOT.models || []).indexOf(state.model) < 0) state.model = (BOOT.models || [])[0];
  if ((BOOT.optionModels || []).indexOf(state.optionModel) < 0) state.optionModel = (BOOT.optionModels || [])[0];

  var content = document.getElementById("content");
  var masthead = document.getElementById("masthead");
  var modelPicker = document.getElementById("model-picker");
  var greekPicker = document.getElementById("greek-picker");
  var themeButton = document.getElementById("theme-button");

  var PLOT_CONFIG = { displayModeBar: false, responsive: true };

  function key(parts) { return parts.filter(Boolean).join("-"); }

  function load(k) {
    if (cache[k]) return Promise.resolve(cache[k]);
    var url = (BOOT.manifest || {})[k];
    if (!url) return Promise.resolve(null);
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + url);
      return r.json();
    }).then(function (payload) { cache[k] = payload; return payload; });
  }

  function plot(target, id, figure) {
    var el = target.querySelector("#" + CSS.escape(id));
    if (!el) return;
    // Each figure is drawn independently. A figure that fails -- a bad trace, or
    // Plotly itself missing -- must not take down the panel around it. The
    // earlier version let one throw escape into the panel's catch block, which
    // replaced the whole view (tables, cards and all) with an error line.
    try {
      Plotly.newPlot(el, figure.data, figure.layout, PLOT_CONFIG);
    } catch (err) {
      el.innerHTML = '<div class="loading-note">chart unavailable</div>';
      if (window.console) console.error("figure " + id + ":", err);
    }
  }

  function draw(target, payload, greek) {
    target.innerHTML = payload.html;
    Object.keys(payload.figures).forEach(function (id) {
      if (id.indexOf("::") >= 0) return;             // greek variants, handled below
      plot(target, id, payload.figures[id]);
    });
    if (greek && greek !== "theta") {
      Object.keys(payload.figures).forEach(function (id) {
        var split = id.split("::");
        if (split.length === 2 && split[1] === greek) plot(target, split[0], payload.figures[id]);
      });
    }
  }

  function panelKey() {
    if (state.tab === "forecast") return key(["forecast", state.theme, state.horizon]);
    if (state.tab === "models")   return key(["models", state.theme, state.horizon, state.model]);
    if (state.tab === "backtest") return key(["backtest", state.theme, state.horizon, state.model]);
    if (state.tab === "options")  return key(["options", state.theme, state.optionModel]);
    return key(["method", state.theme]);
  }

  function syncControls() {
    document.documentElement.setAttribute("data-theme", state.theme);
    themeButton.textContent = state.theme === "dark" ? "Light" : "Dark";

    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (b) {
      b.classList.toggle("tab--selected", b.dataset.tab === state.tab);
    });
    Array.prototype.forEach.call(document.querySelectorAll("#horizon-toggle input"), function (i) {
      i.checked = i.value === state.horizon;
    });

    var horizonUsed = state.tab === "forecast" || state.tab === "models" || state.tab === "backtest";
    document.getElementById("horizon-toggle").style.display = horizonUsed ? "" : "none";

    var wantsModel = state.tab === "models" || state.tab === "backtest" || state.tab === "options";
    modelPicker.hidden = !wantsModel;
    if (wantsModel) {
      var list = state.tab === "options" ? BOOT.optionModels : BOOT.models;
      var current = state.tab === "options" ? state.optionModel : state.model;
      modelPicker.innerHTML = list.map(function (m) {
        return '<option value="' + m + '"' + (m === current ? " selected" : "") + ">"
          + ((BOOT.modelNames || {})[m] || m) + "</option>";
      }).join("");
    }
    greekPicker.hidden = state.tab !== "options";
    if (state.tab === "options" && !greekPicker.innerHTML) {
      greekPicker.innerHTML = (BOOT.greeks || []).map(function (g) {
        return '<option value="' + g + '">' + g.charAt(0).toUpperCase() + g.slice(1) + "</option>";
      }).join("");
    }
    greekPicker.value = state.greek;
  }

  var pending = 0;
  function render() {
    syncControls();
    var token = ++pending;
    var mKey = key(["masthead", state.theme]);
    load(mKey).then(function (p) { if (p && token === pending) draw(masthead, p); });
    load(panelKey()).then(function (p) {
      if (token !== pending) return;
      if (!p) {
        content.innerHTML = '<div class="panel loading-note">This view is not available in this build.</div>';
        return;
      }
      draw(content, p, state.tab === "options" ? state.greek : null);
      window.scrollTo({ top: 0, behavior: "instant" });
    }).catch(function (err) {
      content.innerHTML = '<div class="panel loading-note">Could not load this view: '
        + String(err.message || err) + "</div>";
    });
  }

  document.getElementById("tabs").addEventListener("click", function (e) {
    var btn = e.target.closest(".tab");
    if (btn) { state.tab = btn.dataset.tab; render(); }
  });
  document.getElementById("horizon-toggle").addEventListener("change", function (e) {
    state.horizon = e.target.value;
    localStorage.setItem("btc-horizon", state.horizon);
    render();
  });
  modelPicker.addEventListener("change", function (e) {
    if (state.tab === "options") {
      state.optionModel = e.target.value;
      localStorage.setItem("btc-option-model", state.optionModel);
    } else {
      state.model = e.target.value;
      localStorage.setItem("btc-model", state.model);
    }
    render();
  });
  greekPicker.addEventListener("change", function (e) {
    state.greek = e.target.value;
    localStorage.setItem("btc-greek", state.greek);
    render();
  });
  themeButton.addEventListener("click", function () {
    state.theme = state.theme === "dark" ? "light" : "dark";
    localStorage.setItem("btc-theme", state.theme);
    render();
  });

  document.documentElement.setAttribute("data-theme", state.theme);
  render();
})();
