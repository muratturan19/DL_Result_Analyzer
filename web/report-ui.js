(function () {
  const STORAGE_KEY = "dlra-report-theme";
  const STATUS_COPY = {
    good: "Hedef üstü",
    warning: "Sınırda",
    critical: "Beklenenin altında",
    info: "Bilgi amaçlı"
  };

  const METRIC_THRESHOLDS = {
    precision: { target: 0.75, tolerance: 0.02, direction: "higher" },
    recall: { target: 0.85, tolerance: 0.02, direction: "higher" },
    f1: { target: 0.8, tolerance: 0.02, direction: "higher" },
    map50: { target: 0.75, tolerance: 0.03, direction: "higher" },
    map75: { target: 0.7, tolerance: 0.03, direction: "higher" },
    map50_95: { target: 0.55, tolerance: 0.03, direction: "higher" },
    loss: { target: 0.02, tolerance: 0.02, direction: "lower" }
  };

  const htmlEl = document.documentElement;

  function safeLocalStorage(action, key, value) {
    try {
      if (!window.localStorage) {
        return null;
      }
      if (action === "get") {
        return localStorage.getItem(key);
      }
      if (action === "set") {
        localStorage.setItem(key, value);
      }
      if (action === "remove") {
        localStorage.removeItem(key);
      }
    } catch (error) {
      return null;
    }
    return null;
  }

  function applyTheme(theme, persist = true) {
    if (!theme) {
      return;
    }
    htmlEl.setAttribute("data-theme", theme);
    if (persist) {
      safeLocalStorage("set", STORAGE_KEY, theme);
    }
    updateThemeToggle(theme);
  }

  function getStoredTheme() {
    const stored = safeLocalStorage("get", STORAGE_KEY);
    return stored === "dark" || stored === "light" ? stored : null;
  }

  function resolvePreferredTheme() {
    const stored = getStoredTheme();
    if (stored) {
      return stored;
    }
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }
    return "light";
  }

  function updateThemeToggle(theme) {
    const toggle = document.querySelector("[data-theme-toggle]");
    if (!toggle) {
      return;
    }
    const iconUse = toggle.querySelector("use");
    const text = toggle.querySelector("[data-theme-toggle-label]");
    if (iconUse) {
      iconUse.setAttribute("href", theme === "dark" ? "#icon-sun" : "#icon-moon");
      iconUse.setAttribute("xlink:href", theme === "dark" ? "#icon-sun" : "#icon-moon");
    }
    if (text) {
      text.textContent = theme === "dark" ? "Aydınlık tema" : "Koyu tema";
    }
  }

  function initThemeToggle() {
    const toggle = document.querySelector("[data-theme-toggle]");
    if (!toggle) {
      return;
    }
    toggle.addEventListener("click", function () {
      const currentTheme = htmlEl.getAttribute("data-theme") === "dark" ? "dark" : "light";
      const nextTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(nextTheme, true);
    });
  }

  function initSystemThemeWatcher() {
    if (!window.matchMedia) {
      return;
    }
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener("change", function (event) {
      if (getStoredTheme()) {
        return;
      }
      applyTheme(event.matches ? "dark" : "light", false);
    });
  }

  function initPrintButton() {
    const printBtn = document.querySelector("[data-print-report]");
    if (!printBtn) {
      return;
    }
    printBtn.addEventListener("click", function () {
      window.print();
    });
  }

  function determineStatus(metric, value) {
    if (Number.isNaN(value)) {
      return "info";
    }
    const config = METRIC_THRESHOLDS[metric];
    if (!config || typeof config.target !== "number") {
      return "info";
    }
    const direction = config.direction || "higher";
    const tolerance = typeof config.tolerance === "number" ? config.tolerance : 0;
    if (direction === "lower") {
      if (value <= config.target) {
        return "good";
      }
      if (value <= config.target + tolerance) {
        return "warning";
      }
      return "critical";
    }

    if (value >= config.target) {
      return "good";
    }
    if (value >= config.target - tolerance) {
      return "warning";
    }
    return "critical";
  }

  function initMetricCards() {
    const cards = document.querySelectorAll("[data-metric-card]");
    if (!cards.length) {
      return;
    }
    cards.forEach(function (card) {
      const metric = card.getAttribute("data-metric");
      const raw = parseFloat(card.getAttribute("data-raw"));
      const label = card.getAttribute("data-label") || metric;
      const status = determineStatus(metric, raw);
      card.setAttribute("data-status", status);
      const statusEl = card.querySelector("[data-status-text]");
      if (statusEl) {
        statusEl.textContent = STATUS_COPY[status] || STATUS_COPY.info;
      }
      if (label) {
        card.setAttribute("aria-label", label + " - " + (STATUS_COPY[status] || ""));
      }
    });
  }

  function initCollapsibles() {
    const containers = document.querySelectorAll("[data-collapsible]");
    if (!containers.length) {
      return;
    }
    containers.forEach(function (container, index) {
      const trigger = container.querySelector("[data-collapsible-trigger]");
      const content = container.querySelector("[data-collapsible-content]");
      if (!trigger || !content) {
        return;
      }
      const contentId = content.id || "collapsible-" + index;
      content.id = contentId;
      trigger.setAttribute("aria-controls", contentId);
      const defaultOpen = container.hasAttribute("data-open");
      trigger.setAttribute("aria-expanded", defaultOpen ? "true" : "false");
      content.hidden = !defaultOpen;
      trigger.addEventListener("click", function () {
        const expanded = trigger.getAttribute("aria-expanded") === "true";
        trigger.setAttribute("aria-expanded", expanded ? "false" : "true");
        content.hidden = expanded;
        if (expanded) {
          container.removeAttribute("data-open");
        } else {
          container.setAttribute("data-open", "");
        }
      });
    });
  }

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  ready(function () {
    applyTheme(resolvePreferredTheme(), false);
    initThemeToggle();
    initSystemThemeWatcher();
    initPrintButton();
    initMetricCards();
    initCollapsibles();
  });
})();
