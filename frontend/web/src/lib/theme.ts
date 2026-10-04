// Chart colors, read from the CSS design tokens (styles/index.css) so the
// canvas-drawn charts can never drift from the rest of the UI again.
function token(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function chartColors() {
  return {
    up: token("--color-up", "#3fd08a"),
    down: token("--color-down", "#f25763"),
    pred: token("--color-accent-hi", "#ff8a4c"),
    sma: token("--color-sky", "#6cb8ff"),
    ema: token("--color-rose", "#ee7fa8"),
    sky: token("--color-sky", "#6cb8ff"),
    bollinger: token("--color-dim", "#a69f90"),
    fib: token("--color-faint", "#6f695d"),
    vwap: token("--color-violet", "#b393ff"),
    pattern: token("--color-violet", "#b393ff"),
    text: token("--color-faint", "#6f695d"),
    grid: "rgba(236, 230, 216, 0.05)",
    border: "rgba(236, 230, 216, 0.1)",
    band: "rgba(255, 106, 31, 0.22)",
    bandFade: "rgba(255, 106, 31, 0)",
    volUp: "rgba(63, 208, 138, 0.28)",
    volDown: "rgba(242, 87, 99, 0.28)",
  };
}

export const CHART_FONT = "'Geist Variable', ui-sans-serif, system-ui, sans-serif";
