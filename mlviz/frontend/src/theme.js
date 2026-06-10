// ─── mlviz design system ──────────────────────────────────────────────────────
// Dark-first, engineering-tool aesthetic: near-black surfaces, hairline
// borders, a single indigo accent, mono numerals for data.

export const FONT_UI =
  '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif';
export const FONT_MONO =
  '"SF Mono", ui-monospace, "JetBrains Mono", Menlo, monospace';

export const DARK = {
  bg:               "#09090B",
  canvas:           "#0C0C0E",
  gridDot:          "#1B1B1F",
  surface:          "#101013",
  surfaceSecondary: "#17171B",
  nodeFill:         "#131316",
  border:           "#26262B",
  borderHover:      "#47474F",
  text:             "#F4F4F5",
  textSecondary:    "#A1A1AA",
  textTertiary:     "#63636B",
  separator:        "#1F1F24",
  accent:           "#7B86E2",
  accentFill:       "rgba(123, 134, 226, 0.13)",
  green:            "#4CC38A",
  red:              "#EC5E66",
  classFill: [
    "rgba(94, 158, 255, 0.14)",
    "rgba(76, 195, 138, 0.14)",
    "rgba(245, 165, 36, 0.14)",
    "rgba(181, 131, 232, 0.14)",
    "rgba(236, 94, 123, 0.14)",
    "rgba(45, 212, 191, 0.14)",
  ],
};

export const LIGHT = {
  bg:               "#FAFAFA",
  canvas:           "#FCFCFC",
  gridDot:          "#E8E8EA",
  surface:          "#FFFFFF",
  surfaceSecondary: "#F4F4F5",
  nodeFill:         "#FFFFFF",
  border:           "#E4E4E7",
  borderHover:      "#A1A1AA",
  text:             "#18181B",
  textSecondary:    "#71717A",
  textTertiary:     "#A1A1AA",
  separator:        "#ECECEF",
  accent:           "#5E6AD2",
  accentFill:       "rgba(94, 106, 210, 0.09)",
  green:            "#179C56",
  red:              "#DC2626",
  classFill: [
    "rgba(46, 124, 245, 0.10)",
    "rgba(23, 156, 86, 0.10)",
    "rgba(217, 137, 7, 0.12)",
    "rgba(147, 84, 211, 0.10)",
    "rgba(221, 60, 96, 0.10)",
    "rgba(13, 170, 152, 0.12)",
  ],
};

// Class accents tuned to read on both themes.
export const CLASS_COLORS = [
  "#5E9EFF", "#4CC38A", "#F5A524", "#B583E8", "#EC5E7B", "#2DD4BF",
];

export const CLASS_COLORS_LIGHT = [
  "#2E7CF5", "#179C56", "#D98907", "#9354D3", "#DD3C60", "#0DAA98",
];

export const getTheme = (dark) => (dark ? DARK : LIGHT);
export const getClassColors = (dark) => (dark ? CLASS_COLORS : CLASS_COLORS_LIGHT);

// ─── Text helpers ─────────────────────────────────────────────────────────────
export const sf = (f) => (f ? f.replace(" (cm)", "") : "");
export const clampLabel = (s, max = 19) =>
  s && s.length > max ? s.slice(0, max - 1) + "…" : (s ?? "");
