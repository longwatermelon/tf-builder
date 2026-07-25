// shared palette tokens — vscode dark+ inspired
export const COLORS = {
  bg: "#1e1e1e",
  panel: "#252526",
  panelBorder: "#3c3c3c",
  surface: "#2d2d2d",
  surfaceAlt: "#333334",
  accent: "#0098ff",
  accentDim: "#0098ff20",
  negative: "#f44747",
  negativeDim: "#f4474720",
  text: "#cccccc",
  textMuted: "#858585",
  textBright: "#e8e8e8",
  success: "#4ec9b0",
  successDim: "#4ec9b020",
  warn: "#dcdcaa",
  violet: "#c586c0",
};

// each module type gets a stable accent so the stack is scannable
export const MODULE_COLORS = {
  embed: "#569cd6",
  attn: "#c586c0",
  mlp: "#dcdcaa",
  linear: "#4ec9b0",
  softmax: "#9cdcfe",
};

// background tint for a probability cell, so more probability mass reads as a stronger fill
export function probabilityFill(p) {
  const clamped = Math.min(Math.max(p, 0), 1);
  return `rgba(0, 152, 255, ${(0.04 + 0.46 * clamped).toFixed(3)})`;
}

export const DIFFICULTY_COLORS = {
  tutorial: "#4ec9b0",
  easy: "#569cd6",
  medium: "#dcdcaa",
  hard: "#f44747",
};

// base action button style
export const btnStyle = {
  background: COLORS.surface,
  borderWidth: 1,
  borderStyle: "solid",
  borderColor: COLORS.panelBorder,
  borderRadius: 4,
  color: COLORS.text,
  cursor: "pointer",
  padding: "5px 12px",
  fontSize: 12,
  fontFamily: "'Sora', sans-serif",
  fontWeight: 500,
  transition: "background 0.12s, border-color 0.12s",
};

export const subtleBtnStyle = {
  background: "transparent",
  borderWidth: 1,
  borderStyle: "solid",
  borderColor: COLORS.panelBorder,
  borderRadius: 4,
  color: COLORS.textMuted,
  cursor: "pointer",
  padding: "4px 9px",
  fontSize: 11,
  fontFamily: "'Sora', sans-serif",
  fontWeight: 500,
  transition: "background 0.12s, border-color 0.12s",
};

// compact icon-style controls used in module headers and steppers
export const smallBtnStyle = {
  background: COLORS.bg,
  borderWidth: 1,
  borderStyle: "solid",
  borderColor: COLORS.panelBorder,
  borderRadius: 3,
  color: COLORS.textMuted,
  cursor: "pointer",
  padding: "0 6px",
  fontSize: 12,
  fontWeight: 600,
  lineHeight: "18px",
};

export const MONO = "'JetBrains Mono', ui-monospace, monospace";
