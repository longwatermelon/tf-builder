// parsing and display helpers for hand-edited numeric cells

// parses a hand-typed cell; inf/-inf are only meaningful for attention masks, so they are opt-in
export function parseNumberText(text, allowInfinity = false) {
  const trimmed = String(text).trim();
  if (trimmed === "") return 0;
  const lowered = trimmed.toLowerCase();
  if (lowered === "inf" || lowered === "+inf" || lowered === "infinity") return allowInfinity ? Infinity : null;
  if (lowered === "-inf" || lowered === "-infinity") return allowInfinity ? -Infinity : null;
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(trimmed)) return null;
  const value = Number(trimmed);
  // an overflowing literal like 1e999 must not sneak infinity past allowInfinity
  return Number.isFinite(value) ? value : null;
}

// editable text for a stored value; round-trippable so the cell always shows the real weight
export function numberToText(value) {
  if (value === Infinity) return "inf";
  if (value === -Infinity) return "-inf";
  if (!Number.isFinite(value)) return "0";
  return String(value);
}

// compact read-only display for computed activations
export function formatActivation(value) {
  if (value === Infinity) return "∞";
  if (value === -Infinity) return "-∞";
  if (!Number.isFinite(value)) return "NaN";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e4 || abs < 1e-3) return value.toExponential(1);
  return value.toFixed(abs >= 100 ? 1 : 2);
}

export function formatProbability(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(3);
}
