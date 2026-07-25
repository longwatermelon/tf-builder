// user-supplied names for the row/column axes of every matrix in the app.
//
// a store is a plain object keyed by axis, each holding an { index: name } map:
//   { d: { 0: "in0" }, "attn-3:h": { 1: "cnt" } }
//
// the stream, position and vocab axes are shared by the whole stack, so naming d0 in the
// embedding names it in every later module too. head dims, hidden units and a linear's output
// dims are separate vector spaces per module, so their keys carry the owning module's id.

export const MAX_LABEL_LEN = 4;

export const RESIDUAL_AXIS = "d";
export const POSITION_AXIS = "p";
export const VOCAB_AXIS = "vocab";

// key for an axis that only exists inside one module
export function moduleAxis(moduleId, prefix) {
  return `${moduleId}:${prefix}`;
}

// the axis a linear emits: the vocab axis once it is wide enough to be the logits
function linearOutAxis(module, puzzle) {
  return module.dOut === puzzle.vocab.length ? VOCAB_AXIS : moduleAxis(module.id, "u");
}

// stream axis entering and leaving each module, indexed the same as model.modules; the stream
// starts on the residual axis and takes on the output axis of every linear it passes through
export function computeStreamAxes(model, puzzle) {
  const inputs = [];
  const outputs = [];
  let key = RESIDUAL_AXIS;
  for (const module of model.modules) {
    inputs.push(key);
    if (module.type === "linear") key = linearOutAxis(module, puzzle);
    outputs.push(key);
  }
  return { inputs, outputs };
}

// d0 / h0 / ... names for an axis the user has not renamed
export function defaultLabels(count, prefix) {
  return Array.from({ length: count }, (_, i) => `${prefix}${i}`);
}

// defaults for a stream axis, which is spelled with the vocabulary once it holds the logits
export function streamDefaults(key, count, puzzle) {
  if (key === VOCAB_AXIS) return puzzle.vocab.slice(0, count);
  return defaultLabels(count, key === RESIDUAL_AXIS ? "d" : "u");
}

// custom names where the user set them, defaults everywhere else
export function resolveLabels(labels, key, defaults) {
  const axis = key ? labels[key] : null;
  if (!axis) return defaults;
  return defaults.map((fallback, i) => axis[i] ?? fallback);
}

// whether the user named this index; the resolved label alone cannot tell, since a custom name
// may happen to match the default spelling of one of the axis's matrices
export function hasLabel(labels, key, index) {
  return key ? labels[key]?.[index] !== undefined : false;
}

// name one index of an axis; an empty name drops back to the default
export function setLabel(labels, key, index, name) {
  const trimmed = name.trim().slice(0, MAX_LABEL_LEN);
  const axis = { ...(labels[key] ?? {}) };
  if (trimmed) axis[index] = trimmed;
  else delete axis[index];

  const next = { ...labels };
  if (Object.keys(axis).length) next[key] = axis;
  else delete next[key];
  return next;
}
