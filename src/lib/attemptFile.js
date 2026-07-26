const ATTEMPT_FORMAT = "tf-builder-attempt";

const ATTEMPT_VERSION = 1;

const MAX_WIDTH = 16;

// convert non-finite mask entries into JSON-safe spellings without changing ordinary weights
function encodeNumber(_key, value) {
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  return value;
}

// report a validation failure with the exact field that made the file unusable
function invalid(path, message) {
  throw new Error(`${path} ${message}`);
}

// read one bounded architecture width
function readWidth(value, path) {
  if (!Number.isInteger(value) || value < 1 || value > MAX_WIDTH) {
    invalid(path, `must be an integer from 1 to ${MAX_WIDTH}.`);
  }
  return value;
}

// read one strict boolean toggle
function readBoolean(value, path) {
  if (typeof value !== "boolean") invalid(path, "must be true or false.");
  return value;
}

// read one weight, optionally decoding the special values supported by attention masks
function readNumber(value, path, allowInfinity = false) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (allowInfinity && value === "Infinity") return Infinity;
  if (allowInfinity && value === "-Infinity") return -Infinity;
  invalid(path, allowInfinity ? "must be a number, \"Infinity\", or \"-Infinity\"." : "must be a finite number.");
}

// read a vector with the exact shape required by the architecture
function readVector(value, length, path, allowInfinity = false) {
  if (!Array.isArray(value) || value.length !== length) invalid(path, `must have length ${length}.`);
  return value.map((entry, index) => readNumber(entry, `${path}[${index}]`, allowInfinity));
}

// read a dense matrix with the exact shape required by the architecture
function readMatrix(value, rows, cols, path, allowInfinity = false) {
  if (!Array.isArray(value) || value.length !== rows) invalid(path, `must have ${rows} rows.`);
  return value.map((row, index) => readVector(row, cols, `${path}[${index}]`, allowInfinity));
}

// preserve stable React keys while rejecting IDs that cannot safely identify a module
function readId(value, path, ids) {
  if (typeof value !== "string" || value.length === 0 || value.length > 100) {
    invalid(path, "must be a non-empty string of at most 100 characters.");
  }
  if (ids.has(value)) invalid(path, "must be unique.");
  ids.add(value);
  return value;
}

// validate and copy an imported model so unknown file fields never enter application state
function readModel(value, puzzle) {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid("model", "must be an object.");
  const dModel = readWidth(value.dModel, "model.dModel");
  if (!Array.isArray(value.modules) || value.modules.length < 1) {
    invalid("model.modules", "must contain at least one module.");
  }

  const ids = new Set();
  let width = dModel;
  const modules = value.modules.map((module, index) => {
    const path = `model.modules[${index}]`;
    if (!module || typeof module !== "object" || Array.isArray(module)) invalid(path, "must be an object.");
    const id = readId(module.id, `${path}.id`, ids);

    if (module.type === "embed") {
      if (index !== 0) invalid(`${path}.type`, "can only be embed for the first module.");
      return {
        id,
        type: "embed",
        useE: readBoolean(module.useE, `${path}.useE`),
        useP: readBoolean(module.useP, `${path}.useP`),
        W_E: readMatrix(module.W_E, puzzle.vocab.length, dModel, `${path}.W_E`),
        W_P: readMatrix(module.W_P, puzzle.maxLen, dModel, `${path}.W_P`),
      };
    }
    if (index === 0) invalid(`${path}.type`, "must be embed for the first module.");

    if (module.type === "attn") {
      const dHead = readWidth(module.dHead, `${path}.dHead`);
      return {
        id,
        type: "attn",
        dHead,
        useMask: readBoolean(module.useMask, `${path}.useMask`),
        W_Q: readMatrix(module.W_Q, width, dHead, `${path}.W_Q`),
        W_K: readMatrix(module.W_K, width, dHead, `${path}.W_K`),
        W_V: readMatrix(module.W_V, width, dHead, `${path}.W_V`),
        W_O: readMatrix(module.W_O, dHead, width, `${path}.W_O`),
        mask: readMatrix(module.mask, puzzle.maxLen, puzzle.maxLen, `${path}.mask`, true),
      };
    }

    if (module.type === "mlp") {
      const dHidden = readWidth(module.dHidden, `${path}.dHidden`);
      return {
        id,
        type: "mlp",
        dHidden,
        useB1: readBoolean(module.useB1, `${path}.useB1`),
        useB2: readBoolean(module.useB2, `${path}.useB2`),
        W1: readMatrix(module.W1, width, dHidden, `${path}.W1`),
        b1: readVector(module.b1, dHidden, `${path}.b1`),
        W2: readMatrix(module.W2, dHidden, width, `${path}.W2`),
        b2: readVector(module.b2, width, `${path}.b2`),
      };
    }

    if (module.type === "linear") {
      const dOut = readWidth(module.dOut, `${path}.dOut`);
      const imported = {
        id,
        type: "linear",
        dOut,
        useB: readBoolean(module.useB, `${path}.useB`),
        W: readMatrix(module.W, width, dOut, `${path}.W`),
        b: readVector(module.b, dOut, `${path}.b`),
      };
      width = dOut;
      return imported;
    }

    invalid(`${path}.type`, "must be embed, attn, mlp, or linear.");
  });

  return { dModel, modules };
}

// encode one puzzle attempt as a readable, versioned JSON document
export function encodeAttempt(puzzleId, model) {
  return JSON.stringify(
    {
      format: ATTEMPT_FORMAT,
      version: ATTEMPT_VERSION,
      puzzleId,
      model,
    },
    encodeNumber,
    2,
  );
}

// decode and validate an attempt for the currently selected puzzle
export function decodeAttempt(text, puzzle) {
  let document;
  try {
    document = JSON.parse(text);
  } catch {
    throw new Error("The selected file is not valid JSON.");
  }
  if (!document || typeof document !== "object" || Array.isArray(document)) {
    throw new Error("The attempt file must contain a JSON object.");
  }
  if (document.format !== ATTEMPT_FORMAT || document.version !== ATTEMPT_VERSION) {
    throw new Error(`This is not a supported ${ATTEMPT_FORMAT} v${ATTEMPT_VERSION} file.`);
  }
  if (document.puzzleId !== puzzle.id) {
    throw new Error(`This attempt belongs to “${document.puzzleId}”, not the current “${puzzle.id}” puzzle.`);
  }
  return readModel(document.model, puzzle);
}
