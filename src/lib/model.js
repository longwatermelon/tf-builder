import {
  addMatrix,
  addRowVec,
  identity,
  matmul,
  reluMatrix,
  resizeMatrix,
  resizeVec,
  sliceBlock,
  softmaxRows,
  transpose,
  zeroVec,
  zeros,
} from "./linalg";

// module kinds the player can stack; embed is always the first module
export const MODULE_KINDS = {
  embed: { label: "Embedding", tex: "\\mathrm{Embed}" },
  attn: { label: "Attention Head", tex: "\\mathrm{Attn}" },
  mlp: { label: "MLP", tex: "\\mathrm{MLP}" },
  linear: { label: "Linear", tex: "\\mathrm{Linear}" },
};

let nextModuleId = 1;

// unique id so react keys and selection survive reordering
function makeId(type) {
  nextModuleId += 1;
  return `${type}-${nextModuleId}`;
}

export function createEmbed({ useE = true, useP = false } = {}) {
  return { id: makeId("embed"), type: "embed", useE, useP, W_E: [], W_P: [] };
}

export function createAttn({ dHead = 2, useMask = false } = {}) {
  return { id: makeId("attn"), type: "attn", dHead, useMask, W_Q: [], W_K: [], W_V: [], W_O: [], mask: [] };
}

export function createMlp({ dHidden = 2 } = {}) {
  return { id: makeId("mlp"), type: "mlp", dHidden, useB1: true, useB2: true, W1: [], b1: [], W2: [], b2: [] };
}

export function createLinear({ dOut = 2 } = {}) {
  return { id: makeId("linear"), type: "linear", dOut, useB: true, W: [], b: [] };
}

// starting model for a fresh puzzle attempt: a single embedding, sized so the stream is already
// the logit axis — the player adds everything else themselves
export function createInitialModel(puzzle) {
  const model = { dModel: puzzle.vocab.length, modules: [createEmbed()] };
  return reconcileShapes(model, puzzle);
}

// output width of a module given its input width
export function moduleOutputWidth(module, dIn) {
  return module.type === "linear" ? module.dOut : dIn;
}

// walk the stack and resize every matrix so adjacent module widths line up
export function reconcileShapes(model, puzzle) {
  const vocabSize = puzzle.vocab.length;
  const maxLen = puzzle.maxLen;
  let width = model.dModel;

  const modules = model.modules.map((module) => {
    if (module.type === "embed") {
      width = model.dModel;
      return {
        ...module,
        W_E: resizeMatrix(module.W_E, vocabSize, width),
        W_P: resizeMatrix(module.W_P, maxLen, width),
      };
    }
    if (module.type === "attn") {
      const dh = module.dHead;
      return {
        ...module,
        W_Q: resizeMatrix(module.W_Q, width, dh),
        W_K: resizeMatrix(module.W_K, width, dh),
        W_V: resizeMatrix(module.W_V, width, dh),
        W_O: resizeMatrix(module.W_O, dh, width),
        mask: resizeMatrix(module.mask, maxLen, maxLen),
      };
    }
    if (module.type === "mlp") {
      const dh = module.dHidden;
      return {
        ...module,
        W1: resizeMatrix(module.W1, width, dh),
        b1: resizeVec(module.b1, dh),
        W2: resizeMatrix(module.W2, dh, width),
        b2: resizeVec(module.b2, width),
      };
    }
    // linear is the only module that changes the stream width
    const resized = {
      ...module,
      W: resizeMatrix(module.W, width, module.dOut),
      b: resizeVec(module.b, module.dOut),
    };
    width = module.dOut;
    return resized;
  });

  return { ...model, modules };
}

// input width seen by each module, indexed the same as model.modules
export function computeInputWidths(model) {
  const widths = [];
  let width = model.dModel;
  for (const module of model.modules) {
    widths.push(width);
    width = moduleOutputWidth(module, width);
  }
  return widths;
}

// parameter count for one module, used for the per-module badge
export function moduleParamCount(module, dIn, puzzle) {
  const vocabSize = puzzle.vocab.length;
  const maxLen = puzzle.maxLen;
  if (module.type === "embed") {
    return (module.useE ? vocabSize * dIn : 0) + (module.useP ? maxLen * dIn : 0);
  }
  if (module.type === "attn") {
    const dh = module.dHead;
    return dIn * dh * 3 + dh * dIn + (module.useMask ? maxLen * maxLen : 0);
  }
  if (module.type === "mlp") {
    const dh = module.dHidden;
    return dIn * dh + (module.useB1 ? dh : 0) + dh * dIn + (module.useB2 ? dIn : 0);
  }
  return dIn * module.dOut + (module.useB ? module.dOut : 0);
}

// total allocated float parameters across the stack
export function countParams(model, puzzle) {
  const widths = computeInputWidths(model);
  return model.modules.reduce((sum, module, i) => sum + moduleParamCount(module, widths[i], puzzle), 0);
}

// one attention head: scores are raw QK^T (no 1/sqrt(d) scaling) plus the optional mask
function runAttention(module, x, T) {
  const Q = matmul(x, module.W_Q);
  const K = matmul(x, module.W_K);
  const V = matmul(x, module.W_V);
  let scores = matmul(Q, transpose(K));
  if (module.useMask) {
    const mask = sliceBlock(module.mask, T, T);
    scores = scores.map((row, i) => row.map((v, j) => v + mask[i][j]));
  }
  const pattern = softmaxRows(scores);
  const out = matmul(matmul(pattern, V), module.W_O);
  return { Q, K, V, scores, pattern, out };
}

// forward pass over one token sequence; returns a snapshot after every module
export function forward(model, puzzle, tokenIds) {
  const T = tokenIds.length;
  const stages = [];
  let x = zeros(T, model.dModel);
  let width = model.dModel;
  let error = null;

  for (const module of model.modules) {
    const dIn = width;
    const extras = [];

    if (module.type === "embed") {
      x = zeros(T, model.dModel);
      if (module.useE) {
        for (let i = 0; i < T; i++) {
          const row = module.W_E[tokenIds[i]] ?? zeroVec(model.dModel);
          for (let j = 0; j < model.dModel; j++) x[i][j] += row[j];
        }
      }
      if (module.useP) {
        for (let i = 0; i < T; i++) {
          const row = module.W_P[i] ?? zeroVec(model.dModel);
          for (let j = 0; j < model.dModel; j++) x[i][j] += row[j];
        }
      }
    } else if (module.type === "attn") {
      const result = runAttention(module, x, T);
      extras.push({ key: "pattern", label: "Attention pattern A", matrix: result.pattern, kind: "pattern" });
      x = addMatrix(x, result.out);
    } else if (module.type === "mlp") {
      const hiddenInput = matmul(x, module.W1);
      const hidden = reluMatrix(module.useB1 ? addRowVec(hiddenInput, module.b1) : hiddenInput);
      extras.push({ key: "hidden", label: "Hidden (post-ReLU)", matrix: hidden, kind: "stream" });
      const mlpOutput = matmul(hidden, module.W2);
      x = addMatrix(x, module.useB2 ? addRowVec(mlpOutput, module.b2) : mlpOutput);
    } else {
      const linearOutput = matmul(x, module.W);
      x = module.useB ? addRowVec(linearOutput, module.b) : linearOutput;
      width = module.dOut;
    }

    stages.push({ moduleId: module.id, type: module.type, dIn, width, matrix: x, extras });
  }

  const vocabSize = puzzle.vocab.length;
  if (width !== vocabSize) {
    error = `Final width is ${width}, but the output layer must be ${vocabSize} wide (one logit per vocab token).`;
  }
  const probs = error ? null : softmaxRows(x);

  return { stages, logits: x, probs, error };
}

// per-position verdict for one test sequence
export function gradeSequence(probs, targetIds, epsilon) {
  if (!probs) return null;
  return probs.map((row, i) => {
    const targetId = targetIds[i];
    let topId = 0;
    let topP = -Infinity;
    let runnerP = -Infinity;
    for (let v = 0; v < row.length; v++) {
      if (row[v] > topP) {
        runnerP = topP;
        topP = row[v];
        topId = v;
      } else if (row[v] > runnerP) {
        runnerP = row[v];
      }
    }
    if (targetId === null || targetId === undefined) return { skipped: true, ok: true, topId, margin: topP - runnerP };
    const margin = row[targetId] - Math.max(...row.filter((_, v) => v !== targetId));
    return { skipped: false, ok: topId === targetId && margin >= epsilon, topId, margin };
  });
}

// grade one set of cases, retaining detailed passes only when the UI needs them
function evaluateTestSet(model, puzzle, tests, includeDetails) {
  const results = [];
  let passed = 0;
  for (const test of tests) {
    const tokenIds = test.tokens.map((t) => puzzle.vocab.indexOf(t));
    const targetIds = test.targets.map((t) => (t === null ? null : puzzle.vocab.indexOf(t)));
    const pass = forward(model, puzzle, tokenIds);
    const grades = gradeSequence(pass.probs, targetIds, puzzle.epsilon);
    const ok = !!grades && grades.every((g) => g.ok);
    if (ok) passed += 1;
    if (includeDetails) results.push({ test, tokenIds, targetIds, pass, grades, ok });
  }
  return { results, passed, allPassed: tests.length > 0 && passed === tests.length };
}

// grade visible samples plus the puzzle's exhaustive rule-derived validation domain
export function evaluatePuzzle(model, puzzle) {
  const samples = evaluateTestSet(model, puzzle, puzzle.tests, true);
  const validationTests = puzzle.validationTests ?? puzzle.tests;
  const validation = evaluateTestSet(model, puzzle, validationTests, false);
  const params = countParams(model, puzzle);
  return {
    results: samples.results,
    solved: validation.allPassed,
    validationPassed: validation.passed,
    validationTotal: validationTests.length,
    params,
    elegant: validation.allPassed && params <= puzzle.canonicalParams,
  };
}

// helper used by canonical solutions to fill a matrix from a sparse entry list
export function withEntries(rows, cols, entries) {
  const m = zeros(rows, cols);
  for (const [i, j, v] of entries) m[i][j] = v;
  return m;
}

export { identity };
