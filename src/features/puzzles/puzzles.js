import {
  countParams,
  createAttn,
  createEmbed,
  createLinear,
  reconcileShapes,
  withEntries,
} from "../../lib/model";

// logit gap used by canonical solutions; large enough to clear epsilon comfortably
const LOGIT_SCALE = 5;

// predict the current token: the embedding alone can already be the logits
function echoSolution() {
  return {
    dModel: 3,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 3, [[0, 0, LOGIT_SCALE], [1, 1, LOGIT_SCALE], [2, 2, LOGIT_SCALE]]) },
    ],
  };
}

// predict the cyclic successor of the current token
function successorSolution() {
  return {
    dModel: 3,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 3, [[0, 1, LOGIT_SCALE], [1, 2, LOGIT_SCALE], [2, 0, LOGIT_SCALE]]) },
    ],
  };
}

// output depends only on position parity, so token embeddings are switched off
function alternateSolution() {
  return {
    dModel: 2,
    modules: [
      {
        ...createEmbed({ useE: false, useP: true }),
        W_P: withEntries(4, 2, [[0, 0, LOGIT_SCALE], [1, 1, LOGIT_SCALE], [2, 0, LOGIT_SCALE], [3, 1, LOGIT_SCALE]]),
      },
    ],
  };
}

// previous-token head: the mask alone picks position i-1, OV copies the token across
function prevTokenSolution() {
  const NEG = -Infinity;
  const mask = [
    [0, NEG, NEG, NEG],
    [0, NEG, NEG, NEG],
    [NEG, 0, NEG, NEG],
    [NEG, NEG, 0, NEG],
  ];
  return {
    dModel: 6,
    modules: [
      // dims 0-2 hold the current token, dims 3-5 are left free for the head to write into
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 6, [[0, 0, 1], [1, 1, 1], [2, 2, 1]]) },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(6, 3, [[0, 0, 1], [1, 1, 1], [2, 2, 1]]),
        W_O: withEntries(3, 6, [[0, 3, 1], [1, 4, 1], [2, 5, 1]]),
        mask,
      },
      {
        ...createLinear({ dOut: 3 }),
        W: withEntries(6, 3, [[3, 0, LOGIT_SCALE], [4, 1, LOGIT_SCALE], [5, 2, LOGIT_SCALE]]),
      },
    ],
  };
}

const PUZZLE_DEFS = [
  {
    id: "echo",
    name: "Echo",
    difficulty: "tutorial",
    blurb: "Predict the token at the current position.",
    formula: "y_i = x_i",
    hint: "The residual stream is already 3 wide, and the final softmax reads it directly — no extra modules needed.",
    vocab: ["a", "b", "c"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "c", "a"], targets: ["a", "b", "c", "a"] },
      { tokens: ["c", "a", "b", "b"], targets: ["c", "a", "b", "b"] },
    ],
    solutionFactory: echoSolution,
  },
  {
    id: "successor",
    name: "Successor",
    difficulty: "tutorial",
    blurb: "Predict the next letter in the cycle a → b → c → a.",
    formula: "y_i = \\mathrm{next}(x_i)",
    hint: "Same shape as Echo, but the embedding rows point at a different logit.",
    vocab: ["a", "b", "c"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "c", "a"], targets: ["b", "c", "a", "b"] },
      { tokens: ["c", "c", "a", "b"], targets: ["a", "a", "b", "c"] },
    ],
    solutionFactory: successorSolution,
  },
  {
    id: "alternate",
    name: "Alternate",
    difficulty: "easy",
    blurb: "Output a at even positions and b at odd positions, whatever the input tokens are.",
    formula: "y_i = \\begin{cases} a, & i \\text{ even} \\\\ b, & i \\text{ odd} \\end{cases}",
    hint: "The answer never depends on the token — turn W_E off entirely and pay only for W_P.",
    vocab: ["a", "b"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "a", "b", "b"], targets: ["a", "b", "a", "b"] },
      { tokens: ["b", "a", "a", "b"], targets: ["a", "b", "a", "b"] },
    ],
    solutionFactory: alternateSolution,
  },
  {
    id: "prev_token",
    name: "Previous Token",
    difficulty: "medium",
    blurb: "Predict the token one position back. Position 0 has no predecessor, so it should predict the ⋄ marker it already sits on.",
    formula: "y_i = x_{i-1}, \\quad y_0 = \\diamond",
    hint: "A hand-written mask can point every query at position i-1 by itself, leaving W_Q and W_K at zero. Give the head its own slice of the stream to write into.",
    vocab: ["⋄", "a", "b"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["⋄", "a", "b", "a"], targets: ["⋄", "⋄", "a", "b"] },
      { tokens: ["⋄", "b", "b", "a"], targets: ["⋄", "⋄", "b", "b"] },
      { tokens: ["⋄", "a", "a", "b"], targets: ["⋄", "⋄", "a", "a"] },
    ],
    solutionFactory: prevTokenSolution,
  },
];

// build a fully shaped canonical model for a puzzle
export function buildSolution(puzzle) {
  return reconcileShapes(puzzle.solutionFactory(), puzzle);
}

// the elegance bar is derived from the canonical solution, never hand-written
export const PUZZLES = PUZZLE_DEFS.map((puzzle) => ({
  ...puzzle,
  canonicalParams: countParams(reconcileShapes(puzzle.solutionFactory(), puzzle), puzzle),
}));

export function getPuzzle(id) {
  return PUZZLES.find((puzzle) => puzzle.id === id) ?? PUZZLES[0];
}
