import {
  countParams,
  createAttn,
  createEmbed,
  createLinear,
  createMlp,
  reconcileShapes,
  withEntries,
} from "../../lib/model";

// logit gap used by canonical solutions; large enough to clear epsilon comfortably
const LOGIT_SCALE = 5;

const NEG = -Infinity;

// mask where query i attends only to key targetOf(i); a null target attends nowhere
function pointMask(len, targetOf) {
  return Array.from({ length: len }, (_, i) => {
    const target = targetOf(i);
    return Array.from({ length: len }, (_, j) => (j === target ? 0 : NEG));
  });
}

// standard causal mask: query i attends to every key at or before i
function causalMask(len) {
  return Array.from({ length: len }, (_, i) => Array.from({ length: len }, (_, j) => (j <= i ? 0 : NEG)));
}

// sparse entries copying a contiguous run of rows into a contiguous run of columns
function blockEntries(fromStart, toStart, count, value) {
  return Array.from({ length: count }, (_, k) => [fromStart + k, toStart + k, value]);
}

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
        useB: false,
        W: withEntries(6, 3, [[3, 0, LOGIT_SCALE], [4, 1, LOGIT_SCALE], [5, 2, LOGIT_SCALE]]),
      },
    ],
  };
}

// copy the token, but every ? row of the embedding points at the a logit instead
function repairSolution() {
  return {
    dModel: 3,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 3, [[0, 1, LOGIT_SCALE], [1, 1, LOGIT_SCALE], [2, 2, LOGIT_SCALE]]) },
    ],
  };
}

// echo plus a position-0 marker: W_P outranks W_E only on the first row
function startMarkerSolution() {
  return {
    dModel: 3,
    modules: [
      {
        ...createEmbed({ useE: true, useP: true }),
        W_E: withEntries(3, 3, [[0, 0, LOGIT_SCALE], [1, 1, LOGIT_SCALE], [2, 2, LOGIT_SCALE]]),
        // twice the token logit, so the marker wins at position 0 whatever the token is
        W_P: withEntries(4, 3, [[0, 0, 2 * LOGIT_SCALE]]),
      },
    ],
  };
}

// one dimension is enough to hold the class; the linear layer expands it back to the vocab
function classifySolution() {
  return {
    dModel: 1,
    modules: [
      {
        ...createEmbed({ useE: true, useP: false }),
        W_E: withEntries(7, 1, [[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, -1], [4, 0, -1]]),
      },
      {
        ...createLinear({ dOut: 7 }),
        useB: false,
        W: withEntries(1, 7, [[0, 5, LOGIT_SCALE], [0, 6, -LOGIT_SCALE]]),
      },
    ],
  };
}

// every query attends to position 0 and the head writes that token back over the stream
function broadcastSolution() {
  return {
    dModel: 3,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 3, blockEntries(0, 0, 3, 1)) },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(3, 3, blockEntries(0, 0, 3, 1)),
        // the copied token lands on the same dims at 5x the scale, so it outvotes the local token
        W_O: withEntries(3, 3, blockEntries(0, 0, 3, LOGIT_SCALE)),
        mask: pointMask(4, () => 0),
      },
    ],
  };
}

// same overwrite trick as broadcast, but the mask is the anti-diagonal
function reverseSolution() {
  return {
    dModel: 3,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 3, blockEntries(0, 0, 3, 1)) },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(3, 3, blockEntries(0, 0, 3, 1)),
        W_O: withEntries(3, 3, blockEntries(0, 0, 3, LOGIT_SCALE)),
        mask: pointMask(4, (i) => 3 - i),
      },
    ],
  };
}

// token sign in dim 0, position sign in dim 1; only a relu can turn their product into a logit gap
function agreementSolution() {
  return {
    dModel: 2,
    modules: [
      {
        ...createEmbed({ useE: true, useP: true }),
        W_E: withEntries(2, 2, [[0, 0, 1], [1, 0, -1]]),
        W_P: withEntries(4, 2, [[0, 1, 1], [1, 1, -1], [2, 1, 1], [3, 1, -1]]),
      },
      {
        ...createMlp({ dHidden: 2 }),
        // h1 fires only on (+1, +1), h2 only on (-1, -1), so h1 + h2 is the agreement bit
        W1: [[1, -1], [1, -1]],
        b1: [-1, -1],
        // 10 * agreement - 4 swamps the +-2 gap the raw token and position signs leave behind
        W2: [[LOGIT_SCALE, -LOGIT_SCALE], [LOGIT_SCALE, -LOGIT_SCALE]],
        b2: [-2, 2],
      },
    ],
  };
}

// copy token 0 into its own slot, then an mlp turns the overlap with the current token into one bit
function matchFirstSolution() {
  return {
    dModel: 7,
    modules: [
      // dims 0-2 current token, dims 3-5 token 0, dim 6 the match bit
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(5, 7, blockEntries(0, 0, 3, 1)) },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(7, 3, blockEntries(0, 0, 3, 1)),
        W_O: withEntries(3, 7, blockEntries(0, 3, 3, 1)),
        mask: pointMask(4, () => 0),
      },
      {
        ...createMlp({ dHidden: 3 }),
        useB2: false,
        // hidden unit k fires only when both one-hots carry a 1 in dim k
        W1: withEntries(7, 3, [...blockEntries(0, 0, 3, 1), ...blockEntries(3, 0, 3, 1)]),
        b1: [-1, -1, -1],
        W2: withEntries(3, 7, [[0, 6, 1], [1, 6, 1], [2, 6, 1]]),
      },
      {
        ...createLinear({ dOut: 5 }),
        W: withEntries(7, 5, [[6, 3, 2 * LOGIT_SCALE]]),
        // n sits at a constant logit that only a firing match bit can clear
        b: [0, 0, 0, 0, LOGIT_SCALE],
      },
    ],
  };
}

// two composed heads: head 1 tags each position with its predecessor, head 2 matches that tag
function inductionSolution() {
  const MATCH_SCALE = 10;
  return {
    dModel: 9,
    modules: [
      // dims 0-2 current token, dims 3-5 previous token, dims 6-8 the retrieved token
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(3, 9, blockEntries(0, 0, 3, 1)) },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(9, 3, blockEntries(0, 0, 3, 1)),
        W_O: withEntries(3, 9, blockEntries(0, 3, 3, 1)),
        // position 0 has no predecessor, so its row attends nowhere and leaves the slot at zero
        mask: pointMask(6, (i) => (i === 0 ? null : i - 1)),
      },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        // query is the current token, key is each position's predecessor tag
        W_Q: withEntries(9, 3, blockEntries(0, 0, 3, MATCH_SCALE)),
        W_K: withEntries(9, 3, blockEntries(3, 0, 3, 1)),
        W_V: withEntries(9, 3, blockEntries(0, 0, 3, 1)),
        W_O: withEntries(3, 9, blockEntries(0, 6, 3, 1)),
        mask: causalMask(6),
      },
      { ...createLinear({ dOut: 3 }), useB: false, W: withEntries(9, 3, blockEntries(6, 0, 3, LOGIT_SCALE)) },
    ],
  };
}

// uniform causal attention averages the a-indicator, and the linear layer thresholds that mean
function majoritySolution() {
  return {
    dModel: 2,
    modules: [
      // dim 0 is 1 on a, dim 1 receives the prefix mean of dim 0
      { ...createEmbed({ useE: true, useP: false }), W_E: withEntries(4, 2, [[0, 0, 1]]) },
      {
        ...createAttn({ dHead: 1, useMask: true }),
        // W_Q and W_K stay zero, so the causal mask alone makes every score equal
        W_V: withEntries(2, 1, [[0, 0, 1]]),
        W_O: withEntries(1, 2, [[0, 1, 1]]),
        mask: causalMask(6),
      },
      {
        ...createLinear({ dOut: 4 }),
        W: withEntries(2, 4, [[1, 2, 30], [1, 3, -30]]),
        // the y - n gap is 60m - 35, positive exactly when the mean m clears 7/12; no reachable
        // mean lands on that threshold, and both biases keep y and n above the unused a/b logits
        b: [0, 0, 7.5, 42.5],
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
    id: "repair",
    name: "Repair",
    difficulty: "tutorial",
    blurb: "Copy each token through, except that every ? is a corrupted a and must come out as an a.",
    formula: "y_i = \\begin{cases} a, & x_i = \\texttt{?} \\\\ x_i, & \\text{otherwise} \\end{cases}",
    vocab: ["?", "a", "b"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["?", "a", "b", "?"], targets: ["a", "a", "b", "a"] },
      { tokens: ["b", "?", "b", "a"], targets: ["b", "a", "b", "a"] },
    ],
    solutionFactory: repairSolution,
  },
  {
    id: "alternate",
    name: "Alternate",
    difficulty: "easy",
    blurb: "Output a at even positions and b at odd positions, whatever the input tokens are.",
    formula: "y_i = \\begin{cases} a, & i \\text{ even} \\\\ b, & i \\text{ odd} \\end{cases}",
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
    id: "start_marker",
    name: "Start Marker",
    difficulty: "easy",
    blurb: "Echo every token, but flag position 0 with ▶ no matter which token sits there.",
    formula: "y_0 = \\blacktriangleright, \\quad y_i = x_i \\;(i > 0)",
    vocab: ["▶", "a", "b"],
    inputVocab: ["a", "b"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "a", "b"], targets: ["▶", "b", "a", "b"] },
      { tokens: ["b", "b", "a", "a"], targets: ["▶", "b", "a", "a"] },
      { tokens: ["a", "a", "b", "b"], targets: ["▶", "a", "b", "b"] },
    ],
    solutionFactory: startMarkerSolution,
  },
  {
    id: "classify",
    name: "Vowel or Consonant",
    difficulty: "easy",
    blurb: "Label every token V if it is a vowel and C if it is a consonant.",
    formula: "y_i = \\begin{cases} V, & x_i \\in \\{a, e, i\\} \\\\ C, & \\text{otherwise} \\end{cases}",
    vocab: ["a", "e", "i", "b", "c", "V", "C"],
    inputVocab: ["a", "e", "i", "b", "c"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "e", "c"], targets: ["V", "C", "V", "C"] },
      { tokens: ["i", "i", "c", "a"], targets: ["V", "V", "C", "V"] },
      { tokens: ["b", "c", "b", "e"], targets: ["C", "C", "C", "V"] },
    ],
    solutionFactory: classifySolution,
  },
  {
    id: "prev_token",
    name: "Previous Token",
    difficulty: "medium",
    blurb: "Predict the token one position back. Position 0 has no predecessor, so it should predict the ⋄ marker it already sits on.",
    formula: "y_i = x_{i-1}, \\quad y_0 = \\diamond",
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
  {
    id: "broadcast",
    name: "Broadcast",
    difficulty: "medium",
    blurb: "Every position outputs the token that sits at position 0.",
    formula: "y_i = x_0",
    vocab: ["a", "b", "c"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "c", "b"], targets: ["a", "a", "a", "a"] },
      { tokens: ["c", "a", "a", "b"], targets: ["c", "c", "c", "c"] },
      { tokens: ["b", "c", "c", "a"], targets: ["b", "b", "b", "b"] },
    ],
    solutionFactory: broadcastSolution,
  },
  {
    id: "reverse",
    name: "Reverse",
    difficulty: "medium",
    blurb: "Turn the sequence around: every position outputs the token mirrored across the middle. Sequences are always 4 long here.",
    formula: "y_i = x_{3-i}",
    vocab: ["a", "b", "c"],
    maxLen: 4,
    // the anti-diagonal mask is tied to one length, so the scratch sequence cannot be resized
    fixedLen: true,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "c", "a"], targets: ["a", "c", "b", "a"] },
      { tokens: ["c", "a", "b", "b"], targets: ["b", "b", "a", "c"] },
      { tokens: ["b", "b", "c", "a"], targets: ["a", "c", "b", "b"] },
    ],
    solutionFactory: reverseSolution,
  },
  {
    id: "agreement",
    name: "Agreement",
    difficulty: "medium",
    blurb: "Output a when the token and the position agree — an a at an even position, or a b at an odd one — and b otherwise.",
    formula: "y_i = \\begin{cases} a, & [x_i = a] = [i \\text{ even}] \\\\ b, & \\text{otherwise} \\end{cases}",
    vocab: ["a", "b"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "a", "b", "b"], targets: ["a", "b", "b", "a"] },
      { tokens: ["b", "a", "a", "b"], targets: ["b", "b", "a", "a"] },
      { tokens: ["a", "b", "a", "b"], targets: ["a", "a", "a", "a"] },
    ],
    solutionFactory: agreementSolution,
  },
  {
    id: "match_first",
    name: "Same as First",
    difficulty: "hard",
    blurb: "Answer y where the token repeats the very first token of the sequence, and n everywhere else.",
    formula: "y_i = \\begin{cases} y, & x_i = x_0 \\\\ n, & \\text{otherwise} \\end{cases}",
    vocab: ["a", "b", "c", "y", "n"],
    inputVocab: ["a", "b", "c"],
    maxLen: 4,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "a", "c"], targets: ["y", "n", "y", "n"] },
      { tokens: ["b", "b", "c", "b"], targets: ["y", "y", "n", "y"] },
      { tokens: ["c", "a", "a", "c"], targets: ["y", "n", "n", "y"] },
    ],
    solutionFactory: matchFirstSolution,
  },
  {
    id: "majority",
    name: "Majority So Far",
    difficulty: "hard",
    blurb: "At each position, answer y if strictly more than half of the tokens up to and including it are a, and n otherwise.",
    formula: "y_i = \\begin{cases} y, & \\#\\{j \\le i : x_j = a\\} > \\frac{i+1}{2} \\\\ n, & \\text{otherwise} \\end{cases}",
    vocab: ["a", "b", "y", "n"],
    inputVocab: ["a", "b"],
    maxLen: 6,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "a", "b", "b", "b", "b"], targets: ["y", "y", "y", "n", "n", "n"] },
      { tokens: ["b", "a", "a", "a", "b", "a"], targets: ["n", "n", "y", "y", "y", "y"] },
      { tokens: ["a", "b", "a", "b", "b", "a"], targets: ["y", "n", "y", "n", "n", "n"] },
    ],
    solutionFactory: majoritySolution,
  },
  {
    id: "induction",
    name: "Induction",
    difficulty: "hard",
    blurb:
      "Find where this token appeared earlier and output whatever followed it there. Positions where the token is new, or where two earlier copies are followed by different tokens, are marked · and are not graded.",
    formula: "y_i = x_{j+1} \\;\\; \\text{for every } j < i \\text{ with } x_j = x_i",
    vocab: ["a", "b", "c"],
    maxLen: 6,
    epsilon: 0.05,
    tests: [
      { tokens: ["a", "b", "c", "a", "b", "c"], targets: [null, null, null, "b", "c", "a"] },
      { tokens: ["c", "a", "b", "c", "a", "b"], targets: [null, null, null, "a", "b", "c"] },
      { tokens: ["a", "b", "b", "a", "c", "c"], targets: [null, null, "b", "b", null, "c"] },
      // two earlier copies of a, both followed by b, so the retrieval is still unambiguous
      { tokens: ["a", "b", "a", "b", "c", "a"], targets: [null, null, "b", "a", null, "b"] },
      // the two earlier copies of a disagree here, so the last position is left ungraded
      { tokens: ["a", "b", "a", "c", "b", "a"], targets: [null, null, "b", null, "a", null] },
    ],
    solutionFactory: inductionSolution,
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
