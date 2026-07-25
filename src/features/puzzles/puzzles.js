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

const TRIANGLE_CODES = [
  [1, 0],
  [-0.5, Math.sqrt(3) / 2],
  [-0.5, -Math.sqrt(3) / 2],
];

const TRIANGLE_UNEMBED = [
  [1, -0.5, -0.5],
  [0, Math.sqrt(3) / 2, -Math.sqrt(3) / 2],
];

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
    dModel: 1,
    modules: [
      {
        ...createEmbed({ useE: false, useP: true }),
        W_P: [[1], [-1], [1], [-1]],
      },
      { ...createLinear({ dOut: 2 }), useB: false, W: [[LOGIT_SCALE, -LOGIT_SCALE]] },
    ],
  };
}

// a large scalar copy separates the selected token into one of three affine regions
function maskedCopySolution(mask) {
  return {
    dModel: 1,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: [[-1], [0], [1]] },
      {
        ...createAttn({ dHead: 1, useMask: true }),
        W_V: [[1]],
        W_O: [[10]],
        mask,
      },
      { ...createLinear({ dOut: 3 }), W: [[-1, 0, 1]], b: [-5, 0, -5] },
    ],
  };
}

// previous-token head: the mask alone picks position i-1
function prevTokenSolution() {
  return maskedCopySolution(pointMask(4, (i) => Math.max(0, i - 1)));
}

// one scalar sign separates the repaired a class from b
function repairSolution() {
  return {
    dModel: 1,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: [[1], [1], [-1]] },
      { ...createLinear({ dOut: 3 }), useB: false, W: [[0, LOGIT_SCALE, -LOGIT_SCALE]] },
    ],
  };
}

// four scalar locations let three affine logits mark the start and echo later tokens
function startMarkerSolution() {
  return {
    dModel: 1,
    modules: [
      {
        ...createEmbed({ useE: true, useP: true }),
        W_E: [[0], [-1], [1]],
        W_P: [[0], [4], [4], [4]],
      },
      { ...createLinear({ dOut: 3 }), W: [[0, 5, 10]], b: [0, -10, -30] },
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
  return maskedCopySolution(pointMask(4, () => 0));
}

// same compact copy as broadcast, but the mask is the anti-diagonal
function reverseSolution() {
  return maskedCopySolution(pointMask(4, (i) => 3 - i));
}

// one relu folds the two agreement cases onto the same positive scalar
function agreementSolution() {
  return {
    dModel: 1,
    modules: [
      {
        ...createEmbed({ useE: true, useP: true }),
        W_E: [[0.25], [-0.75]],
        W_P: [[0.25], [-0.75], [0.25], [-0.75]],
      },
      {
        ...createMlp({ dHidden: 1 }),
        useB2: false,
        W1: [[-1]],
        b1: [-1],
        W2: [[4]],
      },
      { ...createLinear({ dOut: 2 }), useB: false, W: [[LOGIT_SCALE, -LOGIT_SCALE]] },
    ],
  };
}

// subtract the first token, then fold both nonzero difference signs below zero
function matchFirstSolution() {
  return {
    dModel: 1,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: [[-2], [0], [2], [0], [0]] },
      {
        ...createAttn({ dHead: 1, useMask: true }),
        W_V: [[1]],
        W_O: [[-1]],
        mask: pointMask(4, () => 0),
      },
      {
        ...createMlp({ dHidden: 1 }),
        useB1: false,
        W1: [[1]],
        W2: [[-2]],
        b2: [1],
      },
      {
        ...createLinear({ dOut: 5 }),
        useB: false,
        W: [[0, 0, 0, LOGIT_SCALE, -LOGIT_SCALE]],
      },
    ],
  };
}

// two compact heads: one stores a predecessor code and the other retrieves its follower
function inductionSolution() {
  const MATCH_SCALE = 10;
  return {
    dModel: 4,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: TRIANGLE_CODES.map((row) => [...row, 0, 0]) },
      {
        ...createAttn({ dHead: 2, useMask: true }),
        W_V: withEntries(4, 2, [[0, 0, 1], [1, 1, 1]]),
        W_O: withEntries(2, 4, [[0, 2, 1], [1, 3, 1]]),
        // position 0 has no predecessor, so its row attends nowhere and leaves the slot at zero
        mask: pointMask(6, (i) => (i === 0 ? null : i - 1)),
      },
      {
        ...createAttn({ dHead: 2, useMask: true }),
        // query is the current token, key is each position's predecessor tag
        W_Q: withEntries(4, 2, [[0, 0, MATCH_SCALE], [1, 1, MATCH_SCALE]]),
        W_K: withEntries(4, 2, [[2, 0, 1], [3, 1, 1]]),
        W_V: withEntries(4, 2, [[0, 0, 1], [1, 1, 1]]),
        W_O: withEntries(2, 4, [[0, 0, LOGIT_SCALE], [1, 1, LOGIT_SCALE]]),
        mask: causalMask(6),
      },
      {
        ...createLinear({ dOut: 3 }),
        useB: false,
        W: [...TRIANGLE_UNEMBED, [0, 0, 0], [0, 0, 0]],
      },
    ],
  };
}

// a large prefix-mean term makes one scalar separate strict majorities from every other prefix
function majoritySolution() {
  const meanScale = 20;
  const baseline = -11.5 / (meanScale + 1);
  return {
    dModel: 1,
    modules: [
      { ...createEmbed({ useE: true, useP: false }), W_E: [[baseline + 1], [baseline], [0], [0]] },
      {
        ...createAttn({ dHead: 1, useMask: true }),
        // W_Q and W_K stay zero, so the causal mask alone makes every score equal
        W_V: [[1]],
        W_O: [[meanScale]],
        mask: causalMask(6),
      },
      {
        ...createLinear({ dOut: 4 }),
        useB: false,
        W: [[0, 0, LOGIT_SCALE, -LOGIT_SCALE]],
      },
    ],
  };
}

// turn compact input strings into graded cases using the puzzle's actual rule
function makeTests(inputs, targetFactory) {
  return inputs.map((input) => {
    const tokens = Array.from(input);
    return { tokens, targets: targetFactory(tokens) };
  });
}

// enumerate every valid non-empty input up to maxLen, optionally after a fixed prefix
function enumerateInputs(alphabet, maxLen, { fixedLen = false, prefix = [] } = {}) {
  const inputs = [];
  let frontier = [[...prefix]];
  for (let length = prefix.length; length <= maxLen; length++) {
    if (length > 0 && (!fixedLen || length === maxLen)) inputs.push(...frontier.map((tokens) => tokens.join("")));
    if (length === maxLen) break;
    frontier = frontier.flatMap((tokens) => alphabet.map((token) => [...tokens, token]));
  }
  return inputs;
}

// echo keeps every token unchanged
function echoTargets(tokens) {
  return [...tokens];
}

// successor advances each token around the three-token cycle
function successorTargets(tokens) {
  const next = { a: "b", b: "c", c: "a" };
  return tokens.map((token) => next[token]);
}

// repair treats each question mark as a corrupted a
function repairTargets(tokens) {
  return tokens.map((token) => (token === "?" ? "a" : token));
}

// alternate ignores token identity and follows position parity
function alternateTargets(tokens) {
  return tokens.map((_, i) => (i % 2 === 0 ? "a" : "b"));
}

// start marker overwrites position zero and echoes every later token
function startMarkerTargets(tokens) {
  return tokens.map((token, i) => (i === 0 ? "▶" : token));
}

// classify maps vowels and consonants to their output-only labels
function classifyTargets(tokens) {
  const vowels = new Set(["a", "e", "i"]);
  return tokens.map((token) => (vowels.has(token) ? "V" : "C"));
}

// previous token emits the fixed start marker followed by each predecessor
function previousTargets(tokens) {
  return tokens.map((_, i) => (i === 0 ? "⋄" : tokens[i - 1]));
}

// broadcast copies the first token to every position
function broadcastTargets(tokens) {
  return tokens.map(() => tokens[0]);
}

// reverse mirrors the complete fixed-length sequence
function reverseTargets(tokens) {
  return [...tokens].reverse();
}

// agreement checks whether token identity agrees with position parity
function agreementTargets(tokens) {
  return tokens.map((token, i) => ((token === "a") === (i % 2 === 0) ? "a" : "b"));
}

// match first compares every token with the first token in its sequence
function matchFirstTargets(tokens) {
  return tokens.map((token) => (token === tokens[0] ? "y" : "n"));
}

// majority labels each prefix by whether a is strictly ahead
function majorityTargets(tokens) {
  let countA = 0;
  return tokens.map((token, i) => {
    if (token === "a") countA += 1;
    return countA > (i + 1) / 2 ? "y" : "n";
  });
}

// induction returns the shared follower of prior matches and skips ambiguous positions
function inductionTargets(tokens) {
  return tokens.map((token, i) => {
    const followers = [];
    for (let j = 0; j < i; j++) {
      if (tokens[j] === token) followers.push(tokens[j + 1]);
    }
    if (followers.length === 0 || followers.some((follower) => follower !== followers[0])) return null;
    return followers[0];
  });
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
    tests: makeTests(["abca", "cabb"], echoTargets),
    targetFactory: echoTargets,
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
    tests: makeTests(["abca", "ccab"], successorTargets),
    targetFactory: successorTargets,
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
    tests: makeTests(["?ab?", "b?ba"], repairTargets),
    targetFactory: repairTargets,
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
    tests: makeTests(["aabb", "baab"], alternateTargets),
    targetFactory: alternateTargets,
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
    tests: makeTests(
      ["abab", "bbaa", "aabb"],
      startMarkerTargets,
    ),
    targetFactory: startMarkerTargets,
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
    tests: makeTests(["abec", "iica", "bcbe"], classifyTargets),
    targetFactory: classifyTargets,
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
    tests: makeTests(["⋄aba", "⋄bba", "⋄aab"], previousTargets),
    targetFactory: previousTargets,
    validationVocab: ["a", "b"],
    validationPrefix: ["⋄"],
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
    tests: makeTests(["abcb", "caab", "bcca"], broadcastTargets),
    targetFactory: broadcastTargets,
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
    tests: makeTests(["abca", "cabb", "bbca"], reverseTargets),
    targetFactory: reverseTargets,
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
    tests: makeTests(["aabb", "baab", "abab"], agreementTargets),
    targetFactory: agreementTargets,
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
    tests: makeTests(["abac", "bbcb", "caac"], matchFirstTargets),
    targetFactory: matchFirstTargets,
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
    tests: makeTests(["aabbbb", "baaaba", "ababba"], majorityTargets),
    targetFactory: majorityTargets,
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
    tests: makeTests(["abcabc", "cabcab", "abbacc", "ababca", "abacba"], inductionTargets),
    targetFactory: inductionTargets,
    solutionFactory: inductionSolution,
  },
];

// build a fully shaped canonical model for a puzzle
export function buildSolution(puzzle) {
  return reconcileShapes(puzzle.solutionFactory(), puzzle);
}

// derive exhaustive rule-validation cases and the elegance bar from each puzzle definition
export const PUZZLES = PUZZLE_DEFS.map((puzzle) => {
  const validationInputs = enumerateInputs(
    puzzle.validationVocab ?? puzzle.inputVocab ?? puzzle.vocab,
    puzzle.maxLen,
    { fixedLen: puzzle.fixedLen, prefix: puzzle.validationPrefix },
  );
  return {
    ...puzzle,
    validationTests: makeTests(validationInputs, puzzle.targetFactory),
    canonicalParams: countParams(reconcileShapes(puzzle.solutionFactory(), puzzle), puzzle),
  };
});

export function getPuzzle(id) {
  return PUZZLES.find((puzzle) => puzzle.id === id) ?? PUZZLES[0];
}
