import {
  countParams,
  createAttn,
  createEmbed,
  createLinear,
  createMlp,
  reconcileShapes,
  withEntries,
} from "../../lib/model";
import { decodeAttempt } from "../../lib/attemptFile";
import agreementAuthorAttempt from "./author-solutions/agreement.json";
import prevTokenAuthorAttempt from "./author-solutions/prev_token.json";
import repairAuthorAttempt from "./author-solutions/repair.json";
import startMarkerAuthorAttempt from "./author-solutions/start_marker.json";

const AUTHOR_ATTEMPTS = {
  agreement: agreementAuthorAttempt,
  prev_token: prevTokenAuthorAttempt,
  repair: repairAuthorAttempt,
  start_marker: startMarkerAuthorAttempt,
};

const ELEGANT_AUTHOR_IDS = new Set(["echo", "successor", "alternate", "classify"]);

// logit gap used by elegant solutions; large enough to clear epsilon comfortably
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

// strict causal mask: query i attends only to positions before i
function strictCausalMask(len) {
  return Array.from({ length: len }, (_, i) => Array.from({ length: len }, (_, j) => (j < i ? 0 : NEG)));
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

// uniform attention counts a tokens while position embeddings set each sorted slot's threshold
function binarySortSolution() {
  return {
    dModel: 2,
    modules: [
      {
        ...createEmbed({ useE: true, useP: true }),
        W_E: [[1, 0], [0, 0]],
        W_P: [[0, -0.5], [0, -1.5], [0, -2.5], [0, -3.5]],
      },
      {
        ...createAttn({ dHead: 1, useMask: false }),
        // zero queries and keys give every token equal weight
        W_V: [[1], [0]],
        W_O: [[0, 4]],
      },
      {
        ...createLinear({ dOut: 2 }),
        useB: false,
        W: [[0, 0], [LOGIT_SCALE, -LOGIT_SCALE]],
      },
    ],
  };
}

// content attention retrieves prior matches and an mlp detects their agreement with the query
function seenBeforeSolution() {
  const MATCH_SCALE = 10;
  return {
    dModel: 3,
    modules: [
      {
        ...createEmbed({ useE: true, useP: false }),
        W_E: [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
      },
      {
        ...createAttn({ dHead: 1, useMask: true }),
        W_Q: [[MATCH_SCALE], [0], [0]],
        W_K: [[1], [0], [0]],
        W_V: [[1], [0], [0]],
        W_O: [[0, 1, 0]],
        mask: strictCausalMask(4),
      },
      {
        ...createMlp({ dHidden: 2 }),
        W1: [[1, -1], [1, -1], [0, 0]],
        b1: [-1.5, -1.5],
        W2: [[0, 0, 1], [0, 0, 1]],
        useB2: false,
      },
      {
        ...createLinear({ dOut: 4 }),
        W: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, LOGIT_SCALE, -LOGIT_SCALE]],
        b: [0, 0, -1, 1],
      },
    ],
  };
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

// copy one scalar feature from a fixed source position into a residual dimension
function fixedPositionCopyHead(maxLen, sourcePosition, inputDimension, outputDimension, dModel) {
  return {
    ...createAttn({ dHead: 1, useMask: true }),
    W_V: withEntries(dModel, 1, [[inputDimension, 0, 1]]),
    W_O: withEntries(1, dModel, [[0, outputDimension, 1]]),
    mask: pointMask(maxLen, () => sourcePosition),
  };
}

// route four operand bits, then use one exact finite-domain detector per output bit and input case
function binaryAdditionSolution() {
  const dModel = 8;
  const detectorMlps = Array.from({ length: 3 }, (_, outputIndex) => {
    const W1 = withEntries(dModel, 16, []);
    const b1 = new Array(16).fill(0);
    const W2 = withEntries(16, dModel, []);
    for (let value = 0; value < 16; value++) {
      const inputBits = [3, 2, 1, 0].map((shift) => (value >> shift) & 1);
      const sum = (inputBits[0] * 2 + inputBits[1]) + (inputBits[2] * 2 + inputBits[3]);
      const sumBit = (sum >> (2 - outputIndex)) & 1;
      let zeroCount = 0;
      for (let bitIndex = 0; bitIndex < 4; bitIndex++) {
        const bit = inputBits[bitIndex];
        W1[bitIndex + 1][value] = bit === 1 ? 1 : -1;
        if (bit === 0) zeroCount += 1;
      }
      W1[outputIndex + 5][value] = 1;
      b1[value] = zeroCount - 4.5;
      W2[value][0] = sumBit === 1 ? 20 : -20;
    }
    return {
      ...createMlp({ dHidden: 16 }),
      W1,
      b1,
      W2,
      useB2: false,
    };
  });

  return {
    dModel,
    modules: [
      {
        ...createEmbed({ useE: true, useP: true }),
        W_E: withEntries(5, dModel, [[1, 0, 1]]),
        W_P: withEntries(9, dModel, [[6, 5, 1], [7, 6, 1], [8, 7, 1]]),
      },
      fixedPositionCopyHead(9, 0, 0, 1, dModel),
      fixedPositionCopyHead(9, 1, 0, 2, dModel),
      fixedPositionCopyHead(9, 3, 0, 3, dModel),
      fixedPositionCopyHead(9, 4, 0, 4, dModel),
      ...detectorMlps,
      {
        ...createLinear({ dOut: 5 }),
        useB: false,
        W: withEntries(dModel, 5, [[0, 0, -1], [0, 1, 1]]),
      },
    ],
  };
}

// prepare a three-entry lookup table, then query it twice at the answer position
function pointerChasingSolution() {
  const dModel = 12;
  const mapKeyPositions = new Set([0, 2, 4]);
  const tableMask = pointMask(9, (i) => (mapKeyPositions.has(i) ? i + 1 : null));
  const answerMask = pointMask(9, (i) => (i === 8 ? 7 : null));
  const lookupMask = Array.from({ length: 9 }, (_, i) =>
    Array.from({ length: 9 }, (_, j) => (i === 8 && mapKeyPositions.has(j) ? 0 : NEG)),
  );
  const tokenIdentity = [[0, 0, 1], [1, 1, 1], [2, 2, 1]];

  return {
    dModel,
    modules: [
      {
        ...createEmbed({ useE: true, useP: false }),
        W_E: withEntries(5, dModel, tokenIdentity),
      },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(dModel, 3, tokenIdentity),
        W_O: withEntries(3, dModel, [[0, 3, 1], [1, 4, 1], [2, 5, 1]]),
        mask: tableMask,
      },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_V: withEntries(dModel, 3, tokenIdentity),
        W_O: withEntries(3, dModel, tokenIdentity),
        mask: answerMask,
      },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_Q: withEntries(dModel, 3, [[0, 0, 20], [1, 1, 20], [2, 2, 20]]),
        W_K: withEntries(dModel, 3, tokenIdentity),
        W_V: withEntries(dModel, 3, [[3, 0, 1], [4, 1, 1], [5, 2, 1]]),
        W_O: withEntries(3, dModel, [[0, 6, 1], [1, 7, 1], [2, 8, 1]]),
        mask: lookupMask,
      },
      {
        ...createAttn({ dHead: 3, useMask: true }),
        W_Q: withEntries(dModel, 3, [[6, 0, 20], [7, 1, 20], [8, 2, 20]]),
        W_K: withEntries(dModel, 3, tokenIdentity),
        W_V: withEntries(dModel, 3, [[3, 0, 1], [4, 1, 1], [5, 2, 1]]),
        W_O: withEntries(3, dModel, [[0, 9, 1], [1, 10, 1], [2, 11, 1]]),
        mask: lookupMask,
      },
      {
        ...createLinear({ dOut: 5 }),
        useB: false,
        W: withEntries(dModel, 5, [[9, 0, LOGIT_SCALE], [10, 1, LOGIT_SCALE], [11, 2, LOGIT_SCALE]]),
      },
    ],
  };
}

// collect all six bracket signs and score exact balance minus every negative-prefix violation
function balancedParenthesesSolution() {
  const dModel = 8;
  const W1 = withEntries(dModel, 9, []);
  const b1 = [1, 0, -1, 0, 0, 0, 0, 0, 0];
  const W2 = withEntries(9, dModel, [[0, 7, 2], [1, 7, -4], [2, 7, 2]]);

  for (let signDimension = 1; signDimension <= 6; signDimension++) {
    W1[signDimension][0] = 1;
    W1[signDimension][1] = 1;
    W1[signDimension][2] = 1;
    for (let prefixEnd = signDimension; prefixEnd <= 6; prefixEnd++) {
      W1[signDimension][prefixEnd + 2] = -1;
    }
  }
  for (let prefixIndex = 0; prefixIndex < 6; prefixIndex++) W2[prefixIndex + 3][7] = -2;

  return {
    dModel,
    modules: [
      {
        ...createEmbed({ useE: true, useP: false }),
        W_E: withEntries(4, dModel, [[0, 0, 1], [1, 0, -1]]),
      },
      ...Array.from({ length: 6 }, (_, i) => fixedPositionCopyHead(6, i, 0, i + 1, dModel)),
      {
        ...createMlp({ dHidden: 9 }),
        W1,
        b1,
        W2,
        b2: [0, 0, 0, 0, 0, 0, 0, -1],
      },
      {
        ...createLinear({ dOut: 4 }),
        useB: false,
        W: withEntries(dModel, 4, [[7, 2, LOGIT_SCALE], [7, 3, -LOGIT_SCALE]]),
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

// binary sort places every a before every b
function binarySortTargets(tokens) {
  const countA = tokens.filter((token) => token === "a").length;
  return tokens.map((_, i) => (i < countA ? "a" : "b"));
}

// seen before marks tokens that already occurred earlier in the sequence
function seenBeforeTargets(tokens) {
  const seen = new Set();
  return tokens.map((token) => {
    const target = seen.has(token) ? "y" : "n";
    seen.add(token);
    return target;
  });
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

// write the three-bit sum into the result slots and leave the expression slots ungraded
function binaryAdditionTargets(tokens) {
  const left = Number.parseInt(`${tokens[0]}${tokens[1]}`, 2);
  const right = Number.parseInt(`${tokens[3]}${tokens[4]}`, 2);
  const sumBits = (left + right).toString(2).padStart(3, "0");
  return tokens.map((_, i) => (i >= 6 ? sumBits[i - 6] : null));
}

// apply the complete three-symbol lookup table twice to the query token
function pointerChasingTargets(tokens) {
  const mapping = { a: tokens[1], b: tokens[3], c: tokens[5] };
  return tokens.map((_, i) => (i === 8 ? mapping[mapping[tokens[7]]] : null));
}

// accept only strings whose running depth never drops below zero and finishes at zero
function balancedParenthesesTargets(tokens) {
  let depth = 0;
  let valid = true;
  for (const token of tokens) {
    depth += token === "(" ? 1 : -1;
    if (depth < 0) valid = false;
  }
  return tokens.map((_, i) => (i === tokens.length - 1 ? (valid && depth === 0 ? "y" : "n") : null));
}

// enumerate a small assignment space and place each assignment into a fixed token format
function structuredInputs(alphabet, assignmentLength, format) {
  return enumerateInputs(alphabet, assignmentLength, { fixedLen: true }).map((input) => format(Array.from(input)));
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
    id: "binary_sort",
    name: "Binary Sort",
    difficulty: "medium",
    blurb: "Sort the sequence so every a comes before every b. Sequences are always 4 long here.",
    formula: "y_i = \\begin{cases} a, & i < \\#\\{j : x_j = a\\} \\\\ b, & \\text{otherwise} \\end{cases}",
    vocab: ["a", "b"],
    maxLen: 4,
    fixedLen: true,
    epsilon: 0.05,
    tests: makeTests(["baba", "bbba", "abaa"], binarySortTargets),
    targetFactory: binarySortTargets,
    solutionFactory: binarySortSolution,
  },
  {
    id: "match_first",
    name: "Same as First",
    difficulty: "medium",
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
    id: "seen_before",
    name: "Seen Before",
    difficulty: "hard",
    blurb: "Answer y when the current token appeared earlier in the sequence, and n when this is its first appearance.",
    formula: "y_i = \\begin{cases} y, & \\exists j < i : x_j = x_i \\\\ n, & \\text{otherwise} \\end{cases}",
    vocab: ["a", "b", "y", "n"],
    inputVocab: ["a", "b"],
    maxLen: 4,
    epsilon: 0.05,
    tests: makeTests(["aaaa", "aaba", "babb"], seenBeforeTargets),
    targetFactory: seenBeforeTargets,
    solutionFactory: seenBeforeSolution,
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
  {
    id: "binary_addition",
    name: "Binary Addition",
    difficulty: "insane",
    blurb: "Add two unsigned two-bit numbers and write their three-bit sum into the slots after the equals sign.",
    formula: "x_1x_0 + z_1z_0 = y_2y_1y_0",
    vocab: ["0", "1", "+", "=", "_"],
    inputVocab: ["0", "1"],
    inputFormat: "bit bit + bit bit = _ _ _",
    scratchChoices: [["0", "1"], ["0", "1"], ["+"], ["0", "1"], ["0", "1"], ["="], ["_"], ["_"], ["_"]],
    maxLen: 9,
    fixedLen: true,
    epsilon: 0.05,
    tests: makeTests(["01+11=___", "11+11=___", "10+01=___"], binaryAdditionTargets),
    validationInputs: structuredInputs(["0", "1"], 4, ([x1, x0, z1, z0]) => [x1, x0, "+", z1, z0, "=", "_", "_", "_"]),
    targetFactory: binaryAdditionTargets,
    solutionFactory: binaryAdditionSolution,
  },
  {
    id: "pointer_chasing",
    name: "Pointer Chasing",
    difficulty: "insane",
    blurb: "Read the table a→A, b→B, c→C and apply it twice to the query token after the question mark.",
    formula: String.raw`y_8 = f(f(x_7)), \quad f(a)=x_1,\ f(b)=x_3,\ f(c)=x_5`,
    vocab: ["a", "b", "c", "?", "_"],
    inputVocab: ["a", "b", "c"],
    inputFormat: "a value b value c value ? query _",
    scratchChoices: [["a"], ["a", "b", "c"], ["b"], ["a", "b", "c"], ["c"], ["a", "b", "c"], ["?"], ["a", "b", "c"], ["_"]],
    maxLen: 9,
    fixedLen: true,
    epsilon: 0.05,
    tests: makeTests(["abbcca?a_", "acbacb?b_", "aabbcc?c_"], pointerChasingTargets),
    validationInputs: structuredInputs(["a", "b", "c"], 4, ([aValue, bValue, cValue, query]) =>
      ["a", aValue, "b", bValue, "c", cValue, "?", query, "_"],
    ),
    targetFactory: pointerChasingTargets,
    solutionFactory: pointerChasingSolution,
  },
  {
    id: "balanced_parentheses",
    name: "Balanced Parentheses",
    difficulty: "insane",
    blurb: "Answer y at the final position only when every prefix has nonnegative depth and the complete six-token string is balanced.",
    formula: String.raw`y_5 = [\forall k,\ d_k \ge 0]\,[d_5=0], \quad d_k=\sum_{j=0}^{k}\bigl([x_j=(]-[x_j=)]\bigr)`,
    vocab: ["(", ")", "y", "n"],
    inputVocab: ["(", ")"],
    inputFormat: "six opening or closing parentheses",
    scratchChoices: Array.from({ length: 6 }, () => ["(", ")"]),
    maxLen: 6,
    fixedLen: true,
    epsilon: 0.05,
    tests: makeTests(["((()))", "()()()", ")(()()", "(()))("], balancedParenthesesTargets),
    targetFactory: balancedParenthesesTargets,
    solutionFactory: balancedParenthesesSolution,
  },
];

// build a fresh, fully shaped copy of one provided solution
export function buildSolution(puzzle, kind = "elegant") {
  const source = kind === "author" ? puzzle.authorSolution : puzzle.solutionFactory();
  return source ? reconcileShapes(source, puzzle) : null;
}

// derive exhaustive rule-validation cases and the elegance bar from each puzzle definition
export const PUZZLES = PUZZLE_DEFS.map((puzzle, index) => {
  const validationInputs = puzzle.validationInputs ?? enumerateInputs(
    puzzle.validationVocab ?? puzzle.inputVocab ?? puzzle.vocab,
    puzzle.maxLen,
    { fixedLen: puzzle.fixedLen, prefix: puzzle.validationPrefix },
  );
  const authorAttempt = AUTHOR_ATTEMPTS[puzzle.id];
  const authorSolution = authorAttempt
    ? decodeAttempt(JSON.stringify(authorAttempt), puzzle)
    : ELEGANT_AUTHOR_IDS.has(puzzle.id)
      ? reconcileShapes(puzzle.solutionFactory(), puzzle)
      : null;
  return {
    ...puzzle,
    number: index + 1,
    authorSolution,
    validationTests: makeTests(validationInputs, puzzle.targetFactory),
    canonicalParams: countParams(reconcileShapes(puzzle.solutionFactory(), puzzle), puzzle),
  };
});

export function getPuzzle(id) {
  return PUZZLES.find((puzzle) => puzzle.id === id) ?? PUZZLES[0];
}
