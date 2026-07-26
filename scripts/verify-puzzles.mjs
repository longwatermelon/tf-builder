import { build } from "esbuild";

const verificationSource = String.raw`
  import assert from "node:assert/strict";
  import { buildSolution, PUZZLES } from "./src/features/puzzles/puzzles.js";
  import { decodeAttempt, encodeAttempt } from "./src/lib/attemptFile.js";
  import { evaluatePuzzle } from "./src/lib/model.js";

  const expectedValidationTotals = {
    binary_addition: 16,
    pointer_chasing: 81,
    balanced_parentheses: 64,
  };

  const binaryInputs = Array.from({ length: 16 }, (_, value) => {
    const bits = value.toString(2).padStart(4, "0");
    return bits.slice(0, 2) + "+" + bits.slice(2) + "=___";
  });
  const pointerSymbols = ["a", "b", "c"];
  const pointerInputs = Array.from({ length: 81 }, (_, value) => {
    const assignment = Array.from({ length: 4 }, (_, i) => pointerSymbols[Math.floor(value / (3 ** (3 - i))) % 3]);
    return "a" + assignment[0] + "b" + assignment[1] + "c" + assignment[2] + "?" + assignment[3] + "_";
  });
  const parenthesesInputs = Array.from({ length: 64 }, (_, value) =>
    value.toString(2).padStart(6, "0").replaceAll("0", "(").replaceAll("1", ")"),
  );
  const expectedInputs = {
    binary_addition: binaryInputs,
    pointer_chasing: pointerInputs,
    balanced_parentheses: parenthesesInputs,
  };
  const expectedGradedPositions = {
    binary_addition: [6, 7, 8],
    pointer_chasing: [8],
    balanced_parentheses: [5],
  };

  for (const [puzzleId, expectedTotal] of Object.entries(expectedValidationTotals)) {
    const puzzle = PUZZLES.find((candidate) => candidate.id === puzzleId);
    assert.ok(puzzle, "missing puzzle: " + puzzleId);
    assert.equal(puzzle.difficulty, "insane");
    assert.equal(puzzle.validationTests.length, expectedTotal);
    const actualInputs = puzzle.validationTests.map((test) => test.tokens.join(""));
    assert.equal(new Set(actualInputs).size, expectedTotal, puzzleId + " validation inputs are not unique");
    assert.deepEqual([...actualInputs].sort(), [...expectedInputs[puzzleId]].sort(), puzzleId + " validation domain is wrong");

    for (const test of [...puzzle.tests, ...puzzle.validationTests]) {
      assert.equal(test.tokens.length, puzzle.maxLen, puzzleId + " has a malformed input");
      assert.equal(test.targets.length, puzzle.maxLen, puzzleId + " has malformed targets");
      assert.ok(test.tokens.every((token) => puzzle.vocab.includes(token)), puzzleId + " has an unknown input token");
      assert.ok(test.targets.every((token) => token === null || puzzle.vocab.includes(token)), puzzleId + " has an unknown target token");
      const gradedPositions = test.targets.flatMap((target, i) => (target === null ? [] : [i]));
      assert.deepEqual(gradedPositions, expectedGradedPositions[puzzleId], puzzleId + " grades the wrong positions");

      const expectedTargets = new Array(puzzle.maxLen).fill(null);
      if (puzzleId === "binary_addition") {
        const left = Number.parseInt(test.tokens.slice(0, 2).join(""), 2);
        const right = Number.parseInt(test.tokens.slice(3, 5).join(""), 2);
        const sum = (left + right).toString(2).padStart(3, "0");
        expectedTargets.splice(6, 3, ...sum);
      } else if (puzzleId === "pointer_chasing") {
        const mapping = { a: test.tokens[1], b: test.tokens[3], c: test.tokens[5] };
        expectedTargets[8] = mapping[mapping[test.tokens[7]]];
      } else {
        let depth = 0;
        let minDepth = 0;
        for (const token of test.tokens) {
          depth += token === "(" ? 1 : -1;
          minDepth = Math.min(minDepth, depth);
        }
        expectedTargets[5] = depth === 0 && minDepth >= 0 ? "y" : "n";
      }
      assert.deepEqual(test.targets, expectedTargets, puzzleId + " target rule is wrong for " + test.tokens.join(""));
    }

    const canonicalModel = buildSolution(puzzle);
    const encodedModel = encodeAttempt(puzzle.id, canonicalModel, { compact: true });
    const roundTrippedModel = decodeAttempt(encodedModel, puzzle);
    assert.deepEqual(roundTrippedModel, canonicalModel, puzzleId + " canonical model does not round-trip");
    const evaluation = evaluatePuzzle(roundTrippedModel, puzzle);
    assert.equal(evaluation.validationTotal, expectedTotal);
    assert.equal(evaluation.validationPassed, expectedTotal);
    assert.equal(evaluation.solved, true, puzzleId + " canonical solution does not solve its rule");
    assert.equal(evaluation.params, puzzle.canonicalParams);
    console.log(puzzleId + ": " + evaluation.validationPassed + "/" + evaluation.validationTotal + ", " + evaluation.params + "p");
  }
`;

// bundle browser-style extensionless imports before executing the verification entry
const bundle = await build({
  stdin: {
    contents: verificationSource,
    resolveDir: process.cwd(),
    sourcefile: "verify-puzzles-entry.js",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  write: false,
});

const encodedBundle = Buffer.from(bundle.outputFiles[0].contents).toString("base64");
await import(`data:text/javascript;base64,${encodedBundle}`);
