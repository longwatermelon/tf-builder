# README

This project is a website for hand-crafting small transformers. The player picks a **puzzle**, assembles a module stack over a residual stream, and edits every weight by hand until the model's per-position output tokens match the puzzle's required outputs.

## Summary

- A **puzzle** is one input/output rule with a vocabulary, a small visible set of representative samples, and rule-derived validation over every valid input up to `maxLen`. Puzzles are graded `tutorial` / `easy` / `medium` / `hard` / `insane`. A `null` target marks a position the rule does not determine, which is displayed as `·` and left ungraded. Optionally a puzzle also declares `inputVocab`, the subset of the vocabulary that can appear as input when the rest of it is output-only labels, `fixedLen`, which pins every sequence to `maxLen`, or an explicit structured input domain for fixed-format tasks.
- The **module stack** is the player's architecture. Every puzzle starts with a single **Embedding** module, sized so the residual stream is already the token axis; the player inserts and deletes **Attention Head**, **MLP**, and **Linear** modules themselves, at any point in the stack. Attention and MLP add into the residual stream; Linear is the only module that changes the stream width, so it doubles as the unembedding.
- Every row/column label is an **annotation** the player owns: clicking one names that dimension in up to 4 characters, so `d0 d1 d2` can read `in0 in1 in2`. A name belongs to the axis, not the matrix, so it follows that dimension through every module and into the computed value grids. The residual stream, the positions (`W_P` rows and both mask axes) and the vocabulary are shared by the whole stack; attention head dims, MLP hidden units, and a `Linear`'s own output dims are separate vector spaces, so each module names those itself. Names live per puzzle and last for the session.
- There is no LayerNorm, and attention scores are raw `QKᵀ` with no `1/√d` scaling — both would only make hand-crafting harder without teaching anything.
- A puzzle is **solved** when, for every valid rule-generated input and every graded position, the required token is the argmax of the output softmax *and* beats the runner-up by the puzzle's `epsilon`. The final softmax is over the vocabulary at each position — it is a per-position output distribution, not a next-token prediction.
- A solve is **elegant** when the total allocated parameter count is at most the elegant solution's. Optional blocks (`W_E`, `W_P`, the attention mask `M`, and every MLP or Linear bias) can be switched off to drop their parameters from the count.
- Each puzzle ships an **elegant solution**, and puzzles can also include an **author's solution**. Revealing either locks progress for the current attempt only — resetting back to a blank model (or reloading) lets a from-scratch rebuild earn the mark again.
- The current puzzle's **attempt** can be exported to or imported from a versioned JSON file, or moved through the clipboard with copy / paste controls (with a manual paste box when the browser blocks clipboard reading). The file records the puzzle ID and complete model architecture and weights; special infinite attention-mask values are preserved as `"Infinity"` and `"-Infinity"`. Imports are shape-checked and must match the selected puzzle before they replace the current model.

## File Structure

- `src/` - Frontend app source for the transformer builder.
  - `src/main.jsx` - React entry point that mounts `App` and loads global + KaTeX styles.
  - `src/App.jsx` - Top-level orchestration: puzzle selection, per-puzzle model state, module add/remove/reorder, progress persistence, and the four-panel layout.
  - `src/index.css` - Global baseline styles loaded once at startup.
  - `src/styles/theme.js` - Shared color tokens, per-module accent colors, and reusable inline button styles.
  - `src/lib/linalg.js` - Dense matrix helpers (matmul, softmax, resize, slice) used by the forward pass.
  - `src/lib/format.js` - Parsing and display helpers for hand-edited floats, including `inf` / `-inf` spellings for attention masks.
  - `src/lib/uiScale.js` - Responsive viewport-based UI zoom calculation and resize synchronization.
  - `src/lib/axisLabels.js` - The player's custom dimension names: axis identity across the stack, default labels, and the per-puzzle label store.
  - `src/lib/model.js` - Module definitions, shape reconciliation across the stack, the forward pass, parameter counting, and puzzle grading.
  - `src/lib/attemptFile.js` - Versioned JSON encoding and strict validation for imported and exported puzzle attempts.
  - `src/features/puzzles/puzzles.js` - Puzzle catalog, target rules, exhaustive validation inputs, and elegant solution factories.
  - `src/features/puzzles/author-solutions/` - Optional versioned attempt files for author's solutions.
  - `src/components/MathText.jsx` - Shared KaTeX renderer for math expressions in the UI.
  - `src/components/NumberCell.jsx` - One hand-editable float: draft-preserving input with select-on-focus and arrow-key navigation hooks.
  - `src/components/EditableLabel.jsx` - One row/column label, click-to-rename when its axis can carry a custom name.
  - `src/components/MatrixEditor.jsx` - Labeled, keyboard-navigable grid of editable weights, with quick fills (zero / identity / causal mask) and renamable axis labels.
  - `src/components/ValueGrid.jsx` - Read-only labeled grid for computed values (residual stream, attention patterns, probabilities).
  - `src/components/ModuleStack.jsx` - Architecture column: the ordered module cards, the stream width each one emits, and the inline insert / delete / reorder controls.
  - `src/components/ObjectiveCard.jsx` - Always-visible statement of the task: goal, formula, vocabulary, exhaustive rule-validation status, representative samples, and the solved / elegant rules.
  - `src/components/ModuleInspector.jsx` - Weight editors and shape controls for the selected module, headed by that module's equation in LaTeX.
  - `src/components/TestPanel.jsx` - Detail for the selected sample: the editable scratch sequence, input/required/produced alignment, output distribution, and expandable intermediate values.
  - `src/components/PuzzleLibrary.jsx` - Left sidebar puzzle list grouped by difficulty, with solved / elegant marks.
  - `src/components/AttemptFileControls.jsx` - Header controls for moving the current model through a JSON attempt file or the clipboard, validating every import.
  - `src/components/ResizeHandle.jsx` - Draggable gutter between two panels; supports pointer drag and arrow-key resizing.

- `nn-builder/` - Git submodule holding the earlier feedforward-network builder, kept only as a design reference.
- `scripts/verify-puzzles.mjs` - Bundled Node verification for structured puzzle domains and canonical solutions.
- `idea.md` - Original design notes for the game.
- `README.md` - Source-of-truth map for this structure. If you add, remove, or repurpose files/directories, update this document in the same change.
