# README

This project is a website for hand-crafting small transformers. The player picks a **puzzle**, assembles a module stack over a residual stream, and edits every weight by hand until the model's per-position next-token predictions match the puzzle's targets.

## Summary

- A **puzzle** is one input/output specification: a vocabulary, a fixed and fully visible set of test sequences, and a per-position target token for each. Puzzles are graded `tutorial` / `easy` / `medium` / `hard`.
- The **module stack** is the player's architecture. It always begins with an **Embedding** module and can be extended with **Attention Head**, **MLP**, and **Linear** modules. Attention and MLP add into the residual stream; Linear is the only module that changes the stream width, so it doubles as the unembedding.
- There is no LayerNorm, and attention scores are raw `QKᵀ` with no `1/√d` scaling — both would only make hand-crafting harder without teaching anything.
- A puzzle is **solved** when, for every test sequence and every position, the target token is the argmax of the output softmax *and* beats the runner-up by the puzzle's `epsilon`.
- A solve is **elegant** when the total allocated parameter count is at most the canonical solution's. Optional blocks (`W_E`, `W_P`, the attention mask `M`) can be switched off to drop their parameters from the count.
- Each puzzle ships a **canonical solution** the player can reveal; the elegance bar is derived from it rather than hand-written. Revealing locks progress for the current attempt only — resetting back to a blank model (or reloading) lets a from-scratch rebuild earn the mark again.

## File Structure

- `src/` - Frontend app source for the transformer builder.
  - `src/main.jsx` - React entry point that mounts `App` and loads global + KaTeX styles.
  - `src/App.jsx` - Top-level orchestration: puzzle selection, per-puzzle model state, module add/remove/reorder, progress persistence, and the four-panel layout.
  - `src/index.css` - Global baseline styles loaded once at startup.
  - `src/styles/theme.js` - Shared color tokens, per-module accent colors, and reusable inline button styles.
  - `src/lib/linalg.js` - Dense matrix helpers (matmul, softmax, resize, slice) used by the forward pass.
  - `src/lib/format.js` - Parsing and display helpers for hand-edited floats, including `inf` / `-inf` spellings for attention masks.
  - `src/lib/model.js` - Module definitions, shape reconciliation across the stack, the forward pass, parameter counting, and puzzle grading.
  - `src/features/puzzles/puzzles.js` - Puzzle catalog and canonical solution factories.
  - `src/components/MathText.jsx` - Shared KaTeX renderer for math expressions in the UI.
  - `src/components/NumberCell.jsx` - One hand-editable float: draft-preserving input with select-on-focus and arrow-key navigation hooks.
  - `src/components/MatrixEditor.jsx` - Labeled, keyboard-navigable grid of editable weights, with quick fills (zero / identity / causal mask).
  - `src/components/ValueGrid.jsx` - Read-only labeled grid for computed values (residual stream, attention patterns, probabilities).
  - `src/components/ModuleStack.jsx` - Architecture column: the ordered module cards, stream widths between them, and the add-module controls.
  - `src/components/ModuleInspector.jsx` - Weight editors and shape controls for the selected module, headed by that module's equation in LaTeX.
  - `src/components/TestPanel.jsx` - Forward-pass column: test tabs, the editable scratch sequence, token/target/prediction alignment, output distribution, and expandable intermediate values.
  - `src/components/PuzzleLibrary.jsx` - Left sidebar puzzle list grouped by difficulty, with solved / elegant marks.

- `nn-builder/` - Git submodule holding the earlier feedforward-network builder, kept only as a design reference.
- `idea.md` - Original design notes for the game.
- `README.md` - Source-of-truth map for this structure. If you add, remove, or repurpose files/directories, update this document in the same change.
