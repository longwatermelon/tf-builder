import { useEffect, useMemo, useState } from "react";
import ModuleInspector from "./components/ModuleInspector";
import ModuleStack from "./components/ModuleStack";
import ObjectiveCard from "./components/ObjectiveCard";
import PuzzleLibrary from "./components/PuzzleLibrary";
import TestPanel from "./components/TestPanel";
import { buildSolution, getPuzzle, PUZZLES } from "./features/puzzles/puzzles";
import {
  computeInputWidths,
  createAttn,
  createInitialModel,
  createLinear,
  createMlp,
  evaluatePuzzle,
  reconcileShapes,
} from "./lib/model";
import { COLORS, DIFFICULTY_COLORS, MONO, btnStyle, subtleBtnStyle } from "./styles/theme";

const PROGRESS_KEY = "tf-builder:progress";

const PROGRESS_STATUSES = ["solved", "elegant"];

// fresh sequence for the scratch tab, defaulting to the first test's tokens
function defaultScratch(puzzle) {
  return [...(puzzle.tests[0]?.tokens ?? [puzzle.vocab[0]])];
}

// read persisted progress, discarding anything that is not a known status map
function loadProgress() {
  if (typeof window === "undefined") return {};
  try {
    const parsed = JSON.parse(window.localStorage.getItem(PROGRESS_KEY) ?? "{}");
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
    return Object.fromEntries(Object.entries(parsed).filter(([, status]) => PROGRESS_STATUSES.includes(status)));
  } catch {
    return {};
  }
}

function Panel({ children, style }) {
  return (
    <div
      style={{
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelBorder}`,
        borderRadius: 6,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export default function App() {
  const [activePuzzleId, setActivePuzzleId] = useState(PUZZLES[0].id);
  const puzzle = getPuzzle(activePuzzleId);

  // one in-progress model per puzzle so switching puzzles never loses work
  const [models, setModels] = useState(() => ({ [PUZZLES[0].id]: createInitialModel(PUZZLES[0]) }));
  const [scratchByPuzzle, setScratchByPuzzle] = useState(() => ({ [PUZZLES[0].id]: defaultScratch(PUZZLES[0]) }));
  const [selectedModuleId, setSelectedModuleId] = useState(() => models[PUZZLES[0].id].modules[0].id);
  const [activeTab, setActiveTab] = useState(0);
  const [revealedIds, setRevealedIds] = useState(() => new Set());
  const [progress, setProgress] = useState(loadProgress);

  const model = models[activePuzzleId];
  const scratchTokens = scratchByPuzzle[activePuzzleId] ?? defaultScratch(puzzle);
  const inputWidths = useMemo(() => computeInputWidths(model), [model]);
  const evaluation = useMemo(() => evaluatePuzzle(model, puzzle), [model, puzzle]);
  const selectedIndex = model.modules.findIndex((m) => m.id === selectedModuleId);
  const selectedModule = selectedIndex >= 0 ? model.modules[selectedIndex] : model.modules[0];
  const selectedWidth = selectedIndex >= 0 ? inputWidths[selectedIndex] : model.dModel;

  // persist solved / elegant marks across sessions
  useEffect(() => {
    try {
      window.localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
    } catch {
      // ignore storage failures
    }
  }, [progress]);

  // record the best status reached; the reveal lock is attempt-scoped, so an attempt that started
  // from the revealed canonical never counts, but resetting to a blank model earns credit again
  function recordProgress(nextModel) {
    if (revealedIds.has(activePuzzleId)) return;
    const result = evaluatePuzzle(nextModel, puzzle);
    if (!result.solved) return;
    const status = result.elegant ? "elegant" : "solved";
    setProgress((prev) => {
      if (prev[activePuzzleId] === "elegant" || prev[activePuzzleId] === status) return prev;
      return { ...prev, [activePuzzleId]: status };
    });
  }

  function updateModel(nextModel) {
    const reconciled = reconcileShapes(nextModel, puzzle);
    setModels((prev) => ({ ...prev, [activePuzzleId]: reconciled }));
    recordProgress(reconciled);
  }

  function selectPuzzle(id) {
    const next = getPuzzle(id);
    const target = models[id] ?? createInitialModel(next);
    if (!models[id]) setModels((prev) => ({ ...prev, [id]: target }));
    if (!scratchByPuzzle[id]) setScratchByPuzzle((prev) => ({ ...prev, [id]: defaultScratch(next) }));
    setActivePuzzleId(id);
    setActiveTab(0);
    setSelectedModuleId(target.modules[0].id);
  }

  // insert a new module at the given position; index 0 is reserved for the embedding
  function addModule(type, index) {
    const factory = type === "attn" ? createAttn : type === "mlp" ? createMlp : createLinear;
    const created = type === "linear" ? factory({ dOut: puzzle.vocab.length }) : factory({});
    const at = Math.min(Math.max(index ?? model.modules.length, 1), model.modules.length);
    const modules = [...model.modules];
    modules.splice(at, 0, created);
    updateModel({ ...model, modules });
    setSelectedModuleId(created.id);
  }

  function removeModule(id) {
    const modules = model.modules.filter((m) => m.id !== id);
    updateModel({ ...model, modules });
    if (id === selectedModuleId) setSelectedModuleId(modules[0].id);
  }

  // swap a module with its neighbour; the embed at index 0 stays pinned
  function moveModule(id, direction) {
    const index = model.modules.findIndex((m) => m.id === id);
    const target = index + direction;
    if (index <= 0 || target <= 0 || target >= model.modules.length) return;
    const modules = [...model.modules];
    [modules[index], modules[target]] = [modules[target], modules[index]];
    updateModel({ ...model, modules });
  }

  function resetModel() {
    const fresh = createInitialModel(puzzle);
    setModels((prev) => ({ ...prev, [activePuzzleId]: fresh }));
    setSelectedModuleId(fresh.modules[0].id);
    setRevealedIds((prev) => {
      const next = new Set(prev);
      next.delete(activePuzzleId);
      return next;
    });
  }

  function revealSolution() {
    const solution = buildSolution(puzzle);
    setModels((prev) => ({ ...prev, [activePuzzleId]: solution }));
    setSelectedModuleId(solution.modules[0].id);
    setRevealedIds((prev) => new Set(prev).add(activePuzzleId));
  }

  const isRevealed = revealedIds.has(activePuzzleId);
  const statusColor = evaluation.elegant ? COLORS.violet : evaluation.solved ? COLORS.success : COLORS.textMuted;
  const statusText = evaluation.elegant ? "Elegant" : evaluation.solved ? "Solved" : "Unsolved";
  const passedCount = evaluation.results.filter((r) => r.ok).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: COLORS.bg, color: COLORS.text }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 14,
          padding: "10px 16px",
          borderBottom: `1px solid ${COLORS.panelBorder}`,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 700, color: COLORS.textBright, letterSpacing: 0.3 }}>tf-builder</span>
        <span style={{ color: COLORS.panelBorder }}>│</span>
        <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.textBright }}>{puzzle.name}</span>
        <span
          style={{
            fontSize: 10,
            letterSpacing: 0.6,
            textTransform: "uppercase",
            color: DIFFICULTY_COLORS[puzzle.difficulty],
          }}
        >
          {puzzle.difficulty}
        </span>

        <div style={{ display: "flex", alignItems: "center", gap: 12, marginLeft: "auto" }}>
          <span style={{ fontFamily: MONO, fontSize: 11, color: COLORS.textMuted }}>
            {passedCount}/{evaluation.results.length} tests
          </span>
          <span style={{ fontFamily: MONO, fontSize: 11, color: COLORS.textMuted }}>
            {evaluation.params}p / canonical {puzzle.canonicalParams}p
          </span>
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: statusColor,
              border: `1px solid ${statusColor}`,
              borderRadius: 999,
              padding: "2px 10px",
            }}
          >
            {statusText}
            {isRevealed ? " (revealed)" : ""}
          </span>
          <button type="button" style={subtleBtnStyle} onClick={resetModel}>
            reset
          </button>
          <button type="button" style={btnStyle} onClick={revealSolution}>
            reveal canonical
          </button>
        </div>
      </header>

      <div style={{ flex: 1, display: "flex", gap: 8, padding: 8, minHeight: 0 }}>
        <Panel style={{ flex: "0 0 218px" }}>
          <PuzzleLibrary puzzles={PUZZLES} activeId={activePuzzleId} progress={progress} onSelect={selectPuzzle} />
        </Panel>

        <Panel style={{ flex: "0 0 250px" }}>
          <ModuleStack
            model={model}
            puzzle={puzzle}
            inputWidths={inputWidths}
            selectedId={selectedModule?.id}
            onSelect={setSelectedModuleId}
            onAdd={addModule}
            onRemove={removeModule}
            onMove={moveModule}
          />
        </Panel>

        <Panel style={{ flex: "1 1 520px" }}>
          <div
            style={{
              padding: "10px 12px 6px",
              fontSize: 11,
              letterSpacing: 0.8,
              textTransform: "uppercase",
              color: COLORS.textMuted,
              display: "flex",
              gap: 8,
            }}
          >
            Weights
            {puzzle.hint ? (
              <span style={{ marginLeft: "auto", textTransform: "none", letterSpacing: 0, color: COLORS.warn, fontSize: 10 }}>
                hint: {puzzle.hint}
              </span>
            ) : null}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: "0 14px 14px" }}>
            {selectedModule ? (
              <ModuleInspector
                module={selectedModule}
                dIn={selectedWidth}
                dModel={model.dModel}
                puzzle={puzzle}
                onChange={(next) =>
                  updateModel({ ...model, modules: model.modules.map((m) => (m.id === next.id ? next : m)) })
                }
                onChangeDModel={(dModel) => updateModel({ ...model, dModel })}
              />
            ) : null}
          </div>
        </Panel>

        <Panel style={{ flex: "1 1 400px" }}>
          <div
            style={{
              padding: "10px 12px 8px",
              fontSize: 11,
              letterSpacing: 0.8,
              textTransform: "uppercase",
              color: COLORS.textMuted,
            }}
          >
            Objective
          </div>
          {/* the objective stays pinned; only the per-case detail below it scrolls */}
          <div style={{ flexShrink: 0, maxHeight: "60%", overflowY: "auto" }}>
            <ObjectiveCard puzzle={puzzle} evaluation={evaluation} activeTab={activeTab} onSelectTab={setActiveTab} />
          </div>
          <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
            <TestPanel
              puzzle={puzzle}
              model={model}
              evaluation={evaluation}
              scratchTokens={scratchTokens}
              onChangeScratch={(tokens) => setScratchByPuzzle((prev) => ({ ...prev, [activePuzzleId]: tokens }))}
              activeTab={activeTab}
            />
          </div>
        </Panel>
      </div>
    </div>
  );
}
