import { useEffect, useMemo, useRef, useState } from "react";
import ModuleInspector from "./components/ModuleInspector";
import ModuleStack from "./components/ModuleStack";
import ObjectiveCard from "./components/ObjectiveCard";
import PuzzleLibrary from "./components/PuzzleLibrary";
import ResizeHandle, { HANDLE_WIDTH } from "./components/ResizeHandle";
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
import { COLORS, DIFFICULTY_COLORS, MONO, UI_ZOOM, btnStyle, subtleBtnStyle } from "./styles/theme";

const PROGRESS_KEY = "tf-builder:progress:rule-validation-v1";

const PROGRESS_STATUSES = ["solved", "elegant"];

const ROW_PADDING = 8;

const MIN_PANEL_WIDTH = 150;

// pixel widths of the puzzles, objective and architecture panels; the weights panel between the
// last two absorbs whatever space is left
const DEFAULT_WIDTHS = [250, 500, 360];

// space the panels themselves can occupy, excluding the row padding and the gutters
function measureAvailable(row) {
  if (!row) return 0;
  return row.clientWidth - ROW_PADDING * 2 - HANDLE_WIDTH * DEFAULT_WIDTHS.length;
}

// hold a panel between its minimum and the width the flexible weights panel can still spare
function clampPanelWidth(widths, index, width, available) {
  const used = widths.reduce((sum, w) => sum + w, 0);
  const slack = Math.max(0, available - used - MIN_PANEL_WIDTH);
  return Math.min(Math.max(width, MIN_PANEL_WIDTH), widths[index] + slack);
}

// shrink fixed panels from the right until they fit alongside a minimum-width weights panel
function fitWidths(widths, available) {
  const next = widths.map((w) => Math.max(MIN_PANEL_WIDTH, w));
  let overflow = next.reduce((sum, w) => sum + w, 0) + MIN_PANEL_WIDTH - available;
  for (let i = next.length - 1; i >= 0 && overflow > 0; i--) {
    const shrink = Math.min(overflow, next[i] - MIN_PANEL_WIDTH);
    next[i] -= shrink;
    overflow -= shrink;
  }
  return next;
}

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

function Panel({ children, style, ref }) {
  return (
    <div
      ref={ref}
      style={{
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelBorder}`,
        borderRadius: 6,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        // keep wide content from stretching a panel past the width the drag handles assign it
        minWidth: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// guideRefs are optional walkthrough anchors supplied by RootApp
export default function App({ guideRefs = {} }) {
  const {
    headerRef = null,
    puzzlesRef = null,
    objectiveRef = null,
    weightsRef = null,
    stackRef = null,
  } = guideRefs;

  const [activePuzzleId, setActivePuzzleId] = useState(PUZZLES[0].id);
  const puzzle = getPuzzle(activePuzzleId);

  // one in-progress model per puzzle so switching puzzles never loses work
  const [models, setModels] = useState(() => ({ [PUZZLES[0].id]: createInitialModel(PUZZLES[0]) }));
  const [scratchByPuzzle, setScratchByPuzzle] = useState(() => ({ [PUZZLES[0].id]: defaultScratch(PUZZLES[0]) }));
  const [selectedModuleId, setSelectedModuleId] = useState(() => models[PUZZLES[0].id].modules[0].id);
  const [activeTab, setActiveTab] = useState(null);
  const [revealedIds, setRevealedIds] = useState(() => new Set());
  const [progress, setProgress] = useState(loadProgress);
  const [panelWidths, setPanelWidths] = useState(DEFAULT_WIDTHS);
  const [availableWidth, setAvailableWidth] = useState(0);
  const rowRef = useRef(null);
  const dragWidthsRef = useRef(DEFAULT_WIDTHS);

  const model = models[activePuzzleId];
  const scratchTokens = scratchByPuzzle[activePuzzleId] ?? defaultScratch(puzzle);
  const inputWidths = useMemo(() => computeInputWidths(model), [model]);
  const evaluation = useMemo(() => evaluatePuzzle(model, puzzle), [model, puzzle]);
  const selectedIndex = model.modules.findIndex((m) => m.id === selectedModuleId);
  const selectedModule = selectedIndex >= 0 ? model.modules[selectedIndex] : model.modules[0];
  const selectedWidth = selectedIndex >= 0 ? inputWidths[selectedIndex] : model.dModel;
  // how far any one panel can still grow before the last panel hits its minimum
  const panelSlack = Math.max(0, availableWidth - panelWidths.reduce((sum, w) => sum + w, 0) - MIN_PANEL_WIDTH);

  // persist solved / elegant marks across sessions
  useEffect(() => {
    try {
      window.localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
    } catch {
      // ignore storage failures
    }
  }, [progress]);

  // record the rendered exhaustive result once, while revealed attempts remain ineligible
  useEffect(() => {
    if (revealedIds.has(activePuzzleId) || !evaluation.solved) return undefined;
    const status = evaluation.elegant ? "elegant" : "solved";
    const timeoutId = window.setTimeout(() => {
      setProgress((prev) => {
        if (prev[activePuzzleId] === "elegant" || prev[activePuzzleId] === status) return prev;
        return { ...prev, [activePuzzleId]: status };
      });
    }, 0);
    return () => window.clearTimeout(timeoutId);
  }, [activePuzzleId, evaluation.elegant, evaluation.solved, revealedIds]);

  // keep the last panel usable when the window shrinks, and on first layout
  useEffect(() => {
    function clampToWindow() {
      const available = measureAvailable(rowRef.current);
      if (!available) return;
      setAvailableWidth(available);
      setPanelWidths((prev) => {
        const next = fitWidths(prev, available);
        return next.every((w, i) => w === prev[i]) ? prev : next;
      });
    }
    clampToWindow();
    window.addEventListener("resize", clampToWindow);
    return () => window.removeEventListener("resize", clampToWindow);
  }, []);

  // snapshot the widths a drag starts from so every move applies to the same baseline
  function startResize() {
    dragWidthsRef.current = panelWidths;
  }

  // dragging a gutter resizes only the panel it belongs to; the weights panel takes up the slack.
  // the snapshot supplies the requested width, but the clamp reads live state so a window resize
  // mid-drag is not undone by the next pointer move
  function resizePanel(index, dx) {
    const requested = dragWidthsRef.current[index] + dx;
    setPanelWidths((prev) => {
      const width = clampPanelWidth(prev, index, requested, availableWidth);
      return prev.map((w, i) => (i === index ? width : w));
    });
  }

  // arrow keys step from the latest width, so repeats accumulate even when renders are batched
  function nudgePanel(index, delta) {
    setPanelWidths((prev) => {
      const width = clampPanelWidth(prev, index, prev[index] + delta, availableWidth);
      return prev.map((w, i) => (i === index ? width : w));
    });
  }

  function updateModel(nextModel) {
    const reconciled = reconcileShapes(nextModel, puzzle);
    setModels((prev) => ({ ...prev, [activePuzzleId]: reconciled }));
  }

  function selectPuzzle(id) {
    const next = getPuzzle(id);
    const target = models[id] ?? createInitialModel(next);
    if (!models[id]) setModels((prev) => ({ ...prev, [id]: target }));
    if (!scratchByPuzzle[id]) setScratchByPuzzle((prev) => ({ ...prev, [id]: defaultScratch(next) }));
    setActivePuzzleId(id);
    setActiveTab(null);
    setSelectedModuleId(target.modules[0].id);
  }

  // select a case for inspection, or clear it when the selected row is clicked again
  function toggleActiveTab(tab) {
    setActiveTab((current) => (current === tab ? null : tab));
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

  return (
    // the zoom scales the whole app one step up; every px below is laid out inside that scale
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: COLORS.bg,
        color: COLORS.text,
        zoom: UI_ZOOM,
      }}
    >
      <header
        ref={headerRef}
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
            {evaluation.validationPassed}/{evaluation.validationTotal} valid inputs
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

      {/* the gutters between panels double as drag handles, so the row itself has no gap;
          below four minimum-width panels it scrolls rather than clipping the last one */}
      <div ref={rowRef} style={{ flex: 1, display: "flex", padding: ROW_PADDING, minHeight: 0, overflowX: "auto" }}>
        <Panel ref={puzzlesRef} style={{ flex: `0 0 ${panelWidths[0]}px` }}>
          <PuzzleLibrary puzzles={PUZZLES} activeId={activePuzzleId} progress={progress} onSelect={selectPuzzle} />
        </Panel>

        <ResizeHandle
          label="Resize puzzles panel"
          width={panelWidths[0]}
          minWidth={MIN_PANEL_WIDTH}
          maxWidth={panelWidths[0] + panelSlack}
          onDragStart={startResize}
          onDragMove={(dx) => resizePanel(0, dx)}
          onNudge={(delta) => nudgePanel(0, delta)}
        />

        <Panel ref={objectiveRef} style={{ flex: `0 0 ${panelWidths[1]}px` }}>
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
            <ObjectiveCard
              puzzle={puzzle}
              evaluation={evaluation}
              scratchTokens={scratchTokens}
              activeTab={activeTab}
              onSelectTab={toggleActiveTab}
            />
          </div>
          <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
            {activeTab !== null ? (
              <TestPanel
                puzzle={puzzle}
                model={model}
                evaluation={evaluation}
                scratchTokens={scratchTokens}
                onChangeScratch={(tokens) => setScratchByPuzzle((prev) => ({ ...prev, [activePuzzleId]: tokens }))}
                activeTab={activeTab}
              />
            ) : null}
          </div>
        </Panel>

        <ResizeHandle
          label="Resize objective panel"
          width={panelWidths[1]}
          minWidth={MIN_PANEL_WIDTH}
          maxWidth={panelWidths[1] + panelSlack}
          onDragStart={startResize}
          onDragMove={(dx) => resizePanel(1, dx)}
          onNudge={(delta) => nudgePanel(1, delta)}
        />

        <Panel ref={weightsRef} style={{ flex: "1 1 0", minWidth: MIN_PANEL_WIDTH }}>
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
            <span style={{ fontSize: 10, letterSpacing: 0, textTransform: "none" }}>q: − · w: 0 · e: inf</span>
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

        {/* this gutter sits to the right of the panel it sizes, so its drag direction is inverted */}
        <ResizeHandle
          label="Resize architecture panel"
          width={panelWidths[2]}
          minWidth={MIN_PANEL_WIDTH}
          maxWidth={panelWidths[2] + panelSlack}
          onDragStart={startResize}
          onDragMove={(dx) => resizePanel(2, -dx)}
          onNudge={(delta) => nudgePanel(2, -delta)}
        />

        <Panel ref={stackRef} style={{ flex: `0 0 ${panelWidths[2]}px` }}>
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
      </div>
    </div>
  );
}
