import { COLORS, MONO } from "../styles/theme";
import MathText from "./MathText";

// shared shape for every selectable row in the case list, including the scratch row
function rowStyle(isActive) {
  return {
    display: "flex",
    alignItems: "center",
    gap: 8,
    width: "100%",
    padding: "5px 8px",
    borderRadius: 4,
    cursor: "pointer",
    textAlign: "left",
    fontFamily: "'Sora', sans-serif",
    border: `1px solid ${isActive ? COLORS.accent : "transparent"}`,
    background: isActive ? COLORS.accentDim : "transparent",
  };
}

// one test case as "input sequence → required output sequence", clickable to inspect
function CaseRow({ index, test, ok, isActive, onSelect }) {
  const arrowColor = ok ? COLORS.success : COLORS.negative;
  return (
    <button type="button" onClick={onSelect} style={rowStyle(isActive)}>
      <span style={{ fontSize: 10, color: COLORS.textMuted, minWidth: 14 }}>{index + 1}</span>
      <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textBright, letterSpacing: 2 }}>
        {test.tokens.join(" ")}
      </span>
      <span style={{ color: COLORS.textMuted, fontSize: 11 }}>→</span>
      <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.warn, letterSpacing: 2 }}>
        {test.targets.map((t) => t ?? "·").join(" ")}
      </span>
      <span style={{ marginLeft: "auto", fontSize: 11, color: arrowColor }}>{ok ? "✓" : "✗"}</span>
    </button>
  );
}

// the always-visible statement of what the player has to build and how it is judged
export default function ObjectiveCard({ puzzle, evaluation, scratchTokens, activeTab, onSelectTab }) {
  const passed = evaluation.results.filter((r) => r.ok).length;
  const inputAlphabet = puzzle.validationVocab ?? puzzle.inputVocab;

  return (
    <div style={{ padding: "0 12px 12px" }}>
      <div
        style={{
          border: `1px solid ${COLORS.panelBorder}`,
          background: "rgba(0,0,0,0.22)",
          borderRadius: 5,
          padding: "10px 12px",
        }}
      >
        <div style={{ fontSize: 12, color: COLORS.textBright, lineHeight: 1.5 }}>{puzzle.blurb}</div>

        <MathText tex={puzzle.formula} displayMode style={{ color: COLORS.text, margin: "10px 0 4px" }} />

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 8 }}>
          <span style={{ fontSize: 10, color: COLORS.textMuted }}>
            vocabulary{" "}
            <span style={{ fontFamily: MONO, color: COLORS.textBright, letterSpacing: 2 }}>{puzzle.vocab.join(" ")}</span>
          </span>
          {puzzle.inputFormat ? (
            <span style={{ fontSize: 10, color: COLORS.textMuted }}>
              input format <span style={{ fontFamily: MONO, color: COLORS.textBright }}>{puzzle.inputFormat}</span>
            </span>
          ) : puzzle.validationPrefix ? (
            <span style={{ fontSize: 10, color: COLORS.textMuted }}>
              inputs{" "}
              <span style={{ fontFamily: MONO, color: COLORS.textBright, letterSpacing: 2 }}>
                {puzzle.validationPrefix.join(" ")} first; then {inputAlphabet.join(" ")}
              </span>
            </span>
          ) : puzzle.inputVocab ? (
            <span style={{ fontSize: 10, color: COLORS.textMuted }}>
              inputs{" "}
              <span style={{ fontFamily: MONO, color: COLORS.textBright, letterSpacing: 2 }}>
                {puzzle.inputVocab.join(" ")}
              </span>
            </span>
          ) : null}
          <span style={{ fontSize: 10, color: COLORS.textMuted }}>
            {puzzle.fixedLen ? "exactly" : "up to"}{" "}
            <span style={{ fontFamily: MONO, color: COLORS.textBright }}>{puzzle.maxLen}</span> positions
          </span>
        </div>
      </div>

      <div style={{ marginTop: 10, fontSize: 10, color: COLORS.textMuted, lineHeight: 1.6 }}>
        <div>
          <span style={{ color: COLORS.success, fontWeight: 600 }}>Solved</span> when every valid input generated from
          the rule passes: the required token must lead the runner-up by at least{" "}
          <span style={{ fontFamily: MONO, color: COLORS.text }}>{puzzle.epsilon}</span>. Rule validation currently{" "}
          <span style={{ fontFamily: MONO, color: COLORS.text }}>
            {evaluation.validationPassed}/{evaluation.validationTotal}
          </span>
          .
        </div>
        <div>
          <span style={{ color: COLORS.violet, fontWeight: 600 }}>Elegant</span> when solved using at most{" "}
          <span style={{ fontFamily: MONO, color: COLORS.text }}>{puzzle.canonicalParams}</span> total parameters —
          currently <span style={{ fontFamily: MONO, color: COLORS.text }}>{evaluation.params}</span>.
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "12px 0 4px" }}>
        <span style={{ fontSize: 10, letterSpacing: 0.6, textTransform: "uppercase", color: COLORS.textMuted }}>
          Representative samples
        </span>
        <span
          style={{
            fontFamily: MONO,
            fontSize: 10,
            color: passed === evaluation.results.length ? COLORS.success : COLORS.textMuted,
            marginLeft: "auto",
          }}
        >
          {passed}/{evaluation.results.length}
        </span>
      </div>

      <div style={{ display: "flex", gap: 10, padding: "0 8px 3px", fontSize: 9, color: COLORS.textMuted }}>
        <span style={{ minWidth: 14 }} />
        <span>input</span>
        <span style={{ marginLeft: 4 }}>required output</span>
      </div>

      {evaluation.results.map((result, i) => (
        <CaseRow
          key={i}
          index={i}
          test={result.test}
          ok={result.ok}
          isActive={activeTab === i}
          onSelect={() => onSelectTab(i)}
        />
      ))}

      {/* the scratch row is not graded, but it selects like a case so it wears the same shape */}
      <button type="button" onClick={() => onSelectTab("scratch")} style={rowStyle(activeTab === "scratch")}>
        <span style={{ fontSize: 10, color: COLORS.textMuted, minWidth: 14 }}>·</span>
        <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textBright, letterSpacing: 2 }}>
          {scratchTokens.join(" ")}
        </span>
        <span style={{ color: COLORS.textMuted, fontSize: 11 }}>→</span>
        <span style={{ fontSize: 11, color: COLORS.textMuted }}>scratch — try your own input</span>
      </button>
    </div>
  );
}
