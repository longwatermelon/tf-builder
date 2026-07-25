import { COLORS, MONO, subtleBtnStyle } from "../styles/theme";
import MathText from "./MathText";

// one test case as "input sequence → required output sequence", clickable to inspect
function CaseRow({ index, test, ok, isActive, onSelect }) {
  const arrowColor = ok ? COLORS.success : COLORS.negative;
  return (
    <div
      onClick={onSelect}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "5px 8px",
        borderRadius: 4,
        cursor: "pointer",
        border: `1px solid ${isActive ? COLORS.accent : "transparent"}`,
        background: isActive ? COLORS.accentDim : "transparent",
      }}
    >
      <span style={{ fontSize: 10, color: COLORS.textMuted, minWidth: 14 }}>{index + 1}</span>
      <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textBright, letterSpacing: 2 }}>
        {test.tokens.join(" ")}
      </span>
      <span style={{ color: COLORS.textMuted, fontSize: 11 }}>→</span>
      <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.warn, letterSpacing: 2 }}>
        {test.targets.map((t) => t ?? "·").join(" ")}
      </span>
      <span style={{ marginLeft: "auto", fontSize: 11, color: arrowColor }}>{ok ? "✓" : "✗"}</span>
    </div>
  );
}

// the always-visible statement of what the player has to build and how it is judged
export default function ObjectiveCard({ puzzle, evaluation, activeTab, onSelectTab }) {
  const passed = evaluation.results.filter((r) => r.ok).length;

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
          <span style={{ fontSize: 10, color: COLORS.textMuted }}>
            up to <span style={{ fontFamily: MONO, color: COLORS.textBright }}>{puzzle.maxLen}</span> positions
          </span>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "12px 0 4px" }}>
        <span style={{ fontSize: 10, letterSpacing: 0.6, textTransform: "uppercase", color: COLORS.textMuted }}>
          Test cases — all must pass
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

      <button
        type="button"
        onClick={() => onSelectTab("scratch")}
        style={{
          ...subtleBtnStyle,
          marginTop: 6,
          width: "100%",
          textAlign: "left",
          borderColor: activeTab === "scratch" ? COLORS.accent : COLORS.panelBorder,
          background: activeTab === "scratch" ? COLORS.accentDim : "transparent",
        }}
      >
        scratch sequence — try your own input
      </button>

      <div
        style={{
          marginTop: 12,
          paddingTop: 10,
          borderTop: `1px solid ${COLORS.panelBorder}`,
          fontSize: 10,
          color: COLORS.textMuted,
          lineHeight: 1.6,
        }}
      >
        <div>
          <span style={{ color: COLORS.success, fontWeight: 600 }}>Solved</span> when, at every position of every test
          case, the required token is the highest-probability token and leads the runner-up by at least{" "}
          <span style={{ fontFamily: MONO, color: COLORS.text }}>{puzzle.epsilon}</span>.
        </div>
        <div>
          <span style={{ color: COLORS.violet, fontWeight: 600 }}>Elegant</span> when solved using at most{" "}
          <span style={{ fontFamily: MONO, color: COLORS.text }}>{puzzle.canonicalParams}</span> total parameters —
          currently <span style={{ fontFamily: MONO, color: COLORS.text }}>{evaluation.params}</span>.
        </div>
      </div>
    </div>
  );
}
