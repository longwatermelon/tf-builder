import { COLORS, DIFFICULTY_COLORS, MONO } from "../styles/theme";
import MathText from "./MathText";

const DIFFICULTY_ORDER = ["tutorial", "easy", "medium", "hard"];

export default function PuzzleLibrary({ puzzles, activeId, progress, onSelect }) {
  const groups = DIFFICULTY_ORDER.map((difficulty) => ({
    difficulty,
    items: puzzles.filter((puzzle) => puzzle.difficulty === difficulty),
  })).filter((group) => group.items.length > 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ padding: "10px 12px 6px", fontSize: 11, letterSpacing: 0.8, textTransform: "uppercase", color: COLORS.textMuted }}>
        Puzzles
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "0 10px 12px" }}>
        {groups.map((group) => (
          <div key={group.difficulty} style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 9,
                letterSpacing: 1,
                textTransform: "uppercase",
                color: DIFFICULTY_COLORS[group.difficulty],
                padding: "4px 4px 5px",
              }}
            >
              {group.difficulty}
            </div>
            {group.items.map((puzzle) => {
              const isActive = puzzle.id === activeId;
              const state = progress[puzzle.id];
              return (
                <div
                  key={puzzle.id}
                  onClick={() => onSelect(puzzle.id)}
                  style={{
                    border: `1px solid ${isActive ? COLORS.accent : COLORS.panelBorder}`,
                    background: isActive ? COLORS.accentDim : COLORS.surface,
                    borderRadius: 5,
                    padding: "7px 9px",
                    marginBottom: 5,
                    cursor: "pointer",
                    transition: "border-color 0.12s, background 0.12s",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.textBright }}>{puzzle.name}</span>
                    {state === "elegant" ? (
                      <span style={{ fontSize: 10, color: COLORS.violet, marginLeft: "auto" }}>★ elegant</span>
                    ) : state === "solved" ? (
                      <span style={{ fontSize: 10, color: COLORS.success, marginLeft: "auto" }}>✓ solved</span>
                    ) : null}
                  </div>
                  <MathText tex={puzzle.formula} style={{ color: COLORS.textMuted, fontSize: 11, marginTop: 3 }} />
                  <div style={{ fontFamily: MONO, fontSize: 9, color: COLORS.textMuted, marginTop: 3 }}>
                    elegant {puzzle.canonicalParams}p
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
