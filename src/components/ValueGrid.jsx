import { formatActivation } from "../lib/format";
import { COLORS, MONO } from "../styles/theme";

const CELL_WIDTH = 58;

// read-only labeled grid for computed values (stream, attention pattern, probabilities)
export default function ValueGrid({
  matrix,
  rowLabels,
  colLabels,
  rowAxis,
  colAxis,
  format = formatActivation,
  cellStyleAt,
}) {
  const rows = matrix.length;
  const cols = rows === 0 ? 0 : matrix[0].length;
  if (rows === 0 || cols === 0) return null;

  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
      {rowAxis ? (
        <span
          style={{
            fontSize: 9,
            letterSpacing: 0.6,
            textTransform: "uppercase",
            color: COLORS.textMuted,
            writingMode: "vertical-rl",
            transform: "rotate(180deg)",
            marginTop: 20,
          }}
        >
          {rowAxis}
        </span>
      ) : null}
      <div>
        {colAxis ? (
          <span
            style={{
              display: "block",
              fontSize: 9,
              letterSpacing: 0.6,
              textTransform: "uppercase",
              color: COLORS.textMuted,
              marginBottom: 3,
              marginLeft: 34,
            }}
          >
            {colAxis}
          </span>
        ) : null}
        <div style={{ display: "grid", gridTemplateColumns: `auto repeat(${cols}, ${CELL_WIDTH}px)`, gap: 3 }}>
          <div />
          {colLabels.map((label, j) => (
            <div key={`c-${j}`} style={{ fontSize: 10, color: COLORS.textMuted, fontFamily: MONO, textAlign: "center" }}>
              {label}
            </div>
          ))}
          {matrix.map((row, i) => (
            <GridRow
              key={`r-${i}`}
              row={row}
              i={i}
              label={rowLabels[i]}
              format={format}
              cellStyleAt={cellStyleAt}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// one value row: left label plus its formatted cells
function GridRow({ row, i, label, format, cellStyleAt }) {
  return (
    <>
      <div
        style={{
          fontSize: 10,
          color: COLORS.textMuted,
          fontFamily: MONO,
          textAlign: "right",
          paddingRight: 4,
          minWidth: 30,
          lineHeight: "21px",
        }}
      >
        {label}
      </div>
      {row.map((value, j) => (
        <div
          key={`v-${i}-${j}`}
          style={{
            fontFamily: MONO,
            fontSize: 11,
            textAlign: "center",
            padding: "3px 4px",
            borderRadius: 3,
            background: "rgba(30,30,30,0.55)",
            border: `1px solid ${COLORS.panelBorder}`,
            color: value === 0 ? COLORS.textMuted : COLORS.text,
            ...(cellStyleAt ? cellStyleAt(i, j, value) : null),
          }}
        >
          {format(value)}
        </div>
      ))}
    </>
  );
}
