import { useRef } from "react";
import { cloneMatrix } from "../lib/linalg";
import { FILL_IDENTITY, FILL_ZERO } from "../lib/matrixFills";
import { COLORS, MONO, smallBtnStyle } from "../styles/theme";
import MathText from "./MathText";
import NumberCell from "./NumberCell";

const CELL_WIDTH = 62;

// small caption used for the row/column axis names
function AxisLabel({ text, style }) {
  return (
    <span style={{ fontSize: 9, letterSpacing: 0.6, textTransform: "uppercase", color: COLORS.textMuted, ...style }}>
      {text}
    </span>
  );
}

// labeled, keyboard-navigable grid of editable floats
export default function MatrixEditor({
  matrix,
  onChange,
  titleTex,
  rowLabels,
  colLabels,
  rowAxis,
  colAxis,
  fills = [FILL_ZERO, FILL_IDENTITY],
  accent = COLORS.accent,
  allowInfinity = false,
}) {
  const cellRefs = useRef(new Map());
  const rows = matrix.length;
  const cols = rows === 0 ? 0 : matrix[0].length;

  // move focus by a (dRow, dCol) step, clamped to the grid
  function focusCell(row, col) {
    const target = cellRefs.current.get(`${row}:${col}`);
    if (target) {
      target.focus();
      target.select();
    }
  }

  function setCell(row, col, value) {
    const next = cloneMatrix(matrix);
    next[row][col] = value;
    onChange(next);
  }

  if (rows === 0 || cols === 0) {
    return (
      <div style={{ color: COLORS.textMuted, fontSize: 11, padding: "6px 0" }}>
        <MathText tex={titleTex} /> <span style={{ marginLeft: 6 }}>is empty at this size</span>
      </div>
    );
  }

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
        <MathText tex={titleTex} style={{ color: accent, fontSize: 13 }} />
        <span style={{ fontSize: 10, color: COLORS.textMuted, fontFamily: MONO }}>
          {rows} × {cols}
        </span>
        <div style={{ display: "flex", gap: 4, marginLeft: "auto" }}>
          {fills.map((fill) => (
            <button
              key={fill.label}
              type="button"
              title={fill.title}
              onClick={() => onChange(fill.build(rows, cols))}
              style={smallBtnStyle}
            >
              {fill.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "flex-start", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
        {rowAxis ? (
          <AxisLabel
            text={rowAxis}
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", marginTop: 22, alignSelf: "center" }}
          />
        ) : null}
        <div>
          {colAxis ? <AxisLabel text={colAxis} style={{ display: "block", marginBottom: 3, marginLeft: 34 }} /> : null}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `auto repeat(${cols}, ${CELL_WIDTH}px)`,
              gap: 3,
              alignItems: "center",
            }}
          >
            <div />
            {colLabels.map((label, j) => (
              <div
                key={`col-${j}`}
                style={{ fontSize: 10, color: COLORS.textMuted, fontFamily: MONO, textAlign: "center" }}
              >
                {label}
              </div>
            ))}
            {matrix.map((row, i) => (
              <Row
                key={`row-${i}`}
                row={row}
                i={i}
                rows={rows}
                cols={cols}
                label={rowLabels[i]}
                cellRefs={cellRefs}
                setCell={setCell}
                focusCell={focusCell}
                allowInfinity={allowInfinity}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// one matrix row: left label plus its editable cells
function Row({ row, i, rows, cols, label, cellRefs, setCell, focusCell, allowInfinity }) {
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
        }}
      >
        {label}
      </div>
      {row.map((value, j) => (
        <NumberCell
          key={`cell-${i}-${j}`}
          value={value}
          allowInfinity={allowInfinity}
          inputRef={(element) => {
            if (element) cellRefs.current.set(`${i}:${j}`, element);
            else cellRefs.current.delete(`${i}:${j}`);
          }}
          onCommit={(next) => setCell(i, j, next)}
          onNav={(dRow, dCol) => {
            const nextRow = Math.min(rows - 1, Math.max(0, i + dRow));
            const nextCol = Math.min(cols - 1, Math.max(0, j + dCol));
            focusCell(nextRow, nextCol);
          }}
        />
      ))}
    </>
  );
}
