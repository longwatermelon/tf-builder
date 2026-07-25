import { useState } from "react";
import { MODULE_KINDS, moduleOutputWidth, moduleParamCount } from "../lib/model";
import { COLORS, MODULE_COLORS, MONO, smallBtnStyle, subtleBtnStyle } from "../styles/theme";

const ADDABLE_TYPES = ["attn", "mlp", "linear"];

// short description of the knob that defines each module's shape
function moduleSubtitle(module) {
  if (module.type === "embed") {
    return module.useE && module.useP ? "W_E + W_P" : module.useE ? "W_E" : module.useP ? "W_P" : "off";
  }
  if (module.type === "attn") return `d_head ${module.dHead}${module.useMask ? " · masked" : ""}`;
  if (module.type === "mlp") return `d_hidden ${module.dHidden}`;
  return `d_out ${module.dOut}`;
}

// connector under a module: shows the width it emits and offers an insert point below it
function Connector({ width, bad, insertIndex, isOpen, onToggle, onAdd }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0 3px 10px", minHeight: 22 }}>
        <span style={{ color: COLORS.panelBorder, fontSize: 12, lineHeight: 1 }}>│</span>
        <span style={{ fontFamily: MONO, fontSize: 10, color: bad ? COLORS.negative : COLORS.textMuted }}>
          width {width}
        </span>
        <button
          type="button"
          title="insert a module here"
          onClick={onToggle}
          style={{
            ...smallBtnStyle,
            marginLeft: "auto",
            color: isOpen ? COLORS.accent : COLORS.textMuted,
            borderColor: isOpen ? COLORS.accent : COLORS.panelBorder,
          }}
        >
          +
        </button>
      </div>
      {isOpen ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "2px 0 6px 18px" }}>
          {ADDABLE_TYPES.map((type) => (
            <button
              key={type}
              type="button"
              onClick={() => onAdd(type, insertIndex)}
              style={{ ...subtleBtnStyle, textAlign: "left", borderColor: MODULE_COLORS[type], color: COLORS.text }}
            >
              + {MODULE_KINDS[type].label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// the stack always ends in a softmax; it holds no weights, so it is drawn rather than modelled
function SoftmaxCard({ inputWidth, expected }) {
  const bad = inputWidth !== expected;
  return (
    <div
      style={{
        border: `1px solid ${bad ? COLORS.negative : COLORS.panelBorder}`,
        background: COLORS.surface,
        borderRadius: 5,
        padding: "7px 9px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span style={{ width: 7, height: 7, borderRadius: 2, background: MODULE_COLORS.softmax, flexShrink: 0 }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.textBright }}>Softmax</span>
        <span style={{ fontFamily: MONO, fontSize: 10, color: COLORS.textMuted, marginLeft: "auto" }}>0p</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
        <span style={{ fontFamily: MONO, fontSize: 10, color: bad ? COLORS.negative : COLORS.textMuted }}>
          {bad ? `needs width ${expected}` : `over ${expected} tokens`}
        </span>
        <span style={{ fontSize: 9, color: COLORS.textMuted, marginLeft: "auto" }}>always last</span>
      </div>
    </div>
  );
}

export default function ModuleStack({ model, puzzle, inputWidths, selectedId, onSelect, onAdd, onRemove, onMove }) {
  // which connector has its insert menu expanded, keyed by the module above it so that
  // reordering or deleting modules cannot silently move the menu to a different slot
  const [openInsertAfterId, setOpenInsertAfterId] = useState(null);
  const moduleCount = model.modules.length;
  const finalWidth = moduleCount
    ? moduleOutputWidth(model.modules[moduleCount - 1], inputWidths[moduleCount - 1])
    : model.dModel;

  function handleAdd(type, index) {
    setOpenInsertAfterId(null);
    onAdd(type, index);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div
        style={{
          padding: "10px 12px 6px",
          fontSize: 11,
          letterSpacing: 0.8,
          textTransform: "uppercase",
          color: COLORS.textMuted,
        }}
      >
        Architecture
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "0 12px 12px" }}>
        <div style={{ fontFamily: MONO, fontSize: 10, color: COLORS.textMuted, paddingLeft: 10, paddingBottom: 4 }}>
          token sequence
        </div>

        {model.modules.map((module, i) => {
          const isSelected = module.id === selectedId;
          const accent = MODULE_COLORS[module.type];
          const params = moduleParamCount(module, inputWidths[i], puzzle);
          const width = moduleOutputWidth(module, inputWidths[i]);
          const isLast = i === moduleCount - 1;
          return (
            <div key={module.id}>
              <div
                onClick={() => onSelect(module.id)}
                style={{
                  border: `1px solid ${isSelected ? accent : COLORS.panelBorder}`,
                  background: isSelected ? `${accent}18` : COLORS.surface,
                  borderRadius: 5,
                  padding: "7px 9px",
                  cursor: "pointer",
                  transition: "border-color 0.12s, background 0.12s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <span style={{ width: 7, height: 7, borderRadius: 2, background: accent, flexShrink: 0 }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.textBright }}>
                    {MODULE_KINDS[module.type].label}
                  </span>
                  <span style={{ fontFamily: MONO, fontSize: 10, color: COLORS.textMuted, marginLeft: "auto" }}>
                    {params}p
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
                  <span style={{ fontFamily: MONO, fontSize: 10, color: COLORS.textMuted }}>
                    {moduleSubtitle(module)}
                  </span>
                  {module.type === "embed" ? (
                    <span style={{ fontSize: 9, color: COLORS.textMuted, marginLeft: "auto" }}>always first</span>
                  ) : (
                    <div style={{ display: "flex", gap: 3, marginLeft: "auto" }} onClick={(event) => event.stopPropagation()}>
                      <button type="button" style={smallBtnStyle} title="move up" disabled={i <= 1} onClick={() => onMove(module.id, -1)}>
                        ↑
                      </button>
                      <button
                        type="button"
                        style={smallBtnStyle}
                        title="move down"
                        disabled={isLast}
                        onClick={() => onMove(module.id, 1)}
                      >
                        ↓
                      </button>
                      <button type="button" style={smallBtnStyle} title="delete this module" onClick={() => onRemove(module.id)}>
                        ×
                      </button>
                    </div>
                  )}
                </div>
              </div>
              <Connector
                width={width}
                bad={isLast && width !== puzzle.vocab.length}
                insertIndex={i + 1}
                isOpen={openInsertAfterId === module.id}
                onToggle={() => setOpenInsertAfterId((prev) => (prev === module.id ? null : module.id))}
                onAdd={handleAdd}
              />
            </div>
          );
        })}

        <SoftmaxCard inputWidth={finalWidth} expected={puzzle.vocab.length} />
      </div>
    </div>
  );
}
