import { MODULE_KINDS, moduleOutputWidth, moduleParamCount } from "../lib/model";
import { COLORS, MODULE_COLORS, MONO, smallBtnStyle, subtleBtnStyle } from "../styles/theme";

// short description of the knob that defines each module's shape
function moduleSubtitle(module) {
  if (module.type === "embed") return module.useE && module.useP ? "W_E + W_P" : module.useE ? "W_E" : module.useP ? "W_P" : "off";
  if (module.type === "attn") return `d_head ${module.dHead}${module.useMask ? " · masked" : ""}`;
  if (module.type === "mlp") return `d_hidden ${module.dHidden}`;
  return `d_out ${module.dOut}`;
}

// marker between cards showing the residual stream width at that point
function StreamWidth({ width, isFinal, expected }) {
  const bad = isFinal && width !== expected;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0 3px 10px" }}>
      <span style={{ color: COLORS.panelBorder, fontSize: 12, lineHeight: 1 }}>│</span>
      <span style={{ fontFamily: MONO, fontSize: 10, color: bad ? COLORS.negative : COLORS.textMuted }}>
        stream width {width}
        {isFinal ? (bad ? ` (needs ${expected})` : " → softmax") : ""}
      </span>
    </div>
  );
}

export default function ModuleStack({ model, puzzle, inputWidths, selectedId, onSelect, onAdd, onRemove, onMove }) {
  const lastWidth = model.modules.length
    ? moduleOutputWidth(model.modules[model.modules.length - 1], inputWidths[model.modules.length - 1])
    : model.dModel;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ padding: "10px 12px 6px", fontSize: 11, letterSpacing: 0.8, textTransform: "uppercase", color: COLORS.textMuted }}>
        Architecture
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "0 12px 8px" }}>
        <div style={{ fontFamily: MONO, fontSize: 10, color: COLORS.textMuted, paddingLeft: 10, paddingBottom: 2 }}>
          tokens
        </div>
        {model.modules.map((module, i) => {
          const isSelected = module.id === selectedId;
          const accent = MODULE_COLORS[module.type];
          const params = moduleParamCount(module, inputWidths[i], puzzle);
          const width = moduleOutputWidth(module, inputWidths[i]);
          return (
            <div key={module.id}>
              <StreamWidth
                width={i === 0 ? model.dModel : inputWidths[i]}
                isFinal={false}
              />
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
                  <span style={{ fontFamily: MONO, fontSize: 10, color: COLORS.textMuted }}>{moduleSubtitle(module)}</span>
                  {i > 0 ? (
                    <div style={{ display: "flex", gap: 3, marginLeft: "auto" }} onClick={(event) => event.stopPropagation()}>
                      <button type="button" style={smallBtnStyle} title="move up" disabled={i <= 1} onClick={() => onMove(module.id, -1)}>
                        ↑
                      </button>
                      <button
                        type="button"
                        style={smallBtnStyle}
                        title="move down"
                        disabled={i >= model.modules.length - 1}
                        onClick={() => onMove(module.id, 1)}
                      >
                        ↓
                      </button>
                      <button type="button" style={smallBtnStyle} title="remove" onClick={() => onRemove(module.id)}>
                        ×
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
              {i === model.modules.length - 1 ? (
                <StreamWidth width={width} isFinal expected={puzzle.vocab.length} />
              ) : null}
            </div>
          );
        })}
        <div style={{ fontFamily: MONO, fontSize: 10, color: lastWidth === puzzle.vocab.length ? COLORS.success : COLORS.negative, paddingLeft: 10 }}>
          softmax → next-token distribution
        </div>
      </div>

      <div style={{ borderTop: `1px solid ${COLORS.panelBorder}`, padding: "8px 12px", display: "flex", gap: 6, flexWrap: "wrap" }}>
        {["attn", "mlp", "linear"].map((type) => (
          <button key={type} type="button" style={subtleBtnStyle} onClick={() => onAdd(type)}>
            + {MODULE_KINDS[type].label}
          </button>
        ))}
      </div>
    </div>
  );
}
