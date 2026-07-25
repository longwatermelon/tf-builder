import { FILL_CAUSAL, FILL_IDENTITY, FILL_ZERO } from "../lib/matrixFills";
import { COLORS, MODULE_COLORS, MONO, smallBtnStyle } from "../styles/theme";
import MatrixEditor from "./MatrixEditor";
import MathText from "./MathText";

// dim labels shared by every stream-facing axis
function dimLabels(n, prefix) {
  return Array.from({ length: n }, (_, i) => `${prefix}${i}`);
}

// +/- control for a module's width knob
function Stepper({ label, value, min, max, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 11, color: COLORS.textMuted }}>{label}</span>
      <button type="button" style={smallBtnStyle} disabled={value <= min} onClick={() => onChange(value - 1)}>
        −
      </button>
      <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textBright, minWidth: 16, textAlign: "center" }}>
        {value}
      </span>
      <button type="button" style={smallBtnStyle} disabled={value >= max} onClick={() => onChange(value + 1)}>
        +
      </button>
    </div>
  );
}

// on/off switch for optional parameter blocks (W_E, W_P, attention mask)
function Toggle({ label, checked, onChange }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: COLORS.text, cursor: "pointer" }}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <MathText tex={label} />
    </label>
  );
}

// section heading with the module's defining equation in latex
function Equation({ tex }) {
  return (
    <div
      style={{
        background: "rgba(0,0,0,0.25)",
        border: `1px solid ${COLORS.panelBorder}`,
        borderRadius: 4,
        padding: "8px 10px",
        marginBottom: 12,
        color: COLORS.text,
        overflowX: "auto",
      }}
    >
      <MathText tex={tex} displayMode />
    </div>
  );
}

export default function ModuleInspector({ module, dIn, dModel, puzzle, onChange, onChangeDModel }) {
  const accent = MODULE_COLORS[module.type];
  const vocab = puzzle.vocab;
  const maxLen = puzzle.maxLen;

  if (module.type === "embed") {
    const parts = [module.useE ? "W_E[t_i]" : null, module.useP ? "W_P[i]" : null].filter(Boolean);
    return (
      <div>
        <Equation tex={`x_i = ${parts.length ? parts.join(" + ") : "0"}`} />
        <div style={{ display: "flex", gap: 18, alignItems: "center", marginBottom: 14, flexWrap: "wrap" }}>
          <Stepper label="d_model" value={dModel} min={1} max={16} onChange={onChangeDModel} />
          <Toggle label="W_E" checked={module.useE} onChange={(useE) => onChange({ ...module, useE })} />
          <Toggle label="W_P" checked={module.useP} onChange={(useP) => onChange({ ...module, useP })} />
        </div>
        {module.useE ? (
          <MatrixEditor
            titleTex="W_E"
            matrix={module.W_E}
            onChange={(W_E) => onChange({ ...module, W_E })}
            rowLabels={vocab}
            colLabels={dimLabels(dModel, "d")}
            rowAxis="token"
            colAxis="residual dim"
            accent={accent}
          />
        ) : null}
        {module.useP ? (
          <MatrixEditor
            titleTex="W_P"
            matrix={module.W_P}
            onChange={(W_P) => onChange({ ...module, W_P })}
            rowLabels={dimLabels(maxLen, "p")}
            colLabels={dimLabels(dModel, "d")}
            rowAxis="position"
            colAxis="residual dim"
            accent={accent}
          />
        ) : null}
      </div>
    );
  }

  if (module.type === "attn") {
    const maskTerm = module.useMask ? " + M" : "";
    return (
      <div>
        <Equation
          tex={`Q = XW_Q,\\quad K = XW_K,\\quad V = XW_V \\\\[4pt] A = \\mathrm{softmax}\\!\\left(QK^{\\top}${maskTerm}\\right) \\\\[4pt] X \\mathrel{+}= (AV)\\,W_O`}
        />
        <div style={{ display: "flex", gap: 18, alignItems: "center", marginBottom: 14, flexWrap: "wrap" }}>
          <Stepper label="d_head" value={module.dHead} min={1} max={16} onChange={(dHead) => onChange({ ...module, dHead })} />
          <Toggle label="M" checked={module.useMask} onChange={(useMask) => onChange({ ...module, useMask })} />
          <span style={{ fontSize: 10, color: COLORS.textMuted }}>
            scores are raw QKᵀ — no 1/√d scaling{module.useMask ? "; type inf / -inf in M" : ""}
          </span>
        </div>
        {["W_Q", "W_K", "W_V"].map((name) => (
          <MatrixEditor
            key={name}
            titleTex={name}
            matrix={module[name]}
            onChange={(next) => onChange({ ...module, [name]: next })}
            rowLabels={dimLabels(dIn, "d")}
            colLabels={dimLabels(module.dHead, "h")}
            rowAxis="residual dim"
            colAxis="head dim"
            accent={accent}
          />
        ))}
        <MatrixEditor
          titleTex="W_O"
          matrix={module.W_O}
          onChange={(W_O) => onChange({ ...module, W_O })}
          rowLabels={dimLabels(module.dHead, "h")}
          colLabels={dimLabels(dIn, "d")}
          rowAxis="head dim"
          colAxis="residual dim"
          accent={accent}
        />
        {module.useMask ? (
          <MatrixEditor
            titleTex="M"
            matrix={module.mask}
            onChange={(mask) => onChange({ ...module, mask })}
            rowLabels={dimLabels(maxLen, "q")}
            colLabels={dimLabels(maxLen, "k")}
            rowAxis="query pos"
            colAxis="key pos"
            fills={[FILL_ZERO, FILL_CAUSAL]}
            accent={accent}
            allowInfinity
          />
        ) : null}
      </div>
    );
  }

  if (module.type === "mlp") {
    return (
      <div>
        <Equation tex={"H = \\mathrm{ReLU}(XW_1 + b_1) \\\\[4pt] X \\mathrel{+}= HW_2 + b_2"} />
        <div style={{ marginBottom: 14 }}>
          <Stepper
            label="d_hidden"
            value={module.dHidden}
            min={1}
            max={16}
            onChange={(dHidden) => onChange({ ...module, dHidden })}
          />
        </div>
        <MatrixEditor
          titleTex="W_1"
          matrix={module.W1}
          onChange={(W1) => onChange({ ...module, W1 })}
          rowLabels={dimLabels(dIn, "d")}
          colLabels={dimLabels(module.dHidden, "h")}
          rowAxis="residual dim"
          colAxis="hidden unit"
          accent={accent}
        />
        <MatrixEditor
          titleTex="b_1"
          matrix={[module.b1]}
          onChange={(next) => onChange({ ...module, b1: next[0] })}
          rowLabels={[""]}
          colLabels={dimLabels(module.dHidden, "h")}
          colAxis="hidden unit"
          fills={[FILL_ZERO]}
          accent={accent}
        />
        <MatrixEditor
          titleTex="W_2"
          matrix={module.W2}
          onChange={(W2) => onChange({ ...module, W2 })}
          rowLabels={dimLabels(module.dHidden, "h")}
          colLabels={dimLabels(dIn, "d")}
          rowAxis="hidden unit"
          colAxis="residual dim"
          accent={accent}
        />
        <MatrixEditor
          titleTex="b_2"
          matrix={[module.b2]}
          onChange={(next) => onChange({ ...module, b2: next[0] })}
          rowLabels={[""]}
          colLabels={dimLabels(dIn, "d")}
          colAxis="residual dim"
          fills={[FILL_ZERO]}
          accent={accent}
        />
      </div>
    );
  }

  // linear is the only module that resizes the stream, so its output may be the logit axis
  const isLogitLayer = module.dOut === vocab.length;
  const outLabels = isLogitLayer ? vocab : dimLabels(module.dOut, "u");
  return (
    <div>
      <Equation tex={"X \\leftarrow XW + b"} />
      <div style={{ display: "flex", gap: 18, alignItems: "center", marginBottom: 14, flexWrap: "wrap" }}>
        <Stepper label="d_out" value={module.dOut} min={1} max={16} onChange={(dOut) => onChange({ ...module, dOut })} />
        {isLogitLayer ? (
          <span style={{ fontSize: 10, color: COLORS.success }}>width matches the vocabulary — usable as unembedding</span>
        ) : null}
      </div>
      <MatrixEditor
        titleTex="W"
        matrix={module.W}
        onChange={(W) => onChange({ ...module, W })}
        rowLabels={dimLabels(dIn, "d")}
        colLabels={outLabels}
        rowAxis="input dim"
        colAxis={isLogitLayer ? "token" : "output dim"}
        fills={[FILL_ZERO, FILL_IDENTITY]}
        accent={accent}
      />
      <MatrixEditor
        titleTex="b"
        matrix={[module.b]}
        onChange={(next) => onChange({ ...module, b: next[0] })}
        rowLabels={[""]}
        colLabels={outLabels}
        colAxis={isLogitLayer ? "token" : "output dim"}
        fills={[FILL_ZERO]}
        accent={accent}
      />
    </div>
  );
}
