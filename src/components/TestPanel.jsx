import { useState } from "react";
import { formatProbability } from "../lib/format";
import { forward, streamColLabels } from "../lib/model";
import { COLORS, MODULE_COLORS, MONO, smallBtnStyle, subtleBtnStyle } from "../styles/theme";
import ValueGrid from "./ValueGrid";

// "0:a" style position labels so rows are unambiguous
function positionLabels(tokens) {
  return tokens.map((token, i) => `${i}:${token}`);
}

// tab strip across the fixed tests plus the free scratch sequence
function Tabs({ count, active, onSelect }) {
  const items = [...Array.from({ length: count }, (_, i) => ({ key: i, label: `Test ${i + 1}` })), { key: "scratch", label: "Scratch" }];
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 10 }}>
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          onClick={() => onSelect(item.key)}
          style={{
            ...subtleBtnStyle,
            color: active === item.key ? COLORS.textBright : COLORS.textMuted,
            borderColor: active === item.key ? COLORS.accent : COLORS.panelBorder,
            background: active === item.key ? COLORS.accentDim : "transparent",
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

// token / target / prediction alignment table for one sequence
function SequenceTable({ tokens, targets, grades, vocab }) {
  const cell = {
    fontFamily: MONO,
    fontSize: 11,
    textAlign: "center",
    padding: "3px 6px",
    borderRadius: 3,
    background: "rgba(30,30,30,0.55)",
    border: `1px solid ${COLORS.panelBorder}`,
  };
  const label = { fontSize: 10, color: COLORS.textMuted, textAlign: "right", paddingRight: 6, lineHeight: "23px" };
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `auto repeat(${tokens.length}, minmax(48px, 1fr))`,
        gap: 3,
        marginBottom: 12,
      }}
    >
      <div style={label}>position</div>
      {tokens.map((_, i) => (
        <div key={`p-${i}`} style={{ ...cell, background: "transparent", border: "none", color: COLORS.textMuted }}>
          {i}
        </div>
      ))}
      <div style={label}>input</div>
      {tokens.map((token, i) => (
        <div key={`t-${i}`} style={{ ...cell, color: COLORS.textBright }}>
          {token}
        </div>
      ))}
      {targets ? (
        <>
          <div style={label}>target</div>
          {targets.map((target, i) => (
            <div key={`g-${i}`} style={{ ...cell, color: COLORS.warn }}>
              {target ?? "·"}
            </div>
          ))}
        </>
      ) : null}
      <div style={label}>predicted</div>
      {tokens.map((_, i) => {
        const grade = grades?.[i];
        // scratch sequences have no targets, so they get neutral styling
        const verdictColor = !grade || grade.ok === null ? null : grade.ok ? COLORS.success : COLORS.negative;
        return (
          <div
            key={`o-${i}`}
            style={{
              ...cell,
              color: verdictColor ?? COLORS.text,
              borderColor: verdictColor ?? COLORS.panelBorder,
            }}
          >
            {grade ? vocab[grade.topId] : "—"}
          </div>
        );
      })}
    </div>
  );
}

export default function TestPanel({
  puzzle,
  model,
  evaluation,
  scratchTokens,
  onChangeScratch,
  activeTab,
  onSelectTab,
}) {
  const [showInternals, setShowInternals] = useState(false);
  const isScratch = activeTab === "scratch";
  const vocab = puzzle.vocab;

  const tokens = isScratch ? scratchTokens : evaluation.results[activeTab]?.test.tokens ?? [];
  const targets = isScratch ? null : evaluation.results[activeTab]?.test.targets ?? null;
  const grades = isScratch ? null : evaluation.results[activeTab]?.grades ?? null;
  const pass = isScratch
    ? forward(model, puzzle, tokens.map((t) => vocab.indexOf(t)))
    : evaluation.results[activeTab]?.pass;

  const probs = pass?.probs ?? null;
  const scratchGrades = isScratch && probs ? probs.map((row) => ({ ok: null, topId: row.indexOf(Math.max(...row)) })) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ padding: "10px 12px 6px", fontSize: 11, letterSpacing: 0.8, textTransform: "uppercase", color: COLORS.textMuted }}>
        Forward Pass
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: "0 12px 12px" }}>
        <Tabs count={puzzle.tests.length} active={activeTab} onSelect={onSelectTab} />

        {isScratch ? (
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
            <span style={{ fontSize: 10, color: COLORS.textMuted }}>click a token to cycle it</span>
            <div style={{ display: "flex", gap: 4, marginLeft: "auto" }}>
              <button
                type="button"
                style={smallBtnStyle}
                disabled={scratchTokens.length <= 1}
                onClick={() => onChangeScratch(scratchTokens.slice(0, -1))}
              >
                −
              </button>
              <button
                type="button"
                style={smallBtnStyle}
                disabled={scratchTokens.length >= puzzle.maxLen}
                onClick={() => onChangeScratch([...scratchTokens, vocab[0]])}
              >
                +
              </button>
            </div>
          </div>
        ) : null}

        {isScratch ? (
          <div style={{ display: "flex", gap: 4, marginBottom: 12, flexWrap: "wrap" }}>
            {scratchTokens.map((token, i) => (
              <button
                key={`s-${i}`}
                type="button"
                onClick={() => {
                  const next = [...scratchTokens];
                  next[i] = vocab[(vocab.indexOf(token) + 1) % vocab.length];
                  onChangeScratch(next);
                }}
                style={{ ...subtleBtnStyle, fontFamily: MONO, color: COLORS.textBright, minWidth: 34 }}
              >
                {token}
              </button>
            ))}
          </div>
        ) : null}

        {pass?.error ? (
          <div
            style={{
              border: `1px solid ${COLORS.negative}`,
              background: COLORS.negativeDim,
              borderRadius: 4,
              padding: "8px 10px",
              fontSize: 11,
              color: COLORS.text,
            }}
          >
            {pass.error}
          </div>
        ) : (
          <>
            <SequenceTable tokens={tokens} targets={targets} grades={grades ?? scratchGrades} vocab={vocab} />

            <div style={{ fontSize: 10, letterSpacing: 0.6, textTransform: "uppercase", color: COLORS.textMuted, marginBottom: 5 }}>
              Output distribution
            </div>
            {probs ? (
              <ValueGrid
                matrix={probs}
                rowLabels={positionLabels(tokens)}
                colLabels={vocab}
                rowAxis="position"
                colAxis="next token"
                format={formatProbability}
                cellStyleAt={(i, j) => {
                  const targetId = targets ? vocab.indexOf(targets[i]) : -1;
                  if (j === targetId) {
                    const ok = grades?.[i]?.ok;
                    return { borderColor: ok ? COLORS.success : COLORS.negative, color: COLORS.textBright };
                  }
                  return null;
                }}
              />
            ) : null}

            <button
              type="button"
              style={{ ...subtleBtnStyle, marginTop: 12 }}
              onClick={() => setShowInternals((prev) => !prev)}
            >
              {showInternals ? "hide" : "show"} intermediate values
            </button>

            {showInternals && pass
              ? pass.stages.map((stage, index) => {
                  const accent = MODULE_COLORS[stage.type];
                  const isLogits = index === pass.stages.length - 1 && stage.width === vocab.length;
                  return (
                    <div key={stage.moduleId} style={{ marginTop: 14 }}>
                      <div style={{ fontSize: 10, color: accent, marginBottom: 4, fontWeight: 600 }}>
                        after {index + 1}. {stage.type}
                      </div>
                      {stage.extras.map((extra) => {
                        const isPattern = extra.kind === "pattern";
                        const hiddenWidth = extra.matrix[0]?.length ?? 0;
                        return (
                          <div key={extra.key} style={{ marginBottom: 8 }}>
                            <div style={{ fontSize: 9, color: COLORS.textMuted, marginBottom: 3 }}>{extra.label}</div>
                            <ValueGrid
                              matrix={extra.matrix}
                              rowLabels={positionLabels(tokens)}
                              colLabels={
                                isPattern
                                  ? positionLabels(tokens)
                                  : Array.from({ length: hiddenWidth }, (_, h) => `h${h}`)
                              }
                              rowAxis={isPattern ? "query" : "position"}
                              colAxis={isPattern ? "key" : "hidden unit"}
                            />
                          </div>
                        );
                      })}
                      <ValueGrid
                        matrix={stage.matrix}
                        rowLabels={positionLabels(tokens)}
                        colLabels={streamColLabels(stage.width, puzzle, isLogits)}
                        rowAxis="position"
                        colAxis={isLogits ? "logit" : "residual dim"}
                      />
                    </div>
                  );
                })
              : null}
          </>
        )}
      </div>
    </div>
  );
}
