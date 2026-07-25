import { useState } from "react";
import {
  defaultLabels,
  moduleAxis,
  POSITION_AXIS,
  resolveLabels,
  streamDefaults,
  VOCAB_AXIS,
} from "../lib/axisLabels";
import { formatProbability } from "../lib/format";
import { forward } from "../lib/model";
import { COLORS, MODULE_COLORS, MONO, probabilityFill, smallBtnStyle, subtleBtnStyle } from "../styles/theme";
import ValueGrid from "./ValueGrid";

// "0:a" style position labels so rows are unambiguous, carrying any name given to the position
function positionLabels(tokens, labels) {
  const names = resolveLabels(
    labels,
    POSITION_AXIS,
    tokens.map((_, i) => String(i)),
  );
  return tokens.map((token, i) => `${names[i]}:${token}`);
}

// token / required / produced alignment table for one sequence
function SequenceTable({ tokens, targets, grades, vocab }) {
  const cell = {
    fontFamily: MONO,
    fontSize: 11,
    textAlign: "center",
    padding: "3px 6px",
    borderRadius: 3,
    background: "rgba(30,30,30,0.55)",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: COLORS.panelBorder,
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
        <div key={`p-${i}`} style={{ ...cell, background: "transparent", borderWidth: 0, color: COLORS.textMuted }}>
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
          <div style={label}>required</div>
          {targets.map((target, i) => (
            <div key={`g-${i}`} style={{ ...cell, color: COLORS.warn }}>
              {target ?? "·"}
            </div>
          ))}
        </>
      ) : null}
      <div style={label}>produced</div>
      {tokens.map((_, i) => {
        const grade = grades?.[i];
        // scratch sequences have no required output, so they get neutral styling
        const verdictColor = !grade || grade.ok === null ? null : grade.ok ? COLORS.success : COLORS.negative;
        return (
          <div
            key={`o-${i}`}
            style={{ ...cell, color: verdictColor ?? COLORS.text, borderColor: verdictColor ?? COLORS.panelBorder }}
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
  streamAxes,
  labels,
}) {
  const [showInternals, setShowInternals] = useState(false);
  const isScratch = activeTab === "scratch";
  const vocab = puzzle.vocab;
  // computed values carry the same dimension names the weight editors show
  const vocabLabels = resolveLabels(labels, VOCAB_AXIS, vocab);
  // some puzzles keep label-only tokens in the vocabulary, so scratch cycles the input alphabet
  const inputVocab = puzzle.validationVocab ?? puzzle.inputVocab ?? vocab;
  const fixedPrefix = puzzle.validationPrefix ?? [];

  const tokens = isScratch ? scratchTokens : evaluation.results[activeTab]?.test.tokens ?? [];
  const targets = isScratch ? null : evaluation.results[activeTab]?.test.targets ?? null;
  const grades = isScratch ? null : evaluation.results[activeTab]?.grades ?? null;
  const pass = isScratch
    ? forward(model, puzzle, tokens.map((t) => vocab.indexOf(t)))
    : evaluation.results[activeTab]?.pass;

  const posLabels = positionLabels(tokens, labels);
  const probs = pass?.probs ?? null;
  const scratchGrades =
    isScratch && probs ? probs.map((row) => ({ ok: null, topId: row.indexOf(Math.max(...row)) })) : null;

  return (
    <div style={{ padding: "0 12px 12px", borderTop: `1px solid ${COLORS.panelBorder}`, paddingTop: 12 }}>
      <div style={{ fontSize: 10, letterSpacing: 0.6, textTransform: "uppercase", color: COLORS.textMuted, marginBottom: 8 }}>
        {isScratch ? "Scratch sequence" : `Sample ${activeTab + 1}`}
      </div>

      {isScratch ? (
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 10, color: COLORS.textMuted }}>click a token to cycle it</span>
          <div style={{ display: "flex", gap: 4, marginLeft: "auto" }}>
            <button
              type="button"
              style={smallBtnStyle}
              disabled={puzzle.fixedLen || scratchTokens.length <= 1}
              onClick={() => onChangeScratch(scratchTokens.slice(0, -1))}
            >
              −
            </button>
            <button
              type="button"
              style={smallBtnStyle}
              disabled={puzzle.fixedLen || scratchTokens.length >= puzzle.maxLen}
              onClick={() => onChangeScratch([...scratchTokens, inputVocab[0]])}
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
              disabled={i < fixedPrefix.length}
              type="button"
              onClick={() => {
                const next = [...scratchTokens];
                const choices = i < fixedPrefix.length ? [fixedPrefix[i]] : inputVocab;
                const at = choices.indexOf(token);
                next[i] = choices[(at + 1) % choices.length];
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
              rowLabels={posLabels}
              colLabels={vocabLabels}
              rowAxis="position"
              colAxis="output token"
              format={formatProbability}
              cellStyleAt={(i, j, value) => {
                // fill intensity tracks the probability mass, so the winning token is visible at a glance
                const heat = { background: probabilityFill(value), color: value >= 0.5 ? COLORS.textBright : COLORS.text };
                const targetId = targets ? vocab.indexOf(targets[i]) : -1;
                if (j === targetId) {
                  const ok = grades?.[i]?.ok;
                  return { ...heat, borderColor: ok ? COLORS.success : COLORS.negative, color: COLORS.textBright };
                }
                return heat;
              }}
            />
          ) : null}

          <button type="button" style={{ ...subtleBtnStyle, marginTop: 12 }} onClick={() => setShowInternals((prev) => !prev)}>
            {showInternals ? "hide" : "show"} intermediate values
          </button>

          {showInternals && pass
            ? pass.stages.map((stage, index) => {
                const accent = MODULE_COLORS[stage.type];
                const isLogits = index === pass.stages.length - 1 && stage.width === vocab.length;
                // names come from the axis the stream is actually on, but the final stage is read as
                // logits, so it is spelled with the vocabulary even when no linear moved it there
                const streamKey = streamAxes.outputs[index];
                const streamNames = isLogits ? vocab : streamDefaults(streamKey, stage.width, puzzle);
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
                            rowLabels={posLabels}
                            colLabels={
                              isPattern
                                ? posLabels
                                : resolveLabels(
                                    labels,
                                    moduleAxis(stage.moduleId, "h"),
                                    defaultLabels(hiddenWidth, "h"),
                                  )
                            }
                            rowAxis={isPattern ? "query" : "position"}
                            colAxis={isPattern ? "key" : "hidden unit"}
                          />
                        </div>
                      );
                    })}
                    <ValueGrid
                      matrix={stage.matrix}
                      rowLabels={posLabels}
                      colLabels={resolveLabels(labels, streamKey, streamNames)}
                      rowAxis="position"
                      colAxis={isLogits ? "token" : "residual dim"}
                    />
                  </div>
                );
              })
            : null}
        </>
      )}
    </div>
  );
}
