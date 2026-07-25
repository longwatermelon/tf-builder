import { useRef, useState } from "react";
import { numberToText, parseNumberText } from "../lib/format";
import { COLORS, MONO } from "../styles/theme";

// one hand-editable float; keeps a local draft so partial input like "-" survives typing
export default function NumberCell({
  value,
  onCommit,
  onNav,
  width = 62,
  inputRef,
  disabled = false,
  allowInfinity = false,
}) {
  const [draft, setDraft] = useState(null);
  const [invalid, setInvalid] = useState(false);
  const localRef = useRef(null);
  const ref = inputRef ?? localRef;
  const text = draft === null ? numberToText(value) : draft;

  return (
    <input
      ref={ref}
      type="text"
      spellCheck={false}
      disabled={disabled}
      value={text}
      onFocus={(event) => {
        setDraft(numberToText(value));
        event.currentTarget.select();
      }}
      onMouseUp={(event) => {
        // clicking a cell always highlights its contents so typing replaces the value outright
        event.preventDefault();
        event.currentTarget.select();
      }}
      onChange={(event) => {
        const next = event.target.value;
        setDraft(next);
        const parsed = parseNumberText(next, allowInfinity);
        if (parsed === null) {
          setInvalid(true);
          return;
        }
        setInvalid(false);
        onCommit(parsed);
      }}
      onBlur={() => {
        setDraft(null);
        setInvalid(false);
      }}
      onKeyDown={(event) => {
        if (!onNav) return;
        const input = event.currentTarget;
        // a fully selected cell counts as both edges so the first arrow press already moves
        const allSelected = input.selectionStart === 0 && input.selectionEnd === input.value.length;
        const atStart = allSelected || (input.selectionStart === 0 && input.selectionEnd === 0);
        const atEnd =
          allSelected
          || (input.selectionStart === input.value.length && input.selectionEnd === input.value.length);
        if (event.key === "ArrowUp") {
          event.preventDefault();
          onNav(-1, 0);
        } else if (event.key === "ArrowDown" || event.key === "Enter") {
          event.preventDefault();
          onNav(1, 0);
        } else if (event.key === "ArrowLeft" && atStart) {
          event.preventDefault();
          onNav(0, -1);
        } else if (event.key === "ArrowRight" && atEnd) {
          event.preventDefault();
          onNav(0, 1);
        } else if (event.key === "Escape") {
          input.blur();
        }
      }}
      style={{
        width,
        background: value === 0 ? "rgba(30,30,30,0.55)" : "rgba(0,152,255,0.10)",
        border: `1px solid ${invalid ? COLORS.negative : COLORS.panelBorder}`,
        borderRadius: 3,
        padding: "3px 5px",
        color: value === 0 ? COLORS.textMuted : COLORS.textBright,
        fontFamily: MONO,
        fontSize: 11,
        textAlign: "center",
        outline: "none",
      }}
    />
  );
}
