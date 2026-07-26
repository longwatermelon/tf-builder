import { useRef, useState } from "react";
import { numberToText, parseNumberText } from "../lib/format";
import { COLORS, MONO } from "../styles/theme";

const VALUE_SHORTCUTS = {
  a: ".",
  q: "-",
  w: "0",
  e: "1000",
  r: "inf",
  s: "inf",
  d: "-inf",
};

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
        const input = event.currentTarget;
        // a fully selected cell counts as both edges so the first arrow press already moves
        const allSelected = input.selectionStart === 0 && input.selectionEnd === input.value.length;
        const shortcut = !event.metaKey && !event.ctrlKey && !event.altKey
          ? VALUE_SHORTCUTS[event.key.toLowerCase()]
          : null;
        if (shortcut) {
          event.preventDefault();
          const selectionStart = input.selectionStart ?? input.value.length;
          const selectionEnd = input.selectionEnd ?? selectionStart;
          const next = `${input.value.slice(0, selectionStart)}${shortcut}${input.value.slice(selectionEnd)}`;
          const nextCursor = selectionStart + shortcut.length;
          setDraft(next);
          const parsed = parseNumberText(next, allowInfinity);
          setInvalid(parsed === null);
          if (parsed !== null) onCommit(parsed);
          requestAnimationFrame(() => input.setSelectionRange(nextCursor, nextCursor));
          return;
        }
        if (!onNav) return;
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
