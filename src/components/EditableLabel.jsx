import { useRef, useState } from "react";
import { MAX_LABEL_LEN } from "../lib/axisLabels";
import { COLORS, MONO } from "../styles/theme";

const INPUT_WIDTH = 46;

// one row/column label. with onRename it becomes click-to-edit, so the player can name what a
// dimension means; committing an empty name restores the generated default
export default function EditableLabel({ name, fallback, isNamed = false, align = "center", onRename, style }) {
  const [draft, setDraft] = useState(null);
  // remembers that an edit was already committed or cancelled, so the blur that follows is a no-op
  const handledRef = useRef(false);

  const base = {
    fontSize: 10,
    fontFamily: MONO,
    color: isNamed ? COLORS.textBright : COLORS.textMuted,
    textAlign: align,
    ...style,
  };

  if (!onRename) return <div style={base}>{name}</div>;

  function open() {
    handledRef.current = false;
    setDraft(isNamed ? name : "");
  }

  function finish(save) {
    if (save) onRename(draft);
    handledRef.current = true;
    setDraft(null);
  }

  if (draft !== null) {
    return (
      <input
        autoFocus
        value={draft}
        maxLength={MAX_LABEL_LEN}
        placeholder={fallback}
        onChange={(event) => setDraft(event.target.value)}
        onFocus={(event) => event.target.select()}
        onBlur={() => {
          if (!handledRef.current) finish(true);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") finish(true);
          else if (event.key === "Escape") finish(false);
        }}
        style={{
          ...base,
          width: INPUT_WIDTH,
          margin: align === "center" ? "0 auto" : "0 0 0 auto",
          display: "block",
          background: COLORS.bg,
          border: `1px solid ${COLORS.accent}`,
          borderRadius: 3,
          color: COLORS.textBright,
          padding: "1px 3px",
          outline: "none",
        }}
      />
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      title={isNamed ? `${fallback} — click to rename` : `click to name ${fallback}`}
      onClick={open}
      onKeyDown={(event) => {
        // both keys that activate a button, so space opens the editor instead of scrolling
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          open();
        }
      }}
      style={{
        ...base,
        cursor: "pointer",
        borderBottom: `1px dashed ${isNamed ? COLORS.accent : "transparent"}`,
      }}
    >
      {name}
    </div>
  );
}
