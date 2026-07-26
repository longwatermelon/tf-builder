import { useEffect, useId, useRef, useState } from "react";
import { btnStyle, COLORS, subtleBtnStyle } from "../styles/theme";

const menuItemStyle = {
  ...subtleBtnStyle,
  display: "block",
  width: "100%",
  textAlign: "left",
  borderColor: "transparent",
  whiteSpace: "nowrap",
};

const popoverStyle = {
  position: "absolute",
  top: "100%",
  right: 0,
  marginTop: 4,
  zIndex: 20,
  minWidth: 150,
  background: COLORS.panel,
  border: `1px solid ${COLORS.panelBorder}`,
  borderRadius: 4,
  padding: 4,
  display: "flex",
  flexDirection: "column",
  gap: 2,
};

// let the player choose which provided solution to reveal
export default function SolutionMenu({ hasAuthorSolution, onSelect }) {
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const popoverId = useId();
  const [open, setOpen] = useState(false);

  // close the dropdown on an outside click or Escape
  useEffect(() => {
    if (!open) return undefined;
    function handlePointerDown(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    }
    function handleKeyDown(event) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    }
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  // close the dropdown and restore focus before loading the selected solution
  function selectSolution(kind) {
    setOpen(false);
    triggerRef.current?.focus();
    onSelect(kind);
  }

  return (
    <div ref={rootRef} style={{ position: "relative" }}>
      <button
        ref={triggerRef}
        type="button"
        style={btnStyle}
        aria-expanded={open}
        aria-controls={popoverId}
        onClick={() => setOpen((current) => !current)}
      >
        Show solution ▾
      </button>
      {open ? (
        <div id={popoverId} style={popoverStyle}>
          <button
            type="button"
            disabled={!hasAuthorSolution}
            title={hasAuthorSolution ? undefined : "No author's solution is available for this puzzle."}
            style={{
              ...menuItemStyle,
              color: hasAuthorSolution ? COLORS.text : COLORS.textMuted,
              cursor: hasAuthorSolution ? "pointer" : "not-allowed",
              opacity: hasAuthorSolution ? 1 : 0.45,
            }}
            onClick={() => selectSolution("author")}
          >
            Author&apos;s solution
          </button>
          <button type="button" style={menuItemStyle} onClick={() => selectSolution("elegant")}>
            Elegant solution
          </button>
        </div>
      ) : null}
    </div>
  );
}
