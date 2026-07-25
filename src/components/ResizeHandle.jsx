import { useEffect, useRef, useState } from "react";
import { COLORS } from "../styles/theme";

export const HANDLE_WIDTH = 8;

// pixels moved per arrow key press
const KEY_STEP = 16;

// draggable gutter between two panels; reports the horizontal offset from the drag origin
export default function ResizeHandle({ label, uiZoom, width, minWidth, maxWidth, onDragStart, onDragMove, onNudge }) {
  const [dragging, setDragging] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const startXRef = useRef(0);
  const pointerIdRef = useRef(null);

  // while dragging, keep the resize cursor and kill text selection everywhere
  useEffect(() => {
    if (!dragging) return undefined;
    const previousCursor = document.body.style.cursor;
    const previousSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousSelect;
    };
  }, [dragging]);

  // only a left-button primary pointer drags, so a second touch or a right-click cannot start a
  // resize; pointer capture keeps move events coming even when the cursor leaves the gutter
  function handlePointerDown(event) {
    if (!event.isPrimary || event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerIdRef.current = event.pointerId;
    startXRef.current = event.clientX;
    setDragging(true);
    onDragStart();
  }

  // pointer coordinates are screen pixels, but panel widths live inside the app's zoom
  function handlePointerMove(event) {
    if (pointerIdRef.current !== event.pointerId) return;
    onDragMove((event.clientX - startXRef.current) / uiZoom);
  }

  function handlePointerUp(event) {
    if (pointerIdRef.current !== event.pointerId) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    pointerIdRef.current = null;
    setDragging(false);
  }

  // arrow keys nudge the same width the pointer drags
  function handleKeyDown(event) {
    const step = event.key === "ArrowLeft" ? -KEY_STEP : event.key === "ArrowRight" ? KEY_STEP : 0;
    if (!step) return;
    event.preventDefault();
    onNudge(step);
  }

  const active = dragging || hovered || focused;

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(width)}
      aria-valuemin={Math.round(minWidth)}
      aria-valuemax={Math.round(maxWidth)}
      tabIndex={0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onKeyDown={handleKeyDown}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        flex: `0 0 ${HANDLE_WIDTH}px`,
        cursor: "col-resize",
        display: "flex",
        justifyContent: "center",
        touchAction: "none",
        outline: "none",
      }}
    >
      {/* thin rule that lights up on hover, focus or drag so the gutter reads as grabbable */}
      <div
        style={{
          width: 2,
          borderRadius: 999,
          background: active ? COLORS.accent : "transparent",
          transition: "background 0.12s",
        }}
      />
    </div>
  );
}
