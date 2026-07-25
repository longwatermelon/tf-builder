import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { btnStyle, COLORS, subtleBtnStyle } from "../styles/theme";

const SPOTLIGHT_PADDING = 10;
const VIEWPORT_EDGE_PADDING = 8;
const TOOLTIP_WIDTH = 332;
const TOOLTIP_ESTIMATED_HEIGHT = 170;
const OVERLAY_COLOR = "rgba(0, 0, 0, 0.58)";

// keep a value inside [min, max]
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// step-by-step spotlight tour: dims the app, cuts a hole around the active
// stop's target, and shows a tooltip card with back / next / skip controls
export default function WorkspaceGuideOverlay({ stops = [], uiZoom, onClose, onComplete }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [highlightRect, setHighlightRect] = useState(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0 });

  const cardRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastAutoScrollStepRef = useRef(-1);

  const activeStop = stops[stepIndex] ?? null;
  const activeTargetRef = activeStop?.targetRef ?? null;

  useEffect(() => {
    cardRef.current?.focus();
  }, [stepIndex]);

  // measure the active target and convert it into the overlay's zoomed coordinate space;
  // getBoundingClientRect reports visual pixels while the overlay root renders under uiZoom,
  // so every measured value is divided by the zoom before use
  const updateHighlight = useCallback(() => {
    if (!activeTargetRef?.current) {
      setHighlightRect((prev) => (prev === null ? prev : null));
      return;
    }

    const targetElement = activeTargetRef.current;
    const targetRect = targetElement.getBoundingClientRect();
    const viewportWidth = window.innerWidth / uiZoom;
    const viewportHeight = window.innerHeight / uiZoom;

    setViewport((prev) =>
      prev.width === viewportWidth && prev.height === viewportHeight
        ? prev
        : { width: viewportWidth, height: viewportHeight }
    );

    const viewportTop = VIEWPORT_EDGE_PADDING;
    const viewportLeft = VIEWPORT_EDGE_PADDING;
    const viewportBottom = viewportHeight - VIEWPORT_EDGE_PADDING;
    const viewportRight = viewportWidth - VIEWPORT_EDGE_PADDING;

    // scroll a fully offscreen target into view once per step so the spotlight has something to frame
    const isOutsideHorizontalViewport = targetRect.right / uiZoom <= viewportLeft || targetRect.left / uiZoom >= viewportRight;
    const isOutsideVerticalViewport = targetRect.bottom / uiZoom <= viewportTop || targetRect.top / uiZoom >= viewportBottom;
    if ((isOutsideHorizontalViewport || isOutsideVerticalViewport) && lastAutoScrollStepRef.current !== stepIndex) {
      lastAutoScrollStepRef.current = stepIndex;
      targetElement.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
    }

    // clamp the spotlight to the viewport so oversized panels still get a visible frame
    const minHighlightWidth = 40;
    const minHighlightHeight = 32;
    const maxLeft = Math.max(viewportLeft, viewportRight - minHighlightWidth);
    const maxTop = Math.max(viewportTop, viewportBottom - minHighlightHeight);
    const left = clamp(targetRect.left / uiZoom - SPOTLIGHT_PADDING, viewportLeft, maxLeft);
    const top = clamp(targetRect.top / uiZoom - SPOTLIGHT_PADDING, viewportTop, maxTop);
    const right = clamp(targetRect.right / uiZoom + SPOTLIGHT_PADDING, left + minHighlightWidth, viewportRight);
    const bottom = clamp(targetRect.bottom / uiZoom + SPOTLIGHT_PADDING, top + minHighlightHeight, viewportBottom);

    const nextRect = {
      left,
      top,
      width: Math.max(minHighlightWidth, right - left),
      height: Math.max(minHighlightHeight, bottom - top),
    };

    // skip state updates for sub-pixel jitter so resize storms stay cheap
    setHighlightRect((prev) => {
      if (!prev) return nextRect;
      if (
        Math.abs(prev.left - nextRect.left) < 0.5
        && Math.abs(prev.top - nextRect.top) < 0.5
        && Math.abs(prev.width - nextRect.width) < 0.5
        && Math.abs(prev.height - nextRect.height) < 0.5
      ) {
        return prev;
      }
      return nextRect;
    });
  }, [activeTargetRef, stepIndex, uiZoom]);

  // re-measure on every resize / scroll, coalesced through one animation frame
  useLayoutEffect(() => {
    const scheduleHighlightUpdate = () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = requestAnimationFrame(() => {
        animationFrameRef.current = null;
        updateHighlight();
      });
    };

    scheduleHighlightUpdate();
    window.addEventListener("resize", scheduleHighlightUpdate);
    window.addEventListener("scroll", scheduleHighlightUpdate, true);

    return () => {
      window.removeEventListener("resize", scheduleHighlightUpdate);
      window.removeEventListener("scroll", scheduleHighlightUpdate, true);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [updateHighlight]);

  // step back one stop, clamped at the first
  const handleBack = useCallback(() => {
    setStepIndex((prev) => (prev <= 0 ? 0 : prev - 1));
  }, []);

  // advance one stop, completing the guide from the last
  const handleNext = useCallback(() => {
    const lastStepIndex = Math.max(0, stops.length - 1);
    if (stepIndex >= lastStepIndex) {
      onComplete?.();
      return;
    }
    setStepIndex((prev) => Math.min(lastStepIndex, prev + 1));
  }, [onComplete, stepIndex, stops.length]);

  if (!activeStop) return null;

  // place the tooltip fully outside the spotlight, trying below, above, right, then left;
  // only when no side fits does it fall back to a clamped position that may overlap
  const tooltipWidth = Math.max(260, Math.min(TOOLTIP_WIDTH, viewport.width - 24));
  let tooltipLeft = 12;
  let tooltipTop = 14;
  if (highlightRect) {
    const gap = 12;
    const maxLeft = Math.max(12, viewport.width - tooltipWidth - 12);
    const maxTop = Math.max(12, viewport.height - TOOLTIP_ESTIMATED_HEIGHT - 12);
    const fitsBelow = highlightRect.top + highlightRect.height + gap + TOOLTIP_ESTIMATED_HEIGHT <= viewport.height - 12;
    const fitsAbove = highlightRect.top - gap - TOOLTIP_ESTIMATED_HEIGHT >= 12;
    const fitsRight = highlightRect.left + highlightRect.width + gap + tooltipWidth <= viewport.width - 12;
    const fitsLeft = highlightRect.left - gap - tooltipWidth >= 12;

    if (fitsBelow) {
      tooltipLeft = clamp(highlightRect.left, 12, maxLeft);
      tooltipTop = highlightRect.top + highlightRect.height + gap;
    } else if (fitsAbove) {
      tooltipLeft = clamp(highlightRect.left, 12, maxLeft);
      tooltipTop = highlightRect.top - gap - TOOLTIP_ESTIMATED_HEIGHT;
    } else if (fitsRight) {
      tooltipLeft = highlightRect.left + highlightRect.width + gap;
      tooltipTop = clamp(highlightRect.top, 12, maxTop);
    } else if (fitsLeft) {
      tooltipLeft = highlightRect.left - gap - tooltipWidth;
      tooltipTop = clamp(highlightRect.top, 12, maxTop);
    } else {
      tooltipLeft = clamp(highlightRect.left, 12, maxLeft);
      tooltipTop = clamp(highlightRect.top + highlightRect.height + gap, 12, maxTop);
    }
  }
  const spotlightShadowSpread = Math.max(viewport.width, viewport.height, 2000);

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 120, zoom: uiZoom, fontFamily: "'Sora', sans-serif" }}>
      {/* full dim while the target has not been measured yet */}
      {!highlightRect && <div style={{ position: "absolute", inset: 0, background: OVERLAY_COLOR }} />}

      {/* the spotlight: a bordered hole whose giant box shadow dims everything around it */}
      {highlightRect && (
        <div
          style={{
            position: "absolute",
            left: highlightRect.left,
            top: highlightRect.top,
            width: highlightRect.width,
            height: highlightRect.height,
            borderRadius: 8,
            border: `2px solid ${COLORS.accent}`,
            boxShadow: `0 0 0 ${spotlightShadowSpread}px ${OVERLAY_COLOR}, 0 0 0 1px ${COLORS.accent}55, 0 0 24px ${COLORS.accent}55`,
          }}
        />
      )}

      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-label="Workspace guide"
        tabIndex={-1}
        style={{
          position: "absolute",
          top: tooltipTop,
          left: tooltipLeft,
          width: tooltipWidth,
          maxWidth: "calc(100vw - 24px)",
          background: COLORS.panel,
          border: `1px solid ${COLORS.panelBorder}`,
          borderRadius: 6,
          padding: 12,
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <div style={{ fontSize: 10, color: COLORS.accent, textTransform: "uppercase", letterSpacing: 0.8, fontWeight: 700 }}>
          {`Guide ${stepIndex + 1} / ${stops.length}`}
        </div>
        <div style={{ fontSize: 14, color: COLORS.textBright, fontWeight: 700 }}>{activeStop.title}</div>
        <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.45 }}>{activeStop.description}</div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap", marginTop: 2 }}>
          <button onClick={onClose} style={subtleBtnStyle}>
            Skip guide
          </button>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handleBack}
              disabled={stepIndex === 0}
              style={{ ...btnStyle, opacity: stepIndex === 0 ? 0.45 : 1 }}
            >
              Back
            </button>
            <button
              onClick={handleNext}
              style={{ ...btnStyle, borderColor: `${COLORS.accent}70`, color: COLORS.accent }}
            >
              {stepIndex >= stops.length - 1 ? "Done" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
