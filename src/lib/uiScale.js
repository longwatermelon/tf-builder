import { useEffect, useState } from "react";

const DESIGN_WIDTH = 1920;
const DESIGN_HEIGHT = 1080;
const BASE_UI_ZOOM = 1.1;
const MIN_UI_ZOOM = 0.9;
const MAX_UI_ZOOM = 2.2;

// calculate a bounded zoom that preserves the 1080p visual scale
export function calculateUiZoom(width, height) {
  const viewportScale = Math.min(width / DESIGN_WIDTH, height / DESIGN_HEIGHT);
  return Math.min(MAX_UI_ZOOM, Math.max(MIN_UI_ZOOM, BASE_UI_ZOOM * viewportScale));
}

// read the current browser viewport and return its matching UI zoom
function readUiZoom() {
  if (typeof window === "undefined") return BASE_UI_ZOOM;
  return calculateUiZoom(window.innerWidth, window.innerHeight);
}

// keep the UI zoom synchronized with viewport size changes
export function useUiZoom() {
  const [uiZoom, setUiZoom] = useState(readUiZoom);

  useEffect(() => {
    function updateUiZoom() {
      setUiZoom(readUiZoom());
    }

    window.addEventListener("resize", updateUiZoom);
    return () => window.removeEventListener("resize", updateUiZoom);
  }, []);

  return uiZoom;
}
