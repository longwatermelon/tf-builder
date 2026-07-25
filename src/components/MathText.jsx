import { useEffect, useMemo, useState } from "react";

let katexPromise;
let katexModule;

// memoize the dynamic import so many labels reuse one katex instance
function loadKatex() {
  if (katexModule) return Promise.resolve(katexModule);
  if (!katexPromise) {
    katexPromise = import("katex")
      .then((module) => {
        katexModule = module.default ?? module;
        return katexModule;
      })
      .catch((error) => {
        katexPromise = undefined;
        throw error;
      });
  }
  return katexPromise;
}

loadKatex();

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export default function MathText({ tex, style, className, displayMode = false }) {
  const [katex, setKatex] = useState(() => katexModule ?? null);

  useEffect(() => {
    let mounted = true;
    loadKatex()
      .then((loadedKatex) => {
        if (!mounted) return;
        setKatex(() => loadedKatex);
      })
      .catch(() => {
        // keep the plain-text fallback when katex fails to load
      });
    return () => {
      mounted = false;
    };
  }, []);

  const html = useMemo(() => {
    if (tex === null || tex === undefined || tex === "") return "";
    if (!katex) return escapeHtml(String(tex));
    try {
      return katex.renderToString(String(tex), {
        displayMode,
        throwOnError: true,
        strict: "ignore",
        output: "htmlAndMathml",
      });
    } catch {
      return escapeHtml(String(tex));
    }
  }, [tex, displayMode, katex]);

  return (
    <span
      className={className}
      style={{ display: displayMode ? "block" : "inline-flex", alignItems: "center", ...style }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
