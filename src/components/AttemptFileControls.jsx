import { useRef, useState } from "react";
import { decodeAttempt, encodeAttempt } from "../lib/attemptFile";
import { COLORS, subtleBtnStyle } from "../styles/theme";

const MAX_FILE_BYTES = 5 * 1024 * 1024;

// make a predictable filename even when a future puzzle ID contains punctuation
function fileNameFor(puzzleId) {
  const safeId = puzzleId.replace(/[^a-z0-9_-]+/gi, "-");
  return `tf-builder-${safeId}-attempt.json`;
}

// trigger a browser download for the encoded attempt
function downloadAttempt(fileName, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

// provide file-based model transfer without exposing browser details to the main app
export default function AttemptFileControls({ puzzle, model, onImport }) {
  const inputRef = useRef(null);
  const [notice, setNotice] = useState(null);

  // encode and download the active puzzle's model
  function handleExport() {
    downloadAttempt(fileNameFor(puzzle.id), encodeAttempt(puzzle.id, model));
    setNotice({ ok: true, text: "attempt exported" });
  }

  // validate a selected JSON file before handing its model to the main app
  async function handleImport(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      if (file.size > MAX_FILE_BYTES) throw new Error("The attempt file must be smaller than 5 MB.");
      const importedModel = decodeAttempt(await file.text(), puzzle);
      onImport(importedModel);
      setNotice({ ok: true, text: "attempt imported" });
    } catch (error) {
      setNotice({ ok: false, text: error instanceof Error ? error.message : "The attempt could not be imported." });
    }
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        onChange={handleImport}
        style={{ display: "none" }}
      />
      <button type="button" style={subtleBtnStyle} onClick={() => inputRef.current?.click()}>
        import JSON
      </button>
      <button type="button" style={subtleBtnStyle} onClick={handleExport}>
        export JSON
      </button>
      {notice ? (
        <span
          role="status"
          title={notice.text}
          style={{ color: notice.ok ? COLORS.success : COLORS.negative, fontSize: 10, maxWidth: 180 }}
        >
          {notice.text}
        </span>
      ) : null}
    </div>
  );
}
