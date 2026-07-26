import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { decodeAttempt, encodeAttempt } from "../lib/attemptFile";
import { COLORS, MONO, subtleBtnStyle } from "../styles/theme";

const MAX_FILE_BYTES = 5 * 1024 * 1024;

// one left-aligned action row inside a dropdown menu
const menuItemStyle = {
  ...subtleBtnStyle,
  display: "block",
  width: "100%",
  textAlign: "left",
  borderColor: "transparent",
  whiteSpace: "nowrap",
};

// shared shell for the dropdown menus and the paste popover
const popoverStyle = {
  position: "absolute",
  top: "100%",
  marginTop: 4,
  zIndex: 20,
  background: COLORS.panel,
  border: `1px solid ${COLORS.panelBorder}`,
  borderRadius: 4,
  padding: 4,
  display: "flex",
  flexDirection: "column",
  gap: 2,
};

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

// provide file- and clipboard-based model transfer without exposing browser details to the main app
export default function AttemptFileControls({ puzzle, model, onImport }) {
  const inputRef = useRef(null);
  const rootRef = useRef(null);
  const [notice, setNotice] = useState(null);
  const [menuOpen, setMenuOpen] = useState(null); // null | "import" | "export"
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");

  // track the live puzzle so async reads can detect a puzzle switch mid-flight;
  // layout effect updates the ref synchronously on commit, before pending reads resume
  const puzzleIdRef = useRef(puzzle.id);
  useLayoutEffect(() => {
    puzzleIdRef.current = puzzle.id;
  }, [puzzle.id]);

  // dismiss any open menu or paste box when the pointer goes down outside the controls
  useEffect(() => {
    if (!menuOpen && !pasteOpen) return;
    function handlePointerDown(event) {
      if (rootRef.current?.contains(event.target)) return;
      setMenuOpen(null);
      setPasteOpen(false);
    }
    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [menuOpen, pasteOpen]);

  // true when the active puzzle changed while this handler's async read was in flight
  function importIsStale() {
    return puzzleIdRef.current !== puzzle.id;
  }

  // surface any import failure as a readable notice
  function failNotice(error) {
    setNotice({ ok: false, text: error instanceof Error ? error.message : "The attempt could not be imported." });
  }

  // validate attempt JSON from any source and hand its model to the main app
  function importText(text) {
    if (new Blob([text]).size > MAX_FILE_BYTES) throw new Error("The attempt JSON must be smaller than 5 MB.");
    onImport(decodeAttempt(text, puzzle));
    setNotice({ ok: true, text: "attempt imported" });
  }

  // toggle one of the two dropdown menus, dismissing the paste box
  function toggleMenu(name) {
    setPasteOpen(false);
    setMenuOpen((open) => (open === name ? null : name));
  }

  // encode and download the active puzzle's model
  function handleExport() {
    setMenuOpen(null);
    downloadAttempt(fileNameFor(puzzle.id), encodeAttempt(puzzle.id, model));
    setNotice({ ok: true, text: "attempt exported" });
  }

  // encode the active puzzle's model onto the clipboard
  async function handleCopy() {
    setMenuOpen(null);
    try {
      await navigator.clipboard.writeText(encodeAttempt(puzzle.id, model));
      setNotice({ ok: true, text: "attempt copied to clipboard" });
    } catch {
      setNotice({ ok: false, text: "Clipboard access was blocked; use the file export instead." });
    }
  }

  // validate a selected JSON file before handing its model to the main app
  async function handleImportFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      if (file.size > MAX_FILE_BYTES) throw new Error("The attempt file must be smaller than 5 MB.");
      const text = await file.text();
      if (importIsStale()) return;
      importText(text);
    } catch (error) {
      failNotice(error);
    }
  }

  // import straight from the clipboard, falling back to a manual paste box when reading is blocked
  async function handlePaste() {
    setMenuOpen(null);
    let text = null;
    try {
      text = await navigator.clipboard.readText();
    } catch {
      // clipboard reading unsupported or denied; fall back to the manual paste box
    }
    if (importIsStale()) return;
    if (text === null) {
      setPasteText("");
      setPasteOpen(true);
      return;
    }
    try {
      importText(text);
    } catch (error) {
      failNotice(error);
    }
  }

  // import whatever the player pasted into the manual box
  function handleManualImport() {
    try {
      importText(pasteText);
      setPasteOpen(false);
      setPasteText("");
    } catch (error) {
      failNotice(error);
    }
  }

  return (
    <div
      ref={rootRef}
      style={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        gap: 6,
        flexWrap: "wrap",
        justifyContent: "flex-end",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        onChange={handleImportFile}
        style={{ display: "none" }}
      />
      <div style={{ position: "relative" }}>
        <button type="button" style={subtleBtnStyle} onClick={() => toggleMenu("import")}>
          import JSON ▾
        </button>
        {menuOpen === "import" ? (
          <div style={{ ...popoverStyle, left: 0, minWidth: 150 }}>
            <button
              type="button"
              style={menuItemStyle}
              onClick={() => {
                setMenuOpen(null);
                inputRef.current?.click();
              }}
            >
              from file…
            </button>
            <button type="button" style={menuItemStyle} onClick={handlePaste}>
              paste from clipboard
            </button>
          </div>
        ) : null}
      </div>
      <div style={{ position: "relative" }}>
        <button type="button" style={subtleBtnStyle} onClick={() => toggleMenu("export")}>
          export JSON ▾
        </button>
        {menuOpen === "export" ? (
          <div style={{ ...popoverStyle, left: 0, minWidth: 150 }}>
            <button type="button" style={menuItemStyle} onClick={handleExport}>
              download file
            </button>
            <button type="button" style={menuItemStyle} onClick={handleCopy}>
              copy to clipboard
            </button>
          </div>
        ) : null}
      </div>
      {notice ? (
        <span
          role="status"
          title={notice.text}
          style={{ color: notice.ok ? COLORS.success : COLORS.negative, fontSize: 10, maxWidth: 180 }}
        >
          {notice.text}
        </span>
      ) : null}
      {pasteOpen ? (
        <div style={{ ...popoverStyle, right: 0, marginTop: 6, padding: 8, width: 300, gap: 6 }}>
          <textarea
            value={pasteText}
            onChange={(event) => setPasteText(event.target.value)}
            placeholder="paste attempt JSON here"
            autoFocus
            style={{
              width: "100%",
              height: 110,
              resize: "vertical",
              boxSizing: "border-box",
              background: COLORS.bg,
              color: COLORS.text,
              border: `1px solid ${COLORS.panelBorder}`,
              borderRadius: 3,
              fontFamily: MONO,
              fontSize: 10,
              padding: 6,
            }}
          />
          <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
            <button type="button" style={subtleBtnStyle} onClick={() => setPasteOpen(false)}>
              cancel
            </button>
            <button type="button" style={subtleBtnStyle} onClick={handleManualImport}>
              import
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
