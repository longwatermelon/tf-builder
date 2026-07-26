import { useCallback, useEffect, useRef, useState } from "react";
import App from "./App";
import WorkspaceGuideOverlay from "./components/WorkspaceGuideOverlay";
import { useUiZoom } from "./lib/uiScale";
import { btnStyle, COLORS } from "./styles/theme";

const ONBOARDING_KEY = "tf-builder:onboarding";

// detect phones and tablets, including iPads that report a desktop-style user agent
function detectMobileDevice() {
  if (typeof window === "undefined" || typeof navigator === "undefined") return false;

  const hasUaDataMobileFlag = typeof navigator.userAgentData?.mobile === "boolean"
    ? navigator.userAgentData.mobile
    : false;
  const userAgent = navigator.userAgent || "";
  const hasMobileUserAgent = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile/i.test(userAgent);
  const isLikelyIPadDesktopUa = navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;

  return hasUaDataMobileFlag || hasMobileUserAgent || isLikelyIPadDesktopUa;
}

// baseline onboarding state for first-time visitors
function createDefaultOnboardingState() {
  return {
    status: "prompt",
    hasCompletedGuide: false,
  };
}

// read persisted onboarding state, falling back to the prompt on anything malformed
function loadOnboardingState() {
  if (typeof window === "undefined") return createDefaultOnboardingState();

  try {
    const raw = window.localStorage.getItem(ONBOARDING_KEY);
    if (!raw) return createDefaultOnboardingState();

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return createDefaultOnboardingState();

    if (parsed.status === "ready") {
      return {
        status: "ready",
        hasCompletedGuide: parsed.hasCompletedGuide === true,
      };
    }

    return createDefaultOnboardingState();
  } catch {
    return createDefaultOnboardingState();
  }
}

// persist onboarding state once the prompt has been answered
function persistOnboardingState(state) {
  if (typeof window === "undefined") return;
  if (state.status === "prompt") return;
  try {
    window.localStorage.setItem(ONBOARDING_KEY, JSON.stringify(state));
  } catch {
    // ignore storage write failures
  }
}

// wraps the app with first-visit onboarding: an intro prompt, the spotlight
// walkthrough overlay, and a persistent replay button
export default function RootApp() {
  const [onboardingState, setOnboardingState] = useState(loadOnboardingState);
  const [isGuideActive, setIsGuideActive] = useState(false);
  const [isMobileBlocked] = useState(detectMobileDevice);
  const uiZoom = useUiZoom();

  const promptCardRef = useRef(null);

  const headerGuideRef = useRef(null);
  const puzzlesGuideRef = useRef(null);
  const objectiveGuideRef = useRef(null);
  const weightsGuideRef = useRef(null);
  const stackGuideRef = useRef(null);

  const guideRefs = {
    headerRef: headerGuideRef,
    puzzlesRef: puzzlesGuideRef,
    objectiveRef: objectiveGuideRef,
    weightsRef: weightsGuideRef,
    stackRef: stackGuideRef,
  };

  const guideStops = [
    {
      id: "puzzle-library",
      title: "Puzzle library",
      description:
        "Pick a puzzle here. Each one keeps its own in-progress model, and your solved / elegant marks are shown next to the names.",
      targetRef: puzzlesGuideRef,
    },
    {
      id: "objective",
      title: "Objective",
      description:
        "The task statement, exhaustive rule-validation status, and representative samples. Click a sample to trace the model's computation on it step by step.",
      targetRef: objectiveGuideRef,
    },
    {
      id: "module-stack",
      title: "Architecture",
      description:
        "Your module stack over the residual stream. Insert attention heads, MLPs, and linear layers anywhere after the embedding, reorder or delete them, and click a module to edit it.",
      targetRef: stackGuideRef,
    },
    {
      id: "weights",
      title: "Weights",
      description:
        "Hand-edit every weight of the selected module here. Type decimal values directly, navigate with arrow keys, and use a to insert a decimal point, q to insert a minus sign, w to insert zero, e to insert 1000, and r to insert infinity in attention masks. Click any row or column label to give that dimension a short name — it follows the axis through the whole stack.",
      targetRef: weightsGuideRef,
    },
    {
      id: "header",
      title: "Status bar",
      description:
        "Track valid inputs passed, your parameter count against the elegant solution, and the solve status. You can also reset the model or reveal an author's or elegant solution from here.",
      targetRef: headerGuideRef,
    },
  ];

  const isPromptOpen = onboardingState.status === "prompt";

  useEffect(() => {
    persistOnboardingState(onboardingState);
  }, [onboardingState]);

  // move focus into the intro prompt so keyboard users start inside the dialog
  useEffect(() => {
    if (isPromptOpen) promptCardRef.current?.focus();
  }, [isPromptOpen]);

  // begin the walkthrough from the intro prompt
  const handleStartGuide = useCallback(() => {
    setOnboardingState({ status: "ready", hasCompletedGuide: false });
    setIsGuideActive(true);
  }, []);

  // dismiss the intro prompt without touring
  const handleSkipGuide = useCallback(() => {
    setOnboardingState((prev) => ({ status: "ready", hasCompletedGuide: prev.hasCompletedGuide }));
  }, []);

  // reopen the walkthrough from the replay button
  const handleOpenGuide = useCallback(() => {
    setIsGuideActive(true);
  }, []);

  // close the walkthrough early via skip
  const handleCloseGuide = useCallback(() => {
    setIsGuideActive(false);
  }, []);

  // finish the walkthrough from its last stop
  const handleCompleteGuide = useCallback(() => {
    setIsGuideActive(false);
    setOnboardingState({ status: "ready", hasCompletedGuide: true });
  }, []);

  if (isMobileBlocked) {
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          background: COLORS.bg,
          color: COLORS.text,
          fontFamily: "'Sora', sans-serif",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 16,
        }}
      >
        <div
          role="alert"
          style={{
            width: "min(560px, 100%)",
            background: COLORS.panel,
            border: `1px solid ${COLORS.panelBorder}`,
            borderRadius: 8,
            padding: "22px 18px",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.textBright }}>Desktop only</div>
          <div style={{ fontSize: 14, lineHeight: 1.6 }}>
            This website is intended to be accessed on desktop.
          </div>
          <div style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.55 }}>
            Please open this app from a desktop or laptop browser to use the transformer builder.
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* the app is inert while the prompt or guide is up so the dialog owns all input */}
      <div
        aria-hidden={isPromptOpen || isGuideActive}
        inert={isPromptOpen || isGuideActive}
        style={{ width: "100%", height: "100%" }}
      >
        <App guideRefs={guideRefs} uiZoom={uiZoom} />
      </div>

      {isPromptOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 80,
            zoom: uiZoom,
            background: "rgba(0, 0, 0, 0.6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 16,
            overflowY: "auto",
            fontFamily: "'Sora', sans-serif",
          }}
        >
          <div
            ref={promptCardRef}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-labelledby="onboarding-title"
            aria-describedby="onboarding-description onboarding-replay-note"
            style={{
              width: "min(520px, 100%)",
              background: COLORS.panel,
              border: `1px solid ${COLORS.panelBorder}`,
              borderRadius: 8,
              padding: 16,
              maxHeight: "calc(100dvh - 32px)",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <div id="onboarding-title" style={{ fontSize: 16, fontWeight: 700, color: COLORS.textBright }}>
              Quick workspace walkthrough?
            </div>
            <div id="onboarding-description" style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.5 }}>
              Each puzzle gives you a vocabulary and a set of test sequences with required outputs. You assemble a
              stack of transformer modules over the residual stream and hand-edit every weight until the model
              produces the required token at every graded position.
            </div>
            <div id="onboarding-replay-note" style={{ fontSize: 12, color: COLORS.textMuted, lineHeight: 1.45 }}>
              Want a spotlight tour of the UI first? You can replay it any time from the bottom-right button.
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, flexWrap: "wrap" }}>
              <button onClick={handleSkipGuide} style={btnStyle}>
                Explore on your own
              </button>
              <button
                onClick={handleStartGuide}
                style={{
                  ...btnStyle,
                  borderColor: `${COLORS.accent}70`,
                  color: COLORS.accent,
                }}
              >
                Start guided walkthrough
              </button>
            </div>
          </div>
        </div>
      )}

      {onboardingState.status === "ready" && !isGuideActive && (
        <button
          onClick={handleOpenGuide}
          style={{
            ...btnStyle,
            position: "fixed",
            right: 14,
            bottom: 14,
            zIndex: 50,
            zoom: uiZoom,
            borderColor: `${COLORS.accent}60`,
            color: COLORS.accent,
            background: COLORS.panel,
          }}
        >
          {onboardingState.hasCompletedGuide ? "Replay guide" : "Start guide"}
        </button>
      )}

      {isGuideActive && (
        <WorkspaceGuideOverlay
          stops={guideStops}
          uiZoom={uiZoom}
          onClose={handleCloseGuide}
          onComplete={handleCompleteGuide}
        />
      )}
    </>
  );
}
