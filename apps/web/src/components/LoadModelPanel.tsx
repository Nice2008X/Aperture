import { useEffect, useRef } from "react";
import { ModelLoader } from "./ModelLoader.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  open: boolean;
  onClose: () => void;
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  /** The currently-loaded model's id, if any — passed straight through to ModelLoader so it's left out of the catalog list. */
  excludeModelId?: string;
  onLoad: (modelId: string, quantization?: "4bit" | "8bit") => void;
}

/** Popover version of ModelLoader for switching models mid-session — same catalog list as the first-load screen, minus the currently-loaded model, closing itself once a load is kicked off. Mirrors SettingsPanel's open/close-on-outside-click/Escape shape. */
export function LoadModelPanel({ open, onClose, status, error, excludeModelId, onLoad }: Props) {
  const { t } = useTranslation();
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    function onPointerDown(e: PointerEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="load-model-panel" ref={panelRef}>
      <div className="load-model-panel-header">
        <span>{t("app.loadDifferentModel")}</span>
        <button className="load-model-panel-close" onClick={onClose} aria-label={t("loader.close")} title={t("loader.close")}>
          ×
        </button>
      </div>
      <ModelLoader
        status={status}
        error={error}
        excludeModelId={excludeModelId}
        embedded
        onLoad={(modelId, quantization) => {
          onLoad(modelId, quantization);
          onClose();
        }}
      />
    </div>
  );
}
