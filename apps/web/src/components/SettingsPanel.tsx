import { useEffect, useRef } from "react";
import { useTranslation } from "./LanguageContext.js";
import { ThemeSwitcher, type Theme } from "./ThemeSwitcher.js";
import { LANGUAGES, type Language } from "../i18n.js";

interface Props {
  open: boolean;
  onClose: () => void;
  theme: Theme;
  onThemeChange: (t: Theme) => void;
  unloadOnHome: boolean;
  onUnloadOnHomeChange: (v: boolean) => void;
  modelsPerPage: number;
  onModelsPerPageChange: (v: number) => void;
  showGpuStatus: boolean;
  onShowGpuStatusChange: (v: boolean) => void;
}

/** Keeps a stray localStorage value (hand-edited, or from a future/older build) from producing zero or negative pages. */
const MODELS_PER_PAGE_MIN = 1;
const MODELS_PER_PAGE_MAX = 50;

export function SettingsButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const { t } = useTranslation();
  return (
    <button className={"settings-toggle-btn" + (open ? " active" : "")} onClick={onToggle} title={t("app.settings")} aria-label={t("app.settings")} aria-expanded={open}>
      ⚙
    </button>
  );
}

export function SettingsPanel({
  open,
  onClose,
  theme,
  onThemeChange,
  unloadOnHome,
  onUnloadOnHomeChange,
  modelsPerPage,
  onModelsPerPageChange,
  showGpuStatus,
  onShowGpuStatusChange,
}: Props) {
  const { t, language, setLanguage } = useTranslation();
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
    <div className="settings-panel" ref={panelRef}>
      <div className="settings-panel-header">
        <span>{t("settings.title")}</span>
        <button className="settings-panel-close" onClick={onClose} aria-label={t("settings.close")} title={t("settings.close")}>
          ×
        </button>
      </div>
      <div className="settings-section">
        <div className="settings-section-title">{t("settings.theme")}</div>
        <ThemeSwitcher theme={theme} onChange={onThemeChange} />
      </div>
      <div className="settings-section">
        <div className="settings-section-title">{t("settings.language")}</div>
        <select className="settings-language-select" value={language} onChange={(e) => setLanguage(e.target.value as Language)}>
          {LANGUAGES.map((l) => (
            <option key={l.id} value={l.id}>
              {l.nativeLabel}
            </option>
          ))}
        </select>
      </div>
      <div className="settings-section">
        <div className="settings-section-title">{t("settings.behavior")}</div>
        <label className="settings-checkbox-row">
          <input type="checkbox" checked={unloadOnHome} onChange={(e) => onUnloadOnHomeChange(e.target.checked)} />
          {t("settings.unloadOnHome")}
        </label>
        <div className="settings-checkbox-hint">{t("settings.unloadOnHomeHint")}</div>
        <label className="settings-number-row">
          {t("settings.modelsPerPage")}
          <input
            type="number"
            className="settings-number-input"
            min={MODELS_PER_PAGE_MIN}
            max={MODELS_PER_PAGE_MAX}
            value={modelsPerPage}
            onChange={(e) => {
              const n = Math.round(Number(e.target.value));
              if (Number.isFinite(n)) onModelsPerPageChange(Math.min(MODELS_PER_PAGE_MAX, Math.max(MODELS_PER_PAGE_MIN, n)));
            }}
          />
        </label>
        <div className="settings-checkbox-hint">{t("settings.modelsPerPageHint")}</div>
        <label className="settings-checkbox-row">
          <input type="checkbox" checked={showGpuStatus} onChange={(e) => onShowGpuStatusChange(e.target.checked)} />
          {t("settings.showGpuStatus")}
        </label>
        <div className="settings-checkbox-hint">{t("settings.showGpuStatusHint")}</div>
      </div>
    </div>
  );
}
