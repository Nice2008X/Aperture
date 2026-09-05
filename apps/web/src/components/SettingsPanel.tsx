import { useEffect, useRef, useState } from "react";
import { useTranslation } from "./LanguageContext.js";
import { ThemeSwitcher, type Theme } from "./ThemeSwitcher.js";
import { LANGUAGES, type Language, type TranslationKey } from "../i18n.js";
import type { GenerationParams } from "../useGeneration.js";

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
  maxNewTokens: number;
  onMaxNewTokensChange: (v: number) => void;
  /** The loaded model's real context length (config.json's max_position_embeddings) — the true ceiling for how many new tokens a single Generate call could ever need, since the model was never trained on positions past this. Falls back to a generic cap when no model is loaded yet (this panel is reachable from the loader screen too, before that number is known). */
  maxNewTokensLimit: number;
  /** The live, user-editable sampling parameters Generate actually uses. */
  generationParams: GenerationParams;
  onGenerationParamChange: <K extends keyof GenerationParams>(key: K, value: GenerationParams[K]) => void;
  /** Restores generationParams to this loaded model's own recommended values (its generation_config.json wherever set, this app's defaults otherwise) — App owns the actual values (modelGenerationDefaults), this panel just triggers it. */
  onResetGenerationDefaults: () => void;
}

/** Keeps a stray localStorage value (hand-edited, or from a future/older build) from producing zero or negative pages. */
const MODELS_PER_PAGE_MIN = 1;
const MODELS_PER_PAGE_MAX = 30;

const MAX_NEW_TOKENS_MIN = 1;
/** Ceiling shown before any model has been loaded (SettingsPanel is reachable from the loader screen too) — once a model is loaded, maxNewTokensLimit is that model's real context length instead, which can be much larger or, occasionally, smaller than this. */
export const MAX_NEW_TOKENS_FALLBACK_LIMIT = 512;

const TEMPERATURE_MIN = 0;
const TEMPERATURE_MAX = 2;
const TEMPERATURE_STEP = 0.05;
const TOP_P_MIN = 0;
const TOP_P_MAX = 1;
const TOP_P_STEP = 0.05;
const TOP_K_MIN = 0;
const TOP_K_MAX = 100;
const REPETITION_PENALTY_MIN = 1;
const REPETITION_PENALTY_MAX = 2;
const REPETITION_PENALTY_STEP = 0.05;
const NO_REPEAT_NGRAM_MIN = 0;
const NO_REPEAT_NGRAM_MAX = 10;

type SettingsTab = "appearance" | "behavior" | "generation";
const SETTINGS_TABS: SettingsTab[] = ["appearance", "behavior", "generation"];
const SETTINGS_TAB_LABEL_KEYS: Record<SettingsTab, TranslationKey> = {
  appearance: "settings.tab.appearance",
  behavior: "settings.tab.behavior",
  generation: "settings.tab.generation",
};

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
  maxNewTokens,
  onMaxNewTokensChange,
  maxNewTokensLimit,
  generationParams,
  onGenerationParamChange,
  onResetGenerationDefaults,
}: Props) {
  const { t, language, setLanguage } = useTranslation();
  const panelRef = useRef<HTMLDivElement>(null);
  // Not lifted to App or persisted — which tab is showing is pure
  // in-the-moment UI state, same reasoning as InferencePanel's
  // outputCollapsed.
  const [activeTab, setActiveTab] = useState<SettingsTab>("appearance");

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
      <div className="settings-tabs" role="tablist">
        {SETTINGS_TABS.map((tabId) => (
          <button
            key={tabId}
            type="button"
            role="tab"
            aria-selected={activeTab === tabId}
            className={"settings-tab" + (activeTab === tabId ? " active" : "")}
            onClick={() => setActiveTab(tabId)}
          >
            {t(SETTINGS_TAB_LABEL_KEYS[tabId])}
          </button>
        ))}
      </div>

      {activeTab === "appearance" && (
        <>
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
        </>
      )}

      {activeTab === "behavior" && (
        <div className="settings-section">
          <label className="settings-checkbox-row">
            <input type="checkbox" checked={unloadOnHome} onChange={(e) => onUnloadOnHomeChange(e.target.checked)} />
            {t("settings.unloadOnHome")}
          </label>
          <div className="settings-checkbox-hint">{t("settings.unloadOnHomeHint")}</div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.modelsPerPage")}</span>
              <span className="settings-slider-value">{Math.min(modelsPerPage, MODELS_PER_PAGE_MAX)}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={MODELS_PER_PAGE_MIN}
              max={MODELS_PER_PAGE_MAX}
              value={Math.min(modelsPerPage, MODELS_PER_PAGE_MAX)}
              onChange={(e) => onModelsPerPageChange(Math.round(Number(e.target.value)))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.modelsPerPageHint")}</div>
          <label className="settings-checkbox-row">
            <input type="checkbox" checked={showGpuStatus} onChange={(e) => onShowGpuStatusChange(e.target.checked)} />
            {t("settings.showGpuStatus")}
          </label>
          <div className="settings-checkbox-hint">{t("settings.showGpuStatusHint")}</div>
        </div>
      )}

      {activeTab === "generation" && (
        <div className="settings-section">
          <div className="settings-section-header">
            <div className="settings-section-title">{t("settings.generation")}</div>
            <button type="button" className="settings-reset-btn" onClick={onResetGenerationDefaults}>
              {t("settings.resetGenerationDefaults")}
            </button>
          </div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.maxNewTokens")}</span>
              <span className="settings-slider-value">{Math.min(maxNewTokens, maxNewTokensLimit)}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={MAX_NEW_TOKENS_MIN}
              max={maxNewTokensLimit}
              value={Math.min(maxNewTokens, maxNewTokensLimit)}
              onChange={(e) => onMaxNewTokensChange(Math.round(Number(e.target.value)))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.maxNewTokensHint").replace("{limit}", maxNewTokensLimit.toLocaleString())}</div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.temperature")}</span>
              <span className="settings-slider-value">{generationParams.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={TEMPERATURE_MIN}
              max={TEMPERATURE_MAX}
              step={TEMPERATURE_STEP}
              value={generationParams.temperature}
              onChange={(e) => onGenerationParamChange("temperature", Number(e.target.value))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.temperatureHint")}</div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.topP")}</span>
              <span className="settings-slider-value">{generationParams.topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={TOP_P_MIN}
              max={TOP_P_MAX}
              step={TOP_P_STEP}
              value={generationParams.topP}
              onChange={(e) => onGenerationParamChange("topP", Number(e.target.value))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.topPHint")}</div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.topK")}</span>
              <span className="settings-slider-value">{generationParams.topK}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={TOP_K_MIN}
              max={TOP_K_MAX}
              value={generationParams.topK}
              onChange={(e) => onGenerationParamChange("topK", Math.round(Number(e.target.value)))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.topKHint")}</div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.repetitionPenalty")}</span>
              <span className="settings-slider-value">{generationParams.repetitionPenalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={REPETITION_PENALTY_MIN}
              max={REPETITION_PENALTY_MAX}
              step={REPETITION_PENALTY_STEP}
              value={generationParams.repetitionPenalty}
              onChange={(e) => onGenerationParamChange("repetitionPenalty", Number(e.target.value))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.repetitionPenaltyHint")}</div>
          <div className="settings-slider-row">
            <div className="settings-slider-label-row">
              <span>{t("settings.noRepeatNgramSize")}</span>
              <span className="settings-slider-value">{generationParams.noRepeatNgramSize}</span>
            </div>
            <input
              type="range"
              className="settings-slider-input"
              min={NO_REPEAT_NGRAM_MIN}
              max={NO_REPEAT_NGRAM_MAX}
              value={generationParams.noRepeatNgramSize}
              onChange={(e) => onGenerationParamChange("noRepeatNgramSize", Math.round(Number(e.target.value)))}
            />
          </div>
          <div className="settings-checkbox-hint">{t("settings.noRepeatNgramSizeHint")}</div>
        </div>
      )}
    </div>
  );
}
