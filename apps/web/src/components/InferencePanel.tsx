import type { InferenceState } from "../useInference.js";
import type { GenerationState } from "../useGeneration.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  supported: boolean;
  state: InferenceState;
  /** Lifted to App (rather than this panel's own useState) so Apply-a-prediction can rewrite it to match the token sequence actually just run, not just append to whatever text was there before. */
  prompt: string;
  onPromptChange: (prompt: string) => void;
  onRun: (prompt: string) => void;
  selectedTokenIndex: number | null;
  onSelectToken: (i: number) => void;
  compareEnabled: boolean;
  onToggleCompare: () => void;
  promptBState: InferenceState;
  promptBText: string;
  onPromptBTextChange: (prompt: string) => void;
  onRunB: (prompt: string) => void;
  /** Independent from selectedTokenIndex — clicking a Prompt B token should only affect Prompt B's own next-token PredictionPanel, not Prompt A's. */
  selectedTokenIndexB: number | null;
  onSelectTokenB: (i: number) => void;
  /** Real multi-token generation (PLAN.md §8.5) — a separate streamed mode from `state`'s single forward pass, started from the same Prompt A text. */
  generationState: GenerationState;
  onGenerate: (prompt: string) => void;
  onStopGeneration: () => void;
  /** Clicking a generated token re-runs inspection (the same `state` this panel already renders token chips for) on the prefix ending at that token, so the rest of the app can drop into any generated step. */
  onInspectStep: (tokenIndex: number) => void;
}

export function InferencePanel({
  supported,
  state,
  prompt,
  onPromptChange,
  onRun,
  selectedTokenIndex,
  onSelectToken,
  compareEnabled,
  onToggleCompare,
  promptBState,
  promptBText,
  onPromptBTextChange,
  onRunB,
  selectedTokenIndexB,
  onSelectTokenB,
  generationState,
  onGenerate,
  onStopGeneration,
  onInspectStep,
}: Props) {
  const { t } = useTranslation();

  if (!supported) {
    return (
      <div className="inference-panel inference-panel-disabled">
        {t("inference.noTokenizer")}
      </div>
    );
  }

  const generating = generationState.status === "streaming";

  return (
    <div className="inference-panel">
      <form
        className="inference-form"
        onSubmit={(e) => {
          e.preventDefault();
          onRun(prompt);
        }}
      >
        <span className="inference-label">{t("inference.promptA")}</span>
        <input value={prompt} onChange={(e) => onPromptChange(e.target.value)} placeholder={t("inference.placeholderA")} />
        <button type="submit" disabled={state.status === "running"}>
          {state.status === "running" ? t("inference.running") : t("inference.run")}
        </button>
        <button type="button" className="compare-toggle" onClick={onToggleCompare}>
          {compareEnabled ? t("inference.hidePromptB") : t("inference.comparePromptB")}
        </button>
        {generating ? (
          <button type="button" className="generate-btn generate-stop" onClick={onStopGeneration}>
            <span className="spinner spinner-inline" /> {t("generation.stop")}
          </button>
        ) : (
          <button type="button" className="generate-btn" onClick={() => onGenerate(prompt)}>
            {t("generation.generate")}
          </button>
        )}
      </form>

      {state.status === "error" && <div className="inference-error">{state.error}</div>}

      {state.displayTokens && (
        <div className="token-chips">
          {state.displayTokens.map((t, i) => {
            const id = state.result?.tokenIds[i];
            return (
              <button
                key={i}
                className={"token-chip" + (i === selectedTokenIndex ? " selected" : "")}
                onClick={() => onSelectToken(i)}
                title={`position ${i}${id !== undefined ? ` · token id ${id}` : ""}`}
              >
                <span className="token-chip-text">{t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}</span>
                {id !== undefined && <span className="token-chip-id">{id}</span>}
              </button>
            );
          })}
        </div>
      )}

      {generationState.status !== "idle" && (
        <div className="generation-output">
          {generationState.error && <div className="inference-error">{generationState.error}</div>}
          <div className="generation-tokens">
            {generationState.tokens.map((tok, i) => (
              <button
                key={i}
                className="generation-token"
                title={t("generation.inspectStep")}
                onClick={() => onInspectStep(i)}
              >
                {tok.text}
              </button>
            ))}
            {generating && <span className="spinner spinner-inline" />}
          </div>
        </div>
      )}

      {compareEnabled && (
        <form
          className="inference-form prompt-b-form"
          onSubmit={(e) => {
            e.preventDefault();
            onRunB(promptBText);
          }}
        >
          <span className="inference-label">{t("inference.promptB")}</span>
          <input value={promptBText} onChange={(e) => onPromptBTextChange(e.target.value)} placeholder={t("inference.placeholderB")} />
          <button type="submit" disabled={promptBState.status === "running"}>
            {promptBState.status === "running" ? t("inference.running") : t("inference.runB")}
          </button>
        </form>
      )}
      {compareEnabled && promptBState.status === "error" && <div className="inference-error">{promptBState.error}</div>}
      {compareEnabled && promptBState.displayTokens && (
        <div className="token-chips token-chips-b">
          {promptBState.displayTokens.map((t, i) => {
            const id = promptBState.result?.tokenIds[i];
            return (
              <button
                key={i}
                className={"token-chip" + (i === selectedTokenIndexB ? " selected" : "")}
                onClick={() => onSelectTokenB(i)}
                title={`position ${i}${id !== undefined ? ` · token id ${id}` : ""}`}
              >
                <span className="token-chip-text">{t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}</span>
                {id !== undefined && <span className="token-chip-id">{id}</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
