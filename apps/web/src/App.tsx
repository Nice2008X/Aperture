import { useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { Model, Tensor } from "@aperture/model-ir";
import { unloadModel, getGenerationConfigDefaults } from "@aperture/api-client";
import { useModel } from "./useModel.js";
import { useInference } from "./useInference.js";
import { useGeneration, DEFAULT_GENERATION_PARAMS, type GenerationMode, type GenerationParams } from "./useGeneration.js";
import { useLocalStorageState } from "./useLocalStorageState.js";
import { useTheme } from "./components/ThemeSwitcher.js";
import { useTranslation } from "./components/LanguageContext.js";
import { SettingsButton, SettingsPanel, MAX_NEW_TOKENS_FALLBACK_LIMIT } from "./components/SettingsPanel.js";
import { ModelLoader } from "./components/ModelLoader.js";
import { LoadModelPanel } from "./components/LoadModelPanel.js";
import { ModelInfoBar } from "./components/ModelInfoBar.js";
import { ModelTree } from "./components/ModelTree.js";
import { ArchitectureGraph, type GraphView } from "./components/ArchitectureGraph.js";
import { Inspector } from "./components/Inspector.js";
import { TensorExplorer, type TensorSourceRequest } from "./components/TensorExplorer.js";
import { InferencePanel } from "./components/InferencePanel.js";
import { PredictionPanel } from "./components/PredictionPanel.js";
import { LogitLensPanel } from "./components/LogitLensPanel.js";
import { TokenAttributionPanel } from "./components/TokenAttributionPanel.js";
import { ExperimentPanel } from "./components/ExperimentPanel.js";

type BottomTab = "tensor" | "logitlens" | "attribution" | "experiment";

const BOTTOM_PANEL_DEFAULT_HEIGHT = 360;
const BOTTOM_PANEL_MIN_HEIGHT = 160;
/** Leaves at least this much vertical space for the tree/graph/inspector row above, however tall the window is. */
const BOTTOM_PANEL_TOP_RESERVE = 240;
/** Default for the Settings panel's "Max generation length" — user-adjustable up to the loaded model's real context length (see MAX_NEW_TOKENS_FALLBACK_LIMIT for the pre-load fallback ceiling). */
const DEFAULT_MAX_NEW_TOKENS = 64;

const TREE_PANEL_DEFAULT_WIDTH = 260;
// Below this, tree rows (indented 14px per depth) and long node names start
// truncating illegibly rather than just looking cramped.
const TREE_PANEL_MIN_WIDTH = 160;
const INSPECTOR_PANEL_DEFAULT_WIDTH = 320;
// Below this, the Inspector's io-rows (a label and a monospace value on one
// line — see .io-row) start wrapping instead of staying legible.
const INSPECTOR_PANEL_MIN_WIDTH = 220;
/** Leaves at least this much horizontal space for the graph pane, however wide the tree/inspector panels are dragged. */
const GRAPH_PANEL_MIN_WIDTH = 320;
/** The two 6px resize handles plus each pane's own 1px border (×3 panes) — real chrome the max-width math below would otherwise ignore, letting the graph pane end up a bit under GRAPH_PANEL_MIN_WIDTH rather than at least that wide. */
const PANE_CHROME_ALLOWANCE = 24;

function computeActivationMagnitude(t: Tensor): number {
  let sum = 0;
  for (let i = 0; i < t.data.length; i++) sum += t.data[i] * t.data[i];
  return Math.sqrt(sum);
}

/** The transformer block that owns `nodeId` — itself if it is one, else the nearest ancestor — or null for a top-level node (embeddings, final norm, LM head, ...) that isn't inside any block. */
function containingBlockId(model: Model, nodeId: string): string | null {
  let cur: string | null = nodeId;
  while (cur) {
    if (model.nodes[cur].type === "transformer_block") return cur;
    cur = model.nodes[cur].parentId ?? null;
  }
  return null;
}

export function App() {
  const { state, load, reset } = useModel();
  const { theme, setTheme } = useTheme();
  const { t } = useTranslation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [loadModelOpen, setLoadModelOpen] = useState(false);
  const [unloadOnHome, setUnloadOnHome] = useLocalStorageState("settings:unloadOnHome", false);
  const [modelsPerPage, setModelsPerPage] = useLocalStorageState("settings:modelsPerPage", 5);
  const [showGpuStatus, setShowGpuStatus] = useLocalStorageState("settings:showGpuStatus", true);
  const [maxNewTokens, setMaxNewTokens] = useLocalStorageState("settings:maxNewTokens", DEFAULT_MAX_NEW_TOKENS);
  const [homeBusy, setHomeBusy] = useState(false);
  // Lifted out of InferencePanel's own state so Apply-a-prediction (below)
  // can rewrite the input text to match whatever token sequence was
  // actually just run, not merely append to it.
  const [promptAText, setPromptAText] = useState("The cat sat on the");
  const [promptBText, setPromptBText] = useState("The dog sat on the");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<GraphView>({ kind: "architecture" });
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  // Separate from selectedTokenIndex — that one is shared across the
  // position-matching panels (TensorExplorer/LogitLens/Attribution, which
  // intentionally compare the *same* token position between A and B), but
  // each prompt's own next-token PredictionPanel should react only to
  // clicks on that prompt's own tokens, not the other one's.
  const [selectedTokenIndexB, setSelectedTokenIndexB] = useState<number | null>(null);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [bottomTab, setBottomTab] = useState<BottomTab>("tensor");
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [tensorSourceRequest, setTensorSourceRequest] = useState<TensorSourceRequest | null>(null);
  /** Which prompt's tokens Token Attribution attributes — controlled here (rather than as the panel's own local state) so the Prediction panel's "Why?" link can request the matching side instead of always landing back on Prompt A. */
  const [attributionSource, setAttributionSource] = useState<"A" | "B">("A");
  const [treeCollapsed, setTreeCollapsed] = useLocalStorageState("panel:tree-collapsed", false);
  const [inspectorCollapsed, setInspectorCollapsed] = useLocalStorageState("panel:inspector-collapsed", false);
  const [treeWidth, setTreeWidth] = useLocalStorageState("panel:tree-width", TREE_PANEL_DEFAULT_WIDTH);
  const [inspectorWidth, setInspectorWidth] = useLocalStorageState("panel:inspector-width", INSPECTOR_PANEL_DEFAULT_WIDTH);
  const [resizingTree, setResizingTree] = useState(false);
  const [resizingInspector, setResizingInspector] = useState(false);
  const [bottomCollapsed, setBottomCollapsed] = useLocalStorageState("panel:bottom-collapsed", false);
  const [bottomHeight, setBottomHeight] = useLocalStorageState("panel:bottom-height", BOTTOM_PANEL_DEFAULT_HEIGHT);
  const [resizingBottom, setResizingBottom] = useState(false);
  const [predictionCollapsed, setPredictionCollapsed] = useLocalStorageState("panel:prediction-collapsed", false);
  // Wraps InferencePanel (Prompt A/B forms, tokens, Generate) and the
  // prediction-panels-row together as one collapsible group — separate
  // from predictionCollapsed above, which only hides the prediction rows
  // within an already-visible group.
  const [promptSectionCollapsed, setPromptSectionCollapsed] = useLocalStorageState("panel:prompt-section-collapsed", false);
  // Mirrors ArchitectureGraph's own on-canvas zoom badge — kept here too so
  // the status footer can show it without either component owning the
  // other's state.
  const [zoomPercent, setZoomPercent] = useState(100);

  const inference = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);
  const promptB = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);
  const generation = useGeneration(state.weightProvider, state.tokenizer);

  // The true ceiling for Settings' "Max generation length" — a model's own
  // trained context length (config.json's max_position_embeddings), not
  // one flat number for every checkpoint; a short-context model can't
  // safely generate as far as a long-context one. Falls back to a generic
  // cap before any model is loaded (Settings is reachable from the loader
  // screen too). Re-clamped here (not just at the input) in case a stored
  // preference from a previous, longer-context model now exceeds this one.
  const maxNewTokensLimit = state.model?.config.contextLength ?? MAX_NEW_TOKENS_FALLBACK_LIMIT;
  const effectiveMaxNewTokens = Math.min(maxNewTokens, maxNewTokensLimit);

  // Whether the loaded model's tokenizer defines a chat_template at all
  // (most base/completion checkpoints don't) — decides whether Generate's
  // chat/raw toggle is actually offered, and which one it defaults to.
  const [chatTemplateAvailable, setChatTemplateAvailable] = useState(false);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("raw");
  // Settings' "Generation" sliders (temperature/topP/topK/repetitionPenalty/
  // noRepeatNgramSize) — modelGenerationDefaults is what "Reset to model
  // defaults" restores, generationParams is the live, user-editable value
  // actually sent to /api/generate. Deliberately plain useState, not
  // useLocalStorageState like maxNewTokens: these are reseeded from the
  // *model's own* generation_config.json on every load (see
  // GenerationParams' doc comment), so persisting a stale override across a
  // switch to a different model would fight that model's own tuning rather
  // than respect it.
  const [modelGenerationDefaults, setModelGenerationDefaults] = useState<GenerationParams>(DEFAULT_GENERATION_PARAMS);
  const [generationParams, setGenerationParams] = useState<GenerationParams>(DEFAULT_GENERATION_PARAMS);
  useEffect(() => {
    const modelId = state.weightProvider?.id;
    if (!modelId) {
      setChatTemplateAvailable(false);
      setModelGenerationDefaults(DEFAULT_GENERATION_PARAMS);
      setGenerationParams(DEFAULT_GENERATION_PARAMS);
      return;
    }
    let cancelled = false;
    getGenerationConfigDefaults(modelId)
      .then((info) => {
        if (cancelled) return;
        setChatTemplateAvailable(info.hasChatTemplate);
        // Chat is the better default whenever it's actually available —
        // an instruct-tuned model's own trained format, not a completion
        // prompt it was never tuned to continue from. Re-derived per model
        // load rather than remembered across model switches, since a
        // different model may not have a template at all.
        setGenerationMode(info.hasChatTemplate ? "chat" : "raw");
        const defaults: GenerationParams = {
          temperature: info.temperature,
          topP: info.topP,
          topK: info.topK,
          repetitionPenalty: info.repetitionPenalty,
          noRepeatNgramSize: info.noRepeatNgramSize,
        };
        setModelGenerationDefaults(defaults);
        setGenerationParams(defaults);
      })
      .catch(() => {
        if (cancelled) return;
        setChatTemplateAvailable(false);
        setModelGenerationDefaults(DEFAULT_GENERATION_PARAMS);
        setGenerationParams(DEFAULT_GENERATION_PARAMS);
      });
    return () => {
      cancelled = true;
    };
  }, [state.weightProvider]);

  const updateGenerationParam = <K extends keyof GenerationParams>(key: K, value: GenerationParams[K]) => {
    setGenerationParams((prev) => ({ ...prev, [key]: value }));
  };

  // A different model can have completely different node ids (fewer/more
  // blocks, different architecture) — stale selection/view referencing the
  // old model's ids would otherwise crash ArchitectureGraph's breadcrumb.
  useEffect(() => {
    setSelectedId(null);
    setView({ kind: "architecture" });
    setSelectedTokenIndex(null);
    setBottomTab("tensor");
    setAttributionSource("A");
    inference.reset();
    promptB.reset();
    generation.reset();
  }, [state.model]);

  // Per-node activation magnitude (L2 norm) from the last run — computed
  // once here and shared by the model tree's per-row ticks and the
  // Inspector's "This run" section, rather than each recomputing it.
  const activationMagnitudeById = useMemo(() => {
    const result = inference.state.result;
    if (!result) return undefined;
    const map: Record<string, number> = {};
    for (const nodeId in result.activations) map[nodeId] = computeActivationMagnitude(result.activations[nodeId]);
    return map;
  }, [inference.state.result]);

  if (state.status !== "ready" || !state.model || !state.weightProvider) {
    return (
      <div className="app-loader-screen">
        <div className="top-right-controls">
          <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
          <SettingsPanel
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            theme={theme}
            onThemeChange={setTheme}
            unloadOnHome={unloadOnHome}
            onUnloadOnHomeChange={setUnloadOnHome}
            modelsPerPage={modelsPerPage}
            onModelsPerPageChange={setModelsPerPage}
            showGpuStatus={showGpuStatus}
            onShowGpuStatusChange={setShowGpuStatus}
            maxNewTokens={maxNewTokens}
            onMaxNewTokensChange={setMaxNewTokens}
            maxNewTokensLimit={maxNewTokensLimit}
            generationParams={generationParams}
            onGenerationParamChange={updateGenerationParam}
            onResetGenerationDefaults={() => setGenerationParams(modelGenerationDefaults)}
          />
        </div>
        <ModelLoader
          status={state.status}
          error={state.error}
          onLoad={load}
          loadProgress={state.loadProgress}
          modelsPerPage={modelsPerPage}
          showGpuStatus={showGpuStatus}
        />
      </div>
    );
  }

  const model = state.model;
  const selectedNode = selectedId ? model.nodes[selectedId] ?? null : null;
  // Defends against the one render where a just-finished model load has
  // landed but `view`/`selectedId` haven't been reset to match yet (the
  // effect above runs after this render, not before it) — without this, a
  // stale blockId from the previous model would crash ArchitectureGraph.
  const safeView: GraphView = view.kind === "block" && !model.nodes[view.blockId] ? { kind: "architecture" } : view;

  // Shared by the graph's double-click-to-expand gesture and the tree's
  // double-click on any row: jump to whichever graph view actually contains
  // this node (its own block's detail view, or back out to the top-level
  // architecture view for a node — e.g. Logits — that isn't inside a block
  // at all) and select it there, instead of only handling the
  // transformer-block case and leaving other double-clicks a no-op.
  const navigateToNode = (nodeId: string) => {
    const blockId = containingBlockId(model, nodeId);
    setView(blockId ? { kind: "block", blockId } : { kind: "architecture" });
    setSelectedId(nodeId);
  };

  // A stale selectedTokenIndex from a previous, possibly-longer prompt would
  // otherwise silently point past the new run's token count (App only
  // resets it on model change, not on every re-run) — clear it so every
  // panel's `?? tokenIds.length - 1` default kicks in fresh for the new
  // result, matching "just ran, so show me the last position" intuition.
  const runPromptA = (prompt: string) => {
    setSelectedTokenIndex(null);
    setPredictionCollapsed(false);
    inference.run(prompt);
  };

  const runPromptB = (prompt: string) => {
    setSelectedTokenIndexB(null);
    setPredictionCollapsed(false);
    promptB.run(prompt);
  };

  const runGeneration = (prompt: string) => {
    setPredictionCollapsed(false);
    generation.generate(prompt, { maxNewTokens: effectiveMaxNewTokens, ...generationParams }, generationMode);
  };

  // Drops into the same inspection UI (Prediction panel, tree, graph, every
  // bottom tab) at a specific generated step, by re-running inference on
  // the token sequence up to and including it — the prompt's own tokens
  // plus however many of the streamed continuation this step covers.
  const inspectGenerationStep = (tokenIndex: number) => {
    const prefix = [...generation.state.promptTokenIds, ...generation.state.tokens.slice(0, tokenIndex + 1).map((t) => t.tokenId)];
    setSelectedTokenIndex(null);
    setPredictionCollapsed(false);
    inference.runTokenIds(prefix);
  };

  // "Apply" a Prediction-panel row: same idea as inspectGenerationStep
  // above (truncate-and-append, then re-run), just sourced from a
  // *predicted* token (topKFromLogits) rather than one already generated —
  // this is how a prediction becomes part of the actual sequence. Each
  // prompt keeps its own tokenIds/selection, so applying a Prompt A
  // prediction never touches Prompt B's sequence and vice versa.
  const applyPredictionA = (tokenIndex: number, tokenId: number) => {
    const prefix = [...inference.state.result!.tokenIds.slice(0, tokenIndex + 1), tokenId];
    setSelectedTokenIndex(null);
    setPredictionCollapsed(false);
    setPromptAText(state.tokenizer!.decode(prefix));
    inference.runTokenIds(prefix);
  };

  const applyPredictionB = (tokenIndex: number, tokenId: number) => {
    const prefix = [...promptB.state.result!.tokenIds.slice(0, tokenIndex + 1), tokenId];
    setSelectedTokenIndexB(null);
    setPredictionCollapsed(false);
    setPromptBText(state.tokenizer!.decode(prefix));
    promptB.runTokenIds(prefix);
  };

  // Switching tabs while the bottom panel is collapsed should actually show
  // the tab, not just change which one is "active" behind a collapsed strip
  // — every place that jumps to a specific bottom tab (the tab bar itself,
  // Inspector's quick actions, Prediction panel's "Why?" link) goes through
  // this instead of setBottomTab directly.
  const selectBottomTab = (tab: BottomTab) => {
    setBottomTab(tab);
    setBottomCollapsed(false);
  };

  const requestTensorSource = (value: "weights" | "activations") => {
    selectBottomTab("tensor");
    setTensorSourceRequest({ value, nonce: Date.now() });
  };

  // Inspector's "This run" input/output links — jumps to Tensor Explorer's
  // Input/Output tab, pre-selected to whichever side (and, for an input,
  // which upstream source) the user actually clicked.
  const requestTensorIO = (io: "input" | "output", sourceId?: string) => {
    selectBottomTab("tensor");
    setTensorSourceRequest({ value: "io", io, sourceId, nonce: Date.now() });
  };

  const viewWhy = (source: "A" | "B") => {
    setAttributionSource(source);
    selectBottomTab("attribution");
  };

  const hasResult = inference.state.status === "ready" && !!inference.state.result;
  const hasResultB = compareEnabled && promptB.state.status === "ready" && !!promptB.state.result;
  // Deliberately looser than hasResult/hasResultB above (no status ===
  // "ready" check) — those two still gate the bottom-tab analyses
  // (LogitLens/Attribution/Experiment), which should wait for a genuinely
  // fresh result. The token chips and prediction-panels-row, though,
  // should just keep showing the previous result while a re-run (e.g. from
  // Apply) is in flight and swap to the new one once it lands, rather than
  // blanking out and popping back in — useInference now keeps `result`
  // populated through "running" specifically to make that possible.
  const showPredictionA = !!inference.state.result;
  const showPredictionB = compareEnabled && !!promptB.state.result;
  // Logit Lens only re-projects tensors an already-run capture already has
  // (see packages/interpretability's computeLogitLens) — it needs a real
  // ActivationCapture, not interventions, so it's ready as soon as any
  // runInference exists. Token Attribution and Experiment both actually
  // call runInference with a non-empty `interventions`, so they need an
  // adapter that's confirmed to do something with those, not just accept
  // the call.
  const logitLensEnabled = hasResult && !!state.adapter?.runInference;
  const interventionTabsEnabled = hasResult && !!state.adapter?.supportsInterventions;
  const currentModelId = state.source?.kind === "backend" ? state.source.modelId : undefined;

  // "Home" (settings.unloadOnHome off, the default) is pure client-side
  // navigation — the model stays resident on the GPU, so a refresh or
  // picking it again from the catalog resumes/re-serves it instantly via
  // the same fast path useModel's own resume-on-mount effect uses. With
  // the setting on, this frees GPU memory immediately instead, at the
  // cost of the next load (here or after a refresh) taking as long as the
  // first one again. Awaits the unload before navigating so a same-tab
  // "load a different model" click right after can't race it — the
  // backend serializes load/unload internally, but there's no reason to
  // rely on request-arrival-order semantics when just waiting is simple.
  const goHome = async () => {
    if (unloadOnHome && currentModelId) {
      setHomeBusy(true);
      try {
        await unloadModel(currentModelId);
      } catch {
        // Best-effort: still navigate home even if the unload call
        // itself failed (e.g. backend hiccup) — the model may still be
        // resident, which just means the next load resumes it instead
        // of reloading, not a broken state.
      } finally {
        setHomeBusy(false);
      }
    }
    reset();
  };

  // Derived, not a separate stored flag: "max frame" just means every
  // surrounding panel is currently collapsed. The prediction panel only
  // counts when it's actually rendered (a result exists) — otherwise its
  // stored collapse preference shouldn't stop the other three panels from
  // reading as "already maximized".
  const isMaxFrame = treeCollapsed && inspectorCollapsed && bottomCollapsed && (!hasResult || predictionCollapsed);
  const toggleMaxFrame = () => {
    const next = !isMaxFrame;
    setTreeCollapsed(next);
    setInspectorCollapsed(next);
    setBottomCollapsed(next);
    setPredictionCollapsed(next);
  };

  // Drag-to-resize for the bottom panel. Height is tracked in state (not
  // just read from the DOM after drag) so it can be persisted; the max
  // clamp is computed live off window.innerHeight rather than a fixed
  // constant so it stays sane across window resizes.
  const handleBottomResizeStart = (e: ReactMouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = bottomHeight;
    setResizingBottom(true);
    const onMove = (moveEvent: MouseEvent) => {
      const maxHeight = Math.max(BOTTOM_PANEL_MIN_HEIGHT, window.innerHeight - BOTTOM_PANEL_TOP_RESERVE);
      const next = startHeight + (startY - moveEvent.clientY);
      setBottomHeight(Math.min(maxHeight, Math.max(BOTTOM_PANEL_MIN_HEIGHT, Math.round(next))));
    };
    const onUp = () => {
      setResizingBottom(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Drag-to-resize for the tree and inspector panels — same pattern as the
  // bottom panel above, mirrored horizontally. The max clamp for each also
  // accounts for the *other* side panel's current width (full or collapsed
  // to its 36px strip), so dragging one out doesn't silently starve the
  // graph pane below GRAPH_PANEL_MIN_WIDTH just because the other panel
  // also happens to be wide.
  const handleTreeResizeStart = (e: ReactMouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = treeWidth;
    setResizingTree(true);
    const onMove = (moveEvent: MouseEvent) => {
      const inspectorSpace = inspectorCollapsed ? 36 : inspectorWidth;
      const maxWidth = Math.max(TREE_PANEL_MIN_WIDTH, window.innerWidth - inspectorSpace - GRAPH_PANEL_MIN_WIDTH - PANE_CHROME_ALLOWANCE);
      const next = startWidth + (moveEvent.clientX - startX);
      setTreeWidth(Math.min(maxWidth, Math.max(TREE_PANEL_MIN_WIDTH, Math.round(next))));
    };
    const onUp = () => {
      setResizingTree(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const handleInspectorResizeStart = (e: ReactMouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = inspectorWidth;
    setResizingInspector(true);
    const onMove = (moveEvent: MouseEvent) => {
      const treeSpace = treeCollapsed ? 36 : treeWidth;
      const maxWidth = Math.max(INSPECTOR_PANEL_MIN_WIDTH, window.innerWidth - treeSpace - GRAPH_PANEL_MIN_WIDTH - PANE_CHROME_ALLOWANCE);
      const next = startWidth + (startX - moveEvent.clientX);
      setInspectorWidth(Math.min(maxWidth, Math.max(INSPECTOR_PANEL_MIN_WIDTH, Math.round(next))));
    };
    const onUp = () => {
      setResizingInspector(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Status-footer fields — all read from state this component already
  // holds, computed here rather than inside the JSX so the render below
  // stays about layout, not derivation.
  const runningA = inference.state.status === "running";
  const runningB = compareEnabled && promptB.state.status === "running";
  const generating = generation.state.status === "streaming";
  const footerBusy = analysisBusy || runningA || runningB || generating;
  const footerHasError = inference.state.status === "error" || (compareEnabled && promptB.state.status === "error") || generation.state.status === "error";
  // Only generation has a meaningful fraction to show (a fixed token cap
  // streamed one at a time) — a plain forward pass is one atomic backend
  // call with no partial progress to report.
  const footerProgress = generating ? { completed: generation.state.tokens.length, total: effectiveMaxNewTokens } : undefined;
  const footerRunningLabel = generating
    ? t("footer.generating")
    : runningA && runningB
      ? t("footer.runningBoth")
      : runningB
        ? t("footer.runningB")
        : t("inference.running");
  const footerStatus: { label: string; tone: "ready" | "busy" | "error" } = footerHasError
    ? { label: t("footer.error"), tone: "error" }
    : footerBusy
      ? { label: analysisBusy ? t("footer.analyzing") : footerRunningLabel, tone: "busy" }
      : { label: t("footer.ready"), tone: "ready" };

  const footerBlockId = selectedId ? containingBlockId(model, selectedId) : safeView.kind === "block" ? safeView.blockId : null;
  const footerBlockLabel = footerBlockId ? model.nodes[footerBlockId]?.name : t("footer.architectureView");
  const footerTitle = selectedNode && footerBlockId !== selectedNode.id ? `${footerBlockLabel} › ${selectedNode.name}` : (selectedNode?.name ?? footerBlockLabel);
  const footerTensor = selectedId ? inference.state.result?.activations[selectedId] : undefined;
  const footerShape = footerTensor
    ? `[${footerTensor.shape.join(" × ")}]`
    : selectedNode?.parameters[0]
      ? `[${selectedNode.parameters[0].logicalShape.join(" × ")}]`
      : "—";
  const footerDtype = footerTensor?.dtype ?? selectedNode?.parameters[0]?.dtype ?? "—";
  const footerNorm = selectedId ? activationMagnitudeById?.[selectedId] : undefined;

  return (
    <div
      className={
        "app" +
        (analysisBusy ? " app-busy" : "") +
        (resizingBottom ? " app-resizing-panel" : "") +
        (resizingTree || resizingInspector ? " app-resizing-col-panel" : "")
      }
    >
      <ModelInfoBar model={model} />
      <div className="top-right-controls">
        <div className="control-group">
          <button className="go-home" onClick={goHome} disabled={homeBusy} title={t("app.backToHome")}>
            {t("app.backToHome")}
          </button>
          <button className="load-different" onClick={() => setLoadModelOpen((v) => !v)}>
            {t("app.loadDifferentModel")}
          </button>
          <LoadModelPanel
            open={loadModelOpen}
            onClose={() => setLoadModelOpen(false)}
            status={state.status}
            error={state.error}
            excludeModelId={currentModelId}
            onLoad={load}
            modelsPerPage={modelsPerPage}
            showGpuStatus={showGpuStatus}
          />
        </div>
        <div className="control-group">
          <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
          <SettingsPanel
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            theme={theme}
            onThemeChange={setTheme}
            unloadOnHome={unloadOnHome}
            onUnloadOnHomeChange={setUnloadOnHome}
            modelsPerPage={modelsPerPage}
            onModelsPerPageChange={setModelsPerPage}
            showGpuStatus={showGpuStatus}
            onShowGpuStatusChange={setShowGpuStatus}
            maxNewTokens={maxNewTokens}
            onMaxNewTokensChange={setMaxNewTokens}
            maxNewTokensLimit={maxNewTokensLimit}
            generationParams={generationParams}
            onGenerationParamChange={updateGenerationParam}
            onResetGenerationDefaults={() => setGenerationParams(modelGenerationDefaults)}
          />
        </div>
      </div>
      <div className="prompt-section">
        <div className="prompt-section-header">
          <button
            type="button"
            className="prompt-section-collapse-btn"
            onClick={() => setPromptSectionCollapsed((v) => !v)}
            title={promptSectionCollapsed ? t("app.expandPanel") : t("app.collapsePanel")}
          >
            {promptSectionCollapsed ? "▸" : "▾"}
          </button>
          <span className="prompt-section-title">{t("inference.sectionTitle")}</span>
        </div>
        {!promptSectionCollapsed && (
          <>
            <InferencePanel
              supported={!!state.tokenizer}
              state={inference.state}
              prompt={promptAText}
              onPromptChange={setPromptAText}
              onRun={runPromptA}
              selectedTokenIndex={selectedTokenIndex}
              onSelectToken={setSelectedTokenIndex}
              compareEnabled={compareEnabled}
              onToggleCompare={() => setCompareEnabled((v) => !v)}
              promptBState={promptB.state}
              promptBText={promptBText}
              onPromptBTextChange={setPromptBText}
              onRunB={runPromptB}
              selectedTokenIndexB={selectedTokenIndexB}
              onSelectTokenB={setSelectedTokenIndexB}
              generationState={generation.state}
              onGenerate={runGeneration}
              onStopGeneration={generation.stop}
              onInspectStep={inspectGenerationStep}
              generationMode={generationMode}
              onToggleGenerationMode={() => setGenerationMode((m) => (m === "chat" ? "raw" : "chat"))}
              chatTemplateAvailable={chatTemplateAvailable}
              maxNewTokens={effectiveMaxNewTokens}
            />
            {showPredictionA && state.tokenizer && (
              <div className="prediction-panels-row">
                <PredictionPanel
                  result={inference.state.result!}
                  tokenizer={state.tokenizer}
                  selectedTokenIndex={selectedTokenIndex}
                  onViewWhy={() => viewWhy("A")}
                  onApplyPrediction={applyPredictionA}
                  collapsed={predictionCollapsed}
                  onToggleCollapsed={() => setPredictionCollapsed((v) => !v)}
                  promptLabel={showPredictionB ? t("inference.promptA") : undefined}
                />
                {showPredictionB && (
                  <PredictionPanel
                    result={promptB.state.result!}
                    tokenizer={state.tokenizer}
                    selectedTokenIndex={selectedTokenIndexB}
                    onViewWhy={() => viewWhy("B")}
                    onApplyPrediction={applyPredictionB}
                    collapsed={predictionCollapsed}
                    onToggleCollapsed={() => setPredictionCollapsed((v) => !v)}
                    promptLabel={t("inference.promptB")}
                  />
                )}
              </div>
            )}
          </>
        )}
      </div>
      <div className="app-body">
        <aside
          className={"pane pane-tree" + (treeCollapsed ? " collapsed" : "") + (resizingTree ? " resizing" : "")}
          style={treeCollapsed ? undefined : { width: treeWidth }}
        >
          <div className="pane-header">
            {!treeCollapsed && <span className="pane-header-title">{t("app.modelTree")}</span>}
            <button className="pane-collapse-btn" onClick={() => setTreeCollapsed((v) => !v)} title={treeCollapsed ? t("app.expandTree") : t("app.collapseTree")}>
              {treeCollapsed ? "›" : "‹"}
            </button>
          </div>
          {treeCollapsed ? (
            <span className="pane-vertical-label">{t("app.modelTree")}</span>
          ) : (
            <div className="pane-tree-body">
              <ModelTree
                model={model}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onNavigate={navigateToNode}
                activationMagnitudeById={activationMagnitudeById}
              />
            </div>
          )}
        </aside>
        {!treeCollapsed && (
          <div
            className={"pane-resize-handle pane-resize-handle-vertical" + (resizingTree ? " resizing" : "")}
            onMouseDown={handleTreeResizeStart}
            title={t("app.resizePanel")}
          />
        )}
        <main className="pane pane-graph">
          <ArchitectureGraph
            model={model}
            view={safeView}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onEnterBlock={navigateToNode}
            onExitBlock={() => setView({ kind: "architecture" })}
            isMaxFrame={isMaxFrame}
            onToggleMaxFrame={toggleMaxFrame}
            onZoomChange={setZoomPercent}
          />
        </main>
        {!inspectorCollapsed && (
          <div
            className={"pane-resize-handle pane-resize-handle-vertical" + (resizingInspector ? " resizing" : "")}
            onMouseDown={handleInspectorResizeStart}
            title={t("app.resizePanel")}
          />
        )}
        <aside
          className={"pane pane-inspector" + (inspectorCollapsed ? " collapsed" : "") + (resizingInspector ? " resizing" : "")}
          style={inspectorCollapsed ? undefined : { width: inspectorWidth }}
        >
          <div className="pane-header">
            <button className="pane-collapse-btn" onClick={() => setInspectorCollapsed((v) => !v)} title={inspectorCollapsed ? t("app.expandInspector") : t("app.collapseInspector")}>
              {inspectorCollapsed ? "‹" : "›"}
            </button>
            {!inspectorCollapsed && <span className="pane-header-title">{t("app.inspector")}</span>}
          </div>
          {inspectorCollapsed ? (
            <span className="pane-vertical-label">{t("app.inspector")}</span>
          ) : (
            <div className="pane-inspector-body">
              <Inspector
                model={model}
                node={selectedNode}
                activationShape={selectedId ? inference.state.result?.activations[selectedId]?.shape : undefined}
                activationMagnitude={selectedId ? activationMagnitudeById?.[selectedId] : undefined}
                onViewActivation={() => requestTensorSource("activations")}
                onViewWeights={() => requestTensorSource("weights")}
                onViewInput={(sourceId) => requestTensorIO("input", sourceId)}
                onViewOutput={() => requestTensorIO("output")}
                onDeselect={() => setSelectedId(null)}
                inferenceResult={inference.state.result}
                tokenizer={state.tokenizer}
                elapsedMs={inference.state.elapsedMs}
              />
            </div>
          )}
        </aside>
      </div>
      <section
        className={"pane pane-tensor" + (bottomCollapsed ? " collapsed" : "") + (resizingBottom ? " resizing" : "")}
        style={bottomCollapsed ? undefined : { height: bottomHeight }}
      >
        {!bottomCollapsed && (
          <div
            className="pane-tensor-resize-handle"
            onMouseDown={handleBottomResizeStart}
            title={t("app.resizePanel")}
          />
        )}
        <div className="bottom-tabs">
          <button className={bottomTab === "tensor" ? "active" : ""} onClick={() => selectBottomTab("tensor")}>
            {t("app.tensorExplorer")}
          </button>
          <button className={bottomTab === "logitlens" ? "active" : ""} disabled={!logitLensEnabled} onClick={() => selectBottomTab("logitlens")} title={!logitLensEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.logitLens")}
          </button>
          <button className={bottomTab === "attribution" ? "active" : ""} disabled={!interventionTabsEnabled} onClick={() => selectBottomTab("attribution")} title={!interventionTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.tokenAttribution")}
          </button>
          <button className={bottomTab === "experiment" ? "active" : ""} disabled={!interventionTabsEnabled} onClick={() => selectBottomTab("experiment")} title={!interventionTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.experiment")}
          </button>
          <span className="bottom-tabs-spacer" />
          <button className="bottom-collapse-btn" onClick={() => setBottomCollapsed((v) => !v)} title={bottomCollapsed ? t("app.expandPanel") : t("app.collapsePanel")}>
            {bottomCollapsed ? "▴" : "▾"}
          </button>
        </div>

        {!bottomCollapsed && bottomTab === "tensor" && (
          <TensorExplorer
            model={model}
            weightProvider={state.weightProvider}
            selectedNode={selectedNode}
            inference={inference.state}
            selectedTokenIndex={selectedTokenIndex}
            promptBInference={promptB.state}
            sourceRequest={tensorSourceRequest}
          />
        )}
        {!bottomCollapsed && bottomTab === "logitlens" && logitLensEnabled && state.tokenizer && (
          <LogitLensPanel
            model={model}
            weightProvider={state.weightProvider}
            capture={inference.state.result!}
            promptBCapture={hasResultB ? promptB.state.result : undefined}
            tokenizer={state.tokenizer}
            selectedTokenIndex={selectedTokenIndex}
            onSelectToken={setSelectedTokenIndex}
            onBusyChange={setAnalysisBusy}
          />
        )}
        {!bottomCollapsed && bottomTab === "attribution" && interventionTabsEnabled && state.tokenizer && (
          <TokenAttributionPanel
            model={model}
            weightProvider={state.weightProvider}
            adapter={state.adapter!}
            tokenIds={inference.state.result!.tokenIds}
            promptBTokenIds={hasResultB ? promptB.state.result!.tokenIds : undefined}
            tokenizer={state.tokenizer}
            selectedTokenIndex={selectedTokenIndex}
            source={attributionSource}
            onSourceChange={setAttributionSource}
            onSelectNode={setSelectedId}
            onBusyChange={setAnalysisBusy}
          />
        )}
        {!bottomCollapsed && bottomTab === "experiment" && interventionTabsEnabled && state.tokenizer && (
          <ExperimentPanel
            model={model}
            weightProvider={state.weightProvider}
            adapter={state.adapter!}
            tokenizer={state.tokenizer}
            selectedNode={selectedNode}
            mainTokenIds={inference.state.result!.tokenIds}
            mainResult={inference.state.result!}
            promptBResult={promptB.state.result ?? null}
            onBusyChange={setAnalysisBusy}
          />
        )}
      </section>
      <div className="status-footer">
        <span className={"status-footer-dot status-footer-dot-" + footerStatus.tone} />
        <span className="status-footer-item status-footer-status">{footerStatus.label}</span>
        {footerBusy && footerProgress && (
          <span
            className="status-footer-item status-footer-progress"
            title={t("footer.tokenProgress").replace("{completed}", String(footerProgress.completed)).replace("{total}", String(footerProgress.total))}
          >
            <span className="status-footer-progress-track">
              <span className="status-footer-progress-fill" style={{ width: `${Math.round((footerProgress.completed / footerProgress.total) * 100)}%` }} />
            </span>
            <span className="status-footer-progress-label">
              {footerProgress.completed}/{footerProgress.total}
            </span>
          </span>
        )}
        <span className="status-footer-sep" />
        <span className="status-footer-item status-footer-title" title={footerTitle}>
          {footerTitle}
        </span>
        <span className="status-footer-sep" />
        <span className="status-footer-item">{footerShape}</span>
        <span className="status-footer-item">{footerDtype}</span>
        {footerNorm !== undefined && <span className="status-footer-item">{t("footer.norm").replace("{value}", footerNorm.toFixed(4))}</span>}
        <span className="status-footer-sep" />
        <span className="status-footer-item">{t("footer.zoom").replace("{percent}", String(zoomPercent))}</span>
        <span className="status-footer-spacer" />
        <span className="status-footer-item status-footer-compute" title={t("footer.gpuTooltip")}>
          {t("footer.gpu")}
        </span>
      </div>
    </div>
  );
}
