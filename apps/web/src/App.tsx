import { useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { Model, Tensor } from "@aperture/model-ir";
import { unloadModel } from "@aperture/api-client";
import { useModel } from "./useModel.js";
import { useInference } from "./useInference.js";
import { useGeneration } from "./useGeneration.js";
import { useLocalStorageState } from "./useLocalStorageState.js";
import { useTheme } from "./components/ThemeSwitcher.js";
import { useTranslation } from "./components/LanguageContext.js";
import { SettingsButton, SettingsPanel } from "./components/SettingsPanel.js";
import { ModelLoader } from "./components/ModelLoader.js";
import { LoadModelPanel } from "./components/LoadModelPanel.js";
import { ModelInfoBar } from "./components/ModelInfoBar.js";
import { ModelTree } from "./components/ModelTree.js";
import { ArchitectureGraph, type GraphView } from "./components/ArchitectureGraph.js";
import { Inspector } from "./components/Inspector.js";
import { TensorExplorer } from "./components/TensorExplorer.js";
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
  const [homeBusy, setHomeBusy] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<GraphView>({ kind: "architecture" });
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [bottomTab, setBottomTab] = useState<BottomTab>("tensor");
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [tensorSourceRequest, setTensorSourceRequest] = useState<{ value: "weights" | "activations"; nonce: number } | null>(null);
  /** Which prompt's tokens Token Attribution attributes — controlled here (rather than as the panel's own local state) so the Prediction panel's "Why?" link can request the matching side instead of always landing back on Prompt A. */
  const [attributionSource, setAttributionSource] = useState<"A" | "B">("A");
  const [treeCollapsed, setTreeCollapsed] = useLocalStorageState("panel:tree-collapsed", false);
  const [inspectorCollapsed, setInspectorCollapsed] = useLocalStorageState("panel:inspector-collapsed", false);
  const [bottomCollapsed, setBottomCollapsed] = useLocalStorageState("panel:bottom-collapsed", false);
  const [bottomHeight, setBottomHeight] = useLocalStorageState("panel:bottom-height", BOTTOM_PANEL_DEFAULT_HEIGHT);
  const [resizingBottom, setResizingBottom] = useState(false);
  const [predictionCollapsed, setPredictionCollapsed] = useLocalStorageState("panel:prediction-collapsed", false);

  const inference = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);
  const promptB = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);
  const generation = useGeneration(state.weightProvider, state.tokenizer);

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
          <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} theme={theme} onThemeChange={setTheme} unloadOnHome={unloadOnHome} onUnloadOnHomeChange={setUnloadOnHome} />
        </div>
        <ModelLoader status={state.status} error={state.error} onLoad={load} loadProgress={state.loadProgress} />
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

  const runGeneration = (prompt: string) => {
    setPredictionCollapsed(false);
    generation.generate(prompt, { maxNewTokens: 64, temperature: 0.7 });
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

  const viewWhy = (source: "A" | "B") => {
    setAttributionSource(source);
    selectBottomTab("attribution");
  };

  const hasResult = inference.state.status === "ready" && !!inference.state.result;
  const hasResultB = compareEnabled && promptB.state.status === "ready" && !!promptB.state.result;
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

  return (
    <div className={"app" + (analysisBusy ? " app-busy" : "") + (resizingBottom ? " app-resizing-panel" : "")}>
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
          />
        </div>
        <div className="control-group">
          <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
          <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} theme={theme} onThemeChange={setTheme} unloadOnHome={unloadOnHome} onUnloadOnHomeChange={setUnloadOnHome} />
        </div>
      </div>
      <InferencePanel
        supported={!!state.tokenizer}
        state={inference.state}
        onRun={runPromptA}
        selectedTokenIndex={selectedTokenIndex}
        onSelectToken={setSelectedTokenIndex}
        compareEnabled={compareEnabled}
        onToggleCompare={() => setCompareEnabled((v) => !v)}
        promptBState={promptB.state}
        onRunB={promptB.run}
        generationState={generation.state}
        onGenerate={runGeneration}
        onStopGeneration={generation.stop}
        onInspectStep={inspectGenerationStep}
      />
      {hasResult && state.tokenizer && (
        <div className="prediction-panels-row">
          <PredictionPanel
            result={inference.state.result!}
            tokenizer={state.tokenizer}
            selectedTokenIndex={selectedTokenIndex}
            onViewWhy={() => viewWhy("A")}
            collapsed={predictionCollapsed}
            onToggleCollapsed={() => setPredictionCollapsed((v) => !v)}
            promptLabel={hasResultB ? t("inference.promptA") : undefined}
          />
          {hasResultB && (
            <PredictionPanel
              result={promptB.state.result!}
              tokenizer={state.tokenizer}
              selectedTokenIndex={selectedTokenIndex}
              onViewWhy={() => viewWhy("B")}
              collapsed={predictionCollapsed}
              onToggleCollapsed={() => setPredictionCollapsed((v) => !v)}
              promptLabel={t("inference.promptB")}
            />
          )}
        </div>
      )}
      <div className="app-body">
        <aside className={"pane pane-tree" + (treeCollapsed ? " collapsed" : "")}>
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
          />
        </main>
        <aside className={"pane pane-inspector" + (inspectorCollapsed ? " collapsed" : "")}>
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
    </div>
  );
}
