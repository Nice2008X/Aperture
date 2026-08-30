// Model IR — the normalized representation every visualization and every
// model adapter agrees on. Nothing in here knows about GPT-2, Llama, or any
// specific architecture; adapters translate a real model INTO this shape.

/**
 * The vocabulary of node kinds the UI knows how to render.
 * Adding a new architecture should rarely require adding a new NodeType —
 * prefer composing existing types (e.g. RoPE can be a "custom" node with a
 * distinguishing `metadata.op`, or promoted here if it recurs across models).
 */
export type NodeType =
  | "model"
  | "block_group"
  | "input"
  | "embedding"
  | "positional_embedding"
  | "transformer_block"
  | "layer_norm"
  | "rms_norm"
  | "attention"
  | "q_projection"
  | "k_projection"
  | "v_projection"
  | "qkv_projection"
  | "output_projection"
  | "rope"
  | "ffn"
  | "linear"
  | "activation"
  | "elementwise_mul"
  | "residual"
  | "lm_head"
  | "output"
  | "moe_layer"
  | "router"
  | "expert"
  /** Fallback for a module the backend's generic graph builder doesn't recognize — keeps an unfamiliar architecture visible (with real shapes/params) instead of failing to load. */
  | "custom";

export interface TensorSpec {
  /** Dimension sizes; a string entry is a symbolic/dynamic dim, e.g. "sequence_length". */
  dims: Array<number | string>;
}

export interface ParameterRef {
  /** Fully-qualified name as it appears in the weight file (e.g. HF state-dict key). */
  name: string;
  /** Full shape of the underlying weight-file tensor (before any `slice` below is applied). */
  shape: number[];
  dtype: string;
  /** Element count / byte size of the full underlying tensor. */
  numElements: number;
  bytes: number;
  /** Which WeightProvider owns this parameter's actual data. */
  providerId: string;
  /**
   * Set when this ref stands for a sub-range of `name`'s tensor rather than
   * the whole thing — e.g. GPT-2 fuses Q/K/V into one c_attn matrix, so the
   * "Q projection" node's ParameterRef points at c_attn with a column slice.
   * Consumers should pass this straight through to WeightProvider.loadTensor.
   */
  slice?: TensorSlice;
  /** The effective shape after `slice` is applied (== shape when slice is absent). */
  logicalShape: number[];
}

export interface ModelNode {
  id: string;
  type: NodeType;
  name: string;

  inputs: TensorSpec[];
  outputs: TensorSpec[];

  /** Weights/biases owned directly by this node (empty for structural/container nodes). */
  parameters: ParameterRef[];

  /** Child node ids, for the logical hierarchy (Model -> Block -> Attention -> Q ...). */
  children: string[];
  parentId: string | null;

  metadata: Record<string, unknown>;
}

export interface ModelEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface ModelConfig {
  modelType: string;
  numLayers: number;
  numHeads: number;
  hiddenSize: number;
  intermediateSize: number;
  vocabSize: number;
  contextLength: number;
  /** Anything architecture-specific that doesn't fit the common fields above. */
  extra: Record<string, unknown>;
}

export interface Model {
  id: string;
  name: string;
  architecture: string;
  config: ModelConfig;

  inputs: TensorSpec[];
  outputs: TensorSpec[];

  /** Flat lookup of every node in the hierarchy, keyed by id. */
  nodes: Record<string, ModelNode>;
  /** Execution-order edges between node ids — the "what flows into what" graph. */
  edges: ModelEdge[];

  rootId: string;
}

// ---------------------------------------------------------------------------
// Weight access — deliberately separate from the graph above. A node knows
// *which* parameters it owns (name/shape/dtype); it does not hold the actual
// bytes. Those are fetched lazily through a WeightProvider, on demand.
// ---------------------------------------------------------------------------

export interface TensorSlice {
  /** Per-dimension [start, end) ranges. Omitted dims are taken in full. */
  ranges?: Array<{ start: number; end: number }>;
}

export interface Tensor {
  shape: number[];
  dtype: string;
  /** Always materialized as float64 for the UI's sake, regardless of source dtype. */
  data: Float64Array;
}

export interface WeightProvider {
  id: string;
  listParameters(): Promise<ParameterRef[]>;
  getParameterInfo(parameterId: string): Promise<ParameterRef>;
  loadTensor(parameterId: string, options?: TensorSlice): Promise<Tensor>;
}

// ---------------------------------------------------------------------------
// Model adapters — the only thing that has to change to support a new
// architecture. Everything above and everything in the UI stays fixed.
// ---------------------------------------------------------------------------

export type ModelSource =
  | { kind: "huggingface"; repo: string; revision?: string }
  | { kind: "local"; name: string; files: Record<string, ArrayBuffer> }
  /** A model the GPU backend already has on disk (data/models/<modelId>/) — see PLAN.md. */
  | { kind: "backend"; modelId: string; quantization?: "4bit" | "8bit" };

/** A human-readable label for a source — the HF repo id, the display name chosen when local files were picked, or the backend's model id. */
export function modelSourceLabel(source: ModelSource): string {
  if (source.kind === "huggingface") return source.repo;
  if (source.kind === "local") return source.name;
  return source.modelId;
}

export interface ModelMetadata {
  architecture: string;
  config: ModelConfig;
  /** name -> {shape, dtype}, taken from the weight file's header, no tensor data loaded. */
  weightIndex: Record<string, { shape: number[]; dtype: string }>;
  source: ModelSource;
  /**
   * Set when an adapter's backend already computed the full graph server-side
   * (see @aperture/api-client's BackendAdapter) — buildGraph() can just
   * return this instead of re-deriving nodes/edges from weightIndex the way
   * a browser-side adapter has to.
   */
  prebuiltModel?: Model;
  /** Raw bytes backing the WeightProvider this metadata will build, if already fetched. */
  weightsBuffer?: ArrayBuffer;
}

/**
 * The result of an actual forward pass — Mode B from the project notes
 * (Runtime -> Input -> Forward pass -> Activations -> Visualization), kept
 * fully separate from Mode A (static architecture/weight visualization,
 * which needs none of this).
 */
export interface ActivationCapture {
  tokenIds: number[];
  tokens: string[];
  /** nodeId -> the tensor that actually flowed through that node during this forward pass. */
  activations: Record<string, Tensor>;
  /** nodeId -> per-head attention weights [numHeads, queryPos, keyPos], softmax already applied. Attention nodes only. */
  attentionWeights: Record<string, Tensor>;
  /** moe_layer nodeId -> softmax'd top-k gate weights [sequence_length, numExpertsPerTok]. MoE layers only. */
  routerWeights?: Record<string, Tensor>;
  /** moe_layer nodeId -> selected expert ids [sequence_length, numExpertsPerTok], same token/row order as routerWeights. MoE layers only. */
  expertAssignment?: Record<string, Tensor>;
  /** [sequence_length, vocabSize] */
  logits: Tensor;
}

/**
 * A single edit to the forward pass, applied at the point a node's
 * activation is computed and threaded through everything downstream — this
 * is the mechanism behind ablation, cross-prompt activation patching, and
 * occlusion-based attribution (they're all just different Interventions).
 *
 * Deliberately simpler than "arbitrary tensor surgery": every op targets one
 * node's activation, optionally narrowed to one token position and/or one
 * attention head. That covers "zero this component", "zero this attention
 * head", and "replace this activation with the one captured from a
 * different run" — the three things the interpretability workflows in this
 * app actually need — without a general tensor-patching DSL.
 */
export interface Intervention {
  /** Which node's activation to intervene on. */
  nodeId: string;
  operation: "zero" | "zero_head" | "scale" | "replace";
  /** Restrict to one sequence position; omitted = every position. */
  tokenIndex?: number;
  /** Required for "zero_head" — only meaningful on an "attention" node. */
  headIndex?: number;
  /** Required for "scale". */
  scale?: number;
  /** Required for "replace" — typically another run's activation for the same nodeId. */
  replacementValue?: Tensor;
}

/**
 * One step of an adapter's loadMetadata() progress — deliberately generic
 * (a free-form `phase` name, not an enum) since what "loading" breaks down
 * into is adapter-specific (a backend adapter's phases come from whatever
 * the server reports; a hypothetical future adapter might have none at
 * all, or completely different ones). The UI treats an unrecognized phase
 * generically rather than requiring an exhaustive list here.
 */
export interface LoadProgressEvent {
  phase: string;
  /** Human-ish sub-label, if the adapter has one (e.g. which file/shard). */
  detail?: string;
  /** Progress within this phase, if known (e.g. parameters materialized so far). */
  current?: number;
  total?: number;
}

export interface ModelAdapter {
  id: string;
  displayName: string;
  canLoad(source: ModelSource, metadata?: { architectures?: string[]; model_type?: string }): boolean;
  /** `onProgress`, when given, is called zero or more times before this resolves — a real checkpoint can take tens of seconds to materialize, which the caller (see apps/web's useModel) surfaces as a progress bar rather than leaving the loading screen with no feedback. */
  loadMetadata(source: ModelSource, onProgress?: (event: LoadProgressEvent) => void): Promise<ModelMetadata>;
  buildGraph(metadata: ModelMetadata): Model;
  getWeightProvider(metadata: ModelMetadata): WeightProvider;
  /**
   * Optional: runs a real forward pass over the weights this adapter's
   * WeightProvider exposes, capturing intermediate tensors. Computing actual
   * model math is inherently architecture-specific — unlike everything else
   * in this interface, there's no way to make this generic — so an adapter
   * that only supports Mode A (static visualization) can simply omit it.
   *
   * `interventions`, when given, are applied as each targeted node's
   * activation is computed, so everything downstream sees the edited value
   * — this is what makes ablation/patching an actual re-execution rather
   * than a cosmetic overlay on the original run's numbers.
   */
  runInference?(model: Model, weightProvider: WeightProvider, tokenIds: number[], interventions?: Intervention[]): Promise<ActivationCapture>;
  /**
   * Whether a non-empty `interventions` array passed to runInference()
   * actually does something, rather than being ignored or rejected. An
   * adapter can support a plain forward pass (PredictionPanel's needs)
   * before it supports interventions (Experiment/Token Attribution's
   * needs, and — for its per-layer *fetches*, not interventions
   * themselves — Logit Lens's) — the UI gates those three panels on this
   * flag specifically, separate from `!!runInference`, so that gap is
   * representable instead of forcing an all-or-nothing rollout.
   */
  supportsInterventions?: boolean;
  /**
   * Optional: scores many independent intervention variants against one
   * target sequence position — what occlusion-based attribution (Token
   * Attribution, Head Attribution) needs — in as few real forward passes
   * as practical, rather than one runInference() call per variant.
   * `variants[i]` is applied only to that variant (an empty array is a
   * valid "no intervention" variant, used for a baseline); returns one
   * logits row (length vocabSize) per variant, at `predictIndex`.
   *
   * An adapter without this falls back to one runInference() call per
   * variant (see packages/interpretability's getVariantLogitRows) —
   * correct either way, just far slower once each pass is a real
   * computation rather than in-browser math a tiny debug model made free.
   */
  runAttributionSweep?(model: Model, weightProvider: WeightProvider, tokenIds: number[], predictIndex: number, variants: Intervention[][]): Promise<Float64Array[]>;
}

// ---------------------------------------------------------------------------
// Small helpers shared by every adapter/consumer.
// ---------------------------------------------------------------------------

/**
 * Sums each *underlying* weight-file tensor exactly once. Necessary because
 * several nodes can share one tensor via `slice` (e.g. GPT-2's fused c_attn
 * backs its Q, K, and V projection nodes) — naively summing every node's
 * ParameterRef would count that tensor three times over.
 */
export function uniqueParameters(model: Model): ParameterRef[] {
  const seen = new Map<string, ParameterRef>();
  for (const node of Object.values(model.nodes)) {
    for (const p of node.parameters) if (!seen.has(p.name)) seen.set(p.name, p);
  }
  return [...seen.values()];
}

export function totalParameterCount(model: Model): number {
  return uniqueParameters(model).reduce((sum, p) => sum + p.numElements, 0);
}

export function totalParameterBytes(model: Model): number {
  return uniqueParameters(model).reduce((sum, p) => sum + p.bytes, 0);
}

export function getChildren(model: Model, nodeId: string): ModelNode[] {
  const node = model.nodes[nodeId];
  if (!node) return [];
  return node.children.map((id) => model.nodes[id]).filter((n): n is ModelNode => !!n);
}

export function dtypeSize(dtype: string): number {
  switch (dtype) {
    case "F64":
    case "I64":
      return 8;
    case "F32":
    case "I32":
      return 4;
    case "F16":
    case "BF16":
    case "I16":
      return 2;
    case "I8":
    case "U8":
    case "BOOL":
      return 1;
    default:
      return 4;
  }
}

export function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}
