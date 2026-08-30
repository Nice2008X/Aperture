import type { ReactNode } from "react";
import type { Model, ModelNode } from "@aperture/model-ir";
import { componentRegistry } from "../registry.js";

function formatDims(dims: Array<number | string>): string {
  return `[${dims.join(", ")}]`;
}

/** True if `ancestorId` is a strict ancestor of `node` (walking `parentId`, not including `node` itself). */
function isAncestor(model: Model, ancestorId: string, node: ModelNode): boolean {
  let cur = node.parentId;
  while (cur) {
    if (cur === ancestorId) return true;
    cur = model.nodes[cur]?.parentId ?? null;
  }
  return false;
}

interface InputSource {
  label: string;
  isBlockBoundary: boolean;
}

/**
 * Which upstream node(s) actually feed this one, and — where it's safe to
 * say so — how they combine. Two things make this trickier than just
 * reading `model.edges`:
 *
 * - A container's own id is sometimes used as an edge source purely so the
 *   *collapsed* graph view stays connected once its children are hidden
 *   (e.g. GPT-2's `edge(ffnContainer, fc)` alongside the real `edge(ln2,
 *   fc)` — the same pattern `buildLevel2Graph` already has to work around).
 *   Those are dropped whenever a real sibling/leaf edge already covers the
 *   connection. When a container edge is the *only* incoming edge, though
 *   (e.g. `edge(block, firstNorm)`), it's genuine — that's the block's own
 *   boundary — and is kept, labeled as such rather than by the container's
 *   own name (matching the graph's "Block Input" convention).
 * - Once real edges are settled, the combining operator is only asserted
 *   when it's actually knowable in general: "addition" category or a
 *   "skip"-labeled edge means +, "elementwise" means ×. A "linear" node
 *   with several incoming edges (e.g. Output Projection reading Q/K/V) is
 *   doing real attention math, not a simple combine — its own formula
 *   (shown separately) already covers that, so no operator is guessed here.
 */
function describeInputConstruction(model: Model, node: ModelNode): { sources: InputSource[]; operator: "+" | "×" | null } {
  const incoming = model.edges.filter((e) => e.target === node.id);
  const nonAncestor = incoming.filter((e) => e.label === "skip" || !isAncestor(model, e.source, node));
  // Container-sourced edges only survive if nothing more specific covers
  // the connection — otherwise they're the redundant "collapsed view" kind.
  const kept = nonAncestor.length > 0 ? nonAncestor : incoming;

  const sources: InputSource[] = kept.map((e) => {
    // Ancestor-sourced edges get relabeled regardless of "skip" — a
    // residual's skip edge is very often exactly this pattern (the block's
    // own original input, carried around the sub-layer), and deserves the
    // same "Block Input" wording the graph itself uses rather than the
    // container's literal name.
    const boundary = isAncestor(model, e.source, node);
    return { label: boundary ? "Block input (from outside this block)" : model.nodes[e.source]?.name ?? e.source, isBlockBoundary: boundary };
  });

  if (sources.length < 2) return { sources, operator: null };

  const info = componentRegistry[node.type];
  if (info.category === "elementwise") return { sources, operator: "×" };
  if (info.category === "addition" || kept.some((e) => e.label === "skip")) return { sources, operator: "+" };
  if (info.category === "linear" || info.category === "other") return { sources, operator: null };
  // A structural node (e.g. a transformer block) combining two or more
  // untagged, non-container sources — every current instance of this
  // (token + positional embedding feeding the first block) is a plain sum.
  return { sources, operator: "+" };
}

interface Props {
  model: Model;
  node: ModelNode | null;
  /** The selected node's real captured activation shape/magnitude from the last run — undefined when no run has happened yet, or this node has no recorded activation (e.g. a purely organizational container). */
  activationShape?: number[];
  activationMagnitude?: number;
  onViewActivation?: () => void;
  onViewWeights?: () => void;
}

export function Inspector({ model, node, activationShape, activationMagnitude, onViewActivation, onViewWeights }: Props) {
  if (!node) {
    return (
      <div className="inspector">
        <div className="empty-hint">Click a component in the graph or tree to inspect it.</div>
      </div>
    );
  }

  const info = componentRegistry[node.type];
  const totalParams = node.parameters.reduce((a, p) => a + (p.slice ? p.logicalShape.reduce((x, y) => x * y, 1) : p.numElements), 0);
  const hasThisRun = activationShape !== undefined && activationMagnitude !== undefined;
  const { sources: inputSources, operator: inputOperator } = describeInputConstruction(model, node);

  return (
    <div className="inspector">
      <div className="inspector-title" style={{ borderColor: info.color }}>
        <span className="inspector-badge" style={{ background: info.color }}>
          {info.label}
        </span>
        <span className="inspector-name">{node.name}</span>
      </div>

      <Section title="What is it?">
        <p>{info.description}</p>
      </Section>

      {inputSources.length > 0 && (
        <Section title="Input construction">
          {inputOperator ? (
            <code className="formula">input = {inputSources.map((s) => s.label).join(` ${inputOperator} `)}</code>
          ) : inputSources.length === 1 ? (
            <code className="formula">input = {inputSources[0].label}</code>
          ) : (
            <>
              <p>Assembled from multiple sources:</p>
              <ul className="input-source-list">
                {inputSources.map((s, i) => (
                  <li key={i}>{s.label}</li>
                ))}
              </ul>
            </>
          )}
        </Section>
      )}

      {hasThisRun && (
        <Section title="This run">
          <div className="io-row">
            <span className="io-label">activation shape</span>
            <span className="io-shape">{formatDims(activationShape!)}</span>
          </div>
          <div className="io-row">
            <span className="io-label">magnitude (L2 norm)</span>
            <span className="io-shape">{activationMagnitude!.toFixed(4)}</span>
          </div>
        </Section>
      )}

      {(hasThisRun || node.parameters.length > 0) && (
        <div className="inspector-actions">
          {hasThisRun && onViewActivation && (
            <button type="button" onClick={onViewActivation}>
              View activation
            </button>
          )}
          {node.parameters.length > 0 && onViewWeights && (
            <button type="button" onClick={onViewWeights}>
              View weights
            </button>
          )}
        </div>
      )}

      {info.formula && (
        <Section title="Show me the math">
          <code className="formula">{info.formula}</code>
        </Section>
      )}

      {(node.inputs.length > 0 || node.outputs.length > 0) && (
        <Section title="Shapes">
          {node.inputs.map((s, i) => (
            <div key={`in-${i}`} className="io-row">
              <span className="io-label">input</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
          {node.outputs.map((s, i) => (
            <div key={`out-${i}`} className="io-row">
              <span className="io-label">output</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
        </Section>
      )}

      {node.parameters.length > 0 && (
        <Section title={`Parameters (${totalParams.toLocaleString()})`}>
          {node.parameters.map((p, i) => (
            <div key={i} className="io-row">
              <span className="io-label">{p.slice ? `${p.name} (slice)` : p.name}</span>
              <span className="io-shape">
                {p.logicalShape.join(" × ")} · {p.dtype}
              </span>
            </div>
          ))}
        </Section>
      )}

      {Object.keys(node.metadata).length > 0 && (
        <Section title="Metadata">
          {Object.entries(node.metadata).map(([k, v]) => (
            <div key={k} className="io-row">
              <span className="io-label">{k}</span>
              <span className="io-shape">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="inspector-section">
      <div className="inspector-section-title">{title}</div>
      {children}
    </div>
  );
}
