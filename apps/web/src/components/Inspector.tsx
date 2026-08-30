import type { ReactNode } from "react";
import type { Model, ModelNode } from "@aperture/model-ir";
import { componentRegistry } from "../registry.js";
import { describeInputConstruction } from "../nodeInputs.js";

function formatDims(dims: Array<number | string>): string {
  return `[${dims.join(", ")}]`;
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
