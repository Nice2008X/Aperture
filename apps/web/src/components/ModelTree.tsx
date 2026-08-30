import { useEffect, useMemo, useRef, useState } from "react";
import type { Model, ModelNode } from "@aperture/model-ir";

interface Props {
  model: Model;
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** Double-clicking any row jumps the graph to whichever view actually contains that node (a transformer block's detail view, or back out to the top-level architecture view for a node outside any block) and selects it there — not just a transformer-block-only "expand" gesture. */
  onNavigate: (nodeId: string) => void;
  /** Per-node activation magnitude (L2 norm) from the last run — lets a user spot an unusually "loud" layer without opening every node. Undefined (or a node missing from it) means no run yet / no activation recorded for that node. */
  activationMagnitudeById?: Record<string, number>;
}

/** Node ids open by default: the root and its immediate children — deep enough to orient a user, shallow enough that a many-layer model doesn't flood the tree with every transformer block (and its attention/FFN internals) expanded. */
function defaultOpenIds(model: Model): Set<string> {
  const open = new Set<string>();
  function walk(nodeId: string, depth: number) {
    if (depth < 2) open.add(nodeId);
    for (const childId of model.nodes[nodeId].children) walk(childId, depth + 1);
  }
  walk(model.rootId, 0);
  return open;
}

/** Every ancestor of `nodeId`, walking parentId up to the root. */
function ancestorsOf(model: Model, nodeId: string | null): string[] {
  const ancestors: string[] = [];
  let current = nodeId ? (model.nodes[nodeId]?.parentId ?? null) : null;
  while (current) {
    ancestors.push(current);
    current = model.nodes[current]?.parentId ?? null;
  }
  return ancestors;
}

export function ModelTree({ model, selectedId, onSelect, onNavigate, activationMagnitudeById }: Props) {
  const [openIds, setOpenIds] = useState<Set<string>>(() => defaultOpenIds(model));

  // Node count is bounded by architecture size (dozens to a couple hundred
  // rows even for a many-layer model) — never vocab-sized — so a plain loop
  // here is just consistent caution, not a required safeguard the way it
  // was for the earlier vocab-sized Math.max(...) bugs.
  const maxMagnitude = useMemo(() => {
    let max = 1e-9;
    for (const v of Object.values(activationMagnitudeById ?? {})) if (v > max) max = v;
    return max;
  }, [activationMagnitudeById]);

  // A different model was loaded — node ids are reused across architectures
  // (every adapter names its root "model", its layer group "blocks", etc.),
  // so the old open/closed set could otherwise persist into a differently
  // shaped tree. Recompute the defaults for the tree actually being shown.
  useEffect(() => {
    setOpenIds(defaultOpenIds(model));
  }, [model]);

  // A node selected from outside the tree (single- or double-click in the
  // architecture graph) needs to actually be visible here — force every one
  // of its ancestors open, even ones the user had collapsed, rather than
  // just flipping a `selected` class on a row that isn't rendered at all.
  // The node itself is included too: double-clicking a transformer block to
  // enter it selects that block, and its own children (rms1/attn/rms2/ffn)
  // need to be revealed, not just the path down to it.
  useEffect(() => {
    if (!selectedId) return;
    const idsToOpen = [selectedId, ...ancestorsOf(model, selectedId)];
    setOpenIds((prev) => {
      const next = new Set(prev);
      let changed = false;
      for (const id of idsToOpen) {
        if (!next.has(id)) {
          next.add(id);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [model, selectedId]);

  const toggle = (nodeId: string) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  };

  return (
    <div className="model-tree">
      <TreeNode
        model={model}
        nodeId={model.rootId}
        depth={0}
        selectedId={selectedId}
        onSelect={onSelect}
        onNavigate={onNavigate}
        openIds={openIds}
        onToggle={toggle}
        activationMagnitudeById={activationMagnitudeById}
        maxMagnitude={maxMagnitude}
      />
    </div>
  );
}

function TreeNode({
  model,
  nodeId,
  depth,
  selectedId,
  onSelect,
  onNavigate,
  openIds,
  onToggle,
  activationMagnitudeById,
  maxMagnitude,
}: {
  model: Model;
  nodeId: string;
  depth: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNavigate: (nodeId: string) => void;
  openIds: Set<string>;
  onToggle: (nodeId: string) => void;
  activationMagnitudeById?: Record<string, number>;
  maxMagnitude: number;
}) {
  const node: ModelNode = model.nodes[nodeId];
  const hasChildren = node.children.length > 0;
  const open = openIds.has(nodeId);
  const isSelected = nodeId === selectedId;
  const rowRef = useRef<HTMLDivElement>(null);

  // Runs whenever this row becomes the selected one — including the first
  // render after a collapsed ancestor was force-opened to reveal it, since
  // that mounts this row fresh with isSelected already true. `block:
  // "nearest"` scrolls the tree pane's own overflow container just enough
  // to bring the row into view, without also yanking the rest of the page.
  useEffect(() => {
    if (isSelected) rowRef.current?.scrollIntoView({ block: "nearest" });
  }, [isSelected]);

  return (
    <div>
      <div
        ref={rowRef}
        className={"tree-row" + (isSelected ? " selected" : "")}
        style={{ paddingLeft: 8 + depth * 14 }}
        onClick={() => onSelect(nodeId)}
        onDoubleClick={() => onNavigate(nodeId)}
      >
        {hasChildren ? (
          <button
            className="tree-toggle"
            onClick={(e) => {
              e.stopPropagation();
              onToggle(nodeId);
            }}
          >
            {open ? "▾" : "▸"}
          </button>
        ) : (
          <span className="tree-toggle-spacer" />
        )}
        <span className="tree-label">{node.name}</span>
        {node.parameters.length > 0 && <span className="tree-param-dot" title="has weights" />}
        {activationMagnitudeById?.[nodeId] !== undefined && (
          <span className="tree-activation-tick" title={`activation magnitude: ${activationMagnitudeById[nodeId].toFixed(4)}`}>
            <span className="tree-activation-tick-fill" style={{ width: `${Math.min(100, (activationMagnitudeById[nodeId] / maxMagnitude) * 100)}%` }} />
          </span>
        )}
      </div>
      {hasChildren && open && (
        <div>
          {node.children.map((childId) => (
            <TreeNode
              key={childId}
              model={model}
              nodeId={childId}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
              onNavigate={onNavigate}
              openIds={openIds}
              onToggle={onToggle}
              activationMagnitudeById={activationMagnitudeById}
              maxMagnitude={maxMagnitude}
            />
          ))}
        </div>
      )}
    </div>
  );
}
