import type { ModelEdge } from "@aperture/model-ir";

export interface LayoutPosition {
  x: number;
  y: number;
}

const CHAIN_COL_WIDTH = 220;
const BRANCH_COL_WIDTH = 240;
const CHAIN_ROW_HEIGHT = 120;
/** Extra vertical room wherever a rank fans out to (or in from) more than one node — gives the junction dot and offset ports space to read clearly instead of being cramped against the next rank. */
const BRANCH_ROW_HEIGHT = 170;

/**
 * Minimal layered ("Sugiyama-style") layout: rank nodes by longest path from
 * a source, then spread each rank horizontally. Works for a straight chain
 * (level 1) and for branching subgraphs like Attention's Q/K/V (level 2)
 * without any architecture-specific logic — it only reads the IR's edges.
 */
export function layeredLayout(nodeIds: string[], edges: ModelEdge[]): Map<string, LayoutPosition> {
  const idSet = new Set(nodeIds);
  const relevant = edges.filter((e) => idSet.has(e.source) && idSet.has(e.target));

  const preds = new Map<string, string[]>();
  const succs = new Map<string, string[]>();
  for (const id of nodeIds) {
    preds.set(id, []);
    succs.set(id, []);
  }
  for (const e of relevant) {
    preds.get(e.target)!.push(e.source);
    succs.get(e.source)!.push(e.target);
  }

  const rank = new Map<string, number>();
  const order = topologicalOrder(nodeIds, relevant);
  for (const id of order) {
    const p = preds.get(id) ?? [];
    const r = p.length === 0 ? 0 : Math.max(...p.map((x) => rank.get(x) ?? 0)) + 1;
    rank.set(id, r);
  }

  const byRank = new Map<number, string[]>();
  for (const id of nodeIds) {
    const r = rank.get(id) ?? 0;
    if (!byRank.has(r)) byRank.set(r, []);
    byRank.get(r)!.push(id);
  }
  const maxRank = Math.max(0, ...byRank.keys());

  // Cumulative y per rank rather than a fixed `r * ROW_HEIGHT` — the gap
  // leading into or out of any rank with more than one node (a branch or a
  // merge) gets extra room; a plain chain segment stays compact.
  const rankY = new Map<number, number>();
  rankY.set(0, 0);
  for (let r = 1; r <= maxRank; r++) {
    const prevBranches = (byRank.get(r - 1)?.length ?? 1) > 1;
    const thisBranches = (byRank.get(r)?.length ?? 1) > 1;
    const gap = prevBranches || thisBranches ? BRANCH_ROW_HEIGHT : CHAIN_ROW_HEIGHT;
    rankY.set(r, (rankY.get(r - 1) ?? 0) + gap);
  }

  const positions = new Map<string, LayoutPosition>();
  for (const [r, ids] of byRank) {
    const n = ids.length;
    const colWidth = n > 1 ? BRANCH_COL_WIDTH : CHAIN_COL_WIDTH;
    ids.forEach((id, i) => {
      const x = (i - (n - 1) / 2) * colWidth;
      positions.set(id, { x, y: rankY.get(r) ?? r * CHAIN_ROW_HEIGHT });
    });
  }
  return positions;
}

function topologicalOrder(nodeIds: string[], edges: ModelEdge[]): string[] {
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const id of nodeIds) {
    inDegree.set(id, 0);
    adj.set(id, []);
  }
  for (const e of edges) {
    adj.get(e.source)!.push(e.target);
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
  }
  const queue = nodeIds.filter((id) => inDegree.get(id) === 0);
  const result: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    result.push(id);
    for (const next of adj.get(id) ?? []) {
      inDegree.set(next, (inDegree.get(next) ?? 0) - 1);
      if (inDegree.get(next) === 0) queue.push(next);
    }
  }
  // any nodes left (cycle, shouldn't happen) — append in original order
  for (const id of nodeIds) if (!result.includes(id)) result.push(id);
  return result;
}
