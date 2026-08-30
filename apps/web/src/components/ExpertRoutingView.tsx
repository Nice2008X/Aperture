import type { Tensor } from "@aperture/model-ir";

interface Props {
  routerWeights: Tensor; // [seqLen, topK] — softmax'd gate weight per selected expert
  expertAssignment: Tensor; // [seqLen, topK] — selected expert id per selected slot, same order
  tokens: string[];
  numExperts: number;
}

// Deterministic per-expert hue so the same expert id always reads as the
// same color across tokens/rows — golden-angle spacing keeps adjacent
// expert ids visually distinct even for a model with dozens of experts.
function expertColor(expertId: number): string {
  const hue = (expertId * 137.508) % 360;
  return `hsl(${hue.toFixed(0)}, 65%, 55%)`;
}

export function ExpertRoutingView({ routerWeights, expertAssignment, tokens, numExperts }: Props) {
  const [seqLen, topK] = routerWeights.shape;

  const rows = Array.from({ length: seqLen }, (_, t) => {
    const slots = Array.from({ length: topK }, (_, k) => ({
      expertId: expertAssignment.data[t * topK + k],
      weight: routerWeights.data[t * topK + k],
    }));
    slots.sort((a, b) => b.weight - a.weight);
    return slots;
  });

  return (
    <div className="expert-routing-view">
      <div className="expert-routing-header">
        <span>
          Top-{topK} expert{topK === 1 ? "" : "s"} selected per token (of {numExperts} total)
        </span>
      </div>
      <div className="expert-routing-rows">
        {rows.map((slots, t) => (
          <div key={t} className="expert-routing-row">
            <span className="expert-routing-token" title={tokens[t]}>
              {tokens[t]?.trim() || "·"}
            </span>
            <div className="expert-routing-bar-track">
              {slots.map((slot, i) => (
                <div
                  key={i}
                  className="expert-routing-bar-segment"
                  style={{ width: `${slot.weight * 100}%`, background: expertColor(slot.expertId) }}
                  title={`Expert ${slot.expertId} — ${(slot.weight * 100).toFixed(1)}%`}
                />
              ))}
            </div>
            <div className="expert-routing-labels">
              {slots.map((slot, i) => (
                <span key={i} className="expert-routing-label" style={{ color: expertColor(slot.expertId) }}>
                  E{slot.expertId} {(slot.weight * 100).toFixed(0)}%
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
