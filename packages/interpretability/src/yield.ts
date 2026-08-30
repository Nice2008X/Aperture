/**
 * Hands control back to the browser for one real event-loop turn.
 *
 * Every `await` in this package's hot loops (loadTensor, runInference) only
 * resolves through a microtask — and a chain of microtasks runs to
 * completion before the browser gets a chance to paint anything, including
 * a "loading…" message a component already set via setState. Without a
 * genuine macrotask yield like this one, a multi-forward-pass computation
 * (token attribution's S+1 passes, logit lens's per-layer projections on a
 * large vocabulary) blocks the tab from ever showing that it's working —
 * the UI just freezes until the whole thing finishes.
 */
export function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
