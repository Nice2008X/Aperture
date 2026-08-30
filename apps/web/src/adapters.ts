import type { ModelAdapter } from "@aperture/model-ir";
import { BackendAdapter } from "@aperture/api-client";

/**
 * The GPU backend (apps/api) now does everything architecture-specific —
 * see PLAN.md §1 and §6. This replaced 8 hand-written client-side adapters
 * (packages/model-adapters/*, still present but no longer wired in here);
 * adding a new architecture is now a matter of the backend's generic graph
 * builder recognizing it, not a new package in this list.
 */
export const ADAPTERS: ModelAdapter[] = [BackendAdapter];
