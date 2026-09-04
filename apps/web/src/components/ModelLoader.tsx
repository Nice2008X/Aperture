import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import type { LoadProgressEvent } from "@aperture/model-ir";
import {
  listModels,
  getGpuStatus,
  streamDownload,
  cancelDownload,
  isDownloadDone,
  isDownloadCancelledEvent,
  isDownloadError,
  cancelLoad,
  unloadModel,
  deleteModel,
  type CatalogEntry,
  type GpuStatus,
  type DownloadPhase,
} from "@aperture/api-client";
import { useTranslation } from "./LanguageContext.js";
import { formatBytes } from "../format.js";
import { LoadProgressBar, type UnifiedLoadProgress } from "./LoadProgressBar.js";
import { DeleteModelDialog } from "./DeleteModelDialog.js";

type Quantization = "4bit" | "8bit" | undefined;

interface Props {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  onLoad: (modelId: string, quantization?: "4bit" | "8bit") => void;
  /** The currently-loaded model's id, if any — left out of the list so it isn't offered back as if it were a fresh option. */
  excludeModelId?: string;
  /** Updated while status === "loading" — a real checkpoint takes tens of seconds to materialize onto the GPU, which this renders as a progress bar instead of leaving the screen with no feedback. */
  loadProgress?: LoadProgressEvent;
  /** Drops the built-in title/subtitle and card chrome (background/border/padding) — used when this is embedded inside a panel that already provides its own header, e.g. the "load a different model" popover. */
  embedded?: boolean;
  /** From Settings ("Models per page") — how many catalog entries the library shows before paging. */
  modelsPerPage: number;
  /** From Settings ("Show GPU status") — hides the VRAM name/usage bar entirely when off. */
  showGpuStatus: boolean;
}

type CatalogState = { status: "loading" } | { status: "ready"; entries: CatalogEntry[] } | { status: "error"; error: string };

type DownloadState =
  | { status: "idle" }
  | { status: "downloading"; downloadedBytes: number; totalBytes: number; phase?: DownloadPhase }
  // Paused keeps the last progress numbers around purely for display (the
  // frozen bar) — resuming re-downloads from these bytes forward for
  // whichever single file was actively transferring at pause time, every
  // other already-finished file is skipped server-side. See
  // apps/api's downloads.py DownloadCancelled doc comment for why that's
  // the real granularity huggingface_hub supports here, not true
  // byte-exact resume.
  | { status: "paused"; downloadedBytes: number; totalBytes: number; phase?: DownloadPhase }
  | { status: "complete"; modelId: string; displayName: string }
  | { status: "error"; error: string };

const IDLE_DOWNLOAD: DownloadState = { status: "idle" };

// Rough VRAM multiplier relative to a catalog entry's on-disk sizeBytes
// (itself just the checkpoint's stored dtype, usually bf16 for what HF
// serves). Real bitsandbytes quantization overhead varies by architecture,
// so this is deliberately a *labeled estimate* (todo11.txt §7's "don't show
// fake precision" caveat) — good enough to catch an obviously-too-large
// pick before the backend rejects it, not a promise of the exact byte count
// that will land on the GPU.
const QUANT_MEMORY_FACTOR: Record<"4bit" | "8bit", number> = { "8bit": 0.5, "4bit": 0.25 };

function estimatedBytesFor(sizeBytes: number, quantization: Quantization): number {
  return sizeBytes * (quantization ? QUANT_MEMORY_FACTOR[quantization] : 1);
}

type FitLevel = "fits" | "tight" | "no";

function fitLevel(estimatedBytes: number, freeBytes: number): FitLevel {
  if (estimatedBytes > freeBytes) return "no";
  if (estimatedBytes > freeBytes * 0.85) return "tight";
  return "fits";
}

const FIT_ICON: Record<FitLevel, string> = { fits: "✓", tight: "⚠", no: "✕" };

// A real Hugging Face repo id is exactly "namespace/name", each half
// alphanumeric-bounded with only -._ in between (mirrors HF's own naming
// rules closely enough for this purpose) — deliberately stricter than
// "anything without a slash-slash", since this string ends up both in a
// network request to the Hub and (server-side, model_id_for) as a
// filesystem directory name. Rejecting anything else here means a segment
// like ".." or "org/.." never reaches either.
const SAFE_REPO_ID_RE = /^[A-Za-z0-9]([A-Za-z0-9._-]{0,94}[A-Za-z0-9])?\/[A-Za-z0-9]([A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;

/**
 * Pulls "org/model" out of a pasted Hugging Face URL — plain paths
 * (huggingface.co/org/model), full URLs, and URLs with a /tree/<rev>,
 * /blob/..., query string, or trailing slash all reduce to just the first
 * two path segments. Anything that isn't recognizably a huggingface.co URL
 * (including a bare "org/model" id, the common case) passes through
 * untouched — validation of the *result* is SAFE_REPO_ID_RE's job, not
 * this function's.
 */
function extractRepoId(raw: string): string {
  const trimmed = raw.trim();
  const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : /^(www\.)?huggingface\.co\//i.test(trimmed) ? `https://${trimmed}` : null;
  if (withScheme) {
    try {
      const url = new URL(withScheme);
      if (url.hostname.toLowerCase() === "huggingface.co" || url.hostname.toLowerCase() === "www.huggingface.co") {
        const segments = url.pathname.split("/").filter(Boolean);
        if (segments.length >= 2) return `${segments[0]}/${segments[1]}`;
      }
    } catch {
      // Not actually a valid URL — fall through and treat it as a plain id.
    }
  }
  return trimmed;
}

/** Mirrors apps/api's downloads.py model_id_for — the on-disk directory name (data/models/<id>/) for a given repo id, needed so a cancelled download can clean up its own partial files by the same path the backend would use. */
function modelIdForRepo(repo: string): string {
  return repo.replace(/\//g, "__");
}

/**
 * Real checkpoints live in data/models/ on the GPU backend (PLAN.md §3) —
 * the catalog list is whatever's already downloaded there. Pasting a
 * Hugging Face repo id below triggers the backend to pull it on demand
 * (PLAN.md §8.6) — same "paste an id, it loads" UX the very first version
 * of this loader had, just downloading server-side now instead of
 * fetching straight into the browser.
 *
 * Selection is a distinct step from loading (todo11.txt §4): clicking an
 * available card selects it, expanding that card in place to show its
 * configuration controls (precision, memory estimate, VRAM fit-check, the
 * Load button) rather than loading immediately — except for a card that's
 * already resident on the GPU (`entry.loaded`), where there's nothing to
 * configure and the original one-click behavior is kept.
 */
export function ModelLoader({ status, error, onLoad, excludeModelId, loadProgress, embedded, modelsPerPage, showGpuStatus }: Props) {
  const { t } = useTranslation();
  const [catalog, setCatalog] = useState<CatalogState>({ status: "loading" });
  const [gpu, setGpu] = useState<GpuStatus | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [repo, setRepo] = useState("");
  const [download, setDownload] = useState<DownloadState>(IDLE_DOWNLOAD);
  const [quantization, setQuantization] = useState<Quantization>(undefined);
  const [page, setPage] = useState(0);
  const perPage = Math.max(1, modelsPerPage);
  // Tracks the one catalog entry an unload/delete is in flight for, so its
  // row can disable itself without freezing the rest of the list.
  const [busyId, setBusyId] = useState<string | null>(null);
  const [unloadAllBusy, setUnloadAllBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<CatalogEntry | null>(null);
  // Set right before calling cancelDownload() so the loop in runDownload
  // knows, once the stream actually ends with a cancelled event, whether
  // to land on "paused" (keep the repo id, offer Resume) or reset back to
  // idle (a real Cancel) — the backend request is identical either way.
  // Clicking Cancel after Pause (or either one twice) just overwrites this
  // with the latest intent — cancelDownload is idempotent server-side, and
  // neither button is disabled while a request is in flight (see
  // requestDownloadStop), so there's nothing to reconcile beyond "last
  // click wins".
  const pendingDownloadActionRef = useRef<"pause" | "cancel" | null>(null);

  // The page-size setting can change while this screen is already up
  // (Settings is reachable from here too) — a stale page index would
  // otherwise show a confusing "page 4 of 2" slice once clamped.
  useEffect(() => {
    setPage(0);
  }, [perPage]);

  const refreshCatalog = useCallback(() => {
    return listModels()
      .then((fetched) => {
        setCatalog({ status: "ready", entries: fetched });
        return fetched;
      })
      .catch((err) => {
        setCatalog({ status: "error", error: err instanceof Error ? err.message : String(err) });
        return [] as CatalogEntry[];
      });
  }, []);

  // Re-fetched after every unload/unload-all/delete below, not just on
  // mount — those free real GPU memory (delete unloads first if the
  // deleted entry was resident), and the status bar/fit-checks would
  // otherwise keep showing the pre-unload numbers until this screen were
  // remounted.
  const refreshGpu = useCallback(() => {
    return getGpuStatus()
      .then(setGpu)
      .catch(() => setGpu({ available: false }));
  }, []);

  useEffect(() => {
    refreshCatalog();
    refreshGpu();
  }, [refreshCatalog, refreshGpu]);

  // Shared by a fresh submit and by Resume — resuming is just calling this
  // again with the same repo id (see DownloadState's "paused" doc comment
  // for what that actually resumes at).
  const runDownload = async (repoId: string) => {
    pendingDownloadActionRef.current = null;
    setDownload({ status: "downloading", downloadedBytes: 0, totalBytes: 0 });
    let last = { downloadedBytes: 0, totalBytes: 0, phase: undefined as DownloadPhase | undefined };
    try {
      for await (const event of streamDownload(repoId)) {
        if (isDownloadCancelledEvent(event)) {
          if (pendingDownloadActionRef.current === "pause") {
            setDownload({ status: "paused", ...last });
          } else {
            // Cancel (unlike Pause) discards the attempt entirely — remove
            // whatever partial files just got written to data/models/ for
            // this repo, silently, rather than leaving a half-downloaded,
            // un-catalogued directory behind. Best-effort: the UI already
            // moves on regardless of whether this cleanup call succeeds.
            setDownload(IDLE_DOWNLOAD);
            setRepo("");
            void deleteModel(modelIdForRepo(repoId)).catch(() => {});
          }
          pendingDownloadActionRef.current = null;
          return;
        }
        if (isDownloadError(event)) throw new Error(event.error);
        if (isDownloadDone(event)) {
          setRepo("");
          const fetched = await refreshCatalog();
          // Land on the configuration step for the freshly-downloaded model
          // instead of auto-loading it (todo11.txt §8) — downloading and
          // loading are different commitments (the latter occupies the
          // GPU), so the user gets a chance to pick precision first. Also
          // jump to whichever page it actually landed on — it's appended
          // at the end of the catalog, so with pagination on it would
          // otherwise be selected-but-invisible on page 1.
          const idx = fetched.filter((e) => e.id !== excludeModelId).findIndex((e) => e.id === event.manifest.id);
          if (idx >= 0) setPage(Math.floor(idx / perPage));
          setSelectedId(event.manifest.id);
          setDownload({ status: "complete", modelId: event.manifest.id, displayName: event.manifest.displayName });
          return;
        }
        last = { downloadedBytes: event.downloadedBytes, totalBytes: event.totalBytes, phase: event.phase };
        setDownload({ status: "downloading", ...last });
      }
    } catch (err) {
      pendingDownloadActionRef.current = null;
      setDownload({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  };

  const startDownload = async (e: FormEvent) => {
    e.preventDefault();
    // Re-derived here rather than trusted from state — the input's onChange
    // already normalizes a pasted URL down to "org/model" as the user types,
    // but this is the actual gate: nothing downloads unless what's about to
    // be sent is a safe, well-formed repo id, full URL or not.
    const parsed = extractRepoId(repo);
    if (!SAFE_REPO_ID_RE.test(parsed)) return;
    setRepo(parsed);
    await runDownload(parsed);
  };

  const requestDownloadStop = (action: "pause" | "cancel") => {
    pendingDownloadActionRef.current = action;
    void cancelDownload(repo);
  };

  const resumeDownload = () => {
    void runDownload(repo);
  };

  // Cancel while *paused* is a different situation from cancel while
  // actively downloading (requestDownloadStop, above): there's no longer a
  // live stream for the backend's cancel-flag mechanism to interrupt, or
  // for a {cancelled: true} event to come back on and trigger cleanup — the
  // download already stopped and exited when it paused. So this deletes
  // the partial directory directly instead of going through
  // cancelDownload/runDownload at all.
  const cancelPausedDownload = () => {
    const repoId = repo;
    setDownload(IDLE_DOWNLOAD);
    setRepo("");
    void deleteModel(modelIdForRepo(repoId)).catch(() => {});
  };

  const entries = catalog.status === "ready" ? catalog.entries.filter((e) => e.id !== excludeModelId) : [];
  const downloading = download.status === "downloading";
  const downloadPaused = download.status === "paused";
  const parsedRepo = extractRepoId(repo);
  const repoValid = SAFE_REPO_ID_RE.test(parsedRepo);
  const modelLoading = status === "loading";
  const gpuInfo = gpu && gpu.available ? gpu : null;
  const pageCount = Math.max(1, Math.ceil(entries.length / perPage));
  const clampedPage = Math.min(page, pageCount - 1);
  const pagedEntries = entries.slice(clampedPage * perPage, clampedPage * perPage + perPage);

  // Clicking a card always just expands/collapses it — even an already-loaded
  // one, which used to jump straight into onLoad's fast path on click. That
  // silently re-triggered a "load" the moment the card was clicked, with no
  // chance to reach for Unload/Delete instead; expanding first and putting
  // an explicit Load model button inside (see the expanded-loaded branch
  // below) lets the user decide.
  function selectEntry(entry: CatalogEntry) {
    setSelectedId((cur) => (cur === entry.id ? null : entry.id));
  }

  // Checked against the *unfiltered* catalog, not `entries` — the embedded
  // switch-model panel excludes the current session's model from the pick
  // list (excludeModelId), but that model is still genuinely resident on
  // the GPU and "Unload all" needs to see it.
  const anyLoaded = catalog.status === "ready" && catalog.entries.some((e) => e.loaded);
  const actionsBusy = busyId !== null || unloadAllBusy;

  async function handleUnloadAll() {
    if (catalog.status !== "ready") return;
    const loadedEntry = catalog.entries.find((e) => e.loaded);
    if (!loadedEntry) return;
    setUnloadAllBusy(true);
    setActionError(null);
    try {
      await unloadModel(loadedEntry.id);
      await Promise.all([refreshCatalog(), refreshGpu()]);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setUnloadAllBusy(false);
    }
  }

  async function handleUnloadEntry(entry: CatalogEntry) {
    setBusyId(entry.id);
    setActionError(null);
    try {
      await unloadModel(entry.id);
      await Promise.all([refreshCatalog(), refreshGpu()]);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  // Opens the confirm dialog (see DeleteModelDialog) instead of deleting
  // right away — the actual delete only runs from confirmDelete, once the
  // user has seen the model named back to them and clicked through.
  function handleDeleteEntry(entry: CatalogEntry) {
    setDeleteTarget(entry);
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    const entry = deleteTarget;
    setBusyId(entry.id);
    setActionError(null);
    try {
      await deleteModel(entry.id);
      setSelectedId((cur) => (cur === entry.id ? null : cur));
      setDeleteTarget(null);
      // Deleting the resident model unloads it server-side first (see
      // ModelRegistry.delete) — refresh GPU status too, not just the catalog,
      // in case that's what just happened here.
      await Promise.all([refreshCatalog(), refreshGpu()]);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  // Download and GPU-load are sequential, never simultaneous (a completed
  // download lands on the configuration step above instead of calling
  // onLoad itself) — but they stay in their own two positions below
  // (download progress right under the repo-id form the user just
  // submitted; load progress up near the catalog, where a catalog click or
  // Load model click lands next) rather than collapsing into one shared
  // slot, which would yank a download's progress away from where the user
  // is actually looking. What's genuinely unified is the LoadProgressBar
  // component itself — one bar/label/status implementation instead of two
  // duplicated ones.
  const loadingProgress: UnifiedLoadProgress | null = modelLoading
    ? { phase: loadProgress?.phase ?? "loading_weights", current: loadProgress?.current, total: loadProgress?.total, unit: "count" }
    : null;
  const downloadProgress: UnifiedLoadProgress | null = downloading
    ? { phase: download.phase ?? "downloading", current: download.downloadedBytes, total: download.totalBytes || undefined, unit: "bytes" }
    : null;
  const pausedProgress: UnifiedLoadProgress | null = downloadPaused
    ? { phase: download.phase ?? "downloading", current: download.downloadedBytes, total: download.totalBytes || undefined, unit: "bytes" }
    : null;

  return (
    <div className={"model-loader" + (embedded ? " embedded" : "")}>
      {!embedded && (
        <>
          <div className="model-loader-title">{t("loader.title")}</div>
          <div className="model-loader-sub">{t("loader.subtitle")}</div>
          <div className="model-loader-cross-promo">
            {t("loader.crossPromo")}{" "}
            <a href="https://nice2008x.github.io/Tensorium/" target="_blank" rel="noopener noreferrer">
              {t("loader.crossPromoDemo")}
            </a>
            {" · "}
            <a href="https://github.com/Nice2008X/Tensorium" target="_blank" rel="noopener noreferrer">
              {t("loader.crossPromoGithub")}
            </a>
            <div className="model-loader-cross-promo-aside">{t("loader.crossPromoAside")}</div>
          </div>
        </>
      )}

      {showGpuStatus && gpuInfo && (
        <div className="model-loader-gpu">
          <div className="model-loader-gpu-header">
            <span className="model-loader-gpu-label">{t("loader.gpuStatus")}</span>
            <span className="model-loader-gpu-name">{gpuInfo.name}</span>
            <span className="model-loader-gpu-mem">
              {formatBytes(gpuInfo.usedBytes)} / {formatBytes(gpuInfo.totalBytes)}
            </span>
          </div>
          <div className="load-progress-track model-loader-gpu-bar">
            <div className="load-progress-fill" style={{ width: `${Math.min(100, (gpuInfo.usedBytes / gpuInfo.totalBytes) * 100)}%` }} />
          </div>
        </div>
      )}

      <div className="model-loader-download-section">
        <div className="model-loader-section-title">{t("loader.addAModel")}</div>
        <div className="model-loader-source-tabs">
          <span className="model-loader-source-tab active">{t("loader.sourceHuggingFace")}</span>
        </div>
        <p className="model-loader-source-hint">{t("loader.sourceHuggingFaceHint")}</p>
        <form className="model-loader-download-form" onSubmit={startDownload}>
          <input
            value={repo}
            onChange={(e) => setRepo(extractRepoId(e.target.value))}
            placeholder={t("loader.inputPlaceholder")}
            disabled={downloading || downloadPaused}
          />
          {/* Stays mounted and just gets disabled once a download is under
              way, rather than being swapped out for other controls —
              avoids the layout jumping around the input on every state
              change, and keeping the standard "same button, now disabled"
              affordance for "can't do this right now" instead of reaching
              for a different one (icon swap, spinner) that same button
              never actually shows here (Pause/Resume/Cancel and the
              progress bar already carry that feedback in their own row). */}
          <button type="submit" className="model-loader-download-submit" disabled={downloading || downloadPaused || !repoValid}>
            {t("loader.download")}
          </button>
        </form>
        {!downloading && !downloadPaused && repo.trim().length > 0 && !repoValid && (
          <div className="model-loader-error">{t("loader.invalidRepoId")}</div>
        )}
        {(downloadProgress || pausedProgress) && (
          <div className="model-loader-download-progress-row">
            <div className="model-loader-download-progress">
              {downloadPaused && <div className="model-loader-download-paused-label">{t("loader.paused")}</div>}
              <LoadProgressBar progress={(downloadProgress ?? pausedProgress)!} />
            </div>
            {downloading && (
              <>
                <button
                  type="button"
                  className="model-loader-card-action icon"
                  onClick={() => requestDownloadStop("pause")}
                  aria-label={t("loader.pause")}
                  title={t("loader.pause")}
                >
                  ⏸
                </button>
                <button
                  type="button"
                  className="model-loader-card-action delete icon"
                  onClick={() => requestDownloadStop("cancel")}
                  aria-label={t("loader.cancel")}
                  title={t("loader.cancel")}
                >
                  ✕
                </button>
              </>
            )}
            {downloadPaused && (
              <>
                <button
                  type="button"
                  className="model-loader-card-action icon"
                  onClick={resumeDownload}
                  aria-label={t("loader.resume")}
                  title={t("loader.resume")}
                >
                  ▶
                </button>
                <button
                  type="button"
                  className="model-loader-card-action delete icon"
                  onClick={cancelPausedDownload}
                  aria-label={t("loader.cancel")}
                  title={t("loader.cancel")}
                >
                  ✕
                </button>
              </>
            )}
          </div>
        )}
        {download.status === "complete" && download.modelId === selectedId && (
          <div className="model-loader-download-complete">
            ✓ {t("loader.downloadComplete")} — {t("loader.downloadCompleteHint").replace("{model}", download.displayName)}
          </div>
        )}
        {download.status === "error" && <div className="model-loader-error">{download.error}</div>}
      </div>

      {catalog.status === "loading" && <div className="model-loader-catalog-status">Loading model catalog…</div>}
      {catalog.status === "error" && (
        <div className="model-loader-error">Couldn't reach the backend at load time: {catalog.error}</div>
      )}
      {catalog.status === "ready" && (entries.length > 0 || anyLoaded) && (
        <div className="model-loader-catalog-header">
          <div className="model-loader-section-title">{t("loader.modelLibrary")}</div>
          <button className="model-loader-unload-all-btn" disabled={!anyLoaded || actionsBusy || modelLoading} onClick={handleUnloadAll}>
            {unloadAllBusy ? t("loader.unloading") : t("loader.unloadAll")}
          </button>
        </div>
      )}
      {catalog.status === "ready" && entries.length === 0 && (
        <div className="model-loader-catalog-status">No models downloaded yet — paste a Hugging Face repo id above to pull one.</div>
      )}
      {actionError && <div className="model-loader-error">{actionError}</div>}

      <div className="model-loader-catalog">
        {pagedEntries.map((entry) => {
          const isSelected = entry.id === selectedId;
          const classes = ["model-loader-catalog-entry"];
          if (isSelected) classes.push("expanded", "selected");
          if (entry.loaded) classes.push("loaded");
          // Every card checks its own fit (todo12.txt §6) against the
          // precision currently chosen (shared across cards — it's "the
          // precision you'd load at"), falling back to the default
          // (full/bf16) before anything's been configured yet or when
          // there's no GPU info to check against.
          const entryEstimatedBytes = estimatedBytesFor(entry.sizeBytes, quantization);
          const cardFit = gpuInfo ? fitLevel(entryEstimatedBytes, gpuInfo.freeBytes) : null;
          const fitLabelKey = cardFit === "fits" ? "loader.fitsShort" : cardFit === "tight" ? "loader.tightShort" : "loader.noFitShort";
          const precisionLabel = quantization ? t(quantization === "8bit" ? "loader.precision8bit" : "loader.precision4bit") : t("loader.precisionFull");

          // Collapsed (the common case — everything but the selected card):
          // one line, just enough to answer "what is it, will it fit" — a
          // loaded model keeps its badge instead of a fit icon (there's
          // nothing to check, it's already resident); an unloaded model
          // gets an icon-only fit indicator rather than repeating the word
          // on every row.
          if (!isSelected) {
            const rowBusy = busyId === entry.id;
            const rowDisabled = status === "loading" || entry.status !== "ready";
            if (rowDisabled) classes.push("disabled");
            return (
              <div key={entry.id} className={classes.join(" ")}>
                {entry.loaded ? (
                  <span className="model-loader-catalog-badge loaded">● {t("loader.loaded")}</span>
                ) : (
                  cardFit && (
                    <span className={`model-loader-card-fit-icon ${cardFit}`} title={t(fitLabelKey)}>
                      {FIT_ICON[cardFit]}
                    </span>
                  )
                )}
                <button className="model-loader-card-toggle-row" disabled={rowDisabled} onClick={() => selectEntry(entry)}>
                  <span className="model-loader-catalog-name">{entry.displayName}</span>
                  <span className="model-loader-catalog-meta-inline">
                    {entry.family}
                    {entry.isMoe ? " · MoE" : ""} · {formatBytes(entry.sizeBytes)} · {precisionLabel}
                    {entry.status !== "ready" ? ` · ${entry.status}` : ""}
                  </span>
                </button>
                {entry.loaded && (
                  <button
                    className="model-loader-card-action icon"
                    disabled={rowBusy || actionsBusy || modelLoading}
                    onClick={() => handleUnloadEntry(entry)}
                    aria-label={t("loader.unload")}
                    title={t("loader.unload")}
                  >
                    ⏏
                  </button>
                )}
                <button
                  className="model-loader-card-action delete icon"
                  disabled={rowBusy || actionsBusy || modelLoading || downloading}
                  onClick={() => handleDeleteEntry(entry)}
                  aria-label={t("loader.delete")}
                  title={t("loader.delete")}
                >
                  🗑
                </button>
              </div>
            );
          }

          // Expanded (the selected card): full detail, plus whatever
          // controls actually apply — a <select>/<button> can't nest inside
          // another <button>, so this branch swaps the card's outer element
          // from <button> to <div> and confines the "click to collapse"
          // toggle to just the name/family header. A loaded entry has
          // nothing to configure (it's already resident at whatever
          // precision it was loaded with), so it skips straight to an
          // explicit Load model (just re-confirms the same fast path a
          // click used to trigger automatically) alongside Unload — the
          // choice this whole expand-instead-of-auto-load change exists to
          // hand back to the user.
          return (
            <div key={entry.id} className={classes.join(" ")}>
              <button
                className="model-loader-card-toggle"
                disabled={status === "loading" || entry.status !== "ready"}
                onClick={() => selectEntry(entry)}
              >
                <span className="model-loader-catalog-entry-header">
                  <span className="model-loader-catalog-name">{entry.displayName}</span>
                  <span className={"model-loader-catalog-badge " + (entry.loaded ? "loaded" : "selected")}>
                    {entry.loaded ? `● ${t("loader.loaded")}` : t("loader.selected")}
                  </span>
                </span>
                <span className="model-loader-catalog-family">
                  {entry.family}
                  {entry.isMoe ? " · MoE" : ""} · {formatBytes(entry.sizeBytes)}
                </span>
              </button>

              <div className="model-loader-card-config">
                {entry.loaded ? (
                  <>
                    <button className="model-loader-load-btn" disabled={modelLoading} onClick={() => onLoad(entry.id, entry.loadedQuantization ?? undefined)}>
                      {t("loader.loadModel")}
                    </button>
                    <button
                      className="model-loader-card-action model-loader-card-delete-expanded"
                      disabled={busyId === entry.id || actionsBusy || modelLoading}
                      onClick={() => handleUnloadEntry(entry)}
                    >
                      {t("loader.unload")}
                    </button>
                  </>
                ) : (
                  <>
                    <div className="model-loader-config-row">
                      <span className="model-loader-config-label">{t("loader.precision")}</span>
                      <select
                        value={quantization ?? "none"}
                        disabled={modelLoading}
                        onChange={(e) => setQuantization(e.target.value === "none" ? undefined : (e.target.value as "4bit" | "8bit"))}
                      >
                        <option value="none">{t("loader.precisionFull")}</option>
                        <option value="8bit">{t("loader.precision8bit")}</option>
                        <option value="4bit">{t("loader.precision4bit")}</option>
                      </select>
                    </div>
                    {cardFit && (
                      <div className={`model-loader-fit-note ${cardFit}`}>
                        <div className="model-loader-fit-headline">
                          {cardFit === "fits" && `${FIT_ICON.fits} ${t("loader.fitsComfortably")}`}
                          {cardFit === "tight" && `${FIT_ICON.tight} ${t("loader.fitsLimited")}`}
                          {cardFit === "no" && `${FIT_ICON.no} ${t("loader.requires").replace("{size}", formatBytes(entryEstimatedBytes))}`}
                        </div>
                        <div className="model-loader-fit-detail">
                          {cardFit !== "no"
                            ? t("loader.usesOfAvailable").replace("{used}", formatBytes(entryEstimatedBytes)).replace("{avail}", formatBytes(gpuInfo!.freeBytes))
                            : t("loader.moreThanAvailable").replace("{delta}", formatBytes(entryEstimatedBytes - gpuInfo!.freeBytes))}
                        </div>
                      </div>
                    )}
                    <button className="model-loader-load-btn" disabled={modelLoading} onClick={() => onLoad(entry.id, quantization)}>
                      {t("loader.loadModel")}
                    </button>
                  </>
                )}
                <button
                  className="model-loader-card-action delete model-loader-card-delete-expanded"
                  disabled={busyId === entry.id || actionsBusy || modelLoading}
                  onClick={() => handleDeleteEntry(entry)}
                >
                  {t("loader.delete")}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {pageCount > 1 && (
        <div className="model-loader-pagination">
          {Array.from({ length: pageCount }, (_, i) => (
            <button
              key={i}
              className={"model-loader-page-dot" + (i === clampedPage ? " active" : "")}
              onClick={() => setPage(i)}
              aria-current={i === clampedPage ? "page" : undefined}
              aria-label={t("loader.pageIndicator").replace("{current}", String(i + 1)).replace("{total}", String(pageCount))}
              title={t("loader.pageIndicator").replace("{current}", String(i + 1)).replace("{total}", String(pageCount))}
            >
              {i + 1}
            </button>
          ))}
        </div>
      )}

      {status === "error" && <div className="model-loader-error">{error}</div>}

      {loadingProgress && (
        <div className="model-loader-footer model-loader-download-progress-row">
          <div className="model-loader-download-progress">
            <LoadProgressBar progress={loadingProgress} />
          </div>
          {selectedId && (
            <button
              type="button"
              className="model-loader-card-action delete icon"
              onClick={() => cancelLoad(selectedId)}
              aria-label={t("loader.stopLoading")}
              title={t("loader.stopLoading")}
            >
              ⏹
            </button>
          )}
        </div>
      )}

      <DeleteModelDialog
        open={deleteTarget !== null}
        modelName={deleteTarget?.displayName}
        busy={deleteTarget !== null && busyId === deleteTarget.id}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
}
