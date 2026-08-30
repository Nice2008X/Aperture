import { useCallback, useEffect, useState, type FormEvent } from "react";
import type { LoadProgressEvent } from "@aperture/model-ir";
import { listModels, streamDownload, isDownloadDone, isDownloadError, type CatalogEntry } from "@aperture/api-client";
import { useTranslation } from "./LanguageContext.js";
import { formatBytes } from "../format.js";
import { LoadProgressBar, type UnifiedLoadProgress } from "./LoadProgressBar.js";

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
}

type CatalogState = { status: "loading" } | { status: "ready"; entries: CatalogEntry[] } | { status: "error"; error: string };

type DownloadState = { status: "idle" } | { status: "downloading"; downloadedBytes: number; totalBytes: number } | { status: "error"; error: string };

const IDLE_DOWNLOAD: DownloadState = { status: "idle" };

/**
 * Real checkpoints live in data/models/ on the GPU backend (PLAN.md §3) —
 * the catalog list is whatever's already downloaded there. Pasting a
 * Hugging Face repo id below triggers the backend to pull it on demand
 * (PLAN.md §8.6) — same "paste an id, it loads" UX the very first version
 * of this loader had, just downloading server-side now instead of
 * fetching straight into the browser.
 */
export function ModelLoader({ status, error, onLoad, excludeModelId, loadProgress, embedded }: Props) {
  const { t } = useTranslation();
  const [catalog, setCatalog] = useState<CatalogState>({ status: "loading" });
  const [repo, setRepo] = useState("");
  const [download, setDownload] = useState<DownloadState>(IDLE_DOWNLOAD);
  const [quantization, setQuantization] = useState<Quantization>(undefined);

  const refreshCatalog = useCallback(() => {
    return listModels()
      .then((entries) => setCatalog({ status: "ready", entries }))
      .catch((err) => setCatalog({ status: "error", error: err instanceof Error ? err.message : String(err) }));
  }, []);

  useEffect(() => {
    refreshCatalog();
  }, [refreshCatalog]);

  const startDownload = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = repo.trim();
    if (!trimmed) return;
    setDownload({ status: "downloading", downloadedBytes: 0, totalBytes: 0 });
    try {
      for await (const event of streamDownload(trimmed)) {
        if (isDownloadError(event)) throw new Error(event.error);
        if (isDownloadDone(event)) {
          setDownload(IDLE_DOWNLOAD);
          setRepo("");
          await refreshCatalog();
          onLoad(event.manifest.id, quantization);
          return;
        }
        setDownload({ status: "downloading", downloadedBytes: event.downloadedBytes, totalBytes: event.totalBytes });
      }
    } catch (err) {
      setDownload({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  };

  const entries = catalog.status === "ready" ? catalog.entries.filter((e) => e.id !== excludeModelId) : [];
  const downloading = download.status === "downloading";
  const modelLoading = status === "loading";
  // Download and GPU-load are sequential, never simultaneous (a download
  // completes and calls onLoad() itself — see startDownload below) — but
  // they stay in their own two positions below (download progress right
  // under the repo-id form the user just submitted; load progress up near
  // the catalog, where a catalog click or a just-finished download lands
  // next) rather than collapsing into one shared slot, which would yank a
  // download's progress away from where the user is actually looking.
  // What's genuinely unified is the LoadProgressBar component itself —
  // one bar/label/status implementation instead of two duplicated ones.
  const loadingProgress: UnifiedLoadProgress | null = modelLoading
    ? { phase: loadProgress?.phase ?? "loading_weights", current: loadProgress?.current, total: loadProgress?.total, unit: "count" }
    : null;
  const downloadProgress: UnifiedLoadProgress | null = downloading
    ? { phase: "downloading", current: download.downloadedBytes, total: download.totalBytes || undefined, unit: "bytes" }
    : null;

  return (
    <div className={"model-loader" + (embedded ? " embedded" : "")}>
      {!embedded && (
        <>
          <div className="model-loader-title">{t("loader.title")}</div>
          <div className="model-loader-sub">{t("loader.subtitle")}</div>
        </>
      )}

      {loadingProgress && (
        <div className="model-loader-loading-progress">
          <LoadProgressBar progress={loadingProgress} />
        </div>
      )}

      {catalog.status === "loading" && <div className="model-loader-catalog-status">Loading model catalog…</div>}
      {catalog.status === "error" && (
        <div className="model-loader-error">Couldn't reach the backend at load time: {catalog.error}</div>
      )}
      {catalog.status === "ready" && entries.length === 0 && (
        <div className="model-loader-catalog-status">No models downloaded yet — paste a Hugging Face repo id below to pull one.</div>
      )}

      <div className="model-loader-catalog">
        {entries.map((entry) => (
          <button
            key={entry.id}
            className="model-loader-catalog-entry"
            disabled={status === "loading" || entry.status !== "ready"}
            onClick={() => onLoad(entry.id, quantization)}
          >
            <span className="model-loader-catalog-name">{entry.displayName}</span>
            <span className="model-loader-catalog-meta">
              {entry.family}
              {entry.isMoe ? " · MoE" : ""} · {formatBytes(entry.sizeBytes)}
              {entry.status !== "ready" ? ` · ${entry.status}` : ""}
            </span>
          </button>
        ))}
      </div>

      <div className="model-loader-quantization">
        <span className="model-loader-quantization-label">{t("loader.precision")}</span>
        <select
          value={quantization ?? "none"}
          disabled={status === "loading" || downloading}
          onChange={(e) => setQuantization(e.target.value === "none" ? undefined : (e.target.value as "4bit" | "8bit"))}
        >
          <option value="none">{t("loader.precisionFull")}</option>
          <option value="8bit">{t("loader.precision8bit")}</option>
          <option value="4bit">{t("loader.precision4bit")}</option>
        </select>
      </div>

      {status === "error" && <div className="model-loader-error">{error}</div>}

      <form className="model-loader-download-form" onSubmit={startDownload}>
        <input
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          placeholder={t("loader.inputPlaceholder")}
          disabled={downloading}
        />
        <button type="submit" disabled={downloading || !repo.trim()}>
          {downloading ? t("loader.downloading") : t("loader.download")}
        </button>
      </form>
      {downloadProgress && (
        <div className="model-loader-download-progress">
          <LoadProgressBar progress={downloadProgress} />
        </div>
      )}
      {download.status === "error" && <div className="model-loader-error">{download.error}</div>}
    </div>
  );
}
