import { useCallback, useEffect, useState, type FormEvent } from "react";
import type { LoadProgressEvent } from "@aperture/model-ir";
import { listModels, streamDownload, isDownloadDone, isDownloadError, type CatalogEntry } from "@aperture/api-client";
import { useTranslation } from "./LanguageContext.js";
import { formatBytes, formatCount } from "../format.js";

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

function loadPhaseLabel(phase: string): string {
  switch (phase) {
    case "loading_weights":
      return "Loading weights onto GPU";
    case "building_graph":
      return "Building architecture graph";
    default:
      return phase;
  }
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
  const downloadPct = downloading && download.totalBytes > 0 ? Math.min(100, (download.downloadedBytes / download.totalBytes) * 100) : 0;
  const modelLoading = status === "loading";
  const loadPct = loadProgress?.total ? Math.min(100, ((loadProgress.current ?? 0) / loadProgress.total) * 100) : undefined;

  return (
    <div className={"model-loader" + (embedded ? " embedded" : "")}>
      {!embedded && (
        <>
          <div className="model-loader-title">{t("loader.title")}</div>
          <div className="model-loader-sub">{t("loader.subtitle")}</div>
        </>
      )}

      {modelLoading && (
        <div className="model-loader-loading-progress">
          <div className="model-loader-loading-title">Loading model onto the GPU…</div>
          <div className="model-loader-download-bar-track">
            <div className="model-loader-download-bar-fill" style={loadPct !== undefined ? { width: `${loadPct}%` } : { width: "100%", opacity: 0.4 }} />
          </div>
          <span className="model-loader-download-status">
            {loadProgress ? (
              <>
                {loadPhaseLabel(loadProgress.phase)}
                {loadProgress.total ? ` — ${formatCount(loadProgress.current ?? 0)} / ${formatCount(loadProgress.total)}` : ""}
              </>
            ) : (
              "Starting…"
            )}
          </span>
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
      {downloading && (
        <div className="model-loader-download-progress">
          <div className="model-loader-download-bar-track">
            <div className="model-loader-download-bar-fill" style={{ width: `${downloadPct}%` }} />
          </div>
          <span className="model-loader-download-status">
            {download.totalBytes > 0 ? `${formatBytes(download.downloadedBytes)} / ${formatBytes(download.totalBytes)}` : "Starting…"}
          </span>
        </div>
      )}
      {download.status === "error" && <div className="model-loader-error">{download.error}</div>}
    </div>
  );
}
