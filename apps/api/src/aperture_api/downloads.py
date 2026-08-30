"""Downloads a Hugging Face repo into data/models/<id>/ and writes our
manifest.json — shared by the on-demand backend endpoint
(POST /api/models/download, PLAN.md §8.6) and scripts/download_model.py
(a thin CLI wrapper around download_sync, for when data/models is
populated out-of-band instead).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import queue
from collections.abc import AsyncIterator
from pathlib import Path

import os

# huggingface_hub disables its tqdm bars whenever stdout isn't a real TTY
# (`disable=None`'s auto-detection, see huggingface_hub.utils.tqdm.
# is_tqdm_disabled) — true for this process, running under uvicorn with
# output redirected to a log file. A *disabled* tqdm instance skips normal
# state tracking entirely (its `desc` attribute is never even set, and
# `.update()` stops advancing `.n`), not just the terminal rendering.
# `enable_progress_bars()` alone doesn't override this TTY check; the
# documented escape hatch is this env var, which must be set before
# huggingface_hub's tqdm module is imported (its disable decision reads it
# at bar-creation time, but importing early to be safe).
os.environ.setdefault("TQDM_POSITION", "-1")

from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import tqdm as hf_tqdm  # noqa: E402

# Only the files a model adapter actually needs — skip .bin duplicates,
# .gguf/.onnx exports, and other repo cruft some HF repos ship alongside
# the safetensors weights.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",
]


def model_id_for(repo: str) -> str:
    return repo.replace("/", "__")


def existing_manifest(models_dir: Path, repo: str) -> dict | None:
    manifest_path = models_dir / model_id_for(repo) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def _write_manifest(target_dir: Path, repo: str, revision: str) -> dict:
    size_bytes = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
    config = json.loads((target_dir / "config.json").read_text())
    is_moe = any("moe" in str(v).lower() or "expert" in k.lower() for k, v in config.items())
    manifest = {
        "id": model_id_for(repo),
        "hfRepo": repo,
        "revision": revision,
        "displayName": repo.split("/")[-1],
        "family": config.get("model_type", "unknown"),
        "isMoe": is_moe,
        "sizeBytes": size_bytes,
        "downloadedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "ready",
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def download_sync(models_dir: Path, repo: str, revision: str = "main") -> dict:
    """Blocking, no progress reporting — what the CLI script uses."""
    target_dir = models_dir / model_id_for(repo)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=target_dir, allow_patterns=ALLOW_PATTERNS)
    return _write_manifest(target_dir, repo, revision)


async def download_with_progress(models_dir: Path, repo: str, revision: str = "main") -> AsyncIterator[dict]:
    """Yields `{downloadedBytes, totalBytes, currentFile}` while
    downloading, then exactly one final `{done: True, manifest}` or
    `{error: "..."}`. A no-op fast path: if this repo already has a
    manifest, yields the done event immediately without touching the
    network — snapshot_download itself is already incremental (skips
    files whose ETag already matches), but skipping it entirely avoids
    even the "is everything up to date" round trip for the common case of
    re-selecting an already-downloaded model.
    """
    existing = existing_manifest(models_dir, repo)
    if existing is not None:
        yield {"done": True, "manifest": existing}
        return

    target_dir = models_dir / model_id_for(repo)
    target_dir.mkdir(parents=True, exist_ok=True)
    progress_queue: queue.Queue = queue.Queue()

    # snapshot_download already aggregates every file's bytes into one real
    # "Downloading bytes" tqdm bar internally (see its _AggregatedTqdm) —
    # individual per-file downloads get a *fake* tqdm that feeds into it,
    # not our tqdm_class, so there's no need (and no way) to sum per-file
    # progress ourselves. It also creates an unrelated "Fetching N files"
    # bar (file *count*, not bytes) and a "Reconstructing..." bar (bytes
    # written to disk, not received over the network) with the same
    # tqdm_class — desc is what tells these apart.
    class _ProgressTqdm(hf_tqdm):
        def update(self, n: int = 1) -> None:
            super().update(n)
            if self.desc == "Downloading bytes":
                progress_queue.put({"downloadedBytes": self.n, "totalBytes": self.total or 0})

    def run() -> None:
        snapshot_download(repo_id=repo, revision=revision, local_dir=target_dir, allow_patterns=ALLOW_PATTERNS, tqdm_class=_ProgressTqdm)

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        while True:
            try:
                yield progress_queue.get_nowait()
            except queue.Empty:
                if task.done():
                    break
                await asyncio.sleep(0.2)
        await task  # re-raises anything run() raised
        yield {"done": True, "manifest": _write_manifest(target_dir, repo, revision)}
    except Exception as e:  # noqa: BLE001 — any failure (bad repo id, network error, disk full, ...) becomes one clear event for the client
        yield {"error": str(e)}
