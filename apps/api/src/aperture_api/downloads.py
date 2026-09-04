"""Downloads a Hugging Face repo into data/models/<id>/ and writes our
manifest.json — shared by the on-demand backend endpoint
(POST /api/models/download, PLAN.md §8.6) and scripts/download_model.py
(a thin CLI wrapper around download_sync, for when data/models is
populated out-of-band instead).
"""

from __future__ import annotations

import asyncio
import datetime
import fnmatch
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

# Xet-backed repos (hf_xet is installed — increasingly the default storage
# for popular Hub repos) hand the actual transfer to a native Rust download
# loop, which only special-cases KeyboardInterrupt for stopping early (see
# huggingface_hub.file_download.xet_get's `except KeyboardInterrupt:
# abort_xet_session(); raise`) — a DownloadCancelled raised from *our*
# Python progress callback below doesn't reliably interrupt it, since
# that's not the mechanism Xet's own cancellation path is built around.
# Forcing every download through the plain HTTP path keeps Pause/Cancel
# reliable, at the cost of Xet's dedup/CDN speed advantage. Must be set
# before huggingface_hub.constants is first imported — it reads the env
# var once into a module-level constant, not per-call.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from huggingface_hub.utils import tqdm as hf_tqdm  # noqa: E402
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # noqa: E402

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


class DownloadCancelled(Exception):
    """Raised (from inside the progress-tracking tqdm hook below) when a
    client calls request_cancel() for this repo while its download is in
    flight. Both "Pause" and "Cancel" in the UI trigger this exact same
    thing — huggingface_hub always discards a file's partial bytes on any
    interrupted download (its `_download_to_tmp_and_move` unlinks the temp
    file in a `finally`, unconditional on why the download stopped), so
    there's no lower-level distinction between the two to make here. What a
    frontend "Resume" gets that a fresh download doesn't is every *other*
    file: snapshot_download skips a file whose destination already exists,
    so re-calling download_with_progress for the same repo only re-fetches
    whichever single file was actively transferring at the moment of the
    interrupt.
    """


# Keyed by repo id — this app loads/downloads one model at a time in
# practice, but keying by repo (rather than one bare global flag) means a
# stale cancel from a previous repo can't accidentally abort an unrelated
# one that happens to start next. _active_downloads gates request_cancel so
# a cancel click that arrives after (or between) downloads can't linger in
# _cancel_requested and instantly abort a *future*, unrelated download for
# the same repo.
_active_downloads: set[str] = set()
_cancel_requested: set[str] = set()


def request_cancel(repo: str) -> bool:
    """Marks repo's in-flight download for cancellation. Returns False (and
    does nothing) if no download for this repo is currently running."""
    if repo not in _active_downloads:
        return False
    _cancel_requested.add(repo)
    return True


def _non_weight_bytes(repo: str, revision: str) -> int | None:
    """Total size of every ALLOW_PATTERNS file that *isn't* a weights
    shard — used to tell "downloading model weights" apart from
    "downloading tokenizer/config files" purely from the byte counter
    snapshot_download already reports (see download_with_progress's phase
    logic), since huggingface_hub's own progress hooks don't expose which
    literal file is in flight (only anonymous, aggregated byte/file
    counts — the `tqdm_class` passed to `snapshot_download` is never
    forwarded to the individual per-file downloads it fans out to). A
    metadata-only Hub API call, no repo content downloaded. Returns None
    on any failure (network hiccup, gated repo, unexpected response shape)
    so the caller can fall back to an undifferentiated "downloading"
    phase instead of blocking the real download on this.
    """
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        return sum(
            s.size or 0
            for s in info.siblings or []
            if any(fnmatch.fnmatch(s.rfilename, pat) for pat in ALLOW_PATTERNS) and not fnmatch.fnmatch(s.rfilename, "*.safetensors*")
        )
    except Exception:  # noqa: BLE001 — best-effort UI enhancement, never worth failing the download over
        return None


def check_model_support(repo: str, revision: str = "main") -> dict:
    """Best-effort pre-download check: is repo's config.json `model_type`
    one that transformers' AutoModelForCausalLM actually knows how to
    instantiate? A repo can have a perfectly ordinary, up-to-date
    config.json and still never load here — its architecture might just
    not be a `transformers` model at all (see PLAN.md §11's `needle2`
    case: a proprietary mobile-runtime format whose `model_type: "needle"`
    isn't registered anywhere in `transformers`, so
    `AutoModelForCausalLM.from_pretrained` fails immediately regardless of
    how the download itself goes). Fetches only config.json (a few KB, not
    the weights), so this is cheap to run before committing to a real
    download.

    Returns `{"modelType": ..., "supported": ...}` with `supported: None`
    (never False) whenever this can't be determined confidently — a
    network hiccup, a gated repo, a bad revision, or a config with no
    `model_type` field — so a caller only ever warns on an actual,
    known-unsupported architecture, never on a merely-inconclusive check.
    """
    try:
        config_path = hf_hub_download(repo_id=repo, filename="config.json", revision=revision)
        config = json.loads(Path(config_path).read_text())
    except Exception:  # noqa: BLE001 — gated repo, bad revision, network error, etc. all just mean "can't tell"
        return {"modelType": None, "supported": None}

    model_type = config.get("model_type")
    if not model_type:
        return {"modelType": None, "supported": None}
    return {"modelType": model_type, "supported": model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}


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
    """Yields `{downloadedBytes, totalBytes, phase}` while downloading, then
    exactly one final `{done: True, manifest}`, `{cancelled: True}`, or
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
    non_weight_bytes = _non_weight_bytes(repo, revision)
    _active_downloads.add(repo)

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
            # Checked on every bar this tqdm_class drives (not just
            # "Downloading bytes") so a cancel/pause lands promptly
            # regardless of which phase the download is in when it's
            # requested — this fires once per network chunk during the
            # actual transfer (~10KB by default), so in practice this is
            # sub-second, not "wait for the next file".
            if repo in _cancel_requested:
                raise DownloadCancelled(repo)
            if self.desc == "Downloading bytes":
                downloaded = self.n
                # Small metadata files (config/tokenizer) finish almost
                # instantly relative to a multi-GB weight shard — once
                # cumulative transfer bytes pass everything that *isn't* a
                # weights file, whatever's left flowing in is the weights.
                # Not literally "this exact file", since up to 8 files
                # download concurrently (max_workers), but accurate for
                # the overwhelming common case those tiny files really are
                # tiny next to the weights.
                phase = "downloading_weights" if non_weight_bytes is not None and downloaded >= non_weight_bytes else "downloading_config"
                event: dict = {"downloadedBytes": downloaded, "totalBytes": self.total or 0}
                if non_weight_bytes is not None:
                    event["phase"] = phase
                progress_queue.put(event)

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
    except DownloadCancelled:
        yield {"cancelled": True}
    except Exception as e:  # noqa: BLE001 — any failure (bad repo id, network error, disk full, ...) becomes one clear event for the client
        yield {"error": str(e)}
    finally:
        # Always clear, regardless of how this generator exits (done, error,
        # or cancelled) — otherwise a resumed download for the same repo
        # would see its *previous* cancel request still armed and abort on
        # its very first progress tick.
        _active_downloads.discard(repo)
        _cancel_requested.discard(repo)
