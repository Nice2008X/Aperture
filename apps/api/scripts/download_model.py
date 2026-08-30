"""Download a Hugging Face model repo into data/models/<id>/ and write our manifest.json.

A thin CLI wrapper around aperture_api.downloads.download_sync — the same
logic POST /api/models/download uses (PLAN.md §8.6), minus progress
streaming, for populating data/models/ out-of-band instead of through the
running app.

Usage: python scripts/download_model.py <hf_repo_id> [--revision main]
"""

import argparse
import sys

from aperture_api.downloads import download_sync
from aperture_api.paths import MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face repo id, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    print(f"Downloading {args.repo}@{args.revision} -> {MODELS_DIR}", file=sys.stderr)
    manifest = download_sync(MODELS_DIR, args.repo, args.revision)
    print(f"Done. {manifest['sizeBytes'] / 1e9:.2f} GB, manifest written.", file=sys.stderr)


if __name__ == "__main__":
    main()
