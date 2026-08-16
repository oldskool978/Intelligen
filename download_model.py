#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def align_tokenizer_metadata(cache_root: Path) -> None:
    for config_path in cache_root.glob("**/tokenizer/tokenizer_config.json"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("fix_mistral_regex", False):
                data["fix_mistral_regex"] = True
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logging.info("Aligned tokenizer regex metadata: %s", config_path)
        except Exception as e:
            logging.warning("Tokenizer metadata alignment skipped for %s: %s", config_path, str(e))

def download_weights(repo_id: str, max_workers: int = 8) -> None:
    logging.info("Target Cache Anchor: %s", CACHE_DIR)
    logging.info("Initiating acquisition for repository: %s", repo_id)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        max_workers=max_workers,
        resume_download=True
    )
    align_tokenizer_metadata(CACHE_DIR)
    logging.info("Acquisition complete. Snapshot synchronized into local hub cache.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire model weights into localized Hugging Face cache.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="MiniMaxAI/MiniMax-Music3",
        help="Hugging Face model repository ID."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads."
    )
    args = parser.parse_args()

    try:
        download_weights(repo_id=args.repo_id, max_workers=args.max_workers)
    except Exception as e:
        logging.error("Acquisition failed: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()