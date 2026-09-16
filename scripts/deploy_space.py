"""Upload app/ to a Hugging Face Space. Run `huggingface-cli login` first."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="e.g. your-username/player-value-forecaster")
    args = ap.parse_args()
    api = HfApi()
    # exist_ok keeps reruns idempotent
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=APP_DIR, repo_id=args.repo_id, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    print(f"Deployed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
