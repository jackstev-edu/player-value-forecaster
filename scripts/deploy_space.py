"""Upload app/ to a Hugging Face Space. Run `huggingface-cli login` first."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="e.g. your-username/player-value-forecaster")
    ap.add_argument("--folder", default="app",
                    help="folder under the repo root to upload (default: app)")
    ap.add_argument("--private", action="store_true",
                    help="create the Space private, and refuse to upload if it is public")
    args = ap.parse_args()
    folder = APP_DIR if args.folder == "app" else ROOT / args.folder
    api = HfApi()
    # exist_ok keeps reruns idempotent; None leaves visibility to the Hub default
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True,
                    private=True if args.private else None)
    # create_repo never changes an existing Space, so check it really is private
    if args.private and not api.repo_info(args.repo_id, repo_type="space").private:
        raise SystemExit(f"{args.repo_id} already exists and is public; not uploading")
    api.upload_folder(folder_path=folder, repo_id=args.repo_id, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    print(f"Deployed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
