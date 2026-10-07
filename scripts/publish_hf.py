"""Publish the prototype package to the Hugging Face Hub under one account.

Uploads the staged dataset (scripts/build_hf_dataset.py), the two model repos
(scripts/export_models.py) and the app as a Space. Every {HF_USER} in a Markdown file is replaced
with the account name in a temporary copy, so the repo files stay generic. Run `hf auth login`
first. scripts/deploy_space.py remains for uploading the app alone, e.g. to a preview Space.

Usage:
    python scripts/publish_hf.py <hf-username> [--only dataset lgbm chronos space] [--dry-run]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "data" / "processed"
# target name -> (repo type, repo name, local folder)
TARGETS = {
    "dataset": ("dataset", "football-contracts-500", STAGED / "hf_dataset"),
    "lgbm": ("model", "player-value-lgbm-core", STAGED / "hf_models" / "lgbm-core"),
    "chronos": ("model", "player-value-chronos-bolt", STAGED / "hf_models" / "chronos"),
    "space": ("space", "player-value-forecaster", ROOT / "app"),
}


def filled_copy(folder: Path, user: str, into: Path) -> Path:
    """Copy the folder with {HF_USER} filled in every Markdown file."""
    staged = into / folder.name
    shutil.copytree(folder, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for md in staged.rglob("*.md"):
        md.write_text(md.read_text(encoding="utf-8").replace("{HF_USER}", user), encoding="utf-8")
    return staged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("user", help="Hugging Face username that will own all four repos")
    ap.add_argument("--only", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    ap.add_argument("--dry-run", action="store_true", help="stage and list files, upload nothing")
    args = ap.parse_args()
    api = HfApi()
    if not args.dry_run:
        who = api.whoami()["name"]
        if who != args.user:
            raise SystemExit(f"logged in as {who!r}, not {args.user!r}; run `hf auth login`")

    for name in args.only:
        repo_type, repo_name, folder = TARGETS[name]
        if not folder.is_dir():
            raise SystemExit(f"{folder} is missing; build it first (see the README run order)")
        repo_id = f"{args.user}/{repo_name}"
        with tempfile.TemporaryDirectory() as tmp:
            staged = filled_copy(folder, args.user, Path(tmp))
            left = [p for p in staged.rglob("*.md") if "{HF_USER}" in p.read_text(encoding="utf-8")]
            if left:
                raise SystemExit(f"placeholder left in {left}")
            if args.dry_run:
                files = sorted(str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file())
                print(f"[dry run] {repo_type} {repo_id}: {len(files)} files: {', '.join(files)}")
                continue
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=True,
                            space_sdk="gradio" if repo_type == "space" else None)
            api.upload_folder(folder_path=staged, repo_id=repo_id, repo_type=repo_type,
                              commit_message="Publish the CMU 24-679 Project 1 prototype package",
                              ignore_patterns=["__pycache__/*", "*.pyc"])
        prefix = {"dataset": "datasets/", "space": "spaces/", "model": ""}[repo_type]
        print(f"published https://huggingface.co/{prefix}{repo_id}")


if __name__ == "__main__":
    main()
