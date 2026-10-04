"""Upload app/ to a Hugging Face Space. Run `huggingface-cli login` first."""
import argparse
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
# Front matter line the Space shows as the page title
TITLE_LINE = re.compile(r"^title:.*$", re.MULTILINE)


def retitled_copy(folder: Path, title: str, into: Path) -> Path:
    """Copy the folder and rewrite the README title, leaving the repo untouched."""
    staged = into / folder.name
    shutil.copytree(folder, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    readme = staged / "README.md"
    if not readme.is_file():
        raise SystemExit(f"{folder} has no README.md to retitle")
    text = readme.read_text(encoding="utf-8")
    # Only the first match, so a title mentioned later in the body is left alone
    new_text, count = TITLE_LINE.subn(f"title: {title}", text, count=1)
    if not count:
        raise SystemExit(f"{readme} has no title line in its front matter")
    readme.write_text(new_text, encoding="utf-8")
    return staged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="e.g. your-username/player-value-forecaster")
    ap.add_argument("--folder", default="app",
                    help="folder under the repo root to upload (default: app)")
    ap.add_argument("--private", action="store_true",
                    help="create the Space private, and refuse to upload if it is public")
    ap.add_argument("--title", help="override the README title in the upload only, so a "
                                    "preview Space cannot be mistaken for the live one")
    args = ap.parse_args()
    folder = APP_DIR if args.folder == "app" else ROOT / args.folder
    api = HfApi()
    # exist_ok keeps reruns idempotent; None leaves visibility to the Hub default
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True,
                    private=True if args.private else None)
    # create_repo never changes an existing Space, so check it really is private
    if args.private and not api.repo_info(args.repo_id, repo_type="space").private:
        raise SystemExit(f"{args.repo_id} already exists and is public; not uploading")

    with tempfile.TemporaryDirectory() as tmp:
        # A retitled copy is staged outside the repo, so git never sees the change
        upload_from = retitled_copy(folder, args.title, Path(tmp)) if args.title else folder
        api.upload_folder(folder_path=upload_from, repo_id=args.repo_id, repo_type="space",
                          ignore_patterns=["__pycache__/*", "*.pyc"])
    if args.title:
        print(f"Uploaded with the title {args.title!r}; {folder}/README.md is unchanged")
    print(f"Deployed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
