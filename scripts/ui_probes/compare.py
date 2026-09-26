"""Compare two screenshot folders, e.g. a local SSR run against the preview Space."""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).parent


def diff(a: Path, b: Path, tol: int, out: Path) -> tuple[float, float]:
    """Share of pixels that differ by more than tol, and the mean difference."""
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    if ia.size != ib.size:
        # Different page heights still compare over the shared top area
        w, h = min(ia.width, ib.width), min(ia.height, ib.height)
        ia, ib = ia.crop((0, 0, w, h)), ib.crop((0, 0, w, h))
    d = ImageChops.difference(ia, ib).convert("L")
    hist = d.histogram()
    total = sum(hist)
    changed = sum(hist[tol + 1:]) / total
    mean = sum(i * n for i, n in enumerate(hist)) / total
    # Changed pixels in red over a faded copy, to see where they differ
    mask = d.point(lambda v: 255 if v > tol else 0)
    faded = Image.blend(ia, Image.new("RGB", ia.size, "white"), 0.6)
    faded.paste(Image.new("RGB", ia.size, (230, 0, 0)), mask=mask)
    faded.save(out / f"diff_{a.name}")
    return changed, mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left", nargs="?", default=str(HERE / "out" / "local"))
    ap.add_argument("right", nargs="?", default=str(HERE / "out" / "space"))
    ap.add_argument("--tol", type=int, default=24, help="grey level difference ignored (default 24)")
    ap.add_argument("--max-changed", type=float, default=0.02,
                    help="share of changed pixels that still counts as a match (default 0.02)")
    args = ap.parse_args()
    left, right = Path(args.left), Path(args.right)
    out = HERE / "out" / "diff"
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in left.glob("*.png") if (right / p.name).exists())
    if not names:
        sys.exit(f"No matching PNG names in {left} and {right}")
    worst = 0.0
    for name in names:
        changed, mean = diff(left / name, right / name, args.tol, out)
        worst = max(worst, changed)
        flag = "match" if changed <= args.max_changed else "DIFFERS"
        print(f"{flag:8s} {name:24s} {changed:6.2%} of pixels changed, mean diff {mean:4.1f}")
    print(f"Diff images in {out}")
    sys.exit(0 if worst <= args.max_changed else 1)


if __name__ == "__main__":
    main()
