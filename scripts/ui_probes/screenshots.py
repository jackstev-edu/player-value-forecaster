"""Screenshot probe: all four screens at desktop and phone width, plus font check."""
from pathlib import Path

from playwright.sync_api import sync_playwright

import probe_common as pc

OUT_ROOT = Path(__file__).parent / "out"

FONT_JS = """async (family) => {
  await document.fonts.ready;
  const faces = [...document.fonts].filter(f => f.family.replace(/"/g, '') === family);
  return {loaded: faces.filter(f => f.status === 'loaded').map(f => f.weight),
          check: document.fonts.check(`700 40px "${family}"`)};
}"""
OVERFLOW_JS = "() => document.documentElement.scrollWidth > innerWidth"


def shoot(page, out: Path, name: str, full: bool, problems: list) -> None:
    """Save one screenshot and note it if the page scrolls sideways."""
    # Fixed layers like the pitch land mid page in full page shots
    page.screenshot(path=str(out / f"{name}.png"), full_page=full)
    if page.evaluate(OVERFLOW_JS):
        problems.append(f"{name}: page scrolls sideways")


def main():
    ap = pc.base_args(__doc__)
    ap.add_argument("--out", help="output folder (default scripts/ui_probes/out/<local|space>)")
    ap.add_argument("--font", default="Barlow Condensed", help="web font that must load")
    ap.add_argument("--full-page", action="store_true", help="whole page, not just the viewport")
    ap.add_argument("--settle", type=int, default=800, help="ms to wait before each shot")
    args = ap.parse_args()
    base, space = pc.resolve(args)
    path = pc.page_path(args)
    out = Path(args.out) if args.out else OUT_ROOT / ("space" if space else "local")
    out.mkdir(parents=True, exist_ok=True)
    problems = []

    with sync_playwright() as pw:
        for tag, (w, h) in (("desktop", pc.DESKTOP), ("phone390", pc.PHONE)):
            s = pc.Session(pw, base, space, w, h)
            page = s.open(path)
            page.wait_for_timeout(args.settle)
            font = page.evaluate(FONT_JS, args.font)
            print(f"{tag}: {args.font} weights loaded {font['loaded']}, check {font['check']}")
            if not font["check"]:
                problems.append(f"{tag}: {args.font} did not load")
            shoot(page, out, f"{tag}_1_home", args.full_page, problems)
            pc.to_search(page)
            page.wait_for_timeout(args.settle)
            shoot(page, out, f"{tag}_2_search", args.full_page, problems)
            page.locator(pc.GO).click()
            pc.wait_cards(page, 1)
            page.wait_for_timeout(args.settle)
            shoot(page, out, f"{tag}_3_results", args.full_page, problems)
            pc.open_card(page, 0)
            # The chart draws after the header, so give it longer
            page.wait_for_timeout(args.settle * 2)
            shoot(page, out, f"{tag}_4_player", args.full_page, problems)
            s.close()

    print(f"Saved 8 screenshots to {out}")
    print("Problems: " + ("; ".join(problems) if problems else "none"))


if __name__ == "__main__":
    main()
