"""Contrast probe: every visible text node against the pixels really behind it.

For each screen it records each text node's colour and line boxes, hides all
text, screenshots, and samples the background under those boxes. Gradients,
translucent panels and the CSS pitch are therefore judged as rendered.
"""
import io
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

import probe_common as pc

# Collects text nodes fully inside the viewport, with their own line boxes
TEXT_JS = """() => {
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const text = n.textContent.trim();
    const el = n.parentElement;
    if (!text || !el || el.closest('svg, script, style, noscript')) continue;
    if (!el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
    const cs = getComputedStyle(el);
    const range = document.createRange(); range.selectNodeContents(n);
    const rects = [...range.getClientRects()].filter(r => r.width > 1 && r.height > 1)
      .filter(r => r.top >= 0 && r.left >= 0 && r.bottom <= innerHeight && r.right <= innerWidth)
      .map(r => [r.left, r.top, r.right, r.bottom]);
    if (!rects.length) continue;
    const cls = (typeof el.className === 'string' ? el.className : '').split(' ')
      .filter(c => c && !c.startsWith('svelte-')).slice(0, 2).join('.');
    out.push({text: text.slice(0, 40), color: cs.color, size: parseFloat(cs.fontSize),
              weight: parseInt(cs.fontWeight) || 400, where: el.tagName.toLowerCase() + (cls ? '.' + cls : ''),
              rects});
  }
  return out;
}"""

# Hides every glyph but keeps layout, so the shot shows only backgrounds
HIDE_TEXT = """* , *::before, *::after, ::placeholder {
  color: transparent !important; -webkit-text-fill-color: transparent !important;
  text-shadow: none !important; caret-color: transparent !important; }"""


def parse_rgba(css: str):
    """Chrome reports rgb(a) for sRGB colours; anything else is skipped."""
    if not css.startswith("rgb"):
        return None
    parts = [float(v) for v in css[css.index("(") + 1:-1].replace("/", ",").replace(" ", ",").split(",") if v]
    return (*parts[:3], parts[3] if len(parts) > 3 else 1.0)


def luminance(c) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def worst_contrast(color, pixels, pct: float) -> tuple[float, tuple]:
    """Contrast against the pct worst background pixels, and that pixel."""
    scored = []
    for bg in pixels:
        # Semi transparent text is blended over each background pixel first
        a = color[3]
        fg = tuple(color[i] * a + bg[i] * (1 - a) for i in range(3))
        scored.append((ratio(fg, bg), bg))
    scored.sort(key=lambda t: t[0])
    return scored[min(len(scored) - 1, int(len(scored) * pct))]


def check_view(page, dpr: float, pct: float) -> list[dict]:
    """All text in the current viewport, each with its worst contrast."""
    items = page.evaluate(TEXT_JS)
    handle = page.add_style_tag(content=HIDE_TEXT)
    page.wait_for_timeout(100)
    shot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    handle.evaluate("el => el.remove()")
    results = []
    for it in items:
        color = parse_rgba(it["color"])
        if color is None or color[3] == 0:
            continue
        pixels = []
        for l, t, r, b in it["rects"]:
            box = tuple(int(v * dpr) for v in (l, t, r, b))
            # Sample every second pixel; boxes are small so this stays quick
            crop = shot.crop(box)
            data = crop.get_flattened_data() if hasattr(crop, "get_flattened_data") else crop.getdata()
            pixels.extend(list(data)[::2])
        if not pixels:
            continue
        cr, bg = worst_contrast(color, pixels, pct)
        results.append({**it, "ratio": cr, "bg": bg})
    return results


def scan_screen(page, dpr: float, pct: float) -> list[dict]:
    """Scroll the page one viewport at a time and check each view."""
    seen, out = set(), []
    height = page.evaluate("() => innerHeight")
    total = page.evaluate("() => document.documentElement.scrollHeight")
    y = 0
    while True:
        page.evaluate(f"() => window.scrollTo(0, {y})")
        page.wait_for_timeout(250)
        # The last step can be clamped, so read where the page really is
        at = page.evaluate("() => scrollY")
        for res in check_view(page, dpr, pct):
            # The same node can appear in two views; page position identifies it
            key = (res["where"], res["text"], round(res["rects"][0][1] + at), round(res["rects"][0][0]))
            if key in seen:
                continue
            seen.add(key)
            out.append(res)
        if y + height >= total:
            break
        y += int(height * 0.8)
    page.evaluate("() => window.scrollTo(0, 0)")
    return out


def main():
    ap = pc.base_args(__doc__)
    ap.add_argument("--min", type=float, default=4.5, help="required ratio (default 4.5)")
    ap.add_argument("--allow-large", action="store_true",
                    help="accept 3:1 for large text (24px, or 18.66px bold), as WCAG AA does")
    ap.add_argument("--pct", type=float, default=0.01,
                    help="judge against the worst this share of background pixels (default 0.01)")
    ap.add_argument("--show-all", action="store_true", help="list passing text too")
    args = ap.parse_args()
    base, space = pc.resolve(args)
    path = pc.page_path(args)
    failures, checked = [], 0

    with sync_playwright() as pw:
        for tag, (w, h) in (("desktop", pc.DESKTOP), ("phone390", pc.PHONE)):
            s = pc.Session(pw, base, space, w, h)
            page = s.open(path)
            dpr = page.evaluate("() => devicePixelRatio")
            steps = {"home": lambda: None,
                     "search": lambda: pc.to_search(page),
                     "results": lambda: (page.locator(pc.GO).click(), pc.wait_cards(page, 1)),
                     "player": lambda: pc.open_card(page, 0)}
            for screen, go in steps.items():
                go()
                # Park the mouse so no card is judged in its hover state
                page.mouse.move(0, 0)
                page.wait_for_timeout(1200)
                for res in scan_screen(page, dpr, args.pct):
                    checked += 1
                    large = res["size"] >= 24 or (res["size"] >= 18.66 and res["weight"] >= 700)
                    need = 3.0 if (args.allow_large and large) else args.min
                    ok = res["ratio"] >= need
                    if not ok or args.show_all:
                        line = (f"{'ok  ' if ok else 'FAIL'} {res['ratio']:5.2f}:1 need {need}  "
                                f"{tag}/{screen}  {res['where']}  \"{res['text']}\"  "
                                f"text {res['color']} on rgb{res['bg']}")
                        print(line)
                    if not ok:
                        failures.append(res)
            s.close()

    print(f"\nChecked {checked} text items on 4 screens at 2 widths; {len(failures)} below the bar.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
