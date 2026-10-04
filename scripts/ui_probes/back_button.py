"""Back button probe: in app Back keeps state, browser Back walks the screens."""
import sys

from playwright.sync_api import sync_playwright

import probe_common as pc

EXPECTED = [["results"], ["search"], ["home"]]


def in_app_back(page, query: str) -> list[str]:
    """Results scroll and the search text must survive two in app Backs."""
    problems = []
    pc.to_search(page)
    box = page.locator(pc.NAME_BOX).first
    box.fill(query)
    page.locator(pc.GO).click()
    pc.wait_cards(page, 1)
    # Let the app's own scroll to top run first, as it would for a person
    page.wait_for_timeout(800)
    # Mid page, since at the very bottom the restore is clamped short
    page.evaluate("() => scrollTo(0, Math.floor((document.documentElement.scrollHeight"
                  " - innerHeight) / 2))")
    page.wait_for_timeout(400)
    before = page.evaluate("() => scrollY")
    if before == 0:
        problems.append("Results did not scroll, so scroll restore was not tested")
    pc.open_card(page, min(6, pc.visible_cards(page).count() - 1))
    pc.back_from(page, "player", "results")
    page.wait_for_timeout(400)
    after = page.evaluate("() => scrollY")
    print(f"In app Back to Results: scroll {before} then {after}")
    if abs(after - before) > 5:
        problems.append(f"Results scroll not restored ({before} then {after})")
    pc.back_from(page, "results", "search")
    kept = page.locator(pc.NAME_BOX).first.input_value()
    print(f"In app Back to Search: search text {query!r} then {kept!r}")
    if kept != query:
        problems.append("search text lost on Back")
    return problems


def browser_back(root, back) -> list:
    """Walk to Player, then press browser Back four times."""
    root.locator(pc.START).first.click()
    root.locator(pc.screen_sel("search")).first.wait_for(state="visible")
    root.locator(pc.GO).click()
    root.locator(pc.CARD).first.wait_for(state="visible")
    root.locator(pc.CARD).nth(2).click()
    root.locator(pc.PLAYER_HEAD).first.wait_for(state="visible")
    steps = []
    for _ in range(4):
        back()
        steps.append(pc.visible_screens(None, root) if root_alive(root) else [])
    return steps


def root_alive(root) -> bool:
    try:
        root.locator("body").count()
        return True
    except Exception:  # noqa: BLE001, a page that left has no frame to ask
        return False


def main():
    ap = pc.base_args(__doc__)
    ap.add_argument("--query", default="a", help="text typed in the name box (default a)")
    ap.add_argument("--iframe", action="store_true",
                    help="also run inside an iframe, as huggingface.co/spaces embeds the app")
    args = ap.parse_args()
    base, space = pc.resolve(args)
    path = pc.page_path(args)
    problems = []

    with sync_playwright() as pw:
        s = pc.Session(pw, base, space, *pc.DESKTOP)
        page = s.open(path)
        problems += in_app_back(page, args.query)
        page.close()

        page = s.open(path)
        # history.back() is what the browser button does; go_back waits for a load
        steps = browser_back(page, lambda: (page.evaluate("history.back()"),
                                            page.wait_for_timeout(1500)))
        print(f"Browser Back x4: {steps} (last one should leave the app)")
        if steps[:3] != EXPECTED or steps[3]:
            problems.append(f"browser Back walked {steps}, expected {EXPECTED} then leave")

        if args.iframe:
            # A signed URL lets a private Space load inside a third party frame
            src = f"{base}{path or '/'}{s.sign_query}"
            host = s.ctx.new_page()
            host.goto("about:blank")
            host.set_content(f'<iframe src="{src}" style="width:100vw;height:95vh;border:0"></iframe>')
            frame = host.frame_locator("iframe")
            frame.locator(pc.screen_sel("home")).first.wait_for(state="visible", timeout=90000)
            steps = browser_back(frame, lambda: (host.evaluate("history.back()"),
                                                 host.wait_for_timeout(1500)))
            # In an iframe the last Back may leave the frame on Home or unload it
            print(f"Browser Back x4 inside an iframe: {steps}")
            if steps[:3] != EXPECTED:
                problems.append(f"iframe Back walked {steps}")
        s.close()

    print("Problems: " + ("; ".join(problems) if problems else "none"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
