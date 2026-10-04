"""Timing probe: SEARCH to cards, card to Player, and screen changes."""
from playwright.sync_api import sync_playwright

import probe_common as pc


def main():
    ap = pc.base_args(__doc__)
    ap.add_argument("--reps", type=int, default=6, help="timed runs after one warm up (default 6)")
    ap.add_argument("--cards", type=int, default=24, help="cards a search must show (default 24)")
    ap.add_argument("--toggle", default="Midfield",
                    help="filter label flipped each run so the first card changes (default Midfield)")
    args = ap.parse_args()
    base, space = pc.resolve(args)
    path = pc.page_path(args)
    print(f"Timing {base}{path or '/'}, {args.reps} runs after a warm up")

    with sync_playwright() as pw:
        s = pc.Session(pw, base, space, *pc.DESKTOP)
        page = s.open(path)
        to_search, to_home, search, open_ms, back_ms = [], [], [], [], []
        ids_ok = True
        # Screen changes without data: Home to Search and Back again
        for rep in range(args.reps + 1):
            fwd = pc.timed(lambda: pc.to_search(page))
            back = pc.back_from(page, "search", "home")
            if rep:
                to_search.append(fwd)
                to_home.append(back)
        pc.to_search(page)
        first = None
        for rep in range(args.reps + 1):
            # Same filters twice would reuse old cards and time nothing
            page.locator(".pvf-search label", has_text=args.toggle).first.click()
            t = pc.timed(lambda: (page.locator(pc.GO).click(),
                                  pc.wait_cards(page, args.cards, not_first_pid=first)))
            first = pc.visible_cards(page).first.get_attribute("data-pid")
            o, want, got = pc.open_card(page, (rep * 5) % args.cards)
            b = pc.back_from(page, "player", "results")
            if rep:  # the first run warms caches and is dropped
                search.append(t)
                open_ms.append(o)
                back_ms.append(b)
                ids_ok &= want == got
            pc.back_from(page, "results", "search")
        s.close()

    print(f"Home to Search        {pc.summary(to_search)}")
    print(f"Search to Home (Back) {pc.summary(to_home)}")
    print(f"SEARCH to {args.cards} cards   {pc.summary(search)}")
    print(f"Card to Player        {pc.summary(open_ms)}")
    print(f"Player to Results     {pc.summary(back_ms)}")
    print(f"Clicked id opened every time: {ids_ok}")


if __name__ == "__main__":
    main()
