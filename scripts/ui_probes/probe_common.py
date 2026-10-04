"""Shared Playwright helpers for the UI probes.

The probes find the app through the class hooks below, the "probe contract".
Any screen design works as long as it keeps these hooks.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

# Probe contract: class hooks the redesign must keep
SCREENS = ["home", "search", "results", "player"]
SCREEN = ".pvf-{name}"                                  # one per screen, plus .pvf-screen
START = ".pvf-home button.pvf-start, .pvf-home button.pvf-fwd"   # Home to Search
GO = ".pvf-search button.pvf-go"                         # runs the search
NAME_BOX = ".pvf-search input[type=text], .pvf-search textarea"
CARD = "button.pvf-card[data-pid]"                       # one per player card
PLAYER_HEAD = ".pvf-player-head[data-pid]"               # Player screen header
BACK = ".pvf-{name} button.pvf-back"                     # in app Back per screen

DESKTOP = (1366, 900)
PHONE = (390, 844)
DEFAULT_SPACE = "jackstev/pvf-preview"


def base_args(description: str) -> argparse.ArgumentParser:
    """Arguments every probe shares: where the app is and which page."""
    ap = argparse.ArgumentParser(description=description)
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--local", nargs="?", const="http://127.0.0.1:7860", metavar="URL",
                       help="probe a local run (default URL http://127.0.0.1:7860)")
    where.add_argument("--space", nargs="?", const=DEFAULT_SPACE, metavar="OWNER/NAME",
                       help=f"probe a Hugging Face Space (default {DEFAULT_SPACE})")
    ap.add_argument("--path", default="",
                    help="page path, e.g. tabs; leave off the slash in Git Bash")
    return ap


def resolve(args) -> tuple[str, str | None]:
    """Base URL and Space id; a local run is the default target."""
    if args.space:
        # Space subdomains are lower case with / _ . turned into dashes
        host = args.space.lower().replace("/", "-").replace("_", "-").replace(".", "-")
        return f"https://{host}.hf.space", args.space
    return (args.local or "http://127.0.0.1:7860").rstrip("/"), None


def page_path(args) -> str:
    # Git Bash rewrites a leading slash into a Windows path, so accept "tabs"
    p = args.path.strip().replace("\\", "/")
    if ":" in p:
        p = "/" + p.rsplit("/", 1)[-1]
    return "" if p in ("", "/") else "/" + p.lstrip("/")


def hf_token() -> str:
    """Token from HF_TOKEN, else the file huggingface-cli login writes."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return (home / "token").read_text(encoding="utf-8").strip()


def space_jwt(space: str) -> str:
    """Short lived signed token for a private Space; the HF token stays here."""
    req = urllib.request.Request(f"https://huggingface.co/api/spaces/{space}/jwt",
                                 headers={"Authorization": f"Bearer {hf_token()}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["token"]


class Session:
    """One browser context at one viewport, signed in to a private Space if needed."""

    def __init__(self, pw, base: str, space: str | None, width: int, height: int):
        self.base = base
        self.browser = pw.chromium.launch()
        phone = width < 600
        self.ctx = self.browser.new_context(
            viewport={"width": width, "height": height}, device_scale_factor=2 if phone else 1,
            is_mobile=phone, has_touch=phone)
        self.sign_query = ""
        if space:
            self.sign_query = f"?__sign={space_jwt(space)}"
            # One signed load sets the spaces-jwt cookie; later loads are plain
            page = self.ctx.new_page()
            page.goto(f"{base}/{self.sign_query}", wait_until="load", timeout=180000)
            page.close()

    def open(self, path: str = "", query: str = ""):
        """New tab on the Home screen; networkidle never fires with Gradio."""
        page = self.ctx.new_page()
        page.goto(f"{self.base}{path}{query}", wait_until="load", timeout=180000)
        wait_screen(page, "home", 90000)
        return page

    def close(self):
        self.browser.close()


def screen_sel(name: str) -> str:
    return SCREEN.format(name=name)


def wait_screen(page, name: str, timeout: int = 30000):
    page.locator(screen_sel(name)).first.wait_for(state="visible", timeout=timeout)


def visible_screens(page, frame=None) -> list[str]:
    """Which of the four screens are showing right now."""
    root = frame or page
    out = []
    for n in SCREENS:
        loc = root.locator(screen_sel(n))
        if loc.count() and loc.first.is_visible():
            out.append(n)
    return out


def timed(fn) -> float:
    """Milliseconds a callable takes, measured on the driver side."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def to_search(page):
    page.locator(START).first.click()
    wait_screen(page, "search")


def visible_cards(page):
    return page.locator(f"{CARD} >> visible=true")


def wait_cards(page, n: int = 24, not_first_pid: str | None = None, timeout: int = 90000):
    """Wait for n visible cards; optionally require a new first card."""
    page.wait_for_function(
        """([sel, n, prev]) => {
             const els = [...document.querySelectorAll(sel)].filter(e => e.offsetParent !== null);
             return els.length >= n && els[0].dataset.pid !== prev;
           }""", arg=[CARD, n, not_first_pid or ""], timeout=timeout)


def open_card(page, i: int) -> tuple[float, str, str]:
    """Click card i; ms until the Player header shows that exact id."""
    card = visible_cards(page).nth(i)
    pid = card.get_attribute("data-pid")
    t0 = time.perf_counter()
    card.click()
    # A stale header from an earlier player must not end the wait
    page.wait_for_function(
        """([sel, pid]) => { const h = document.querySelector(sel);
             return !!h && h.offsetParent !== null && h.dataset.pid === pid; }""",
        arg=[PLAYER_HEAD, pid], timeout=60000)
    ms = (time.perf_counter() - t0) * 1000
    return ms, pid, page.locator(PLAYER_HEAD).get_attribute("data-pid")


def back_from(page, name: str, to: str) -> float:
    return timed(lambda: (page.locator(BACK.format(name=name)).first.click(), wait_screen(page, to)))


def summary(values: list[float]) -> str:
    return (f"median {statistics.median(values):4.0f} ms "
            f"(min {min(values):.0f}, max {max(values):.0f}, n={len(values)})")
