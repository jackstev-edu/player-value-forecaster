"""Generate a fake prediction bundle so the GUI can be built before models exist.

Players and clubs are invented on purpose: no real person gets a made-up value.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvf.config import load_config  # noqa: E402
from pvf.export.predictions import write_bundle  # noqa: E402

# Real competition codes so league filters look like the final app
LEAGUES = [
    ("GB1", "Premier League", "England", 3.0), ("ES1", "LaLiga", "Spain", 2.4),
    ("IT1", "Serie A", "Italy", 2.1), ("L1", "Bundesliga", "Germany", 2.2),
    ("FR1", "Ligue 1", "France", 1.8), ("NL1", "Eredivisie", "Netherlands", 1.1),
    ("PO1", "Liga Portugal", "Portugal", 1.1), ("TR1", "Super Lig", "Turkey", 0.9),
]
POSITIONS = {"Attack": ["Centre-Forward", "Left Winger", "Right Winger"],
             "Midfield": ["Central Midfield", "Defensive Midfield", "Attacking Midfield"],
             "Defender": ["Centre-Back", "Left-Back", "Right-Back"],
             "Goalkeeper": ["Goalkeeper"]}
NATIONS = ["England", "Spain", "France", "Brazil", "Argentina", "Germany", "Portugal",
           "Netherlands", "Italy", "Nigeria", "Japan", "United States", "Turkey", "Belgium"]


def age_drift(age: float) -> float:
    """Rough log-value drift per year: rises early, falls after the late 20s."""
    return 0.25 - 0.022 * (age - 18) if age < 28 else -0.05 - 0.04 * (age - 28)


def main(n_players: int = 400, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp("2026-07-01")
    players, history, forecasts = [], [], []

    for i in range(n_players):
        pid = 900000 + i
        code, lname, country, strength = LEAGUES[rng.integers(len(LEAGUES))]
        pos = rng.choice(list(POSITIONS), p=[0.3, 0.33, 0.3, 0.07])
        age = float(rng.integers(17, 36))
        # League strength shifts the typical value level
        log_v = rng.normal(np.log(3e6) + 0.6 * strength, 1.0)

        # Walk backwards to fake a value history of twice-yearly updates
        dates = pd.date_range(end=anchor - pd.Timedelta(days=20), periods=int(min(age - 15, 10) * 2),
                              freq="6MS")
        path = [log_v]
        for d in dates[:-1][::-1]:
            a = age - (anchor - d).days / 365.25
            path.append(path[-1] - age_drift(a) / 2 + rng.normal(0, 0.18))
        vals = np.exp(np.array(path[::-1]))
        vals = np.clip(np.round(vals / 5e4) * 5e4, 5e4, 2e8)
        history += [{"player_id": pid, "date": d, "value_eur": float(v)} for d, v in zip(dates, vals)]

        players.append({
            "player_id": pid, "name": f"Mock Player {i + 1:03d}", "age": int(age),
            "position": pos, "sub_position": rng.choice(POSITIONS[pos]),
            "nationality": rng.choice(NATIONS), "club_name": f"Mock {lname} Club {rng.integers(1, 19):02d}",
            "league_id": code, "league_name": lname, "league_country": country,
            "current_value_eur": float(vals[-1]), "value_date": dates[-1],
        })

        # Forecast band widens with horizon, centre follows the age drift
        mu, now = 0.0, float(vals[-1])
        for h in (1, 2, 3):
            mu += age_drift(age + h - 1)
            sd = 0.35 * np.sqrt(h)
            forecasts.append({
                "player_id": pid, "horizon": h, "anchor_date": anchor,
                "target_date": anchor + pd.DateOffset(years=h),
                "p10_eur": now * np.exp(mu - 1.2816 * sd), "p50_eur": now * np.exp(mu),
                "p90_eur": now * np.exp(mu + 1.2816 * sd), "model": "mock",
            })

    cfg = load_config()
    out = write_bundle(cfg["paths"]["bundle"], pd.DataFrame(players), pd.DataFrame(history),
                       pd.DataFrame(forecasts), is_mock=True, model_version="mock-0",
                       data_versions={"player-scores": "n/a", "football-datasets": "n/a"},
                       notes="Synthetic bundle for GUI development. Not real players.")
    print(f"Wrote mock bundle with {n_players} players to {out}")


if __name__ == "__main__":
    main()
