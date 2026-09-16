"""Contract between pipeline and GUI. The app reads exactly these tables."""

PLAYERS_COLUMNS = [
    "player_id", "name", "age", "position", "sub_position", "nationality",
    "club_name", "league_id", "league_name", "league_country", "current_value_eur",
    "value_date",
]
HISTORY_COLUMNS = ["player_id", "date", "value_eur"]
FORECASTS_COLUMNS = [
    "player_id", "horizon", "anchor_date", "target_date", "p10_eur", "p50_eur", "p90_eur", "model",
]
MANIFEST_KEYS = ["created_at", "is_mock", "model_version", "data_versions", "notes"]


def validate(players, history, forecasts) -> None:
    """Fail fast if a bundle is missing columns or has bad bands."""
    for name, df, cols in [("players", players, PLAYERS_COLUMNS),
                           ("history", history, HISTORY_COLUMNS),
                           ("forecasts", forecasts, FORECASTS_COLUMNS)]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    # Quantile bands must be ordered for the chart to make sense
    if not ((forecasts["p10_eur"] <= forecasts["p50_eur"])
            & (forecasts["p50_eur"] <= forecasts["p90_eur"])).all():
        raise ValueError("forecast quantiles are not ordered")
    orphans = set(forecasts["player_id"]) - set(players["player_id"])
    if orphans:
        raise ValueError(f"{len(orphans)} forecast rows reference unknown players")
