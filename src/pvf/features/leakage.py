"""Guard against snapshot columns that only hold scrape-time values.

Source: Analysis Guide section 3. Using any of these on historical rows
makes the backtest look great and live forecasts poor.
"""

# Snapshot columns that leak the future into past anchors
LEAKY_COLUMNS = {
    "players": {
        "market_value_in_eur", "highest_market_value_in_eur", "current_club_id",
        "current_club_name", "current_club_domestic_competition_id",
        "contract_expiration_date", "agent_name", "international_caps",
        "international_goals", "current_national_team_id", "last_season",
    },
    # Almost every profile column is a snapshot, so allowlist instead
    "player_profiles": "ALL_EXCEPT_STATIC",
}

# Fixed facts that are safe at any anchor date
STATIC_SAFE = {"player_id", "date_of_birth", "foot", "height", "height_in_cm",
               "country_of_citizenship", "citizenship", "position", "sub_position",
               "main_position", "is_eu"}


def is_leaky(table: str, column: str) -> bool:
    """True if a column from this table cannot be used for historical rows."""
    rule = LEAKY_COLUMNS.get(table)
    if rule is None:
        return False
    if rule == "ALL_EXCEPT_STATIC":
        return column not in STATIC_SAFE
    return column in rule


def assert_no_leakage(used: dict[str, list[str]]) -> None:
    """Raise if any selected feature source column is on the leakage list."""
    bad = [f"{t}.{c}" for t, cols in used.items() for c in cols if is_leaky(t, c)]
    if bad:
        raise ValueError(f"Leaky snapshot columns selected: {bad}")
