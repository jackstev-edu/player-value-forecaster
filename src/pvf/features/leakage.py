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


# What each feature module is allowed to read, declared rather than inferred. Task 12:
# every slot's inputs go past `assert_no_leakage`, and `tests/test_leakage.py` fails if a
# module appears in `pvf/features` without a row here — so a new slot cannot be added
# without saying what it reads.
#
# Passing this list is necessary, not sufficient: it catches a snapshot column by name,
# and says nothing about reading a real column at the wrong date. The behavioural test in
# `tests/test_leakage.py` covers that half by rebuilding the panel with the future added
# and asserting no feature column moves.
FEATURE_SOURCES = {
    "build_panel": {
        "games": ["game_id", "competition_id", "season", "home_club_id", "away_club_id"],
        "game_lineups": ["game_id", "player_id", "club_id"],
        "appearances": ["game_id", "player_id", "player_club_id", "minutes_played"],
        "players": ["player_id", "date_of_birth", "position", "sub_position", "foot",
                    "height_in_cm", "country_of_citizenship"],
        "player_valuations": ["player_id", "date", "market_value_in_eur"],
        "player_profiles": ["player_id", "is_eu"],
    },
    "history": {
        "player_valuations": ["player_id", "date", "market_value_in_eur"],
    },
    "performance": {
        "games": ["game_id", "season"],
        "game_lineups": ["game_id", "player_id", "club_id", "type"],
        "appearances": ["game_id", "player_id", "player_club_id", "goals", "assists"],
    },
    "context": {
        "games": ["game_id", "competition_id", "season", "home_club_id", "away_club_id"],
        "game_lineups": ["game_id", "player_id", "club_id"],
        "appearances": ["game_id", "player_id", "player_club_id"],
        "player_valuations": ["player_id", "date", "market_value_in_eur"],
        "players": ["player_id", "position"],
    },
    "health": {
        "player_injuries": ["player_id", "from_date", "end_date", "days_missed"],
        "transfer_history": ["player_id", "transfer_date", "transfer_type",
                             "transfer_fee"],
    },
}
