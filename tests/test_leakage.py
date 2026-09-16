import pytest

from pvf.features.leakage import assert_no_leakage, is_leaky


def test_snapshot_columns_flagged():
    assert is_leaky("players", "market_value_in_eur")
    assert is_leaky("player_profiles", "current_club_name")


def test_static_columns_allowed():
    assert not is_leaky("players", "date_of_birth")
    assert not is_leaky("player_profiles", "height")
    assert not is_leaky("player_valuations", "market_value_in_eur")


def test_assert_raises():
    with pytest.raises(ValueError):
        assert_no_leakage({"players": ["foot", "agent_name"]})
