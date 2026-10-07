"""Off-the-shelf model: Chronos-Bolt, a pretrained time-series model, used zero-shot.

It sees only the player's own Transfermarkt value history up to the anchor, resampled to a
monthly step series, and forecasts 12 months ahead. It is never trained on our samples, so
the contract data cannot help it; it is the "history alone" counterpart to the GBM.
"""
import numpy as np
import pandas as pd

DEFAULT_MODEL = "amazon/chronos-bolt-small"
QUANTILES = [0.1, 0.5, 0.9]


def monthly_context(valuations: pd.DataFrame, player_id: int, anchor: pd.Timestamp,
                    max_months: int = 240) -> np.ndarray:
    """Value on the anchor and on each month before it, oldest first, back to the first valuation.

    Each point is the latest valuation on or before that date, so a value is held until the
    next one, as Transfermarkt shows it. Valuations after the anchor are never read.
    """
    v = valuations[(valuations["player_id"] == player_id) & (valuations["date"] <= anchor)]
    v = v.dropna(subset=["market_value_in_eur"]).sort_values("date")
    if v.empty:
        return np.array([], dtype=float)
    first = v["date"].iloc[0]
    months = min(max_months, (anchor.year - first.year) * 12 + anchor.month - first.month)
    grid = pd.DatetimeIndex([anchor - pd.DateOffset(months=k) for k in range(months, -1, -1)])
    pos = np.searchsorted(v["date"].to_numpy(), grid.to_numpy(), side="right") - 1
    values = v["market_value_in_eur"].to_numpy(dtype=float)
    return values[pos[pos >= 0]]


def contexts_for(valuations: pd.DataFrame, player_ids, anchors) -> list[np.ndarray]:
    """monthly_context for each (player, anchor) pair, grouping the lookups by player."""
    by_player = {pid: g for pid, g in valuations.groupby("player_id")}
    empty = valuations.iloc[0:0]
    return [monthly_context(by_player.get(pid, empty), pid, pd.Timestamp(a))
            for pid, a in zip(player_ids, anchors)]


class ChronosForecaster:
    """Wraps a Chronos-Bolt pipeline: value series in, one-season log-ratio quantiles out."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu", pipeline=None):
        self.model_id = model_id
        if pipeline is None:
            import torch
            from chronos import BaseChronosPipeline
            pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device,
                                                           torch_dtype=torch.float32)
        self.pipeline = pipeline

    def predict(self, contexts: list[np.ndarray], value_now: np.ndarray, months: int = 12,
                batch_size: int = 64) -> pd.DataFrame:
        """q10, q50, q90 of log(value in `months` / value_now), plus `fallback`.

        A player with no history, or no current value, gets the no-change forecast with no band
        (`fallback` True), so every row has a median and the scores stay comparable.
        """
        import torch
        value_now = np.asarray(value_now, dtype=float)
        n = len(contexts)
        out = np.full((n, len(QUANTILES)), np.nan)
        ok = np.array([len(c) > 0 for c in contexts]) & np.isfinite(value_now) & (value_now > 0)
        idx = np.flatnonzero(ok)
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            batch = [torch.tensor(contexts[i], dtype=torch.float32) for i in chunk]
            q, _ = self.pipeline.predict_quantiles(batch, prediction_length=months,
                                                   quantile_levels=QUANTILES)
            eur = np.asarray(q[:, months - 1, :], dtype=float)
            # A value cannot fall to zero; floor the forecast at 1% of today's value
            eur = np.maximum(eur, 0.01 * value_now[chunk, None])
            out[chunk] = np.sort(np.log(eur / value_now[chunk, None]), axis=1)
        frame = pd.DataFrame(out, columns=["q10", "q50", "q90"])
        frame["fallback"] = ~ok
        frame.loc[frame["fallback"], "q50"] = 0.0
        return frame


def conformal_shift(q10: np.ndarray, q90: np.ndarray, y: np.ndarray, level: float = 0.8) -> float:
    """How far to widen (or narrow, if negative) a band so it holds `level` of the rows given.

    The same split-conformal rule QuantileGBM.calibrate uses, for a model we cannot refit.
    """
    keep = np.isfinite(q10) & np.isfinite(q90) & np.isfinite(y)
    scores = np.maximum(q10[keep] - y[keep], y[keep] - q90[keep])
    n = len(scores)
    if n == 0:
        return 0.0
    rank = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, rank, method="higher"))
