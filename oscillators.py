"""
oscillators.py
==============

RSI + MACD computation using the pure-Python `ta` library
(https://github.com/bukosabino/ta).

    * RSI            -> ta.momentum.RSIIndicator(...).rsi()
    * MACD line      -> ta.trend.MACD(...).macd()
    * MACD signal    -> ta.trend.MACD(...).macd_signal()
    * MACD histogram -> ta.trend.MACD(...).macd_diff()

All indicators are created with `fillna=True` so the leading NaN warmup values
are filled, and every point is additionally guarded to be finite before being
serialized — this prevents Lightweight Charts from choking on NaN/Infinity.

These oscillators are returned as their own payload keys (`rsi`, `macd`) so the
frontend can render them in DEDICATED sub-panes below the price chart, with their
time scales synchronized to the main chart.

Entry point:
    compute_oscillators(df, cfg) -> dict
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD


DEFAULTS = {
    "rsi": True,
    "rsi_period": 14,
    "macd": True,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}


def _series_to_points(time, series):
    """Zip a time array with a value series, skipping any non-finite point."""
    out = []
    for t, v in zip(time, series):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        out.append({"time": int(t), "value": fv})
    return out


def compute_oscillators(df: pd.DataFrame, cfg: Optional[dict] = None) -> dict:
    """Compute RSI and/or MACD for a candle DataFrame.

    Args:
        df:  DataFrame with columns time, close.
        cfg: dict overriding DEFAULTS (feature toggles + periods).

    Returns:
        dict possibly containing:
            "rsi":  [{"time","value"}, ...]
            "macd": {"macd":[...], "signal":[...], "hist":[...]}
    """
    c = dict(DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if v is not None})

    result = {}
    if len(df) < 2:
        return result

    close = df["close"].astype(float)
    time = df["time"].to_numpy()

    # ---- RSI ----
    if c["rsi"]:
        period = max(2, int(c["rsi_period"]))
        rsi = RSIIndicator(close=close, window=period, fillna=True).rsi()
        result["rsi"] = _series_to_points(time, rsi)

    # ---- MACD ----
    if c["macd"]:
        fast = max(1, int(c["macd_fast"]))
        slow = max(2, int(c["macd_slow"]))
        signal = max(1, int(c["macd_signal"]))
        # ta expects window_slow > window_fast; swap defensively if misconfigured.
        if slow <= fast:
            slow, fast = fast + 1, fast
        macd = MACD(
            close=close,
            window_slow=slow,
            window_fast=fast,
            window_sign=signal,
            fillna=True,
        )
        result["macd"] = {
            "macd": _series_to_points(time, macd.macd()),
            "signal": _series_to_points(time, macd.macd_signal()),
            "hist": _series_to_points(time, macd.macd_diff()),
        }

    return result
