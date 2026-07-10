"""
data_source.py
==============

Pluggable Data & Strategy Core.

This module is a clean, decoupled abstraction layer that fetches OHLCV
(Open / High / Low / Close / Volume) time-series data from multiple markets:

    * Binance  -> live crypto (public REST klines endpoint)
    * yfinance -> USA stocks / ETFs

The design goal is that ANY new broker / data provider (Coinglass, Crypto.com,
Coinalize, Kraken, etc.) can be plugged in by writing ONE function with the
signature:

        def fetch_<name>(symbol: str, interval: str, limit: int) -> pandas.DataFrame

and registering it in the ``DATA_SOURCES`` dictionary at the bottom of the file.

Every fetched DataFrame is pushed through ``apply_custom_strategies(df)`` before
it leaves this module. That single hook is where YOU paste your own indicator /
strategy code. Any extra columns you create there automatically travel all the
way to the browser and get plotted — no frontend edits required.
"""

from __future__ import annotations

# IMPORTANT: bootstrap the Windows CA trust store BEFORE importing requests /
# yfinance, so HTTPS calls verify against the certs Windows trusts (fixes the
# antivirus/proxy "unable to get local issuer certificate" SSL errors).
import certs_bootstrap  # noqa: F401  (must be imported first)

import time
from typing import Callable, Dict, List

import pandas as pd
import requests

try:
    # yfinance is only needed for the stock bridge. We import lazily-safe so the
    # crypto side still works even if yfinance is not installed.
    import yfinance as yf
except Exception:  # pragma: no cover - defensive import
    yf = None

# Optional per-indicator style hints forwarded to the frontend.
# Populate this dict (name -> {"color", "type", "overlay", "scale", "base", ...})
# if you want a custom indicator to render with a specific color / on a separate
# oscillator scale. Columns with no entry here fall back to frontend defaults
# (a colored line overlaid on the price scale). Empty = no custom indicators.
INDICATOR_META: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
# Every fetch_* function MUST return a pandas DataFrame that contains at least
# these columns. "time" is a UNIX timestamp in SECONDS (UTC), which is exactly
# what TradingView Lightweight Charts expects.
BASE_COLUMNS: List[str] = ["time", "open", "high", "low", "close", "volume"]


class DataSourceError(Exception):
    """Raised when a data source cannot return valid data (bad ticker, rate
    limit, network failure, empty response, etc.)."""


# ---------------------------------------------------------------------------
# Interval mapping helpers
# ---------------------------------------------------------------------------
# The frontend speaks a single, normalized set of timeframes. Each data source
# translates those into whatever its own API expects.
NORMALIZED_INTERVALS: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Binance accepts these strings directly.
_BINANCE_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# yfinance uses a different vocabulary and, critically, has strict rules about
# how far back intraday data can go. We map each normalized interval to a
# (yf_interval, yf_period) pair that Yahoo will actually honour.
_YF_INTERVAL_MAP: Dict[str, Dict[str, str]] = {
    "1m": {"interval": "1m", "period": "5d"},
    "5m": {"interval": "5m", "period": "1mo"},
    "15m": {"interval": "15m", "period": "1mo"},
    # Yahoo has no native 4h; we approximate with 1h (still very usable) and a
    # longer window. 60m is Yahoo's hourly bar id.
    "1h": {"interval": "60m", "period": "3mo"},
    "4h": {"interval": "60m", "period": "6mo"},
    "1d": {"interval": "1d", "period": "5y"},
}


# ---------------------------------------------------------------------------
# THE STRATEGY HOOK  (this is the function you will edit later)
# ---------------------------------------------------------------------------
def apply_custom_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Apply YOUR custom technical indicators / strategies to the candle data.

    -----------------------------------------------------------------------
    WHAT THIS FUNCTION IS FOR
    -----------------------------------------------------------------------
    Right now this function does the bare minimum that is always required:

        1. Guarantees the DataFrame is sorted chronologically (ascending
           ``time``) — TradingView Lightweight Charts REQUIRES strictly
           ascending, de-duplicated timestamps or it will throw.
        2. Drops duplicate timestamps (keeps the last occurrence).
        3. Returns the DataFrame unchanged otherwise.

    -----------------------------------------------------------------------
    HOW TO PLUG IN YOUR OWN SCRIPT  (READ THIS CAREFULLY)
    -----------------------------------------------------------------------
    The `df` passed in ALWAYS contains these columns:

        time    -> UNIX timestamp in seconds (UTC), int
        open    -> float
        high    -> float
        low     -> float
        close   -> float
        volume  -> float

    To add an indicator, simply create a NEW COLUMN on `df`. Its name becomes
    the series name shown on the chart, and its numeric values become the plotted
    line. The rest of the pipeline (app.py serialization + static/app.js) will
    automatically detect any column that is NOT one of the six base columns and
    render it as its own line series on the chart — you do NOT touch the
    frontend at all.

    -------------------------------------------------------------------
    EXAMPLE A — using pandas-ta  (pip install pandas-ta)
    -------------------------------------------------------------------
        import pandas_ta as ta

        # Simple / Exponential moving averages
        df["SMA_20"] = ta.sma(df["close"], length=20)
        df["EMA_50"] = ta.ema(df["close"], length=50)

        # RSI (note: this will plot on the SAME price pane; for oscillators you
        # may prefer a separate lower pane — see app.js `separatePane` note)
        df["RSI_14"] = ta.rsi(df["close"], length=14)

        # MACD returns a DataFrame of several columns — join them in and they
        # will each become their own line automatically:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = df.join(macd)   # adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    -------------------------------------------------------------------
    EXAMPLE B — using TA-Lib  (pip install TA-Lib  + native lib)
    -------------------------------------------------------------------
        import talib

        df["SMA_20"]  = talib.SMA(df["close"], timeperiod=20)
        df["EMA_50"]  = talib.EMA(df["close"], timeperiod=50)
        df["RSI_14"]  = talib.RSI(df["close"], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df["close"])
        df["MACD"]        = macd
        df["MACD_signal"] = macdsignal
        df["MACD_hist"]   = macdhist

    -------------------------------------------------------------------
    EXAMPLE C — a full pasted custom strategy
    -------------------------------------------------------------------
    If you already have a script, wrap it as a function and call it here:

        df = my_existing_strategy(df)   # must return the same df + new columns

    -----------------------------------------------------------------------
    IMPORTANT RULES
    -----------------------------------------------------------------------
      * Only add NUMERIC columns. NaN values are fine — they are stripped out
        per-point on serialization (a moving average simply won't draw until it
        has enough bars).
      * Do NOT rename or drop the six base columns.
      * Keep the DataFrame indexed by row order; do not set a fancy index — the
        pipeline reads columns, not the index.
      * The sorting / de-duplication below MUST remain (or run first in your own
        code), otherwise the chart library will reject the data.

    Paste your indicator code in the clearly marked block below.
    """

    # --- Mandatory hygiene: sort + de-duplicate the time series ------------
    df = df.copy()
    df = df.sort_values("time", ascending=True)
    df = df.drop_duplicates(subset="time", keep="last")
    df = df.reset_index(drop=True)

    # ======================================================================
    # >>> PASTE YOUR CUSTOM INDICATOR / STRATEGY CODE BELOW THIS LINE <<<
    #
    #   e.g.
    #       import pandas_ta as ta
    #       df["SMA_20"] = ta.sma(df["close"], length=20)
    #       df["EMA_50"] = ta.ema(df["close"], length=50)
    #
    #   Any new columns you create here will be plotted automatically.
    # ======================================================================

    # (No custom indicators enabled — the DataFrame is returned intact.)

    # ======================================================================
    # >>> END OF YOUR CUSTOM CODE <<<
    # ======================================================================

    return df


# ---------------------------------------------------------------------------
# Data source #1: Binance (crypto)
# ---------------------------------------------------------------------------
# Public data host (data-api.binance.vision) en vez de api.binance.com: el
# primero NO tiene bloqueo geográfico, así que funciona desde datacenters / EE.UU.
# (ej. los runners de GitHub Actions, que Binance bloquea con HTTP 451).
_BINANCE_REST_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch historical klines from Binance's public REST endpoint.

    Args:
        symbol:   Binance trading pair, e.g. ``BTCUSDT`` (case-insensitive).
        interval: One of NORMALIZED_INTERVALS.
        limit:    Number of candles (max 1000 on Binance).

    Returns:
        DataFrame with the canonical BASE_COLUMNS.
    """
    binance_interval = _BINANCE_INTERVAL_MAP.get(interval)
    if binance_interval is None:
        raise DataSourceError(f"Unsupported interval for Binance: {interval!r}")

    params = {
        "symbol": symbol.upper().strip(),
        "interval": binance_interval,
        "limit": max(1, min(int(limit), 1000)),
    }

    try:
        resp = requests.get(_BINANCE_REST_URL, params=params, timeout=10)
    except requests.RequestException as exc:
        raise DataSourceError(f"Network error contacting Binance: {exc}") from exc

    if resp.status_code == 429:
        raise DataSourceError("Binance rate limit hit (HTTP 429). Slow down.")
    if resp.status_code == 400:
        # Binance returns 400 for unknown symbols.
        raise DataSourceError(f"Unknown Binance symbol {symbol!r} (HTTP 400).")
    if resp.status_code != 200:
        raise DataSourceError(
            f"Binance returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    rows = resp.json()
    if not rows:
        raise DataSourceError(f"Binance returned no data for {symbol!r}.")

    # Binance kline row layout:
    # [ openTime(ms), open, high, low, close, volume, closeTime, ... ]
    records = []
    for r in rows:
        records.append(
            {
                "time": int(r[0]) // 1000,  # ms -> seconds
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
        )

    return pd.DataFrame.from_records(records, columns=BASE_COLUMNS)


# ---------------------------------------------------------------------------
# Data source #2: yfinance (USA stocks)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Symbol resolution for Yahoo Finance
# ---------------------------------------------------------------------------
# Yahoo uses its own tickers that differ from the broker / TradingView style
# names traders are used to (e.g. gold is "GC=F", not "XAUUSD"). This map + a
# forex heuristic translate common symbols so they "just work".
_YF_ALIASES = {
    # Metals
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    "XCUUSD": "HG=F", "COPPER": "HG=F",
    # Energy
    "WTI": "CL=F", "USOIL": "CL=F", "OIL": "CL=F", "CRUDE": "CL=F",
    "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NGAS": "NG=F", "NATGAS": "NG=F",
    # Indices
    "US30": "^DJI", "DJI": "^DJI", "DJIA": "^DJI", "DOW": "^DJI",
    "US500": "^GSPC", "SPX": "^GSPC", "SPX500": "^GSPC", "SP500": "^GSPC",
    "US100": "^NDX", "NAS100": "^NDX", "NDX": "^NDX", "NASDAQ": "^IXIC",
    "US2000": "^RUT", "RUSSELL": "^RUT",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI", "DAX": "^GDAXI",
    "UK100": "^FTSE", "FTSE": "^FTSE",
    "FRA40": "^FCHI", "CAC": "^FCHI", "CAC40": "^FCHI",
    "EU50": "^STOXX50E", "ESP35": "^IBEX",
    "JP225": "^N225", "JPN225": "^N225", "NIKKEI": "^N225",
    "HK50": "^HSI", "HSI": "^HSI", "AUS200": "^AXJO", "VIX": "^VIX",
    # Crypto as Yahoo tickers (in case chosen under the Stock source)
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
}

_CURRENCIES = {
    "USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD",
    "CNH", "HKD", "SGD", "MXN", "ZAR", "TRY", "SEK", "NOK",
}


def _resolve_yahoo_symbol(symbol: str) -> str:
    """Translate a broker/TradingView-style symbol into a Yahoo Finance ticker."""
    s = symbol.upper().strip()
    if s in _YF_ALIASES:
        return _YF_ALIASES[s]
    # Already Yahoo-native (futures "GC=F", indices "^DJI", forex "EURUSD=X").
    if "=" in s or s.startswith("^"):
        return s
    # Forex heuristic: 6 letters made of two known currencies -> append "=X".
    if len(s) == 6 and s[:3] in _CURRENCIES and s[3:] in _CURRENCIES:
        return s + "=X"
    return s


def fetch_yfinance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch historical candles for a US stock/ETF via yfinance.

    Args:
        symbol:   Ticker, e.g. ``AAPL``.
        interval: One of NORMALIZED_INTERVALS.
        limit:    Max number of most-recent candles to keep.
    """
    if yf is None:
        raise DataSourceError(
            "yfinance is not installed. Run: pip install yfinance"
        )

    mapping = _YF_INTERVAL_MAP.get(interval)
    if mapping is None:
        raise DataSourceError(f"Unsupported interval for yfinance: {interval!r}")

    ticker = _resolve_yahoo_symbol(symbol)

    try:
        raw = yf.download(
            tickers=ticker,
            interval=mapping["interval"],
            period=mapping["period"],
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # yfinance raises a grab-bag of exceptions
        raise DataSourceError(f"yfinance error for {ticker!r}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataSourceError(
            f"No data returned for stock {ticker!r} "
            f"(bad ticker, market closed, or rate limited)."
        )

    # yfinance can return a MultiIndex column set when multiple tickers are
    # requested; flatten defensively so single-ticker access is uniform.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()

    # The datetime column is named "Datetime" for intraday and "Date" for daily.
    dt_col = "Datetime" if "Datetime" in raw.columns else "Date"

    dt_series = pd.to_datetime(raw[dt_col], utc=True)

    # Robust across pandas versions: convert each tz-aware datetime to epoch seconds.
    epoch_seconds = dt_series.map(lambda ts: int(ts.timestamp()))

    frame = pd.DataFrame(
        {
            "time": epoch_seconds.astype("int64"),
            "open": pd.to_numeric(raw["Open"], errors="coerce"),
            "high": pd.to_numeric(raw["High"], errors="coerce"),
            "low": pd.to_numeric(raw["Low"], errors="coerce"),
            "close": pd.to_numeric(raw["Close"], errors="coerce"),
            "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
        }
    )

    # Drop rows where the core price data is missing (holidays / halts).
    frame = frame.dropna(subset=["open", "high", "low", "close"])

    if frame.empty:
        raise DataSourceError(f"No valid candles for stock {ticker!r}.")

    # Keep only the most recent `limit` candles.
    frame = frame.tail(int(limit)).reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Live quote helper (used by the stock polling bridge in the frontend)
# ---------------------------------------------------------------------------
def fetch_stock_quote(symbol: str) -> float:
    """Return the latest available price for a stock ticker.

    Uses yfinance's fast_info when available and falls back to the last close of
    a 1-minute history pull. Raises DataSourceError on failure.
    """
    if yf is None:
        raise DataSourceError("yfinance is not installed.")

    ticker = _resolve_yahoo_symbol(symbol)
    try:
        t = yf.Ticker(ticker)
        # fast_info is the cheapest path.
        fast = getattr(t, "fast_info", None)
        if fast is not None:
            price = fast.get("last_price") or fast.get("lastPrice")
            if price:
                return float(price)

        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        raise DataSourceError(f"Could not fetch quote for {ticker!r}: {exc}") from exc

    raise DataSourceError(f"No live price available for {ticker!r}.")


# ---------------------------------------------------------------------------
# PLUGGABLE REGISTRY
# ---------------------------------------------------------------------------
# To add a new broker, write a fetch_<name>(symbol, interval, limit) function
# above and register it here. Nothing else in the codebase needs to change.
#
#   Example (future):
#       from my_coinglass import fetch_coinglass
#       DATA_SOURCES["coinglass"] = fetch_coinglass
#
DATA_SOURCES: Dict[str, Callable[[str, str, int], pd.DataFrame]] = {
    "crypto": fetch_binance,   # alias used by the frontend market selector
    "binance": fetch_binance,
    "stock": fetch_yfinance,   # alias used by the frontend market selector
    "yfinance": fetch_yfinance,
}


def get_ohlcv(source: str, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Top-level dispatcher: fetch data from `source`, then run the strategy hook.

    This is the ONLY function the Flask layer should call.

    Args:
        source:   Registered key in DATA_SOURCES, e.g. "crypto" or "stock".
        symbol:   Ticker / trading pair.
        interval: Normalized timeframe.
        limit:    Candle count.

    Returns:
        A fully processed DataFrame (base columns + any custom indicator columns).
    """
    key = (source or "").lower().strip()
    fetcher = DATA_SOURCES.get(key)
    if fetcher is None:
        raise DataSourceError(
            f"Unknown data source {source!r}. "
            f"Available: {', '.join(sorted(DATA_SOURCES))}"
        )

    df = fetcher(symbol, interval, limit)

    if df is None or df.empty:
        raise DataSourceError(f"No data for {symbol!r} from {source!r}.")

    # Run the user-editable strategy / indicator hook.
    df = apply_custom_strategies(df)
    return df


# ---------------------------------------------------------------------------
# Symbol lists (for the ticker search / autocomplete)
# ---------------------------------------------------------------------------
# Curated list of popular US tickers + ETFs for the stock search (yfinance has
# no practical "list everything" endpoint).
POPULAR_STOCKS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "BRK-B", "JPM", "V", "MA", "UNH", "HD", "PG", "COST", "JNJ", "ABBV", "WMT",
    "NFLX", "AMD", "INTC", "QCOM", "TXN", "MU", "ADBE", "CRM", "ORCL", "CSCO",
    "IBM", "NOW", "SHOP", "UBER", "ABNB", "PYPL", "SQ", "COIN", "PLTR", "SNOW",
    "MRNA", "PFE", "MRK", "LLY", "TMO", "DHR", "BMY", "AMGN", "GILD", "CVS",
    "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "SPGI",
    "XOM", "CVX", "COP", "SLB", "OXY", "PSX",
    "KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "BA", "CAT", "GE", "HON", "LMT", "RTX", "DE", "UPS", "FDX",
    "F", "GM", "RIVN", "LCID", "NIO",
    "BABA", "JD", "PDD", "TSM", "ASML", "SAP", "TM", "SONY",
    "SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "ARKK", "XLK", "XLF", "XLE",
    "GLD", "SLV", "USO", "TLT", "HYG", "SMH", "SOXL", "TQQQ", "SQQQ",
    "MSTR", "MARA", "RIOT", "HOOD", "SOFI", "GME", "AMC",
    # Commodities / forex / indices (broker-style names, resolved to Yahoo tickers)
    "XAUUSD", "XAGUSD", "XPTUSD", "COPPER", "WTI", "BRENT", "NGAS",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP",
    "US30", "US500", "US100", "US2000", "GER40", "UK100", "FRA40",
    "JP225", "HK50", "VIX",
]


# In-memory cache for the (large) Binance symbol list.
_binance_symbols_cache = {"ts": 0.0, "data": []}
_BINANCE_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/price"
# Quote assets kept in the crypto search, in priority order (USDT first).
_QUOTE_PRIORITY = ["USDT", "USDC", "FDUSD", "BTC", "ETH", "BNB"]


def _list_binance_symbols() -> List[str]:
    """Fetch & cache tradable Binance symbols (10-minute TTL)."""
    now = time.time()
    if _binance_symbols_cache["data"] and now - _binance_symbols_cache["ts"] < 600:
        return _binance_symbols_cache["data"]

    try:
        resp = requests.get(_BINANCE_TICKER_URL, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
    except requests.RequestException as exc:
        # On failure, return whatever we cached before (possibly empty).
        if _binance_symbols_cache["data"]:
            return _binance_symbols_cache["data"]
        raise DataSourceError(f"Could not fetch Binance symbols: {exc}") from exc

    symbols = [r["symbol"] for r in rows if "symbol" in r]

    def rank(sym: str) -> int:
        for i, q in enumerate(_QUOTE_PRIORITY):
            if sym.endswith(q):
                return i
        return len(_QUOTE_PRIORITY)

    filtered = [s for s in symbols if any(s.endswith(q) for q in _QUOTE_PRIORITY)]
    filtered.sort(key=lambda s: (rank(s), s))

    _binance_symbols_cache["data"] = filtered
    _binance_symbols_cache["ts"] = now
    return filtered


def list_symbols(source: str) -> List[str]:
    """Return the searchable symbol list for a market source."""
    key = (source or "").lower().strip()
    if key in ("crypto", "binance"):
        return _list_binance_symbols()
    if key in ("stock", "yfinance"):
        return list(POPULAR_STOCKS)
    return []


def dataframe_to_payload(df: pd.DataFrame) -> dict:
    """Serialize a processed DataFrame into the JSON payload the frontend expects.

    Output structure::

        {
            "candles": [
                {"time": 1700000000, "open": .., "high": .., "low": .., "close": .., "volume": ..},
                ...
            ],
            "indicators": {
                "SMA_20": [{"time": .., "value": ..}, ...],   # NaNs stripped per-point
                ...
            }
        }

    ANY column that is not one of the six BASE_COLUMNS is treated as a custom
    indicator line and packed into ``indicators`` automatically. This is what
    makes the strategy hook "just work" end-to-end.
    """
    # Candles (base OHLCV).
    candles = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        candles.append(
            {
                "time": int(d["time"]),
                "open": float(d["open"]),
                "high": float(d["high"]),
                "low": float(d["low"]),
                "close": float(d["close"]),
                "volume": float(d["volume"]) if pd.notna(d["volume"]) else 0.0,
            }
        )

    # Indicators (every extra column).
    indicator_columns = [c for c in df.columns if c not in BASE_COLUMNS]
    indicators: Dict[str, list] = {}
    for col in indicator_columns:
        series = []
        for t, v in zip(df["time"], df[col]):
            # Strip NaN / inf per point so lines only draw where data exists.
            if pd.isna(v):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):  # NaN / inf guard
                continue
            series.append({"time": int(t), "value": fv})
        # Only include indicators that actually have data.
        if series:
            indicators[str(col)] = series

    # Forward per-indicator style hints (color / type / scale / base) for any
    # column we have metadata for. Columns without metadata simply fall back to
    # frontend defaults (a colored line overlaid on the price scale).
    indicator_meta = {
        name: INDICATOR_META[name]
        for name in indicators
        if name in INDICATOR_META
    }

    return {
        "candles": candles,
        "indicators": indicators,
        "indicator_meta": indicator_meta,
    }
