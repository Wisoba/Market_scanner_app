from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


_HTTP_JSON_CACHE: dict[str, tuple[pd.Timestamp, dict]] = {}


def _cache_ttl_seconds() -> int:
    raw = os.getenv("MARKET_DATA_CACHE_SECONDS", "20")
    try:
        return max(0, int(raw))
    except ValueError:
        return 20


def _http_timeout_seconds() -> int:
    raw = os.getenv("MARKET_DATA_HTTP_TIMEOUT_SECONDS", "8")
    try:
        return max(3, int(raw))
    except ValueError:
        return 8


def load_env_file(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _alpaca_credentials() -> tuple[Optional[str], Optional[str]]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    return key, secret


def _alpaca_feed(default: str = "sip") -> str:
    return (os.getenv("ALPACA_FEED") or default).strip().lower() or default


@dataclass
class EngineConfig:
    lookback: int = 60
    atr_window: int = 14
    fast_ma: int = 10
    slow_ma: int = 30
    breakout_window: int = 20
    chop_window: int = 14
    trend_window: int = 20
    min_signal_strength: float = 0.18
    min_confidence: float = 0.55
    max_chop_for_trade: float = 0.62
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    max_hold_bars: int = 12
    transaction_cost_bps: float = 4.0
    slippage_bps: float = 2.0
    initial_capital: float = 10000.0
    risk_fraction: float = 0.15
    account_risk_fraction: float = 0.01


def _normalize(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def _readability_label(score: float) -> str:
    if score >= 72.0:
        return "CLEAN"
    if score >= 48.0:
        return "MIXED"
    return "CHAOTIC"


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path)

    # Yahoo-style exports can come with an extra metadata row:
    # Price,Close,High,Low,Open,Volume
    # Ticker,SPY,SPY,SPY,SPY,SPY
    # Date,...
    if list(raw.columns) and str(raw.columns[0]).lower() == "price":
        raw = pd.read_csv(path, skiprows=[1])
        renamed = {}
        for col in raw.columns:
            col_str = str(col)
            lower = col_str.lower()
            if lower.startswith("price"):
                renamed[col] = "Date"
            elif lower.startswith("close"):
                renamed[col] = "Close"
            elif lower.startswith("high"):
                renamed[col] = "High"
            elif lower.startswith("low"):
                renamed[col] = "Low"
            elif lower.startswith("open"):
                renamed[col] = "Open"
            elif lower.startswith("volume"):
                renamed[col] = "Volume"
        raw = raw.rename(columns=renamed)
        if "Date" in raw.columns:
            raw = raw[raw["Date"].astype(str).str.lower() != "date"].copy()

    cols = {c.lower(): c for c in raw.columns}

    date_col = cols.get("date")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    volume_col = cols.get("volume")

    if date_col is None or close_col is None:
        raise ValueError("CSV must include at least Date and Close columns.")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_col]),
            "Open": pd.to_numeric(raw[open_col], errors="coerce") if open_col else pd.to_numeric(raw[close_col], errors="coerce"),
            "High": pd.to_numeric(raw[high_col], errors="coerce") if high_col else pd.to_numeric(raw[close_col], errors="coerce"),
            "Low": pd.to_numeric(raw[low_col], errors="coerce") if low_col else pd.to_numeric(raw[close_col], errors="coerce"),
            "Close": pd.to_numeric(raw[close_col], errors="coerce"),
            "Volume": pd.to_numeric(raw[volume_col], errors="coerce") if volume_col else np.nan,
        }
    ).dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if df["Volume"].isna().all():
        df["Volume"] = 1.0
    else:
        df["Volume"] = df["Volume"].fillna(df["Volume"].median())

    return df


def make_synthetic_market(n: int = 360, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close = [100.0]
    open_ = [100.0]
    high = [101.0]
    low = [99.0]
    volume = [1_000_000.0]

    drift = 0.0009
    sigma = 0.010
    for i in range(1, n):
        if i == 80:
            drift = -0.0014
            sigma = 0.013
        elif i == 150:
            drift = 0.0018
            sigma = 0.009
        elif i == 240:
            drift = 0.0001
            sigma = 0.006
        elif i == 300:
            drift = -0.0020
            sigma = 0.015

        gap = rng.normal(0.0, sigma * 0.35)
        intraday = rng.normal(drift, sigma)
        prev_close = close[-1]
        o = prev_close * (1.0 + gap)
        c = o * (1.0 + intraday)
        intraday_range = abs(rng.normal(0.012, 0.004))
        h = max(o, c) * (1.0 + intraday_range * 0.5)
        l = min(o, c) * (1.0 - intraday_range * 0.5)
        v = volume[-1] * (1.0 + rng.normal(0.0, 0.16))

        open_.append(o)
        close.append(c)
        high.append(h)
        low.append(l)
        volume.append(max(v, 100_000.0))

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def make_synthetic_symbol_market(symbol: str, n: int = 360) -> pd.DataFrame:
    seed = int(sum((i + 1) * ord(ch) for i, ch in enumerate(symbol.upper()))) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    symbol = symbol.upper()
    profiles = {
        "NVDA": (0.0018, 0.018),
        "AAPL": (0.0010, 0.011),
        "MSFT": (0.0011, 0.010),
        "META": (0.0014, 0.014),
        "TSLA": (0.0003, 0.025),
        "AMD": (0.0013, 0.019),
        "AMZN": (0.0009, 0.012),
        "QQQ": (0.0008, 0.009),
        "SPY": (0.0006, 0.007),
    }
    drift, sigma = profiles.get(symbol, (0.0007, 0.012))

    close = [100.0 + rng.normal(0.0, 3.0)]
    open_ = [close[0]]
    high = [close[0] * 1.01]
    low = [close[0] * 0.99]
    volume = [float(rng.integers(800_000, 8_000_000))]

    for i in range(1, n):
        regime_shift = 0.0
        if 60 < i < 110:
            regime_shift = drift * 1.8
        elif 160 < i < 210:
            regime_shift = -drift * 1.2
        elif 260 < i < 320:
            regime_shift = drift * 2.2

        gap = rng.normal(0.0, sigma * 0.25)
        intraday = rng.normal(drift + regime_shift, sigma)
        prev_close = close[-1]
        o = prev_close * (1.0 + gap)
        c = o * (1.0 + intraday)
        bar_range = abs(rng.normal(0.014, 0.005))
        h = max(o, c) * (1.0 + bar_range * 0.5)
        l = min(o, c) * (1.0 - bar_range * 0.5)
        v = volume[-1] * (1.0 + rng.normal(0.0, 0.18))

        open_.append(o)
        close.append(max(c, 1.0))
        high.append(max(h, 1.0))
        low.append(max(min(l, h), 1.0))
        volume.append(max(v, 100_000.0))

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def make_synthetic_market_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {symbol: make_synthetic_symbol_market(symbol) for symbol in symbols}


def load_symbol_csv_dir(csv_dir: str | Path) -> dict[str, pd.DataFrame]:
    csv_dir = Path(csv_dir)
    symbol_to_df: dict[str, pd.DataFrame] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        try:
            df = load_ohlcv_csv(path)
        except Exception:
            continue
        if len(df) < 80:
            continue
        symbol_to_df[path.stem.upper()] = df
    return symbol_to_df


def _http_get_json(url: str, headers: Optional[dict[str, str]] = None, cache_seconds: int | None = None) -> dict:
    ttl = _cache_ttl_seconds() if cache_seconds is None else max(0, cache_seconds)
    now = pd.Timestamp.now(tz="UTC")
    if ttl > 0:
        cached = _HTTP_JSON_CACHE.get(url)
        if cached is not None:
            fetched_at, payload = cached
            if (now - fetched_at).total_seconds() <= ttl:
                return payload

    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=_http_timeout_seconds()) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if ttl > 0:
        _HTTP_JSON_CACHE[url] = (now, payload)
    return payload


def _alpaca_headers() -> dict[str, str]:
    key, secret = _alpaca_credentials()
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY or ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY in environment.")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _bars_to_df(bars: list[dict], utc: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(df["t"], utc=utc),
            "Open": pd.to_numeric(df["o"], errors="coerce"),
            "High": pd.to_numeric(df["h"], errors="coerce"),
            "Low": pd.to_numeric(df["l"], errors="coerce"),
            "Close": pd.to_numeric(df["c"], errors="coerce"),
            "Volume": pd.to_numeric(df["v"], errors="coerce"),
        }
    ).dropna().sort_values("Date").reset_index(drop=True)


def _fetch_alpaca_bars_payload(
    symbols: list[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feed: str,
    limit: int = 10000,
) -> dict:
    normalized = sorted({symbol.upper() for symbol in symbols if symbol.strip()})
    if not normalized:
        return {}
    all_bars: dict[str, list[dict]] = {symbol: [] for symbol in normalized}
    page_token: str | None = None
    while True:
        params_dict = {
            "symbols": ",".join(normalized),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": limit,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        }
        if page_token:
            params_dict["page_token"] = page_token
        url = f"https://data.alpaca.markets/v2/stocks/bars?{urlencode(params_dict)}"
        payload = _http_get_json(url, headers=_alpaca_headers())
        for symbol, bars in payload.get("bars", {}).items():
            all_bars.setdefault(symbol.upper(), []).extend(bars or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            return all_bars


def fetch_alpaca_history_batch(
    symbols: list[str],
    timeframe: str = "1Day",
    months: int = 6,
    feed: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    feed = feed or _alpaca_feed()
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=30 * months)
    payload = _fetch_alpaca_bars_payload(symbols, timeframe, start, end, feed)
    symbol_to_df: dict[str, pd.DataFrame] = {}
    for symbol in sorted({symbol.upper() for symbol in symbols}):
        bars = payload.get(symbol, [])
        if bars:
            symbol_to_df[symbol] = _bars_to_df(bars)
        else:
            print(f"Skipping {symbol}: No Alpaca bars returned.")
    return symbol_to_df


def fetch_alpaca_intraday_batch(
    symbols: list[str],
    timeframe: str = "5Min",
    days: int = 5,
    feed: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    feed = feed or _alpaca_feed()
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=max(days, 1) + 2)
    payload = _fetch_alpaca_bars_payload(symbols, timeframe, start, end, feed)
    symbol_to_df: dict[str, pd.DataFrame] = {}
    for symbol in sorted({symbol.upper() for symbol in symbols}):
        bars = payload.get(symbol, [])
        if bars:
            symbol_to_df[symbol] = _bars_to_df(bars, utc=True)
        else:
            print(f"Skipping {symbol}: No Alpaca intraday bars returned.")
    return symbol_to_df


def fetch_alpaca_symbol_history(
    symbol: str,
    timeframe: str = "1Day",
    months: int = 6,
    feed: Optional[str] = None,
) -> pd.DataFrame:
    feed = feed or _alpaca_feed()
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=30 * months)
    params = urlencode(
        {
            "symbols": symbol.upper(),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        }
    )
    url = f"https://data.alpaca.markets/v2/stocks/bars?{params}"
    payload = _http_get_json(url, headers=_alpaca_headers())
    bars = payload.get("bars", {}).get(symbol.upper(), [])
    if not bars:
        raise RuntimeError(f"No Alpaca bars returned for {symbol}.")

    return _bars_to_df(bars)


def fetch_alpaca_symbol_intraday(
    symbol: str,
    timeframe: str = "5Min",
    days: int = 5,
    feed: Optional[str] = None,
) -> pd.DataFrame:
    feed = feed or _alpaca_feed()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=max(days, 1) + 2)
    params = urlencode(
        {
            "symbols": symbol.upper(),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        }
    )
    url = f"https://data.alpaca.markets/v2/stocks/bars?{params}"
    payload = _http_get_json(url, headers=_alpaca_headers())
    bars = payload.get("bars", {}).get(symbol.upper(), [])
    if not bars:
        raise RuntimeError(f"No Alpaca intraday bars returned for {symbol}.")

    return _bars_to_df(bars, utc=True)


def fetch_yfinance_symbol_intraday(
    symbol: str,
    interval: str = "5m",
) -> pd.DataFrame:
    return fetch_yfinance_symbol_history(symbol, months=1, interval=interval).tail(5000).reset_index(drop=True)


def fetch_intraday_watchlist(
    symbols: list[str],
    provider: str = "alpaca",
    interval: str = "5m",
    days: int = 5,
    feed: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    feed = feed or _alpaca_feed()
    alpaca_timeframes = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1Hour",
    }
    symbol_to_df: dict[str, pd.DataFrame] = {}
    if provider == "alpaca":
        timeframe = alpaca_timeframes.get(interval.lower(), interval)
        try:
            batch = fetch_alpaca_intraday_batch(symbols, timeframe=timeframe, days=days, feed=feed)
        except Exception as exc:
            print(f"Alpaca intraday batch unavailable: {exc}")
            return {}
        for symbol, df in batch.items():
            if len(df) >= 20:
                symbol_to_df[symbol] = df
        missing_symbols = [symbol.upper() for symbol in symbols if symbol.upper() not in symbol_to_df]
        if not missing_symbols:
            return symbol_to_df
    else:
        missing_symbols = [symbol.upper() for symbol in symbols]

    for symbol in missing_symbols:
        symbol = symbol.upper()
        try:
            if provider == "alpaca":
                timeframe = alpaca_timeframes.get(interval.lower(), interval)
                df = fetch_alpaca_symbol_intraday(symbol, timeframe=timeframe, days=days, feed=feed)
            elif provider == "yfinance":
                df = fetch_yfinance_symbol_intraday(symbol, interval=interval)
            else:
                raise ValueError(f"Unsupported intraday provider: {provider}")
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")
            continue
        if len(df) >= 20:
            symbol_to_df[symbol] = df
    return symbol_to_df


def fetch_polygon_symbol_history(
    symbol: str,
    multiplier: int = 1,
    timespan: str = "day",
    months: int = 6,
) -> pd.DataFrame:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("Missing POLYGON_API_KEY in environment.")

    end = pd.Timestamp.now(tz="UTC").normalize().date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=30 * months)).date()
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/"
        f"{multiplier}/{timespan}/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    payload = _http_get_json(url)
    bars = payload.get("results", [])
    if not bars:
        raise RuntimeError(f"No Polygon bars returned for {symbol}.")

    df = pd.DataFrame(bars)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(df["t"], unit="ms", utc=True),
            "Open": pd.to_numeric(df["o"], errors="coerce"),
            "High": pd.to_numeric(df["h"], errors="coerce"),
            "Low": pd.to_numeric(df["l"], errors="coerce"),
            "Close": pd.to_numeric(df["c"], errors="coerce"),
            "Volume": pd.to_numeric(df["v"], errors="coerce"),
        }
    ).dropna().sort_values("Date").reset_index(drop=True)


def fetch_live_watchlist(
    symbols: list[str],
    provider: str = "alpaca",
    months: int = 6,
) -> dict[str, pd.DataFrame]:
    symbol_to_df: dict[str, pd.DataFrame] = {}
    if provider == "alpaca":
        try:
            batch = fetch_alpaca_history_batch(symbols, months=months)
        except Exception as exc:
            print(f"Alpaca history batch unavailable: {exc}")
            return {}
        for symbol, df in batch.items():
            if len(df) >= 80:
                symbol_to_df[symbol] = df
        missing_symbols = [symbol.upper() for symbol in symbols if symbol.upper() not in symbol_to_df]
        if not missing_symbols:
            return symbol_to_df
    else:
        missing_symbols = [symbol.upper() for symbol in symbols]

    for symbol in missing_symbols:
        symbol = symbol.upper()
        try:
            if provider == "alpaca":
                df = fetch_alpaca_symbol_history(symbol, months=months)
            elif provider == "polygon":
                df = fetch_polygon_symbol_history(symbol, months=months)
            elif provider == "finnhub":
                df = fetch_finnhub_symbol_history(symbol, months=months)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")
            continue
        if len(df) >= 80:
            symbol_to_df[symbol] = df
    return symbol_to_df


def fetch_yfinance_symbol_history(
    symbol: str,
    months: int = 6,
    interval: str = "1d",
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is not available in this environment.") from exc

    period = f"{months}mo"
    data = yf.download(
        symbol.upper(),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data is None or len(data) == 0:
        raise RuntimeError(f"No yfinance bars returned for {symbol}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    df = data.reset_index()
    rename_map = {}
    for col in df.columns:
        lower = str(col).lower()
        if lower in {"date", "datetime"}:
            rename_map[col] = "Date"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower == "close":
            rename_map[col] = "Close"
        elif lower == "volume":
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    return df.dropna().sort_values("Date").reset_index(drop=True)


def fetch_yfinance_watchlist(symbols: list[str], months: int = 6) -> dict[str, pd.DataFrame]:
    symbol_to_df: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = fetch_yfinance_symbol_history(symbol, months=months)
        except Exception:
            continue
        if len(df) >= 80:
            symbol_to_df[symbol.upper()] = df
    return symbol_to_df


def fetch_finnhub_symbol_history(
    symbol: str,
    resolution: str = "D",
    months: int = 6,
) -> pd.DataFrame:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FINNHUB_API_KEY in environment.")

    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    start = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30 * months)).timestamp())
    params = urlencode(
        {
            "symbol": symbol.upper(),
            "resolution": resolution,
            "from": start,
            "to": now,
            "token": api_key,
        }
    )
    url = f"https://finnhub.io/api/v1/stock/candle?{params}"
    payload = _http_get_json(url)
    if payload.get("s") != "ok":
        raise RuntimeError(f"No Finnhub bars returned for {symbol}.")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(payload["t"], unit="s", utc=True),
            "Open": pd.to_numeric(payload["o"], errors="coerce"),
            "High": pd.to_numeric(payload["h"], errors="coerce"),
            "Low": pd.to_numeric(payload["l"], errors="coerce"),
            "Close": pd.to_numeric(payload["c"], errors="coerce"),
            "Volume": pd.to_numeric(payload["v"], errors="coerce"),
        }
    )
    return df.dropna().sort_values("Date").reset_index(drop=True)


class MarketReadingEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig()

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()

        out["ret_1"] = out["Close"].pct_change().fillna(0.0)
        out["ret_5"] = out["Close"].pct_change(5).fillna(0.0)
        out["ret_10"] = out["Close"].pct_change(10).fillna(0.0)

        out["ma_fast"] = out["Close"].rolling(cfg.fast_ma).mean()
        out["ma_slow"] = out["Close"].rolling(cfg.slow_ma).mean()
        out["ma_gap"] = (out["ma_fast"] - out["ma_slow"]) / out["ma_slow"]

        out["highest_breakout"] = out["High"].rolling(cfg.breakout_window).max().shift(1)
        out["lowest_breakout"] = out["Low"].rolling(cfg.breakout_window).min().shift(1)
        out["breakout_up"] = (out["Close"] - out["highest_breakout"]) / out["Close"]
        out["breakout_down"] = (out["lowest_breakout"] - out["Close"]) / out["Close"]

        prev_close = out["Close"].shift(1)
        tr_parts = pd.concat(
            [
                out["High"] - out["Low"],
                (out["High"] - prev_close).abs(),
                (out["Low"] - prev_close).abs(),
            ],
            axis=1,
        )
        out["tr"] = tr_parts.max(axis=1)
        out["atr"] = out["tr"].rolling(cfg.atr_window).mean()
        out["atr_pct"] = out["atr"] / out["Close"]

        out["vol_ma"] = out["Volume"].rolling(20).mean()
        out["volume_impulse"] = (out["Volume"] / out["vol_ma"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        delta = out["Close"].diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        rs = up.rolling(14).mean() / down.rolling(14).mean().replace(0.0, np.nan)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        out["rsi"] = out["rsi"].fillna(50.0)

        dm = out["Close"].diff(cfg.chop_window).abs()
        path = out["Close"].diff().abs().rolling(cfg.chop_window).sum()
        out["chop_ratio"] = (1.0 - (dm / path.replace(0.0, np.nan))).clip(lower=0.0, upper=1.0).fillna(1.0)

        out["trend_slope"] = (
            out["Close"].rolling(cfg.trend_window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / max(float(np.mean(x)), 1e-12),
                raw=False,
            )
        )
        out["trend_slope"] = out["trend_slope"].fillna(0.0)

        out["trend_score"] = (
            0.45 * _normalize(out["ma_gap"]).clip(-2.0, 2.0)
            + 0.35 * _normalize(out["trend_slope"]).clip(-2.0, 2.0)
            + 0.20 * _normalize(out["ret_10"]).clip(-2.0, 2.0)
        )

        out["reversal_score"] = (
            -0.50 * _normalize(out["ret_5"]).clip(-2.0, 2.0)
            - 0.20 * _normalize(out["ret_1"]).clip(-2.0, 2.0)
            + 0.30 * _normalize((50.0 - out["rsi"]).abs()).clip(-2.0, 2.0)
        )

        out["breakout_score"] = (
            0.60 * _normalize(out["breakout_up"].fillna(0.0) - out["breakout_down"].fillna(0.0)).clip(-2.0, 2.0)
            + 0.40 * _normalize(out["volume_impulse"]).clip(-2.0, 2.0)
        )

        trend_regime = (out["trend_score"].abs() > 0.55).astype(float)
        meanrev_regime = (out["chop_ratio"] > 0.60).astype(float)
        out["regime_bias"] = trend_regime - meanrev_regime * 0.5

        out["long_pressure"] = (
            0.40 * out["trend_score"]
            + 0.35 * out["breakout_score"]
            + 0.15 * ((50.0 - out["rsi"]) / 50.0) * (-1.0)
            + 0.10 * out["regime_bias"]
        )
        out["short_pressure"] = (
            -0.40 * out["trend_score"]
            - 0.35 * out["breakout_score"]
            + 0.15 * ((out["rsi"] - 50.0) / 50.0)
            - 0.10 * out["regime_bias"]
        )

        out["signal_strength"] = out[["long_pressure", "short_pressure"]].abs().max(axis=1)
        out["signal_direction"] = np.where(out["long_pressure"] >= out["short_pressure"], 1, -1)

        directional_edge = pd.Series(
            np.where(out["signal_direction"] > 0, out["long_pressure"], out["short_pressure"]),
            index=out.index,
        )
        stability = (1.0 - out["chop_ratio"]).clip(0.0, 1.0)
        volatility_penalty = (out["atr_pct"] / out["atr_pct"].rolling(50).median().replace(0.0, np.nan)).fillna(1.0)
        volatility_penalty = (1.0 / volatility_penalty).clip(0.0, 1.0)
        volume_bonus = ((out["volume_impulse"] - 1.0) / 2.0 + 0.5).clip(0.0, 1.0)

        out["confidence"] = (
            0.45 * directional_edge.rank(pct=True).fillna(0.0)
            + 0.25 * stability
            + 0.15 * volatility_penalty
            + 0.15 * volume_bonus
        ).clip(0.0, 1.0)

        rel_atr = (out["atr_pct"] / out["atr_pct"].rolling(50).median().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        volatility_quality = (1.0 - ((rel_atr.fillna(1.0) - 1.0).abs() / 1.5)).clip(0.0, 1.0)
        volume_quality = (1.0 - ((out["volume_impulse"].fillna(1.0) - 1.0).abs() / 3.0)).clip(0.0, 1.0)
        direction_quality = (out["signal_strength"] / (out["signal_strength"].rolling(50).quantile(0.85).replace(0.0, np.nan))).fillna(0.0).clip(0.0, 1.0)
        out["readability"] = (
            100.0
            * (
                0.35 * stability
                + 0.25 * out["confidence"]
                + 0.20 * volatility_quality
                + 0.10 * volume_quality
                + 0.10 * direction_quality
            )
        ).clip(0.0, 100.0)
        out["readability_label"] = out["readability"].map(_readability_label)

        out["trade_ok"] = (
            (out["signal_strength"] >= cfg.min_signal_strength)
            & (out["confidence"] >= cfg.min_confidence)
            & (out["chop_ratio"] <= cfg.max_chop_for_trade)
            & out["atr_pct"].notna()
        )

        out["trade_label"] = np.where(
            out["trade_ok"] & (out["signal_direction"] > 0),
            "LONG",
            np.where(out["trade_ok"] & (out["signal_direction"] < 0), "SHORT", "NO_TRADE"),
        )

        return out

    def latest_read(self, df: pd.DataFrame) -> dict:
        enriched = self.enrich(df)
        row = enriched.iloc[-1]
        reasons = []
        if row["signal_strength"] < self.cfg.min_signal_strength:
            reasons.append("signal too weak")
        if row["confidence"] < self.cfg.min_confidence:
            reasons.append("confidence too low")
        if row["chop_ratio"] > self.cfg.max_chop_for_trade:
            reasons.append("market too choppy")
        if pd.isna(row["atr_pct"]):
            reasons.append("not enough history")
        if row["readability"] < 48.0:
            reasons.append("structure chaotic")

        hotness = (
            0.45 * float(row["confidence"])
            + 0.35 * float(row["signal_strength"])
            + 0.20 * abs(float(row["breakout_score"]))
        ) * (1.0 if row["trade_label"] != "SHORT" else 0.9)

        if row["trade_label"] == "NO_TRADE":
            setup_grade = "AVOID"
        elif row["confidence"] >= 0.75 and row["signal_strength"] >= 0.75:
            setup_grade = "A"
        elif row["confidence"] >= 0.65 and row["signal_strength"] >= 0.55:
            setup_grade = "B"
        else:
            setup_grade = "C"

        atr_value = float(row["atr"]) if pd.notna(row["atr"]) else None
        stop_distance = atr_value * self.cfg.stop_atr_mult if atr_value is not None else None
        risk_budget = self.cfg.initial_capital * self.cfg.account_risk_fraction
        suggested_shares = int(risk_budget / stop_distance) if stop_distance and stop_distance > 0 else 0

        return {
            "date": row["Date"],
            "label": row["trade_label"],
            "confidence": float(row["confidence"]),
            "readability": float(row["readability"]),
            "readability_label": row["readability_label"],
            "signal_strength": float(row["signal_strength"]),
            "trend_score": float(row["trend_score"]),
            "breakout_score": float(row["breakout_score"]),
            "chop_ratio": float(row["chop_ratio"]),
            "atr_pct": float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else None,
            "atr": atr_value,
            "long_pressure": float(row["long_pressure"]),
            "short_pressure": float(row["short_pressure"]),
            "hotness": hotness,
            "setup_grade": setup_grade,
            "stop_distance": stop_distance,
            "suggested_shares": suggested_shares,
            "reasons": reasons or ["conditions aligned"],
        }

    def hotness_table(self, symbol_to_df: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for symbol, df in symbol_to_df.items():
            read = self.latest_read(df)
            direction_sign = 0
            if read["label"] == "LONG":
                direction_sign = 1
            elif read["label"] == "SHORT":
                direction_sign = -1

            hotness = (
                0.45 * read["confidence"]
                + 0.35 * read["signal_strength"]
                + 0.20 * abs(read["breakout_score"])
            ) * (1.0 if direction_sign >= 0 else 0.9)

            rows.append(
                {
                    "symbol": symbol,
                    "date": read["date"],
                    "label": read["label"],
                    "confidence": read["confidence"],
                    "readability": read["readability"],
                    "readability_label": read["readability_label"],
                    "signal_strength": read["signal_strength"],
                    "trend_score": read["trend_score"],
                    "breakout_score": read["breakout_score"],
                    "chop_ratio": read["chop_ratio"],
                    "hotness": hotness,
                    "setup_grade": read["setup_grade"],
                    "atr": read["atr"],
                    "stop_distance": read["stop_distance"],
                    "suggested_shares": read["suggested_shares"],
                    "reason": ", ".join(read["reasons"]),
                }
            )

        table = pd.DataFrame(rows).sort_values(
            ["hotness", "confidence", "signal_strength"],
            ascending=[False, False, False],
        )
        if not table.empty:
            table["rank"] = np.arange(1, len(table) + 1)
            table = table[
                [
                    "rank",
                    "symbol",
                    "label",
                    "setup_grade",
                    "hotness",
                    "confidence",
                    "readability",
                    "readability_label",
                    "signal_strength",
                    "trend_score",
                    "breakout_score",
                    "chop_ratio",
                    "atr",
                    "stop_distance",
                    "suggested_shares",
                    "reason",
                ]
            ]
        return table

    def backtest(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        cfg = self.cfg
        data = self.enrich(df)

        cash = cfg.initial_capital
        units = 0.0
        position = 0
        entry_price = None
        entry_idx = None
        stop_price = None
        target_price = None

        equity_rows = []
        trades = []

        for i in range(cfg.lookback, len(data) - 1):
            row = data.iloc[i]
            current_close = float(row["Close"])
            next_open = float(data.iloc[i + 1]["Open"])

            if position != 0 and entry_price is not None and entry_idx is not None:
                hold_bars = i - entry_idx
                exit_now = False
                reason = None

                if position > 0:
                    if float(row["Low"]) <= float(stop_price):
                        exit_now = True
                        reason = "stop"
                    elif float(row["High"]) >= float(target_price):
                        exit_now = True
                        reason = "target"
                else:
                    if float(row["High"]) >= float(stop_price):
                        exit_now = True
                        reason = "stop"
                    elif float(row["Low"]) <= float(target_price):
                        exit_now = True
                        reason = "target"

                if not exit_now and hold_bars >= cfg.max_hold_bars:
                    exit_now = True
                    reason = "time"
                if not exit_now and row["trade_label"] != ("LONG" if position > 0 else "SHORT"):
                    exit_now = True
                    reason = "signal_flip"

                if exit_now:
                    exec_price = next_open * (1.0 - position * cfg.slippage_bps / 10000.0)
                    trade_value = abs(units) * exec_price
                    costs = trade_value * (cfg.transaction_cost_bps / 10000.0)
                    cash += units * exec_price - costs

                    pnl_pct = position * (exec_price - entry_price) / entry_price
                    trades.append(
                        {
                            "entry_date": data.iloc[entry_idx]["Date"],
                            "exit_date": data.iloc[i + 1]["Date"],
                            "side": "LONG" if position > 0 else "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "bars_held": hold_bars,
                            "pnl_pct": pnl_pct,
                            "reason": reason,
                        }
                    )
                    units = 0.0
                    position = 0
                    entry_price = None
                    entry_idx = None
                    stop_price = None
                    target_price = None

            if position == 0 and row["trade_label"] in {"LONG", "SHORT"}:
                side = 1 if row["trade_label"] == "LONG" else -1
                alloc_cash = cash * cfg.risk_fraction
                exec_price = next_open * (1.0 + side * cfg.slippage_bps / 10000.0)
                if alloc_cash > 0.0 and exec_price > 0.0:
                    units = side * (alloc_cash / exec_price)
                    trade_value = abs(units) * exec_price
                    costs = trade_value * (cfg.transaction_cost_bps / 10000.0)
                    cash -= units * exec_price + costs
                    position = side
                    entry_price = exec_price
                    entry_idx = i + 1
                    atr = max(float(row["atr"]), 1e-8)
                    if side > 0:
                        stop_price = entry_price - cfg.stop_atr_mult * atr
                        target_price = entry_price + cfg.target_atr_mult * atr
                    else:
                        stop_price = entry_price + cfg.stop_atr_mult * atr
                        target_price = entry_price - cfg.target_atr_mult * atr

            equity = cash + units * current_close
            equity_rows.append(
                {
                    "Date": row["Date"],
                    "Equity": equity,
                    "Close": current_close,
                    "Position": position,
                    "Signal": row["trade_label"],
                    "Confidence": float(row["confidence"]),
                    "Strength": float(row["signal_strength"]),
                }
            )

        equity_df = pd.DataFrame(equity_rows)
        if not equity_df.empty:
            equity_df["Return"] = equity_df["Equity"].pct_change().fillna(0.0)
            rolling_max = equity_df["Equity"].cummax()
            equity_df["Drawdown"] = equity_df["Equity"] / rolling_max - 1.0
        else:
            equity_df["Return"] = []
            equity_df["Drawdown"] = []

        closed = pd.DataFrame(trades)
        win_rate = float((closed["pnl_pct"] > 0).mean()) if not closed.empty else 0.0
        summary = {
            "initial_capital": cfg.initial_capital,
            "final_equity": float(equity_df["Equity"].iloc[-1]) if not equity_df.empty else cfg.initial_capital,
            "total_return_pct": 100.0 * ((float(equity_df["Equity"].iloc[-1]) / cfg.initial_capital) - 1.0) if not equity_df.empty else 0.0,
            "max_drawdown_pct": 100.0 * float(equity_df["Drawdown"].min()) if not equity_df.empty else 0.0,
            "num_closed_trades": int(len(closed)),
            "win_rate_pct": 100.0 * win_rate,
        }
        return equity_df, closed, summary


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.tz_convert("America/New_York")
    out["session_date"] = out["Date"].dt.date
    out["session_time"] = out["Date"].dt.time
    market_open = pd.Timestamp("09:30").time()
    market_close = pd.Timestamp("16:00").time()
    return out[(out["session_time"] >= market_open) & (out["session_time"] <= market_close)].copy()


def market_bias_from_intraday(symbol_to_df: dict[str, pd.DataFrame]) -> Optional[float]:
    scores = []
    for symbol in ("SPY", "QQQ"):
        df = symbol_to_df.get(symbol)
        if df is None or df.empty:
            continue
        regular = _regular_session(df)
        if regular.empty:
            continue
        latest_session = regular["session_date"].max()
        today = regular[regular["session_date"] == latest_session].copy()
        if len(today) < 4:
            continue
        typical = (today["High"] + today["Low"] + today["Close"]) / 3.0
        vwap = (typical * today["Volume"]).cumsum() / today["Volume"].cumsum().replace(0.0, np.nan)
        price = float(today["Close"].iloc[-1])
        trend = float(price / float(today["Close"].tail(4).iloc[0]) - 1.0)
        scores.append((1.0 if price > float(vwap.iloc[-1]) else -1.0) + (0.5 if trend > 0 else -0.5))
    if not scores:
        return None
    return float(np.mean(scores))


def daytrade_read(
    symbol: str,
    df: pd.DataFrame,
    market_bias: Optional[float] = None,
    opening_range_minutes: int = 15,
    min_rel_volume: float = 0.9,
) -> dict:
    regular = _regular_session(df)
    if regular.empty:
        raise RuntimeError(f"No regular-session intraday bars for {symbol}.")

    latest_session = regular["session_date"].max()
    today = regular[regular["session_date"] == latest_session].copy()
    history = regular[regular["session_date"] < latest_session].copy()
    if len(today) < 4:
        raise RuntimeError(f"Not enough current-session bars for {symbol}.")

    typical = (today["High"] + today["Low"] + today["Close"]) / 3.0
    today["vwap"] = (typical * today["Volume"]).cumsum() / today["Volume"].cumsum().replace(0.0, np.nan)
    latest = today.iloc[-1]
    price = float(latest["Close"])
    vwap = float(latest["vwap"])

    open_time = pd.Timestamp(str(latest_session) + " 09:30", tz="America/New_York")
    range_end = open_time + pd.Timedelta(minutes=opening_range_minutes)
    opening = today[(today["Date"] >= open_time) & (today["Date"] < range_end)]
    if opening.empty:
        opening = today.head(max(opening_range_minutes // 5, 1))
    or_high = float(opening["High"].max())
    or_low = float(opening["Low"].min())

    latest_minute = latest["Date"].hour * 60 + latest["Date"].minute
    today_volume = float(today["Volume"].sum())
    rel_volume = 1.0
    if not history.empty:
        history = history.copy()
        history["minute_of_day"] = history["Date"].dt.hour * 60 + history["Date"].dt.minute
        comparable = history[history["minute_of_day"] <= latest_minute]
        prior_by_session = comparable.groupby("session_date")["Volume"].sum()
        if not prior_by_session.empty and float(prior_by_session.mean()) > 0:
            rel_volume = today_volume / float(prior_by_session.mean())

    recent = today.tail(4)
    trend_pct = float(price / float(recent["Close"].iloc[0]) - 1.0) if len(recent) >= 2 else 0.0
    range_width = max(or_high - or_low, price * 0.002)
    above_vwap = price > vwap
    below_vwap = price < vwap
    breaks_high = price > or_high
    breaks_low = price < or_low
    near_high = price >= or_high - 0.25 * range_width
    near_low = price <= or_low + 0.25 * range_width
    aligned_long = market_bias is None or market_bias >= 0
    aligned_short = market_bias is None or market_bias <= 0

    setup = "NO_TRADE"
    reasons = []
    direction_score = 0.0
    if breaks_high and above_vwap and rel_volume >= min_rel_volume and trend_pct >= -0.001 and aligned_long:
        setup = "ALERT_LONG"
        reasons.append("opening range break above VWAP")
        direction_score = 1.0
    elif breaks_low and below_vwap and rel_volume >= min_rel_volume and trend_pct <= 0.001 and aligned_short:
        setup = "ALERT_SHORT"
        reasons.append("opening range break below VWAP")
        direction_score = -1.0
    elif near_high and above_vwap and aligned_long:
        setup = "LONG_WATCH"
        reasons.append("near range high above VWAP")
        direction_score = 0.5
    elif near_low and below_vwap and aligned_short:
        setup = "SHORT_WATCH"
        reasons.append("near range low below VWAP")
        direction_score = -0.5
    else:
        if rel_volume < min_rel_volume:
            reasons.append("relative volume is light")
        elif market_bias is not None and abs(market_bias) < 0.25:
            reasons.append("market ETFs are mixed")
        else:
            reasons.append("no clean opening range setup")

    if rel_volume >= 1.5:
        reasons.append("strong relative volume")
    if market_bias is not None and abs(market_bias) >= 1.0:
        reasons.append("market aligned")

    setup_score = (
        abs(direction_score) * 2.0
        + min(rel_volume, 3.0) * 0.35
        + min(abs(trend_pct) * 100.0, 1.0) * 0.25
        + (0.25 if "ALERT" in setup else 0.0)
    )
    stop_distance = max(abs(price - vwap), range_width * 0.35)
    trend_quality = max(0.0, 1.0 - min(abs(trend_pct) / 0.006, 1.0))
    volume_quality = max(0.0, 1.0 - min(abs(rel_volume - 1.0) / 2.5, 1.0))
    vwap_distance_quality = max(0.0, 1.0 - min(abs(price - vwap) / max(range_width * 1.5, 1e-12), 1.0))
    structure_quality = 1.0 if "ALERT" in setup else (0.65 if "WATCH" in setup else 0.35)
    readability = 100.0 * (
        0.35 * structure_quality
        + 0.25 * volume_quality
        + 0.20 * trend_quality
        + 0.20 * vwap_distance_quality
    )

    return {
        "symbol": symbol.upper(),
        "time": latest["Date"],
        "setup": setup,
        "score": setup_score,
        "readability": readability,
        "readability_label": _readability_label(readability),
        "price": price,
        "vwap": vwap,
        "or_high": or_high,
        "or_low": or_low,
        "rel_volume": rel_volume,
        "trend_pct": trend_pct,
        "stop_distance": stop_distance,
        "reason": ", ".join(reasons),
    }


def daytrade_table(
    symbol_to_df: dict[str, pd.DataFrame],
    opening_range_minutes: int = 15,
    min_rel_volume: float = 0.9,
) -> pd.DataFrame:
    bias = market_bias_from_intraday(symbol_to_df)
    rows = []
    for symbol, df in symbol_to_df.items():
        try:
            rows.append(
                daytrade_read(
                    symbol,
                    df,
                    market_bias=bias,
                    opening_range_minutes=opening_range_minutes,
                    min_rel_volume=min_rel_volume,
                )
            )
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    priority = {"ALERT_LONG": 0, "ALERT_SHORT": 0, "LONG_WATCH": 1, "SHORT_WATCH": 1, "NO_TRADE": 2}
    table["priority"] = table["setup"].map(priority).fillna(3)
    table = table.sort_values(["priority", "score", "rel_volume"], ascending=[True, False, False])
    table["rank"] = np.arange(1, len(table) + 1)
    return table[
        [
            "rank",
            "symbol",
            "time",
            "setup",
            "score",
            "readability",
            "readability_label",
            "price",
            "vwap",
            "or_high",
            "or_low",
            "rel_volume",
            "trend_pct",
            "stop_distance",
            "reason",
        ]
    ]


def market_is_open_now(now: Optional[pd.Timestamp] = None) -> bool:
    current = now or pd.Timestamp.now(tz="America/New_York")
    if current.tzinfo is None:
        current = current.tz_localize("America/New_York")
    else:
        current = current.tz_convert("America/New_York")
    if current.weekday() >= 5:
        return False
    market_open = current.normalize() + pd.Timedelta(hours=9, minutes=30)
    market_close = current.normalize() + pd.Timedelta(hours=16)
    return bool(market_open <= current <= market_close)


def append_paper_log(path: str | Path, table: pd.DataFrame) -> int:
    alerts = table[table["setup"].isin(["ALERT_LONG", "ALERT_SHORT"])].copy()
    if alerts.empty:
        return 0

    log_path = Path(path)
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    alerts.insert(0, "logged_at", pd.Timestamp.now(tz="America/New_York").isoformat())
    alerts["side"] = np.where(alerts["setup"] == "ALERT_LONG", "LONG", "SHORT")
    alerts["entry_price"] = alerts["price"]
    alerts["stop_price"] = np.where(
        alerts["side"] == "LONG",
        alerts["entry_price"] - alerts["stop_distance"],
        alerts["entry_price"] + alerts["stop_distance"],
    )
    alerts["target_1r"] = np.where(
        alerts["side"] == "LONG",
        alerts["entry_price"] + alerts["stop_distance"],
        alerts["entry_price"] - alerts["stop_distance"],
    )
    alerts["target_2r"] = np.where(
        alerts["side"] == "LONG",
        alerts["entry_price"] + 2.0 * alerts["stop_distance"],
        alerts["entry_price"] - 2.0 * alerts["stop_distance"],
    )
    alerts["alert_key"] = alerts["time"].astype(str) + "|" + alerts["symbol"].astype(str) + "|" + alerts["setup"].astype(str)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        if "alert_key" in existing.columns:
            existing_keys = set(existing["alert_key"].dropna().astype(str))
            alerts = alerts[~alerts["alert_key"].astype(str).isin(existing_keys)].copy()
            if alerts.empty:
                return 0
    cols = [
        "alert_key",
        "logged_at",
        "time",
        "symbol",
        "side",
        "setup",
        "entry_price",
        "stop_price",
        "target_1r",
        "target_2r",
        "score",
        "rel_volume",
        "trend_pct",
        "vwap",
        "or_high",
        "or_low",
        "reason",
    ]
    alerts[cols].to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return int(len(alerts))


def _r_multiple(side: str, entry: float, price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    if side == "LONG":
        return (price - entry) / risk
    return (entry - price) / risk


def evaluate_alert_outcome(row: pd.Series, df: pd.DataFrame) -> dict:
    side = str(row["side"]).upper()
    entry_time = pd.Timestamp(row["time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("America/New_York")
    else:
        entry_time = entry_time.tz_convert("America/New_York")

    entry = float(row["entry_price"])
    stop = float(row["stop_price"])
    target_1r = float(row["target_1r"])
    target_2r = float(row["target_2r"])
    risk = abs(entry - stop)
    if risk <= 0:
        return {
            "outcome": "invalid_risk",
            "exit_time": "",
            "exit_price": np.nan,
            "pnl_r": 0.0,
            "max_favorable_r": 0.0,
            "max_adverse_r": 0.0,
            "bars_observed": 0,
        }

    regular = _regular_session(df)
    session = regular[regular["session_date"] == entry_time.date()].copy()
    future = session[session["Date"] > entry_time].copy()
    if future.empty:
        outcome = "too_late_to_evaluate" if entry_time.time() >= pd.Timestamp("15:55").time() else "pending"
        return {
            "outcome": outcome,
            "exit_time": "",
            "exit_price": np.nan,
            "pnl_r": 0.0,
            "max_favorable_r": 0.0,
            "max_adverse_r": 0.0,
            "bars_observed": 0,
        }

    max_favorable_r = 0.0
    max_adverse_r = 0.0
    for _, bar in future.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])
        if side == "LONG":
            max_favorable_r = max(max_favorable_r, (high - entry) / risk)
            max_adverse_r = max(max_adverse_r, (entry - low) / risk)
            stop_hit = low <= stop
            target_1_hit = high >= target_1r
            target_2_hit = high >= target_2r
            if stop_hit and (target_1_hit or target_2_hit):
                return {
                    "outcome": "ambiguous_stop_first",
                    "exit_time": bar["Date"],
                    "exit_price": stop,
                    "pnl_r": -1.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
            if stop_hit:
                return {
                    "outcome": "hit_stop",
                    "exit_time": bar["Date"],
                    "exit_price": stop,
                    "pnl_r": -1.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
            if target_2_hit:
                return {
                    "outcome": "hit_2r",
                    "exit_time": bar["Date"],
                    "exit_price": target_2r,
                    "pnl_r": 2.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
            if target_1_hit:
                return {
                    "outcome": "hit_1r",
                    "exit_time": bar["Date"],
                    "exit_price": target_1r,
                    "pnl_r": 1.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
        else:
            max_favorable_r = max(max_favorable_r, (entry - low) / risk)
            max_adverse_r = max(max_adverse_r, (high - entry) / risk)
            stop_hit = high >= stop
            target_1_hit = low <= target_1r
            target_2_hit = low <= target_2r
            if stop_hit and (target_1_hit or target_2_hit):
                return {
                    "outcome": "ambiguous_stop_first",
                    "exit_time": bar["Date"],
                    "exit_price": stop,
                    "pnl_r": -1.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
            if stop_hit:
                return {
                    "outcome": "hit_stop",
                    "exit_time": bar["Date"],
                    "exit_price": stop,
                    "pnl_r": -1.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
            if target_2_hit:
                return {
                    "outcome": "hit_2r",
                    "exit_time": bar["Date"],
                    "exit_price": target_2r,
                    "pnl_r": 2.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }
            if target_1_hit:
                return {
                    "outcome": "hit_1r",
                    "exit_time": bar["Date"],
                    "exit_price": target_1r,
                    "pnl_r": 1.0,
                    "max_favorable_r": max_favorable_r,
                    "max_adverse_r": max_adverse_r,
                    "bars_observed": int(len(future[future["Date"] <= bar["Date"]])),
                }

    last = future.iloc[-1]
    exit_price = float(last["Close"])
    return {
        "outcome": "eod_exit" if last["Date"].time() >= pd.Timestamp("15:55").time() else "pending",
        "exit_time": last["Date"],
        "exit_price": exit_price,
        "pnl_r": _r_multiple(side, entry, exit_price, risk),
        "max_favorable_r": max_favorable_r,
        "max_adverse_r": max_adverse_r,
        "bars_observed": int(len(future)),
    }


def evaluate_paper_log(
    path: str | Path,
    provider: str = "alpaca",
    interval: str = "5m",
    days: int = 10,
    feed: str = "iex",
) -> tuple[pd.DataFrame, dict]:
    log_path = Path(path)
    if not log_path.exists():
        raise RuntimeError(f"Paper log does not exist: {log_path}")

    log = pd.read_csv(log_path)
    required = {"time", "symbol", "side", "entry_price", "stop_price", "target_1r", "target_2r"}
    missing = required - set(log.columns)
    if missing:
        raise RuntimeError(f"Paper log is missing required columns: {sorted(missing)}")

    symbols = sorted(log["symbol"].dropna().astype(str).str.upper().unique())
    symbol_to_df = fetch_intraday_watchlist(symbols, provider=provider, interval=interval, days=days, feed=feed)
    outcomes = []
    for _, row in log.iterrows():
        symbol = str(row["symbol"]).upper()
        df = symbol_to_df.get(symbol)
        if df is None or df.empty:
            outcomes.append(
                {
                    "outcome": "missing_bars",
                    "exit_time": "",
                    "exit_price": np.nan,
                    "pnl_r": 0.0,
                    "max_favorable_r": 0.0,
                    "max_adverse_r": 0.0,
                    "bars_observed": 0,
                }
            )
            continue
        outcomes.append(evaluate_alert_outcome(row, df))

    outcome_df = pd.DataFrame(outcomes)
    evaluated = pd.concat([log.drop(columns=[c for c in outcome_df.columns if c in log.columns], errors="ignore"), outcome_df], axis=1)
    resolved = evaluated[evaluated["outcome"].isin(["hit_stop", "hit_1r", "hit_2r", "ambiguous_stop_first", "eod_exit"])]
    wins = resolved[resolved["pnl_r"] > 0]
    losses = resolved[resolved["pnl_r"] < 0]
    summary = {
        "alerts": int(len(evaluated)),
        "resolved": int(len(resolved)),
        "pending": int((evaluated["outcome"] == "pending").sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": 100.0 * float(len(wins) / len(resolved)) if len(resolved) else 0.0,
        "total_r": float(resolved["pnl_r"].sum()) if len(resolved) else 0.0,
        "expectancy_r": float(resolved["pnl_r"].mean()) if len(resolved) else 0.0,
    }
    return evaluated, summary


def _format_latest(reading: dict) -> str:
    reasons = ", ".join(reading["reasons"])
    return (
        f"{reading['date'].date()} | {reading['label']} | "
        f"grade={reading['setup_grade']} | "
        f"hotness={reading['hotness']:.3f} | "
        f"confidence={reading['confidence']:.3f} | "
        f"strength={reading['signal_strength']:.3f} | "
        f"trend={reading['trend_score']:.3f} | "
        f"breakout={reading['breakout_score']:.3f} | "
        f"chop={reading['chop_ratio']:.3f} | "
        f"reason={reasons}"
    )


def _print_bucket(title: str, table: pd.DataFrame, limit: int = 3) -> None:
    print(f"\n{title}:")
    if table.empty:
        print("none")
        return
    cols = [
        "rank",
        "symbol",
        "label",
        "setup_grade",
        "hotness",
        "confidence",
        "signal_strength",
        "readability",
        "readability_label",
        "suggested_shares",
        "reason",
    ]
    print(table.head(limit)[cols].to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market reading engine with explicit trade / no-trade decisions.")
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV. If omitted, yfinance watchlist scanning is used.")
    parser.add_argument("--csv-dir", type=str, default=None, help="Directory of per-symbol OHLCV CSVs. File stem is used as symbol.")
    parser.add_argument("--symbols", type=str, default="NVDA,AAPL,MSFT,META,TSLA,AMD,AMZN,QQQ,SPY", help="Comma-separated tickers to scan with yfinance.")
    parser.add_argument("--months", type=int, default=6, help="How many months of bars to fetch from the provider.")
    parser.add_argument("--export-csv", type=str, default=None, help="Optional path to export the hotness leaderboard as CSV.")
    parser.add_argument("--bucketed", action="store_true", help="Print Best Longs / Best Shorts / Avoid buckets.")
    parser.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic basket instead of yfinance.")
    parser.add_argument("--provider", choices=["yfinance", "alpaca"], default="yfinance", help="Data provider for watchlist scans.")
    parser.add_argument("--feed", default=_alpaca_feed(), help="Alpaca market-data feed, such as sip or iex.")
    parser.add_argument("--env-file", default=".env", help="Optional .env file containing API keys.")
    parser.add_argument("--daytrade", action="store_true", help="Run an intraday VWAP/opening-range scanner instead of daily hotness.")
    parser.add_argument("--interval", default="5m", help="Intraday bar interval, such as 1m, 5m, or 15m.")
    parser.add_argument("--days", type=int, default=5, help="How many recent calendar days to fetch for intraday scans.")
    parser.add_argument("--opening-range-minutes", type=int, default=15, help="Opening range window for daytrade scans.")
    parser.add_argument("--only-alerts", action="store_true", help="Only print ALERT_LONG and ALERT_SHORT rows for daytrade scans.")
    parser.add_argument("--min-rel-volume", type=float, default=0.9, help="Minimum relative volume required for daytrade alerts.")
    parser.add_argument("--market-hours-only", action="store_true", help="Skip daytrade scans outside regular U.S. market hours.")
    parser.add_argument("--paper-log", type=str, default=None, help="Append daytrade alerts to a paper-trade CSV log.")
    parser.add_argument("--evaluate-paper-log", type=str, default=None, help="Evaluate a paper-trade CSV log against later intraday bars.")
    parser.add_argument("--evaluated-log", type=str, default=None, help="Optional output path for evaluated paper log. Defaults to overwriting --evaluate-paper-log.")
    parser.add_argument("--tail", type=int, default=10, help="How many recent readings to print.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    engine = MarketReadingEngine()
    if args.evaluate_paper_log:
        evaluated, summary = evaluate_paper_log(
            args.evaluate_paper_log,
            provider=args.provider,
            interval=args.interval,
            days=args.days,
            feed=args.feed,
        )
        output_path = args.evaluated_log or args.evaluate_paper_log
        evaluated.to_csv(output_path, index=False)
        print("Paper log evaluation:")
        print(summary)
        print(f"\nWrote evaluated log to: {output_path}")
        display_cols = [
            "time",
            "symbol",
            "side",
            "entry_price",
            "outcome",
            "exit_price",
            "pnl_r",
            "max_favorable_r",
            "max_adverse_r",
        ]
        cols = [c for c in display_cols if c in evaluated.columns]
        if cols:
            print("\nRecent evaluated alerts:")
            print(evaluated.tail(args.tail)[cols].to_string(index=False))
    elif args.daytrade:
        if args.market_hours_only and not market_is_open_now():
            print("Market is closed. Skipping daytrade scan because --market-hours-only is set.")
            raise SystemExit(0)
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        symbol_to_df = fetch_intraday_watchlist(
            symbols,
            provider=args.provider,
            interval=args.interval,
            days=args.days,
            feed=args.feed,
        )
        table = daytrade_table(
            symbol_to_df,
            opening_range_minutes=args.opening_range_minutes,
            min_rel_volume=args.min_rel_volume,
        )
        if args.paper_log and not table.empty:
            logged_count = append_paper_log(args.paper_log, table)
            if logged_count:
                print(f"Appended {logged_count} alert(s) to paper log: {args.paper_log}")
        if args.only_alerts and not table.empty:
            table = table[table["setup"].isin(["ALERT_LONG", "ALERT_SHORT"])].copy()
        print(f"{args.provider} intraday daytrade scanner ({args.interval} bars):")
        if table.empty:
            print("No usable intraday data returned.")
        else:
            print(table.to_string(index=False))
        if args.export_csv and not table.empty:
            table.to_csv(args.export_csv, index=False)
            print(f"\nExported daytrade scan to: {args.export_csv}")
    elif args.csv_dir:
        csv_dir = Path(args.csv_dir)
        symbol_to_df = load_symbol_csv_dir(csv_dir)
        table = engine.hotness_table(symbol_to_df)
        print("Hotness leaderboard:")
        print(table.to_string(index=False))
        if args.bucketed and not table.empty:
            _print_bucket("Best Longs", table[table["label"] == "LONG"])
            _print_bucket("Best Shorts", table[table["label"] == "SHORT"])
            _print_bucket("Avoid", table[table["label"] == "NO_TRADE"])
        if args.export_csv:
            table.to_csv(args.export_csv, index=False)
            print(f"\nExported leaderboard to: {args.export_csv}")
        if not table.empty:
            leader = table.iloc[0]
            print(
                f"\nLeader: {leader['symbol']} | {leader['label']} | "
                f"grade={leader['setup_grade']} | hotness={leader['hotness']:.3f} | "
                f"suggested_shares={leader['suggested_shares']}"
            )
    elif args.csv:
        df = load_ohlcv_csv(args.csv)
        enriched = engine.enrich(df)
        latest = engine.latest_read(df)
        equity, trades, summary = engine.backtest(df)

        print("Latest reading:")
        print(_format_latest(latest))
        print("\nRecent decisions:")
        for _, row in enriched.tail(args.tail).iterrows():
            print(
                f"{row['Date'].date()} | {row['trade_label']} | "
                f"conf={row['confidence']:.3f} | strength={row['signal_strength']:.3f}"
            )
        print("\nBacktest summary:")
        print(summary)
        if not trades.empty:
            print("\nLast 5 trades:")
            print(trades.tail(5).to_string(index=False))
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.synthetic:
            symbol_to_df = make_synthetic_market_universe(symbols)
        elif args.provider == "alpaca":
            symbol_to_df = fetch_live_watchlist(symbols, provider="alpaca", months=args.months)
        else:
            symbol_to_df = fetch_yfinance_watchlist(symbols, months=args.months)
        table = engine.hotness_table(symbol_to_df)
        print(f"{'Synthetic' if args.synthetic else args.provider} hotness leaderboard:")
        print(table.to_string(index=False))
        if args.bucketed and not table.empty:
            _print_bucket("Best Longs", table[table["label"] == "LONG"])
            _print_bucket("Best Shorts", table[table["label"] == "SHORT"])
            _print_bucket("Avoid", table[table["label"] == "NO_TRADE"])
        if args.export_csv:
            table.to_csv(args.export_csv, index=False)
            print(f"\nExported leaderboard to: {args.export_csv}")
        if not table.empty:
            leader_symbol = table.iloc[0]["symbol"]
            print(f"\nLeader: {leader_symbol}")
            leader_df = symbol_to_df[leader_symbol]
            latest = engine.latest_read(leader_df)
            print(_format_latest(latest))
