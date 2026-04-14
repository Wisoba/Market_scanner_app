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


def _http_get_json(url: str, headers: Optional[dict[str, str]] = None) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_alpaca_symbol_history(
    symbol: str,
    timeframe: str = "1Day",
    months: int = 6,
    feed: str = "iex",
) -> pd.DataFrame:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in environment.")

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
    payload = _http_get_json(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        },
    )
    bars = payload.get("bars", {}).get(symbol.upper(), [])
    if not bars:
        raise RuntimeError(f"No Alpaca bars returned for {symbol}.")

    df = pd.DataFrame(bars)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(df["t"]),
            "Open": pd.to_numeric(df["o"], errors="coerce"),
            "High": pd.to_numeric(df["h"], errors="coerce"),
            "Low": pd.to_numeric(df["l"], errors="coerce"),
            "Close": pd.to_numeric(df["c"], errors="coerce"),
            "Volume": pd.to_numeric(df["v"], errors="coerce"),
        }
    ).dropna().sort_values("Date").reset_index(drop=True)


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
    for symbol in symbols:
        symbol = symbol.upper()
        if provider == "alpaca":
            df = fetch_alpaca_symbol_history(symbol, months=months)
        elif provider == "polygon":
            df = fetch_polygon_symbol_history(symbol, months=months)
        elif provider == "finnhub":
            df = fetch_finnhub_symbol_history(symbol, months=months)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
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
        if lower == "date":
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
    parser.add_argument("--tail", type=int, default=10, help="How many recent readings to print.")
    args = parser.parse_args()

    engine = MarketReadingEngine()
    if args.csv_dir:
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
        else:
            symbol_to_df = fetch_yfinance_watchlist(symbols, months=args.months)
        table = engine.hotness_table(symbol_to_df)
        print(f"{'Synthetic' if args.synthetic else 'yfinance'} hotness leaderboard:")
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
