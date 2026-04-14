from __future__ import annotations

from pathlib import Path
from typing import List
import json as pyjson
import sqlite3
import os
from urllib.parse import urlparse, unquote

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from delivery import send_notification
from market_reading_engine_v2 import MarketReadingEngine, fetch_yfinance_watchlist


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "market_scan_users.db"
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app = FastAPI(title="Market Hotness Scanner")


DEFAULT_SYMBOLS = ["NVDA", "AAPL", "MSFT", "META", "TSLA", "AMD", "AMZN", "QQQ", "SPY"]
TABLE_COLUMNS = [
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


def _resolve_db_path() -> str:
    raw = os.environ.get("MARKET_SCAN_DB_PATH") or os.environ.get("DATABASE_URL")
    if not raw:
        return str(DB_PATH)
    if raw.startswith("sqlite:///"):
        parsed = urlparse(raw)
        db_path = unquote(parsed.path)
        return db_path or str(DB_PATH)
    return raw


def _db():
    conn = sqlite3.connect(_resolve_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT NOT NULL UNIQUE,
                email TEXT,
                phone TEXT,
                channel TEXT NOT NULL,
                watchlist TEXT NOT NULL,
                alerts_enabled INTEGER NOT NULL DEFAULT 1,
                notify_at TEXT NOT NULL DEFAULT '05:00',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


_init_db()


def _parse_symbols(symbols_raw: str | None) -> List[str]:
    if not symbols_raw:
        return DEFAULT_SYMBOLS
    parsed = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    return parsed or DEFAULT_SYMBOLS


def _normalize_identifier(identifier: str) -> str:
    return identifier.strip()


def _infer_channel(identifier: str) -> str:
    return "email" if "@" in identifier else "sms"


def _split_contact(identifier: str):
    if "@" in identifier:
        return identifier, None
    digits = "".join(ch for ch in identifier if ch.isdigit() or ch == "+")
    return None, digits or identifier


def _save_user(identifier: str, watchlist: str, alerts_enabled: bool, notify_at: str):
    identifier = _normalize_identifier(identifier)
    email, phone = _split_contact(identifier)
    channel = _infer_channel(identifier)
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO users (identifier, email, phone, channel, watchlist, alerts_enabled, notify_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(identifier) DO UPDATE SET
                email=excluded.email,
                phone=excluded.phone,
                channel=excluded.channel,
                watchlist=excluded.watchlist,
                alerts_enabled=excluded.alerts_enabled,
                notify_at=excluded.notify_at,
                updated_at=CURRENT_TIMESTAMP
            """,
            (identifier, email, phone, channel, watchlist, int(alerts_enabled), notify_at),
        )


def _get_user(identifier: str | None):
    if not identifier:
        return None
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE identifier = ?", (identifier,)).fetchone()
    return dict(row) if row else None


def _current_user(request: Request):
    return _get_user(request.cookies.get("market_scanner_user"))


def _market_summary(table):
    if table.empty:
        return {
            "market_stance": "No data",
            "top_pick": None,
            "runner_up": None,
            "best_short": None,
            "danger_zone": [],
        }

    longs = table[table["label"] == "LONG"]
    shorts = table[table["label"] == "SHORT"]
    avoids = table[table["label"] == "NO_TRADE"]

    top_pick = longs.iloc[0]["symbol"] if not longs.empty else None
    runner_up = longs.iloc[1]["symbol"] if len(longs) > 1 else None
    best_short = shorts.iloc[0]["symbol"] if not shorts.empty else None
    danger_zone = avoids.head(3)["symbol"].tolist()

    if not longs.empty and shorts.empty:
        stance = "Selective risk-on"
    elif longs.empty and not shorts.empty:
        stance = "Selective risk-off"
    elif longs.empty and shorts.empty:
        stance = "Mostly choppy"
    else:
        stance = "Mixed"

    return {
        "market_stance": stance,
        "top_pick": top_pick,
        "runner_up": runner_up,
        "best_short": best_short,
        "danger_zone": danger_zone,
    }


def _notification_copy(user, summary, leader, longs, avoids):
    if not user:
        return None
    top_longs = [row["symbol"] for row in longs[:2]]
    avoid_names = [row["symbol"] for row in avoids[:2]]
    pieces = [f"5AM Pulse: {summary['market_stance']}."]
    if leader:
        pieces.append(f"Leader {leader['symbol']} ({leader['label']} {leader['setup_grade']}).")
    if top_longs:
        pieces.append(f"Focus on {', '.join(top_longs)}.")
    if avoid_names:
        pieces.append(f"Avoid chasing {', '.join(avoid_names)}.")
    return {
        "channel": user["channel"],
        "destination": user["email"] or user["phone"] or user["identifier"],
        "notify_at": user["notify_at"],
        "enabled": bool(user["alerts_enabled"]),
        "subject": f"5AM Market Pulse: {summary['market_stance']}",
        "message": " ".join(pieces),
    }


def _notification_html(preview):
    return (
        "<div style='font-family:Georgia,serif;padding:20px;'>"
        "<div style='font-size:12px;letter-spacing:0.12em;text-transform:uppercase;color:#6d655d;margin-bottom:10px;'>Market Pulse</div>"
        f"<h1 style='margin:0 0 12px;font-size:28px;'>"
        f"{preview['subject']}"
        "</h1>"
        f"<p style='font-size:16px;line-height:1.6;color:#16120e;'>{preview['message']}</p>"
        "</div>"
    )


def _fetch_intraday_snapshot(symbol: str):
    try:
        import yfinance as yf
    except Exception:
        return None

    data = yf.download(
        symbol.upper(),
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
    )
    if data is None or len(data) < 3:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    intraday = data.reset_index()
    keep = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in intraday.columns]
    intraday = intraday[keep].dropna().reset_index(drop=True)
    if len(intraday) < 3:
        return None

    first_open = float(intraday["Open"].iloc[0])
    latest_close = float(intraday["Close"].iloc[-1])
    day_high = float(intraday["High"].max())
    day_low = float(intraday["Low"].min())
    latest_volume = float(intraday["Volume"].iloc[-1]) if "Volume" in intraday else 0.0
    range_size = max(day_high - day_low, 1e-6)

    close_series = intraday["Close"].astype(float)
    velocity_5m = latest_close - float(close_series.iloc[-2])
    velocity_15m = latest_close - float(close_series.iloc[max(0, len(close_series) - 4)])
    momentum_30m = latest_close - float(close_series.iloc[max(0, len(close_series) - 7)])
    returns = close_series.pct_change().fillna(0.0)
    accel = returns.iloc[-1] - returns.iloc[-2] if len(returns) >= 2 else 0.0

    if "Volume" in intraday and intraday["Volume"].sum() > 0:
        typical = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3.0
        cum_vol = intraday["Volume"].cumsum()
        vwap = float((typical * intraday["Volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])
        vwap_drift_pct = ((latest_close / vwap) - 1.0) * 100.0 if vwap else 0.0
    else:
        vwap_drift_pct = None

    return {
        "live_price": round(latest_close, 2),
        "day_move_pct": round(((latest_close / first_open) - 1.0) * 100.0, 2) if first_open else 0.0,
        "from_open": round(latest_close - first_open, 2),
        "range_position_pct": round(((latest_close - day_low) / range_size) * 100.0, 1),
        "velocity_5m": round(velocity_5m, 3),
        "velocity_15m": round(velocity_15m, 3),
        "momentum_30m": round(momentum_30m, 3),
        "accel_pct": round(accel * 100.0, 3),
        "day_high": round(day_high, 2),
        "day_low": round(day_low, 2),
        "latest_volume": int(latest_volume),
        "vwap_drift_pct": round(vwap_drift_pct, 2) if vwap_drift_pct is not None else None,
        "intraday": close_series.tail(36).tolist(),
    }


def _ticker_payload(symbol: str, df, engine: MarketReadingEngine):
    enriched = engine.enrich(df)
    read = engine.latest_read(df)
    row = enriched.iloc[-1]
    closes = df["Close"].tail(30).astype(float).tolist()
    intraday = _fetch_intraday_snapshot(symbol)
    change_1d = float(row["ret_1"]) if "ret_1" in row else 0.0
    change_5d = float(row["ret_5"]) if "ret_5" in row else 0.0
    latest_close = float(df["Close"].iloc[-1])
    return {
        "symbol": symbol,
        "label": read["label"],
        "setup_grade": read["setup_grade"],
        "hotness": round(read["hotness"], 3),
        "confidence": round(read["confidence"], 3),
        "signal_strength": round(read["signal_strength"], 3),
        "trend_score": round(read["trend_score"], 3),
        "breakout_score": round(read["breakout_score"], 3),
        "chop_ratio": round(read["chop_ratio"], 3),
        "atr": round(read["atr"], 3) if read["atr"] is not None else None,
        "stop_distance": round(read["stop_distance"], 3) if read["stop_distance"] is not None else None,
        "suggested_shares": read["suggested_shares"],
        "reasons": read["reasons"],
        "close": round(latest_close, 2),
        "change_1d_pct": round(change_1d * 100.0, 2),
        "change_5d_pct": round(change_5d * 100.0, 2),
        "sparkline": closes,
        "live": intraday,
        "date": str(read["date"].date()),
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = _current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "default_symbols": ",".join(DEFAULT_SYMBOLS),
        },
    )


@app.get("/enter")
async def login_submit(
    request: Request,
    identifier: str,
    watchlist: str = Query(",".join(DEFAULT_SYMBOLS)),
    notify_at: str = Query("05:00"),
    alerts_enabled: str | None = Query(None),
):
    normalized = _normalize_identifier(identifier)
    _save_user(
        identifier=normalized,
        watchlist=",".join(_parse_symbols(watchlist)),
        alerts_enabled=alerts_enabled is not None,
        notify_at=notify_at or "05:00",
    )
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie("market_scanner_user", normalized, httponly=True, samesite="lax")
    return response


@app.get("/preferences")
async def update_preferences(
    request: Request,
    watchlist: str = Query(",".join(DEFAULT_SYMBOLS)),
    notify_at: str = Query("05:00"),
    alerts_enabled: str | None = Query(None),
):
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    _save_user(
        identifier=user["identifier"],
        watchlist=",".join(_parse_symbols(watchlist)),
        alerts_enabled=alerts_enabled is not None,
        notify_at=notify_at or user["notify_at"],
    )
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("market_scanner_user")
    return response


@app.get("/send-test-alert")
async def send_test_alert(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    symbol_list = _parse_symbols(user["watchlist"])
    engine = MarketReadingEngine()
    symbol_to_df = fetch_yfinance_watchlist(symbol_list, months=6)
    table = engine.hotness_table(symbol_to_df) if symbol_to_df else pd.DataFrame(columns=TABLE_COLUMNS)
    longs = table[table["label"] == "LONG"].head(5).to_dict(orient="records") if not table.empty else []
    avoids = table[table["label"] == "NO_TRADE"].head(5).to_dict(orient="records") if not table.empty else []
    leader = table.iloc[0].to_dict() if not table.empty else None
    summary = _market_summary(table)
    preview = _notification_copy(user, summary, leader, longs, avoids)

    if not preview:
        return RedirectResponse(url="/?delivery=No+preview+available", status_code=303)

    result = send_notification(
        channel=preview["channel"],
        destination=preview["destination"],
        subject=preview["subject"],
        text=preview["message"],
        html=_notification_html(preview),
    )
    status = "sent" if result.ok else "failed"
    detail = f"{status}:{result.detail}"
    return RedirectResponse(url=f"/?delivery={detail}", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    symbols: str | None = Query(None),
    months: int = Query(6, ge=1, le=24),
    delivery: str | None = Query(None),
):
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    active_symbols = symbols or user["watchlist"]
    symbol_list = _parse_symbols(active_symbols)
    engine = MarketReadingEngine()

    symbol_to_df = fetch_yfinance_watchlist(symbol_list, months=months)
    table = engine.hotness_table(symbol_to_df) if symbol_to_df else pd.DataFrame(columns=TABLE_COLUMNS)
    ticker_details = {
        symbol: _ticker_payload(symbol, df, engine)
        for symbol, df in symbol_to_df.items()
    }

    longs = table[table["label"] == "LONG"].head(5).to_dict(orient="records") if not table.empty else []
    shorts = table[table["label"] == "SHORT"].head(5).to_dict(orient="records") if not table.empty else []
    avoids = table[table["label"] == "NO_TRADE"].head(5).to_dict(orient="records") if not table.empty else []
    leader = table.iloc[0].to_dict() if not table.empty else None
    summary = _market_summary(table)
    detail_cards = [ticker_details[row["symbol"]] for row in table.to_dict(orient="records")] if not table.empty else []
    notification_preview = _notification_copy(user, summary, leader, longs, avoids)

    return templates.TemplateResponse(
        request,
        "market_scan_app.html",
        {
            "symbols": ",".join(symbol_list),
            "months": months,
            "leader": leader,
            "summary": summary,
            "longs": longs,
            "shorts": shorts,
            "avoids": avoids,
            "table": table.to_dict(orient="records") if not table.empty else [],
            "detail_cards": detail_cards,
            "detail_cards_json": pyjson.dumps(detail_cards),
            "user": user,
            "notification_preview": notification_preview,
            "delivery_status": delivery,
        },
    )
