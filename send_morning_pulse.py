from __future__ import annotations

import sqlite3
from pathlib import Path
import os

import pandas as pd

from delivery import send_notification
from market_reading_engine_v2 import MarketReadingEngine, fetch_yfinance_watchlist
from market_scan_app import (
    TABLE_COLUMNS,
    _market_summary,
    _notification_copy,
    _notification_html,
    _parse_symbols,
    _resolve_db_path,
)


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".market_scan_env"


def load_env_file():
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'").strip('"')
        os.environ[key.strip()] = value


def iter_users():
    conn = sqlite3.connect(_resolve_db_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM users WHERE alerts_enabled = 1").fetchall()
        for row in rows:
            yield dict(row)
    finally:
        conn.close()


def main():
    load_env_file()
    engine = MarketReadingEngine()
    for user in iter_users():
        symbols = _parse_symbols(user["watchlist"])
        symbol_to_df = fetch_yfinance_watchlist(symbols, months=6)
        table = engine.hotness_table(symbol_to_df) if symbol_to_df else pd.DataFrame(columns=TABLE_COLUMNS)
        longs = table[table["label"] == "LONG"].head(5).to_dict(orient="records") if not table.empty else []
        avoids = table[table["label"] == "NO_TRADE"].head(5).to_dict(orient="records") if not table.empty else []
        leader = table.iloc[0].to_dict() if not table.empty else None
        summary = _market_summary(table)
        preview = _notification_copy(user, summary, leader, longs, avoids)
        if not preview:
            print(f"skip {user['identifier']}: no preview")
            continue
        result = send_notification(
            channel=preview["channel"],
            destination=preview["destination"],
            subject=preview["subject"],
            text=preview["message"],
            html=_notification_html(preview),
        )
        print(f"{user['identifier']}: {'sent' if result.ok else 'failed'} -> {result.detail}")


if __name__ == "__main__":
    main()
