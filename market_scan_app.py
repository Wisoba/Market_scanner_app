from __future__ import annotations

import asyncio
import html
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from enum import Enum
from typing import List
import json as pyjson
import sqlite3
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, urlparse, unquote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from delivery import send_notification
from market_reading_engine_v2 import (
    MarketReadingEngine,
    daytrade_table,
    fetch_intraday_watchlist,
    fetch_live_watchlist,
    fetch_yfinance_watchlist,
    load_env_file,
)


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "market_scan_users.db"
SITE_DIR = APP_DIR / "site"
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app = FastAPI(title="Market Hotness Scanner")

load_env_file(APP_DIR / ".env")


@app.get("/healthz")
def healthz() -> dict:
    """Lightweight liveness probe for Railway — no DB or template dependency."""
    return {"status": "ok"}


def _site_page(filename: str) -> FileResponse:
    path = SITE_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Page not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/legal/privacy", include_in_schema=False)
@app.get("/privacy.html", include_in_schema=False)
def privacy_policy() -> FileResponse:
    return _site_page("privacy.html")


@app.get("/legal/terms", include_in_schema=False)
@app.get("/terms.html", include_in_schema=False)
def terms_of_service() -> FileResponse:
    return _site_page("terms.html")


@app.get("/support", include_in_schema=False)
@app.get("/support.html", include_in_schema=False)
def support_page() -> FileResponse:
    return _site_page("support.html")


def _require_admin_token(provided: str | None) -> None:
    """Guard privileged endpoints.

    If MARKET_SCAN_ADMIN_TOKEN is unset (typical for local dev) the endpoint stays open.
    When the env var is set (production), a matching X-Admin-Token header is required.
    """
    expected = (os.environ.get("MARKET_SCAN_ADMIN_TOKEN") or "").strip()
    if not expected:
        return
    if not provided or provided.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "TQQQ", "SQQQ", "SOXL", "SMH",
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "AVGO",
    "PLTR", "SMCI", "COIN", "MSTR", "MARA", "RIOT", "HOOD", "SOFI",
    "RIVN", "JPM", "BAC", "XLF", "XLE", "GLD",
]
ALPACA_NEWS_STREAM_URL = os.environ.get(
    "ALPACA_NEWS_STREAM_URL",
    "wss://stream.data.alpaca.markets/v1beta1/news",
)
ALPACA_NEWS_SANDBOX_STREAM_URL = os.environ.get(
    "ALPACA_NEWS_SANDBOX_STREAM_URL",
    "wss://stream.data.sandbox.alpaca.markets/v1beta1/news",
)
ALPACA_FEED = os.environ.get("ALPACA_FEED", "sip").strip().lower() or "sip"
TABLE_COLUMNS = [
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
INTRADAY_TABLE_COLUMNS = [
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


class AttentionState(str, Enum):
    HEATING = "HEATING"
    ACTIVE_HEAT = "ACTIVE_HEAT"
    COOLING = "COOLING"
    CHAOTIC = "CHAOTIC"


class ReadabilityScore(str, Enum):
    CLEAN = "CLEAN"
    MIXED = "MIXED"
    CHAOTIC = "CHAOTIC"


class HypotheticalPlan(BaseModel):
    direction: str
    entry_zone_start: float
    entry_zone_end: float
    stop_loss: float
    profit_target: float
    risk_reward: float = Field(default=1.5)
    label: str = "Hypothetical scenario"


class SignalCardModel(BaseModel):
    ticker: str
    source: str
    state: AttentionState
    pulse_status: str
    lane: str | None = None
    best_bid_rank: int | None = None
    raw_heat_rank: int | None = None
    attention_score: int = Field(default=0, ge=0, le=100)
    heat_score: int = Field(ge=0, le=100)
    readability_score: int = Field(ge=0, le=100)
    readability: ReadabilityScore
    rank: int | None = None
    rank_reason: list[str] = Field(default_factory=list)
    heat_reason: list[str] = Field(default_factory=list)
    risk_note: str | None = None
    risk_reasons: list[str] = Field(default_factory=list)
    pillar_alignment: str | None = None
    alignment_score: int | None = Field(default=None, ge=0, le=100)
    alignment_reasoning: list[str] = Field(default_factory=list)
    market_read: str | None = None
    conviction_level: str | None = None
    price: float | None = None
    vwap: float | None = None
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    relative_volume: float | None = None
    stop_distance: float | None = None
    why_metrics: list[str]
    plan: HypotheticalPlan | None = None
    expires_at: datetime | None = None
    seconds_remaining: int | None = None
    disclaimer: str = "For market-attention research only. Not financial advice."


def _direction_label(direction: str | None) -> str:
    normalized = (direction or "").upper()
    if normalized == "SHORT":
        return "falling"
    if normalized == "LONG":
        return "rising"
    return "mixed"


def _signal_direction(signal: SignalCardModel) -> str:
    if signal.plan:
        return _direction_label(signal.plan.direction)
    status = signal.pulse_status.upper()
    if "SHORT" in status:
        return "falling"
    if "LONG" in status:
        return "rising"
    if "CHAOTIC" in status or signal.readability == ReadabilityScore.CHAOTIC:
        return "chaotic"
    return "mixed"


class LiveNowResponse(BaseModel):
    market_stance: str
    timestamp: datetime
    refresh_after_seconds: int = 5
    symbols: list[str]
    signals: list[SignalCardModel]


_LIVE_NOW_CACHE: dict[str, tuple[datetime, LiveNowResponse]] = {}


class IntelItemModel(BaseModel):
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime | None = None
    symbols: list[str] = []
    category: str = "market"
    impact: str = "Watch"
    confidence: str = "Medium"
    media_type: str = "article"
    catalyst_theme: str = "Market Structure"
    market_relevance: str = "Medium"
    affected_symbols: list[str] = Field(default_factory=list)
    impact_summary: str = "May influence current market behavior."
    impact_direction: str = "Watch"


class IntelFeedResponse(BaseModel):
    timestamp: datetime
    refresh_after_seconds: int = 300
    symbols: list[str]
    items: list[IntelItemModel]


_INTEL_CACHE: dict[str, tuple[datetime, IntelFeedResponse]] = {}


def _snapshot_cache_seconds() -> int:
    raw = os.environ.get("MARKET_SNAPSHOT_CACHE_SECONDS", "180")
    try:
        return max(0, int(raw))
    except ValueError:
        return 180


class AlertRegistrationRequest(BaseModel):
    identifier: str
    channel: str | None = None
    device_token: str | None = None
    timezone: str
    watchlist: str | None = None
    notify_at: str = "05:00"


class AlertRegistrationResponse(BaseModel):
    ok: bool
    channel: str
    notify_at: str
    timezone: str
    message: str | None = None


class AlertEvaluationResponse(BaseModel):
    ok: bool
    users_checked: int
    alerts_sent: int
    alerts_skipped: int
    details: list[dict]


class AlertEventResponse(BaseModel):
    id: int
    alert_type: str
    alert_key: str
    symbols: str
    channel: str
    destination: str
    subject: str
    message: str
    provider: str | None = None
    delivery_ok: bool
    delivery_detail: str | None = None
    created_at: str


def _resolve_db_path() -> str:
    raw = os.environ.get("MARKET_SCAN_DB_PATH") or os.environ.get("DATABASE_URL")
    if not raw:
        return str(DB_PATH)

    if raw.startswith("sqlite:///"):
        parsed = urlparse(raw)
        raw = unquote(parsed.path)
    elif "://" in raw:
        # This service currently uses SQLite. Railway may inject a PostgreSQL
        # DATABASE_URL for another service; do not treat that URL as a file path.
        return str(DB_PATH)

    db_path = Path(raw).expanduser()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return str(DB_PATH)
    return str(db_path)


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
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "timezone" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN timezone TEXT NOT NULL DEFAULT 'America/New_York'")
        if "device_token" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN device_token TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identifier TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                alert_key TEXT NOT NULL,
                symbols TEXT NOT NULL,
                channel TEXT NOT NULL,
                destination TEXT NOT NULL,
                subject TEXT NOT NULL,
                message TEXT NOT NULL,
                provider TEXT,
                delivery_ok INTEGER NOT NULL DEFAULT 0,
                delivery_detail TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_alert_events_user_type_key_time
            ON alert_events (user_identifier, alert_type, alert_key, created_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_state (
                user_identifier TEXT NOT NULL,
                state_key TEXT NOT NULL,
                state_value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_identifier, state_key)
            )
            """
        )


_init_db()


def _parse_symbols(symbols_raw: str | None) -> List[str]:
    if not symbols_raw:
        return DEFAULT_SYMBOLS
    parsed = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    return parsed or DEFAULT_SYMBOLS


def _intel_cache_seconds() -> int:
    raw = os.environ.get("INTEL_CACHE_SECONDS", "300")
    try:
        return max(0, int(raw))
    except ValueError:
        return 300


def _http_text(url: str, headers: dict[str, str] | None = None, timeout: float = 4.0) -> str | None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "GACE-Scan/1.0 market-intel",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _clean_summary(raw: str | None, limit: int = 180) -> str:
    text = html.unescape(raw or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _story_symbols(title: str, summary: str, symbols: list[str]) -> list[str]:
    haystack = f" {title} {summary} ".upper()
    matched = [symbol for symbol in symbols if f" {symbol} " in haystack or f"${symbol}" in haystack]
    return matched[:6]


def _intel_impact(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(word in text for word in ("fed", "inflation", "cpi", "jobs", "treasury", "rates", "yield")):
        return "Macro"
    if any(word in text for word in ("war", "tariff", "sanction", "election", "oil", "opec")):
        return "Political"
    if any(word in text for word in ("earnings", "guidance", "revenue", "profit", "forecast")):
        return "Earnings"
    if any(word in text for word in ("bitcoin", "crypto", "ethereum")):
        return "Crypto"
    return "Market"


def _catalyst_theme(title: str, summary: str, impact: str) -> str:
    text = f"{title} {summary}".lower()
    if any(word in text for word in ("ai", "nvidia", "data center", "semiconductor", "chips")):
        return "AI Infrastructure"
    if any(word in text for word in ("bank", "underwriting", "loan", "credit", "regional bank")):
        return "Banking"
    if any(word in text for word in ("bitcoin", "crypto", "ethereum")):
        return "Crypto Liquidity"
    if any(word in text for word in ("oil", "opec", "energy", "crude")):
        return "Energy"
    if impact == "Macro":
        return "Macro Conditions"
    if impact == "Earnings":
        return "Earnings"
    if impact == "Political":
        return "Policy Risk"
    return "Market Structure"


def _market_relevance(title: str, summary: str, matched_symbols: list[str], impact: str) -> str:
    text = f"{title} {summary}".lower()
    if impact in {"Macro", "Political"}:
        return "High"
    if len(matched_symbols) >= 2:
        return "High"
    if any(word in text for word in ("nasdaq", "s&p", "dow", "fed", "inflation", "rates", "oil")):
        return "High"
    if matched_symbols:
        return "Medium"
    return "Low"


def _impact_direction(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(word in text for word in ("falls", "fall", "slides", "slump", "sell-off", "decline", "miss", "cuts")):
        return "Bearish"
    if any(word in text for word in ("rises", "rise", "surge", "beats", "upgrade", "approved", "expands", "partnership")):
        return "Bullish"
    return "Watch"


def _impact_summary(theme: str, relevance: str, direction: str, matched_symbols: list[str]) -> str:
    subject = ", ".join(matched_symbols[:3]) if matched_symbols else "market behavior"
    if direction == "Bullish":
        return f"Supports the {theme.lower()} narrative for {subject}."
    if direction == "Bearish":
        return f"Pressures the {theme.lower()} narrative for {subject}."
    if relevance == "High":
        return f"High-relevance catalyst that may affect {subject}."
    return f"May influence {subject}."


def _dedupe_intel(items: list[IntelItemModel], limit: int) -> list[IntelItemModel]:
    seen: set[str] = set()
    unique: list[IntelItemModel] = []
    for item in items:
        key = item.url or item.title.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda item: item.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return unique[:limit]


def _fetch_rss_items(feed_url: str, source: str, symbols: list[str], media_type: str = "article") -> list[IntelItemModel]:
    body = _http_text(feed_url)
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []

    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "media": "http://search.yahoo.com/mrss/",
    }
    entries = root.findall(".//item") or root.findall(".//atom:entry", namespaces)
    items: list[IntelItemModel] = []
    for entry in entries[:12]:
        title = entry.findtext("title") or entry.findtext("atom:title", namespaces=namespaces) or ""
        link = entry.findtext("link") or ""
        if not link:
            link_node = entry.find("atom:link", namespaces=namespaces)
            link = link_node.attrib.get("href", "") if link_node is not None else ""
        summary = (
            entry.findtext("description")
            or entry.findtext("summary")
            or entry.findtext("atom:summary", namespaces=namespaces)
            or ""
        )
        published = (
            entry.findtext("pubDate")
            or entry.findtext("published")
            or entry.findtext("atom:published", namespaces=namespaces)
            or entry.findtext("atom:updated", namespaces=namespaces)
        )
        clean_title = _clean_summary(title, limit=120)
        clean = _clean_summary(summary)
        matched = _story_symbols(clean_title, clean, symbols)
        impact = _intel_impact(clean_title, clean)
        theme = _catalyst_theme(clean_title, clean, impact)
        relevance = _market_relevance(clean_title, clean, matched, impact)
        direction = _impact_direction(clean_title, clean)
        if clean_title and link:
            items.append(
                IntelItemModel(
                    id=f"{source}:{link}",
                    title=clean_title,
                    summary=clean or impact,
                    source=source,
                    url=link,
                    published_at=_parse_datetime(published),
                    symbols=matched,
                    category=impact.lower(),
                    impact=impact,
                    confidence="High" if matched else "Medium",
                    media_type=media_type,
                    catalyst_theme=theme,
                    market_relevance=relevance,
                    affected_symbols=matched[:6],
                    impact_summary=_impact_summary(theme, relevance, direction, matched),
                    impact_direction=direction,
                )
            )
    return items


def _fetch_alpaca_news(symbols: list[str]) -> list[IntelItemModel]:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        return []

    query = urlencode({"symbols": ",".join(symbols[:30]), "limit": "20", "sort": "desc"})
    body = _http_text(
        f"https://data.alpaca.markets/v1beta1/news?{query}",
        headers={"Apca-Api-Key-Id": key, "Apca-Api-Secret-Key": secret},
    )
    if not body:
        return []
    try:
        payload = pyjson.loads(body)
    except ValueError:
        return []

    items: list[IntelItemModel] = []
    for story in payload.get("news", [])[:20]:
        title = _clean_summary(story.get("headline"), limit=120)
        summary = _clean_summary(story.get("summary"))
        url = story.get("url") or ""
        if not title or not url:
            continue
        story_symbols = [str(symbol).upper() for symbol in story.get("symbols", []) if str(symbol).upper() in symbols]
        impact = _intel_impact(title, summary)
        theme = _catalyst_theme(title, summary, impact)
        relevance = _market_relevance(title, summary, story_symbols, impact)
        direction = _impact_direction(title, summary)
        items.append(
            IntelItemModel(
                id=f"alpaca:{story.get('id') or url}",
                title=title,
                summary=summary or impact,
                source=story.get("source") or "Alpaca News",
                url=url,
                published_at=_parse_datetime(story.get("created_at") or story.get("updated_at")),
                symbols=story_symbols[:6],
                category=impact.lower(),
                impact=impact,
                confidence="High" if story_symbols else "Medium",
                media_type="article",
                catalyst_theme=theme,
                market_relevance=relevance,
                affected_symbols=story_symbols[:6],
                impact_summary=_impact_summary(theme, relevance, direction, story_symbols),
                impact_direction=direction,
            )
        )
    return items


def _fetch_gdelt_items(symbols: list[str]) -> list[IntelItemModel]:
    query = os.environ.get(
        "GDELT_INTEL_QUERY",
        "(market OR stocks OR inflation OR Federal Reserve OR oil OR tariffs OR earnings OR bitcoin)",
    )
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(
        {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": "12",
            "sort": "HybridRel",
        }
    )
    body = _http_text(url)
    if not body:
        return []
    try:
        payload = pyjson.loads(body)
    except ValueError:
        return []
    items: list[IntelItemModel] = []
    for article in payload.get("articles", [])[:12]:
        title = _clean_summary(article.get("title"), limit=120)
        url = article.get("url") or ""
        domain = article.get("domain") or "GDELT"
        if not title or not url:
            continue
        summary = _clean_summary(article.get("seendate") or article.get("sourceCountry") or "Global market event")
        matched = _story_symbols(title, summary, symbols)
        impact = _intel_impact(title, summary)
        theme = _catalyst_theme(title, summary, impact)
        relevance = _market_relevance(title, summary, matched, impact)
        direction = _impact_direction(title, summary)
        items.append(
            IntelItemModel(
                id=f"gdelt:{url}",
                title=title,
                summary=summary,
                source=domain,
                url=url,
                published_at=_parse_datetime(article.get("seendate")),
                symbols=matched,
                category=impact.lower(),
                impact=impact,
                confidence="Medium",
                media_type="article",
                catalyst_theme=theme,
                market_relevance=relevance,
                affected_symbols=matched[:6],
                impact_summary=_impact_summary(theme, relevance, direction, matched),
                impact_direction=direction,
            )
        )
    return items


def _intel_feed(symbols: str | None, limit: int = 30) -> IntelFeedResponse:
    symbol_list = _parse_symbols(symbols)
    cache_ttl = _intel_cache_seconds()
    cache_key = f"{limit}:{','.join(symbol_list)}"
    now = datetime.now(timezone.utc)
    if cache_ttl > 0:
        cached = _INTEL_CACHE.get(cache_key)
        if cached is not None:
            cached_at, cached_feed = cached
            if (now - cached_at).total_seconds() <= cache_ttl:
                return cached_feed

    yahoo_symbols = ",".join(symbol_list[:20])
    rss_feeds = [
        (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yahoo_symbols}&region=US&lang=en-US", "Yahoo Finance", "article"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC Markets", "article"),
        ("https://www.youtube.com/feeds/videos.xml?channel_id=UCrp_UI8XtuYfpiqluWLD7Lw", "CNBC Television", "video"),
    ]
    items: list[IntelItemModel] = []
    items.extend(_fetch_alpaca_news(symbol_list))
    items.extend(_fetch_gdelt_items(symbol_list))
    for feed_url, source, media_type in rss_feeds:
        items.extend(_fetch_rss_items(feed_url, source, symbol_list, media_type=media_type))

    feed = IntelFeedResponse(
        timestamp=now,
        refresh_after_seconds=cache_ttl or 300,
        symbols=symbol_list,
        items=_dedupe_intel(items, limit=max(5, min(50, limit))),
    )
    if cache_ttl > 0:
        _INTEL_CACHE[cache_key] = (now, feed)
    return feed


def _market_data_provider() -> str:
    provider = os.environ.get("MARKET_DATA_PROVIDER", "").strip().lower()
    if provider:
        return provider
    has_alpaca = bool(
        (os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID"))
        and (os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY"))
    )
    return "alpaca" if has_alpaca else "yfinance"


def _fetch_watchlist(symbol_list: list[str], months: int):
    provider = _market_data_provider()
    if provider in {"alpaca", "polygon", "finnhub"}:
        symbol_to_df = fetch_live_watchlist(symbol_list, provider=provider, months=months)
        if symbol_to_df:
            return symbol_to_df, provider
        # Primary daily/history provider returned nothing (e.g. Alpaca historical-data
        # access is unavailable while intraday still works). Fall back to free yfinance
        # daily bars so the Daily tab and full leaderboard aren't silently empty.
        return fetch_yfinance_watchlist(symbol_list, months=months), "yfinance"
    return fetch_yfinance_watchlist(symbol_list, months=months), "yfinance"


def _fetch_intraday_table(symbol_list: list[str]) -> tuple[pd.DataFrame, str]:
    provider = _market_data_provider()
    interval = os.environ.get("INTRADAY_SCAN_INTERVAL", "1m")

    if provider == "alpaca":
        # Try the configured feed first, then fall back to the free "iex" feed so
        # intraday / Raw Heat isn't empty when "sip" (paid) isn't on the account.
        feeds = [ALPACA_FEED] + (["iex"] if ALPACA_FEED != "iex" else [])
        for feed in feeds:
            try:
                symbol_to_df = fetch_intraday_watchlist(
                    symbol_list,
                    provider="alpaca",
                    interval=interval,
                    days=5,
                    feed=feed,
                )
                table = daytrade_table(symbol_to_df)
            except Exception as exc:  # noqa: BLE001
                print(f"intraday fetch failed on feed={feed}: {exc}")
                continue
            if not table.empty:
                return table, f"alpaca_{feed}"
        print("intraday Alpaca feeds produced no rows; trying yfinance fallback")
    else:
        print(f"intraday provider={provider}; trying yfinance fallback")

    try:
        fallback_limit = max(5, int(os.environ.get("INTRADAY_YFINANCE_SYMBOL_LIMIT", "12")))
    except ValueError:
        fallback_limit = 12
    fallback_interval = os.environ.get("INTRADAY_YFINANCE_INTERVAL", "5m").strip() or "5m"
    try:
        symbol_to_df = fetch_intraday_watchlist(
            symbol_list[:fallback_limit],
            provider="yfinance",
            interval=fallback_interval,
        )
        table = daytrade_table(symbol_to_df)
    except Exception as exc:  # noqa: BLE001
        print(f"intraday yfinance fallback failed: {exc}")
    else:
        if not table.empty:
            return table, "yfinance"
    return pd.DataFrame(columns=INTRADAY_TABLE_COLUMNS), "unavailable"


def _split_reasons(reason: str | None) -> list[str]:
    if not reason:
        return []
    return [piece.strip() for piece in str(reason).split(",") if piece.strip()]


def _coerce_readability_label(label: str | None) -> ReadabilityScore:
    normalized = str(label or "MIXED").upper()
    if normalized == "CLEAN":
        return ReadabilityScore.CLEAN
    if normalized == "CHAOTIC":
        return ReadabilityScore.CHAOTIC
    return ReadabilityScore.MIXED


def _state_sort_key(signal: SignalCardModel) -> tuple[int, int, int, int, str]:
    order = {
        AttentionState.ACTIVE_HEAT: 0,
        AttentionState.HEATING: 1,
        AttentionState.COOLING: 2,
        AttentionState.CHAOTIC: 3,
    }
    source_order = 0 if signal.source.startswith("intraday") else 1
    return (
        source_order,
        signal.rank if signal.rank is not None else 999,
        -signal.heat_score,
        order[signal.state],
        signal.ticker,
    )


def _intraday_state(row: dict) -> AttentionState:
    setup = str(row.get("setup", "NO_TRADE")).upper()
    readability = str(row.get("readability_label", "MIXED")).upper()
    score = float(row.get("score") or 0.0)
    if readability == "CHAOTIC":
        return AttentionState.CHAOTIC
    if setup.startswith("ALERT"):
        return AttentionState.ACTIVE_HEAT
    if setup.endswith("WATCH"):
        return AttentionState.HEATING
    if score >= 1.25:
        return AttentionState.COOLING
    return AttentionState.CHAOTIC if readability == "CHAOTIC" else AttentionState.COOLING


def _daily_state(row: dict) -> AttentionState:
    label = str(row.get("label", "NO_TRADE")).upper()
    readability = str(row.get("readability_label", "MIXED")).upper()
    hotness = float(row.get("hotness") or 0.0)
    if readability == "CHAOTIC":
        return AttentionState.CHAOTIC
    if label in {"LONG", "SHORT"} and hotness >= 0.55:
        return AttentionState.HEATING
    if hotness >= 0.50:
        return AttentionState.COOLING
    return AttentionState.CHAOTIC if readability == "CHAOTIC" else AttentionState.COOLING


def _build_plan(direction: str, price: float | None, stop_distance: float | None) -> HypotheticalPlan | None:
    if price is None or stop_distance is None or stop_distance <= 0:
        return None

    buffer = max(price * 0.0005, stop_distance * 0.08)
    reward_multiple = 1.5
    if direction == "SHORT":
        stop_loss = price + stop_distance
        profit_target = price - (stop_distance * reward_multiple)
    else:
        stop_loss = price - stop_distance
        profit_target = price + (stop_distance * reward_multiple)

    return HypotheticalPlan(
        direction=direction,
        entry_zone_start=round(price - buffer, 2),
        entry_zone_end=round(price + buffer, 2),
        stop_loss=round(stop_loss, 2),
        profit_target=round(profit_target, 2),
        risk_reward=reward_multiple,
        label=f"Hypothetical {direction.title()} Scenario",
    )


def _intraday_signal_card(row: dict) -> SignalCardModel:
    setup = str(row.get("setup", "NO_TRADE")).upper()
    direction = "SHORT" if "SHORT" in setup else "LONG"
    price = float(row["price"]) if pd.notna(row.get("price")) else None
    stop_distance = float(row["stop_distance"]) if pd.notna(row.get("stop_distance")) else None
    timestamp = row.get("time")
    expires_at = None
    seconds_remaining = None
    if timestamp is not None and pd.notna(timestamp):
        expires_at = pd.Timestamp(timestamp).to_pydatetime() + pd.Timedelta(minutes=15)
        seconds_remaining = max(0, int((expires_at - datetime.now(expires_at.tzinfo)).total_seconds()))

    why = _split_reasons(row.get("reason"))
    if row.get("vwap") is not None and pd.notna(row.get("vwap")):
        why.append("VWAP-aware")
    if row.get("or_high") is not None and pd.notna(row.get("or_high")):
        why.append("Opening range mapped")

    return SignalCardModel(
        ticker=str(row.get("symbol", "")).upper(),
        source=f"intraday_{os.environ.get('INTRADAY_SCAN_INTERVAL', '1m')}",
        state=_intraday_state(row),
        pulse_status=setup,
        heat_score=max(0, min(100, int(round((float(row.get("score") or 0.0) / 3.5) * 100)))),
        readability_score=max(0, min(100, int(round(float(row.get("readability") or 0.0))))),
        readability=_coerce_readability_label(row.get("readability_label")),
        rank=int(row["rank"]) if pd.notna(row.get("rank")) else None,
        price=round(price, 2) if price is not None else None,
        vwap=round(float(row["vwap"]), 2) if pd.notna(row.get("vwap")) else None,
        opening_range_high=round(float(row["or_high"]), 2) if pd.notna(row.get("or_high")) else None,
        opening_range_low=round(float(row["or_low"]), 2) if pd.notna(row.get("or_low")) else None,
        relative_volume=round(float(row["rel_volume"]), 2) if pd.notna(row.get("rel_volume")) else None,
        stop_distance=round(stop_distance, 2) if stop_distance is not None else None,
        why_metrics=why or ["No clean opening range setup"],
        plan=_build_plan(direction, price, stop_distance) if "ALERT" in setup or "WATCH" in setup else None,
        expires_at=expires_at,
        seconds_remaining=seconds_remaining,
    )


def _daily_signal_card(row: dict) -> SignalCardModel:
    label = str(row.get("label", "NO_TRADE")).upper()
    direction = "SHORT" if label == "SHORT" else "LONG"
    stop_distance = float(row["stop_distance"]) if pd.notna(row.get("stop_distance")) else None
    price = None
    reasons = _split_reasons(row.get("reason"))
    if label in {"LONG", "SHORT"}:
        reasons.append("Daily structure aligned")
    if row.get("trend_score") is not None:
        reasons.append("Trend score measured")
    if row.get("breakout_score") is not None:
        reasons.append("Breakout score measured")

    return SignalCardModel(
        ticker=str(row.get("symbol", "")).upper(),
        source="daily",
        state=_daily_state(row),
        pulse_status=f"DAILY_{label}",
        heat_score=max(0, min(100, int(round(float(row.get("hotness") or 0.0) * 100)))),
        readability_score=max(0, min(100, int(round(float(row.get("readability") or 0.0))))),
        readability=_coerce_readability_label(row.get("readability_label")),
        rank=int(row["rank"]) if pd.notna(row.get("rank")) else None,
        stop_distance=round(stop_distance, 2) if stop_distance is not None else None,
        why_metrics=reasons or ["Daily setup not clean enough"],
        plan=_build_plan(direction, price, stop_distance) if label in {"LONG", "SHORT"} else None,
    )


def _attention_signals(daily_table: pd.DataFrame, intraday_table: pd.DataFrame) -> list[SignalCardModel]:
    signals: list[SignalCardModel] = []
    if not intraday_table.empty:
        signals.extend(_intraday_signal_card(row) for row in intraday_table.to_dict(orient="records"))
    if not daily_table.empty:
        signals.extend(_daily_signal_card(row) for row in daily_table.to_dict(orient="records"))

    intraday_signals = [signal for signal in signals if signal.source.startswith("intraday")]
    daily_signals = [signal for signal in signals if signal.source == "daily"]
    for signal in intraday_signals:
        signal.attention_score = signal.heat_score
        is_raw_heat = signal.readability == ReadabilityScore.CHAOTIC or signal.readability_score < 55
        signal.lane = "RAW_HEAT" if is_raw_heat else "BEST_BID"
        signal.rank_reason = _rank_reasons(signal)
        signal.heat_reason = _heat_reasons(signal)
        signal.risk_note = _risk_note(signal)
        signal.risk_reasons = _risk_reasons(signal)
        _apply_market_read(signal)

    for signal in daily_signals:
        signal.attention_score = signal.heat_score
        signal.rank_reason = _rank_reasons(signal)
        signal.risk_note = _risk_note(signal)
        signal.risk_reasons = _risk_reasons(signal)
        _apply_market_read(signal)

    best_bids = sorted(
        (signal for signal in intraday_signals if signal.lane == "BEST_BID"),
        key=lambda signal: (
            -(signal.attention_score * (signal.readability_score / 100.0)),
            -signal.readability_score,
            -signal.attention_score,
            signal.ticker,
        ),
    )
    for index, signal in enumerate(best_bids, start=1):
        signal.best_bid_rank = index

    raw_heat = sorted(
        intraday_signals,
        key=lambda signal: (
            -signal.attention_score,
            -signal.readability_score,
            signal.ticker,
        ),
    )
    for index, signal in enumerate(raw_heat, start=1):
        signal.raw_heat_rank = index

    return sorted(signals, key=_state_sort_key)


def _apply_market_read(signal: SignalCardModel) -> None:
    alignment_score = _alignment_score(signal)
    signal.alignment_score = alignment_score
    signal.pillar_alignment = _pillar_alignment(signal, alignment_score)
    signal.alignment_reasoning = _alignment_reasoning(signal)
    signal.conviction_level = _conviction_level(alignment_score)
    signal.market_read = _market_read(signal)


def _alignment_score(signal: SignalCardModel) -> int:
    score = int(round((signal.attention_score * 0.45) + (signal.readability_score * 0.55)))

    if signal.plan is not None:
        score += 8
    if signal.relative_volume is not None and signal.relative_volume >= 1.0:
        score += 5
    if signal.lane == "RAW_HEAT":
        score -= 12
    if signal.readability == ReadabilityScore.CHAOTIC:
        score -= 8

    return max(0, min(100, score))


def _pillar_alignment(signal: SignalCardModel, alignment_score: int) -> str:
    if signal.lane == "RAW_HEAT":
        if alignment_score >= 55:
            return "Attention active, structure contested"
        return "Attention active, structure weak"
    if alignment_score >= 70:
        return "Attention and structure aligned"
    if alignment_score >= 55:
        return "Attention and structure forming"
    return "Attention present, confirmation limited"


def _alignment_reasoning(signal: SignalCardModel) -> list[str]:
    reasons: list[str] = []

    if signal.attention_score >= 25:
        reasons.append("Attention is increasing")
    elif signal.attention_score >= 15:
        reasons.append("Attention is active")
    else:
        reasons.append("Attention is present")

    if signal.readability == ReadabilityScore.CLEAN:
        reasons.append("Structure is supportive")
    elif signal.readability == ReadabilityScore.MIXED:
        reasons.append("Structure is still forming")
    else:
        reasons.append("Structure is unstable")

    if signal.plan is not None:
        reasons.append("Timing confirms the setup")
    elif signal.lane == "RAW_HEAT":
        reasons.append("Timing remains unresolved")
    else:
        reasons.append("Timing needs confirmation")

    return reasons[:3]


def _conviction_level(alignment_score: int) -> str:
    if alignment_score >= 72:
        return "High"
    if alignment_score >= 52:
        return "Medium"
    return "Low"


def _market_read(signal: SignalCardModel) -> str:
    if signal.source == "daily":
        if signal.readability == ReadabilityScore.CLEAN and signal.plan is not None:
            return "Daily structure is readable and directional, giving the setup stronger follow-through potential."
        if signal.readability == ReadabilityScore.MIXED:
            return "Daily structure is present, but confirmation is still mixed across the broader setup."
        if signal.readability == ReadabilityScore.CHAOTIC:
            return "Daily activity is visible, but structure remains too unstable for strong conviction."
        return "Daily structure is being monitored, but the setup has not produced a clear read yet."

    if signal.lane == "RAW_HEAT":
        if signal.readability == ReadabilityScore.CHAOTIC:
            return "Participation is accelerating, but structure remains unstable. Conviction stays limited until the move becomes readable."
        return "Activity is increasing, but quality control has not fully confirmed the trade path."

    if signal.readability == ReadabilityScore.CLEAN and signal.plan is not None:
        return "Participation is building inside a readable structure while timing and direction are aligned."
    if signal.readability == ReadabilityScore.CLEAN:
        return "Attention is present inside a clean structure, with confirmation still developing."
    if signal.readability == ReadabilityScore.MIXED:
        return "Readable pressure is forming, but confirmation remains incomplete."
    return "Attention is present, but the current structure is not clear enough for strong conviction."


def _rank_reasons(signal: SignalCardModel) -> list[str]:
    reasons: list[str] = []

    if signal.readability == ReadabilityScore.CLEAN:
        reasons.append("Structure is clean")
    elif signal.readability == ReadabilityScore.MIXED:
        reasons.append("Structure is partially aligned")
    else:
        reasons.append("Activity is elevated")

    if signal.relative_volume is not None:
        if signal.relative_volume >= 1.0:
            reasons.append("Live participation is active")
        else:
            reasons.append("Participation is still developing")

    if signal.plan is not None:
        reasons.append("Timing and direction are aligned")
    elif signal.vwap is not None:
        reasons.append("VWAP context is mapped")

    for metric in signal.why_metrics:
        if metric not in reasons and len(reasons) < 4:
            reasons.append(metric)

    return reasons[:4]


def _heat_reasons(signal: SignalCardModel) -> list[str]:
    reasons: list[str] = []

    if signal.attention_score >= 25:
        reasons.append("Participation surge detected")
    elif signal.attention_score >= 15:
        reasons.append("Activity is building")
    else:
        reasons.append("Attention is emerging")

    if signal.relative_volume is not None:
        if signal.relative_volume >= 1.0:
            reasons.append("Volume expansion present")
        else:
            reasons.append("Relative volume is developing")

    if signal.vwap is not None:
        reasons.append("VWAP pressure is active")

    for metric in signal.why_metrics:
        if metric not in reasons and len(reasons) < 3:
            reasons.append(metric)

    return reasons[:3]


def _risk_note(signal: SignalCardModel) -> str | None:
    if signal.lane == "RAW_HEAT":
        return "High activity, low readability. Trade path is unstable."
    if signal.readability == ReadabilityScore.MIXED:
        return "Readable pressure is forming, but confirmation is incomplete."
    return None


def _risk_reasons(signal: SignalCardModel) -> list[str]:
    if signal.lane != "RAW_HEAT":
        if signal.readability == ReadabilityScore.MIXED:
            return ["Confirmation is incomplete"]
        return []

    reasons: list[str] = []
    if signal.readability == ReadabilityScore.CHAOTIC:
        reasons.append("Structure unstable")
    elif signal.readability_score < 55:
        reasons.append("Readability below quality threshold")

    if signal.plan is None:
        reasons.append("Timing conflict detected")

    if signal.stop_distance is not None and signal.price is not None:
        stop_pct = signal.stop_distance / max(signal.price, 0.01)
        if stop_pct > 0.012:
            reasons.append("Stop distance is elevated")

    if not reasons:
        reasons.append("Trade path is unstable")

    return reasons[:3]


def _model_json_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return pyjson.loads(model.json())


def _normalize_identifier(identifier: str) -> str:
    return identifier.strip()


def _infer_channel(identifier: str) -> str:
    return "email" if "@" in identifier else "push"


def _normalize_channel(channel: str | None, identifier: str) -> str:
    normalized = str(channel or "").strip().lower()
    if normalized in {"push", "email"}:
        return normalized
    return _infer_channel(identifier)


def _delivery_readiness(channel: str, device_token: str | None) -> tuple[bool, str | None]:
    if channel == "email":
        if not os.environ.get("RESEND_API_KEY") or not os.environ.get("RESEND_FROM_EMAIL"):
            return False, "Email delivery is not configured on this backend."
    if channel == "push":
        if not device_token:
            return False, "Push delivery needs an APNs device token from the app."
        missing = [
            name
            for name in ("APNS_KEY_ID", "APNS_TEAM_ID")
            if not os.environ.get(name)
        ]
        if missing or not (os.environ.get("APNS_AUTH_KEY") or os.environ.get("APNS_AUTH_KEY_PATH")):
            return False, "Push delivery is missing APNs credentials."
    return True, None


def _split_contact(identifier: str):
    if "@" in identifier:
        return identifier, None
    return None, None


def _save_user(
    identifier: str,
    watchlist: str,
    alerts_enabled: bool,
    notify_at: str,
    timezone_name: str = "America/New_York",
    channel_override: str | None = None,
    device_token: str | None = None,
):
    identifier = _normalize_identifier(identifier)
    email, phone = _split_contact(identifier)
    channel = _normalize_channel(channel_override, identifier)
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO users (identifier, email, phone, channel, watchlist, alerts_enabled, notify_at, timezone, device_token, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(identifier) DO UPDATE SET
                email=excluded.email,
                phone=excluded.phone,
                channel=excluded.channel,
                watchlist=excluded.watchlist,
                alerts_enabled=excluded.alerts_enabled,
                notify_at=excluded.notify_at,
                timezone=excluded.timezone,
                device_token=COALESCE(excluded.device_token, users.device_token),
                updated_at=CURRENT_TIMESTAMP
            """,
            (identifier, email, phone, channel, watchlist, int(alerts_enabled), notify_at, timezone_name, device_token),
        )


def _get_user(identifier: str | None):
    if not identifier:
        return None
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE identifier = ?", (identifier,)).fetchone()
    return dict(row) if row else None


def _current_user(request: Request):
    return _get_user(request.cookies.get("market_scanner_user"))


def _all_alert_users() -> list[dict]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM users WHERE alerts_enabled = 1").fetchall()
    return [dict(row) for row in rows]


def _alert_destination(user: dict) -> str:
    if user.get("channel") == "push" and user.get("device_token"):
        return user["device_token"]
    return user.get("email") or user.get("phone") or user.get("identifier") or ""


def _db_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _user_local_now(user: dict, now: datetime) -> datetime:
    try:
        zone = ZoneInfo(user.get("timezone") or "America/New_York")
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("America/New_York")
    return now.astimezone(zone)


def _parse_notify_time(raw: str | None) -> time:
    try:
        hour, minute = str(raw or "05:00").split(":", 1)
        return time(hour=max(0, min(23, int(hour))), minute=max(0, min(59, int(minute[:2]))))
    except Exception:
        return time(hour=5, minute=0)


def _recent_alert_exists(user_identifier: str, alert_type: str, alert_key: str, cooldown: timedelta, now: datetime) -> bool:
    cutoff = _db_timestamp(now - cooldown)
    with _db() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM alert_events
            WHERE user_identifier = ?
              AND alert_type = ?
              AND alert_key = ?
              AND created_at >= ?
            LIMIT 1
            """,
            (user_identifier, alert_type, alert_key, cutoff),
        ).fetchone()
    return row is not None


def _get_alert_state(user_identifier: str, state_key: str, default: str = "0") -> str:
    with _db() as conn:
        row = conn.execute(
            "SELECT state_value FROM alert_state WHERE user_identifier = ? AND state_key = ?",
            (user_identifier, state_key),
        ).fetchone()
    return str(row["state_value"]) if row else default


def _set_alert_state(user_identifier: str, state_key: str, state_value: str):
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO alert_state (user_identifier, state_key, state_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_identifier, state_key) DO UPDATE SET
                state_value=excluded.state_value,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_identifier, state_key, state_value),
        )


def _record_alert_event(
    user: dict,
    alert_type: str,
    alert_key: str,
    symbols: list[str],
    subject: str,
    message: str,
    result,
):
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO alert_events (
                user_identifier, alert_type, alert_key, symbols, channel, destination,
                subject, message, provider, delivery_ok, delivery_detail
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["identifier"],
                alert_type,
                alert_key,
                ",".join(symbols),
                user["channel"],
                _alert_destination(user),
                subject,
                message,
                getattr(result, "provider", None),
                int(bool(getattr(result, "ok", False))),
                getattr(result, "detail", None),
            ),
        )


def _recent_alert_events(identifier: str, limit: int = 20) -> list[AlertEventResponse]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, alert_type, alert_key, symbols, channel, destination, subject,
                   message, provider, delivery_ok, delivery_detail, created_at
            FROM alert_events
            WHERE user_identifier = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (identifier, max(1, min(100, limit))),
        ).fetchall()

    return [
        AlertEventResponse(
            id=int(row["id"]),
            alert_type=row["alert_type"],
            alert_key=row["alert_key"],
            symbols=row["symbols"],
            channel=row["channel"],
            destination=row["destination"],
            subject=row["subject"],
            message=row["message"],
            provider=row["provider"],
            delivery_ok=bool(row["delivery_ok"]),
            delivery_detail=row["delivery_detail"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def _send_alert(user: dict, alert_type: str, alert_key: str, symbols: list[str], subject: str, message: str) -> dict:
    destination = _alert_destination(user)
    if not destination:
        return {"sent": False, "alert_type": alert_type, "reason": "missing destination"}

    result = send_notification(
        channel=user["channel"],
        destination=destination,
        subject=subject,
        text=message,
        html=_notification_html({"subject": subject, "message": message}),
    )

    _record_alert_event(user, alert_type, alert_key, symbols, subject, message, result)
    return {
        "sent": bool(getattr(result, "ok", False)),
        "alert_type": alert_type,
        "symbols": symbols,
        "channel": user["channel"],
        "detail": getattr(result, "detail", ""),
    }


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
    pieces = [f"5 AM Attention Brief: {summary['market_stance']}."]
    if leader:
        pieces.append(f"Structure leader {leader['symbol']} ({leader['label']} {leader['setup_grade']}).")
    if top_longs:
        pieces.append(f"Best long candidates: {', '.join(top_longs)}.")
    if avoid_names:
        pieces.append(f"Low-readability names: {', '.join(avoid_names)}.")
    return {
        "channel": user["channel"],
        "destination": user["email"] or user["phone"] or user["identifier"],
        "notify_at": user["notify_at"],
        "enabled": bool(user["alerts_enabled"]),
        "subject": f"5 AM Attention Brief: {summary['market_stance']}",
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


def _signal_attention_score(signal: SignalCardModel) -> int:
    return int(signal.attention_score or signal.heat_score or 0)


def _is_best_bid_signal(signal: SignalCardModel, minimum_attention: int = 25) -> bool:
    return (
        signal.source.startswith("intraday")
        and signal.state in {AttentionState.ACTIVE_HEAT, AttentionState.HEATING}
        and (signal.lane == "BEST_BID" or signal.readability != ReadabilityScore.CHAOTIC)
        and signal.readability_score >= 55
        and _signal_attention_score(signal) >= minimum_attention
    )


def _is_priority_attention(signal: SignalCardModel, minimum_attention: int = 25) -> bool:
    return (
        signal.source.startswith("intraday")
        and signal.state in {AttentionState.ACTIVE_HEAT, AttentionState.HEATING}
        and _signal_attention_score(signal) >= minimum_attention
    )


def _clean_active_signals(signals: list[SignalCardModel]) -> list[SignalCardModel]:
    return [
        signal
        for signal in signals
        if _is_best_bid_signal(signal, minimum_attention=25) and signal.readability == ReadabilityScore.CLEAN
    ]


def _should_send_5am(user: dict, now: datetime) -> bool:
    local_now = _user_local_now(user, now)
    notify_at = _parse_notify_time(user.get("notify_at"))
    scheduled = local_now.replace(hour=notify_at.hour, minute=notify_at.minute, second=0, microsecond=0)
    return timedelta(0) <= local_now - scheduled < timedelta(minutes=8)


def _evaluate_5am_alert(user: dict, snapshot: LiveNowResponse, now: datetime) -> dict | None:
    if not _should_send_5am(user, now):
        return None

    local_day = _user_local_now(user, now).date().isoformat()
    alert_key = f"5am:{local_day}"
    if _recent_alert_exists(user["identifier"], "premarket_5am", alert_key, timedelta(hours=26), now):
        return {"sent": False, "alert_type": "premarket_5am", "reason": "daily cooldown"}

    leaders = sorted(
        (signal for signal in snapshot.signals if _is_best_bid_signal(signal, minimum_attention=20)),
        key=lambda signal: (
            signal.best_bid_rank if signal.best_bid_rank is not None else 999,
            -(_signal_attention_score(signal) * (signal.readability_score / 100.0)),
        ),
    )
    top = leaders[:3]
    symbols = [signal.ticker for signal in top]
    if symbols:
        leader_bias = ", ".join(f"{signal.ticker} {_signal_direction(signal)}" for signal in top)
        message = f"5 AM Attention Brief: {snapshot.market_stance}. Best Bids to watch: {leader_bias}. Structure has survived the readability filter."
    else:
        message = f"5 AM Attention Brief: {snapshot.market_stance}. No clean Best Bid yet; protect attention until structure improves."
    return _send_alert(
        user,
        "premarket_5am",
        alert_key,
        symbols,
        f"5 AM Attention Brief: {snapshot.market_stance}",
        message,
    )


def _evaluate_priority_alerts(user: dict, snapshot: LiveNowResponse, now: datetime) -> list[dict]:
    results: list[dict] = []
    watched = set(_parse_symbols(user.get("watchlist")))
    candidates = [
        signal
        for signal in snapshot.signals
        if signal.ticker in watched and _is_priority_attention(signal, minimum_attention=25)
    ]

    for signal in candidates:
        alert_key = f"priority:{signal.ticker}"
        if _recent_alert_exists(user["identifier"], "priority_symbol", alert_key, timedelta(minutes=30), now):
            results.append({"sent": False, "alert_type": "priority_symbol", "symbol": signal.ticker, "reason": "30m cooldown"})
            continue

        lane = "Raw Heat" if signal.lane == "RAW_HEAT" else "Best Bid"
        message = (
            f"{signal.ticker} is drawing {lane} attention. "
            f"Attention {_signal_attention_score(signal)}, readability {signal.readability_score}; "
            f"{(signal.market_read or signal.pulse_status).replace('_', ' ').lower()}."
        )
        results.append(
            _send_alert(
                user,
                "priority_symbol",
                alert_key,
                [signal.ticker],
                f"Priority watch: {signal.ticker}",
                message,
            )
        )

    return results


def _evaluate_sustained_alerts(user: dict, snapshot: LiveNowResponse, now: datetime) -> list[dict]:
    results: list[dict] = []
    active = [signal for signal in snapshot.signals if _is_best_bid_signal(signal, minimum_attention=30)]
    active_symbols = {signal.ticker for signal in active}

    for symbol in _parse_symbols(user.get("watchlist")):
        state_key = f"active_since:{symbol}"
        if symbol not in active_symbols:
            _set_alert_state(user["identifier"], state_key, "")

    for signal in active:
        state_key = f"active_since:{signal.ticker}"
        raw_since = _get_alert_state(user["identifier"], state_key, "")
        if not raw_since:
            _set_alert_state(user["identifier"], state_key, now.isoformat())
            continue

        try:
            active_since = datetime.fromisoformat(raw_since)
        except ValueError:
            _set_alert_state(user["identifier"], state_key, now.isoformat())
            continue

        if now - active_since < timedelta(minutes=5):
            continue

        alert_key = f"sustained:{signal.ticker}"
        if _recent_alert_exists(user["identifier"], "sustained_attention", alert_key, timedelta(hours=2), now):
            results.append({"sent": False, "alert_type": "sustained_attention", "symbol": signal.ticker, "reason": "2h cooldown"})
            continue

        results.append(
            _send_alert(
                user,
                "sustained_attention",
                alert_key,
                [signal.ticker],
                f"{signal.ticker} held readable attention",
                f"{signal.ticker} has held Best Bid pressure for 5+ minutes. Attention {_signal_attention_score(signal)}, readability {signal.readability_score}; {signal.conviction_level or 'conviction unscored'}.",
            )
        )

    return results


def _evaluate_market_hot_alert(user: dict, snapshot: LiveNowResponse, now: datetime) -> dict | None:
    hot = sorted(
        _clean_active_signals(snapshot.signals),
        key=lambda signal: _signal_attention_score(signal) * (signal.readability_score / 100.0),
        reverse=True,
    )
    if len(hot) < 3:
        return None

    alert_key = "market_hot"
    if _recent_alert_exists(user["identifier"], "market_hot", alert_key, timedelta(minutes=30), now):
        return {"sent": False, "alert_type": "market_hot", "reason": "30m cooldown"}

    symbols = [signal.ticker for signal in hot[:4]]
    display_name = str(user.get("identifier") or "").split("@", 1)[0].split("+", 1)[0].strip()
    name_prefix = f"{display_name}, " if display_name else ""
    return _send_alert(
        user,
        "market_hot",
        alert_key,
        symbols,
        f"{name_prefix}Best Bids are forming",
        f"Readable attention is clustering in {', '.join(symbols)}. This is a broad Best Bid event, not a single-stock blip.",
    )


def _evaluate_slowdown_alert(user: dict, snapshot: LiveNowResponse, now: datetime) -> dict | None:
    local_now = _user_local_now(user, now)
    clean_count = len(_clean_active_signals(snapshot.signals))
    previous = int(_get_alert_state(user["identifier"], "previous_clean_active_count", "0") or "0")
    _set_alert_state(user["identifier"], "previous_clean_active_count", str(clean_count))

    if local_now.hour < 12:
        return None
    if previous < 3 or clean_count > 1:
        return None

    alert_key = "afternoon_slowdown"
    if _recent_alert_exists(user["identifier"], "afternoon_slowdown", alert_key, timedelta(minutes=90), now):
        return {"sent": False, "alert_type": "afternoon_slowdown", "reason": "90m cooldown"}

    return _send_alert(
        user,
        "afternoon_slowdown",
        alert_key,
        [],
        "Readable bids are thinning",
        f"The clean Best Bid list cooled from {previous} names to {clean_count}. This may be a review moment rather than a chase moment.",
    )


def _evaluate_alerts_for_user(user: dict, snapshot: LiveNowResponse, now: datetime) -> list[dict]:
    details: list[dict] = []
    for evaluator in (
        _evaluate_5am_alert,
        _evaluate_market_hot_alert,
        _evaluate_slowdown_alert,
    ):
        result = evaluator(user, snapshot, now)
        if result:
            details.append(result)
    details.extend(_evaluate_priority_alerts(user, snapshot, now))
    details.extend(_evaluate_sustained_alerts(user, snapshot, now))
    return details


def _fetch_intraday_snapshot(symbol: str):
    try:
        import yfinance as yf
    except Exception:
        return None

    try:
        data = yf.download(
            symbol.upper(),
            period="1d",
            interval="5m",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return None
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
        "readability": round(read["readability"], 1),
        "readability_label": read["readability_label"],
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
    symbol_to_df, _provider = _fetch_watchlist(symbol_list, months=6)
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


@app.post("/api/v1/alerts/register", response_model=AlertRegistrationResponse)
async def api_register_alerts(payload: AlertRegistrationRequest):
    identifier = _normalize_identifier(payload.identifier)
    channel = _normalize_channel(payload.channel, identifier)
    if not identifier:
        return AlertRegistrationResponse(
            ok=False,
            channel=channel,
            notify_at=payload.notify_at,
            timezone=payload.timezone,
            message="Enter a valid alert destination.",
        )

    ready, readiness_message = _delivery_readiness(channel, payload.device_token)
    if not ready:
        return AlertRegistrationResponse(
            ok=False,
            channel=channel,
            notify_at=payload.notify_at or "05:00",
            timezone=payload.timezone or "America/New_York",
            message=readiness_message,
        )

    watchlist = ",".join(_parse_symbols(payload.watchlist or ",".join(DEFAULT_SYMBOLS)))
    _save_user(
        identifier=identifier,
        watchlist=watchlist,
        alerts_enabled=True,
        notify_at=payload.notify_at or "05:00",
        timezone_name=payload.timezone or "America/New_York",
        channel_override=channel,
        device_token=payload.device_token,
    )
    return AlertRegistrationResponse(
        ok=True,
        channel=channel,
        notify_at=payload.notify_at or "05:00",
        timezone=payload.timezone or "America/New_York",
        message="Alerts are active.",
    )


@app.get("/api/v1/alerts/events", response_model=list[AlertEventResponse])
async def api_alert_events(
    identifier: str = Query(...),
    limit: int = Query(20, ge=1, le=100),
):
    return _recent_alert_events(_normalize_identifier(identifier), limit=limit)


@app.post("/api/v1/alerts/evaluate", response_model=AlertEvaluationResponse)
async def api_evaluate_alerts(
    symbols: str | None = Query(None),
    months: int = Query(6, ge=1, le=24),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
):
    _require_admin_token(x_admin_token)
    users = _all_alert_users()
    if not users:
        return AlertEvaluationResponse(ok=True, users_checked=0, alerts_sent=0, alerts_skipped=0, details=[])

    scan_symbols = set(_parse_symbols(symbols))
    for user in users:
        scan_symbols.update(_parse_symbols(user.get("watchlist")))

    snapshot = _live_now_snapshot(symbols=",".join(sorted(scan_symbols)), months=months)
    now = datetime.now(timezone.utc)
    details: list[dict] = []
    for user in users:
        details.extend(_evaluate_alerts_for_user(user, snapshot, now))

    alerts_sent = sum(1 for detail in details if detail.get("sent"))
    alerts_skipped = sum(1 for detail in details if not detail.get("sent"))
    return AlertEvaluationResponse(
        ok=True,
        users_checked=len(users),
        alerts_sent=alerts_sent,
        alerts_skipped=alerts_skipped,
        details=details,
    )


def _live_now_snapshot(symbols: str | None, months: int, refresh_after_seconds: int = 5) -> LiveNowResponse:
    symbol_list = _parse_symbols(symbols)
    cache_ttl = _snapshot_cache_seconds()
    cache_key = f"{months}:{','.join(symbol_list)}"
    now = datetime.now(timezone.utc)
    if cache_ttl > 0:
        cached = _LIVE_NOW_CACHE.get(cache_key)
        if cached is not None:
            cached_at, cached_snapshot = cached
            if (now - cached_at).total_seconds() <= cache_ttl:
                return cached_snapshot

    engine = MarketReadingEngine()
    symbol_to_df, _provider = _fetch_watchlist(symbol_list, months=months)
    daily_table = engine.hotness_table(symbol_to_df) if symbol_to_df else pd.DataFrame(columns=TABLE_COLUMNS)
    intraday_table, _intraday_provider = _fetch_intraday_table(symbol_list)
    summary = _market_summary(daily_table)
    response_timestamp = datetime.now(timezone.utc)
    snapshot = LiveNowResponse(
        market_stance=summary["market_stance"],
        timestamp=response_timestamp,
        refresh_after_seconds=refresh_after_seconds,
        symbols=symbol_list,
        signals=_attention_signals(daily_table, intraday_table),
    )
    if cache_ttl > 0:
        _LIVE_NOW_CACHE[cache_key] = (now, snapshot)
    return snapshot


async def _prewarm_snapshot_loop() -> None:
    """Keep the default /api/v1/now snapshot warm so no real user hits a cold 20s+ fetch."""
    while True:
        try:
            await asyncio.to_thread(_live_now_snapshot, None, 6)
        except Exception as exc:  # noqa: BLE001
            print(f"snapshot prewarm error: {exc}")
        await asyncio.sleep(max(30, _snapshot_cache_seconds() - 30))


@app.on_event("startup")
async def _start_snapshot_prewarm() -> None:
    asyncio.create_task(_prewarm_snapshot_loop())


@app.get("/api/v1/now", response_model=LiveNowResponse)
def api_now(
    symbols: str | None = Query(None),
    months: int = Query(6, ge=1, le=24),
):
    return _live_now_snapshot(symbols=symbols, months=months)


@app.get("/api/v1/intel", response_model=IntelFeedResponse)
def api_intel(
    symbols: str | None = Query(None),
    limit: int = Query(30, ge=5, le=50),
):
    return _intel_feed(symbols=symbols, limit=limit)


@app.websocket("/api/v1/now/stream")
async def api_now_stream(websocket: WebSocket):
    await websocket.accept()
    symbols = websocket.query_params.get("symbols")
    months = int(websocket.query_params.get("months", "6"))
    refresh_seconds = max(2, min(60, int(websocket.query_params.get("refresh", "5"))))
    try:
        while True:
            snapshot = _live_now_snapshot(
                symbols=symbols,
                months=months,
                refresh_after_seconds=refresh_seconds,
            )
            await websocket.send_json(_model_json_dict(snapshot))
            await asyncio.sleep(refresh_seconds)
    except WebSocketDisconnect:
        return


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

    symbol_to_df, provider = _fetch_watchlist(symbol_list, months=months)
    table = engine.hotness_table(symbol_to_df) if symbol_to_df else pd.DataFrame(columns=TABLE_COLUMNS)
    intraday_table, intraday_provider = _fetch_intraday_table(symbol_list)
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
            "intraday_table": intraday_table.to_dict(orient="records") if not intraday_table.empty else [],
            "detail_cards": detail_cards,
            "detail_cards_json": pyjson.dumps(detail_cards),
            "user": user,
            "notification_preview": notification_preview,
            "delivery_status": delivery,
            "data_provider": provider,
            "intraday_provider": intraday_provider,
        },
    )
