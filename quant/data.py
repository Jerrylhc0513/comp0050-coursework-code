#!/usr/bin/env python3
"""
data.py - US equity daily data pipeline (free Alpaca data)

What it maintains
  1. Security master: US common stocks on the major exchanges. Delisted stocks are included
     as well (INCLUDE_DELISTED) so that backtests are not limited to today's survivors.
  2. Daily bars: raw OHLCV plus total return `ret` (splits and dividends included),
     stored as one parquet file per month, updated incrementally every day.
  3. Universe: on the last trading day of each month, the top 1000 stocks by 60-day average
     dollar volume (price >= $5, traded on >= 50 of the last 60 days, ADV >= $5M).
     The list is used for every day of the following month and expanded to a (date, symbol) table.
  4. 1-minute bars (optional): only for stocks in that day's universe, regular hours 9:30-16:00,
     one parquet file per day.
  5. Benchmark: SPY daily bars for comparison.

Usage
  python data.py update                    # first run = full download; later runs = incremental
  python data.py update --minute           # also download / update 1-minute bars
  python data.py update --refresh-master   # force a refresh of the security master
  python data.py status                    # summary of local data

  If daily data already exists but starts after DAILY_START (e.g. an older install that started
  in 2020), `update` back-fills the missing early history automatically, once.

Reading data from your own code
  from data import load_daily, load_universe, current_universe, load_minute, panel
  df    = load_daily(start="2023-01-01", adjusted=True)     # long table with adj_close etc.
  close = panel("adj_close", start="2023-01-01")            # wide date x symbol, universe only
  pool  = current_universe()                                # latest universe list

Environment variables (can be put in a .env file next to this script)
  ALPACA_API_KEY, ALPACA_SECRET_KEY   required. A free Alpaca (paper) account is enough
  MASSIVE_API_KEY                     optional but recommended. Free Massive (formerly Polygon.io) key,
                                      used to identify common stocks precisely; without it a
                                      name-based rule removes ETFs, warrants, preferreds etc.
  ALPACA_TRADING_URL                  optional. Defaults to paper trading
  QUANT_DATA_DIR                      optional. Data directory, default ./data next to this file

Adjustment
  Raw prices are stored and never rewritten. A separate column `ret` holds the total return
  versus the previous trading day (splits + dividends). It is computed from Alpaca adjusted
  prices at download time and does not change afterwards. load_daily(adjusted=True) rebuilds
  adj_open / adj_high / adj_low / adj_close by compounding `ret`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# ============================== Settings ==============================
DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", Path(__file__).resolve().parent / "data"))

DAILY_START = "2016-01-01"     # first date of daily history (Alpaca SIP history starts in 2016)
MINUTE_START = "2023-10-01"    # first date of 1-minute history

UNIVERSE_SIZE = 1000           # stocks in the universe
ADV_WINDOW = 60                # dollar-volume window (trading days)
ADV_MIN_DAYS = 50              # must have traded on this many days in the window
MIN_PRICE = 5.0                # minimum share price (USD)
MIN_ADV = 5_000_000            # minimum average daily dollar volume (USD)

EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}   # no OTC
MASSIVE_TYPES = ["CS"]         # CS = common stock; use ["CS", "ADRC"] to include ADRs such as TSM
INCLUDE_DELISTED = True        # also download delisted common stocks (reduces survivorship bias)

DAILY_READY_HOUR_ET = 18       # a day's daily bar is considered complete after this hour (ET)
UPDATE_LOOKBACK_DAYS = 14      # extra calendar days fetched on incremental updates (return continuity)
MASTER_MAX_AGE_DAYS = 7        # refresh the security master every N days
REGULAR_HOURS_ONLY = True      # minute bars: regular session only
BATCH_DAILY = 100              # symbols per request
BATCH_MINUTE = 100

ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_TRADING_URL = os.environ.get("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets")
MASSIVE_URL = "https://api.massive.com"
ALPACA_REQ_PER_MIN = 180       # free tier is ~200/min, keep some headroom
MASSIVE_REQ_PER_MIN = 5        # free tier is 5/min

ET = ZoneInfo("America/New_York")
DAILY_DIR = DATA_DIR / "daily"
MINUTE_DIR = DATA_DIR / "minute"
TMP_DIR = DATA_DIR / "_tmp"
MASTER_FILE = DATA_DIR / "master.parquet"
UNIVERSE_FILE = DATA_DIR / "universe.parquet"
SELECTION_FILE = DATA_DIR / "universe_selections.parquet"
BENCHMARK_FILE = DATA_DIR / "benchmark.parquet"
NO_DATA_FILE = DATA_DIR / "no_data_symbols.json"   # symbols that returned no bars (not retried)
BACKFILL_MARK = DATA_DIR / "backfill_done.json"
BENCHMARKS = ["SPY"]           # benchmark ETFs (never part of the universe)

BAR_COLS = ["ts", "symbol", "open", "high", "low", "close", "volume", "vwap", "trades"]
DAILY_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "vwap", "trades", "ret"]
SYMBOL_RE = r"^[A-Z]{1,5}(?:\.[A-Z])?$"
NON_COMMON_RE = (r"\b(?:ETF|ETN|Fund|iShares|SPDR|ProShares|Direxion|Warrants?|Rights?|Units?|"
                 r"Preferred|Depositary|Notes|Debentures|Leveraged|Inverse)\b")

log = logging.getLogger("data")


# ============================== HTTP ==============================
class _RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


_session = requests.Session()
_alpaca_limiter = _RateLimiter(ALPACA_REQ_PER_MIN)
_massive_limiter = _RateLimiter(MASSIVE_REQ_PER_MIN)


def _get_json(url: str, params: dict, headers: dict, limiter: _RateLimiter, retries: int = 6):
    for attempt in range(retries):
        limiter.wait()
        try:
            r = _session.get(url, params=params, headers=headers, timeout=60)
        except requests.RequestException as e:
            wait = min(60, 2 ** (attempt + 1))
            log.warning("Network error: %s, retrying in %d s", e, wait)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            wait = min(60, 2 ** (attempt + 1))
            log.warning("HTTP %d, retrying in %d s", r.status_code, wait)
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {url}\n{r.text[:500]}")
    raise RuntimeError(f"Still failing after {retries} retries: {url}")


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing environment variable {name} (put it in the .env file)")
    return v


def alpaca_get(url: str, params: dict | None = None):
    headers = {"APCA-API-KEY-ID": _env("ALPACA_API_KEY"),
               "APCA-API-SECRET-KEY": _env("ALPACA_SECRET_KEY")}
    return _get_json(url, dict(params or {}), headers, _alpaca_limiter)


def massive_get(url: str, params: dict | None = None):
    params = dict(params or {})
    params["apiKey"] = _env("MASSIVE_API_KEY")
    return _get_json(url, params, {}, _massive_limiter)


# ============================== Time helpers ==============================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def last_complete_date() -> pd.Timestamp:
    """Most recent trading day whose daily bar is complete (weekends skipped; holidays have no data)."""
    now = now_utc().astimezone(ET)
    d = now.date() if now.hour >= DAILY_READY_HOUR_ET else now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return pd.Timestamp(d)


def _et(d: pd.Timestamp, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET)


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ============================== Security master ==============================
def _massive_common_stocks(active: bool) -> pd.DataFrame:
    rows = []
    for t in MASSIVE_TYPES:
        url: str | None = f"{MASSIVE_URL}/v3/reference/tickers"
        params = {"market": "stocks", "type": t, "active": str(active).lower(), "limit": 1000}
        while url:
            js = massive_get(url, params)
            rows += [{"symbol": r["ticker"], "type": r.get("type", t)} for r in js.get("results", [])]
            url, params = js.get("next_url"), {}
    return pd.DataFrame(rows, columns=["symbol", "type"]).drop_duplicates("symbol")


def _alpaca_assets(status: str) -> pd.DataFrame:
    a = pd.DataFrame(alpaca_get(f"{ALPACA_TRADING_URL}/v2/assets",
                                {"status": status, "asset_class": "us_equity"}))
    if a.empty:
        return pd.DataFrame(columns=["symbol", "name", "exchange"])
    if status == "active":
        a = a[a["tradable"].astype(bool)]
    a = a[a["exchange"].isin(EXCHANGES) & a["symbol"].str.match(SYMBOL_RE)]
    return a[["symbol", "name", "exchange"]].drop_duplicates("symbol")


def update_master(force: bool = False) -> pd.DataFrame:
    if MASTER_FILE.exists() and not force:
        age_days = (time.time() - MASTER_FILE.stat().st_mtime) / 86400
        if age_days < MASTER_MAX_AGE_DAYS:
            return pd.read_parquet(MASTER_FILE)

    log.info("Refreshing the security master")
    frames = [_alpaca_assets("active").assign(active=True)]
    if INCLUDE_DELISTED:
        frames.append(_alpaca_assets("inactive").assign(active=False))
    a = pd.concat(frames, ignore_index=True).drop_duplicates("symbol", keep="first")
    log.info("Alpaca listed securities: %d active, %d inactive",
             int(a["active"].sum()), int((~a["active"]).sum()))

    if os.environ.get("MASSIVE_API_KEY"):
        cs = _massive_common_stocks(True)
        if INCLUDE_DELISTED:
            cs = pd.concat([cs, _massive_common_stocks(False)]).drop_duplicates("symbol")
        master = a.merge(cs, on="symbol", how="inner")
        log.info("Massive %s tickers: %d; matched with Alpaca: %d",
                 "/".join(MASSIVE_TYPES), len(cs), len(master))
    else:
        log.warning("MASSIVE_API_KEY not set: using a name rule to drop ETFs, warrants, preferreds etc.")
        mask = a["name"].fillna("").str.contains(NON_COMMON_RE, case=False, regex=True)
        master = a[~mask].assign(type="CS(est)")
        log.info("Kept by the name rule: %d", len(master))

    master = master.sort_values("symbol").reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_FILE, index=False)
    return master


# ============================== Bar download ==============================
def fetch_bars(symbols: list[str], timeframe: str, start: datetime, end: datetime,
               adjustment: str) -> pd.DataFrame:
    """Bars for several symbols from Alpaca. `ts` is US/Eastern time (tz-naive)."""
    end = min(end, now_utc() - timedelta(minutes=16))   # free tier cannot query the last 15 minutes of SIP
    if end <= start or not symbols:
        return _empty(BAR_COLS)
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "start": _rfc3339(start), "end": _rfc3339(end),
              "adjustment": adjustment, "feed": "sip", "limit": 10000, "sort": "asc"}
    frames = []
    while True:
        js = alpaca_get(f"{ALPACA_DATA_URL}/v2/stocks/bars", params)
        for sym, bars in (js.get("bars") or {}).items():
            if bars:
                frames.append(pd.DataFrame(bars).assign(symbol=sym))
        token = js.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not frames:
        return _empty(BAR_COLS)
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "t": "ts", "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vw": "vwap", "n": "trades"})
    for c in BAR_COLS:
        if c not in df:
            df[c] = np.nan
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601").dt.tz_convert(ET).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume", "vwap", "trades"]:
        df[c] = df[c].astype("float64")
    return df[BAR_COLS]


def _empty(cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    for c in ("ts", "date"):
        if c in df:
            df[c] = pd.Series(dtype="datetime64[ns]")
    if "symbol" in df:
        df["symbol"] = pd.Series(dtype="object")
    return df


def _daily_bars(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily bars for [start, end]: raw prices + total return `ret` computed from adjusted prices."""
    t0, t1 = _et(start), _et(end + pd.Timedelta(days=1))
    raw = fetch_bars(symbols, "1Day", t0, t1, "raw")
    if raw.empty:
        return _empty(DAILY_COLS)
    adj = fetch_bars(symbols, "1Day", t0, t1, "all")
    raw["date"] = raw["ts"].dt.normalize()
    adj["date"] = adj["ts"].dt.normalize()
    adj = adj.sort_values(["symbol", "date"])
    # Adjusted prices from one request share one set of adjustment factors,
    # so the ratio of consecutive days is the true total return.
    adj["ret"] = adj.groupby("symbol")["close"].pct_change(fill_method=None)
    df = raw.merge(adj[["symbol", "date", "ret"]], on=["symbol", "date"], how="left")
    df = df[df["date"] <= end]
    return df[DAILY_COLS].sort_values(["date", "symbol"]).reset_index(drop=True)


def _download_history(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
                      tag: str) -> tuple[pd.DataFrame, Path]:
    """Download full history in batches. Each batch is saved to a temp file so an interrupted
    run resumes where it stopped."""
    tmp = TMP_DIR / tag
    key = {"symbols": symbols, "start": str(start.date()), "end": str(end.date())}
    manifest = tmp / "manifest.json"
    if tmp.exists():
        try:
            same = json.loads(manifest.read_text()) == key
        except (OSError, ValueError):
            same = False
        if not same:
            shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(key))

    batches = list(_chunks(symbols, BATCH_DAILY))
    parts = []
    t_start = time.time()
    for i, batch in enumerate(batches, 1):
        f = tmp / f"part_{i:05d}.parquet"
        if not f.exists():
            _daily_bars(batch, start, end).to_parquet(f, index=False)
        parts.append(f)
        if i % 5 == 0 or i == len(batches):
            log.info("Daily bars: batch %d/%d, %.0f s elapsed", i, len(batches), time.time() - t_start)
    dfs = [d for d in (pd.read_parquet(f) for f in parts) if len(d)]
    return (pd.concat(dfs, ignore_index=True) if dfs else _empty(DAILY_COLS)), tmp


# ============================== Daily storage ==============================
def _daily_files() -> list[Path]:
    return sorted(DAILY_DIR.glob("*.parquet"))


def _write_daily(df: pd.DataFrame) -> None:
    if df.empty:
        return
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    for month, g in df.groupby(df["date"].dt.to_period("M")):
        f = DAILY_DIR / f"{month}.parquet"
        if f.exists():
            g = pd.concat([pd.read_parquet(f), g], ignore_index=True)
            g = g.drop_duplicates(["date", "symbol"], keep="last")
        g.sort_values(["date", "symbol"]).to_parquet(f, index=False)


def _daily_state() -> tuple[set[str], pd.Timestamp | None, pd.Timestamp | None]:
    """(symbols stored, first date, last date)."""
    files = _daily_files()
    if not files:
        return set(), None, None
    symbols: set[str] = set()
    for f in files:
        symbols.update(pd.read_parquet(f, columns=["symbol"])["symbol"].unique())
    first = pd.read_parquet(files[0], columns=["date"])["date"].min()
    last = pd.read_parquet(files[-1], columns=["date"])["date"].max()
    return symbols, pd.Timestamp(first), pd.Timestamp(last)


def _no_data_symbols() -> set[str]:
    try:
        return set(json.loads(NO_DATA_FILE.read_text()))
    except (OSError, ValueError):
        return set()


def _remember_no_data(requested: list[str], df: pd.DataFrame) -> None:
    got = set(df["symbol"].unique()) if len(df) else set()
    missing = _no_data_symbols() | {s for s in requested if s not in got}
    NO_DATA_FILE.write_text(json.dumps(sorted(missing)))


def _backfill(symbols: list[str], first: pd.Timestamp) -> None:
    """Download [DAILY_START, first] once for an install whose history starts later than DAILY_START.
    The overlap day `first` is rewritten so its `ret` (previously missing) gets filled in."""
    start = pd.Timestamp(DAILY_START)
    if BACKFILL_MARK.exists() or first <= start + pd.Timedelta(days=7):
        return
    log.info("Back-filling daily history %s to %s for %d symbols (one-off, may take a while)",
             start.date(), first.date(), len(symbols))
    df, tmp = _download_history(symbols, start, first, "daily_backfill")
    _write_daily(df)
    shutil.rmtree(tmp, ignore_errors=True)
    BACKFILL_MARK.write_text(json.dumps({"start": DAILY_START, "until": str(first.date())}))
    log.info("Back-fill done: %d rows", len(df))


def update_daily(master: pd.DataFrame) -> None:
    end = last_complete_date()
    is_active = master["active"].astype(bool) if "active" in master else pd.Series(True, index=master.index)
    active = sorted(master.loc[is_active, "symbol"])
    everyone = sorted(master["symbol"])
    stored, first, last = _daily_state()
    skip = _no_data_symbols()

    if last is None:
        log.info("First daily download: %d symbols, %s to %s (several minutes to an hour)",
                 len(everyone), DAILY_START, end.date())
        df, tmp = _download_history(everyone, pd.Timestamp(DAILY_START), end, "daily_full")
        _write_daily(df)
        _remember_no_data(everyone, df)
        BACKFILL_MARK.write_text(json.dumps({"start": DAILY_START, "until": None}))
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("Daily bars written: %d rows", len(df))
        return

    _backfill(sorted(stored), first)

    new_syms = [s for s in everyone if s not in stored and s not in skip]
    if new_syms:
        log.info("%d symbols without history (new listings or delisted names): downloading", len(new_syms))
        df, tmp = _download_history(new_syms, pd.Timestamp(DAILY_START), end, "daily_new")
        _write_daily(df)
        _remember_no_data(new_syms, df)
        shutil.rmtree(tmp, ignore_errors=True)

    if last >= end:
        log.info("Daily bars are up to date (%s)", last.date())
        return

    old_syms = [s for s in active if s in stored]
    start = last - pd.Timedelta(days=UPDATE_LOOKBACK_DAYS)
    log.info("Incremental daily update: after %s to %s, %d symbols", last.date(), end.date(), len(old_syms))
    frames = [_daily_bars(b, start, end) for b in _chunks(old_syms, BATCH_DAILY)]
    frames = [f for f in frames if len(f)]
    if not frames:
        log.warning("No new daily bars (holiday?)")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] > last]
    _write_daily(df)
    log.info("Added %d rows, latest date %s", len(df), df["date"].max().date() if len(df) else last.date())


def update_benchmark() -> None:
    """Only one or two ETFs: re-download the whole series every time (a few seconds)."""
    end = last_complete_date()
    df = _daily_bars(BENCHMARKS, pd.Timestamp(DAILY_START), end)
    if df.empty:
        log.warning("No benchmark data: %s", BENCHMARKS)
        return
    df.to_parquet(BENCHMARK_FILE, index=False)
    log.info("Benchmark %s updated to %s", "/".join(BENCHMARKS), df["date"].max().date())


def load_benchmark(symbol: str = "SPY") -> pd.DataFrame:
    """Benchmark daily long table with adj_open / adj_close etc."""
    df = pd.read_parquet(BENCHMARK_FILE)
    return add_adjusted(df[df["symbol"] == symbol])


# ============================== Universe ==============================
def build_universe(master: pd.DataFrame) -> None:
    files = _daily_files()
    if not files:
        return
    d = pd.concat([pd.read_parquet(f, columns=["date", "symbol", "close", "volume"]) for f in files],
                  ignore_index=True)
    d = d[d["symbol"].isin(set(master["symbol"]))]
    d["dv"] = (d["close"] * d["volume"]).astype("float64")     # dollar volume of the day
    d = d.sort_values("date").reset_index(drop=True)

    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    months = dates.to_period("M")
    month_end = pd.Series(dates, index=dates).groupby(months).max()
    periods = list(month_end.index)
    date_vals = d["date"].to_numpy()

    sels, unis = [], []
    for prev, cur in zip(periods[:-1], periods[1:]):
        r = month_end[prev]                                 # select after the close of last month's final day
        i_end = dates.get_loc(r)
        if i_end + 1 < ADV_MIN_DAYS:
            continue
        lo = np.searchsorted(date_vals, dates[max(0, i_end - ADV_WINDOW + 1)].to_datetime64(), "left")
        hi = np.searchsorted(date_vals, r.to_datetime64(), "right")
        w = d.iloc[lo:hi]                                   # the last ADV_WINDOW trading days
        g = w.groupby("symbol")
        stats = pd.DataFrame({"adv": g["dv"].mean(), "n": g["dv"].count()})
        last_close = w[w["date"] == r].set_index("symbol")["close"]
        stats["close"] = last_close.reindex(stats.index)
        ok = (stats["close"] >= MIN_PRICE) & (stats["n"] >= ADV_MIN_DAYS) & (stats["adv"] >= MIN_ADV)
        top = stats.loc[ok, "adv"].sort_values(ascending=False).head(UNIVERSE_SIZE)
        if top.empty:
            continue
        sels.append(pd.DataFrame({"rebalance_date": r, "symbol": top.index.values,
                                  "adv": top.values, "rank": np.arange(1, len(top) + 1)}))
        days = dates[months == cur]                         # used on every day of the month
        unis.append(pd.DataFrame({"date": np.repeat(days.values, len(top)),
                                  "symbol": np.tile(top.index.values, len(days))}))

    if not sels:
        log.warning("Not enough history yet (need at least %d trading days) to build a universe", ADV_MIN_DAYS)
        return
    sel = pd.concat(sels, ignore_index=True)
    uni = pd.concat(unis, ignore_index=True)
    sel.to_parquet(SELECTION_FILE, index=False)
    uni.to_parquet(UNIVERSE_FILE, index=False)
    latest = sel[sel["rebalance_date"] == sel["rebalance_date"].max()]
    log.info("Universe updated: %d monthly rebalances, latest %s with %d stocks (smallest ADV $%.1fM)",
             sel["rebalance_date"].nunique(), latest["rebalance_date"].iloc[0].date(),
             len(latest), latest["adv"].min() / 1e6)


# ============================== 1-minute bars ==============================
def update_minute() -> None:
    if not UNIVERSE_FILE.exists():
        raise SystemExit("No universe yet: run python data.py update first")
    uni = pd.read_parquet(UNIVERSE_FILE)
    end = last_complete_date()
    uni = uni[(uni["date"] >= pd.Timestamp(MINUTE_START)) & (uni["date"] <= end)]
    by_day = uni.groupby("date")["symbol"].apply(list)

    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in MINUTE_DIR.glob("*.parquet")}
    todo = [pd.Timestamp(d) for d in by_day.index if pd.Timestamp(d).strftime("%Y-%m-%d") not in done]
    if not todo:
        log.info("Minute bars are up to date")
        return

    n_req = sum(math.ceil(len(by_day[d]) / BATCH_MINUTE) * 4 for d in todo)
    log.info("%d days of minute bars to download, about %d requests / %.1f hours "
             "(safe to interrupt, resumes next time)", len(todo), n_req, n_req / ALPACA_REQ_PER_MIN / 60)

    h0, h1 = ((9, 30), (16, 0)) if REGULAR_HOURS_ONLY else ((4, 0), (20, 0))
    t_start = time.time()
    for i, d in enumerate(todo, 1):
        syms = sorted(by_day[d])
        frames = [fetch_bars(b, "1Min", _et(d, *h0), _et(d, *h1), "raw")
                  for b in _chunks(syms, BATCH_MINUTE)]
        frames = [f for f in frames if len(f)]
        if not frames:
            log.warning("%s has no minute bars, skipped (retried next run)", d.date())
            continue
        df = pd.concat(frames, ignore_index=True)
        if REGULAR_HOURS_ONLY:
            t = df["ts"].dt.time
            df = df[(t >= dtime(9, 30)) & (t < dtime(16, 0))]
        df.sort_values(["symbol", "ts"]).to_parquet(MINUTE_DIR / f"{d:%Y-%m-%d}.parquet", index=False)
        if i % 5 == 0 or i == len(todo):
            el = time.time() - t_start
            log.info("Minute bars: %d/%d days, %.0f min elapsed, about %.0f min left",
                     i, len(todo), el / 60, el / i * (len(todo) - i) / 60)


# ============================== Readers ==============================
def load_daily(start=None, end=None, symbols=None, columns=None, adjusted: bool = False) -> pd.DataFrame:
    """Daily long table. adjusted=True adds adj_open/adj_high/adj_low/adj_close (compounded from ret)."""
    files = _daily_files()
    if start is not None:
        files = [f for f in files if f.stem >= pd.Timestamp(start).strftime("%Y-%m")]
    if end is not None:
        files = [f for f in files if f.stem <= pd.Timestamp(end).strftime("%Y-%m")]
    if not files:
        return _empty(DAILY_COLS)
    cols = None
    if columns is not None:
        need = ["date", "symbol", *columns]
        if adjusted:
            need += ["open", "high", "low", "close", "ret"]
        cols = list(dict.fromkeys(need))
    sym_set = set(symbols) if symbols is not None else None
    parts = []
    for f in files:
        x = pd.read_parquet(f, columns=cols)
        if sym_set is not None:
            x = x[x["symbol"].isin(sym_set)]
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return add_adjusted(df) if adjusted else df


def add_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Compound `ret` into adjusted prices; the last adjusted close equals the last raw close."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    r = df["ret"].where(df["ret"].notna(), g["close"].pct_change(fill_method=None))
    r = r.where(g.cumcount() > 0, 0.0).fillna(0.0)        # each symbol's first day is the base
    level = (1.0 + r).groupby(df["symbol"], sort=False).cumprod()
    scale = g["close"].transform("last") / level.groupby(df["symbol"], sort=False).transform("last")
    df["adj_close"] = level * scale
    k = df["adj_close"] / df["close"]
    for c in ("open", "high", "low"):
        if c in df:
            df[f"adj_{c}"] = df[c] * k
    return df


def load_universe(start=None, end=None) -> pd.DataFrame:
    """Daily universe membership as a (date, symbol) long table."""
    uni = pd.read_parquet(UNIVERSE_FILE)
    if start is not None:
        uni = uni[uni["date"] >= pd.Timestamp(start)]
    if end is not None:
        uni = uni[uni["date"] <= pd.Timestamp(end)]
    return uni.reset_index(drop=True)


def current_universe() -> list[str]:
    """Universe from the latest monthly selection (used for tomorrow's signals)."""
    sel = pd.read_parquet(SELECTION_FILE)
    latest = sel[sel["rebalance_date"] == sel["rebalance_date"].max()]
    return latest.sort_values("rank")["symbol"].tolist()


def load_minute(start, end=None, symbols=None) -> pd.DataFrame:
    """1-minute bars for [start, end]; pass only `start` to read a single day."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    files = [f for f in sorted(MINUTE_DIR.glob("*.parquet"))
             if start.strftime("%Y-%m-%d") <= f.stem <= end.strftime("%Y-%m-%d")]
    if not files:
        return _empty(BAR_COLS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]
    return df.reset_index(drop=True)


def panel(field: str, start=None, end=None, universe_only: bool = True) -> pd.DataFrame:
    """Wide date x symbol table. field can be close / volume / ret / adj_close etc.
    With universe_only=True, cells outside that day's universe are NaN."""
    adjusted = field.startswith("adj_")
    base = field[4:] if adjusted else field
    df = load_daily(start, end, columns=[base], adjusted=adjusted)
    if universe_only:
        df = df.merge(load_universe(start, end), on=["date", "symbol"], how="inner")
    return df.pivot(index="date", columns="symbol", values=field).sort_index()


# ============================== Status ==============================
def status() -> None:
    print(f"Data directory: {DATA_DIR}")
    if MASTER_FILE.exists():
        m = pd.read_parquet(MASTER_FILE)
        age = (time.time() - MASTER_FILE.stat().st_mtime) / 86400
        n_act = int(m["active"].sum()) if "active" in m else len(m)
        print(f"Security master: {len(m)} symbols ({n_act} active), refreshed {age:.1f} days ago")
    files = _daily_files()
    if files:
        syms, first, last = _daily_state()
        print(f"Daily bars: {first.date()} to {last.date()}, {len(syms)} symbols, {len(files)} monthly files")
    if SELECTION_FILE.exists():
        sel = pd.read_parquet(SELECTION_FILE)
        r = sel["rebalance_date"].max()
        print(f"Universe: latest selection {r.date()}, {(sel['rebalance_date'] == r).sum()} stocks; "
              f"{sel['rebalance_date'].nunique()} monthly selections")
    mfiles = sorted(MINUTE_DIR.glob("*.parquet"))
    if mfiles:
        print(f"Minute bars: {mfiles[0].stem} to {mfiles[-1].stem}, {len(mfiles)} trading days")
    size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()) if DATA_DIR.exists() else 0
    print(f"Disk usage: {size / 1e9:.2f} GB")


def main() -> None:
    p = argparse.ArgumentParser(description="US equity data pipeline (free Alpaca data)")
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update", help="download / update data (the first run downloads everything)")
    u.add_argument("--minute", action="store_true", help="also download / update 1-minute bars")
    u.add_argument("--refresh-master", action="store_true", help="force a refresh of the security master")
    sub.add_parser("status", help="summary of local data")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "update":
        master = update_master(force=args.refresh_master)
        update_daily(master)
        update_benchmark()
        build_universe(master)
        if args.minute:
            update_minute()
        log.info("All done")
    else:
        status()


if __name__ == "__main__":
    main()
