#!/usr/bin/env python3
"""
fundamentals.py - quarterly financial statements from SEC EDGAR (free), point-in-time

What it does
  1. Maps tickers to SEC CIK numbers (sec.gov/files/company_tickers.json) and gets each
     company's SIC code, which is turned into a coarse sector (used for sector neutralisation
     and the per-sector position cap).
  2. Downloads each company's XBRL "company facts" and extracts a small set of items:
       flows (quarterly and trailing-twelve-month): net income, revenue, gross profit,
                                                    operating income, operating cash flow
       stocks (balance sheet, as of period end):    total assets, shareholders' equity, shares outstanding
     plus a few derived items (SUE, year-on-year growth of assets and sales).
  3. Every value carries the date it was FILED with the SEC. Factors only use a value from its
     filing date onwards, so there is no look-ahead. When the same period is reported again
     later (comparatives, amendments) the first filing is kept.

Usage
  python fundamentals.py update      # first run downloads ~2-3k companies (about 15-20 minutes);
                                     # later runs only refresh companies older than REFRESH_DAYS
  python fundamentals.py status

Requires SEC_USER_AGENT in .env, e.g.  SEC_USER_AGENT=Your Name your@email.com
(the SEC asks every client to identify itself; falls back to EMAIL_SENDER).

Reading from code
  import fundamentals as FU
  sectors = FU.load_sectors()                       # symbol -> sector code
  panel = FU.pit_panel("ni_ttm", dates, symbols)    # date x symbol, latest value known on each date

Known limits
  - company_tickers.json only lists current registrants, so delisted companies usually have
    no fundamentals (their price factors still work).
  - XBRL tags differ between companies; a few companies will have gaps.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import data as D

FUND_DIR = D.DATA_DIR / "fundamentals"
CIK_DIR = FUND_DIR / "cik"
COMPANY_FILE = FUND_DIR / "companies.parquet"
FACTS_FILE = FUND_DIR / "facts.parquet"

REFRESH_DAYS = 7             # re-download a company's facts after this many days
SIC_REFRESH_DAYS = 90        # SIC codes rarely change
SEC_REQ_PER_SEC = 8          # SEC allows up to 10 requests per second
MAX_AGE_DAYS = 400           # a value is considered stale this many days after its period end

FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT"}

# item -> candidate XBRL tags in priority order (taxonomy, tag)
FLOW_TAGS = {
    "ni": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"),
           ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic")],
    "rev": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"), ("us-gaap", "SalesRevenueNet")],
    "gp": [("us-gaap", "GrossProfit")],
    "cogs": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold"),
             ("us-gaap", "CostOfGoodsSold")],
    "opinc": [("us-gaap", "OperatingIncomeLoss")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
}
STOCK_TAGS = {
    "assets": [("us-gaap", "Assets")],
    "equity": [("us-gaap", "StockholdersEquity"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
}

SECTORS = ["Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples",
           "Health Care", "Financials", "Real Estate", "Technology", "Communication", "Utilities", "Other"]

log = logging.getLogger("fundamentals")


# ============================== HTTP ==============================
_session = requests.Session()
_last_call = [0.0]


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT") or os.environ.get("EMAIL_SENDER")
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT in .env, e.g.  SEC_USER_AGENT=Your Name your@email.com")
    return ua if " " in ua else f"quant-research {ua}"


def sec_get(url: str, retries: int = 5):
    for attempt in range(retries):
        delay = _last_call[0] + 1.0 / SEC_REQ_PER_SEC - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last_call[0] = time.monotonic()
        try:
            r = _session.get(url, headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"},
                             timeout=60)
        except requests.RequestException as e:
            time.sleep(min(30, 2 ** (attempt + 1)))
            log.warning("Network error %s, retrying", e)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            wait = min(60, 5 * 2 ** attempt)
            log.warning("SEC HTTP %d, waiting %d s", r.status_code, wait)
            time.sleep(wait)
            continue
        raise RuntimeError(f"SEC HTTP {r.status_code}: {url}")
    log.warning("Giving up on %s", url)
    return None


# ============================== Sectors ==============================
def sic_to_sector(sic) -> int:
    """Coarse sector code (index into SECTORS) from a 4-digit SIC code."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return SECTORS.index("Other")
    d2, d3 = s // 100, s // 10
    name = "Other"
    if d2 <= 9:
        name = "Consumer Staples"
    elif d2 == 13 or d2 == 29:
        name = "Energy"
    elif d2 in (10, 12, 14):
        name = "Materials"
    elif 15 <= d2 <= 17:
        name = "Industrials"
    elif d2 in (20, 21):
        name = "Consumer Staples"
    elif d2 in (22, 23, 25, 31, 39):
        name = "Consumer Discretionary"
    elif d2 in (24, 26, 30, 32, 33):
        name = "Materials"
    elif d2 == 27:
        name = "Communication"
    elif d2 == 28:
        name = "Health Care" if d3 == 283 else "Materials"
    elif d2 == 34:
        name = "Industrials"
    elif d2 == 35:
        name = "Technology" if d3 == 357 else "Industrials"
    elif d2 == 36:
        name = "Technology"
    elif d2 == 37:
        name = "Consumer Discretionary" if d3 == 371 else "Industrials"
    elif d2 == 38:
        name = "Health Care" if d3 in (384, 385) else "Technology"
    elif 40 <= d2 <= 47:
        name = "Industrials"
    elif d2 == 48:
        name = "Communication"
    elif d2 == 49:
        name = "Utilities"
    elif d2 in (50, 51):
        name = "Health Care" if d3 == 512 else "Industrials"
    elif d2 == 54:
        name = "Consumer Staples"
    elif 52 <= d2 <= 59:
        name = "Consumer Discretionary"
    elif d2 == 65 or s == 6798:
        name = "Real Estate"
    elif 60 <= d2 <= 67:
        name = "Financials"
    elif d2 == 73:
        name = "Technology" if d3 == 737 else "Industrials"
    elif d2 == 80:
        name = "Health Care"
    elif d2 in (78, 79):
        name = "Communication"
    elif d2 == 70:
        name = "Consumer Discretionary"
    elif 70 <= d2 <= 89:
        name = "Industrials"
    return SECTORS.index(name)


# ============================== Companies ==============================
def update_companies(symbols: set[str]) -> pd.DataFrame:
    js = sec_get("https://www.sec.gov/files/company_tickers.json") or {}
    rows = [{"symbol": str(v["ticker"]).upper().replace("-", "."), "cik": int(v["cik_str"]),
             "title": v.get("title")} for v in js.values()]
    tick = pd.DataFrame(rows, columns=["symbol", "cik", "title"]).drop_duplicates("symbol")
    tick = tick[tick["symbol"].isin(symbols)]
    old = pd.read_parquet(COMPANY_FILE) if COMPANY_FILE.exists() else pd.DataFrame(
        columns=["symbol", "cik", "title", "sic", "sic_time"])
    comp = tick.merge(old[["cik", "sic", "sic_time"]].drop_duplicates("cik"), on="cik", how="left")
    now = time.time()
    need = comp[comp["sic"].isna() | (now - comp["sic_time"].fillna(0) > SIC_REFRESH_DAYS * 86400)]
    ciks = need["cik"].drop_duplicates().tolist()
    if ciks:
        log.info("Fetching SIC codes for %d companies", len(ciks))
    sic = {}
    for i, cik in enumerate(ciks, 1):
        sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json") or {}
        sic[cik] = sub.get("sic") or "0"
        if i % 200 == 0:
            log.info("SIC %d/%d", i, len(ciks))
    if sic:
        upd = comp["cik"].map(sic)
        comp.loc[upd.notna(), "sic"] = upd[upd.notna()]
        comp.loc[upd.notna(), "sic_time"] = now
    comp["sic"] = comp["sic"].astype(str)
    comp["sector"] = comp["sic"].map(sic_to_sector).astype("int16")
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_parquet(COMPANY_FILE, index=False)
    log.info("Companies with a CIK: %d of %d symbols", len(comp), len(symbols))
    return comp


def load_companies() -> pd.DataFrame:
    return pd.read_parquet(COMPANY_FILE) if COMPANY_FILE.exists() else pd.DataFrame(
        columns=["symbol", "cik", "title", "sic", "sector"])


def load_sectors() -> pd.Series:
    """symbol -> sector code (int). Empty if fundamentals were never downloaded."""
    c = load_companies()
    return c.set_index("symbol")["sector"].astype(int) if len(c) else pd.Series(dtype=int)


# ============================== Facts extraction ==============================
def _entries(facts: dict, tags: list[tuple[str, str]]) -> pd.DataFrame:
    """All 10-K/10-Q entries for the candidate tags; earlier tags win when periods overlap."""
    frames = []
    for prio, (tax, tag) in enumerate(tags):
        node = facts.get(tax, {}).get(tag)
        if not node:
            continue
        for unit, vals in node.get("units", {}).items():
            if unit not in ("USD", "shares"):
                continue
            df = pd.DataFrame(vals)
            if df.empty or "form" not in df:
                continue
            df = df[df["form"].isin(FORMS)]
            if df.empty:
                continue
            df["prio"] = prio
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "prio"])
    df = pd.concat(frames, ignore_index=True)
    if "start" not in df:
        df["start"] = np.nan
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "filed", "val"])
    # first filing of each period; among same-day filings prefer the higher-priority tag
    df = df.sort_values(["filed", "prio"]).drop_duplicates(["start", "end"], keep="first")
    return df[["start", "end", "val", "filed", "prio"]]


def _quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Discrete quarterly values (end, val, filed) from a flow concept, deriving Q4 = FY - Q1..Q3."""
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    dur = (df["end"] - df["start"]).dt.days
    q = df[(dur >= 75) & (dur <= 105)][["start", "end", "val", "filed"]]
    a = df[(dur >= 340) & (dur <= 390)]
    extra = []
    for _, r in a.iterrows():
        inside = q[(q["start"] >= r["start"] - pd.Timedelta(days=10)) & (q["end"] < r["end"] - pd.Timedelta(days=30))]
        inside = inside.drop_duplicates("end")
        if len(inside) == 3 and not (q["end"] == r["end"]).any():
            extra.append({"start": inside["end"].max() + pd.Timedelta(days=1), "end": r["end"],
                          "val": r["val"] - inside["val"].sum(), "filed": r["filed"]})
    if extra:
        q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True)
    q = q.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return q[["end", "val", "filed"]].reset_index(drop=True)


def _ttm(q: pd.DataFrame) -> pd.DataFrame:
    """Trailing four quarters, only when the four quarter ends span about one year."""
    if len(q) < 4:
        return pd.DataFrame(columns=["end", "val", "filed"])
    q = q.sort_values("end").reset_index(drop=True)
    span = (q["end"] - q["end"].shift(3)).dt.days
    val = q["val"].rolling(4).sum()
    filed = pd.Series([q["filed"].iloc[max(0, i - 3):i + 1].max() for i in range(len(q))])
    ok = span.between(250, 300)
    return pd.DataFrame({"end": q["end"], "val": val, "filed": filed})[ok].reset_index(drop=True)


def _instant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    x = df[df["start"].isna()] if df["start"].notna().any() else df
    x = x.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return x[["end", "val", "filed"]].reset_index(drop=True)


def _lagged(x: pd.DataFrame, lo: int = 330, hi: int = 400) -> pd.Series:
    """For every row, the value about one year earlier (period end between lo and hi days before)."""
    out = np.full(len(x), np.nan)
    ends, vals = x["end"].to_numpy(), x["val"].to_numpy(dtype=float)
    for i in range(len(x)):
        gap = (ends[i] - ends[:i]) / np.timedelta64(1, "D")
        j = np.flatnonzero((gap >= lo) & (gap <= hi))
        if len(j):
            out[i] = vals[j[-1]]
    return pd.Series(out, index=x.index)


def extract(facts_json: dict) -> pd.DataFrame:
    """Company facts JSON -> long table (item, end, filed, value)."""
    facts = facts_json.get("facts", {})
    out = []

    def add(item: str, x: pd.DataFrame) -> None:
        if len(x):
            out.append(pd.DataFrame({"item": item, "end": x["end"].values, "filed": x["filed"].values,
                                     "value": x["val"].astype(float).values}))

    quarters = {k: _quarterly(_entries(facts, tags)) for k, tags in FLOW_TAGS.items()}
    gp = quarters["gp"]
    if gp.empty and len(quarters["rev"]) and len(quarters["cogs"]):
        m = quarters["rev"].merge(quarters["cogs"], on="end", suffixes=("_r", "_c"))
        gp = pd.DataFrame({"end": m["end"], "val": m["val_r"] - m["val_c"],
                           "filed": m[["filed_r", "filed_c"]].max(axis=1)})
    quarters["gp"] = gp
    for k in ("ni", "rev", "gp", "opinc", "cfo"):
        add(f"{k}_ttm", _ttm(quarters[k]))
    ni = quarters["ni"]
    add("ni_q", ni)

    # SUE: year-on-year change in quarterly net income / std of the last 8 such changes
    if len(ni) >= 8:
        ni = ni.sort_values("end").reset_index(drop=True)
        d4 = ni["val"] - _lagged(ni, 330, 400)
        sd = d4.rolling(8, min_periods=4).std().shift(1)
        sue = (d4 / sd.where(sd > 0)).clip(-10, 10)
        add("sue", pd.DataFrame({"end": ni["end"], "val": sue, "filed": ni["filed"]}).dropna())

    rev_ttm = _ttm(quarters["rev"])
    if len(rev_ttm):
        g = rev_ttm["val"] / _lagged(rev_ttm).where(lambda v: v > 0) - 1
        add("sales_growth", pd.DataFrame({"end": rev_ttm["end"], "val": g, "filed": rev_ttm["filed"]}).dropna())

    for k, tags in STOCK_TAGS.items():
        x = _instant(_entries(facts, tags))
        add(k, x)
        if k == "assets" and len(x):
            g = x["val"] / _lagged(x).where(lambda v: v > 0) - 1
            add("asset_growth", pd.DataFrame({"end": x["end"], "val": g, "filed": x["filed"]}).dropna())
    if not out:
        return pd.DataFrame(columns=["item", "end", "filed", "value"])
    res = pd.concat(out, ignore_index=True)
    return res[np.isfinite(res["value"])]


# ============================== Update ==============================
def update(force: bool = False) -> None:
    if not D.MASTER_FILE.exists():
        raise SystemExit("No security master yet: run python data.py update first")
    symbols = set(pd.read_parquet(D.UNIVERSE_FILE)["symbol"].unique()) if D.UNIVERSE_FILE.exists() \
        else set(pd.read_parquet(D.MASTER_FILE)["symbol"])
    comp = update_companies(symbols)
    CIK_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    ciks = comp["cik"].drop_duplicates().astype(int).tolist()
    todo = [c for c in ciks if force or not (CIK_DIR / f"{c}.parquet").exists()
            or now - (CIK_DIR / f"{c}.parquet").stat().st_mtime > REFRESH_DAYS * 86400]
    log.info("Company facts to download: %d of %d", len(todo), len(ciks))
    t0 = time.time()
    for i, cik in enumerate(todo, 1):
        js = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        try:
            x = extract(js) if js else pd.DataFrame(columns=["item", "end", "filed", "value"])
        except Exception as e:                       # one odd filer should not stop the run
            log.warning("CIK %d: extraction failed (%s)", cik, e)
            x = pd.DataFrame(columns=["item", "end", "filed", "value"])
        x.assign(cik=cik).to_parquet(CIK_DIR / f"{cik}.parquet", index=False)
        if i % 100 == 0 or i == len(todo):
            el = time.time() - t0
            log.info("Facts %d/%d, %.0f s elapsed, about %.0f s left", i, len(todo), el, el / i * (len(todo) - i))
    parts = [pd.read_parquet(CIK_DIR / f"{c}.parquet") for c in ciks if (CIK_DIR / f"{c}.parquet").exists()]
    parts = [p for p in parts if len(p)]
    facts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["item", "end", "filed", "value", "cik"])
    facts.to_parquet(FACTS_FILE, index=False)
    log.info("Fundamentals saved: %d rows, %d companies", len(facts), facts["cik"].nunique())


def status() -> None:
    if not FACTS_FILE.exists():
        print("No fundamentals yet: run python fundamentals.py update")
        return
    f = pd.read_parquet(FACTS_FILE)
    c = load_companies()
    print(f"Companies: {len(c)} symbols, {f['cik'].nunique()} with facts; latest filing {f['filed'].max().date()}")
    print(f.groupby("item")["cik"].nunique().rename("companies").to_string())
    print("Sectors: " + ", ".join(f"{SECTORS[k]} {v}" for k, v in c["sector"].value_counts().sort_index().items()))


# ============================== Point-in-time panels ==============================
_FACTS_CACHE: dict = {}


def load_facts() -> pd.DataFrame:
    if not FACTS_FILE.exists():
        return pd.DataFrame(columns=["item", "end", "filed", "value", "cik"])
    key = FACTS_FILE.stat().st_mtime
    if _FACTS_CACHE.get("key") != key:
        _FACTS_CACHE.update(key=key, df=pd.read_parquet(FACTS_FILE))
    return _FACTS_CACHE["df"]


def pit_panel(item: str, dates: pd.DatetimeIndex, symbols: pd.Index, max_age_days: int = MAX_AGE_DAYS,
              with_filed: bool = False):
    """date x symbol panel of the most recent value of `item` that had been filed on or before each date.
    Values whose period ended more than max_age_days before the date are treated as missing.
    with_filed=True also returns the panel of filing dates (as day numbers) for the values used."""
    facts = load_facts()
    comp = load_companies()
    empty = pd.DataFrame(np.nan, index=dates, columns=symbols)
    if facts.empty or comp.empty:
        return (empty, empty.copy()) if with_filed else empty
    f = facts[facts["item"] == item]
    f = f.merge(comp[["symbol", "cik"]], on="cik")
    f = f[f["symbol"].isin(symbols)]
    if f.empty:
        return (empty, empty.copy()) if with_filed else empty
    # Row i on date t: the newest period end among everything filed so far (ignore late
    # re-filings of older periods).
    f = f.sort_values(["symbol", "filed", "end"])
    newest = f.groupby("symbol")["end"].cummax()
    f = f[f["end"] >= newest]
    # first trading date on or after the filing date
    pos = dates.searchsorted(f["filed"].values, side="left")
    f = f.assign(pos=pos)
    f = f[f["pos"] < len(dates)]
    f = f.drop_duplicates(["symbol", "pos"], keep="last")
    sym_idx = symbols.get_indexer(f["symbol"])
    val = np.full((len(dates), len(symbols)), np.nan)
    end = np.full((len(dates), len(symbols)), np.nan)
    fil = np.full((len(dates), len(symbols)), np.nan)
    val[f["pos"].values, sym_idx] = f["value"].values
    end[f["pos"].values, sym_idx] = f["end"].values.astype("datetime64[D]").astype(float)
    fil[f["pos"].values, sym_idx] = f["filed"].values.astype("datetime64[D]").astype(float)
    V = pd.DataFrame(val, index=dates, columns=symbols).ffill()
    E_ = pd.DataFrame(end, index=dates, columns=symbols).ffill()
    today = dates.values.astype("datetime64[D]").astype(float)[:, None]
    V = V.where(today - E_.to_numpy() <= max_age_days)
    if with_filed:
        Fd = pd.DataFrame(fil, index=dates, columns=symbols).ffill().where(V.notna())
        return V, Fd
    return V


def main() -> None:
    p = argparse.ArgumentParser(description="SEC EDGAR fundamentals")
    p.add_argument("cmd", choices=["update", "status"])
    p.add_argument("--force", action="store_true", help="re-download every company")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "update":
        update(args.force)
    else:
        status()


if __name__ == "__main__":
    main()
