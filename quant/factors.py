#!/usr/bin/env python3
"""
factors.py - factor library (about 50 factors)

Three groups
  price    from adjusted daily bars: reversal, momentum (plain / residual / intermediate / 52-week high),
           trend, volatility, beta, downside beta, skewness, lottery (MAX), liquidity, volume,
           overnight vs intraday returns, close vs VWAP, sector-relative return, sector momentum
  minute   from 1-minute bars, compressed to one row of intraday features per stock per day
           (realised vol / skew, first and last 30 minutes, volume profile), then rolled into daily factors
  fund     from SEC filings (fundamentals.py), point-in-time by filing date: value (E/P, B/M, S/P, CF/P),
           quality (gross profitability, ROE, operating margin, accruals), investment (asset growth),
           growth, net share issuance, earnings surprise (SUE) and earnings-announcement return (EAR)

Rules
  - The value on day t only uses information available at the close of day t, so it can be traded at t+1.
  - The factor table only keeps stocks that are in that day's universe. Values are raw;
    standardisation / neutralisation happens when they are used (preprocess()).
  - The "literature sign" in FACTOR_INFO is a reference only; evaluate.py measures the real sign.
  - Factors are computed one at a time and written to the long table immediately to keep memory low.

Usage
  python factors.py              # all factors (minute factors too if minute bars exist)
  python factors.py --no-minute  # skip minute factors

From code
  from factors import load_factors, preprocess, FACTOR_INFO, factor_columns
  fac = load_factors(start="2023-01-01", names=["ret_5", "bm"])     # (date, symbol, sector, factors...)
  z   = preprocess(fac, ["ret_5", "bm"], neutralize=["sector", "log_adv_60"])
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Iterator

import numpy as np
import pandas as pd

import data as D
import fundamentals as FU

FACTOR_DIR = D.DATA_DIR / "factors"
FACTOR_FILE = FACTOR_DIR / "factors.parquet"
MINUTE_FEAT_DIR = D.DATA_DIR / "minute_features"
META_COLS = ("date", "symbol", "sector")

# name: (description, literature sign; "?" = no consensus, group)
FACTOR_INFO: dict[str, tuple[str, str, str]] = {
    # ---- price: reversal / momentum ----
    "ret_1":          ("1-day return (daily reversal)", "-", "price"),
    "ret_5":          ("5-day return (weekly reversal)", "-", "price"),
    "resid_ret_5":    ("5-day market-adjusted residual return (idiosyncratic reversal)", "-", "price"),
    "ret_21":         ("21-day return (monthly reversal)", "-", "price"),
    "ret_63":         ("63-day return", "?", "price"),
    "mom_6_1":        ("6-month return skipping the last month", "+", "price"),
    "mom_12_1":       ("12-month return skipping the last month (momentum)", "+", "price"),
    "mom_12_7":       ("return from month -12 to -7 (intermediate momentum)", "+", "price"),
    "resid_mom_12_1": ("12-1 month residual momentum, scaled by residual vol", "+", "price"),
    "high_52w":       ("close / 52-week high", "+", "price"),
    "trend_200":      ("close / 200-day moving average - 1", "+", "price"),
    "sector_rel_21":  ("21-day return minus the sector average", "-", "price"),
    "sector_mom_6":   ("sector average 6-1 month momentum (industry momentum)", "+", "price"),
    # ---- price: risk ----
    "vol_20":         ("20-day return volatility (low-vol anomaly)", "-", "price"),
    "vol_60":         ("60-day return volatility", "-", "price"),
    "beta_60":        ("60-day beta to the equal-weighted universe", "-", "price"),
    "idio_vol_60":    ("60-day idiosyncratic volatility", "-", "price"),
    "downside_beta_120": ("120-day beta on down-market days", "?", "price"),
    "skew_60":        ("60-day daily-return skewness", "-", "price"),
    "max_21":         ("largest daily return in 21 days (lottery)", "-", "price"),
    "hl_range_20":    ("20-day average log(high/low)", "-", "price"),
    # ---- price: liquidity / volume ----
    "amihud_20":      ("log Amihud illiquidity, |ret| / dollar volume, 20-day mean", "+", "price"),
    "log_adv_60":     ("log 60-day average dollar volume (size proxy, used for neutralisation)", "-", "price"),
    "abn_vol_5_60":   ("log 5-day / 60-day average dollar volume (abnormal volume)", "?", "price"),
    "turn_cv_60":     ("60-day coefficient of variation of dollar volume", "-", "price"),
    "pv_corr_20":     ("20-day corr of return and log volume change", "?", "price"),
    "overnight_20":   ("20-day cumulative overnight return (prev close to open)", "?", "price"),
    "intraday_20":    ("20-day cumulative intraday return (open to close)", "?", "price"),
    "close_vwap_5":   ("5-day mean log(close / VWAP)", "?", "price"),
    # ---- minute ----
    "rskew_20":       ("20-day mean intraday realised skewness (1-minute returns)", "-", "minute"),
    "rv_ratio_5_20":  ("log 5-day / 20-day mean intraday realised vol", "?", "minute"),
    "open30_ret_5":   ("5-day cumulative first-30-minute return", "?", "minute"),
    "last30_ret_5":   ("5-day cumulative last-30-minute return", "?", "minute"),
    "last30_vshare_20": ("20-day mean last-30-minute volume share", "?", "minute"),
    "vwap_dev_5":     ("5-day mean log(close / intraday VWAP), minute-exact", "?", "minute"),
    "vprofile_dev_5": ("5-day mean distance of the half-hour volume profile from its 20-day norm", "?", "minute"),
    # ---- fundamentals (SEC, point-in-time) ----
    "log_mcap":       ("log market cap (price x shares outstanding)", "-", "fund"),
    "ep":             ("earnings yield: TTM net income / market cap", "+", "fund"),
    "bm":             ("book-to-market: equity / market cap", "+", "fund"),
    "sp":             ("sales-to-price: TTM revenue / market cap", "+", "fund"),
    "cfp":            ("cash-flow yield: TTM operating cash flow / market cap", "+", "fund"),
    "gpa":            ("gross profitability: TTM gross profit / assets", "+", "fund"),
    "roe":            ("TTM net income / equity", "+", "fund"),
    "op_margin":      ("TTM operating income / revenue", "+", "fund"),
    "accruals":       ("(TTM net income - operating cash flow) / assets", "-", "fund"),
    "asset_growth":   ("year-on-year total asset growth (investment)", "-", "fund"),
    "sales_growth":   ("year-on-year TTM revenue growth", "?", "fund"),
    "net_issue_12m":  ("12-month log change in market cap minus log return (net share issuance)", "-", "fund"),
    "sue":            ("standardised unexpected earnings (latest quarter, fades after ~110 days)", "+", "fund"),
    "ear":            ("3-day market-adjusted return around the latest earnings announcement", "+", "fund"),
}
PRICE_FACTORS = [k for k, v in FACTOR_INFO.items() if v[2] == "price"]
MINUTE_FACTORS = [k for k, v in FACTOR_INFO.items() if v[2] == "minute"]
FUND_FACTORS = [k for k, v in FACTOR_INFO.items() if v[2] == "fund"]

N_BUCKETS = 13              # 9:30-16:00 = 13 half hours
FULL_DAY_MINUTES = 390
EAR_HOLD_DAYS = 63          # EAR stays valid for about one quarter
EAR_SEARCH_DAYS = 25        # look this many trading days before the filing for the announcement day

log = logging.getLogger("factors")


# ============================== Helpers ==============================
def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="symbol", values=col).sort_index()


def _safe_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


def _universe_mask(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    uni = D.load_universe()
    arr = np.zeros((len(index), len(columns)), dtype=bool)
    uni = uni[uni["date"].isin(index) & uni["symbol"].isin(columns)]
    arr[index.get_indexer(uni["date"]), columns.get_indexer(uni["symbol"])] = True
    return pd.DataFrame(arr, index=index, columns=columns)


def _take(p: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """Values of a date x symbol panel at the (date, symbol) rows of `keys`, as float32."""
    vals = np.full(len(keys), np.nan, dtype="float32")
    ri = p.index.get_indexer(keys["date"])
    ci = p.columns.get_indexer(keys["symbol"])
    ok = (ri >= 0) & (ci >= 0)
    arr = p.to_numpy(dtype="float64", na_value=np.nan)
    v = arr[ri[ok], ci[ok]]
    v[~np.isfinite(v)] = np.nan
    vals[ok] = v
    return vals


def _rolling_beta(R: pd.DataFrame, M: pd.DataFrame, w: int, mp: int):
    ex, ey = R.rolling(w, mp).mean(), M.rolling(w, mp).mean()
    cov = (R * M).rolling(w, mp).mean() - ex * ey
    var_m = (M * M).rolling(w, mp).mean() - ey * ey
    var_x = (R * R).rolling(w, mp).mean() - ex * ex
    return cov / var_m.where(var_m > 0), cov, var_x


def _sector_mean(X: pd.DataFrame, sectors: pd.Series, mask: pd.DataFrame) -> pd.DataFrame:
    """Each cell = mean of X over the universe stocks of the same sector on that day."""
    out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    sec = sectors.reindex(X.columns)
    for k in sec.dropna().unique():
        cols = sec.index[sec == k]
        m = X[cols].where(mask[cols]).mean(axis=1)
        out[cols] = np.broadcast_to(m.to_numpy()[:, None], (len(X), len(cols)))
    return out


# ============================== Price factors ==============================
def price_factors(daily: pd.DataFrame, in_universe: pd.DataFrame | None,
                  sectors: pd.Series) -> Iterator[tuple[str, pd.DataFrame]]:
    """daily: load_daily(adjusted=True) long table. Yields (name, date x symbol panel) one at a time."""
    C = _pivot(daily, "adj_close")
    O = _pivot(daily, "adj_open").reindex_like(C)
    H = _pivot(daily, "adj_high").reindex_like(C)
    L = _pivot(daily, "adj_low").reindex_like(C)
    close = _pivot(daily, "close").reindex_like(C)
    volume = _pivot(daily, "volume").reindex_like(C)
    vwap = _pivot(daily, "vwap").reindex_like(C)
    R = _pivot(daily, "ret").reindex_like(C)
    R = R.where(R.notna(), C.pct_change(fill_method=None))
    DV = close * volume                                    # dollar volume
    lr = np.log1p(R)
    mask = in_universe if in_universe is not None else C.notna()

    yield "ret_1", R
    yield "ret_5", np.expm1(lr.rolling(5, min_periods=5).sum())
    yield "ret_21", np.expm1(lr.rolling(21, min_periods=21).sum())
    yield "ret_63", np.expm1(lr.rolling(63, min_periods=60).sum())
    yield "mom_6_1", np.expm1(lr.shift(21).rolling(105, min_periods=90).sum())
    yield "mom_12_1", np.expm1(lr.shift(21).rolling(231, min_periods=200).sum())
    yield "mom_12_7", np.expm1(lr.shift(126).rolling(126, min_periods=110).sum())
    yield "high_52w", C / C.rolling(252, min_periods=200).max()
    yield "trend_200", C / C.rolling(200, min_periods=180).mean() - 1
    yield "vol_20", R.rolling(20, min_periods=15).std()
    yield "vol_60", R.rolling(60, min_periods=45).std()
    yield "max_21", R.rolling(21, min_periods=15).max()
    yield "skew_60", R.rolling(60, min_periods=45).skew()
    yield "hl_range_20", _safe_log(H / L).rolling(20, min_periods=15).mean()

    # market = equal-weighted universe return (all stocks before the universe starts)
    m = R.where(mask).mean(axis=1)
    m = m.where(mask.sum(axis=1) > 0, R.mean(axis=1))
    M = pd.DataFrame(np.broadcast_to(m.to_numpy()[:, None], R.shape),
                     index=R.index, columns=R.columns).where(R.notna())
    beta, cov, var_x = _rolling_beta(R, M, 60, 40)
    yield "beta_60", beta
    yield "idio_vol_60", np.sqrt((var_x - beta * cov).clip(lower=0))
    resid = R - beta.shift(1) * M                          # yesterday's beta: no same-day fit
    yield "resid_ret_5", resid.rolling(5, min_periods=5).sum()
    rs = resid.shift(21)
    yield "resid_mom_12_1", rs.rolling(231, min_periods=200).sum() / \
        (rs.rolling(231, min_periods=200).std() * np.sqrt(231))
    del resid, rs, cov, var_x
    down = M < 0
    dbeta, _, _ = _rolling_beta(R.where(down), M.where(down), 120, 30)
    yield "downside_beta_120", dbeta
    del dbeta, beta, M

    if len(sectors):
        r21 = np.expm1(lr.rolling(21, min_periods=21).sum())
        yield "sector_rel_21", r21 - _sector_mean(r21, sectors, mask)
        del r21
        m61 = np.expm1(lr.shift(21).rolling(105, min_periods=90).sum())
        yield "sector_mom_6", _sector_mean(m61, sectors, mask).where(m61.notna())
        del m61

    illiq = R.abs() / DV.where(DV > 0)
    yield "amihud_20", _safe_log(illiq.rolling(20, min_periods=15).mean())
    adv60 = DV.rolling(60, min_periods=40).mean()
    yield "log_adv_60", _safe_log(adv60)
    yield "abn_vol_5_60", _safe_log(DV.rolling(5, min_periods=4).mean() / adv60)
    yield "turn_cv_60", DV.rolling(60, min_periods=40).std() / adv60
    dlogv = _safe_log(volume).diff()
    yield "pv_corr_20", R.rolling(20, min_periods=15).corr(dlogv)
    yield "overnight_20", _safe_log(O / C.shift(1)).rolling(20, min_periods=15).sum()
    yield "intraday_20", _safe_log(C / O).rolling(20, min_periods=15).sum()
    yield "close_vwap_5", _safe_log(close / vwap).rolling(5, min_periods=4).mean()


# ============================== Fundamental factors ==============================
def _ear(R: pd.DataFrame, M: pd.Series, volume: pd.DataFrame, filed: pd.DataFrame) -> pd.DataFrame:
    """Earnings-announcement return. For every new quarterly filing, the announcement day is taken to be
    the day of highest abnormal volume in the EAR_SEARCH_DAYS trading days up to the filing (the press
    release usually comes before the 10-Q). EAR = market-adjusted return over that day and its neighbours.
    The value becomes usable on the later of the filing day and the day after the announcement."""
    T, N = R.shape
    ex = (R.sub(M, axis=0)).to_numpy()
    vol = volume.to_numpy(dtype=float)
    base = volume.rolling(60, min_periods=20).median().shift(1).to_numpy(dtype=float)
    av = vol / np.where(base > 0, base, np.nan)
    F = filed.to_numpy()
    out = np.full((T, N), np.nan)
    for j in range(N):
        f = F[:, j]
        new = np.flatnonzero(np.isfinite(f) & (np.r_[np.nan, f[:-1]] != f))
        for p in new:
            lo = max(1, p - EAR_SEARCH_DAYS)
            w = av[lo:p + 1, j]
            if not np.isfinite(w).any():
                continue
            d0 = lo + int(np.nanargmax(w))
            hi = min(T - 1, d0 + 1)
            e = ex[d0 - 1:hi + 1, j]
            if np.isnan(e).all():
                continue
            start = max(p, hi)
            out[start:min(T, start + EAR_HOLD_DAYS), j] = np.nansum(e)
    return pd.DataFrame(out, index=R.index, columns=R.columns)


def fund_factors(daily_close: pd.DataFrame, C: pd.DataFrame, R: pd.DataFrame, volume: pd.DataFrame,
                 mask: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    if not FU.FACTS_FILE.exists():
        log.info("No fundamentals (run python fundamentals.py update): fundamental factors skipped")
        return
    dates, syms = C.index, C.columns
    pit = lambda item, age=FU.MAX_AGE_DAYS: FU.pit_panel(item, dates, syms, age)
    shares = pit("shares")
    mcap = daily_close * shares
    mcap = mcap.where(mcap > 0)
    del shares
    yield "log_mcap", np.log(mcap)
    ni = pit("ni_ttm")
    yield "ep", ni / mcap
    equity = pit("equity")
    pos_eq = equity.where(equity > 0)
    yield "bm", pos_eq / mcap
    yield "roe", ni / pos_eq
    del equity, pos_eq
    rev = pit("rev_ttm")
    yield "sp", rev / mcap
    yield "op_margin", pit("opinc_ttm") / rev.where(rev > 0)
    del rev
    cfo = pit("cfo_ttm")
    yield "cfp", cfo / mcap
    assets = pit("assets").where(lambda a: a > 0)
    yield "gpa", pit("gp_ttm") / assets
    yield "accruals", (ni - cfo) / assets
    del cfo, assets, ni
    yield "asset_growth", pit("asset_growth").clip(-0.9, 5)
    yield "sales_growth", pit("sales_growth").clip(-0.9, 5)
    yield "net_issue_12m", (np.log(mcap / mcap.shift(252)) - np.log(C / C.shift(252))).clip(-1, 1)
    del mcap
    yield "sue", pit("sue", 150)
    _, filed = FU.pit_panel("ni_q", dates, syms, 150, with_filed=True)
    m = R.where(mask).mean(axis=1)
    m = m.where(mask.sum(axis=1) > 0, R.mean(axis=1))
    yield "ear", _ear(R, m, volume, filed)


# ============================== Minute factors ==============================
def minute_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """One day of 1-minute bars -> one row of intraday features per stock.
    On half trading days the last-30-minute and profile features are set to NaN."""
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    mins = (df["ts"].dt.hour * 60 + df["ts"].dt.minute - 570).to_numpy()
    df["m"] = mins
    half_day = mins.max() < FULL_DAY_MINUTES - 1 - 30          # last bar before 15:29 = half day
    g = df.groupby("symbol", sort=True)

    lr = np.log(df["close"]).groupby(df["symbol"]).diff()
    rv2 = (lr ** 2).groupby(df["symbol"]).sum()
    r3 = (lr ** 3).groupby(df["symbol"]).sum()
    n = lr.groupby(df["symbol"]).count()
    rv = np.sqrt(rv2)
    out = pd.DataFrame(index=rv.index)
    out["rv"] = rv
    out["rskew"] = (np.sqrt(n) * r3 / rv ** 3).where(rv > 0)
    out["n_bars"] = g.size()

    first_open = g["open"].first()
    last_close = g["close"].last()
    c_1000 = df[df["m"] < 30].groupby("symbol")["close"].last()
    c_1530 = df[df["m"] < 360].groupby("symbol")["close"].last()
    out["open30_ret"] = np.log(c_1000 / first_open)
    out["last30_ret"] = np.log(last_close / c_1530)

    vol_total = g["volume"].sum()
    out["last30_vshare"] = df[df["m"] >= 360].groupby("symbol")["volume"].sum().reindex(out.index, fill_value=0) \
        / vol_total.where(vol_total > 0)
    pv = (df["vwap"].fillna(df["close"]) * df["volume"]).groupby(df["symbol"]).sum()
    day_vwap = pv / vol_total.where(vol_total > 0)
    out["vwap_dev"] = np.log(last_close / day_vwap)

    bucket = np.clip(df["m"].to_numpy() // 30, 0, N_BUCKETS - 1)
    bv = df.assign(b=bucket).groupby(["symbol", "b"])["volume"].sum().unstack(fill_value=0)
    bv = bv.reindex(index=out.index, columns=range(N_BUCKETS), fill_value=0)
    shares = bv.div(vol_total.where(vol_total > 0), axis=0)
    for b in range(N_BUCKETS):
        out[f"vs{b:02d}"] = shares[b]

    if half_day:
        cols = ["last30_ret", "last30_vshare"] + [f"vs{b:02d}" for b in range(N_BUCKETS)]
        out[cols] = np.nan
    return out.reset_index()


def update_minute_features() -> int:
    """Compute and cache intraday features for every minute file not yet processed. Returns new days."""
    files = sorted(D.MINUTE_DIR.glob("*.parquet"))
    MINUTE_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in MINUTE_FEAT_DIR.glob("*.parquet")}
    todo = [f for f in files if f.stem not in done]
    t0 = time.time()
    for i, f in enumerate(todo, 1):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        feat = minute_day_features(df)
        feat.insert(0, "date", pd.Timestamp(f.stem))
        feat.to_parquet(MINUTE_FEAT_DIR / f.name, index=False)
        if i % 50 == 0 or i == len(todo):
            log.info("Minute features %d/%d days, %.0f s elapsed", i, len(todo), time.time() - t0)
    return len(todo)


def minute_factors() -> Iterator[tuple[str, pd.DataFrame]]:
    files = sorted(MINUTE_FEAT_DIR.glob("*.parquet"))
    if not files:
        return
    mf = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    P = lambda c: _pivot(mf, c)
    yield "rskew_20", P("rskew").rolling(20, min_periods=10).mean()
    rv = P("rv")
    yield "rv_ratio_5_20", _safe_log(rv.rolling(5, min_periods=4).mean() / rv.rolling(20, min_periods=15).mean())
    yield "open30_ret_5", P("open30_ret").rolling(5, min_periods=4).sum()
    yield "last30_ret_5", P("last30_ret").rolling(5, min_periods=4).sum()
    yield "last30_vshare_20", P("last30_vshare").rolling(20, min_periods=15).mean()
    yield "vwap_dev_5", P("vwap_dev").rolling(5, min_periods=4).mean()

    # volume-profile deviation: total-variation distance 0.5 * sum_b |share_b today - 20-day mean share_b|
    dev = None
    for b in range(N_BUCKETS):
        s = P(f"vs{b:02d}")
        base = s.shift(1).rolling(20, min_periods=15).mean()
        d = (s - base).abs()
        dev = d if dev is None else dev.add(d, fill_value=np.nan)
    yield "vprofile_dev_5", (0.5 * dev).rolling(5, min_periods=3).mean()


# ============================== Build ==============================
def build_factors(use_minute: bool = True) -> pd.DataFrame:
    uni = D.load_universe()
    symbols = sorted(uni["symbol"].unique())          # every stock that was ever in the universe
    log.info("Reading daily bars: %d symbols that were ever in the universe", len(symbols))
    daily = D.load_daily(symbols=symbols, adjusted=True,
                         columns=["open", "high", "low", "close", "volume", "vwap", "ret"])
    C_index = pd.Index(sorted(daily["date"].unique()))
    C_cols = pd.Index(sorted(daily["symbol"].unique()))
    in_uni = _universe_mask(C_index, C_cols)
    sectors = FU.load_sectors()
    if sectors.empty:
        log.info("No sector data yet (python fundamentals.py update); sector factors skipped")

    keys = uni[["date", "symbol"]].sort_values(["date", "symbol"]).reset_index(drop=True)
    out = {"date": keys["date"].to_numpy(), "symbol": keys["symbol"].to_numpy(),
           "sector": keys["symbol"].map(sectors).fillna(-1).astype("int16").to_numpy()}
    t0 = time.time()

    def collect(gen, label):
        n = 0
        for name, p in gen:
            out[name] = _take(p, keys)
            n += 1
        log.info("%s factors: %d (%.0f s)", label, n, time.time() - t0)

    collect(price_factors(daily, in_uni, sectors), "Price")
    C = _pivot(daily, "adj_close")
    R = _pivot(daily, "ret").reindex_like(C)
    R = R.where(R.notna(), C.pct_change(fill_method=None))
    close = _pivot(daily, "close").reindex_like(C)
    volume = _pivot(daily, "volume").reindex_like(C)
    del daily
    collect(fund_factors(close, C, R, volume, in_uni.reindex_like(C).fillna(False)), "Fundamental")
    del C, R, close, volume

    if use_minute and any(D.MINUTE_DIR.glob("*.parquet")):
        n = update_minute_features()
        if n:
            log.info("New minute-feature days: %d", n)
        collect(minute_factors(), "Minute")
    elif use_minute:
        log.info("No minute bars: minute factors skipped (python data.py update --minute)")

    fac = pd.DataFrame(out)
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    fac.to_parquet(FACTOR_FILE, index=False)
    names = factor_columns(fac)
    cov = fac[names].notna().mean()
    log.info("Factor table saved: %d rows x %d factors (%s to %s)", len(fac), len(names),
             fac["date"].min().date(), fac["date"].max().date())
    log.info("Coverage: %s", ", ".join(f"{k} {v:.0%}" for k, v in cov.items()))
    return fac


def factor_columns(fac_or_cols) -> list[str]:
    cols = fac_or_cols.columns if hasattr(fac_or_cols, "columns") else fac_or_cols
    return [c for c in cols if c not in META_COLS]


def available_factors() -> list[str]:
    import pyarrow.parquet as pq
    return factor_columns(pq.read_schema(FACTOR_FILE).names)


def load_factors(start=None, end=None, names: list[str] | None = None) -> pd.DataFrame:
    """(date, symbol, sector, factors...) long table. Pass `names` to read only some columns."""
    cols = None if names is None else list(dict.fromkeys([*META_COLS, *names]))
    fac = pd.read_parquet(FACTOR_FILE, columns=cols)
    if start is not None:
        fac = fac[fac["date"] >= pd.Timestamp(start)]
    if end is not None:
        fac = fac[fac["date"] <= pd.Timestamp(end)]
    return fac.reset_index(drop=True)


# ============================== Preprocessing ==============================
def preprocess(df: pd.DataFrame, cols: list[str], neutralize: list[str] | str | None = None,
               mad_k: float = 5.0) -> pd.DataFrame:
    """Per day, cross-sectionally: MAD winsorise -> z-score -> optional neutralisation -> z-score again.
    neutralize: "sector" removes sector means; any numeric column (e.g. "log_adv_60") is regressed out.
    Returns a new (date, symbol, cols...) table; the input is not modified."""
    if isinstance(neutralize, str):
        neutralize = [neutralize]
    neutralize = list(neutralize or [])
    out = df[["date", "symbol"]].copy()
    by = df["date"]
    x = df[cols].astype("float64")
    med = x.groupby(by).transform("median")
    mad = (x - med).abs().groupby(by).transform("median") * 1.4826
    x = x.clip(lower=med - mad_k * mad, upper=med + mad_k * mad, axis=None)
    x = (x - x.groupby(by).transform("mean")) / x.groupby(by).transform("std")

    if "sector" in neutralize and "sector" in df:
        sec = df["sector"].fillna(-1)
        x = x - x.groupby([by, sec]).transform("mean")
    for nc in [c for c in neutralize if c != "sector" and c in df]:
        s = df[nc].astype("float64")
        s = (s - s.groupby(by).transform("mean")) / s.groupby(by).transform("std")
        for c in cols:
            if c == nc:
                continue
            y = x[c]
            ok = y.notna() & s.notna()
            ss, yy = s.where(ok), y.where(ok)
            sd = ss - ss.groupby(by).transform("mean")
            yd = yy - yy.groupby(by).transform("mean")
            b = (sd * yd).groupby(by).transform("sum") / (sd * sd).groupby(by).transform("sum")
            x[c] = yd - b * sd
    if neutralize:
        x = (x - x.groupby(by).transform("mean")) / x.groupby(by).transform("std")
    out[cols] = x
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compute factors")
    p.add_argument("--no-minute", action="store_true", help="skip minute factors")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not D.UNIVERSE_FILE.exists():
        raise SystemExit("No universe yet: run python data.py update first")
    build_factors(use_minute=not args.no_minute)


if __name__ == "__main__":
    main()
