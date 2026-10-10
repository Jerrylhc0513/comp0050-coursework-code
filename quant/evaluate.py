#!/usr/bin/env python3
"""
evaluate.py - single-factor tests and automatic factor selection

Per factor and holding period h:
  - Rank IC: daily cross-sectional Spearman correlation of the factor with the next-h-day return
  - IC mean / std, ICIR, share of days with IC > 0
  - Newey-West t-stat (lag h-1): overlapping h-day windows make plain t-stats too large
  - IC decay over h = 1, 5, 10, 20
  - quintile test: every h days sort into 5 groups and hold h days (no overlap); annualised group
    returns, long-short (Q5 - Q1) return and Sharpe, monotonicity, top-group turnover
  - average cross-sectional rank correlation between factors
  - IC by calendar year

Automatic selection (select_factors) - with ~50 candidates some will look significant by luck,
so the bar is high and every rule only uses data up to the cutoff date:
  1. |NW t| >= T_MIN (3.0) AND passes Benjamini-Hochberg FDR at FDR_Q across all candidates
  2. stable: the IC sign is the same in at least MIN_YEAR_CONSISTENCY of the calendar years,
     and both halves of the sample have the same sign with |t| >= 1
  3. still profitable after costs: sign-adjusted long-short return minus turnover x round-trip cost > 0
  4. enough coverage (fundamental factors miss some stocks)
  5. greedy de-duplication: walk down the list by |t|; skip a factor whose |rank correlation| with
     an already selected factor is >= MAX_CORR; stop at MAX_FACTORS
  The sign used later is the sign of the IC measured before the cutoff (not the literature sign).

Entry price (--entry)
  open   signal after the close of t, buy at the open of t+1, sell at the open h days later (default)
  vwap   buy / sell at the full-day VWAP
  close  buy at the close of t (not achievable, for comparison only)

Returns here are before costs (except the cost screen in step 3). Costs are modelled in backtest.py.

Usage
  python evaluate.py                                  # all factors, main horizon 20 days
  python evaluate.py --end 2023-12-31                 # test and select using data up to 2023 only
  python evaluate.py --factors ret_5 vol_20 --main-horizon 5 --no-neutralize
Output (reports/)
  factor_summary.csv, ic_decay.csv, ic_by_year.csv, quantile_returns.csv, factor_corr.csv,
  ic_daily.parquet, ls_curve.parquet, selection_table.csv, selected_factors.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import factors as F

REPORT_DIR = Path(os.environ.get("QUANT_REPORT_DIR", Path(__file__).resolve().parent / "reports"))
SELECTED_FILE = REPORT_DIR / "selected_factors.json"
HORIZONS = [1, 5, 10, 20]
MAIN_H = 20                # main holding period (trading days); matches the mid/low-frequency strategy
NEUTRALIZE = ["sector", "log_adv_60"]   # default neutralisation: sector and size
N_QUANTILES = 5
MIN_STOCKS = 50            # skip days with fewer valid stocks
CORR_STEP = 10             # sample every N days for factor-factor correlations

# selection rules
T_MIN = 3.0
FDR_Q = 0.05
MIN_YEAR_CONSISTENCY = 0.6
MAX_CORR = 0.5
MAX_FACTORS = 12
MIN_FACTORS = 3
MIN_COVERAGE = 0.4         # average stocks covered / average universe size
ROUND_TRIP_COST = 0.0020   # cost per unit of one-leg turnover, used in the cost screen

log = logging.getLogger("evaluate")


# ============================== Forward returns ==============================
def entry_price_panel(entry: str, start=None, symbols=None) -> pd.DataFrame:
    """Adjusted entry-price panel (date x symbol)."""
    daily = D.load_daily(start=start, symbols=symbols, adjusted=True, columns=["open", "close", "vwap", "ret"])
    if entry == "open":
        col = "adj_open"
    elif entry == "close":
        col = "adj_close"
    elif entry == "vwap":
        daily["adj_vwap"] = daily["vwap"] * daily["adj_close"] / daily["close"]
        col = "adj_vwap"
    else:
        raise ValueError(f"Unknown entry: {entry}")
    return daily.pivot(index="date", columns="symbol", values=col).sort_index()


def forward_returns(P: pd.DataFrame, h: int, entry: str) -> pd.DataFrame:
    """Row t = return of holding h days after the signal at the close of t.
    open/vwap start at t+1; close starts at t."""
    if entry == "close":
        return P.shift(-h) / P - 1
    return P.shift(-(1 + h)) / P.shift(-1) - 1


def label_lag(h: int, entry: str) -> int:
    """At the close of t, the newest date whose h-day label is fully known is t - label_lag."""
    return h if entry == "close" else h + 1


# ============================== Statistics ==============================
def rowwise_rank_corr(X: pd.DataFrame, Y: pd.DataFrame, min_n: int = MIN_STOCKS) -> pd.Series:
    """Per row (day), Spearman correlation of X and Y using stocks where both are present."""
    ok = X.notna() & Y.notna()
    xr = X.where(ok).rank(axis=1)
    yr = Y.where(ok).rank(axis=1)
    xd = xr.sub(xr.mean(axis=1), axis=0)
    yd = yr.sub(yr.mean(axis=1), axis=0)
    num = (xd * yd).sum(axis=1)
    den = np.sqrt((xd * xd).sum(axis=1) * (yd * yd).sum(axis=1))
    ic = num / den.where(den > 0)
    return ic.where(ok.sum(axis=1) >= min_n).dropna()


def newey_west_t(x: pd.Series, lags: int) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def ic_stats(ic: pd.Series, h: int) -> dict:
    return {
        "ic_mean": ic.mean(),
        "ic_std": ic.std(),
        "icir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "t_nw": newey_west_t(ic, max(h - 1, 0)),
        "ic_pos": (ic > 0).mean(),
        "n_days": len(ic),
    }


def quantile_analysis(X: pd.DataFrame, Y: pd.DataFrame, h: int, nq: int = N_QUANTILES,
                      min_n: int = MIN_STOCKS) -> tuple[pd.DataFrame, dict]:
    """Rebalance every h days (no overlap). Returns per-period group returns (+ LS and top-group
    turnover columns) and summary stats."""
    ok = X.notna() & Y.notna()
    valid_days = ok.sum(axis=1) >= min_n
    days = X.index[valid_days.to_numpy()][::h]
    Xs, Ys, oks = X.loc[days].where(ok.loc[days]), Y.loc[days], ok.loc[days]
    pct = Xs.rank(axis=1, pct=True, method="first")
    Q = np.ceil(pct * nq).clip(1, nq)
    per = pd.DataFrame({q: Ys.where(Q == q).mean(axis=1) for q in range(1, nq + 1)})
    per["LS"] = per[nq] - per[1]

    top = (Q == nq) & oks
    prev = top.shift(1, fill_value=False)
    overlap = (top & prev).sum(axis=1)
    size_prev = prev.sum(axis=1)
    per["turnover"] = 1 - overlap / size_prev.where(size_prev > 0)

    ppy = 252 / h
    ann = per[[*range(1, nq + 1), "LS"]].mean() * ppy
    sharpe_ls = per["LS"].mean() / per["LS"].std() * np.sqrt(ppy) if per["LS"].std() > 0 else np.nan
    mono = pd.Series(ann[list(range(1, nq + 1))].to_numpy()).corr(pd.Series(range(1, nq + 1)), method="spearman")
    stats = {f"Q{q}_ann": ann[q] for q in range(1, nq + 1)}
    stats.update({"LS_ann": ann["LS"], "LS_sharpe": sharpe_ls, "monotonic": mono,
                  "top_turnover": per["turnover"].iloc[1:].mean()})
    return per, stats


# ============================== Research (per-factor time series) ==============================
@dataclass
class Research:
    """Everything the selection needs, as small time series, so selection at any cutoff date is cheap."""
    h: int
    entry: str
    neutralize: list
    ic: pd.DataFrame = field(default_factory=pd.DataFrame)          # date x factor, main-horizon IC
    ls: pd.DataFrame = field(default_factory=pd.DataFrame)          # period start x factor, LS return
    turnover: pd.DataFrame = field(default_factory=pd.DataFrame)    # period start x factor
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)    # date x factor, stocks with a value
    universe_n: pd.Series = field(default_factory=pd.Series)        # date -> universe size
    ranks: dict = field(default_factory=dict)                       # factor -> sampled pct-rank panel
    decay: dict = field(default_factory=dict)                       # factor -> {h: IC series}
    quant: dict = field(default_factory=dict)                       # factor -> quintile stats (full sample)


def _cache_file(h: int, entry: str, neutralize: list) -> Path:
    tag = "-".join(neutralize) if neutralize else "raw"
    return REPORT_DIR / "cache" / f"research_h{h}_{entry}_{tag}.pkl"


def research(names: list[str] | None = None, entry: str = "open", h: int = MAIN_H,
             neutralize: list | None = None, horizons: list[int] | None = None,
             use_cache: bool = True) -> Research:
    """Compute per-factor IC / long-short / turnover series over the whole history, one factor at a time.
    The result is cached and reused while the factor table is unchanged."""
    neutralize = list(NEUTRALIZE if neutralize is None else neutralize)
    all_names = F.available_factors()
    names = names or all_names
    missing = [n for n in names if n not in all_names]
    if missing:
        raise SystemExit(f"Not in the factor table: {missing}")
    horizons = sorted(set(horizons or [h]) | {h})
    cf = _cache_file(h, entry, neutralize)
    stamp = (F.FACTOR_FILE.stat().st_mtime, tuple(names), tuple(horizons))
    if use_cache and cf.exists():
        try:
            with open(cf, "rb") as fh:
                saved_stamp, res = pickle.load(fh)
            if saved_stamp == stamp:
                log.info("Using cached research %s", cf.name)
                return res
        except Exception:
            pass

    keys = F.load_factors(names=[])
    idx = pd.DatetimeIndex(sorted(keys["date"].unique()))
    cols = pd.Index(sorted(keys["symbol"].unique()))
    universe_n = keys.groupby("date").size()
    del keys
    P = entry_price_panel(entry, symbols=list(cols)).reindex(columns=cols)
    fwd = {hh: forward_returns(P, hh, entry).reindex(index=idx) for hh in horizons}
    del P
    neut_cols = [c for c in neutralize if c != "sector" and c in all_names]
    res = Research(h=h, entry=entry, neutralize=neutralize, universe_n=universe_n)
    ic, ls, to, cov = {}, {}, {}, {}
    for i, n in enumerate(names, 1):
        if n in neut_cols:
            continue                                   # a neutraliser is zero after neutralisation
        fac = F.load_factors(names=[n, *neut_cols])
        if fac[n].notna().sum() == 0:
            log.warning("%s has no data, skipped", n)
            continue
        if neutralize:
            fac = fac.assign(**{n: F.preprocess(fac, [n], neutralize)[n]})
        X = fac.pivot(index="date", columns="symbol", values=n).reindex(index=idx, columns=cols)
        del fac
        res.decay[n] = {hh: rowwise_rank_corr(X, fwd[hh]) for hh in horizons}
        ic[n] = res.decay[n][h]
        per, qst = quantile_analysis(X, fwd[h], h)
        ls[n], to[n] = per["LS"], per["turnover"]
        res.quant[n] = qst | {"rank_autocorr": rowwise_rank_corr(X, X.shift(-h)).mean()}
        cov[n] = X.notna().sum(axis=1)
        res.ranks[n] = X.iloc[::CORR_STEP].rank(axis=1, pct=True).astype("float32")
        if i % 5 == 0 or i == len(names):
            log.info("Research %d/%d factors", i, len(names))
    res.ic, res.ls, res.turnover = pd.DataFrame(ic), pd.DataFrame(ls), pd.DataFrame(to)
    res.coverage = pd.DataFrame(cov)
    cf.parent.mkdir(parents=True, exist_ok=True)
    with open(cf, "wb") as fh:
        pickle.dump((stamp, res), fh)
    return res


# ============================== Selection ==============================
def _bh_pass(p: pd.Series, q: float) -> pd.Series:
    """Benjamini-Hochberg: which hypotheses are rejected at FDR q."""
    p = p.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool)
    thresh = q * np.arange(1, m + 1) / m
    below = np.flatnonzero(p.to_numpy() <= thresh)
    k = below.max() + 1 if len(below) else 0
    return pd.Series(np.arange(m) < k, index=p.index)


def factor_table(res: Research, cutoff=None, start=None) -> pd.DataFrame:
    """Statistics per factor using only labels known at the close of `cutoff`."""
    ic = res.ic
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        dates = res.universe_n.index
        known = dates[dates <= cutoff]
        if len(known) <= label_lag(res.h, res.entry):
            return pd.DataFrame()
        last_label = known[-1 - label_lag(res.h, res.entry)]
        ic = ic[ic.index <= last_label]
    if start is not None:
        ic = ic[ic.index >= pd.Timestamp(start)]
    if ic.empty:
        return pd.DataFrame()
    # a period is indexed by its start date, whose label is known by the same rule as the IC
    ls = res.ls.loc[(res.ls.index >= ic.index.min()) & (res.ls.index <= ic.index.max())]
    to = res.turnover.reindex(ls.index)
    cov = res.coverage.reindex(ic.index)
    uni = res.universe_n.reindex(ic.index)
    rows = []
    ppy = 252 / res.h
    for n in ic.columns:
        s = ic[n].dropna()
        if len(s) < 120:
            continue
        t = newey_west_t(s, max(res.h - 1, 0))
        sign = 1.0 if s.mean() > 0 else -1.0
        yearly = s.groupby(s.index.year).mean()
        yearly = yearly[s.groupby(s.index.year).size() >= 60]
        consist = (np.sign(yearly) == sign).mean() if len(yearly) else np.nan
        half = len(s) // 2
        t1 = newey_west_t(s.iloc[:half], max(res.h - 1, 0))
        t2 = newey_west_t(s.iloc[half:], max(res.h - 1, 0))
        ls_n = ls[n].dropna() * sign
        net = ls_n.mean() * ppy - 2 * to[n].mean() * ROUND_TRIP_COST * ppy if len(ls_n) else np.nan
        rows.append({"factor": n, "sign": int(sign), "ic_mean": s.mean(), "icir": s.mean() / s.std(),
                     "t_nw": t, "p": erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
                     "year_consistency": consist, "n_years": len(yearly), "t_half1": t1, "t_half2": t2,
                     "ls_ann": ls_n.mean() * ppy if len(ls_n) else np.nan, "ls_net_ann": net,
                     "turnover": to[n].mean(), "coverage": (cov[n] / uni).mean(), "n_days": len(s)})
    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()


def select_factors(res: Research, cutoff=None, start=None, t_min: float = T_MIN,
                   use_fdr: bool = True, verbose: bool = True) -> tuple[dict[str, int], pd.DataFrame]:
    """Returns ({factor: sign}, table with the reason each factor passed or failed)."""
    tab = factor_table(res, cutoff, start)
    if tab.empty:
        return {}, tab
    tab["fdr_pass"] = _bh_pass(tab["p"], FDR_Q).reindex(tab.index).fillna(False).astype(bool) \
        if use_fdr else True
    tab["sig"] = (tab["t_nw"].abs() >= t_min) & tab["fdr_pass"]
    tab["stable"] = (tab["year_consistency"].fillna(0) >= MIN_YEAR_CONSISTENCY) & \
        (np.sign(tab["t_half1"]) == tab["sign"]) & (np.sign(tab["t_half2"]) == tab["sign"]) & \
        (tab["t_half1"].abs() >= 1) & (tab["t_half2"].abs() >= 1)
    tab["net_ok"] = tab["ls_net_ann"] > 0
    tab["cov_ok"] = tab["coverage"] >= MIN_COVERAGE
    tab["candidate"] = tab["sig"] & tab["stable"] & tab["net_ok"] & tab["cov_ok"]
    tab["selected"] = False
    tab["dropped_for"] = ""

    hi = None if cutoff is None else pd.Timestamp(cutoff)
    chosen: list[str] = []
    for n in tab[tab["candidate"]].sort_values("t_nw", key=abs, ascending=False).index:
        if len(chosen) >= MAX_FACTORS:
            break
        clash = ""
        for c in chosen:
            a, b = res.ranks[n], res.ranks[c]
            if hi is not None:
                a, b = a[a.index <= hi], b[b.index <= hi]
            if start is not None:
                a, b = a[a.index >= pd.Timestamp(start)], b[b.index >= pd.Timestamp(start)]
            rho = rowwise_rank_corr(a, b).mean()
            if abs(rho) >= MAX_CORR:
                clash = f"{c} ({rho:+.2f})"
                break
        if clash:
            tab.loc[n, "dropped_for"] = clash
        else:
            chosen.append(n)
            tab.loc[n, "selected"] = True

    if len(chosen) < MIN_FACTORS and (t_min > 2.0 or use_fdr):
        if verbose:
            log.warning("Only %d factors passed the strict rules up to %s; relaxing to |t| >= 2 without FDR",
                        len(chosen), cutoff.date() if hasattr(cutoff, "date") else cutoff)
        return select_factors(res, cutoff, start, t_min=2.0, use_fdr=False, verbose=verbose)
    signs = {n: int(tab.loc[n, "sign"]) for n in chosen}
    return signs, tab


def save_selection(signs: dict[str, int], tab: pd.DataFrame, res: Research, cutoff, path: Path = SELECTED_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cutoff": str(pd.Timestamp(cutoff).date()) if cutoff is not None else None,
               "horizon": res.h, "entry": res.entry, "neutralize": res.neutralize, "factors": signs,
               "stats": {n: {k: (None if pd.isna(v) else float(v)) for k, v in
                             tab.loc[n, ["ic_mean", "icir", "t_nw", "ls_net_ann", "turnover"]].items()}
                         for n in signs}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_selection(path: Path = SELECTED_FILE) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ============================== Full report ==============================
def factor_corr_matrix(res: Research, names: list[str], end=None) -> pd.DataFrame:
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = res.ranks[a], res.ranks[b]
            if end is not None:
                x, y = x[x.index <= pd.Timestamp(end)], y[y.index <= pd.Timestamp(end)]
            M.loc[a, b] = M.loc[b, a] = rowwise_rank_corr(x, y).mean()
    return M


def evaluate(names: list[str] | None = None, entry: str = "open", horizons: list[int] = HORIZONS,
             main_h: int = MAIN_H, neutralize: list | None = None, start=None, end=None) -> pd.DataFrame:
    res = research(names, entry, main_h, neutralize, horizons)
    lag = max(main_h - 1, 0)
    end_ts = pd.Timestamp(end) if end is not None else None
    start_ts = pd.Timestamp(start) if start is not None else None

    def clip(s: pd.Series) -> pd.Series:
        if end_ts is not None:
            s = s[s.index <= end_ts]
        if start_ts is not None:
            s = s[s.index >= start_ts]
        return s

    summary, decay, yearly, quant, ic_daily, ls_curves = [], [], {}, {}, {}, {}
    for n in res.ic.columns:
        row_decay = {"factor": n}
        for hh, s in res.decay[n].items():
            s = clip(s)
            row_decay[f"ic_{hh}d"] = s.mean()
            row_decay[f"t_{hh}d"] = newey_west_t(s, max(hh - 1, 0))
        decay.append(row_decay)
        ic = clip(res.ic[n].dropna())
        if ic.empty:
            continue
        st = ic_stats(ic, main_h)
        st["t_nw"] = newey_west_t(ic, lag)
        ls = clip(res.ls[n].dropna())
        ppy = 252 / main_h
        q = res.quant[n]
        info = F.FACTOR_INFO.get(n, ("", "?", ""))
        summary.append({"factor": n, "desc": info[0], "group": info[2], "lit_sign": info[1],
                        "avg_stocks": clip(res.coverage[n])[lambda v: v > 0].mean(), **st,
                        "LS_ann": ls.mean() * ppy,
                        "LS_sharpe": ls.mean() / ls.std() * np.sqrt(ppy) if ls.std() > 0 else np.nan,
                        "monotonic": q["monotonic"], "top_turnover": clip(res.turnover[n]).mean(),
                        "rank_autocorr": q["rank_autocorr"]})
        quant[n] = {f"Q{k}": q[f"Q{k}_ann"] for k in range(1, N_QUANTILES + 1)} | {"LS": q["LS_ann"]}
        yearly[n] = ic.groupby(ic.index.year).mean()
        ic_daily[n] = ic
        ls_curves[n] = (1 + ls.fillna(0)).cumprod()
    if not summary:
        raise SystemExit("No factor could be evaluated (history too short or too few stocks)")
    summ = pd.DataFrame(summary).set_index("factor")
    summ = summ.reindex(summ["t_nw"].abs().sort_values(ascending=False).index)

    signs, tab = select_factors(res, cutoff=end_ts, start=start_ts)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summ["selected"] = summ.index.isin(list(signs))
    summ.to_csv(REPORT_DIR / "factor_summary.csv")
    pd.DataFrame(decay).set_index("factor").to_csv(REPORT_DIR / "ic_decay.csv")
    pd.DataFrame(yearly).T.to_csv(REPORT_DIR / "ic_by_year.csv")
    pd.DataFrame(quant).T.to_csv(REPORT_DIR / "quantile_returns.csv")
    pd.DataFrame(ic_daily).to_parquet(REPORT_DIR / "ic_daily.parquet")
    pd.DataFrame(ls_curves).to_parquet(REPORT_DIR / "ls_curve.parquet")
    tab.to_csv(REPORT_DIR / "selection_table.csv")
    save_selection(signs, tab, res, end_ts if end_ts is not None else res.ic.index.max(), SELECTED_FILE)
    factor_corr_matrix(res, list(summ.index), end_ts).to_csv(REPORT_DIR / "factor_corr.csv")

    _print_report(summ, pd.DataFrame(decay).set_index("factor"), tab, signs, entry, main_h, res.neutralize)
    return summ


def _print_report(summ: pd.DataFrame, decay: pd.DataFrame, tab: pd.DataFrame, signs: dict, entry: str,
                  h: int, neutralize: list) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(f"\n===== Single-factor tests: entry={entry}, main horizon={h} days, "
          f"neutralised on {neutralize or 'nothing'} =====")
    show = summ[["group", "lit_sign", "ic_mean", "icir", "t_nw", "ic_pos", "LS_ann", "LS_sharpe",
                 "monotonic", "top_turnover", "avg_stocks", "selected"]].copy()
    fmt = {"ic_mean": "{:+.4f}", "icir": "{:+.3f}", "t_nw": "{:+.2f}", "ic_pos": "{:.0%}",
           "LS_ann": "{:+.1%}", "LS_sharpe": "{:+.2f}", "monotonic": "{:+.2f}", "top_turnover": "{:.0%}",
           "avg_stocks": "{:.0f}"}
    for c, f in fmt.items():
        show[c] = show[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    show["selected"] = np.where(summ["selected"], "YES", "")
    print(show.to_string())
    print("\n----- IC decay (IC mean / NW t) -----")
    hs = sorted({int(c.split("_")[1][:-1]) for c in decay.columns if c.startswith("ic_")})
    dd = pd.DataFrame({f"{x}d": decay[f"ic_{x}d"].map("{:+.4f}".format) + " / " + decay[f"t_{x}d"].map("{:+.1f}".format)
                       for x in hs}).reindex(summ.index)
    print(dd.to_string())
    if len(tab):
        print("\n----- Selection (sig = |t| and FDR, stable = by year and by half, net = LS after costs) -----")
        t = tab[["sign", "t_nw", "year_consistency", "t_half1", "t_half2", "ls_net_ann", "coverage",
                 "sig", "stable", "net_ok", "selected", "dropped_for"]].sort_values("t_nw", key=abs, ascending=False)
        print(t.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\nSelected: " + (", ".join(f"{k}({'+' if v > 0 else '-'})" for k, v in signs.items()) or "none"))
    print(f"Saved to {SELECTED_FILE}. model.py uses it by default; the walk-forward method re-selects every year.")


def main() -> None:
    p = argparse.ArgumentParser(description="Single-factor tests and factor selection")
    p.add_argument("--factors", nargs="*", help="factors to test (default: all)")
    p.add_argument("--entry", choices=["open", "vwap", "close"], default="open")
    p.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    p.add_argument("--main-horizon", type=int, default=MAIN_H, help="holding period for quintiles and selection")
    p.add_argument("--no-neutralize", action="store_true", help="do not neutralise on sector and size")
    p.add_argument("--start")
    p.add_argument("--end", help="only use data up to this date (keep later data for out-of-sample tests)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not F.FACTOR_FILE.exists():
        raise SystemExit("No factor table yet: run python factors.py")
    evaluate(args.factors, args.entry, args.horizons, args.main_horizon,
             [] if args.no_neutralize else None, args.start, args.end)


if __name__ == "__main__":
    main()
