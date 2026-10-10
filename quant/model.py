#!/usr/bin/env python3
"""
model.py - combine factors into one score per stock per day (higher = more attractive)

Methods (all strictly use information available at the time)
  wf     walk-forward (default, used for live trading): every January the factors are re-selected with
         evaluate.select_factors() using only data up to the previous year end, and that year's scores are
         the sign-weighted average of the selected (sector/size-neutralised) factors. Backtesting this
         score is a genuine out-of-sample test, and today's live score is produced by exactly the same rule.
  equal  equal weights on a fixed list: --factors name:sign ..., else reports/selected_factors.json
         (written by evaluate.py), else DEFAULT_EQUAL
  ic     IC-weighted: weight = rolling mean IC of the last IC_WINDOW days, lagged so the labels are known
  lgbm   LightGBM retrained monthly on the last TRAIN_YEARS of data; label = cross-sectional rank of the
         next-h-day return; training ends at t-h-1 (purged) so no future information is used

Usage
  python model.py                                   # = --method wf
  python model.py --method wf --refresh             # forget cached yearly selections and redo them
  python model.py --method equal                    # uses reports/selected_factors.json
  python model.py --method equal --factors ret_5:-1 vol_20:-1 mom_12_1:+1
  python model.py --method ic / --method lgbm
Output
  data/scores/<method>.parquet    (date, symbol, score), universe stocks only
  reports/wf_selection/<year>.json  factors used in each walk-forward year
  console: IC and quintile test of the score itself (same definitions as evaluate.py)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time

import numpy as np
import pandas as pd

import data as D
import evaluate as E
import factors as F

SCORE_DIR = D.DATA_DIR / "scores"
WF_DIR = E.REPORT_DIR / "wf_selection"

HORIZON = E.MAIN_H          # forecast horizon in trading days
ENTRY = "open"              # same as the backtest: open / vwap / close
NEUTRALIZE = E.NEUTRALIZE   # sector + size
WF_MIN_TRAIN_DAYS = 504     # walk-forward: need ~2 years of history before the first selection
IC_WINDOW = 252             # ic: look-back for the IC weights
IC_MIN_DAYS = 126           # ic: minimum IC history before producing scores
TRAIN_YEARS = 3             # lgbm: training window
MIN_TRAIN_DAYS = 252        # lgbm: minimum training days
RETRAIN_MONTHS = 1          # lgbm: retrain every N months
LGBM_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42)

# Fallback for `equal` when there is no selection file yet (literature signs)
DEFAULT_EQUAL = {"mom_12_1": +1, "ret_21": -1, "vol_60": -1, "high_52w": +1, "gpa": +1, "bm": +1}

log = logging.getLogger("model")


# ============================== Helpers ==============================
def _combine(z: pd.DataFrame, names: list[str], weights: np.ndarray) -> np.ndarray:
    """Weighted sum of the columns of z; a missing factor counts as 0 (cross-sectional mean).
    NaN when fewer than half of the factors are present. weights: 1-D or 2-D (one row per row of z)."""
    vals = z[names].to_numpy(dtype="float64")
    ok = np.isfinite(vals)
    w = np.broadcast_to(weights, vals.shape)
    num = np.where(ok, vals * w, 0.0).sum(axis=1)
    den = np.where(ok, np.abs(w), 0.0).sum(axis=1)
    score = num / np.where(den > 0, den, np.nan)
    score[ok.sum(axis=1) < np.ceil(len(names) / 2)] = np.nan
    return score


def _forward(idx: pd.Index, cols: pd.Index, h: int, entry: str) -> pd.DataFrame:
    P = E.entry_price_panel(entry, symbols=list(cols)).reindex(columns=cols)
    return E.forward_returns(P, h, entry).reindex(index=idx)


def _neutral_numeric(neutralize: list) -> list[str]:
    return [c for c in neutralize if c != "sector"]


def _prep(names: list[str], neutralize: list) -> pd.DataFrame:
    extra = [c for c in _neutral_numeric(neutralize) if c not in names]
    fac = F.load_factors(names=[*names, *extra])
    return F.preprocess(fac, names, neutralize)


# ============================== Methods ==============================
def score_equal(signs: dict[str, int], neutralize: list) -> pd.DataFrame:
    names = list(signs)
    z = _prep(names, neutralize)
    w = np.array([float(signs[n]) for n in names])
    return z[["date", "symbol"]].assign(score=_combine(z, names, w))


def walk_forward_selections(h: int, entry: str, neutralize: list, refresh: bool = False) -> dict[int, dict]:
    """{year: {factor: sign}} using data up to the end of the previous year. Cached in reports/wf_selection/."""
    if refresh and WF_DIR.exists():
        shutil.rmtree(WF_DIR)
    WF_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(sorted(F.load_factors(names=[])["date"].unique()))
    res = None
    out = {}
    for year in sorted(set(dates.year)):
        before = dates[dates < pd.Timestamp(year=year, month=1, day=1)]
        if len(before) < WF_MIN_TRAIN_DAYS + E.label_lag(h, entry):
            continue
        cutoff = before[-1]
        f = WF_DIR / f"{year}.json"
        sel = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        if sel is None or sel.get("horizon") != h or sel.get("entry") != entry or sel.get("neutralize") != neutralize:
            if res is None:
                res = E.research(None, entry, h, neutralize)
            signs, tab = E.select_factors(res, cutoff=cutoff)
            if not signs:
                log.warning("%d: no factor selected, year skipped", year)
                continue
            E.save_selection(signs, tab, res, cutoff, f)
            sel = json.loads(f.read_text(encoding="utf-8"))
        out[year] = sel["factors"]
        log.info("%d (selected with data to %s): %s", year, sel["cutoff"],
                 ", ".join(f"{k}({'+' if v > 0 else '-'})" for k, v in sel["factors"].items()))
    if out:
        last = max(out)
        shutil.copy(WF_DIR / f"{last}.json", E.SELECTED_FILE)
    return out


def score_walk_forward(h: int, entry: str, neutralize: list, refresh: bool = False) -> pd.DataFrame:
    sels = walk_forward_selections(h, entry, neutralize, refresh)
    if not sels:
        raise SystemExit("Walk-forward needs at least ~2 years of factor history before the first test year")
    union = sorted({n for s in sels.values() for n in s})
    z = _prep(union, neutralize)
    score = np.full(len(z), np.nan)
    years = z["date"].dt.year.to_numpy()
    for year, signs in sels.items():
        rows = years == year
        names = list(signs)
        score[rows] = _combine(z.loc[rows], names, np.array([float(signs[n]) for n in names]))
    return z[["date", "symbol"]].assign(score=score)


def score_ic(names: list[str], h: int, entry: str, neutralize: list) -> pd.DataFrame:
    z = _prep(names, neutralize)
    panels = {n: z.pivot(index="date", columns="symbol", values=n) for n in names}
    idx, sym = panels[names[0]].index, panels[names[0]].columns
    fwd = _forward(idx, sym, h, entry)

    ic = pd.DataFrame({n: E.rowwise_rank_corr(panels[n], fwd) for n in names}).reindex(idx)
    known = ic.shift(E.label_lag(h, entry))                    # what is known on day t
    W = known.rolling(IC_WINDOW, min_periods=IC_MIN_DAYS).mean()
    W = W.div(W.abs().sum(axis=1), axis=0)
    E.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    W.to_csv(E.REPORT_DIR / "ic_weights.csv")

    wrow = W.reindex(z["date"]).to_numpy()
    no_weight = ~np.isfinite(wrow).any(axis=1)             # dates without enough IC history
    wrow = np.nan_to_num(wrow, nan=0.0)                     # a factor without a weight yet counts as 0
    score = _combine(z, names, wrow)
    score[no_weight] = np.nan
    last = W.dropna(how="all").iloc[-1] if W.notna().any().any() else None
    if last is not None:
        log.info("Latest weights: %s", ", ".join(f"{k} {v:+.2f}" for k, v in last.sort_values(key=abs, ascending=False).items()))
    return z[["date", "symbol"]].assign(score=score)


def score_lgbm(names: list[str], h: int, entry: str, neutralize: list) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError:
        raise SystemExit("The lgbm method needs: pip install lightgbm")
    z = _prep(names, neutralize)
    idx = pd.Index(sorted(z["date"].unique()))
    sym = pd.Index(sorted(z["symbol"].unique()))
    fwd = _forward(idx, sym, h, entry)
    label = fwd.where(fwd.notna()).rank(axis=1, pct=True)            # cross-sectional rank as the label
    lab_long = label.rename_axis(index="date", columns="symbol").reset_index() \
        .melt(id_vars="date", var_name="symbol", value_name="label").dropna(subset=["label"])
    df = z.merge(lab_long, on=["date", "symbol"], how="left")
    df["pos"] = idx.get_indexer(df["date"])

    months = idx.to_period("M")
    starts = [i for i in range(len(idx)) if i == 0 or months[i] != months[i - 1]]
    starts = starts[::RETRAIN_MONTHS]
    lag = E.label_lag(h, entry)
    window = TRAIN_YEARS * 252
    scores = np.full(len(df), np.nan)
    imps = []
    t0 = time.time()
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(idx)
        train_end = s - lag
        train_start = max(0, train_end - window + 1)
        if train_end - train_start + 1 < MIN_TRAIN_DAYS:
            continue
        tr = df[(df["pos"] >= train_start) & (df["pos"] <= train_end) & df["label"].notna()]
        if len(tr) < 1000:
            continue
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(tr[names], tr["label"])
        te = (df["pos"] >= s) & (df["pos"] < e)
        scores[te.to_numpy()] = model.predict(df.loc[te, names])
        imps.append(pd.Series(model.booster_.feature_importance("gain"), index=names, name=idx[s]))
        log.info("Trained %s: %d rows (%s to %s), %.0f s elapsed", idx[s].date(), len(tr),
                 idx[train_start].date(), idx[train_end].date(), time.time() - t0)
    if imps:
        imp = pd.DataFrame(imps)
        E.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        imp.to_csv(E.REPORT_DIR / "lgbm_importance.csv")
        avg = (imp.div(imp.sum(axis=1), axis=0)).mean().sort_values(ascending=False)
        log.info("Mean feature importance: %s", ", ".join(f"{k} {v:.0%}" for k, v in avg.head(10).items()))
    out = df[["date", "symbol"]].copy()
    out["score"] = scores
    return out


# ============================== Main ==============================
def _equal_signs(factors) -> dict[str, int]:
    if isinstance(factors, dict):
        return factors
    sel = E.load_selection()
    if sel and sel.get("factors"):
        log.info("Using %s (cutoff %s)", E.SELECTED_FILE.name, sel.get("cutoff"))
        return {k: int(v) for k, v in sel["factors"].items()}
    log.warning("No selected_factors.json yet (run evaluate.py); using DEFAULT_EQUAL")
    return DEFAULT_EQUAL


def build_scores(method: str, factors: list[str] | dict[str, int] | None = None, h: int = HORIZON,
                 entry: str = ENTRY, neutralize: list | None = None, refresh: bool = False) -> pd.DataFrame:
    neutralize = list(NEUTRALIZE if neutralize is None else neutralize)
    available = F.available_factors()
    if method == "wf":
        log.info("Walk-forward: horizon %d days, entry %s, neutralised on %s", h, entry, neutralize or "nothing")
        sc = score_walk_forward(h, entry, neutralize, refresh)
    elif method == "equal":
        signs = _equal_signs(factors)
        bad = [n for n in signs if n not in available]
        if bad:
            raise SystemExit(f"Not in the factor table: {bad}")
        log.info("Equal weights: %s", ", ".join(f"{k}({'+' if v > 0 else '-'})" for k, v in signs.items()))
        sc = score_equal(signs, neutralize)
    else:
        names = list(factors) if factors else [n for n in available if n not in _neutral_numeric(neutralize)]
        bad = [n for n in names if n not in available]
        if bad:
            raise SystemExit(f"Not in the factor table: {bad}")
        log.info("%s: %d factors, horizon %d days, entry %s", method, len(names), h, entry)
        sc = (score_ic if method == "ic" else score_lgbm)(names, h, entry, neutralize)

    sc = sc.dropna(subset=["score"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    if sc.empty:
        raise SystemExit("No scores were produced (history too short?)")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(SCORE_DIR / f"{method}.parquet", index=False)
    log.info("Scores saved: %s to %s, %d rows", sc["date"].min().date(), sc["date"].max().date(), len(sc))
    report_score(sc, h, entry, method)
    return sc


def report_score(sc: pd.DataFrame, h: int, entry: str, name: str) -> pd.DataFrame:
    """IC and quintile test of the score itself."""
    X = sc.pivot(index="date", columns="symbol", values="score")
    rows = []
    yearly = pd.Series(dtype=float)
    for hh in sorted({5, h}):
        fwd = _forward(X.index, X.columns, hh, entry)
        ic = E.rowwise_rank_corr(X, fwd)
        if ic.empty:
            continue
        st = E.ic_stats(ic, hh)
        _, q = E.quantile_analysis(X, fwd, hh)
        rows.append({"horizon": f"{hh}d", "IC mean": st["ic_mean"], "ICIR": st["icir"], "NW t": st["t_nw"],
                     "IC>0": st["ic_pos"], "LS ann": q["LS_ann"], "LS Sharpe": q["LS_sharpe"],
                     "Q5 ann": q[f"Q{E.N_QUANTILES}_ann"], "Q1 ann": q["Q1_ann"], "top turnover": q["top_turnover"]})
        if hh == h:
            yearly = ic.groupby(ic.index.year).mean()
    if not rows:
        print("(not enough history to test the score yet)")
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("horizon")
    print(f"\n===== Combined score [{name}] (entry={entry}, before costs) =====")
    print(out.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("IC by year: " + "  ".join(f"{y}: {v:+.4f}" for y, v in yearly.items()))
    return out


def load_scores(method: str) -> pd.DataFrame:
    return pd.read_parquet(SCORE_DIR / f"{method}.parquet")


def _parse_factors(items: list[str] | None, method: str):
    if not items:
        return None
    if method == "equal":
        out = {}
        for it in items:
            if ":" not in it:
                raise SystemExit(f"equal needs a sign, e.g. ret_5:-1 (got {it})")
            k, v = it.split(":")
            out[k] = 1 if float(v) > 0 else -1
        return out
    return [it.split(":")[0] for it in items]


def main() -> None:
    p = argparse.ArgumentParser(description="Combine factors into a score")
    p.add_argument("--method", choices=["wf", "equal", "ic", "lgbm"], default="wf")
    p.add_argument("--factors", nargs="*", help="equal: name:sign (e.g. ret_5:-1); ic/lgbm: names only")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--entry", choices=["open", "vwap", "close"], default=ENTRY)
    p.add_argument("--no-neutralize", action="store_true", help="do not neutralise on sector and size")
    p.add_argument("--refresh", action="store_true", help="wf: recompute every yearly selection")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not F.FACTOR_FILE.exists():
        raise SystemExit("No factor table yet: run python factors.py")
    build_scores(args.method, _parse_factors(args.factors, args.method), args.horizon, args.entry,
                 [] if args.no_neutralize else None, args.refresh)


if __name__ == "__main__":
    main()
