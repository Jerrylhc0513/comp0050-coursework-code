#!/usr/bin/env python3
"""
backtest.py - portfolio backtest: turn model.py scores into holdings, charge costs, compare with SPY

Two portfolio profiles (PROFILES), shared with daily_signal.py so live rules == backtest rules
  model   the $100,000 paper account that takes every signal: 25 stocks, rebalanced weekly,
          inverse-volatility weights (capped), at most 6 stocks per sector, pulled back to target
          weights at every rebalance (small drifts are left alone to save costs)
  real    the small hand-traded account: 5 stocks, rebalanced monthly (every 21 trading days),
          equal weights, at most 2 per sector, kept positions are not resized (fewer orders)
Both keep a buffer: a holding is only sold when its rank falls below HOLD_RANK; free slots are filled
with the best-ranked new names.

Timeline (no look-ahead)
  after the close of day t: scores of day t -> target portfolio (every `rebalance` trading days)
  day t+1: trade at the open (or the day's VWAP); NAV is marked from fill price to fill price

Costs (per stock, on traded dollars)
  half spread HALF_SPREAD_BPS + commission COMMISSION_BPS
  + square-root impact IMPACT_COEF x daily vol x sqrt(traded dollars / 20-day ADV)
  impact grows with size, which is why each profile carries its own capital

Benchmarks (same fill-price convention, no costs): SPY and the equal-weighted universe

Usage
  python backtest.py                                   # wf scores, model profile
  python backtest.py --profile real                    # the concentrated 5-stock profile
  python backtest.py --score equal --top 30 --rebalance 10 --start 2022-01-01
  python backtest.py --mode ls                         # research: long top / short bottom
Output (reports/)
  backtest_<name>_nav.csv / _trades.csv, console summary and calendar-year returns
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import data as D
import evaluate as E
import fundamentals as FU
import model as M

HALF_SPREAD_BPS = 5.0
COMMISSION_BPS = 0.0
IMPACT_COEF = 0.5
BORROW_BPS = 50.0          # annual borrow fee for shorts in ls mode
VOL_WINDOW = 60            # inverse-volatility weights use this look-back


@dataclass(frozen=True)
class Profile:
    name: str
    top_n: int              # number of holdings
    hold_rank: int          # a holding is sold only when its rank falls below this
    rebalance: int          # trading days between rebalances
    weighting: str          # "equal" or "inv_vol"
    max_weight: float       # cap per stock (inv_vol)
    max_per_sector: int | None
    reweight: bool          # resize kept positions back to target at each rebalance
    drift_tol: float        # with reweight: leave a position alone if within this of its target weight
    capital: float          # starting capital (drives impact cost)


PROFILES = {
    "model": Profile("model", top_n=25, hold_rank=60, rebalance=5, weighting="inv_vol", max_weight=0.08,
                     max_per_sector=6, reweight=True, drift_tol=0.01, capital=100_000),
    "real": Profile("real", top_n=5, hold_rank=15, rebalance=21, weighting="equal", max_weight=0.30,
                    max_per_sector=2, reweight=False, drift_tol=0.0, capital=1_000),
}
DEFAULT_SCORE = "wf"

log = logging.getLogger("backtest")


# ============================== Selection and weights ==============================
def select_portfolio(score: pd.Series, current: list[str], top_n: int, hold_rank: int,
                     sectors: pd.Series | None = None, max_per_sector: int | None = None) -> list[str]:
    """score: today's scores of tradable stocks (higher = better). current: current holdings.
    Returns the new holdings. daily_signal.py uses the same function."""
    hold_rank = max(hold_rank, top_n)                      # the buffer cannot be tighter than the book
    ranked = score.dropna().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    count: dict = {}

    def fits(s: str) -> bool:
        if max_per_sector is None or sectors is None or s not in sectors.index:
            return True
        return count.get(sectors[s], 0) < max_per_sector

    def take(s: str) -> None:
        if sectors is not None and s in sectors.index:
            count[sectors[s]] = count.get(sectors[s], 0) + 1

    out: list[str] = []
    for s in sorted([s for s in current if s in rank.index and rank[s] <= hold_rank], key=lambda s: rank[s]):
        if len(out) < top_n and fits(s):
            out.append(s)
            take(s)
    for s in ranked.index:
        if len(out) >= top_n:
            break
        if s not in out and fits(s):
            out.append(s)
            take(s)
    return out


def target_weights(names: list[str], vol: pd.Series | None, weighting: str, max_weight: float) -> pd.Series:
    """Weights summing to 1. inv_vol: 1/vol, missing vols get the median, then capped at max_weight."""
    if not names:
        return pd.Series(dtype=float)
    if weighting == "equal" or vol is None:
        return pd.Series(1.0 / len(names), index=names)
    v = vol.reindex(names).astype(float)
    v = v.where(v > 0)
    v = v.fillna(v.median() if v.notna().any() else 1.0)
    w = 1.0 / v
    w = w / w.sum()
    cap = max(max_weight, 1.0 / len(names))
    for _ in range(20):                                     # redistribute what is above the cap
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        w[under] += excess * w[under] / w[under].sum()
    return w


def plan_weights(new_hold: list[str], current_w: pd.Series, vol: pd.Series | None, prof: Profile,
                 first: bool) -> pd.Series:
    """Target weights after a rebalance, given current weights (fractions of NAV, cash excluded)."""
    tw = target_weights(new_hold, vol, prof.weighting, prof.max_weight)
    if first or prof.reweight:
        cur = current_w.reindex(new_hold).fillna(0.0)
        frozen = [s for s in new_hold if cur[s] > 0 and abs(tw[s] - cur[s]) < prof.drift_tol]
        if frozen and len(frozen) < len(new_hold):
            rest = [s for s in new_hold if s not in frozen]
            room = max(0.0, 1.0 - cur[frozen].sum())
            tw[frozen] = cur[frozen]
            tw[rest] = tw[rest] / tw[rest].sum() * room
        return tw
    kept = [s for s in new_hold if current_w.get(s, 0.0) > 0]
    new = [s for s in new_hold if s not in kept]
    out = pd.Series(0.0, index=new_hold)
    out[kept] = current_w.reindex(kept).values
    freed = max(0.0, 1.0 - out.sum())
    if new:
        out[new] = tw[new] / tw[new].sum() * freed
    return out


# ============================== Engine ==============================
def _panels(symbols: list[str], entry: str):
    daily = D.load_daily(symbols=symbols, adjusted=True, columns=["open", "close", "volume", "vwap", "ret"])
    piv = lambda c: daily.pivot(index="date", columns="symbol", values=c).sort_index()
    if entry == "vwap":
        daily["adj_vwap"] = daily["vwap"] * daily["adj_close"] / daily["close"]
    adj_col = {"open": "adj_open", "vwap": "adj_vwap", "close": "adj_close"}[entry]
    raw_col = {"open": "open", "vwap": "vwap", "close": "close"}[entry]
    P = piv(adj_col)
    raw = piv(raw_col)
    ret = piv("ret")
    sigma = ret.rolling(20, min_periods=10).std()
    vol = ret.rolling(VOL_WINDOW, min_periods=40).std()
    adv = (piv("close") * piv("volume")).rolling(20, min_periods=10).mean()
    return P, raw, sigma, vol, adv


def run_backtest(scores: pd.DataFrame, prof: Profile, entry: str = "open", mode: str = "long",
                 start=None, end=None, half_spread_bps: float = HALF_SPREAD_BPS,
                 commission_bps: float = COMMISSION_BPS, impact_coef: float = IMPACT_COEF,
                 borrow_bps: float = BORROW_BPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    if start is not None:
        scores = scores[scores["date"] >= pd.Timestamp(start)]
    if end is not None:
        scores = scores[scores["date"] <= pd.Timestamp(end)]
    S = scores.pivot(index="date", columns="symbol", values="score").sort_index()
    if S.empty:
        raise SystemExit("No scores in the selected period")
    P, raw, sigma, vol, adv = _panels(list(S.columns), entry)
    cols = S.columns
    P, raw, sigma, vol, adv = (x.reindex(columns=cols) for x in (P, raw, sigma, vol, adv))
    Pf = P.ffill()
    R = (Pf / Pf.shift(1) - 1).fillna(0.0)                 # suspended days earn 0
    sectors = FU.load_sectors()
    sectors = sectors if len(sectors) else None

    dates = P.index[(P.index >= S.index[0])]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    sig_dates = [d for d in S.index if d in dates][::prof.rebalance]
    exec_of = {}
    for t in sig_dates:
        i = dates.get_loc(t)
        if i + 1 < len(dates):
            exec_of[dates[i + 1]] = t

    n = len(cols)
    w = np.zeros(n)
    nav, nav_g = float(prof.capital), float(prof.capital)
    hold: list[str] = []
    recs, trades = [], []
    col_index = {s: k for k, s in enumerate(cols)}
    started = False
    for d in dates:
        r = R.loc[d].to_numpy()
        if started:
            pnl = float(w @ r)
            borrow = borrow_bps / 1e4 / 252 * float(-w[w < 0].sum()) if mode == "ls" else 0.0
            nav *= 1 + pnl - borrow
            nav_g *= 1 + pnl
            denom = 1 + pnl
            w = w * (1 + r) / denom if denom != 0 else w
        turnover = cost = 0.0
        if d in exec_of:
            t = exec_of[d]
            tradable = P.loc[d].notna()
            sc = S.loc[t].where(tradable.reindex(cols, fill_value=False))
            target = np.zeros(n)
            if mode == "long":
                new_hold = select_portfolio(sc, hold, prof.top_n, prof.hold_rank, sectors, prof.max_per_sector)
                cur = pd.Series(w, index=cols)
                cur = cur[cur > 1e-12]
                tw = plan_weights(new_hold, cur, vol.loc[t], prof, first=not started)
                for s, v in tw.items():
                    target[col_index[s]] = v
                hold = new_hold
            else:
                ranked = sc.dropna().sort_values(ascending=False)
                k = min(prof.top_n, len(ranked) // 2)
                for s in ranked.index[:k]:
                    target[col_index[s]] = 1.0 / k
                for s in ranked.index[-k:]:
                    target[col_index[s]] = -1.0 / k
                hold = list(ranked.index[:k]) + list(ranked.index[-k:])

            dw = target - w
            trade_usd = np.abs(dw) * nav
            sig = sigma.loc[t].to_numpy()
            a = adv.loc[t].to_numpy()
            impact = np.where(np.isfinite(sig) & np.isfinite(a) & (a > 0),
                              impact_coef * np.nan_to_num(sig) * np.sqrt(trade_usd / np.where(a > 0, a, 1)), 0.0)
            rate = (half_spread_bps + commission_bps) / 1e4 + impact
            cost = float((rate * trade_usd).sum())
            turnover = float(np.abs(dw).sum()) / 2
            nav_pre = nav
            nav -= cost                                      # costs are paid from cash
            w = target * nav_pre / nav
            started = True
            for k_ in np.flatnonzero(np.abs(dw) > 1e-9):
                w0, w1 = target[k_] - dw[k_], target[k_]
                if w0 == 0:
                    side = "BUY" if w1 > 0 else "SHORT"
                elif w1 == 0:
                    side = "SELL" if w0 > 0 else "COVER"
                else:
                    side = "ADD" if abs(w1) > abs(w0) else "TRIM"
                trades.append({"exec_date": d, "signal_date": t, "symbol": cols[k_], "side": side,
                               "weight_from": w0, "weight_to": w1, "price": raw.at[d, cols[k_]],
                               "cost_bps": rate[k_] * 1e4})
        if started:
            recs.append({"date": d, "nav": nav, "nav_gross": nav_g, "turnover": turnover,
                         "cost": cost, "n_hold": int((np.abs(w) > 1e-12).sum()),
                         "gross_exposure": float(np.abs(w).sum())})

    navdf = pd.DataFrame(recs).set_index("date")
    navdf = _add_benchmarks(navdf, R, entry)
    return navdf, pd.DataFrame(trades)


def _add_benchmarks(navdf: pd.DataFrame, R: pd.DataFrame, entry: str) -> pd.DataFrame:
    idx = navdf.index
    # equal-weighted universe: yesterday's members, today's average return
    uni = D.load_universe()
    U = uni.assign(v=True).pivot(index="date", columns="symbol", values="v").reindex(index=R.index, columns=R.columns)
    U = U.notna().shift(1, fill_value=False)
    ew = R.where(U).mean(axis=1).reindex(idx).fillna(0.0)
    ew.iloc[0] = 0.0
    navdf["ew"] = navdf["nav_gross"].iloc[0] * (1 + ew).cumprod()
    if D.BENCHMARK_FILE.exists():
        b = D.load_benchmark("SPY").set_index("date")
        col = {"open": "adj_open", "close": "adj_close", "vwap": None}[entry]
        if col is None:
            b["adj_vwap"] = b["vwap"] * b["adj_close"] / b["close"]
            col = "adj_vwap"
        px = b[col].reindex(idx).ffill()
        rb = (px / px.shift(1) - 1).fillna(0.0)
        navdf["spy"] = navdf["nav_gross"].iloc[0] * (1 + rb).cumprod()
    return navdf


# ============================== Metrics ==============================
def perf(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    r = nav.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {}
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = nav / nav.cummax() - 1
    out = {"ann_return": ann, "ann_vol": vol, "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
           "max_drawdown": dd.min(), "calmar": ann / abs(dd.min()) if dd.min() < 0 else np.nan}
    if bench is not None:
        rb = bench.pct_change().reindex(r.index).fillna(0.0)
        act = r - rb
        te = act.std() * np.sqrt(252)
        out.update({"excess_ann": act.mean() * 252, "tracking_error": te,
                    "info_ratio": act.mean() * 252 / te if te > 0 else np.nan,
                    "beta": np.cov(r, rb)[0, 1] / rb.var() if rb.var() > 0 else np.nan})
    return out


def report(navdf: pd.DataFrame, trades: pd.DataFrame, name: str, capital: float) -> pd.DataFrame:
    bench_col = "spy" if "spy" in navdf else "ew"
    rows = {"strategy (net)": perf(navdf["nav"], navdf[bench_col]),
            "strategy (gross)": perf(navdf["nav_gross"], navdf[bench_col]),
            "universe equal-weight": perf(navdf["ew"], navdf[bench_col] if bench_col == "spy" else None)}
    if "spy" in navdf:
        rows["SPY"] = perf(navdf["spy"])
    tab = pd.DataFrame(rows).T
    days = len(navdf)
    extra = {"annual turnover (one-way)": navdf["turnover"].sum() * 252 / days,
             "cost drag (annual)": (navdf["cost"] / navdf["nav"].shift(1).fillna(capital)).sum() * 252 / days,
             "avg holdings": navdf["n_hold"].mean(),
             "rebalances": int((navdf["turnover"] > 0).sum()),
             "trades": len(trades)}
    pd.set_option("display.width", 200)
    pct = {"ann_return", "ann_vol", "max_drawdown", "excess_ann", "tracking_error"}
    show = tab.copy().astype(object)
    for c in tab.columns:
        show[c] = tab[c].map(lambda v, c=c: "" if pd.isna(v) else (f"{v:+.1%}" if c in pct else f"{v:+.2f}"))
    print(f"\n===== Backtest [{name}]: {navdf.index[0].date()} to {navdf.index[-1].date()}, "
          f"benchmark = {'SPY' if bench_col == 'spy' else 'universe equal-weight'} =====")
    print(show.to_string())
    print("  ".join(f"{k}: {v:.1%}" if "turnover" in k or "cost" in k else f"{k}: {v:.0f}" for k, v in extra.items()))

    cols = {"nav": "strategy net", "nav_gross": "strategy gross", "ew": "universe EW", "spy": "SPY"}
    yearly = {}
    for c, label in cols.items():
        if c not in navdf:
            continue
        last = navdf[c].groupby(navdf.index.year).last()
        base = last.shift(1)
        base.iloc[0] = navdf[c].iloc[0]
        yearly[label] = last / base - 1
    ytab = pd.DataFrame(yearly)
    print("\nCalendar-year returns:")
    print(ytab.to_string(float_format=lambda v: f"{v:+.1%}"))
    print("\nNote: Sharpe ignores the risk-free rate. Only the 'wf' score is fully out-of-sample; "
          "for other scores use --start after the period used to pick factors.")
    return tab


def main() -> None:
    p = argparse.ArgumentParser(description="Portfolio backtest")
    p.add_argument("--score", default=DEFAULT_SCORE, help="score file in data/scores/ (wf / equal / ic / lgbm)")
    p.add_argument("--profile", choices=list(PROFILES), default="model")
    p.add_argument("--mode", choices=["long", "ls"], default="long")
    p.add_argument("--entry", choices=["open", "vwap", "close"], default="open")
    p.add_argument("--top", type=int, help="override the profile's number of holdings")
    p.add_argument("--hold-rank", type=int)
    p.add_argument("--rebalance", type=int, help="trading days between rebalances")
    p.add_argument("--capital", type=float)
    p.add_argument("--weighting", choices=["equal", "inv_vol"])
    p.add_argument("--spread-bps", type=float, default=HALF_SPREAD_BPS, help="half spread in bps")
    p.add_argument("--impact", type=float, default=IMPACT_COEF, help="square-root impact coefficient, 0 = none")
    p.add_argument("--start")
    p.add_argument("--end")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    prof = PROFILES[args.profile]
    over = {k: v for k, v in {"top_n": args.top, "hold_rank": args.hold_rank, "rebalance": args.rebalance,
                              "capital": args.capital, "weighting": args.weighting}.items() if v is not None}
    prof = replace(prof, **over)
    scores = M.load_scores(args.score)
    navdf, trades = run_backtest(scores, prof, args.entry, args.mode, args.start, args.end,
                                 half_spread_bps=args.spread_bps, impact_coef=args.impact)
    name = f"{args.score}_{prof.name}_{args.mode}_top{prof.top_n}_reb{prof.rebalance}_{args.entry}"
    E.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    navdf.to_csv(E.REPORT_DIR / f"backtest_{name}_nav.csv")
    trades.to_csv(E.REPORT_DIR / f"backtest_{name}_trades.csv", index=False)
    report(navdf, trades, name, prof.capital)
    print(f"Details saved to {E.REPORT_DIR}/backtest_{name}_*.csv")


if __name__ == "__main__":
    main()
