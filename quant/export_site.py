#!/usr/bin/env python3
"""
export_site.py - export what the web page needs into site/ (a small snapshot)

app.py only reads site/, so the same app.py runs locally and on Streamlit Community Cloud:
push site/ to GitHub and the page updates. Raw data (data/), raw holdings (portfolio/) and .env
are never uploaded.

Contents
  meta.json             update time, data date, strategy settings of both accounts
  ranking.csv           today's ranking (top 50 scores)
  model/, real/         NAV, holdings, trades and the latest trade list of each account
  spy.csv               SPY adjusted close
  backtest_nav.csv ...  one backtest (the newest by default)
  factors/              factor test results and the selected factors

Settings (.env)
  SITE_SHOW_REAL_AMOUNTS=1   show the real account's dollar amounts and shares (default);
                             0 = only returns and weights
  SITE_BACKTEST=<name>       which backtest to show: the middle part of reports/backtest_<name>_nav.csv

Usage: python export_site.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import daily_signal as S
import data as D
import evaluate as E

SITE_DIR = Path(os.environ.get("SITE_DIR", Path(__file__).resolve().parent / "site"))
log = logging.getLogger("export")


def _show_real_amounts() -> bool:
    return os.environ.get("SITE_SHOW_REAL_AMOUNTS", "1") == "1"


def _latest_orders(acct: S.Account) -> pd.DataFrame | None:
    files = sorted(acct.orders_dir.glob("*.csv")) if acct.orders_dir.exists() else []
    return pd.read_csv(files[-1]) if files else None


def export_account(acct: S.Account, mk: S.Market, out: Path, show_amounts: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    info = {"name": acct.name, "exists": acct.holdings.exists()}
    if not acct.holdings.exists():
        return info
    pos, cash = S.load_holdings(acct)
    pos = S.value_positions(pos, mk.close)
    nav = cash + pos["value"].sum()
    pos["weight"] = pos["value"] / nav if nav else np.nan
    hist = pd.read_csv(acct.history) if acct.history.exists() else pd.DataFrame(columns=["date", "nav"])
    trades = pd.read_csv(acct.trade_log) if acct.trade_log.exists() else pd.DataFrame()
    orders = _latest_orders(acct)
    st = S.load_state(acct)
    info.update({"last_rebalance": st.get("last_rebalance"), "pending": bool(st.get("pending")),
                 "n_hold": int(len(pos)), "show_amounts": show_amounts})

    if show_amounts:
        info.update({"nav": round(nav, 2), "cash": round(cash, 2)})
        hist.to_csv(out / "history.csv", index=False)
        pos[["symbol", "shares", "cost_basis", "price", "value", "weight", "pnl_pct"]].to_csv(out / "holdings.csv", index=False)
        trades.to_csv(out / "trades.csv", index=False)
        if orders is not None:
            orders.to_csv(out / "orders_latest.csv", index=False)
    else:                                     # publish returns and weights only
        if len(hist):
            hist = hist.assign(nav=(hist["nav"] / hist["nav"].iloc[0] * 100).round(3))[["date", "nav"]]
        hist.to_csv(out / "history.csv", index=False)
        pos[["symbol", "price", "weight", "pnl_pct"]].to_csv(out / "holdings.csv", index=False)
        keep = [c for c in ["record_time", "signal_date", "symbol", "side", "price", "slippage_bps"] if c in trades]
        trades[keep].to_csv(out / "trades.csv", index=False)
        if orders is not None:
            orders.drop(columns=["shares", "amount_usd"], errors="ignore").to_csv(out / "orders_latest.csv", index=False)
    return info


def export() -> Path:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir(parents=True)
    mk = S.load_market()
    method = S.SCORE_METHOD
    score_date, scores = S.latest_scores(method)

    accounts = {}
    for acct in [S.MODEL, S.REAL]:
        show = True if acct.auto_fill else _show_real_amounts()
        accounts[acct.key] = export_account(acct, mk, SITE_DIR / acct.key, show)

    if len(scores):
        ranked = scores.dropna().sort_values(ascending=False).head(50)
        held = {k: set(pd.read_csv(SITE_DIR / k / "holdings.csv")["symbol"]) if (SITE_DIR / k / "holdings.csv").exists()
                else set() for k in ("model", "real")}
        pd.DataFrame({"rank": range(1, len(ranked) + 1), "symbol": ranked.index, "score": ranked.values.round(4),
                      "close": [mk.close.get(s, np.nan) for s in ranked.index],
                      "model_held": [s in held["model"] for s in ranked.index],
                      "real_held": [s in held["real"] for s in ranked.index]}).to_csv(SITE_DIR / "ranking.csv", index=False)

    bt_name = None
    navs = sorted(E.REPORT_DIR.glob("backtest_*_nav.csv"), key=lambda f: f.stat().st_mtime) if E.REPORT_DIR.exists() else []
    want = os.environ.get("SITE_BACKTEST")
    pick = [f for f in navs if want and f.name == f"backtest_{want}_nav.csv"] or navs[-1:]
    if pick:
        f = pick[0]
        bt_name = f.name[len("backtest_"):-len("_nav.csv")]
        shutil.copy(f, SITE_DIR / "backtest_nav.csv")
        tf = E.REPORT_DIR / f"backtest_{bt_name}_trades.csv"
        if tf.exists():
            pd.read_csv(tf).tail(2000).to_csv(SITE_DIR / "backtest_trades.csv", index=False)

    starts = []
    for k in ("model", "real"):
        h = SITE_DIR / k / "history.csv"
        if h.exists():
            d = pd.read_csv(h)["date"]
            if len(d):
                starts.append(pd.Timestamp(d.min()))
    if (SITE_DIR / "backtest_nav.csv").exists():
        starts.append(pd.Timestamp(pd.read_csv(SITE_DIR / "backtest_nav.csv")["date"].min()))
    if D.BENCHMARK_FILE.exists() and starts:
        spy = D.load_benchmark("SPY")
        spy = spy[spy["date"] >= min(starts) - pd.Timedelta(days=7)]
        spy[["date", "adj_close"]].rename(columns={"adj_close": "spy"}).to_csv(SITE_DIR / "spy.csv", index=False)

    fdir = SITE_DIR / "factors"
    fdir.mkdir()
    for name in ["factor_summary.csv", "ic_decay.csv", "ic_by_year.csv", "factor_corr.csv",
                 "selection_table.csv", "selected_factors.json"]:
        if (E.REPORT_DIR / name).exists():
            shutil.copy(E.REPORT_DIR / name, fdir / name)
    for name in ["ic_daily", "ls_curve"]:
        if (E.REPORT_DIR / f"{name}.parquet").exists():
            pd.read_parquet(E.REPORT_DIR / f"{name}.parquet").round(6).to_csv(fdir / f"{name}.csv")

    profiles = {k: {"top_n": p.top_n, "hold_rank": p.hold_rank, "rebalance": p.rebalance,
                    "weighting": p.weighting, "max_per_sector": p.max_per_sector, "capital": p.capital}
                for k, p in B.PROFILES.items()}
    meta = {"updated": datetime.now().isoformat(timespec="minutes"), "data_date": mk.last.date().isoformat(),
            "score_date": score_date.date().isoformat() if score_date is not None else None,
            "method": method, "profiles": profiles,
            "universe_size": D.UNIVERSE_SIZE, "model_capital": S.MODEL_CAPITAL, "backtest": bt_name,
            "costs": {"half_spread_bps": B.HALF_SPREAD_BPS, "impact_coef": B.IMPACT_COEF},
            "accounts": accounts}
    (SITE_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in SITE_DIR.rglob("*") if f.is_file())
    log.info("Site data exported to %s (%.1f MB)", SITE_DIR, size / 1e6)
    return SITE_DIR


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    export()
