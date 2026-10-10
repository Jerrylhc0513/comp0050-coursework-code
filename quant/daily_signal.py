#!/usr/bin/env python3
"""
daily_signal.py - daily trade lists for two accounts, using the rules in backtest.PROFILES

  model account   portfolio/model/   $100,000 paper account that takes EVERY signal (profile "model":
                                     25 stocks, weekly, inverse-vol weights). Orders fill automatically at
                                     the next day's real open, with half spread + square-root impact
                                     (same parameters as backtest.py). Booked daily from the first run and
                                     never rewritten: a true out-of-sample record.
  real account    portfolio/         the small hand-traded account (profile "real": 5 stocks, monthly,
                                     equal weights). You place the orders and enter the fills with `record`.

Run once a day after the close (run_daily.bat does it):
  1. model account: fill yesterday's orders at today's open
  2. both accounts: mark to today's close, append to account_history.csv
  3. for each account whose rebalance is due (its own `rebalance` interval), make a trade list:
       sell: holdings that fell below the account's HOLD_RANK or lost their score (left the universe)
       buy:  the best-ranked new stocks until the account holds TOP_N, sized by the profile's weights
  4. write the HTML report (reports/daily_report.html) and email it

Commands
  python daily_signal.py                       today's signals (non-rebalance days only update values)
  python daily_signal.py --force               force a trade list for both accounts today
  python daily_signal.py record                enter the real fills of the latest real-account list
  python daily_signal.py record --as-planned   record the suggested shares at the reference prices
  python daily_signal.py status                holdings and NAV of both accounts
  python daily_signal.py test-email            send a test email
  python daily_signal.py notify-error <log>    email the end of a log (used by run_daily.bat on failure)

Settings (.env)
  EMAIL_SENDER / EMAIL_APP_PASSWORD / EMAIL_TO   sender, app password, recipient
  EMAIL_DAILY=1     also email on days without a rebalance (default: only rebalances, warnings, errors)
  SCORE_METHOD      which score to use (wf / equal / ic / lgbm), default wf
  MODEL_ACCOUNT=0   switch the model account off
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import data as D
import fundamentals as FU
import model as M

# ============================== Settings ==============================
SCORE_METHOD = os.environ.get("SCORE_METHOD", "wf")
INITIAL_CASH = 1_000.0       # real account cash on the very first run
CASH_BUFFER = 0.02           # real account keeps 2% cash in case the open is above the reference price
MODEL_CAPITAL = float(B.PROFILES["model"].capital)
MODEL_ENABLED = os.environ.get("MODEL_ACCOUNT", "1") == "1"
MIN_ORDER_USD = 1.0          # Alpaca minimum notional order

PORTFOLIO_DIR = Path(os.environ.get("PORTFOLIO_DIR", Path(__file__).resolve().parent / "portfolio"))
REPORT_FILE = Path(os.environ.get("QUANT_REPORT_DIR", Path(__file__).resolve().parent / "reports")) / "daily_report.html"


@dataclass(frozen=True)
class Account:
    key: str
    name: str
    dir: Path
    initial_cash: float
    cash_buffer: float
    auto_fill: bool          # True = model account, orders fill automatically at the open
    profile: B.Profile

    @property
    def holdings(self) -> Path: return self.dir / "holdings.csv"
    @property
    def state(self) -> Path: return self.dir / "state.json"
    @property
    def history(self) -> Path: return self.dir / "account_history.csv"
    @property
    def trade_log(self) -> Path: return self.dir / "trade_log.csv"
    @property
    def orders_dir(self) -> Path: return self.dir / "orders"
    @property
    def signals_dir(self) -> Path: return self.dir / "signals"


REAL = Account("real", "Real account", PORTFOLIO_DIR, INITIAL_CASH, CASH_BUFFER, False, B.PROFILES["real"])
MODEL = Account("model", "Model account", PORTFOLIO_DIR / "model", MODEL_CAPITAL, 0.0, True, B.PROFILES["model"])

SMTP_HOSTS = {"gmail.com": "smtp.gmail.com", "163.com": "smtp.163.com", "126.com": "smtp.126.com",
              "qq.com": "smtp.qq.com", "foxmail.com": "smtp.qq.com",
              "outlook.com": "smtp-mail.outlook.com", "hotmail.com": "smtp-mail.outlook.com"}

log = logging.getLogger("signal")
_input = input               # replaceable in tests


# ============================== Holdings and state ==============================
def load_holdings(acct: Account = REAL) -> tuple[pd.DataFrame, float]:
    if not acct.holdings.exists():
        save_holdings(pd.DataFrame(columns=["symbol", "shares", "cost_basis"]), acct.initial_cash, acct)
        log.info("Created %s (%s) with $%.2f cash", acct.holdings, acct.name, acct.initial_cash)
    df = pd.read_csv(acct.holdings, dtype={"symbol": str})
    df["symbol"] = df["symbol"].str.strip().str.upper()
    cash = float(df.loc[df["symbol"] == "CASH", "shares"].sum())
    pos = df[(df["symbol"] != "CASH") & (df["shares"].abs() > 1e-9)].reset_index(drop=True)
    return pos[["symbol", "shares", "cost_basis"]], cash


def save_holdings(pos: pd.DataFrame, cash: float, acct: Account = REAL) -> None:
    out = pd.concat([pos[["symbol", "shares", "cost_basis"]],
                     pd.DataFrame([{"symbol": "CASH", "shares": round(cash, 2), "cost_basis": 1.0}])],
                    ignore_index=True)
    acct.dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(acct.holdings, index=False)


def load_state(acct: Account = REAL) -> dict:
    if acct.state.exists():
        return json.loads(acct.state.read_text(encoding="utf-8"))
    return {"last_rebalance": None, "pending": None}


def save_state(state: dict, acct: Account = REAL) -> None:
    acct.dir.mkdir(parents=True, exist_ok=True)
    acct.state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_trades(rows: list[dict], acct: Account) -> None:
    if not rows:
        return
    cols = ["record_time", "signal_date", "symbol", "side", "shares", "price", "ref_price", "slippage_bps", "cost_bps"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(acct.trade_log, mode="a", header=not acct.trade_log.exists(),
                                                    index=False)


# ============================== Market data and scores ==============================
@dataclass
class Market:
    last: pd.Timestamp
    dates: list
    close: pd.Series             # latest raw close per stock
    O: pd.DataFrame              # raw opens, date x symbol
    sigma: pd.DataFrame          # 20-day return vol (impact cost)
    vol: pd.DataFrame            # VOL_WINDOW-day return vol (inverse-vol weights)
    adv: pd.DataFrame            # 20-day average dollar volume


def load_market(n_files: int = 5) -> Market:
    files = D._daily_files()
    if not files:
        raise SystemExit("No daily data yet: run python data.py update")
    df = pd.concat([pd.read_parquet(f, columns=["date", "symbol", "open", "close", "volume", "ret"])
                    for f in files[-n_files:]]).sort_values("date")
    piv = lambda c: df.pivot(index="date", columns="symbol", values=c).sort_index()
    Cl = piv("close")
    R = piv("ret")
    return Market(last=pd.Timestamp(df["date"].max()), dates=[pd.Timestamp(d) for d in Cl.index],
                  close=df.groupby("symbol")["close"].last(), O=piv("open"),
                  sigma=R.rolling(20, min_periods=10).std(),
                  vol=R.rolling(B.VOL_WINDOW, min_periods=40).std(),
                  adv=(Cl * piv("volume")).rolling(20, min_periods=10).mean())


def latest_scores(method: str) -> tuple[pd.Timestamp | None, pd.Series]:
    f = M.SCORE_DIR / f"{method}.parquet"
    if not f.exists():
        return None, pd.Series(dtype=float)
    sc = pd.read_parquet(f)
    d = sc["date"].max()
    return pd.Timestamp(d), sc[sc["date"] == d].set_index("symbol")["score"]


def value_positions(pos: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    pos = pos.copy()
    pos["price"] = pos["symbol"].map(px).fillna(pos["cost_basis"])
    pos["value"] = pos["shares"] * pos["price"]
    pos["pnl_pct"] = pos["price"] / pos["cost_basis"] - 1
    return pos


def record_history(date: pd.Timestamp, nav: float, cash: float, pos_value: float, n_hold: int,
                   acct: Account = REAL) -> None:
    row = pd.DataFrame([{"date": date.date().isoformat(), "nav": round(nav, 2), "cash": round(cash, 2),
                         "positions_value": round(pos_value, 2), "n_hold": n_hold}])
    if acct.history.exists():
        h = pd.read_csv(acct.history)
        h = pd.concat([h[h["date"] != row["date"].iloc[0]], row], ignore_index=True).sort_values("date")
    else:
        h = row
    acct.dir.mkdir(parents=True, exist_ok=True)
    h.to_csv(acct.history, index=False)


def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    n = d + pd.Timedelta(days=1)
    while n.weekday() >= 5:
        n += pd.Timedelta(days=1)
    return n


# ============================== Orders ==============================
def make_orders(pos: pd.DataFrame, cash: float, scores: pd.Series, prof: B.Profile, vol: pd.Series | None = None,
                sectors: pd.Series | None = None, cash_buffer: float = CASH_BUFFER) -> tuple[pd.DataFrame, list[str]]:
    """pos must already be valued (value_positions). Returns (orders, new holdings)."""
    held = list(pos["symbol"])
    ranked = scores.dropna().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    new_hold = B.select_portfolio(scores, held, prof.top_n, prof.hold_rank, sectors, prof.max_per_sector)
    p = pos.set_index("symbol")
    nav = cash + (float(p["value"].sum()) if len(p) else 0.0)
    cur_w = (p["value"] / nav) if nav > 0 and len(p) else pd.Series(dtype=float)
    tw = B.plan_weights(new_hold, cur_w, vol, prof, first=len(held) == 0)
    sells = [s for s in held if s not in new_hold]

    rows = []
    proceeds = 0.0
    for s in sells:
        sh, price = float(p.at[s, "shares"]), float(p.at[s, "price"])
        reason = f"rank {rank[s]}, below the top {prof.hold_rank}" if s in rank.index else \
            "no score (left the universe or suspended)"
        rows.append({"action": "SELL", "symbol": s, "shares": sh, "amount_usd": round(sh * price, 2),
                     "ref_price": price, "rank": rank.get(s, np.nan), "score": scores.get(s, np.nan), "reason": reason})
        proceeds += sh * price

    if prof.reweight or not held:
        investable = nav * (1 - cash_buffer)
        tol = max(MIN_ORDER_USD, prof.drift_tol * nav)
        for s in new_hold:
            have = float(p.at[s, "value"]) if s in p.index else 0.0
            diff = float(tw[s]) * investable - have
            if s in p.index and abs(diff) < tol:
                continue
            if s not in p.index and diff < MIN_ORDER_USD:
                continue
            price = float(p.at[s, "price"]) if s in p.index else np.nan
            act = "BUY" if s not in p.index else ("ADD" if diff > 0 else "TRIM")
            rows.append({"action": act, "symbol": s, "shares": np.nan, "amount_usd": round(abs(diff), 2),
                         "ref_price": price, "rank": rank[s], "score": scores[s],
                         "reason": f"rank {rank[s]}, target weight {tw[s]:.1%}"})
    else:
        new = [s for s in new_hold if s not in p.index]
        budget = (cash + proceeds) * (1 - cash_buffer)
        wsum = float(tw[new].sum()) if new else 0.0
        for s in new:
            amt = budget * float(tw[s]) / wsum if wsum > 0 else 0.0
            if amt < MIN_ORDER_USD:
                continue
            rows.append({"action": "BUY", "symbol": s, "shares": np.nan, "amount_usd": round(amt, 2),
                         "ref_price": np.nan, "rank": rank[s], "score": scores[s], "reason": f"rank {rank[s]}"})
    orders = pd.DataFrame(rows, columns=["action", "symbol", "shares", "amount_usd", "ref_price", "rank", "score", "reason"])
    return orders, new_hold


def _fill_ref_prices(orders: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    orders = orders.copy()
    miss = orders["ref_price"].isna()
    orders.loc[miss, "ref_price"] = orders.loc[miss, "symbol"].map(px)
    need = orders["action"].isin(["BUY", "ADD", "TRIM"]) & orders["shares"].isna()
    orders.loc[need, "shares"] = (orders.loc[need, "amount_usd"] / orders.loc[need, "ref_price"]).round(4)
    return orders


# ============================== Model account auto-fill ==============================
def _cost_rate(mk: Market, t: pd.Timestamp, sym: str, usd: float) -> float:
    rate = (B.HALF_SPREAD_BPS + B.COMMISSION_BPS) / 1e4
    try:
        sig, a = float(mk.sigma.at[t, sym]), float(mk.adv.at[t, sym])
    except KeyError:
        return rate
    if math.isfinite(sig) and math.isfinite(a) and a > 0:
        rate += B.IMPACT_COEF * sig * math.sqrt(usd / a)
    return rate


def fill_model(acct: Account, mk: Market) -> list[dict]:
    """Fill pending orders at the open of the first trading day after the signal (with costs).
    Orders for stocks without an open price are skipped."""
    st = load_state(acct)
    if not st.get("pending"):
        return []
    orders = pd.read_csv(acct.orders_dir / st["pending"])
    if orders.empty:
        st["pending"] = None
        save_state(st, acct)
        return []
    t = pd.Timestamp(orders["signal_date"].iloc[0])
    later = [d for d in mk.dates if d > t]
    if not later:
        return []                                   # the fill day has not come yet
    ex = later[0]
    pos, cash = load_holdings(acct)
    p = pos.set_index("symbol")
    fills = []

    def open_px(sym):
        try:
            v = float(mk.O.at[ex, sym])
        except KeyError:
            return None
        return v if math.isfinite(v) and v > 0 else None

    for _, o in orders[orders["action"].isin(["SELL", "TRIM"])].iterrows():
        sym, op = o["symbol"], open_px(o["symbol"])
        if op is None or sym not in p.index:
            log.warning("Model account: %s has no open on %s, sell skipped", sym, ex.date())
            continue
        have = float(p.at[sym, "shares"])
        sh = have if o["action"] == "SELL" else min(float(o["shares"]), have)
        c = _cost_rate(mk, t, sym, sh * op)
        fp = op * (1 - c)
        cash += sh * fp
        p.at[sym, "shares"] = have - sh
        fills.append({"record_time": ex.date().isoformat(), "signal_date": t.date().isoformat(), "symbol": sym,
                      "side": "SELL", "shares": round(sh, 6), "price": round(fp, 4), "ref_price": o["ref_price"],
                      "slippage_bps": round((o["ref_price"] / fp - 1) * 1e4, 1), "cost_bps": round(c * 1e4, 2)})

    all_buys = orders[orders["action"].isin(["BUY", "ADD"])]
    buys = all_buys[[open_px(s) is not None for s in all_buys["symbol"]]]
    for s in all_buys["symbol"]:
        if open_px(s) is None:
            log.warning("Model account: %s has no open on %s, buy skipped; the cash waits for the next rebalance",
                        s, ex.date())
    if len(buys):
        # scale planned amounts down if the open moved against us; an unfilled buy leaves its cash in place
        budget = cash * (1 - acct.cash_buffer)
        planned = all_buys["amount_usd"].astype(float)
        amounts = buys["amount_usd"].astype(float) * min(1.0, budget / planned.sum())
        for i, o in buys.iterrows():
            sym, op, amt = o["symbol"], open_px(o["symbol"]), float(amounts[i])
            c = _cost_rate(mk, t, sym, amt)
            fp = op * (1 + c)
            sh = amt / fp
            old_sh = float(p.at[sym, "shares"]) if sym in p.index else 0.0
            old_cb = float(p.at[sym, "cost_basis"]) if sym in p.index else 0.0
            p.loc[sym, ["shares", "cost_basis"]] = [old_sh + sh, (old_sh * old_cb + sh * fp) / (old_sh + sh)]
            cash -= amt
            fills.append({"record_time": ex.date().isoformat(), "signal_date": t.date().isoformat(), "symbol": sym,
                          "side": "BUY", "shares": round(sh, 6), "price": round(fp, 4), "ref_price": o["ref_price"],
                          "slippage_bps": round((fp / o["ref_price"] - 1) * 1e4, 1), "cost_bps": round(c * 1e4, 2)})
    pos = p.reset_index()
    save_holdings(pos[pos["shares"] > 1e-9], cash, acct)
    append_trades(fills, acct)
    st["pending"] = None
    save_state(st, acct)
    log.info("Model account: %d fills at the open of %s", len(fills), ex.date())
    return fills


# ============================== Text ==============================
def _fmt_orders(orders: pd.DataFrame) -> str:
    lines = []
    sells = orders[orders["action"].isin(["SELL", "TRIM"])]
    buys = orders[orders["action"].isin(["BUY", "ADD"])]
    if len(sells):
        lines.append("Sell first:")
        for _, o in sells.iterrows():
            what = f"all {o.shares:g} sh" if o.action == "SELL" else f"{o.shares:g} sh"
            lines.append(f"  {o.symbol:<6} {what:<14} ref ${o.ref_price:,.2f}  ~ ${o.amount_usd:,.2f}   {o.reason}")
    if len(buys):
        lines.append("Then buy (dollar-amount orders):")
        for _, o in buys.iterrows():
            lines.append(f"  {o.symbol:<6} ${o.amount_usd:,.2f}   ~ {o.shares:g} sh  ref ${o.ref_price:,.2f}   {o.reason}")
    if not lines:
        lines.append("No trades needed: every holding is still inside the buffer.")
    return "\n".join(lines)


def _fmt_holdings(pos: pd.DataFrame, cash: float) -> str:
    if pos.empty:
        return f"No positions, cash ${cash:,.2f}"
    nav = cash + pos["value"].sum()
    lines = [f"{'symbol':<6} {'shares':>10} {'price':>9} {'value':>10} {'weight':>6} {'P&L':>7}"]
    for _, r in pos.sort_values("value", ascending=False).iterrows():
        lines.append(f"{r.symbol:<6} {r.shares:>10.4f} {r.price:>9.2f} {r.value:>10.2f} "
                     f"{r.value / nav:>6.1%} {r.pnl_pct:>+7.1%}")
    lines.append(f"{'cash':<6} {'':>10} {'':>9} {cash:>10.2f} {cash / nav:>6.1%}")
    return "\n".join(lines)


# ============================== Email ==============================
def send_email(subject: str, body: str, html: str | None = None, images: dict[str, bytes] | None = None) -> bool:
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_APP_PASSWORD")
    if not sender or not password:
        log.info("Email not configured (EMAIL_SENDER / EMAIL_APP_PASSWORD), skipped")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = os.environ.get("EMAIL_TO") or sender
    msg.set_content(body, charset="utf-8")
    if html:
        msg.add_alternative(html, subtype="html", charset="utf-8")
        part = msg.get_payload()[1]
        for cid, png in (images or {}).items():
            part.add_related(png, maintype="image", subtype="png", cid=f"<{cid}>")
    password = password.replace(" ", "")          # app passwords are often shown with spaces
    domain = sender.split("@")[-1].lower()
    host = os.environ.get("EMAIL_SMTP_HOST") or SMTP_HOSTS.get(domain, "smtp.gmail.com")
    ports = [int(os.environ["EMAIL_SMTP_PORT"])] if os.environ.get("EMAIL_SMTP_PORT") else [465, 587]
    errors = []
    for port in ports:                            # fall back from 465 (SSL) to 587 (STARTTLS)
        try:
            if port == 465:
                s = smtplib.SMTP_SSL(host, port, timeout=30)
            else:
                s = smtplib.SMTP(host, port, timeout=30)
                s.ehlo()
                s.starttls()
                s.ehlo()
            with s:
                s.login(sender, password)
                s.send_message(msg)
            log.info("Email sent to %s (%s:%d)", msg["To"], host, port)
            return True
        except smtplib.SMTPAuthenticationError as e:   # wrong password: another port will not help
            log.error("Email login rejected (%s:%d): %s. Use an app password / authorisation code",
                      host, port, e.smtp_code)
            return False
        except Exception as e:                    # an email failure never blocks the signal itself
            errors.append(f"{host}:{port} {type(e).__name__}: {e}")
    log.error("Email failed: %s. Run python check_email.py to diagnose", "; ".join(errors))
    return False


# ============================== Main flow ==============================
def process_account(acct: Account, mk: Market, score_date, scores: pd.Series, force: bool, method: str,
                    sectors: pd.Series | None = None) -> dict:
    prof = acct.profile
    pos, cash = load_holdings(acct)
    state = load_state(acct)
    pos = value_positions(pos, mk.close)
    pos_value = float(pos["value"].sum())
    nav = cash + pos_value
    record_history(mk.last, nav, cash, pos_value, len(pos), acct)

    notes = []
    if state.get("pending") and not acct.auto_fill:
        notes.append(f"Warning: the last trade list ({state['pending']}) has not been recorded yet, so holdings "
                     f"may be out of date. After trading run: python daily_signal.py record")
    if score_date is None:
        notes.append(f"Warning: no score file data/scores/{method}.parquet; run python model.py --method {method}")
    elif score_date < mk.last:
        notes.append(f"Warning: scores ({score_date.date()}) are older than prices ({mk.last.date()}); "
                     f"run factors.py and model.py")

    lr = pd.Timestamp(state["last_rebalance"]) if state.get("last_rebalance") else None
    days_since = None if lr is None else (sum(1 for d in mk.dates if lr < d <= mk.last) if lr >= mk.dates[0] else 10 ** 6)
    due = lr is None or days_since >= prof.rebalance
    fresh = score_date is not None and score_date >= mk.last
    do_rebalance = (due and fresh) or (force and score_date is not None)

    header = (f"[{acct.name}] after the close of {mk.last.date()} (score: {method}, {prof.top_n} stocks, "
              f"sell below rank {prof.hold_rank}, rebalance every {prof.rebalance} trading days, "
              f"{prof.weighting} weights)\nNAV ${nav:,.2f} (positions ${pos_value:,.2f} + cash ${cash:,.2f})")
    res = {"account": acct, "date": mk.last, "rebalance": do_rebalance, "nav": nav, "cash": cash, "pos": pos,
           "orders": None, "new_hold": None, "notes": notes, "days_left": None}

    if do_rebalance:
        vol = mk.vol.iloc[-1] if len(mk.vol) else None
        orders, new_hold = make_orders(pos, cash, scores, prof, vol, sectors, acct.cash_buffer)
        orders = _fill_ref_prices(orders, mk.close)
        exec_day = next_weekday(mk.last)
        acct.orders_dir.mkdir(parents=True, exist_ok=True)
        acct.signals_dir.mkdir(parents=True, exist_ok=True)
        ofile = acct.orders_dir / f"{mk.last.date()}.csv"
        orders.assign(signal_date=mk.last.date().isoformat()).to_csv(ofile, index=False)
        ranked = scores.dropna().sort_values(ascending=False)
        target = "  ".join(f"{s}(#{int(np.flatnonzero(ranked.index == s)[0]) + 1})" for s in new_hold)
        how = ("The model account fills automatically at the next open (spread and impact included)." if acct.auto_fill else
               "Steps (Alpaca):\n"
               "  1. After the open (9:30-9:45) sell first: market order, sell all shares\n"
               "  2. Then buy: market order by dollar amount (Dollars)\n"
               "  3. After the fills run on your computer: python daily_signal.py record")
        body = "\n\n".join(filter(None, [
            header, f">> Rebalance: trade on {exec_day.date()} (next trading day) after the open", _fmt_orders(orders),
            f"Target holdings (rank):\n  {target}", how, "\n".join(notes), "Current holdings:\n" + _fmt_holdings(pos, cash)]))
        (acct.signals_dir / f"{mk.last.date()}.txt").write_text(body, encoding="utf-8")
        state.update({"last_rebalance": mk.last.date().isoformat(), "pending": ofile.name if len(orders) else None})
        save_state(state, acct)
        n_sell = int(orders["action"].isin(["SELL", "TRIM"]).sum())
        n_buy = int(orders["action"].isin(["BUY", "ADD"]).sum())
        res.update(orders=orders, new_hold=new_hold, exec_day=exec_day,
                   subject=(f"[Quant] {mk.last.date()} {acct.name} rebalance: sell {n_sell}, buy {n_buy}" if len(orders)
                            else f"[Quant] {mk.last.date()} {acct.name} rebalance day, no trades"))
    else:
        if due and not fresh:
            why = "Rebalance is due, but the scores are not up to date: no rebalance this time"
        else:
            res["days_left"] = prof.rebalance - (days_since or 0)
            why = f"No rebalance today; next one in {res['days_left']} trading days"
        body = "\n\n".join(filter(None, [header, why, "\n".join(notes), "Current holdings:\n" + _fmt_holdings(pos, cash)]))
        res["subject"] = f"[Quant] {mk.last.date()} no rebalance, NAV ${nav:,.2f}"
    res["body"] = body
    return res


def run(force: bool = False, method: str = SCORE_METHOD) -> dict:
    mk = load_market()
    score_date, scores = latest_scores(method)
    sectors = FU.load_sectors()
    sectors = sectors if len(sectors) else None
    fills = fill_model(MODEL, mk) if MODEL_ENABLED else []
    real = process_account(REAL, mk, score_date, scores, force, method, sectors)
    model = process_account(MODEL, mk, score_date, scores, force, method, sectors) if MODEL_ENABLED else None
    real["model"] = model
    real["model_fills"] = fills

    text = real["body"] + ("\n\n" + "=" * 40 + "\n\n" + model["body"] if model else "")
    print(text)
    html = images = None
    try:
        import report
        html, images, standalone = report.build(real, model, scores, score_date, method)
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(standalone, encoding="utf-8")
        log.info("HTML report: %s", REPORT_FILE)
    except Exception as e:                       # a report failure never blocks the signal
        log.error("Building the HTML report failed: %s", e)
    rebalanced = real["rebalance"] or bool(model and model["rebalance"])
    if rebalanced or os.environ.get("EMAIL_DAILY") == "1" or real["notes"]:
        subject = real["subject"] if real["rebalance"] or not model else model["subject"]
        send_email(subject, text, html, images)
    return real


def record(as_planned: bool = False) -> None:
    acct = REAL
    state = load_state(acct)
    if not state.get("pending"):
        print("Nothing to record.")
        return
    ofile = acct.orders_dir / state["pending"]
    orders = pd.read_csv(ofile)
    pos, cash = load_holdings(acct)
    p = pos.set_index("symbol")
    logs = []
    print(f"Recording fills for {ofile.name}. Enter = use the suggestion in brackets; shares 0 = not filled.")
    for _, o in orders.sort_values("action", key=lambda s: s.isin(["BUY", "ADD"])).iterrows():
        sh_def, px_def = float(o["shares"]), float(o["ref_price"])
        if as_planned:
            sh, price = sh_def, px_def
        else:
            label = {"SELL": "Sold", "TRIM": "Trimmed", "BUY": "Bought", "ADD": "Added"}[o["action"]]
            sh = _ask(f"{label} {o['symbol']} shares [{sh_def:g}]: ", sh_def)
            if sh <= 0:
                continue
            price = _ask(f"    average price [{px_def:.2f}]: ", px_def)
        sym = o["symbol"]
        if o["action"] in ("SELL", "TRIM"):
            have = float(p.at[sym, "shares"]) if sym in p.index else 0.0
            sh = min(sh, have)
            cash += sh * price
            if sym in p.index:
                p.at[sym, "shares"] = have - sh
            slip = (o["ref_price"] / price - 1) * 1e4
            side = "SELL"
        else:
            old_sh = float(p.at[sym, "shares"]) if sym in p.index else 0.0
            old_cb = float(p.at[sym, "cost_basis"]) if sym in p.index else 0.0
            new_sh = old_sh + sh
            p.loc[sym, ["shares", "cost_basis"]] = [new_sh, (old_sh * old_cb + sh * price) / new_sh]
            cash -= sh * price
            slip = (price / o["ref_price"] - 1) * 1e4
            side = "BUY"
        logs.append({"record_time": datetime.now().isoformat(timespec="seconds"), "signal_date": o.get("signal_date"),
                     "symbol": sym, "side": side, "shares": sh, "price": price, "ref_price": o["ref_price"],
                     "slippage_bps": round(slip, 1)})
    pos = p.reset_index()
    pos = pos[pos["shares"] > 1e-9]
    save_holdings(pos, cash, acct)
    append_trades(logs, acct)
    state["pending"] = None
    save_state(state, acct)
    if cash < -0.01:
        print(f"Warning: cash is negative (${cash:,.2f}); check the shares and prices you entered.")
    print(f"Holdings updated: {len(pos)} stocks, cash ${cash:,.2f}")
    if logs:
        avg = pd.DataFrame(logs)["slippage_bps"].mean()
        print(f"Average slippage: {avg:+.1f} bps (positive = worse than the reference close)")


def _ask(prompt: str, default: float) -> float:
    while True:
        s = _input(prompt).strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            print("Please enter a number")


def status() -> None:
    mk = load_market()
    for acct in [REAL] + ([MODEL] if MODEL_ENABLED else []):
        pos, cash = load_holdings(acct)
        pos = value_positions(pos, mk.close)
        st = load_state(acct)
        print(f"\n===== {acct.name} =====")
        print(_fmt_holdings(pos, cash))
        print(f"NAV ${cash + pos['value'].sum():,.2f}; last rebalance: {st.get('last_rebalance') or 'none'}; "
              f"pending list: {st.get('pending') or 'none'}")
        if acct.history.exists():
            print(pd.read_csv(acct.history).tail(5).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Daily trade signals")
    p.add_argument("cmd", nargs="?", default="run", choices=["run", "record", "status", "test-email", "notify-error"])
    p.add_argument("logfile", nargs="?", help="notify-error: the failed log file")
    p.add_argument("--force", action="store_true", help="force a trade list today")
    p.add_argument("--as-planned", action="store_true", help="record: use the suggested values")
    p.add_argument("--method", default=SCORE_METHOD)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "run":
        run(args.force, args.method)
    elif args.cmd == "record":
        record(args.as_planned)
    elif args.cmd == "status":
        status()
    elif args.cmd == "notify-error":
        tail = Path(args.logfile).read_text(encoding="utf-8", errors="replace")[-4000:] if args.logfile else ""
        send_email("[Quant] daily job failed", "The daily job failed. End of the log:\n\n" + tail)
    else:
        ok = send_email("[Quant] test email", "If you received this, the email settings work.")
        print("Sent" if ok else "Failed: check EMAIL_SENDER and EMAIL_APP_PASSWORD in .env")


if __name__ == "__main__":
    main()
