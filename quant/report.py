"""
report.py - HTML daily report (called by daily_signal.py)

Builds two HTML versions with the same content:
  email       charts referenced as inline images (cid), readable in the Gmail mobile app
  standalone  charts embedded as base64, saved as reports/daily_report.html (double-click to open)
Only inline styles and PNG images, no JavaScript, so email clients render it correctly.
"""
from __future__ import annotations

import base64
import html as H
import io

import numpy as np
import pandas as pd

import data as D

# Colours match the dashboard (model/strategy = blue, SPY = orange, real account = aqua); checked with the dataviz validator
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, LINE, BG = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
POS, NEG = "#1a7f37", "#c62828"


def _spy() -> pd.Series | None:
    if not D.BENCHMARK_FILE.exists():
        return None
    return D.load_benchmark("SPY").set_index("date")["adj_close"]


def _history(acct) -> pd.Series | None:
    if not acct.history.exists():
        return None
    h = pd.read_csv(acct.history, parse_dates=["date"]).set_index("date")["nav"]
    return h if len(h) else None


def _chart_png(series: dict[str, tuple[pd.Series, str]]) -> bytes | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    series = {k: v for k, v in series.items() if v[0] is not None and len(v[0].dropna()) >= 2}
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for name, (s, color) in series.items():
        s = s.dropna()
        ax.plot(s.index, s.values * 100, color=color, lw=2, label=name, solid_capstyle="round")
        ax.annotate(f"{name} {s.values[-1]:+.1%}", (s.index[-1], s.values[-1] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.axhline(0, color=LINE, lw=1, zorder=0)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="y", color=LINE, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series),
                  borderaxespad=0.2)
    ax.margins(x=0.02)
    fig.autofmt_xdate()
    fig.subplots_adjust(right=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


def _cum(s: pd.Series | None) -> pd.Series | None:
    return None if s is None or len(s) < 1 else s / s.iloc[0] - 1


def _table(df: pd.DataFrame, align: dict | None = None) -> str:
    align = align or {}
    th = "".join(f'<th style="text-align:{align.get(c, "left")};padding:6px 8px;border-bottom:1px solid {LINE};'
                 f'color:{INK2};font-weight:600;font-size:12px">{H.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = "".join(f'<td style="text-align:{align.get(c, "left")};padding:6px 8px;border-bottom:1px solid {LINE};'
                      f'font-size:13px;color:{INK}">{v}</td>' for c, v in r.items())
        rows.append(f"<tr>{tds}</tr>")
    return (f'<table style="border-collapse:collapse;width:100%;margin:6px 0 14px">'
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _pct(v: float, signed: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    color = POS if v > 0 else NEG if v < 0 else INK
    return f'<span style="color:{color}">{v:+.1%}</span>' if signed else f"{v:.1%}"


def _kpi(acct_res: dict, spy: pd.Series | None) -> str:
    acct = acct_res["account"]
    h = _history(acct)
    nav = acct_res["nav"]
    day = total = vs = None
    if h is not None and len(h) >= 2:
        day = h.iloc[-1] / h.iloc[-2] - 1
        total = h.iloc[-1] / h.iloc[0] - 1
        if spy is not None:
            s = spy.reindex(h.index).ffill()
            if s.notna().all():
                vs = total - (s.iloc[-1] / s.iloc[0] - 1)
    since = h.index[0].date() if h is not None else acct_res["date"].date()
    return (f'<td style="vertical-align:top;padding:10px 12px;border:1px solid {LINE};border-radius:8px">'
            f'<div style="font-size:12px;color:{INK2}">{H.escape(acct.name)} (since {since})</div>'
            f'<div style="font-size:22px;font-weight:700;color:{INK};margin:2px 0">${nav:,.2f}</div>'
            f'<div style="font-size:12px;color:{INK2}">today {_pct(day)} · total {_pct(total)} · vs SPY {_pct(vs)}</div></td>')


def _orders_table(orders: pd.DataFrame) -> str:
    if orders is None or orders.empty:
        return f'<p style="color:{INK2}">No trades needed: every holding is still inside the buffer.</p>'
    o = orders.copy()
    o["sort"] = o["action"].isin(["BUY", "ADD"])
    o = o.sort_values(["sort", "rank"])
    label = {"SELL": "Sell", "TRIM": "Trim", "BUY": "Buy", "ADD": "Add"}
    t = pd.DataFrame({
        "Action": [f'<b style="color:{NEG if a in ("SELL", "TRIM") else POS}">{label[a]}</b>' for a in o["action"]],
        "Symbol": [f"<b>{H.escape(s)}</b>" for s in o["symbol"]],
        "Quantity": [f"all {sh:g} sh" if a == "SELL" else f"${amt:,.2f} (~{sh:g} sh)"
               for a, sh, amt in zip(o["action"], o["shares"], o["amount_usd"])],
        "Ref price": [f"${p:,.2f}" for p in o["ref_price"]],
        "Reason": [H.escape(str(r)) for r in o["reason"]],
    })
    return _table(t, {"Ref price": "right"})


def _holdings_table(pos: pd.DataFrame, cash: float) -> str:
    if pos is None or pos.empty:
        return f'<p style="color:{INK2}">No positions, cash ${cash:,.2f}</p>'
    nav = cash + pos["value"].sum()
    p = pos.sort_values("value", ascending=False)
    t = pd.DataFrame({"Symbol": [f"<b>{H.escape(s)}</b>" for s in p["symbol"]],
                      "Shares": [f"{v:,.4f}".rstrip("0").rstrip(".") for v in p["shares"]],
                      "Price": [f"${v:,.2f}" for v in p["price"]],
                      "Value": [f"${v:,.2f}" for v in p["value"]],
                      "Weight": [f"{v / nav:.1%}" for v in p["value"]],
                      "P&L": [_pct(v) for v in p["pnl_pct"]]})
    t.loc[len(t)] = ["Cash", "", "", f"${cash:,.2f}", f"{cash / nav:.1%}", ""]
    return _table(t, {c: "right" for c in ["Shares", "Price", "Value", "Weight", "P&L"]})


def build(real: dict, model: dict | None, scores: pd.Series, score_date, method: str):
    """Returns (email html, {cid: png}, standalone html)."""
    spy = _spy()
    date = real["date"].date()
    h_model = _history(model["account"]) if model else None
    h_real = _history(real["account"])

    # ---- cumulative returns: each account from its own start, SPY from the earliest start ----
    starts = [h.index[0] for h in (h_model, h_real) if h is not None]
    spy_cum = None
    if spy is not None and starts:
        s = spy[spy.index >= min(starts)]
        spy_cum = _cum(s) if len(s) else None
    series = {}
    if h_model is not None:
        series["Model"] = (_cum(h_model), BLUE)
    if h_real is not None:
        series["Real"] = (_cum(h_real), AQUA)
    series["SPY"] = (spy_cum, ORANGE)
    png = _chart_png(series)
    images = {"nav_chart": png} if png else {}

    # ---- today's ranking ----
    rec = ""
    if scores is not None and len(scores):
        ranked = scores.dropna().sort_values(ascending=False).head(max(10, len(real.get("new_hold") or [])))
        held_m = set(model["pos"]["symbol"]) if model else set()
        held_r = set(real["pos"]["symbol"])
        px = D.load_daily(start=pd.Timestamp(date) - pd.Timedelta(days=7), columns=["close"])
        px = px.sort_values("date").groupby("symbol")["close"].last()
        t = pd.DataFrame({"Rank": range(1, len(ranked) + 1),
                          "Symbol": [f"<b>{H.escape(s)}</b>" for s in ranked.index],
                          "Close": [f"${px.get(s, np.nan):,.2f}" for s in ranked.index],
                          "Score": [f"{v:+.3f}" for v in ranked.values],
                          "Model": ["held" if s in held_m else "" for s in ranked.index],
                          "Real": ["held" if s in held_r else "" for s in ranked.index]})
        rec = _table(t, {"Rank": "right", "Close": "right", "Score": "right"})

    def section(title: str, body: str) -> str:
        return (f'<h2 style="font-size:16px;color:{INK};margin:22px 0 6px;padding-bottom:4px;'
                f'border-bottom:2px solid {INK}">{H.escape(title)}</h2>{body}')

    notes = "".join(f'<p style="background:#fff4e5;border-left:3px solid #eda100;padding:8px 10px;'
                    f'font-size:13px;color:{INK}">{H.escape(n)}</p>' for n in real["notes"])

    rp = real["account"].profile
    if real["rebalance"]:
        action = (f'<p style="font-size:14px;color:{INK}"><b>&#9654; Rebalance</b>: trade on <b>{real["exec_day"].date()}</b>'
                  f' (next trading day) after the open. Sell first (all shares), then buy (market orders by dollar '
                  f'amount). Afterwards run <code>python daily_signal.py record</code>.</p>' + _orders_table(real["orders"]))
    else:
        left = real.get("days_left")
        action = (f'<p style="font-size:14px;color:{INK2}">No rebalance today'
                  f'{f"; next one in {left} trading days" if left else ""}.</p>')

    model_html = ""
    if model:
        mp = model["account"].profile
        fills = real.get("model_fills") or []
        fill_txt = ""
        if fills:
            f = pd.DataFrame(fills)
            fill_txt = (f'<p style="font-size:13px;color:{INK2}">{len(f)} automatic fills at today\'s open, '
                        f'average cost {f["cost_bps"].mean():.1f} bps:</p>' +
                        _table(pd.DataFrame({"Side": ["Buy" if s == "BUY" else "Sell" for s in f["side"]],
                                             "Symbol": f["symbol"], "Shares": [f"{v:,.4f}" for v in f["shares"]],
                                             "Fill price": [f"${v:,.2f}" for v in f["price"]]}),
                               {"Shares": "right", "Fill price": "right"}))
        nxt = (f'<p style="font-size:13px;color:{INK2}">{len(model["orders"])} orders generated today; '
               f'they fill automatically at the next open.</p>' if model["rebalance"] and model["orders"] is not None
               and len(model["orders"]) else "")
        model_html = section(f"Model account (${mp.capital:,.0f}, every signal, {mp.top_n} stocks, "
                             f"rebalance every {mp.rebalance} days)",
                             fill_txt + nxt + _holdings_table(model["pos"], model["cash"]))

    def page(img_src: str | None) -> str:
        chart = (f'<img src="{img_src}" alt="Cumulative return: model account, real account and SPY" '
                 f'style="width:100%;max-width:720px;height:auto;display:block;margin:4px 0 2px">'
                 if img_src else f'<p style="color:{INK2}">The chart appears once an account has 2 days of history.</p>')
        kpis = "".join([_kpi(model, spy)] if model else []) + _kpi(real, spy)
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quant report {date}</title></head>
<body style="margin:0;background:{BG};font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif">
<div style="max-width:760px;margin:0 auto;padding:16px">
<div style="font-size:12px;color:{INK2}">Quant report · data to the close of {date} · score {H.escape(method)}</div>
<h1 style="font-size:22px;color:{INK};margin:4px 0 12px">Multi-factor stock selection</h1>
<table style="width:100%;border-collapse:separate;border-spacing:8px 0;margin:0 -8px"><tr>{kpis}</tr></table>
{notes}
{section(f"Real account: what to do today ({rp.top_n} stocks, rebalance every {rp.rebalance} days)", action)}
{section("Cumulative return", chart)}
{section("Today's ranking (by score)", rec or '<p>No scores</p>')}
{model_html}
{section("Real account holdings", _holdings_table(real["pos"], real["cash"]))}
<p style="font-size:11px;color:{MUTED};margin-top:24px">Model signals are a personal research record, not investment advice.
The model account is simulated: fills at the real next-day open, minus half the spread and square-root impact.</p>
</div></body></html>"""

    email_html = page("cid:nav_chart" if png else None)
    standalone = page(f"data:image/png;base64,{base64.b64encode(png).decode()}" if png else None)
    return email_html, images, standalone
