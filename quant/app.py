"""
app.py - strategy web page (Streamlit)

Reads only the snapshot in site/ (written by export_site.py), never the raw data, so:
  locally:  python export_site.py  then  streamlit run app.py
  online:   push the code and site/ to GitHub and deploy this file on Streamlit Community Cloud

Tabs
  Overview     both accounts vs SPY, today's ranking
  Model        the $100,000 account that takes every signal at the next open (costs included)
  Real         the hand-traded account
  Backtest     NAV, drawdown, metrics, calendar-year returns
  Factors      IC, decay, cumulative IC, long-short NAV, correlations, selected factors
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SITE = Path(os.environ.get("SITE_DIR", Path(__file__).resolve().parent / "site"))

st.set_page_config(page_title="US multi-factor strategy", page_icon="📈", layout="wide")

# ---------- colours: model/strategy = blue, SPY = orange, real = aqua; EW benchmark = grey dashed ----------
# The three hues were checked with the dataviz validator (light and dark, pairwise distinguishable)
PALETTE = {
    "light": {"model": "#2a78d6", "spy": "#eb6834", "real": "#1baf7a", "ref": "#8a8984",
              "pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec", "grid": "rgba(0,0,0,0.08)"},
    "dark": {"model": "#3987e5", "spy": "#d95926", "real": "#199e70", "ref": "#a3a29b",
             "pos": "#3987e5", "neg": "#e66767", "mid": "#383835", "grid": "rgba(255,255,255,0.10)"},
}
try:
    _mode = st.context.theme.type or "light"
except Exception:
    _mode = "light"
C = PALETTE["dark" if _mode == "dark" else "light"]
W = "stretch"


# ============================== Reading ==============================
@st.cache_data(ttl=300)
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame | None:
    df = pd.read_csv(path, **kw)
    return df if len(df.columns) else None


def csv(rel: str, **kw) -> pd.DataFrame | None:
    p = SITE / rel
    return _read_csv(str(p), p.stat().st_mtime, **kw) if p.exists() else None   # cache refreshes when the file changes


def meta() -> dict:
    p = SITE / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def series(rel: str, col: str) -> pd.Series | None:
    df = csv(rel, parse_dates=["date"])
    if df is None or df.empty or col not in df:
        return None
    return df.set_index("date")[col].astype(float)


def perf(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    r = nav.pct_change().dropna()
    if len(r) < 2:
        return {}
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    out = {"Ann. return": ann, "Ann. vol": r.std() * np.sqrt(252),
           "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
           "Max drawdown": dd, "Calmar": ann / abs(dd) if dd < 0 else np.nan}
    if bench is not None:
        rb = bench.pct_change().reindex(r.index).fillna(0.0)
        act = r - rb
        te = act.std() * np.sqrt(252)
        out.update({"Excess ann.": act.mean() * 252, "Tracking error": te,
                    "Info ratio": act.mean() * 252 / te if te > 0 else np.nan,
                    "Beta": np.cov(r, rb)[0, 1] / rb.var() if rb.var() > 0 else np.nan})
    return out


# ============================== Charts ==============================
def chart(lines: list[tuple[str, pd.Series, str, str]], y_title: str = "", pct: bool = True,
          height: int = 380) -> go.Figure:
    """lines: [(name, data, colour key, dash)]. Line ends are labelled directly; a legend shows for 2+ lines."""
    fig = go.Figure()
    lines = [(n, s.dropna(), c, d) for n, s, c, d in lines if s is not None and len(s.dropna()) >= 2]
    for name, s, ck, dash in lines:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(width=2, color=C[ck], dash=dash),
                                 hovertemplate=f"%{{y:{'+.1%' if pct else ',.2f'}}}"))
        if len(lines) > 1:
            fig.add_annotation(x=s.index[-1], y=s.values[-1], text=name, showarrow=False,
                               xanchor="left", xshift=6, font=dict(size=12))
    fig.update_layout(height=height, margin=dict(l=10, r=100, t=30, b=10), hovermode="x unified",
                      showlegend=len(lines) > 1, legend=dict(orientation="h", y=1.08, x=0), yaxis_title=y_title)
    fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across", spikethickness=1, spikedash="solid")
    fig.update_yaxes(gridcolor=C["grid"], zeroline=pct, zerolinecolor=C["grid"], tickformat=".0%" if pct else None)
    return fig


def cum(s: pd.Series | None, start=None) -> pd.Series | None:
    if s is None:
        return None
    if start is not None:
        s = s[s.index >= start]
    return s / s.iloc[0] - 1 if len(s) else None


def fmt_pct(v, signed=True) -> str:
    return "—" if v is None or not np.isfinite(v) else (f"{v:+.1%}" if signed else f"{v:.1%}")


def empty(msg: str) -> None:
    st.info(msg)


def describe(p: dict) -> str:
    return (f"{p.get('top_n')} stocks, sold below rank {p.get('hold_rank')}, rebalance every "
            f"{p.get('rebalance')} trading days, {p.get('weighting')} weights")


# ============================== Page ==============================
m = meta()
st.title("US multi-factor strategy")
if not m:
    st.info("No data yet. Run `python export_site.py` locally to create site/, then refresh.")
    st.stop()

prof = m.get("profiles", {})
st.caption(f"Data to the close of {m.get('data_date')} · updated {m.get('updated', '').replace('T', ' ')} · "
           f"universe: top {m.get('universe_size')} US stocks by dollar volume · score: {m.get('method')} · "
           f"model account: {describe(prof.get('model', {}))} · real account: {describe(prof.get('real', {}))}")

spy = series("spy.csv", "spy")
h_model = series("model/history.csv", "nav")
h_real = series("real/history.csv", "nav")
acc = m.get("accounts", {})

tab_all, tab_model, tab_real, tab_bt, tab_fac = st.tabs(["Overview", "Model account", "Real account",
                                                          "Backtest", "Factors"])

# ---------------------------------------------------------------- Overview
with tab_all:
    starts = [h.index[0] for h in (h_model, h_real) if h is not None and len(h)]
    k1, k2, k3 = st.columns(3)
    for col, h, name in [(k1, h_model, "Model account"), (k2, h_real, "Real account")]:
        if h is not None and len(h) >= 2:
            col.metric(f"{name} total return (since {h.index[0].date()})", fmt_pct(h.iloc[-1] / h.iloc[0] - 1),
                       f"today {fmt_pct(h.iloc[-1] / h.iloc[-2] - 1)}", delta_color="off")
        else:
            col.metric(f"{name} total return", "—", "shown after 2 days of history", delta_color="off")
    if spy is not None and starts:
        s = cum(spy, min(starts))
        k3.metric(f"SPY over the same period (since {min(starts).date()})", fmt_pct(s.iloc[-1]) if s is not None else "—")

    if starts and (h_model is not None and len(h_model) >= 2 or h_real is not None and len(h_real) >= 2):
        st.plotly_chart(chart([("Model", cum(h_model), "model", "solid"),
                               ("Real", cum(h_real), "real", "solid"),
                               ("SPY", cum(spy, min(starts)), "spy", "solid")], "Cumulative return"),
                        width=W, theme="streamlit")
        st.caption("Each account is measured from its own first day; SPY from the earliest start.")
    else:
        empty("Once an account has 2 days of history, this compares the model account, the real account and SPY.")

    st.subheader(f"Today's ranking (scores after the close of {m.get('score_date')})")
    rk = csv("ranking.csv")
    if rk is None:
        empty("No scores yet.")
    else:
        top = rk.head(max(20, int(prof.get("real", {}).get("hold_rank") or 20))).copy()
        top["Model holds"] = np.where(top["model_held"], "✓", "")
        top["Real holds"] = np.where(top["real_held"], "✓", "")
        st.dataframe(top[["rank", "symbol", "close", "score", "Model holds", "Real holds"]]
                     .rename(columns={"rank": "Rank", "symbol": "Symbol", "close": "Close", "score": "Score"}),
                     hide_index=True, width=W,
                     column_config={"Close": st.column_config.NumberColumn(format="$%.2f"),
                                    "Score": st.column_config.NumberColumn(format="%+.3f")})


# ---------------------------------------------------------------- Account pages
def account_page(key: str, h: pd.Series | None, color: str, intro: str) -> None:
    info = acc.get(key, {})
    st.caption(intro)
    if not info.get("exists"):
        empty("No records for this account yet. They appear after the first `python daily_signal.py`.")
        return
    show = info.get("show_amounts", True)
    k = st.columns(5)
    if show and "nav" in info:
        k[0].metric("NAV", f"${info['nav']:,.2f}")
    else:
        k[0].metric("NAV index (start = 100)", f"{h.iloc[-1]:.2f}" if h is not None and len(h) else "—")
    if h is not None and len(h) >= 2:
        tot = h.iloc[-1] / h.iloc[0] - 1
        k[1].metric("Total return", fmt_pct(tot))
        if spy is not None:
            s = cum(spy, h.index[0])
            k[2].metric("vs SPY", fmt_pct(tot - s.iloc[-1]) if s is not None else "—")
        k[3].metric("Max drawdown", fmt_pct((h / h.cummax() - 1).min(), signed=False))
    k[4].metric("Holdings / last rebalance", f"{info.get('n_hold', 0)} / {info.get('last_rebalance') or '—'}")
    if info.get("pending") and key == "real":
        st.warning("The latest trade list has not been recorded yet.")

    if h is not None and len(h) >= 2:
        st.plotly_chart(chart([("Account", cum(h), color, "solid"), ("SPY", cum(spy, h.index[0]), "spy", "solid")],
                              "Cumulative return"), width=W, theme="streamlit")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Holdings")
        hold = csv(f"{key}/holdings.csv")
        if hold is None or hold.empty:
            st.write("No positions")
        else:
            t = hold.sort_values("weight", ascending=False).assign(weight=lambda d: d["weight"] * 100,
                                                                    pnl_pct=lambda d: d["pnl_pct"] * 100)
            cols = [c for c in ["symbol", "shares", "price", "value", "weight", "pnl_pct"] if c in t]
            st.dataframe(t[cols].rename(columns={"symbol": "Symbol", "shares": "Shares", "price": "Price",
                                                 "value": "Value", "weight": "Weight", "pnl_pct": "P&L"}),
                         hide_index=True, width=W,
                         column_config={"Weight": st.column_config.NumberColumn(format="%.1f%%"),
                                        "P&L": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                                        "Value": st.column_config.NumberColumn(format="$%.2f"),
                                        "Shares": st.column_config.NumberColumn(format="%.4f")})
            if show and "cash" in info:
                st.caption(f"Cash ${info['cash']:,.2f}")
    with c2:
        st.subheader("Latest trade list")
        o = csv(f"{key}/orders_latest.csv")
        if o is None or o.empty:
            st.write("None yet")
        else:
            st.caption(f"Signal date {o['signal_date'].iloc[0]}, executed at the next open")
            cols = [c for c in ["action", "symbol", "amount_usd", "rank", "reason"] if c in o]
            st.dataframe(o[cols].rename(columns={"action": "Action", "symbol": "Symbol", "amount_usd": "Amount ($)",
                                                 "rank": "Rank", "reason": "Reason"}), hide_index=True, width=W)

    st.subheader("Fills")
    tr = csv(f"{key}/trades.csv")
    if tr is None or tr.empty:
        st.write("No fills yet")
    else:
        msg = f"{len(tr)} fills · average slippage {tr['slippage_bps'].mean():+.1f} bps (vs the signal-day close, positive = worse)"
        if "cost_bps" in tr and tr["cost_bps"].notna().any():
            msg += f" · simulated cost {tr['cost_bps'].mean():.1f} bps on average"
        st.caption(msg)
        tr = tr.iloc[::-1].drop(columns=[c for c in ["cost_bps"] if c in tr and tr[c].isna().all()])
        st.dataframe(tr.rename(columns={"record_time": "Fill date", "signal_date": "Signal date", "symbol": "Symbol",
                                        "side": "Side", "shares": "Shares", "price": "Price", "ref_price": "Ref price",
                                        "slippage_bps": "Slippage (bp)", "cost_bps": "Cost (bp)"}),
                     hide_index=True, width=W)


with tab_model:
    cap = m.get("model_capital", 100000)
    account_page("model", h_model, "model",
                 f"Starting capital ${cap:,.0f}; {describe(prof.get('model', {}))}. Every signal fills at the next "
                 f"day's real open, minus half the spread ({m.get('costs', {}).get('half_spread_bps', 5):g} bps) and "
                 "square-root impact. Booked daily since go-live and never rewritten: the result of following every signal.")
with tab_real:
    account_page("real", h_real, "real", f"Hand-traded on Alpaca with real fills; {describe(prof.get('real', {}))}. "
                 "Differences from the model account come from the smaller, more concentrated book, timing and prices.")

# ---------------------------------------------------------------- Backtest
with tab_bt:
    nd = csv("backtest_nav.csv", parse_dates=["date"])
    if nd is None:
        empty("No backtest yet. Run `python backtest.py` locally, then `python export_site.py`.")
    else:
        nd = nd.set_index("date")
        st.caption(f"Backtest: {m.get('backtest')}. Scores after the close of t, trades at the open of t+1, "
                   "spread and impact charged. The 'wf' score re-selects factors every year with past data only.")
        bench = "spy" if "spy" in nd else "ew"
        rows = {"Strategy (net)": perf(nd["nav"], nd[bench]), "Strategy (gross)": perf(nd["nav_gross"], nd[bench]),
                "Universe EW": perf(nd["ew"], nd["spy"] if bench == "spy" else None)}
        if "spy" in nd:
            rows["SPY"] = perf(nd["spy"])
        mm = rows["Strategy (net)"]
        k = st.columns(4)
        k[0].metric("Ann. return (net)", fmt_pct(mm.get("Ann. return", np.nan)))
        k[1].metric("Sharpe", f"{mm.get('Sharpe', np.nan):.2f}")
        k[2].metric("Max drawdown", fmt_pct(mm.get("Max drawdown", np.nan), signed=False))
        k[3].metric("Excess ann. (vs " + ("SPY" if bench == "spy" else "EW") + ")", fmt_pct(mm.get("Excess ann.", np.nan)))
        lines = [("Strategy (net)", nd["nav"] / nd["nav"].iloc[0] - 1, "model", "solid")]
        if "spy" in nd:
            lines.append(("SPY", nd["spy"] / nd["spy"].iloc[0] - 1, "spy", "solid"))
        lines.append(("Universe EW", nd["ew"] / nd["ew"].iloc[0] - 1, "ref", "dash"))
        st.plotly_chart(chart(lines, "Cumulative return"), width=W, theme="streamlit")

        dd = nd["nav"] / nd["nav"].cummax() - 1
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
                                   line=dict(width=2, color=C["neg"]), hovertemplate="%{y:.1%}"))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, hovermode="x unified",
                          title=dict(text="Strategy drawdown", font=dict(size=14)))
        fig.update_yaxes(tickformat=".0%", gridcolor=C["grid"])
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, width=W, theme="streamlit")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Metrics")
            tab = pd.DataFrame(rows).T
            pcts = [c for c in ["Ann. return", "Ann. vol", "Max drawdown", "Excess ann.", "Tracking error"] if c in tab]
            tab[pcts] = tab[pcts] * 100
            st.dataframe(tab, width=W, column_config={
                **{c: st.column_config.NumberColumn(format="%.1f%%") for c in pcts},
                **{c: st.column_config.NumberColumn(format="%.2f") for c in ["Sharpe", "Calmar", "Info ratio", "Beta"] if c in tab}})
            days = len(nd)
            st.caption(f"Annual one-way turnover {nd['turnover'].sum() * 252 / days:.0%} · "
                       f"average holdings {nd['n_hold'].mean():.1f}")
        with c2:
            st.subheader("Calendar-year returns")
            yearly = {}
            for c, lab in {"nav": "Strategy net", "nav_gross": "Strategy gross", "ew": "Universe EW", "spy": "SPY"}.items():
                if c in nd:
                    last = nd[c].groupby(nd.index.year).last()
                    base = last.shift(1)
                    base.iloc[0] = nd[c].iloc[0]
                    yearly[lab] = (last / base - 1) * 100
            yt = pd.DataFrame(yearly)
            yt.index = yt.index.astype(str)
            st.dataframe(yt, width=W, column_config={c: st.column_config.NumberColumn(format="%+.1f%%") for c in yt})
        bt = csv("backtest_trades.csv")
        if bt is not None and len(bt):
            with st.expander(f"Backtest trades (latest {len(bt)})"):
                st.dataframe(bt.iloc[::-1], hide_index=True, width=W)

# ---------------------------------------------------------------- Factors
with tab_fac:
    summ = csv("factors/factor_summary.csv", index_col=0)
    if summ is None:
        empty("No factor tests yet. Run `python evaluate.py` locally, then `python export_site.py`.")
    else:
        sel_p = SITE / "factors" / "selected_factors.json"
        if sel_p.exists():
            sel = json.loads(sel_p.read_text(encoding="utf-8"))
            st.success("Selected factors (data up to " + str(sel.get("cutoff")) + "): " +
                       ", ".join(f"{k} ({'+' if v > 0 else '−'})" for k, v in sel.get("factors", {}).items()))
        st.caption("Single-factor tests (before costs). IC = cross-sectional rank correlation with future returns; "
                   "|NW t| > 3 is the bar for 'reliable'.")
        s = summ.sort_values("ic_mean")
        fig = go.Figure(go.Bar(x=s["ic_mean"], y=s.index, orientation="h",
                               marker=dict(color=[C["pos"] if v > 0 else C["neg"] for v in s["ic_mean"]], cornerradius=4),
                               customdata=np.stack([s["t_nw"], s.get("desc", pd.Series("", index=s.index)).fillna("")], axis=1),
                               hovertemplate="%{y}<br>IC %{x:+.4f}<br>NW t %{customdata[0]:+.2f}<br>%{customdata[1]}<extra></extra>"))
        fig.update_layout(height=max(300, 22 * len(s)), margin=dict(l=10, r=10, t=36, b=10), hovermode="closest",
                          title=dict(text="Mean IC per factor (blue = positive, red = negative)", font=dict(size=14)))
        fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"])
        st.plotly_chart(fig, width=W, theme="streamlit")

        show = summ.drop(columns=["desc"], errors="ignore").copy()
        for c in ["ic_pos", "LS_ann", "top_turnover"]:
            if c in show:
                show[c] = show[c] * 100
        st.dataframe(show.rename(columns={"group": "Group", "lit_sign": "Lit. sign", "avg_stocks": "Avg stocks",
                                          "ic_mean": "IC mean", "ic_std": "IC std", "icir": "ICIR", "t_nw": "NW t",
                                          "ic_pos": "IC>0", "n_days": "Days", "LS_ann": "LS ann.",
                                          "LS_sharpe": "LS Sharpe", "monotonic": "Monotonic",
                                          "top_turnover": "Top turnover", "rank_autocorr": "Rank autocorr",
                                          "selected": "Selected"}),
                     width=W, column_config={
                         "IC>0": st.column_config.NumberColumn(format="%.0f%%"),
                         "LS ann.": st.column_config.NumberColumn(format="%+.1f%%"),
                         "Top turnover": st.column_config.NumberColumn(format="%.0f%%"),
                         "IC mean": st.column_config.NumberColumn(format="%+.4f"),
                         **{c: st.column_config.NumberColumn(format="%+.2f")
                            for c in ["ICIR", "NW t", "LS Sharpe", "Monotonic", "Rank autocorr", "IC std"]},
                         "Avg stocks": st.column_config.NumberColumn(format="%.0f")})
        decay = csv("factors/ic_decay.csv", index_col=0)
        if decay is not None:
            with st.expander("IC decay (mean IC / NW t per horizon)"):
                st.dataframe(decay, width=W)
        seltab = csv("factors/selection_table.csv", index_col=0)
        if seltab is not None:
            with st.expander("Selection details (why each factor passed or failed)"):
                st.dataframe(seltab, width=W)

        ic_daily = csv("factors/ic_daily.csv", index_col=0, parse_dates=True)
        ls = csv("factors/ls_curve.csv", index_col=0, parse_dates=True)
        if ic_daily is not None:
            st.subheader("Single factor")
            f = st.selectbox("Factor", list(summ.index))
            c1, c2 = st.columns(2)
            with c1:
                if f in ic_daily:
                    fig = chart([("Cumulative IC", ic_daily[f].dropna().cumsum(), "model", "solid")], "Cumulative IC", pct=False)
                    fig.update_layout(title=dict(text="Cumulative IC (a steady slope = stable)", font=dict(size=14)))
                    st.plotly_chart(fig, width=W, theme="streamlit")
            with c2:
                if ls is not None and f in ls:
                    fig = chart([("Long-short NAV", ls[f].dropna(), "model", "solid")], "NAV", pct=False)
                    fig.update_layout(title=dict(text="Long-short NAV (Q5 − Q1, before costs)", font=dict(size=14)))
                    st.plotly_chart(fig, width=W, theme="streamlit")
            yr = csv("factors/ic_by_year.csv", index_col=0)
            if yr is not None and f in yr.index:
                st.caption("IC by year: " + "  ".join(f"{c}: {v:+.4f}" for c, v in yr.loc[f].dropna().items()))

        corr = csv("factors/factor_corr.csv", index_col=0)
        if corr is not None:
            st.subheader("Factor correlations (mean cross-sectional rank correlation)")
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                       colorscale=[[0, C["neg"]], [0.5, C["mid"]], [1, C["pos"]]], xgap=2, ygap=2,
                                       hovertemplate="%{y} × %{x}<br>%{z:+.2f}<extra></extra>"))
            fig.update_layout(height=max(400, 20 * len(corr)), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width=W, theme="streamlit")
            st.caption("The selection skips a factor whose |correlation| with an already selected one is 0.5 or more.")

st.divider()
st.caption("Personal quant research record. Model signals are not investment advice.")
