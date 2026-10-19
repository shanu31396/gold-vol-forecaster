"""
Gold Volatility Forecaster - dashboard
Run locally:   streamlit run app/app.py
Reads the result files produced by the step scripts (results/ folder).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import json

import plotly.graph_objects as go
import streamlit as st

import live

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GITHUB = "https://github.com/shanu31396/gold-vol-forecaster"
BLOCKS_PER_YEAR = 6 * 252

NAMES = {
    "actual": "Actual",
    "persistence": "Persistence (repeat last 4h)",
    "seasonal_naive": "Seasonal naive",
    "har": "HAR-RV",
    "har_season": "HAR-RV + time of day",
    "har_season_events": "HAR-RV + time of day + calendar",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (price only)",
    "grad_boost": "Gradient boosting (price only)",
    "xgboost_events": "XGBoost + calendar",
    "xgboost_events_news": "XGBoost + calendar + news",
    "lstm": "LSTM",
    "transformer": "Transformer",
}

st.set_page_config(page_title="Gold Volatility Forecaster", page_icon="🥇", layout="wide")


# ------------------------------------------------------------------ data loading
def latest(*names):
    for n in names:
        if (RESULTS / n).exists():
            return RESULTS / n
    return None


@st.cache_data
def load_predictions():
    f = latest("step7_predictions.csv", "step5_predictions.csv", "step4_predictions.csv")
    return None if f is None else pd.read_csv(f, index_col=0, parse_dates=True)


@st.cache_data
def load_scores(kind):
    f = latest(f"step7_scores_{kind}.csv", f"step5_scores_{kind}.csv", f"step4_scores_{kind}.csv")
    return None if f is None else pd.read_csv(f, index_col=0)


def to_vol_pct(log_rv):
    """log realized variance of a 4h block -> annualised volatility in % (easier to read)."""
    return np.sqrt(np.exp(log_rv) * BLOCKS_PER_YEAR) * 100


def pretty(df):
    out = df.copy()
    out.index = [NAMES.get(i, i) for i in out.index]
    return out.rename(columns={"RMSE_logRV": "RMSE", "R2_oos": "R²"})


p = load_predictions()
scores_all, scores_ev = load_scores("all"), load_scores("event_blocks")
if p is None or scores_all is None:
    st.error("No result files found. Run the step scripts first (results/ folder is empty).")
    st.stop()

models = [c for c in p.columns if c in NAMES and c not in ("actual",) and not c.endswith("_s2")]
best = scores_all["QLIKE"].idxmin()

# ------------------------------------------------------------------ header
st.title("🥇 Gold Volatility Forecaster")
st.markdown(
    "Predicts **how much gold (XAUUSD) will move in the next 4 hours**: calm or wild, "
    "not up or down. Built on 12 years of 5-minute prices, the US economic calendar and "
    f"deep learning. [Code on GitHub]({GITHUB})")

# ------------------------------------------------------------------ LIVE forecast
@st.cache_resource
def load_live_model():
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(ROOT / "model" / "xgb_live.json")
    meta = json.loads((ROOT / "model" / "meta.json").read_text())
    events = pd.read_csv(ROOT / "model" / "events.csv", parse_dates=["broker_time"])
    return model, meta, events


def twelve_data_key():
    try:
        return st.secrets["TWELVE_DATA_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)          # re-download prices at most every 5 minutes
def get_prices(key, offset):
    notes = []
    if key:
        try:
            return live.fetch_twelve_data(key, offset), "Twelve Data (XAU/USD spot)", False, notes
        except Exception as e:
            notes.append(f"Twelve Data unavailable ({e}); using backup source.")
    return live.fetch_yahoo(offset), "Yahoo Finance (COMEX gold futures)", True, notes


def fmt_time(broker_ts, offset, tz):
    ny = (broker_ts - pd.Timedelta(hours=offset)).tz_localize("America/New_York")
    return ny.tz_convert(tz)


st.header("🔴 Live forecast")
if not (ROOT / "model" / "xgb_live.json").exists():
    st.warning("Live model not found. Run step8_live_model.py and push the model/ folder.")
else:
    try:
        model, meta, events = load_live_model()
        off = meta["broker_hours_ahead_of_ny"]
        with st.spinner("Downloading the latest gold prices and running the model..."):
            m5, source, clip, notes = get_prices(twelve_data_key(), off)
            f = live.predict_now(m5, model, meta, events, live.broker_now(off), clip_jumps=clip)
        for n in notes:
            st.caption(n)

        start_ist, end_ist = fmt_time(f["window_start"], off, "Asia/Kolkata"), fmt_time(f["window_end"], off, "Asia/Kolkata")
        start_utc, end_utc = fmt_time(f["window_start"], off, "UTC"), fmt_time(f["window_end"], off, "UTC")
        if f["market_open"]:
            left = f["window_end"] - live.broker_now(off)
            mins = max(int(left.total_seconds() // 60), 0)
            st.markdown(f"**Forecast for the current 4-hour window:** {start_ist:%d %b, %H:%M}–{end_ist:%H:%M} IST "
                        f"({start_utc:%H:%M}–{end_utc:%H:%M} UTC) · ends in {mins // 60}h {mins % 60}m")
        else:
            st.markdown(f"**Market is closed.** Forecast for the first window after it reopens: "
                        f"{start_ist:%a %d %b, %H:%M}–{end_ist:%H:%M} IST ({start_utc:%H:%M}–{end_utc:%H:%M} UTC)")

        icon = {"Calm": "🟢", "Normal": "🟡", "Wild": "🔴"}[f["mood"]]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected", f"{icon} {f['mood']}", help="Compared with the same time of day over the past 12 years")
        k2.metric("Typical move", f"± ${f['usd_move']:.0f}", help="One standard deviation of gold's price change over the 4 hours")
        k3.metric("Annualised volatility", f"{f['vol_pct']:.0f}%")
        if f["vol_so_far_pct"] is not None and f["market_open"]:
            k4.metric("So far in this window", f"{f['vol_so_far_pct']:.0f}%", help=f"From the {f['bars_so_far']} five-minute candles already in this window")
        if f["events"]:
            st.error(f"⚠️ {' + '.join(f['events'])} announcement in this window: expect bigger, sudden moves.")
        elif f["hours_to_event"] < 24:
            st.info(f"Next US announcement in about {f['hours_to_event']:.0f} hours.")

        latest = fmt_time(f["latest_bar"], off, "UTC")
        st.caption(f"Prices: {source}, latest candle {latest:%d %b %H:%M} UTC · refreshes every 5 minutes · "
                   f"model trained on data up to {meta['trained_until'][:10]} · not financial advice")
        if st.button("Refresh now"):
            get_prices.clear()
            st.rerun()
    except Exception as e:
        st.warning(f"Live forecast is unavailable right now ({e}). The research results below still work.")


# ------------------------------------------------------------------ LIVE track record
LOG_URL = f"{GITHUB}/blob/main/live_log/forecasts.csv"
HISTORY_URL = f"{GITHUB}/commits/main/live_log/forecasts.csv"
ICON = {"Calm": "🟢", "Normal": "🟡", "Wild": "🔴"}
LEVEL = {"Calm": 0, "Normal": 1, "Wild": 2}


def load_log():
    f = ROOT / "live_log" / "forecasts.csv"
    if not f.exists():
        return None
    lg = pd.read_csv(f, parse_dates=["window_start"])
    for c in ["pred_usd_move", "actual_usd_move", "price"]:
        if c not in lg.columns:
            lg[c] = np.nan
    off_h = json.loads((ROOT / "model" / "meta.json").read_text())["broker_hours_ahead_of_ny"]
    lg["start_utc"] = (lg["window_start"] - pd.Timedelta(hours=off_h)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    lg["made_utc"] = pd.to_datetime(lg["made_at_utc"]).dt.tz_localize("UTC")
    gap = lg["actual_mood"].map(LEVEL) - lg["mood"].map(LEVEL)
    lg["result"] = np.select([gap.abs() == 0, gap.abs() == 1, gap.abs() == 2],
                             ["✅ Right", "🟡 Close", "❌ Wrong"], default="⏳ Waiting")
    return lg.sort_values("window_start", ascending=False)


def describe(mood, usd):
    if pd.isna(mood):
        return "⏳ in progress"
    return f"{ICON[mood]} {mood}" + (f" · ±${usd:.0f}" if pd.notna(usd) else "")


lg = load_log()

# the most recent finished window, right under the live forecast
if lg is not None and lg["actual_mood"].notna().any():
    last = lg.dropna(subset=["actual_mood"]).iloc[0]
    s_ist = last["start_utc"].tz_convert("Asia/Kolkata")
    e_ist = s_ist + pd.Timedelta(hours=4)
    st.markdown(f"**Previous window ({s_ist:%d %b, %H:%M}–{e_ist:%H:%M} IST):** "
                f"model said {describe(last['mood'], last['pred_usd_move'])} → actually "
                f"{describe(last['actual_mood'], last['actual_usd_move'])} **{last['result']}**")

st.subheader("📈 Every forecast vs what really happened")
if lg is None:
    st.caption("The hourly robot hasn't recorded any forecasts yet. They will appear here automatically.")
else:
    scored = lg.dropna(subset=["actual_mood"])
    week = scored[scored["start_utc"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Forecasts saved", f"{len(lg):,}", help=f"Since {lg['start_utc'].min():%d %b %Y}")
    if len(scored):
        right = scored["result"].eq("✅ Right")
        close = scored["result"].isin(["✅ Right", "🟡 Close"])
        t2.metric("Exactly right", f"{right.mean():.0%}", help="Calm / Normal / Wild matched. Random guessing: about 33%")
        t3.metric("Right or one step off", f"{close.mean():.0%}")
        if len(week):
            t4.metric("Exactly right, last 7 days", f"{week['result'].eq('✅ Right').mean():.0%}", help=f"{len(week)} windows")

        lay = dict(height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25))
        ist = lambda ts: ts.dt.tz_convert("Asia/Kolkata")
        tab_v, tab_d, tab_r, tab_t = st.tabs(["Volatility", "$ move", "Right / Close / Wrong", "Accuracy trend"])

        with tab_v:   # forecast vs reality, every window
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ist(scored["start_utc"]), y=scored["actual_vol_pct"], name="What happened",
                                      mode="lines+markers", line=dict(color="rgba(120,120,120,0.7)")))
            fig2.add_trace(go.Scatter(x=ist(lg["start_utc"]), y=lg["pred_vol_pct"], name="What the model said beforehand",
                                      mode="lines+markers"))
            fig2.update_layout(yaxis_title="Annualised volatility (%)", **lay)
            st.plotly_chart(fig2, width="stretch")

        with tab_d:   # expected vs actual dollar move, last 30 finished windows
            dm = scored.dropna(subset=["pred_usd_move", "actual_usd_move"]).head(30).iloc[::-1]
            if len(dm):
                labels = ist(dm["start_utc"]).dt.strftime("%d %b %H:%M")
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=labels, y=dm["pred_usd_move"], name="Model expected (±$)"))
                fig3.add_trace(go.Bar(x=labels, y=dm["actual_usd_move"], name="Actually moved (±$)",
                                      marker_color="rgba(120,120,120,0.7)"))
                fig3.update_layout(barmode="group", yaxis_title="Typical move in $ (up or down)", **lay)
                st.plotly_chart(fig3, width="stretch")
                ratio = (dm["actual_usd_move"] / dm["pred_usd_move"]).median()
                st.caption(f"Last {len(dm)} finished windows (IST). On a typical window gold moved "
                           f"{ratio:.1f}× what the model expected (1.0× = spot on).")
            else:
                st.caption("Dollar moves appear here once windows with the newer robot version finish.")

        with tab_r:   # how often right / close / wrong
            order = ["✅ Right", "🟡 Close", "❌ Wrong"]
            counts = scored["result"].value_counts().reindex(order, fill_value=0)
            pct = counts / counts.sum()
            fig4 = go.Figure(go.Bar(x=order, y=counts.values, marker_color=["#2e9e44", "#e0b400", "#d0342c"],
                                    text=[f"{c} ({p:.0%})" for c, p in zip(counts.values, pct.values)],
                                    textposition="outside"))
            fig4.update_layout(yaxis_title="Number of windows", showlegend=False, **lay)
            fig4.update_yaxes(range=[0, max(counts.max() * 1.25, 1)])
            st.plotly_chart(fig4, width="stretch")
            st.caption("Right = same category. Close = one step off (e.g. said Normal, was Wild). "
                       "Wrong = said Calm but it was Wild, or the other way round.")

        with tab_t:   # accuracy week by week
            wk = scored.assign(week=ist(scored["start_utc"]).dt.tz_localize(None).dt.to_period("W").dt.start_time)
            g = wk.groupby("week")["result"].agg(right=lambda r: r.eq("✅ Right").mean(), n="size").reset_index()
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(x=g["week"], y=g["right"] * 100, name="Exactly right that week",
                                  text=[f"{v:.0%}<br>{n} windows" for v, n in zip(g["right"], g["n"])],
                                  textposition="outside"))
            fig5.add_hline(y=100 / 3, line_dash="dash", line_color="gray",
                           annotation_text="Random guessing (33%)", annotation_position="bottom right")
            fig5.update_layout(yaxis_title="% exactly right", yaxis_range=[0, 115], showlegend=False, **lay)
            fig5.update_xaxes(title="Week starting")
            st.plotly_chart(fig5, width="stretch")
            st.caption("Bars above the dashed line mean the model beat random guessing that week.")

    mins = ((lg["made_utc"] - lg["start_utc"]).dt.total_seconds() / 60).round().astype("Int64")
    table = pd.DataFrame({
        "Window (IST)": lg["start_utc"].dt.tz_convert("Asia/Kolkata").dt.strftime("%a %d %b, %H:%M")
                        + "–" + (lg["start_utc"] + pd.Timedelta(hours=4)).dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M"),
        "Model said": [describe(m, u) for m, u in zip(lg["mood"], lg["pred_usd_move"])],
        "What happened": [describe(m, u) for m, u in zip(lg["actual_mood"], lg["actual_usd_move"])],
        "Result": lg["result"],
        "US announcement": lg["events"].fillna(""),
        "Saved": [("before the window" if m <= 0 else f"{m} min into the window") if pd.notna(m) else ""
                  for m in mins],
    })
    st.dataframe(table, hide_index=True, width="stretch", height=420)
    st.download_button("⬇️ Download all forecasts (CSV)", lg.drop(columns=["start_utc", "made_utc", "result"])
                       .to_csv(index=False), "gold_volatility_forecasts.csv", "text/csv")
    st.markdown(
        "**How to read this:** \"±$15\" means gold usually moves up or down by about $15 in that window. "
        "Each forecast only uses prices from *before* its window started, and is saved by a robot within minutes "
        "of the window opening. Past forecasts are never edited: new retrained models only affect future forecasts.")
    st.markdown(f"🔍 **Check it yourself:** [the full forecast file]({LOG_URL}) · "
                f"[the robot's timestamped change history]({HISTORY_URL}), where every forecast is recorded publicly by GitHub.")
    hist = ROOT / "live_log" / "retrain_history.csv"
    if hist.exists():
        h = pd.read_csv(hist).iloc[-1]
        verdict = "a new model was adopted" if h["replaced"] else "the current model was kept (the new one wasn't better)"
        st.caption(f"Last weekly retrain: {h['date_utc']} UTC on {int(h['train_rows']):,} examples; {verdict}.")

st.divider()

# ------------------------------------------------------------------ headline numbers
st.header("📊 Research results (2017–2026 backtest)")
base = "har_season"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", NAMES.get(best, best))
if base in scores_all.index:
    gain = 1 - scores_all.loc[best, "QLIKE"] / scores_all.loc[base, "QLIKE"]
    c2.metric("Better than HAR-RV benchmark", f"{gain:.0%}", help="Improvement in QLIKE, the standard volatility forecast loss")
if scores_ev is not None and base in scores_ev.index and "har_season_events" in scores_ev.index:
    cut = 1 - scores_ev.loc["har_season_events", "QLIKE"] / scores_ev.loc[base, "QLIKE"]
    c3.metric("Error cut on announcement days", f"{cut:.0%}", help="CPI, NFP and FOMC periods, HAR-RV with vs without calendar features")
c4.metric("4-hour periods tested", f"{len(p):,}", help=f"{p.index.min():%b %Y} to {p.index.max():%b %Y}, walk-forward")

# ------------------------------------------------------------------ chart
st.subheader("Forecast vs what actually happened")
left, right = st.columns([2, 1])
chosen = left.multiselect("Models to show", models, default=[m for m in [base, best] if m in models],
                          format_func=lambda m: NAMES.get(m, m))
days = right.select_slider("Period", options=[30, 90, 180, 365, 730], value=90,
                           format_func=lambda d: f"last {d} days")
view = p[p.index >= p.index.max() - pd.Timedelta(days=days)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=view.index, y=to_vol_pct(view["actual"]), name="Actual",
                         line=dict(color="rgba(120,120,120,0.6)", width=1)))
for m in chosen:
    fig.add_trace(go.Scatter(x=view.index, y=to_vol_pct(view[m]), name=NAMES.get(m, m), line=dict(width=2)))
fig.update_layout(yaxis_title="Annualised volatility (%)", height=430, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")
st.caption("Each point is one 4-hour period. Spikes are usually US data releases (CPI, jobs report, Fed decisions).")

# ------------------------------------------------------------------ score tables
st.subheader("How the models compare")
tab1, tab2 = st.tabs(["All periods", "Announcement periods only"])
with tab1:
    st.dataframe(pretty(scores_all).style.highlight_min(subset=["RMSE", "QLIKE"], color="#c8f7c5")
                 .highlight_max(subset=["R²"], color="#c8f7c5").format("{:.4f}"), width="stretch")
with tab2:
    if scores_ev is not None:
        st.dataframe(pretty(scores_ev).style.highlight_min(subset=["RMSE", "QLIKE"], color="#c8f7c5")
                     .highlight_max(subset=["R²"], color="#c8f7c5").format("{:.4f}"), width="stretch")
st.caption("Lower RMSE and QLIKE = better. Higher R² = better. QLIKE punishes predicting 'calm' before a storm.")

# ------------------------------------------------------------------ news
news = RESULTS / "step6_scores_all.csv"
if news.exists():
    st.subheader("Does news help?")
    st.markdown("Gold headlines from GDELT (2017 onwards), each read by **FinBERT** "
                "(positive / negative / neutral). Same XGBoost model with vs without news clues, scored 2018 onwards:")
    st.dataframe(pretty(pd.read_csv(news, index_col=0)).style.format("{:.4f}"), width="stretch")

# ------------------------------------------------------------------ feature importance
fi = RESULTS / "feature_importance.png"
if fi.exists():
    st.subheader("What the model pays attention to")
    st.image(str(fi), caption="Permutation importance, XGBoost. 'next_hour' = which trading session comes next.")

# ------------------------------------------------------------------ method
with st.expander("How it was built"):
    st.markdown("""
- **Target:** realized volatility of the next 4 hours (sum of squared 5-minute log returns).
- **Benchmark:** HAR-RV, the standard academic volatility model, plus time-of-day effects.
- **Models:** Random Forest, XGBoost, LSTM and Transformer (PyTorch).
- **Calendar:** CPI and jobs-report dates from the FRED API, FOMC dates; the broker's timezone is detected automatically from price spikes at the 8:30 a.m. New York jobs report.
- **News:** GDELT headlines scored with FinBERT.
- **Testing:** walk-forward by year. Every model is trained only on the past and tested on the next, unseen year. No random splits, no look-ahead.
""")