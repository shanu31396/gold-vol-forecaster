from pathlib import Path
import numpy as np, pandas as pd
from step2_har_baseline import make_features
from step3_ml_models import add_ml_features, FEATURES, make_boost, BOOST_NAME
from step4_calendar import add_event_features, EVENT_FEATURES, run_model, scores_for
from step2_har_baseline import walk_forward
RV_FILE = Path("data/processed/rv_4h.csv")
EVENTS_FILE = Path("data/processed/events.csv")
NEWS_FILE = Path("data/processed/news_scored.csv")
RESULTS = Path("results")
BROKER_HOURS_AHEAD_OF_NY = 7
NEWS_START = pd.Timestamp("2017-01-02")
FIRST_SCORED_YEAR = 2018
NEWS_FEATURES = ["news_n_now", "news_sent_now", "news_n_day", "news_sent_day", "news_neg_day"]
def news_by_block(news: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(news["time_utc"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    news["broker_time"] = t.dt.tz_localize(None) + pd.Timedelta(hours=BROKER_HOURS_AHEAD_OF_NY)
    news["sentiment"] = news["p_positive"] - news["p_negative"]
    news["is_neg"] = (news["p_negative"] > 0.5).astype(float)
    g = news.groupby(news["broker_time"].dt.floor("4h"))
    return pd.DataFrame({"n": g.size(), "sent_sum": g["sentiment"].sum(), "neg_sum": g["is_neg"].sum()})
def add_news_features(d: pd.DataFrame, nb: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    x = nb.reindex(d.index).fillna(0)
    day = x.rolling(6, min_periods=1).sum()
    d["news_n_now"] = x["n"]
    d["news_sent_now"] = np.where(x["n"] > 0, x["sent_sum"] / x["n"].clip(lower=1), 0)
    d["news_n_day"] = day["n"]
    d["news_sent_day"] = np.where(day["n"] > 0, day["sent_sum"] / day["n"].clip(lower=1), 0)
    d["news_neg_day"] = np.where(day["n"] > 0, day["neg_sum"] / day["n"].clip(lower=1), 0)
    d.loc[d.index < NEWS_START, NEWS_FEATURES] = np.nan
    return d
if __name__ == "__main__":
    rv = pd.read_csv(RV_FILE, index_col=0, parse_dates=True)
    ev = pd.read_csv(EVENTS_FILE, parse_dates=["broker_time"])
    news = pd.read_csv(NEWS_FILE)
    print(f"Using {len(news):,} scored headlines")
    d = make_features(add_event_features(add_ml_features(rv), ev)).dropna(subset=FEATURES)
    d = add_news_features(d, news_by_block(news))
    recent = d[d.index >= NEWS_START]
    busy = recent["news_n_now"] > recent["news_n_now"].quantile(0.9)
    ratio = np.exp(recent.loc[busy, "target"].mean() - recent.loc[~busy, "target"].mean())
    print(f"After the busiest 10% of news blocks, the next block is {ratio:.1f}x more volatile.\n")
    print("Training (one model per test year)...")
    p = walk_forward(d)
    base = FEATURES + EVENT_FEATURES
    run_model(BOOST_NAME + "_events", make_boost, d, base, p)
    run_model(BOOST_NAME + "_events_news", make_boost, d, base + NEWS_FEATURES, p)
    p = p[p.index.year >= FIRST_SCORED_YEAR]
    names = ["har_season", BOOST_NAME + "_events", BOOST_NAME + "_events_news"]
    ev_blocks = d.loc[p.index, ["cpi_next", "nfp_next", "fomc_next"]].max(axis=1) == 1
    all_tab, ev_tab = scores_for(p, names), scores_for(p, names, ev_blocks)
    RESULTS.mkdir(exist_ok=True)
    all_tab.to_csv(RESULTS / "step6_scores_all.csv")
    ev_tab.to_csv(RESULTS / "step6_scores_event_blocks.csv")
    p.to_csv(RESULTS / "step6_predictions.csv")
    print(f"\nTest period: {p.index.min().date()} to {p.index.max().date()}  ({len(p):,} blocks)")
    print("\nALL blocks:")
    print(all_tab.to_string())
    print(f"\nONLY blocks with an announcement ({ev_blocks.sum()} blocks):")
    print(ev_tab.to_string())
    print("\nLower RMSE/QLIKE = better. Higher R2 = better.")