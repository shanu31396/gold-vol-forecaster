"""
STEP 4 - Economic calendar
Gold's biggest volatility spikes happen at scheduled US announcements:
  CPI  (inflation)       8:30 a.m. New York time
  NFP  (jobs report)     8:30 a.m. New York time
  FOMC (Fed decision)    2:00 p.m. New York time
The dates are known weeks in advance, so it is fair (not cheating) to tell the model
"an announcement is coming in the next 4 hours".

What this script does:
  1. Downloads CPI + NFP release dates from FRED (uses your fred_key.txt)
  2. Figures out your broker's clock automatically from the NFP price spikes
  3. Adds calendar features to every 4h block
  4. Retrains the models and compares them with and without the calendar
"""
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from step1_prepare_data import load_m5, clean, RAW_FILE
from step2_har_baseline import make_features, walk_forward, score, qlike, MIN_TRAIN_YEARS
from step3_ml_models import add_ml_features, FEATURES, make_boost, BOOST_NAME

RV_FILE = Path("data/processed/rv_4h.csv")
EVENTS_FILE = Path("data/processed/events.csv")
RESULTS = Path("results")
NY = ZoneInfo("America/New_York")

FRED_RELEASES = {"cpi": 10, "nfp": 50}        # FRED release ids: Consumer Price Index, Employment Situation
EVENT_TIME_NY = {"cpi": (8, 30), "nfp": (8, 30), "fomc": (14, 0)}

# FOMC decision days (2nd day of each meeting), from federalreserve.gov
FOMC_DATES = """
2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09
""".split()

EVENT_FEATURES = ["cpi_next", "nfp_next", "fomc_next", "event_now",
                  "hours_to_event", "hours_since_event"]


# ---------------------------------------------------------------- 1. download dates
def fred_release_dates(name: str, rid: int) -> list:
    cache = Path(f"data/raw/fred_{name}_dates.csv")
    if cache.exists():
        return pd.read_csv(cache)["date"].tolist()

    key = Path("fred_key.txt").read_text(encoding="utf-8").strip().lstrip("\ufeff")
    url = ("https://api.stlouisfed.org/fred/release/dates"
           f"?release_id={rid}&api_key={key}&file_type=json&realtime_start=2010-01-01"
           "&realtime_end=9999-12-31&include_release_dates_with_no_data=true&limit=10000")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"FRED said: {e.code}. Check that fred_key.txt contains only your 32-character key.")
    except urllib.error.URLError:
        raise SystemExit("Could not reach FRED. Check your internet connection and try again.")

    dates = sorted({d["date"] for d in data["release_dates"] if d["date"] >= "2013-01-01"})
    pd.DataFrame({"date": dates}).to_csv(cache, index=False)
    return dates


# ---------------------------------------------------------------- 2. broker clock
def us_dst(date_str: str) -> bool:
    d = datetime.fromisoformat(date_str).replace(hour=12, tzinfo=NY)
    return bool(d.dst())


def detect_broker_offset(m5: pd.DataFrame, nfp_dates: list) -> dict:
    """On NFP days the biggest 5-min candle happens right at 8:30 New York time.
    The gap between that candle's broker time and 8:30 tells us the broker's clock."""
    ret = np.log(m5["close"]).diff().abs()
    found = {True: [], False: []}
    for ds in nfp_dates:
        day = ret[ret.index.normalize() == pd.Timestamp(ds)]
        if len(day) < 100:
            continue
        spike = day.idxmax()
        announce = pd.Timestamp(ds) + pd.Timedelta(hours=8, minutes=30)
        # the spike candle can START at 8:30 or the candle before it (bar labelled 8:25)
        hours = round(((spike - announce).total_seconds() / 3600) * 2) / 2
        found[us_dst(ds)].append(hours)

    offsets = {}
    for dst, vals in found.items():
        label = "US summer time" if dst else "US winter time"
        if not vals:
            raise SystemExit("Could not find NFP days in your data to detect the broker clock.")
        s = pd.Series(vals)
        best = s.round().mode()[0]
        share = (s.round() == best).mean()
        print(f"  {label}: broker clock = New York + {best:g} h   ({share:.0%} of {len(s)} NFP days agree)")
        offsets[dst] = best
    return offsets


def build_events(m5: pd.DataFrame) -> pd.DataFrame:
    dates = {name: fred_release_dates(name, rid) for name, rid in FRED_RELEASES.items()}
    dates["fomc"] = FOMC_DATES
    for name, ds in dates.items():
        print(f"  {name.upper()}: {len(ds)} dates, {ds[0]} to {ds[-1]}")

    print("\nDetecting your broker's clock from NFP spikes...")
    offset = detect_broker_offset(m5, dates["nfp"])

    rows = []
    for name, ds in dates.items():
        h, m = EVENT_TIME_NY[name]
        for d in ds:
            t = pd.Timestamp(d) + pd.Timedelta(hours=h + offset[us_dst(d)], minutes=m)
            rows.append({"event": name, "broker_time": t})
    ev = pd.DataFrame(rows).sort_values("broker_time").reset_index(drop=True)
    ev.to_csv(EVENTS_FILE, index=False)
    return ev


# ---------------------------------------------------------------- 3. features
def add_event_features(d: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    start = d.index.to_series()
    next_start = start.shift(-1)
    ev_block = ev["broker_time"].dt.floor("4h")

    for name in ["cpi", "nfp", "fomc"]:
        blocks = set(ev_block[ev["event"] == name])
        d[f"{name}_next"] = next_start.isin(blocks).astype(int)
    d["event_now"] = start.isin(set(ev_block)).astype(int)

    times = ev["broker_time"].sort_values().values
    block_end = (start + pd.Timedelta(hours=4)).values
    i = np.searchsorted(times, block_end)
    nxt = times[np.minimum(i, len(times) - 1)]
    prv = times[np.maximum(i - 1, 0)]
    d["hours_to_event"] = np.clip((nxt - block_end) / np.timedelta64(1, "h"), 0, 168)
    d["hours_since_event"] = np.clip((block_end - prv) / np.timedelta64(1, "h"), 0, 168)
    return d


# ---------------------------------------------------------------- 4. models
def run_model(name, make_model, d, feats, p):
    X, y = d[feats], d["target"]
    years = sorted(d.index.year.unique())
    for yr in years[MIN_TRAIN_YEARS:]:
        tr, te = d.index.year < yr, d.index.year == yr
        last = d.index.year == yr - 1
        m = make_model().fit(X[tr & ~last], y[tr & ~last])
        s2 = np.var(y[last] - m.predict(X[last]))
        m = make_model().fit(X[tr], y[tr])
        p.loc[d.index[te], name] = m.predict(X[te])
        p.loc[d.index[te], name + "_s2"] = s2
    print(f"  {name} done")


def scores_for(p, names, mask=None):
    q = p if mask is None else p[mask]
    rows = {}
    for n in names:
        err = q["actual"] - q[n]
        rows[n] = {
            "RMSE_logRV": np.sqrt(np.mean(err ** 2)),
            "R2_oos": 1 - np.sum(err ** 2) / np.sum((q["actual"] - q["actual"].mean()) ** 2),
            "QLIKE": qlike(q["actual_rv"], np.exp(q[n] + 0.5 * q.get(n + "_s2", 0))),
        }
    return pd.DataFrame(rows).T.round(4)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    print("Loading M5 data (for the broker clock check)...")
    m5 = clean(load_m5(RAW_FILE))

    print("\nGetting announcement dates...")
    ev = build_events(m5)
    print(f"\nSaved {len(ev)} events -> {EVENTS_FILE}")

    rv = pd.read_csv(RV_FILE, index_col=0, parse_dates=True)
    d = make_features(add_event_features(add_ml_features(rv), ev)).dropna(subset=FEATURES)
    hour_dummies = pd.get_dummies(d["next_hour"].astype(int), prefix="h", drop_first=True).astype(float)
    d = pd.concat([d, hour_dummies], axis=1)
    HOUR_COLS = list(hour_dummies.columns)
    HAR = ["har_h", "har_d", "har_w", "har_m"]

    # quick sanity check: are event blocks really more volatile?
    ev_next = d[["cpi_next", "nfp_next", "fomc_next"]].max(axis=1) == 1
    ratio = np.exp(d.loc[ev_next, "target"].mean() - d.loc[~ev_next, "target"].mean())
    print(f"\nBlocks WITH an announcement are {ratio:.1f}x more volatile (variance) than blocks without.")

    print("\nTraining (one model per test year)...")
    p = walk_forward(d)
    run_model("har_season_events", LinearRegression, d, HAR + HOUR_COLS + EVENT_FEATURES[:4], p)
    run_model(BOOST_NAME, make_boost, d, FEATURES, p)
    run_model(BOOST_NAME + "_events", make_boost, d, FEATURES + EVENT_FEATURES, p)

    names = ["har_season", "har_season_events", BOOST_NAME, BOOST_NAME + "_events"]
    ev_blocks = d.loc[p.index, ["cpi_next", "nfp_next", "fomc_next"]].max(axis=1) == 1
    all_tab = scores_for(p, names)
    ev_tab = scores_for(p, names, ev_blocks)
    all_tab.to_csv(RESULTS / "step4_scores_all.csv")
    ev_tab.to_csv(RESULTS / "step4_scores_event_blocks.csv")
    p.to_csv(RESULTS / "step4_predictions.csv")

    print(f"\nTest period: {p.index.min().date()} to {p.index.max().date()}  ({len(p):,} blocks)")
    print("\nALL blocks:")
    print(all_tab.to_string())
    print(f"\nONLY blocks with an announcement ({ev_blocks.sum()} blocks):")
    print(ev_tab.to_string())
    print("\nLower RMSE/QLIKE = better. Higher R2 = better.")