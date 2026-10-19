"""
STEP 8 - Train the final model for the live website
  1. Trains XGBoost + calendar (your best model) on ALL data up to today
  2. Saves it into the model/ folder (small files that go to GitHub, so the website can use them)
  3. Safety check: rebuilds the last ~2 months of features with the WEBSITE's code (app/live.py)
     and confirms they are identical to the training features. If the website computed clues
     even slightly differently, the model would silently give wrong answers.

Re-run this every few months (and after new FOMC dates are added) to keep the model fresh.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from step1_prepare_data import load_m5, clean, RAW_FILE
from step2_har_baseline import make_features
from step3_ml_models import add_ml_features, FEATURES
from step4_calendar import add_event_features, EVENT_FEATURES
from app.live import build_rv, feature_table, make_model

RV_FILE = Path("data/processed/rv_4h.csv")
EVENTS_FILE = Path("data/processed/events.csv")
MODEL_DIR = Path("model")
BROKER_HOURS_AHEAD_OF_NY = 7
ALL_FEATURES = FEATURES + EVENT_FEATURES


LIVE_BLOCKS = Path("live_data/blocks.csv")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    rv = pd.read_csv(RV_FILE, index_col=0, parse_dates=True)
    ev = pd.read_csv(EVENTS_FILE, parse_dates=["broker_time"])
    d = make_features(add_event_features(add_ml_features(rv), ev)).dropna(subset=FEATURES)
    X, y = d[ALL_FEATURES], d["target"]

    # 1. honest error estimate on the most recent year (model trained only on older data)
    recent = d.index >= d.index.max() - pd.Timedelta(days=365)
    m = make_model().fit(X[~recent], y[~recent])
    resid = y[recent] - m.predict(X[recent])
    r2 = 1 - (resid ** 2).sum() / ((y[recent] - y[recent].mean()) ** 2).sum()
    print(f"Check on the last 12 months (unseen): R2 = {r2:.3f}")

    # 2. final model on everything
    model = make_model().fit(X, y)
    model.save_model(MODEL_DIR / "xgb_live.json")

    q_by_hour = {str(int(h)): list(np.percentile(g, [33, 66])) for h, g in y.groupby(d["next_hour"])}
    meta = {
        "features": ALL_FEATURES,
        "residual_var": float(np.var(resid)),
        "q_all": list(np.percentile(y, [33, 66])),
        "q_by_hour": q_by_hour,
        "broker_hours_ahead_of_ny": BROKER_HOURS_AHEAD_OF_NY,
        "trained_until": str(d.index.max()),
        "n_train": int(len(d)),
        "recent_r2": float(r2),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(EVENTS_FILE, MODEL_DIR / "events.csv")
    # seed the robot's growing dataset with your history (only the first time)
    LIVE_BLOCKS.parent.mkdir(exist_ok=True)
    if not LIVE_BLOCKS.exists():
        rv.to_csv(LIVE_BLOCKS)
        print(f"Created {LIVE_BLOCKS} with {len(rv):,} historical blocks (the robot will keep adding to it)")
    else:
        print(f"{LIVE_BLOCKS} already exists - kept as is (it contains blocks the robot collected)")
    print(f"Saved model/xgb_live.json, model/meta.json, model/events.csv  (trained on {len(d):,} blocks)")
    print(f"Calendar known until: {ev['broker_time'].max():%Y-%m-%d}")

    # 3. training-serving skew check
    m5 = clean(load_m5(RAW_FILE))
    tail = m5[m5.index >= m5.index.max() - pd.Timedelta(days=75)]
    live = feature_table(build_rv(tail), ev)
    common = live.index.intersection(d.index)[140:]           # skip rows whose rolling windows aren't full yet
    diff = (live.loc[common, ALL_FEATURES].astype(float) - d.loc[common, ALL_FEATURES].astype(float)).abs().max()
    worst = diff.max()
    print(f"\nSkew check on {len(common)} recent blocks: largest feature difference = {worst:.2e}")
    if worst < 1e-6:
        print("PASS - the website computes exactly the same clues as training.")
    else:
        print("FAIL - these features differ:\n" + diff[diff >= 1e-6].to_string())