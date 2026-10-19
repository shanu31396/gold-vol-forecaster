"""
RETRAIN ROBOT - runs every Sunday on GitHub Actions (also works on your PC: python retrain_bot.py)
Champion vs challenger:
  - CHAMPION   = the model the website uses right now
  - CHALLENGER = a new model trained on everything EXCEPT the last 8 weeks
  Both are scored on those last 8 weeks (data the challenger has never seen).
  Only if the challenger is at least as good is a new model trained on ALL data and saved.
  This way automatic retraining can keep the model current but can never quietly make it worse.
History of every decision: live_log/retrain_history.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app import live

MODEL_DIR = Path("model")
BLOCKS_FILE = Path("live_data/blocks.csv")
HISTORY = Path("live_log/retrain_history.csv")
HOLDOUT = pd.Timedelta(weeks=8)
TOLERANCE = 1.02          # challenger may be at most 2% worse (noise) and still win
MIN_NEW_BLOCKS = 30       # don't bother retraining for less than ~1 week of new data


def mse(model, X, y):
    return float(np.mean((y - model.predict(X)) ** 2))


if __name__ == "__main__":
    from xgboost import XGBRegressor
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    feats = meta["features"]
    events = pd.read_csv(MODEL_DIR / "events.csv", parse_dates=["broker_time"])
    blocks = pd.read_csv(BLOCKS_FILE, index_col=0, parse_dates=True)
    d = live.training_table(blocks, events, feats)
    X, y = d[feats].astype(float), d["target"]

    new_blocks = int((d.index > pd.Timestamp(meta["trained_until"])).sum())
    print(f"{len(d):,} training rows, {new_blocks} new since the last training ({meta['trained_until'][:10]})")
    if new_blocks < MIN_NEW_BLOCKS:
        print("Not enough new data yet - keeping the current model.")
        raise SystemExit

    hold = d.index >= d.index.max() - HOLDOUT
    champion = XGBRegressor()
    champion.load_model(MODEL_DIR / "xgb_live.json")
    challenger = live.make_model().fit(X[~hold], y[~hold])
    mse_champ, mse_chall = mse(champion, X[hold], y[hold]), mse(challenger, X[hold], y[hold])
    replace = mse_chall <= mse_champ * TOLERANCE
    print(f"Last 8 weeks error  champion: {mse_champ:.4f}   challenger: {mse_chall:.4f}")

    if replace:
        final = live.make_model().fit(X, y)
        final.save_model(MODEL_DIR / "xgb_live.json")
        resid = y[hold] - challenger.predict(X[hold])
        meta.update(live.mood_thresholds(d))
        meta.update({"trained_until": str(d.index.max()), "n_train": int(len(d)),
                     "residual_var": float(np.var(resid)),
                     "recent_r2": float(1 - (resid ** 2).sum() / ((y[hold] - y[hold].mean()) ** 2).sum())})
        (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
        print("Challenger won - new model saved.")
    else:
        print("Champion is better - keeping the current model.")

    HISTORY.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{"date_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
                         "train_rows": len(d), "new_blocks": new_blocks,
                         "mse_champion": round(mse_champ, 5), "mse_challenger": round(mse_chall, 5),
                         "replaced": bool(replace)}])
    row.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)