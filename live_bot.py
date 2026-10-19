"""
LIVE ROBOT - runs every hour on GitHub Actions (also works on your PC: python live_bot.py)
  1. Downloads the latest 5-minute gold prices
  2. When a new 4-hour window starts, records the model's forecast for it BEFORE it happens
  3. When a window ends, fills in what actually happened
  4. Adds every newly completed 4-hour block to live_data/blocks.csv (the growing dataset
     the weekly retraining learns from)
Results are saved in live_log/forecasts.csv, which the website reads.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app import live

MODEL_DIR = Path("model")
BLOCKS_FILE = Path("live_data/blocks.csv")
LOG_FILE = Path("live_log/forecasts.csv")
LOG_COLS = ["window_start", "made_at_utc", "pred_log_rv", "pred_vol_pct", "mood", "pred_usd_move", "events",
            "source", "model_trained_until", "price", "actual_log_rv", "actual_vol_pct", "actual_mood",
            "actual_usd_move"]


def load_model():
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(MODEL_DIR / "xgb_live.json")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    events = pd.read_csv(MODEL_DIR / "events.csv", parse_dates=["broker_time"])
    return model, meta, events


def get_prices(offset):
    key = os.environ.get("TWELVE_DATA_KEY")
    if key:
        try:
            return live.fetch_twelve_data(key, offset), "twelvedata", False
        except Exception as e:
            print(f"Twelve Data failed ({e}), using Yahoo Finance")
    return live.fetch_yahoo(offset), "yahoo", True


def run(now=None, prices=None):
    model, meta, events = load_model()
    off = meta["broker_hours_ahead_of_ny"]
    now = now if now is not None else live.broker_now(off)
    m5, source, clip = prices if prices is not None else get_prices(off)
    print(f"Broker time now: {now:%Y-%m-%d %H:%M} | latest candle: {m5.index.max():%Y-%m-%d %H:%M} | source: {source}")

    text_cols = {c: "object" for c in ["mood", "events", "source", "model_trained_until", "actual_mood"]}
    log = (pd.read_csv(LOG_FILE, parse_dates=["window_start"], dtype=text_cols) if LOG_FILE.exists()
           else pd.DataFrame(columns=LOG_COLS).astype(text_cols))
    log = log.reindex(columns=LOG_COLS)          # older logs may miss newer columns

    # 1. forecast for the current (or next, if the market is closed) window
    f = live.predict_now(m5, model, meta, events, now, clip_jumps=clip)
    if f["window_start"] not in set(log["window_start"]):
        row = {"window_start": f["window_start"],
               "made_at_utc": (now - pd.Timedelta(hours=off)).tz_localize("America/New_York")
                              .tz_convert("UTC").strftime("%Y-%m-%d %H:%M"),
               "pred_log_rv": round(f["pred_log_rv"], 5), "pred_vol_pct": round(f["vol_pct"], 2),
               "mood": f["mood"], "pred_usd_move": round(f["usd_move"], 2),
               "events": "+".join(f["events"]), "source": source,
               "model_trained_until": meta["trained_until"][:10], "price": round(f["last_price"], 2)}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        print(f"NEW forecast for window starting {f['window_start']:%Y-%m-%d %H:%M}: {f['mood']} ({f['vol_pct']:.1f}%)")
    else:
        print("Forecast for the current window already recorded")

    # 2. fill in what actually happened for finished windows
    rv = live.build_rv(m5, clip)
    done = rv[rv.index + live.BLOCK <= m5.index.max() + pd.Timedelta(minutes=5)]
    filled = 0
    for i, r in log[log["actual_log_rv"].isna()].iterrows():
        ws = pd.Timestamp(r["window_start"])
        if ws in done.index:
            a = float(done.loc[ws, "log_rv"])
            log.loc[i, "actual_log_rv"] = round(a, 5)
            log.loc[i, "actual_vol_pct"] = round(float(live.to_vol_pct(a)), 2)
            log.loc[i, "actual_mood"] = live.mood_of(a, ws.hour, meta)
            if pd.notna(r["price"]):   # same definition as the forecast: 1 std of the 4h price change, in $
                log.loc[i, "actual_usd_move"] = round(float(r["price"]) * float(np.sqrt(np.exp(a))), 2)
            filled += 1
    print(f"Filled in actual results for {filled} finished window(s)")

    # 3. grow the dataset with newly completed blocks
    blocks = pd.read_csv(BLOCKS_FILE, index_col=0, parse_dates=True)
    new = done[done.index > blocks.index.max()][["rv", "n_bars", "ret", "log_rv"]]
    if len(new):
        pd.concat([blocks, new]).to_csv(BLOCKS_FILE)
    print(f"Added {len(new)} new 4h block(s) to {BLOCKS_FILE} (total {len(blocks) + len(new):,})")

    LOG_FILE.parent.mkdir(exist_ok=True)
    log["window_start"] = pd.to_datetime(log["window_start"])
    log.sort_values("window_start")[LOG_COLS].to_csv(LOG_FILE, index=False)


if __name__ == "__main__":
    run()