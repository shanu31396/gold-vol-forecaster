"""
Live forecasting engine (used by the website).
Downloads the latest 5-minute gold prices, rebuilds EXACTLY the same features the model was
trained on, and predicts volatility for the 4-hour window after the last completed one.

No Streamlit code here, so the training script can import it and check that live features
match training features (a "training-serving skew" test).
"""
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

BLOCK = pd.Timedelta(hours=4)
MODEL_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
MIN_BARS_PER_BLOCK = 36
BLOCKS_PER_YEAR = 6 * 252


# ------------------------------------------------------------------ price sources
def _ny_to_broker(index_ny_naive: pd.DatetimeIndex, offset_hours: int) -> pd.DatetimeIndex:
    return index_ny_naive + pd.Timedelta(hours=offset_hours)


def fetch_twelve_data(api_key: str, offset_hours: int) -> pd.DataFrame:
    """XAU/USD 5-minute candles from Twelve Data (2 requests, ~10,000 candles, ~5 weeks)."""
    frames, end_date = [], None
    for _ in range(2):
        params = {"symbol": "XAU/USD", "interval": "5min", "outputsize": 5000,
                  "timezone": "America/New_York", "apikey": api_key, "format": "JSON"}
        if end_date:
            params["end_date"] = end_date
        url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        if data.get("status") != "ok":
            raise RuntimeError(data.get("message", "Twelve Data error"))
        df = pd.DataFrame(data["values"])
        frames.append(df)
        end_date = (pd.Timestamp(df["datetime"].min()) - pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat(frames)
    df.index = _ny_to_broker(pd.DatetimeIndex(pd.to_datetime(df["datetime"])), offset_hours)
    df = df[["open", "high", "low", "close"]].astype(float)
    return df[~df.index.duplicated()].sort_index()


def fetch_yahoo(offset_hours: int) -> pd.DataFrame:
    """Backup: COMEX gold futures (GC=F) 5-minute candles from Yahoo Finance, last 60 days."""
    import yfinance as yf
    h = yf.Ticker("GC=F").history(period="60d", interval="5m", auto_adjust=False)
    if h.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    idx = h.index.tz_convert("America/New_York").tz_localize(None)
    df = h.rename(columns=str.lower)[["open", "high", "low", "close"]]
    df.index = _ny_to_broker(idx, offset_hours)
    return df[~df.index.duplicated()].sort_index()


# ------------------------------------------------------------------ features (mirror of steps 1-4)
def build_rv(m5: pd.DataFrame, clip_jumps: bool = False) -> pd.DataFrame:
    ret = np.log(m5["close"]).diff()
    gap = m5.index.to_series().diff() != pd.Timedelta(minutes=5)
    ret[gap.values] = np.nan
    if clip_jumps:   # futures contract roll-overs create one fake giant jump; remove it
        med = ret.abs().median()
        ret[ret.abs() > 50 * med] = np.nan
    block = m5.index.floor("4h")
    g = pd.DataFrame({"r2": ret ** 2, "r": ret}, index=m5.index).groupby(block)
    out = pd.DataFrame({"rv": g["r2"].sum(min_count=1), "n_bars": g["r"].count(), "ret": g["r"].sum()})
    out = out[(out["n_bars"] >= MIN_BARS_PER_BLOCK) & (out["rv"] > 0)].copy()
    out["log_rv"] = np.log(out["rv"])
    return out


def next_block_start(start: pd.Timestamp) -> pd.Timestamp:
    """The block after `start`; Friday's last block is followed by Monday 00:00."""
    nxt = start + BLOCK
    while nxt.dayofweek >= 5:
        nxt = (nxt + pd.Timedelta(days=1)).normalize()
    return nxt


def feature_table(rv: pd.DataFrame, events: pd.DataFrame, last_next_start=None) -> pd.DataFrame:
    d = rv.copy()
    start = d.index.to_series()
    next_start = start.shift(-1)
    next_start.iloc[-1] = last_next_start if last_next_start is not None else next_block_start(start.iloc[-1])

    # step 3 features
    for k in range(2, 7):
        d[f"lag{k}"] = d["log_rv"].shift(k - 1)
    d["vol_of_vol"] = d["log_rv"].rolling(30).std()
    d["abs_ret"] = d["ret"].abs()
    d["abs_ret_d"] = d["abs_ret"].rolling(6).mean()
    d["ret_d"] = d["ret"].rolling(6).sum()
    d["next_dow"] = next_start.dt.dayofweek
    d["gap_hours"] = (next_start - start).dt.total_seconds() / 3600

    # step 4 calendar features
    ev_block = events["broker_time"].dt.floor("4h")
    for name in ["cpi", "nfp", "fomc"]:
        d[f"{name}_next"] = next_start.isin(set(ev_block[events["event"] == name])).astype(int)
    d["event_now"] = start.isin(set(ev_block)).astype(int)
    times = events["broker_time"].sort_values().values
    block_end = (start + BLOCK).values
    i = np.searchsorted(times, block_end)
    nxt = times[np.minimum(i, len(times) - 1)]
    prv = times[np.maximum(i - 1, 0)]
    d["hours_to_event"] = np.clip((nxt - block_end) / np.timedelta64(1, "h"), 0, 168)
    d["hours_since_event"] = np.clip((block_end - prv) / np.timedelta64(1, "h"), 0, 168)

    # step 2 features
    d["hour"] = d.index.hour
    d["har_h"] = d["log_rv"]
    d["har_d"] = np.log(d["rv"].rolling(6).mean())
    d["har_w"] = np.log(d["rv"].rolling(30).mean())
    d["har_m"] = np.log(d["rv"].rolling(132).mean())
    d["next_hour"] = next_start.dt.hour.astype(float)
    d["next_start"] = next_start
    return d


def make_model():
    from xgboost import XGBRegressor
    return XGBRegressor(**MODEL_PARAMS)


def training_table(blocks: pd.DataFrame, events: pd.DataFrame, features: list) -> pd.DataFrame:
    """Features + target (next block's log RV) for every block that has a known next block."""
    d = feature_table(blocks, events)
    d["target"] = d["log_rv"].shift(-1)
    return d.dropna(subset=features + ["target"])


def mood_thresholds(d: pd.DataFrame) -> dict:
    return {
        "q_all": list(np.percentile(d["target"], [33, 66])),
        "q_by_hour": {str(int(h)): list(np.percentile(g, [33, 66])) for h, g in d["target"].groupby(d["next_hour"])},
    }


def mood_of(log_rv: float, hour: int, meta: dict) -> str:
    q33, q66 = meta["q_by_hour"].get(str(int(hour)), meta["q_all"])
    return "Calm" if log_rv < q33 else "Normal" if log_rv < q66 else "Wild"


# ------------------------------------------------------------------ prediction
def to_vol_pct(log_rv):
    return np.sqrt(np.exp(log_rv) * BLOCKS_PER_YEAR) * 100


def broker_now(offset_hours: int) -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None) + pd.Timedelta(hours=offset_hours)


def predict_now(m5: pd.DataFrame, model, meta: dict, events: pd.DataFrame, now: pd.Timestamp,
                clip_jumps=False) -> dict:
    latest_bar = m5.index.max()
    market_open = (now - latest_bar) <= pd.Timedelta(minutes=30)
    rv = build_rv(m5, clip_jumps)
    if market_open:
        # keep only COMPLETED blocks (the block containing the latest candle is still in progress)
        rv = rv[rv.index + BLOCK <= latest_bar + pd.Timedelta(minutes=5)]
    if len(rv) < 140:
        raise RuntimeError(f"Not enough history ({len(rv)} complete 4h blocks, need 140)")

    d = feature_table(rv, events)
    row = d.iloc[[-1]]
    X = row[meta["features"]].astype(float)
    if X.isna().any().any():
        raise RuntimeError("Some features could not be computed from the live data")
    pred = float(model.predict(X)[0])

    target_start = row["next_start"].iloc[0]
    mood = mood_of(pred, target_start.hour, meta)

    # how much of the target window has already happened
    so_far = m5[(m5.index >= target_start) & (m5.index < target_start + BLOCK)]
    r = np.log(so_far["close"]).diff().dropna()
    last_price = float(m5["close"].iloc[-1])

    upcoming = [e for e in ("cpi", "nfp", "fomc") if row[f"{e}_next"].iloc[0] == 1]
    return {
        "pred_log_rv": pred,
        "vol_pct": float(to_vol_pct(pred)),
        "mood": mood,
        "usd_move": last_price * float(np.sqrt(np.exp(pred))),   # 1 standard deviation over 4h, in $
        "last_price": last_price,
        "window_start": target_start,
        "window_end": target_start + BLOCK,
        "latest_bar": latest_bar,
        "market_open": bool(market_open),
        "bars_so_far": len(so_far),
        "vol_so_far_pct": float(np.sqrt((r ** 2).sum() * (48 / max(len(r), 1)) * BLOCKS_PER_YEAR) * 100) if len(r) >= 6 else None,
        "events": [e.upper() for e in upcoming],
        "hours_to_event": float(row["hours_to_event"].iloc[0]),
    }