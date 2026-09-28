"""
STEP 5 - LSTM (deep learning)
XGBoost sees ONE row of clues. The LSTM reads the last week of 4-hour blocks IN ORDER
(like a short video) and learns patterns in the sequence. It also gets the calendar clues
about the next block (announcement coming? which session? after a weekend?).

Same fair test as before: walk-forward by year.
For each test year Y:  train on years < Y-1,  early-stop on year Y-1,  predict year Y.
Runs on GPU if available (Google Colab), otherwise CPU.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from step2_har_baseline import make_features, MIN_TRAIN_YEARS
from step3_ml_models import add_ml_features, FEATURES
from step4_calendar import add_event_features, scores_for

RV_FILE = Path("data/processed/rv_4h.csv")
EVENTS_FILE = Path("data/processed/events.csv")
STEP4_PRED = Path("results/step4_predictions.csv")
RESULTS = Path("results")

SEQ_LEN = 30          # how many past 4h blocks the LSTM reads (~1 trading week)
HIDDEN = 32           # size of the LSTM's memory
SEEDS = [0, 1, 2]     # train 3 times with different random starts and average (more stable)
MAX_EPOCHS = 100
PATIENCE = 8          # stop when validation hasn't improved for 8 epochs
BATCH = 256

SEQ_COLS = ["log_rv", "ret", "abs_ret", "hour_sin", "hour_cos", "dow", "gap_prev", "event_now"]
NEXT_COLS = ["next_hour_sin", "next_hour_cos", "next_dow", "gap_hours",
             "cpi_next", "nfp_next", "fomc_next", "hours_to_event", "hours_since_event"]

device = "cuda" if torch.cuda.is_available() else "cpu"


class VolLSTM(nn.Module):
    def __init__(self, n_seq, n_next):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + n_next, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, seq, nxt):
        _, (h, _) = self.lstm(seq)                      # h = LSTM's summary of the whole week
        return self.head(torch.cat([h[-1], nxt], dim=1)).squeeze(1)


def build_data():
    rv = pd.read_csv(RV_FILE, index_col=0, parse_dates=True)
    ev = pd.read_csv(EVENTS_FILE, parse_dates=["broker_time"])
    d = make_features(add_event_features(add_ml_features(rv), ev)).dropna(subset=FEATURES)
    start = d.index.to_series()
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * d["hour"] / 24), np.cos(2 * np.pi * d["hour"] / 24)
    d["next_hour_sin"] = np.sin(2 * np.pi * d["next_hour"] / 24)
    d["next_hour_cos"] = np.cos(2 * np.pi * d["next_hour"] / 24)
    d["dow"] = start.dt.dayofweek
    d["gap_prev"] = (start - start.shift(1)).dt.total_seconds().fillna(4 * 3600) / 3600
    return d


def make_windows(arr):
    """Turn a (rows, features) table into (rows, SEQ_LEN, features) 'videos'.
    Window for row i = rows i-SEQ_LEN+1 ... i (only the past, never the future)."""
    w = np.lib.stride_tricks.sliding_window_view(arr, SEQ_LEN, axis=0)   # (rows-SEQ+1, feat, SEQ)
    w = np.transpose(w, (0, 2, 1))
    pad = np.full((SEQ_LEN - 1,) + w.shape[1:], np.nan)
    return np.concatenate([pad, w]).astype(np.float32)


def train_one(seed, Xs, Xn, y, tr, va):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VolLSTM(Xs.shape[2], Xn.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    to_t = lambda a: torch.tensor(a, device=device)
    Xs_tr, Xn_tr, y_tr = to_t(Xs[tr]), to_t(Xn[tr]), to_t(y[tr])
    Xs_va, Xn_va, y_va = to_t(Xs[va]), to_t(Xn[va]), to_t(y[va])

    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(y_tr), device=device)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xs_tr[b], Xn_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xs_va, Xn_va), y_va).item()
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1


def predict(model, Xs, Xn, mask):
    with torch.no_grad():
        return model(torch.tensor(Xs[mask], device=device),
                     torch.tensor(Xn[mask], device=device)).cpu().numpy()


if __name__ == "__main__":
    print(f"Running on: {device.upper()}")
    t0 = time.time()
    d = build_data()
    p = pd.read_csv(STEP4_PRED, index_col=0, parse_dates=True)   # earlier models' predictions

    seq_raw = d[SEQ_COLS].values.astype(np.float64)
    nxt_raw = d[NEXT_COLS].values.astype(np.float64)
    y_raw = d["target"].values
    years = d.index.year.values
    has_window = np.arange(len(d)) >= SEQ_LEN - 1

    for Y in sorted(set(years))[MIN_TRAIN_YEARS:]:
        tr = (years < Y - 1) & has_window
        va = (years == Y - 1) & has_window
        te = (years == Y) & has_window

        # scale every input using TRAINING statistics only (no peeking at the future)
        ms, ss = seq_raw[tr].mean(0), seq_raw[tr].std(0) + 1e-8
        mn, sn = nxt_raw[tr].mean(0), nxt_raw[tr].std(0) + 1e-8
        my, sy = y_raw[tr].mean(), y_raw[tr].std()
        Xs = make_windows((seq_raw - ms) / ss)
        Xn = ((nxt_raw - mn) / sn).astype(np.float32)
        y = ((y_raw - my) / sy).astype(np.float32)

        va_preds, te_preds, epochs = [], [], []
        for seed in SEEDS:
            model, n_ep = train_one(seed, Xs, Xn, y, tr, va)
            va_preds.append(predict(model, Xs, Xn, va))
            te_preds.append(predict(model, Xs, Xn, te))
            epochs.append(n_ep)

        va_pred = np.mean(va_preds, 0) * sy + my
        te_pred = np.mean(te_preds, 0) * sy + my
        s2 = np.var(y_raw[va] - va_pred)
        idx = d.index[te]
        p.loc[idx, "lstm"] = te_pred
        p.loc[idx, "lstm_s2"] = s2
        print(f"  {Y} done  (epochs per seed: {epochs})")

    p = p.dropna(subset=["lstm"])
    names = ["har_season", "har_season_events", "xgboost_events", "lstm"]
    names = [n for n in names if n in p.columns]
    ev_blocks = d.loc[p.index, ["cpi_next", "nfp_next", "fomc_next"]].max(axis=1) == 1
    all_tab = scores_for(p, names)
    ev_tab = scores_for(p, names, ev_blocks)

    RESULTS.mkdir(exist_ok=True)
    all_tab.to_csv(RESULTS / "step5_scores_all.csv")
    ev_tab.to_csv(RESULTS / "step5_scores_event_blocks.csv")
    p.to_csv(RESULTS / "step5_predictions.csv")

    print(f"\nTest period: {p.index.min().date()} to {p.index.max().date()}  ({len(p):,} blocks)")
    print("\nALL blocks:")
    print(all_tab.to_string())
    print(f"\nONLY blocks with an announcement ({ev_blocks.sum()} blocks):")
    print(ev_tab.to_string())
    print(f"\nLower RMSE/QLIKE = better. Higher R2 = better.   Total time: {(time.time() - t0) / 60:.1f} min")