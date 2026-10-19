"""
STEP 7 - Transformer (attention)
Same inputs and same fair test as the LSTM (Step 5), but the LSTM is replaced by a
Transformer encoder, the architecture behind BERT and GPT.

LSTM reads the week block-by-block, in order, like reading a sentence word by word.
Transformer uses ATTENTION: every block in the week can directly "look at" every other block
and decide which ones matter most for the forecast (for example, the same session yesterday).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from step2_har_baseline import MIN_TRAIN_YEARS
from step4_calendar import scores_for
from step5_lstm import (build_data, make_windows, predict, SEQ_COLS, NEXT_COLS, SEQ_LEN,
                        SEEDS, MAX_EPOCHS, PATIENCE, BATCH, device)

STEP5_PRED = Path("results/step5_predictions.csv")
RESULTS = Path("results")
D_MODEL = 32          # size of each block's vector inside the Transformer
HEADS = 4             # attention heads: 4 different "ways of looking" at the week
LAYERS = 2


class VolTransformer(nn.Module):
    def __init__(self, n_seq, n_next):
        super().__init__()
        self.embed = nn.Linear(n_seq, D_MODEL)                        # block features -> vector
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))     # learned "position in the week"
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, dim_feedforward=64, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL + n_next, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, seq, nxt):
        h = self.encoder(self.embed(seq) + self.pos)
        summary = h[:, -1]                                            # the most recent block, after attending to the whole week
        return self.head(torch.cat([summary, nxt], dim=1)).squeeze(1)


def train_one(seed, Xs, Xn, y, tr, va):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VolTransformer(Xs.shape[2], Xn.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
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
            loss_fn(model(Xs_tr[b], Xn_tr[b]), y_tr[b]).backward()
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


if __name__ == "__main__":
    print(f"Running on: {device.upper()}")
    t0 = time.time()
    d = build_data()
    p = pd.read_csv(STEP5_PRED, index_col=0, parse_dates=True)

    seq_raw = d[SEQ_COLS].values.astype(np.float64)
    nxt_raw = d[NEXT_COLS].values.astype(np.float64)
    y_raw = d["target"].values
    years = d.index.year.values
    has_window = np.arange(len(d)) >= SEQ_LEN - 1

    for Y in sorted(set(years))[MIN_TRAIN_YEARS:]:
        tr = (years < Y - 1) & has_window
        va = (years == Y - 1) & has_window
        te = (years == Y) & has_window

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
        idx = d.index[te]
        p.loc[idx, "transformer"] = np.mean(te_preds, 0) * sy + my
        p.loc[idx, "transformer_s2"] = np.var(y_raw[va] - va_pred)
        print(f"  {Y} done  (epochs per seed: {epochs})")

    p = p.dropna(subset=["transformer"])
    names = [n for n in ["har_season", "har_season_events", "xgboost_events", "lstm", "transformer"]
             if n in p.columns]
    ev_blocks = d.loc[p.index, ["cpi_next", "nfp_next", "fomc_next"]].max(axis=1) == 1
    all_tab, ev_tab = scores_for(p, names), scores_for(p, names, ev_blocks)

    RESULTS.mkdir(exist_ok=True)
    all_tab.to_csv(RESULTS / "step7_scores_all.csv")
    ev_tab.to_csv(RESULTS / "step7_scores_event_blocks.csv")
    p.to_csv(RESULTS / "step7_predictions.csv")

    print(f"\nTest period: {p.index.min().date()} to {p.index.max().date()}  ({len(p):,} blocks)")
    print("\nALL blocks:")
    print(all_tab.to_string())
    print(f"\nONLY blocks with an announcement ({ev_blocks.sum()} blocks):")
    print(ev_tab.to_string())
    print(f"\nLower RMSE/QLIKE = better. Higher R2 = better.   Total time: {(time.time() - t0) / 60:.1f} min")