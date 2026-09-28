"""
STEP 2 - Baselines (the bar every ML model must beat)
  1. Persistence      : next block vol = this block vol
  2. Seasonal naive   : next block vol = same time-of-day block, previous day
  3. HAR-RV           : classic volatility model (Corsi, 2009) on log RV
  4. HAR-RV + season  : HAR-RV plus time-of-day dummies (gold is quiet in Asia, wild at London/NY)

Evaluation: walk-forward, train on all years before Y, test on year Y. Never a random split.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

IN_FILE = Path("data/processed/rv_4h.csv")
RESULTS = Path("results")
MIN_TRAIN_YEARS = 3


def make_features(rv: pd.DataFrame) -> pd.DataFrame:
    d = rv.copy()
    d["hour"] = d.index.hour
    # HAR components (all use information available at the END of the current block)
    d["har_h"] = d["log_rv"]                                        # last 4h
    d["har_d"] = np.log(d["rv"].rolling(6).mean())                  # last ~1 day
    d["har_w"] = np.log(d["rv"].rolling(30).mean())                 # last ~1 week
    d["har_m"] = np.log(d["rv"].rolling(132).mean())                # last ~1 month
    # target = NEXT block
    d["target"] = d["log_rv"].shift(-1)
    d["target_rv"] = d["rv"].shift(-1)
    d["next_hour"] = d["hour"].shift(-1)
    # seasonal naive: previous day's block at the same hour as the NEXT block
    d["seasonal_naive"] = d["log_rv"].groupby(d["hour"]).shift(1).shift(-1)
    return d.dropna(subset=["har_m", "target", "seasonal_naive"])


def qlike(actual_var, forecast_var):
    x = actual_var / forecast_var
    return np.mean(x - np.log(x) - 1)


def walk_forward(d: pd.DataFrame):
    har_cols = ["har_h", "har_d", "har_w", "har_m"]
    hour_dummies = pd.get_dummies(d["next_hour"].astype(int), prefix="h", drop_first=True).astype(float)
    X_season = pd.concat([d[har_cols], hour_dummies], axis=1)

    years = sorted(d.index.year.unique())
    preds = []
    for y in years[MIN_TRAIN_YEARS:]:
        tr, te = d.index.year < y, d.index.year == y
        out = pd.DataFrame(index=d.index[te])
        out["actual"] = d.loc[te, "target"]
        out["actual_rv"] = d.loc[te, "target_rv"]
        out["persistence"] = d.loc[te, "har_h"]
        out["seasonal_naive"] = d.loc[te, "seasonal_naive"]

        for name, X in [("har", d[har_cols]), ("har_season", X_season)]:
            m = LinearRegression().fit(X[tr], d.loc[tr, "target"])
            out[name] = m.predict(X[te])
            # variance of train residuals -> bias correction when converting log forecast back to variance
            out[name + "_s2"] = np.var(d.loc[tr, "target"] - m.predict(X[tr]))
        preds.append(out)
    return pd.concat(preds)


def score(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["persistence", "seasonal_naive", "har", "har_season"]:
        err = p["actual"] - p[name]
        s2 = p.get(name + "_s2", 0)
        f_var = np.exp(p[name] + 0.5 * s2)
        rows.append({
            "model": name,
            "RMSE_logRV": np.sqrt(np.mean(err ** 2)),
            "R2_oos": 1 - np.sum(err ** 2) / np.sum((p["actual"] - p["actual"].mean()) ** 2),
            "QLIKE": qlike(p["actual_rv"], f_var),
        })
    return pd.DataFrame(rows).set_index("model").round(4)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    d = make_features(pd.read_csv(IN_FILE, index_col=0, parse_dates=True))
    p = walk_forward(d)
    p.to_csv(RESULTS / "baseline_predictions.csv")

    table = score(p)
    table.to_csv(RESULTS / "baseline_scores.csv")
    print(f"Test period: {p.index.min().date()} to {p.index.max().date()}  ({len(p):,} blocks)\n")
    print(table.to_string())
    print("\nLower RMSE/QLIKE = better. Higher R2 = better.")

    # plot last 60 days: actual vs best baseline
    last = p[p.index >= p.index.max() - pd.Timedelta(days=60)]
    plt.figure(figsize=(12, 4))
    plt.plot(last.index, last["actual"], label="Actual log RV", lw=0.8)
    plt.plot(last.index, last["har_season"], label="HAR-RV + season", lw=1.2)
    plt.title("Next-4h volatility: actual vs HAR-RV forecast (last 60 days)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "har_last60days.png", dpi=130)
    print(f"\nSaved plot -> {RESULTS / 'har_last60days.png'}")