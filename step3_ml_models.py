"""
STEP 3 - First real ML models
  - Random Forest
  - XGBoost (falls back to sklearn's HistGradientBoosting if xgboost isn't installed)
Both get more clues (features) than HAR, and are tested exactly like the baselines:
walk-forward, train on years before Y, test on year Y.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from step2_har_baseline import make_features, walk_forward, score, qlike, MIN_TRAIN_YEARS

try:
    from xgboost import XGBRegressor
    def make_boost():
        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    BOOST_NAME = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def make_boost():
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03,
                                             max_leaf_nodes=15, min_samples_leaf=50, random_state=42)
    BOOST_NAME = "grad_boost"

IN_FILE = Path("data/processed/rv_4h.csv")
RESULTS = Path("results")

FEATURES = [
    "har_h", "har_d", "har_w", "har_m",          # recent volatility at different time scales
    "lag2", "lag3", "lag4", "lag5", "lag6",      # the previous few 4h blocks individually
    "vol_of_vol",                                # how unstable volatility itself has been
    "abs_ret", "abs_ret_d", "ret_d",             # size + direction of recent price moves
    "next_hour", "next_dow", "gap_hours",        # WHEN the next block is (session, weekday, after weekend?)
]


def add_ml_features(rv: pd.DataFrame) -> pd.DataFrame:
    d = rv.copy()
    for k in range(2, 7):
        d[f"lag{k}"] = d["log_rv"].shift(k - 1)
    d["vol_of_vol"] = d["log_rv"].rolling(30).std()
    d["abs_ret"] = d["ret"].abs()
    d["abs_ret_d"] = d["abs_ret"].rolling(6).mean()
    d["ret_d"] = d["ret"].rolling(6).sum()       # falling gold tends to raise volatility
    starts = d.index.to_series()
    d["next_dow"] = starts.shift(-1).dt.dayofweek
    d["gap_hours"] = (starts.shift(-1) - starts).dt.total_seconds() / 3600
    return d


def fit_predict(make_model, d, tr, te):
    """Fit, predict the test year, and estimate residual variance on the last training year
    (needed to convert log forecasts back to variance fairly for QLIKE)."""
    X, y = d[FEATURES], d["target"]
    last_year = d.index.year == d.index[tr].year.max()
    inner_tr = tr & ~last_year
    m = make_model().fit(X[inner_tr], y[inner_tr])
    s2 = np.var(y[last_year] - m.predict(X[last_year]))
    m = make_model().fit(X[tr], y[tr])
    return m, m.predict(X[te]), s2


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    d = make_features(add_ml_features(pd.read_csv(IN_FILE, index_col=0, parse_dates=True)))
    d = d.dropna(subset=FEATURES)

    p = walk_forward(d)                          # baselines, on the exact same rows
    models = {
        "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                                       max_features=0.5, n_jobs=-1, random_state=42),
        BOOST_NAME: make_boost,
    }

    years = sorted(d.index.year.unique())
    for name, make_model in models.items():
        print(f"Training {name} (one model per test year)...")
        for y in years[MIN_TRAIN_YEARS:]:
            tr, te = d.index.year < y, d.index.year == y
            m, pred, s2 = fit_predict(make_model, d, tr, te)
            p.loc[d.index[te], name] = pred
            p.loc[d.index[te], name + "_s2"] = s2
            print(f"  {y} done")

    # score everything together
    table = score(p)
    for name in models:
        err = p["actual"] - p[name]
        table.loc[name] = [
            np.sqrt(np.mean(err ** 2)),
            1 - np.sum(err ** 2) / np.sum((p["actual"] - p["actual"].mean()) ** 2),
            qlike(p["actual_rv"], np.exp(p[name] + 0.5 * p[name + "_s2"])),
        ]
    table = table.round(4)
    table.to_csv(RESULTS / "step3_scores.csv")
    p.to_csv(RESULTS / "step3_predictions.csv")

    print(f"\nTest period: {p.index.min().date()} to {p.index.max().date()}  ({len(p):,} blocks)\n")
    print(table.to_string())
    best = table["QLIKE"].idxmin()
    print(f"\nBest model by QLIKE: {best}")

    # which clues mattered most? (measured on the latest test year)
    y = years[-1]
    tr, te = d.index.year < y, d.index.year == y
    m, _, _ = fit_predict(models[BOOST_NAME], d, tr, te)
    imp = permutation_importance(m, d.loc[te, FEATURES], d.loc[te, "target"],
                                 n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    imp.plot.barh()
    plt.title(f"Which features matter most ({BOOST_NAME}, test year {y})")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=130)
    print("\nTop 5 features:\n" + imp.sort_values(ascending=False).head(5).round(4).to_string())
    print(f"\nSaved plot -> {RESULTS / 'feature_importance.png'}")