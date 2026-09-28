"""
STEP 1 - Data pipeline
Loads XAUUSD M5 candles, cleans them, and builds a 4-hour realized volatility (RV) dataset.

Output: data/processed/rv_4h.csv  (one row per 4-hour block)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- CONFIG (edit these) ----------------
RAW_FILE = Path("data/raw/XAUUSD_M5.csv")   # your M5 export
BLOCK_HOURS = 4                             # forecast horizon
MIN_BARS_PER_BLOCK = 36                     # 4h = 48 bars; skip blocks with too little data
OUT_FILE = Path("data/processed/rv_4h.csv")
# -----------------------------------------------------


def load_m5(path: Path) -> pd.DataFrame:
    """Reads MT5-style exports (<DATE> <TIME> <OPEN>...) or plain CSVs with a datetime column."""
    with open(path, "r") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ";" if ";" in first else ","

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        stamp = df["date"].astype(str).str.replace(".", "-", regex=False) + " " + df["time"].astype(str)
    else:
        dt_col = next(c for c in df.columns if c in ("datetime", "time", "date", "timestamp"))
        stamp = df[dt_col].astype(str).str.replace(".", "-", regex=False)

    df["datetime"] = pd.to_datetime(stamp)
    df = df[["datetime", "open", "high", "low", "close"]]
    return df.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[(df["high"] >= df["low"])]
    print(f"Cleaning removed {before - len(df):,} bad rows")
    return df


def build_rv(df: pd.DataFrame) -> pd.DataFrame:
    # 5-min log returns, but ONLY between consecutive bars (ignores weekend/daily-break gaps)
    ret = np.log(df["close"]).diff()
    gap = df.index.to_series().diff() != pd.Timedelta(minutes=5)
    ret[gap] = np.nan

    block = df.index.floor(f"{BLOCK_HOURS}h")
    g = pd.DataFrame({"r2": ret ** 2, "r": ret}, index=df.index).groupby(block)

    out = pd.DataFrame({
        "rv": g["r2"].sum(min_count=1),          # realized variance
        "n_bars": g["r"].count(),
        "ret": g["r"].sum(),                      # block return (for features later)
    })
    out = out[out["n_bars"] >= MIN_BARS_PER_BLOCK].copy()
    out = out[out["rv"] > 0]
    out["log_rv"] = np.log(out["rv"])
    out.index.name = "block_start"
    return out


if __name__ == "__main__":
    df = clean(load_m5(RAW_FILE))
    print(f"Loaded {len(df):,} M5 bars from {df.index.min()} to {df.index.max()}")

    rv = build_rv(df)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rv.to_csv(OUT_FILE)

    print(f"Built {len(rv):,} four-hour blocks -> {OUT_FILE}")
    print("\nBlocks per year:\n", rv.groupby(rv.index.year).size().to_string())
    print("\nAnnualised vol (approx) by year:")
    ann = np.sqrt(rv["rv"].groupby(rv.index.year).mean() * (252 * 24 / BLOCK_HOURS))
    print((ann * 100).round(1).astype(str).add("%").to_string())