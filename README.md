# Gold Volatility Forecaster

Forecasting how much gold (XAUUSD) will move in the next 4 hours, using 12 years of 5-minute price data, the US economic calendar, and deep learning.

The goal is to predict **volatility** (how calm or wild the market will be), not price direction. Volatility forecasts are used for risk management and position sizing: trade smaller when a storm is coming.

---

## Key results

Tested walk-forward on **12,850 four-hour periods (Jan 2017 – Sep 2026)**. Every model is trained only on years before the year it predicts, so it never sees the future.

| Model | R² (higher = better) | QLIKE (lower = better) |
|---|---|---|
| Persistence (repeat last 4h) | 0.013 | 0.718 |
| HAR-RV (industry-standard benchmark) | 0.373 | 0.394 |
| HAR-RV + time-of-day | 0.574 | 0.303 |
| XGBoost (price features only) | 0.593 | 0.310 |
| LSTM (sequence model + calendar) | 0.620 | 0.269 |
| **XGBoost + economic calendar** | **0.632** | **0.269** |

**On periods containing a CPI, NFP, or FOMC announcement (314 periods):**

| Model | R² | QLIKE |
|---|---|---|
| HAR-RV + time-of-day | −1.638 | 1.961 |
| **HAR-RV + time-of-day + calendar** | **0.239** | **0.350** |

### Findings
1. **Economic-calendar features improved forecast accuracy (QLIKE) by 11%** over the HAR-RV benchmark, and **cut error on announcement periods by 82%**. Without them, every model predicted "normal" right before major US data releases. Announcement periods turned out to be 3.3× more volatile than normal ones.
2. **Time of day is the single most important feature.** Gold is quiet in the Asian session and active during London/New York hours.
3. **Deep learning did not beat gradient boosting on this dataset.** The LSTM matched XGBoost (QLIKE 0.2693 vs 0.2685) but took ~10 minutes to train versus seconds. On ~12,000 samples of tabular data, this is consistent with what is commonly observed in practice.
4. Price-only ML models gained little over HAR-RV. The big improvement came from **new information** (the calendar), not from a more complex model.

---

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_prepare_data.py` | Cleans 5-min candles and computes 4-hour **realized volatility** (sum of squared 5-min log returns), ignoring weekend and daily-break gaps |
| 2 | `step2_har_baseline.py` | Benchmarks: persistence, seasonal naive, HAR-RV (Corsi, 2009), HAR-RV with time-of-day effects |
| 3 | `step3_ml_models.py` | Random Forest and XGBoost with engineered features; permutation feature importance |
| 4 | `step4_calendar.py` | Downloads CPI/NFP release dates from the FRED API, adds FOMC dates, and **auto-detects the broker's timezone** from price spikes at the 8:30 a.m. NY jobs report |
| 5 | `step5_lstm.py` | PyTorch LSTM that reads the last ~week of 4-hour blocks as a sequence, combined with calendar features; early stopping; 3-seed ensemble |
| 6 | `step6*_*.py` | *(in progress)* Gold news headlines from GDELT, sentiment scored with **FinBERT** |

### Methodology choices
- **Walk-forward validation**, never random splits. Random splits leak future information in time series.
- **No look-ahead:** features only use data available at prediction time. Calendar dates are published weeks in advance, so they are legitimate inputs.
- **Scaling fitted on training data only** in each fold.
- **Metrics:** RMSE and R² on log volatility, plus **QLIKE**, the standard loss for volatility forecasts, which penalizes under-predicting a volatility spike.

---

## Run it yourself

```bash
git clone https://github.com/shanu31396/gold-vol-forecaster.git
cd gold-vol-forecaster
pip install -r requirements.txt
```

You need:
- 5-minute XAUUSD candles exported from MetaTrader 5, saved as `data/raw/XAUUSD_M5.csv` (not included in this repo)
- A free FRED API key in a file named `fred_key.txt` (for Step 4)

Then run the steps in order: `python step1_prepare_data.py`, `python step2_har_baseline.py`, and so on.

---

## Limitations
- Price data comes from one broker's feed; other brokers' data may differ slightly.
- The FOMC dates are a fixed list; emergency meetings (e.g. March 2020) are not included.
- The target is realized volatility only; the models do not predict price direction.

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch · Hugging Face Transformers · FRED API · GDELT