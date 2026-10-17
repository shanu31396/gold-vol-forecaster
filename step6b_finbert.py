import time, numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
IN_FILE = Path("data/processed/news_headlines.csv")
OUT = Path("data/processed/news_scored.csv")
MODEL = "ProsusAI/finbert"
BATCH = 32
SAVE_EVERY = 2000
device = "cuda" if torch.cuda.is_available() else "cpu"
def score(titles, tok, model):
    enc = tok(titles, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()
    labels = [model.config.id2label[i].lower() for i in range(probs.shape[1])]
    return pd.DataFrame(probs, columns=[f"p_{l}" for l in labels])
if __name__ == "__main__":
    news = pd.read_csv(IN_FILE)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    start = len(done)
    print(f"{len(news):,} headlines, {start:,} already scored. Running on {device.upper()}.")
    if start >= len(news):
        print("Everything is already scored.")
        raise SystemExit
    print("Loading FinBERT (first time downloads ~440 MB)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    example = news["title"].iloc[0]
    print(f"\nExample headline: {example}")
    print(f"Tokens BERT sees: {tok.tokenize(example)[:20]}\n")
    parts, t0 = [done], time.time()
    for i in range(start, len(news), BATCH):
        batch = news.iloc[i:i + BATCH]
        s = score(batch["title"].astype(str).tolist(), tok, model)
        s.insert(0, "time_utc", batch["time_utc"].values)
        s.insert(1, "title", batch["title"].values)
        parts.append(s)
        n = i + len(batch)
        if n % SAVE_EVERY < BATCH or n == len(news):
            pd.concat(parts, ignore_index=True).to_csv(OUT, index=False)
            rate = (n - start) / (time.time() - t0)
            left = (len(news) - n) / rate / 60
            print(f"  {n:,}/{len(news):,} scored  ({rate:.0f}/sec, ~{left:.0f} min left) - saved")
    out = pd.read_csv(OUT)
    out["sentiment"] = out["p_positive"] - out["p_negative"]
    out.to_csv(OUT, index=False)
    print(f"\nDone -> {OUT}")
    print("\nMost NEGATIVE headlines:")
    for t in out.nsmallest(3, "sentiment")["title"]:
        print("  -", t)
    print("\nMost POSITIVE headlines:")
    for t in out.nlargest(3, "sentiment")["title"]:
        print("  +", t)