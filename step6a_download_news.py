import http.client, json, time, urllib.error, urllib.parse, urllib.request, pandas as pd
from email.utils import parsedate_to_datetime
from pathlib import Path
import argparse
import sys
QUERY = '"gold price" sourcelang:english'
parser = argparse.ArgumentParser(description="Download gold news from GDELT API")
parser.add_argument("--start", type=lambda x: pd.Timestamp(x), default="2017-01-02",
                    help="Start date (default: 2017-01-02)")
parser.add_argument("--end", type=lambda x: pd.Timestamp(x),
                    help="End date (default: today)")
parser.add_argument("--per-day", action="store_true",
                    help="Download per day instead of per week (much slower, for testing)")
parser.add_argument("--stop-after", type=int,
                    help="Stop after downloading N records (for quick testing)")
parser.add_argument("--max-weeks", type=int, default=None,
                    help="Maximum number of weeks to download (None = all weeks)")
parser.add_argument("--status-file", type=str, default="data/raw/news/download_status.json",
                    help="Checkpoint status file to resume from")
args = parser.parse_args()
START = args.start
END = args.end if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
CACHE = Path("data/raw/news")
OUT = Path("data/processed/news_headlines.csv")
WAIT = 10
REQUEST_GAP = 10
MAX_ATTEMPTS = 5  
RETRY_BASE = 30
RETRY_MAX = 300   
MAX_RECORDS_PER_REQUEST = 75
HEADERS = {
    "User-Agent": "Mozilla/5.0 (student research project: gold volatility forecasting)",
    "Accept": "application/json",
}
def _retry_after(error: urllib.error.HTTPError):
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None
def _wait_seconds(attempt: int, retry_after=None) -> float:
    if retry_after is not None:
        return retry_after
    return min(RETRY_MAX, RETRY_BASE * (2 ** attempt))
_last_request_at = 0.0
def _save_checkpoint(weeks_downloaded: list, failed_weeks: list, progress_file: Path):
    """Save download progress to a checkpoint file."""
    unique_downloaded = sorted(list(set(weeks_downloaded)))
    unique_failed = sorted(list(set(failed_weeks) - set(unique_downloaded)))
    checkpoint = {
        "weeks_downloaded": [w.isoformat() for w in unique_downloaded],
        "failed_weeks": [w.isoformat() for w in unique_failed],
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat()
    }
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(checkpoint, indent=2))
    print(f"    ✓ Progress saved to {progress_file.name}")
def _load_checkpoint(progress_file: Path) -> dict | None:
    """Load download progress from a checkpoint file."""
    if not progress_file.exists():
        return None
    try:
        checkpoint = json.loads(progress_file.read_text())
        return checkpoint
    except (json.JSONDecodeError, IOError):
        return None
def _scan_existing_files(cache_dir: Path, progress_file: Path) -> list[pd.Timestamp]:
    """Scan data/raw/news for already downloaded .json files."""
    existing_weeks = []
    for f in sorted(cache_dir.glob("*.json")):
        if f.name == progress_file.name:
            continue
        try:
            if f.stat().st_size >= 2:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    existing_weeks.append(pd.Timestamp(f.stem))
        except (ValueError, json.JSONDecodeError, OSError):
            continue
    return existing_weeks
def _fetch_articles(start: pd.Timestamp, end: pd.Timestamp, query: str,
                    maxrecords: int, progress_file: Path) -> list | None:
    global _last_request_at
    params = {
        "query": query, "mode": "ArtList", "format": "json", "maxrecords": str(maxrecords),
        "sort": "HybridRel",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params) 
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            gap = REQUEST_GAP - (time.monotonic() - _last_request_at)
            if gap > 0:
                time.sleep(gap)
            req = urllib.request.Request(url, headers=HEADERS)
            _last_request_at = time.monotonic()
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8-sig").strip()
            if not text:
                raise ValueError("empty response")
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')            
            try:
                # FIXED: Lenient parsing to ignore structural glitches
                payload = json.loads(text, strict=False)
            except json.JSONDecodeError:
                # FIXED: If GDELT truncated the file completely, don't waste 5 attempts
                if attempt >= 1:
                    print(f"\n    ⚠ GDELT returned permanently corrupted text for this week. Skipping.")
                    return []
                preview = " ".join(text.split())[:120]
                raise ValueError(f"non-JSON response: {preview!r}")
            if isinstance(payload, dict) and "articles" not in payload:
                return []
            if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
                raise ValueError("JSON response has no articles list")
            if not all(isinstance(article, dict) for article in payload["articles"]):
                raise ValueError("JSON articles list contains invalid entries")
            count = len(payload["articles"])
            if count < maxrecords and count > 0:
                print(f"\n    ℹ Warning: Expected ~{maxrecords} records, got {count}")
            return payload["articles"]
        except urllib.error.HTTPError as e:
            reason = "too many requests" if e.code == 429 else f"error {e.code}"
            retry_after = _retry_after(e)
            e.close()
            if e.code == 429:
                wait = max(60, retry_after or RETRY_BASE)
                print(f"\n    ⚠ GDELT rate limit is active. Waiting {wait}s...")
                time.sleep(wait)
                continue
            retryable = e.code in (408, 425) or 500 <= e.code < 600
            if not retryable:
                print(f"\n    ✗ GDELT request failed ({reason}); skipping this request")
                return None
            wait = _wait_seconds(attempt, retry_after)
            problem = f"GDELT busy ({reason})"
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, ValueError) as e:
            wait = _wait_seconds(attempt)
            problem = f"GDELT response/network problem ({type(e).__name__}: {e})"
        except KeyboardInterrupt:
            raise
        if attempt == MAX_ATTEMPTS - 1:
            print(f"\n    ✗ {problem} - failed after {MAX_ATTEMPTS} attempts")
            return None
        print(f"\n    {problem} - waiting {wait:.0f}s, then retrying ({attempt + 2}/{MAX_ATTEMPTS})")
        sys.stderr.write("\r")
        sys.stderr.flush()
        time.sleep(wait)
        attempt += 1
    return None
if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    progress_file = Path(args.status_file)
    checkpoint = _load_checkpoint(progress_file)
    checkpoint_weeks = [pd.Timestamp(w) for w in checkpoint.get("weeks_downloaded", [])] if checkpoint else []
    disk_weeks = _scan_existing_files(CACHE, progress_file)
    already_downloaded = sorted(list(set(checkpoint_weeks + disk_weeks)))
    already_set = set(already_downloaded)
    if already_downloaded:
        print(f"✓ Resuming: Found {len(disk_weeks)} files on disk & {len(checkpoint_weeks)} in checkpoint ({len(already_downloaded)} unique weeks).")
        if len(already_downloaded) != len(checkpoint_weeks):
            _save_checkpoint(already_downloaded, [], progress_file)
    if args.per_day:
        dates = pd.date_range(START, END, freq="1D")
        weeks_to_download = [
            d for d in dates
            if d not in already_set and not (CACHE / f"{d:%Y-%m-%d}.json").exists()
        ]
        print(f"⚠ WARNING: Per-day mode is extremely slow (~{len(weeks_to_download)} requests)")
    else:
        weeks = pd.date_range(START, END, freq="7D")
        weeks_to_download = [
            w for w in weeks
            if w not in already_set and not (CACHE / f"{w:%Y-%m-%d}.json").exists()
        ]
        if args.max_weeks:
            weeks_to_download = weeks_to_download[:args.max_weeks]
            print(f"⚠ DEV MODE: Limited to {args.max_weeks} weeks only")
        else:
            print(f"✓ Download configuration:")
            print(f"  • Range: {START.date()} to {END.date()}")
            print(f"  • Mode: WEEKLY (recommended)")
            print(f"  • Total weeks: {len(weeks)} ({len(weeks) - len(weeks_to_download)} already downloaded, {len(weeks_to_download)} to go)")
            print(f"  • Press Ctrl+C anytime to stop (progress is saved)")
        print()
    total_tasks = len(weeks_to_download)
    downloaded_weeks = already_downloaded.copy()
    failed_weeks = []
    total_records = 0
    open_records = 0
    if total_tasks == 0:
        print("✓ Nothing new to download! All weeks in your date range are already on disk.")
    else:
        try:
            for i, w in enumerate(weeks_to_download, 1):
                progress_pct = i / total_tasks
                sys.stderr.write(f"\r[{('=' * 40)[:int(40 * progress_pct)]}{'-' * (40 - int(40 * progress_pct))}] {progress_pct*100:5.1f}% ({w.strftime('%Y-%m-%d')})")
                sys.stderr.flush()
                if args.per_day:
                    days = pd.date_range(w, w + pd.Timedelta(days=6), freq="1D")
                    week_records = []
                    for day in days:
                        arts = _fetch_articles(
                            day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                            QUERY, 10, progress_file
                        )
                        if arts is None:
                            failed_weeks.append(w)
                        else:
                            week_records.extend(arts)
                    keep = [{"seendate": a.get("seendate"), "title": a.get("title"), "domain": a.get("domain")}
                            for a in week_records]
                else:
                    arts = _fetch_articles(
                        w, w + pd.Timedelta(days=7) - pd.Timedelta(seconds=1),
                        QUERY, MAX_RECORDS_PER_REQUEST, progress_file
                    )
                    if arts is None:
                        failed_weeks.append(w)
                        continue
                    keep = [{"seendate": a.get("seendate"), "title": a.get("title"), "domain": a.get("domain")}
                            for a in arts]
                    if not keep:
                        print(f"\n    ℹ Week {w.strftime('%Y-%m-%d')}: No news found - saving empty marker")
                (CACHE / f"{w:%Y-%m-%d}.json").write_text(json.dumps(keep))
                total_records += len(keep)
                downloaded_weeks.append(w)
                if not args.per_day or i % 10 == 0:
                    _save_checkpoint(downloaded_weeks, failed_weeks, progress_file)
                if not args.per_day:
                    time.sleep(WAIT)
            sys.stderr.write("\r" + " " * 80 + "\r")
            print(f"\n✓ Download complete!")
            print(f"  ✓ Downloaded this session: {total_tasks - len(failed_weeks)}/{total_tasks} weeks")
            print(f"  ✓ Total weeks on disk: {len(downloaded_weeks)}")
            print(f"  ✓ New records: {total_records:,}")
            if failed_weeks:
                print(f"\n✗ Failed: {len(failed_weeks)} weeks")
                print("\n  Run this script again to retry failed weeks.")
        except KeyboardInterrupt:
            print(f"\n\n✓ Download interrupted by user.")
            _save_checkpoint(downloaded_weeks, failed_weeks, progress_file)
            print(f"✓ Total weeks safely stored: {len(downloaded_weeks)}")
            sys.exit(1)
    print("\n✓ Merging all downloaded news...")
    rows = []
    for f in sorted(CACHE.glob("*.json")):
        if f.name == progress_file.name:
            continue
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                rows.extend(data)
                open_records += len(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"    ⚠ Warning: Could not parse {f.name}: {e}")
    df = pd.DataFrame(rows, columns=["seendate", "title", "domain"]).dropna(subset=["seendate", "title"])
    df["time_utc"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df = df.dropna(subset=["time_utc"])
    df["title"] = df["title"].str.strip()
    df = df.drop_duplicates(subset=["title"]).sort_values("time_utc")
    if args.stop_after and len(df) > args.stop_after:
        print(f"✓ Applying stop-after limit of {args.stop_after} records")
        df = df.head(args.stop_after)
    df[["time_utc", "title", "domain"]].to_csv(OUT, index=False)
    print(f"\n✓ Saved {len(df):,} unique headlines -> {OUT}")
    if not df.empty:
        print("\n  Headlines per year:")
        print(df.groupby(df["time_utc"].dt.year).size().to_string())
    if failed_weeks:
        print(f"\n✗ {len(failed_weeks)} weeks failed. Run this script again to retry.")
    print("\n✓ All done!")