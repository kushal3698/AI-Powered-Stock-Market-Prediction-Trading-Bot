import os
import time
import argparse
import requests
import pandas as pd
import yfinance as yf

def ingest_data(ticker, start_date, end_date, output_dir="data/raw", retries=3, wait=5):
    """
    Ingest stock data from yFinance with headers, falling back to default downloader
    and local cached CSV files if Yahoo rate-limiting is active.
    """
    print(f"Starting ingestion for {ticker} from {start_date} to {end_date}...")
    output_file = os.path.join(output_dir, f"{ticker}.csv")

    for attempt in range(1, retries + 1):
        try:
            # Config 1: Session with User-Agent and Referer (often bypasses rate limits)
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://finance.yahoo.com/"
            })

            print(f"[Attempt {attempt}] Downloading with custom session...")
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                session=session,
                progress=False,
                threads=False,
            )

            if df.empty:
                # Config 2: Fallback to default yfinance downloader (no custom session)
                print(f"[Attempt {attempt}] Custom session returned empty. Trying default downloader...")
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

            if df.empty:
                # Check for cached local file before raising error
                if os.path.exists(output_file):
                    print(f"[Warning] yfinance returned empty. Falling back to local raw cache: {output_file}")
                    df = pd.read_csv(output_file)
                    return df
                raise ValueError(f"No data returned for {ticker}.")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.reset_index(inplace=True)

            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_file, index=False)

            print(f"Saved to {output_file}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(df.tail(3))
            return df

        except Exception as e:
            err = str(e)
            # If we hit an error but have a cached file
            if os.path.exists(output_file):
                print(f"[Warning] yfinance failed with error: {err}. Falling back to local raw cache: {output_file}")
                df = pd.read_csv(output_file)
                return df

            if attempt < retries:
                print(f"[Attempt {attempt}/{retries}] Error: {err[:80]}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"All {retries} attempts failed. Last error: {err}")
                raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()
    ingest_data(args.ticker, args.start, args.end, args.output_dir)
