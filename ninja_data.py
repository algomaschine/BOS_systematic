from __future__ import division

import argparse
import concurrent.futures
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial

import psutil
from binance.client import Client
from tqdm import tqdm

import asyncio
import polars as pl

# ==================== Configuration ====================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Load API keys from environment variables for security
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')

# ========================================================

def wait_for_low_cpu(threshold=10.0, delay=30 * 60):
    """
    Wait until the CPU utilization is below a certain threshold.
    This is just an optional function if you want to wait before big downloads.
    """
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent <= threshold:
            logger.info(f"CPU utilization is now {cpu_percent}%. Resuming...")
            break
        else:
            logger.info(f"CPU utilization is {cpu_percent}%. Waiting for {delay/60} minutes before checking again...")
            time.sleep(delay)

def ts(message):
    """Shortcut for timestamped logging."""
    logger.info(message)

def download_historical_data(client, ticker, interval, start_str, end_str):
    """Download historical klines from Binance and return a cleaned DataFrame."""
    ts(f'Downloading {ticker} from {start_str} to {end_str}')
    try:
        klines = client.get_historical_klines(ticker, interval, start_str, end_str)
        if not klines:
            ts(f"No data returned for {ticker} from {start_str} to {end_str}")
            return pd.DataFrame()
    except Exception as e:
        ts(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]

    data = pd.DataFrame(klines, columns=columns)

    # Convert numeric columns from strings to floats
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Add a 'Date' column
    data['Date'] = pd.to_datetime(data['open_time'], unit='ms').dt.floor('min')

    return data[['open', 'high', 'low', 'close', 'volume', 'Date']]


def get_binance_data(client, ticker, start_date, end_date, interval='1m', num_threads=6, strictly_fresh_date=True):
    """
    Fetch data from Binance in 1-minute intervals, only filling in
    missing dates (fewer than 1440 rows per day).
    """
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, f"{ticker}_{interval}_binance.csv")

    missing_dates = set()
    existing_data = pd.DataFrame([])

    # Ensure start_date and end_date are datetime.date objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    # If file exists, load it
    if os.path.exists(file_path):
        import polars as pl
        existing_data = pl.read_csv(file_path).to_pandas() ; existing_data['Date'] = pd.to_datetime(existing_data['Date'],errors='coerce')

        # Ensure numeric columns remain numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in existing_data.columns:
                existing_data[col] = pd.to_numeric(existing_data[col], errors='coerce')

        # Filter out data beyond the requested end_date
        existing_data = existing_data[existing_data['Date'].dt.date < end_date]

        # Create a new column 'date_only' to group by calendar day
        existing_data['date_only'] = existing_data['Date'].dt.date

        # Count the number of 1-minute bars per date
        date_counts = existing_data.groupby('date_only').size()

        # Identify days that don't have 1440 one-minute bars
        missing_dates = set(date_counts[date_counts < 1440].index)

        # Figure out the full set of days we actually want
        all_dates = pd.date_range(start=max(existing_data['Date'].min().date(), start_date),
                                  end=end_date, freq='D').date
        # Add days that are completely missing
        missing_dates.update(set(all_dates) - set(existing_data['date_only'].unique()))


        # Only keep "strictly fresh" dates (same month/year as today) if requested
        if strictly_fresh_date:
            today = pd.Timestamp.today()
            missing_dates = {
                d for d in missing_dates
                if  d.year == today.year
            }


    else:
        existing_data = pd.DataFrame()

        # If no file, everything from start_date to end_date is missing
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D').date
        missing_dates = set(all_dates)

    missing_dates = sorted(missing_dates)

    # If there's nothing missing, just return what we have
    if not missing_dates:
        ts(f"No missing dates for {ticker}, everything up to date.")
        return existing_data

    ts(f"Downloading missing data for {ticker} from {start_date} to {end_date}")
    # Create day-by-day date ranges
    date_ranges = [(d, d + timedelta(days=1)) for d in missing_dates]

    all_data = existing_data.copy()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                download_historical_data,
                client,
                ticker,
                interval,
                d.strftime('%Y-%m-%d'),
                (d + timedelta(days=1)).strftime('%Y-%m-%d')
            ): d for d in missing_dates
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if not result.empty:
                all_data = pd.concat([all_data, result], ignore_index=True)

    # Drop duplicates and sort by Date
    all_data.drop_duplicates(subset=['Date'], inplace=True)
    all_data.sort_values(by='Date', inplace=True)

    # Async writing with polars
    async def quick_write(fp, df):
    #  def quick_write(fp, df):
        import polars as pl
        # If 'date_only' is still there, drop it
        if 'date_only' in df.columns:
            df = df.drop(columns=['date_only'])

        # Convert to polars, then write
        dfp = pl.from_pandas(df)
        dfp.write_csv(fp)

    try:
        asyncio.run(quick_write(file_path, all_data)) # this can not be called withing another async
        #quick_write( file_path, all_data )
        ts(f"Data for {ticker} has been updated and saved to {file_path}")
    except Exception as e:
        ts(f"Error saving data for {ticker}: {e}")

    return all_data


def get_fresh_enriched_data(client, ticker, start_date):
    """
    Fetch the data up to (and including) today.
    """
    # You can adjust the end date (0 days offset => up to current day)
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        return get_binance_data(client, ticker, start_date, end_date)
    except Exception as e:
        ts(f"Error fetching fresh data for {ticker}: {e}")
        return pd.DataFrame()


def update_data(ticker, timeframe, start_date):
    """
    Download/Update data for a specific ticker.
    Returns a DataFrame (could be empty if there's none).
    """
    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    df = get_fresh_enriched_data(client, ticker, start_date)
    if df.empty:
        logger.warning(f"No data available for {ticker}. Skipping resampling.")
        return df  # Return an empty DataFrame, so we don't break downstream.
    return df


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Binance Data Downloader and Resampler")
    parser.add_argument(
        '-t', '--tickers',
        nargs='+',
        default=[], #['SOLUSDT'],
        help='List of ticker symbols (e.g., SOLUSDT ETHUSDT BTCUSDT)'
    )
    parser.add_argument(
        '-f', '--timeframe',
        type=int,
        default=60,
        help='Base timeframe in minutes for resampling (default: 60)'
    )
    parser.add_argument(
        '-s', '--start_date',
        type=str,
        default='2020-01-01',
        help='Start date for data downloading in YYYY-MM-DD format (default: 2020-01-01)'
    )
    return parser.parse_args()


def calculate_typical_price(df):
    """
    Calculate the typical price = (High + Low + Close) / 3
    Ensures numeric conversion first.
    """
    for col in ['high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    return df


def convert_to_timeframe(df, timeframe):
    """
    Convert the data to the specified timeframe in minutes.
    """
    if 'Date' not in df.columns:
        raise ValueError("The 'Date' column is missing from the dataframe.")

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Resample data to the new timeframe
    df_resampled = df.resample(f'{timeframe}min').agg({
        'typical_price': 'ohlc',
        'volume': 'sum'
    }).dropna()

    # Flatten columns from multi-index
    df_resampled.columns = [
        'typical_price_open', 'typical_price_high',
        'typical_price_low', 'typical_price_close',
        'volume'
    ]

    # Calculate the typical price (average of O, H, L, C)
    df_resampled['typical_price'] = df_resampled[
        ['typical_price_open', 'typical_price_high',
         'typical_price_low', 'typical_price_close']
    ].mean(axis=1)

    # Final format
    df_resampled = df_resampled[['typical_price', 'volume']]
    df_resampled.columns = ['time', 'volume']
    df_resampled['time'] = df_resampled.index
    df_resampled = df_resampled[['time', 'value']]

    return df_resampled


def add_pct_chg(df):
    """
    Calculate percent change of 'value' column and add it as 'pct_chg'.
    """
    df['pct_chg'] = df['value'].pct_change() * 100
    df.dropna(inplace=True)
    return df


def do_convos(ticker, df, timeframes=[5, 8, 13, 15, 30, 60, 120, 180, 240, 480, 1440]):
    """
    Process each ticker's data:
    - Convert to typical price
    - Resample to given timeframes
    - Save multiple outputs, including pct_chg variations
    """
    df = calculate_typical_price(df)

    # Ensure numeric volume
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    df_list = []
    for tf in timeframes:
        df_resampled = convert_to_timeframe(df.copy(), tf)

        # 1) Original resampled data
        df_resampled.to_csv(f'data/{ticker}_{tf}min_orig.csv', index=False)

        # 2) Last 1000 rows (for example)
        df_resampled.tail(1000).to_csv(f'data/{ticker}_{tf}min_FSC.csv', index=False)

        # 3) Percent change data
        df_pct_chg = add_pct_chg(df_resampled.copy())
        df_pct_chg[['date', 'value']].to_csv(f'data/{ticker}_{tf}min_pct_chg.csv', index=False)

        # 4) Last 1000 rows of pct_chg
        df_pct_chg[['date', 'value']].tail(1000).to_csv(f'data/{ticker}_{tf}min_pct_chg_FSC.csv', index=False)

        df_list.append(df_resampled)

    return df_list


async def process_ticker(ticker_df_pair):
    """
    Worker function for each ticker.
    Calls do_convos to generate resampled data & saves to disk.
    """
    ticker, df = ticker_df_pair
    try:
        do_convos(ticker, df)
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")


def main():
    args = parse_arguments()
    tickers = args.tickers
    timeframe = args.timeframe
    start_date = args.start_date

    # If user passes "ALL", define your entire ticker list here
    if 'ALL' in tickers or tickers==[]:
        tickers = [
            
            "BTCUSDT",
            "ETHUSDT"
           
        ]


    # Gather DataFrames for valid tickers
    dfs = []
    for ticker in tickers:
        if not ticker.endswith('USDT'):
            logger.warning(f"Ticker {ticker} does not end with 'USDT'. Skipping.")
            continue

        try:
            df = update_data(ticker, timeframe, start_date)
            # If update_data returns an empty df or None, handle that
            if df is None or df.empty:
                print(f'-ISSUE WITH {ticker}: No data available')
                continue

            # Drop 'date_only' if it exists, just to clean up
            if 'date_only' in df.columns:
                df = df.drop(columns=['date_only'])

            #dfs.append((ticker, df.copy()))
            process_ticker( (ticker, df.copy()) )
            print(f'+GOOD: {ticker}')
        except Exception as e:
            print(f'-ISSUE WITH {ticker}: {e}')

    # Process each ticker's DataFrame concurrently
    # with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
    #    executor.map(process_ticker, dfs)


if __name__ == '__main__':
    main()
