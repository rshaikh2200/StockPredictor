from flask import Flask, render_template_string, jsonify, request
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import utils
from yfinance import Ticker
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import glob
from pytickersymbols import PyTickerSymbols
from datetime import datetime, timedelta
import json
import plotly.graph_objs as go
import plotly.utils
import warnings
import traceback
import logging
warnings.filterwarnings('ignore')

# Additional imports for TA analysis
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator

import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas_ta as ta_pat  # For candlestick pattern recognition

# Additional imports for push notifications
import requests
import schedule
import time
import threading
import sqlite3

# Configure logging for error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_app_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

app = Flask(__name__)

# Constants
MIN_START = pd.Timestamp("1998-01-01")
END_DATE = datetime.now().strftime("%Y-%m-%d")
WINDOW = 20
HORIZON = 20

# Candlestick patterns - short-term (1-4 weeks) and long-term (>1 month)
SHORT_TERM_PATTERNS = [
    'DOJI', 'HAMMER', 'INVERTED_HAMMER', 'ENGULFING', 'HARAMI',
    'PIERCING', 'DARK_CLOUD_COVER', 'KICKING', 'MORNING_STAR', 'EVENING_STAR'
]

LONG_TERM_PATTERNS = [
    'HEAD_AND_SHOULDERS', 'DOUBLE_BOTTOM', 'DOUBLE_TOP',
    'TRIPLE_BOTTOM', 'TRIPLE_TOP', 'RISING_WEDGE',
    'FALLING_WEDGE', 'BELT_HOLD'
]

def get_sp500_tickers():
    """Return a list of S&P 500 tickers using the pytickersymbols package."""
    try:
        symbols = PyTickerSymbols()
        sp500_stocks = symbols.get_stocks_by_index("S&P 500")
        tickers = [stock['symbol'] for stock in sp500_stocks]
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers")
        return sorted(tickers)
    except Exception as e:
        error_msg = f"Error fetching S&P 500 tickers: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)
        return sorted([
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JNJ', 'V', 'WMT', 'UNH', 'PG', 'HD', 'MA', 'BAC', 'ADBE', 'PFE',
            'DIS', 'NFLX', 'CRM', 'XOM', 'TMO', 'ABT', 'COST', 'CSCO', 'VZ',
            'INTC', 'CVX', 'CMCSA', 'DHR', 'ORCL', 'PEP', 'NKE', 'KO', 'ACN',
            'MRK', 'LLY', 'ABBV', 'MDT', 'TXN', 'QCOM', 'NEE', 'HON', 'UPS',
            'LOW', 'IBM', 'AMD', 'T', 'BMY', 'CAT', 'COP', 'UNP', 'GS', 'MS',
            'LMT', 'BA', 'SPGI', 'AMGN', 'BLK', 'SYK', 'AXP', 'MMM', 'MDLZ'
        ])


def get_available_tickers():
    """Get list of tickers that either have trained models OR are in S&P 500."""
    model_files = glob.glob("saved_models/*_model_epoch_*.keras")
    trained_tickers = []
    for file in model_files:
        ticker = file.split('/')[-1].split('_model_')[0]
        if ticker not in trained_tickers:
            trained_tickers.append(ticker)
    sp500_tickers = get_sp500_tickers()
    all_tickers = list(dict.fromkeys(trained_tickers + sp500_tickers))
    return sorted(all_tickers)


def assign_label_text(label_int):
    """Convert integer label to text description."""
    labels = {
        0: "Strong Down",
        1: "Slight Down",
        2: "Slight Up",
        3: "Strong Up"
    }
    return labels.get(label_int, "Unknown")


def get_available_models():
    """Get list of tickers that have trained models."""
    model_files = glob.glob("saved_models/*_model_epoch_*.keras")
    tickers = []
    for file in model_files:
        ticker = file.split('/')[-1].split('_model_')[0]
        if ticker not in tickers:
            tickers.append(ticker)
    return sorted(tickers)


def load_best_model(ticker):
    """Load the best saved model for a ticker."""
    try:
        model_files = glob.glob(f'saved_models/{ticker}_model_epoch_*.keras')
        if not model_files:
            logger.warning(f"No model files found for ticker {ticker}")
            return None
        best_model_path = sorted(model_files)[-1]
        logger.info(f"Loading model for {ticker}: {best_model_path}")
        return load_model(best_model_path)
    except Exception as e:
        error_msg = f"Error loading model for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None


def get_stock_data(ticker, days_back=500):
    """Fetch recent stock data for predictions - Enhanced version."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    try:
        logger.info(f"Fetching stock data for {ticker}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Use yfinance Ticker object for more reliable data fetching
        ticker_obj = yf.Ticker(ticker)
        
        # Try to get historical data
        data = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            period="1d"
        )
        
        if data.empty:
            error_msg = f"No data returned from yfinance for {ticker}"
            logger.error(error_msg)
            print(error_msg)
            return None
            
        # Reset index to get Date as a column
        data = data.reset_index()
        
        # Ensure we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f"Missing columns for {ticker}: {missing_columns}"
            logger.error(error_msg)
            print(error_msg)
            return None
            
        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
        print(f"Successfully fetched {len(data)} rows for {ticker}")
        return data[required_columns]
        
    except Exception as e:
        error_msg = f"Error fetching data for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)
        return None


def get_historical_comparison(ticker, days=60):
    """Get historical data for model vs actual comparison with flexible date handling."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days+100)
    try:
        logger.info(f"Fetching historical comparison data for {ticker}: {days} days")
        
        data = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        ).reset_index()
        
        if data.empty:
            error_msg = f"No historical data found for {ticker}"
            logger.error(error_msg)
            return None
            
        # Ensure we have enough data points
        if len(data) < WINDOW + 1:
            error_msg = f"Insufficient historical data for {ticker}: {len(data)} rows, need at least {WINDOW + 1}"
            logger.error(error_msg)
            return None
            
        logger.info(f"Successfully fetched historical comparison data for {ticker}: {len(data)} rows")
        return data[['Date', 'Close', 'Volume']]
        
    except Exception as e:
        error_msg = f"Error fetching historical data for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)
        return None

# ─────────────────────────────────────────────────────────────
# Update these to match pandas_ta’s CDL_* function names:
SHORT_TERM_PATTERNS = [
    'DOJI', 'HAMMER', 'INVERTED_HAMMER', 'ENGULFING', 'HARAMI',
    'PIERCING', 'DARK_CLOUD_COVER', 'KICKING', 'MORNING_STAR', 'EVENING_STAR'
]

LONG_TERM_PATTERNS = [
    'HEAD_AND_SHOULDERS', 'DOUBLE_BOTTOM', 'DOUBLE_TOP',
    'TRIPLE_BOTTOM', 'TRIPLE_TOP', 'RISING_WEDGE',
    'FALLING_WEDGE', 'BELT_HOLD'
]
# ─────────────────────────────────────────────────────────────

def detect_candlestick_patterns(df, time_frame):
    """
    Detect candlestick patterns based on the time frame.

    For each pattern:
      1) Look at the *last* bar:
         >0 → "Bullish", <0 → "Bearish"
      2) If last bar is 0, scan entire series:
         any >0 → "Bullish"
         any <0 → "Bearish"
      3) Otherwise → "No"
    """
    # pick short vs. long term by your existing logic
    patterns = SHORT_TERM_PATTERNS if time_frame == '5' else LONG_TERM_PATTERNS
    results = {}

    for pat in patterns:
        try:
            # build the pandas_ta function name, e.g. "CDLDOJI", "CDL_HAMMER" etc.
            func_name = 'CDL' + pat
            series = getattr(ta_pat, func_name)(
                df['Open'], df['High'], df['Low'], df['Close']
            )

            # if the pattern function isn't present or returns nothing:
            if series is None or series.size == 0:
                results[pat] = "No"
                continue

            # 1) check the last candle
            last = int(series.iloc[-1])
            if last > 0:
                results[pat] = "Bullish"
                continue
            elif last < 0:
                results[pat] = "Bearish"
                continue

            # 2) fallback: scan the full window
            if (series > 0).any():
                results[pat] = "Bullish"
            elif (series < 0).any():
                results[pat] = "Bearish"
            else:
                results[pat] = "No"

        except AttributeError:
            # function not found in pandas_ta
            logger.error(f"Pattern function not found: {func_name}")
            results[pat] = "No"
        except Exception as e:
            logger.error(f"Error detecting {pat}: {e}")
            results[pat] = "Error"

    return results



@app.route('/')
def index():
    available_tickers = get_available_tickers()
    return render_template_string(HTML_TEMPLATE, tickers=available_tickers)


@app.route('/predict/<ticker>')
def predict_stock(ticker):
    try:
        logger.info(f"Starting prediction for ticker: {ticker}")
        
        model = load_best_model(ticker)
        if model is None:
            logger.warning(f"No trained model found for {ticker}")
            return jsonify({
                'error': f'No trained model found for {ticker}. This ticker is available for viewing but predictions require a trained model.',
                'ticker': ticker,
                'has_model': False
            })
            
        data = get_stock_data(ticker)
        if data is None:
            error_msg = f'Unable to fetch data for {ticker}'
            logger.error(error_msg)
            return jsonify({'error': error_msg})
            
        close_prices = data['Close'].values.astype('float32').reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        if len(scaled_prices) < WINDOW:
            logger.warning(f"Insufficient data for prediction: {len(scaled_prices)} < {WINDOW}")
            return jsonify({'error': 'Insufficient data for prediction'})
        last_window = scaled_prices[-WINDOW:, 0]
        X = np.array([last_window])
        X = np.reshape(X, (X.shape[0], 1, WINDOW))
        current_price = close_prices[-1][0]
        
        # Make predictions
        predictions = {}
        pred_next, pred_month, pred_dir = model.predict(X, verbose=0)
        direction_label = int(np.argmax(pred_dir[0]))
        direction_text = assign_label_text(direction_label)
        next_day_price = float(scaler.inverse_transform([[pred_next[0][0]]])[0][0])
        month_price = float(scaler.inverse_transform([[pred_month[0][0]]])[0][0])
        daily_change_rate = (month_price - current_price) / 20  # 20-day horizon from model
        
        # Define the time periods we're interested in
        time_periods = [5, 30, 90, 365]
        for days in time_periods:
            predicted_price = current_price + (daily_change_rate * days)
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            predictions[f'{days}_days'] = {
                'price': f"${predicted_price:.2f}",
                'change': round(float(price_change), 2),
                'percent_change': round(float(percent_change), 2)
            }
        
        predictions['direction'] = {
            'label': direction_text,
            'confidence': round(float(np.max(pred_dir[0])) * 100, 1)
        }
        predictions['current_price'] = f"${current_price:.2f}"
        predictions['ticker'] = ticker
        predictions['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predictions['has_model'] = True
        
        # Fetch additional metrics
        try:
            logger.info(f"Fetching additional metrics for {ticker}")
            tk = yf.Ticker(ticker)
            info = tk.info
            
            def format_metric(value, format_type='general'):
                if value is None or value == 'N/A':
                    return 'N/A'
                if format_type == 'currency' and isinstance(value, (int, float)):
                    return f"${value:,.2f}"
                if format_type == 'percentage' and isinstance(value, (int, float)):
                    return f"{value:.2%}"
                if format_type == 'large_number' and isinstance(value, (int, float)):
                    if value >= 1e12:
                        return f"${value/1e12:.2f}T"
                    elif value >= 1e9:
                        return f"${value/1e9:.2f}B"
                    elif value >= 1e6:
                        return f"${value/1e6:.2f}M"
                    else:
                        return f"${value:,.0f}"
                if format_type == 'volume' and isinstance(value, (int, float)):
                    if value >= 1e6:
                        return f"{value/1e6:.1f}M"
                    elif value >= 1e3:
                        return f"{value/1e3:.1f}K"
                    else:
                        return f"{value:,.0f}"
                return str(value)
                
            predictions['metrics'] = {
                'Market Cap': format_metric(info.get('marketCap'), 'large_number'),
                'P/E Ratio': format_metric(info.get('trailingPE')),
                'Forward P/E': format_metric(info.get('forwardPE')),
                'Dividend Yield': format_metric(info.get('dividendYield'), 'percentage'),
                'Beta': format_metric(info.get('beta')),
                'Volume': format_metric(info.get('volume'), 'volume'),
                'Avg Volume': format_metric(info.get('averageVolume'), 'volume'),
                '52W High': format_metric(info.get('fiftyTwoWeekHigh'), 'currency'),
                '52W Low': format_metric(info.get('fiftyTwoWeekLow'), 'currency'),
                'EPS': format_metric(info.get('trailingEps'), 'currency')
            }
            logger.info(f"Successfully fetched metrics for {ticker}")
            
        except Exception as e:
            error_msg = f"Error fetching metrics for {ticker}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            print(error_msg)
            predictions['metrics'] = {
                'Market Cap': 'N/A',
                'P/E Ratio': 'N/A',
                'Forward P/E': 'N/A',
                'Dividend Yield': 'N/A',
                'Beta': 'N/A',
                'Volume': 'N/A',
                'Avg Volume': 'N/A',
                '52W High': 'N/A',
                '52W Low': 'N/A',
                'EPS': 'N/A'
            }
            
        logger.info(f"Successfully completed prediction for {ticker}")
        return jsonify(predictions)
        
    except Exception as e:
        error_msg = f"Unexpected error in predict_stock for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})


@app.route('/historical/<ticker>')
def historical_comparison(ticker):
    try:
        logger.info(f"Fetching historical comparison for {ticker}")
        
        model = load_best_model(ticker)
        if model is None:
            error_msg = f'No trained model found for {ticker}'
            logger.error(error_msg)
            return jsonify({'error': error_msg})
            
        data = get_historical_comparison(ticker, days=60)
        if data is None:
            error_msg = f'Insufficient historical data for {ticker}'
            logger.error(error_msg)
            return jsonify({'error': error_msg})
            
        close_prices = data['Close'].values.astype('float32').reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        
        dates = []
        actual_prices = []
        predicted_prices = []
        volumes = []
        
        # Start from the point where we have enough data for a full window
        start_idx = WINDOW
        for i in range(start_idx, len(data)):
            window_data = scaled_prices[i-WINDOW:i, 0]
            X = np.array([window_data]).reshape(1, 1, WINDOW)
            pred_next, _, _ = model.predict(X, verbose=0)
            pred_price = scaler.inverse_transform([[pred_next[0][0]]])[0][0]
            dates.append(data['Date'].iloc[i].strftime('%Y-%m-%d'))
            actual_prices.append(round(float(data['Close'].iloc[i]), 2))
            predicted_prices.append(round(float(pred_price), 2))
            volumes.append(int(data['Volume'].iloc[i]))
                
        logger.info(f"Successfully generated historical comparison for {ticker}: {len(dates)} data points")
        return jsonify({
            'dates': dates,
            'actual': actual_prices,
            'predicted': predicted_prices,
            'volume': volumes,
            'ticker': ticker
        })
        
    except Exception as e:
        error_msg = f"Error in historical_comparison for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})


@app.route('/prices/<ticker>')
def prices(ticker):
    """Return actual closing prices for a given range of days with flexible date handling."""
    try:
        range_days = int(request.args.get('range', 30))
        logger.info(f"Fetching prices for {ticker}, range: {range_days} days")
        
        # Fetch data with buffer to account for missing days
        data = get_stock_data(ticker, days_back=range_days + 20)
        if data is None or data.empty:
            error_msg = f'Unable to fetch price data for {ticker}'
            logger.error(error_msg)
            return jsonify({'error': error_msg})
            
        # Take the last 'range_days' entries
        df = data.tail(range_days)
        logger.info(f"Successfully fetched {len(df)} price data points for {ticker}")
        
        return jsonify({
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'open': df['Open'].tolist(),
            'high': df['High'].tolist(),
            'low': df['Low'].tolist(),
            'close': df['Close'].tolist(),
            'volume': df['Volume'].tolist(),
            'ticker': ticker
        })
        
    except Exception as e:
        error_msg = f"Error in prices endpoint for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})


# --- Monkey-patch fundamentals retrieval ---
def _patched_get_fundamentals(self, proxy=None):
    url  = f"{self._scrape_url}/{self.ticker}/financials"
    return utils.get_json(url, proxy)

Ticker._get_fundamentals = _patched_get_fundamentals

# --- Helper to convert DFs to JSON-safe records ---
def df_to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    df = df.copy()
    df.columns = [
        c.strftime('%Y-%m-%d') if isinstance(c, (pd.Timestamp, pd.Period)) else str(c)
        for c in df.columns
    ]
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient='records')



@app.route('/fundamentals/<ticker>')
def fundamentals(ticker):
    try:
        logger.info(f"Fetching fundamentals for {ticker}")
        tk = yf.Ticker(ticker)

        # Annual earnings
        earnings = (
            tk.earnings.reset_index()
            if getattr(tk, 'earnings', None) is not None else pd.DataFrame()
        )
        logger.info(f"Earnings rows: {len(earnings)}")

        # Quarterly earnings
        quarterly = (
            tk.quarterly_earnings.reset_index()
            if getattr(tk, 'quarterly_earnings', None) is not None else pd.DataFrame()
        )
        logger.info(f"Quarterly earnings rows: {len(quarterly)}")

        # Balance sheet
        bs = (
            tk.balance_sheet.reset_index()
            if getattr(tk, 'balance_sheet', None) is not None else pd.DataFrame()
        )
        logger.info(f"Balance sheet rows: {len(bs)}")

        # Financials (income statement)
        fin = (
            tk.financials.reset_index()
            if getattr(tk, 'financials', None) is not None else pd.DataFrame()
        )
        logger.info(f"Financials rows: {len(fin)}")

        # Cashflow
        cf = (
            tk.cashflow.reset_index()
            if getattr(tk, 'cashflow', None) is not None else pd.DataFrame()
        )
        logger.info(f"Cashflow rows: {len(cf)}")

        logger.info(f"Successfully fetched all fundamentals for {ticker}")
        return jsonify({
            'earnings': df_to_records(earnings),
            'quarterlyEarnings': df_to_records(quarterly),
            'balanceSheet': df_to_records(bs),
            'financials': df_to_records(fin),
            'cashflow': df_to_records(cf)
        })
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/indicators/<ticker>')
def technical_indicators(ticker):
    """Calculate and return technical indicators optimized for day trading."""
    try:
        # keep 20 extra days so that short‐period EMAs stabilize
        buffer_days = 20
        data = get_stock_data(ticker, days_back=30 + buffer_days)
        if data is None or data.empty:
            return jsonify({'error': 'Unable to fetch data for ticker'}), 400

        # use only the last 30 days for generating signals
        df = data.tail(30).copy()
        df.set_index('Date', inplace=True)

        # === MACD optimized for day trading: fast=5, slow=35, signal=5 ===
        macd = MACD(
            close=df['Close'],
            window_fast=5,
            window_slow=35,
            window_sign=5
        )
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # === RSI optimized for day trading: window=7 ===
        rsi = RSIIndicator(close=df['Close'], window=7)
        df['rsi'] = rsi.rsi()

        # === SMAs optimized for day trading: 5 & 13 periods ===
        sma_s = SMAIndicator(close=df['Close'], window=5)
        df['ma_short'] = sma_s.sma_indicator()
        sma_l = SMAIndicator(close=df['Close'], window=13)
        df['ma_long']  = sma_l.sma_indicator()

        # === Stochastic Oscillator optimized: %K=9, %D=3 ===
        sto = StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=9,
            smooth_window=3
        )
        df['stoch_k'] = sto.stoch()
        df['stoch_d'] = sto.stoch_signal()

        # === Bollinger Bands optimized: 10-period SMA, 1.5 std dev ===
        bb = BollingerBands(
            close=df['Close'],
            window=10,
            window_dev=1.5
        )
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_upper']  = bb.bollinger_hband()
        df['bb_lower']  = bb.bollinger_lband()

        # === On-Balance Volume (no parameters to tune) ===
        obv = OnBalanceVolumeIndicator(
            close=df['Close'],
            volume=df['Volume']
        )
        df['obv'] = obv.on_balance_volume()

        # Determine signals for each day
        signals = []
        for i in range(len(df)):
            date = df.index[i].strftime('%Y-%m-%d')

            # MACD crossover
            macd_signal = "None"
            if i > 0:
                if df['macd'].iat[i] > df['macd_signal'].iat[i] and df['macd'].iat[i-1] <= df['macd_signal'].iat[i-1]:
                    macd_signal = "Buy"
                elif df['macd'].iat[i] < df['macd_signal'].iat[i] and df['macd'].iat[i-1] >= df['macd_signal'].iat[i-1]:
                    macd_signal = "Sell"

            # RSI extremes
            rsi_signal = "None"
            if df['rsi'].iat[i] < 30:
                rsi_signal = "Buy"
            elif df['rsi'].iat[i] > 70:
                rsi_signal = "Sell"

            # SMA crossover
            ma_signal = "None"
            if i > 0:
                if df['ma_short'].iat[i] > df['ma_long'].iat[i] and df['ma_short'].iat[i-1] <= df['ma_long'].iat[i-1]:
                    ma_signal = "Buy"
                elif df['ma_short'].iat[i] < df['ma_long'].iat[i] and df['ma_short'].iat[i-1] >= df['ma_long'].iat[i-1]:
                    ma_signal = "Sell"

            # Stochastic crossover
            stoch_signal = "None"
            if i > 0:
                if df['stoch_k'].iat[i] > df['stoch_d'].iat[i] and df['stoch_k'].iat[i-1] <= df['stoch_d'].iat[i-1]:
                    stoch_signal = "Buy"
                elif df['stoch_k'].iat[i] < df['stoch_d'].iat[i] and df['stoch_k'].iat[i-1] >= df['stoch_d'].iat[i-1]:
                    stoch_signal = "Sell"

            # Bollinger Bands
            bb_signal = "None"
            if df['Close'].iat[i] < df['bb_lower'].iat[i]:
                bb_signal = "Buy"
            elif df['Close'].iat[i] > df['bb_upper'].iat[i]:
                bb_signal = "Sell"

            # OBV trend
            obv_signal = "None"
            if i > 0:
                if df['obv'].iat[i] > df['obv'].iat[i-1]:
                    obv_signal = "Buy"
                elif df['obv'].iat[i] < df['obv'].iat[i-1]:
                    obv_signal = "Sell"

            signals.append({
                'date': date,
                'macd': macd_signal,
                'rsi': rsi_signal,
                'moving_averages': ma_signal,
                'stochastic': stoch_signal,
                'bollinger': bb_signal,
                'obv': obv_signal
            })

        logger.info(f"Successfully calculated technical signals for {ticker} (day-trader settings)")
        return jsonify({'ticker': ticker, 'signals': signals})

    except Exception as e:
        logger.error(f"Error calculating technical indicators for {ticker}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500



@app.route('/closing_price/<ticker>')
def closing_price_data(ticker):
    """Return closing prices for different time frames."""
    try:
        range_days = int(request.args.get('range', 30))
        logger.info(f"Fetching closing prices for {ticker}, range: {range_days} days")
        
        # Fetch data with buffer to account for missing days
        data = get_stock_data(ticker, days_back=range_days + 20)
        if data is None or data.empty:
            error_msg = f'Unable to fetch price data for {ticker}'
            logger.error(error_msg)
            return jsonify({'error': error_msg})
            
        # Take the last 'range_days' entries
        df = data.tail(range_days)
        logger.info(f"Successfully fetched {len(df)} closing price data points for {ticker}")
        
        return jsonify({
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'close': df['Close'].tolist(),
            'ticker': ticker
        })
        
    except Exception as e:
        error_msg = f"Error in closing_price endpoint for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})


@app.route('/candlestick/<ticker>')
def candlestick_data(ticker):
    """Return candlestick data with pattern recognition for different time frames."""
    try:
        range_days = int(request.args.get('range', 30))
        logger.info(f"Fetching candlestick data for {ticker}, range: {range_days} days")
        
        # Fetch data with buffer to account for missing days
        data = get_stock_data(ticker, days_back=range_days + 20)
        if data is None or data.empty:
            error_msg = f'Unable to fetch price data for {ticker}'
            logger.error(error_msg)
            return jsonify({'error': error_msg})
            
        # Take the last 'range_days' entries
        df = data.tail(range_days)
        logger.info(f"Successfully fetched {len(df)} candlestick data points for {ticker}")
        
        # Detect candlestick patterns
        time_frame = str(range_days)
        patterns = detect_candlestick_patterns(df, time_frame)
        
        return jsonify({
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'open': df['Open'].tolist(),
            'high': df['High'].tolist(),
            'low': df['Low'].tolist(),
            'close': df['Close'].tolist(),
            'volume': df['Volume'].tolist(),
            'patterns': patterns,
            'ticker': ticker
        })
        
    except Exception as e:
        error_msg = f"Error in candlestick endpoint for {ticker}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})


# =============================================================================
# PUSH NOTIFICATION SYSTEM (PUSHOVER)
# =============================================================================

class PushNotifier:
    def __init__(self, pushover_token, pushover_user):
        self.token = pushover_token
        self.user = pushover_user
        self.url = "https://api.pushover.net/1/messages.json"
    
    def send_push_notification(self, predictions):
        """Send push notification via Pushover"""
        try:
            message = f"🚀 {len(predictions)} High-Confidence Stock Predictions:\n\n"
            for pred in predictions:
                trend_emoji = "🚀" if "Up" in pred['direction'] else "📉"
                message += f"{trend_emoji} {pred['ticker']}: {pred['direction']} ({pred['confidence']}%)\n"
                message += f"Current: ${pred['current_price']} → 30d: ${pred['30_day_price']}\n"
                message += f"Change: {pred['30_day_change']:+.1f}%\n\n"
            data = {
                "token": self.token,
                "user": self.user,
                "title": "📈 Stock Predictions Alert",
                "message": message[:1024],
                "priority": 1,
                "sound": "cashregister"
            }
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                logger.info("Push notification sent successfully")
                print("Push notification sent successfully")
                return True
            else:
                error_msg = f"Failed to send push notification: {response.text}"
                logger.error(error_msg)
                print(error_msg)
                return False
        except Exception as e:
            error_msg = f"Error sending push notification: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            print(error_msg)
            return False

# =============================================================================
# MAIN NOTIFICATION SYSTEM
# =============================================================================

class StockNotificationSystem:
    def __init__(self, app, pushover_token=None, pushover_user=None):
        self.app = app
        self.db_path = "notifications.db"
        self.init_database()
        if pushover_token and pushover_user:
            self.push_notifier = PushNotifier(pushover_token, pushover_user)
        else:
            self.push_notifier = None
            print("Push notifications disabled - no Pushover credentials provided")
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                confidence REAL,
                direction TEXT,
                price REAL,
                prediction_date TEXT,
                sent_date TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_high_confidence_predictions(self, min_confidence=50):
        high_confidence_predictions = []
        with self.app.app_context():
            available_tickers = get_available_models()
            for ticker in available_tickers:
                try:
                    model = load_best_model(ticker)
                    if model is None:
                        continue
                    data = get_stock_data(ticker)
                    if data is None:
                        continue
                    close_prices = data['Close'].values.astype('float32').reshape(-1, 1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_prices = scaler.fit_transform(close_prices)
                    if len(scaled_prices) < WINDOW:
                        continue
                    last_window = scaled_prices[-WINDOW:, 0]
                    X = np.array([last_window]).reshape(1, 1, WINDOW)
                    current_price = close_prices[-1][0]
                    
                    # Make predictions
                    pred_next, pred_month, pred_dir = model.predict(X, verbose=0)
                    direction_label = int(np.argmax(pred_dir[0]))
                    direction_text = assign_label_text(direction_label)
                    confidence = round(float(np.max(pred_dir[0])) * 100, 1)
                    
                    if confidence > min_confidence:
                        # Calculate 30-day prediction
                        next_day_price = float(scaler.inverse_transform([[pred_next[0][0]]])[0][0])
                        month_price = float(scaler.inverse_transform([[pred_month[0][0]]])[0][0])
                        daily_change_rate = (month_price - current_price) / 20
                        predicted_30_price = current_price + (daily_change_rate * 30)
                        price_change = predicted_30_price - current_price
                        percent_change = (price_change / current_price) * 100
                        
                        pred_data = {
                            'ticker': ticker,
                            'current_price': round(current_price, 2),
                            'direction': direction_text,
                            'confidence': confidence,
                            '30_day_price': round(predicted_30_price, 2),
                            '30_day_change': round(percent_change, 2)
                        }
                        high_confidence_predictions.append(pred_data)
                except Exception as e:
                    error_msg = f"Error processing {ticker}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    print(error_msg)
                    continue
        return high_confidence_predictions
    
    def send_notifications(self, predictions):
        if not predictions:
            logger.info("No high-confidence predictions to send")
            print("No high-confidence predictions to send")
            return
        if self.push_notifier:
            success = self.push_notifier.send_push_notification(predictions)
            if success:
                self.log_notifications(predictions)
        else:
            print("Push notifications not configured. Set PUSHOVER_TOKEN and PUSHOVER_USER.")
    
    def log_notifications(self, predictions):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for pred in predictions:
                cursor.execute('''
                    INSERT INTO notifications (ticker, confidence, direction, price, prediction_date, sent_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pred['ticker'],
                    pred['confidence'],
                    pred['direction'],
                    pred['current_price'],
                    datetime.now().strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
            conn.commit()
            conn.close()
            logger.info(f"Logged {len(predictions)} notifications to database")
        except Exception as e:
            error_msg = f"Error logging notifications: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
    
    def start_scheduler(self):
        def run_scheduler():
            schedule.every().day.at("09:00").do(self.daily_notification_job)
            schedule.every().hour.do(self.hourly_notification_job)
            while True:
                schedule.run_pending()
                time.sleep(60)
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("📱 Push notification scheduler started!")
        print("📱 Push notification scheduler started!")
        print("Daily notifications at 9:00 AM, hourly checks during market hours")
    
    def daily_notification_job(self):
        try:
            logger.info("🔍 Running daily high-confidence prediction check...")
            print("🔍 Running daily high-confidence prediction check...")
            predictions = self.get_high_confidence_predictions(min_confidence=50)
            if predictions:
                logger.info(f"📈 Found {len(predictions)} high-confidence predictions")
                print(f"📈 Found {len(predictions)} high-confidence predictions")
                self.send_notifications(predictions)
            else:
                logger.info("📊 No high-confidence predictions found today")
                print("📊 No high-confidence predictions found today")
        except Exception as e:
            error_msg = f"Error in daily notification job: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
    
    def hourly_notification_job(self):
        try:
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:
                predictions = self.get_high_confidence_predictions(min_confidence=75)
                if predictions:
                    logger.info(f"⚡ Hourly check: Found {len(predictions)} very high-confidence predictions")
                    print(f"⚡ Hourly check: Found {len(predictions)} very high-confidence predictions")
                    self.send_notifications(predictions)
        except Exception as e:
            error_msg = f"Error in hourly notification job: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

# =============================================================================
# NOTIFICATION ROUTES
# =============================================================================

def setup_notifications(app, pushover_token=None, pushover_user=None):
    notification_system = StockNotificationSystem(app, pushover_token, pushover_user)
    
    @app.route('/test-notifications')
    def test_notifications():
        try:
            predictions = notification_system.get_high_confidence_predictions(min_confidence=30)
            notification_system.send_notifications(predictions)
            return jsonify({
                'message': f'Test notification sent for {len(predictions)} predictions',
                'predictions': predictions
            })
        except Exception as e:
            error_msg = f"Error in test notifications: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)})
    
    @app.route('/notification-history')  
    def notification_history():
        try:
            conn = sqlite3.connect(notification_system.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM notifications ORDER BY sent_date DESC LIMIT 50')
            history = cursor.fetchall()
            conn.close()
            return jsonify({
                'history': [
                    {
                        'ticker': row[1],
                        'confidence': row[2],
                        'direction': row[3],
                        'price': row[4],
                        'prediction_date': row[5],
                        'sent_date': row[6]
                    } for row in history
                ]
            })
        except Exception as e:
            error_msg = f"Error fetching notification history: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)})
    
    @app.route('/available-tickers')
    def available_tickers():
        try:
            all_tickers = get_available_tickers()
            trained_tickers = get_available_models()
            ticker_status = []
            for ticker in all_tickers:
                status = {
                    'ticker': ticker,
                    'has_model': ticker in trained_tickers,
                    'in_sp500': ticker in get_sp500_tickers()
                }
                ticker_status.append(status)
            return jsonify({
                'tickers': ticker_status,
                'total_count': len(all_tickers),
                'trained_count': len(trained_tickers),
                'sp500_count': len(get_sp500_tickers())
            })
        except Exception as e:
            error_msg = f"Error fetching available tickers: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)})
    
    notification_system.start_scheduler()
    return notification_system

# HTML Template with updated time periods
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&P 500 Stock Price Predictor</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #2c3e50;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 10px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px 15px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .header p {
            font-size: 1rem;
            color: #7f8c8d;
            font-weight: 400;
        }

        .content {
            display: grid;
            gap: 25px;
        }

        .ticker-selection {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .ticker-selection h2 {
            font-size: 1.4rem;
            margin-bottom: 18px;
            color: #2c3e50;
            font-weight: 600;
        }

        .search-box {
            width: 100%;
            padding: 14px 18px;
            font-size: 1rem;
            border: 2px solid #e1e8ed;
            border-radius: 10px;
            outline: none;
            transition: all 0.3s ease;
            background: #f8f9fa;
            margin-bottom: 18px;
        }

        .search-box:focus {
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .ticker-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 18px;
            padding: 12px 0;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .ticker-stat {
            text-align: center;
        }

        .ticker-stat-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #667eea;
        }

        .ticker-stat-label {
            font-size: 0.75rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .ticker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
            gap: 8px;
            max-height: 350px;
            overflow-y: auto;
            padding: 8px;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
        }

        .ticker-btn {
            background: #f8f9fa;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            padding: 10px 6px;
            cursor: pointer;
            text-align: center;
            font-weight: 600;
            color: #2c3e50;
            transition: all 0.3s ease;
            font-size: 0.75rem;
            position: relative;
        }

        .ticker-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        }

        .ticker-btn.active {
            background: #27ae60;
            color: white;
            border-color: #27ae60;
        }

        .ticker-btn.has-model::after {
            content: '●';
            position: absolute;
            top: 2px;
            right: 5px;
            color: #27ae60;
            font-size: 10px;
        }

        .ticker-btn.has-model:hover::after {
            color: white;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 50px 15px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 14px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results-section {
            display: none;
            grid-template-columns: 1fr;
            gap: 25px;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .section-card h3 {
            font-size: 1.25rem;
            margin-bottom: 18px;
            color: #2c3e50;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .stock-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid #ecf0f1;
        }

        .stock-symbol {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
        }

        .current-price {
            font-size: 1.75rem;
            font-weight: 600;
            color: #27ae60;
        }

        .predictions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 18px;
        }

        .prediction-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #dee2e6;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .prediction-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }

        .prediction-card h3 {
            font-size: 0.95rem;
            color: #6c757d;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .prediction-price {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 6px;
        }

        .prediction-change {
            font-size: 0.9rem;
            font-weight: 600;
        }

        .positive {
            color: #27ae60;
        }

        .negative {
            color: #e74c3c;
        }

        .direction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .direction-card h3 {
            color: rgba(255, 255, 255, 0.9);
        }

        .direction-label {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .direction-confidence {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .table-wrapper {
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin-top: 12px;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        .comparison-table th {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .comparison-table td {
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
            transition: background-color 0.2s ease;
        }

        .comparison-table tr:hover {
            background-color: #f8f9fa;
        }

        .price-cell {
            font-weight: 600;
            color: #2c3e50;
        }

        .difference-positive {
            color: #27ae60;
            font-weight: 600;
        }

        .difference-negative {
            color: #e74c3c;
            font-weight: 600;
        }

        .accuracy-high {
            background: #27ae60;
            color: white;
            padding: 3px 6px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .accuracy-medium {
            background: #f39c12;
            color: white;
            padding: 3px 6px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .accuracy-low {
            background: #e74c3c;
            color: white;
            padding: 3px 6px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .signal-buy {
            color: #27ae60;
            font-weight: 600;
        }

        .signal-sell {
            color: #e74c3c;
            font-weight: 600;
        }

        .error {
            padding: 18px;
            background: #fee;
            border: 1px solid #fcc;
            border-radius: 8px;
            color: #c66;
            text-align: center;
            font-weight: 500;
        }

        .warning {
            padding: 18px;
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            color: #856404;
            text-align: center;
            font-weight: 500;
            margin-bottom: 18px;
        }

        .error-display {
            background: #ffebee;
            border: 1px solid #f44336;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: #d32f2f;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
        }

        .error-display h4 {
            color: #c62828;
            margin-bottom: 10px;
            font-family: inherit;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            background: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        .tablinks {
            padding: 10px 20px;
            background: #f8f9fa;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            color: #2c3e50;
            transition: all 0.3s ease;
        }
        
        .tablinks:hover {
            background: #e9ecef;
        }
        
        .tablinks.active {
            background: #667eea;
            color: white;
        }
        
        .tabcontent {
            display: none;
        }
        
        .tabcontent.active {
            display: grid;
            gap: 25px;
        }
        
        .chart-container {
            height: 400px;
            margin-top: 20px;
        }
        
        .pattern-table {
            margin-top: 20px;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8rem;
            }
            
            .predictions-grid {
                grid-template-columns: 1fr;
            }
            
            .stock-header {
                flex-direction: column;
                gap: 8px;
                text-align: center;
            }
            
            .ticker-grid {
                grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
            }
            
            .tabs {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> S&P 500 Stock Price Predictor</h1>
            <p>AI-Powered Stock Price Predictions & Technical Analysis</p>
        </div>
        
        <div class="content">
            <div class="ticker-selection">
                <h2><i class="fas fa-search"></i> Select a Stock Ticker</h2>
                
                <div class="ticker-stats" id="tickerStats">
                    <div class="ticker-stat">
                        <div class="ticker-stat-value">{{ tickers|length }}</div>
                        <div class="ticker-stat-label">Total Tickers</div>
                    </div>
                    <div class="ticker-stat">
                        <div class="ticker-stat-value" id="trainedCount">-</div>
                        <div class="ticker-stat-label">With Models</div>
                    </div>
                    <div class="ticker-stat">
                        <div class="ticker-stat-value" id="sp500Count">-</div>
                        <div class="ticker-stat-label">S&P 500</div>
                    </div>
                </div>
                
                <input type="text" class="search-box" placeholder="Search for a ticker symbol (e.g., AAPL, GOOGL, MSFT...)" id="searchBox">
                
                <div class="ticker-grid" id="tickerGrid">
                    {% for ticker in tickers %}
                    <div class="ticker-btn" onclick="selectTicker('{{ ticker }}')" data-ticker="{{ ticker }}">{{ ticker }}</div>
                    {% endfor %}
                </div>
                
                <div style="margin-top: 12px; padding: 8px; background: #e8f4f8; border-radius: 8px; font-size: 0.85rem; color: #2c3e50;">
                    <i class="fas fa-info-circle"></i> 
                    <strong>Legend:</strong> Green dot (●) indicates tickers with trained AI models for predictions. 
                    All S&P 500 stocks are available for basic information viewing.
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Loading predictions...</p>
            </div>
            
            <div class="results-section" id="results">
                <div class="tabs">
                    <button class="tablinks active" onclick="openTab(event, 'overview')">Overview</button>
                    <button class="tablinks" onclick="openTab(event, 'closingPrice')">Closing Price</button>
                    <button class="tablinks" onclick="openTab(event, 'candlestick')">Candlestick</button>
                </div>
                
                <div id="overview" class="tabcontent active">
                    <div class="section-card">
                        <h3><i class="fas fa-chart-bar"></i> Key Stock Metrics</h3>
                        <div class="table-wrapper">
                            <table id="metricsTable" class="comparison-table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                
                    <div class="section-card" id="stockInfo">
                        <!-- Stock information will be populated here -->
                    </div>
                    
                    <div class="section-card">
                        <h3><i class="fas fa-exchange-alt"></i> Model Predictions vs Actual Prices</h3>
                        <div id="tableContainer" style="margin-top: 20px;">
                            <div class="table-header">
                                <label for="tableRangeSelect">Days:</label>
                                <select id="tableRangeSelect">
                                    <option value="10">10 Days</option>
                                    <option value="30">30 Days</option>
                                    <option value="60" selected>60 Days</option>
                                </select>
                            </div>
                            <div class="table-wrapper">
                                <table id="comparisonTable" class="comparison-table">
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Volume</th>
                                            <th>Actual Price</th>
                                            <th>Predicted Price</th>
                                            <th>Difference</th>
                                            <th>% Error</th>
                                            <th>Accuracy</th>
                                        </tr>
                                    </thead>
                                    <tbody id="tableBody"></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section-card" id="signals">
    <h3><i class="fas fa-signal"></i> Technical Signals (Last 30 Days)</h3>
    <div class="table-wrapper">
        <table id="signalsTable" class="comparison-table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>MACD</th>
                    <th>RSI</th>
                    <th>Moving Averages</th>
                    <th>Stochastic</th>
                    <th>Bollinger</th>
                    <th>OBV</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
</div>
                    
                    <div class="section-card" id="fundamentals">
                        <h3><i class="fas fa-building"></i> Fundamentals</h3>
                        <p>Loading...</p>
                    </div>
                </div>
                
                <div id="closingPrice" class="tabcontent">
                    <div class="section-card">
                        <h3><i class="fas fa-chart-line"></i> Closing Price</h3>
                        <div>
                            <label for="closingPriceRange">Time Frame:</label>
                            <select id="closingPriceRange">
                                <option value="5">5 Days</option>
                                <option value="30">30 Days</option>
                                <option value="90">90 Days</option>
                                <option value="365">12 Months</option>
                            </select>
                        </div>
                        <div id="closingPriceChart" class="chart-container"></div>
                    </div>
                </div>
                
                <div id="candlestick" class="tabcontent">
                    <div class="section-card">
                        <h3><i class="fas fa-candle-stick"></i> Candlestick Chart</h3>
                        <div>
                            <label for="candlestickRange">Time Frame:</label>
                            <select id="candlestickRange">
                                <option value="5">5 Days</option>
                                <option value="30">30 Days</option>
                                <option value="90">90 Days</option>
                                <option value="365">12 Months</option>
                            </select>
                        </div>
                        <div id="candlestickChart" class="chart-container"></div>
                    </div>
                    <div class="section-card pattern-table">
                        <h3><i class="fas fa-table"></i> Candlestick Pattern Recognition</h3>
                        <div id="patternTable"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedTicker = null;
        let tickerData = [];
        let historicalData = null;

        // Enhanced error display function
        function displayError(containerId, error, context = '') {
            const container = document.getElementById(containerId);
            const timestamp = new Date().toLocaleString();
            const errorHtml = `<div class="error-display">
                    <h4>Error ${context ? 'in ' + context : ''} [${timestamp}]</h4>
                    <div>${error}</div>
                </div>`;
            if (container) {
                container.innerHTML = errorHtml;
            }
            console.error(`[${timestamp}] Error ${context ? 'in ' + context : ''}: ${error}`);
        }

        // Enhanced logging function
        function logInfo(message, data = null) {
            const timestamp = new Date().toLocaleString();
            console.log(`[${timestamp}] ${message}`, data || '');
        }

        // Tab switching
        function openTab(evt, tabName) {
            try {
                // Hide all tabcontent
                const tabcontent = document.getElementsByClassName("tabcontent");
                for (let i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].classList.remove("active");
                }
                
                // Remove active class from tablinks
                const tablinks = document.getElementsByClassName("tablinks");
                for (let i = 0; i < tablinks.length; i++) {
                    tablinks[i].classList.remove("active");
                }
                
                // Show the current tab, add active class to the button
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.classList.add("active");
                
                // Load data if needed
                if (tabName === 'closingPrice' && selectedTicker) {
                    loadClosingPriceChart(selectedTicker);
                } else if (tabName === 'candlestick' && selectedTicker) {
                    loadCandlestickChart(selectedTicker);
                }
            } catch (error) {
                const errorMsg = `Error switching tabs: ${error.message}`;
                console.error(errorMsg);
                displayError('results', errorMsg, 'tab switching');
            }
        }

        // Load ticker data and update display
        async function loadTickerData() {
            try {
                logInfo('Loading ticker data...');
                const response = await fetch('/available-tickers');
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                tickerData = data.tickers;
                
                // Update stats
                document.getElementById('trainedCount').textContent = data.trained_count;
                document.getElementById('sp500Count').textContent = data.sp500_count;
                
                // Update ticker buttons with model indicators
                tickerData.forEach(ticker => {
                    const btn = document.querySelector(`[data-ticker="${ticker.ticker}"]`);
                    if (btn && ticker.has_model) {
                        btn.classList.add('has-model');
                        btn.title = `${ticker.ticker} - AI Model Available`;
                    } else if (btn) {
                        btn.title = `${ticker.ticker} - S&P 500 Stock (Basic Info Only)`;
                    }
                });
                
                logInfo('Ticker data loaded successfully', {
                    total: data.tickers.length,
                    trained: data.trained_count,
                    sp500: data.sp500_count
                });
                
            } catch (error) {
                const errorMsg = `Failed to load ticker data: ${error.message}`;
                console.error(errorMsg);
                displayError('tickerStats', errorMsg, 'ticker data loading');
            }
        }

        // Search functionality
        document.getElementById('searchBox').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const tickers = document.querySelectorAll('.ticker-btn');
            tickers.forEach(ticker => {
                const tickerText = ticker.textContent.toLowerCase();
                ticker.style.display = tickerText.includes(searchTerm) ? 'block' : 'none';
            });
        });

        function selectTicker(ticker) {
            try {
                selectedTicker = ticker;
                document.querySelectorAll('.ticker-btn').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                // Reset data
                historicalData = null;
                
                logInfo(`Selecting ticker: ${ticker}`);
                
                // Fetch all data with detailed logging
                Promise.all([
                    fetchPredictions(ticker).catch(e => {
                        console.error('Predictions failed:', e);
                        return null;
                    }),
                    fetchHistoricalData(ticker).catch(e => {
                        console.error('Historical data failed:', e);
                        return null;
                    }),
                    fetchFundamentals(ticker).catch(e => {
                        console.error('Fundamentals failed:', e);
                        return null;
                    }),
                    fetchSignals(ticker).catch(e => {
                        console.error('Signals failed:', e);
                        return null;
                    })
                ]).then(() => {
                    logInfo('All data fetching complete for ' + ticker);
                    logInfo('Historical data available: ' + !!historicalData);
                    
                    // Show results after all data is loaded
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').style.display = 'grid';
                }).catch(error => {
                    console.error('Error in Promise.all:', error);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').style.display = 'grid';
                });
                
            } catch (error) {
                const errorMsg = `Error selecting ticker ${ticker}: ${error.message}`;
                console.error(errorMsg);
                displayError('stockInfo', errorMsg, 'ticker selection');
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').style.display = 'grid';
            }
        }

        async function fetchPredictions(ticker) {
            try {
                logInfo(`Fetching predictions for ${ticker}`);
                const response = await fetch(`/predict/${ticker}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    if (!data.has_model) {
                        showWarning(`${ticker} is an S&P 500 stock but no AI model is available for predictions. Showing available information only.`);
                        displayBasicInfo(data);
                    } else {
                        throw new Error(data.error);
                    }
                    return null;
                }
                
                displayMetrics(data.metrics);
                displayPredictions(data);
                logInfo('Predictions displayed successfully for ' + ticker);
                return data;
                
            } catch (error) {
                const errorMsg = `Failed to fetch predictions for ${ticker}: ${error.message}`;
                console.error(errorMsg);
                displayError('stockInfo', errorMsg, 'predictions fetch');
                return null;
            }
        }

        async function fetchHistoricalData(ticker) {
            try {
                logInfo(`Fetching historical data for ${ticker}`);
                const response = await fetch(`/historical/${ticker}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    logInfo('Historical data not available: ' + data.error);
                    return null;
                }
                
                historicalData = data;
                populateComparisonTable(data, 60);
                logInfo('Historical data fetched successfully for ' + ticker);
                return data;
                
            } catch (error) {
                const errorMsg = `Failed to fetch historical data for ${ticker}: ${error.message}`;
                console.error(errorMsg);
                displayError('tableBody', errorMsg, 'historical data fetch');
                return null;
            }
        }

        function displayMetrics(metrics) {
            try {
                const tbody = document.querySelector('#metricsTable tbody');
                tbody.innerHTML = '';
                for (const [key, val] of Object.entries(metrics)) {
                    const row = `<tr>
                        <td>${key}</td>
                        <td>${val !== null ? val : 'N/A'}</td>
                    </tr>`;
                    tbody.innerHTML += row;
                }
                logInfo('Metrics displayed successfully');
            } catch (error) {
                const errorMsg = `Error displaying metrics: ${error.message}`;
                console.error(errorMsg);
                displayError('metricsTable', errorMsg, 'metrics display');
            }
        }

        function displayPredictions(data) {
            try {
                const stockInfo = document.getElementById('stockInfo');
                stockInfo.innerHTML = 
                    `<div class="stock-header">
                        <div class="stock-symbol">${data.ticker}</div>
                        <div class="current-price">${data.current_price}</div>
                    </div>
                    <div class="predictions-grid">
                        <div class="prediction-card">
                            <h3>5 Days</h3>
                            <div class="prediction-price">${data['5_days'].price}</div>
                            <div class="prediction-change ${data['5_days'].change >= 0 ? 'positive' : 'negative'}">
                                ${data['5_days'].change >= 0 ? '+' : ''}${data['5_days'].change} (${data['5_days'].percent_change}%)
                            </div>
                        </div>
                        <div class="prediction-card">
                            <h3>30 Days</h3>
                            <div class="prediction-price">${data['30_days'].price}</div>
                            <div class="prediction-change ${data['30_days'].change >= 0 ? 'positive' : 'negative'}">
                                ${data['30_days'].change >= 0 ? '+' : ''}${data['30_days'].change} (${data['30_days'].percent_change}%)
                            </div>
                        </div>
                        <div class="prediction-card">
                            <h3>90 Days</h3>
                            <div class="prediction-price">${data['90_days'].price}</div>
                            <div class="prediction-change ${data['90_days'].change >= 0 ? 'positive' : 'negative'}">
                                ${data['90_days'].change >= 0 ? '+' : ''}${data['90_days'].change} (${data['90_days'].percent_change}%)
                            </div>
                        </div>
                        <div class="prediction-card">
                            <h3>365 Days</h3>
                            <div class="prediction-price">${data['365_days'].price}</div>
                            <div class="prediction-change ${data['365_days'].change >= 0 ? 'positive' : 'negative'}">
                                ${data['365_days'].change >= 0 ? '+' : ''}${data['365_days'].change} (${data['365_days'].percent_change}%)
                            </div>
                        </div>
                        <div class="prediction-card direction-card">
                            <div class="direction-label">${data.direction.label}</div>
                            <div class="direction-confidence">Confidence: ${data.direction.confidence}%</div>
                        </div>
                    </div>
                    <div style="text-align: center; color: #7f8c8d; margin-top: 15px; font-style: italic;">
                        <i class="fas fa-clock"></i> Last Updated: ${data.last_updated}
                    </div>`;
                logInfo('Predictions displayed successfully');
            } catch (error) {
                const errorMsg = `Error displaying predictions: ${error.message}`;
                console.error(errorMsg);
                displayError('stockInfo', errorMsg, 'predictions display');
            }
        }

        function displayBasicInfo(data) {
            try {
                const stockInfo = document.getElementById('stockInfo');
                stockInfo.innerHTML = 
                    `<div class="stock-header">
                        <div class="stock-symbol">${data.ticker}</div>
                        <div style="font-size: 1.1rem; color: #f39c12;">
                            <i class="fas fa-info-circle"></i> Basic Info Only
                        </div>
                    </div>
                    <div style="text-align: center; padding: 30px; background: #fff3cd; border-radius: 10px; margin: 15px 0;">
                        <h3 style="color: #856404; margin-bottom: 12px;">
                            <i class="fas fa-robot"></i> AI Predictions Not Available
                        </h3>
                        <p style="color: #856404; margin-bottom: 12px;">
                            This S&P 500 stock doesn't have a trained AI model yet. You can view basic stock information and fundamentals below.
                        </p>
                        <p style="color: #856404; font-size: 0.85rem;">
                            Look for tickers with green dots (●) for AI predictions.
                        </p>
                    </div>`;
                logInfo('Basic info displayed successfully');
            } catch (error) {
                const errorMsg = `Error displaying basic info: ${error.message}`;
                console.error(errorMsg);
                displayError('stockInfo', errorMsg, 'basic info display');
            }
        }

        function populateComparisonTable(data, days) {
            try {
                const tableBody = document.getElementById('tableBody');
                tableBody.innerHTML = '';
                const startIdx = data.dates.length - days;
                
                for (let i = startIdx; i < data.dates.length; i++) {
                    const date = data.dates[i];
                    const vol = data.volume[i];
                    const actual = data.actual[i];
                    const pred = data.predicted[i];
                    const diff = pred - actual;
                    const pctErr = (diff / actual) * 100;
                    const absErr = Math.abs(pctErr);
                    
                    let accClass, accText;
                    if (absErr <= 2) { 
                        accClass='accuracy-high'; accText='High'; 
                    } else if (absErr <= 5) { 
                        accClass='accuracy-medium'; accText='Medium'; 
                    } else { 
                        accClass='accuracy-low'; accText='Low'; 
                    }
                    
                    const diffClass = diff >= 0 ? 'difference-positive' : 'difference-negative';
                    const sign = diff >= 0 ? '+' : '';
                    
                    tableBody.innerHTML += 
                        `<tr>
                            <td>${date}</td>
                            <td>${vol.toLocaleString()}</td>
                            <td class="price-cell">${actual.toFixed(2)}</td>
                            <td class="price-cell">${pred.toFixed(2)}</td>
                            <td class="${diffClass}">${sign}${diff.toFixed(2)}</td>
                            <td class="${diffClass}">${sign}${pctErr.toFixed(2)}%</td>
                            <td><span class="${accClass}">${accText}</span></td>
                        </tr>`;
                }
                logInfo('Comparison table populated successfully');
            } catch (error) {
                const errorMsg = `Error populating comparison table: ${error.message}`;
                console.error(errorMsg);
                displayError('tableBody', errorMsg, 'comparison table population');
            }
        }

        async function fetchSignals(ticker) {
            try {
                logInfo(`Fetching signals for ${ticker}`);
                const response = await fetch(`/indicators/${ticker}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displaySignals(data);
                logInfo('Signals fetched and displayed successfully for ' + ticker);
                return data;
                
            } catch (error) {
                const errorMsg = `Failed to load signals for ${ticker}: ${error.message}`;
                console.error(errorMsg);
                displayError('signals', errorMsg, 'signals fetch');
                return null;
            }
        }

   function displaySignals(data) {
    try {
        const tbody = document.querySelector('#signalsTable tbody');
        tbody.innerHTML = '';
        
        data.signals.forEach(signal => {
            const macdClass = signal.macd === 'Buy' ? 'signal-buy' : signal.macd === 'Sell' ? 'signal-sell' : '';
            const rsiClass = signal.rsi === 'Buy' ? 'signal-buy' : signal.rsi === 'Sell' ? 'signal-sell' : '';
            const maClass = signal.moving_averages === 'Buy' ? 'signal-buy' : signal.moving_averages === 'Sell' ? 'signal-sell' : '';
            const stochClass = signal.stochastic === 'Buy' ? 'signal-buy' : signal.stochastic === 'Sell' ? 'signal-sell' : '';
            const bbClass = signal.bollinger === 'Buy' ? 'signal-buy' : signal.bollinger === 'Sell' ? 'signal-sell' : '';
            const obvClass = signal.obv === 'Buy' ? 'signal-buy' : signal.obv === 'Sell' ? 'signal-sell' : '';
            
            tbody.innerHTML += 
                `<tr>
                    <td>${signal.date}</td>
                    <td class="${macdClass}">${signal.macd}</td>
                    <td class="${rsiClass}">${signal.rsi}</td>
                    <td class="${maClass}">${signal.moving_averages}</td>
                    <td class="${stochClass}">${signal.stochastic}</td>
                    <td class="${bbClass}">${signal.bollinger}</td>
                    <td class="${obvClass}">${signal.obv}</td>
                </tr>`;
        });
        logInfo('Signals table populated successfully');
    } catch (error) {
        const errorMsg = `Error displaying signals: ${error.message}`;
        console.error(errorMsg);
        displayError('signalsTable', errorMsg, 'signals display');
    }
}
        async function fetchFundamentals(ticker) {
            try {
                logInfo(`Fetching fundamentals for ${ticker}`);
                const response = await fetch(`/fundamentals/${ticker}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayFundamentals(data);
                logInfo('Fundamentals fetched and displayed successfully for ' + ticker);
                return data;
                
            } catch (error) {
                const errorMsg = `Failed to load fundamentals for ${ticker}: ${error.message}`;
                console.error(errorMsg);
                displayError('fundamentals', errorMsg, 'fundamentals fetch');
                return null;
            }
        }

        function displayFundamentals(data) {
            try {
                const cont = document.getElementById('fundamentals');
                cont.innerHTML = '<h3><i class="fas fa-building"></i> Fundamentals</h3>';
                
                function genTable(records, title, icon) {
                    if (!records || !records.length) {
                        return `<div style="margin: 15px 0;">
                            <h4><i class="${icon}"></i> ${title}</h4>
                            <p style="color: #7f8c8d; font-style: italic;">No data available</p>
                        </div>`;
                    }
                    const cols = Object.keys(records[0]);
                    let html = `<div style="margin: 15px 0;">
                        <h4><i class="${icon}"></i> ${title}</h4>
                        <div class="table-wrapper">
                            <table class="comparison-table">
                                <thead><tr>`;
                    html += cols.map(c => `<th>${c}</th>`).join('');
                    html += `</tr></thead><tbody>`;
                    records.forEach(r => {
                        html += '<tr>' + cols.map(c => `<td>${r[c] || 'N/A'}</td>`).join('') + '</tr>';
                    });
                    html += `</tbody></table></div></div>`;
                    return html;
                }
                
                cont.innerHTML += genTable(data.earnings, 'Annual Earnings', 'fas fa-calendar-alt');
                cont.innerHTML += genTable(data.quarterlyEarnings, 'Quarterly Earnings', 'fas fa-chart-pie');
                cont.innerHTML += genTable(data.balanceSheet.slice(0,1), 'Balance Sheet (Latest)', 'fas fa-balance-scale');
                cont.innerHTML += genTable(data.financials.slice(0,1), 'Financials (Latest)', 'fas fa-dollar-sign');
                cont.innerHTML += genTable(data.cashflow.slice(0,1), 'Cash Flow (Latest)', 'fas fa-money-bill-wave');
                
                logInfo('Fundamentals displayed successfully');
            } catch (error) {
                const errorMsg = `Error displaying fundamentals: ${error.message}`;
                console.error(errorMsg);
                displayError('fundamentals', errorMsg, 'fundamentals display');
            }
        }

        function showError(message) {
            const stockInfo = document.getElementById('stockInfo');
            stockInfo.innerHTML = `<div class="error"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
            document.getElementById('loading').style.display = 'none';
            document.getElementById('results').style.display = 'grid';
        }

        function showWarning(message) {
            const stockInfo = document.getElementById('stockInfo');
            stockInfo.innerHTML = `<div class="warning"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
        }

        // Table range selector event listener
        document.getElementById('tableRangeSelect').addEventListener('change', () => {
            try {
                const days = parseInt(document.getElementById('tableRangeSelect').value);
                if (historicalData) {
                    populateComparisonTable(historicalData, days);
                }
            } catch (error) {
                const errorMsg = `Error updating table range: ${error.message}`;
                console.error(errorMsg);
                displayError('tableBody', errorMsg, 'table range update');
            }
        });
        
        // Closing price chart
        async function loadClosingPriceChart(ticker) {
            try {
                const range = document.getElementById('closingPriceRange').value;
                logInfo(`Loading closing price chart for ${ticker}, range: ${range} days`);
                
                const response = await fetch(`/closing_price/${ticker}?range=${range}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Create Plotly trace
                const trace = {
                    x: data.dates,
                    y: data.close,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Closing Price',
                    line: {color: '#667eea', width: 2},
                    marker: {size: 6, color: '#764ba2'}
                };
                
                const layout = {
                    title: `${ticker} Closing Price (${range} Days)`,
                    xaxis: {title: 'Date'},
                    yaxis: {title: 'Price (USD)'},
                    showlegend: true,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#2c3e50'},
                    margin: {l: 50, r: 30, t: 50, b: 50}
                };
                
                Plotly.newPlot('closingPriceChart', [trace], layout);
                logInfo('Closing price chart rendered successfully');
                
            } catch (error) {
                const errorMsg = `Error loading closing price chart: ${error.message}`;
                console.error(errorMsg);
                displayError('closingPriceChart', errorMsg, 'closing price chart');
            }
        }
        
        // Candlestick chart
        async function loadCandlestickChart(ticker) {
            try {
                const range = document.getElementById('candlestickRange').value;
                logInfo(`Loading candlestick chart for ${ticker}, range: ${range} days`);
                
                const response = await fetch(`/candlestick/${ticker}?range=${range}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Create candlestick trace
                const trace = {
                    x: data.dates,
                    close: data.close,
                    high: data.high,
                    low: data.low,
                    open: data.open,
                    type: 'candlestick',
                    name: 'Candlestick',
                    increasing: {line: {color: '#27ae60'}},
                    decreasing: {line: {color: '#e74c3c'}}
                };
                
                const layout = {
                    title: `${ticker} Candlestick (${range} Days)`,
                    xaxis: {title: 'Date', rangeslider: {visible: false}},
                    yaxis: {title: 'Price (USD)'},
                    showlegend: false,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#2c3e50'},
                    margin: {l: 50, r: 30, t: 50, b: 50}
                };
                
                Plotly.newPlot('candlestickChart', [trace], layout);
                
                // Display patterns
                displayPatterns(data.patterns);
                logInfo('Candlestick chart rendered successfully');
                
            } catch (error) {
                const errorMsg = `Error loading candlestick chart: ${error.message}`;
                console.error(errorMsg);
                displayError('candlestickChart', errorMsg, 'candlestick chart');
            }
        }
        
        // Display candlestick patterns
        function displayPatterns(patterns) {
            try {
                const tableContainer = document.getElementById('patternTable');
                if (!patterns || Object.keys(patterns).length === 0) {
                    tableContainer.innerHTML = '<p>No candlestick patterns detected in this time frame.</p>';
                    return;
                }
                
                let tableHtml = `
                    <div class="table-wrapper">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Pattern</th>
                                    <th>Detected</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                for (const [pattern, detected] of Object.entries(patterns)) {
                    const formattedPattern = pattern.charAt(0).toUpperCase() + pattern.slice(1);
                    tableHtml += `
                        <tr>
                            <td>${formattedPattern}</td>
                            <td class="${detected === 'Yes' ? 'signal-buy' : 'signal-sell'}">
                                ${detected}
                            </td>
                        </tr>
                    `;
                }
                
                tableHtml += `
                            </tbody>
                        </table>
                    </div>
                `;
                
                tableContainer.innerHTML = tableHtml;
                logInfo('Pattern table displayed successfully');
            } catch (error) {
                const errorMsg = `Error displaying patterns: ${error.message}`;
                console.error(errorMsg);
                displayError('patternTable', errorMsg, 'pattern display');
            }
        }
        
        // Add event listeners for new range selectors
        document.getElementById('closingPriceRange').addEventListener('change', () => {
            if (selectedTicker) {
                loadClosingPriceChart(selectedTicker);
            }
        });
        
        document.getElementById('candlestickRange').addEventListener('change', () => {
            if (selectedTicker) {
                loadCandlestickChart(selectedTicker);
            }
        });

        // Add debugging to check if Plotly is loaded
        document.addEventListener('DOMContentLoaded', function() {
            try {
                logInfo('DOM loaded');
                loadTickerData();
            } catch (error) {
                const errorMsg = `Error during DOM initialization: ${error.message}`;
                console.error(errorMsg);
            }
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    import os
    os.makedirs('saved_models', exist_ok=True)
    
    try:
        logger.info("🔍 Loading S&P 500 tickers using PyTickerSymbols...")
        print("🔍 Loading S&P 500 tickers using PyTickerSymbols...")
        sp500_tickers = get_sp500_tickers()
        logger.info(f"📊 Found {len(sp500_tickers)} S&P 500 tickers")
        print(f"📊 Found {len(sp500_tickers)} S&P 500 tickers")
        
        # =============================================================================
        # PUSH NOTIFICATION CONFIGURATION
        # =============================================================================
        PUSHOVER_TOKEN = "an1pxdrmpsyfscpng5s3pg8c16qzxz"
        PUSHOVER_USER = "u3e6rkshozsnaookryxjdwwu5odk75"
        notification_system = setup_notifications(app, PUSHOVER_TOKEN, PUSHOVER_USER)
        
        import socket
        def find_available_port(start_port=5001):
            port = start_port
            while port < start_port + 100:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(('localhost', port))
                    sock.close()
                    return port
                except OSError:
                    port += 1
            return None
            
        available_port = find_available_port(5001)
        if available_port:
            logger.info(f"🚀 Starting S&P 500 Stock Prediction App on port {available_port}")
            print(f"🚀 Starting S&P 500 Stock Prediction App on port {available_port}")
            print(f"🌐 Access the application at: http://localhost:{available_port}")
            print(f"📱 Push notifications: {'Enabled' if PUSHOVER_TOKEN and PUSHOVER_USER else 'Disabled (configure Pushover credentials)'}")
            print(f"🧪 Test notifications at: http://localhost:{available_port}/test-notifications")
            print(f"📊 Notification history at: http://localhost:{available_port}/notification-history")
            print(f"📋 Available tickers API: http://localhost:{available_port}/available-tickers")
            
            trained_models = get_available_models()
            print(f"\n📈 Ticker Summary:")
            print(f"   • Total S&P 500 tickers: {len(sp500_tickers)}")
            print(f"   • Tickers with trained models: {len(trained_models)}")
            print(f"   • Total available for display: {len(get_available_tickers())}")
            
            if trained_models:
                print(f"\n🤖 Tickers with AI models: {', '.join(sorted(trained_models)[:10])}{'...' if len(trained_models) > 10 else ''}")
                
            print(f"\n🆕 New Features:")
            print(f"   • Added closing price charts with time frame selector")
            print(f"   • Added candlestick charts with pattern recognition")
            print(f"   • Added technical signals table")
            
            logger.info("Starting Flask application...")
            app.run(debug=True, host='0.0.0.0', port=available_port)
        else:
            error_msg = "❌ No available ports found. Please free up some ports and try again."
            logger.error(error_msg)
            print(error_msg)
            
    except Exception as e:
        error_msg = f"❌ Critical error during application startup: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)