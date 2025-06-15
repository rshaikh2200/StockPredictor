from flask import Flask, render_template_string, jsonify, request
import numpy as np
import pandas as pd
import yfinance as yf
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
warnings.filterwarnings('ignore')

# Additional imports for push notifications
import requests
import schedule
import time
import threading
import sqlite3

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


def get_sp500_tickers():
    """Return a list of S&P 500 tickers using the pytickersymbols package."""
    try:
        symbols = PyTickerSymbols()
        sp500_stocks = symbols.get_stocks_by_index("S&P 500")
        tickers = [stock['symbol'] for stock in sp500_stocks]
        return sorted(tickers)
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
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
    model_files = glob.glob(f'saved_models/{ticker}_model_epoch_*.keras')
    if not model_files:
        return None
    best_model_path = sorted(model_files)[-1]
    return load_model(best_model_path)


def get_stock_data(ticker, days_back=500):
    """Fetch recent stock data for predictions."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    try:
        data = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        ).reset_index()
        if data.empty:
            return None
        return data[['Date', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def get_closing_price_data(ticker, days):
    """Fetch closing price data for specific time frames."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # Add buffer for weekends/holidays
    try:
        data = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        ).reset_index()
        if data.empty:
            return None
        # Get the last 'days' entries
        return data[['Date', 'Close']].tail(days)
    except Exception as e:
        print(f"Error fetching closing price data for {ticker}: {e}")
        return None


def prepare_prediction_data(data):
    """Prepare data for model prediction."""
    close_prices = data['Close'].values.astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    if len(scaled_prices) < WINDOW:
        return None, None, None
    last_window = scaled_prices[-WINDOW:, 0]
    X = np.array([last_window])
    X = np.reshape(X, (X.shape[0], 1, WINDOW))
    return X, scaler, close_prices[-1][0]


def make_predictions(model, X, scaler, current_price, days_list=[30, 60, 90, 120]):
    """Make price predictions for multiple time horizons."""
    predictions = {}
    pred_next, pred_month, pred_dir = model.predict(X, verbose=0)
    direction_label = int(np.argmax(pred_dir[0]))
    direction_text = assign_label_text(direction_label)
    next_day_price = float(scaler.inverse_transform([[pred_next[0][0]]])[0][0])
    month_price = float(scaler.inverse_transform([[pred_month[0][0]]])[0][0])
    daily_change_rate = (month_price - current_price) / 20  # 20-day horizon from model
    for days in days_list:
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
    return predictions


def get_historical_comparison(ticker, days=60):
    """Get historical data for model vs actual comparison."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days+100)
    try:
        data = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        ).reset_index()
        if data.empty or len(data) < days + WINDOW:
            return None
        return data[['Date', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return None


@app.route('/')
def index():
    available_tickers = get_available_tickers()
    return render_template_string(HTML_TEMPLATE, tickers=available_tickers)


@app.route('/predict/<ticker>')
def predict_stock(ticker):
    try:
        model = load_best_model(ticker)
        if model is None:
            return jsonify({
                'error': f'No trained model found for {ticker}. This ticker is available for viewing but predictions require a trained model.',
                'ticker': ticker,
                'has_model': False
            })
        data = get_stock_data(ticker)
        if data is None:
            return jsonify({'error': f'Unable to fetch data for {ticker}'})
        X, scaler, current_price = prepare_prediction_data(data)
        if X is None:
            return jsonify({'error': f'Insufficient data for {ticker}'})
        predictions = make_predictions(model, X, scaler, current_price)
        predictions['current_price'] = f"${current_price:.2f}"
        predictions['ticker'] = ticker
        predictions['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predictions['has_model'] = True
        try:
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
        except Exception as e:
            print(f"Error fetching metrics for {ticker}: {e}")
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
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/historical/<ticker>')
def historical_comparison(ticker):
    try:
        model = load_best_model(ticker)
        if model is None:
            return jsonify({'error': f'No trained model found for {ticker}'})
        data = get_historical_comparison(ticker, days=60)
        if data is None:
            return jsonify({'error': f'Insufficient historical data for {ticker}'})
        close_prices = data['Close'].values.astype('float32').reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        dates = []
        actual_prices = []
        predicted_prices = []
        volumes = []
        start_idx = len(data) - 60
        for i in range(start_idx, len(data)):
            if i >= WINDOW:
                window_data = scaled_prices[i-WINDOW:i, 0]
                X = np.array([window_data]).reshape(1, 1, WINDOW)
                pred_next, _, _ = model.predict(X, verbose=0)
                pred_price = scaler.inverse_transform([[pred_next[0][0]]])[0][0]
                dates.append(data['Date'].iloc[i].strftime('%Y-%m-%d'))
                actual_prices.append(round(float(data['Close'].iloc[i]), 2))
                predicted_prices.append(round(float(pred_price), 2))
                volumes.append(int(data['Volume'].iloc[i]))
        return jsonify({
            'dates': dates,
            'actual': actual_prices,
            'predicted': predicted_prices,
            'volume': volumes,
            'ticker': ticker
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/prices/<ticker>')
def prices(ticker):
    """Return actual closing prices for a given range of days."""
    try:
        range_days = int(request.args.get('range', 30))
        data = get_stock_data(ticker, days_back=range_days + 10)  # extra buffer
        if data is None or data.empty:
            return jsonify({'error': f'Unable to fetch price data for {ticker}'})
        df = data.tail(range_days)
        return jsonify({
            'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].round(2).tolist(),
            'ticker': ticker
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/closing-prices/<ticker>')
def closing_prices_timeframe(ticker):
    """Return closing price data for multiple time frames (5, 30, 90, 365 days)."""
    try:
        time_frames = [5, 30, 90, 365]
        results = {}
        
        for days in time_frames:
            data = get_closing_price_data(ticker, days)
            if data is not None and not data.empty:
                results[f'{days}_days'] = {
                    'dates': data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                    'prices': data['Close'].round(2).tolist(),
                    'days': days
                }
            else:
                results[f'{days}_days'] = {
                    'dates': [],
                    'prices': [],
                    'days': days,
                    'error': f'No data available for {days} days'
                }
        
        return jsonify({
            'ticker': ticker,
            'timeframes': results,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/fundamentals/<ticker>')
def fundamentals(ticker):
    try:
        tk = yf.Ticker(ticker)
        try:
            earnings = tk.earnings.reset_index() if hasattr(tk, 'earnings') and tk.earnings is not None else pd.DataFrame()
        except:
            earnings = pd.DataFrame()
        try:
            quarterly_earnings = tk.quarterly_earnings.reset_index() if hasattr(tk, 'quarterly_earnings') and tk.quarterly_earnings is not None else pd.DataFrame()
        except:
            quarterly_earnings = pd.DataFrame()
        try:
            bs = tk.balance_sheet.reset_index() if hasattr(tk, 'balance_sheet') and tk.balance_sheet is not None else pd.DataFrame()
        except:
            bs = pd.DataFrame()
        try:
            fin = tk.financials.reset_index() if hasattr(tk, 'financials') and tk.financials is not None else pd.DataFrame()
        except:
            fin = pd.DataFrame()
        try:
            cf = tk.cashflow.reset_index() if hasattr(tk, 'cashflow') and tk.cashflow is not None else pd.DataFrame()
        except:
            cf = pd.DataFrame()
        return jsonify({
            'earnings': earnings.to_dict(orient='records') if not earnings.empty else [],
            'quarterlyEarnings': quarterly_earnings.to_dict(orient='records') if not quarterly_earnings.empty else [],
            'balanceSheet': bs.to_dict(orient='records') if not bs.empty else [],
            'financials': fin.to_dict(orient='records') if not fin.empty else [],
            'cashflow': cf.to_dict(orient='records') if not cf.empty else []
        })
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
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
                print("Push notification sent successfully")
                return True
            else:
                print(f"Failed to send push notification: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending push notification: {str(e)}")
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
                    X, scaler, current_price = prepare_prediction_data(data)
                    if X is None:
                        continue
                    predictions = make_predictions(model, X, scaler, current_price)
                    if predictions['direction']['confidence'] > min_confidence:
                        pred_data = {
                            'ticker': ticker,
                            'current_price': predictions['current_price'],
                            'direction': predictions['direction']['label'],
                            'confidence': predictions['direction']['confidence'],
                            '30_day_price': predictions['30_days']['price'],
                            '30_day_change': predictions['30_days']['percent_change']
                        }
                        high_confidence_predictions.append(pred_data)
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    continue
        return high_confidence_predictions
    
    def send_notifications(self, predictions):
        if not predictions:
            print("No high-confidence predictions to send")
            return
        if self.push_notifier:
            success = self.push_notifier.send_push_notification(predictions)
            if success:
                self.log_notifications(predictions)
        else:
            print("Push notifications not configured. Set PUSHOVER_TOKEN and PUSHOVER_USER.")
    
    def log_notifications(self, predictions):
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
    
    def start_scheduler(self):
        def run_scheduler():
            schedule.every().day.at("09:00").do(self.daily_notification_job)
            schedule.every().hour.do(self.hourly_notification_job)
            while True:
                schedule.run_pending()
                time.sleep(60)
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print("📱 Push notification scheduler started!")
        print("Daily notifications at 9:00 AM, hourly checks during market hours")
    
    def daily_notification_job(self):
        print("🔍 Running daily high-confidence prediction check...")
        predictions = self.get_high_confidence_predictions(min_confidence=50)
        if predictions:
            print(f"📈 Found {len(predictions)} high-confidence predictions")
            self.send_notifications(predictions)
        else:
            print("📊 No high-confidence predictions found today")
    
    def hourly_notification_job(self):
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:
            predictions = self.get_high_confidence_predictions(min_confidence=75)
            if predictions:
                print(f"⚡ Hourly check: Found {len(predictions)} very high-confidence predictions")
                self.send_notifications(predictions)

# =============================================================================
# NOTIFICATION ROUTES
# =============================================================================

def setup_notifications(app, pushover_token=None, pushover_user=None):
    notification_system = StockNotificationSystem(app, pushover_token, pushover_user)
    
    @app.route('/test-notifications')
    def test_notifications():
        predictions = notification_system.get_high_confidence_predictions(min_confidence=30)
        notification_system.send_notifications(predictions)
        return jsonify({
            'message': f'Test notification sent for {len(predictions)} predictions',
            'predictions': predictions
        })
    
    @app.route('/notification-history')  
    def notification_history():
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
    
    @app.route('/available-tickers')
    def available_tickers():
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
    
    notification_system.start_scheduler()
    return notification_system

# HTML Template with closing price plotting functionality
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&P 500 Stock Price Predictor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
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

        .plot-switcher {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }

        .plot-switcher button {
            padding: 6px 12px;
            border: none;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            background: #e1e8ed;
            color: #2c3e50;
            transition: background 0.3s ease;
        }

        .plot-switcher button.active {
            background: #667eea;
            color: #fff;
        }

        #actualControls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }

        #actualControls select {
            padding: 5px 8px;
            font-size: 0.85rem;
            border-radius: 6px;
            border: 1px solid #ccc;
        }

        #closingPriceControls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }

        #closingPriceControls button {
            padding: 5px 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            cursor: pointer;
            background: #f8f9fa;
            color: #2c3e50;
            transition: all 0.3s ease;
        }

        #closingPriceControls button.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        #tableContainer .table-header {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        #tableContainer select {
            padding: 5px 8px;
            font-size: 0.85rem;
            border-radius: 6px;
            border: 1px solid #ccc;
        }

        /* Ensure charts have visible height */
        #chart, #actualChart, #closingPriceChart {
            width: 100%;
            height: 400px;
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
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> S&P 500 Stock Price Predictor</h1>
            <p>AI-Powered Stock Price Predictions using LSTM Neural Networks</p>
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
                    <div class="plot-switcher">
                        <button id="modelBtn" class="active">Model vs Actual</button>
                        <button id="actualBtn">Actual Price</button>
                        <button id="closingPriceBtn">Closing Price Analysis</button>
                    </div>
                    
                    <div id="modelSection">
                        <h3><i class="fas fa-chart-area"></i> Model Predictions vs Actual Prices</h3>
                        <div id="chart"></div>
                        
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
                    
                    <div id="actualSection" style="display: none; margin-top: 20px;">
                        <div id="actualControls">
                            <label for="actualRangeSelect">View Actual Price for:</label>
                            <select id="actualRangeSelect">
                                <option value="5">5 Days</option>
                                <option value="30" selected>30 Days</option>
                                <option value="90">90 Days</option>
                                <option value="100">100 Days</option>
                            </select>
                        </div>
                        <div id="actualChart"></div>
                    </div>

                    <div id="closingPriceSection" style="display: none; margin-top: 20px;">
                        <h3><i class="fas fa-chart-line"></i> Closing Price Analysis</h3>
                        <div id="closingPriceControls">
                            <label>Time Frame:</label>
                            <button onclick="showClosingPriceChart(5)" data-days="5" class="active">5 Days</button>
                            <button onclick="showClosingPriceChart(30)" data-days="30">30 Days</button>
                            <button onclick="showClosingPriceChart(90)" data-days="90">90 Days</button>
                            <button onclick="showClosingPriceChart(365)" data-days="365">1 Year</button>
                        </div>
                        <div id="closingPriceChart"></div>
                    </div>
                </div>
                
                <div class="section-card" id="fundamentals">
                    <h3><i class="fas fa-building"></i> Fundamentals</h3>
                    <p>Loading...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedTicker = null;
        let tickerData = [];
        let historicalData = null;
        let closingPriceData = null;

        // Load ticker data and update display
        async function loadTickerData() {
            try {
                const response = await fetch('/available-tickers');
                const data = await response.json();
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
            } catch (error) {
                console.error('Error loading ticker data:', error);
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
            selectedTicker = ticker;
            document.querySelectorAll('.ticker-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            fetchPredictions(ticker);
            fetchHistoricalData(ticker);
            fetchFundamentals(ticker);
            fetchClosingPriceData(ticker);
        }

        async function fetchPredictions(ticker) {
            try {
                const response = await fetch(`/predict/${ticker}`);
                const data = await response.json();
                if (data.error) {
                    if (!data.has_model) {
                        showWarning(`${ticker} is an S&P 500 stock but no AI model is available for predictions. Showing available information only.`);
                        displayBasicInfo(data);
                    } else {
                        showError(data.error);
                    }
                    return;
                }
                displayMetrics(data.metrics);
                displayPredictions(data);
            } catch (error) {
                showError('Failed to fetch predictions: ' + error.message);
            }
        }

        async function fetchHistoricalData(ticker) {
            try {
                const response = await fetch(`/historical/${ticker}`);
                const data = await response.json();
                if (data.error) {
                    console.log('Historical data not available:', data.error);
                    document.getElementById('modelSection').style.display = 'none';
                    return;
                }
                historicalData = data;
                displayModelSection(data);
                document.getElementById('modelSection').style.display = 'block';
                document.getElementById('actualSection').style.display = 'none';
                document.getElementById('closingPriceSection').style.display = 'none';
                document.getElementById('modelBtn').classList.add('active');
                document.getElementById('actualBtn').classList.remove('active');
                document.getElementById('closingPriceBtn').classList.remove('active');
            } catch (error) {
                console.error('Failed to fetch historical data:', error);
            }
        }

        async function fetchClosingPriceData(ticker) {
            try {
                const response = await fetch(`/closing-prices/${ticker}`);
                const data = await response.json();
                if (data.error) {
                    console.log('Closing price data not available:', data.error);
                    return;
                }
                closingPriceData = data;
            } catch (error) {
                console.error('Failed to fetch closing price data:', error);
            }
        }

        async function fetchActualPrices(ticker, days) {
            try {
                const response = await fetch(`/prices/${ticker}?range=${days}`);
                const data = await response.json();
                if (data.error) {
                    console.log('Actual prices not available:', data.error);
                    return null;
                }
                return data;
            } catch (error) {
                console.error('Failed to fetch actual prices:', error);
                return null;
            }
        }

        function displayMetrics(metrics) {
            const tbody = document.querySelector('#metricsTable tbody');
            tbody.innerHTML = '';
            for (const [key, val] of Object.entries(metrics)) {
                const row = `<tr>
                    <td>${key}</td>
                    <td>${val !== null ? val : 'N/A'}</td>
                </tr>`;
                tbody.innerHTML += row;
            }
        }

        function displayPredictions(data) {
            const stockInfo = document.getElementById('stockInfo');
            stockInfo.innerHTML = `
                <div class="stock-header">
                    <div class="stock-symbol">${data.ticker}</div>
                    <div class="current-price">${data.current_price}</div>
                </div>
                <div class="predictions-grid">
                    <div class="prediction-card">
                        <h3>30 Days</h3>
                        <div class="prediction-price">${data['30_days'].price}</div>
                        <div class="prediction-change ${data['30_days'].change >= 0 ? 'positive' : 'negative'}">
                            ${data['30_days'].change >= 0 ? '+' : ''}${data['30_days'].change} (${data['30_days'].percent_change}%)
                        </div>
                    </div>
                    <div class="prediction-card">
                        <h3>60 Days</h3>
                        <div class="prediction-price">${data['60_days'].price}</div>
                        <div class="prediction-change ${data['60_days'].change >= 0 ? 'positive' : 'negative'}">
                            ${data['60_days'].change >= 0 ? '+' : ''}${data['60_days'].change} (${data['60_days'].percent_change}%)
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
                        <h3>120 Days</h3>
                        <div class="prediction-price">${data['120_days'].price}</div>
                        <div class="prediction-change ${data['120_days'].change >= 0 ? 'positive' : 'negative'}">
                            ${data['120_days'].change >= 0 ? '+' : ''}${data['120_days'].change} (${data['120_days'].percent_change}%)
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
            document.getElementById('loading').style.display = 'none';
            document.getElementById('results').style.display = 'grid';
        }

        function displayBasicInfo(data) {
            const stockInfo = document.getElementById('stockInfo');
            stockInfo.innerHTML = `
                <div class="stock-header">
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
            document.getElementById('loading').style.display = 'none';
            document.getElementById('results').style.display = 'grid';
        }

        function displayModelSection(data) {
            const sliceCount = 5;
            const dates = data.dates.slice(-sliceCount);
            const actual = data.actual.slice(-sliceCount);
            const predicted = data.predicted.slice(-sliceCount);

            const trace1 = {
                x: dates,
                y: actual,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual Price',
                line: { width: 2, color: '#27ae60' },
                marker: { size: 5, color: '#27ae60' }
            };

            const trace2 = {
                x: dates,
                y: predicted,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: { width: 2, dash: 'dash', color: '#667eea' },
                marker: { size: 5, color: '#667eea' }
            };

            const layout = {
                title: {
                    text: `${data.ticker} - Model vs Actual (Last 5 Days)`,
                    font: { size: 16, color: '#2c3e50' }
                },
                xaxis: { 
                    title: 'Date',
                    gridcolor: '#ecf0f1'
                },
                yaxis: { 
                    title: 'Price ($)',
                    gridcolor: '#ecf0f1'
                },
                hovermode: 'x unified',
                showlegend: true,
                margin: { l: 50, r: 30, t: 50, b: 60 },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };

            Plotly.newPlot('chart', [trace1, trace2], layout, { responsive: true });
            populateComparisonTable(data, 60);
        }

        function populateComparisonTable(data, days) {
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
                tableBody.innerHTML += `
                    <tr>
                        <td>${date}</td>
                        <td>${vol.toLocaleString()}</td>
                        <td class="price-cell">${actual.toFixed(2)}</td>
                        <td class="price-cell">${pred.toFixed(2)}</td>
                        <td class="${diffClass}">${sign}${diff.toFixed(2)}</td>
                        <td class="${diffClass}">${sign}${pctErr.toFixed(2)}%</td>
                        <td><span class="${accClass}">${accText}</span></td>
                    </tr>`;
            }
        }

        function displayActualSection(data) {
            const trace = {
                x: data.dates,
                y: data.prices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Close Price',
                line: { width: 2, color: '#e74c3c' },
                marker: { size: 4, color: '#e74c3c' }
            };

            const layout = {
                title: {
                    text: `${data.ticker} - Actual Close Price (${data.dates.length} Days)`,
                    font: { size: 16, color: '#2c3e50' }
                },
                xaxis: { 
                    title: 'Date',
                    gridcolor: '#ecf0f1'
                },
                yaxis: { 
                    title: 'Price ($)',
                    gridcolor: '#ecf0f1'
                },
                hovermode: 'x unified',
                showlegend: false,
                margin: { l: 50, r: 30, t: 50, b: 60 },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };

            Plotly.newPlot('actualChart', [trace], layout, { responsive: true });
        }

        function showClosingPriceChart(days) {
            if (!closingPriceData || !closingPriceData.timeframes) {
                console.log('No closing price data available');
                return;
            }

            const timeframeKey = `${days}_days`;
            const timeframeData = closingPriceData.timeframes[timeframeKey];

            if (!timeframeData || !timeframeData.dates || timeframeData.dates.length === 0) {
                console.log(`No data available for ${days} days`);
                return;
            }

            // Update active button
            document.querySelectorAll('#closingPriceControls button').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`#closingPriceControls button[data-days="${days}"]`).classList.add('active');

            const trace = {
                x: timeframeData.dates,
                y: timeframeData.prices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Closing Price',
                line: { width: 3, color: '#3498db' },
                marker: { size: 5, color: '#2980b9' },
                fill: 'tonexty',
                fillcolor: 'rgba(52, 152, 219, 0.1)'
            };

            const layout = {
                title: {
                    text: `${closingPriceData.ticker} - Closing Price Analysis (${days} Days)`,
                    font: { size: 16, color: '#2c3e50' }
                },
                xaxis: { 
                    title: 'Date',
                    gridcolor: '#ecf0f1',
                    tickangle: -45
                },
                yaxis: { 
                    title: 'Closing Price ($)',
                    gridcolor: '#ecf0f1'
                },
                hovermode: 'x unified',
                showlegend: false,
                margin: { l: 60, r: 30, t: 60, b: 100 },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };

            Plotly.newPlot('closingPriceChart', [trace], layout, { responsive: true });
        }

        function displayFundamentals(data) {
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
        }

        async function fetchFundamentals(ticker) {
            try {
                const response = await fetch(`/fundamentals/${ticker}`);
                const data = await response.json();
                displayFundamentals(data);
            } catch (error) {
                console.error('Failed to fetch fundamentals:', error);
                document.getElementById('fundamentals').innerHTML = '<h3><i class="fas fa-building"></i> Fundamentals</h3><p class="error">Failed to load fundamentals data</p>';
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

        // Button event listeners
        document.getElementById('modelBtn').addEventListener('click', () => {
            if (!selectedTicker || !historicalData) return;
            document.getElementById('modelSection').style.display = 'block';
            document.getElementById('actualSection').style.display = 'none';
            document.getElementById('closingPriceSection').style.display = 'none';
            document.getElementById('modelBtn').classList.add('active');
            document.getElementById('actualBtn').classList.remove('active');
            document.getElementById('closingPriceBtn').classList.remove('active');
            displayModelSection(historicalData);
        });

        document.getElementById('actualBtn').addEventListener('click', async () => {
            if (!selectedTicker) return;
            document.getElementById('modelSection').style.display = 'none';
            document.getElementById('actualSection').style.display = 'block';
            document.getElementById('closingPriceSection').style.display = 'none';
            document.getElementById('modelBtn').classList.remove('active');
            document.getElementById('actualBtn').classList.add('active');
            document.getElementById('closingPriceBtn').classList.remove('active');
            const days = parseInt(document.getElementById('actualRangeSelect').value);
            const data = await fetchActualPrices(selectedTicker, days);
            if (data) displayActualSection(data);
        });

        document.getElementById('closingPriceBtn').addEventListener('click', () => {
            if (!selectedTicker || !closingPriceData) return;
            document.getElementById('modelSection').style.display = 'none';
            document.getElementById('actualSection').style.display = 'none';
            document.getElementById('closingPriceSection').style.display = 'block';
            document.getElementById('modelBtn').classList.remove('active');
            document.getElementById('actualBtn').classList.remove('active');
            document.getElementById('closingPriceBtn').classList.add('active');
            // Show default 5-day chart
            showClosingPriceChart(5);
        });

        document.getElementById('tableRangeSelect').addEventListener('change', () => {
            const days = parseInt(document.getElementById('tableRangeSelect').value);
            if (historicalData) populateComparisonTable(historicalData, days);
        });

        document.getElementById('actualRangeSelect').addEventListener('change', async () => {
            if (!selectedTicker) return;
            const days = parseInt(document.getElementById('actualRangeSelect').value);
            const data = await fetchActualPrices(selectedTicker, days);
            if (data) displayActualSection(data);
        });

        document.addEventListener('DOMContentLoaded', function() {
            loadTickerData();
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    import os
    os.makedirs('saved_models', exist_ok=True)
    print("🔍 Loading S&P 500 tickers using PyTickerSymbols...")
    sp500_tickers = get_sp500_tickers()
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
        print(f"🚀 Starting S&P 500 Stock Prediction App on port {available_port}")
        print(f"🌐 Access the application at: http://localhost:{available_port}")
        print(f"📱 Push notifications: {'Enabled' if PUSHOVER_TOKEN and PUSHOVER_USER else 'Disabled (configure Pushover credentials)'}")
        print(f"🧪 Test notifications at: http://localhost:{available_port}/test-notifications")
        print(f"📊 Notification history at: http://localhost:{available_port}/notification-history")
        print(f"📋 Available tickers API: http://localhost:{available_port}/available-tickers")
        print(f"📈 Closing prices API: http://localhost:{available_port}/closing-prices/<ticker>")
        trained_models = get_available_models()
        print(f"\n📈 Ticker Summary:")
        print(f"   • Total S&P 500 tickers: {len(sp500_tickers)}")
        print(f"   • Tickers with trained models: {len(trained_models)}")
        print(f"   • Total available for display: {len(get_available_tickers())}")
        if trained_models:
            print(f"\n🤖 Tickers with AI models: {', '.join(sorted(trained_models)[:10])}{'...' if len(trained_models) > 10 else ''}")
        print(f"\n🆕 New Features:")
        print(f"   • Closing price analysis for 5, 30, 90, and 365 days")
        print(f"   • Interactive time frame switching")
        print(f"   • Enhanced visualization with fill areas")
        app.run(debug=True, host='0.0.0.0', port=available_port)
    else:
        print("❌ No available ports found. Please free up some ports and try again.")
