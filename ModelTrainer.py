# Importing modules
# ----------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import glob
from pytickersymbols import PyTickerSymbols
from datetime import datetime   # ✅ NEW

# ----------------------------------------------
# Constants
# ----------------------------------------------
MIN_START = pd.Timestamp("1998-01-01")   # earliest allowed start date
END_DATE  = "2025-01-01"                 # fixed end date

# ----------------------------------------------
# Fetch S&P 500 tickers
# ----------------------------------------------
def get_sp500_tickers() -> list[str]:
    """
    Return a list of S&P 500 tickers using the pytickersymbols package.
    """
    symbols = PyTickerSymbols()
    sp500_stocks = symbols.get_stocks_by_index("S&P 500")
    return [stock['symbol'] for stock in sp500_stocks]

TICKERS = get_sp500_tickers()

# ----------------------------------------------
# Helper for direction label
# ----------------------------------------------
def assign_label(pct: float, up: float = 0.05, down: float = -0.05) -> int:
    """
    Map percentage change to a 4-class label.
        3 – strongly up   ( ≥ +up )
        2 – slightly up   ( 0 … +up )
        1 – slightly down ( down … 0 )
        0 – strongly down ( ≤ down )
    """
    if pct >= up:
        return 3
    elif pct > 0:
        return 2
    elif pct <= down:
        return 0
    else:
        return 1

for ticker in TICKERS:
    print(f"Processing {ticker}...")

    # Skip if a trained model already exists for this ticker
    existing_models = glob.glob(f"saved_models/{ticker}_model_epoch_*.keras")
    if existing_models:
        print(f"Model files for {ticker} already exist. Skipping.")
        continue

    # ----------------------------------------------
    # Pull full history, then trim to ≥ 1998-01-01
    # ----------------------------------------------
    tesla = yf.download(
        tickers=ticker,
        period="max",          # grab earliest-to-latest data
        end=END_DATE,          # fixed end date
        progress=False
    ).reset_index()

    # Skip tickers with no data (e.g., delisted)
    if tesla.empty:
        print(f"No data for {ticker}, skipping.")
        continue

    # Determine dynamic start date (≥ 1998-01-01)
    first_available = tesla['Date'].iloc[0]
    start_date = max(first_available, MIN_START)

    # Trim data to the dynamic start date
    tesla = tesla[tesla['Date'] >= start_date].reset_index(drop=True)

    # ----------------------------------------------
    # Isolating the date and close price
    # ----------------------------------------------
    tesla = tesla[['Date', 'Close']]

    # ----------------------------------------------
    # Selecting the required part of the dataset
    # (keep full span after dynamic start)
    # ----------------------------------------------
    new_tesla = tesla.copy()

    # Preserve raw closing prices (unscaled) for classification labels
    raw_prices = new_tesla['Close'].values.astype('float32').reshape(-1)

    # ----------------------------------------------
    # Feature preprocessing
    # ----------------------------------------------
    new_tesla = new_tesla.drop('Date', axis=1)
    new_tesla = new_tesla.reset_index(drop=True)
    T = new_tesla.values.astype('float32').reshape(-1, 1)

    # ----------------------------------------------
    # Min-max scaling to get values in the range [0, 1]
    # ----------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    T_scaled = scaler.fit_transform(T)

    # ----------------------------------------------
    # 80-20 split (time-ordered)
    # ----------------------------------------------
    train_size = int(len(T_scaled) * 0.80)
    test_size  = len(T_scaled) - train_size
    train_scaled, test_scaled = (
        T_scaled[0:train_size, :],
        T_scaled[train_size:len(T_scaled), :]
    )
    train_raw, test_raw = (
        raw_prices[0:train_size],
        raw_prices[train_size:]
    )

    # ----------------------------------------------
    # Method to create features + 3 targets
    # ----------------------------------------------
    HORIZON = 20             # ≈ one month of trading days
    WINDOW  = 20             # same as original window_size

    def create_features(data_scaled, data_raw, window_size, horizon):
        """
        Returns:
            X        – window of past window_size scaled prices
            y_next   – next-day scaled price        (regression)
            y_month  – +horizon-day scaled price    (regression)
            y_dir    – 4-class movement label       (classification)
        """
        X, y_next, y_month, y_dir = [], [], [], []
        max_i = len(data_scaled) - window_size - horizon
        for i in range(max_i):
            window_scaled = data_scaled[i : i + window_size, 0]
            X.append(window_scaled)

            # regression targets (scaled)
            next_p  = data_scaled[i + window_size, 0]
            month_p = data_scaled[i + window_size + horizon, 0]
            y_next.append(next_p)
            y_month.append(month_p)

            # classification target (raw % change)
            last_raw   = data_raw[i + window_size - 1]
            month_raw  = data_raw[i + window_size + horizon]
            pct_change = (month_raw - last_raw) / last_raw
            y_dir.append(assign_label(pct_change))
        return (
            np.array(X),
            np.array(y_next),
            np.array(y_month),
            np.array(y_dir)
        )

    X_train, y_next_train, y_month_train, y_dir_train_int = create_features(
        train_scaled, train_raw, WINDOW, HORIZON
    )
    X_test, y_next_test, y_month_test, y_dir_test_int = create_features(
        test_scaled, test_raw, WINDOW, HORIZON
    )

    # One-hot encode direction labels
    y_dir_train = to_categorical(y_dir_train_int, num_classes=4)
    y_dir_test  = to_categorical(y_dir_test_int,  num_classes=4)

    # Reshape to [samples, time-steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, WINDOW))
    X_test  = np.reshape(X_test,  (X_test.shape[0],  1, WINDOW))

    T_shape     = T_scaled.shape
    train_shape = train_scaled.shape
    test_shape  = test_scaled.shape

    def isLeak(T_shape, train_shape, test_shape):
        return not (T_shape[0] == (train_shape[0] + test_shape[0]))

    print(isLeak(T_shape, train_shape, test_shape))

    # ----------------------------------------------
    # Building multi-task model
    # ----------------------------------------------
    tf.random.set_seed(11)
    np.random.seed(11)

    inp = Input(shape=(X_train.shape[1], WINDOW), name='price_history')
    x   = LSTM(50, activation='relu')(inp)
    x   = Dropout(0.2)(x)
    x   = Dense(25, activation='relu')(x)

    out_next  = Dense(1,  name='next_day')(x)
    out_month = Dense(1,  name='one_month')(x)
    out_dir   = Dense(4,  activation='softmax', name='direction')(x)

    model = Model(inputs=inp, outputs=[out_next, out_month, out_dir])

    model.compile(
        optimizer='adam',
        loss={
            'next_day' : 'mse',
            'one_month': 'mse',
            'direction': 'categorical_crossentropy'
        },
        loss_weights={
            'next_day' : 1.0,
            'one_month': 1.0,
            'direction': 1.0
        },
        metrics={'direction': 'accuracy'}
    )

    # ----------------------------------------------
    # Save models
    # ----------------------------------------------
    filepath = f'saved_models/{ticker}_model_epoch_{{epoch:02d}}.keras'
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    history = model.fit(
        X_train,
        {
            'next_day' : y_next_train,
            'one_month': y_month_train,
            'direction': y_dir_train
        },
        epochs=100,
        batch_size=20,
        validation_data=(
            X_test,
            {
                'next_day' : y_next_test,
                'one_month': y_month_test,
                'direction': y_dir_test
            }
        ),
        callbacks=[checkpoint],
        verbose=1,
        shuffle=False
    )

    # ----------------------------------------------
    # Loading the best model and predicting
    # ----------------------------------------------
    model_files = glob.glob(f'saved_models/{ticker}_model_epoch_*.keras')

    if not model_files:
        raise FileNotFoundError(f"No model files found for {ticker} in 'saved_models/'.")

    best_model_path = sorted(model_files)[-1]
    print(f"Loading best model for {ticker}: {best_model_path}")
    best_model = load_model(best_model_path)

    # Predict
    train_preds_next, train_preds_month, train_preds_dir = best_model.predict(X_train)
    test_preds_next,  test_preds_month,  test_preds_dir  = best_model.predict(X_test)

    # Inverse-transform regression outputs
    Y_hat_train_next = scaler.inverse_transform(train_preds_next).ravel()
    Y_hat_test_next  = scaler.inverse_transform(test_preds_next).ravel()

    # Inverse-transform actual values
    y_next_train_inv = scaler.inverse_transform(y_next_train.reshape(-1, 1)).ravel()
    y_next_test_inv  = scaler.inverse_transform(y_next_test.reshape(-1, 1)).ravel()

    # ----------------------------------------------
    # Model performance evaluation
    # ----------------------------------------------
    train_RMSE = np.sqrt(mean_squared_error(y_next_train_inv, Y_hat_train_next))
    test_RMSE  = np.sqrt(mean_squared_error(y_next_test_inv,  Y_hat_test_next))

    dir_pred_test = np.argmax(test_preds_dir, axis=1)
    dir_accuracy  = (dir_pred_test == y_dir_test_int).mean()

    print(f'{ticker} Train RMSE (next-day):', train_RMSE)
    print(f'{ticker} Test  RMSE (next-day):',  test_RMSE)
    print(f'{ticker} Direction accuracy   :', dir_accuracy)
