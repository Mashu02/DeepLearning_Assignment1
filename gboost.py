import numpy as np
import scipy.io
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, r2_score)
from pathlib import Path
import matplotlib.pyplot as plt

TRAIN_FILE = Path('Xtrain.mat')
TEST_FILE = Path('Xtest.mat')
LOOKBACK = 64 # how deep the lookback is, i.e. how many past points to use for predicting the next one
VALIDATION_RATIO = 0.2 # to evaluate on the last 20% of the data, not random samples


def load_intensity_series(path: Path) -> np.ndarray:
    # load values from a .mat file and return the first non-meta variable as a 1D float array
    mat = scipy.io.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    if len(keys) == 0:
        raise ValueError(f"No data variables found in {path}")

    series = np.asarray(mat[keys[0]]).ravel()
    if series.ndim != 1:
        series = series.ravel()

    return series.astype(float)


def create_lag_features(series: np.ndarray, lookback: int):
    # build simple lagged features: last lookback points predict the next point
    n_samples = len(series) - lookback
    X = np.zeros((n_samples, lookback), dtype=float)
    y = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        X[i] = series[i:i + lookback]
        y[i] = series[i + lookback]

    return X, y


def time_series_split(X: np.ndarray, y: np.ndarray, validation_ratio: float):
    # validation uses the last chunk of the time series, not random samples
    n_val = max(1, int(len(X) * validation_ratio))
    X_train = X[:-n_val]
    X_val = X[-n_val:]
    y_train = y[:-n_val]
    y_val = y[-n_val:]
    return X_train, X_val, y_train, y_val


def forecast_future(model, series: np.ndarray, lookback: int, steps: int):
    # Start forecasting from the end of the series
    current_input = series[-lookback:].copy()
    forecasts = []

    for _ in range(steps):
        # Predict the next value
        next_pred = model.predict(current_input.reshape(1, -1))[0]
        forecasts.append(next_pred)
        # Update the input by shifting and adding the prediction
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_pred

    return np.array(forecasts)


def plot_forecast(series: np.ndarray, forecasts: np.ndarray, lookback: int):
    plt.figure(figsize=(14, 6))
    plt.plot(series, label='Historical Data')
    forecast_start = len(series)
    forecast_end = forecast_start + len(forecasts)
    plt.plot(range(forecast_start, forecast_end), forecasts, label='Forecast', linestyle='--')
    plt.axvline(x=len(series) - lookback, linestyle=':', label='Training End')
    plt.title('Time Series Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_test_vs_pred(y_true, y_pred):
    plt.figure(figsize=(14, 6))

    plt.plot(y_true, label="Real values")
    plt.plot(y_pred, label="Predictions")


    plt.title("Gradient Boosting Predictions vs Real Values")
    plt.xlabel("Time Step")
    plt.ylabel("Signal")

    plt.legend()
    plt.savefig('plotvspred.png', dpi=300, bbox_inches='tight')

    plt.show()


def main():
    train_series = load_intensity_series(TRAIN_FILE)
    test_series = load_intensity_series(TEST_FILE)

    print(f"Loaded train series, length = {len(train_series)}")
    print(f"Loaded test series, length = {len(test_series)}")
    print(f"Using the last {LOOKBACK} points to predict the next one")

    # Create lagged training data
    X, y = create_lag_features(train_series, LOOKBACK)

    # Time-series split
    X_train, X_val, y_train, y_val = time_series_split(
        X, y, VALIDATION_RATIO
    )

    # Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=3,
        verbose=1,
    )

    print("Training the model now...")
    model.fit(X_train, y_train)

    # One-step predictions for train/validation
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # =========================
    # RECURSIVE TEST FORECAST
    # =========================

    forecast_steps = min(200, len(test_series))

    # recursively forecast 200 future values
    y_pred_test = forecast_future(
        model=model,
        series=train_series,
        lookback=LOOKBACK,
        steps=forecast_steps
    )

    # real test values
    test_y = test_series[:forecast_steps]

    # =========================
    # METRICS
    # =========================

    train_mae = mean_absolute_error(y_train, y_pred_train)

    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_pred_val)

    test_mae = mean_absolute_error(test_y, y_pred_test)
    test_mse = mean_squared_error(test_y, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_y, y_pred_test)

    print(f"\nTraining MAE: {train_mae:.4f}")

    print(f"\nValidation MAE: {val_mae:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")

    print(f"\nRecursive Test Forecast (200 steps)")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # =========================
    # PLOTS
    # =========================

    plot_forecast(train_series, y_pred_test, LOOKBACK)
    plot_test_vs_pred(test_y, y_pred_test)

    # =========================
    # DIRECTION ACCURACY
    # =========================

    # previous value used for direction comparison
    prev_values = np.concatenate([
        [train_series[-1]],
        y_pred_test[:-1]
    ])

    y_test_dir = (test_y > prev_values).astype(int)
    y_pred_test_dir = (y_pred_test > prev_values).astype(int)

    print("\nDirection prediction performance on recursive forecast:")
    print(f"Accuracy: {accuracy_score(y_test_dir, y_pred_test_dir):.4f}")
    print(f"Precision: {precision_score(y_test_dir, y_pred_test_dir, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test_dir, y_pred_test_dir, zero_division=0):.4f}")
    print(f"F1 score: {f1_score(y_test_dir, y_pred_test_dir, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_dir, y_pred_test_dir))

    # =========================
    # SHOW SAMPLE PREDICTIONS
    # =========================

    print("\nFirst 10 recursive predictions:")

    for i in range(min(10, forecast_steps)):
        print(
            f"Step {i+1}: "
            f"real={test_y[i]:.4f}, "
            f"predicted={y_pred_test[i]:.4f}"
        ) 
        print(f"Direction: real={y_test_dir[i]}, predicted={y_pred_test_dir[i]}")


if __name__ == '__main__':
    main()