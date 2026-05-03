import numpy as np
import scipy.io
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, r2_score)
from pathlib import Path

DATA_FILE = Path('DataFiles/Xtrain.mat')
LOOKBACK = 10 # how deep the lookback is, i.e. how many past points to use for predicting the next one
VALIDATION_RATIO = 0.2 # to evaluate on the last 20% of the data, not random samples


def load_intensity_series(path: Path) -> np.ndarray:
    # load the values, i assume it was intensity values, but it could be anything really, as long as it's a 1D series of floats
    mat = scipy.io.loadmat(path)
    series = np.asarray(mat['Xtrain']).ravel()
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


def main():
    series = load_intensity_series(DATA_FILE)
    print(f"Loaded intensity series, length = {len(series)}")
    print(f"Using the last {LOOKBACK} points to guess the next one")

    X, y = create_lag_features(series, LOOKBACK)
    X_train, X_val, y_train, y_val = time_series_split(X, y, VALIDATION_RATIO)

    # this is a basic gradient boosting regressor, should work fine for a start
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=3,
        verbose=1,
    )

    print("Training the model now...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)

    print(f"\nTrainingsfout (MAE): {train_mae:.4f}")
    print(f"Validatiefout (MAE): {val_mae:.4f}")
    print(f"Validatie RMSE: {val_rmse:.4f}")
    print(f"Validatie R²: {val_r2:.4f}")

    y_train_dir = (y_train > X_train[:, -1]).astype(int)
    y_val_dir = (y_val > X_val[:, -1]).astype(int)
    y_pred_train_dir = (y_pred_train > X_train[:, -1]).astype(int)
    y_pred_val_dir = (y_pred_val > X_val[:, -1]).astype(int)

    print("\nHoe vaak is de richting voorspeld?")
    print(f"Accuracy: {accuracy_score(y_val_dir, y_pred_val_dir):.4f}")
    print(f"Precision: {precision_score(y_val_dir, y_pred_val_dir, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val_dir, y_pred_val_dir, zero_division=0):.4f}")
    print(f"F1 score: {f1_score(y_val_dir, y_pred_val_dir, zero_division=0):.4f}")
    print("Confusiematrix:")
    print(confusion_matrix(y_val_dir, y_pred_val_dir))

    print("\nVoorbeeld voorspellingen:")
    for i in range(min(10, len(y_val))):
        print(f"punt {i + 1}: echt={y_val[i]:.2f}, voorspeld={y_pred_val[i]:.2f}, richting echt={y_val_dir[i]}, richting voorspeld={y_pred_val_dir[i]}")


if __name__ == '__main__':
    main()
