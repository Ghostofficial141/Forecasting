"""
lstm_model.py
=============
LSTM deep learning model for multi-step time-series forecasting.
Implementation details:
  - Sequence-to-one architecture (predict next step)
  - Multi-layer LSTM with dropout
  - MinMaxScaling per state
  - Keras early stopping + model checkpointing
  - Recursive multi-step inference
  - TensorFlow / Keras backend
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.exception import ModelTrainingError, PredictionError
from src.utils.helpers import load_yaml, save_pickle, load_pickle, ensure_dir
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, MODELS_DIR

logger = get_logger(__name__)


class LSTMModel:
    """
    Multi-layer LSTM for univariate time-series forecasting.
    Uses a sliding window (sequence_length) to create input sequences.
    """

    MODEL_NAME = "LSTM"

    def __init__(self, config_path=None):
        cfg = load_yaml(config_path or CONFIG_PATH)
        self.model_cfg = cfg["models"]["lstm"]
        self.horizon = cfg["models"]["forecast_horizon"]
        self.seq_len: int = self.model_cfg.get("sequence_length", 12)
        self.units: List[int] = self.model_cfg.get("units", [64, 32])
        self.dropout: float = self.model_cfg.get("dropout", 0.2)
        self.dense_units: int = self.model_cfg.get("dense_units", 16)
        self.epochs: int = self.model_cfg.get("epochs", 100)
        self.batch_size: int = self.model_cfg.get("batch_size", 32)
        self.lr: float = self.model_cfg.get("learning_rate", 0.001)
        self.patience: int = self.model_cfg.get("patience", 15)
        self._models: Dict[str, object] = {}
        self._scalers: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        series: np.ndarray,
        state: str = "all",
    ) -> None:
        """
        Fit LSTM on a 1-D numpy array (target values only).

        Parameters
        ----------
        series : np.ndarray
            1-D array of target values in chronological order.
        state : str
            State label for model serialisation.
        """
        logger.info(
            f"[LSTM] Fitting for state='{state}' | "
            f"len={len(series)} | seq_len={self.seq_len}"
        )
        try:
            # Suppress TF INFO/WARNING logs
            import os
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import (
                EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            )
            from sklearn.preprocessing import MinMaxScaler

            tf.random.set_seed(self.model_cfg.get("random_state", 42))

            # Scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
            self._scalers[state] = scaler

            # Build sequences
            X, y = self._create_sequences(scaled, self.seq_len)
            if len(X) == 0:
                raise ModelTrainingError(
                    f"[LSTM] Not enough data to build sequences for state='{state}'. "
                    f"Need at least seq_len+1={self.seq_len + 1} data points."
                )

            # Time-series split: last 20% for validation
            split = int(len(X) * 0.8)
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]

            X_tr = X_tr.reshape(-1, self.seq_len, 1)
            X_val = X_val.reshape(-1, self.seq_len, 1)

            # Build model
            model = Sequential(name=f"LSTM_{state}")
            model.add(Input(shape=(self.seq_len, 1)))
            for i, u in enumerate(self.units):
                return_seq = (i < len(self.units) - 1)  # last layer doesn't return seq
                model.add(LSTM(u, return_sequences=return_seq, name=f"lstm_{i}"))
                model.add(Dropout(self.dropout, name=f"dropout_{i}"))
            if self.dense_units:
                model.add(Dense(self.dense_units, activation="relu", name="dense_hidden"))
            model.add(Dense(1, name="output"))

            model.compile(
                optimizer=Adam(learning_rate=self.lr),
                loss="huber",             # robust to outliers
                metrics=["mae"],
            )

            # Callbacks
            ckpt_path = MODELS_DIR / self.MODEL_NAME / f"{state}_best.keras"
            ensure_dir(ckpt_path.parent)
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ModelCheckpoint(
                    str(ckpt_path),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ]

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0,
                shuffle=False,   # do NOT shuffle time series data
            )

            best_epoch = np.argmin(history.history["val_loss"])
            best_val_loss = history.history["val_loss"][best_epoch]
            logger.info(
                f"[LSTM][{state}] Training complete. "
                f"Best epoch={best_epoch + 1}, val_loss={best_val_loss:.6f}"
            )

            self._models[state] = model
            self._save(state)

        except Exception as e:
            raise ModelTrainingError(
                f"[LSTM] Training failed for state='{state}'", error=e
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        series: np.ndarray,
        state: str = "all",
        n_periods: Optional[int] = None,
    ) -> np.ndarray:
        """
        Recursive multi-step LSTM forecasting.

        Parameters
        ----------
        series : np.ndarray
            Full historical target series (unscaled).
        state : str
            State label.
        n_periods : int
            Number of future periods to forecast.

        Returns
        -------
        np.ndarray
            1-D array of (unscaled) future predictions.
        """
        n = n_periods or self.horizon
        model = self._get_model(state)
        scaler = self._get_scaler(state)

        try:
            if len(series) < self.seq_len:
                raise PredictionError(
                    f"Not enough history to predict using LSTM for state='{state}'. "
                    f"Expected >= {self.seq_len} points, got {len(series)}."
                )

            # Scale history using the fitted scaler
            scaled = scaler.transform(series.reshape(-1, 1)).flatten()

            # Seed sequence: last seq_len values
            window = list(scaled[-self.seq_len:])
            preds_scaled = []

            for _ in range(n):
                x = np.array(window[-self.seq_len:]).reshape(1, self.seq_len, 1)
                pred = float(model.predict(x, verbose=0)[0, 0])
                preds_scaled.append(pred)
                window.append(pred)

            preds = scaler.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).flatten()
            return np.maximum(preds, 0).astype(float)

        except Exception as e:
            raise PredictionError(
                f"[LSTM] Prediction failed for state='{state}'", error=e
            )

    def predict_in_sample(
        self,
        series: np.ndarray,
        state: str = "all",
    ) -> np.ndarray:
        """Return predictions for training data evaluation."""
        model = self._get_model(state)
        scaler = self._get_scaler(state)

        try:
            scaled = scaler.transform(series.reshape(-1, 1)).flatten()
            X, _ = self._create_sequences(scaled, self.seq_len)
            X = X.reshape(-1, self.seq_len, 1)
            preds_scaled = model.predict(X, verbose=0).flatten()
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            return np.maximum(preds, 0).astype(float)
        except Exception as e:
            raise PredictionError(
                f"[LSTM] In-sample prediction failed for state='{state}'", error=e
            )

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_sequences(
        data: np.ndarray,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (X, y) sliding-window sequences from 1-D array.

        X[i] = data[i : i + seq_len]
        y[i] = data[i + seq_len]
        """
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i: i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save(self, state: str) -> None:
        ensure_dir(MODELS_DIR / self.MODEL_NAME)
        # Save scaler
        save_pickle(self._scalers[state], MODELS_DIR / self.MODEL_NAME / f"{state}_scaler.pkl")
        # Save Keras model
        self._models[state].save(
            str(MODELS_DIR / self.MODEL_NAME / f"{state}.keras")
        )
        logger.debug(f"[LSTM] Model + scaler saved for state='{state}'")

    def load(self, state: str) -> None:
        import tensorflow as tf
        model_path = MODELS_DIR / self.MODEL_NAME / f"{state}.keras"
        scaler_path = MODELS_DIR / self.MODEL_NAME / f"{state}_scaler.pkl"
        self._models[state] = tf.keras.models.load_model(str(model_path))
        self._scalers[state] = load_pickle(scaler_path)

    def _get_model(self, state: str):
        if state not in self._models:
            self.load(state)
        return self._models[state]

    def _get_scaler(self, state: str):
        if state not in self._scalers:
            self.load(state)
        return self._scalers[state]
