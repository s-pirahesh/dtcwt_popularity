"""
Advanced Baseline Methods for Prediction
ARMA and LSTM models for comparison
"""
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")


class ARMAPredictor:
    """
    ARMA/ARIMA-based popularity prediction
    
    Classic time series forecasting method
    """
    
    def __init__(self, p: int = 3, q: int = 2, d: int = 0):
        """
        Initialize ARMA predictor
        
        Args:
            p: AR order
            q: MA order
            d: Differencing order (0 for ARMA, >0 for ARIMA)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARMA")
        
        self.p = p
        self.q = q
        self.d = d
        self.model = None
    
    def fit(self, time_series: np.ndarray):
        """
        Fit ARMA model
        
        Args:
            time_series: Training time series
        """
        if len(time_series) < self.p + self.q + 5:
            raise ValueError("Time series too short for ARMA")
        
        try:
            # Fit ARIMA model
            self.model = ARIMA(
                time_series,
                order=(self.p, self.d, self.q)
            )
            self.model_fit = self.model.fit()
        
        except Exception as e:
            print(f"ARMA fitting failed: {e}")
            self.model_fit = None
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Predict future values
        
        Args:
            steps: Number of steps ahead
            
        Returns:
            Predicted values
        """
        if self.model_fit is None:
            return np.zeros(steps)
        
        try:
            forecast = self.model_fit.forecast(steps=steps)
            return np.array(forecast)
        
        except Exception as e:
            print(f"ARMA prediction failed: {e}")
            return np.zeros(steps)
    
    def predict_popularity(self, time_series: np.ndarray, 
                          horizon: int = 7) -> float:
        """
        Predict future popularity score
        
        Args:
            time_series: Historical time series
            horizon: Prediction horizon (days)
            
        Returns:
            Predicted popularity score (mean of forecast)
        """
        try:
            self.fit(time_series)
            forecast = self.predict(steps=horizon)
            
            # Score = mean of predicted values
            score = np.mean(forecast)
            
            return float(max(0.0, score))
        
        except:
            # Fallback to last value
            return float(time_series[-1]) if len(time_series) > 0 else 0.0


class LSTMPredictor:
    """
    LSTM-based popularity prediction
    
    Deep learning approach for time series
    """
    
    def __init__(self, lookback: int = 10, units: int = 64, 
                 epochs: int = 50, batch_size: int = 32):
        """
        Initialize LSTM predictor
        
        Args:
            lookback: Number of past time steps to use
            units: Number of LSTM units
            epochs: Training epochs
            batch_size: Batch size
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM")
        
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
    
    def _create_model(self):
        """Create LSTM model architecture"""
        model = keras.Sequential([
            layers.LSTM(self.units, activation='tanh', 
                       input_shape=(self.lookback, 1)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            time_series: Input time series
            
        Returns:
            (X, y) sequences
        """
        # Normalize
        self.scaler_mean = np.mean(time_series)
        self.scaler_std = np.std(time_series) + 1e-8
        
        normalized = (time_series - self.scaler_mean) / self.scaler_std
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(normalized) - self.lookback):
            X.append(normalized[i:i+self.lookback])
            y.append(normalized[i+self.lookback])
        
        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)
        
        return X, y
    
    def fit(self, time_series: np.ndarray, verbose: int = 0):
        """
        Train LSTM model
        
        Args:
            time_series: Training time series
            verbose: Verbosity level
        """
        if len(time_series) < self.lookback + 10:
            raise ValueError("Time series too short for LSTM")
        
        try:
            # Prepare data
            X, y = self._prepare_sequences(time_series)
            
            if len(X) == 0:
                raise ValueError("No training sequences created")
            
            # Create and train model
            self.model = self._create_model()
            
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=verbose,
                validation_split=0.2
            )
        
        except Exception as e:
            print(f"LSTM training failed: {e}")
            self.model = None
    
    def predict(self, time_series: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict future values
        
        Args:
            time_series: Historical time series
            steps: Number of steps ahead
            
        Returns:
            Predicted values
        """
        if self.model is None:
            return np.zeros(steps)
        
        try:
            # Normalize input
            normalized = (time_series - self.scaler_mean) / self.scaler_std
            
            # Take last lookback steps
            current_sequence = normalized[-self.lookback:].reshape(1, self.lookback, 1)
            
            predictions = []
            
            for _ in range(steps):
                # Predict next value
                next_val = self.model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(next_val)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_val
            
            # Denormalize predictions
            predictions = np.array(predictions) * self.scaler_std + self.scaler_mean
            
            return predictions
        
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            return np.zeros(steps)
    
    def predict_popularity(self, time_series: np.ndarray, 
                          horizon: int = 7) -> float:
        """
        Predict future popularity score
        
        Args:
            time_series: Historical time series
            horizon: Prediction horizon
            
        Returns:
            Predicted popularity score
        """
        try:
            self.fit(time_series, verbose=0)
            forecast = self.predict(time_series, steps=horizon)
            
            # Score = mean of forecast
            score = np.mean(forecast)
            
            return float(max(0.0, score))
        
        except:
            # Fallback
            return float(np.mean(time_series[-7:])) if len(time_series) >= 7 else 0.0


class SimplePredictor:
    """
    Simple prediction baselines
    """
    
    @staticmethod
    def naive(time_series: np.ndarray, horizon: int = 7) -> float:
        """
        Naive forecast: last value repeated
        
        Args:
            time_series: Historical data
            horizon: Forecast horizon
            
        Returns:
            Predicted score
        """
        if len(time_series) == 0:
            return 0.0
        
        return float(time_series[-1])
    
    @staticmethod
    def moving_average(time_series: np.ndarray, 
                      window: int = 7, horizon: int = 7) -> float:
        """
        Moving average forecast
        
        Args:
            time_series: Historical data
            window: MA window size
            horizon: Forecast horizon
            
        Returns:
            Predicted score
        """
        if len(time_series) < window:
            window = len(time_series)
        
        if window == 0:
            return 0.0
        
        ma = np.mean(time_series[-window:])
        
        return float(ma)
    
    @staticmethod
    def linear_extrapolation(time_series: np.ndarray, horizon: int = 7) -> float:
        """
        Linear trend extrapolation
        
        Args:
            time_series: Historical data
            horizon: Forecast horizon
            
        Returns:
            Predicted score at horizon
        """
        if len(time_series) < 2:
            return float(time_series[-1]) if len(time_series) > 0 else 0.0
        
        # Fit linear trend
        x = np.arange(len(time_series))
        coeffs = np.polyfit(x, time_series, 1)
        
        # Extrapolate
        future_x = len(time_series) + horizon - 1
        prediction = np.polyval(coeffs, future_x)
        
        return float(max(0.0, prediction))
