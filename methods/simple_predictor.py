"""
Simple ML-based Predictor (Contribution 5)
Uses DTCWT features for Random Forest prediction
Secondary component (20% of research)
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from .dtcwt_assessment import DTCWTAssessment
from .hybrid_assessment import HybridAssessment
from config import ML_CONFIG


class SimplePredictor:
    """
    Simple predictor using DTCWT features and Random Forest
    
    Purpose: Demonstrate practical application of DTCWT features
    Note: This is 20% of research (assessment is 80%)
    """
    
    def __init__(self, feature_method='hybrid', **rf_params):
        """
        Initialize predictor
        
        Args:
            feature_method: 'dtcwt' or 'hybrid' for feature extraction
            **rf_params: Random Forest parameters
        """
        self.feature_method = feature_method
        
        # Initialize feature extractor
        if feature_method == 'dtcwt':
            self.feature_extractor = DTCWTAssessment()
        else:  # hybrid
            self.feature_extractor = HybridAssessment()
        
        # Initialize Random Forest
        rf_config = ML_CONFIG['random_forest'].copy()
        rf_config.update(rf_params)
        self.model = RandomForestRegressor(**rf_config)
        
        self.is_trained = False
    
    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract features from time series
        
        Args:
            time_series: Input time series
            
        Returns:
            Feature vector
        """
        return self.feature_extractor.get_feature_vector(time_series)
    
    def prepare_data(self, 
                    time_series_list: List[np.ndarray],
                    prediction_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from time series
        
        Args:
            time_series_list: List of time series (each should be window_size + horizon long)
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            Tuple of (X_features, y_targets)
        """
        X = []
        y = []
        
        for ts in time_series_list:
            if len(ts) < prediction_horizon + 10:
                continue
            
            # Split into current window and future
            current = ts[:-prediction_horizon]
            future = ts[-prediction_horizon:]
            
            # Extract features from current window
            features = self.extract_features(current)
            
            # Target: sum of future accesses
            target = np.sum(future)
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the predictor
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        if validation_split > 0:
            # Split for validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=validation_split,
                random_state=42
            )
            
            # Train
            self.model.fit(X_tr, y_tr)
            
            # Validate
            y_pred = self.model.predict(X_val)
            
            # Compute metrics
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'train_score': self.model.score(X_tr, y_tr),
                'val_score': self.model.score(X_val, y_val),
            }
        else:
            # Train on all data
            self.model.fit(X_train, y_train)
            
            metrics = {
                'train_score': self.model.score(X_train, y_train),
            }
        
        self.is_trained = True
        
        return metrics
    
    def predict(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Predict future popularity
        
        Args:
            time_series_list: List of time series (current window)
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        # Extract features
        X = np.array([self.extract_features(ts) for ts in time_series_list])
        
        # Predict
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_single(self, time_series: np.ndarray) -> float:
        """
        Predict for a single time series
        
        Args:
            time_series: Input time series
            
        Returns:
            Prediction value
        """
        return float(self.predict([time_series])[0])
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances from Random Forest
        
        Returns:
            Array of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        return self.model.feature_importances_
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Test metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': np.mean(np.abs(y_test - y_pred)),
            'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2)),
            'r2': self.model.score(X_test, y_test),
        }
        
        # Add MAPE if possible
        mask = y_test != 0
        if np.any(mask):
            metrics['mape'] = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        return metrics


class BaselinePredictor:
    """
    Baseline prediction methods for comparison
    """
    
    @staticmethod
    def last_value(time_series: np.ndarray, horizon: int = 1) -> float:
        """Predict using last observed value"""
        return float(time_series[-1]) * horizon
    
    @staticmethod
    def moving_average(time_series: np.ndarray, window: int = 7, horizon: int = 1) -> float:
        """Predict using moving average"""
        return float(np.mean(time_series[-window:])) * horizon
    
    @staticmethod
    def linear_trend(time_series: np.ndarray, horizon: int = 1) -> float:
        """Predict using linear extrapolation"""
        x = np.arange(len(time_series))
        coeffs = np.polyfit(x, time_series, deg=1)
        
        # Predict future points
        future_x = np.arange(len(time_series), len(time_series) + horizon)
        predictions = np.polyval(coeffs, future_x)
        
        return float(np.sum(np.maximum(0, predictions)))
    
    @staticmethod
    def exponential_smoothing(time_series: np.ndarray, alpha: float = 0.3, horizon: int = 1) -> float:
        """Predict using exponential smoothing"""
        # Compute EWMA
        ewma = time_series[0]
        for value in time_series[1:]:
            ewma = alpha * value + (1 - alpha) * ewma
        
        return float(ewma) * horizon


def compare_predictors(time_series_list: List[np.ndarray],
                      prediction_horizon: int = 7,
                      test_size: float = 0.2) -> Dict[str, Dict]:
    """
    Compare DTCWT predictor with baselines
    
    Args:
        time_series_list: List of time series
        prediction_horizon: Prediction horizon
        test_size: Fraction for testing
        
    Returns:
        Dictionary of results for each method
    """
    # Initialize predictor
    dtcwt_pred = SimplePredictor(feature_method='hybrid')
    
    # Prepare data
    X, y = dtcwt_pred.prepare_data(time_series_list, prediction_horizon)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train DTCWT predictor
    train_metrics = dtcwt_pred.train(X_train, y_train, validation_split=0.0)
    test_metrics = dtcwt_pred.evaluate(X_test, y_test)
    
    results = {
        'DTCWT-RF': test_metrics
    }
    
    # Baseline predictions
    baseline_preds = {
        'Last Value': [],
        'Moving Avg': [],
        'Linear Trend': [],
        'Exp Smoothing': [],
    }
    
    for ts in [time_series_list[i] for i in range(len(y_test))]:
        current = ts[:-prediction_horizon]
        
        baseline_preds['Last Value'].append(
            BaselinePredictor.last_value(current, prediction_horizon)
        )
        baseline_preds['Moving Avg'].append(
            BaselinePredictor.moving_average(current, horizon=prediction_horizon)
        )
        baseline_preds['Linear Trend'].append(
            BaselinePredictor.linear_trend(current, prediction_horizon)
        )
        baseline_preds['Exp Smoothing'].append(
            BaselinePredictor.exponential_smoothing(current, horizon=prediction_horizon)
        )
    
    # Evaluate baselines
    for method, preds in baseline_preds.items():
        preds = np.array(preds)
        results[method] = {
            'mae': np.mean(np.abs(y_test - preds)),
            'rmse': np.sqrt(np.mean((y_test - preds) ** 2)),
        }
    
    return results
