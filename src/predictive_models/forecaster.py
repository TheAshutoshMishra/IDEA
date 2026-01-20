"""
Predictive Modeling & Forecasting Module
Time series forecasting and predictive analytics
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class Forecaster:
    """Predictive modeling and forecasting for Aadhaar data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize forecaster"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load forecasting parameters
        self.methods = config['predictive_models']['time_series']['methods']
    
    def forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main forecasting pipeline
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of forecasts and predictions
        """
        self.logger.info("Starting predictive modeling")
        
        predictions = {
            'enrollment_forecast': self.forecast_enrollment(df),
            'update_forecast': self.forecast_updates(df),
            'resource_predictions': self.predict_resource_needs(df),
            'trend_predictions': self.predict_trends(df)
        }
        
        self.logger.info("Predictive modeling complete")
        return predictions
    
    def forecast_enrollment(self, df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]:
        """
        Forecast future enrollment numbers
        
        Args:
            df: DataFrame with enrollment data
            periods: Number of periods to forecast
            
        Returns:
            Forecast results
        """
        if 'enrollment_date' not in df.columns:
            self.logger.warning("No enrollment_date column found")
            return {}
        
        # Prepare time series data
        ts_data = df.groupby('enrollment_date').size().reset_index()
        ts_data.columns = ['date', 'enrollments']
        ts_data = ts_data.sort_values('date')
        
        results = {}
        
        # Prophet Forecast
        if 'prophet' in self.methods:
            try:
                prophet_forecast = self._forecast_with_prophet(ts_data, periods)
                results['prophet'] = prophet_forecast
            except Exception as e:
                self.logger.error(f"Prophet forecast failed: {e}")
        
        # ARIMA Forecast
        if 'arima' in self.methods:
            try:
                arima_forecast = self._forecast_with_arima(ts_data, periods)
                results['arima'] = arima_forecast
            except Exception as e:
                self.logger.error(f"ARIMA forecast failed: {e}")
        
        # Simple moving average baseline
        ma_forecast = self._forecast_with_moving_average(ts_data, periods)
        results['moving_average'] = ma_forecast
        
        # Ensemble prediction (average of all methods)
        if len(results) > 1:
            ensemble = self._create_ensemble_forecast(results, periods)
            results['ensemble'] = ensemble
        
        return results
    
    def _forecast_with_prophet(self, ts_data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Forecast using Facebook Prophet"""
        # Prepare data for Prophet
        prophet_df = ts_data.copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and fit Prophet model
        model = Prophet(
            seasonality_mode=self.config['predictive_models']['time_series']['prophet']['seasonality_mode'],
            changepoint_prior_scale=self.config['predictive_models']['time_series']['prophet']['changepoint_prior_scale']
        )
        model.fit(prophet_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)
        
        # Extract predictions
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        return {
            'predictions': predictions['yhat'].tolist(),
            'lower_bound': predictions['yhat_lower'].tolist(),
            'upper_bound': predictions['yhat_upper'].tolist(),
            'dates': predictions['ds'].dt.strftime('%Y-%m-%d').tolist()
        }
    
    def _forecast_with_arima(self, ts_data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Forecast using ARIMA"""
        # Prepare data
        y = ts_data['enrollments'].values
        
        # Fit ARIMA model (simplified - in production would use auto_arima)
        try:
            model = ARIMA(y, order=(1, 1, 1))
            fitted = model.fit()
            
            # Forecast
            forecast = fitted.forecast(steps=periods)
            
            return {
                'predictions': forecast.tolist(),
                'model_order': (1, 1, 1)
            }
        except:
            # Fallback to simpler model
            model = ARIMA(y, order=(0, 1, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=periods)
            
            return {
                'predictions': forecast.tolist(),
                'model_order': (0, 1, 0)
            }
    
    def _forecast_with_moving_average(self, ts_data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Simple moving average forecast as baseline"""
        window = min(30, len(ts_data) // 2)
        ma = ts_data['enrollments'].rolling(window=window).mean().iloc[-1]
        
        predictions = [ma] * periods
        
        return {
            'predictions': predictions,
            'window_size': window
        }
    
    def _create_ensemble_forecast(self, forecasts: Dict[str, Any], periods: int) -> Dict[str, Any]:
        """Create ensemble forecast by averaging multiple methods"""
        all_predictions = []
        
        for method, forecast in forecasts.items():
            if 'predictions' in forecast:
                all_predictions.append(forecast['predictions'])
        
        if not all_predictions:
            return {}
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        return {
            'predictions': ensemble_pred.tolist(),
            'methods_used': list(forecasts.keys()),
            'n_methods': len(all_predictions)
        }
    
    def forecast_updates(self, df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]:
        """Forecast update activity"""
        if 'last_update_date' not in df.columns:
            return {}
        
        # Prepare time series
        ts_data = df.groupby('last_update_date').size().reset_index()
        ts_data.columns = ['date', 'updates']
        ts_data = ts_data.sort_values('date')
        
        # Use similar approach as enrollment forecast
        try:
            return self._forecast_with_prophet(ts_data, periods)
        except:
            return self._forecast_with_moving_average(ts_data, periods)
    
    def predict_resource_needs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict resource requirements based on patterns"""
        predictions = {}
        
        # Predict enrollment center load
        if 'state' in df.columns and 'enrollment_date' in df.columns:
            state_daily = df.groupby(['state', df['enrollment_date'].dt.date]).size()
            state_avg = state_daily.groupby('state').mean()
            
            predictions['avg_daily_load_by_state'] = state_avg.to_dict()
            predictions['peak_load_states'] = state_avg.nlargest(5).to_dict()
        
        # Predict update processing capacity needed
        if 'update_count' in df.columns:
            total_updates = df['update_count'].sum()
            avg_per_day = total_updates / len(df) if len(df) > 0 else 0
            
            predictions['estimated_daily_updates'] = float(avg_per_day)
            predictions['recommended_capacity'] = float(avg_per_day * 1.5)  # 50% buffer
        
        return predictions
    
    def predict_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict emerging trends"""
        trends = {}
        
        # Age group trends
        if 'age' in df.columns and 'enrollment_date' in df.columns:
            recent = df[df['enrollment_date'] >= df['enrollment_date'].max() - pd.Timedelta(days=90)]
            overall = df
            
            recent_age_dist = recent['age'].describe()
            overall_age_dist = overall['age'].describe()
            
            trends['age_demographics'] = {
                'recent_mean_age': float(recent_age_dist['mean']),
                'overall_mean_age': float(overall_age_dist['mean']),
                'trend': 'younger' if recent_age_dist['mean'] < overall_age_dist['mean'] else 'older'
            }
        
        # Geographic trends
        if 'state' in df.columns and 'enrollment_date' in df.columns:
            recent = df[df['enrollment_date'] >= df['enrollment_date'].max() - pd.Timedelta(days=30)]
            recent_states = recent['state'].value_counts()
            
            trends['emerging_states'] = recent_states.head(5).to_dict()
        
        return trends
