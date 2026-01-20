"""
Predictive Risk Scoring Engine
Predicts likelihood of high-failure scenarios
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from datetime import datetime, timedelta
import logging


class RiskScorer:
    """Predict risk of authentication failures and system issues"""
    
    def __init__(self, model_type='gradient_boosting'):
        """
        Initialize risk scorer
        
        Args:
            model_type: 'gradient_boosting' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 30,
            'medium': 60,
            'high': 80
        }
    
    def prepare_training_data(self, df: pd.DataFrame, window_minutes=60) -> pd.DataFrame:
        """
        Prepare data for training by creating time-window features
        
        Args:
            df: Transaction DataFrame
            window_minutes: Size of prediction window
            
        Returns:
            DataFrame with features and target
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{window_minutes}min')
        
        # Aggregate features per window
        window_features = df.groupby('time_window').agg({
            'result': lambda x: (x == 'Failure').sum() / len(x),  # Failure rate
            'retry_count': ['mean', 'max', 'sum'],
            'response_time_ms': ['mean', 'max'],
            'transaction_id': 'count'  # Transaction volume
        }).reset_index()
        
        # Flatten column names
        window_features.columns = [
            'time_window', 'failure_rate', 'avg_retry', 'max_retry', 'total_retries',
            'avg_response_time', 'max_response_time', 'transaction_count'
        ]
        
        # Add time-based features
        window_features['hour'] = window_features['time_window'].dt.hour
        window_features['day_of_week'] = window_features['time_window'].dt.dayofweek
        window_features['is_weekend'] = window_features['day_of_week'].isin([5, 6]).astype(int)
        window_features['is_off_hours'] = (
            (window_features['hour'] >= 23) | (window_features['hour'] <= 6)
        ).astype(int)
        
        # Create lagged features (previous window)
        for col in ['failure_rate', 'avg_retry', 'transaction_count']:
            window_features[f'{col}_prev'] = window_features[col].shift(1)
        
        # Create target: high failure rate in NEXT window
        window_features['target'] = (window_features['failure_rate'].shift(-1) > 0.15).astype(int)
        
        # Drop rows with NaN (first and last)
        window_features = window_features.dropna()
        
        return window_features
    
    def train(self, df: pd.DataFrame, test_size=0.2):
        """
        Train risk prediction model
        
        Args:
            df: Prepared training DataFrame
            test_size: Proportion of data for testing
        """
        self.logger.info("Training risk scoring model...")
        
        # Prepare data if not already done
        if 'target' not in df.columns:
            df = self.prepare_training_data(df)
        
        # Define features
        self.feature_columns = [
            'failure_rate', 'avg_retry', 'max_retry', 'total_retries',
            'avg_response_time', 'max_response_time', 'transaction_count',
            'hour', 'day_of_week', 'is_weekend', 'is_off_hours',
            'failure_rate_prev', 'avg_retry_prev', 'transaction_count_prev'
        ]
        
        X = df[self.feature_columns]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Initialize model
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.logger.info(f"Model trained. Test accuracy: {(y_pred == y_test).mean():.3f}")
        self.logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        self.is_trained = True
        
        # Print classification report
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop Features:")
            print(feature_importance.head(5))
    
    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk scores for transactions
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with risk scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        if 'target' not in df.columns:
            features_df = self.prepare_training_data(df)
        else:
            features_df = df.copy()
        
        X = features_df[self.feature_columns]
        
        # Predict probability
        risk_probabilities = self.model.predict_proba(X)[:, 1]
        
        # Convert to risk score (0-100)
        risk_scores = (risk_probabilities * 100).round(1)
        
        # Assign risk labels
        risk_labels = pd.cut(
            risk_scores,
            bins=[-1, self.risk_thresholds['low'], self.risk_thresholds['medium'], 
                  self.risk_thresholds['high'], 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Create results DataFrame
        results = features_df[['time_window']].copy()
        results['risk_score'] = risk_scores
        results['risk_label'] = risk_labels
        results['risk_probability'] = risk_probabilities
        
        return results
    
    def get_risk_factors(self, df: pd.DataFrame, top_n=5) -> dict:
        """
        Get top risk contributing factors
        
        Args:
            df: DataFrame with features
            top_n: Number of top factors to return
            
        Returns:
            Dictionary of risk factors
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_factors = feature_importance.head(top_n)
        
        # Get actual values for these features
        factors = {}
        for _, row in top_factors.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature in df.columns:
                value = df[feature].iloc[-1] if len(df) > 0 else None
                factors[feature] = {
                    'importance': float(importance),
                    'current_value': float(value) if value is not None else None
                }
        
        return factors
    
    def predict_next_window(self, df: pd.DataFrame, window_minutes=60) -> dict:
        """
        Predict risk for the next time window
        
        Args:
            df: Recent transaction DataFrame
            window_minutes: Window size in minutes
            
        Returns:
            Prediction dictionary
        """
        # Prepare latest window data
        features_df = self.prepare_training_data(df, window_minutes=window_minutes)
        
        if len(features_df) == 0:
            return {
                'risk_score': 50,
                'risk_label': 'Unknown',
                'message': 'Insufficient data for prediction'
            }
        
        # Get latest window
        latest = features_df.iloc[[-1]]
        
        # Predict
        risk_result = self.predict_risk(latest)
        
        # Get risk factors
        risk_factors = self.get_risk_factors(latest)
        
        return {
            'time_window': latest['time_window'].iloc[0],
            'risk_score': float(risk_result['risk_score'].iloc[0]),
            'risk_label': str(risk_result['risk_label'].iloc[0]),
            'risk_probability': float(risk_result['risk_probability'].iloc[0]),
            'top_risk_factors': risk_factors,
            'current_metrics': {
                'failure_rate': float(latest['failure_rate'].iloc[0]),
                'avg_retry': float(latest['avg_retry'].iloc[0]),
                'transaction_count': int(latest['transaction_count'].iloc[0])
            }
        }
    
    def save_model(self, filepath='models/risk_model.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'risk_thresholds': self.risk_thresholds,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/risk_model.pkl'):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.risk_thresholds = model_data['risk_thresholds']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")


def main():
    """Demo the risk scorer"""
    print("Testing Predictive Risk Scoring...")
    
    # Load data
    df = pd.read_csv('data/raw/auth_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df):,} transactions")
    
    # Initialize scorer
    scorer = RiskScorer(model_type='gradient_boosting')
    
    # Prepare training data
    print("\nPreparing training data...")
    training_data = scorer.prepare_training_data(df, window_minutes=60)
    
    print(f"Created {len(training_data)} time windows")
    print(f"Target distribution: {training_data['target'].value_counts().to_dict()}")
    
    # Train model
    print("\nTraining model...")
    scorer.train(training_data)
    
    # Make predictions
    print("\nMaking risk predictions...")
    predictions = scorer.predict_risk(training_data)
    
    print(f"\nRisk Score Distribution:")
    print(predictions['risk_label'].value_counts())
    
    # Predict next window
    recent_data = df[df['timestamp'] >= df['timestamp'].max() - timedelta(hours=2)]
    next_prediction = scorer.predict_next_window(recent_data)
    
    print(f"\n🔮 Next Window Prediction:")
    print(f"   Time: {next_prediction['time_window']}")
    print(f"   Risk Score: {next_prediction['risk_score']:.1f}")
    print(f"   Risk Label: {next_prediction['risk_label']}")
    print(f"   Probability: {next_prediction['risk_probability']:.2%}")
    
    print(f"\n   Current Metrics:")
    for metric, value in next_prediction['current_metrics'].items():
        print(f"      {metric}: {value}")
    
    print(f"\n   Top Risk Factors:")
    for factor, data in next_prediction['top_risk_factors'].items():
        print(f"      {factor}: importance={data['importance']:.3f}, value={data['current_value']}")
    
    # Save model
    print("\nSaving model...")
    scorer.save_model('models/risk_model.pkl')
    print("✅ Model saved successfully")


if __name__ == "__main__":
    main()
