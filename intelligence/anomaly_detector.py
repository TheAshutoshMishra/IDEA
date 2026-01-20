"""
Real-Time Anomaly Detection Engine
Detects suspicious patterns in authentication transactions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging


class RealTimeAnomalyDetector:
    """Detect anomalies in Aadhaar authentication streams"""
    
    def __init__(self, contamination=0.05):
        """
        Initialize detector
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Thresholds for rule-based detection
        self.thresholds = {
            'max_retry_count': 5,
            'max_failure_rate_1h': 0.30,  # 30% failures in 1 hour
            'max_response_time': 3000,  # 3 seconds
            'off_hours_start': 23,  # 11 PM
            'off_hours_end': 6,  # 6 AM
        }
    
    def train(self, df: pd.DataFrame):
        """
        Train anomaly detection model on historical data
        
        Args:
            df: Historical transaction DataFrame
        """
        self.logger.info("Training anomaly detection model...")
        
        # Prepare features
        X = self._extract_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        self.is_trained = True
        self.logger.info("Model trained successfully")
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in transaction data
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        results = df.copy()
        
        # ML-based detection
        if self.is_trained and len(df) > 0:
            ml_anomalies = self._detect_ml_anomalies(df)
            results = results.merge(ml_anomalies, left_index=True, right_index=True, how='left')
        
        # Rule-based detection
        rule_anomalies = self._detect_rule_based_anomalies(df)
        results = results.merge(rule_anomalies, left_index=True, right_index=True, how='left')
        
        # Combine anomaly flags
        results['is_anomaly'] = (
            results.get('ml_anomaly', 0) | results.get('rule_anomaly', 0)
        ).astype(int)
        
        # Calculate severity
        results['severity'] = results.apply(self._calculate_severity, axis=1)
        
        # Calculate confidence
        results['confidence'] = results.apply(self._calculate_confidence, axis=1)
        
        # Get explanations
        results['anomaly_reasons'] = results.apply(
            lambda row: self._get_explanation(row) if row['is_anomaly'] else None,
            axis=1
        )
        
        return results
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        # Numeric features
        numeric_cols = ['retry_count', 'response_time_ms', 'hour']
        for col in numeric_cols:
            if col in df.columns:
                features.append(df[col].values.reshape(-1, 1))
        
        # Categorical features (encoded)
        if 'result' in df.columns:
            result_encoded = (df['result'] == 'Failure').astype(int).values.reshape(-1, 1)
            features.append(result_encoded)
        
        if 'auth_type' in df.columns:
            # One-hot encode auth_type
            auth_dummies = pd.get_dummies(df['auth_type'], prefix='auth')
            features.append(auth_dummies.values)
        
        # Concatenate all features
        X = np.hstack(features)
        return X
    
    def _detect_ml_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """ML-based anomaly detection using Isolation Forest"""
        X = self._extract_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 for anomaly, 1 for normal)
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)
        
        results = pd.DataFrame(index=df.index)
        results['ml_anomaly'] = (predictions == -1).astype(int)
        results['ml_score'] = -anomaly_scores  # Convert to positive (higher = more anomalous)
        
        return results
    
    def _detect_rule_based_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule-based anomaly detection"""
        results = pd.DataFrame(index=df.index)
        results['rule_anomaly'] = 0
        results['rule_violations'] = [[] for _ in range(len(df))]
        
        # Rule 1: Excessive retries
        if 'retry_count' in df.columns:
            excessive_retries = df['retry_count'] > self.thresholds['max_retry_count']
            results.loc[excessive_retries, 'rule_anomaly'] = 1
            results.loc[excessive_retries, 'rule_violations'] = results.loc[excessive_retries, 'rule_violations'].apply(
                lambda x: x + ['excessive_retries']
            )
        
        # Rule 2: High response time
        if 'response_time_ms' in df.columns:
            slow_response = df['response_time_ms'] > self.thresholds['max_response_time']
            results.loc[slow_response, 'rule_anomaly'] = 1
            results.loc[slow_response, 'rule_violations'] = results.loc[slow_response, 'rule_violations'].apply(
                lambda x: x + ['slow_response']
            )
        
        # Rule 3: Off-hours activity (unusual)
        if 'hour' in df.columns:
            off_hours = (df['hour'] >= self.thresholds['off_hours_start']) | \
                       (df['hour'] <= self.thresholds['off_hours_end'])
            results.loc[off_hours, 'rule_violations'] = results.loc[off_hours, 'rule_violations'].apply(
                lambda x: x + ['off_hours_activity']
            )
        
        # Rule 4: Failure with high retries
        if 'result' in df.columns and 'retry_count' in df.columns:
            failed_with_retries = (df['result'] == 'Failure') & (df['retry_count'] >= 3)
            results.loc[failed_with_retries, 'rule_anomaly'] = 1
            results.loc[failed_with_retries, 'rule_violations'] = results.loc[failed_with_retries, 'rule_violations'].apply(
                lambda x: x + ['failed_with_high_retries']
            )
        
        return results
    
    def _calculate_severity(self, row) -> str:
        """Calculate anomaly severity"""
        if not row['is_anomaly']:
            return 'None'
        
        # Count violations
        violation_count = 0
        if 'rule_violations' in row and isinstance(row['rule_violations'], list):
            violation_count = len(row['rule_violations'])
        
        if 'ml_anomaly' in row and row['ml_anomaly']:
            violation_count += 1
        
        # Determine severity
        if violation_count >= 3:
            return 'High'
        elif violation_count >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_confidence(self, row) -> float:
        """Calculate confidence score (0-100)"""
        if not row['is_anomaly']:
            return 0.0
        
        confidence = 50.0  # Base confidence
        
        # Increase confidence based on ML score
        if 'ml_score' in row and row['ml_anomaly']:
            confidence += min(row['ml_score'] * 20, 30)
        
        # Increase confidence based on rule violations
        if 'rule_violations' in row and isinstance(row['rule_violations'], list):
            confidence += len(row['rule_violations']) * 10
        
        return min(confidence, 100.0)
    
    def _get_explanation(self, row) -> str:
        """Generate human-readable explanation"""
        reasons = []
        
        # ML-based reason
        if 'ml_anomaly' in row and row['ml_anomaly']:
            reasons.append("ML model flagged as unusual pattern")
        
        # Rule-based reasons
        if 'rule_violations' in row and isinstance(row['rule_violations'], list):
            for violation in row['rule_violations']:
                if violation == 'excessive_retries':
                    reasons.append(f"Excessive retries ({row.get('retry_count', 'N/A')} attempts)")
                elif violation == 'slow_response':
                    reasons.append(f"Slow response time ({row.get('response_time_ms', 'N/A')}ms)")
                elif violation == 'off_hours_activity':
                    reasons.append(f"Off-hours activity (hour: {row.get('hour', 'N/A')})")
                elif violation == 'failed_with_high_retries':
                    reasons.append("Authentication failed with multiple retries")
        
        return "; ".join(reasons) if reasons else "Unknown"
    
    def detect_regional_anomalies(self, df: pd.DataFrame, window='1H') -> pd.DataFrame:
        """
        Detect region-wise anomalies (sudden spikes)
        
        Args:
            df: Transaction DataFrame with 'timestamp' and 'region'
            window: Time window for aggregation
            
        Returns:
            DataFrame with regional anomaly scores
        """
        if 'region' not in df.columns or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        # Group by region and time window
        df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor(window)
        
        regional_stats = df.groupby(['region', 'time_window']).agg({
            'result': lambda x: (x == 'Failure').sum() / len(x),  # Failure rate
            'retry_count': 'mean',
            'response_time_ms': 'mean'
        }).reset_index()
        
        regional_stats.columns = ['region', 'time_window', 'failure_rate', 
                                  'avg_retries', 'avg_response_time']
        
        # Detect anomalies using Z-score
        for col in ['failure_rate', 'avg_retries', 'avg_response_time']:
            mean = regional_stats[col].mean()
            std = regional_stats[col].std()
            regional_stats[f'{col}_zscore'] = (regional_stats[col] - mean) / std if std > 0 else 0
        
        # Flag anomalies (Z-score > 2)
        regional_stats['is_regional_anomaly'] = (
            (regional_stats['failure_rate_zscore'].abs() > 2) |
            (regional_stats['avg_retries_zscore'].abs() > 2) |
            (regional_stats['avg_response_time_zscore'].abs() > 2)
        ).astype(int)
        
        return regional_stats
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        """Generate anomaly summary statistics"""
        if 'is_anomaly' not in df.columns:
            df = self.detect_anomalies(df)
        
        anomalies = df[df['is_anomaly'] == 1]
        
        summary = {
            'total_transactions': len(df),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) if len(df) > 0 else 0,
            'severity_distribution': anomalies['severity'].value_counts().to_dict() if 'severity' in anomalies.columns else {},
            'avg_confidence': anomalies['confidence'].mean() if 'confidence' in anomalies.columns else 0,
            'high_severity_count': len(anomalies[anomalies['severity'] == 'High']) if 'severity' in anomalies.columns else 0
        }
        
        return summary


def main():
    """Demo the anomaly detector"""
    print("Testing Real-Time Anomaly Detection...")
    
    # Load data
    df = pd.read_csv('data/raw/auth_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df):,} transactions")
    
    # Initialize detector
    detector = RealTimeAnomalyDetector()
    
    # Train on first 80% of data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    print(f"\nTraining on {len(train_df):,} transactions...")
    detector.train(train_df)
    
    # Detect anomalies in test set
    print(f"Detecting anomalies in {len(test_df):,} transactions...")
    results = detector.detect_anomalies(test_df)
    
    # Get summary
    summary = detector.get_anomaly_summary(results)
    
    print("\nAnomaly Detection Summary:")
    print(f"   Total transactions: {summary['total_transactions']:,}")
    print(f"   Anomalies detected: {summary['total_anomalies']:,}")
    print(f"   Anomaly rate: {summary['anomaly_rate']:.2%}")
    print(f"   High severity: {summary['high_severity_count']:,}")
    print(f"   Avg confidence: {summary['avg_confidence']:.1f}%")
    
    if summary['severity_distribution']:
        print(f"\n   Severity distribution:")
        for severity, count in summary['severity_distribution'].items():
            print(f"      {severity}: {count}")
    
    # Show sample anomalies
    anomalies = results[results['is_anomaly'] == 1].head(5)
    if len(anomalies) > 0:
        print(f"\n🚨 Sample Anomalies:")
        for idx, row in anomalies.iterrows():
            print(f"\n   Transaction: {row.get('transaction_id', idx)}")
            print(f"   Severity: {row['severity']}")
            print(f"   Confidence: {row['confidence']:.1f}%")
            print(f"   Reason: {row['anomaly_reasons']}")
    
    # Regional anomalies
    print(f"\n🗺️ Detecting regional anomalies...")
    regional_anomalies = detector.detect_regional_anomalies(test_df)
    regional_issues = regional_anomalies[regional_anomalies['is_regional_anomaly'] == 1]
    
    if len(regional_issues) > 0:
        print(f"   Found {len(regional_issues)} regional anomaly windows")
        print(f"\n   Top affected regions:")
        print(regional_issues.groupby('region').size().sort_values(ascending=False).head())


if __name__ == "__main__":
    main()
