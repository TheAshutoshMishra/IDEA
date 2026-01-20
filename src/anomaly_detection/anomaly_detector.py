"""
Anomaly Detection Module
Detects outliers, suspicious patterns, and data quality issues
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import logging
from typing import Dict, List, Any, Tuple


class AnomalyDetector:
    """Detect anomalies and suspicious patterns in Aadhaar data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize anomaly detector"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load anomaly detection parameters
        self.methods = config['anomaly_detection']['methods']
        self.contamination = config['anomaly_detection']['isolation_forest']['contamination']
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main anomaly detection pipeline
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of detected anomalies
        """
        self.logger.info("Starting anomaly detection")
        
        anomalies = {
            'statistical_outliers': self._detect_statistical_outliers(df),
            'ml_anomalies': self._detect_ml_anomalies(df),
            'data_quality_issues': self._detect_data_quality_issues(df),
            'suspicious_patterns': self._detect_suspicious_patterns(df),
            'fraud_indicators': self.detect_fraud(df)
        }
        
        # Summarize
        total_anomalies = sum(
            len(v) if isinstance(v, list) else v.get('count', 0)
            for v in anomalies.values() if v
        )
        
        anomalies['summary'] = {
            'total_records': len(df),
            'total_anomalies_detected': total_anomalies,
            'anomaly_rate': total_anomalies / len(df) if len(df) > 0 else 0
        }
        
        self.logger.info(f"Anomaly detection complete: {total_anomalies} anomalies found")
        return anomalies
    
    def _detect_statistical_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical outliers using Z-score and IQR"""
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_outliers = []
            
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_outliers = np.where(z_scores > 3)[0]
            
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = df[
                (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            ].index.tolist()
            
            if len(z_outliers) > 0 or len(iqr_outliers) > 0:
                outliers[col] = {
                    'z_score_outliers': len(z_outliers),
                    'iqr_outliers': len(iqr_outliers),
                    'outlier_percentage': (len(iqr_outliers) / len(df)) * 100
                }
        
        return outliers
    
    def _detect_ml_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using ML algorithms"""
        results = {}
        
        # Prepare data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(0)
        
        if len(X) < 10:
            self.logger.warning("Insufficient data for ML anomaly detection")
            return results
        
        # Isolation Forest
        if 'isolation_forest' in self.methods:
            try:
                iso_forest = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                predictions = iso_forest.fit_predict(X)
                anomaly_indices = np.where(predictions == -1)[0]
                
                results['isolation_forest'] = {
                    'anomalies_detected': len(anomaly_indices),
                    'anomaly_indices': anomaly_indices.tolist()[:100],  # Limit output
                    'anomaly_percentage': (len(anomaly_indices) / len(df)) * 100
                }
            except Exception as e:
                self.logger.error(f"Isolation Forest failed: {e}")
        
        # Local Outlier Factor
        if 'local_outlier_factor' in self.methods:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=self.contamination
                )
                predictions = lof.fit_predict(X)
                anomaly_indices = np.where(predictions == -1)[0]
                
                results['local_outlier_factor'] = {
                    'anomalies_detected': len(anomaly_indices),
                    'anomaly_indices': anomaly_indices.tolist()[:100],
                    'anomaly_percentage': (len(anomaly_indices) / len(df)) * 100
                }
            except Exception as e:
                self.logger.error(f"LOF failed: {e}")
        
        # One-Class SVM
        if 'one_class_svm' in self.methods:
            try:
                # Limit data for SVM due to computational constraints
                sample_size = min(10000, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)
                
                oc_svm = OneClassSVM(gamma='auto', nu=self.contamination)
                predictions = oc_svm.fit_predict(X_sample)
                anomaly_indices = np.where(predictions == -1)[0]
                
                results['one_class_svm'] = {
                    'sample_size': sample_size,
                    'anomalies_detected': len(anomaly_indices),
                    'anomaly_percentage': (len(anomaly_indices) / sample_size) * 100
                }
            except Exception as e:
                self.logger.error(f"One-Class SVM failed: {e}")
        
        return results
    
    def _detect_data_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect data quality issues"""
        issues = {}
        
        # Missing values
        missing = df.isnull().sum()
        issues['missing_values'] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        issues['duplicate_rows'] = int(duplicates)
        
        # Invalid values
        if 'age' in df.columns:
            invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
            issues['invalid_age_count'] = len(invalid_age)
        
        # Inconsistent formats
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            unique_formats = df[col].apply(lambda x: len(str(x)) if pd.notna(x) else 0).nunique()
            if unique_formats > 10:  # Many different formats
                issues[f'{col}_format_inconsistency'] = {
                    'unique_formats': int(unique_formats)
                }
        
        return issues
    
    def _detect_suspicious_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect suspicious behavioral patterns"""
        suspicious = {}
        
        # Multiple enrollments from same location in short time
        if 'enrollment_date' in df.columns and 'location' in df.columns:
            df['date_hour'] = df['enrollment_date'].dt.floor('H')
            location_counts = df.groupby(['location', 'date_hour']).size()
            high_frequency = location_counts[location_counts > 50]  # Threshold
            
            suspicious['high_frequency_locations'] = {
                'count': len(high_frequency),
                'locations': high_frequency.head(10).to_dict()
            }
        
        # Unusual update patterns
        if 'update_count' in df.columns:
            excessive_updates = df[df['update_count'] > 10]  # Threshold
            suspicious['excessive_updates'] = {
                'count': len(excessive_updates),
                'percentage': (len(excessive_updates) / len(df)) * 100
            }
        
        # Same mobile number for multiple enrollments
        if 'mobile_number' in df.columns:
            mobile_counts = df['mobile_number'].value_counts()
            duplicate_mobiles = mobile_counts[mobile_counts > 5]  # Threshold
            
            suspicious['duplicate_mobile_numbers'] = {
                'count': len(duplicate_mobiles),
                'top_duplicates': duplicate_mobiles.head(10).to_dict()
            }
        
        return suspicious
    
    def detect_fraud(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential fraud indicators"""
        fraud_indicators = {}
        
        # Rapid enrollment/update patterns
        if 'enrollment_date' in df.columns:
            df['date'] = df['enrollment_date'].dt.date
            daily_counts = df.groupby('date').size()
            
            # Detect sudden spikes
            mean = daily_counts.mean()
            std = daily_counts.std()
            spikes = daily_counts[daily_counts > mean + 3 * std]
            
            fraud_indicators['enrollment_spikes'] = {
                'spike_dates': len(spikes),
                'dates': spikes.to_dict() if len(spikes) < 10 else {}
            }
        
        # Age/demographic inconsistencies
        if 'age' in df.columns and 'enrollment_date' in df.columns:
            df['age_at_enrollment'] = df['enrollment_date'].dt.year - df['age']
            # Check for inconsistent birth years
            age_variance = df.groupby('id')['age_at_enrollment'].std() if 'id' in df.columns else None
            
            if age_variance is not None:
                inconsistent = age_variance[age_variance > 1]
                fraud_indicators['age_inconsistencies'] = len(inconsistent)
        
        # Biometric failures
        if 'biometric_status' in df.columns:
            failed = df[df['biometric_status'] == 'failed']
            fraud_indicators['biometric_failures'] = {
                'count': len(failed),
                'percentage': (len(failed) / len(df)) * 100
            }
        
        return fraud_indicators
