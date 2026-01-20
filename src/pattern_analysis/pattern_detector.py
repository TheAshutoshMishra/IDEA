"""
Pattern Detection Module
Identifies meaningful patterns and trends in enrollment/update data
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy import stats
import logging
from typing import Dict, List, Any


class PatternDetector:
    """Detect patterns, trends, and insights in Aadhaar data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pattern detector"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load pattern analysis parameters
        self.clustering_method = config['pattern_analysis']['clustering']['method']
        self.n_clusters = config['pattern_analysis']['clustering']['n_clusters']
        self.trend_method = config['pattern_analysis']['trend_detection']['method']
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main pattern detection pipeline
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of detected patterns
        """
        self.logger.info("Starting pattern detection")
        
        patterns = {
            'enrollment_patterns': self.analyze_enrollment(df),
            'update_patterns': self.analyze_updates(df),
            'geographic_patterns': self.analyze_geographic(df),
            'demographic_patterns': self.analyze_demographic(df),
            'temporal_patterns': self.analyze_temporal(df),
            'clusters': self.perform_clustering(df)
        }
        
        self.logger.info(f"Pattern detection complete")
        return patterns
    
    def analyze_enrollment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enrollment-specific patterns"""
        results = {}
        
        # Enrollment rate trends
        if 'enrollment_date' in df.columns:
            enrollment_by_date = df.groupby(
                pd.Grouper(key='enrollment_date', freq='D')
            ).size()
            
            results['daily_enrollment'] = {
                'mean': float(enrollment_by_date.mean()),
                'std': float(enrollment_by_date.std()),
                'min': float(enrollment_by_date.min()),
                'max': float(enrollment_by_date.max()),
                'trend': self._detect_trend(enrollment_by_date.values)
            }
        
        # Age group analysis
        if 'age' in df.columns:
            age_bins = [0, 18, 30, 45, 60, 100]
            age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
            
            results['age_distribution'] = df['age_group'].value_counts().to_dict()
        
        # Gender distribution
        if 'gender' in df.columns:
            results['gender_distribution'] = df['gender'].value_counts().to_dict()
        
        # Coverage analysis
        results['total_enrollments'] = len(df)
        
        return results
    
    def analyze_updates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze update-specific patterns"""
        results = {}
        
        # Update frequency
        if 'update_count' in df.columns:
            results['update_statistics'] = {
                'mean_updates': float(df['update_count'].mean()),
                'median_updates': float(df['update_count'].median()),
                'max_updates': int(df['update_count'].max()),
                'users_with_updates': int((df['update_count'] > 0).sum())
            }
        
        # Update types analysis
        update_type_cols = [col for col in df.columns if 'update_type' in col.lower()]
        if update_type_cols:
            results['update_types'] = df[update_type_cols[0]].value_counts().to_dict()
        
        # Time between updates
        if 'last_update_date' in df.columns and 'enrollment_date' in df.columns:
            df['days_since_enrollment'] = (
                df['last_update_date'] - df['enrollment_date']
            ).dt.days
            
            results['update_timing'] = {
                'mean_days_to_update': float(df['days_since_enrollment'].mean()),
                'median_days_to_update': float(df['days_since_enrollment'].median())
            }
        
        return results
    
    def analyze_geographic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic patterns"""
        results = {}
        
        # State-wise distribution
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            results['state_distribution'] = state_counts.to_dict()
            results['top_5_states'] = state_counts.head(5).to_dict()
            results['bottom_5_states'] = state_counts.tail(5).to_dict()
        
        # District-wise distribution
        if 'district' in df.columns:
            district_counts = df['district'].value_counts()
            results['district_distribution'] = district_counts.head(50).to_dict()
        
        # Urban vs Rural
        if 'area_type' in df.columns:
            results['urban_rural_split'] = df['area_type'].value_counts().to_dict()
        
        return results
    
    def analyze_demographic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demographic patterns"""
        results = {}
        
        # Age statistics
        if 'age' in df.columns:
            results['age_statistics'] = {
                'mean': float(df['age'].mean()),
                'median': float(df['age'].median()),
                'std': float(df['age'].std()),
                'min': int(df['age'].min()),
                'max': int(df['age'].max())
            }
        
        # Gender by age group
        if 'gender' in df.columns and 'age_group' in df.columns:
            gender_age = pd.crosstab(df['age_group'], df['gender'], normalize='index')
            results['gender_by_age_group'] = gender_age.to_dict()
        
        # Education level (if available)
        if 'education' in df.columns:
            results['education_distribution'] = df['education'].value_counts().to_dict()
        
        return results
    
    def analyze_temporal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        results = {}
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        for date_col in date_cols:
            # Monthly trends
            monthly = df.groupby(df[date_col].dt.to_period('M')).size()
            results[f'{date_col}_monthly_trend'] = {
                'data': monthly.to_dict(),
                'trend': self._detect_trend(monthly.values)
            }
            
            # Day of week patterns
            dow = df.groupby(df[date_col].dt.dayofweek).size()
            results[f'{date_col}_day_of_week'] = dow.to_dict()
            
            # Seasonal patterns
            quarterly = df.groupby(df[date_col].dt.quarter).size()
            results[f'{date_col}_quarterly'] = quarterly.to_dict()
        
        return results
    
    def perform_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering to find user segments"""
        results = {}
        
        # Select numeric features for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            self.logger.warning("Not enough numeric features for clustering")
            return results
        
        # Prepare data
        X = df[numeric_cols].fillna(0)
        
        # Perform clustering
        if self.clustering_method == 'kmeans':
            model = KMeans(n_clusters=self.n_clusters, random_state=42)
            clusters = model.fit_predict(X)
            
            results['cluster_labels'] = clusters.tolist()
            results['cluster_centers'] = model.cluster_centers_.tolist()
            results['n_clusters'] = self.n_clusters
            
            # Analyze each cluster
            df['cluster'] = clusters
            for i in range(self.n_clusters):
                cluster_data = df[df['cluster'] == i]
                results[f'cluster_{i}_size'] = len(cluster_data)
                results[f'cluster_{i}_characteristics'] = {
                    col: float(cluster_data[col].mean()) 
                    for col in numeric_cols if col in cluster_data.columns
                }
        
        return results
    
    def _detect_trend(self, values: np.ndarray) -> str:
        """Detect trend in time series data"""
        if len(values) < 3:
            return 'insufficient_data'
        
        try:
            if self.trend_method == 'mann_kendall':
                # Mann-Kendall trend test
                n = len(values)
                s = 0
                for i in range(n-1):
                    for j in range(i+1, n):
                        s += np.sign(values[j] - values[i])
                
                # Simplified interpretation
                if s > 0:
                    return 'increasing'
                elif s < 0:
                    return 'decreasing'
                else:
                    return 'stable'
            else:
                # Linear regression slope
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                
                if slope > 0.01:
                    return 'increasing'
                elif slope < -0.01:
                    return 'decreasing'
                else:
                    return 'stable'
        except:
            return 'unknown'
