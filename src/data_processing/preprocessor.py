"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional


class DataPreprocessor:
    """Preprocessor for Aadhaar enrollment and update data"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load processing parameters
        self.chunk_size = config['data_processing']['chunk_size']
        self.missing_threshold = config['data_processing']['missing_threshold']
        self.outlier_method = config['data_processing']['outlier_method']
        
    def process(self, data_path: str) -> pd.DataFrame:
        """
        Main processing pipeline
        
        Args:
            data_path: Path to raw data file
            
        Returns:
            Processed DataFrame
        """
        self.logger.info(f"Processing data from: {data_path}")
        
        # Load data
        df = self._load_data(data_path)
        
        # Clean data
        df = self._clean_data(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Save processed data
        self._save_processed_data(df, data_path)
        
        self.logger.info(f"Processing complete. Shape: {df.shape}")
        return df
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path, chunksize=None)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            self.logger.info(f"Loaded {len(df)} records from {data_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data - remove duplicates, fix dtypes, etc."""
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicates")
        
        # Strip whitespace from string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
        
        # Convert date columns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Drop columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index
        df = df.drop(columns=cols_to_drop)
        self.logger.info(f"Dropped {len(cols_to_drop)} columns with >{self.missing_threshold*100}% missing")
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from numeric columns"""
        if self.outlier_method == 'IQR':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                initial_rows = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = initial_rows - len(df)
                if removed > 0:
                    self.logger.info(f"Removed {removed} outliers from {col}")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        # Time-based features
        if self.config['features']['time_features']:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            for col in date_cols:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
        
        # Add more feature engineering based on domain knowledge
        # This will be customized based on actual UIDAI data structure
        
        return df
    
    def _save_processed_data(self, df: pd.DataFrame, original_path: str):
        """Save processed data"""
        output_dir = Path(self.config['paths']['processed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(original_path).stem + '_processed.parquet'
        output_path = output_dir / filename
        
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved processed data to: {output_path}")
