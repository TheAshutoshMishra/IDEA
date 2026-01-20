"""
Simulated Aadhaar Authentication Data Generator
Generates realistic authentication transaction data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib


class AadhaarAuthSimulator:
    """Generate realistic Aadhaar authentication transactions"""
    
    def __init__(self, seed=42):
        """Initialize simulator with seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Indian states for geographic simulation
        self.states = [
            'Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu', 'Uttar Pradesh',
            'West Bengal', 'Gujarat', 'Rajasthan', 'Kerala', 'Madhya Pradesh',
            'Telangana', 'Andhra Pradesh', 'Punjab', 'Haryana', 'Bihar'
        ]
        
        # Authentication types
        self.auth_types = ['OTP', 'Biometric', 'Both']
        
        # Failure reasons
        self.failure_reasons = [
            'Invalid_OTP', 'Biometric_Mismatch', 'Timeout', 
            'Network_Error', 'Invalid_Request', 'Aadhaar_Locked',
            'Exceeds_Retry_Limit', 'Device_Error'
        ]
        
    def generate_transactions(self, n_records=10000, start_date=None, end_date=None):
        """
        Generate authentication transaction data
        
        Args:
            n_records: Number of transactions to generate
            start_date: Start datetime (default: 30 days ago)
            end_date: End datetime (default: now)
            
        Returns:
            DataFrame with simulated transactions
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Generate timestamps
        time_range = (end_date - start_date).total_seconds()
        timestamps = [
            start_date + timedelta(seconds=random.random() * time_range)
            for _ in range(n_records)
        ]
        timestamps.sort()
        
        # Generate base data
        data = {
            'timestamp': timestamps,
            'transaction_id': [self._generate_hash(f"txn_{i}") for i in range(n_records)],
            'region': np.random.choice(self.states, n_records),
            'auth_type': np.random.choice(self.auth_types, n_records, p=[0.45, 0.35, 0.20]),
            'device_id': [self._generate_hash(f"device_{np.random.randint(1, 5000)}") 
                         for _ in range(n_records)],
            'operator_id': [self._generate_hash(f"operator_{np.random.randint(1, 500)}") 
                           for _ in range(n_records)]
        }
        
        df = pd.DataFrame(data)
        
        # Generate success/failure with realistic patterns
        df['result'] = self._generate_results(df)
        
        # Add retry counts (higher for failures)
        df['retry_count'] = df['result'].apply(
            lambda x: np.random.poisson(2) if x == 'Failure' else 0
        )
        
        # Response time (failures take longer)
        df['response_time_ms'] = df['result'].apply(
            lambda x: np.random.normal(800, 200) if x == 'Success' 
            else np.random.normal(1500, 400)
        ).clip(100, 5000).astype(int)
        
        # Failure reasons (only for failures)
        df['failure_reason'] = df['result'].apply(
            lambda x: np.random.choice(self.failure_reasons) if x == 'Failure' else None
        )
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Inject anomalies (5% of data)
        df = self._inject_anomalies(df)
        
        return df
    
    def _generate_hash(self, value):
        """Generate consistent hash for IDs"""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _generate_results(self, df):
        """Generate success/failure with realistic patterns"""
        results = []
        
        for idx, row in df.iterrows():
            # Base success rate: 95%
            success_prob = 0.95
            
            # Lower success during off-hours (11 PM - 6 AM)
            hour = row['timestamp'].hour
            if hour >= 23 or hour <= 6:
                success_prob -= 0.10
            
            # Lower success for biometric vs OTP
            if row['auth_type'] == 'Biometric':
                success_prob -= 0.03
            
            # Some regions have lower success
            if row['region'] in ['Bihar', 'Uttar Pradesh']:
                success_prob -= 0.05
            
            # Generate result
            result = 'Success' if np.random.random() < success_prob else 'Failure'
            results.append(result)
        
        return results
    
    def _inject_anomalies(self, df):
        """Inject realistic anomalies into data"""
        n_records = len(df)
        
        # Type 1: Sudden spike in failures (0.5%)
        spike_count = int(n_records * 0.005)
        spike_indices = np.random.choice(n_records, spike_count, replace=False)
        
        # Create a failure cluster
        for idx in spike_indices:
            # Affect nearby transactions (within 10 minutes)
            time_window = df.iloc[idx]['timestamp']
            mask = (df['timestamp'] >= time_window - timedelta(minutes=5)) & \
                   (df['timestamp'] <= time_window + timedelta(minutes=5))
            df.loc[mask, 'result'] = 'Failure'
            df.loc[mask, 'failure_reason'] = 'Network_Error'
        
        # Type 2: Excessive retries from same device (1%)
        excessive_retry_count = int(n_records * 0.01)
        retry_indices = np.random.choice(n_records, excessive_retry_count, replace=False)
        df.loc[retry_indices, 'retry_count'] = np.random.randint(5, 15, excessive_retry_count)
        
        # Type 3: Off-hours spike (0.5%)
        off_hours_mask = (df['hour'] >= 23) | (df['hour'] <= 4)
        off_hours_indices = df[off_hours_mask].sample(frac=0.1).index
        df.loc[off_hours_indices, 'retry_count'] = np.random.randint(3, 10, len(off_hours_indices))
        
        # Mark anomalies
        df['is_anomaly'] = 0
        df.loc[spike_indices, 'is_anomaly'] = 1
        df.loc[retry_indices, 'is_anomaly'] = 1
        df.loc[off_hours_indices, 'is_anomaly'] = 1
        
        return df
    
    def generate_realtime_stream(self, duration_minutes=60, rate_per_minute=100):
        """
        Generate real-time streaming data
        
        Args:
            duration_minutes: How long to simulate
            rate_per_minute: Transactions per minute
            
        Yields:
            Single transaction record
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            # Generate one transaction
            df = self.generate_transactions(n_records=1, 
                                           start_date=datetime.now(),
                                           end_date=datetime.now())
            yield df.iloc[0].to_dict()
            
            # Wait for next transaction
            import time
            time.sleep(60.0 / rate_per_minute)


def main():
    """Demo the simulator"""
    print("🔄 Generating simulated Aadhaar authentication data...")
    
    simulator = AadhaarAuthSimulator()
    
    # Generate 30 days of data
    df = simulator.generate_transactions(
        n_records=50000,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    # Save to CSV
    output_file = 'data/raw/auth_transactions.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Generated {len(df):,} transactions")
    print(f"📁 Saved to: {output_file}")
    
    # Show statistics
    print("\n📊 Data Statistics:")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Success rate: {(df['result'] == 'Success').mean():.2%}")
    print(f"   Failure rate: {(df['result'] == 'Failure').mean():.2%}")
    print(f"   Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.2%})")
    print(f"\n   Authentication types:")
    print(df['auth_type'].value_counts())
    print(f"\n   Top regions:")
    print(df['region'].value_counts().head())
    print(f"\n   Top failure reasons:")
    print(df[df['result'] == 'Failure']['failure_reason'].value_counts())


if __name__ == "__main__":
    main()
