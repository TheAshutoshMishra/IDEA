"""
Quick Demo Script - Test All MVP Components
Run this before the dashboard demo
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data.simulator import AadhaarAuthSimulator
from intelligence.anomaly_detector import RealTimeAnomalyDetector
from intelligence.risk_scorer import RiskScorer

def test_data_generation():
    """Test data simulator"""
    print("\n🔹 Testing Data Generation...")
    
    simulator = AadhaarAuthSimulator()
    # Generate 1000 transactions for testing
    df = simulator.generate_transactions(n_records=1000)
    
    print(f"✅ Generated {len(df):,} transactions")
    print(f"   - Success rate: {(df['result'] == 'Success').mean():.1%}")
    print(f"   - Failure rate: {(df['result'] == 'Failure').mean():.1%}")
    print(f"   - Unique regions: {df['region'].nunique()}")
    print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def test_anomaly_detection(df):
    """Test anomaly detector"""
    print("\n🔹 Testing Anomaly Detection...")
    
    detector = RealTimeAnomalyDetector()
    
    # Train on first 80%
    train_size = int(len(df) * 0.8)
    detector.train(df[:train_size])
    print(f"✅ Trained on {train_size:,} records")
    
    # Detect on last 20%
    results = detector.detect_anomalies(df[train_size:])
    anomalies = results[results['is_anomaly'] == 1]
    
    print(f"✅ Detected {len(anomalies)} anomalies in test set")
    print(f"   - High severity: {len(anomalies[anomalies['severity'] == 'High'])}")
    print(f"   - Medium severity: {len(anomalies[anomalies['severity'] == 'Medium'])}")
    print(f"   - Low severity: {len(anomalies[anomalies['severity'] == 'Low'])}")
    print(f"   - Avg confidence: {anomalies['confidence'].mean():.1f}%")
    
    if len(anomalies) > 0:
        print(f"\n   Example anomaly:")
        example = anomalies.iloc[0]
        print(f"   - Region: {example['region']}")
        print(f"   - Severity: {example['severity']}")
        print(f"   - Reasons: {example['anomaly_reasons']}")
    
    return detector

def test_risk_scoring(df):
    """Test risk scorer"""
    print("\n🔹 Testing Risk Scoring...")
    
    scorer = RiskScorer()
    
    # Prepare training data
    train_size = int(len(df) * 0.8)
    training_data = scorer.prepare_training_data(df[:train_size])
    
    print(f"✅ Prepared {len(training_data):,} training windows")
    
    # Train model
    accuracy = scorer.train(training_data)
    print(f"✅ Model trained")
    
    # Predict on recent data
    test_data = scorer.prepare_training_data(df[train_size:])
    
    if len(test_data) > 0:
        predictions = scorer.predict_risk(test_data)
        
        high_risk = len(predictions[predictions['risk_label'].isin(['High', 'Critical'])])
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"   - High/Critical risk: {high_risk}")
        print(f"   - Average risk score: {predictions['risk_score'].mean():.1f}")
        
        # Predict next window
        next_window = scorer.predict_next_window(df[train_size:])
        print(f"\n   Next hour prediction:")
        print(f"   - Risk score: {next_window['risk_score']:.1f}")
        print(f"   - Risk label: {next_window['risk_label']}")
        print(f"   - Probability: {next_window['risk_probability']:.1%}")
    
    return scorer

def test_full_pipeline():
    """Test complete pipeline"""
    print("\n" + "="*60)
    print("🚀 AadhaarSecure360 MVP - Component Testing")
    print("="*60)
    
    try:
        # Test all components
        df = test_data_generation()
        detector = test_anomaly_detection(df)
        scorer = test_risk_scoring(df)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - MVP READY FOR DEMO")
        print("="*60)
        
        print("\n📊 To launch dashboard:")
        print("   streamlit run dashboard/app_mvp.py")
        
        print("\n🔐 Demo credentials:")
        print("   Admin: admin / admin123")
        print("   Analyst: analyst / analyst123")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
