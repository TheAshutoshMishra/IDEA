"""
AadhaarInsight360 - Main Analysis Pipeline
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_processing.preprocessor import DataPreprocessor
from pattern_analysis.pattern_detector import PatternDetector
from anomaly_detection.anomaly_detector import AnomalyDetector
from predictive_models.forecaster import Forecaster
from utils.logger import setup_logger
from utils.report_generator import ReportGenerator


class AadhaarInsight360:
    """Main class for Aadhaar data analysis"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the analysis pipeline"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logger
        self.logger = setup_logger(
            self.config['logging']['level'],
            self.config['logging']['file']
        )
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.forecaster = Forecaster(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        self.logger.info("AadhaarInsight360 initialized successfully")
    
    def run_full_analysis(self, data_path):
        """
        Run complete analysis pipeline
        
        Args:
            data_path: Path to raw data file
        """
        self.logger.info(f"Starting full analysis on: {data_path}")
        
        try:
            # Step 1: Data Preprocessing
            self.logger.info("Step 1: Data Preprocessing")
            processed_data = self.preprocessor.process(data_path)
            self.logger.info(f"Processed {len(processed_data)} records")
            
            # Step 2: Pattern Analysis
            self.logger.info("Step 2: Pattern Analysis")
            patterns = self.pattern_detector.detect_patterns(processed_data)
            self.logger.info(f"Detected {len(patterns)} patterns")
            
            # Step 3: Anomaly Detection
            self.logger.info("Step 3: Anomaly Detection")
            anomalies = self.anomaly_detector.detect_anomalies(processed_data)
            self.logger.info(f"Found {len(anomalies)} anomalies")
            
            # Step 4: Predictive Modeling
            self.logger.info("Step 4: Predictive Modeling")
            predictions = self.forecaster.forecast(processed_data)
            self.logger.info("Predictions generated successfully")
            
            # Step 5: Generate Report
            self.logger.info("Step 5: Generating Report")
            results = {
                'processed_data': processed_data,
                'patterns': patterns,
                'anomalies': anomalies,
                'predictions': predictions,
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'data_source': data_path,
                    'records_processed': len(processed_data)
                }
            }
            
            report_path = self.report_generator.generate_report(results)
            self.logger.info(f"Report generated: {report_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
    
    def analyze_enrollment_patterns(self, data_path):
        """Analyze enrollment-specific patterns"""
        self.logger.info("Analyzing enrollment patterns")
        processed_data = self.preprocessor.process(data_path)
        return self.pattern_detector.analyze_enrollment(processed_data)
    
    def analyze_update_patterns(self, data_path):
        """Analyze update-specific patterns"""
        self.logger.info("Analyzing update patterns")
        processed_data = self.preprocessor.process(data_path)
        return self.pattern_detector.analyze_updates(processed_data)
    
    def detect_fraud(self, data_path):
        """Run fraud detection algorithms"""
        self.logger.info("Running fraud detection")
        processed_data = self.preprocessor.process(data_path)
        return self.anomaly_detector.detect_fraud(processed_data)
    
    def forecast_enrollment(self, data_path, periods=12):
        """Forecast future enrollment"""
        self.logger.info(f"Forecasting enrollment for {periods} periods")
        processed_data = self.preprocessor.process(data_path)
        return self.forecaster.forecast_enrollment(processed_data, periods)


def main():
    """Main entry point"""
    # Initialize
    analyzer = AadhaarInsight360()
    
    # Define data paths
    data_dir = Path('data/raw')
    
    # Check if data exists
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("⚠️  No data files found in data/raw/")
        print("Please place UIDAI datasets in the data/raw/ directory")
        return
    
    # Run analysis on all files
    for data_file in data_dir.glob('*.csv'):
        print(f"\n{'='*60}")
        print(f"Analyzing: {data_file.name}")
        print(f"{'='*60}\n")
        
        results = analyzer.run_full_analysis(str(data_file))
        
        print("\n✅ Analysis completed successfully!")
        print(f"Results saved to: data/outputs/")
    
    print("\n" + "="*60)
    print("🎉 All analyses completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review generated reports in data/outputs/")
    print("2. Launch dashboard: streamlit run dashboard/app.py")
    print("3. Explore notebooks in notebooks/")


if __name__ == "__main__":
    main()
