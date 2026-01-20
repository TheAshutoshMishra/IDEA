"""
Report Generation Utilities
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ReportGenerator:
    """Generate analysis reports in various formats"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize report generator"""
        self.config = config
        self.output_dir = Path(config['paths']['outputs'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive report
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate JSON report
        json_path = self.output_dir / f'analysis_report_{timestamp}.json'
        self._generate_json_report(results, json_path)
        
        # Generate summary text report
        txt_path = self.output_dir / f'summary_{timestamp}.txt'
        self._generate_text_summary(results, txt_path)
        
        # Generate Excel report if data is available
        if 'processed_data' in results:
            excel_path = self.output_dir / f'data_export_{timestamp}.xlsx'
            self._generate_excel_report(results, excel_path)
        
        return str(json_path)
    
    def _generate_json_report(self, results: Dict[str, Any], output_path: Path):
        """Generate JSON report"""
        # Remove DataFrame from results for JSON serialization
        json_results = {k: v for k, v in results.items() if k != 'processed_data'}
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def _generate_text_summary(self, results: Dict[str, Any], output_path: Path):
        """Generate human-readable text summary"""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AADHAAR INSIGHT360 - ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            if 'metadata' in results:
                f.write("ANALYSIS METADATA\n")
                f.write("-" * 80 + "\n")
                for key, value in results['metadata'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Patterns
            if 'patterns' in results:
                f.write("DETECTED PATTERNS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Pattern categories: {len(results['patterns'])}\n")
                f.write("\n")
            
            # Anomalies
            if 'anomalies' in results and 'summary' in results['anomalies']:
                f.write("ANOMALY DETECTION\n")
                f.write("-" * 80 + "\n")
                summary = results['anomalies']['summary']
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Predictions
            if 'predictions' in results:
                f.write("PREDICTIVE INSIGHTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Forecast models: {len(results['predictions'])}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
    
    def _generate_excel_report(self, results: Dict[str, Any], output_path: Path):
        """Generate Excel report with multiple sheets"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Processed data
            if 'processed_data' in results:
                df = results['processed_data']
                df.head(10000).to_excel(writer, sheet_name='Sample Data', index=False)
            
            # Patterns summary
            if 'patterns' in results:
                patterns_df = self._flatten_dict_to_df(results['patterns'])
                patterns_df.to_excel(writer, sheet_name='Patterns', index=False)
            
            # Anomalies summary
            if 'anomalies' in results:
                anomalies_df = self._flatten_dict_to_df(results['anomalies'])
                anomalies_df.to_excel(writer, sheet_name='Anomalies', index=False)
    
    def _flatten_dict_to_df(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Flatten nested dictionary to DataFrame"""
        flat_data = []
        
        def flatten(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key))
                else:
                    items.append((new_key, v))
            return items
        
        flat_data = flatten(data)
        return pd.DataFrame(flat_data, columns=['Metric', 'Value'])
