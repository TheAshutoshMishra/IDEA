"""
Aadhaar Enrolment Data Analysis
Comprehensive analysis of UIDAI enrolment dataset
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AadhaarEnrolmentAnalyzer:
    """Analyze Aadhaar enrolment patterns from UIDAI dataset"""
    
    def __init__(self, file_pattern='data/datasets/api_data_aadhar_enrolment_*.csv'):
        """
        Initialize analyzer with enrolment data files
        
        Args:
            file_pattern: Glob pattern to match CSV files
        """
        self.file_pattern = file_pattern
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and merge all enrolment CSV files"""
        print("📊 Loading Aadhaar Enrolment Dataset...")
        
        files = glob.glob(self.file_pattern)
        if not files:
            raise FileNotFoundError(f"No files found matching: {self.file_pattern}")
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  ✓ Loaded {file}: {len(df):,} records")
        
        # Combine all files
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d-%m-%Y')
        
        # Add derived columns
        self.df['total_enrolments'] = (
            self.df['age_0_5'] + 
            self.df['age_5_17'] + 
            self.df['age_18_greater']
        )
        
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['week'] = self.df['date'].dt.isocalendar().week
        
        print(f"\n✅ Total dataset loaded: {len(self.df):,} records")
        print(f"📅 Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"🗺️  States: {self.df['state'].nunique()}, Districts: {self.df['district'].nunique()}")
        
    def get_summary_statistics(self):
        """Get comprehensive summary statistics"""
        print("\n" + "="*60)
        print("📈 SUMMARY STATISTICS")
        print("="*60)
        
        stats = {
            'Total Records': f"{len(self.df):,}",
            'Date Range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}",
            'Total Days': self.df['date'].nunique(),
            'States': self.df['state'].nunique(),
            'Districts': self.df['district'].nunique(),
            'PIN Codes': self.df['pincode'].nunique(),
            'Total Enrolments': f"{self.df['total_enrolments'].sum():,}",
            'Avg Enrolments/Day': f"{self.df.groupby('date')['total_enrolments'].sum().mean():.0f}",
            'Age 0-5 Total': f"{self.df['age_0_5'].sum():,}",
            'Age 5-17 Total': f"{self.df['age_5_17'].sum():,}",
            'Age 18+ Total': f"{self.df['age_18_greater'].sum():,}",
        }
        
        for key, value in stats.items():
            print(f"{key:.<40} {value:>18}")
        
        return stats
    
    def analyze_age_distribution(self):
        """Analyze age group distribution"""
        print("\n" + "="*60)
        print("👥 AGE GROUP DISTRIBUTION")
        print("="*60)
        
        age_totals = {
            'Age 0-5': self.df['age_0_5'].sum(),
            'Age 5-17': self.df['age_5_17'].sum(),
            'Age 18+': self.df['age_18_greater'].sum()
        }
        
        total = sum(age_totals.values())
        
        for age_group, count in age_totals.items():
            percentage = (count / total) * 100
            print(f"{age_group:.<30} {count:>12,} ({percentage:>5.2f}%)")
        
        return age_totals
    
    def analyze_top_states(self, top_n=10):
        """Analyze top performing states"""
        print("\n" + "="*60)
        print(f"🏆 TOP {top_n} STATES BY ENROLMENT")
        print("="*60)
        
        state_stats = self.df.groupby('state').agg({
            'total_enrolments': 'sum',
            'district': 'nunique',
            'pincode': 'nunique'
        }).sort_values('total_enrolments', ascending=False)
        
        state_stats.columns = ['Total_Enrolments', 'Districts', 'PIN_Codes']
        
        print(f"\n{'State':<25} {'Enrolments':>12} {'Districts':>10} {'PINs':>8}")
        print("-" * 60)
        
        for state, row in state_stats.head(top_n).iterrows():
            print(f"{state:<25} {row['Total_Enrolments']:>12,.0f} {row['Districts']:>10} {row['PIN_Codes']:>8}")
        
        return state_stats
    
    def analyze_top_districts(self, top_n=15):
        """Analyze top performing districts"""
        print("\n" + "="*60)
        print(f"🏅 TOP {top_n} DISTRICTS BY ENROLMENT")
        print("="*60)
        
        district_stats = self.df.groupby(['state', 'district']).agg({
            'total_enrolments': 'sum',
            'pincode': 'nunique',
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).sort_values('total_enrolments', ascending=False)
        
        print(f"\n{'District':<30} {'State':<20} {'Total':>10}")
        print("-" * 65)
        
        for (state, district), row in district_stats.head(top_n).iterrows():
            print(f"{district:<30} {state:<20} {row['total_enrolments']:>10,.0f}")
        
        return district_stats
    
    def analyze_temporal_trends(self):
        """Analyze temporal trends"""
        print("\n" + "="*60)
        print("📅 TEMPORAL TRENDS")
        print("="*60)
        
        # Monthly trends
        monthly = self.df.groupby('year_month')['total_enrolments'].sum()
        print("\nMonthly Enrolments:")
        for month, count in monthly.items():
            print(f"  {month}: {count:>12,}")
        
        # Day of week trends
        print("\nEnrolments by Day of Week:")
        dow = self.df.groupby('day_of_week')['total_enrolments'].sum()
        dow = dow.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        for day, count in dow.items():
            print(f"  {day:<12}: {count:>12,}")
        
        return monthly, dow
    
    def identify_low_performing_districts(self, bottom_n=20):
        """Identify districts with lowest enrolment"""
        print("\n" + "="*60)
        print(f"⚠️  BOTTOM {bottom_n} DISTRICTS (Need Intervention)")
        print("="*60)
        
        district_stats = self.df.groupby(['state', 'district']).agg({
            'total_enrolments': 'sum',
            'date': 'count'
        }).sort_values('total_enrolments', ascending=True)
        
        district_stats.columns = ['Total_Enrolments', 'Records']
        
        print(f"\n{'District':<30} {'State':<20} {'Total':>10}")
        print("-" * 65)
        
        for (state, district), row in district_stats.head(bottom_n).iterrows():
            print(f"{district:<30} {state:<20} {row['Total_Enrolments']:>10,.0f}")
        
        return district_stats
    
    def detect_anomalies(self):
        """Detect unusual patterns and anomalies"""
        print("\n" + "="*60)
        print("🔍 ANOMALY DETECTION")
        print("="*60)
        
        # Daily total enrolments
        daily = self.df.groupby('date')['total_enrolments'].sum()
        
        # Statistical thresholds
        mean_daily = daily.mean()
        std_daily = daily.std()
        upper_threshold = mean_daily + 3 * std_daily
        lower_threshold = max(0, mean_daily - 3 * std_daily)
        
        anomalies_high = daily[daily > upper_threshold]
        anomalies_low = daily[daily < lower_threshold]
        
        print(f"\nDaily Enrolment Statistics:")
        print(f"  Mean: {mean_daily:,.0f}")
        print(f"  Std Dev: {std_daily:,.0f}")
        print(f"  Upper Threshold (μ + 3σ): {upper_threshold:,.0f}")
        print(f"  Lower Threshold (μ - 3σ): {lower_threshold:,.0f}")
        
        print(f"\n🔴 High Anomaly Days: {len(anomalies_high)}")
        if len(anomalies_high) > 0:
            for date, count in anomalies_high.head(5).items():
                print(f"  {date.date()}: {count:>12,}")
        
        print(f"\n🔵 Low Anomaly Days: {len(anomalies_low)}")
        if len(anomalies_low) > 0:
            for date, count in anomalies_low.head(5).items():
                print(f"  {date.date()}: {count:>12,}")
        
        return anomalies_high, anomalies_low
    
    def generate_insights(self):
        """Generate actionable insights"""
        print("\n" + "="*60)
        print("💡 KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Age group percentages
        age_totals = {
            'Age 0-5': self.df['age_0_5'].sum(),
            'Age 5-17': self.df['age_5_17'].sum(),
            'Age 18+': self.df['age_18_greater'].sum()
        }
        total = sum(age_totals.values())
        
        dominant_age = max(age_totals, key=age_totals.get)
        weakest_age = min(age_totals, key=age_totals.get)
        
        # State performance
        state_stats = self.df.groupby('state')['total_enrolments'].sum().sort_values(ascending=False)
        top_state = state_stats.index[0]
        bottom_state = state_stats.index[-1]
        
        # Temporal insights
        monthly = self.df.groupby('year_month')['total_enrolments'].sum()
        peak_month = monthly.idxmax()
        low_month = monthly.idxmin()
        
        insights = [
            f"1. Age Group Focus:",
            f"   - {dominant_age} has highest enrolments ({age_totals[dominant_age]:,})",
            f"   - {weakest_age} needs more attention ({age_totals[weakest_age]:,})",
            f"",
            f"2. Geographic Performance:",
            f"   - Best performing state: {top_state} ({state_stats[top_state]:,.0f} enrolments)",
            f"   - Needs intervention: {bottom_state} ({state_stats[bottom_state]:,.0f} enrolments)",
            f"",
            f"3. Temporal Patterns:",
            f"   - Peak month: {peak_month} ({monthly[peak_month]:,.0f} enrolments)",
            f"   - Low month: {low_month} ({monthly[low_month]:,.0f} enrolments)",
            f"",
            f"4. Recommendations:",
            f"   - Launch targeted campaigns in {bottom_state}",
            f"   - Focus on {weakest_age} demographic across all regions",
            f"   - Replicate success strategies from {top_state}",
            f"   - Increase resources during {peak_month} period",
        ]
        
        for insight in insights:
            print(insight)
        
        return insights
    
    def save_processed_data(self, output_file='data/processed/enrolment_processed.csv'):
        """Save processed dataset"""
        self.df.to_csv(output_file, index=False)
        print(f"\n💾 Processed data saved to: {output_file}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🇮🇳 AADHAAR ENROLMENT ANALYSIS SYSTEM")
    print("="*60)
    print("Analyzing UIDAI Enrolment Dataset\n")
    
    # Initialize analyzer
    analyzer = AadhaarEnrolmentAnalyzer()
    
    # Run all analyses
    analyzer.get_summary_statistics()
    analyzer.analyze_age_distribution()
    analyzer.analyze_top_states(top_n=10)
    analyzer.analyze_top_districts(top_n=15)
    analyzer.analyze_temporal_trends()
    analyzer.identify_low_performing_districts(bottom_n=20)
    analyzer.detect_anomalies()
    analyzer.generate_insights()
    
    # Save processed data
    analyzer.save_processed_data()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
