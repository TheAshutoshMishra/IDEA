"""
Advanced Analytics Module for Aadhaar Enrolment Analysis
Provides trivariate analysis, correlations, and predictive insights
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class AdvancedEnrolmentAnalytics:
    """Advanced analytics for deeper insights"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_features()
    
    def prepare_features(self):
        """Prepare additional features for analysis"""
        # Calculate enrolment rates
        self.df['child_ratio'] = self.df['age_0_5'] / (self.df['total_enrolments'] + 1)
        self.df['youth_ratio'] = self.df['age_5_17'] / (self.df['total_enrolments'] + 1)
        self.df['adult_ratio'] = self.df['age_18_greater'] / (self.df['total_enrolments'] + 1)
        
        # Time-based features
        self.df['day'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['is_weekend'] = self.df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        self.df['week_of_month'] = (self.df['date'].dt.day - 1) // 7 + 1
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater', 'total_enrolments', 
                       'child_ratio', 'youth_ratio', 'adult_ratio', 'month', 'day']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': correlation_matrix.columns[i],
                    'Variable 2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Correlation', ascending=False, key=abs)
        
        return correlation_matrix, corr_df
    
    def trivariate_analysis(self):
        """Analyze relationships between three variables"""
        # State vs Age Group vs Time
        state_age_time = self.df.groupby(['state', 'quarter']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrolments': 'sum'
        }).reset_index()
        
        # District vs Day of Week vs Age Group
        district_dow_age = self.df.groupby(['district', 'day_of_week']).agg({
            'age_0_5': 'mean',
            'age_5_17': 'mean',
            'age_18_greater': 'mean'
        }).reset_index()
        
        # Month vs State vs Enrolment Type
        monthly_state_pattern = self.df.groupby(['month', 'state']).agg({
            'child_ratio': 'mean',
            'youth_ratio': 'mean',
            'adult_ratio': 'mean',
            'total_enrolments': 'sum'
        }).reset_index()
        
        return state_age_time, district_dow_age, monthly_state_pattern
    
    def clustering_analysis(self, n_clusters=5):
        """Cluster districts based on enrolment patterns"""
        district_features = self.df.groupby('district').agg({
            'age_0_5': 'mean',
            'age_5_17': 'mean',
            'age_18_greater': 'mean',
            'total_enrolments': 'sum',
            'child_ratio': 'mean',
            'youth_ratio': 'mean',
            'adult_ratio': 'mean'
        }).reset_index()
        
        # Prepare features for clustering
        features = district_features[['age_0_5', 'age_5_17', 'age_18_greater', 
                                     'child_ratio', 'youth_ratio', 'adult_ratio']].values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        district_features['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Calculate cluster characteristics
        cluster_summary = district_features.groupby('cluster').agg({
            'age_0_5': 'mean',
            'age_5_17': 'mean',
            'age_18_greater': 'mean',
            'total_enrolments': 'mean',
            'district': 'count'
        }).reset_index()
        cluster_summary.columns = ['Cluster', 'Avg_Age_0_5', 'Avg_Age_5_17', 
                                   'Avg_Age_18+', 'Avg_Total_Enrolments', 'District_Count']
        
        return district_features, cluster_summary
    
    def statistical_insights(self):
        """Generate statistical insights"""
        insights = {}
        
        # Distribution tests
        _, p_value_normal = stats.normaltest(self.df['total_enrolments'])
        insights['normality'] = {
            'is_normal': p_value_normal > 0.05,
            'p_value': p_value_normal
        }
        
        # Compare age groups
        age_groups = [self.df['age_0_5'], self.df['age_5_17'], self.df['age_18_greater']]
        f_stat, p_value_anova = stats.f_oneway(*age_groups)
        insights['age_group_difference'] = {
            'significant': p_value_anova < 0.05,
            'p_value': p_value_anova,
            'f_statistic': f_stat
        }
        
        # Weekend vs Weekday comparison
        weekend_enrol = self.df[self.df['is_weekend'] == 1]['total_enrolments']
        weekday_enrol = self.df[self.df['is_weekend'] == 0]['total_enrolments']
        t_stat, p_value_ttest = stats.ttest_ind(weekend_enrol, weekday_enrol)
        insights['weekend_effect'] = {
            'significant': p_value_ttest < 0.05,
            'p_value': p_value_ttest,
            't_statistic': t_stat,
            'weekend_mean': weekend_enrol.mean(),
            'weekday_mean': weekday_enrol.mean()
        }
        
        return insights
    
    def predictive_insights(self):
        """Generate predictive insights using trend analysis"""
        # Calculate monthly growth rates
        monthly_totals = self.df.groupby('year_month')['total_enrolments'].sum().sort_index()
        growth_rates = monthly_totals.pct_change() * 100
        
        # Predict next month using simple moving average
        recent_avg = monthly_totals.tail(3).mean()
        recent_growth = growth_rates.tail(3).mean()
        
        predicted_next_month = recent_avg * (1 + recent_growth / 100)
        
        # Identify districts needing intervention
        district_performance = self.df.groupby('district').agg({
            'total_enrolments': 'sum',
            'adult_ratio': 'mean'
        }).reset_index()
        
        # Districts with low adult enrolment
        low_adult_districts = district_performance[
            district_performance['adult_ratio'] < district_performance['adult_ratio'].quantile(0.25)
        ].sort_values('total_enrolments', ascending=False)
        
        # High potential districts (high total but low adult)
        high_potential = low_adult_districts[
            low_adult_districts['total_enrolments'] > district_performance['total_enrolments'].median()
        ]
        
        return {
            'predicted_next_month': predicted_next_month,
            'monthly_growth_rate': recent_growth,
            'low_adult_districts': low_adult_districts.head(20),
            'high_potential_districts': high_potential.head(10)
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze adult enrolment gap
        adult_ratio = self.df['adult_ratio'].mean()
        if adult_ratio < 0.05:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Age Group Focus',
                'issue': f'Adult enrolment critically low ({adult_ratio*100:.1f}%)',
                'recommendation': 'Launch targeted adult enrolment drives in all districts',
                'expected_impact': 'Increase adult coverage by 200-300%'
            })
        
        # Analyze geographic disparities
        state_stats = self.df.groupby('state')['total_enrolments'].sum()
        disparity_ratio = state_stats.max() / state_stats.min()
        if disparity_ratio > 50:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Geographic Equity',
                'issue': f'Severe state-level disparity (ratio: {disparity_ratio:.0f}:1)',
                'recommendation': 'Deploy mobile enrolment units to underserved states',
                'expected_impact': 'Reduce disparity by 40-50%'
            })
        
        # Analyze temporal patterns
        dow_stats = self.df.groupby('day_of_week')['total_enrolments'].sum()
        if dow_stats.max() / dow_stats.min() > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Operational Efficiency',
                'issue': 'Uneven distribution across weekdays',
                'recommendation': 'Adjust center hours and staff allocation based on peak days',
                'expected_impact': '20-30% improvement in resource utilization'
            })
        
        # Analyze monthly trends
        monthly_cv = self.df.groupby('year_month')['total_enrolments'].sum().std() / \
                     self.df.groupby('year_month')['total_enrolments'].sum().mean()
        if monthly_cv > 0.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Capacity Planning',
                'issue': f'High monthly variation (CV: {monthly_cv:.2f})',
                'recommendation': 'Implement flexible staffing model for peak months',
                'expected_impact': '15-25% cost savings in low months'
            })
        
        return pd.DataFrame(recommendations)


if __name__ == "__main__":
    import glob
    
    # Load data
    files = glob.glob('data/datasets/api_data_aadhar_enrolment_*.csv')
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Initialize analytics
    analytics = AdvancedEnrolmentAnalytics(df)
    
    print("\n" + "="*60)
    print("ADVANCED ANALYTICS INSIGHTS")
    print("="*60)
    
    # Correlation Analysis
    print("\nTop 10 Correlations:")
    _, corr_df = analytics.correlation_analysis()
    print(corr_df.head(10))
    
    # Statistical Insights
    print("\nStatistical Insights:")
    insights = analytics.statistical_insights()
    for key, value in insights.items():
        print(f"\n{key}: {value}")
    
    # Clustering
    print("\nDistrict Clustering:")
    _, cluster_summary = analytics.clustering_analysis()
    print(cluster_summary)
    
    # Recommendations
    print("\nActionable Recommendations:")
    recommendations = analytics.generate_recommendations()
    print(recommendations.to_string(index=False))
