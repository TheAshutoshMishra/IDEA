import pandas as pd

df = pd.read_csv('data/raw/auth_transactions.csv')
print(f'Total Records: {len(df):,}')
print(f'\nColumns: {list(df.columns)}')
print(f'\nDate Range: {df["timestamp"].min()} to {df["timestamp"].max()}')
print(f'\nRegions: {df["region"].nunique()}')
print(f'\nAuth Types: {df["auth_type"].unique()}')
print(f'\nSuccess Rate: {(df["result"]=="Success").mean()*100:.1f}%')
print(f'\nAnomalies: {df["is_anomaly"].sum():,}')
print(f'\nFailure Reasons:')
print(df['failure_reason'].value_counts())
