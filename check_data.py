import pandas as pd
import glob

files = glob.glob('api_data_aadhar_enrolment_*.csv')
print(f'Total files: {len(files)}\n')

total = 0
for f in files:
    df = pd.read_csv(f)
    print(f'{f}: {len(df):,} rows')
    total += len(df)

print(f'\nTotal records across all files: {total:,}')

# Load one file to check unique states and districts
df = pd.read_csv(files[0])
print(f'\nUnique states: {df["state"].nunique()}')
print(f'Unique districts: {df["district"].nunique()}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
