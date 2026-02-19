import pandas as pd
from datetime import datetime
import os
import time

# Read the CSV file
df = pd.read_csv('startup data.csv')

# Extract year from founded_at column
# Handle various date formats
def extract_year(date_str):
    if pd.isna(date_str) or date_str == '':
        return None
    
    try:
        # Try to parse the date string
        # pandas.to_datetime can handle multiple formats
        date_obj = pd.to_datetime(date_str)
        return date_obj.year
    except:
        return None

# Apply the function to extract year
df['founded_year'] = df['founded_at'].apply(extract_year)

# Save the modified CSV with a backup
backup_file = 'startup data_backup.csv'
if os.path.exists('startup data.csv'):
    if os.path.exists(backup_file):
        os.remove(backup_file)
    os.rename('startup data.csv', backup_file)
    time.sleep(0.5)

df.to_csv('startup data.csv', index=False)

print("Year extraction complete!")
print(f"Total rows processed: {len(df)}")
print(f"Rows with extracted year: {df['founded_year'].notna().sum()}")
print("\nFirst few entries:")
print(df[['name', 'founded_at', 'founded_year']].head(10))
