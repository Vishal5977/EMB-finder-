import pandas as pd

df = pd.read_csv(r'C:\Users\Asus\embroidery-finder\design_database.csv')
codes = sorted(df['design_name'].unique())
print("Total unique designs indexed:", len(codes))
print("Design codes:", codes)