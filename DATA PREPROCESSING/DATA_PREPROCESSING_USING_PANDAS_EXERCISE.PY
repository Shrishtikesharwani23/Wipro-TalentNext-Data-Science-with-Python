"""Q2)Perform Data Preprocessing on melb_data.csv dataset with statistical perspective.
The dataset can be downloaded from https://www.kaggle.com/datasets/gunjanpathak/melb-data?
resource=download"""

import pandas as pd
import numpy as np

# 1. Load Dataset
df = pd.read_csv("melb_data.csv")

print("=== Initial Data Overview ===")
print("Shape:", df.shape)
print(df.info())
print(df.head())

# 2. Statistical Summary
print("\n=== Numerical Features Summary ===")
print(df.describe().T)

print("\n=== Categorical Features Overview ===")
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {df[col].nunique()}")
    print(df[col].value_counts().head())

# 3. Handle Inappropriate Data
df.drop_duplicates(inplace=True)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

irrelevant_cols = ['Address', 'SellerG', 'Postcode']
df.drop(columns=irrelevant_cols, inplace=True, errors='ignore')

# 4. Missing Data Analysis & Handling
print("\n=== Missing Data Count ===")
missing_count = df.isnull().sum()
print(missing_count[missing_count > 0])

threshold = len(df) * 0.3
df.dropna(axis=1, thresh=threshold, inplace=True)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 5. Outlier Detection (IQR Method)
print("\n=== Outlier Count (IQR method) ===")
for col in num_cols:
    if col in df.columns:  
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        print(f"{col}: {outliers} outliers detected")

# 6. Encode Categorical Variables (Pure Pandas)
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes  

# 7. Final Data Overview
print("\n=== Cleaned Data Overview ===")
print("Shape:", df.shape)
print(df.head())
print(df.describe().T)
print(df.info())
