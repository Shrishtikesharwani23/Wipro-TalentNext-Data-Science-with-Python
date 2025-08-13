"""Use-Case:House Price Prediction
Dataset: melb data.csv
The dataset can be downloaded melb data.csv | Kaggle

Perform the following tasks:
1. Load the data in dataframe (Pandas)
2. Handle inappropriate data
3. Handle the missing data
4. Handle the categorical data"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("melb_data.csv")
print("Initial Data:")
print(df.head())
print(df.info())

df.drop_duplicates(inplace=True) 

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

cols_to_drop = ['Address', 'SellerG', 'Postcode']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

threshold = len(df) * 0.3
df.dropna(axis=1, thresh=threshold, inplace=True)
          
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])

print("\nCleaned Data:")
print(df.head())
print(df.info())
