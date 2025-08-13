"""Q1)Perform Data Preprocessing on melb_data.csv dataset with statistical perspective.
The dataset can be downloaded from https://www.kaggle.com/datasets/gunjanpathak/melb-data?
resource=download"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Load Dataset
df = pd.read_csv("melb_data.csv")

print("=== Initial Data Overview ===")
print("Shape:", df.shape)
print(df.info())
print(df.head())

# 2. Statistical Summary
print("\n=== Numerical Summary ===")
print(df.describe().T)

print("\n=== Categorical Summary ===")
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head())

# 3. Handle Inappropriate Data
df.drop_duplicates(inplace=True)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

irrelevant_cols = ['Address', 'SellerG', 'Postcode']
df.drop(columns=irrelevant_cols, inplace=True, errors='ignore')

# 4. Define Features & Target
X = df.drop(columns=['Price'])  
y = df['Price']  

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 5. Create Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 6. Apply Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n=== Processed Feature Matrix ===")
print("X_train shape:", X_train_processed.shape)
print("X_test shape:", X_test_processed.shape)
