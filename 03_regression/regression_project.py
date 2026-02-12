"""
California Housing Price Prediction - Regression Pipeline
Author: Perla Thebian
Purpose: Demonstrate ML workflow for continuous value prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load data (built into sklearn)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target  # Target in $100,000s

print("DATASET OVERVIEW")
print("\n")
print(f"Shape: {df.shape}")
print(f"\nFeature names: {housing.feature_names}")
print(f"\nFirst 5 rows:")
print(df.head())

print("\n")
print("COLUMN INFO")
print("\n")
print(df.info())

print("\n")
print("MISSING VALUES")
print("\n")
print(df.isnull().sum())

print("\n")
print("STATISTICAL SUMMARY")
print("\n")
print(df.describe())

print("\n")
print("TARGET DISTRIBUTION (Price in $100k)")
print("\n")
print(df['PRICE'].describe())


# DATA QUALITY CHECK

# Check for missing values
print("\nMissing values:", df.isnull().sum().sum())

# Check for outliers using 3-sigma rule
print("\n")
print("OUTLIER DETECTION")
print("\n")

for col in df.columns:
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")


# FEATURE ENGINEERING

# Create derived features based on domain knowledge
# Ratios often more informative than raw counts

# Feature 1: Rooms per person (housing spaciousness)
# More rooms per person = more spacious living
df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup']

# Feature 2: Bedroom ratio (what % of rooms are bedrooms)
# Higher ratio might indicate family homes vs apartments
df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']

# Feature 3: People per room (crowd indicator)
# Higher = more crowded living conditions
df['PeoplePerRoom'] = df['AveOccup'] / df['AveRooms']

print("\n")
print("NEW FEATURES CREATED")
print("\n")
print(['PopulationPerHousehold', 'BedroomRatio', 'PeoplePerRoom'])
print(f"\nFinal feature count: {df.shape[1] - 1}")  # -1 for target column