"""
Titanic Survival Prediction - Binary Classification
Author: Perla Thebian
Purpose: Demonstrate clean ML workflow for classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/titanic.csv')

# Check basics
print("DATASET OVERVIEW")
print(f"\nShape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print('\n')
print("FIRST 5 ROWS")
print()
print(df.head())

print('\n')
print("COLUMN INFO")
print()
print(df.info())

print('\n')
print("MISSING VALUES")
print()
print(df.isnull().sum())

print('\n')
print("TARGET DISTRIBUTION")
print()
print(df['Survived'].value_counts())
print(f"Survival rate: {df['Survived'].mean():.2%}")

# Identify feature types
print('\n')
print("FEATURE TYPES")
print()
print(f"Numeric features: {df.select_dtypes(include=['int64', 'float64']).columns.tolist()}")
print(f"Categorical features: {df.select_dtypes(include=['object']).columns.tolist()}")