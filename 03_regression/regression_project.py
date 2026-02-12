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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
os.makedirs('results', exist_ok=True)

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
print(['RoomsPerPerson', 'BedroomRatio', 'PeoplePerRoom'])
print(f"\nFinal feature count: {df.shape[1] - 1}")  # -1 for target column


# TRAIN/TEST SPLIT

# Separate features from target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n")
print("TRAIN/TEST SPLIT")
print("\n")
print(f"Training set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")
print(f"Train price mean: ${y_train.mean()*100:.2f}k")
print(f"Test price mean: ${y_test.mean()*100:.2f}k")

# Feature Scaling
# Features on vastly different scales (population: thousands, latitude: tens)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete!!")


# MODEL SELECTION

# Will train 2 models and compare:

# 1. LINEAR REGRESSION
#    Pros: Fast, interpretable, extrapolates
#    Cons: Assumes linear relationship
#    Use case: Good baseline

# 2. RANDOM FOREST REGRESSOR
#    Pros: Handles non-linearity, feature interactions
#    Cons: Doesn't extrapolate, less interpretable
#    Use case: Usually better RMSE

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

print("\n")
print("MODEL TRAINING")
print("\n")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Store
    results[name] = {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    print(f"{name} trained!!")


# MODEL EVALUATION

for name, result in results.items():
    print("\n")
    print(f"{name} EVALUATION")
    print("\n")
    
    # Calculating metrics
    train_mae = mean_absolute_error(y_train, result['y_pred_train'])
    test_mae = mean_absolute_error(y_test, result['y_pred_test'])
    train_rmse = np.sqrt(mean_squared_error(y_train, result['y_pred_train']))
    test_rmse = np.sqrt(mean_squared_error(y_test, result['y_pred_test']))
    train_r2 = r2_score(y_train, result['y_pred_train'])
    test_r2 = r2_score(y_test, result['y_pred_test'])
    
    results[name]['metrics'] = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    

    print(f"Train MAE:  ${train_mae*100:.2f}k")
    print(f"Test MAE:   ${test_mae*100:.2f}k")
    print(f"   On average, predictions are off by ${test_mae*100:.2f}k")
    
    print(f"\nTrain RMSE: ${train_rmse*100:.2f}k")
    print(f"Test RMSE:  ${test_rmse*100:.2f}k")
    print(f"   RMSE penalizes large errors more than MAE")
    
    print(f"\nTrain R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2:.4f}")
    print(f"   Model explains {test_r2*100:.1f}% of price variance")
    
    # Checking for overfitting
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.1:
        print(f"\nOverfitting detected (R² gap: {r2_gap:.4f})")
    else:
        print(f"\nGood generalization (R² gap: {r2_gap:.4f})")



# VISUALIZATIONS

best_model_name = max(results, key=lambda x: results[x]['metrics']['test_r2'])
best_result = results[best_model_name]

print("\n")
print(f"BEST MODEL: {best_model_name}")
print("\n")

# Creating plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot1: Predictions vs Actual
axes[0].scatter(y_test, best_result['y_pred_test'], alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($100k)')
axes[0].set_ylabel('Predicted Price ($100k)')
axes[0].set_title(f'{best_model_name}\nPredictions vs Actual')
axes[0].grid(True, alpha=0.3)

# Plot2: Residuals
residuals = y_test - best_result['y_pred_test']
axes[1].scatter(best_result['y_pred_test'], residuals, alpha=0.5, s=20)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price ($100k)')
axes[1].set_ylabel('Residuals (Actual - Predicted)')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/prediction_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Prediction plot saved to results/prediction_plot.png")


# SAVING REPORT

with open('results/evaluation_report.txt', 'w') as f:
    f.write("REGRESSION PROJECT EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Test MAE:  ${results[best_model_name]['metrics']['test_mae']*100:.2f}k\n")
    f.write(f"Test RMSE: ${results[best_model_name]['metrics']['test_rmse']*100:.2f}k\n")
    f.write(f"Test R²:   {results[best_model_name]['metrics']['test_r2']:.4f}\n\n")
    
    f.write("INTERPRETATION:\n")
    f.write(f"- Predictions are off by ${results[best_model_name]['metrics']['test_mae']*100:.2f}k on average (MAE)\n")
    f.write(f"- Model explains {results[best_model_name]['metrics']['test_r2']*100:.1f}% of price variation (R²)\n")
    f.write(f"- RMSE of ${results[best_model_name]['metrics']['test_rmse']*100:.2f}k penalizes large errors\n\n")
    
    f.write("=" * 60 + "\n")
    f.write("BOTH MODELS COMPARISON\n")
    f.write("=" * 60 + "\n\n")
    
    for name, result in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Test MAE:  ${result['metrics']['test_mae']*100:.2f}k\n")
        f.write(f"  Test RMSE: ${result['metrics']['test_rmse']*100:.2f}k\n")
        f.write(f"  Test R²:   {result['metrics']['test_r2']:.4f}\n")
        f.write(f"  Overfit Gap: {result['metrics']['train_r2'] - result['metrics']['test_r2']:.4f}\n\n")

print("Report saved to results/evaluation_report.txt")
print("\n")
print("REGRESSION PROJECT COMPLETE!!")