"""
Titanic Survival Prediction - Binary Classification
Author: Perla Thebian
Purpose: Demonstrate clean ML workflow for classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

import os
os.makedirs('results', exist_ok=True)

# Load data
df = pd.read_csv('data/titanic.csv')

# # Check basics
# print("DATASET OVERVIEW")
# print(f"\nShape: {df.shape}")
# print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# print('\n')
# print("FIRST 5 ROWS")
# print()
# print(df.head())

# print('\n')
# print("COLUMN INFO")
# print()
# print(df.info())

# print('\n')
# print("MISSING VALUES")
# print()
# print(df.isnull().sum())

# print('\n')
# print("TARGET DISTRIBUTION")
# print()
# print(df['Survived'].value_counts())
# print(f"Survival rate: {df['Survived'].mean():.2%}")

# # Identify feature types
# print('\n')
# print("FEATURE TYPES")
# print()
# print(f"Numeric features: {df.select_dtypes(include=['int64', 'float64']).columns.tolist()}")
# print(f"Categorical features: {df.select_dtypes(include=['object']).columns.tolist()}")


# DATA CLEANING

# Dropping useless columns
# PassengerId: just an index, no predictive value
# Name: also no predictive valye
# Ticket: random strings, no clear pattern
# Cabin: 687/891 missing values (~77%), too incomplete to use
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

print("\n")
print("AFTER DROPPING USELESS COLUMNS")
print(f"\nRemaining columns: {df.columns.tolist()}")

# Handling missing values
# Age: 177/891 missing values (~19.8%) - fill with median
# Why median: Less affected by outliers compared to mean
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked: 2 missing values (~0.2%) - fill with mode (first element = most common)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\n")
print("AFTER HANDLING MISSING VALUES")
print()
print(df.isnull().sum())


# FEATURE ENGINEERING

# Encode Sex: male=0, female=1
# Why: Binary categorical -> simple numeric mapping
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked (C, Q, S)
# Why: Nominal categorical (no order) -> one-hot encoding
# drop_first=True to avoid multicollinearity (one column being deduced from the otehrs)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Creating new features
# FamilySize: indicates traveling alone or with family
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# IsAlone: binary indicator for solo travelers (solo travelers might have different survival rates)
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("\n")
print("FINAL FEATURE SET")
print("\n")
print(df.columns.tolist())
print(f"\nFinal shape: {df.shape}")


# TRAIN/TEST SPLIT

# Separate features (X) from target/label (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split: 80% train, 20% test
# random_state=42 for reproducibility
# stratify=y ensures same survival ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n")
print("TRAIN/TEST SPLIT")
print(f"Training set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")
print(f"Train survival rate: {y_train.mean():.2%}")
print(f"Test survival rate: {y_test.mean():.2%}")

# Feature Scaling
# Fit and trasnform on train, transform only on test to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  

print("\nData preprocessing complete!!")
print("Ready for model training")


# MODEL SELECTION

# Two models will be trained and compared:
#
# 1. LOGISTIC REGRESSION
#    Pros: Simple, fast, interpretable
#    Cons: Assumes linear decision boundary
#    Use case: Good baseline model
#
# 2. RANDOM FOREST CLASSIFIER  
#    Pros: Handles non-linear relationships, feature interactions
#    Cons: Less interpretable, slower training
#    Use case: Usually better accuracy than logistic regression
#
# Strategy: Start with simple (Logistic), then try complex (RF)
# Choose based on: test accuracy + overfitting chekc

# Dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store results
results = {}

print("\n")
print("MODEL TRAINING")
print("\n")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Store predictions for later evaluation
    results[name] = {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    print(f"{name} trained successfully!!")

print("\nBoth models trained. Ready for evaluation!")



# MODEL EVALUATION

for name, result in results.items():
    print("\n")
    print(f"{name} EVALUATION")
    print("\n")
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, result['y_pred_train'])
    test_acc = accuracy_score(y_test, result['y_pred_test'])
    precision = precision_score(y_test, result['y_pred_test'])
    recall = recall_score(y_test, result['y_pred_test'])
    f1 = f1_score(y_test, result['y_pred_test'])
    
    # Store in results dictionary
    results[name]['metrics'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    
    # Checking for overfitting
    gap = train_acc - test_acc
    if gap > 0.05:
        print(f"\nPossible overfitting (gap: {gap:.4f})")
        print("   Train score significantly higher than test score")
    else:
        print(f"\nGood generalization (gap: {gap:.4f})")
        print("   Model performs similarly on train and test data")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, result['y_pred_test'])
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nInterpretation:")
    print(f"  True Negatives (TN):  {cm[0,0]} - Correctly predicted 'not survived'")
    print(f"  False Positives (FP): {cm[0,1]} - Predicted 'survived' but actually didn't")
    print(f"  False Negatives (FN): {cm[1,0]} - Predicted 'not survived' but actually did")
    print(f"  True Positives (TP):  {cm[1,1]} - Correctly predicted 'survived'")


# VISUALIZATIONS

# Choose best model based on test accuracy
best_model_name = max(results, key=lambda x: results[x]['metrics']['test_accuracy'])
best_result = results[best_model_name]

print("\n")
print(f"BEST MODEL: {best_model_name}")
print("\n")

# Create confusion matrix heatmap
cm = confusion_matrix(y_test, best_result['y_pred_test'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title(f'{best_model_name} - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Confusion matrix saved to results/confusion_matrix.png")


# SAVE EVALUATION REPORT


with open('results/evaluation_report.txt', 'w') as f:
    f.write("CLASSIFICATION PROJECT EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Test Accuracy: {best_result['metrics']['test_accuracy']:.4f}\n")
    f.write(f"Precision: {best_result['metrics']['precision']:.4f}\n")
    f.write(f"Recall: {best_result['metrics']['recall']:.4f}\n")
    f.write(f"F1-Score: {best_result['metrics']['f1']:.4f}\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n\n")
    
    f.write("Detailed Classification Report:\n")
    f.write(classification_report(y_test, best_result['y_pred_test'],
                                 target_names=['Not Survived', 'Survived']))
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("COMPARISON OF BOTH MODELS\n")
    f.write("=" * 60 + "\n\n")
    
    for name, result in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Test Accuracy: {result['metrics']['test_accuracy']:.4f}\n")
        f.write(f"  Overfitting Gap: {result['metrics']['train_accuracy'] - result['metrics']['test_accuracy']:.4f}\n")
        f.write("\n")

print("Evaluation report saved to results/evaluation_report.txt")
print("\n")
print("CLASSIFICATION PROJECT COMPLETE")
print("\n")