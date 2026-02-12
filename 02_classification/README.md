# Titanic Survival Prediction - Classification Project

**Task:** Binary classification to predict passenger survival  
**Dataset:** Kaggle Titanic (891 passengers)  
**Best Model:** Random Forest Classifier  
**Test Accuracy:** 82.68%

---

## Results Summary

| Metric              | Logistic Regression | Random Forest | Winner              |
| ------------------- | ------------------- | ------------- | ------------------- |
| **Test Accuracy**   | 80.45%              | **82.68%**    | Random Forest       |
| **Precision**       | 78.33%              | **79.69%**    | Random Forest       |
| **Recall**          | 68.12%              | **73.91%**    | Random Forest       |
| **F1-Score**        | 72.87%              | **76.69%**    | Random Forest       |
| **Train Accuracy**  | 80.06%              | 98.17%        | -                   |
| **Overfitting Gap** | **-0.39%**          | 15.49%        | Logistic Regression |

---

## Model Selection Decision

### Why Random Forest Despite Overfitting?

**The Question:** Random Forest shows significant overfitting (15.5% gap between train and test accuracy). Why choose it over Logistic Regression which has near-perfect generalization?

**The Answer:** Test accuracy is the deciding metric for production deployment.

#### Performance Comparison

**Random Forest:**

- Train Accuracy: 98.17% (nearly memorized training data)
- Test Accuracy: **82.68%** (performance on unseen data)
- Gap: 15.49% (indicates overfitting)

**Logistic Regression:**

- Train Accuracy: 80.06%
- Test Accuracy: 80.45%
- Gap: -0.39% (greatt generalization)

**Key Insight:** Random Forest still achieves **2.23% higher accuracy on unseen data** despite the overfitting.

#### What This Means

**Overfitting is a warning, not a disqualifier:**

- The model learned real patterns in the data
- The model also memorized some training-specific details
- **But** it still makes better predictions on new passengers

**Real-world impact:**

- Random Forest: Correctly predicts survival for **83 out of 100** new passengers
- Logistic Regression: Correctly predicts survival for **80 out of 100** new passengers
- **3 more correct predictions per 100 passengers**

---

## Detailed Analysis

### When Overfitting Becomes a Problem

Overfitting is bad when test performance is worse than alternatives:

```
Bad Overfitting Example:
Model A: Train 95%, Test 70% : Overfitting hurt performance
Model B: Train 75%, Test 74% : Choose Model B
```

Overfitting is accepted when test performance is better:

```
Our Case:
Random Forest: Train 98%, Test 83% : Overfitting but still better
Logistic Reg:  Train 80%, Test 80% : Random Forest wins
```

---

### Production Recommendations

**Chosen Model:** Random Forest

**Monitoring Plan:**

1. Track performance on fresh data batches
2. If test accuracy drops below 80%, switch to Logistic Regression
3. Monitor confusion matrix for class-specific degradation

**Improvement Options (if needed):**

```python
# Reduce overfitting via hyperparameter tuning:
RandomForestClassifier(
    n_estimators=50,        # Reduce from 100
    max_depth=10,           # Add depth limit
    min_samples_split=10,   # Require more samples to split
    random_state=42
)
```

**Alternative Approach:**
If interpretability is valued over 2.2% accuracy gain, Logistic Regression is also defensible:

- More stable predictions
- Easier to explain
- Lower risk of performance degradation

---

## Confusion Matrix Interpretation

### Random Forest Results

```
                Predicted
              Not Survived | Survived
Actual  ─────────────────────────────
Not Survived  │     97     │    13    │ = 110
Survived      │     18     │    51    │ = 69
              └─────────────────────────┘
                  115          64
```

**Breakdown:**

- **True Negatives (97):** Correctly identified non-survivors
- **True Positives (51):** Correctly identified survivors
- **False Positives (13):** Predicted survival when passenger didn't survive
  - _Cost:_ Low - disappointing prediction but not life-threatening
- **False Negatives (18):** Predicted death when passenger survived
  - _Cost:_ Medium - missed identifying survival factors

**Class Performance:**

- Non-survivors: 97/110 = **88.2% recall** (found most non-survivors)
- Survivors: 51/69 = **73.9% recall** (found most survivors)

**Why FN > FP matters:**
In this dataset, predicting death when someone survived (FN=18) is more common than predicting survival when someone died (FP=13). This suggests the model is slightly conservative, which is acceptable given the 38% survival rate (imbalanced classes).

---

## Technical Implementation

### Data Preprocessing

**Missing Values:**

```python
# Age: 19.8% missing (177 values)
df['Age'] = df['Age'].fillna(df['Age'].median())
# Why median? Robust to outliers (few very old passengers)

# Embarked: 0.2% missing (2 values)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Why mode? Most common port (Southampton)
```

**Feature Engineering:**

```python
# Created features based on survival hypotheses:
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Hypothesis: Solo travelers had different survival rates
# Result: IsAlone improved accuracy by ~2%
```

**Encoding:**

```python
# Sex: Binary categorical -> Label encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Embarked: Nominal categorical -> One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# drop_first=True avoids multicollinearity
```

**Scaling:**

```python
# StandardScaler for Logistic Regression (required)
# Not needed for Random Forest (tree-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)        # Transform test
```

---

## Model Training

**Logistic Regression:**

```python
LogisticRegression(random_state=42, max_iter=1000)
```

- `max_iter=1000`: Ensures convergence
- `random_state=42`: Ensures reproducibility

**Random Forest:**

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

- `n_estimators=100`: 100 decision trees
- Default hyperparameters used

---

## Metrics Explanation

**Why Multiple Metrics?**

Accuracy alone is insufficient for imbalanced datasets (38% survival rate).

**Precision (79.69%):**

- Of passengers predicted to survive, 79.69% actually survived
- Important when: False alarms are costly
- Context: Predicting survival when they died (FP) has low cost

**Recall (73.91%):**

- Of passengers who survived, we correctly identified 73.91%
- Important when: Missing positives is costly
- Context: Missing survival cases (FN) means we failed to identify survival factors

**F1-Score (76.69%):**

- Harmonic mean balances precision and recall
- Useful when: Both false positives and false negatives matter
- 76.69% indicates good balance

---

## Key Learnings

### 1. Feature Engineering Matters

Created features (FamilySize, IsAlone) improved accuracy by **~2%**. Domain knowledge drives good features.

### 2. Data Leakage Prevention

```python
# WRONG (data leakage):
scaler.fit(X)  # Uses all data including test
X_train, X_test = train_test_split(X)

# RIGHT (no leakage):
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # Only train data
X_test_scaled = scaler.transform(X_test)
```

### 3. Overfitting Context Matters

- Overfitting isn't automatically bad if test performance is superior
- Monitor production performance to ensure it doesn't degrade
- Consider the tradeoff: 2.2% accuracy vs 15.5% overfitting gap

### 4. Class Imbalance Awareness

- 38% survival rate (imbalanced)
- Stratified split maintains class distribution
- Accuracy alone would be misleading
- Precision/Recall provide class-specific insights

---

## Files

- **`classification_project.py`** - Complete pipeline from data loading to evaluation
- **`data/train.csv`** - Titanic dataset (source: Kaggle)
- **`data/README.md`** - Data source citation
- **`results/confusion_matrix.png`** - Visual confusion matrix
- **`results/evaluation_report.txt`** - Detailed metrics report

---

## Usage

```bash
# Ensure virtual environment is activated
cd 02_classification

# Run complete pipeline
python classification_project.py

# Output:
# Prints data exploration
# Shows model evaluation metrics
# Saves confusion matrix to results/
# Saves evaluation report to results/
```
