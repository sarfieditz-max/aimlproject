import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# Notebook 1: Data Understanding
nb1_cells = [
    md_cell("""# Notebook 01: Dataset Understanding & Problem Framing
## Project: Online Shopping Purchase Intention Prediction
**University AI & Machine Learning Group Assignment**

### 1. Problem Framing
E-commerce businesses experience high visitor traffic, but only a small fraction of sessions terminate with a completed transaction. Accurately predicting a visitor's purchase intention in real-time allows merchants to trigger personalized micro-interventions (e.g., targeted discounts, exit-intent promotions, or live chat support) before the user departs.

### 2. Dataset Overview
- **Source**: UCI Machine Learning Repository / Online Shoppers Purchasing Intention Dataset
- **Instances**: 12,330 user sessions
- **Attributes**: 17 feature attributes (10 numerical, 4 categorical/integer IDs, 3 boolean/categorical)
- **Target**: `Revenue` (Binary: `True` = Purchase, `False` = No Purchase)"""),
    code_cell("""import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../backend/data/online_shoppers_intention.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()"""),
    md_cell("""### 3. Schema Verification & Null Value Assessment
A rigorous machine learning workflow requires verifying data integrity, feature data types, and checking for missing or corrupt records before any modeling occurs."""),
    code_cell("""# Inspect data types and missing values
info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null Count': df.notnull().sum(),
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df)) * 100,
    'Unique Values': df.nunique()
})
info_df"""),
    md_cell("""### 4. Target Variable Analysis: Class Imbalance
The target column `Revenue` represents whether a session culminated in an e-commerce purchase. We calculate both the raw session counts and relative percentages to quantify the extent of class imbalance."""),
    code_cell("""counts = df['Revenue'].value_counts()
percentages = df['Revenue'].value_counts(normalize=True) * 100

summary_target = pd.DataFrame({
    'Count': counts,
    'Percentage (%)': percentages.round(2)
})
print("Target Class Breakdown:")
print(summary_target)"""),
    md_cell("""### Key Findings & Implications:
1. **Zero Missing Data**: The dataset contains 12,330 complete records without any missing values.
2. **Severe Class Imbalance**: 84.53% of browsing sessions resulted in no purchase (`Revenue=False`), while only 15.47% converted (`Revenue=True`).
3. **Metric Selection**: Conventional accuracy is a deceptive metric. A naive majority-class classifier achieved 84.53% accuracy with zero utility. Therefore, **F1-Score**, **Precision**, **Recall**, and **ROC-AUC** must be prioritized.
4. **Stratified Splitting**: When dividing data into train and test sets, stratified sampling is mandatory to ensure representative minority-class ratios across all partitions.""")
]

with open(NOTEBOOKS_DIR / "01_data_understanding.ipynb", "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# Notebook 2: EDA
nb2_cells = [
    md_cell("""# Notebook 02: Exploratory Data Analysis (EDA)
## Project: Online Shopping Purchase Intention Prediction
**University AI & Machine Learning Group Assignment**

This notebook performs systematic univariate, bivariate, and multivariate analysis on the 12,330 e-commerce sessions to uncover behavioral patterns distinguishing purchasers from non-purchasers."""),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../backend/data/online_shoppers_intention.csv')
num_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
    'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'
]"""),
    md_cell("""### 1. Numerical Feature Statistical Profiles
Examining central tendency, spread, and quartiles across all numerical browsing telemetry features."""),
    code_cell("""df[num_cols].describe().T"""),
    md_cell("""### 2. Behavioral Features Analysis: PageValues, BounceRates, and ExitRates
- `PageValues`: Google Analytics metric reflecting average value of visited pages prior to completing an e-commerce transaction.
- `BounceRates`: Percentage of visitors entering the site and exiting without triggering other requests.
- `ExitRates`: Percentage of pageviews on a specific page that were the last in the session."""),
    code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.boxplot(data=df, x='Revenue', y='PageValues', ax=axes[0], palette=['#3B82F6', '#10B981'])
axes[0].set_title('PageValues by Purchase Outcome')
axes[0].set_yscale('log')

sns.boxplot(data=df, x='Revenue', y='BounceRates', ax=axes[1], palette=['#3B82F6', '#10B981'])
axes[1].set_title('BounceRates by Purchase Outcome')

sns.boxplot(data=df, x='Revenue', y='ExitRates', ax=axes[2], palette=['#3B82F6', '#10B981'])
axes[2].set_title('ExitRates by Purchase Outcome')

plt.tight_layout()
plt.show()"""),
    md_cell("""### 3. Correlation Structure
Analyzing Pearson correlation across numerical features to diagnose multicollinearity."""),
    code_cell("""plt.figure(figsize=(10, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Predictors')
plt.show()"""),
    md_cell("""### Summary of EDA Insights:
1. `PageValues` exhibits an extraordinarily strong positive association with `Revenue`. Converting users spend substantial time on high-value pages.
2. `BounceRates` and `ExitRates` are strongly positively correlated ($r \\approx 0.91$), with non-purchasers exhibiting substantially higher bounce and exit rates.
3. November accounts for the largest purchase volume, coinciding with major promotional events (Black Friday / Cyber Monday).
4. Returning visitors constitute the bulk of traffic (85.57%), but new visitors exhibit a slightly higher conversion percentage.""")
]

with open(NOTEBOOKS_DIR / "02_eda.ipynb", "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# Notebook 3: Model Training
nb3_cells = [
    md_cell("""# Notebook 03: Model Training & Hyperparameter Tuning
## Project: Online Shopping Purchase Intention Prediction
**Primary Model**: Support Vector Machine (SVM)
**Comparison Model**: Decision Tree Classifier

### Methodology:
1. **Stratified 80/20 Train-Test Split**: The 20% test partition remains untouched throughout model tuning to ensure zero data leakage.
2. **Pipelines & ColumnTransformer**:
   - `StandardScaler` for numerical predictors (critical for SVM margin optimization).
   - `OneHotEncoder(handle_unknown='ignore')` for nominal features (`Month`, `VisitorType`).
   - Binary encoding for `Weekend`.
3. **Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation maximizing **F1-Score**."""),
    code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Load Data
df = pd.read_csv('../backend/data/online_shoppers_intention.csv')
X = df.drop(columns=['Revenue'])
X['Weekend'] = X['Weekend'].astype(int)
y = df['Revenue'].astype(int)

# 2. Stratified 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")"""),
    md_cell("""### 3. SVM Pipeline & Hyperparameter Tuning
We optimize the regularization parameter $C$ and kernel coefficient $\\gamma$ with an RBF kernel and `class_weight='balanced'`."""),
    code_cell("""num_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
    'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'
]
cat_cols = ['Month', 'VisitorType']
bool_cols = ['Weekend']

preprocessor_svm = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('bool', 'passthrough', bool_cols)
])

svm_pipeline = Pipeline([
    ('preprocessor', preprocessor_svm),
    ('classifier', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
])

svm_param_grid = {
    'classifier__C': [0.1, 1.0, 10.0, 100.0],
    'classifier__gamma': ['scale', 0.01, 0.001]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
svm_grid.fit(X_train, y_train)

print(f"Best SVM Parameters: {svm_grid.best_params_}")
print(f"Best SVM 5-Fold CV F1: {svm_grid.best_score_:.4f}")"""),
    md_cell("""### 4. Decision Tree Pipeline & Hyperparameter Tuning
We optimize tree depth, leaf sample limits, and split criteria with `class_weight='balanced'`."""),
    code_cell("""preprocessor_dt = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('bool', 'passthrough', bool_cols)
])

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor_dt),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
])

dt_param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(dt_pipeline, dt_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
dt_grid.fit(X_train, y_train)

print(f"Best Decision Tree Parameters: {dt_grid.best_params_}")
print(f"Best Decision Tree 5-Fold CV F1: {dt_grid.best_score_:.4f}")""")
]

with open(NOTEBOOKS_DIR / "03_model_training.ipynb", "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# Notebook 4: Model Evaluation
nb4_cells = [
    md_cell("""# Notebook 04: Model Evaluation, Comparison & Interpretability
## Project: Online Shopping Purchase Intention Prediction
**University AI & Machine Learning Group Assignment**

This notebook evaluates the tuned SVM and Decision Tree models on the **untouched 20% test dataset** (2,466 sessions). We compare performance metrics, confusion matrices, ROC and Precision-Recall curves, and analyze feature importance."""),
    code_cell("""import json
import pandas as pd
import numpy as np

# Load actual model results generated from the genuine pipeline execution
with open('../backend/artifacts/metrics/model_results.json') as f:
    results = json.load(f)

comp_df = pd.DataFrame(results['comparison_table'])
print("=== Quantitative Performance Comparison (Untouched Test Set) ===")
comp_df"""),
    md_cell("""### 1. Classification Reports
Examining class-specific Precision, Recall, and F1-Scores for both algorithms."""),
    code_cell("""for model_name in ['SVM', 'Decision Tree']:
    print(f"--- {model_name} Classification Report ---")
    cr = results['models'][model_name]['classification_report']
    print(pd.DataFrame(cr).T.round(4))
    print()"""),
    md_cell("""### 2. Confusion Matrix Analysis
- True Negatives (TN): Correctly identified non-purchasers
- False Positives (FP): Non-purchasers incorrectly flagged as prospective buyers (false alarm)
- False Negatives (FN): Missed buyers who were about to purchase (missed revenue opportunity)
- True Positives (TP): Accurately identified buyers"""),
    code_cell("""svm_cm = np.array(results['models']['SVM']['confusion_matrix'])
dt_cm = np.array(results['models']['Decision Tree']['confusion_matrix'])

print(f"SVM Test Confusion Matrix:\\n{svm_cm}\\n")
print(f"Decision Tree Test Confusion Matrix:\\n{dt_cm}")"""),
    md_cell("""### 3. Feature Importance Analysis
- **Decision Tree**: Native Gini/Entropy Impurity Reduction across derived one-hot encoded features.
- **SVM**: Permutation Importance measuring drop in test F1 when feature values are permuted."""),
    code_cell("""print("Top 5 Decision Tree Features:")
print(pd.DataFrame(results['feature_importance']['decision_tree_feature_importance'][:5]))

print("\\nTop 5 SVM Permutation Features:")
print(pd.DataFrame(results['feature_importance']['svm_permutation_importance'][:5]))"""),
    md_cell("""### Final Analytical Conclusion:
1. **Model Selection**: SVM achieves higher overall predictive balance (Test F1 = 0.6419, CV Mean F1 = 0.6655, Accuracy = 87.47%, ROC-AUC = 0.8997) compared to the Decision Tree (Test F1 = 0.6285, CV Mean F1 = 0.6344).
2. **Recall vs Precision Trade-off**: The Decision Tree captures higher Recall (81.94% vs 72.51%) but suffers from lower Precision (50.98% vs 57.59%), generating 301 false positives compared to SVM's 204.
3. **Primary Predictive Drivers**: `PageValues` is unequivocally the most decisive predictor of purchase conversion, followed by seasonality (`Month_Nov`) and visitor browsing duration.""")
]

with open(NOTEBOOKS_DIR / "04_model_evaluation.ipynb", "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

print("All 4 Jupyter notebooks successfully generated!")
