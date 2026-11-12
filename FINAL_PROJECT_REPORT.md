# University AI/ML Assignment: Final Project Report

## Project Title
**Online Shopping Purchase Intention Prediction & Visitor Segmentation Using Machine Learning**

---

### Academic Information
- **Course**: IT2011 — Artificial Intelligence and Machine Learning (Year 2 Semester 1)
- **Dataset**: UCI Online Shoppers Purchasing Intention Dataset (`online_shoppers_intention.csv`, 12,330 sessions, 18 attributes)
- **Target Variable**: `Revenue` (Binary: `False` = No Purchase [84.53%], `True` = Purchase [15.47%])
- **Group Models Implemented**:
  1. **Logistic Regression (`IT25100737`)** — Supervised Linear Baseline Classifier
  2. **Random Forest (`IT25100153`)** — Supervised Bagging Ensemble Classifier
  3. **Support Vector Machine (`IT25101477`)** — Supervised Non-Linear RBF Margin Classifier
  4. **Decision Tree (`IT25102733` — Fernando W M M P D)** — Supervised Rule-Based Tree Classifier
  5. **K-Means Clustering + PCA (`IT25102287` — Eragoda W M S P)** — Unsupervised Behavioral Segmentation

---

## 1. Executive Summary & Problem Framing
In modern e-commerce environments, understanding and predicting whether an active visitor session will result in a purchase—and grouping visitors into actionable behavioral segments—is essential for maximizing conversion rates, personalizing user experience, and reducing cart abandonment. Because web visitor volume is large but actual checkout events are relatively infrequent (15.47% positive purchase rate), this problem is characterized by severe class imbalance (~5.46 : 1).

This project implements a rigorous, leak-free machine learning workflow across 12,330 historical browsing sessions. Four supervised classification algorithms (**Logistic Regression**, **Random Forest**, **Support Vector Machine**, and **Decision Tree**) and one unsupervised clustering algorithm (**K-Means + PCA**) were developed, tuned via **5-Fold Stratified Cross-Validation** on an 80% training partition (9,864 sessions), and verified on a strictly untouched 20% test partition (2,466 sessions).

---

## 2. Dataset Understanding & Schema Audit
- **Total Records (Instances)**: 12,330
- **Total Attributes (Features)**: 17 predictor features + 1 target (`Revenue`)
- **Missing Values**: 0 across all 18 columns
- **Target Distribution**:
  - `Revenue = False`: 10,422 sessions (84.53%)
  - `Revenue = True`: 1,908 sessions (15.47%)
  - **Imbalance Ratio**: 5.46 : 1

### Feature Categorization
| Feature Name | Type | Standard Processing | Description |
| :--- | :--- | :--- | :--- |
| `Administrative` | Numerical | `StandardScaler` / Log1p | Number of administrative pages visited |
| `Administrative_Duration` | Numerical | `StandardScaler` / Log1p | Time spent on administrative pages (seconds) |
| `Informational` | Numerical | `StandardScaler` / Log1p | Number of informational pages visited |
| `Informational_Duration` | Numerical | `StandardScaler` / Log1p | Time spent on informational pages (seconds) |
| `ProductRelated` | Numerical | `StandardScaler` / Log1p | Number of product pages visited |
| `ProductRelated_Duration` | Numerical | `StandardScaler` / Log1p | Time spent on product pages (seconds) |
| `BounceRates` | Numerical | `StandardScaler` | Average bounce rate of visited pages |
| `ExitRates` | Numerical | `StandardScaler` | Average exit rate of visited pages |
| `PageValues` | Numerical | `StandardScaler` | Google Analytics page value prior to transaction |
| `SpecialDay` | Numerical | `StandardScaler` | Closeness of visit date to a special day/holiday |
| `OperatingSystems` | Numerical (ID) | `StandardScaler` / OneHot | Operating system identifier (1–8) |
| `Browser` | Numerical (ID) | `StandardScaler` / OneHot | Browser identifier (1–13) |
| `Region` | Numerical (ID) | `StandardScaler` / OneHot | Geographic region identifier (1–9) |
| `TrafficType` | Numerical (ID) | `StandardScaler` / OneHot | Traffic channel identifier (1–20) |
| `Month` | Categorical | `OneHotEncoder` / Ordinal | Calendar month (`Feb`, `Mar`, `May`, `June`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec`) |
| `VisitorType` | Categorical | `OneHotEncoder` | `Returning_Visitor`, `New_Visitor`, `Other` |
| `Weekend` | Boolean | Numeric (`0`/`1`) | Whether the session took place on Saturday/Sunday |
| `Revenue` | Boolean | Target (`0`/`1`) | Whether a transaction was completed |

---

## 3. Data Preprocessing & Leakage Prevention
To guarantee scientific validity and prevent data leakage across all five models:
1. **Strict Stratified Split**: The dataset was partitioned into 80% training (9,864 samples) and 20% testing (2,466 samples) using `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`.
2. **Untouched Test Partition**: The 2,466 test sessions were isolated upfront and never exposed during feature scaling, outlier capping, SMOTE resampling, or hyperparameter tuning.
3. **Pipeline Encapsulation**: All transformations (`StandardScaler`, `OneHotEncoder`, log transformations, and in-fold `SMOTE` / `SMOTENC`) were encapsulated inside scikit-learn `Pipeline` and `ColumnTransformer` objects so that statistics were learned strictly within each training fold.

---

## 4. Exploratory Data Analysis (EDA) Insights
Publication-grade figures generated directly from the dataset (`reports/figures/`) revealed five core domain findings:
1. **Severe Class Imbalance (`01_revenue_class_distribution.png` & `02_revenue_percentage.png`)**:
   84.53% of sessions do not convert, making raw Accuracy a misleading metric; **F1-Score**, **ROC-AUC**, **PR-AUC**, **Precision**, and **Recall** are used as primary evaluation metrics.
2. **Seasonal Conversion Spikes (`03_monthly_visitor_distribution.png` & `04_revenue_by_month.png`)**:
   Traffic volume peaks in May (3,364 sessions) and November (2,998 sessions), with November exhibiting the highest conversion rate due to Black Friday and holiday shopping.
3. **Visitor Type Paradox (`05_visitortype_distribution.png` & `06_revenue_by_visitortype.png`)**:
   Returning visitors generate 85.57% of sessions, but New Visitors convert at a higher relative rate (~24.9% vs. ~13.9%).
4. **Dominance of `PageValues` & Exit Telemetry (`08_pagevalues_distribution.png`–`10_exitrates_distribution.png`)**:
   `PageValues` is heavily right-skewed and serves as the strongest discriminator of purchase intent. Conversely, high `BounceRates` and `ExitRates` strongly signal non-converting sessions.
5. **Multicollinearity (`12_correlation_heatmap.png`)**:
   Strong positive correlation exists between `BounceRates` and `ExitRates` ($r = 0.91$) and between `ProductRelated` page count and `ProductRelated_Duration` ($r = 0.86$).

---

## 5. Individual Model Architectures & Hyperparameter Tuning

### 5.1 Model 1: Logistic Regression (`IT25100737`)
- **Folder**: `individual_models/IT25100737_Logistic_Regression/`
- **Methodology**: Evaluated 8 experimental varieties comparing default L2 (`lbfgs`), `class_weight='balanced'`, in-fold `SMOTE`, L1 sparsity (`saga` and `liblinear`), and out-of-fold probability threshold optimization—explicitly demonstrating why pre-split SMOTE (V8) causes data leakage and must be avoided.
- **Optimal Configuration**:
  - **V5 (Default Threshold = 0.50)**: `C=0.01`, `penalty='l1'`, `solver='saga'`, with `SMOTE` inside each CV fold ($\text{CV F1} = 0.6698$, $\text{CV PR-AUC} = 0.6669$).
  - **V7 (Tuned Threshold = 0.57)**: Uses V5 pipeline with decision threshold tuned on out-of-fold probabilities ($\text{CV F1} = 0.6739$).
- **Held-Out Test Performance (V7)**:
  - $\text{Precision} = 0.6113$, $\text{Recall} = 0.7618$, $\text{F1-Score} = 0.6783$, $\text{ROC-AUC} = 0.9205$, $\text{PR-AUC} = 0.6783$.

### 5.2 Model 2: Random Forest Classifier (`IT25100153`)
- **Folder**: `individual_models/IT25100153_Random_Forest/`
- **Methodology**: Compared 9 forest configurations across 5-fold stratified CV (baseline unregularized, `balanced` weights, `balanced_subsample`, regularized depth/leaf constraints, `SMOTENC`, random undersampling, and an ablation without `PageValues`), followed by `RandomizedSearchCV` (24 candidates) and `GridSearchCV` (9 candidates).
- **Optimal Hyperparameters**:
  - `n_estimators=100`, `max_depth=12`, `min_samples_leaf=5`, `max_features=0.5`, `criterion='gini'`, `class_weight='balanced'`, `random_state=42`.
- **5-Fold Cross-Validation Score**: $\text{Mean F1} = 0.6837$.
- **Held-Out Test Performance**:
  - $\text{Accuracy} = 0.8849$, $\text{Precision} = 0.6068$, $\text{Recall} = 0.7513$, $\text{F1-Score} = 0.6713$, $\text{ROC-AUC} = 0.9320$, $\text{PR-AUC} = 0.7398$.

### 5.3 Model 3: Support Vector Machine (`IT25101477`)
- **Folder**: `individual_models/IT25101477_SVM/` & `backend/`
- **Methodology**: Constructed a leak-free `ColumnTransformer` (`StandardScaler` on 14 numeric columns, `OneHotEncoder` on categorical columns) paired with `SVC(kernel='rbf', probability=True, class_weight='balanced')`, tuned via 5-fold `GridSearchCV` over $C \in [0.1, 1.0, 10.0, 100.0]$ and $\gamma \in [\text{'scale'}, 0.01, 0.001]$.
- **Optimal Hyperparameters**: $C = 100.0$, $\gamma = 0.001$, $\text{kernel} = \text{'rbf'}$.
- **5-Fold Cross-Validation Score**: $\text{Mean F1} = 0.6655 \pm 0.0164$.
- **Held-Out Test Performance**:
  - $\text{Accuracy} = 0.8747$, $\text{Precision} = 0.5759$, $\text{Recall} = 0.7251$, $\text{F1-Score} = 0.6419$, $\text{ROC-AUC} = 0.8997$.
  - **Confusion Matrix**: $\text{TN} = 1,880$, $\text{FP} = 204$, $\text{FN} = 105$, $\text{TP} = 277$.

### 5.4 Model 4: Decision Tree Classifier (`IT25102733` — Fernando W M M P D)
- **Folder**: `individual_models/IT25102733_Decision_Tree/`
- **Methodology**: Investigated tree depth overfitting across $\text{max\_depth} \in [1, 3, 5, 10, \text{None}]$ (showing unrestricted depth-29 trees memorize training data with train accuracy = 1.0 but collapse to 0.741 test ROC-AUC), followed by 180-candidate 5-fold Stratified `GridSearchCV`.
- **Optimal Hyperparameters**:
  - **Individual Tuned Tree (`IT25102733`)**: `max_depth=3`, `criterion='entropy'`, `class_weight='balanced'` ($\text{Test Accuracy} = 0.8750$, $\text{Precision} = 0.5710$, $\text{Recall} = 0.8120$, $\text{F1} = 0.6700$, $\text{ROC-AUC} = 0.9080$).
  - **Depth-7 Gini Comparison Tree**: `max_depth=7`, `criterion='gini'`, `class_weight='balanced'` ($\text{CV F1} = 0.6344 \pm 0.0069$, $\text{Test Accuracy} = 0.8500$, $\text{Precision} = 0.5098$, $\text{Recall} = 0.8194$, $\text{F1} = 0.6285$, $\text{ROC-AUC} = 0.8928$, $\text{TN} = 1,783, \text{FP} = 301, \text{FN} = 69, \text{TP} = 313$).

### 5.5 Model 5: K-Means Clustering + PCA (`IT25102287` — Eragoda W M S P)
- **Folder**: `individual_models/IT25102287_KMeans_Clustering/` & `deployment/`
- **Methodology**: Evaluated 13 preprocessing and feature-selection varieties (`V01`–`V13`) across $k \in \{2..8\}$ using 5-fold cross-validated Silhouette, Davies-Bouldin, and Calinski-Harabasz indices. Excluded `PageValues` in **V09** so segments reflect pure browsing behavior before checkout pages are reached.
- **Optimal Configuration (`V09`)**:
  - Features: 9 behavioral attributes (`Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `SpecialDay`) with `log1p` transform on skewed counts/durations + `StandardScaler`.
  - Hyperparameters: `n_clusters=3`, `init='k-means++'`, `n_init=10`, `algorithm='lloyd'`.
- **Cluster Quality Metrics**:
  - **5-Fold CV Silhouette**: `0.400 ± 0.007` | **Test Silhouette**: `0.4090`
  - **Test Davies-Bouldin Index**: `1.0985` | **Test Calinski-Harabasz Index**: `837.00`
- **Discovered Shopper Personas**:
  1. **Segment 1 — Quick Exits (7.74% of traffic, 0.26% purchase rate)**: Bounce immediately (`BounceRates ≈ 0.176`, `ExitRates ≈ 0.187`, ~2 product pages).
  2. **Segment 0 — Casual Browsers (74.19% of traffic, 14.36% purchase rate)**: Typical visitors (~21 product pages, ~787s product duration).
  3. **Segment 2 — Engaged Shoppers (18.08% of traffic, 26.53% purchase rate)**: High-intent visitors (~85 product pages, ~3,324s duration, minimal bounce/exit).

---

## 6. Consolidated Multi-Model Comparison & Business Alignment

| Model | Student ID | Paradigm | CV Score | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Primary Operational Strength |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | `IT25100153` | Ensemble | **0.6837** (F1) | **0.8849** | 0.6068 | 0.7513 | 0.6713 | **0.9320** | Highest overall discrimination (ROC-AUC & PR-AUC) & accuracy |
| **Logistic Regression** | `IT25100737` | Linear | 0.6739 (F1) | — | **0.6113** | 0.7618 | **0.6783** | 0.9205 | Best precision & threshold-tuned F1 with fast linear inference |
| **Decision Tree** | `IT25102733` | Rule Tree | ~0.6600 (F1) | 0.8750 | 0.5710 | **0.8120** | 0.6700 | 0.9080 | Highest recall (catches >81% of buyers) & human-readable rules |
| **SVM (RBF)** | `IT25101477` | Margin | 0.6655 (F1) | 0.8747 | 0.5759 | 0.7251 | 0.6419 | 0.8997 | Strong non-linear boundary & high specificity (90.21%) |
| **K-Means (+PCA)** | `IT25102287` | Clustering | 0.400 (Sil) | — | — | — | — | — | Real-time behavioral persona segmentation without page values |

### Business Trade-Off Recommendations
- **When Promotional Budgets Are Constrained (Precision-Critical)**: Use **Logistic Regression (`IT25100737`)** or **Random Forest (`IT25100153`)** (Precision ~61%), minimizing costly coupon leakage to non-buyers.
- **When Maximizing Total Conversions (Recall-Critical)**: Use the **Decision Tree (`IT25102733`)** (Recall ~81.2%–81.9%) to trigger low-cost web interventions (e.g., exit-intent banners) for the widest pool of prospective buyers.
- **When Early-Session Personalization Is Needed (Pre-Checkout)**: Use **K-Means (`IT25102287`)** to classify visitors into *Quick Exits*, *Casual Browsers*, or *Engaged Shoppers* using pure browsing telemetry.

---

## 7. Feature Importance & Interpretability Synthesis
Across all supervised models:
1. **`PageValues` is the dominant predictor**:
   - Accounts for **~78%–79%** of Gini impurity reduction in Decision Trees and causes a **+0.3760 F1 drop** when permuted in SVM/Random Forest.
   - Notably, in the Random Forest ablation experiment (`regularized_balanced_without_pagevalues`), removing `PageValues` caused CV F1 to drop significantly, confirming its primary role while motivated our K-Means (`V09`) design to segment purely on browsing behavior.
2. **Seasonality (`Month_Nov` & `Month_May`)**: Consistently ranks second in importance across tree, linear, and permutation analyses.
3. **Session Depth & Friction (`ProductRelated_Duration`, `ExitRates`, `BounceRates`)**: Longer product engagement combined with low exit/bounce rates strongly increases conversion probability.

---

## 8. Ethical Considerations & Responsible AI
1. **Data Privacy**: The dataset contains anonymized session-level telemetry devoid of PII. Production deployments must enforce explicit cookie/telemetry consent (GDPR/ePrivacy).
2. **Demographic & Algorithmic Bias**: Proxy features (`Region`, `OperatingSystems`, `Browser`) must never be used for predatory price discrimination.
3. **Class Imbalance Fairness**: Balanced class weights and in-fold SMOTE ensure minority-class buyers are accurately represented.
4. **Human Oversight**: Model probabilities and cluster labels are decision-support heuristics and should remain subject to human oversight.

---

## 9. AI Tool Usage Declaration
In compliance with university academic integrity guidelines:

| Tool | Purpose | Verification Protocol |
| :--- | :--- | :--- |
| **Google Antigravity / Gemini** | Project organization, FastAPI/Streamlit scaffolding, documentation formatting, and unit test suites | All code, cross-validation splits, and evaluation metrics were executed and verified against real `online_shoppers_intention.csv` runs by the student team. |
| **Scikit-Learn / Imbalanced-Learn** | Pipeline construction, StratifiedKFold, GridSearchCV, Logistic Regression, Random Forest, SVM, Decision Tree, and K-Means modeling | Metrics, confusion matrices, and serialized `.joblib` models verified programmatically. |

**Academic Integrity Attestation**: The student group confirms that all statistical outputs, cross-validation scores, and test set metrics were genuinely computed from the UCI dataset without fabrication.
