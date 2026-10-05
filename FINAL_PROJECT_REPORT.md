# University AI/ML Assignment: Final Project Report

## Project Title
**Online Shopping Purchase Intention Prediction Using Machine Learning**

---

### Academic Information
- **Course**: Year 2 Semester 1 — Artificial Intelligence and Machine Learning
- **Dataset**: UCI Online Shoppers Purchasing Intention Dataset (`online_shoppers_intention.csv`)
- **Primary Algorithm**: Support Vector Machine (SVM with RBF Kernel)
- **Comparison Algorithm**: Decision Tree Classifier
- **Target Variable**: `Revenue` (Binary: `False` = No Purchase, `True` = Purchase)

---

## 1. Executive Summary & Problem Framing
In modern e-commerce environments, understanding and predicting whether an active visitor session will result in a purchase is essential for maximizing revenue, personalizing user experience, and reducing cart abandonment. Because web visitor volume is large but actual checkout events are infrequent, this problem is characterized by severe class imbalance.

This project implements an end-to-end, scientifically validated machine learning workflow comparing a primary **Support Vector Machine (SVM)** classifier with a **Decision Tree Classifier** across 12,330 historical browsing sessions. Both algorithms were tuned via **5-Fold Stratified Cross-Validation** on an 80% training partition, with final diagnostic verification conducted on a strictly untouched 20% test partition (2,466 sessions).

---

## 2. Dataset Understanding & Schema Audit
- **Total Records (Instances)**: 12,330
- **Total Attributes (Features)**: 17 predictor features + 1 target (`Revenue`)
- **Missing Values**: Exactly 0 across all 18 columns.
- **Target Distribution**:
  - `Revenue = False`: 10,422 sessions (84.53%)
  - `Revenue = True`: 1,908 sessions (15.47%)
  - **Imbalance Ratio**: Approximately 5.46 : 1

### Feature Categorization
| Feature Name | Type | Processing | Description |
| :--- | :--- | :--- | :--- |
| `Administrative` | Numerical | `StandardScaler` | Number of administrative pages visited |
| `Administrative_Duration` | Numerical | `StandardScaler` | Time spent on administrative pages (seconds) |
| `Informational` | Numerical | `StandardScaler` | Number of informational pages visited |
| `Informational_Duration` | Numerical | `StandardScaler` | Time spent on informational pages (seconds) |
| `ProductRelated` | Numerical | `StandardScaler` | Number of product pages visited |
| `ProductRelated_Duration` | Numerical | `StandardScaler` | Time spent on product pages (seconds) |
| `BounceRates` | Numerical | `StandardScaler` | Average bounce rate of visited pages |
| `ExitRates` | Numerical | `StandardScaler` | Average exit rate of visited pages |
| `PageValues` | Numerical | `StandardScaler` | Google Analytics page value before purchase |
| `SpecialDay` | Numerical | `StandardScaler` | Proximity of visit date to a special day |
| `OperatingSystems` | Numerical (ID) | `StandardScaler` | Operating system identifier (1–8) |
| `Browser` | Numerical (ID) | `StandardScaler` | Browser identifier (1–13) |
| `Region` | Numerical (ID) | `StandardScaler` | Geographic region identifier (1–9) |
| `TrafficType` | Numerical (ID) | `StandardScaler` | Traffic channel identifier (1–20) |
| `Month` | Categorical | `OneHotEncoder` | Calendar month (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec) |
| `VisitorType` | Categorical | `OneHotEncoder` | Returning_Visitor, New_Visitor, Other |
| `Weekend` | Boolean | Numeric (0/1) | Whether the session took place on Saturday/Sunday |
| `Revenue` | Boolean | Target (0/1) | Whether transaction was completed |

---

## 3. Data Preprocessing & Leakage Prevention
To guarantee scientific validity and prevent data leakage:
1. **Strict Stratified Split**: The dataset was partitioned into 80% training (9,864 samples) and 20% testing (2,466 samples) using `train_test_split(..., stratify=y, random_state=42)`.
2. **Untouched Test Partition**: The test partition was completely isolated and never exposed during transformer fitting or cross-validation tuning.
3. **Pipeline Encapsulation**: Data transformations were encapsulated inside `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`:
   - `StandardScaler()` applied to the 14 numerical features (crucial for distance-based SVM margin optimization).
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to `Month` and `VisitorType`.
   - Passthrough transformation for numeric `Weekend`.

---

## 4. Exploratory Data Analysis (EDA) Insights
Twelve publication-grade figures were generated directly from the dataset (`backend/artifacts/plots/`):
1. **Revenue Class Distribution (`01_revenue_class_distribution.png`) & Percentage (`02_revenue_percentage.png`)**:
   Revealed an 84.53% to 15.47% majority-to-minority distribution.
2. **Monthly Visitor Trends (`03_monthly_visitor_distribution.png` & `04_revenue_by_month.png`)**:
   Traffic volume concentrates heavily in May (3,364 sessions) and November (2,998 sessions). However, November produces the highest conversion rate, driven by seasonal sales (Black Friday / Cyber Monday).
3. **Visitor Profile Analysis (`05_visitortype_distribution.png` & `06_revenue_by_visitortype.png`)**:
   Returning visitors comprise 85.57% of traffic, but new visitors exhibit higher relative conversion propensity (~24.9% vs ~13.9%).
4. **Behavioral Telemetry (`08_pagevalues_distribution.png`, `09_bouncerates_distribution.png`, `10_exitrates_distribution.png`, `11_productrelated_duration_distribution.png`)**:
   - `PageValues` emerged as the dominant discriminator: non-purchasers cluster near zero, whereas purchasers exhibit substantial positive values.
   - High `BounceRates` and `ExitRates` are strong negative predictors of conversion.
5. **Correlation Heatmap (`12_correlation_heatmap.png`)**:
   Diagnosed strong collinearity between `BounceRates` and `ExitRates` ($r = 0.91$), and between `ProductRelated` count and duration ($r = 0.86$).

---

## 5. Model Training & Hyperparameter Tuning
Cross-validation was conducted strictly on the training set using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` maximizing **F1-score**.

### 5.1 Primary Model: Support Vector Machine (SVC)
- **Base Classifier**: `sklearn.svm.SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)`
- **Search Space**:
  - $C \in [0.1, 1.0, 10.0, 100.0]$
  - $\gamma \in [\text{'scale'}, 0.01, 0.001]$
- **Optimal Hyperparameters**:
  - $C = 100.0$
  - $\gamma = 0.001$
  - $\text{kernel} = \text{'rbf'}$
- **5-Fold Cross-Validation Score**:
  - $\text{Mean F1} = 0.6655$
  - $\text{Std F1} = 0.0164$

### 5.2 Comparison Model: Decision Tree Classifier
- **Base Classifier**: `sklearn.tree.DecisionTreeClassifier(class_weight='balanced', random_state=42)`
- **Search Space**:
  - $\text{max\_depth} \in [3, 5, 7, 10, \text{None}]$
  - $\text{min\_samples\_split} \in [2, 5, 10]$
  - $\text{min\_samples\_leaf} \in [1, 2, 4]$
  - $\text{criterion} \in [\text{'gini'}, \text{'entropy'}]$
- **Optimal Hyperparameters**:
  - $\text{criterion} = \text{'gini'}$
  - $\text{max\_depth} = 7$
  - $\text{min\_samples\_split} = 2$
  - $\text{min\_samples\_leaf} = 1$
- **5-Fold Cross-Validation Score**:
  - $\text{Mean F1} = 0.6344$
  - $\text{Std F1} = 0.0069$

---

## 6. Model Evaluation on Untouched Test Set (N=2,466)
Following hyperparameter tuning, both final pipelines were evaluated on the untouched test partition.

### 6.1 Quantitative Comparison Table
| Metric | SVM (Primary Model) | Decision Tree (Comparison) | Delta (SVM - DT) | Superior Algorithm |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.8747** | 0.8500 | +0.0247 | **SVM** |
| **Precision** | **0.5759** | 0.5098 | +0.0661 | **SVM** |
| **Recall (Sensitivity)** | 0.7251 | **0.8194** | -0.0943 | **Decision Tree** |
| **F1-Score** | **0.6419** | 0.6285 | +0.0134 | **SVM** |
| **ROC-AUC** | **0.8997** | 0.8928 | +0.0069 | **SVM** |
| **5-Fold CV Mean F1** | **0.6655** | 0.6344 | +0.0311 | **SVM** |
| **5-Fold CV Std F1** | 0.0164 | **0.0069** | +0.0095 | **Decision Tree** |

### 6.2 Confusion Matrix Breakdown
#### Support Vector Machine (SVM):
$$\begin{pmatrix} \text{TN: } 1880 & \text{FP: } 204 \\ \text{FN: } 105 & \text{TP: } 277 \end{pmatrix}$$
- **Specificity (True Negative Rate)**: $1880 / (1880 + 204) = 90.21\%$
- **Precision**: $277 / (277 + 204) = 57.59\%$
- **Recall**: $277 / (277 + 105) = 72.51\%$

#### Decision Tree:
$$\begin{pmatrix} \text{TN: } 1783 & \text{FP: } 301 \\ \text{FN: } 69 & \text{TP: } 313 \end{pmatrix}$$
- **Specificity (True Negative Rate)**: $1783 / (1783 + 301) = 85.56\%$
- **Precision**: $313 / (313 + 301) = 50.98\%$
- **Recall**: $313 / (313 + 69) = 81.94\%$

---

## 7. Feature Importance & Interpretability Analysis
1. **Decision Tree Native Gini Feature Importance**:
   - `PageValues`: **78.10%** of total impurity reduction.
   - `Month_Nov`: **6.01%** (November seasonal shopping factor).
   - `ProductRelated_Duration`: **4.76%**.
   - `Administrative_Duration`: **2.03%**.
   - `ExitRates`: **1.73%**.
2. **SVM Permutation Importance (Test F1 Drop)**:
   - `PageValues`: **+0.3760** (shuffling drops F1 from 0.6419 to ~0.2659).
   - `Month`: **+0.0340**.
   - `ProductRelated`: **+0.0112**.
   - `ProductRelated_Duration`: **+0.0033**.
   - `SpecialDay`: **+0.0021**.

**Key Finding**: Both algorithms independently corroborate that `PageValues` is the dominant feature driving purchase intention prediction.

---

## 8. Business Objective Alignment & Algorithm Trade-offs
The application does not enforce a rigid single winner, but frames the selection according to operational objectives:
- **Default Winner (F1-Score: SVM = 0.6419)**: When balancing false alarms against missed conversions on an imbalanced dataset, SVM provides the optimal harmonic mean.
- **Precision-Critical Contexts (SVM Winner: 57.59% vs 50.98%)**: If marketing interventions carry tangible costs (e.g., costly discounts or human sales outreach), SVM minimizes wasted expenditures by producing 97 fewer false alarms.
- **Recall-Critical Contexts (Decision Tree Winner: 81.94% vs 72.51%)**: If customer acquisition value is extraordinarily high and intervention costs are negligible (e.g., unobtrusive web push banners), the Decision Tree successfully detects 36 additional buyers.

---

## 9. Ethical Considerations & Responsible AI
1. **Data Privacy**: The dataset contains session-level browsing telemetry devoid of PII (names, emails, home addresses, payment tokens). Real-world implementations must enforce consent frameworks (GDPR/ePrivacy directive).
2. **Demographic & Algorithmic Bias**: Feature inputs such as `Region`, `OperatingSystems`, or `Browser` can act as socio-economic proxies. Models must never be utilized for predatory price steering or discriminatory service degradation.
3. **Class Imbalance Fairness**: Failing to adjust for class imbalance leads models to systematically ignore minority buyers. Balanced class weighting prevents systematic under-servicing.
4. **Probabilistic Uncertainty**: Predicted probabilities (`predict_proba`) represent model confidence heuristics, not deterministic human behaviors. Predictions must remain subject to human oversight.

---

## 10. AI Tool Usage Declaration
In full compliance with university academic integrity requirements, all generative AI tool usage is declared below:

| Tool | Version | Purpose | Extent of Use | Verification Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **Google Antigravity / Gemini 3.8 Flash** | v2.0 (2026) | Full-stack scaffolding, EDA script generation, FastAPI backend architecture, React dashboard development, unit test authoring | Engineering co-pilot across Python & React codebase | All generated code was executed, debugged, and verified against real dataset training runs by the student team. |
| **Scikit-Learn** | 1.6.1 | Pipeline creation, GridSearchCV, cross-validation, SVM and Decision Tree training | Core Machine Learning modeling | Cross-validation logs, confusion matrices, and test metrics verified programmatically. |
| **FastAPI / Uvicorn** | 0.128.8 / 0.39.0 | RESTful API server, Pydantic v2 validation | Backend web serving | Tested via pytest TestClient and live browser integration. |
| **Vite / React** | 6.4.3 / 18.3.1 | Frontend dashboard and interactive user interface | Client-side application | Validated with vitest, component tests, and responsive screen audits. |

**Academic Integrity Attestation**: The student group confirms that all AI-assisted code was thoroughly reviewed, tested, and validated. No metrics, cross-validation scores, or prediction probabilities were fabricated.
