# Online Shopping Purchase Intention Prediction Using Machine Learning

> **Year 2 Semester 1 University Artificial Intelligence and Machine Learning Group Assignment**  
> Primary Model: **Support Vector Machine (SVM with RBF Kernel)**  
> Comparison Model: **Decision Tree Classifier**  
> Dataset: **UCI Online Shoppers Purchasing Intention Dataset (`online_shoppers_intention.csv`, 12,330 rows, 18 columns)**

---

## 📌 Project Overview
This project delivers a complete, end-to-end Machine Learning web application designed to predict online customer purchase intention in real time. It compares a primary Support Vector Machine (SVC) against a Decision Tree Classifier using an untouched 20% test partition and 5-fold Stratified Cross-Validation on the remaining 80% training data.

The system features:
- **FastAPI Backend**: Serving RESTful endpoints for health checks, dataset distribution, metric summaries, confusion matrices, ROC/PR curves, permutation feature importances, and real-time probabilistic inference (`predict_proba()`).
- **React.js Dashboard**: An academic, data-science UI built with Vite, React 18, and Lucide icons, responsive across Desktop, Laptop, Tablet, and Mobile devices.
- **Jupyter Notebooks**: Four self-contained, reproducible notebooks documenting data understanding, EDA, model training, and evaluation.
- **Academic Documentation**: Transparent ethical considerations, business trade-off analysis, and a university AI tool usage declaration.

---

## 📊 Dataset & Features
- **Total Sessions**: 12,330
- **Features**: 17 attributes (14 numerical, 2 categorical, 1 boolean)
- **Target Variable**: `Revenue` (`False` = 10,422 sessions / 84.53%, `True` = 1,908 sessions / 15.47%)
- **Class Imbalance**: Positive purchase rate is **15.47%**. Accuracy alone is misleading; **F1-Score**, **Precision**, **Recall**, and **ROC-AUC** are the primary evaluation criteria.

### Feature Table
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Administrative` | Integer | Administrative pages visited |
| `Administrative_Duration` | Float | Seconds spent on administrative pages |
| `Informational` | Integer | Informational pages visited |
| `Informational_Duration` | Float | Seconds spent on informational pages |
| `ProductRelated` | Integer | Product-related pages visited |
| `ProductRelated_Duration` | Float | Seconds spent on product pages |
| `BounceRates` | Float | Average bounce rate of pages visited |
| `ExitRates` | Float | Average exit rate of pages visited |
| `PageValues` | Float | Google Analytics value of pages prior to transaction |
| `SpecialDay` | Float | Closeness of browsing date to a special day/holiday |
| `OperatingSystems` | Integer | Operating system code (1-8) |
| `Browser` | Integer | Browser code (1-13) |
| `Region` | Integer | Geographic region code (1-9) |
| `TrafficType` | Integer | Traffic source channel code (1-20) |
| `Month` | String | Calendar month of session |
| `VisitorType` | String | Returning_Visitor, New_Visitor, Other |
| `Weekend` | Boolean | True if browsing session was on weekend |
| `Revenue` | Boolean | Target: True if session completed transaction |

---

## ⚙️ Preprocessing & Data Splitting
1. **Stratified 80/20 Train-Test Split**:
   - 9,864 Training Sessions (80%)
   - 2,466 Testing Sessions (20%) — **Remains strictly untouched throughout training and tuning**
   - `stratify=y`, `random_state=42`
2. **Scikit-Learn Pipeline & ColumnTransformer**:
   - `StandardScaler()` applied to all 14 numerical features for SVM margin distance calculations.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to `Month` and `VisitorType`.
   - `Weekend` converted to numeric 0/1.

---

## 📈 Real Calculated Model Results
*Trained directly on `online_shoppers_intention.csv` with zero fabricated metrics.*

### Quantitative Benchmark Table (Untouched Test Partition, N=2,466)
| Metric | SVM (Primary Model) | Decision Tree (Comparison) | Superior Algorithm |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.8747** | 0.8500 | **SVM** |
| **Precision** | **0.5759** | 0.5098 | **SVM** |
| **Recall (Sensitivity)** | 0.7251 | **0.8194** | **Decision Tree** |
| **F1-Score** | **0.6419** | 0.6285 | **SVM** |
| **ROC-AUC** | **0.8997** | 0.8928 | **SVM** |
| **5-Fold CV Mean F1** | **0.6655** | 0.6344 | **SVM** |
| **5-Fold CV Std F1** | 0.0164 | **0.0069** | **Decision Tree** |

### Hyperparameters Discovered via 5-Fold Stratified GridSearchCV
- **SVM**: `C=100.0`, `gamma=0.001`, `kernel='rbf'`, `class_weight='balanced'`
- **Decision Tree**: `criterion='gini'`, `max_depth=7`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight='balanced'`

### Confusion Matrices
- **SVM**: TN = 1,880, FP = 204, FN = 105, TP = 277
- **Decision Tree**: TN = 1,783, FP = 301, FN = 69, TP = 313

---

## 🚀 How to Run the Project

### 1. Prerequisites
- Python 3.9+ installed
- Node.js 18+ and npm installed

### 2. Backend Setup & Startup
```bash
# Navigate to project root
cd /path/to/AIML_Project

# Activate virtual environment
source .venv/bin/activate

# Install backend dependencies (if needed)
pip install -r backend/requirements.txt

# Start FastAPI backend server
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The FastAPI backend will start at `http://127.0.0.1:8000`. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup & Startup
In a second terminal:
```bash
# Navigate to frontend
cd frontend

# Install dependencies (if not already installed)
npm install

# Start Vite dev server
npm run dev
```
The frontend dashboard will be live at `http://localhost:5173`.

---

## 🧪 Running Tests

### Backend Unit & Integration Tests (pytest)
```bash
PYTHONPATH=. pytest backend/tests/test_backend.py -v
```
Verifies health endpoint, dataset loading, preprocessing, model loading, metrics integrity, and prediction input validation.

### Frontend Component Tests (Vitest)
```bash
cd frontend
npm test
```
Verifies loading states, dashboard statistics rendering, error states, and live prediction form interactions.

---

## 🌐 API Endpoints Reference
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Operational health and model readiness check |
| `GET` | `/api/dataset/summary` | Dataset metadata, total rows, train/test counts |
| `GET` | `/api/dataset/distribution` | Numerical feature 5-number summaries & categorical counts |
| `GET` | `/api/dataset/rows` | Searchable, paginated dataset records for UI explorer |
| `GET` | `/api/models/results` | Complete test evaluation metrics & cross-validation scores |
| `GET` | `/api/models/comparison` | Comparative metrics table & best model determination |
| `GET` | `/api/evaluation/confusion-matrix` | Test confusion matrices and classification reports |
| `GET` | `/api/evaluation/roc` | ROC curve coordinates (FPR, TPR) & AUC values |
| `GET` | `/api/evaluation/precision-recall` | Precision-Recall curve coordinates & baseline |
| `GET` | `/api/features/importance` | SVM Permutation Importance & DT Gini Feature Importance |
| `POST` | `/api/predict` | Single session inference predicting purchase probability |
| `POST` | `/api/retrain` | Triggers retraining & 5-fold cross-validation of both models |

---

## ⚖️ Ethical Considerations
- **Data Privacy**: No Personally Identifiable Information (PII) is collected or processed. Session logs comply with consent standards.
- **Fairness & Bias**: Demographic proxies (e.g., Browser, Region) must not be exploited for price discrimination.
- **Class Imbalance**: Balanced class weights protect minority purchasers from statistical omission.
- **Human in the loop**: Heuristic probabilities (`predict_proba`) must assist rather than replace human commercial judgment.

---

## 🤖 AI Tool Usage Declaration
All AI contributions (Google Antigravity / Gemini 3.8 Flash) were strictly utilized for design, architecture, and coding assistance. All statistical outputs, cross-validation runs, and evaluation metrics were generated by executing genuine scikit-learn models on `online_shoppers_intention.csv` and thoroughly verified by the student team.
