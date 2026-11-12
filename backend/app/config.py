import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file path
DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"
if not DATASET_PATH.exists():
    root_dataset = PROJECT_ROOT / "online_shoppers_intention.csv"
    if root_dataset.exists():
        import shutil
        shutil.copy(root_dataset, DATASET_PATH)

# Model paths
SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
DT_MODEL_PATH = MODELS_DIR / "decision_tree_model.joblib"
METRICS_JSON_PATH = METRICS_DIR / "model_results.json"
ROOT_METRICS_JSON_PATH = PROJECT_ROOT / "MODEL_RESULTS.json"

# Feature definitions
TARGET_COLUMN = "Revenue"

NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
]

BOOLEAN_FEATURES = [
    "Weekend",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Valid categorical choices
VALID_MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VALID_VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

# Split & CV parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
PRIMARY_SCORING = "f1"
