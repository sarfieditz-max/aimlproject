from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ShopperFeatures(BaseModel):
    Administrative: int = Field(..., ge=0, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(..., ge=0.0, description="Total duration on administrative pages in seconds")
    Informational: int = Field(..., ge=0, description="Number of informational pages visited")
    Informational_Duration: float = Field(..., ge=0.0, description="Total duration on informational pages in seconds")
    ProductRelated: int = Field(..., ge=0, description="Number of product-related pages visited")
    ProductRelated_Duration: float = Field(..., ge=0.0, description="Total duration on product-related pages in seconds")
    BounceRates: float = Field(..., ge=0.0, le=1.0, description="Average bounce rate of pages visited")
    ExitRates: float = Field(..., ge=0.0, le=1.0, description="Average exit rate of pages visited")
    PageValues: float = Field(..., ge=0.0, description="Average page value for visited pages")
    SpecialDay: float = Field(..., ge=0.0, le=1.0, description="Closeness of visiting time to a special day")
    Month: str = Field(..., description="Month of the session (e.g., Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec)")
    OperatingSystems: int = Field(..., ge=1, le=8, description="Operating system identifier")
    Browser: int = Field(..., ge=1, le=13, description="Browser identifier")
    Region: int = Field(..., ge=1, le=9, description="Geographic region identifier")
    TrafficType: int = Field(..., ge=1, le=20, description="Traffic source type identifier")
    VisitorType: str = Field(..., description="Visitor type: Returning_Visitor, New_Visitor, or Other")
    Weekend: bool = Field(..., description="Whether the session was on a weekend")
    model: Optional[str] = Field(default="SVM", description="Model to use: 'SVM' or 'Decision Tree'")


class PredictionResponse(BaseModel):
    prediction: bool
    prediction_label: str
    probability: float
    model: str
    features_received: Dict[str, Any]


class MetricSummary(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_mean_f1: float
    cv_std_f1: float
    cv_mean_accuracy: float
    cv_mean_precision: float
    cv_mean_recall: float
    cv_mean_roc_auc: float
    best_params: Dict[str, Any]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


class ModelComparisonResponse(BaseModel):
    best_model_by_f1: str
    comparison_table: List[Dict[str, Any]]
    models: Dict[str, MetricSummary]
    objective_guidance: str


class DatasetSummaryResponse(BaseModel):
    total_rows: int
    total_columns: int
    training_samples: int
    testing_samples: int
    target_distribution: Dict[str, int]
    target_percentages: Dict[str, float]
    missing_values_count: int
    columns: List[str]
    column_types: Dict[str, str]
    numerical_features: List[str]
    categorical_features: List[str]
    boolean_features: List[str]


class RetrainRequest(BaseModel):
    svm_c_values: Optional[List[float]] = None
    svm_gammas: Optional[List[Any]] = None
    dt_max_depths: Optional[List[Optional[int]]] = None
    random_state: Optional[int] = 42


class HealthResponse(BaseModel):
    status: str
    dataset_present: bool
    svm_model_loaded: bool
    dt_model_loaded: bool
    metrics_present: bool
    message: str
