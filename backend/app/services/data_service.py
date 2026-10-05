import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from backend.app.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    ALL_FEATURES,
)
from backend.app.ml.preprocessing import load_dataset


def get_dataset_summary() -> Dict[str, Any]:
    """Provide comprehensive dataset statistics and metadata."""
    df = load_dataset()
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    target_dist = {str(k): int(v) for k, v in target_counts.items()}
    target_pct = {
        str(k): round(float(v / len(df) * 100), 2) for k, v in target_counts.items()
    }

    # 80/20 train/test split sizes
    train_count = int(np.round(len(df) * 0.8))
    test_count = len(df) - train_count

    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "training_samples": train_count,
        "testing_samples": test_count,
        "target_distribution": target_dist,
        "target_percentages": target_pct,
        "missing_values_count": int(df.isnull().sum().sum()),
        "columns": list(df.columns),
        "column_types": col_types,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
    }


def get_dataset_rows(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Retrieve paginated, searchable, and filtered dataset rows."""
    df = load_dataset()

    # Search filter across string columns
    if search:
        search_lower = search.strip().lower()
        mask = (
            df["Month"].astype(str).str.lower().str.contains(search_lower)
            | df["VisitorType"].astype(str).str.lower().str.contains(search_lower)
            | df["Revenue"].astype(str).str.lower().str.contains(search_lower)
        )
        df = df[mask]

    # Sorting
    if sort_by and sort_by in df.columns:
        ascending = sort_order.lower() == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)

    total_records = len(df)
    total_pages = max(1, int(np.ceil(total_records / page_size)))
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Filter columns if requested
    selected_cols = columns if columns else list(df.columns)
    available_cols = [c for c in selected_cols if c in df.columns]

    paged_df = df.iloc[start_idx:end_idx][available_cols]

    # Clean records to Python native types
    records = paged_df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                r[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                r[k] = round(float(v), 4)
            elif isinstance(v, (np.bool_, bool)):
                r[k] = bool(v)

    return {
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": total_pages,
        "columns": available_cols,
        "rows": records,
    }


def get_feature_distribution_data() -> Dict[str, Any]:
    """Provide numerical feature summary stats and categorical value counts."""
    df = load_dataset()
    
    numerical_stats = {}
    for col in NUMERICAL_FEATURES:
        series = df[col]
        numerical_stats[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "q75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }

    categorical_counts = {}
    for col in CATEGORICAL_FEATURES + BOOLEAN_FEATURES + [TARGET_COLUMN]:
        counts = df[col].value_counts().to_dict()
        categorical_counts[col] = {str(k): int(v) for k, v in counts.items()}

    return {
        "numerical_stats": numerical_stats,
        "categorical_counts": categorical_counts,
    }
