from fastapi import APIRouter, Query
from typing import Optional, List
from backend.app.services.data_service import (
    get_dataset_summary,
    get_feature_distribution_data,
    get_dataset_rows,
)

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/summary")
def api_dataset_summary():
    """Retrieve full dataset dimensions, train/test split, and imbalance metrics."""
    return get_dataset_summary()


@router.get("/distribution")
def api_dataset_distribution():
    """Retrieve numerical 5-number summaries and categorical frequencies."""
    return get_feature_distribution_data()


@router.get("/rows")
def api_dataset_rows(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: Optional[str] = Query(default=None),
    sort_by: Optional[str] = Query(default=None),
    sort_order: str = Query(default="asc", pattern="^(asc|desc)$"),
    columns: Optional[str] = Query(default=None, description="Comma-separated column names"),
):
    """Retrieve paginated, filtered, and searchable records for the Dataset Explorer UI."""
    col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    return get_dataset_rows(
        page=page,
        page_size=page_size,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        columns=col_list,
    )
