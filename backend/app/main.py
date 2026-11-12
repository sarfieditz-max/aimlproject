import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import PLOTS_DIR
from backend.app.routes import health, dataset, prediction, models, evaluation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("online_shoppers_api")

app = FastAPI(
    title="Online Shopping Purchase Intention Prediction API",
    description="University AI/ML Assignment Backend implementing SVM and Decision Tree classifiers.",
    version="1.0.0",
)

# CORS configuration - specifically permit frontend development origins
ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler to prevent internal stack trace leakage
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An error occurred while processing your request. Please check input parameters or model status.",
            "path": request.url.path,
        },
    )

# Static files for pre-generated plots
app.mount("/artifacts/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

# Include modular API routers
app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(prediction.router)
app.include_router(models.router)
app.include_router(evaluation.router)


@app.get("/")
def root():
    return {
        "title": "Online Shopping Purchase Intention Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/api/health",
    }
