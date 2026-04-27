"""FastAPI application: NSD-ISS Clinical Digital Twin Dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.data.patient_store import PatientStore
from app.services.model_registry import ModelRegistry

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Singleton instances ─────────────────────────────────────
model_registry = ModelRegistry()
patient_store = PatientStore()

# ── Directories ─────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"


# ── Lifespan (startup / shutdown) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all data and models at startup."""
    logger.info("=" * 60)
    logger.info("  NSD-ISS Clinical Digital Twin Dashboard — Starting up")
    logger.info("=" * 60)

    # 1. Load patient data from CSVs
    patient_store.load()

    # 2. Load ML models (CatBoost, DeepHit, Graph-DT)
    model_registry.load_all()

    logger.info("=" * 60)
    logger.info("  Startup complete — ready to serve requests")
    logger.info("=" * 60)

    yield  # Application runs here

    logger.info("Shutting down dashboard...")


# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(
    title="NSD-ISS Clinical Digital Twin Dashboard",
    description=(
        "Interactive clinical decision support tool integrating "
        "NSD-ISS biological staging, survival prediction, conformal "
        "uncertainty quantification, and patient similarity graphs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Register API routers ────────────────────────────────────
from app.api.patients import router as patients_router      # noqa: E402
from app.api.staging import router as staging_router        # noqa: E402
from app.api.survival import router as survival_router      # noqa: E402
from app.api.whatif import router as whatif_router           # noqa: E402
from app.api.explain import router as explain_router        # noqa: E402
from app.api.cohort import router as cohort_router          # noqa: E402
from app.api.annotations import router as annotations_router  # noqa: E402
from app.api.wearable import router as wearable_router      # noqa: E402

app.include_router(patients_router, prefix="/api/patients", tags=["Patients"])
app.include_router(staging_router, prefix="/api/staging", tags=["Staging"])
app.include_router(survival_router, prefix="/api/survival", tags=["Survival"])
app.include_router(whatif_router, prefix="/api/whatif", tags=["What-If"])
app.include_router(explain_router, prefix="/api/explain", tags=["Explainability"])
app.include_router(cohort_router, prefix="/api/cohort", tags=["Cohort"])
app.include_router(annotations_router, prefix="/api/annotations", tags=["Annotations"])
app.include_router(wearable_router, prefix="/api/wearable", tags=["Wearable"])


# ── HTML page routes ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Patient list / cohort overview."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/patient/{patno}", response_class=HTMLResponse)
async def patient_view(request: Request, patno: int):
    """Single-patient clinical dashboard."""
    return templates.TemplateResponse(
        "patient.html", {"request": request, "patno": patno}
    )


# ── Health check ────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": model_registry.is_loaded,
        "patients_loaded": len(patient_store.patient_ids),
    }
