from datetime import datetime, timezone
import json
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .baseline import predict
from .config import settings
from .faithfulness import area_under_curve, perturbation_curve
from .explanations import explain_text
from .metrics import agreement_metrics
from .risk import critical_risk_metrics
from .transformer_inference import TransformerInferenceService


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class StatusResponse(BaseModel):
    project: str
    dataset_available: bool
    baseline_available: bool
    transformer_available: bool
    translations_available: bool
    evaluation_artifacts_available: bool


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.max_input_length)
    language: str = "en"
    model: str = "baseline"


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model: str
    language: str


class ExplanationRequest(PredictionRequest):
    segments: list[str] | None = None


class FaithfulnessRequest(PredictionRequest):
    segments: list[str] = Field(min_length=1)
    ranked_indices: list[int] | None = None
    mode: str = "deletion"


class ConsistencyRequest(BaseModel):
    english_predictions: list[str] = Field(min_length=1)
    translated_predictions: list[str] = Field(min_length=1)


transformer_service = TransformerInferenceService(settings.transformer_model_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    transformer_service.load()
    yield


app = FastAPI(
    title="SentinelNLP API",
    description="Research API for multilingual vulnerability-severity reliability.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.project_name,
        "status": "ok",
        "message": "SentinelNLP API is running",
        "docs": "/docs",
        "health": "/api/health",
        "status_endpoint": "/api/status",
    }


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        project=settings.project_name,
        dataset_available=(settings.data_dir / "processed").exists(),
        baseline_available=(settings.baseline_model_path / "model.joblib").is_file(),
        transformer_available=transformer_service.available,
        translations_available=(settings.data_dir / "translations").exists(),
        evaluation_artifacts_available=settings.artifact_dir.exists(),
    )


def _baseline_path() -> Path:
    path = settings.baseline_model_path / "model.joblib"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Baseline model is not trained")
    return path


def _prediction(request: PredictionRequest) -> PredictionResponse:
    if request.model not in {"baseline", "xlmr"}:
        raise HTTPException(status_code=400, detail="Unsupported model")
        
    if request.model == "xlmr":
        try:
            pred_label, probabilities = transformer_service.predict(request.text)
            return PredictionResponse(
                prediction=pred_label,
                probabilities=probabilities,
                model="xlm-roberta-base",
                language=request.language,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=f"XLM-R inference unavailable: {exc}") from exc

    predictions, probabilities = predict(_baseline_path(), [request.text])
    model = joblib.load(_baseline_path())
    return PredictionResponse(
        prediction=predictions[0],
        probabilities={label: float(value) for label, value in zip(model.classes_, probabilities[0])},
        model=request.model,
        language=request.language,
    )


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {
        "models": [
            {"id": "baseline", "name": "TF-IDF + Logistic Regression", "available": (settings.baseline_model_path / "model.joblib").is_file()},
            {"id": "xlmr", "name": "XLM-RoBERTa-base", "available": transformer_service.available},
        ]
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    return _prediction(request)


@app.post("/api/explain")
def explain(request: ExplanationRequest) -> dict[str, Any]:
    prediction = _prediction(request)
    try:
        explanation = explain_text(_baseline_path(), request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {**prediction.model_dump(), **explanation}


@app.post("/api/evaluate/faithfulness")
def evaluate_faithfulness(request: FaithfulnessRequest) -> dict[str, Any]:
    if request.model != "baseline":
        raise HTTPException(status_code=503, detail="Only an available model can run faithfulness evaluation")
    model_path = _baseline_path()
    model = joblib.load(model_path)
    ranked = request.ranked_indices or list(range(len(request.segments)))
    if sorted(ranked) != list(range(len(request.segments))):
        raise HTTPException(status_code=400, detail="ranked_indices must contain each segment index once")
    curve = perturbation_curve(
        request.segments,
        ranked,
        lambda text: float(model.predict_proba([text])[0][list(model.classes_).index(model.predict([request.text])[0])]),
        request.mode,
    )
    return {"mode": request.mode, "curve": curve, "area_under_curve": area_under_curve(curve)}


@app.post("/api/evaluate/consistency")
def evaluate_consistency(request: ConsistencyRequest) -> dict[str, Any]:
    try:
        return agreement_metrics(request.english_predictions, request.translated_predictions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    path = settings.artifact_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "not_evaluated"}


@app.get("/api/baseline-metrics")
def baseline_metrics() -> dict[str, Any]:
    path = settings.artifact_dir / "baseline_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "not_evaluated"}


@app.get("/api/dataset/summary")
def dataset_summary() -> dict[str, Any]:
    path = settings.data_dir / "processed" / "nvd_records.json"
    if not path.is_file():
        return {"status": "not_available"}
    records = json.loads(path.read_text(encoding="utf-8"))
    distribution: dict[str, int] = {}
    for record in records:
        distribution[record["severity"]] = distribution.get(record["severity"], 0) + 1
    return {"status": "available", "record_count": len(records), "class_distribution": distribution}


@app.get("/api/faithfulness-summary")
def faithfulness_summary() -> dict[str, Any]:
    path = settings.artifact_dir / "explanation_faithfulness_summary.json"
    if not path.is_file():
        path = settings.artifact_dir / "expanded_faithfulness_recomputed.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "not_evaluated"}


@app.get("/api/evaluate/critical-risk")
def critical_risk() -> dict[str, Any]:
    path = settings.artifact_dir / "critical-risk.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "not_evaluated"}


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    path = settings.artifact_dir / "experiments.json"
    return {"experiments": json.loads(path.read_text(encoding="utf-8"))} if path.is_file() else {"experiments": []}


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict[str, Any]:
    if Path(experiment_id).name != experiment_id:
        raise HTTPException(status_code=400, detail="Invalid experiment identifier")
    path = settings.artifact_dir / "experiments" / f"{experiment_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Experiment not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/exports/{artifact_name}")
def export_artifact(artifact_name: str) -> Any:
    if Path(artifact_name).name != artifact_name or Path(artifact_name).suffix not in {".json", ".csv"}:
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    path = settings.artifact_dir / artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return path.read_text(encoding="utf-8")
