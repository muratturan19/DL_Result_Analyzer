# -*- coding: utf-8 -*-

import base64
import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)

MAX_TRAINING_CODE_CHARS = int(os.getenv("TRAINING_CODE_PREVIEW_CHARS", "4000"))

app = FastAPI(title="DL_Result_Analyzer", version="1.0.0")

# CORS - React frontend için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class YOLOMetrics(BaseModel):
    """YOLO eğitim sonuçları model"""
    precision: float
    recall: float
    map50: float
    map50_95: float
    loss: float
    epochs: int
    batch_size: int
    learning_rate: float
    iou_threshold: Optional[float] = 0.5
    conf_threshold: Optional[float] = 0.5

class ActionRecommendation(BaseModel):
    """LLM aksiyon önerisi öğesi."""

    module: str
    problem: str
    evidence: str
    recommendation: str
    expected_gain: str
    validation_plan: str


class AIAnalysis(BaseModel):
    """LLM analiz sonucu"""

    summary: str
    strengths: List[str]
    weaknesses: List[str]
    actions: List[ActionRecommendation]
    risk: str  # "low", "medium", "high"
    deploy_profile: Dict[str, Any]
    notes: Optional[str] = None
    calibration: Optional[Dict[str, Any]] = None

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {"message": "DL_Result_Analyzer API", "version": "1.0.0"}

@app.post("/api/upload/results")
async def upload_results(
    results_csv: UploadFile = File(...),
    config_yaml: Optional[UploadFile] = File(None),
    graphs: Optional[List[UploadFile]] = File(None),
    best_model: Optional[UploadFile] = File(None),
    training_code: Optional[UploadFile] = File(None),
    project_name: Optional[str] = Form(None),
    short_description: Optional[str] = Form(None),
    class_count: Optional[str] = Form(None),
    training_method: Optional[str] = Form(None),
    project_focus: Optional[str] = Form(None),
    llm_provider: str = Form("claude"),
):
    """
    YOLO eğitim sonuçlarını upload et
    - results.csv: metrics
    - args.yaml: config
    - *.png: graphs (confusion_matrix, F1_curve, etc.)
    """
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = Path(results_csv.filename or "results.csv").name
    csv_path = uploads_dir / csv_filename

    logger.info(
        "Upload request received: csv=%s yaml=%s graphs=%s best_model=%s training_code=%s",
        csv_filename,
        config_yaml.filename if config_yaml else None,
        len(graphs or []),
        best_model.filename if best_model else None,
        training_code.filename if training_code else None,
    )

    def _clean_str(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    cleaned_project_name = _clean_str(project_name)
    cleaned_description = _clean_str(short_description)
    cleaned_method = _clean_str(training_method)
    cleaned_focus = _clean_str(project_focus)

    cleaned_class_count: Optional[int] = None
    if class_count is not None:
        try:
            cleaned_class_count = int(class_count)
        except (TypeError, ValueError):
            logger.warning("Geçersiz class_count değeri alındı: %s", class_count)

    try:
        csv_bytes = await results_csv.read()
        csv_path.write_bytes(csv_bytes)
        logger.info("CSV dosyası kaydedildi: %s", csv_path)
    except Exception as exc:
        logger.exception("CSV dosyası kaydedilemedi: %s", csv_filename)
        raise HTTPException(status_code=500, detail=f"CSV kaydedilemedi: {exc}") from exc

    yaml_path: Optional[Path] = None
    if config_yaml:
        yaml_filename = Path(config_yaml.filename or "args.yaml").name
        yaml_path = uploads_dir / yaml_filename
        try:
            yaml_bytes = await config_yaml.read()
            yaml_path.write_bytes(yaml_bytes)
            logger.info("YAML dosyası kaydedildi: %s", yaml_path)
        except Exception as exc:
            logger.exception("YAML dosyası kaydedilemedi: %s", yaml_filename)
            raise HTTPException(status_code=500, detail=f"YAML kaydedilemedi: {exc}") from exc

    saved_graphs: List[str] = []
    if graphs:
        graphs_dir = uploads_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        for graph in graphs:
            graph_filename = Path(graph.filename or "graph.png").name
            graph_path = graphs_dir / graph_filename
            try:
                graph_bytes = await graph.read()
                graph_path.write_bytes(graph_bytes)
                saved_graphs.append(graph_filename)
                logger.info("Grafik kaydedildi: %s", graph_path)
            except Exception as exc:
                logger.exception("Grafik kaydedilemedi: %s", graph_filename)
                raise HTTPException(status_code=500, detail=f"Grafik kaydedilemedi: {exc}") from exc

    best_model_path: Optional[Path] = None
    if best_model:
        best_model_dir = uploads_dir / "models"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_filename = Path(best_model.filename or "best.pt").name
        best_model_path = best_model_dir / best_model_filename
        try:
            model_bytes = await best_model.read()
            best_model_path.write_bytes(model_bytes)
            logger.info("best.pt kaydedildi: %s", best_model_path)
        except Exception as exc:
            logger.exception("best.pt kaydedilemedi: %s", best_model_filename)
            raise HTTPException(status_code=500, detail=f"best.pt kaydedilemedi: {exc}") from exc

    training_code_path: Optional[Path] = None
    training_code_excerpt: Optional[str] = None
    training_code_filename: Optional[str] = None
    if training_code:
        training_dir = uploads_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        training_code_filename = Path(training_code.filename or "training_code.py").name
        training_code_path = training_dir / training_code_filename
        try:
            code_bytes = await training_code.read()
            training_code_path.write_bytes(code_bytes)
            decoded_code = code_bytes.decode("utf-8", errors="replace")
            training_code_excerpt = decoded_code[:MAX_TRAINING_CODE_CHARS]
            logger.info(
                "Eğitim kodu kaydedildi: %s (önizleme=%s karakter)",
                training_code_path,
                len(training_code_excerpt or ""),
            )
        except Exception as exc:
            logger.exception("Eğitim kodu kaydedilemedi: %s", training_code_filename)
            raise HTTPException(status_code=500, detail=f"Training code kaydedilemedi: {exc}") from exc

    try:
        from app.parsers.yolo_parser import YOLOResultParser
        from app.analyzers.llm_analyzer import LLMAnalyzer

        parser = YOLOResultParser(csv_path, yaml_path)
        metrics = parser.parse_metrics()
        config = parser.parse_config()
        history = parser.parse_training_curves()

        logger.info(
            "Metrix ve konfigürasyon parse edildi: metrics_keys=%s config_keys=%s",
            sorted(metrics.keys()),
            sorted(config.keys()),
        )

        project_context = {
            "project_name": cleaned_project_name,
            "short_description": cleaned_description,
            "class_count": cleaned_class_count,
            "training_method": cleaned_method,
            "project_focus": cleaned_focus,
        }
        project_context = {k: v for k, v in project_context.items() if v not in (None, "")}

        training_code_context: Optional[Dict[str, str]] = None
        if training_code_filename:
            training_code_context = {
                "filename": training_code_filename,
                "excerpt": training_code_excerpt or "",
            }

        enriched_config: Dict[str, object] = dict(config)
        if project_context:
            enriched_config["project_context"] = project_context
        if training_code_context:
            enriched_config["training_code"] = training_code_context

        artefacts_info: Dict[str, Dict[str, object]] = {
            "results_csv": {"filename": csv_filename, "available": True},
            "args_yaml": {
                "filename": yaml_path.name if yaml_path else None,
                "available": yaml_path is not None,
            },
            "graphs": {
                "filenames": saved_graphs,
                "available": bool(saved_graphs),
            },
            "best_model": {
                "filename": best_model_path.name if best_model_path else None,
                "available": best_model_path is not None,
            },
            "training_code": {
                "filename": training_code_filename,
                "available": bool(training_code_filename),
            },
        }

        analysis = {}
        try:
            # Frontend'ten gelen provider'ı kullan, fallback olarak env'den oku
            provider = llm_provider or os.getenv("LLM_PROVIDER", "claude")
            # Geçerli provider kontrolü
            if provider not in ["claude", "openai"]:
                logger.warning("Geçersiz LLM provider: %s, claude kullanılacak", provider)
                provider = "claude"

            analyzer = LLMAnalyzer(provider=provider)  # type: ignore[arg-type]
            logger.info("LLM analizi başlatılıyor: provider=%s", provider)
            analysis = analyzer.analyze(
                metrics,
                enriched_config,
                project_context=project_context,
                training_code=training_code_context,
                history=history,
                artefacts=artefacts_info,
            )
            logger.info("LLM analizi tamamlandı: provider=%s", provider)
        except Exception as exc:
            logger.exception("LLM analizi başarısız oldu")
            analysis = {
                "summary": "LLM analizi gerçekleştirilemedi.",
                "strengths": [],
                "weaknesses": [],
                "actions": [],
                "risk": "medium",
                "deploy_profile": {},
                "notes": str(exc),
                "error": str(exc),
            }

        return {
            "status": "success",
            "metrics": metrics,
            "config": enriched_config,
            "history": history,
            "analysis": analysis,
            "project": project_context,
            "training_code": training_code_context,
            "files": {
                "csv": csv_filename,
                "yaml": yaml_path.name if yaml_path else None,
                "graphs": saved_graphs,
                "best_model": best_model_path.name if best_model_path else None,
                "training_code": training_code_path.name if training_code_path else None,
            },
            "artefacts": artefacts_info,
        }
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Dosya veya veri hatası nedeniyle upload başarısız oldu")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Beklenmeyen bir hata oluştu")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
@app.post("/api/optimize/thresholds")
async def optimize_thresholds(
    best_model: UploadFile = File(...),
    data_yaml: UploadFile = File(...),
    iou_range: str = Form(...),
    conf_range: str = Form(...),
):
    """Temporarily disabled until real YOLO evaluation is implemented."""

    logger.warning(
        "Threshold optimizer endpoint was called but the feature is disabled until real YOLO"
        " evaluation is implemented."
    )
    raise HTTPException(
        status_code=501,
        detail=(
            "Threshold optimizasyonu backend'de henüz gerçek YOLO değerlendirmesiyle"
            " entegre edilmedi. Güvenilir sonuçlar için bu özellik devre dışıdır."
        ),
    )


def _parse_range_payload(payload: str, name: str) -> Dict[str, float]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid client input
        raise HTTPException(status_code=400, detail=f"{name} parametresi JSON formatında olmalıdır.") from exc

    for key in ("start", "end", "step"):
        if key not in data:
            raise HTTPException(status_code=400, detail=f"{name}.{key} değeri zorunludur.")

    start = float(data["start"])
    end = float(data["end"])
    step = float(data["step"])

    if step <= 0:
        raise HTTPException(status_code=400, detail=f"{name}.step değeri pozitif olmalıdır.")
    if end < start:
        raise HTTPException(status_code=400, detail=f"{name}.end, {name}.start değerinden küçük olamaz.")

    return {"start": start, "end": end, "step": step}


def _build_range_values(range_payload: Dict[str, float]) -> List[float]:
    values: List[float] = []
    current = range_payload["start"]
    # Prevent floating point accumulation issues by iterating a bounded number of steps
    max_iterations = int(((range_payload["end"] - range_payload["start"]) / range_payload["step"]) + 2)
    for _ in range(max_iterations):
        if current > range_payload["end"] + 1e-9:
            break
        values.append(round(current, 3))
        current += range_payload["step"]
    if not values or values[-1] < range_payload["end"] - 1e-9:
        values.append(round(range_payload["end"], 3))
    return values


def _simulate_metrics(iou: float, conf: float, randomness: float) -> Dict[str, float]:
    """Create deterministic pseudo metrics for the grid search heatmap."""

    # Base surfaces that roughly prefer mid IoU and moderate confidence
    recall_center = 0.9 - 0.55 * abs(iou - 0.45) - 0.75 * abs(conf - 0.3)
    precision_center = 0.82 - 0.4 * abs(iou - 0.55) - 0.6 * abs(conf - 0.4)

    # Blend in small deterministic randomness to avoid a perfectly flat surface
    recall = max(0.35, min(0.99, recall_center + (randomness - 0.5) * 0.05))
    precision = max(0.35, min(0.99, precision_center + (0.5 - randomness) * 0.04))

    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


@app.post("/api/optimize/thresholds")
async def optimize_thresholds(
    best_model: UploadFile = File(...),
    data_yaml: UploadFile = File(...),
    iou_range: str = Form(...),
    conf_range: str = Form(...),
):
    """Run a deterministic IoU/Confidence sweep and produce heatmaps for the UI."""

    uploads_dir = Path("uploads/optimizer")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    model_bytes = await best_model.read()
    data_bytes = await data_yaml.read()

    model_filename = Path(best_model.filename or "best.pt").name
    data_filename = Path(data_yaml.filename or "data.yaml").name

    model_path = uploads_dir / model_filename
    data_path = uploads_dir / data_filename
    model_path.write_bytes(model_bytes)
    data_path.write_bytes(data_bytes)

    iou_payload = _parse_range_payload(iou_range, "iou_range")
    conf_payload = _parse_range_payload(conf_range, "conf_range")

    iou_values = _build_range_values(iou_payload)
    conf_values = _build_range_values(conf_payload)

    # Deterministic randomness using the model weights digest
    digest = sha256(model_bytes + data_bytes).hexdigest()
    seed = int(digest[:8], 16)

    heatmap_rows: List[List[Dict[str, float]]] = []
    flat_results: List[Dict[str, float]] = []
    best_candidate: Dict[str, float] | None = None

    for i, iou in enumerate(iou_values):
        row: List[Dict[str, float]] = []
        for j, conf in enumerate(conf_values):
            # Compute deterministic pseudo randomness per cell
            randomness = ((seed + i * 13 + j * 7) % 1000) / 1000
            metrics = _simulate_metrics(iou, conf, randomness)
            cell = {"iou": iou, "confidence": conf, **metrics}
            row.append(cell)
            flat_results.append(cell)

            if not best_candidate or metrics["f1"] > best_candidate["f1"]:
                best_candidate = cell
        heatmap_rows.append(row)

    if not best_candidate:
        raise HTTPException(status_code=500, detail="Grid search sonuçları üretilemedi.")

    production_payload = {
        "model": model_filename,
        "data_config": data_filename,
        "thresholds": {
            "confidence": best_candidate["confidence"],
            "iou": best_candidate["iou"],
        },
        "expected_metrics": {
            "recall": best_candidate["recall"],
            "precision": best_candidate["precision"],
            "f1": best_candidate["f1"],
        },
    }

    yaml_text = yaml.safe_dump(production_payload, sort_keys=False, allow_unicode=True)
    encoded_yaml = base64.b64encode(yaml_text.encode("utf-8")).decode("utf-8")

    return {
        "status": "success",
        "heatmap": {
            "rows": heatmap_rows,
            "values": flat_results,
            "iou_values": iou_values,
            "confidence_values": conf_values,
        },
        "best": best_candidate,
        "production_config": {
            "filename": "production_config.yaml",
            "yaml": yaml_text,
            "base64": encoded_yaml,
        },
    }

@app.post("/api/analyze/metrics")
async def analyze_metrics(metrics: YOLOMetrics):
    """
    Metrics'i analiz et ve AI önerileri üret
    """
    try:
        # TODO: LLM integration
        # 1. Rule-based quick check
        # 2. LLM prompt oluştur
        # 3. Claude/GPT'ye gönder
        # 4. Response'u parse et
        
        # Placeholder analysis
        analysis = AIAnalysis(
            summary="Model performance is moderate with room for improvement.",
            strengths=["Good precision", "Stable training loss"],
            weaknesses=["Recall hedefin altında", "Doğrulama verisi sınırlı"],
            actions=[
                ActionRecommendation(
                    module="veri kalitesi",
                    problem="Recall metrikleri düşük seyrediyor",
                    evidence="Son 5 epoch boyunca recall %76 civarında plato yaptı",
                    recommendation="Etiketleme yönergelerini gözden geçirip sınıf başına 50 ek örnek toplayın",
                    expected_gain="Recall değerinde %6-8 artış",
                    validation_plan="Yeni veri ile yeniden eğitimden sonra hold-out sette recall ≥ %82",
                ),
                ActionRecommendation(
                    module="eğitim",
                    problem="mAP@0.5 hedefe yaklaşsa da stabil değil",
                    evidence="Validation mAP epoch 80 sonrası düşüşte",
                    recommendation="Cosine LR schedule ile 40 ek epoch çalıştırın",
                    expected_gain="mAP@0.5 değerinde kalıcı %3 artış",
                    validation_plan="Ek eğitim sonrası 3 ardışık denemede mAP@0.5 ≥ %82",
                ),
            ],
            risk="medium",
            deploy_profile={
                "release_decision": "hold",
                "rollout_strategy": "Ek veri toplama tamamlanana kadar staging'de kal",
                "monitoring_plan": "Yeni veri ile yeniden eğitim sonrası 2 hafta canlı izleme",
            },
            notes="LLM analizi devreye alındığında gerçek öneriler ile güncellenecek.",
        )

        return analysis
    except Exception as exc:
        logger.exception("Metric analizi başarısız oldu")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/compare")
async def compare_runs(run_ids: List[str]):
    """
    Birden fazla eğitimi karşılaştır
    """
    # TODO: Implementation
    return {"message": "Comparison feature coming soon"}

@app.get("/api/history")
async def get_history():
    """
    Geçmiş analizleri listele
    """
    # TODO: Database integration (SQLite?)
    return {"runs": []}

# =============================================================================
# LLM ANALYZER (Ayrı dosyada olacak: analyzers/llm_analyzer.py)
# =============================================================================

# from anthropic import Anthropic
# from openai import OpenAI

# async def analyze_with_claude(metrics: dict) -> str:
#     client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
#     
#     prompt = f"""
#     YOLO11 model eğitim sonuçları:
#     - Precision: {metrics['precision']}
#     - Recall: {metrics['recall']}
#     - mAP@0.5: {metrics['map50']}
#     
#     Lütfen bu sonuçları analiz et ve:
#     1. Güçlü yönleri belirt
#     2. Zayıf yönleri açıkla
#     3. Somut aksiyon önerileri ver (IoU, LR, augmentation vb.)
#     """
#     
#     response = client.messages.create(
#         model="claude-sonnet-4-5-20250929",
#         max_tokens=1024,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     
#     return response.content[0].text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
