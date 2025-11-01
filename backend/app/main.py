# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)

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

class AIAnalysis(BaseModel):
    """LLM analiz sonucu"""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    action_items: List[str]
    risk_level: str  # "low", "medium", "high"

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
    llm_provider: str = "claude"
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
        "Upload request received: csv=%s yaml=%s graphs=%s",
        csv_filename,
        config_yaml.filename if config_yaml else None,
        len(graphs or []),
    )

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

    try:
        from app.parsers.yolo_parser import YOLOResultParser
        from app.analyzers.llm_analyzer import LLMAnalyzer

        parser = YOLOResultParser(csv_path, yaml_path)
        metrics = parser.parse_metrics()
        config = parser.parse_config()

        logger.info(
            "Metrix ve konfigürasyon parse edildi: metrics_keys=%s config_keys=%s",
            sorted(metrics.keys()),
            sorted(config.keys()),
        )

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
            analysis = analyzer.analyze(metrics, config)
            logger.info("LLM analizi tamamlandı: provider=%s", provider)
        except Exception as exc:
            logger.exception("LLM analizi başarısız oldu")
            analysis = {
                "summary": "LLM analizi gerçekleştirilemedi.",
                "strengths": [],
                "weaknesses": [],
                "action_items": [],
                "risk_level": "medium",
                "error": str(exc),
            }

        return {
            "status": "success",
            "metrics": metrics,
            "config": config,
            "analysis": analysis,
            "files": {
                "csv": csv_filename,
                "yaml": yaml_path.name if yaml_path else None,
                "graphs": saved_graphs,
            },
        }
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Dosya veya veri hatası nedeniyle upload başarısız oldu")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Beklenmeyen bir hata oluştu")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
            strengths=["Good precision", "Low loss"],
            weaknesses=["Low recall", "May need more data"],
            action_items=[
                "Lower IoU threshold to 0.3",
                "Increase training epochs to 200",
                "Add more augmentation"
            ],
            risk_level="medium"
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
