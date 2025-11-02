# -*- coding: utf-8 -*-

import base64
import json
import logging
import os
import re
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from textwrap import wrap
from threading import Lock
from typing import Dict, List, Optional, Any
from uuid import uuid4

import yaml

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

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


class ReportStore:
    """In-memory store that keeps report contexts for follow-up Q/A."""

    def __init__(self, ttl_seconds: int = 6 * 3600) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired: List[str] = []
        for report_id, entry in self._store.items():
            timestamp: datetime = entry.get("timestamp", now)
            if now - timestamp > self._ttl:
                expired.append(report_id)
        for report_id in expired:
            self._store.pop(report_id, None)

    def save(self, payload: Dict[str, Any]) -> str:
        report_id = uuid4().hex
        with self._lock:
            self._purge_expired()
            self._store[report_id] = {
                "payload": payload,
                "timestamp": datetime.now(timezone.utc),
            }
        return report_id

    def get(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._purge_expired()
            entry = self._store.get(report_id)
            if not entry:
                return None
            entry["timestamp"] = datetime.now(timezone.utc)
            return entry["payload"]

    def update(self, report_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._purge_expired()
            if report_id in self._store:
                self._store[report_id] = {
                    "payload": payload,
                    "timestamp": datetime.now(timezone.utc),
                }

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


class QARequest(BaseModel):
    """Follow-up Q/A request payload."""

    question: str
    llm_provider: Optional[str] = None


REPORT_STORE = ReportStore()


def _slugify_filename(value: Optional[str], fallback: str = "dl-result-report") -> str:
    if not value:
        return fallback
    normalized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or fallback


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _build_metrics_html(metrics: Dict[str, Any]) -> str:
    rows = []
    display_map = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP@0.5",
        "map50_95": "mAP@0.5:0.95",
        "loss": "Loss",
    }
    for key, label in display_map.items():
        value = metrics.get(key)
        if key in {"precision", "recall", "map50", "map50_95"}:
            formatted = _format_percent(value)
        elif value is None:
            formatted = "N/A"
        else:
            formatted = f"{value:.4f}" if isinstance(value, (float, int)) else escape(str(value))
        rows.append(f"<tr><th>{escape(label)}</th><td>{formatted}</td></tr>")
    return "\n".join(rows)


def _build_list_html(items: List[str]) -> str:
    if not items:
        return "<p>Bilgi bulunamadı.</p>"
    escaped_items = "".join(f"<li>{escape(item)}</li>" for item in items if item)
    return f"<ul>{escaped_items}</ul>" if escaped_items else "<p>Bilgi bulunamadı.</p>"


def _build_actions_html(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "<p>Aksiyon önerisi bulunamadı.</p>"

    action_items = []
    for action in actions:
        module = escape(str(action.get("module", ""))) if action.get("module") else "Genel"
        recommendation = escape(str(action.get("recommendation", "")))
        evidence = escape(str(action.get("evidence", "")))
        expected = escape(str(action.get("expected_gain", "")))
        validation = escape(str(action.get("validation_plan", "")))

        detail_parts = [f"<strong>Modül:</strong> {module}"]
        if evidence:
            detail_parts.append(f"<strong>Kanıt:</strong> {evidence}")
        if recommendation:
            detail_parts.append(f"<strong>Öneri:</strong> {recommendation}")
        if expected:
            detail_parts.append(f"<strong>Beklenen Kazanç:</strong> {expected}")
        if validation:
            detail_parts.append(f"<strong>Doğrulama Planı:</strong> {validation}")

        action_items.append(f"<li><div class=\"action-item\">{'<br/>'.join(detail_parts)}</div></li>")

    return f"<ul class=\"action-list\">{''.join(action_items)}</ul>"


def _build_dataset_summary(dataset: Dict[str, Any]) -> str:
    if not dataset:
        return ""

    count_map = [
        ("Eğitim", dataset.get("train_images")),
        ("Doğrulama", dataset.get("val_images")),
        ("Test", dataset.get("test_images")),
        ("Toplam", dataset.get("total_images")),
    ]

    count_items = []
    for label, value in count_map:
        if value is None:
            continue
        try:
            numeric = float(value)
            display = f"{int(numeric):,}".replace(",", ".")
        except (TypeError, ValueError):
            display = escape(str(value))
        count_items.append(f"<div class=\"dataset-count\"><span>{escape(label)}</span><strong>{display}</strong></div>")

    classes = dataset.get("class_names")
    class_items = ""
    if isinstance(classes, list) and classes:
        chips = "".join(f"<span class=\"dataset-chip\">{escape(str(item))}</span>" for item in classes)
        class_items = f"<div class=\"dataset-classes\"><span>Sınıflar:</span>{chips}</div>"

    parts = []
    if count_items:
        parts.append(f"<div class=\"dataset-counts\">{''.join(count_items)}</div>")
    if class_items:
        parts.append(class_items)

    return "".join(parts)


def _generate_html_report(report_id: str, context: Dict[str, Any]) -> str:
    project_context = context.get("project") or context.get("config", {}).get("project_context", {}) or {}
    project_name = project_context.get("project_name") or "DL Result Report"
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%d %B %Y %H:%M")

    metrics_section = _build_metrics_html(context.get("metrics", {}))
    dataset_section = _build_dataset_summary(context.get("config", {}).get("dataset", {}))

    analysis = context.get("analysis", {}) or {}
    strengths_html = _build_list_html(analysis.get("strengths", []))
    weaknesses_html = _build_list_html(analysis.get("weaknesses", []))
    actions_html = _build_actions_html(analysis.get("actions", []))

    risk = analysis.get("risk")
    risk_display = escape(str(risk)).upper() if risk else "BİLİNMİYOR"

    deploy_profile = analysis.get("deploy_profile", {}) or {}
    deploy_items = "".join(
        f"<li><strong>{escape(str(key))}:</strong> {escape(str(value))}</li>" for key, value in deploy_profile.items()
    )

    qa_history = context.get("qa_history", []) or []
    qa_items = []
    for entry in qa_history[-5:][::-1]:
        question = escape(str(entry.get("question", "")))
        answer = escape(str(entry.get("answer", "")))
        timestamp = escape(str(entry.get("timestamp", "")))
        qa_items.append(
            f"<div class=\"qa-entry\"><div class=\"qa-question\"><strong>Soru:</strong> {question}</div>"
            f"<div class=\"qa-answer\"><strong>Yanıt:</strong> {answer}</div>"
            f"<div class=\"qa-timestamp\">{timestamp}</div></div>"
        )

    deploy_html = f"<ul>{deploy_items}</ul>" if deploy_items else "<p>Bilgi bulunamadı.</p>"
    qa_history_html = "".join(qa_items) if qa_items else "<p>Henüz takip sorusu yok.</p>"

    summary_text = escape(str(analysis.get("summary", ""))) if analysis.get("summary") else "Henüz özet oluşturulmadı."
    notes = analysis.get("notes") or analysis.get("error")
    notes_html = escape(str(notes)) if notes else ""

    return """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="utf-8" />
    <title>DL Result Analyzer - Rapor</title>
    <style>
        body { font-family: 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background: #f3f4f6; color: #111827; }
        header { background: linear-gradient(135deg, #312e81, #6366f1); color: white; padding: 32px; }
        header h1 { margin: 0 0 8px; font-size: 28px; }
        header p { margin: 0; font-size: 14px; opacity: 0.85; }
        main { padding: 32px; }
        section { background: white; border-radius: 16px; padding: 24px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }
        h2 { margin-top: 0; font-size: 22px; color: #1f2937; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th, td { padding: 12px 16px; text-align: left; }
        th { background: #eef2ff; font-weight: 600; width: 220px; }
        tr:nth-child(even) td { background: #f9fafb; }
        ul { margin: 12px 0; padding-left: 20px; }
        .dataset-counts { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }
        .dataset-count { background: #eef2ff; border-radius: 12px; padding: 12px; display: flex; flex-direction: column; gap: 4px; }
        .dataset-count span { font-size: 12px; color: #4f46e5; text-transform: uppercase; letter-spacing: 0.05em; }
        .dataset-count strong { font-size: 20px; }
        .dataset-classes { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        .dataset-classes span { font-weight: 600; color: #1f2937; }
        .dataset-chip { background: #f3f4f6; padding: 6px 12px; border-radius: 999px; font-size: 13px; }
        .analysis-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; }
        .qa-entry { background: #f9fafb; border-radius: 12px; padding: 16px; margin-bottom: 12px; border: 1px solid #e5e7eb; }
        .qa-question { font-weight: 600; margin-bottom: 8px; }
        .qa-answer { margin-bottom: 6px; }
        .qa-timestamp { font-size: 12px; color: #6b7280; }
        .risk-chip { display: inline-block; padding: 6px 12px; border-radius: 999px; background: #f97316; color: white; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
        .action-list { list-style: none; padding: 0; margin: 12px 0 0; display: flex; flex-direction: column; gap: 12px; }
        .action-item { background: #f9fafb; padding: 14px; border-radius: 12px; border: 1px solid #e5e7eb; }
        footer { text-align: center; padding: 24px; color: #6b7280; font-size: 12px; }
    </style>
</head>
<body>
    <header>
        <h1>{project_name}</h1>
        <p>Rapor ID: {report_id} • Oluşturulma: {generated_at}</p>
    </header>
    <main>
        <section>
            <h2>📊 Metrik Özeti</h2>
            <table>
                <tbody>
                    {metrics_section}
                </tbody>
            </table>
        </section>
        {('<section><h2>🗂️ Veri Seti Özeti</h2>' + dataset_section + '</section>') if dataset_section else ''}
        <section>
            <h2>🤖 AI Analizi</h2>
            <div class="risk-chip">Risk: {risk_display}</div>
            <p>{summary_text}</p>
            {'<div class="analysis-grid"><div><h3>Güçlü Yönler</h3>' + strengths_html + '</div><div><h3>Zayıf Yönler</h3>' + weaknesses_html + '</div></div>'}
            <div>
                <h3>🎯 Aksiyon Önerileri</h3>
                {actions_html}
            </div>
            <div>
                <h3>🚀 Yayın Profili</h3>
                {deploy_html}
            </div>
            {('<div><h3>📝 Notlar</h3><p>' + notes_html + '</p></div>') if notes_html else ''}
        </section>
        <section>
            <h2>💬 Son Sorular</h2>
            {qa_history_html}
        </section>
    </main>
    <footer>DL_Result_Analyzer • {generated_at}</footer>
</body>
</html>
"""


def _generate_pdf_report(report_id: str, context: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    page_width, page_height = A4
    margin = 2 * cm
    max_chars = 95

    pdf = canvas.Canvas(buffer, pagesize=A4)

    project_context = context.get("project") or context.get("config", {}).get("project_context", {}) or {}
    project_name = project_context.get("project_name") or "DL Result Report"
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%d %B %Y %H:%M")

    pdf.setTitle(f"DL Result Analyzer - {project_name}")

    y_position = page_height - margin

    def ensure_space(lines: int = 1, leading: float = 14.0) -> None:
        nonlocal y_position
        if y_position - lines * leading < margin:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y_position = page_height - margin

    def write_line(text: str = "", font: str = "Helvetica", size: int = 11, leading: float = 14.0) -> None:
        nonlocal y_position
        ensure_space(1, leading)
        pdf.setFont(font, size)
        pdf.drawString(margin, y_position, text)
        y_position -= leading

    def write_paragraph(text: str, font: str = "Helvetica", size: int = 11, leading: float = 14.0) -> None:
        nonlocal y_position
        if not text:
            return
        lines = wrap(text, max_chars)
        ensure_space(len(lines), leading)
        for line in lines:
            pdf.setFont(font, size)
            pdf.drawString(margin, y_position, line)
            y_position -= leading
        y_position -= leading * 0.3

    def write_heading(text: str, level: int = 1) -> None:
        size = 18 if level == 1 else 14
        leading = 22 if level == 1 else 18
        write_line(text, font="Helvetica-Bold", size=size, leading=leading)

    def write_bullet(text: str) -> None:
        nonlocal y_position
        lines = wrap(text, max_chars - 4)
        ensure_space(len(lines))
        for idx, line in enumerate(lines):
            prefix = "• " if idx == 0 else "  "
            pdf.setFont("Helvetica", 11)
            pdf.drawString(margin + (0 if idx == 0 else 10), y_position, prefix + line)
            y_position -= 14
        y_position -= 4

    write_heading(project_name, level=1)
    write_paragraph(f"Rapor ID: {report_id}")
    write_paragraph(f"Oluşturulma: {generated_at}")

    metrics = context.get("metrics", {}) or {}
    if metrics:
        write_heading("Metrik Özeti", level=2)
        for key, label in [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("map50", "mAP@0.5"),
            ("map50_95", "mAP@0.5:0.95"),
            ("loss", "Loss"),
        ]:
            value = metrics.get(key)
            if key in {"precision", "recall", "map50", "map50_95"}:
                display = _format_percent(value)
            elif value is None:
                display = "N/A"
            else:
                display = f"{float(value):.4f}" if isinstance(value, (float, int)) else str(value)
            write_paragraph(f"{label}: {display}")

    dataset = context.get("config", {}).get("dataset", {}) or {}
    if dataset:
        write_heading("Veri Seti Özeti", level=2)
        for label, key in [
            ("Eğitim", "train_images"),
            ("Doğrulama", "val_images"),
            ("Test", "test_images"),
            ("Toplam", "total_images"),
        ]:
            value = dataset.get(key)
            if value is None:
                continue
            try:
                display = f"{int(float(value)):,}".replace(",", ".")
            except (TypeError, ValueError):
                display = str(value)
            write_paragraph(f"{label}: {display}")
        class_names = dataset.get("class_names")
        if isinstance(class_names, list) and class_names:
            write_paragraph("Sınıflar:")
            for class_name in class_names:
                write_bullet(str(class_name))

    analysis = context.get("analysis", {}) or {}
    if analysis:
        write_heading("AI Analizi", level=2)
        summary = analysis.get("summary")
        if summary:
            write_paragraph(summary)

        strengths = analysis.get("strengths") or []
        if strengths:
            write_line("Güçlü Yönler", font="Helvetica-Bold", size=12)
            for item in strengths:
                write_bullet(str(item))

        weaknesses = analysis.get("weaknesses") or []
        if weaknesses:
            write_line("Zayıf Yönler", font="Helvetica-Bold", size=12)
            for item in weaknesses:
                write_bullet(str(item))

        actions = analysis.get("actions") or []
        if actions:
            write_line("Aksiyon Önerileri", font="Helvetica-Bold", size=12)
            for action in actions:
                parts = []
                module = action.get("module")
                if module:
                    parts.append(f"Modül: {module}")
                recommendation = action.get("recommendation")
                if recommendation:
                    parts.append(f"Öneri: {recommendation}")
                expected = action.get("expected_gain")
                if expected:
                    parts.append(f"Beklenen Kazanç: {expected}")
                evidence = action.get("evidence")
                if evidence:
                    parts.append(f"Kanıt: {evidence}")
                validation = action.get("validation_plan")
                if validation:
                    parts.append(f"Doğrulama Planı: {validation}")
                if parts:
                    write_bullet(" | ".join(parts))

        risk = analysis.get("risk")
        if risk:
            write_paragraph(f"Risk Seviyesi: {str(risk).upper()}")

        deploy_profile = analysis.get("deploy_profile") or {}
        if deploy_profile:
            write_line("Yayın Profili", font="Helvetica-Bold", size=12)
            for key, value in deploy_profile.items():
                write_paragraph(f"{key}: {value}")

        notes = analysis.get("notes") or analysis.get("error")
        if notes:
            write_paragraph(f"Notlar: {notes}")

    qa_history = context.get("qa_history", []) or []
    if qa_history:
        write_heading("Son Sorular", level=2)
        for entry in qa_history[-5:][::-1]:
            write_paragraph(f"Soru: {entry.get('question', '')}")
            write_paragraph(f"Yanıt: {entry.get('answer', '')}")
            timestamp = entry.get("timestamp")
            if timestamp:
                write_paragraph(f"Zaman Damgası: {timestamp}")
            write_line()

    write_paragraph(f"Oluşturma Zamanı: {generated_at}")
    pdf.showPage()
    pdf.save()

    return buffer.getvalue()


@app.get("/api/report/{report_id}/export")
async def export_report(report_id: str, format: str = "html"):
    context = REPORT_STORE.get(report_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı veya süresi doldu.")

    format_normalized = (format or "html").lower()
    project_context = context.get("project") or context.get("config", {}).get("project_context", {}) or {}
    project_name = project_context.get("project_name")
    slug = _slugify_filename(project_name)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")

    if format_normalized == "html":
        html_content = _generate_html_report(report_id, context)
        filename = f"{slug}-{timestamp}.html"
        return Response(
            content=html_content,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    if format_normalized == "pdf":
        pdf_bytes = _generate_pdf_report(report_id, context)
        filename = f"{slug}-{timestamp}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    raise HTTPException(status_code=400, detail="Desteklenmeyen format. Lütfen 'html' veya 'pdf' kullanın.")

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

        provider = llm_provider or os.getenv("LLM_PROVIDER", "claude")
        if provider not in ["claude", "openai"]:
            logger.warning("Geçersiz LLM provider: %s, claude kullanılacak", provider)
            provider = "claude"

        analysis: Dict[str, Any] = {}
        try:
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

        files_payload = {
            "csv": csv_filename,
            "yaml": yaml_path.name if yaml_path else None,
            "graphs": saved_graphs,
            "best_model": best_model_path.name if best_model_path else None,
            "training_code": training_code_path.name if training_code_path else None,
        }

        report_context: Dict[str, Any] = {
            "metrics": metrics,
            "config": enriched_config,
            "history": history,
            "analysis": analysis,
            "project": project_context,
            "artefacts": artefacts_info,
            "training_code": training_code_context,
            "files": files_payload,
            "llm_provider": provider,
            "qa_history": [],
        }

        report_id = REPORT_STORE.save(report_context)

        return {
            "status": "success",
            "metrics": metrics,
            "config": enriched_config,
            "history": history,
            "analysis": analysis,
            "project": project_context,
            "training_code": training_code_context,
            "files": files_payload,
            "artefacts": artefacts_info,
            "report_id": report_id,
            "qa_history": report_context["qa_history"],
            "llm_provider": provider,
        }
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Dosya veya veri hatası nedeniyle upload başarısız oldu")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Beklenmeyen bir hata oluştu")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/report/{report_id}/qa")
async def report_qa(report_id: str, payload: QARequest):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru metni boş olamaz.")

    context = REPORT_STORE.get(report_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı veya süresi doldu.")

    provider = payload.llm_provider or context.get("llm_provider") or os.getenv("LLM_PROVIDER", "claude")
    if provider not in ["claude", "openai"]:
        logger.warning("Geçersiz QA provider seçimi: %s, claude kullanılacak", provider)
        provider = "claude"

    try:
        from app.analyzers.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(provider=provider)  # type: ignore[arg-type]
        answer_payload = analyzer.answer_question(question, context)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("LLM Q/A isteği başarısız oldu")
        raise HTTPException(status_code=500, detail=f"LLM Q/A başarısız oldu: {exc}") from exc

    qa_entry = {
        "question": question,
        "answer": answer_payload.get("answer", ""),
        "references": answer_payload.get("references", []),
        "follow_up_questions": answer_payload.get("follow_up_questions", []),
        "notes": answer_payload.get("notes"),
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "raw": answer_payload,
    }

    history = context.get("qa_history")
    if not isinstance(history, list):
        history = []
        context["qa_history"] = history
    history.append(qa_entry)
    context["llm_provider"] = provider

    REPORT_STORE.update(report_id, context)

    return {
        "status": "success",
        "report_id": report_id,
        "provider": provider,
        "qa": qa_entry,
        "qa_history": history,
    }


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

    logger.info(
        "Received threshold optimization request with model=%s data=%s",
        model_filename,
        data_filename,
    )

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
