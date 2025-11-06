# -*- coding: utf-8 -*-

import json
import logging
import os
import re
from html import escape
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from string import Template
from textwrap import wrap
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

import yaml

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv

from app.api.endpoints.optimization import router as optimization_router, _THRESHOLD_REPORT_STORE
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)

MAX_TRAINING_CODE_CHARS = int(os.getenv("TRAINING_CODE_PREVIEW_CHARS", "4000"))

PDF_FONT_REGULAR = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_DIR = _REPO_ROOT / "web"


def _load_report_asset(filename: str) -> str:
    asset_path = _WEB_DIR / filename
    try:
        return asset_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Rapor varlık dosyası bulunamadı: %s", asset_path)
    except OSError:
        logger.warning("Rapor varlık dosyası okunamadı: %s", asset_path, exc_info=True)
    return ""


REPORT_THEME_CSS = _load_report_asset("report-theme.css")
REPORT_PRINT_CSS = _load_report_asset("print.css")
REPORT_UI_SCRIPT = _load_report_asset("report-ui.js")
REPORT_ICONS_SVG = _load_report_asset("icons.svg")

_FONT_DIR = Path(__file__).resolve().parent / "static" / "fonts"

try:
    regular_path = _FONT_DIR / "DejaVuSans.ttf"
    bold_path = _FONT_DIR / "DejaVuSans-Bold.ttf"

    if regular_path.exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular_path)))
        PDF_FONT_REGULAR = "DejaVuSans"

    if bold_path.exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold_path)))
        PDF_FONT_BOLD = "DejaVuSans-Bold"

    if PDF_FONT_REGULAR == "DejaVuSans":
        addMapping("DejaVuSans", 0, 0, PDF_FONT_REGULAR)
        addMapping("DejaVuSans", 0, 1, PDF_FONT_BOLD)
        addMapping("DejaVuSans", 1, 0, PDF_FONT_REGULAR)
        addMapping("DejaVuSans", 1, 1, PDF_FONT_BOLD)
except Exception:  # pragma: no cover - defensive
    logger.warning(
        "Türkçe karakter destekli PDF fontu yüklenemedi, varsayılan fontlar kullanılacak.",
        exc_info=True,
    )
    PDF_FONT_REGULAR = "Helvetica"
    PDF_FONT_BOLD = "Helvetica-Bold"

app = FastAPI(title="DL_Result_Analyzer", version="1.0.0")

# CORS - React frontend için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(optimization_router)

# =============================================================================
# MODELS
# =============================================================================


class ReportStore:
    """In-memory store that keeps report contexts for follow-up Q/A.

    The store persists entries to disk so that freshly uploaded reports remain
    available even if the application process reloads (for example when running
    the development server with ``uvicorn --reload``)."""

    def __init__(self, ttl_seconds: int = 6 * 3600, storage_dir: Optional[Path] = None) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = Lock()
        storage_env = os.getenv("REPORT_STORAGE_DIR")
        if storage_dir is not None:
            self._storage_dir = Path(storage_dir)
        elif storage_env:
            self._storage_dir = Path(storage_env)
        else:
            self._storage_dir = Path(__file__).resolve().parent.parent / "reports"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load_existing_reports()

    def _report_path(self, report_id: str) -> Path:
        return self._storage_dir / f"{report_id}.json"

    @staticmethod
    def _coerce_timestamp(raw_value: Optional[str]) -> datetime:
        if not raw_value:
            return datetime.now(timezone.utc)
        try:
            timestamp = datetime.fromisoformat(raw_value)
        except ValueError:
            logger.warning("Geçersiz rapor zaman damgası: %s", raw_value)
            return datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _load_existing_reports(self) -> None:
        for json_path in self._storage_dir.glob("*.json"):
            report_id = json_path.stem
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                payload = data.get("payload")
                if payload is None:
                    raise ValueError("payload anahtarı bulunamadı")
                timestamp = self._coerce_timestamp(data.get("timestamp"))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Bozuk rapor dosyası yoksayılıyor: %s (%s)", json_path, exc)
                try:
                    json_path.unlink()
                except OSError:
                    logger.debug("Rapor dosyası silinemedi: %s", json_path)
                continue
            self._store[report_id] = {"payload": payload, "timestamp": timestamp}
        self._purge_expired()

    def _write_entry(self, report_id: str, entry: Dict[str, Any]) -> None:
        payload = entry.get("payload")
        timestamp: datetime = entry.get("timestamp", datetime.now(timezone.utc))
        data = {
            "payload": payload,
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
        }
        json_path = self._report_path(report_id)
        tmp_path = json_path.with_suffix(".json.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            tmp_path.replace(json_path)
        except Exception:
            logger.exception("Rapor dosyası yazılırken hata oluştu: %s", json_path)
            try:
                tmp_path.unlink()
            except OSError:
                logger.debug("Geçici rapor dosyası silinemedi: %s", tmp_path)

    def _delete_entry(self, report_id: str) -> None:
        json_path = self._report_path(report_id)
        try:
            json_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            logger.debug("Rapor dosyası silinemedi: %s", json_path)

    def _load_from_disk(self, report_id: str) -> Optional[Dict[str, Any]]:
        json_path = self._report_path(report_id)
        if not json_path.exists():
            return None
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            payload = data.get("payload")
            if payload is None:
                raise ValueError("payload anahtarı bulunamadı")
            timestamp = self._coerce_timestamp(data.get("timestamp"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Rapor dosyası okunamadı, silinecek: %s (%s)", json_path, exc)
            self._delete_entry(report_id)
            return None
        entry = {"payload": payload, "timestamp": timestamp}
        if datetime.now(timezone.utc) - timestamp > self._ttl:
            self._delete_entry(report_id)
            return None
        return entry

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired: List[str] = []
        for report_id, entry in list(self._store.items()):
            timestamp: datetime = entry.get("timestamp", now)
            if now - timestamp > self._ttl:
                expired.append(report_id)
        for report_id in expired:
            self._store.pop(report_id, None)
            self._delete_entry(report_id)

    def save(self, payload: Dict[str, Any]) -> str:
        report_id = uuid4().hex
        with self._lock:
            self._purge_expired()
            entry = {
                "payload": payload,
                "timestamp": datetime.now(timezone.utc),
            }
            self._store[report_id] = entry
            self._write_entry(report_id, entry)
        return report_id

    def get(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._purge_expired()
            entry = self._store.get(report_id)
            if not entry:
                entry = self._load_from_disk(report_id)
                if not entry:
                    return None
                self._store[report_id] = entry
            entry["timestamp"] = datetime.now(timezone.utc)
            self._write_entry(report_id, entry)
            return entry["payload"]

    def update(self, report_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._purge_expired()
            entry = {
                "payload": payload,
                "timestamp": datetime.now(timezone.utc),
            }
            self._store[report_id] = entry
            self._write_entry(report_id, entry)

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


def _format_metric_value(value: Any, as_percent: bool) -> Tuple[str, Optional[float]]:
    if value is None:
        return "N/A", None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return "N/A", None
        percent_string = cleaned.endswith("%")
        numeric_candidate = cleaned.rstrip("% ").replace(",", ".")
        try:
            numeric = float(numeric_candidate)
        except (TypeError, ValueError):
            return escape(cleaned), None
        if as_percent:
            raw_value = numeric / 100 if percent_string else numeric
            display = f"{numeric:.2f}%" if percent_string else f"{raw_value * 100:.2f}%"
        else:
            raw_value = numeric
            display = f"{numeric:.4f}".rstrip("0").rstrip(".") or f"{numeric:.4f}"
        return display, raw_value

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return escape(str(value)), None

    if as_percent:
        display = f"{numeric_value * 100:.2f}%"
        return display, numeric_value

    display = f"{numeric_value:.4f}".rstrip("0").rstrip(".") or f"{numeric_value:.4f}"
    return display, numeric_value


def _build_metrics_html(metrics: Dict[str, Any]) -> str:
    cards: List[str] = []
    metric_definitions = [
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("f1", "F1 Score", True),
        ("map50", "mAP@0.5", True),
        ("map75", "mAP@0.75", True),
        ("map50_95", "mAP@0.5:0.95", True),
        ("loss", "Loss", False),
    ]

    for key, label, as_percent in metric_definitions:
        display, raw_value = _format_metric_value(metrics.get(key), as_percent)
        raw_attr = "NaN" if raw_value is None else f"{raw_value:.6f}"
        cards.append(
            """
            <div class=\"stat-card\" data-metric-card data-metric=\"{metric}\" data-label=\"{label}\" data-raw=\"{raw}\">
                <span class=\"stat-label\">{label}</span>
                <span class=\"stat-value\">{display}</span>
                <span class=\"status-chip\" data-status-text>Durum değerlendiriliyor</span>
            </div>
            """.strip().format(
                metric=escape(key),
                label=escape(label),
                raw=raw_attr,
                display=display,
            )
        )

    if not cards:
        return "<p class=\"empty-state\">Metrik bilgisi bulunamadı.</p>"

    return "\n".join(cards)


def _build_list_html(items: List[str]) -> str:
    if not items:
        return "<p class=\"text-muted\">Bilgi bulunamadı.</p>"
    escaped_items = "".join(f"<li>{escape(item)}</li>" for item in items if item)
    return f"<ul class=\"bullet-list\">{escaped_items}</ul>" if escaped_items else "<p class=\"text-muted\">Bilgi bulunamadı.</p>"


def _build_actions_html(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "<p class=\"empty-state\">Aksiyon önerisi bulunamadı.</p>"

    action_items: List[str] = []
    for action in actions:
        module = escape(str(action.get("module", ""))) if action.get("module") else "Genel"
        recommendation = escape(str(action.get("recommendation", "")))
        evidence = escape(str(action.get("evidence", "")))
        expected = escape(str(action.get("expected_gain", "")))
        validation = escape(str(action.get("validation_plan", "")))

        meta_lines: List[str] = []
        if recommendation:
            meta_lines.append(f"<div><strong>Öneri:</strong> {recommendation}</div>")
        if evidence:
            meta_lines.append(f"<div><strong>Kanıt:</strong> {evidence}</div>")
        if expected:
            meta_lines.append(f"<div><strong>Beklenen Kazanç:</strong> {expected}</div>")
        if validation:
            meta_lines.append(f"<div><strong>Doğrulama Planı:</strong> {validation}</div>")

        action_items.append(
            """
            <li>
                <article class=\"action-card\">
                    <div>
                        <span class=\"chip\">
                            <svg aria-hidden=\"true\"><use href=\"#icon-info\" xlink:href=\"#icon-info\"></use></svg>
                            {module}
                        </span>
                    </div>
                    <div class=\"action-meta\">
                        {meta}
                    </div>
                </article>
            </li>
            """.strip().format(module=module, meta="".join(meta_lines) or "<div>Detay belirtilmedi.</div>")
        )

    return f"<ul class=\"action-list\">{''.join(action_items)}</ul>"


def _build_dataset_summary(dataset: Dict[str, Any]) -> str:
    if not dataset:
        return "<p class=\"text-muted\">Veri seti bilgisi bulunamadı.</p>"

    count_map = [
        ("Eğitim", dataset.get("train_images")),
        ("Doğrulama", dataset.get("val_images")),
        ("Test", dataset.get("test_images")),
        ("Toplam", dataset.get("total_images")),
    ]

    count_items: List[str] = []
    for label, value in count_map:
        if value is None:
            continue
        try:
            numeric = float(value)
            display = f"{int(numeric):,}".replace(",", ".")
        except (TypeError, ValueError):
            display = escape(str(value))
        count_items.append(
            """
            <div class=\"dataset-card\">
                <span class=\"dataset-label\">{label}</span>
                <span class=\"dataset-value\">{display}</span>
            </div>
            """.strip().format(label=escape(label), display=display)
        )

    classes = dataset.get("class_names")
    class_items = ""
    if isinstance(classes, list) and classes:
        chips = "".join(
            f"<span class=\"dataset-class\" role=\"listitem\">{escape(str(item))}</span>"
            for item in classes
            if item
        )
        if chips:
            class_items = f"<div class=\"dataset-classes\" role=\"list\">{chips}</div>"

    parts: List[str] = []
    if count_items:
        parts.append(f"<div class=\"dataset-grid\">{''.join(count_items)}</div>")
    if class_items:
        parts.append(class_items)

    return "".join(parts) if parts else "<p class=\"text-muted\">Veri seti bilgisi bulunamadı.</p>"


def _normalize_risk_details(raw_value: Any) -> List[str]:
    if not raw_value:
        return []

    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if not cleaned:
            return []
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            tokens = [token.strip() for token in re.split(r"[\n;,]+", cleaned) if token.strip()]
            return tokens or [cleaned]
        else:
            raw_value = parsed

    if isinstance(raw_value, dict):
        items: List[str] = []
        for key, value in raw_value.items():
            key_str = str(key).strip()
            value_str = str(value).strip()
            if key_str and value_str:
                items.append(f"{key_str}: {value_str}")
            elif key_str:
                items.append(key_str)
            elif value_str:
                items.append(value_str)
        return items

    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    return [str(raw_value)]


def _risk_icon_href(level: str) -> str:
    if level == "low":
        return "#icon-success"
    if level in {"medium", "moderate"}:
        return "#icon-warning"
    if level in {"high", "critical"}:
        return "#icon-warning"
    return "#icon-info"


def _build_risk_chips(analysis: Dict[str, Any]) -> str:
    risk_value = analysis.get("risk") or analysis.get("risk_level")
    level_key = ""
    chips: List[str] = []

    if isinstance(risk_value, str):
        normalized = risk_value.strip().lower()
        if normalized in {"low", "medium", "moderate", "high", "critical"}:
            level_key = "critical" if normalized == "high" else normalized
            label_map = {
                "low": "Düşük risk",
                "medium": "Orta risk",
                "moderate": "Orta risk",
                "high": "Yüksek risk",
                "critical": "Kritik risk",
            }
            chips.append(
                """
                <span class=\"risk-chip\" data-level=\"{level}\">
                    <svg aria-hidden=\"true\"><use href=\"{icon}\" xlink:href=\"{icon}\"></use></svg>
                    <span>{label}</span>
                </span>
                """.strip().format(
                    level=escape(level_key),
                    label=escape(label_map.get(level_key, risk_value)),
                    icon=_risk_icon_href(level_key),
                )
            )
        elif normalized:
            chips.append(
                """
                <span class=\"risk-chip\" data-level=\"medium\">
                    <svg aria-hidden=\"true\"><use href=\"#icon-warning\" xlink:href=\"#icon-warning\"></use></svg>
                    <span>{label}</span>
                </span>
                """.strip().format(label=escape(risk_value.strip()))
            )
    elif risk_value:
        chips.append(
            """
            <span class=\"risk-chip\" data-level=\"medium\">
                <svg aria-hidden=\"true\"><use href=\"#icon-warning\" xlink:href=\"#icon-warning\"></use></svg>
                <span>{label}</span>
            </span>
            """.strip().format(label=escape(str(risk_value)))
        )

    risk_details = (
        analysis.get("risk_details")
        or analysis.get("risk_factors")
        or analysis.get("risk_reasons")
        or analysis.get("risk_notes")
    )

    for item in _normalize_risk_details(risk_details):
        chips.append(
            """
            <span class=\"risk-chip\" data-level=\"medium\">
                <svg aria-hidden=\"true\"><use href=\"#icon-warning\" xlink:href=\"#icon-warning\"></use></svg>
                <span>{label}</span>
            </span>
            """.strip().format(label=escape(item))
        )

    if not chips:
        return "<p class=\"text-muted\">Risk verisi paylaşılmadı.</p>"

    return f"<div class=\"risk-chips\">{''.join(chips)}</div>"


def _build_deploy_profile_html(deploy_profile: Any) -> str:
    if not deploy_profile:
        return "<p class=\"empty-state\">Bilgi bulunamadı.</p>"

    if isinstance(deploy_profile, dict):
        rows = []
        for key, value in deploy_profile.items():
            rows.append(
                """
                <tr>
                    <th scope=\"row\">{key}</th>
                    <td>{value}</td>
                </tr>
                """.strip().format(key=escape(str(key)), value=escape(str(value)))
            )
        return f"<div class=\"table-wrapper\"><table class=\"data-table\"><tbody>{''.join(rows)}</tbody></table></div>"

    if isinstance(deploy_profile, list):
        entries: List[str] = []
        for item in deploy_profile:
            if isinstance(item, dict):
                pieces = [f"{str(k)}: {str(v)}" for k, v in item.items() if v is not None]
                entries.append("; ".join(pieces) if pieces else str(item))
            else:
                entries.append(str(item))
        return _build_list_html([entry for entry in entries if entry])

    return f"<p>{escape(str(deploy_profile))}</p>"


def _build_dataset_review_html(dataset_review: Any) -> str:
    """Build HTML for dataset_review section from AI analysis."""
    if not dataset_review or not isinstance(dataset_review, dict):
        return ""

    html_parts = []

    # Size evaluation
    size_eval = dataset_review.get("size_evaluation")
    if size_eval:
        html_parts.append(f"<p><strong>Boyut Değerlendirmesi:</strong> {escape(str(size_eval))}</p>")

    # Split assessment
    split_assess = dataset_review.get("split_assessment")
    if split_assess:
        html_parts.append(f"<p><strong>Bölümleme Değerlendirmesi:</strong> {escape(str(split_assess))}</p>")

    # Folder distribution
    folder_dist = dataset_review.get("folder_distribution")
    if folder_dist and isinstance(folder_dist, list):
        items_html = _build_list_html(folder_dist)
        html_parts.append(f"<div><strong>Klasör Dağılımı:</strong>{items_html}</div>")

    # Recommendations
    recommendations = dataset_review.get("recommendations")
    if recommendations and isinstance(recommendations, list):
        items_html = _build_list_html(recommendations)
        html_parts.append(f"<div><strong>Öneriler:</strong>{items_html}</div>")

    if not html_parts:
        return ""

    return "".join(html_parts)


def _build_architecture_alignment_html(arch_alignment: Any) -> str:
    """Build HTML for architecture_alignment section from AI analysis."""
    if not arch_alignment or not isinstance(arch_alignment, dict):
        return ""

    html_parts = []

    # Model info
    model_name = arch_alignment.get("model_name")
    if model_name:
        html_parts.append(f"<p><strong>Model:</strong> {escape(str(model_name))}</p>")

    min_required = arch_alignment.get("minimum_required_images")
    if min_required:
        html_parts.append(f"<p><strong>Minimum Gerekli Görsel:</strong> {escape(str(min_required))}</p>")

    current_dataset = arch_alignment.get("current_dataset_images")
    if current_dataset:
        html_parts.append(f"<p><strong>Mevcut Veri Seti Görselleri:</strong> {escape(str(current_dataset))}</p>")

    # Fit assessment
    fit_assess = arch_alignment.get("fit_assessment")
    if fit_assess:
        html_parts.append(f"<p><strong>Uyum Değerlendirmesi:</strong> {escape(str(fit_assess))}</p>")

    # Actions
    actions = arch_alignment.get("actions")
    if actions and isinstance(actions, list):
        items_html = _build_list_html(actions)
        html_parts.append(f"<div><strong>Önerilen Aksiyonlar:</strong>{items_html}</div>")

    if not html_parts:
        return ""

    return "".join(html_parts)


def _generate_html_report(report_id: str, context: Dict[str, Any]) -> str:
    project_context = context.get("project") or context.get("config", {}).get("project_context", {}) or {}
    project_name = project_context.get("project_name") or "DL Result Report"
    project_name_html = escape(str(project_name))
    report_id_html = escape(report_id)
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%d %B %Y %H:%M")

    metrics_section = _build_metrics_html(context.get("metrics", {}))
    dataset_section = _build_dataset_summary(context.get("config", {}).get("dataset", {}))

    analysis = context.get("analysis", {}) or {}
    strengths_html = _build_list_html(analysis.get("strengths", []))
    weaknesses_html = _build_list_html(analysis.get("weaknesses", []))
    actions_html = _build_actions_html(analysis.get("actions", []))

    risk_section = _build_risk_chips(analysis)
    deploy_html = _build_deploy_profile_html(analysis.get("deploy_profile"))

    # Extract dataset_review and architecture_alignment from analysis
    dataset_review_html = _build_dataset_review_html(analysis.get("dataset_review"))
    architecture_alignment_html = _build_architecture_alignment_html(analysis.get("architecture_alignment"))

    qa_history = context.get("qa_history", []) or []
    qa_items: List[str] = []
    for entry in qa_history[-5:][::-1]:
        question = escape(str(entry.get("question", "")))
        answer = escape(str(entry.get("answer", "")))
        timestamp = escape(str(entry.get("timestamp", "")))
        qa_items.append(
            """
            <article class=\"qa-card\">
                <div class=\"qa-question\">{question}</div>
                <div class=\"qa-answer\">{answer}</div>
                <div class=\"qa-timestamp\">{timestamp}</div>
            </article>
            """.strip().format(question=question, answer=answer, timestamp=timestamp)
        )

    qa_history_html = "".join(qa_items) if qa_items else "<p class=\"empty-state\">Henüz takip sorusu yok.</p>"
    summary_text = (
        escape(str(analysis.get("summary"))) if analysis.get("summary") else "Henüz özet oluşturulmadı."
    )
    notes_text = analysis.get("notes") or analysis.get("error")

    dataset_section_html = (
        f"""
        <section class=\"section-card\" aria-labelledby=\"dataset-heading\">
            <div class=\"section-heading\">
                <h2 id=\"dataset-heading\">Veri Seti Özeti</h2>
                <p>Veri bölümlerinin dağılımı ve sınıf yapısı.</p>
            </div>
            {dataset_section}
        </section>
        """
        if dataset_section
        else ""
    )

    analysis_lists_html = f"""
        <div class=\"section-block\">
            <div class=\"collapsible\" data-collapsible data-open>
                <button type=\"button\" class=\"collapsible-trigger\" data-collapsible-trigger>
                    <span>Güçlü yönler</span>
                    <svg aria-hidden=\"true\"><use href=\"#icon-chevron\" xlink:href=\"#icon-chevron\"></use></svg>
                </button>
                <div class=\"collapsible-content\" data-collapsible-content>
                    {strengths_html}
                </div>
            </div>
            <div class=\"collapsible\" data-collapsible>
                <button type=\"button\" class=\"collapsible-trigger\" data-collapsible-trigger>
                    <span>Geliştirilmesi gerekenler</span>
                    <svg aria-hidden=\"true\"><use href=\"#icon-chevron\" xlink:href=\"#icon-chevron\"></use></svg>
                </button>
                <div class=\"collapsible-content\" data-collapsible-content>
                    {weaknesses_html}
                </div>
            </div>
        </div>
    """

    notes_block = ""
    if notes_text:
        notes_block = """
        <div class=\"section-block\">
            <div class=\"collapsible\" data-collapsible data-open>
                <button type=\"button\" class=\"collapsible-trigger\" data-collapsible-trigger>
                    <span>Notlar</span>
                    <svg aria-hidden=\"true\"><use href=\"#icon-chevron\" xlink:href=\"#icon-chevron\"></use></svg>
                </button>
                <div class=\"collapsible-content\" data-collapsible-content>
                    <p>{notes}</p>
                </div>
            </div>
        </div>
        """.format(notes=escape(str(notes_text)))

    # Build dataset review section
    dataset_review_section = ""
    if dataset_review_html:
        dataset_review_section = f"""
        <div class=\"section-block\">
            <div class=\"collapsible\" data-collapsible data-open>
                <button type=\"button\" class=\"collapsible-trigger\" data-collapsible-trigger>
                    <span>Veri Seti Değerlendirmesi</span>
                    <svg aria-hidden=\"true\"><use href=\"#icon-chevron\" xlink:href=\"#icon-chevron\"></use></svg>
                </button>
                <div class=\"collapsible-content\" data-collapsible-content>
                    {dataset_review_html}
                </div>
            </div>
        </div>
        """

    # Build architecture alignment section
    architecture_alignment_section = ""
    if architecture_alignment_html:
        architecture_alignment_section = f"""
        <div class=\"section-block\">
            <div class=\"collapsible\" data-collapsible data-open>
                <button type=\"button\" class=\"collapsible-trigger\" data-collapsible-trigger>
                    <span>Model Mimarisi Uyumu</span>
                    <svg aria-hidden=\"true\"><use href=\"#icon-chevron\" xlink:href=\"#icon-chevron\"></use></svg>
                </button>
                <div class=\"collapsible-content\" data-collapsible-content>
                    {architecture_alignment_html}
                </div>
            </div>
        </div>
        """

    # Get token usage info
    token_usage = context.get("token_usage", {})
    total_tokens = token_usage.get("total_tokens", 0) if isinstance(token_usage, dict) else 0
    token_info_text = f"Toplam token kullanımı: {total_tokens:,}" if total_tokens > 0 else ""

    page_title = f"{project_name} - DL Result Analyzer"

    template = Template(
        """<!DOCTYPE html>
<html lang=\"tr\" data-theme=\"dark\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>$page_title</title>
    <script>
      window.tailwind = window.tailwind || {};
      tailwind.config = {
        darkMode: ['class', '[data-theme="dark"]'],
        theme: {
          extend: {
            colors: {
              primary: '#6366f1',
              accent: '#22d3ee',
              slate: '#0f172a'
            }
          }
        },
        corePlugins: {
          preflight: false
        }
      };
    </script>
    <script src=\"https://cdn.tailwindcss.com\" defer></script>
    <style>
$theme_css
    </style>
    <style media=\"print\">
$print_css
    </style>
</head>
<body class=\"report-body\">
$icons_svg
    <header class=\"report-header\">
        <div class=\"report-shell\">
            <div class=\"report-meta\">
                <h1>$project_name</h1>
                <p>Rapor ID: $report_id</p>
                <div class=\"section-meta\">
                    <span>Oluşturulma: $generated_at</span>
                </div>
            </div>
            <div class=\"header-actions\" data-print-hidden>
                <button class=\"btn btn-primary\" type=\"button\" data-theme-toggle aria-label=\"Temayı değiştir\">
                    <svg aria-hidden=\"true\"><use href=\"#icon-moon\" xlink:href=\"#icon-moon\"></use></svg>
                    <span data-theme-toggle-label>Koyu tema</span>
                </button>
                <button class=\"btn btn-outline\" type=\"button\" data-print-report aria-label=\"Raporu yazdır\">
                    <svg aria-hidden=\"true\"><use href=\"#icon-external\" xlink:href=\"#icon-external\"></use></svg>
                    <span>Yazdır</span>
                </button>
            </div>
        </div>
    </header>
    <main class=\"report-main\">
        <div class=\"report-container\">
            <section class=\"section-card elevated\" aria-labelledby=\"metrics-heading\">
                <div class=\"section-heading\">
                    <h2 id=\"metrics-heading\">Performans Özeti</h2>
                    <p>Precision, Recall, mAP ve Loss metriklerinin son değerleri.</p>
                </div>
                <div class=\"stat-grid\">
                    $metrics_section
                </div>
                <div class=\"section-block\">
                    <h3>Risk profili</h3>
                    $risk_section
                </div>
            </section>
            $dataset_section
            <section class=\"section-card\" aria-labelledby=\"analysis-heading\">
                <div class=\"section-heading\">
                    <h2 id=\"analysis-heading\">AI Analizi</h2>
                    <p>Model değerlendirmesi, güçlü/zayıf yönler ve öneriler.</p>
                </div>
                <p>$summary_text</p>
                $analysis_lists
                <div class=\"section-block\">
                    <h3>Aksiyon önerileri</h3>
                    $actions_html
                </div>
                <div class=\"section-block\">
                    <h3>Yayın profili</h3>
                    $deploy_profile
                </div>
                $dataset_review_section
                $architecture_alignment_section
                $notes_block
            </section>
            <section class=\"section-card\" aria-labelledby=\"qa-heading\">
                <div class=\"section-heading\">
                    <h2 id=\"qa-heading\">Takip Soruları</h2>
                    <p>Son otomatik Q/A oturumlarının özeti.</p>
                </div>
                <div class=\"qa-grid\">
                    $qa_history
                </div>
            </section>
        </div>
    </main>
    <footer class=\"report-footer\">
        <span>DL_Result_Analyzer • sürüm $app_version</span>
        <span>Oluşturulma zamanı: $generated_at</span>
        $token_info_span
    </footer>
    <script defer>
$ui_script
    </script>
</body>
</html>
"""
    )

    token_info_span = f"<span>{escape(token_info_text)}</span>" if token_info_text else ""

    return template.substitute(
        page_title=escape(page_title),
        project_name=project_name_html,
        report_id=report_id_html,
        generated_at=generated_at,
        metrics_section=metrics_section,
        risk_section=risk_section,
        dataset_section=dataset_section_html,
        analysis_lists=analysis_lists_html,
        actions_html=actions_html,
        deploy_profile=deploy_html,
        dataset_review_section=dataset_review_section,
        architecture_alignment_section=architecture_alignment_section,
        notes_block=notes_block,
        qa_history=qa_history_html,
        summary_text=summary_text,
        token_info_span=token_info_span,
        icons_svg=REPORT_ICONS_SVG,
        theme_css=REPORT_THEME_CSS,
        print_css=REPORT_PRINT_CSS,
        ui_script=REPORT_UI_SCRIPT,
        app_version=escape(str(app.version) if hasattr(app, "version") else "1.0.0"),
    )


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
            pdf.setFont(PDF_FONT_REGULAR, 11)
            y_position = page_height - margin

    def write_line(text: str = "", font: str = PDF_FONT_REGULAR, size: int = 11, leading: float = 14.0) -> None:
        nonlocal y_position
        ensure_space(1, leading)
        pdf.setFont(font, size)
        pdf.drawString(margin, y_position, text)
        y_position -= leading

    def write_paragraph(text: str, font: str = PDF_FONT_REGULAR, size: int = 11, leading: float = 14.0) -> None:
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
        write_line(text, font=PDF_FONT_BOLD, size=size, leading=leading)

    def write_bullet(text: str) -> None:
        nonlocal y_position
        lines = wrap(text, max_chars - 4)
        ensure_space(len(lines))
        for idx, line in enumerate(lines):
            prefix = "• " if idx == 0 else "  "
            pdf.setFont(PDF_FONT_REGULAR, 11)
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
            ("f1", "F1 Score"),
            ("map50", "mAP@0.5"),
            ("map75", "mAP@0.75"),
            ("map50_95", "mAP@0.5:0.95"),
            ("loss", "Loss"),
        ]:
            value = metrics.get(key)
            if key in {"precision", "recall", "f1", "map50", "map75", "map50_95"}:
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
            write_line("Güçlü Yönler", font=PDF_FONT_BOLD, size=12)
            for item in strengths:
                write_bullet(str(item))

        weaknesses = analysis.get("weaknesses") or []
        if weaknesses:
            write_line("Zayıf Yönler", font=PDF_FONT_BOLD, size=12)
            for item in weaknesses:
                write_bullet(str(item))

        actions = analysis.get("actions") or []
        if actions:
            write_line("Aksiyon Önerileri", font=PDF_FONT_BOLD, size=12)
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
            write_line("Yayın Profili", font=PDF_FONT_BOLD, size=12)
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
    project_type: Optional[str] = Form(None),
    dataset_totals: Optional[str] = Form(None),
    split_ratios: Optional[str] = Form(None),
    folder_distributions: Optional[str] = Form(None),
    llm_provider: str = Form("claude"),
):
    """
    YOLO eğitim sonuçlarını upload et
    - results.csv: metrics
    - args.yaml: config
    - *.png: graphs (confusion_matrix, F1_curve, etc.)
    """
    import time
    request_start_time = time.time()

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = Path(results_csv.filename or "results.csv").name
    csv_path = uploads_dir / csv_filename

    logger.info(
        "=== UPLOAD REQUEST STARTED === csv=%s yaml=%s graphs=%s best_model=%s training_code=%s",
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
    cleaned_project_type = _clean_str(project_type)

    cleaned_class_count: Optional[int] = None
    if class_count is not None:
        try:
            cleaned_class_count = int(class_count)
        except (TypeError, ValueError):
            logger.warning("Geçersiz class_count değeri alındı: %s", class_count)

    def _parse_json_field(raw_value: Optional[str], *, field_name: str) -> Optional[Any]:
        if raw_value is None:
            return None
        candidate = raw_value.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("%s alanı için geçersiz JSON alındı", field_name)
            return None

    def _cast_to_int(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(round(float(stripped.replace(",", "."))))
            except ValueError:
                return None
        return None

    def _cast_to_float(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            normalized = stripped.replace(",", ".")
            try:
                return float(normalized)
            except ValueError:
                return None
        return None

    def _sanitize_numeric_mapping(
        data: Any, *, field_name: str, caster
    ) -> Optional[Dict[str, Any]]:
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.warning("%s alanı için dict bekleniyordu", field_name)
            return None
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key).strip()
            if not key_str:
                continue
            numeric_value = caster(value)
            if numeric_value is None:
                logger.warning(
                    "%s alanındaki '%s' değeri sayıya dönüştürülemedi",
                    field_name,
                    key_str,
                )
                continue
            sanitized[key_str] = numeric_value
        return sanitized or None

    def _sanitize_nested_numeric(data: Any, *, field_name: str, caster):
        if data is None:
            return None

        def _inner(value: Any):
            if isinstance(value, dict):
                nested: Dict[str, Any] = {}
                for nested_key, nested_value in value.items():
                    key_str = str(nested_key).strip()
                    if not key_str:
                        continue
                    cleaned = _inner(nested_value)
                    if cleaned is not None:
                        nested[key_str] = cleaned
                return nested or None
            if isinstance(value, list):
                cleaned_list = []
                for item in value:
                    cleaned = _inner(item)
                    if cleaned is not None:
                        cleaned_list.append(cleaned)
                return cleaned_list or None
            return caster(value)

        sanitized = _inner(data)
        if sanitized is None and data not in (None, ""):
            logger.warning("%s alanı için geçerli bir dağılım elde edilemedi", field_name)
        return sanitized

    parsed_dataset_totals = _sanitize_numeric_mapping(
        _parse_json_field(dataset_totals, field_name="dataset_totals"),
        field_name="dataset_totals",
        caster=_cast_to_int,
    )
    parsed_split_ratios = _sanitize_numeric_mapping(
        _parse_json_field(split_ratios, field_name="split_ratios"),
        field_name="split_ratios",
        caster=_cast_to_float,
    )
    parsed_folder_distributions = _sanitize_nested_numeric(
        _parse_json_field(folder_distributions, field_name="folder_distributions"),
        field_name="folder_distributions",
        caster=_cast_to_int,
    )

    try:
        logger.info("CSV dosyası okunuyor...")
        csv_bytes = await results_csv.read()
        csv_size_mb = len(csv_bytes) / (1024 * 1024)
        logger.info("CSV dosyası okundu: %.2f MB", csv_size_mb)
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

        logger.info("Parsing başlatılıyor...")
        parse_start_time = time.time()

        parser = YOLOResultParser(csv_path, yaml_path)
        metrics = parser.parse_metrics()
        config = parser.parse_config()
        history = parser.parse_training_curves()

        parse_duration = time.time() - parse_start_time
        logger.info(
            "Metrix ve konfigürasyon parse edildi (%.2fs): metrics_keys=%s config_keys=%s",
            parse_duration,
            sorted(metrics.keys()),
            sorted(config.keys()),
        )

        project_context = {
            "project_name": cleaned_project_name,
            "short_description": cleaned_description,
            "class_count": cleaned_class_count,
            "training_method": cleaned_method,
            "project_focus": cleaned_focus,
            "project_type": cleaned_project_type,
        }
        project_context = {k: v for k, v in project_context.items() if v not in (None, "")}

        training_code_context: Optional[Dict[str, str]] = None
        if training_code_filename:
            training_code_context = {
                "filename": training_code_filename,
                "excerpt": training_code_excerpt or "",
            }

        enriched_config: Dict[str, object] = dict(config)
        dataset_section: Dict[str, Any] = {}
        if isinstance(enriched_config.get("dataset"), dict):
            dataset_section = dict(enriched_config["dataset"])

        if parsed_dataset_totals:
            dataset_section["dataset_totals"] = parsed_dataset_totals
            alias_map = {
                "train": "train_images",
                "training": "train_images",
                "val": "val_images",
                "validation": "val_images",
                "test": "test_images",
                "total": "total_images",
            }
            for original_key, numeric_value in parsed_dataset_totals.items():
                lookup = (
                    alias_map.get(original_key.lower())
                    if isinstance(original_key, str)
                    else None
                )
                if lookup:
                    dataset_section[lookup] = numeric_value

        if parsed_split_ratios:
            dataset_section["split_ratios"] = parsed_split_ratios

        if parsed_folder_distributions:
            dataset_section["folder_distributions"] = parsed_folder_distributions

        if dataset_section:
            enriched_config["dataset"] = dataset_section

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

        # Prepare graph image paths for LLM vision analysis
        graph_image_paths: List[Path] = []
        if graphs:
            graphs_dir = uploads_dir / "graphs"
            for graph_filename in saved_graphs:
                graph_path = graphs_dir / graph_filename
                if graph_path.exists():
                    graph_image_paths.append(graph_path)
                    logger.debug("Grafik LLM'e gönderilecek: %s", graph_filename)

        analysis: Dict[str, Any] = {}
        llm_start_time = time.time()
        try:
            analyzer = LLMAnalyzer(provider=provider)  # type: ignore[arg-type]
            logger.info("LLM analizi başlatılıyor: provider=%s, graphs=%d", provider, len(graph_image_paths))

            analysis = analyzer.analyze(
                metrics,
                enriched_config,
                project_context=project_context,
                training_code=training_code_context,
                history=history,
                artefacts=artefacts_info,
                graph_images=graph_image_paths,
            )

            llm_duration = time.time() - llm_start_time
            logger.info(
                "LLM analizi tamamlandı (%.2fs): provider=%s, has_summary=%s",
                llm_duration,
                provider,
                "summary" in analysis,
            )
        except Exception as exc:
            llm_duration = time.time() - llm_start_time
            logger.exception("LLM analizi başarısız oldu (%.2fs): %s", llm_duration, str(exc))
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

        # Extract token usage from analysis
        analysis_token_usage = analysis.pop("token_usage", None) if isinstance(analysis, dict) else None
        token_usage_info: Dict[str, Any] = {
            "analysis": analysis_token_usage,
            "qa_sessions": [],
            "total_input_tokens": analysis_token_usage.get("input_tokens", 0) if analysis_token_usage else 0,
            "total_output_tokens": analysis_token_usage.get("output_tokens", 0) if analysis_token_usage else 0,
            "total_tokens": analysis_token_usage.get("total_tokens", 0) if analysis_token_usage else 0,
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
            "token_usage": token_usage_info,
        }

        report_id = REPORT_STORE.save(report_context)

        total_duration = time.time() - request_start_time
        logger.info(
            "=== UPLOAD REQUEST COMPLETED === Duration: %.2fs, ReportID: %s, Provider: %s",
            total_duration,
            report_id,
            provider,
        )

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
        total_duration = time.time() - request_start_time
        logger.exception(
            "Dosya veya veri hatası nedeniyle upload başarısız oldu (Duration: %.2fs)",
            total_duration,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        total_duration = time.time() - request_start_time
        logger.exception("Beklenmeyen bir hata oluştu (Duration: %.2fs)", total_duration)
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

    # Extract token usage from QA response
    qa_token_usage = answer_payload.pop("token_usage", None) if isinstance(answer_payload, dict) else None

    qa_entry = {
        "question": question,
        "answer": answer_payload.get("answer", ""),
        "references": answer_payload.get("references", []),
        "follow_up_questions": answer_payload.get("follow_up_questions", []),
        "notes": answer_payload.get("notes"),
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "token_usage": qa_token_usage,
        "raw": answer_payload,
    }

    history = context.get("qa_history")
    if not isinstance(history, list):
        history = []
        context["qa_history"] = history
    history.append(qa_entry)
    context["llm_provider"] = provider

    # Update total token usage
    token_usage_info = context.get("token_usage", {})
    if not isinstance(token_usage_info, dict):
        token_usage_info = {
            "analysis": None,
            "qa_sessions": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }

    if qa_token_usage:
        token_usage_info["qa_sessions"].append(qa_token_usage)
        token_usage_info["total_input_tokens"] += qa_token_usage.get("input_tokens", 0)
        token_usage_info["total_output_tokens"] += qa_token_usage.get("output_tokens", 0)
        token_usage_info["total_tokens"] += qa_token_usage.get("total_tokens", 0)

    context["token_usage"] = token_usage_info

    REPORT_STORE.update(report_id, context)

    return {
        "status": "success",
        "report_id": report_id,
        "provider": provider,
        "qa": qa_entry,
        "qa_history": history,
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


def _generate_threshold_suggestions(best: Dict[str, float]) -> str:
    """Generate improvement suggestions based on threshold optimization results."""
    suggestions = []

    precision = best.get("precision", 0)
    recall = best.get("recall", 0)
    f1 = best.get("f1", 0)
    map50 = best.get("map50", 0)
    map75 = best.get("map75", 0)
    iou = best.get("iou", 0)
    conf = best.get("confidence", 0)

    # Analyze precision/recall balance
    if precision > 0 and recall > 0:
        if precision > recall + 0.15:
            suggestions.append({
                "icon": "⚖️",
                "title": "Recall Geliştirme Fırsatı",
                "description": f"Precision ({precision:.1%}) recall'dan ({recall:.1%}) belirgin şekilde yüksek. Daha fazla nesneyi tespit etmek için confidence threshold'u düşürmeyi ({conf:.2f} → {max(0.1, conf-0.05):.2f}) veya daha fazla eğitim verisi eklemeyi değerlendirin.",
                "impact": "Yüksek",
                "color": "#f59e0b"
            })
        elif recall > precision + 0.15:
            suggestions.append({
                "icon": "🎯",
                "title": "Precision İyileştirme Önerisi",
                "description": f"Recall ({recall:.1%}) precision'dan ({precision:.1%}) belirgin şekilde yüksek. False positive'leri azaltmak için confidence threshold'u artırmayı ({conf:.2f} → {min(0.9, conf+0.05):.2f}) veya hard negative mining uygulayın.",
                "impact": "Yüksek",
                "color": "#f59e0b"
            })

    # F1 score analysis
    if f1 < 0.7:
        suggestions.append({
            "icon": "📊",
            "title": "Model Performans İyileştirmesi",
            "description": f"F1 skoru ({f1:.1%}) hedef değerin altında. Model mimarisini güçlendirmeyi (daha büyük model), eğitim süresini artırmayı veya data augmentation stratejilerini zenginleştirmeyi düşünün.",
            "impact": "Kritik",
            "color": "#ef4444"
        })
    elif f1 >= 0.85:
        suggestions.append({
            "icon": "✅",
            "title": "Mükemmel F1 Dengesi",
            "description": f"F1 skoru ({f1:.1%}) hedef aralıkta. Production'a hazır. Farklı IoU threshold'larında performansı izlemeye devam edin.",
            "impact": "Düşük",
            "color": "#10b981"
        })

    # mAP analysis
    if map50 > 0.85 and map75 < 0.65:
        suggestions.append({
            "icon": "🔍",
            "title": "Lokalizasyon Hassasiyeti",
            "description": f"mAP@0.5 ({map50:.1%}) yüksek ama mAP@0.75 ({map75:.1%}) düşük. Bu, bounding box lokalizasyonunun iyileştirilmesi gerektiğini gösterir. IoU loss weight'i artırın veya daha kaliteli annotation'lar kullanın.",
            "impact": "Orta",
            "color": "#3b82f6"
        })

    # Threshold optimization
    if conf < 0.3:
        suggestions.append({
            "icon": "⚡",
            "title": "Düşük Confidence Threshold",
            "description": f"Mevcut confidence threshold ({conf:.2f}) çok düşük. Bu, gerçek dünya senaryolarında fazla false positive üretebilir. Production ortamında en az 0.35-0.40 arası threshold kullanmayı düşünün.",
            "impact": "Orta",
            "color": "#f59e0b"
        })
    elif conf > 0.7:
        suggestions.append({
            "icon": "🎚️",
            "title": "Yüksek Confidence Threshold",
            "description": f"Mevcut confidence threshold ({conf:.2f}) yüksek. Bu kesinlik istiyorsanız iyidir, ancak bazı geçerli tespitleri kaçırabilirsiniz. İhtiyaçlarınıza göre dengelemeyi değerlendirin.",
            "impact": "Düşük",
            "color": "#3b82f6"
        })

    # IoU threshold
    if iou < 0.4:
        suggestions.append({
            "icon": "📐",
            "title": "IoU Threshold Ayarı",
            "description": f"IoU threshold ({iou:.2f}) düşük. NMS (Non-Maximum Suppression) daha esnek ama overlapping detections riski var. Çakışan tespit sayısını azaltmak için 0.45-0.50 aralığını deneyin.",
            "impact": "Düşük",
            "color": "#6b7280"
        })

    # Add general production recommendations
    suggestions.append({
        "icon": "🚀",
        "title": "Production Deployment Önerileri",
        "description": "Bu threshold'ları production'da kullanmadan önce: (1) Farklı lighting/weather koşullarında test edin, (2) Edge case'leri manuel olarak inceleyin, (3) A/B testi yaparak mevcut sisteminizle karşılaştırın.",
        "impact": "Kritik",
        "color": "#8b5cf6"
    })

    if not suggestions:
        suggestions.append({
            "icon": "✨",
            "title": "İyi Dengeli Sonuçlar",
            "description": "Threshold optimizasyonunuz dengeli görünüyor. Production'a geçmeden önce farklı data split'lerde validate etmeyi unutmayın.",
            "impact": "Düşük",
            "color": "#10b981"
        })

    # Generate HTML for suggestions
    suggestions_html = ""
    for suggestion in suggestions:
        impact_class = suggestion["impact"].lower().replace("ü", "u").replace("ı", "i")
        suggestions_html += f"""
        <div class="suggestion-card" style="border-left: 4px solid {suggestion['color']};">
            <div class="suggestion-header">
                <span class="suggestion-icon">{suggestion['icon']}</span>
                <span class="suggestion-title">{suggestion['title']}</span>
                <span class="suggestion-impact impact-{impact_class}">{suggestion['impact']}</span>
            </div>
            <p class="suggestion-description">{suggestion['description']}</p>
        </div>
        """

    return suggestions_html


def _generate_threshold_html_report(report_id: str, context: Dict[str, Any]) -> str:
    """Generate HTML report for threshold optimization results."""
    model_filename = escape(str(context.get("model_filename", "N/A")))
    data_filename = escape(str(context.get("data_filename", "N/A")))
    split = escape(str(context.get("split", "N/A")))
    optimization_date = context.get("optimization_date", datetime.now(timezone.utc).isoformat())
    formatted_date = datetime.fromisoformat(optimization_date).strftime("%d %B %Y %H:%M")

    best = context.get("best", {})
    best_iou = best.get("iou", 0)
    best_conf = best.get("confidence", 0)
    best_precision = _format_percent(best.get("precision", 0))
    best_recall = _format_percent(best.get("recall", 0))
    best_f1 = _format_percent(best.get("f1", 0))
    best_map50 = _format_percent(best.get("map50", 0))
    best_map75 = _format_percent(best.get("map75", 0))
    best_map5095 = _format_percent(best.get("map5095", 0))

    total_combinations = context.get("total_combinations", 0)

    heatmap = context.get("heatmap", {})
    iou_values = heatmap.get("iou_values", [])
    conf_values = heatmap.get("confidence_values", [])

    production_config = context.get("production_config", {})
    config_yaml = production_config.get("yaml", "")

    # Generate improvement suggestions
    suggestions_html = _generate_threshold_suggestions(best)

    template = Template("""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Threshold Optimization Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #1f2937;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            color: white;
            padding: 40px 32px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }
        header h1 {
            margin: 0 0 12px;
            font-size: 36px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        header p {
            margin: 0;
            font-size: 16px;
            opacity: 0.95;
            font-weight: 300;
        }
        main {
            padding: 32px 0;
        }
        section {
            background: white;
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 28px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        section:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }
        h2 {
            margin: 0 0 24px;
            font-size: 24px;
            color: #1f2937;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            padding-bottom: 16px;
            border-bottom: 3px solid #f3f4f6;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 16px;
            overflow: hidden;
            border-radius: 12px;
        }
        th, td {
            padding: 16px 20px;
            text-align: left;
        }
        th {
            background: linear-gradient(135deg, #eef2ff, #e0e7ff);
            font-weight: 600;
            color: #4f46e5;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        tr {
            transition: background 0.2s;
        }
        tr:nth-child(even) td {
            background: #fafafa;
        }
        tr:hover td {
            background: #f0f9ff !important;
        }
        td {
            border-bottom: 1px solid #f3f4f6;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 24px;
        }
        .metric-card {
            background: linear-gradient(135deg, #fafafa 0%, #ffffff 100%);
            border-radius: 16px;
            padding: 24px;
            border: 2px solid #e5e7eb;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(99, 102, 241, 0.15);
            border-color: #6366f1;
        }
        .metric-label {
            font-size: 13px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-top: 4px;
        }
        .best-badge {
            display: inline-block;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        .suggestion-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: all 0.2s;
        }
        .suggestion-card:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }
        .suggestion-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        .suggestion-icon {
            font-size: 28px;
        }
        .suggestion-title {
            font-size: 18px;
            font-weight: 700;
            color: #1f2937;
            flex: 1;
        }
        .suggestion-impact {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .impact-kritik { background: #fee2e2; color: #dc2626; }
        .impact-yuksek { background: #fef3c7; color: #d97706; }
        .impact-orta { background: #dbeafe; color: #2563eb; }
        .impact-dusuk { background: #d1fae5; color: #059669; }
        .suggestion-description {
            color: #4b5563;
            line-height: 1.7;
            font-size: 15px;
        }
        pre {
            background: #1e293b;
            color: #e2e8f0;
            padding: 24px;
            border-radius: 12px;
            overflow-x: auto;
            font-size: 14px;
            line-height: 1.6;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        footer {
            text-align: center;
            padding: 32px 24px;
            color: rgba(255, 255, 255, 0.9);
            font-size: 14px;
            font-weight: 300;
        }
        @media print {
            body { background: white; }
            section { box-shadow: none; page-break-inside: avoid; }
            header { background: #6366f1; }
        }
        @media (max-width: 768px) {
            .container { padding: 12px; }
            header { padding: 24px 16px; }
            header h1 { font-size: 28px; }
            section { padding: 20px; }
            .metric-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header>
        <h1>🎛️ Threshold Optimization Report</h1>
        <p>Rapor ID: $report_id • Tarih: $formatted_date</p>
    </header>
    <main class="container">
        <section>
            <h2>📋 Optimizasyon Detayları</h2>
            <table>
                <tbody>
                    <tr><th>Model</th><td>$model_filename</td></tr>
                    <tr><th>Data Config</th><td>$data_filename</td></tr>
                    <tr><th>Test Split</th><td>$split</td></tr>
                    <tr><th>Toplam Kombinasyon</th><td>$total_combinations</td></tr>
                    <tr><th>IoU Aralığı</th><td>$iou_range</td></tr>
                    <tr><th>Confidence Aralığı</th><td>$conf_range</td></tr>
                </tbody>
            </table>
        </section>
        <section>
            <h2>🎯 En İyi Sonuç</h2>
            <div class="best-badge">IoU: $best_iou · Confidence: $best_conf</div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">$best_precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">$best_recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">$best_f1</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">mAP@0.5</div>
                    <div class="metric-value">$best_map50</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">mAP@0.75</div>
                    <div class="metric-value">$best_map75</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">mAP@0.5:0.95</div>
                    <div class="metric-value">$best_map5095</div>
                </div>
            </div>
        </section>
        <section>
            <h2>💡 Geliştirme Önerileri</h2>
            $suggestions_html
        </section>
        <section>
            <h2>⚙️ Production Config</h2>
            <pre>$config_yaml</pre>
        </section>
    </main>
    <footer>DL_Result_Analyzer • Threshold Optimizer • $formatted_date</footer>
</body>
</html>
""")

    iou_range_str = f"{min(iou_values):.2f} - {max(iou_values):.2f}" if iou_values else "N/A"
    conf_range_str = f"{min(conf_values):.2f} - {max(conf_values):.2f}" if conf_values else "N/A"

    return template.substitute(
        report_id=escape(report_id),
        formatted_date=formatted_date,
        model_filename=model_filename,
        data_filename=data_filename,
        split=split,
        total_combinations=total_combinations,
        iou_range=iou_range_str,
        conf_range=conf_range_str,
        best_iou=f"{best_iou:.2f}",
        best_conf=f"{best_conf:.2f}",
        best_precision=best_precision,
        best_recall=best_recall,
        best_f1=best_f1,
        best_map50=best_map50,
        best_map75=best_map75,
        best_map5095=best_map5095,
        config_yaml=escape(config_yaml),
        suggestions_html=suggestions_html,
    )


def _generate_threshold_pdf_report(report_id: str, context: Dict[str, Any]) -> bytes:
    """Generate PDF report for threshold optimization results."""
    buffer = BytesIO()
    page_width, page_height = A4
    margin = 2 * cm
    max_chars = 95

    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle(f"Threshold Optimization Report - {report_id[:8]}")

    model_filename = context.get("model_filename", "N/A")
    data_filename = context.get("data_filename", "N/A")
    split = context.get("split", "N/A")
    optimization_date = context.get("optimization_date", datetime.now(timezone.utc).isoformat())
    formatted_date = datetime.fromisoformat(optimization_date).strftime("%d %B %Y %H:%M")

    best = context.get("best", {})
    total_combinations = context.get("total_combinations", 0)

    heatmap = context.get("heatmap", {})
    iou_values = heatmap.get("iou_values", [])
    conf_values = heatmap.get("confidence_values", [])

    y_position = page_height - margin

    def ensure_space(lines: int = 1, leading: float = 14.0) -> None:
        nonlocal y_position
        if y_position - lines * leading < margin:
            pdf.showPage()
            pdf.setFont(PDF_FONT_REGULAR, 11)
            y_position = page_height - margin

    def write_line(text: str = "", font: str = PDF_FONT_REGULAR, size: int = 11, leading: float = 14.0) -> None:
        nonlocal y_position
        ensure_space(1, leading)
        pdf.setFont(font, size)
        pdf.drawString(margin, y_position, text)
        y_position -= leading

    def write_paragraph(text: str, font: str = PDF_FONT_REGULAR, size: int = 11, leading: float = 14.0) -> None:
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
        write_line(text, font=PDF_FONT_BOLD, size=size, leading=leading)

    # Header
    write_heading("Threshold Optimization Report", level=1)
    write_paragraph(f"Rapor ID: {report_id}")
    write_paragraph(f"Tarih: {formatted_date}")
    write_line()

    # Optimization details
    write_heading("Optimizasyon Detayları", level=2)
    write_paragraph(f"Model: {model_filename}")
    write_paragraph(f"Data Config: {data_filename}")
    write_paragraph(f"Test Split: {split}")
    write_paragraph(f"Toplam Kombinasyon: {total_combinations}")
    if iou_values:
        write_paragraph(f"IoU Aralığı: {min(iou_values):.2f} - {max(iou_values):.2f}")
    if conf_values:
        write_paragraph(f"Confidence Aralığı: {min(conf_values):.2f} - {max(conf_values):.2f}")
    write_line()

    # Best results
    write_heading("En İyi Sonuç", level=2)
    write_paragraph(f"IoU: {best.get('iou', 0):.2f} · Confidence: {best.get('confidence', 0):.2f}", font=PDF_FONT_BOLD)
    write_paragraph(f"Precision: {_format_percent(best.get('precision'))}")
    write_paragraph(f"Recall: {_format_percent(best.get('recall'))}")
    write_paragraph(f"F1 Score: {_format_percent(best.get('f1'))}")
    write_paragraph(f"mAP@0.5: {_format_percent(best.get('map50'))}")
    write_paragraph(f"mAP@0.75: {_format_percent(best.get('map75'))}")
    write_paragraph(f"mAP@0.5:0.95: {_format_percent(best.get('map5095'))}")
    write_line()

    # Production config
    production_config = context.get("production_config", {})
    config_yaml = production_config.get("yaml", "")
    if config_yaml:
        write_heading("Production Config", level=2)
        for line in config_yaml.split('\n')[:30]:  # Limit to first 30 lines
            write_paragraph(line, size=9, leading=12)

    write_paragraph(f"Oluşturma Zamanı: {formatted_date}")
    pdf.showPage()
    pdf.save()

    return buffer.getvalue()


@app.get("/api/optimize/thresholds/reports/{report_id}/export")
async def export_threshold_report(report_id: str, format: str = "html"):
    """Export threshold optimization report as HTML or PDF."""
    context = _THRESHOLD_REPORT_STORE.get(report_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Threshold raporu bulunamadı veya süresi doldu.")

    format_normalized = (format or "html").lower()
    model_filename = context.get("model_filename", "threshold-report")
    slug = _slugify_filename(model_filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    if format_normalized == "html":
        html_content = _generate_threshold_html_report(report_id, context)
        filename = f"{slug}-threshold-{timestamp}.html"
        return Response(
            content=html_content,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    if format_normalized == "pdf":
        pdf_bytes = _generate_threshold_pdf_report(report_id, context)
        filename = f"{slug}-threshold-{timestamp}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    raise HTTPException(status_code=400, detail="Desteklenmeyen format. Lütfen 'html' veya 'pdf' kullanın.")

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
