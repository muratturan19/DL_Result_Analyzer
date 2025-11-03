# -*- coding: utf-8 -*-
"""Threshold optimization endpoint for YOLO models."""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
try:  # pragma: no cover - optional dependency for tests
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - fallback when Ultralytics is absent
    YOLO = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimize", tags=["optimization"])

_ALLOWED_SPLITS = {"train", "val", "test"}
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")
_WINDOWS_UNC_PATTERN = re.compile(r"^[/\\]{2}[^/\\]+[/\\]+[^/\\]+")


def _normalize_float_values(values: Iterable[float], name: str) -> List[float]:
    seen = set()
    normalized: List[float] = []
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{name} değerleri sayısal olmalıdır.") from exc
        rounded = round(value, 6)
        if rounded in seen:
            continue
        seen.add(rounded)
        normalized.append(round(value, 6))
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{name} için en az bir değer gereklidir.")
    normalized.sort()
    return normalized


def _build_range_from_dict(payload: Dict[str, float], name: str) -> List[float]:
    if {"start", "end", "step"}.issubset(payload.keys()):
        start = float(payload["start"])
        end = float(payload["end"])
        step = float(payload["step"])
        if step <= 0:
            raise HTTPException(status_code=400, detail=f"{name} step pozitif olmalıdır.")
        values: List[float] = []
        current = start
        while current <= end + 1e-9:
            values.append(round(current, 6))
            current += step
        if not values:
            raise HTTPException(status_code=400, detail=f"{name} aralığı boş olamaz.")
        if values[-1] < round(end, 6):
            values.append(round(end, 6))
        return _normalize_float_values(values, name)
    if "values" in payload:
        return _normalize_float_values(payload["values"], name)
    raise HTTPException(status_code=400, detail=f"{name} aralığı geçersiz formatta.")


def _parse_range_payload(raw_payload: str) -> tuple[list[float], list[float]]:
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="ranges JSON formatında olmalıdır.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="ranges sözlük formatında olmalıdır.")

    if "iou" not in payload or "confidence" not in payload:
        raise HTTPException(status_code=400, detail="ranges içinde 'iou' ve 'confidence' anahtarları gerekli.")

    iou_values = (
        _build_range_from_dict(payload["iou"], "IoU")
        if isinstance(payload["iou"], dict)
        else _normalize_float_values(payload["iou"], "IoU")
    )
    conf_values = (
        _build_range_from_dict(payload["confidence"], "Confidence")
        if isinstance(payload["confidence"], dict)
        else _normalize_float_values(payload["confidence"], "Confidence")
    )

    return iou_values, conf_values


def _parse_single_range(raw_payload: str, name: str) -> List[float]:
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{name} aralığı JSON formatında olmalıdır.") from exc

    if isinstance(payload, dict):
        return _build_range_from_dict(payload, name)
    if isinstance(payload, list):
        return _normalize_float_values(payload, name)
    raise HTTPException(status_code=400, detail=f"{name} aralığı geçersiz formatta.")


def _resolve_existing_file(root: Path, filename: str, *, description: str) -> Path:
    if not filename:
        raise HTTPException(status_code=400, detail=f"{description} için dosya adı gerekli.")

    sanitized = Path(filename).name
    candidate = root / sanitized

    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"{description} sunucuda bulunamadı: {sanitized}")
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"{description} bir dosya olmalıdır: {sanitized}")

    return candidate


def _extract_metric(box_metrics: object, attribute: str) -> float:
    value = getattr(box_metrics, attribute, None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _select_best_candidate(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        raise HTTPException(status_code=500, detail="Grid search sonuçları bulunamadı.")

    recall_threshold = 0.85
    precision_threshold = 0.75

    filtered = [item for item in results if item["recall"] >= recall_threshold]
    if not filtered:
        filtered = list(results)
    else:
        precision_filtered = [item for item in filtered if item["precision"] >= precision_threshold]
        if precision_filtered:
            filtered = precision_filtered

    best = max(filtered, key=lambda item: item["f1"])
    return best


def _build_production_config(
    *,
    best: Dict[str, float],
    model_filename: str,
    data_filename: str,
    split: str,
    task: str,
) -> Dict[str, object]:
    payload = {
        "model_info": {
            "model_path": model_filename,
            "data_config": data_filename,
            "task": task,
            "optimization_date": datetime.now(timezone.utc).isoformat(),
            "test_split": split,
        },
        "inference_params": {
            "confidence_threshold": best["confidence"],
            "iou_threshold": best["iou"],
            "imgsz": 640,
            "task": "detect",
        },
        "performance_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_score": best["f1"],
            "map50": best["map50"],
            "map75": best["map75"],
            "map50_95": best["map5095"],
        },
        "usage_example": {
            "python": (
                "model = YOLO('{path}')\nresults = model.predict(source='image.jpg', conf={conf}, iou={iou})"
            ).format(path=model_filename, conf=best["confidence"], iou=best["iou"]),
        },
    }
    yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    encoded = base64.b64encode(yaml_text.encode("utf-8")).decode("utf-8")
    return {
        "filename": "production_config.yaml",
        "yaml": yaml_text,
        "base64": encoded,
    }


def _format_cell(iou: float, conf: float, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "iou": round(float(iou), 6),
        "confidence": round(float(conf), 6),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "map50": round(metrics["map50"], 4),
        "map75": round(metrics["map75"], 4),
        "map5095": round(metrics["map5095"], 4),
    }


def _run_grid_search(
    model: YOLO,
    data_path: Path,
    *,
    split: str,
    iou_values: List[float],
    conf_values: List[float],
) -> tuple[list[list[Dict[str, float]]], List[Dict[str, float]]]:
    heatmap_rows: list[list[Dict[str, float]]] = []
    flat_results: List[Dict[str, float]] = []

    for iou in iou_values:
        row: list[Dict[str, float]] = []
        for conf in conf_values:
            try:
                metrics = model.val(
                    data=str(data_path),
                    iou=iou,
                    conf=conf,
                    split=split,
                    verbose=False,
                    plots=False,
                )
            except Exception as exc:  # pragma: no cover - Ultralytics specific errors
                logger.exception("Grid search sırasında hata: iou=%s conf=%s", iou, conf)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            box_metrics = getattr(metrics, "box", None)
            if box_metrics is None:
                raise HTTPException(status_code=500, detail="Ultralytics metrikleri okunamadı.")

            precision = _extract_metric(box_metrics, "mp")
            recall = _extract_metric(box_metrics, "mr")
            map50 = _extract_metric(box_metrics, "map50")
            map75 = _extract_metric(box_metrics, "map75")
            map5095 = _extract_metric(box_metrics, "map")

            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * precision * recall / (precision + recall)

            cell_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "map50": map50,
                "map75": map75,
                "map5095": map5095,
            }
            cell = _format_cell(iou, conf, cell_metrics)
            row.append(cell)
            flat_results.append(cell)
        heatmap_rows.append(row)

    return heatmap_rows, flat_results


async def _write_upload_to_path(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await upload.seek(0)


@router.post("/thresholds")
async def optimize_thresholds(
    best_model: UploadFile = File(None),
    data_yaml: UploadFile = File(None),
    ranges: Optional[str] = Form(None),
    iou_range: Optional[str] = Form(None),
    conf_range: Optional[str] = Form(None),
    best_model_filename: Optional[str] = Form(None),
    data_yaml_filename: Optional[str] = Form(None),
    split: str = Form("test"),
    dataset_root: Optional[str] = Form(None),
):
    """Optimize IoU and confidence thresholds via grid search."""

    if YOLO is None:
        raise HTTPException(status_code=500, detail="Ultralytics modülü yüklü değil.")

    if split not in _ALLOWED_SPLITS:
        raise HTTPException(status_code=400, detail=f"Geçersiz split: {split}.")

    if best_model is None and not best_model_filename:
        raise HTTPException(status_code=400, detail="best.pt dosyası sağlanmalıdır.")
    if data_yaml is None and not data_yaml_filename:
        raise HTTPException(status_code=400, detail="data.yaml dosyası sağlanmalıdır.")

    if ranges is not None:
        iou_values, conf_values = _parse_range_payload(ranges)
    else:
        if iou_range is None or conf_range is None:
            raise HTTPException(status_code=400, detail="IoU ve confidence aralıkları belirtilmelidir.")
        iou_values = _parse_single_range(iou_range, "IoU")
        conf_values = _parse_single_range(conf_range, "Confidence")

    logger.info(
        "Threshold optimization started for model=%s data=%s",
        best_model.filename if best_model else best_model_filename,
        data_yaml.filename if data_yaml else data_yaml_filename,
    )

    with TemporaryDirectory(prefix="threshold_opt_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        if best_model is not None:
            model_filename_raw = best_model.filename
        else:
            model_filename_raw = best_model_filename
        model_filename = Path(model_filename_raw or "best.pt").name

        if data_yaml is not None:
            data_filename_raw = data_yaml.filename
        else:
            data_filename_raw = data_yaml_filename
        data_filename = Path(data_filename_raw or "data.yaml").name

        model_path = tmp_path / model_filename
        data_path = tmp_path / data_filename

        if best_model is not None:
            await _write_upload_to_path(best_model, model_path)
        else:
            existing_model = _resolve_existing_file(
                Path("uploads") / "models",
                best_model_filename or "best.pt",
                description="best.pt",
            )
            shutil.copy2(existing_model, model_path)

        if data_yaml is not None:
            await _write_upload_to_path(data_yaml, data_path)
            # Use current working directory as base for resolving relative paths in uploaded YAML
            data_base_dir = Path.cwd()
        else:
            existing_yaml = _resolve_existing_file(
                Path("uploads"),
                data_yaml_filename or "data.yaml",
                description="data.yaml",
            )
            shutil.copy2(existing_yaml, data_path)
            # Use the existing YAML's parent directory for resolving relative paths
            data_base_dir = existing_yaml.parent

        _prepare_data_yaml_for_inference(
            data_path,
            base_dir=data_base_dir,
            dataset_root_override=dataset_root,
        )

        try:
            model = await run_in_threadpool(YOLO, str(model_path))
        except Exception as exc:  # pragma: no cover - Ultralytics specific errors
            logger.exception("Model yüklenemedi: %s", model_path)
            raise HTTPException(status_code=400, detail=f"Model yüklenemedi: {exc}") from exc

        heatmap_rows, flat_results = await run_in_threadpool(
            _run_grid_search,
            model,
            data_path,
            split=split,
            iou_values=iou_values,
            conf_values=conf_values,
        )
        task = model.task

    best_candidate = _select_best_candidate(flat_results)
    production_config = _build_production_config(
        best=best_candidate,
        model_filename=model_filename,
        data_filename=data_filename,
        split=split,
        task=task,
    )

    return {
        "status": "success",
        "heatmap": {
            "rows": heatmap_rows,
            "values": flat_results,
            "iou_values": iou_values,
            "confidence_values": conf_values,
        },
        "best": best_candidate,
        "production_config": production_config,
    }


def _is_absolute_path(path_value: str) -> bool:
    """Return True if given path string is absolute for POSIX or Windows systems."""

    if not path_value:
        return False

    if Path(path_value).is_absolute():
        return True

    # Additional checks for Windows-style absolute paths when running on non-Windows systems
    if _WINDOWS_DRIVE_PATTERN.match(path_value):
        return True
    if _WINDOWS_UNC_PATTERN.match(path_value):
        return True

    return False


def _load_data_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML content and ensure dictionary structure."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem issues propagated to client
        raise HTTPException(status_code=400, detail=f"data.yaml dosyası okunamadı: {exc}") from exc

    try:
        content = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail="data.yaml YAML formatında olmalıdır.") from exc

    if not isinstance(content, dict):
        raise HTTPException(status_code=400, detail="data.yaml sözlük formatında olmalıdır.")

    return content


def _resolve_dataset_root(content: Dict[str, Any], *, base_dir: Path) -> Path:
    """Resolve dataset root directory using YAML content and provided base directory."""

    raw_root = content.get("path")
    if isinstance(raw_root, str) and raw_root.strip():
        root_value = raw_root.strip()
        if _is_absolute_path(root_value):
            return Path(root_value)
        return (base_dir / root_value).resolve()
    return base_dir.resolve()


def _resolve_split_directory(split_value: str, dataset_root: Path) -> Path:
    """Resolve split directory path respecting absolute paths and dataset root."""

    split_str = split_value.strip()
    if _is_absolute_path(split_str):
        return Path(split_str)

    candidate = Path(split_str)
    if _is_absolute_path(str(dataset_root)):
        return dataset_root / candidate
    return (dataset_root / candidate).resolve()


def _prepare_data_yaml_for_inference(
    data_yaml_path: Path,
    *,
    base_dir: Path,
    dataset_root_override: Optional[str] = None,
) -> None:
    """Validate dataset configuration and normalise relative paths for Ultralytics."""

    content = _load_data_yaml(data_yaml_path)

    missing_required = [key for key in ("train", "val") if not isinstance(content.get(key), str) or not content.get(key)]
    if missing_required:
        missing_display = ", ".join(sorted(missing_required))
        raise HTTPException(status_code=400, detail=f"data.yaml içinde '{missing_display}' anahtarları zorunludur.")

    dataset_root: Path
    override_path: Optional[Path] = None
    if dataset_root_override and dataset_root_override.strip():
        override_raw = dataset_root_override.strip()
        override_candidate = Path(override_raw).expanduser()
        if not _is_absolute_path(override_raw):
            override_candidate = (base_dir / override_candidate).resolve()
        if not override_candidate.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Veri seti kök klasörü bulunamadı: {override_candidate}",
            )
        dataset_root = override_candidate
        override_path = override_candidate
    else:
        dataset_root = _resolve_dataset_root(content, base_dir=base_dir)

    dataset_root = dataset_root.expanduser()
    updated = False
    missing_directories: List[str] = []

    if override_path is not None:
        if content.get("path") != str(override_path):
            content["path"] = str(override_path)
            updated = True
    elif isinstance(content.get("path"), str) and content["path"].strip() and not _is_absolute_path(content["path"].strip()):
        content["path"] = str(dataset_root)
        updated = True

    for split in ("train", "val", "test"):
        raw_value = content.get(split)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue

        candidate = _resolve_split_directory(raw_value, dataset_root).expanduser()
        if not candidate.is_dir():
            missing_directories.append(f"{split}: {candidate}")
            continue

        if not _is_absolute_path(raw_value.strip()):
            content[split] = str(candidate)
            updated = True

    if missing_directories:
        formatted = ", ".join(missing_directories)
        raise HTTPException(status_code=404, detail=f"Veri seti klasörleri bulunamadı: {formatted}")

    if updated:
        data_yaml_path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
