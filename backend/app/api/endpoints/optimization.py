# -*- coding: utf-8 -*-
"""Threshold optimization endpoint for YOLO models."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ultralytics import YOLO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimize", tags=["optimization"])

_ALLOWED_SPLITS = {"train", "val", "test"}


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
    best_model: UploadFile = File(...),
    data_yaml: UploadFile = File(...),
    ranges: str = Form(...),
    split: str = Form("test"),
):
    """Optimize IoU and confidence thresholds via grid search."""

    if split not in _ALLOWED_SPLITS:
        raise HTTPException(status_code=400, detail=f"Geçersiz split: {split}.")

    iou_values, conf_values = _parse_range_payload(ranges)

    logger.info(
        "Threshold optimization started for model=%s data=%s", best_model.filename, data_yaml.filename
    )

    with TemporaryDirectory(prefix="threshold_opt_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / (Path(best_model.filename or "best.pt").name)
        data_path = tmp_path / (Path(data_yaml.filename or "data.yaml").name)

        await _write_upload_to_path(best_model, model_path)
        await _write_upload_to_path(data_yaml, data_path)

        try:
            model = YOLO(str(model_path))
        except Exception as exc:  # pragma: no cover - Ultralytics specific errors
            logger.exception("Model yüklenemedi: %s", model_path)
            raise HTTPException(status_code=400, detail=f"Model yüklenemedi: {exc}") from exc

        heatmap_rows, flat_results = _run_grid_search(
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
        model_filename=Path(best_model.filename or "best.pt").name,
        data_filename=Path(data_yaml.filename or "data.yaml").name,
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
