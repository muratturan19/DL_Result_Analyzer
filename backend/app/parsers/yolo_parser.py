"""Utilities for parsing YOLO training artefacts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml


class YOLOResultParser:
    """Parse YOLO result artefacts such as results.csv and args.yaml."""

    def __init__(self, csv_path: Path | str, yaml_path: Optional[Path | str] = None) -> None:
        self.csv_path = Path(csv_path)
        self.yaml_path = Path(yaml_path) if yaml_path else None

    def parse_metrics(self) -> Dict[str, float]:
        """Parse the final epoch metrics from YOLO's ``results.csv`` file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"results.csv not found at: {self.csv_path}")

        try:
            dataframe = pd.read_csv(self.csv_path)
        except Exception as exc:  # pragma: no cover - pandas specific errors
            raise ValueError(f"Unable to read CSV file: {self.csv_path}") from exc

        if dataframe.empty:
            raise ValueError("results.csv is empty and cannot be parsed.")

        last_row = dataframe.iloc[-1]

        def _to_float(key: str, default: float = 0.0) -> float:
            value = last_row.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        return {
            "precision": _to_float("metrics/precision(B)"),
            "recall": _to_float("metrics/recall(B)"),
            "map50": _to_float("metrics/mAP50(B)"),
            "map50_95": _to_float("metrics/mAP50-95(B)"),
            "loss": _to_float("train/box_loss"),
        }

    def parse_config(self) -> Dict[str, object]:
        """Parse YOLO's ``args.yaml`` file if present."""
        if not self.yaml_path:
            return {}

        if not self.yaml_path.exists():
            raise FileNotFoundError(f"args.yaml not found at: {self.yaml_path}")

        with self.yaml_path.open("r", encoding="utf-8") as yaml_file:
            try:
                config = yaml.safe_load(yaml_file) or {}
            except yaml.YAMLError as exc:  # pragma: no cover - yaml specific errors
                raise ValueError(f"Unable to parse YAML file: {self.yaml_path}") from exc

        return {
            "epochs": config.get("epochs", config.get("epoch", "N/A")),
            "batch": config.get("batch", config.get("batch_size", "N/A")),
            "lr0": config.get("lr0", config.get("learning_rate", "N/A")),
            "iou": config.get("iou", config.get("iou_threshold", "N/A")),
            "conf": config.get("conf", config.get("conf_threshold", "N/A")),
        }
