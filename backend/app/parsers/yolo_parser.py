"""Utilities for parsing YOLO training artefacts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)


class YOLOResultParser:
    """Parse YOLO result artefacts such as results.csv and args.yaml."""

    def __init__(self, csv_path: Path | str, yaml_path: Optional[Path | str] = None) -> None:
        self.csv_path = Path(csv_path)
        self.yaml_path = Path(yaml_path) if yaml_path else None
        self._dataframe = None

    def _load_dataframe(self):
        """Load results.csv into a cached dataframe."""

        if self._dataframe is not None:
            return self._dataframe

        if not self.csv_path.exists():
            raise FileNotFoundError(f"results.csv not found at: {self.csv_path}")

        try:
            self._dataframe = pd.read_csv(self.csv_path)
        except Exception as exc:  # pragma: no cover - pandas specific errors
            raise ValueError(f"Unable to read CSV file: {self.csv_path}") from exc

        if self._dataframe.empty:
            raise ValueError("results.csv is empty and cannot be parsed.")

        return self._dataframe

    def parse_metrics(self) -> Dict[str, float]:
        """Parse the final epoch metrics from YOLO's ``results.csv`` file."""
        dataframe = self._load_dataframe()
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

    def parse_training_curves(self) -> Dict[str, List[float]]:
        """Extract epoch-wise curves for visualization on the frontend."""

        dataframe = self._load_dataframe()

        def _column_to_list(column: str) -> List[float]:
            if column not in dataframe.columns:
                return []

            series = pd.to_numeric(dataframe[column], errors="coerce")
            filled_series = series.ffill().bfill()
            return [float(value) for value in filled_series]

        epochs: List[float]
        if "epoch" in dataframe.columns:
            epochs = [int(value) for value in dataframe["epoch"].fillna(0).astype(int).tolist()]
        else:
            epochs = list(range(len(dataframe)))

        return {
            "epochs": epochs,
            "train_box_loss": _column_to_list("train/box_loss"),
            "val_box_loss": _column_to_list("val/box_loss"),
            "precision": _column_to_list("metrics/precision(B)"),
            "recall": _column_to_list("metrics/recall(B)"),
            "map50": _column_to_list("metrics/mAP50(B)"),
            "map50_95": _column_to_list("metrics/mAP50-95(B)"),
        }

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Convert values that represent integers into ``int`` safely."""

        if isinstance(value, bool) or value is None:
            return None

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                if "." in stripped:
                    return int(float(stripped))
                return int(stripped)
            except ValueError:
                return None

        return None

    def _resolve_path(self, value: Any) -> Optional[Path]:
        if value is None or not self.yaml_path:
            return None

        try:
            candidate = Path(str(value))
        except (TypeError, ValueError):
            return None

        if not candidate.is_absolute():
            candidate = (self.yaml_path.parent / candidate).resolve()

        return candidate

    @staticmethod
    def _extract_dataset_paths(source: Dict[str, Any]) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        for key in ("train", "val", "test", "path"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                mapped_key = f"{key}_path" if key in {"train", "val", "test"} else "root_path"
                paths.setdefault(mapped_key, value.strip())
        return paths

    @classmethod
    def _extract_dataset_counts(cls, source: Dict[str, Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        alias_map = {
            "train_images": {
                "train_images",
                "train_imgs",
                "train_samples",
                "train_count",
                "n_train",
                "train",
            },
            "val_images": {
                "val_images",
                "val_imgs",
                "val_samples",
                "val_count",
                "n_val",
                "validation",
                "valid",
                "val",
            },
            "test_images": {
                "test_images",
                "test_imgs",
                "test_samples",
                "test_count",
                "n_test",
                "test",
            },
            "total_images": {"total_images", "images_total", "dataset_size", "total"},
        }

        for key, value in source.items():
            if isinstance(value, dict):
                nested = cls._extract_dataset_counts(value)
                for nested_key, nested_value in nested.items():
                    counts.setdefault(nested_key, nested_value)
                continue

            numeric_value = cls._safe_int(value)
            if numeric_value is None:
                continue

            lowered = str(key).lower().strip()
            normalized = lowered.replace("-", "_").replace(" ", "_")
            for target_key, aliases in alias_map.items():
                if normalized in aliases:
                    counts.setdefault(target_key, numeric_value)
                    break

        if "total_images" not in counts:
            subtotal = sum(
                counts.get(item, 0)
                for item in ("train_images", "val_images", "test_images")
                if counts.get(item) is not None
            )
            if subtotal:
                counts["total_images"] = subtotal

        return counts

    @staticmethod
    def _extract_class_info(source: Dict[str, Any]) -> Dict[str, Any]:
        class_info: Dict[str, Any] = {}
        names = source.get("names")
        if isinstance(names, dict):
            try:
                ordered = [names[key] for key in sorted(names.keys(), key=lambda item: int(item))]
            except Exception:  # pragma: no cover - heterogeneous keys
                ordered = [names[key] for key in sorted(names.keys())]
            class_info["class_names"] = [str(item) for item in ordered]
        elif isinstance(names, list):
            class_info["class_names"] = [str(item) for item in names if str(item).strip()]

        if "class_names" in class_info:
            class_info["class_count"] = len(class_info["class_names"])
        else:
            numeric_nc = YOLOResultParser._safe_int(source.get("nc"))
            if numeric_nc is not None:
                class_info["class_count"] = numeric_nc

        return class_info

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as yaml_file:
                return yaml.safe_load(yaml_file) or {}
        except FileNotFoundError:
            LOGGER.debug("Dataset YAML not found at %s", path)
            return {}
        except yaml.YAMLError:
            LOGGER.warning("Dataset YAML could not be parsed at %s", path)
            return {}

    def _extract_dataset_info(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        dataset_info: Dict[str, Any] = {}

        data_value = raw_config.get("data")
        if data_value is not None:
            dataset_info["config_path"] = str(data_value)
            resolved = self._resolve_path(data_value)
            if resolved and resolved.exists():
                dataset_info["config_path_resolved"] = str(resolved)

        sources: List[Dict[str, Any]] = []
        for key in ("data_dict", "dataset_info"):
            value = raw_config.get(key)
            if isinstance(value, dict):
                sources.append(value)

        resolved_data_path = self._resolve_path(data_value) if data_value else None
        if resolved_data_path and resolved_data_path.exists():
            yaml_source = self._load_yaml_file(resolved_data_path)
            if isinstance(yaml_source, dict):
                sources.append(yaml_source)

        for source in sources:
            paths = self._extract_dataset_paths(source)
            for key, value in paths.items():
                dataset_info.setdefault(key, value)

            counts = self._extract_dataset_counts(source)
            for key, value in counts.items():
                dataset_info.setdefault(key, value)

            class_info = self._extract_class_info(source)
            for key, value in class_info.items():
                dataset_info.setdefault(key, value)

        # Fall back to top-level class information if still missing
        if "class_count" not in dataset_info:
            class_details = self._extract_class_info(raw_config)
            for key, value in class_details.items():
                dataset_info.setdefault(key, value)

        return {k: v for k, v in dataset_info.items() if v not in (None, "")}

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

        parsed_config: Dict[str, Any] = {
            "epochs": config.get("epochs", config.get("epoch", "N/A")),
            "batch": config.get("batch", config.get("batch_size", "N/A")),
            "lr0": config.get("lr0", config.get("learning_rate", "N/A")),
            "iou": config.get("iou", config.get("iou_threshold", "N/A")),
            "conf": config.get("conf", config.get("conf_threshold", "N/A")),
        }

        dataset_info = self._extract_dataset_info(config)
        if dataset_info:
            parsed_config["dataset"] = dataset_info

        return parsed_config
