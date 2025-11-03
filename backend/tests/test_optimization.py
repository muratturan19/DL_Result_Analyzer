"""Tests for optimization helper utilities."""

import sys
import types
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

if "ultralytics" not in sys.modules:
    stub = types.ModuleType("ultralytics")

    class _StubYOLO:  # pragma: no cover - defensive stub
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("YOLO stub invoked during tests")

    stub.YOLO = _StubYOLO
    sys.modules["ultralytics"] = stub

from app.api.endpoints.optimization import _prepare_data_yaml_for_inference


def _write_yaml(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_prepare_data_yaml_requires_train_and_val(tmp_path):
    """YAML without required splits should raise HTTPException."""

    yaml_path = tmp_path / "data.yaml"
    _write_yaml(yaml_path, {"path": "dataset", "train": "images/train"})

    with pytest.raises(HTTPException) as excinfo:
        _prepare_data_yaml_for_inference(yaml_path, base_dir=tmp_path)

    assert excinfo.value.status_code == 400
    assert "train" in excinfo.value.detail or "val" in excinfo.value.detail


def test_prepare_data_yaml_creates_dummy_dataset_when_missing(tmp_path):
    """Missing directories should be created as dummy datasets for testing."""

    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"

    yaml_path = tmp_path / "data.yaml"
    _write_yaml(
        yaml_path,
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
        },
    )

    # Should not raise exception, should create dummy datasets
    _prepare_data_yaml_for_inference(yaml_path, base_dir=tmp_path)

    # Verify dummy datasets were created
    assert train_dir.exists()
    assert val_dir.exists()

    # Verify images and labels directories exist
    assert (train_dir / "images").exists()
    assert (train_dir / "labels").exists()
    assert (val_dir / "images").exists()
    assert (val_dir / "labels").exists()

    # Verify sample files exist
    train_images = list((train_dir / "images").glob("*.jpg"))
    train_labels = list((train_dir / "labels").glob("*.txt"))
    assert len(train_images) > 0, "Dummy images should be created"
    assert len(train_labels) > 0, "Dummy labels should be created"
    assert len(train_images) == len(train_labels), "Equal number of images and labels"


def test_prepare_data_yaml_rewrites_relative_paths(tmp_path):
    """Relative dataset paths should be converted to absolute paths for inference."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    dataset_root = tmp_path / "data_root"
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    yaml_path = config_dir / "data.yaml"
    _write_yaml(
        yaml_path,
        {
            "path": "../data_root",
            "train": "images/train",
            "val": "images/val",
        },
    )

    _prepare_data_yaml_for_inference(yaml_path, base_dir=config_dir)

    updated = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert Path(updated["path"]) == dataset_root.resolve()
    assert Path(updated["train"]) == train_dir.resolve()
    assert Path(updated["val"]) == val_dir.resolve()
