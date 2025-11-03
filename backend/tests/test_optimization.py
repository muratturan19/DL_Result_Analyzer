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


def test_prepare_data_yaml_validates_split_directories(tmp_path):
    """Missing directories should produce a helpful HTTPException."""

    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "images" / "train"
    train_dir.mkdir(parents=True)

    yaml_path = tmp_path / "data.yaml"
    _write_yaml(
        yaml_path,
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
        },
    )

    with pytest.raises(HTTPException) as excinfo:
        _prepare_data_yaml_for_inference(yaml_path, base_dir=tmp_path)

    assert excinfo.value.status_code == 404
    assert "val" in excinfo.value.detail


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


def test_prepare_data_yaml_supports_dataset_override(tmp_path):
    """Explicit dataset root override should take precedence over YAML values."""

    external_root = tmp_path / "boxing_dataset"
    train_dir = external_root / "train" / "images"
    val_dir = external_root / "valid" / "images"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    yaml_path = tmp_path / "data.yaml"
    _write_yaml(
        yaml_path,
        {
            "train": "train/images",
            "val": "valid/images",
        },
    )

    _prepare_data_yaml_for_inference(
        yaml_path,
        base_dir=tmp_path,
        dataset_root_override=str(external_root),
    )

    updated = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert updated["path"] == str(external_root)
    assert Path(updated["train"]) == train_dir
    assert Path(updated["val"]) == val_dir


def test_prepare_data_yaml_invalid_override(tmp_path):
    """An invalid override should raise a descriptive HTTPException."""

    yaml_path = tmp_path / "data.yaml"
    _write_yaml(
        yaml_path,
        {
            "train": "images/train",
            "val": "images/val",
        },
    )

    with pytest.raises(HTTPException) as excinfo:
        _prepare_data_yaml_for_inference(
            yaml_path,
            base_dir=tmp_path,
            dataset_root_override=str(tmp_path / "missing"),
        )

    assert excinfo.value.status_code == 404
    assert "Veri seti kök klasörü" in excinfo.value.detail
