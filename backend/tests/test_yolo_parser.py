"""Tests for YOLO result parser."""
import pytest
from app.parsers.yolo_parser import YOLOResultParser


class TestYOLOResultParser:
    """Test suite for YOLOResultParser."""

    def test_parse_metrics_with_valid_csv(self, sample_csv_path, sample_yaml_path):
        """Test parsing metrics from valid CSV file."""
        parser = YOLOResultParser(sample_csv_path, sample_yaml_path)
        metrics = parser.parse_metrics()

        # Check all required keys exist
        assert "precision" in metrics
        assert "recall" in metrics
        assert "map50" in metrics
        assert "map50_95" in metrics
        assert "loss" in metrics

        # Check values are in valid ranges
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["map50"] <= 1.0
        assert 0.0 <= metrics["map50_95"] <= 1.0
        assert metrics["loss"] >= 0.0

    def test_parse_metrics_returns_last_epoch(self, sample_csv_path, sample_yaml_path):
        """Test that parser returns metrics from the last epoch."""
        parser = YOLOResultParser(sample_csv_path, sample_yaml_path)
        metrics = parser.parse_metrics()

        # Last epoch values from sample_results.csv (epoch 99)
        assert abs(metrics["precision"] - 0.7901) < 0.01
        assert abs(metrics["recall"] - 0.8195) < 0.01
        assert abs(metrics["map50"] - 0.8555) < 0.01

    def test_parse_config_with_valid_yaml(self, sample_csv_path, sample_yaml_path):
        """Test parsing config from valid YAML file."""
        parser = YOLOResultParser(sample_csv_path, sample_yaml_path)
        config = parser.parse_config()

        # Check expected keys
        assert "epochs" in config
        assert "batch" in config
        assert "lr0" in config
        assert "iou" in config
        assert "conf" in config

        # Check expected values from sample_args.yaml
        assert config["epochs"] == 100
        assert config["batch"] == 16
        assert config["lr0"] == 0.01

    def test_parse_config_without_yaml(self, sample_csv_path):
        """Test parsing config returns empty dict when no YAML provided."""
        parser = YOLOResultParser(sample_csv_path, yaml_path=None)
        config = parser.parse_config()

        assert config == {}

    def test_parse_metrics_file_not_found(self):
        """Test that FileNotFoundError is raised for missing CSV."""
        parser = YOLOResultParser("nonexistent.csv", None)

        with pytest.raises(FileNotFoundError):
            parser.parse_metrics()

    def test_parse_config_file_not_found(self, sample_csv_path):
        """Test that FileNotFoundError is raised for missing YAML."""
        parser = YOLOResultParser(sample_csv_path, "nonexistent.yaml")

        with pytest.raises(FileNotFoundError):
            parser.parse_config()
