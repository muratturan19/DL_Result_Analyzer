"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


@pytest.fixture
def sample_csv_path():
    """Path to sample results.csv."""
    return Path(__file__).parent.parent.parent / "examples" / "sample_results.csv"


@pytest.fixture
def sample_yaml_path():
    """Path to sample args.yaml."""
    return Path(__file__).parent.parent.parent / "examples" / "sample_args.yaml"


@pytest.fixture
def sample_data_yaml_path():
    """Path to sample data.yaml."""
    return Path(__file__).parent.parent.parent / "examples" / "sample_data.yaml"
