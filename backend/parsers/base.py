"""Base classes and interfaces for parsers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """Abstract base class for dataset parsers."""

    @abstractmethod
    def parse(self, source: str) -> list[dict[str, Any]]:
        """Parse the provided source and return structured records."""
        raise NotImplementedError
