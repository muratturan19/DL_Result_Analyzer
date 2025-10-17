"""Aggregate statistics over parsed experiment records."""

from statistics import mean
from typing import Iterable


class PerformanceAnalyzer:
    """Compute basic metrics from experiment results."""

    def summarize_accuracy(self, values: Iterable[float]) -> float:
        """Return the average accuracy for the provided values."""
        values = list(values)
        if not values:
            return 0.0
        return mean(values)
