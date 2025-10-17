"""CSV parser implementation for experiment results."""

import csv
from io import StringIO
from typing import Any

from .base import BaseParser


class CSVParser(BaseParser):
    """Parse CSV formatted experiment outputs."""

    def parse(self, source: str) -> list[dict[str, Any]]:
        reader = csv.DictReader(StringIO(source))
        return [dict(row) for row in reader]
