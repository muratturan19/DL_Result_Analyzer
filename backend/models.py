"""Pydantic models and domain entities for the backend service."""

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    """Schema describing a request to analyze a set of experiment results."""

    dataset_path: str
    parser: str


class AnalysisResult(BaseModel):
    """Schema for returning summary statistics to the client."""

    accuracy: float
    loss: float
    metadata: dict[str, str] | None = None
