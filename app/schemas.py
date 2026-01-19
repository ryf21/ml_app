from typing import Any, Dict, List

from pydantic import BaseModel


class TextIn(BaseModel):
    text: str


class BatchIn(BaseModel):
    texts: List[str]


class PredItem(BaseModel):
    label: str
    score: float
    raw: Dict[str, Any] | None = None


class PredictOut(BaseModel):
    text: str
    prediction: PredItem


class BatchOut(BaseModel):
    results: List[PredictOut]


class ModelInfoOut(BaseModel):
    model_id: str
    framework: str | None = None
    extra: Dict[str, Any] | None = None
