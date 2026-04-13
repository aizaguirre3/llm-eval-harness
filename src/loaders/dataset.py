from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from src.config import DATA_DIR


class QAPair(BaseModel):
    id: str
    question: str
    expected_answer: str
    context: str = ""
    metadata: Dict = {}


class DatasetLoader:
    """Loads Q&A evaluation datasets from JSON files."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load(self, filename: str) -> List[QAPair]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        raw = json.loads(path.read_text())
        return [QAPair.model_validate(item) for item in raw]

    def load_by_category(self, filename: str, category: str) -> List[QAPair]:
        pairs = self.load(filename)
        return [p for p in pairs if p.metadata.get("category") == category]

    def list_datasets(self) -> List[str]:
        return [f.name for f in self.data_dir.glob("*.json")]
