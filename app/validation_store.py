"""
Stockage persistant des validations cliniques (audit trail).

Fichier JSONL local — adapté API + déploiement Docker/HF avec volume.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "validations" / "clinical_validations.jsonl"


class ValidationStore:
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        task: str,
        model_prediction: str,
        model_confidence: float,
        clinician_verdict: str,
        comment: str = "",
        extra: Optional[dict] = None,
    ) -> dict:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "model_prediction": model_prediction,
            "model_confidence": round(float(model_confidence), 4),
            "clinician_verdict": clinician_verdict,
            "comment": comment.strip(),
            "extra": extra or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def list_recent(self, n: int = 50, task: Optional[str] = None) -> List[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        if task:
            entries = [e for e in entries if e.get("task") == task]
        return entries[-n:]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().splitlines() if line.strip())
