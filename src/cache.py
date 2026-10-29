"""Resumable JSONL scoring cache (Section 8/10/12: cache keys include model
revision, full prompt text, verbalizer set, and scoring settings; resuming a
run skips already-cached (prompt, label) scores instead of recomputing)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EvalCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._mem[row["key"]] = row["value"]

    @staticmethod
    def make_key(*, model_id: str, model_revision: str, prompt_text: str, verbalizers: list[str],
                 aggregation: str, max_input_tokens: int) -> str:
        payload = json.dumps(
            {
                "model_id": model_id,
                "model_revision": model_revision,
                "prompt_text": prompt_text,
                "verbalizers": verbalizers,
                "aggregation": aggregation,
                "max_input_tokens": max_input_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str):
        return self._mem.get(key)

    def put(self, key: str, value: dict) -> None:
        self._mem[key] = value
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "value": value}, default=str) + "\n")

    def __len__(self) -> int:
        return len(self._mem)
