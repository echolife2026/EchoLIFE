from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UsageTracker:
    mode: str
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, stage: str, day: str, usage_metadata: Optional[Dict[str, Any]], response_metadata: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "stage": stage,
            "day": day,
            "usage_metadata": usage_metadata or {},
            "response_metadata": response_metadata or {},
        }
        if extra:
            rec.update(extra)
        self.records.append(rec)

    def summary(self) -> Dict[str, Any]:
        total_input = 0
        total_output = 0
        total = 0
        for r in self.records:
            u = r.get("usage_metadata", {}) or {}
            total_input += int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0)
            total_output += int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0)
            total += int(u.get("total_tokens", 0) or 0)
        return {
            "mode": self.mode,
            "num_calls": len(self.records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total if total else (total_input + total_output),
            "records": self.records,
        }
