from typing import Any


class TriageAgent:
    def __init__(self) -> None:
        self.checkpoint: dict[str, Any] = {}

    async def initialize(self) -> None:
        self.checkpoint = {}

    async def run(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        return {"triage_level": "unknown", "next_step": "awaiting_data"}
