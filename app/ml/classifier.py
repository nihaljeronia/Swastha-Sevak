from typing import Any


class DiseaseClassifier:
    def __init__(self) -> None:
        self.model: Any | None = None

    async def load_model(self) -> None:
        self.model = "xgboost-classifier"

    async def shutdown(self) -> None:
        self.model = None

    async def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return {"disease_code": "unknown", "confidence": 0.0}
