from typing import Any
import logging
import os

logger = logging.getLogger(__name__)

USE_MOCK_CLASSIFIER = os.getenv("MOCK_CLASSIFIER", "false").lower() in ("true", "1", "yes")


class DiseaseClassifier:
    def __init__(self) -> None:
        self.model: Any | None = None

    async def load_model(self) -> None:
        self.model = "xgboost-classifier"

    async def shutdown(self) -> None:
        self.model = None

    async def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return {"disease_code": "unknown", "confidence": 0.0}


# ============================================================================
# Module-level function for agent nodes (with mock support)
# ============================================================================


def predict_triage(
    symptoms: list[str],
    entities: dict | None = None,
) -> tuple[str, list[dict]]:
    """Classify urgency level and predict top-3 disease conditions.

    Args:
        symptoms: List of extracted symptoms.
        entities: Medical entities dict (optional, for future enhancement).

    Returns:
        Tuple of (urgency_level, top_3_conditions).
        - urgency_level: one of "self_care", "routine", "urgent", "emergency"
        - top_3_conditions: list of dicts [{"name": str, "confidence": float}, ...]
    """
    if USE_MOCK_CLASSIFIER:
        logger.info("[predict_triage] MOCK: returning hardcoded triage")
        mock_urgency = "routine"
        mock_conditions = [
            {"name": "Common Cold", "confidence": 0.65},
            {"name": "Upper Respiratory Infection", "confidence": 0.25},
            {"name": "Influenza", "confidence": 0.10},
        ]
        return mock_urgency, mock_conditions

    # Real classifier (placeholder)
    logger.warning("[predict_triage] Real classifier not yet implemented")
    return "routine", [{"name": "Unknown", "confidence": 0.0}]
