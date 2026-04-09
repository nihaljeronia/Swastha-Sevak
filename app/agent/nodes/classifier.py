"""Disease classification and urgency prediction node."""

from __future__ import annotations

import logging

from app.agent.states import TriageState
from app.ml.classifier import predict_triage

logger = logging.getLogger(__name__)


def classifier_node(state: TriageState) -> dict:
    """Classify urgency and predict top-3 disease conditions.

    Calls the XGBoost triage classifier (or mock) with extracted symptoms.

    Args:
        state: Triage state with symptoms extracted.

    Returns:
        Partial state update dict with urgency and conditions.
    """
    logger.info(f"[classifier] Classifying {len(state['symptoms'])} symptoms")

    if not state["symptoms"]:
        logger.warning("[classifier] No symptoms to classify, defaulting to routine")
        return {
            "urgency": "routine",
            "conditions": [],
        }

    # Call the classifier
    urgency, conditions = predict_triage(
        symptoms=state["symptoms"],
        entities=state["medical_entities"],
    )

    logger.info(
        f"[classifier] Urgency: {urgency}, "
        f"Top conditions: {[c['name'] for c in conditions[:3]]}"
    )

    return {
        "urgency": urgency,
        "conditions": conditions,
    }
