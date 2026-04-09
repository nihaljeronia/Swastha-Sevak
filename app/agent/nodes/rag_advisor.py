"""Medical advice retrieval from FAISS knowledge base node."""

from __future__ import annotations

import logging

from app.agent.states import TriageState

logger = logging.getLogger(__name__)


def retrieve_medical_advice(
    conditions: list[dict], urgency: str, symptoms: list[str]
) -> str:
    """Retrieve medical advice from FAISS knowledge base.

    Placeholder for FAISS vector retrieval. Returns template advice.

    Args:
        conditions: List of predicted conditions with confidence.
        urgency: Urgency level (self_care, routine, urgent, emergency).
        symptoms: List of symptoms.

    Returns:
        Medical advice text in English.
    """
    if not conditions:
        return "Please consult a healthcare provider."

    top_condition = conditions[0]["name"]

    if urgency == "emergency":
        return (
            f"⚠️ **EMERGENCY** — {top_condition} appears to be life-threatening.\n"
            "Call 108 (ambulance) immediately."
        )
    elif urgency == "urgent":
        return (
            f"⚠️ **URGENT** — You should see a doctor today for {top_condition}.\n"
            "Visit the nearest health center now."
        )
    elif urgency == "routine":
        return (
            f"Based on your symptoms, you might have {top_condition}.\n"
            "Rest, stay hydrated, monitor your condition.\n"
            "See a doctor within a few days if symptoms persist or worsen."
        )
    else:  # self_care
        return (
            f"Your symptoms suggest {top_condition}, which is usually manageable with self-care.\n"
            "Rest well, drink fluids, and take over-the-counter pain relief if needed.\n"
            "See a doctor if symptoms worsen."
        )


def rag_advisor_node(state: TriageState) -> dict:
    """Retrieve medical advice from FAISS knowledge base.

    Args:
        state: Triage state with conditions.

    Returns:
        Partial state update dict with medical advice.
    """
    logger.info(f"[rag_advisor] Retrieving advice for urgency={state['urgency']}")

    if not state["conditions"]:
        logger.warning("[rag_advisor] No conditions to retrieve advice for")
        advice = "Please consult a healthcare provider."
    else:
        advice = retrieve_medical_advice(
            conditions=state["conditions"],
            urgency=state["urgency"],
            symptoms=state["symptoms"],
        )

    logger.info("[rag_advisor] Medical advice retrieved")
    return {"advice": advice}
