"""Escalation and doctor alert node."""

from __future__ import annotations

import logging

from app.agent.states import TriageState
from app.db.crud import get_nearest_phc

logger = logging.getLogger(__name__)


def escalation_node(state: TriageState) -> dict:
    """Handle escalation for emergency and urgent cases.

    Finds nearest PHC and prepares escalation info if urgency is
    'emergency' or 'urgent'.

    Args:
        state: Triage state with urgency classification.

    Returns:
        Partial state update dict with nearest_phc if escalation needed.
    """
    logger.info(f"[escalation] Checking escalation for urgency={state['urgency']}")

    updates = {}

    if state["urgency"] not in ["emergency", "urgent"]:
        logger.info("[escalation] No escalation needed")
        return updates

    # Find nearest PHC
    logger.info("[escalation] Finding nearest PHC")
    try:
        nearest_phc = get_nearest_phc(state["patient_phone"])
        if nearest_phc:
            updates["nearest_phc"] = nearest_phc
            logger.warning(
                f"[escalation] ESCALATION: {state['urgency'].upper()} for "
                f"{state['patient_phone']} → {nearest_phc.get('name')}"
            )
        else:
            logger.warning(
                f"[escalation] {state['urgency'].upper()} escalation but no PHC found"
            )
    except Exception as e:
        logger.error(f"[escalation] Error finding nearest PHC: {e}")

    return updates


def should_escalate(state: TriageState) -> str:
    """Conditional edge: determine if escalation is needed.

    Routes to 'response_composer' directly (escalation info already retrieved).

    Args:
        state: Current triage state.

    Returns:
        Node name: always 'response_composer' (escalation is prepared, not a node).
    """
    logger.info(f"[should_escalate] Checking urgency={state['urgency']}")
    if state["urgency"] in ["emergency", "urgent"]:
        logger.info("[should_escalate] Escalation prepared, continuing to response_composer")
    return "response_composer"
