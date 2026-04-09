"""Response composition and message formatting node."""

from __future__ import annotations

import logging

from app.agent.states import TriageState
from app.nlp.models import translate_to_patient_lang

logger = logging.getLogger(__name__)


def compose_response(
    urgency: str, conditions: list[dict], advice: str, nearest_phc: dict | None
) -> str:
    """Compose formatted response message combining all information.

    Args:
        urgency: Urgency level.
        conditions: List of predicted conditions.
        advice: Medical advice text.
        nearest_phc: Nearest PHC dict with name, phone, distance_km.

    Returns:
        Formatted response message in English.
    """
    lines = []

    # Add assessment header
    lines.append("🏥 **Summary**")
    lines.append("")

    # Add conditions if available
    if conditions:
        top_2 = [c["name"] for c in conditions[:2]]
        lines.append(f"Possible conditions: {', '.join(top_2)}")
        lines.append("")

    # Add medical advice
    lines.append(advice)
    lines.append("")

    # Add nearest PHC info if urgent/emergency
    if nearest_phc and urgency in ["emergency", "urgent"]:
        lines.append("📍 **Nearest Health Center**")
        lines.append(f"{nearest_phc.get('name', 'Not found')}")
        lines.append(f"📞 {nearest_phc.get('phone', 'N/A')}")
        lines.append(f"Distance: {nearest_phc.get('distance_km', '?')} km")
        lines.append("")

    # Add emergency contact
    if urgency == "emergency":
        lines.append("🚑 **For life-threatening emergencies:**")
        lines.append("Call 108 (ambulance service)")

    # Add follow-up reminder
    lines.append("")
    lines.append("We'll check in with you in 24 hours to see how you're doing.")

    return "\n".join(lines)


def response_composer_node(state: TriageState) -> dict:
    """Compose formatted WhatsApp response in patient's language.

    Combines urgency level, predicted conditions, medical advice,
    and nearest PHC information into a cohesive message.

    Args:
        state: Triage state with all classification results.

    Returns:
        Partial state update dict with response message added to messages.
    """
    logger.info(f"[response_composer] Composing response in {state['language']}")

    # Build response in English first
    response_en = compose_response(
        urgency=state["urgency"],
        conditions=state["conditions"],
        advice=state["advice"],
        nearest_phc=state.get("nearest_phc"),
    )

    # Translate to patient's language
    response_lang = translate_to_patient_lang(response_en, state["language"])

    # Add to messages
    messages = state["messages"].copy() if state["messages"] else []
    messages.append(
        {
            "role": "assistant",
            "content": response_lang,
            "language": state["language"],
        }
    )

    logger.info("[response_composer] Response composed and added to messages")
    return {"messages": messages}
