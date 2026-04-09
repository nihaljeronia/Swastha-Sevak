"""Language detection and initial symptom extraction node."""

from __future__ import annotations

import logging

from app.agent.states import TriageState
from app.nlp.language_id import detect_language
from app.nlp.models import process_message, translate_to_en

logger = logging.getLogger(__name__)


def language_detect_node(state: TriageState) -> dict:
    """Detect patient language from first message and extract initial symptoms.

    This is the entry point node. It:
    1. Detects language from first user message
    2. Translates to English if needed
    3. Extracts initial symptoms and medical entities

    Args:
        state: Current triage state with at least one message.

    Returns:
        Partial state update dict with language, symptoms, and medical_entities.
    """
    logger.info(f"[language_detect] Processing message from {state['patient_phone']}")

    # Get the first user message
    first_message = next(
        (msg for msg in state["messages"] if msg.get("role") == "user"), None
    )
    if not first_message:
        logger.warning("[language_detect] No user message found in state")
        return {
            "language": "en",
            "symptoms": [],
            "medical_entities": {},
            "question_count": 0,
        }

    # Detect language
    content = first_message["content"]
    language = detect_language(content)
    logger.info(f"[language_detect] Detected language: {language}")

    # Translate to English if needed
    if language != "en":
        content_en = translate_to_en(content, language)
    else:
        content_en = content

    # Extract initial symptoms and medical entities
    symptoms, entities = process_message(content_en)
    logger.info(f"[language_detect] Extracted {len(symptoms)} initial symptoms")

    return {
        "language": language,
        "symptoms": symptoms or [],
        "medical_entities": entities or {},
        "question_count": 0,
    }
