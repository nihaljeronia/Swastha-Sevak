"""Webhook service layer — bridges the WhatsApp webhook routes and the database.

The webhook routes (app/webhook/routes.py) must NEVER import from app/db/
directly. All DB access goes through this service module.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.models import Message, Patient

if TYPE_CHECKING:
    from app.nlp.models import NLPModelManager

logger = logging.getLogger(__name__)


async def process_incoming_message(
    session: AsyncSession,
    phone: str,
    message_type: str,
    content: str | None,
    raw_payload: dict[str, Any] | None = None,
) -> tuple[Patient, Message]:
    """Handle a new inbound WhatsApp message.

    1. Get-or-create the Patient from the sender's phone number.
    2. Save the inbound Message to the database.

    Returns:
        (patient, message) tuple for downstream processing.
    """
    patient, _created = await crud.get_or_create_patient(session, phone)
    message = await crud.save_message(
        session=session,
        patient_id=patient.id,
        direction="inbound",
        message_type=message_type,
        content=content,
        raw_payload=raw_payload,
    )
    return patient, message


async def save_outbound_message(
    session: AsyncSession,
    patient_id: uuid.UUID,
    content: str,
    session_id: uuid.UUID | None = None,
) -> Message:
    """Persist an outbound reply we sent back to the patient."""
    message = await crud.save_message(
        session=session,
        patient_id=patient_id,
        direction="outbound",
        message_type="text",
        content=content,
        session_id=session_id,
    )
    return message


async def run_nlp_pipeline(
    session: AsyncSession,
    patient: Patient,
    nlp_manager: NLPModelManager,
    text: str,
) -> dict[str, Any]:
    """Run the full NLP pipeline on an incoming patient text message.

    Steps:
    1. Detect language.
    2. Persist detected language on the Patient record (first message only).
    3. Translate to English for NER.
    4. Extract medical entities (symptoms, duration, severity).
    5. Get or create an active TriageSession.
    6. Accumulate new symptoms and entities into the session.

    Returns:
        Dict with keys: language, english_text, entities,
        triage_session_id, all_symptoms, is_new_session.
    """
    # 1. Detect language
    language = nlp_manager.detect_language(text)
    logger.info("Detected language=%s for patient=%s", language, patient.id)

    # 2. Persist language preference (only when first recorded)
    if not patient.language:
        await crud.update_patient(session, patient.id, language=language)

    # 3. Translate to English for downstream NER
    if language == "en":
        english_text = text
    else:
        english_text = await nlp_manager.translate_to_english(text, language)
        logger.info("Translated to English: %r", english_text[:80])

    # 4. Extract medical entities from English text
    entities = nlp_manager.extract_entities(english_text)

    # 5. Get or create the active triage session
    triage = await crud.get_active_triage_session(session, patient.id)
    is_new_session = triage is None
    if triage is None:
        triage = await crud.create_triage_session(session, patient.id)

    # 6. Merge new symptoms (avoid duplicates) and update entities
    existing_symptoms: list[str] = list(triage.symptoms or [])
    new_symptoms = [s for s in entities["symptoms"] if s not in existing_symptoms]
    all_symptoms = existing_symptoms + new_symptoms

    existing_entities: dict[str, Any] = dict(triage.medical_entities or {})
    if entities.get("duration"):
        existing_entities["duration"] = entities["duration"]
    if entities.get("severity"):
        existing_entities["severity"] = entities["severity"]

    await crud.update_triage_session(
        session,
        triage.id,
        symptoms=all_symptoms,
        medical_entities=existing_entities,
    )
    logger.info(
        "Session %s updated — symptoms=%s entities=%s",
        triage.id,
        all_symptoms,
        existing_entities,
    )

    return {
        "language": language,
        "english_text": english_text,
        "entities": entities,
        "triage_session_id": triage.id,
        "all_symptoms": all_symptoms,
        "is_new_session": is_new_session,
    }


async def compose_triage_reply(
    nlp_result: dict[str, Any],
    nlp_manager: NLPModelManager,
) -> str:
    """Compose a conversational triage reply in the patient's language.

    Keeps the bot helpful for Step 3 before the LangGraph agent (Step 4)
    takes over the full conversation loop.

    Logic:
    - No symptoms found     → greet and ask the patient to describe their problem.
    - 1–2 symptoms found    → acknowledge and ask for more detail / duration.
    - 3+ symptoms collected → acknowledge and inform that analysis is in progress.

    The English response is translated to the patient's detected language before
    being returned.
    """
    language = nlp_result["language"]
    all_symptoms: list[str] = nlp_result["all_symptoms"]

    if not all_symptoms:
        english_reply = (
            "Hello! I am Swastha Sevak, your health assistant. "
            "Please describe what you are feeling — fever, pain, cough, "
            "or any other symptom."
        )
    elif len(all_symptoms) < 3:
        names = ", ".join(s.replace("_", " ") for s in all_symptoms)
        english_reply = (
            f"I understand you have: {names}. "
            "Can you tell me more? How long have you had these symptoms, "
            "and is there anything else troubling you?"
        )
    else:
        names = ", ".join(s.replace("_", " ") for s in all_symptoms)
        english_reply = (
            f"Thank you. I have noted your symptoms: {names}. "
            "I am analysing your condition. "
            "If this is an emergency, please call 108 immediately."
        )

    if language == "en":
        return english_reply

    translated = await nlp_manager.translate_from_english(english_reply, language)
    return translated
