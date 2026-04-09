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

    Logic:
    - No symptoms found     → greet and ask the patient to describe their problem.
    - 1–2 symptoms found    → acknowledge and ask for more detail / duration.
    - 3+ symptoms collected → run classifier and return real triage response.

    The English response is translated to the patient's detected language before
    being returned.
    """
    from app.ml.classifier import predict_from_text

    language = nlp_result["language"]
    all_symptoms: list[str] = nlp_result["all_symptoms"]
    english_text = nlp_result.get("english_text", "")

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
        # Run the classifier for real triage
        result = predict_from_text(english_text, all_symptoms)
        urgency = result.get("urgency", "routine")
        conditions = result.get("conditions", [])
        mapped = result.get("mapped_symptoms", all_symptoms)

        urgency_emoji = {
            "self_care": "\U0001f7e2",
            "routine": "\U0001f7e1",
            "urgent": "\U0001f7e0",
            "emergency": "\U0001f534",
        }
        urgency_advice = {
            "self_care": "You can manage this at home with rest and fluids.",
            "routine": "Please visit a doctor soon.",
            "urgent": "Please visit your nearest health center today.",
            "emergency": "GO TO THE NEAREST HOSPITAL IMMEDIATELY! Call 108 for ambulance!",
        }

        emoji = urgency_emoji.get(urgency, "\U0001f7e1")
        advice = urgency_advice.get(urgency, "Please consult a doctor.")
        symptom_names = ", ".join(s.replace("_", " ") for s in mapped) if mapped else ", ".join(s.replace("_", " ") for s in all_symptoms)

        # Format top condition
        condition_line = ""
        if conditions:
            top = conditions[0]
            confidence_pct = int(top["confidence"] * 100)
            condition_line = f"\nPossible condition: {top['name']} ({confidence_pct}% confidence)"

        english_reply = (
            f"{emoji} *Swastha Sevak Triage*\n\n"
            f"Your symptoms: {symptom_names}\n"
            f"{condition_line}\n\n"
            f"Urgency: *{urgency.upper()}*\n"
            f"{advice}\n\n"
        )

        if urgency in ("urgent", "emergency"):
            english_reply += (
                "Nearest: Community Health Center, Bhopal Road\n"
                "Phone: 0755-XXXXXXX\n"
                "Ambulance: 108\n\n"
            )

        english_reply += "This is an AI-based preliminary assessment, not a replacement for a doctor. Please consult a doctor."

        logger.info(
            "Triage result: urgency=%s conditions=%s symptoms=%s",
            urgency,
            [c["name"] for c in conditions[:3]],
            mapped,
        )

    if language == "en":
        return english_reply

    translated = await nlp_manager.translate_from_english(english_reply, language)
    return translated
