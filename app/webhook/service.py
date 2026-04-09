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


async def run_triage_agent(
    session: AsyncSession,
    patient: Patient,
    text: str,
) -> dict[str, Any]:
    """Run the TriageAgent (LangGraph) on an incoming patient text message.

    Steps:
    1. Get or create an active TriageSession from the database.
    2. Hydrate the LangGraph agent's state with the existing DB session.
    3. Execute the Multi-Agent loop to process NLP, predict urgency, and compose reply.
    4. Save updated conditions back into the PostgreSQL DB.
    """
    from app.agents.graph import TriageAgent

    # 1. Get or create the active triage session
    triage = await crud.get_active_triage_session(session, patient.id)
    if triage is None:
        triage = await crud.create_triage_session(session, patient.id)

    # 2. Fetch conversational context
    past_messages = await crud.get_messages_for_session(session, triage.id)
    formatted_messages = []
    for msg in past_messages:
        # Ignore the current inbound message being processed right now since we append it later
        if msg.content == text and msg.direction == "inbound":
            continue
        role = "user" if msg.direction == "inbound" else "assistant"
        formatted_messages.append({"role": role, "content": msg.content or ""})

    # 3. Setup the LangGraph agent
    agent = TriageAgent()
    await agent.load_model()
    
    # Hydrate LangGraph State from Postgres
    agent.checkpoint["state"] = {
        "patient_phone": patient.phone,
        "language": patient.language or "en",
        "symptoms": list(triage.symptoms or []),
        "medical_entities": dict(triage.medical_entities or {}),
        "question_count": len(triage.symptoms or []),
        "messages": formatted_messages,
    }
    
    # 3. Process the new message using the Agent Graph
    patient_data = {
        "phone": patient.phone,
        "message": text
    }
    result = await agent.run(patient_data)
    
    final_state = result["state"]
    new_symptoms = final_state.get("symptoms", [])
    new_entities = final_state.get("medical_entities", {})
    urgency = final_state.get("urgency", "routine")
    detected_language = final_state.get("language", "en")
    
    # Update language preference if newly detected
    if not patient.language and detected_language != "en":
        await crud.update_patient(session, patient.id, language=detected_language)
        
    session_completed = final_state.get("session_completed", False)
    status_flag = "completed" if session_completed else "in_progress"
    
    # 4. Save updated state back to DB
    from datetime import datetime
    await crud.update_triage_session(
        session,
        triage.id,
        symptoms=new_symptoms,
        medical_entities=new_entities,
        urgency=urgency,
        status=status_flag,
        completed_at=datetime.utcnow() if session_completed else None
    )
    
    # Automatically execute LLM summarization and insert into final summary tables
    if session_completed:
        import os
        from app.db.models import ConversationSummary
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from pydantic import BaseModel, Field
            
            class SummaryOutput(BaseModel):
                topic: str = Field(description="A short 2-3 word topic name (e.g. 'Viral Fever Intake')")
                summary: str = Field(description="A 3-sentence clinical summary of the whole conversation, symptoms, duration, and diagnosis.")
                
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            chain = ChatPromptTemplate.from_template("Summarize this clinical conversation:\n{history}") | llm.with_structured_output(SummaryOutput)
            
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in formatted_messages])
            try:
                res = await chain.ainvoke({"history": history_str})
                summary_record = ConversationSummary(patient_id=patient.id, session_id=triage.id, topic=res.topic, summary_text=res.summary)
                session.add(summary_record)
                await session.commit()
                logger.info(f"Generated Session Summary successfully: {res.topic}")
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
        else:
            # Fallback
            summary_record = ConversationSummary(patient_id=patient.id, session_id=triage.id, topic=urgency.capitalize(), summary_text="System finalized diagnosis without LLM.")
            session.add(summary_record)
            await session.commit()
            
    logger.info(
        "Agent finished session %s [STATUS: %s] — symptoms=%s urgency=%s",
        triage.id,
        status_flag,
        new_symptoms,
        urgency,
    )

    return {
        "triage_session_id": triage.id,
        "reply_text": result.get("reply", "I am having trouble understanding. Please try again.")
    }


async def download_media(media_id: str) -> bytes:
    """Download encrypted media binary from Meta WhatsApp API."""
    import httpx
    from app.core.config import settings

    headers = {"Authorization": f"Bearer {settings.meta_access_token}"}
    async with httpx.AsyncClient() as client:
        # Step 1: Query the Graph API for the private media URL
        metadata_res = await client.get(
            f"https://graph.facebook.com/v21.0/{media_id}",
            headers=headers,
        )
        metadata_res.raise_for_status()
        
        media_url = metadata_res.json().get("url")
        if not media_url:
            raise ValueError("Media URL not found in Graph API response.")
            
        # Step 2: Download the raw bytes using the same Bearer token
        media_res = await client.get(media_url, headers=headers)
        media_res.raise_for_status()
        
        return media_res.content

