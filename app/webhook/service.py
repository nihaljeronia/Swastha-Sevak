"""Webhook service layer — bridges the WhatsApp webhook routes and the database.

The webhook routes (app/webhook/routes.py) must NEVER import from app/db/
directly. All DB access goes through this service module.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.models import Message, Patient


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
