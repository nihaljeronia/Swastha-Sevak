"""CRUD operations for Swasthya Sevak database models.

All database interactions go through these functions.
Never write inline queries elsewhere in the codebase.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Alert, Message, Patient, TriageSession


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------


async def get_or_create_patient(
    session: AsyncSession,
    phone: str,
) -> tuple[Patient, bool]:
    """Return the Patient for *phone*, creating one if it doesn't exist.

    Returns:
        A tuple of (patient, created) where *created* is True when a new
        record was inserted.
    """
    stmt = select(Patient).where(Patient.phone == phone)
    result = await session.execute(stmt)
    patient = result.scalar_one_or_none()

    if patient is not None:
        return patient, False

    patient = Patient(phone=phone)
    session.add(patient)
    await session.commit()
    await session.refresh(patient)
    return patient, True


async def update_patient(
    session: AsyncSession,
    patient_id: uuid.UUID,
    **fields: Any,
) -> Patient | None:
    """Update mutable fields on an existing Patient record."""
    stmt = select(Patient).where(Patient.id == patient_id)
    result = await session.execute(stmt)
    patient = result.scalar_one_or_none()

    if patient is None:
        return None

    allowed = {"name", "language", "district", "state"}
    for key, value in fields.items():
        if key in allowed:
            setattr(patient, key, value)

    await session.commit()
    await session.refresh(patient)
    return patient


async def get_patient_by_phone(
    session: AsyncSession,
    phone: str,
) -> Patient | None:
    """Look up a Patient by phone number. Returns None if not found."""
    stmt = select(Patient).where(Patient.phone == phone)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


async def save_message(
    session: AsyncSession,
    patient_id: uuid.UUID,
    direction: str,
    message_type: str = "text",
    content: str | None = None,
    raw_payload: dict | None = None,
    session_id: uuid.UUID | None = None,
) -> Message:
    """Persist a single inbound or outbound WhatsApp message."""
    message = Message(
        patient_id=patient_id,
        session_id=session_id,
        direction=direction,
        message_type=message_type,
        content=content,
        raw_payload=raw_payload,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


# ---------------------------------------------------------------------------
# TriageSession
# ---------------------------------------------------------------------------


async def create_triage_session(
    session: AsyncSession,
    patient_id: uuid.UUID,
) -> TriageSession:
    """Start a new triage session for a patient."""
    triage = TriageSession(patient_id=patient_id)
    session.add(triage)
    await session.commit()
    await session.refresh(triage)
    return triage


async def get_active_triage_session(
    session: AsyncSession,
    patient_id: uuid.UUID,
) -> TriageSession | None:
    """Return the latest in-progress triage session for a patient, if any."""
    stmt = (
        select(TriageSession)
        .where(
            TriageSession.patient_id == patient_id,
            TriageSession.status == "in_progress",
        )
        .order_by(TriageSession.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def update_triage_session(
    session: AsyncSession,
    triage_session_id: uuid.UUID,
    **fields: Any,
) -> TriageSession | None:
    """Update fields on an existing TriageSession."""
    stmt = select(TriageSession).where(TriageSession.id == triage_session_id)
    result = await session.execute(stmt)
    triage = result.scalar_one_or_none()

    if triage is None:
        return None

    allowed = {
        "symptoms",
        "medical_entities",
        "urgency",
        "conditions",
        "advice",
        "follow_up_at",
        "follow_up_sent",
        "status",
        "completed_at",
    }
    for key, value in fields.items():
        if key in allowed:
            setattr(triage, key, value)

    await session.commit()
    await session.refresh(triage)
    return triage


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


async def create_alert(
    session: AsyncSession,
    session_id: uuid.UUID,
    doctor_phone: str,
    alert_type: str,
    patient_summary: str | None = None,
) -> Alert:
    """Create a new doctor alert for an emergency/urgent triage session."""
    alert = Alert(
        session_id=session_id,
        doctor_phone=doctor_phone,
        alert_type=alert_type,
        patient_summary=patient_summary,
    )
    session.add(alert)
    await session.commit()
    await session.refresh(alert)
    return alert


async def update_alert_status(
    session: AsyncSession,
    alert_id: uuid.UUID,
    status: str,
    sent_at: datetime | None = None,
) -> Alert | None:
    """Update the status of an existing Alert."""
    stmt = select(Alert).where(Alert.id == alert_id)
    result = await session.execute(stmt)
    alert = result.scalar_one_or_none()

    if alert is None:
        return None

    alert.status = status
    if sent_at is not None:
        alert.sent_at = sent_at

    await session.commit()
    await session.refresh(alert)
    return alert
