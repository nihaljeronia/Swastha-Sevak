from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PatientMessage


async def create_patient_message(
    session: AsyncSession,
    whatsapp_id: str,
    phone_number: str,
    message_text: str,
    language: str | None = None,
    message_type: str = "text",
) -> PatientMessage:
    message = PatientMessage(
        whatsapp_id=whatsapp_id,
        phone_number=phone_number,
        message_text=message_text,
        language=language,
        message_type=message_type,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message
