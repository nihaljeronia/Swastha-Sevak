from fastapi import APIRouter, status

from app.webhook.schemas import WhatsAppWebhookEvent
from app.webhook.service import enqueue_patient_message

router = APIRouter()


@router.post("/whatsapp", status_code=status.HTTP_202_ACCEPTED)
async def whatsapp_webhook(event: WhatsAppWebhookEvent):
    """Accept incoming WhatsApp webhook payload and delegate processing."""
    await enqueue_patient_message(event)
    return {"detail": "Accepted"}
