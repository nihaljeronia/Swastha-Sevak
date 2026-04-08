from app.webhook.schemas import WhatsAppWebhookEvent


async def enqueue_patient_message(event: WhatsAppWebhookEvent) -> None:
    """Send patient message to the async processing pipeline."""
    # TODO: persist the incoming event through app/db/crud.py and enqueue a Celery task.
    return None
