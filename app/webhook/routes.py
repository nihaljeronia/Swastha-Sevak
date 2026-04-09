"""WhatsApp webhook routes.

- GET  /webhook — Meta verification (returns hub.challenge)
- POST /webhook — Receives messages, persists them, sends echo reply
"""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.webhook.service import process_incoming_message, save_outbound_message

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET — Webhook verification
# ---------------------------------------------------------------------------


@router.get("")
async def verify_webhook(request: Request) -> Response:
    """Meta WhatsApp webhook verification endpoint."""
    params = dict(request.query_params)

    hub_mode = params.get("hub.mode") or params.get("hub_mode")
    hub_challenge = params.get("hub.challenge") or params.get("hub_challenge")
    hub_verify_token = params.get("hub.verify_token") or params.get("hub_verify_token")

    if hub_mode == "subscribe" and hub_verify_token == settings.meta_verify_token:
        return Response(content=hub_challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification failed")


# ---------------------------------------------------------------------------
# POST — Receive messages
# ---------------------------------------------------------------------------


@router.post("")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Receive incoming WhatsApp messages from Meta webhook.

    Returns 200 immediately, processes the message in a background task.
    """
    body = await request.json()
    logger.info("WEBHOOK RECEIVED: %s", json.dumps(body, indent=2))

    # Extract message data from Meta's nested structure
    entry = body.get("entry", [])
    if not entry:
        return {"status": "ok"}

    changes = entry[0].get("changes", [])
    if not changes:
        return {"status": "ok"}

    value = changes[0].get("value", {})
    messages = value.get("messages")

    if not messages:
        # Status update (delivered, read), not a message
        return {"status": "ok"}

    message = messages[0]
    sender_phone: str = message.get("from", "")
    message_type: str = message.get("type", "text")

    # Determine content based on message type
    if message_type == "text":
        content = message.get("text", {}).get("body", "")
    elif message_type == "audio":
        content = f"[audio:{message.get('audio', {}).get('id', '')}]"
    elif message_type == "interactive":
        interactive = message.get("interactive", {})
        content = (
            interactive.get("button_reply", {}).get("id")
            or interactive.get("list_reply", {}).get("id")
            or ""
        )
    else:
        content = f"[{message_type}]"

    # Process in background so we return 200 immediately to Meta
    background_tasks.add_task(
        _process_and_reply,
        sender_phone,
        message_type,
        content,
        body,
    )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------


async def _process_and_reply(
    sender_phone: str,
    message_type: str,
    content: str,
    raw_payload: dict,
) -> None:
    """Background task: persist message to DB, then send echo reply."""
    try:
        async with AsyncSessionLocal() as session:
            # 1. Persist inbound message
            patient, msg = await process_incoming_message(
                session=session,
                phone=sender_phone,
                message_type=message_type,
                content=content,
                raw_payload=raw_payload,
            )
            logger.info(
                "Saved inbound message %s from patient %s (phone=%s)",
                msg.id,
                patient.id,
                sender_phone,
            )

            # 2. Compose reply (echo for now — will be replaced by agent)
            if message_type == "text":
                reply_text = f"You said: {content}"
            elif message_type == "audio":
                reply_text = "Voice note received! (processing coming soon)"
            elif message_type == "interactive":
                reply_text = f"You selected: {content}"
            else:
                reply_text = f"Received your {message_type} message."

            # 3. Send reply via Meta API
            await _send_whatsapp_reply(sender_phone, reply_text)

            # 4. Persist outbound message
            await save_outbound_message(
                session=session,
                patient_id=patient.id,
                content=reply_text,
            )
            logger.info("Saved outbound reply to patient %s", patient.id)

    except Exception:
        logger.exception("Error processing message from %s", sender_phone)


# ---------------------------------------------------------------------------
# WhatsApp reply helper
# ---------------------------------------------------------------------------


async def _send_whatsapp_reply(to: str, text: str) -> None:
    """Send a text message back to the user via Meta Cloud API."""
    url = (
        f"https://graph.facebook.com/v21.0/"
        f"{settings.meta_phone_number_id}/messages"
    )
    headers = {
        "Authorization": f"Bearer {settings.meta_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        logger.info("SEND REPLY status=%s: %s", response.status_code, response.text)
