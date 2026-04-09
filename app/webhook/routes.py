"""WhatsApp webhook routes.

- GET  /webhook — Meta verification (returns hub.challenge)
- POST /webhook — Receives messages, persists them, sends echo reply
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.webhook.service import (
    process_incoming_message,
    run_triage_agent,
    save_outbound_message,
)

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
async def receive_webhook(request: Request) -> dict[str, str]:
    """Receive incoming WhatsApp messages from Meta webhook."""
    body = await request.json()

    try:
        entry = body.get("entry", [])
        if not entry:
            return {"status": "ok"}

        changes = entry[0].get("changes", [])
        if not changes:
            return {"status": "ok"}

        value = changes[0].get("value", {})
        messages = value.get("messages")

        if not messages:
            return {"status": "ok"}

        message = messages[0]
        sender_phone = message.get("from")
        message_type = message.get("type")

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

        logger.info("Inbound from=%s type=%s content=%r", sender_phone, message_type, content[:80])

        # --- DB: persist inbound message ---
        async with AsyncSessionLocal() as session:
            patient, msg = await process_incoming_message(
                session=session,
                phone=sender_phone,
                message_type=message_type,
                content=content,
                raw_payload=body,
            )
            logger.info("Saved inbound message=%s patient=%s", msg.id, patient.id)

            triage_session_id = None
            if message_type in ("text", "audio"):
                from app.webhook.service import run_triage_agent, download_media
                
                # Automatically transcribe voice notes
                if message_type == "audio":
                    audio_id = message.get("audio", {}).get("id")
                    audio_bytes = await download_media(audio_id)
                    nlp_manager = request.app.state.nlp_manager
                    
                    # Assuming we know patient lang, or whisper auto-detects
                    content = await nlp_manager.transcribe(audio_bytes, source_language=patient.language)
                    
                    if not content:
                        # Fallback natively if missing or bad transcription
                        content = "(System: Transcribing voice note failed. The patient sent audio but it could not be understood.)"

                # Run unified intelligent multi-agent pipeline using Transcribed or Text Content
                result = await run_triage_agent(
                    session=session,
                    patient=patient,
                    text=content,
                )
                triage_session_id = result["triage_session_id"]
                reply_text = result["reply_text"]
            elif message_type == "interactive":
                reply_text = (
                    f"You selected: {content}. "
                    "Please tell me more about your symptoms."
                )
            else:
                reply_text = f"Received your {message_type} message."

            # --- Generate & Send Audio Reply via Meta API ---
            # Automatically parse the response and natively encode back into localized TTS dialects!
            audio_bytes = await request.app.state.nlp_manager.text_to_speech(reply_text, patient.language or "hi")
            if audio_bytes:
                try:
                    media_id = await upload_audio_to_meta(audio_bytes)
                    if media_id:
                        await send_audio_reply(sender_phone, media_id)
                except Exception as e:
                    logger.error("Failed to push audio via Meta: %s", e)

            # --- Send Text Reply via Meta API ---
            await send_reply(sender_phone, reply_text)

            # --- DB: persist outbound message ---
            await save_outbound_message(
                session=session,
                patient_id=patient.id,
                content=reply_text,
                session_id=triage_session_id,
            )
            logger.info("Reply sent and saved for patient=%s", patient.id)

    except Exception:
        logger.exception("Error processing webhook")

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WhatsApp reply helper & Audio upload
# ---------------------------------------------------------------------------


async def upload_audio_to_meta(audio_bytes: bytes) -> str:
    """Upload TTS audio binary to Meta Graph and retain the server Media ID."""
    url = f"https://graph.facebook.com/v21.0/{settings.meta_phone_number_id}/media"
    headers = {"Authorization": f"Bearer {settings.meta_access_token}"}
    files = {
        'file': ('audio.mp3', audio_bytes, 'audio/mpeg'),
        'type': (None, 'audio'),
        'messaging_product': (None, 'whatsapp')
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, files=files)
        res.raise_for_status()
        return res.json().get("id")


async def send_audio_reply(to: str, media_id: str) -> None:
    """Dispatch Meta Media Asset out to the client natively."""
    url = f"https://graph.facebook.com/v21.0/{settings.meta_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.meta_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "audio",
        "audio": {"id": media_id},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        logger.info("Audio Reply to=%s status=%s", to, response.status_code)


async def send_reply(to: str, text: str) -> None:
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
        logger.info("Text Reply to=%s status=%s", to, response.status_code)

