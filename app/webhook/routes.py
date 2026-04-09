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
from app.nlp.models import NLPModelManager
from app.webhook.service import (
    compose_triage_reply,
    process_incoming_message,
    run_nlp_pipeline,
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

    print("=" * 60)
    print("WEBHOOK RAW BODY:")
    print(json.dumps(body, indent=2))
    print("=" * 60)

    try:
        entry = body.get("entry", [])
        print(f"ENTRY: {len(entry)} entries")

        if not entry:
            print("NO ENTRY — returning")
            return {"status": "ok"}

        changes = entry[0].get("changes", [])
        print(f"CHANGES: {len(changes)} changes")

        if not changes:
            print("NO CHANGES — returning")
            return {"status": "ok"}

        value = changes[0].get("value", {})
        print(f"VALUE KEYS: {list(value.keys())}")

        messages = value.get("messages")
        print(f"MESSAGES: {messages}")

        if not messages:
            print("NO MESSAGES — this is a status update, not a user message")
            return {"status": "ok"}

        message = messages[0]
        sender_phone = message.get("from")
        message_type = message.get("type")
        print(f"SENDER: {sender_phone}, TYPE: {message_type}")

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

        print(f"CONTENT: {content}")

        # --- DB: persist inbound message ---
        print("SAVING to database...")
        async with AsyncSessionLocal() as session:
            patient, msg = await process_incoming_message(
                session=session,
                phone=sender_phone,
                message_type=message_type,
                content=content,
                raw_payload=body,
            )
            print(f"DB SAVED: patient={patient.id}, message={msg.id}")

            # --- NLP pipeline (text only) ---
            triage_session_id = None
            if message_type == "text" and content:
                print("RUNNING NLP pipeline...")
                nlp_manager = request.app.state.nlp_manager
                nlp_result = await run_nlp_pipeline(
                    session=session,
                    patient=patient,
                    nlp_manager=nlp_manager,
                    text=content,
                )
                triage_session_id = nlp_result["triage_session_id"]
                print(f"NLP RESULT: lang={nlp_result['language']}, "
                      f"symptoms={nlp_result['all_symptoms']}, "
                      f"session={triage_session_id}")
                reply_text = await compose_triage_reply(nlp_result, nlp_manager)
            elif message_type == "audio":
                reply_text = (
                    "Voice note received! Automatic transcription is coming soon. "
                    "For now, please type your symptoms."
                )
            elif message_type == "interactive":
                reply_text = (
                    f"You selected: {content}. "
                    "Please tell me more about your symptoms."
                )
            else:
                reply_text = f"Received your {message_type} message."

            print(f"REPLY TEXT: {reply_text}")

            # --- Send reply via Meta API ---
            print("CALLING send_reply...")
            await send_reply(sender_phone, reply_text)
            print("send_reply COMPLETED")

            # --- DB: persist outbound message ---
            print("SAVING outbound message...")
            await save_outbound_message(
                session=session,
                patient_id=patient.id,
                content=reply_text,
                session_id=triage_session_id,
            )
            print(f"DB SAVED outbound reply for patient {patient.id}")

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------


async def _process_and_reply(
    sender_phone: str,
    message_type: str,
    content: str,
    raw_payload: dict,
    nlp_manager: NLPModelManager,
) -> None:
    """Background task: persist message, run NLP pipeline, send reply."""
    try:
        async with AsyncSessionLocal() as session:
            # 1. Persist inbound message (and get/create Patient record)
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

            # 2. Run NLP pipeline on text messages; stubs for other types
            triage_session_id = None
            if message_type == "text" and content:
                nlp_result = await run_nlp_pipeline(
                    session=session,
                    patient=patient,
                    nlp_manager=nlp_manager,
                    text=content,
                )
                triage_session_id = nlp_result["triage_session_id"]
                reply_text = await compose_triage_reply(nlp_result, nlp_manager)
            elif message_type == "audio":
                reply_text = (
                    "Voice note received! Automatic transcription is coming soon. "
                    "For now, please type your symptoms."
                )
            elif message_type == "interactive":
                reply_text = (
                    f"You selected: {content}. "
                    "Please tell me more about your symptoms."
                )
            else:
                reply_text = f"Received your {message_type} message."

            # 3. Send reply via Meta API
            await _send_whatsapp_reply(sender_phone, reply_text)

            # 4. Persist outbound message (linked to triage session if available)
            await save_outbound_message(
                session=session,
                patient_id=patient.id,
                content=reply_text,
                session_id=triage_session_id,
            )
            logger.info("Saved outbound reply to patient %s", patient.id)

    except Exception:
        logger.exception("Error processing message from %s", sender_phone)


# ---------------------------------------------------------------------------
# WhatsApp reply helper
# ---------------------------------------------------------------------------


async def send_reply(to: str, text: str) -> None:
    """Send a text message back to the user via Meta Cloud API."""
    phone_number_id = settings.meta_phone_number_id
    access_token = settings.meta_access_token

    print(f"SEND_REPLY: to={to}, phone_number_id={phone_number_id}")
    print(f"SEND_REPLY: token starts with: {access_token[:20] if access_token else 'NONE'}...")

    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {access_token}",
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
        print(f"SEND_REPLY RESPONSE: status={response.status_code}, body={response.text}")
