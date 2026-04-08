from fastapi import APIRouter, HTTPException, Query, Response, Request, status
import os
import json
import httpx

from app.core.config import settings
from app.webhook.schemas import WhatsAppWebhookEvent
from app.webhook.service import enqueue_patient_message

router = APIRouter()


@router.get("")
async def verify_webhook(request: Request):
    """Meta WhatsApp webhook verification endpoint."""
    params = dict(request.query_params)
    
    hub_mode = params.get("hub.mode") or params.get("hub_mode")
    hub_challenge = params.get("hub.challenge") or params.get("hub_challenge")
    hub_verify_token = params.get("hub.verify_token") or params.get("hub_verify_token")
    
    verify_token = os.getenv("META_VERIFY_TOKEN", "swastha-sevak-token")
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        return Response(content=hub_challenge, media_type="text/plain")
    
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("")
async def receive_webhook(request: Request):
    """Receive incoming WhatsApp messages from Meta webhook."""
    body = await request.json()
    
    # Log the full payload for debugging
    print("WEBHOOK RECEIVED:", json.dumps(body, indent=2))
    
    # Extract message data from Meta's nested structure
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
            # This is a status update (delivered, read), not a message
            return {"status": "ok"}
        
        message = messages[0]
        sender_phone = message.get("from")
        message_type = message.get("type")
        
        if message_type == "text":
            message_body = message.get("text", {}).get("body", "")
            print(f"TEXT from {sender_phone}: {message_body}")
            
            # Echo reply for testing
            await send_reply(sender_phone, f"You said: {message_body}")
        
        elif message_type == "audio":
            audio_id = message.get("audio", {}).get("id", "")
            print(f"AUDIO from {sender_phone}: media_id={audio_id}")
            await send_reply(sender_phone, "Voice note received! (processing coming soon)")
        
        elif message_type == "interactive":
            interactive = message.get("interactive", {})
            reply_id = interactive.get("button_reply", {}).get("id") or interactive.get("list_reply", {}).get("id")
            print(f"INTERACTIVE from {sender_phone}: {reply_id}")
            await send_reply(sender_phone, f"You selected: {reply_id}")
        
        else:
            print(f"UNKNOWN type from {sender_phone}: {message_type}")
    
    except Exception as e:
        print(f"Error processing webhook: {e}")
    
    return {"status": "ok"}


async def send_reply(to: str, text: str):
    """Send a text message back to the user via Meta Cloud API."""
    phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
    access_token = os.getenv("META_ACCESS_TOKEN")
    
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
        print(f"SEND REPLY status={response.status_code}: {response.text}")
