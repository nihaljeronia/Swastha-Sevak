import pytest


@pytest.mark.asyncio
async def test_whatsapp_webhook_accepts_request(async_client):
    payload = {
        "message_id": "msg-1",
        "from_number": "+911234567890",
        "message_text": "Hello",
        "language": "hi",
        "message_type": "text",
    }

    response = await async_client.post("/api/webhook/whatsapp", json=payload)

    assert response.status_code == 202
    assert response.json() == {"detail": "Accepted"}
