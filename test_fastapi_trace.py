from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

payload = {
    "entry": [{
        "changes": [{
            "value": {
                "messages": [{
                    "from": "1234567890",
                    "type": "text",
                    "text": {"body": "hello"}
                }]
            }
        }]
    }]
}

response = client.post("/webhook", json=payload)
print(response.status_code)
print(response.content)
