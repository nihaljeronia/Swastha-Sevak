import asyncio
import httpx

async def post():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post('http://127.0.0.1:8000/webhook', json={'entry': [{'changes': [{'value': {'messages': [{'from': '123', 'type': 'text', 'text': {'body': 'hello'}}]}}]}]})
            print(r.status_code, r.text)
        except Exception as e:
            print("Failed to connect:", e)

if __name__ == "__main__":
    asyncio.run(post())
