import asyncio
import logging

from app.db.session import AsyncSessionLocal
from app.webhook.service import (
    process_incoming_message,
    run_triage_agent,
    save_outbound_message,
)

# Silence noisy third-party logs so the CLI is clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app").setLevel(logging.WARNING)

async def main():
    print("="*65)
    print("🩺 SWASTHA SEVAK - LIVE CLINICAL SIMULATION".center(65))
    print("="*65)
    print("Type 'exit' or 'quit' to end the simulation.")
    print("You can chat in Hindi, English, Tamil, etc.\n")
    
    # Use a dummy phone number so we don't mess up real data
    phone = "+919999999999"
    
    async with AsyncSessionLocal() as session:
        while True:
            try:
                text = input("\n👤 Patient: ")
            except KeyboardInterrupt:
                break
                
            if text.lower() in ["exit", "quit", ""]:
                break
                
            # 1. Save inbound message and get patient context
            patient, msg = await process_incoming_message(
                session=session,
                phone=phone,
                message_type="text",
                content=text,
            )
            
            # 2. Run LangGraph Multi-Agent Triage
            print("🤖 [Evaluating Clinical Logic...]")
            result = await run_triage_agent(
                session=session,
                patient=patient,
                text=text,
            )
            
            # 3. Save outbound Doctor reply to Memory
            reply_text = result["reply_text"]
            await save_outbound_message(
                session=session,
                patient_id=patient.id,
                content=reply_text,
                session_id=result["triage_session_id"],
            )
            
            # 4. Display translated output
            print(f"\n👩‍⚕️ Doctor: {reply_text}\n")
            print("-" * 65)

if __name__ == "__main__":
    asyncio.run(main())
