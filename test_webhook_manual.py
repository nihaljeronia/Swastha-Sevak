import asyncio
import traceback
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.db import crud
from app.webhook.service import run_triage_agent

async def main():
    async with AsyncSessionLocal() as session:
        # Create a mock patient
        patient, _ = await crud.get_or_create_patient(session, "1234567890")
        
        try:
            print("Message 1:")
            result = await run_triage_agent(session, patient, "mujhe bukhar hai")
            # print("Reply:", result["reply_text"])
            print("SUCCESS M1")
            
            print("Message 2:")
            result2 = await run_triage_agent(session, patient, "sir dard bhi hai")
            # print("Reply:", result2["reply_text"])
            print("SUCCESS M2")
            
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
