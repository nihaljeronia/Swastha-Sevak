import asyncio
from app.agents.graph import TriageAgent

async def main():
    agent = TriageAgent()
    await agent.load_model()
    await agent.initialize()
    
    print("Agent initialized. Running with sample text...")
    
    sample_patient_data = {
        "phone": "+919999999999",
        "message": "mujhe kal se bahut tej bukhar aur sir dard hai"
    }
    
    result = await agent.run(sample_patient_data)
    print("\nResult run 1:")
    print("Urgency:", result["triage_level"])
    print("Next step:", result["next_step"])
    print("Reply:", result["reply"])
    
    # second turn
    sample_patient_data2 = {
        "phone": "+919999999999",
        "message": "Nahi, aur koi taklif nahi hai"
    }
    print("\nResult run 2:")
    result2 = await agent.run(sample_patient_data2)
    print("Urgency:", result2["triage_level"])
    print("Next step:", result2["next_step"])
    print("Reply:", result2["reply"])
    print("Symptoms Extracted:", result2["state"]["symptoms"])

if __name__ == "__main__":
    asyncio.run(main())
