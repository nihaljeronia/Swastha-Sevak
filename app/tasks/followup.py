from fastapi import BackgroundTasks


async def enqueue_patient_event_task(event_payload: dict, background_tasks: BackgroundTasks) -> dict:
    """BackgroundTasks placeholder for patient follow-up and triage processing."""
    return {"status": "queued", "payload": event_payload}
