from typing import TypedDict, Annotated, Any
import operator

def update_list(current: list, new: list) -> list:
    return current + new

def update_dict(current: dict, new: dict) -> dict:
    updated = current.copy()
    updated.update(new)
    return updated

class TriageState(TypedDict):
    patient_phone: str
    language: str                    # detected language code
    messages: Annotated[list[dict], update_list] # conversation history
    symptoms: Annotated[list[str], update_list]  # extracted symptoms in English
    medical_entities: Annotated[dict, update_dict] # NER output
    question_count: int             # how many Qs asked so far
    urgency: str                    # classifier output
    conditions: list[dict]          # top-3 with confidence scores
    advice: str                     # RAG-generated advice
    nearest_phc: dict               # name, distance, phone
    follow_up_scheduled: bool
    escalated: bool
    next_agent: str
    session_completed: bool
