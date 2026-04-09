"""Agent state definition for the triage workflow.

Defines the TypedDict schema that represents the complete state
of a patient's triage session throughout the LangGraph agent execution.
"""

from __future__ import annotations

from typing import TypedDict


class TriageState(TypedDict):
    """Complete state for a medical triage session.

    This state object is passed through all nodes in the LangGraph
    triage workflow, accumulating information about the patient,
    their symptoms, and the triage decision.

    Attributes:
        patient_phone: WhatsApp number with country code (e.g., "91XXXXXXXXXX").
        language: Detected patient language code (e.g., "hi", "ta", "mr", "en").
        messages: Full conversation history as list of dicts.
            Each dict: {"role": "user"|"assistant", "content": str, "language": str}
        symptoms: Extracted symptoms in English from conversation.
            Built progressively during symptom_collector node.
        medical_entities: NER output dict with detected medical entities.
            Structure: {"body_parts": [...], "duration": [...], "severity": [...]}
        question_count: Number of follow-up questions asked so far (0-5 max).
        urgency: Classifier output urgency level.
            One of: "self_care", "routine", "urgent", "emergency".
        conditions: Top-3 predicted disease conditions from classifier.
            List of dicts: [{"name": str, "confidence": float}, ...]
        advice: RAG-generated medical advice in English (before translation).
        nearest_phc: Nearest Primary Health Center info.
            Dict: {"name": str, "distance_km": float, "phone": str, "address": str}
        follow_up_scheduled: Boolean flag if follow-up task was scheduled.
    """

    patient_phone: str
    language: str
    messages: list[dict]
    symptoms: list[str]
    medical_entities: dict
    question_count: int
    urgency: str
    conditions: list[dict]
    advice: str
    nearest_phc: dict
    follow_up_scheduled: bool
