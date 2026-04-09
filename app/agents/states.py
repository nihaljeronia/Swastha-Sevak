"""Triage agent state definitions."""

from __future__ import annotations

from typing import Any, TypedDict


class TriageState(TypedDict):
    """State for the triage agent graph."""
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