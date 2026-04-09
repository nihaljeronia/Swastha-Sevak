"""Keyword-based medical entity extraction.

v1 placeholder — scans English-translated text for known symptom
keywords, duration patterns, and severity indicators.

Will be replaced with a fine-tuned MuRIL NER model in Step 3.5.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Symptom keywords (English) ──────────────────────────────────────────────

SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "fever": ["fever", "temperature", "bukhar", "taap", "tap"],
    "headache": ["headache", "head pain", "head ache", "sir dard", "sir me dard"],
    "cough": ["cough", "coughing", "khansi", "khaansi"],
    "cold": ["cold", "runny nose", "sneezing", "sardi", "nazla"],
    "vomiting": ["vomiting", "vomit", "ulti", "throwing up", "nausea"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose motions", "loose motion", "dast", "loose stool"],
    "chest_pain": ["chest pain", "seene me dard", "chhati me dard", "chest heaviness"],
    "body_pain": ["body pain", "body ache", "badan dard", "sharir me dard"],
    "joint_pain": ["joint pain", "joint ache", "jodo me dard", "joints pain", "gathiya"],
    "rash": ["rash", "rashes", "skin rash", "puravya", "daane", "spots on skin"],
    "breathlessness": ["breathlessness", "breathless", "shortness of breath", "difficulty breathing", "saans me taklif", "saans phulna"],
    "fatigue": ["fatigue", "tiredness", "weakness", "kamzori", "thakan", "tired", "weak"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "stomach ache", "pet dard", "pet me dard", "tummy pain"],
    "sore_throat": ["sore throat", "throat pain", "gale me dard", "gala kharab"],
    "blood_in_stool": ["blood in stool", "bloody stool", "latrine me khoon"],
    "blood_in_urine": ["blood in urine", "bloody urine", "peshab me khoon"],
    "dizziness": ["dizziness", "dizzy", "giddiness", "chakkar", "sir ghoomna"],
    "swelling": ["swelling", "swollen", "sujan", "soojh"],
    "weight_loss": ["weight loss", "losing weight", "wajan kam"],
    "night_sweats": ["night sweats", "sweating at night", "raat me pasina"],
}

# ── Duration patterns ────────────────────────────────────────────────────────

_DURATION_PATTERN = re.compile(
    r"(?:(?:since|from|for|past|last)\s+)?"       # optional prefix
    r"(\d+)\s*"                                    # number
    r"(days?|din|weeks?|hafta|hafte|months?|mahine" # unit
    r"|hours?|ghante?)",
    re.IGNORECASE,
)

# Also catch "X din se", "X dino se" (Hindi pattern)
_DURATION_HINDI_PATTERN = re.compile(
    r"(\d+)\s*(din|dino|hafte|mahine|ghante)\s*(?:se|say)?",
    re.IGNORECASE,
)

# ── Severity indicators ─────────────────────────────────────────────────────

_SEVERE_WORDS = {
    "severe", "extreme", "unbearable", "very bad", "terrible",
    "intense", "excruciating", "worst", "critical", "bahut",
    "bahut zyada", "asahaniya", "cannot bear",
}
_MILD_WORDS = {
    "mild", "slight", "little", "minor", "thoda", "halka",
}


def extract_medical_entities(text: str) -> dict:
    """Scan *text* (expected in English) for medical entities.

    Returns:
        ``{"symptoms": [...], "duration": "...", "severity": "..."}``
    """
    if not text:
        return {"symptoms": [], "duration": None, "severity": "unknown"}

    text_lower = text.lower()

    # ── Symptoms ─────────────────────────────────────────────────────────
    found_symptoms: list[str] = []
    for symptom_key, aliases in SYMPTOM_KEYWORDS.items():
        for alias in aliases:
            if alias in text_lower:
                if symptom_key not in found_symptoms:
                    found_symptoms.append(symptom_key)
                break  # one match per symptom is enough

    # ── Duration ─────────────────────────────────────────────────────────
    duration: str | None = None
    match = _DURATION_PATTERN.search(text_lower) or _DURATION_HINDI_PATTERN.search(text_lower)
    if match:
        number = match.group(1)
        unit = match.group(2).lower()

        # Normalise Hindi units to English
        unit_map = {
            "din": "days", "dino": "days",
            "hafte": "weeks", "hafta": "weeks",
            "mahine": "months",
            "ghante": "hours", "ghanta": "hours",
        }
        unit = unit_map.get(unit, unit)
        if not unit.endswith("s") and int(number) > 1:
            unit += "s"
        duration = f"{number} {unit}"

    # ── Severity ─────────────────────────────────────────────────────────
    severity = "moderate"  # default
    for w in _SEVERE_WORDS:
        if w in text_lower:
            severity = "severe"
            break
    else:
        for w in _MILD_WORDS:
            if w in text_lower:
                severity = "mild"
                break

    result = {
        "symptoms": found_symptoms,
        "duration": duration,
        "severity": severity,
    }
    logger.info("NER result: text=%r → %s", text[:60], result)
    return result
