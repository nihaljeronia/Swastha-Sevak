"""Symptom text to feature vector preprocessing pipeline."""
from __future__ import annotations
import json
import os
import numpy as np
from pathlib import Path

# Colloquial-to-medical symptom mapping
SYMPTOM_SYNONYMS = {
    "fever": ["fever", "bukhar", "temperature", "body hot", "tap", "badan garam", "high temperature", "feverish"],
    "headache": ["headache", "sir dard", "head pain", "sir me dard", "head ache", "migraine"],
    "cough": ["cough", "khansi", "khaansi", "dry cough", "wet cough"],
    "cold": ["cold", "sardi", "zukam", "runny nose", "naak behna"],
    "vomiting": ["vomiting", "ulti", "throwing up", "nausea", "ji machlana", "vomit"],
    "diarrhea": ["diarrhea", "dast", "loose motion", "loose stool", "pet kharab"],
    "chest_pain": ["chest pain", "seene me dard", "chest tightness", "heart pain", "chhati me dard"],
    "body_pain": ["body pain", "badan dard", "body ache", "muscles pain", "shareer dard"],
    "joint_pain": ["joint pain", "jodon me dard", "joint ache", "gathiya", "jodon ka dard"],
    "rash": ["rash", "daane", "chaale", "skin rash", "red spots", "laal daane"],
    "breathlessness": ["breathlessness", "saans lene me dikkat", "shortness of breath", "difficulty breathing", "saans phoolna"],
    "fatigue": ["fatigue", "thakan", "weakness", "kamzori", "tired", "exhaustion"],
    "abdominal_pain": ["abdominal pain", "pet dard", "stomach pain", "pet me dard", "tummy ache"],
    "sore_throat": ["sore throat", "gale me dard", "throat pain", "gala kharab"],
    "dizziness": ["dizziness", "chakkar", "chakkar aana", "lightheaded", "sir ghoomna"],
    "swelling": ["swelling", "sujan", "soojh", "puffiness"],
    "weight_loss": ["weight loss", "vajan kam hona", "losing weight", "patla hona"],
    "night_sweats": ["night sweats", "raat ko pasina", "sweating at night"],
    "blood_in_stool": ["blood in stool", "potty me khoon", "bloody stool"],
    "eye_pain": ["eye pain", "aankh me dard", "eye ache"],
    "chills": ["chills", "thand lagna", "shivering", "kaampna"],
    "nausea": ["nausea", "ji machlana", "feel like vomiting"],
    "back_pain": ["back pain", "kamar dard", "peeth dard"],
    "muscle_pain": ["muscle pain", "maansapeshiyon me dard", "muscle ache"],
    "loss_of_appetite": ["loss of appetite", "bhookh na lagna", "not hungry"],
    "constipation": ["constipation", "kabz", "qabz"],
    "blood_in_urine": ["blood in urine", "peshab me khoon"],
    "yellow_skin": ["yellow skin", "jaundice", "peeli chamdi", "piliya"],
    "confusion": ["confusion", "confused", "samajh nahi aa raha"],
    "seizure": ["seizure", "fits", "mirgi", "convulsion"],
}

_symptom_list: list[str] | None = None
_synonym_map: dict[str, str] | None = None


def _build_synonym_map() -> tuple[list[str], dict[str, str]]:
    """Build reverse mapping: synonym -> standard symptom name."""
    symptom_list = sorted(SYMPTOM_SYNONYMS.keys())
    synonym_map = {}
    for standard, synonyms in SYMPTOM_SYNONYMS.items():
        for syn in synonyms:
            synonym_map[syn.lower()] = standard
    return symptom_list, synonym_map


def get_symptom_list() -> list[str]:
    global _symptom_list, _synonym_map
    if _symptom_list is None:
        _symptom_list, _synonym_map = _build_synonym_map()
    return _symptom_list


def get_synonym_map() -> dict[str, str]:
    global _symptom_list, _synonym_map
    if _synonym_map is None:
        _symptom_list, _synonym_map = _build_synonym_map()
    return _synonym_map


def map_to_medical_terms(text: str, extracted_entities: list[str] | None = None) -> list[str]:
    """Map colloquial/multilingual text to standard symptom names."""
    synonym_map = get_synonym_map()
    matched = set()
    text_lower = text.lower()

    # Check synonym phrases in text
    for synonym, standard in synonym_map.items():
        if synonym in text_lower:
            matched.add(standard)

    # Also check extracted entities
    if extracted_entities:
        for entity in extracted_entities:
            entity_lower = entity.lower().strip()
            if entity_lower in synonym_map:
                matched.add(synonym_map[entity_lower])
            # Fuzzy: check if entity is substring of any synonym
            for synonym, standard in synonym_map.items():
                if entity_lower in synonym or synonym in entity_lower:
                    matched.add(standard)
                    break

    return list(matched)


def symptoms_to_vector(symptoms: list[str]) -> np.ndarray:
    """Convert symptom list to binary feature vector."""
    symptom_list = get_symptom_list()
    vector = np.zeros(len(symptom_list), dtype=np.float32)
    for s in symptoms:
        s_lower = s.lower().strip().replace(" ", "_")
        if s_lower in symptom_list:
            idx = symptom_list.index(s_lower)
            vector[idx] = 1.0
    return vector
