"""XGBoost disease triage classifier."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.preprocessor import symptoms_to_vector, map_to_medical_terms, get_symptom_list

logger = logging.getLogger(__name__)

USE_MOCK_CLASSIFIER = os.getenv("MOCK_CLASSIFIER", "false").lower() in ("true", "1", "yes")

_urgency_model = None
_disease_model = None
_disease_labels = None

MODEL_DIR = Path(__file__).parent / "models"


def load_models() -> None:
    """Load trained models at startup. Call from FastAPI lifespan."""
    global _urgency_model, _disease_model, _disease_labels

    urgency_path = MODEL_DIR / "urgency_classifier.joblib"
    disease_path = MODEL_DIR / "disease_classifier.joblib"
    labels_path = MODEL_DIR / "disease_labels.json"

    if urgency_path.exists():
        _urgency_model = joblib.load(urgency_path)
        logger.info("Loaded urgency model from %s", urgency_path)
    else:
        logger.warning("No urgency model at %s — using mock mode", urgency_path)

    if disease_path.exists():
        _disease_model = joblib.load(disease_path)
        logger.info("Loaded disease model from %s", disease_path)
    else:
        logger.warning("No disease model at %s — using mock mode", disease_path)

    if labels_path.exists():
        with open(labels_path) as f:
            _disease_labels = json.load(f)
        logger.info("Loaded %d disease labels", len(_disease_labels))


# ============================================================================
# DiseaseClassifier class — used by app/main.py lifespan
# ============================================================================


class DiseaseClassifier:
    """Wrapper class for FastAPI lifespan integration."""

    def __init__(self) -> None:
        self.model: Any | None = None

    async def load_model(self) -> None:
        """Load models (called at startup)."""
        load_models()
        self.model = "xgboost-classifier"

    async def shutdown(self) -> None:
        global _urgency_model, _disease_model, _disease_labels
        _urgency_model = None
        _disease_model = None
        _disease_labels = None
        self.model = None

    async def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Async predict interface."""
        symptoms = features.get("symptoms", [])
        result = _predict_triage_internal(symptoms)
        return result


# ============================================================================
# Module-level function for agent nodes (with mock support)
# ============================================================================


def predict_triage(
    symptoms: list[str],
    entities: dict | None = None,
) -> tuple[str, list[dict]]:
    """Classify urgency level and predict top-3 disease conditions.

    Args:
        symptoms: List of extracted symptoms.
        entities: Medical entities dict (optional, for future enhancement).

    Returns:
        Tuple of (urgency_level, top_3_conditions).
        - urgency_level: one of "self_care", "routine", "urgent", "emergency"
        - top_3_conditions: list of dicts [{"name": str, "confidence": float}, ...]
    """
    if USE_MOCK_CLASSIFIER:
        logger.info("[predict_triage] MOCK: returning hardcoded triage")
        mock_urgency = "routine"
        mock_conditions = [
            {"name": "Common Cold", "confidence": 0.65},
            {"name": "Upper Respiratory Infection", "confidence": 0.25},
            {"name": "Influenza", "confidence": 0.10},
        ]
        return mock_urgency, mock_conditions

    result = _predict_triage_internal(symptoms)
    return result["urgency"], result["conditions"]


def predict_from_text(text: str, entities: list[str] | None = None) -> dict:
    """Convenience: map colloquial text -> medical terms -> predict."""
    medical_symptoms = map_to_medical_terms(text, entities)
    result = _predict_triage_internal(medical_symptoms)
    result["mapped_symptoms"] = medical_symptoms
    return result


# ============================================================================
# Internal prediction logic
# ============================================================================


def _predict_triage_internal(symptoms: list[str]) -> dict:
    """Core prediction: symptoms -> {urgency, conditions, risk_score}."""
    if not symptoms:
        return {
            "urgency": "routine",
            "conditions": [{"name": "Insufficient symptoms", "confidence": 0.0}],
            "risk_score": 0.1,
        }

    vector = symptoms_to_vector(symptoms).reshape(1, -1)

    # Urgency prediction
    if _urgency_model is not None:
        urgency_idx = _urgency_model.predict(vector)[0]
        urgency_proba = _urgency_model.predict_proba(vector)[0]
        urgency_labels = ["self_care", "routine", "urgent", "emergency"]

        if isinstance(urgency_idx, (int, np.integer)):
            urgency = urgency_labels[min(int(urgency_idx), len(urgency_labels) - 1)]
        else:
            urgency = str(urgency_idx)

        risk_score = float(max(urgency_proba))
    else:
        urgency = _mock_urgency(symptoms)
        risk_score = 0.7

    # Disease prediction
    conditions = []
    if _disease_model is not None and _disease_labels is not None:
        disease_proba = _disease_model.predict_proba(vector)[0]
        top_indices = np.argsort(disease_proba)[::-1][:3]
        for idx in top_indices:
            if idx < len(_disease_labels):
                conditions.append({
                    "name": _disease_labels[idx],
                    "confidence": round(float(disease_proba[idx]), 3),
                })
    else:
        conditions = _mock_conditions(symptoms)

    return {
        "urgency": urgency,
        "conditions": conditions,
        "risk_score": round(risk_score, 3),
    }


def _mock_urgency(symptoms: list[str]) -> str:
    emergency_symptoms = {"chest_pain", "seizure", "breathlessness", "confusion", "blood_in_stool"}
    urgent_symptoms = {"fever", "joint_pain", "rash", "yellow_skin", "blood_in_urine"}

    if any(s in emergency_symptoms for s in symptoms):
        return "emergency"
    if len(set(symptoms) & urgent_symptoms) >= 2:
        return "urgent"
    if any(s in urgent_symptoms for s in symptoms):
        return "routine"
    return "self_care"


def _mock_conditions(symptoms: list[str]) -> list[dict]:
    s = set(symptoms)
    if {"fever", "joint_pain", "rash"} & s == {"fever", "joint_pain", "rash"}:
        return [{"name": "Dengue", "confidence": 0.82}, {"name": "Chikungunya", "confidence": 0.12}]
    if {"fever", "chills"} & s:
        return [{"name": "Malaria", "confidence": 0.65}, {"name": "Viral Fever", "confidence": 0.25}]
    if {"cough", "fever"} & s == {"cough", "fever"}:
        return [{"name": "Pneumonia", "confidence": 0.45}, {"name": "TB", "confidence": 0.30}]
    if "chest_pain" in s:
        return [{"name": "Cardiac Event", "confidence": 0.60}, {"name": "GERD", "confidence": 0.25}]
    return [{"name": "Common Viral Infection", "confidence": 0.50}]
