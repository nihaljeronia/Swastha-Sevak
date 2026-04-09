"""NLP model manager — wraps speech, language detection, NER, and translation.

v1 wires lightweight, no-download libraries:
  - Language detection : ``langdetect`` (deterministic, seeded)
  - Translation        : Google Translate via ``deep-translator``
  - NER                : keyword-based (app/nlp/ner.py)
  - ASR                : stub — IndicWhisper will replace when GPU is available

v2 (future) will swap each component for AI4Bharat models:
  - Language detection → fastText lid.176 / MuRIL
  - Translation        → IndicTrans2
  - NER                → fine-tuned MuRIL BERT
  - ASR                → IndicWhisper (22 Indian languages)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.nlp.language_id import detect_language
from app.nlp.ner import extract_medical_entities
from app.nlp.translator import from_english, to_english

logger = logging.getLogger(__name__)


class NLPModelManager:
    """Manages all NLP models used in the triage pipeline.

    Loaded once at startup via FastAPI lifespan; shared across all requests
    through ``app.state.nlp_manager``.
    """

    def __init__(self) -> None:
        self.stt_model: Any | None = None
        self.language_model: str | None = None
        self.translation_model: str | None = None
        self._loaded: bool = False

    async def load_models(self) -> None:
        """Register all NLP model handles at application startup.

        v1 uses libraries that require no model downloads.
        Heavy models (IndicWhisper, MuRIL, IndicTrans2) are stubbed
        until GPU resources and model weights are available.
        """
        self.language_model = "langdetect-v1"
        self.translation_model = "google-translate-v1"
        # TODO: load IndicWhisper when GPU / model weights are available:
        #   from app.nlp.asr import load_indicwhisper
        #   self.stt_model = await load_indicwhisper()
        self._loaded = True
        logger.info(
            "NLP models ready — language=%s, translation=%s, ASR=stub",
            self.language_model,
            self.translation_model,
        )

    async def shutdown(self) -> None:
        """Release all model resources on server shutdown."""
        self.stt_model = None
        self.language_model = None
        self.translation_model = None
        self._loaded = False
        logger.info("NLP models unloaded")

    # ── Language detection ────────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """Return an ISO 639-1 language code for *text*.

        Delegates to :func:`app.nlp.language_id.detect_language`.
        Falls back to ``"hi"`` (Hindi) on short text or low confidence.
        """
        return detect_language(text)

    # ── Speech-to-text ────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_language: str | None = None,
    ) -> str:
        """Transcribe audio bytes to text.

        TODO: Implement with AI4Bharat IndicWhisper (22 Indian languages).
        Returns an empty string until the model is loaded.
        """
        logger.warning(
            "ASR not yet implemented (IndicWhisper stub) — "
            "returning empty transcript for %d-byte audio.",
            len(audio_bytes),
        )
        return ""

    # ── Translation ───────────────────────────────────────────────────────────

    async def translate_to_english(self, text: str, source_language: str) -> str:
        """Translate *text* from *source_language* to English (non-blocking).

        Runs the synchronous ``deep-translator`` call in a thread-pool executor
        so it doesn't block the async event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, to_english, text, source_language)

    async def translate_from_english(self, text: str, target_language: str) -> str:
        """Translate English *text* to *target_language* (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, from_english, text, target_language)

    # ── NER ───────────────────────────────────────────────────────────────────

    def extract_entities(self, english_text: str) -> dict:
        """Extract medical entities from English *english_text*.

        Returns ``{"symptoms": [...], "duration": str|None, "severity": str}``.
        Delegates to :func:`app.nlp.ner.extract_medical_entities`.
        """
        return extract_medical_entities(english_text)


# ============================================================================
# Module-level helper functions for agent nodes (with mock support)
# ============================================================================

import os

USE_MOCK_NLP = os.getenv("MOCK_NLP", "false").lower() in ("true", "1", "yes")


def process_message(text: str) -> tuple[list[str], dict]:
    """Extract symptoms and medical entities from text.

    Args:
        text: Input message text (preferably in English).

    Returns:
        Tuple of (symptoms_list, entities_dict).
        Entities dict structure: {"body_parts": [...], "duration": [...], "severity": [...]}.
    """
    if USE_MOCK_NLP:
        logger.info("[process_message] MOCK: extracting hardcoded symptoms")
        mock_symptoms = ["fever", "cough"]
        mock_entities = {
            "body_parts": ["chest"],
            "duration": ["3 days"],
            "severity": ["moderate"],
        }
        return mock_symptoms, mock_entities

    # Use the actual extraction logic
    logger.info("[process_message] Extracting symptoms via NER")
    entities = extract_medical_entities(text)
    symptoms = entities.get("symptoms", [])
    return symptoms, entities


def translate_to_en(text: str, source_lang: str) -> str:
    """Translate text to English from source language.

    Args:
        text: Input text.
        source_lang: Source language code ('hi', 'ta', 'mr', etc.).

    Returns:
        English translation of text.
    """
    if USE_MOCK_NLP:
        logger.info(f"[translate_to_en] MOCK: pretending to translate from {source_lang}")
        return text

    # Real implementation via translator
    logger.info(f"[translate_to_en] Translating from {source_lang} to English")
    return to_english(text, source_lang)


def translate_to_patient_lang(text: str, target_lang: str) -> str:
    """Translate text to patient's language from English.

    Args:
        text: Input text in English.
        target_lang: Target language code ('hi', 'ta', 'mr', etc.).

    Returns:
        Translated text in target language.
    """
    if USE_MOCK_NLP:
        logger.info(f"[translate_to_patient_lang] MOCK: pretending to translate to {target_lang}")
        return f"{text}\n[{target_lang}]"

    # Real implementation via translator
    logger.info(f"[translate_to_patient_lang] Translating to {target_lang}")
    return from_english(text, target_lang)
