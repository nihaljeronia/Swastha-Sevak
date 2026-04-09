"""Translation between Indian languages and English.

v1 placeholder using ``deep-translator`` (Google Translate free API).
Will be replaced with AI4Bharat IndicTrans2 for offline capability
and better accuracy on Indian languages.
"""

from __future__ import annotations

import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# Mapping from ISO 639-1 codes to language names used by Google Translate
LANG_CODE_MAP: dict[str, str] = {
    "hi": "hindi",
    "ta": "tamil",
    "mr": "marathi",
    "te": "telugu",
    "bn": "bengali",
    "gu": "gujarati",
    "kn": "kannada",
    "ml": "malayalam",
    "pa": "punjabi",
    "or": "odia",
    "as": "assamese",
    "ur": "urdu",
    "en": "english",
}


def _resolve_lang(code: str) -> str:
    """Convert our ISO 639-1 code to the Google Translate language name."""
    return LANG_CODE_MAP.get(code, code)


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate *text* from *source_lang* to *target_lang*.

    Both language args are ISO 639-1 codes (e.g. ``"hi"``, ``"en"``).
    Returns the original text unchanged if source == target or on error.
    """
    if not text or not text.strip():
        return text

    if source_lang == target_lang:
        return text

    try:
        src = _resolve_lang(source_lang)
        tgt = _resolve_lang(target_lang)
        result = GoogleTranslator(source=src, target=tgt).translate(text)
        logger.info(
            "Translated (%s→%s): %r → %r",
            source_lang,
            target_lang,
            text[:60],
            (result or "")[:60],
        )
        return result or text
    except Exception:
        logger.exception(
            "Translation failed (%s→%s) for: %r",
            source_lang,
            target_lang,
            text[:60],
        )
        return text


def to_english(text: str, source_lang: str) -> str:
    """Translate *text* from *source_lang* to English."""
    return translate(text, source_lang, "en")


def from_english(text: str, target_lang: str) -> str:
    """Translate English *text* to *target_lang*."""
    return translate(text, "en", target_lang)
