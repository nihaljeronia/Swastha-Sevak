"""Language detection for incoming patient messages.

Uses the ``langdetect`` library (Google's language-detection port) to
identify the ISO 639-1 language code from a text string.

v1 placeholder — will be upgraded to MuRIL / fastText lid model later.
"""

from __future__ import annotations

import logging

from langdetect import DetectorFactory, detect_langs

# Make langdetect deterministic (otherwise results vary across runs)
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# ISO 639-1 codes for languages we support
SUPPORTED_LANGUAGES: set[str] = {
    "hi",  # Hindi
    "ta",  # Tamil
    "mr",  # Marathi
    "te",  # Telugu
    "bn",  # Bengali
    "gu",  # Gujarati
    "kn",  # Kannada
    "ml",  # Malayalam
    "pa",  # Punjabi
    "or",  # Odia
    "as",  # Assamese
    "ur",  # Urdu
    "en",  # English
}

# Default fallback — Hindi is most spoken in rural India
DEFAULT_LANGUAGE = "hi"

# Minimum confidence to trust the detection result
MIN_CONFIDENCE = 0.4


def detect_language(text: str) -> str:
    """Detect the language of *text* and return an ISO 639-1 code.

    Falls back to ``"hi"`` (Hindi) when:
    - The input is too short for reliable detection.
    - Confidence is below ``MIN_CONFIDENCE``.
    - The detected language is not in our supported set.

    Returns:
        Two-letter ISO 639-1 language code, e.g. ``"hi"``, ``"mr"``, ``"en"``.
    """
    if not text or len(text.strip()) < 3:
        logger.info("Text too short for detection, defaulting to %s", DEFAULT_LANGUAGE)
        return DEFAULT_LANGUAGE

    try:
        results = detect_langs(text)
        if not results:
            return DEFAULT_LANGUAGE

        top = results[0]
        lang_code = str(top.lang)
        confidence = top.prob

        logger.info(
            "Language detection: text=%r → lang=%s conf=%.2f (all=%s)",
            text[:60],
            lang_code,
            confidence,
            results[:3],
        )

        # Check confidence and whether we support this language
        if confidence < MIN_CONFIDENCE:
            logger.info(
                "Low confidence (%.2f), defaulting to %s",
                confidence,
                DEFAULT_LANGUAGE,
            )
            return DEFAULT_LANGUAGE

        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code

        # Unsupported language — fall back to Hindi
        logger.info(
            "Unsupported language %s, defaulting to %s",
            lang_code,
            DEFAULT_LANGUAGE,
        )
        return DEFAULT_LANGUAGE

    except Exception:
        logger.exception("Language detection failed, defaulting to %s", DEFAULT_LANGUAGE)
        return DEFAULT_LANGUAGE
