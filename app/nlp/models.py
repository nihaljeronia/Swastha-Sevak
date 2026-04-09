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
        """Register all NLP model handles at application startup."""
        self.language_model = "langdetect-v1"
        self.translation_model = "google-translate-v1"
        
        # Load whisper small using the main thread directly natively (bypassing HF)
        try:
            import whisper
            import torch
            import os
            
            # Using the small architecture natively
            model_id = os.getenv("INDICWHISPER_MODEL", "small")
            logger.info(f"Loading native Whisper model: '{model_id}'...")
            
            self.stt_model = whisper.load_model(model_id)
                
        except ImportError:
            logger.warning("native 'whisper' library missing! ASR will not work.")
            self.stt_model = None
        except Exception as e:
            logger.error(f"Could not load native whisper: {e}")
            self.stt_model = None
            
        self._loaded = True
        logger.info(
            "NLP models ready — language=%s, translation=%s, ASR=%s",
            self.language_model,
            self.translation_model,
            "Native Whisper (Loaded)" if self.stt_model else "Missing",
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
        """Return an ISO 639-1 language code for *text*."""
        return detect_language(text)

    # ── Speech-to-text ────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_language: str | None = None,
    ) -> str:
        """Transcribe audio directly using local native Whisper model."""
        if not self.stt_model:
            logger.warning("ASR failed: Native Whisper model is not loaded.")
            return ""
            
        import tempfile
        import os
        import torch
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            loop = asyncio.get_running_loop()
            
            def _do_transcribe():
                # Whisper handles parsing internally, no librosa needed
                result = self.stt_model.transcribe(
                    tmp_path, 
                    language=source_language, 
                    fp16=torch.cuda.is_available()
                )
                return result.get("text", "").strip()
                
            text = await loop.run_in_executor(None, _do_transcribe)
            
            logger.info("Transcribed voice note to: %r", text[:60])
            return text
        except Exception:
            logger.exception("Native Whisper Transcribe failed")
            return ""
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

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

    # ── Text to Speech (TTS) ──────────────────────────────────────────────────

    async def text_to_speech(self, text: str, language: str) -> bytes:
        """Convert final text response back into regional audio bytes via gTTS."""
        import tempfile
        import os
        from gtts import gTTS
        
        # Determine language mappings, fallback to hindi
        lang_map = {"hi": "hi", "ta": "ta", "te": "te", "mr": "mr", "bn": "bn", "gu": "gu", "en": "en"}
        tts_lang = lang_map.get(language, "hi")
        
        try:
            loop = asyncio.get_running_loop()
            def _tts():
                tts = gTTS(text=text, lang=tts_lang)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_path = fp.name
                # Save requires a named path
                tts.save(temp_path)
                with open(temp_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                os.remove(temp_path)
                return audio_data
                
            return await loop.run_in_executor(None, _tts)
        except Exception:
            logger.exception("TTS Failed")
            return b""
