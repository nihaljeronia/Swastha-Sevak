"""Audio utility functions for WhatsApp voice note processing.

Handles OGG→WAV conversion (WhatsApp sends OGG Opus) and temp file cleanup.
Requires ffmpeg to be installed on the system.
"""

from __future__ import annotations

import logging
import os
import tempfile

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def convert_ogg_to_wav(input_path: str, output_path: str | None = None) -> str:
    """Convert a WhatsApp ``.ogg`` (Opus) audio file to ``.wav``.

    Args:
        input_path: Path to the source ``.ogg`` file.
        output_path: Optional explicit output path.  When *None*, a temp
            file is created in the same directory.

    Returns:
        Path to the resulting ``.wav`` file.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    logger.info("Converting OGG → WAV: %s → %s", input_path, output_path)

    audio = AudioSegment.from_file(input_path, format="ogg")
    audio.export(output_path, format="wav")

    logger.info("Conversion complete: %s (%.1f KB)", output_path, os.path.getsize(output_path) / 1024)
    return output_path


def cleanup_temp_file(file_path: str) -> None:
    """Delete a temporary audio file if it exists."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Cleaned up temp file: %s", file_path)
    except OSError:
        logger.exception("Failed to clean up temp file: %s", file_path)
