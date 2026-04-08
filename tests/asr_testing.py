''' IndicWhisper ASR Model Testing
Testing the model on audio files from the audio_path folder to check if the model is working correctly 
and transcribes audio accurately'''

import whisper
import librosa
import numpy as np
import time
import json
from pathlib import Path
import os

class IndicWhisperTester:
    def __init__(self):
        print("Loading IndicWhisper model...")
        # Models: tiny, base, small, medium, large
        self.model = whisper.load_model("small")  
        self.results = []
        self.audio_dir = Path(__file__).parent / "audio_path"
    
    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to .wav, .mp3, .m4a, etc.
            language: Language code (e.g., 'hi' for Hindi, 'ta' for Tamil)
                     If None, auto-detect
        """
        start = time.time()
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
            fp16=True  
        )
        
        elapsed = time.time() - start
        
        confidence = None
        if result.get("segments"):
            confidences = [seg.get("confidence", 0) for seg in result["segments"]]
            confidence = sum(confidences) / len(confidences) if confidences else None
        
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "confidence": confidence,
            "duration_seconds": elapsed,
            "audio_file": str(audio_path),
            "segments": len(result.get("segments", []))
        }
    
    def batch_test(self, language=None):
        """Test multiple audio files from audio_path folder"""
        audio_files = sorted(self.audio_dir.glob("*.wav"))
        
        if not audio_files:
            print(f"No .wav files found in {self.audio_dir}")
            return []
        
        print(f"\nTesting {len(audio_files)} audio files...\n")
        
        for idx, audio_file in enumerate(audio_files, 1):
            try:
                print(f"\n[{idx}/{len(audio_files)}] Testing: {audio_file.name}")
                result = self.transcribe(str(audio_file), language)
                self.results.append(result)
                
                print(f"  Status: Success")
                print(f"  Language: {result['language']}")
                print(f"  Duration: {result['duration_seconds']:.2f}s")
                print(f"  Segments: {result['segments']}")
                print(f"  Transcription: {result['text'][:100]}..." if len(result['text']) > 100 else f"  Transcription: {result['text']}")
            
            except Exception as e:
                print(f"  Error: {str(e)}")
                self.results.append({
                    "audio_file": str(audio_file),
                    "error": str(e),
                    "status": "failed"
                })
        
        return self.results
    
    def test_language_detection(self):
        audio_files = sorted(self.audio_dir.glob("*.wav"))
        if not audio_files:
            print("No audio files found")
            return
        print("\n=== LANGUAGE DETECTION TEST ===")
        audio_file = audio_files[0]
        print(f"Testing auto-detection on: {audio_file.name}")
        
        try:
            result = self.transcribe(str(audio_file))
            print(f"Detected language: {result['language']}")
            print(f"Transcription: {result['text']}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    print("INDICWHISPER ASR MODEL TESTING")
    
    tester = IndicWhisperTester()
    
    if not tester.audio_dir.exists():
        print(f"Error: Audio directory not found at {tester.audio_dir}")
        exit(1)
    
    audio_files = list(tester.audio_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files in {tester.audio_dir}")
    for f in audio_files:
        print(f"  - {f.name}")

    print("TEST 1: LANGUAGE AUTO-DETECTION")
    tester.test_language_detection()
    print("TEST 2: BATCH PROCESSING ALL AUDIO FILES")
    results = tester.batch_test()
    total_files = len(audio_files)
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])
    total_time = sum([r.get("duration_seconds", 0) for r in results if "error" not in r])
    print(f"Total processing time: {total_time:.2f}s")
    
    output_file = Path(__file__).parent / "asr_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n GPU-Usage: {results[0].get('gpu_usage', 'N/A') if results else 'N/A'}%")
        print(f"Results saved to {output_file}")



#now testing for the indic wav2vec2 model
