import whisper
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, Generator
from collections import deque
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperSTT:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        enable_streaming: bool = True
    ):
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "float32"
        self.language = language
        self.enable_streaming = enable_streaming
        
        logger.info(f"Loading Whisper model: {model_size} on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)
        
        self.audio_buffer = deque(maxlen=int(16000 * 30))
        self.transcription_buffer = []
        self.is_processing = False
        self.processing_thread = None
        
        self.min_speech_duration = 0.5
        self.max_silence_duration = 1.0
        self.speech_buffer = []
        self.silence_counter = 0
        
    def process_audio_chunk(self, audio_chunk: np.ndarray, is_speech: bool) -> Optional[str]:
        self.audio_buffer.extend(audio_chunk)
        
        if is_speech:
            self.speech_buffer.extend(audio_chunk)
            self.silence_counter = 0
        else:
            self.silence_counter += len(audio_chunk) / 16000
            
        if self.silence_counter > self.max_silence_duration and len(self.speech_buffer) > 0:
            if len(self.speech_buffer) / 16000 > self.min_speech_duration:
                audio_array = np.array(self.speech_buffer, dtype=np.float32)
                transcription = self._transcribe(audio_array)
                self.speech_buffer = []
                return transcription
            else:
                self.speech_buffer = []
                
        return None
        
    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    result = self.model.transcribe(
                        audio,
                        language=self.language,
                        fp16=(self.compute_type == "float16"),
                        without_timestamps=True,
                        task="transcribe"
                    )
            else:
                result = self.model.transcribe(
                    audio,
                    language=self.language,
                    without_timestamps=True,
                    task="transcribe"
                )
                
            return result["text"].strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
            
    def stream_transcribe(self, audio_generator: Generator) -> Generator[str, None, None]:
        for audio_chunk, is_speech in audio_generator:
            transcription = self.process_audio_chunk(audio_chunk, is_speech)
            if transcription:
                yield transcription
                
    def transcribe_file(self, audio_file: str) -> str:
        try:
            result = self.model.transcribe(
                audio_file,
                language=self.language,
                fp16=(self.compute_type == "float16") if self.device == "cuda" else False
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""
            
    def clear_buffers(self):
        self.audio_buffer.clear()
        self.speech_buffer = []
        self.transcription_buffer = []
        self.silence_counter = 0

class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en"
    ):
        try:
            from faster_whisper import WhisperModel
            
            self.device = device if torch.cuda.is_available() else "cpu"
            self.compute_type = compute_type if self.device == "cuda" else "int8"
            
            logger.info(f"Loading Faster-Whisper model: {model_size}")
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.language = language
            self.audio_buffer = []
            
        except ImportError:
            logger.error("faster-whisper not installed, falling back to regular Whisper")
            raise
            
    def transcribe(self, audio: np.ndarray) -> str:
        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=5,
                language=self.language,
                without_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""