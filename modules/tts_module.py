import torch
import numpy as np
import logging
from typing import Optional, Generator, Union
import io
import wave
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoquiTTS:
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        vocoder_name: Optional[str] = "vocoder_models/en/ljspeech/hifigan_v2",
        device: str = "cuda",
        use_gpu: bool = True
    ):
        try:
            from TTS.api import TTS
            
            self.device = device if torch.cuda.is_available() and use_gpu else "cpu"
            
            logger.info(f"Loading TTS model: {model_name}")
            
            # Suppress TTS verbose output
            import logging as tts_logging
            tts_logging.getLogger("TTS").setLevel(tts_logging.WARNING)
            
            self.tts = TTS(model_name=model_name, progress_bar=False)
            
            if self.device == "cuda":
                self.tts.to("cuda")
                
            logger.info("TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            self.tts = None
            
        self.sample_rate = 22050
        self.audio_queue = queue.Queue()
        self.is_generating = False
        
    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        stream: bool = False
    ) -> Union[np.ndarray, Generator[np.ndarray, None, None]]:
        
        if not self.tts:
            logger.error("TTS model not loaded")
            return np.array([])
            
        try:
            if stream:
                return self._synthesize_stream(text, speaker, language)
            else:
                wav = self.tts.tts(text=text, speaker=speaker, language=language)
                return np.array(wav)
                
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return np.array([])
            
    def _synthesize_stream(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None
    ) -> Generator[np.ndarray, None, None]:
        
        sentences = self._split_text(text)
        
        for sentence in sentences:
            if sentence.strip():
                wav = self.tts.tts(text=sentence, speaker=speaker, language=language)
                yield np.array(wav)
                
    def _split_text(self, text: str) -> list:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
        
    def synthesize_to_file(
        self,
        text: str,
        file_path: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None
    ):
        if not self.tts:
            logger.error("TTS model not loaded")
            return
            
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=file_path,
                speaker=speaker,
                language=language
            )
            logger.info(f"Audio saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

class XTTS:
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda",
        use_deepspeed: bool = False
    ):
        try:
            from TTS.api import TTS
            
            self.device = device if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading XTTS model: {model_name}")
            self.tts = TTS(model_name, progress_bar=False)
            
            if self.device == "cuda":
                self.tts.to("cuda")
                
            self.use_deepspeed = use_deepspeed and self.device == "cuda"
            
            logger.info("XTTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            self.tts = None
            
        self.sample_rate = 24000
        self.chunk_size = 512
        
    def clone_voice(
        self,
        reference_audio: str,
        reference_text: Optional[str] = None
    ):
        if not self.tts:
            logger.error("XTTS model not loaded")
            return None
            
        try:
            logger.info(f"Cloning voice from {reference_audio}")
            
            speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=reference_audio,
                gpt_cond_len=30,
                max_ref_length=60
            )
            
            self.speaker_embedding = speaker_embedding
            logger.info("Voice cloned successfully")
            return speaker_embedding
            
        except Exception as e:
            logger.error(f"Voice cloning error: {e}")
            return None
            
    def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        speaker_wav: Optional[str] = None,
        temperature: float = 0.75
    ) -> Generator[np.ndarray, None, None]:
        
        if not self.tts:
            logger.error("XTTS model not loaded")
            return
            
        try:
            chunks = self.tts.tts_with_vc_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                temperature=temperature,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=50,
                top_p=0.85,
                stream=True
            )
            
            for chunk in chunks:
                yield chunk
                
        except Exception as e:
            logger.error(f"XTTS streaming error: {e}")

class EdgeTTS:
    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ):
        try:
            import edge_tts
            self.edge_tts = edge_tts
            
            self.voice = voice
            self.rate = rate
            self.pitch = pitch
            
            logger.info(f"EdgeTTS initialized with voice: {voice}")
            
        except ImportError:
            logger.error("edge-tts not installed")
            self.edge_tts = None
            
    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None
    ) -> bytes:
        
        if not self.edge_tts:
            return b""
            
        voice = voice or self.voice
        rate = rate or self.rate
        pitch = pitch or self.pitch
        
        communicate = self.edge_tts.Communicate(
            text,
            voice,
            rate=rate,
            pitch=pitch
        )
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        return audio_data
        
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        stream: bool = False
    ) -> np.ndarray:
        
        import asyncio
        import io
        from pydub import AudioSegment
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            audio_bytes = loop.run_until_complete(
                self.synthesize_async(text, voice, rate, pitch)
            )
            
            # Convert bytes to numpy array
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Normalize to float32
            samples = samples.astype(np.float32) / 32768.0
            
            # Resample if needed
            if audio_segment.frame_rate != 22050:
                import scipy.signal
                samples = scipy.signal.resample(
                    samples,
                    int(len(samples) * 22050 / audio_segment.frame_rate)
                )
            
            return samples
            
        except Exception as e:
            logger.error(f"EdgeTTS synthesis error: {e}")
            return np.array([])
            
        finally:
            loop.close()
            
    async def list_voices(self):
        if not self.edge_tts:
            return []
            
        voices = await self.edge_tts.list_voices()
        return voices