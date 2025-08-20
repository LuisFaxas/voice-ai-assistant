import sounddevice as sd
import numpy as np
import queue
import threading
import logging
from typing import Optional, Callable
import webrtcvad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        vad_mode: int = 3,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.device_index = device_index
        
        self.vad = webrtcvad.Vad(vad_mode)
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_muted = False
        self.stream = None
        self.thread = None
        
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)
        
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        audio_chunk = indata.copy().flatten()
        
        if len(audio_chunk) == self.chunk_size:
            audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
            
            self.audio_queue.put({
                'data': audio_chunk,
                'is_speech': is_speech,
                'timestamp': time_info.inputBufferAdcTime
            })
            
            for callback in self.callbacks:
                try:
                    callback(audio_chunk, is_speech)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def start(self):
        if self.is_recording:
            return
            
        self.is_recording = True
        
        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=self.device_index,
            dtype=np.float32
        )
        
        self.stream.start()
        logger.info("Audio capture started")
        
    def stop(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        logger.info("Audio capture stopped")
        
    def get_audio_chunk(self, timeout: Optional[float] = None):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
    def mute(self):
        self.is_muted = True
        logger.debug("Audio capture muted")
        
    def unmute(self):
        self.is_muted = False
        logger.debug("Audio capture unmuted")
        
    def list_devices(self):
        return sd.query_devices()
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()