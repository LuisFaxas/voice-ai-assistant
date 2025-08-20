import sounddevice as sd
import numpy as np
import queue
import threading
import logging
from typing import Optional, Union
import time
import io
import wave
from pydub import AudioSegment
from pydub.playback import play

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPlayback:
    def __init__(
        self,
        sample_rate: int = 22050,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stream = None
        self.playback_thread = None
        
        self.volume = 1.0
        self.is_paused = False
        
    def play(self, audio_data: Union[np.ndarray, bytes], blocking: bool = False):
        if isinstance(audio_data, bytes):
            audio_data = self._bytes_to_array(audio_data)
            
        if blocking:
            self._play_blocking(audio_data)
        else:
            self.audio_queue.put(audio_data)
            if not self.is_playing:
                self._start_playback()
                
    def _play_blocking(self, audio_data: np.ndarray):
        try:
            audio_data = audio_data * self.volume
            sd.play(audio_data, self.sample_rate, device=self.device_index)
            sd.wait()
        except Exception as e:
            logger.error(f"Playback error: {e}")
            
    def _start_playback(self):
        if self.is_playing:
            return
            
        self.is_playing = True
        self.playback_thread = threading.Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
    def _playback_worker(self):
        logger.info("Playback worker started")
        
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device_index,
                dtype=np.float32
            )
            self.stream.start()
            
            while self.is_playing:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                    
                try:
                    audio_data = self.audio_queue.get(timeout=0.5)
                    
                    audio_data = audio_data * self.volume
                    
                    audio_data = np.asarray(audio_data, dtype=np.float32)
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(-1, 1)
                        
                    for i in range(0, len(audio_data), self.chunk_size):
                        if not self.is_playing or self.is_paused:
                            break
                            
                        chunk = audio_data[i:i + self.chunk_size]
                        if len(chunk) < self.chunk_size:
                            chunk = np.pad(chunk, ((0, self.chunk_size - len(chunk)), (0, 0)))
                            
                        self.stream.write(chunk)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Playback error: {e}")
                    
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                
            logger.info("Playback worker stopped")
            
    def stream_play(self, audio_generator):
        for audio_chunk in audio_generator:
            if isinstance(audio_chunk, bytes):
                audio_chunk = self._bytes_to_array(audio_chunk)
                
            self.audio_queue.put(audio_chunk)
            
            if not self.is_playing:
                self._start_playback()
                
    def pause(self):
        self.is_paused = True
        logger.info("Playback paused")
        
    def resume(self):
        self.is_paused = False
        logger.info("Playback resumed")
        
    def stop(self):
        self.is_playing = False
        self.clear_queue()
        
        if self.playback_thread:
            self.playback_thread.join(timeout=2)
            
        logger.info("Playback stopped")
        
    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self.volume:.2f}")
        
    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            samples = np.array(audio_segment.get_array_of_samples())
            
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
            else:
                samples = samples.reshape((-1, 1))
                
            samples = samples.astype(np.float32) / (2**15)
            
            if audio_segment.frame_rate != self.sample_rate:
                import scipy.signal
                samples = scipy.signal.resample(
                    samples,
                    int(len(samples) * self.sample_rate / audio_segment.frame_rate)
                )
                
            return samples
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return np.array([])
            
    def save_to_file(self, audio_data: np.ndarray, file_path: str):
        try:
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
                
            logger.info(f"Audio saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            
    def list_devices(self):
        return sd.query_devices()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()