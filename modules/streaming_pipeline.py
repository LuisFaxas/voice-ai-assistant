import asyncio
import queue
import threading
import logging
from typing import Optional, Callable, Any
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingPipeline:
    def __init__(
        self,
        audio_capture,
        stt_module,
        llm_module,
        tts_module,
        audio_playback,
        enable_interruption: bool = True
    ):
        self.audio_capture = audio_capture
        self.stt = stt_module
        self.llm = llm_module
        self.tts = tts_module
        self.audio_playback = audio_playback
        self.enable_interruption = enable_interruption
        
        self.stt_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        
        self.is_running = False
        self.is_processing = False
        self.is_speaking = False
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.stt_thread = None
        self.llm_thread = None
        self.tts_thread = None
        
        self.current_transcription = ""
        self.current_response = ""
        
        self.callbacks = {
            'on_transcription': None,
            'on_response_start': None,
            'on_response_token': None,
            'on_response_end': None,
            'on_speech_start': None,
            'on_speech_end': None,
            'on_error': None
        }
        
    def set_callback(self, event: str, callback: Callable):
        if event in self.callbacks:
            self.callbacks[event] = callback
            
    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        
        self.audio_capture.add_callback(self._handle_audio_chunk)
        self.audio_capture.start()
        
        self.stt_thread = threading.Thread(target=self._stt_worker)
        self.stt_thread.daemon = True
        self.stt_thread.start()
        
        self.llm_thread = threading.Thread(target=self._llm_worker)
        self.llm_thread.daemon = True
        self.llm_thread.start()
        
        self.tts_thread = threading.Thread(target=self._tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()
        
        logger.info("Streaming pipeline started")
        
    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        
        self.audio_capture.stop()
        
        self.stt_queue.put(None)
        self.llm_queue.put(None)
        self.tts_queue.put(None)
        
        if self.stt_thread:
            self.stt_thread.join(timeout=2)
        if self.llm_thread:
            self.llm_thread.join(timeout=2)
        if self.tts_thread:
            self.tts_thread.join(timeout=2)
            
        self.executor.shutdown(wait=False)
        
        logger.info("Streaming pipeline stopped")
        
    def _handle_audio_chunk(self, audio_chunk: np.ndarray, is_speech: bool):
        # Simple approach: don't process audio while speaking
        if self.is_speaking:
            return
                
        # Process audio when not speaking
        self.stt_queue.put((audio_chunk, is_speech))
            
    def _stt_worker(self):
        logger.info("STT worker started")
        
        while self.is_running:
            try:
                item = self.stt_queue.get(timeout=0.5)
                if item is None:
                    break
                    
                audio_chunk, is_speech = item
                
                transcription = self.stt.process_audio_chunk(audio_chunk, is_speech)
                
                if transcription:
                    self.current_transcription = transcription
                    logger.info(f"Transcription: {transcription}")
                    
                    if self.callbacks['on_transcription']:
                        self.callbacks['on_transcription'](transcription)
                        
                    self.llm_queue.put(transcription)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"STT error: {e}")
                if self.callbacks['on_error']:
                    self.callbacks['on_error']('stt', str(e))
                    
        logger.info("STT worker stopped")
        
    def _llm_worker(self):
        logger.info("LLM worker started")
        
        while self.is_running:
            try:
                transcription = self.llm_queue.get(timeout=0.5)
                if transcription is None:
                    break
                    
                self.is_processing = True
                
                if self.callbacks['on_response_start']:
                    self.callbacks['on_response_start']()
                    
                self.llm.add_to_history("user", transcription)
                
                response_tokens = []
                sentence_buffer = []
                
                for token in self.llm.generate(transcription, stream=True):
                    if not self.is_running:
                        break
                        
                    response_tokens.append(token)
                    sentence_buffer.append(token)
                    
                    if self.callbacks['on_response_token']:
                        self.callbacks['on_response_token'](token)
                        
                    if any(punct in token for punct in ['.', '!', '?', '\n']):
                        sentence = ''.join(sentence_buffer)
                        if sentence.strip():
                            self.tts_queue.put(sentence)
                        sentence_buffer = []
                        
                if sentence_buffer:
                    sentence = ''.join(sentence_buffer)
                    if sentence.strip():
                        self.tts_queue.put(sentence)
                        
                self.current_response = ''.join(response_tokens)
                self.llm.add_to_history("assistant", self.current_response)
                
                if self.callbacks['on_response_end']:
                    self.callbacks['on_response_end'](self.current_response)
                    
                self.is_processing = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"LLM error: {e}")
                if self.callbacks['on_error']:
                    self.callbacks['on_error']('llm', str(e))
                self.is_processing = False
                
        logger.info("LLM worker stopped")
        
    def _tts_worker(self):
        logger.info("TTS worker started")
        
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=0.5)
                if text is None:
                    break
                    
                # Skip empty text
                if not text or not text.strip():
                    continue
                    
                self.is_speaking = True
                
                if self.callbacks['on_speech_start']:
                    self.callbacks['on_speech_start']()
                
                try:
                    # Synthesize audio
                    audio_data = self.tts.synthesize(text, stream=False)
                        
                    if audio_data is not None and len(audio_data) > 0:
                        self.audio_playback.play(audio_data, blocking=True)
                    else:
                        logger.warning(f"TTS returned empty audio for text: {text[:50]}...")
                        
                except Exception as e:
                    logger.error(f"TTS synthesis error: {e}")
                    
                # Add small delay to let audio finish
                time.sleep(0.3)
                    
                self.is_speaking = False
                
                if self.callbacks['on_speech_end']:
                    self.callbacks['on_speech_end']()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
                if self.callbacks['on_error']:
                    self.callbacks['on_error']('tts', str(e))
                self.is_speaking = False
                
        logger.info("TTS worker stopped")
        
    def _interrupt_speech(self):
        if self.is_speaking:
            logger.info("Interrupting speech")
            
            self.audio_playback.stop()
            self.audio_playback.clear_queue()
            
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except queue.Empty:
                    break
                    
            self.is_speaking = False
            
    def process_text_input(self, text: str):
        self.llm_queue.put(text)
        
    def get_status(self) -> dict:
        return {
            'is_running': self.is_running,
            'is_processing': self.is_processing,
            'is_speaking': self.is_speaking,
            'stt_queue_size': self.stt_queue.qsize(),
            'llm_queue_size': self.llm_queue.qsize(),
            'tts_queue_size': self.tts_queue.qsize()
        }

class AsyncStreamingPipeline:
    def __init__(
        self,
        audio_capture,
        stt_module,
        llm_module,
        tts_module,
        audio_playback
    ):
        self.audio_capture = audio_capture
        self.stt = stt_module
        self.llm = llm_module
        self.tts = tts_module
        self.audio_playback = audio_playback
        
        self.is_running = False
        self.tasks = []
        
    async def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        
        self.tasks = [
            asyncio.create_task(self._audio_capture_task()),
            asyncio.create_task(self._stt_task()),
            asyncio.create_task(self._llm_task()),
            asyncio.create_task(self._tts_task())
        ]
        
        logger.info("Async streaming pipeline started")
        
    async def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Async streaming pipeline stopped")
        
    async def _audio_capture_task(self):
        while self.is_running:
            await asyncio.sleep(0.01)
            
    async def _stt_task(self):
        while self.is_running:
            await asyncio.sleep(0.01)
            
    async def _llm_task(self):
        while self.is_running:
            await asyncio.sleep(0.01)
            
    async def _tts_task(self):
        while self.is_running:
            await asyncio.sleep(0.01)