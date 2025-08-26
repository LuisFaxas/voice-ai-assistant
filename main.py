#!/usr/bin/env python3
"""
ULTRA-FAST VOICE ASSISTANT with Mistral 7B & Faster-Whisper
Optimized for RTX 4090 - Maximum GPU utilization
"""

import sys
import os
import time
# Fix Windows encoding issues
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
import re
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import pygame
import torch
import threading
import queue
import io
import shlex  # For proper shell escaping
import json
import select
import wave
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque
import concurrent.futures
try:
    import pyttsx3  # Fallback TTS
except ImportError:
    pyttsx3 = None

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Core imports
import sounddevice as sd
from faster_whisper import WhisperModel  # NEW: Faster-whisper instead of OpenAI
from llama_cpp import Llama

# Initialize pygame with stable audio settings
pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=1024)  # Larger buffer for stability
pygame.mixer.init()

class PerformanceMonitor:
    """Lightweight performance monitoring"""
    
    def __init__(self):
        self.metrics = {}
        self.history = deque(maxlen=10)
        self.current_timers = {}
        
    def start_timer(self, name: str):
        self.current_timers[name] = time.perf_counter()
        
    def end_timer(self, name: str) -> float:
        if name in self.current_timers:
            duration = time.perf_counter() - self.current_timers[name]
            self.metrics[name] = duration
            del self.current_timers[name]
            return duration
        return 0.0
    
    def add_to_history(self):
        if self.metrics:
            self.history.append(self.metrics.copy())
            
    def get_average_metrics(self) -> Dict[str, float]:
        if not self.history:
            return {}
        
        avg_metrics = {}
        for metric_name in self.history[0].keys():
            values = [m.get(metric_name, 0) for m in self.history]
            avg_metrics[metric_name] = sum(values) / len(values)
        return avg_metrics

class FasterWhisperVoiceAssistant:
    """Voice Assistant with Faster-Whisper for 4-5x speed improvement"""
    
    def __init__(self):
        """Initialize with Faster-Whisper optimizations"""
        self.show_banner()
        
        # Performance monitoring
        self.perf = PerformanceMonitor()
        
        # Check GPU
        self.device = self._check_gpu()
        
        # State management
        self.is_running = False
        self.is_speaking = False
        self.conversation_history = []
        
        # Threading for parallel processing
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Persistent Piper process for zero-overhead TTS
        self.piper_process = None
        self.piper_lock = threading.Lock()
        
        # Initialize components
        init_start = time.perf_counter()
        
        self._init_audio()
        self._init_faster_whisper()  # NEW METHOD
        self._init_llm()
        self._init_tts_optimized()
        
        init_time = time.perf_counter() - init_start
        print(f"\n[OK] Systems initialized in {init_time:.2f}s\n")
        
    def show_banner(self):
        mode = "FAST MODE" if os.environ.get('VOICE_MODE', 'fast') == 'fast' else "QUALITY MODE"
        print("\n" + "="*60)
        print(f"  VOICE ASSISTANT - {mode}")
        print("  RTX 4090 Optimized | Target: <2s Fast / <5s Quality")
        print("="*60)
        
    def _check_gpu(self) -> str:
        """Check GPU and optimize settings"""
        print("\n[*] Checking GPU...")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [OK] GPU: {gpu_name}")
            
            # Maximum GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.empty_cache()
            
            return "cuda"
        else:
            print("  [!] No GPU - using CPU")
            return "cpu"
            
    def _init_audio(self):
        """Initialize audio with voice activity detection"""
        print("\n[*] Setting up audio...")
        
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
        # Voice Activity Detection settings (optimized for speed)
        self.vad_threshold = 0.015   # Slightly higher for quicker detection
        self.silence_duration = 0.5  # Stop after 0.5s of silence (faster)
        self.max_recording = 2.5     # Maximum 2.5 seconds (faster)
        self.min_recording = 0.3     # Minimum 0.3 seconds
        
        print("  [OK] Audio ready with VAD")
        
    def _init_faster_whisper(self):
        """Initialize Faster-Whisper with CTranslate2 optimizations"""
        print("\n[*] Loading Faster-Whisper (4-5x faster)...")
        
        start = time.perf_counter()
        
        # Determine compute type based on GPU
        if self.device == "cuda":
            # RTX 4090 supports int8_float16 for best speed/quality tradeoff
            compute_type = "int8_float16"
            device = "cuda"
        else:
            compute_type = "int8"
            device = "cpu"
        
        # Model size options:
        # tiny.en: Fastest (39M params)
        # base.en: Good balance (74M params)  
        # small.en: Better accuracy (244M params)
        # medium.en: High accuracy (769M params)
        # large-v3: Best accuracy (1550M params)
        
        model_size = "tiny.en"  # Fastest model (2x faster than base)
        
        # Initialize Faster-Whisper model
        self.whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=2,  # Parallel processing
            download_root="models/whisper",  # Cache models locally
            local_files_only=False  # Download if needed
        )
        
        # Warm up the model with dummy audio
        dummy_audio = np.zeros(16000, dtype=np.float32)
        list(self.whisper_model.transcribe(
            dummy_audio,
            language="en",
            beam_size=1,
            best_of=1
        ))
        
        load_time = time.perf_counter() - start
        print(f"  [OK] Faster-Whisper '{model_size}' ready in {load_time:.2f}s")
        print(f"     Device: {device}, Compute: {compute_type}")
        print(f"     Expected: 4-5x faster than OpenAI Whisper")
        
    def _init_llm(self):
        """Initialize Mistral 7B with RTX 4090 optimizations"""
        print("\n[*] Loading Mistral 7B...")
        
        start = time.perf_counter()
        
        # Check for performance mode from environment or default to fast
        quality_mode = os.environ.get('VOICE_MODE', 'fast').lower() == 'quality'
        
        if quality_mode:
            # Quality mode: Try Mistral first
            model_paths = [
                Path("models/mistral-7b-instruct-v0.3.Q5_K_M.gguf"),  # Best quality
                Path("models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"),  # Good quality
                Path("models/phi-2.Q5_K_M.gguf")  # Fast fallback
            ]
            print("  Mode: QUALITY (Mistral 7B preferred)")
        else:
            # Fast mode (default): Use Phi-2 for speed
            model_paths = [
                Path("models/phi-2.Q5_K_M.gguf"),  # Fast and responsive
                Path("models/mistral-7b-instruct-v0.3.Q5_K_M.gguf"),  # Fallback
                Path("models/mistral-7b-instruct-v0.2.Q5_K_M.gguf")  # Fallback
            ]
            print("  Mode: FAST (Phi-2 for speed)")
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if not model_path:
            print(f"  [ERROR] No models found")
            sys.exit(1)
        
        model_name = model_path.stem
        print(f"  [*] Loading: {model_name}")
        
        # RTX 4090 optimized settings for Mistral 7B
        if "mistral" in model_name.lower():
            # Mistral 7B optimized for faster response
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,      # Reduced context for speed
                n_gpu_layers=-1 if self.device == "cuda" else 0,  # Full GPU offload
                n_batch=1024,    # Larger batch for faster processing
                n_threads=4,     # Fewer threads for less overhead
                use_mmap=True,   # Memory mapping for faster loading
                use_mlock=False,
                seed=42,
                verbose=False,
                # RTX 4090 specific optimizations
                tensor_split=None,  # Use full GPU
                rope_freq_base=10000,  # Standard RoPE
                rope_freq_scale=1.0,
                f16_kv=True,     # Use FP16 for KV cache
                logits_all=False,
                vocab_only=False,
                embedding=False
            )
            # Better prompt for Mistral
            self.system_prompt = "You are a helpful AI assistant. Respond concisely and clearly."
            self.use_instruct_format = True
        else:
            # Phi-2 settings (optimized for speed)
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=512,       # Minimal context for ultra-fast response
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                n_batch=2048,    # Large batch for speed
                n_threads=8,     # Balanced threading
                use_mmap=True,
                use_mlock=False,
                seed=42,
                verbose=False
            )
            self.system_prompt = "Be concise and helpful. Respond naturally."
            self.use_instruct_format = False
        
        # Warm up the model
        self.llm("Hello", max_tokens=1, echo=False)
        
        load_time = time.perf_counter() - start
        print(f"  [OK] {model_name} ready in {load_time:.2f}s")
        
        if "mistral" in model_name.lower():
            print(f"     RTX 4090: Full GPU offload, 24GB VRAM")
            print(f"     Expected: ~130 tokens/sec generation")
            if "v0.3" in model_name.lower():
                print(f"     Version: v0.3 - Latest with improved reasoning")
            elif "v0.2" in model_name.lower():
                print(f"     Version: v0.2 - Previous stable version")
        
    def _init_tts_optimized(self):
        """Initialize TTS with caching and optimizations"""
        print("\n[*] Setting up Natural TTS...")
        
        voices_dir = Path("models/piper_voices")
        
        # Voice options optimized for speed
        voice_options = [
            ("en_US-amy-medium", "Amy (Fast & Clear)"),  # Faster synthesis
            ("en_US-ryan-high", "Ryan (Quality)"),  # Fallback for quality mode
        ]
        
        self.voice_model = None
        self.voice_config = None
        
        # Try Ryan first for better quality
        for voice_name, desc in voice_options:
            model_path = voices_dir / f"{voice_name}.onnx"
            config_path = voices_dir / f"{voice_name}.onnx.json"
            
            if model_path.exists() and config_path.exists():
                self.voice_model = str(model_path.absolute())
                self.voice_config = str(config_path.absolute())
                self.selected_voice = voice_name
                print(f"  [OK] Voice: {desc}")
                
                # Set voice-specific parameters (stable, no glitching)
                if "amy" in voice_name:
                    self.tts_speed = "1.0"   # Natural speed (1.05 causes glitches)
                    self.tts_silence = "0"   # No pauses for speed
                else:  # Ryan
                    self.tts_speed = "1.0"   # Natural speed
                    self.tts_silence = "0"   # No pauses for speed
                break
        
        if not self.voice_model:
            print("  [ERROR] No voice models found")
            sys.exit(1)
            
        # TTS cache for common responses
        self.tts_cache = {}
        self.piper_gpu_flag = "--cuda" if self.device == "cuda" else ""
        
        # Start persistent Piper process
        self._start_piper_process()
            
    def record_audio_vad(self) -> Optional[np.ndarray]:
        """Record with Voice Activity Detection or instant mode"""
        
        # Wait for TTS to finish speaking before recording
        while self.is_speaking:
            time.sleep(0.1)
        
        # Check for instant mode
        instant_mode = os.environ.get('INSTANT_MODE', '0') == '1'
        
        if instant_mode:
            # Fixed 1 second recording, no VAD
            print("\n[MIC] Recording (1s)...", end="", flush=True)
            self.perf.start_timer("Recording")
            
            duration = 1.0  # Fixed 1 second
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate,
                          channels=self.channels,
                          dtype='float32')
            sd.wait()
            
            rec_time = self.perf.end_timer("Recording")
            print(f" OK ({rec_time:.1f}s)")
            return audio.flatten()
        else:
            # Standard VAD mode
            print("\n[MIC] Listening (VAD)...", end="", flush=True)
            
            self.perf.start_timer("Recording")
            
            frames = []
            silence_counter = 0
            speech_detected = False
            start_time = time.perf_counter()
            
            def callback(indata, frames_count, time_info, status):
                if status:
                    print(f"Status: {status}")
                frames.append(indata.copy())
            
            with sd.InputStream(callback=callback,
                               channels=self.channels,
                               samplerate=self.sample_rate,
                               blocksize=self.chunk_size):
                
                while True:
                    time.sleep(0.05)  # Faster checking
                    
                    if len(frames) > 0:
                        # Check recent audio level
                        recent_audio = np.concatenate(frames[-5:]) if len(frames) > 5 else np.concatenate(frames)
                        amplitude = np.max(np.abs(recent_audio))
                        
                        if amplitude > self.vad_threshold:
                            speech_detected = True
                            silence_counter = 0
                        elif speech_detected:
                            silence_counter += 1
                        
                        # Stop conditions
                        elapsed = time.perf_counter() - start_time
                        
                        if speech_detected and silence_counter > (self.silence_duration * 20):  # Adjusted for 0.05s sleep
                            break  # Silence after speech
                        elif elapsed > self.max_recording:
                            break  # Max duration reached
                        elif not speech_detected and elapsed > 1.5:  # Reduced from 2.0
                            break  # No speech detected quickly
            
            rec_time = self.perf.end_timer("Recording")
            
            if frames and speech_detected:
                audio = np.concatenate(frames)
                print(f" OK ({rec_time:.1f}s)")
                return audio.flatten()
            else:
                print(" (no speech)")
                return None
            
    def transcribe_audio_faster(self, audio: np.ndarray) -> Optional[str]:
        """Ultra-fast transcription with Faster-Whisper"""
        if audio is None:
            return None
            
        print("[STT] Transcribing (Faster-Whisper)...", end="", flush=True)
        
        self.perf.start_timer("Transcription")
        
        try:
            # Faster-Whisper transcription with ULTRA optimized settings
            segments, info = self.whisper_model.transcribe(
                audio,
                language="en",
                beam_size=1,  # Fastest
                best_of=1,    # Single attempt
                temperature=0,  # Deterministic
                vad_filter=False,  # Skip VAD for speed
                without_timestamps=True,  # No timestamps
                word_timestamps=False,  # No word-level
                condition_on_previous_text=False,  # No context
            )
            
            # Collect transcribed text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            text = " ".join(text_parts).strip()
            
            trans_time = self.perf.end_timer("Transcription")
            print(f" OK ({trans_time:.2f}s)")
            
            if text:
                print(f"\n[USER] You: {text}")
                return text
            return None
                
        except Exception as e:
            print(f" ERROR: {e}")
            return None
            
    def generate_response_streaming(self, user_input: str):
        """Stream LLM response with concurrent TTS for ultra-low latency"""
        if not user_input:
            return (None, None)
            
        print("[LLM] Thinking...", end="", flush=True)
        
        self.perf.start_timer("LLM")
        
        # Format prompt based on model
        if hasattr(self, 'use_instruct_format') and self.use_instruct_format:
            # Mistral Instruct format
            prompt = f"[INST] {user_input} [/INST]"
            max_tokens = 25  # Reduced for speed
            temperature = 0.5  # More focused
            top_p = 0.85
            top_k = 20
        else:
            # Phi-2 format
            prompt = f"User: {user_input}\nAssistant:"
            max_tokens = 20  # Reduced for speed
            temperature = 0.5
            top_p = 0.85
            top_k = 15
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=1.05,
                stop=["[INST]", "User:", "\n\n"],
                echo=False,
                stream=False  # Could enable streaming in future
            )
            
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                
                # Clean up
                text = text.replace("Assistant:", "").strip()
                text = text.replace("[/INST]", "").strip()
                
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                
                llm_time = self.perf.end_timer("LLM")
                print(f" OK ({llm_time:.2f}s)")
                
                # Start TTS immediately in parallel
                tts_thread = threading.Thread(target=self.speak_text_fast, args=(text,))
                tts_thread.start()
                
                print(f"\n[AI] {text}")
                
                # Return both text and thread for synchronization
                return (text, tts_thread)
                
        except Exception as e:
            print(f" ERROR: {e}")
            return (None, None)
            
    def speak_text_fast(self, text: str):
        """Robust TTS with fallback options"""
        if not text:
            return
            
        print("[TTS] Speaking...", end="", flush=True)
        
        self.perf.start_timer("TTS")
        self.is_speaking = True
        
        # Clean text for TTS
        text = text.strip()
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove problematic chars
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Check cache first
        if text in self.tts_cache:
            try:
                pygame.mixer.music.load(self.tts_cache[text])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.01)
                pygame.mixer.music.unload()
                
                tts_time = self.perf.end_timer("TTS")
                print(f" OK (cached: {tts_time:.2f}s)")
                self.is_speaking = False
                return
            except:
                pass  # Cache failed, continue with generation
        
        # Try Piper TTS first
        success = self._speak_with_piper(text)
        
        # If Piper fails, try fallback
        if not success and pyttsx3:
            success = self._speak_with_pyttsx3(text)
        
        tts_time = self.perf.end_timer("TTS")
        if success:
            print(f" OK ({tts_time:.2f}s)")
        else:
            print(f" FAILED")
        
        self.is_speaking = False
    
    def _start_piper_process(self):
        """Start persistent Piper process for zero-overhead TTS"""
        try:
            # Start Piper in streaming mode
            cmd = [
                'piper',
                '--model', self.voice_model,
                '--config', self.voice_config,
                '--length-scale', '1.0',
                '--sentence-silence', '0',
                '--output-raw'  # Raw audio output for streaming
            ]
            
            self.piper_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered for real-time
            )
            
            print("  [OK] Persistent Piper process started")
            
        except Exception as e:
            print(f"  [WARN] Failed to start persistent Piper: {e}")
            self.piper_process = None
    
    def _speak_with_piper(self, text: str) -> bool:
        """Try to speak with Piper TTS"""
        temp_wav = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
            
            # Use shell command with proper escaping for reliability
            escaped_text = shlex.quote(text)
            cmd = f'echo {escaped_text} | piper --model "{self.voice_model}" --config "{self.voice_config}" --output_file "{temp_wav}" --length-scale 1.0 --sentence-silence 0'
            
            # Run with shell=True for piping
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=2, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                # Play the audio
                try:
                    pygame.mixer.music.load(temp_wav)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.01)
                    
                    pygame.mixer.music.unload()
                    
                    # Cache successful generation
                    if len(text) < 50:
                        self.tts_cache[text] = temp_wav
                        temp_wav = None  # Don't delete cached file
                    
                    return True
                except pygame.error as e:
                    print(f" [pygame error: {e}]", end="")
                    return False
            else:
                if result.stderr:
                    print(f" [piper: {result.stderr[:50]}]", end="")
                return False
                
        except subprocess.TimeoutExpired:
            print(" [timeout]", end="")
            return False
        except Exception as e:
            print(f" [{type(e).__name__}]", end="")
            return False
        finally:
            # Cleanup temp file if not cached
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except:
                    pass
    
    def _speak_with_pyttsx3(self, text: str) -> bool:
        """Fallback TTS using pyttsx3"""
        try:
            if not hasattr(self, 'pyttsx3_engine'):
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', 180)  # Slightly faster
                
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            return True
        except Exception as e:
            print(f" [pyttsx3: {e}]", end="")
            return False
            
    def conversation_loop(self):
        """Main loop with ultra-low latency"""
        mode = os.environ.get('VOICE_MODE', 'fast').upper()
        instant = " (Instant)" if os.environ.get('INSTANT_MODE', '0') == '1' else " (VAD)"
        
        print("\n" + "="*60)
        print(f"  {mode} MODE ACTIVE - RTX 4090 ACCELERATED")
        print("="*60)
        print(f"  - ENTER: Voice input{instant}")
        print("  - Type: Text input")
        print("  - Commands: quit, metrics")
        print("="*60)
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Get input
                user_action = input("\n[ENTER/type]: ").strip()
                
                if user_action.lower() in ['quit', 'exit']:
                    self.show_metrics_summary()
                    break
                elif user_action.lower() == 'metrics':
                    self.show_metrics_summary()
                    continue
                    
                # Start total timer
                self.perf.start_timer("Total")
                
                # Process input
                if user_action:
                    user_input = user_action
                    print(f"\n👤 You: {user_input}")
                else:
                    # Voice with VAD
                    audio = self.record_audio_vad()
                    user_input = self.transcribe_audio_faster(audio)  # Using faster-whisper
                    
                if user_input:
                    # Generate with streaming TTS
                    result = self.generate_response_streaming(user_input)
                    if result:
                        # Unpack response and thread
                        if isinstance(result, tuple):
                            response_text, tts_thread = result
                            # Wait for TTS to complete
                            tts_thread.join(timeout=10.0)
                        else:
                            response_text = result
                        
                        # Show metrics
                        total_time = self.perf.end_timer("Total")
                        
                        print(f"\n[TIME] ", end="")
                        for name, duration in self.perf.metrics.items():
                            if name != "Total":
                                print(f"{name}: {duration:.1f}s | ", end="")
                        print(f"Total: {total_time:.1f}s")
                        
                        # Check if we hit new target
                        if total_time < 2.0:
                            print("  [**] BLAZING FAST! (<2s)")
                        elif total_time < 3.0:
                            print("  [*] ULTRA-FAST! (<3s)")
                        elif total_time < 5.0:
                            print("  [OK] Target achieved! (<5s)")
                        
                        self.perf.add_to_history()
                        
            except KeyboardInterrupt:
                print("\n[!] Interrupted")
                self.show_metrics_summary()
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                
        print("\n[*] Shutting down...")
        
        # Cleanup persistent Piper
        if hasattr(self, 'piper_process') and self.piper_process:
            try:
                self.piper_process.terminate()
                self.piper_process.wait(timeout=1.0)
            except:
                pass
        
    def show_metrics_summary(self):
        """Display performance summary"""
        if self.perf.history:
            avg = self.perf.get_average_metrics()
            print("\n[METRICS] Performance Summary (Faster-Whisper):")
            print("  " + "-"*40)
            total = 0
            for name, duration in avg.items():
                if name != "Total":
                    print(f"  {name:15s}: {duration:.2f}s avg")
                    total += duration
            print("  " + "-"*40)
            print(f"  {'TOTAL':15s}: {total:.2f}s avg")
            
            if total < 2.0:
                print("\n  [**] ACHIEVING BLAZING FAST TARGET! <2s average!")
            elif total < 3.0:
                print("\n  [*] ACHIEVING ULTRA-FAST TARGET! <3s average!")
            elif total < 5.0:
                print("\n  [OK] ACHIEVING TARGET! <5s average!")

def main():
    """Main entry point with performance modes"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Assistant with Performance Modes")
    parser.add_argument("--quality", action="store_true", 
                       help="Quality mode: Use Mistral 7B for better responses (slower)")
    parser.add_argument("--fast", action="store_true", default=True,
                       help="Fast mode: Use Phi-2 for quick responses (default)")
    parser.add_argument("--instant", action="store_true",
                       help="Instant recording: Fixed 1s recording, no VAD")
    args = parser.parse_args()
    
    # Set performance mode
    if args.quality:
        os.environ['VOICE_MODE'] = 'quality'
        mode = "QUALITY MODE (Mistral 7B)"
    else:
        os.environ['VOICE_MODE'] = 'fast'
        mode = "FAST MODE (Phi-2)"
    
    if args.instant:
        os.environ['INSTANT_MODE'] = '1'
        mode += " + INSTANT RECORDING"
    
    try:
        print("\n[*] Initializing Voice Assistant...")
        print(f"   Mode: {mode}")
        print("   RTX 4090 Optimized | Faster-Whisper STT")
        
        assistant = FasterWhisperVoiceAssistant()
        assistant.conversation_loop()
    except Exception as e:
        print(f"\n[ERROR] Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()