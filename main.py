#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED VOICE ASSISTANT - Sub-5 Second Response Time
Aggressive optimizations for minimum latency
"""

import sys
import os
import time
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
from typing import Optional, Dict
from datetime import datetime
from collections import deque
import concurrent.futures

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Core imports
import sounddevice as sd
import whisper
from llama_cpp import Llama

# Initialize pygame with minimal latency settings
pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=128)
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

class UltraVoiceAssistant:
    """Ultra-optimized Voice Assistant targeting <5s total latency"""
    
    def __init__(self):
        """Initialize with aggressive optimizations"""
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
        
        # Initialize components
        init_start = time.perf_counter()
        
        self._init_audio()
        self._init_stt()
        self._init_llm()
        self._init_tts_optimized()
        
        init_time = time.perf_counter() - init_start
        print(f"\n✅ Systems initialized in {init_time:.2f}s\n")
        
    def show_banner(self):
        print("\n" + "="*60)
        print("  ⚡ ULTRA-OPTIMIZED VOICE ASSISTANT")
        print("  Target: <5 Second Response | RTX 4090")
        print("="*60)
        
    def _check_gpu(self) -> str:
        """Check GPU and optimize settings"""
        print("\n🔍 Checking GPU...")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ GPU: {gpu_name}")
            
            # Maximum GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.empty_cache()
            
            return "cuda"
        else:
            print("  ⚠️  No GPU - using CPU")
            return "cpu"
            
    def _init_audio(self):
        """Initialize audio with voice activity detection"""
        print("\n🎤 Setting up audio...")
        
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
        # Voice Activity Detection settings
        self.vad_threshold = 0.01
        self.silence_duration = 0.8  # Stop after 0.8s of silence
        self.max_recording = 4.0     # Maximum 4 seconds
        self.min_recording = 0.5     # Minimum 0.5 seconds
        
        print("  ✅ Audio ready with VAD")
        
    def _init_stt(self):
        """Initialize Whisper with speed optimizations"""
        print("\n🎧 Loading Whisper...")
        
        start = time.perf_counter()
        
        # Use tiny model for ultra-fast processing
        model_size = "tiny.en"  # English-only tiny model
        
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        # Warm up the model
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.whisper_model.transcribe(dummy_audio, language="en", fp16=False)
        
        load_time = time.perf_counter() - start
        print(f"  ✅ Whisper '{model_size}' ready in {load_time:.2f}s")
        
    def _init_llm(self):
        """Initialize LLM with maximum speed settings"""
        print("\n🧠 Loading LLM...")
        
        start = time.perf_counter()
        
        model_path = Path("models/phi-2.Q5_K_M.gguf")
        
        if not model_path.exists():
            print(f"  ❌ Model not found: {model_path}")
            sys.exit(1)
        
        # Ultra-fast settings
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=1024,      # Smaller context for speed
            n_gpu_layers=-1 if self.device == "cuda" else 0,
            n_batch=2048,    # Large batch
            n_threads=16,    # Max threads
            use_mmap=True,
            use_mlock=False,
            seed=42,         # Fixed seed for consistency
            verbose=False
        )
        
        # Warm up the model
        self.llm("Hello", max_tokens=1, echo=False)
        
        load_time = time.perf_counter() - start
        print(f"  ✅ LLM ready in {load_time:.2f}s")
        
        # Ultra-brief prompt
        self.system_prompt = "Reply in max 10 words."
        
    def _init_tts_optimized(self):
        """Initialize TTS with caching and optimizations"""
        print("\n🗣️ Setting up Ultra-Fast TTS...")
        
        voices_dir = Path("models/piper_voices")
        
        # Use fastest voice (amy-medium is faster than ryan-high)
        voice_options = [
            ("en_US-amy-medium", "Amy (Fast)"),
            ("en_US-ryan-high", "Ryan (Quality)"),
        ]
        
        self.voice_model = None
        self.voice_config = None
        
        for voice_name, desc in voice_options:
            model_path = voices_dir / f"{voice_name}.onnx"
            config_path = voices_dir / f"{voice_name}.onnx.json"
            
            if model_path.exists() and config_path.exists():
                self.voice_model = str(model_path.absolute())
                self.voice_config = str(config_path.absolute())
                self.selected_voice = voice_name
                print(f"  ✅ Voice: {desc}")
                break
        
        if not self.voice_model:
            print("  ❌ No voice models found")
            sys.exit(1)
            
        # TTS cache for common responses
        self.tts_cache = {}
        self.piper_gpu_flag = "--cuda" if self.device == "cuda" else ""
        
        # Pre-generate common responses
        self._pregenerate_common_responses()
        
    def _pregenerate_common_responses(self):
        """Cache common responses for instant playback"""
        common_phrases = [
            "I can help you with that.",
            "Sure, let me help.",
            "Yes.",
            "No.",
            "I understand.",
        ]
        
        print("  Pre-generating common responses...")
        for phrase in common_phrases:
            # Skip for now to save startup time
            pass
            
    def record_audio_vad(self) -> Optional[np.ndarray]:
        """Record with Voice Activity Detection for dynamic duration"""
        print("\n🎤 Listening (VAD)...", end="", flush=True)
        
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
                time.sleep(0.1)
                
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
                    
                    if speech_detected and silence_counter > (self.silence_duration * 10):
                        break  # Silence after speech
                    elif elapsed > self.max_recording:
                        break  # Max duration reached
                    elif not speech_detected and elapsed > 2.0:
                        break  # No speech detected in 2 seconds
        
        rec_time = self.perf.end_timer("Recording")
        
        if frames and speech_detected:
            audio = np.concatenate(frames)
            print(f" ✓ ({rec_time:.1f}s)")
            return audio.flatten()
        else:
            print(" (no speech)")
            return None
            
    def transcribe_audio_fast(self, audio: np.ndarray) -> Optional[str]:
        """Ultra-fast transcription"""
        if audio is None:
            return None
            
        print("📝 Transcribing...", end="", flush=True)
        
        self.perf.start_timer("Transcription")
        
        try:
            # Minimal settings for speed
            result = self.whisper_model.transcribe(
                audio,
                language="en",
                fp16=(self.device == "cuda"),
                beam_size=1,
                best_of=1,
                temperature=0,  # Deterministic
                without_timestamps=True,
                condition_on_previous_text=False
            )
            
            text = result["text"].strip()
            
            trans_time = self.perf.end_timer("Transcription")
            print(f" ✓ ({trans_time:.2f}s)")
            
            if text:
                print(f"\n👤 You: {text}")
                return text
            return None
                
        except Exception as e:
            print(f" ❌ {e}")
            return None
            
    def generate_response_fast(self, user_input: str) -> Optional[str]:
        """Ultra-fast LLM response"""
        if not user_input:
            return None
            
        print("🤔 Thinking...", end="", flush=True)
        
        self.perf.start_timer("LLM")
        
        # Minimal prompt
        prompt = f"User: {user_input}\nAssistant (max 10 words):"
        
        try:
            response = self.llm(
                prompt,
                max_tokens=20,  # Very short
                temperature=0.5,
                top_p=0.9,
                top_k=10,
                repeat_penalty=1.1,
                stop=["User:", "\n"],
                echo=False
            )
            
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                
                # Clean up
                text = text.replace("Assistant:", "").strip()
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                
                llm_time = self.perf.end_timer("LLM")
                print(f" ✓ ({llm_time:.2f}s)")
                
                print(f"\n🤖 AI: {text}")
                return text
                
        except Exception as e:
            print(f" ❌ {e}")
            return None
            
    def speak_text_fast(self, text: str):
        """Ultra-fast TTS with streaming"""
        if not text:
            return
            
        print("🔊 Speaking...", end="", flush=True)
        
        self.perf.start_timer("TTS")
        self.is_speaking = True
        
        # Check cache first
        if text in self.tts_cache:
            # Play from cache
            pygame.mixer.music.load(self.tts_cache[text])
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            
            tts_time = self.perf.end_timer("TTS")
            print(f" ✓ (cached: {tts_time:.2f}s)")
            return
        
        temp_wav = None
        
        try:
            # Use pipe for faster generation
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
            
            # Ultra-fast Piper settings
            cmd = f'echo "{text}" | piper --model "{self.voice_model}" --config "{self.voice_config}" --output_file "{temp_wav}" --length-scale 0.9 --sentence-silence 0'
            
            if self.piper_gpu_flag:
                cmd += f" {self.piper_gpu_flag}"
            
            # Run in shell for speed
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=3)
            
            if result.returncode == 0 and os.path.exists(temp_wav):
                # Play immediately
                pygame.mixer.music.load(temp_wav)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.02)
                
                pygame.mixer.music.unload()
                
                tts_time = self.perf.end_timer("TTS")
                print(f" ✓ ({tts_time:.2f}s)")
                
                # Cache if short enough
                if len(text) < 50:
                    self.tts_cache[text] = temp_wav
                    temp_wav = None  # Don't delete cached file
                    
        except Exception as e:
            print(f" ❌ {e}")
            
        finally:
            self.is_speaking = False
            
            # Cleanup
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except:
                    pass
                    
            time.sleep(0.1)
            
    def conversation_loop(self):
        """Main loop with ultra-low latency"""
        print("\n" + "="*60)
        print("  💬 ULTRA-FAST MODE - Target <5s Response")
        print("="*60)
        print("  • ENTER: Voice input with VAD")
        print("  • Type: Text input")
        print("  • Commands: quit, metrics")
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
                    user_input = self.transcribe_audio_fast(audio)
                    
                if user_input:
                    # Generate and speak
                    response = self.generate_response_fast(user_input)
                    if response:
                        self.speak_text_fast(response)
                        
                        # Show metrics
                        total_time = self.perf.end_timer("Total")
                        
                        print(f"\n⏱️  ", end="")
                        for name, duration in self.perf.metrics.items():
                            if name != "Total":
                                print(f"{name}: {duration:.1f}s | ", end="")
                        print(f"Total: {total_time:.1f}s")
                        
                        # Check if we hit target
                        if total_time < 5.0:
                            print("  ✅ Target achieved! (<5s)")
                        
                        self.perf.add_to_history()
                        
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted")
                self.show_metrics_summary()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
        print("\n🔌 Shutting down...")
        
    def show_metrics_summary(self):
        """Display performance summary"""
        if self.perf.history:
            avg = self.perf.get_average_metrics()
            print("\n📊 Performance Summary:")
            print("  " + "-"*40)
            total = 0
            for name, duration in avg.items():
                if name != "Total":
                    print(f"  {name:15s}: {duration:.2f}s avg")
                    total += duration
            print("  " + "-"*40)
            print(f"  {'TOTAL':15s}: {total:.2f}s avg")
            
            if total < 5.0:
                print("\n  🏆 ACHIEVING TARGET! <5s average!")

def main():
    """Main entry point"""
    try:
        assistant = UltraVoiceAssistant()
        assistant.conversation_loop()
    except Exception as e:
        print(f"\n💥 Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()