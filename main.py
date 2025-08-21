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
from typing import Optional, Dict, List
from datetime import datetime
from collections import deque
import concurrent.futures

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Core imports
import sounddevice as sd
from faster_whisper import WhisperModel  # NEW: Faster-whisper instead of OpenAI
from llama_cpp import Llama

# Initialize pygame with balanced quality/latency settings
pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
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
        
        # Initialize components
        init_start = time.perf_counter()
        
        self._init_audio()
        self._init_faster_whisper()  # NEW METHOD
        self._init_llm()
        self._init_tts_optimized()
        
        init_time = time.perf_counter() - init_start
        print(f"\n[OK] Systems initialized in {init_time:.2f}s\n")
        
    def show_banner(self):
        print("\n" + "="*60)
        print("  MISTRAL 7B + FASTER-WHISPER VOICE ASSISTANT")
        print("  RTX 4090 Optimized | Target: <3s Response")
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
        
        # Voice Activity Detection settings
        self.vad_threshold = 0.01
        self.silence_duration = 0.8  # Stop after 0.8s of silence
        self.max_recording = 4.0     # Maximum 4 seconds
        self.min_recording = 0.5     # Minimum 0.5 seconds
        
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
        
        model_size = "base.en"  # Better than tiny, still very fast with faster-whisper
        
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
        
        # Try Mistral v0.3 first, then v0.2, then fallback to Phi-2
        model_paths = [
            Path("models/mistral-7b-instruct-v0.3.Q5_K_M.gguf"),  # Mistral 7B v0.3 (latest)
            Path("models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"),  # Mistral 7B v0.2
            Path("models/phi-2.Q5_K_M.gguf")  # Fallback
        ]
        
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
            # Mistral 7B optimal settings for RTX 4090
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=4096,      # Larger context for better understanding
                n_gpu_layers=-1 if self.device == "cuda" else 0,  # Full GPU offload
                n_batch=512,     # Optimal batch size for 7B model
                n_threads=8,     # Balanced threading
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
            # Phi-2 settings (fallback)
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=1024,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                n_batch=2048,
                n_threads=16,
                use_mmap=True,
                use_mlock=False,
                seed=42,
                verbose=False
            )
            self.system_prompt = "Reply in max 10 words."
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
        
        # Voice options with quality profiles
        voice_options = [
            ("en_US-ryan-high", "Ryan (Natural Quality)"),  # Better for natural speech
            ("en_US-amy-medium", "Amy (Faster Response)"),  # Fallback
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
                
                # Set voice-specific parameters
                if "ryan" in voice_name:
                    self.tts_speed = "0.98"  # Natural speed for Ryan
                    self.tts_silence = "0.2"  # More natural pauses
                else:
                    self.tts_speed = "0.95"  # Slightly faster for Amy
                    self.tts_silence = "0.15"  # Shorter pauses
                break
        
        if not self.voice_model:
            print("  [ERROR] No voice models found")
            sys.exit(1)
            
        # TTS cache for common responses
        self.tts_cache = {}
        self.piper_gpu_flag = "--cuda" if self.device == "cuda" else ""
            
    def record_audio_vad(self) -> Optional[np.ndarray]:
        """Record with Voice Activity Detection for dynamic duration"""
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
            # Faster-Whisper transcription with optimized settings
            segments, info = self.whisper_model.transcribe(
                audio,
                language="en",
                beam_size=1,  # Fastest beam search
                best_of=1,    # Single attempt
                temperature=0,  # Deterministic
                vad_filter=True,  # Enable VAD filter for better accuracy
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence for splitting
                    threshold=0.6,  # VAD threshold
                    min_speech_duration_ms=250,  # Minimum speech duration
                    max_speech_duration_s=float('inf')
                ),
                without_timestamps=True,  # Skip timestamps for speed
                word_timestamps=False,  # Skip word-level timestamps
                condition_on_previous_text=False,  # Don't use context
                compression_ratio_threshold=2.4,  # Skip low quality audio
                log_prob_threshold=-1.0,  # Skip uncertain segments
                no_speech_threshold=0.6  # Threshold for silence detection
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
            
    def generate_response_fast(self, user_input: str) -> Optional[str]:
        """Ultra-fast LLM response with Mistral 7B optimizations"""
        if not user_input:
            return None
            
        print("[LLM] Thinking...", end="", flush=True)
        
        self.perf.start_timer("LLM")
        
        # Format prompt based on model
        if hasattr(self, 'use_instruct_format') and self.use_instruct_format:
            # Mistral Instruct format
            prompt = f"[INST] {user_input} [/INST]"
            max_tokens = 50  # Allow more tokens for better responses
            temperature = 0.7
            top_p = 0.95
            top_k = 40
        else:
            # Phi-2 format
            prompt = f"User: {user_input}\nAssistant (max 10 words):"
            max_tokens = 20
            temperature = 0.5
            top_p = 0.9
            top_k = 10
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=1.1,
                stop=["[INST]", "User:", "\n\n"],
                echo=False
            )
            
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                
                # Clean up based on model
                text = text.replace("Assistant:", "").strip()
                text = text.replace("[/INST]", "").strip()
                
                # Limit response length for speed
                sentences = text.split('.')
                if len(sentences) > 2:
                    text = '.'.join(sentences[:2]) + '.'
                
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                
                llm_time = self.perf.end_timer("LLM")
                print(f" OK ({llm_time:.2f}s)")
                
                print(f"\n[AI] {text}")
                return text
                
        except Exception as e:
            print(f" ERROR: {e}")
            return None
            
    def speak_text_fast(self, text: str):
        """Natural TTS with improved quality"""
        if not text:
            return
            
        print("[TTS] Speaking...", end="", flush=True)
        
        self.perf.start_timer("TTS")
        self.is_speaking = True
        
        # Clean text for TTS (remove special chars that cause issues)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Check cache first
        if text in self.tts_cache:
            # Play from cache
            pygame.mixer.music.load(self.tts_cache[text])
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            
            tts_time = self.perf.end_timer("TTS")
            print(f" OK (cached: {tts_time:.2f}s)")
            self.is_speaking = False
            return
        
        temp_wav = None
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
            
            # Build Piper command with natural settings
            cmd = [
                'piper',
                '--model', self.voice_model,
                '--config', self.voice_config,
                '--output_file', temp_wav,
                '--length-scale', getattr(self, 'tts_speed', '0.98'),  # Voice-specific speed
                '--sentence-silence', getattr(self, 'tts_silence', '0.15'),  # Voice-specific pauses
                '--noise-scale', '0.667',  # Reduce artifacts (2/3 default)
                '--noise-w', '0.8'  # Smoother voice
            ]
            
            if self.piper_gpu_flag:
                cmd.append(self.piper_gpu_flag)
            
            # Use Popen for better control
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send text properly
            stdout, stderr = process.communicate(input=text, timeout=5)
            
            if process.returncode == 0 and os.path.exists(temp_wav):
                # Add small delay to ensure file is fully written
                time.sleep(0.05)
                
                # Load and play with fade-in
                pygame.mixer.music.load(temp_wav)
                pygame.mixer.music.set_volume(0.95)  # Slightly reduce volume to avoid clipping
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.02)
                
                # Proper cleanup
                pygame.mixer.music.stop()
                time.sleep(0.05)  # Small delay before unload
                pygame.mixer.music.unload()
                
                tts_time = self.perf.end_timer("TTS")
                print(f" OK ({tts_time:.2f}s)")
                
                # Cache if short enough
                if len(text) < 50:
                    self.tts_cache[text] = temp_wav
                    temp_wav = None  # Don't delete cached file
            else:
                print(f" ERROR: TTS failed")
                if stderr:
                    print(f"  Details: {stderr[:100]}")
                    
        except subprocess.TimeoutExpired:
            print(f" ERROR: TTS timeout")
            if process:
                process.kill()
        except Exception as e:
            print(f" ERROR: {e}")
            
        finally:
            self.is_speaking = False
            
            # Cleanup
            if temp_wav and os.path.exists(temp_wav):
                try:
                    # Add delay to avoid file lock issues
                    time.sleep(0.1)
                    os.unlink(temp_wav)
                except:
                    pass
            
    def conversation_loop(self):
        """Main loop with ultra-low latency"""
        print("\n" + "="*60)
        print("  MISTRAL 7B + FASTER-WHISPER - RTX 4090 MODE")
        print("="*60)
        print("  - ENTER: Voice input with VAD")
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
                    # Generate and speak
                    response = self.generate_response_fast(user_input)
                    if response:
                        self.speak_text_fast(response)
                        
                        # Show metrics
                        total_time = self.perf.end_timer("Total")
                        
                        print(f"\n[TIME] ", end="")
                        for name, duration in self.perf.metrics.items():
                            if name != "Total":
                                print(f"{name}: {duration:.1f}s | ", end="")
                        print(f"Total: {total_time:.1f}s")
                        
                        # Check if we hit new target
                        if total_time < 3.0:
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
            
            if total < 3.0:
                print("\n  [*] ACHIEVING ULTRA-FAST TARGET! <3s average!")
            elif total < 5.0:
                print("\n  [OK] ACHIEVING TARGET! <5s average!")

def main():
    """Main entry point"""
    try:
        print("\n[*] Initializing Mistral 7B Voice Assistant...")
        print("   RTX 4090 Optimized | Faster-Whisper STT")
        print("   Downloading model if needed...")
        
        assistant = FasterWhisperVoiceAssistant()
        assistant.conversation_loop()
    except Exception as e:
        print(f"\n[ERROR] Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()