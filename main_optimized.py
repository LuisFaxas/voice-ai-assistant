#!/usr/bin/env python3
"""
OPTIMIZED VOICE ASSISTANT - RTX 4090 with Performance Monitoring
Real-time latency tracking and response optimization
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
from typing import Optional, Dict
from datetime import datetime
from collections import deque

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Core imports
import sounddevice as sd
import whisper
from llama_cpp import Llama

# Initialize pygame with optimized settings
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=256)  # Smaller buffer

class PerformanceMonitor:
    """Lightweight performance monitoring without adding latency"""
    
    def __init__(self):
        self.metrics = {}
        self.history = deque(maxlen=10)  # Keep last 10 interactions
        self.current_timers = {}
        
    def start_timer(self, name: str):
        """Start a named timer"""
        self.current_timers[name] = time.perf_counter()
        
    def end_timer(self, name: str) -> float:
        """End a timer and return duration"""
        if name in self.current_timers:
            duration = time.perf_counter() - self.current_timers[name]
            self.metrics[name] = duration
            del self.current_timers[name]
            return duration
        return 0.0
    
    def get_last_metrics(self) -> Dict[str, float]:
        """Get the last recorded metrics"""
        return self.metrics.copy()
    
    def add_to_history(self):
        """Add current metrics to history"""
        if self.metrics:
            self.history.append(self.metrics.copy())
            
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from history"""
        if not self.history:
            return {}
        
        avg_metrics = {}
        for metric_name in self.history[0].keys():
            values = [m.get(metric_name, 0) for m in self.history]
            avg_metrics[metric_name] = sum(values) / len(values)
        return avg_metrics
    
    def display_inline(self):
        """Display metrics inline without blocking"""
        if self.metrics:
            # Build compact metrics string
            parts = []
            total = 0
            for name, duration in self.metrics.items():
                parts.append(f"{name}: {duration:.2f}s")
                total += duration
            parts.append(f"Total: {total:.2f}s")
            
            # Print on same line
            metrics_str = " | ".join(parts)
            print(f"\r⏱️  {metrics_str}", end="", flush=True)
            print()  # New line after metrics

class VoiceAssistant:
    """RTX 4090 Optimized Voice Assistant with Performance Monitoring"""
    
    def __init__(self):
        """Initialize the voice assistant with all components"""
        self.show_banner()
        
        # Performance monitoring
        self.perf = PerformanceMonitor()
        
        # Check GPU status
        self.device = self._check_gpu()
        
        # State management
        self.is_running = False
        self.is_speaking = False
        self.conversation_history = []
        
        # Performance settings
        self.enable_metrics = True
        self.verbose_metrics = False
        
        # Initialize all components with timing
        init_start = time.perf_counter()
        
        self._init_audio()
        self._init_stt()
        self._init_llm()
        self._init_tts()
        
        init_time = time.perf_counter() - init_start
        print(f"\n✅ All systems initialized in {init_time:.2f} seconds\n")
        
    def show_banner(self):
        """Display startup banner"""
        print("\n" + "="*60)
        print("  🚀 RTX 4090 VOICE ASSISTANT - OPTIMIZED")
        print("  Real-time Performance Monitoring | Low Latency")
        print("="*60)
        
    def _check_gpu(self) -> str:
        """Check GPU availability and return device type"""
        print("\n🔍 Checking GPU status...")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ VRAM: {gpu_memory:.1f} GB")
            
            if "4090" in gpu_name:
                print("  ⚡ RTX 4090 detected - Maximum performance!")
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            return "cuda"
        else:
            print("  ⚠️  No GPU found - using CPU (slower)")
            return "cpu"
            
    def _init_audio(self):
        """Initialize audio settings"""
        print("\n🎤 Setting up audio...")
        
        self.sample_rate = 16000
        self.channels = 1
        self.recording_duration = 2.5  # Reduced from 3.0 for faster response
        
        # Test microphone
        try:
            test = sd.rec(
                int(0.1 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()
            print("  ✅ Microphone ready")
        except Exception as e:
            print(f"  ❌ Microphone error: {e}")
            sys.exit(1)
            
    def _init_stt(self):
        """Initialize Whisper STT with GPU acceleration"""
        print("\n🎧 Loading Whisper STT...")
        
        start = time.perf_counter()
        
        # Use 'base' for faster processing, 'small' for better accuracy
        model_size = "base" if self.device == "cuda" else "base"
        
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        load_time = time.perf_counter() - start
        print(f"  ✅ Whisper '{model_size}' loaded on {self.device.upper()} in {load_time:.2f}s")
        
    def _init_llm(self):
        """Initialize LLM with proper context size"""
        print("\n🧠 Loading Language Model...")
        
        start = time.perf_counter()
        
        model_path = Path("models/phi-2.Q5_K_M.gguf")
        
        if not model_path.exists():
            print(f"  ❌ Model not found: {model_path}")
            print("  Run: python download_model.py")
            sys.exit(1)
        
        # Optimized settings for speed
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_gpu_layers=-1 if self.device == "cuda" else 0,
            n_batch=1024,  # Increased batch size
            n_threads=12,   # More threads
            use_mmap=True,  # Memory mapping for faster loading
            use_mlock=False,
            verbose=False
        )
        
        load_time = time.perf_counter() - start
        print(f"  ✅ Phi-2 loaded with {'GPU' if self.device == 'cuda' else 'CPU'} in {load_time:.2f}s")
        
        # Optimized prompt for faster responses
        self.system_prompt = """You are Assistant. Reply in 1 sentence only. Be brief and helpful."""
        
    def _init_tts(self):
        """Initialize Piper TTS with enhanced voice settings"""
        print("\n🗣️ Setting up Text-to-Speech...")
        
        voices_dir = Path("models/piper_voices")
        
        # Check for voice models
        voice_options = [
            ("en_US-ryan-high", "Ryan (High Quality Male)"),
            ("en_US-amy-medium", "Amy (Natural Female)"),
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
            print("  ⚠️  Downloading voice models...")
            self._download_voices()
            
            # Check again after download
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
            print("  ❌ Could not load voice models")
            sys.exit(1)
            
        # Set GPU flag for Piper
        self.piper_gpu_flag = "--cuda" if self.device == "cuda" else ""
        
    def _download_voices(self):
        """Download voice models if missing"""
        try:
            download_script = Path("download_best_voices.py")
            if download_script.exists():
                result = subprocess.run(
                    [sys.executable, str(download_script)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    print(f"  ⚠️  Download failed: {result.stderr}")
        except Exception as e:
            print(f"  ⚠️  Could not download voices: {e}")
            
    def record_audio(self) -> Optional[np.ndarray]:
        """Record audio from microphone with visual feedback"""
        print("\n🎤 Listening...", end="", flush=True)
        
        if self.enable_metrics:
            self.perf.start_timer("Recording")
        
        # Record audio
        recording = sd.rec(
            int(self.recording_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()
        
        if self.enable_metrics:
            rec_time = self.perf.end_timer("Recording")
            print(f" ✓ ({rec_time:.2f}s)")
        else:
            print(" ✓")
        
        # Check if we got meaningful audio
        audio_data = recording.flatten()
        max_amplitude = np.max(np.abs(audio_data))
        
        if max_amplitude < 0.001:
            print(" (no audio detected)")
            return None
            
        return audio_data
        
    def transcribe_audio(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio to text using Whisper"""
        if audio is None:
            return None
            
        print("📝 Transcribing...", end="", flush=True)
        
        if self.enable_metrics:
            self.perf.start_timer("Transcription")
        
        try:
            # Optimized transcription settings
            result = self.whisper_model.transcribe(
                audio,
                language="en",
                fp16=(self.device == "cuda"),
                beam_size=1,  # Faster with beam_size=1
                best_of=1     # Skip multiple attempts
            )
            
            text = result["text"].strip()
            
            if self.enable_metrics:
                trans_time = self.perf.end_timer("Transcription")
                print(f" ✓ ({trans_time:.2f}s)")
            else:
                print(" ✓")
            
            if text:
                print(f"\n👤 You: {text}")
                return text
            else:
                print(" (no speech detected)")
                return None
                
        except Exception as e:
            print(f" ❌ Error: {e}")
            return None
            
    def generate_response(self, user_input: str) -> Optional[str]:
        """Generate LLM response with improved formatting"""
        if not user_input:
            return None
            
        print("🤔 Thinking...", end="", flush=True)
        
        if self.enable_metrics:
            self.perf.start_timer("LLM")
        
        # Build minimal context for speed
        prompt = f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        try:
            response = self.llm(
                prompt,
                max_tokens=40,  # Reduced for faster response
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["User:", "\n\n", "\n"],
                echo=False
            )
            
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                
                # Ensure proper capitalization
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                
                # Add to history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': text
                })
                
                # Keep only last 3 exchanges
                self.conversation_history = self.conversation_history[-3:]
                
                if self.enable_metrics:
                    llm_time = self.perf.end_timer("LLM")
                    print(f" ✓ ({llm_time:.2f}s)")
                else:
                    print(" ✓")
                
                print(f"\n🤖 AI: {text}")
                return text
            else:
                print(" (no response)")
                return None
                
        except Exception as e:
            print(f" ❌ Error: {e}")
            return None
            
    def speak_text(self, text: str):
        """Convert text to speech with enhanced quality and proper file handling"""
        if not text:
            return
            
        print("🔊 Speaking...", end="", flush=True)
        
        if self.enable_metrics:
            self.perf.start_timer("TTS")
        
        self.is_speaking = True
        
        temp_txt = None
        temp_wav = None
        
        try:
            # Create temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                temp_txt = f.name
                # Clean text for better pronunciation
                clean_text = text.replace('"', '').replace("'", '')
                f.write(clean_text)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
            
            # Build Piper command with optimized parameters
            cmd = [
                'piper',
                '--model', self.voice_model,
                '--config', self.voice_config,
                '--input_file', temp_txt,
                '--output_file', temp_wav,
                '--sentence-silence', '0.1',  # Reduced pause
                '--length-scale', '1.0',       # Normal speed
                '--noise-scale', '0.667',
            ]
            
            if self.piper_gpu_flag:
                cmd.append(self.piper_gpu_flag)
            
            # Generate speech
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and os.path.exists(temp_wav):
                # Play audio with proper cleanup
                self._play_audio_safely(temp_wav)
                
                if self.enable_metrics:
                    tts_time = self.perf.end_timer("TTS")
                    print(f" ✓ ({tts_time:.2f}s)")
                else:
                    print(" ✓")
            else:
                raise Exception(f"Piper error: {result.stderr}")
                
        except Exception as e:
            print(f" ❌ TTS Error: {e}")
            print(f"\n📝 Text: {text}")
            
        finally:
            # Clean up temp files
            self.is_speaking = False
            
            # Clean up with retries
            for temp_file in [temp_txt, temp_wav]:
                if temp_file and os.path.exists(temp_file):
                    for _ in range(3):
                        try:
                            os.unlink(temp_file)
                            break
                        except:
                            time.sleep(0.05)
            
            time.sleep(0.2)  # Reduced pause
            
    def _play_audio_safely(self, wav_path: str):
        """Play audio file with proper resource cleanup"""
        try:
            # Load and play
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            
            # CRITICAL: Unload to release file handle
            pygame.mixer.music.unload()
            
            # Extra safety delay
            time.sleep(0.05)
            
        except Exception as e:
            print(f"\n⚠️  Playback error: {e}")
            # Try alternative playback method
            try:
                os.system(f'powershell -c (New-Object Media.SoundPlayer "{wav_path}").PlaySync()')
            except:
                pass
    
    def display_performance_summary(self):
        """Display performance metrics summary"""
        if self.enable_metrics and self.perf.history:
            avg = self.perf.get_average_metrics()
            if avg:
                print("\n📊 Performance Summary (last 10 interactions):")
                print("  " + "-"*40)
                total_avg = 0
                for name, duration in avg.items():
                    print(f"  {name:15s}: {duration:.2f}s avg")
                    total_avg += duration
                print("  " + "-"*40)
                print(f"  {'Total':15s}: {total_avg:.2f}s avg")
                print()
                
    def conversation_loop(self):
        """Main conversation loop with improved UX and metrics"""
        print("\n" + "="*60)
        print("  💬 CONVERSATION MODE - PERFORMANCE MONITORING ENABLED")
        print("="*60)
        print("  • Press ENTER to speak (2.5 seconds recording)")
        print("  • Type your message to skip voice input")
        print("  • Commands: 'quit', 'metrics', 'verbose'")
        print("="*60)
        
        self.is_running = True
        interaction_count = 0
        
        while self.is_running:
            try:
                # Reset metrics for this interaction
                self.perf.metrics = {}
                
                # Get user input
                user_action = input("\n[ENTER=speak, or type]: ").strip()
                
                # Handle commands
                if user_action.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    self.display_performance_summary()
                    break
                elif user_action.lower() == 'metrics':
                    self.display_performance_summary()
                    continue
                elif user_action.lower() == 'verbose':
                    self.verbose_metrics = not self.verbose_metrics
                    print(f"Verbose metrics: {'ON' if self.verbose_metrics else 'OFF'}")
                    continue
                    
                # Start total timer
                if self.enable_metrics:
                    self.perf.start_timer("Total")
                
                # Determine input method
                if user_action:
                    # Text input
                    user_input = user_action
                    print(f"\n👤 You: {user_input}")
                else:
                    # Voice input
                    audio = self.record_audio()
                    user_input = self.transcribe_audio(audio)
                    
                if user_input:
                    # Generate and speak response
                    response = self.generate_response(user_input)
                    if response:
                        self.speak_text(response)
                        
                        # End total timer and display metrics
                        if self.enable_metrics:
                            total_time = self.perf.end_timer("Total")
                            
                            # Display inline metrics
                            print(f"\n⏱️  Latency: ", end="")
                            
                            # Show individual times
                            metrics_parts = []
                            for name, duration in self.perf.metrics.items():
                                if name != "Total":
                                    metrics_parts.append(f"{name}: {duration:.1f}s")
                            
                            print(" | ".join(metrics_parts), end="")
                            print(f" | Total: {total_time:.1f}s")
                            
                            # Add to history
                            self.perf.add_to_history()
                            
                            interaction_count += 1
                            if interaction_count % 5 == 0:
                                # Show summary every 5 interactions
                                self.display_performance_summary()
                        
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                self.display_performance_summary()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
                
        print("\n🔌 Shutting down...")
        self.is_running = False
        
def main():
    """Main entry point"""
    try:
        # Check for required packages
        required = ['pygame', 'whisper', 'torch', 'llama_cpp', 'sounddevice']
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"\n⚠️  Missing packages: {', '.join(missing)}")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)
        
        # Start the assistant
        assistant = VoiceAssistant()
        assistant.conversation_loop()
        
    except Exception as e:
        print(f"\n💥 Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()