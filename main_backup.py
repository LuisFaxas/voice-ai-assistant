#!/usr/bin/env python3
"""
MAIN VOICE ASSISTANT - RTX 4090 Optimized
Polished single-file implementation with all fixes
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
from typing import Optional

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Core imports
import sounddevice as sd
import whisper
from llama_cpp import Llama

# Initialize pygame for audio playback
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

class VoiceAssistant:
    """RTX 4090 Optimized Voice Assistant with Human-like TTS"""
    
    def __init__(self):
        """Initialize the voice assistant with all components"""
        self.show_banner()
        
        # Check GPU status
        self.device = self._check_gpu()
        
        # State management
        self.is_running = False
        self.is_speaking = False
        self.conversation_history = []
        
        # Initialize all components
        self._init_audio()
        self._init_stt()
        self._init_llm()
        self._init_tts()
        
        print("\n✅ All systems initialized successfully!\n")
        
    def show_banner(self):
        """Display startup banner"""
        print("\n" + "="*60)
        print("  🚀 RTX 4090 VOICE ASSISTANT")
        print("  Human-like voice | GPU Accelerated | 100% Offline")
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
            
            return "cuda"
        else:
            print("  ⚠️  No GPU found - using CPU (slower)")
            return "cpu"
            
    def _init_audio(self):
        """Initialize audio settings"""
        print("\n🎤 Setting up audio...")
        
        self.sample_rate = 16000
        self.channels = 1
        self.recording_duration = 3.0  # seconds
        
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
        
        # Use 'small' model for better accuracy on RTX 4090
        model_size = "small" if self.device == "cuda" else "base"
        
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        print(f"  ✅ Whisper '{model_size}' loaded on {self.device.upper()}")
        
    def _init_llm(self):
        """Initialize LLM with proper context size"""
        print("\n🧠 Loading Language Model...")
        
        model_path = Path("models/phi-2.Q5_K_M.gguf")
        
        if not model_path.exists():
            print(f"  ❌ Model not found: {model_path}")
            print("  Run: python download_model.py")
            sys.exit(1)
        
        # Fixed context size to prevent overflow
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,  # Match training context (was 4096, causing warning)
            n_gpu_layers=-1 if self.device == "cuda" else 0,
            n_batch=512,
            n_threads=8,
            verbose=False
        )
        
        print(f"  ✅ Phi-2 loaded with {'GPU' if self.device == 'cuda' else 'CPU'} acceleration")
        
        # Improved prompt for better responses
        self.system_prompt = """You are a helpful AI assistant named Assistant. 
Important rules:
- Respond in 1-2 sentences maximum
- Use proper capitalization and punctuation
- Be conversational and friendly
- Never say "I am not capable" or similar - just provide helpful answers
- Keep responses natural and human-like"""
        
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
        
        # Record audio
        recording = sd.rec(
            int(self.recording_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()
        
        # Check if we got meaningful audio
        audio_data = recording.flatten()
        max_amplitude = np.max(np.abs(audio_data))
        
        if max_amplitude < 0.001:
            print(" (no audio detected)")
            return None
            
        print(" ✓")
        return audio_data
        
    def transcribe_audio(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio to text using Whisper"""
        if audio is None:
            return None
            
        print("📝 Transcribing...", end="", flush=True)
        
        try:
            # Use FP16 on GPU for faster processing
            result = self.whisper_model.transcribe(
                audio,
                language="en",
                fp16=(self.device == "cuda")
            )
            
            text = result["text"].strip()
            
            if text:
                print(" ✓")
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
        
        # Build context with conversation history
        context = self.system_prompt + "\n\n"
        
        # Add last 2 exchanges for context
        for exchange in self.conversation_history[-2:]:
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        
        # Add current input
        prompt = context + f"User: {user_input}\nAssistant:"
        
        try:
            response = self.llm(
                prompt,
                max_tokens=60,  # Reduced for conciseness
                temperature=0.7,
                stop=["User:", "\n\n"],
                echo=False
            )
            
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                
                # Ensure proper capitalization
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                
                # Store in history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': text
                })
                
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
            
            # Build Piper command with quality parameters
            cmd = [
                'piper',
                '--model', self.voice_model,
                '--config', self.voice_config,
                '--input_file', temp_txt,
                '--output_file', temp_wav,
                '--sentence-silence', '0.2',  # Pause between sentences
                '--length-scale', '1.15',      # Slightly slower for clarity
                '--noise-scale', '0.667',      # Natural variation
            ]
            
            if self.piper_gpu_flag:
                cmd.append(self.piper_gpu_flag)
            
            # Generate speech
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and os.path.exists(temp_wav):
                # Play audio with proper cleanup
                self._play_audio_safely(temp_wav)
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
                            time.sleep(0.1)
            
            time.sleep(0.3)  # Prevent audio feedback
            
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
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n⚠️  Playback error: {e}")
            # Try alternative playback method
            try:
                os.system(f'powershell -c (New-Object Media.SoundPlayer "{wav_path}").PlaySync()')
            except:
                pass
                
    def conversation_loop(self):
        """Main conversation loop with improved UX"""
        print("\n" + "="*60)
        print("  💬 CONVERSATION MODE")
        print("="*60)
        print("  • Press ENTER to speak (3 seconds recording)")
        print("  • Type your message to skip voice input")
        print("  • Type 'quit' to exit")
        print("="*60)
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Get user input
                user_action = input("\n[ENTER=speak, or type]: ").strip()
                
                if user_action.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                    
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
                        
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
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