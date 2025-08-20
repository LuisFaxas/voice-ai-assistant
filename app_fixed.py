#!/usr/bin/env python3
"""
FIXED Voice Assistant - Reliable TTS and Better Voice
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
import queue
import threading

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Minimal imports - only what we need
import sounddevice as sd
import whisper
import torch
import pyttsx3  # Offline TTS that actually works

# Import our local LLM
from llama_cpp import Llama

class WorkingVoiceAssistant:
    def __init__(self):
        print("\n[INITIALIZING VOICE ASSISTANT]\n")
        
        # State flags
        self.is_running = False
        self.is_listening = False
        self.is_speaking = False
        
        # Initialize components
        self._init_audio()
        self._init_stt()
        self._init_llm()
        self._init_tts()
        
        print("\n[INITIALIZATION COMPLETE]\n")
        
    def _init_audio(self):
        """Initialize audio settings"""
        print("• Setting up audio...")
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 2.0  # Record 2-second chunks
        
        # Test microphone
        try:
            test_recording = sd.rec(int(0.1 * self.sample_rate), 
                                   samplerate=self.sample_rate, 
                                   channels=self.channels)
            sd.wait()
            print("  ✓ Microphone working")
        except Exception as e:
            print(f"  ✗ Microphone error: {e}")
            
    def _init_stt(self):
        """Initialize Whisper STT"""
        print("• Loading Whisper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=device)
        print(f"  ✓ Whisper loaded on {device}")
        
    def _init_llm(self):
        """Initialize Local LLM"""
        print("• Loading LLM...")
        model_path = "models/phi-2.Q5_K_M.gguf"
        
        if not Path(model_path).exists():
            print(f"  ✗ Model not found: {model_path}")
            print("  Please run: python download_model.py")
            sys.exit(1)
            
        # Load with simple settings
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Use smaller context to avoid overflow
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            verbose=False
        )
        print("  ✓ LLM loaded")
        
        # Simple prompt
        self.system_prompt = "You are a helpful assistant. Give brief, clear responses in 1-2 sentences maximum."
        
    def _init_tts(self):
        """Initialize offline TTS with better voice"""
        print("• Setting up TTS...")
        
        # Create new engine for each session to avoid issues
        self.tts_engine = pyttsx3.init('sapi5')  # Force Windows SAPI5
        
        # List all available voices
        voices = self.tts_engine.getProperty('voices')
        
        if voices:
            print(f"  Found {len(voices)} voices")
            
            # Try to find a better voice (Microsoft David, Zira, or Hazel)
            preferred_voices = ['David', 'Zira', 'Hazel', 'Mark']
            selected_voice = None
            
            for voice in voices:
                voice_name = voice.name.lower()
                print(f"  - {voice.name}")
                for preferred in preferred_voices:
                    if preferred.lower() in voice_name:
                        selected_voice = voice
                        break
                if selected_voice:
                    break
            
            # Use selected voice or default to second voice if available (usually better than first)
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice.id)
                print(f"  ✓ Using voice: {selected_voice.name}")
            elif len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)
                print(f"  ✓ Using voice: {voices[1].name}")
            else:
                self.tts_engine.setProperty('voice', voices[0].id)
                print(f"  ✓ Using voice: {voices[0].name}")
        
        # Set speech properties
        self.tts_engine.setProperty('rate', 180)  # Slightly faster for natural speech
        self.tts_engine.setProperty('volume', 1.0)  # Max volume
        
        # Test TTS
        try:
            self.tts_engine.say(" ")
            self.tts_engine.runAndWait()
            print("  ✓ TTS ready (offline)")
        except:
            print("  ⚠ TTS initialization warning")
        
    def record_audio(self):
        """Record audio from microphone"""
        print("\n[Listening...] ", end="", flush=True)
        
        # Record audio
        duration = 3.0  # Record for 3 seconds
        recording = sd.rec(int(duration * self.sample_rate),
                          samplerate=self.sample_rate,
                          channels=self.channels,
                          dtype=np.float32)
        sd.wait()
        
        # Check if we got audio
        max_amplitude = np.max(np.abs(recording))
        if max_amplitude < 0.001:
            print("(no audio detected)")
            return None
            
        print("✓")
        return recording.flatten()
        
    def transcribe_audio(self, audio):
        """Convert audio to text using Whisper"""
        if audio is None:
            return None
            
        print("[Transcribing...] ", end="", flush=True)
        
        try:
            # Use Whisper to transcribe
            result = self.whisper_model.transcribe(
                audio,
                language="en",
                fp16=False  # Disable FP16 for stability
            )
            
            text = result["text"].strip()
            
            if text:
                print("✓")
                print(f"\n>>> You said: {text}\n")
                return text
            else:
                print("(no speech detected)")
                return None
                
        except Exception as e:
            print(f"Error: {e}")
            return None
            
    def generate_response(self, user_input):
        """Generate LLM response"""
        if not user_input:
            return None
            
        print("[Thinking...] ", end="", flush=True)
        
        # Build simple prompt
        prompt = f"""System: {self.system_prompt}

User: {user_input}
Assistant:"""
        
        try:
            # Generate response with shorter max_tokens for brevity
            response = self.llm(
                prompt,
                max_tokens=50,  # Shorter responses
                temperature=0.7,
                stop=["User:", "\n\n"],
                echo=False
            )
            
            # Extract text
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                print("✓")
                print(f"\n<<< AI: {text}\n")
                return text
            else:
                print("(no response)")
                return None
                
        except Exception as e:
            print(f"Error: {e}")
            return None
            
    def speak_text(self, text):
        """Convert text to speech - FIXED VERSION"""
        if not text:
            return
            
        print("[Speaking...] ", end="", flush=True)
        self.is_speaking = True
        
        try:
            # Re-initialize engine each time to avoid state issues
            engine = pyttsx3.init('sapi5')
            
            # Set voice properties
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)  # Use second voice
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            
            # Speak the text
            engine.say(text)
            engine.runAndWait()
            
            # Clean up
            engine.stop()
            del engine
            
            print("✓")
            
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback: print the text so user can read it
            print(f"\n[TTS Failed - Text output]: {text}")
            
        self.is_speaking = False
        
        # Pause to avoid feedback
        time.sleep(0.5)
        
    def conversation_loop(self):
        """Main conversation loop"""
        print("\n" + "="*50)
        print("  VOICE ASSISTANT READY")
        print("="*50)
        print("\nPress ENTER to speak, or type 'quit' to exit")
        print("-"*50)
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Wait for user input
                user_action = input("\n[Press ENTER to speak, or type message]: ").strip()
                
                if user_action.lower() == 'quit':
                    break
                    
                # If user typed something, use it as input
                if user_action:
                    user_input = user_action
                    print(f"\n>>> You typed: {user_input}\n")
                else:
                    # Record and transcribe audio
                    audio = self.record_audio()
                    user_input = self.transcribe_audio(audio)
                    
                if user_input:
                    # Generate response
                    response = self.generate_response(user_input)
                    
                    if response:
                        # Speak response
                        self.speak_text(response)
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Error in conversation loop: {e}]")
                continue
                
        print("\n[Shutting down...]")
        self.is_running = False

def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("  VOICE ASSISTANT (FIXED)")
    print("  Simple • Reliable • Offline")
    print("="*50)
    
    try:
        # Create and run assistant
        assistant = WorkingVoiceAssistant()
        assistant.conversation_loop()
        
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()