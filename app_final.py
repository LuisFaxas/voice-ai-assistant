#!/usr/bin/env python3
"""
FINAL FIXED Voice Assistant - Reliable TTS with Better Voice
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Core imports
import sounddevice as sd
import whisper
import torch
import pyttsx3
from llama_cpp import Llama

class VoiceAssistant:
    def __init__(self):
        print("\n[INITIALIZING VOICE ASSISTANT]\n")
        
        # State management
        self.is_running = False
        self.is_speaking = False
        
        # Initialize all components
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
        
        # Test microphone
        try:
            test = sd.rec(int(0.1 * self.sample_rate), 
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
            
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            verbose=False
        )
        print("  ✓ LLM loaded")
        
        self.system_prompt = "You are a helpful assistant. Keep responses brief and clear in 1-2 sentences."
        
    def _init_tts(self):
        """Initialize TTS with best available voice"""
        print("• Setting up TTS...")
        
        # Don't store engine as instance variable - create fresh each time
        temp_engine = pyttsx3.init('sapi5')
        voices = temp_engine.getProperty('voices')
        
        if voices:
            print(f"  Found {len(voices)} voices:")
            
            # Find best voice
            best_voice = None
            best_voice_name = None
            
            # Preferred voices in order
            preferred = ['Microsoft David', 'Microsoft Zira', 'Microsoft Hazel', 'Microsoft Mark']
            
            for i, voice in enumerate(voices):
                print(f"    {i}: {voice.name}")
                
                # Check for preferred voices
                for pref in preferred:
                    if pref.lower() in voice.name.lower():
                        if not best_voice:
                            best_voice = voice.id
                            best_voice_name = voice.name
                        break
            
            # If no preferred voice found, use second voice if available
            if not best_voice and len(voices) > 1:
                best_voice = voices[1].id
                best_voice_name = voices[1].name
            elif not best_voice:
                best_voice = voices[0].id
                best_voice_name = voices[0].name
            
            # Store selected voice ID for later use
            self.selected_voice_id = best_voice
            print(f"  ✓ Selected voice: {best_voice_name}")
        else:
            self.selected_voice_id = None
            print("  ⚠ No voices found, using default")
        
        # Clean up temp engine
        temp_engine.stop()
        del temp_engine
        
        print("  ✓ TTS ready (offline)")
        
    def record_audio(self):
        """Record audio from microphone"""
        print("\n[Listening...] ", end="", flush=True)
        
        # Record for 3 seconds
        duration = 3.0
        recording = sd.rec(int(duration * self.sample_rate),
                          samplerate=self.sample_rate,
                          channels=self.channels,
                          dtype=np.float32)
        sd.wait()
        
        # Check amplitude
        max_amplitude = np.max(np.abs(recording))
        if max_amplitude < 0.001:
            print("(no audio detected)")
            return None
            
        print("✓")
        return recording.flatten()
        
    def transcribe_audio(self, audio):
        """Convert audio to text"""
        if audio is None:
            return None
            
        print("[Transcribing...] ", end="", flush=True)
        
        try:
            result = self.whisper_model.transcribe(
                audio,
                language="en",
                fp16=False
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
        
        prompt = f"""System: {self.system_prompt}

User: {user_input}
Assistant:"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=50,  # Keep responses short
                temperature=0.7,
                stop=["User:", "\n\n"],
                echo=False
            )
            
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
        """Text to speech with fresh engine each time"""
        if not text:
            return
            
        print("[Speaking...] ", end="", flush=True)
        self.is_speaking = True
        
        try:
            # Create fresh engine for each speech
            engine = pyttsx3.init('sapi5')
            
            # Set selected voice
            if self.selected_voice_id:
                engine.setProperty('voice', self.selected_voice_id)
            
            # Configure properties
            engine.setProperty('rate', 180)  # Natural speed
            engine.setProperty('volume', 1.0)  # Full volume
            
            # Speak the text
            engine.say(text)
            engine.runAndWait()
            
            # Properly clean up
            engine.stop()
            del engine
            
            print("✓")
            
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"\n[TTS Failed - Text]: {text}")
            
        self.is_speaking = False
        time.sleep(0.5)  # Prevent feedback
        
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
                # Get user action
                user_action = input("\n[Press ENTER to speak, or type message]: ").strip()
                
                if user_action.lower() == 'quit':
                    break
                    
                # Handle text or voice input
                if user_action:
                    user_input = user_action
                    print(f"\n>>> You typed: {user_input}\n")
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
                break
            except Exception as e:
                print(f"\n[Error: {e}]")
                continue
                
        print("\n[Shutting down...]")
        self.is_running = False

def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("  VOICE ASSISTANT (FINAL)")
    print("  Reliable • Offline • Better Voice")
    print("="*50)
    
    try:
        assistant = VoiceAssistant()
        assistant.conversation_loop()
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()