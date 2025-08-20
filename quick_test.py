#!/usr/bin/env python3
"""Quick test to verify all components work"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("\n=== QUICK COMPONENT TEST ===\n")

# Test 1: Audio
print("1. Testing Audio...")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"   ✓ Found {len(devices)} audio devices")
except Exception as e:
    print(f"   ✗ Audio error: {e}")

# Test 2: Whisper
print("2. Testing Whisper STT...")
try:
    import whisper
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   ✓ Whisper ready (using {device})")
except Exception as e:
    print(f"   ✗ Whisper error: {e}")

# Test 3: LLM
print("3. Testing LLM...")
try:
    from llama_cpp import Llama
    model_path = "models/phi-2.Q5_K_M.gguf"
    if Path(model_path).exists():
        print("   ✓ LLM model found")
        
        # Quick test
        print("   Testing LLM response...")
        llm = Llama(model_path=model_path, n_ctx=512, verbose=False, n_gpu_layers=0)
        response = llm("Hello", max_tokens=10, echo=False)
        if response:
            print("   ✓ LLM responding")
    else:
        print(f"   ✗ Model not found: {model_path}")
except Exception as e:
    print(f"   ✗ LLM error: {e}")

# Test 4: TTS
print("4. Testing TTS...")
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("   ✓ TTS ready (Windows SAPI)")
    
    # Test speech
    response = input("   Test TTS? (y/n): ")
    if response.lower() == 'y':
        engine.say("Hello, voice assistant is working")
        engine.runAndWait()
        print("   ✓ TTS working")
except Exception as e:
    print(f"   ✗ TTS error: {e}")

print("\n=== TEST COMPLETE ===")
print("\nIf all components show ✓, run:")
print("  python working_app.py")
print("\nOr double-click:")
print("  run_working.bat")