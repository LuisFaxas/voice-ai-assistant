#!/usr/bin/env python3
"""Test TTS to diagnose speaking issues"""

import pyttsx3
import time

def test_basic_tts():
    """Test basic TTS functionality"""
    print("\n=== BASIC TTS TEST ===")
    
    engine = pyttsx3.init()
    engine.say("Test one. Hello world.")
    engine.runAndWait()
    print("✓ Test 1 complete")
    
    time.sleep(1)
    
    engine.say("Test two. This is the second test.")
    engine.runAndWait()
    print("✓ Test 2 complete")
    
    time.sleep(1)
    
    engine.say("Test three. Final test.")
    engine.runAndWait()
    print("✓ Test 3 complete")
    
    del engine

def test_reinit_tts():
    """Test re-initializing TTS each time"""
    print("\n=== REINIT TTS TEST ===")
    
    for i in range(3):
        engine = pyttsx3.init()
        engine.say(f"Test {i+1}. This is test number {i+1}.")
        engine.runAndWait()
        engine.stop()
        del engine
        print(f"✓ Test {i+1} complete")
        time.sleep(1)

def test_voice_selection():
    """Test different voices"""
    print("\n=== VOICE SELECTION TEST ===")
    
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    print(f"Found {len(voices)} voices:")
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name}")
    
    # Test each voice
    for i, voice in enumerate(voices[:3]):  # Test first 3 voices
        print(f"\nTesting voice {i}: {voice.name}")
        engine.setProperty('voice', voice.id)
        engine.say(f"This is voice {i}. {voice.name}")
        engine.runAndWait()
        time.sleep(1)
    
    del engine

def test_continuous_conversation():
    """Simulate conversation loop"""
    print("\n=== CONVERSATION SIMULATION ===")
    
    messages = [
        "Hello, how are you?",
        "I can help you with many things.",
        "What would you like to know?",
        "That's interesting.",
        "Goodbye!"
    ]
    
    for i, msg in enumerate(messages):
        print(f"\n[Message {i+1}]: {msg}")
        
        # Create new engine each time (like the fix)
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        
        engine.say(msg)
        engine.runAndWait()
        engine.stop()
        del engine
        
        print("✓ Spoken")
        time.sleep(1)

def main():
    print("="*50)
    print("  TTS DIAGNOSTIC TEST")
    print("="*50)
    
    tests = [
        ("Basic TTS", test_basic_tts),
        ("Re-init TTS", test_reinit_tts),
        ("Voice Selection", test_voice_selection),
        ("Conversation Simulation", test_continuous_conversation)
    ]
    
    for name, test_func in tests:
        response = input(f"\nRun {name} test? (y/n): ")
        if response.lower() == 'y':
            try:
                test_func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    
    print("\n=== TESTS COMPLETE ===")
    print("\nIf TTS worked in 'Re-init TTS' and 'Conversation Simulation',")
    print("then app_fixed.py should work correctly.")

if __name__ == "__main__":
    main()