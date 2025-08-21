#!/usr/bin/env python3
"""
Test and optimize TTS settings for natural voice
"""

import os
import sys
import subprocess
import tempfile
import time
import pygame
from pathlib import Path

# Initialize pygame
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

def test_piper_settings(text, voice_model, voice_config, settings_name, **kwargs):
    """Test different Piper TTS settings"""
    print(f"\nTesting: {settings_name}")
    print(f"Settings: {kwargs}")
    
    temp_wav = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_wav = f.name
        
        # Build command with different settings
        cmd = ['piper']
        cmd.extend(['--model', voice_model])
        cmd.extend(['--config', voice_config])
        cmd.extend(['--output_file', temp_wav])
        
        # Add optional settings
        if 'length_scale' in kwargs:
            cmd.extend(['--length-scale', str(kwargs['length_scale'])])
        if 'sentence_silence' in kwargs:
            cmd.extend(['--sentence-silence', str(kwargs['sentence_silence'])])
        if 'noise_scale' in kwargs:
            cmd.extend(['--noise-scale', str(kwargs['noise_scale'])])
        if 'noise_w' in kwargs:
            cmd.extend(['--noise-w', str(kwargs['noise_w'])])
            
        # Run Piper
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=text, timeout=5)
        
        if process.returncode == 0 and os.path.exists(temp_wav):
            # Get file size
            file_size = os.path.getsize(temp_wav) / 1024  # KB
            print(f"Generated: {temp_wav} ({file_size:.1f} KB)")
            
            # Play the audio
            pygame.mixer.music.load(temp_wav)
            pygame.mixer.music.play()
            
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            duration = time.time() - start_time
            
            pygame.mixer.music.unload()
            print(f"Duration: {duration:.2f}s")
            
            # Wait a bit between tests
            time.sleep(0.5)
            
            return True
        else:
            print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False
    finally:
        # Cleanup
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except:
                pass

def main():
    # Setup paths
    voices_dir = Path("models/piper_voices")
    
    # Test with Amy voice (faster)
    amy_model = str(voices_dir / "en_US-amy-medium.onnx")
    amy_config = str(voices_dir / "en_US-amy-medium.onnx.json")
    
    # Test with Ryan voice (quality)
    ryan_model = str(voices_dir / "en_US-ryan-high.onnx")
    ryan_config = str(voices_dir / "en_US-ryan-high.onnx.json")
    
    # Test text
    test_text = "Hello! How can I help you today? This is a test of the text to speech system."
    
    print("="*60)
    print("TTS QUALITY TESTING")
    print("="*60)
    
    # Test different settings with Amy (fast voice)
    print("\n--- TESTING AMY VOICE (MEDIUM) ---")
    
    # Original problematic settings
    test_piper_settings(
        test_text, amy_model, amy_config,
        "Original (problematic)",
        length_scale=0.9,
        sentence_silence=0
    )
    
    # Default settings
    test_piper_settings(
        test_text, amy_model, amy_config,
        "Default settings"
        # No extra parameters - use Piper defaults
    )
    
    # Natural speed with pauses
    test_piper_settings(
        test_text, amy_model, amy_config,
        "Natural (1.0 speed, 0.2s silence)",
        length_scale=1.0,
        sentence_silence=0.2
    )
    
    # Slightly faster but natural
    test_piper_settings(
        test_text, amy_model, amy_config,
        "Balanced (0.95 speed, 0.1s silence)",
        length_scale=0.95,
        sentence_silence=0.1
    )
    
    # With noise reduction
    test_piper_settings(
        test_text, amy_model, amy_config,
        "Clean (1.0 speed, reduced noise)",
        length_scale=1.0,
        sentence_silence=0.15,
        noise_scale=0.5,
        noise_w=0.5
    )
    
    print("\n--- TESTING RYAN VOICE (HIGH) ---")
    
    # Ryan with natural settings
    test_piper_settings(
        test_text, ryan_model, ryan_config,
        "Ryan Natural",
        length_scale=1.0,
        sentence_silence=0.2
    )
    
    # Ryan optimized
    test_piper_settings(
        test_text, ryan_model, ryan_config,
        "Ryan Optimized",
        length_scale=0.98,
        sentence_silence=0.15,
        noise_scale=0.6
    )
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("Use length_scale=1.0 (or 0.95-0.98 max)")
    print("Use sentence_silence=0.1-0.2 for natural pauses")
    print("Consider noise_scale=0.5-0.7 to reduce artifacts")
    print("="*60)

if __name__ == "__main__":
    main()