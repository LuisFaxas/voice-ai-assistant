#!/usr/bin/env python3
"""
Automatically download the best Piper TTS voice models
"""

import os
import requests
from pathlib import Path

def download_file(url, dest_path):
    """Download a file with progress"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"  [OK] Already exists: {dest_path.name}")
        return True
    
    print(f"  Downloading: {dest_path.name}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"    Progress: {progress:.1f}%", end='\r')
        
        print(f"  [OK] Downloaded: {dest_path.name}    ")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    print("="*60)
    print("  DOWNLOADING BEST PIPER TTS VOICES")
    print("="*60)
    
    # Download the two best voices - Amy (female) and Ryan (male)
    voices = [
        {
            'name': 'en_US-amy-medium',
            'desc': 'Amy - Natural female voice',
            'model': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx',
            'config': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json'
        },
        {
            'name': 'en_US-ryan-high',
            'desc': 'Ryan - Natural male voice (HIGH QUALITY)',
            'model': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx',
            'config': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json'
        }
    ]
    
    models_dir = Path("models/piper_voices")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for voice in voices:
        print(f"\n[Downloading] {voice['desc']}")
        
        # Download model
        model_path = models_dir / f"{voice['name']}.onnx"
        download_file(voice['model'], model_path)
        
        # Download config
        config_path = models_dir / f"{voice['name']}.onnx.json"
        download_file(voice['config'], config_path)
    
    print(f"\n{'='*60}")
    print("Voice models downloaded successfully!")
    print(f"Location: {models_dir.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()