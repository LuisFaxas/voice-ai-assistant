#!/usr/bin/env python3
"""
Download additional high-quality Piper voices
"""

import os
import sys
import json
import urllib.request
import tarfile
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, destination, download_progress)
    print("\nDownload complete!")

def main():
    voices_dir = Path("models/piper_voices")
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    # High-quality voice options
    voices = {
        "lessac": {
            "name": "en_US-lessac-high",
            "description": "Lessac (Professional Narrator Quality)",
            "url_onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx",
            "url_json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
        },
        "libritts": {
            "name": "en_US-libritts_r-medium",
            "description": "LibriTTS (Natural Conversational)",
            "url_onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
            "url_json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"
        },
        "joe": {
            "name": "en_US-joe-medium",
            "description": "Joe (Clear Male Voice)",
            "url_onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx",
            "url_json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json"
        },
        "kristin": {
            "name": "en_US-kristin-medium",
            "description": "Kristin (Warm Female Voice)",
            "url_onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx",
            "url_json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium/en_US-kristin-medium.onnx.json"
        }
    }
    
    print("="*60)
    print("HIGH-QUALITY VOICE DOWNLOADER")
    print("="*60)
    print("\nAvailable voices:")
    
    for key, voice in voices.items():
        print(f"{key}: {voice['description']}")
    
    print("\nWhich voices to download?")
    print("1. Lessac (Best quality, larger)")
    print("2. LibriTTS (Natural conversation)")
    print("3. Joe (Male)")
    print("4. Kristin (Female)")
    print("5. All voices")
    print("0. Cancel")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    to_download = []
    if choice == "1":
        to_download = ["lessac"]
    elif choice == "2":
        to_download = ["libritts"]
    elif choice == "3":
        to_download = ["joe"]
    elif choice == "4":
        to_download = ["kristin"]
    elif choice == "5":
        to_download = list(voices.keys())
    elif choice == "0":
        print("Cancelled")
        return
    else:
        print("Invalid choice")
        return
    
    for voice_key in to_download:
        voice = voices[voice_key]
        print(f"\n--- Downloading {voice['description']} ---")
        
        # Download ONNX model
        onnx_path = voices_dir / f"{voice['name']}.onnx"
        if onnx_path.exists():
            print(f"Already exists: {onnx_path}")
        else:
            try:
                download_file(voice['url_onnx'], str(onnx_path))
            except Exception as e:
                print(f"Error downloading ONNX: {e}")
                continue
        
        # Download JSON config
        json_path = voices_dir / f"{voice['name']}.onnx.json"
        if json_path.exists():
            print(f"Already exists: {json_path}")
        else:
            try:
                download_file(voice['url_json'], str(json_path))
            except Exception as e:
                print(f"Error downloading JSON: {e}")
                continue
        
        print(f"[OK] {voice['description']} installed")
    
    print("\n" + "="*60)
    print("Download complete!")
    print("\nTo use a voice, update main.py voice_options with:")
    for voice_key in to_download:
        voice = voices[voice_key]
        print(f'  ("{voice["name"]}", "{voice["description"]}")')
    print("="*60)

if __name__ == "__main__":
    main()