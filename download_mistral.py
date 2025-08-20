#!/usr/bin/env python3
"""
Download Mistral 7B Instruct model optimized for RTX 4090
"""

import os
import sys
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress indicator"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    try:
        urllib.request.urlretrieve(url, destination, download_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Mistral 7B Instruct Q5_K_M - Optimal for RTX 4090 (balance of quality and speed)
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    model_path = models_dir / "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    print("Downloading Mistral 7B Instruct v0.2 (Q5_K_M quantization)")
    print("This is optimized for RTX 4090 with excellent quality/speed balance")
    print("File size: ~5.13 GB")
    print()
    
    if download_file(model_url, str(model_path)):
        print(f"\nModel saved to: {model_path}")
        print("You can now use this model in main.py")
    else:
        print("\nFailed to download model. Please check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()