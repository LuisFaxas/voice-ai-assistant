#!/usr/bin/env python3
"""
Download Mistral 7B Instruct v0.3 - Latest version optimized for RTX 4090
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
    
    # Mistral 7B Instruct v0.3 Q5_K_M - Latest version for RTX 4090
    # v0.3 has improved instruction following and better reasoning
    # Try different sources for v0.3
    model_urls = [
        "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
        "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q5_K_M.gguf"
    ]
    model_path = models_dir / "mistral-7b-instruct-v0.3.Q5_K_M.gguf"
    
    # Check if old v0.2 exists
    old_model = models_dir / "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    if old_model.exists():
        print(f"Found older Mistral v0.2 at {old_model}")
        print("Will keep both versions for compatibility")
        remove_old = False
    else:
        remove_old = False
    
    if model_path.exists():
        print(f"Model v0.3 already exists at {model_path}")
        print("Skipping download.")
        return
    
    print("\n" + "="*60)
    print("Downloading Mistral 7B Instruct v0.3 (Q5_K_M)")
    print("Latest version with improved instruction following")
    print("Optimized for RTX 4090 - Full GPU acceleration")
    print("File size: ~5.13 GB")
    print("="*60 + "\n")
    
    # Try different sources
    downloaded = False
    for url in model_urls:
        print(f"\nTrying source: {url.split('/')[3]}")
        if download_file(url, str(model_path)):
            downloaded = True
            break
        else:
            print("Trying next source...")
    
    if downloaded:
        print(f"\nModel saved to: {model_path}")
        print("\nMistral v0.3 improvements over v0.2:")
        print("- Better instruction following")
        print("- Improved reasoning capabilities")
        print("- Enhanced context understanding")
        print("- More coherent long-form responses")
        
        if remove_old and old_model.exists():
            try:
                os.remove(old_model)
                print(f"\nRemoved old model: {old_model}")
            except Exception as e:
                print(f"\nCouldn't remove old model: {e}")
        
        print("\nYou can now use this model in main.py")
    else:
        print("\nFailed to download model. Please check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()