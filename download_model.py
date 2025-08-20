import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading Phi-2 GGUF model (2.7B parameters, optimized for your RTX 4090)...")
    
    model_url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf"
    model_path = models_dir / "phi-2.Q5_K_M.gguf"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return
    
    print(f"Downloading to {model_path}...")
    download_file(model_url, model_path)
    print("Download complete!")

if __name__ == "__main__":
    main()