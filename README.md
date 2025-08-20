# Voice AI Assistant

A simple, reliable, offline voice assistant powered by Whisper STT, Llama LLM, and Windows TTS.

## Features

- **100% Offline**: No internet connection required
- **Voice Input**: Press ENTER to speak (3-second recording)
- **Text Input**: Type messages directly
- **Local LLM**: Uses Phi-2 model for responses
- **Windows TTS**: Built-in text-to-speech
- **GPU Accelerated**: Supports CUDA for faster processing

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Windows OS (for TTS)
- 4GB+ RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-ai-assistant.git
cd voice-ai-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the LLM model:
```bash
python download_model.py
```

## Usage

### Quick Start
Double-click `run.bat` or run:
```bash
python app.py
```

### How to Use
1. Press ENTER to record 3 seconds of audio
2. Speak your question/command
3. Wait for transcription and response
4. The AI will speak its response
5. Type 'quit' to exit

### Alternative: Type Instead of Speak
Instead of pressing ENTER, you can type your message directly.

## Project Structure

```
voice-ai-assistant/
├── app.py              # Main application
├── download_model.py   # Model downloader
├── requirements.txt    # Python dependencies
├── run.bat            # Windows launcher
├── quick_test.py      # Component tester
├── modules/           # Core modules
│   ├── audio_capture.py
│   ├── audio_playback.py
│   ├── stt_module.py
│   ├── llm_module.py
│   ├── tts_module.py
│   └── streaming_pipeline.py
└── models/            # Model files (created after download)
```

## Troubleshooting

### No audio detected
- Check your microphone is connected
- Run `python quick_test.py` to test components
- Ensure Windows audio service is running

### Model not found
- Run `python download_model.py` to download the Phi-2 model
- Check the `models/` directory exists

### CUDA not available
- The app will automatically fall back to CPU
- Performance will be slower but still functional

## License

MIT