# 🚀 RTX 4090 Voice Assistant

A high-performance, offline voice assistant leveraging RTX 4090 GPU acceleration for real-time speech recognition, language processing, and human-like text-to-speech.

## ✨ Features

- **100% Offline**: Complete privacy with no internet required
- **GPU Accelerated**: Optimized for NVIDIA RTX 4090
- **Human-like Voice**: Neural TTS with Piper (Ryan/Amy voices)
- **Real-time Processing**: Fast transcription and response generation
- **Natural Conversation**: Context-aware responses with conversation history

## 🛠️ System Requirements

- **GPU**: NVIDIA RTX 4090 (or any CUDA-capable GPU)
- **RAM**: 16GB minimum
- **Storage**: 5GB for models
- **OS**: Windows 10/11
- **Python**: 3.9-3.11

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/LuisFaxas/voice-ai-assistant.git
cd voice-ai-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download AI Models
```bash
# Download language model (Phi-2)
python download_model.py

# Download voice models (Ryan & Amy)
python download_best_voices.py
```

## 🎮 Usage

### Start the Assistant
```bash
python main.py
```

### Controls
- **Press ENTER**: Start voice recording (3 seconds)
- **Type message**: Skip voice input and type directly
- **Type 'quit'**: Exit the assistant

## 🏗️ Architecture

```
main.py                 # Main application
├── Whisper STT         # Speech-to-Text (GPU)
├── Phi-2 LLM          # Language Model (GPU)
└── Piper TTS          # Text-to-Speech (GPU)
```

### Models Used
- **STT**: OpenAI Whisper (small model)
- **LLM**: Microsoft Phi-2 (2.7B parameters, GGUF quantized)
- **TTS**: Piper with Ryan (male) or Amy (female) voices

## 📁 Project Structure

```
voice-ai-assistant/
├── main.py                    # Main application
├── download_model.py          # LLM downloader
├── download_best_voices.py    # Voice model downloader
├── requirements.txt           # Python dependencies
├── models/                    # AI models directory
│   ├── phi-2.Q5_K_M.gguf     # Language model
│   └── piper_voices/          # TTS voice models
│       ├── en_US-ryan-high.onnx
│       └── en_US-amy-medium.onnx
└── old_versions/              # Archived old implementations
```

## ⚙️ Configuration

The assistant can be configured by modifying `main.py`:

- **Recording Duration**: `self.recording_duration = 3.0` (seconds)
- **LLM Temperature**: `temperature=0.7` (0.0-1.0, higher = more creative)
- **Max Response Length**: `max_tokens=60` (adjust for longer/shorter responses)
- **TTS Speed**: `--length-scale 1.15` (higher = slower speech)

## 🐛 Troubleshooting

### GPU Not Detected
- Ensure CUDA is installed: `nvidia-smi`
- Install CUDA toolkit if missing
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### TTS Not Working
- Verify Piper is installed: `piper --help`
- Check voice models exist in `models/piper_voices/`
- Re-download voices: `python download_best_voices.py`

### Microphone Issues
- Check Windows microphone permissions
- Verify default recording device in Windows settings
- Test with: `python -c "import sounddevice as sd; print(sd.query_devices())"`

## 🔧 Performance Optimization

### GPU Memory Usage
- Whisper (small): ~1.5GB VRAM
- Phi-2: ~3GB VRAM
- Piper TTS: ~500MB VRAM
- **Total**: ~5GB VRAM (RTX 4090 has 24GB)

### Response Time
- Voice recording: 3 seconds
- Transcription: <1 second
- LLM generation: 1-2 seconds
- TTS generation: <1 second
- **Total**: ~6 seconds per interaction

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- OpenAI for Whisper
- Microsoft for Phi-2
- Rhasspy for Piper TTS
- NVIDIA for CUDA acceleration

---

**Built with ❤️ for RTX 4090**