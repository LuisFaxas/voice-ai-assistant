# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTX 4090-optimized offline voice assistant with real-time STT, LLM, and TTS capabilities. The system achieves <3s response times through GPU acceleration and efficient model selection.

## Critical Commands

### Running the Application
```bash
# Main application (uses Mistral 7B v0.3 if available)
python main.py

# Alternative startup methods
run.bat           # Windows batch launcher with error handling
setup.bat         # First-time dependency installation
```

### Model Management
```bash
# Download LLMs (in priority order)
python download_mistral_v3.py    # Mistral 7B v0.3 (best quality, 5GB)
python download_mistral.py       # Mistral 7B v0.2 (fallback)
python download_model.py         # Phi-2 (lightweight backup)

# Download TTS voices
python download_best_voices.py   # High-quality voices (Lessac, LibriTTS)
python download_better_voices.py # Additional professional voices

# Test TTS quality
python test_tts.py               # Compare different voice settings
```

### Dependencies
```bash
pip install -r requirements.txt
# Note: Faster-Whisper auto-downloads on first run
```

## Architecture & Key Design Decisions

### Model Cascade Strategy
The system implements a fallback chain for reliability:
1. **Primary**: Mistral 7B v0.3 (best reasoning, 4096 context)
2. **Fallback**: Mistral 7B v0.2 (stable version)
3. **Emergency**: Phi-2 (lightweight, always works)

This ensures the assistant never fails to start, automatically using the best available model.

### Performance Pipeline
```
User Speech → VAD → Faster-Whisper → Mistral 7B → Piper TTS → Audio Output
            (0.5-4s)   (<0.5s)         (<1s)        (<1s)
```

Key optimizations:
- **VAD** (Voice Activity Detection): Dynamic recording (0.5-4s) instead of fixed 3s
- **Faster-Whisper**: 4-5x faster than OpenAI Whisper via CTranslate2
- **Full GPU offload**: `n_gpu_layers=-1` for maximum RTX 4090 utilization
- **TTS caching**: Common responses cached to skip regeneration

### Critical File Interactions

**main.py** orchestrates three subsystems:
1. **STT Pipeline**: `_init_faster_whisper()` → `transcribe_audio_faster()`
   - Uses `models/whisper/` cache (auto-downloaded, excluded from git)
   - Compute type: `int8_float16` on RTX 4090

2. **LLM Pipeline**: `_init_llm()` → `generate_response_fast()`
   - Loads from `models/*.gguf` files
   - Mistral uses instruct format: `[INST] query [/INST]`
   - Phi-2 uses conversational format

3. **TTS Pipeline**: `_init_tts_optimized()` → `speak_text_fast()`
   - Prioritizes Ryan voice (natural) over Amy (fast)
   - Uses Popen instead of shell for better control
   - Settings: speed=0.98, sentence-silence=0.15-0.2s

### Windows-Specific Handling
- UTF-8 encoding fix for terminals
- ASCII output (no emojis) to prevent encoding errors
- Batch files for user-friendly execution
- Path handling with proper escaping

## Development Patterns

### Adding New Models
1. Create downloader in pattern of `download_*.py`
2. Add to model_paths list in `_init_llm()` (priority order)
3. Handle model-specific prompt formats (instruct vs conversational)

### TTS Voice Tuning
Adjust in `speak_text_fast()`:
- `--length-scale`: 0.95-1.0 (speed vs naturalness)
- `--sentence-silence`: 0.1-0.2s (pause duration)
- `--noise-scale`: 0.5-0.7 (artifact reduction)

### Performance Monitoring
Built-in `PerformanceMonitor` tracks:
- Recording, Transcription, LLM, TTS, Total times
- Access via typing "metrics" during runtime
- History stored in deque(maxlen=10) for averaging

## Model Storage

```
models/
├── mistral-7b-instruct-v0.3.Q5_K_M.gguf  # 4.8GB, primary
├── mistral-7b-instruct-v0.2.Q5_K_M.gguf  # 4.8GB, fallback
├── phi-2.Q5_K_M.gguf                      # 2.0GB, emergency
├── piper_voices/                          # TTS voices
│   ├── en_US-ryan-high.onnx              # Natural quality
│   └── en_US-amy-medium.onnx             # Faster response
└── whisper/                               # Auto-downloaded STT cache
```

## Common Issues & Solutions

### EOF Errors in Terminal
- Run directly via `python main.py` instead of through automation
- Ensure terminal has proper input capabilities

### TTS Artifacts/Noise
- Check `--length-scale` not below 0.95
- Ensure `--sentence-silence` > 0.1
- Verify pygame buffer size ≥ 512

### Slow Response Times
- Confirm GPU detected: Check for "NVIDIA GeForce RTX 4090" in startup
- Verify CUDA: `torch.cuda.is_available()` should be True
- Check model loading: Mistral should show "Full GPU offload"

### Context Warnings
- Normal: "n_ctx_per_seq (4096) < n_ctx_train (32768)"
- Indicates using subset of model's training context (intentional for speed)