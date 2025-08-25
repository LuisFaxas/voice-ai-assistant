# Performance Guide - Voice Assistant

## Quick Start

### Default (Fast Mode) - **<2 seconds response**
```bash
python main.py
# OR
python main.py --fast
```
- Uses Phi-2 (2GB model)
- Amy voice (faster synthesis)
- Optimized for speed

### Quality Mode - **<5 seconds response**
```bash
python main.py --quality
```
- Uses Mistral 7B (5GB model)
- Better understanding and responses
- More natural conversation

### Instant Mode - **Ultra-fast, 1s recording**
```bash
python main.py --instant
# OR
python main.py --quality --instant
```
- Fixed 1 second recording (no VAD)
- Immediate processing
- Best for short commands

## Performance Comparison

| Mode | Recording | STT | LLM | TTS | Total |
|------|-----------|-----|-----|-----|-------|
| **Fast (default)** | 0.5-1.5s | 0.3s | 0.5s | 0.5s | **~2s** |
| **Fast + Instant** | 1.0s | 0.3s | 0.5s | 0.5s | **~2.3s** |
| **Quality** | 0.5-2.5s | 0.3s | 2s | 0.8s | **~4s** |
| **Quality + Instant** | 1.0s | 0.3s | 2s | 0.8s | **~4.1s** |

## Key Improvements

### Voice Quality Fixes
- Removed noise artifacts (eliminated noise-scale/noise-w parameters)
- Natural speech speed (1.0-1.05 vs 0.9)
- Minimal sentence pauses (0.05s vs 0.2s)
- Amy voice prioritized for clarity

### Speed Optimizations
- Phi-2 as default (2GB vs 5GB Mistral)
- Reduced context windows (512 for Phi-2, 2048 for Mistral)
- Faster VAD detection (0.5s silence vs 0.8s)
- Lower pygame buffer (256 vs 512)
- Instant mode option (1s fixed recording)

### Response Quality
- Removed 10-word limit for Phi-2
- Natural response length (up to 30 tokens)
- Better prompt formatting
- Balanced temperature settings

## Troubleshooting

### Still too slow?
1. Use instant mode: `python main.py --instant`
2. Check GPU is detected (should show RTX 4090 on startup)
3. Ensure models are on SSD, not HDD

### Voice still has artifacts?
1. Update Piper: `pip install --upgrade piper-tts`
2. Try Ryan voice: Edit main.py to prioritize Ryan over Amy
3. Check audio drivers are up to date

### Responses too short/long?
- Fast mode: 30 tokens max (2-3 sentences)
- Quality mode: 40 tokens max (3-4 sentences)
- Edit `max_tokens` in main.py to adjust

## Usage Examples

### Quick Commands (Fast + Instant)
```bash
python main.py --instant
```
Perfect for:
- "What time is it?"
- "Turn on the lights"
- "Play music"

### Conversations (Quality Mode)
```bash
python main.py --quality
```
Better for:
- Complex questions
- Detailed explanations
- Natural dialogue

### Balanced (Fast + VAD)
```bash
python main.py
```
Best default for:
- General use
- Variable input length
- Good speed/quality balance