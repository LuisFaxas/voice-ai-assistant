# Voice Assistant Usage Guide

## Quick Start

```bash
# Default: Mistral 7B + Ryan voice
python main.py

# Custom LLM and voice
python main.py --llm mistral --voice ryan
python main.py --llm phi2 --voice amy
```

## Command Line Options

### Language Models (`--llm`)
- `mistral` (default) - Mistral 7B v0.3, best quality conversations
- `mistral-v2` - Mistral 7B v0.2, stable version
- `phi2` - Microsoft Phi-2, fastest responses

### Voices (`--voice`)
- `ryan` (default) - Male voice, high quality
- `amy` - Female voice, clear and fast

### Recording Modes
- Default: Voice Activity Detection (VAD) - stops when you stop talking
- `--instant` - Fixed 1-second recording, no VAD

## Examples

### Best Quality (Mistral + Ryan)
```bash
python main.py --llm mistral --voice ryan
```

### Fast Mode (Phi-2 + Amy)
```bash
python main.py --llm phi2 --voice amy
```

### Ultra-Fast (Phi-2 + Instant)
```bash
python main.py --llm phi2 --instant
```

### Female Assistant
```bash
python main.py --voice amy
```

## Performance Expectations

With RTX 4090:
- **Mistral 7B**: 2-3 second total response time
- **Phi-2**: <2 second total response time
- **Instant Mode**: Saves 0.5-1.5 seconds on recording

## Voice Commands

While running:
- **ENTER**: Start voice recording
- **Type text**: Skip voice, type your message
- **"quit"**: Exit the assistant
- **"metrics"**: Show performance statistics

## Troubleshooting

### Voice Issues
- If voice sounds robotic, ensure you're using `--voice ryan`
- If voice glitches, check audio drivers are updated

### LLM Issues
- If responses claim to be OpenAI, update to latest version
- If responses are too short, use `--llm mistral` for better quality

### Performance Issues
- Use `--llm phi2` for faster responses
- Use `--instant` to eliminate VAD delay
- Check GPU is detected (should show RTX 4090 on startup)

## Advanced Features (Coming Soon)

### Silero VAD Integration
Advanced neural Voice Activity Detection for perfect speech endpoint detection.

### Personality Modes
```bash
python main.py --personality assistant  # Professional
python main.py --personality friendly   # Casual, warm
```

### Custom Voice Cloning
Using XTTS-v2 for emotional voice synthesis with your own voice.