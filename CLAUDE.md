# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based Text-to-Speech application that converts uploaded text files (.txt, .md) into speech using the OpenAI TTS API. The app provides line-by-line audio generation with caching, editing capabilities, and batch processing.

## Commands

### Setup virtual environment
```bash
uv venv
source .venv/bin/activate
```

### Running the application
```bash
streamlit run text_to_speech_app.py
```

### Installing dependencies
```bash
uv pip install -r requirements.txt
```

### Environment setup
Create a `.env` file with:
```
OPENAI_API_KEY="sk-..."
```

## Architecture

The codebase has two main files:
- `text_to_speech_app.py` - Current refactored application with class-based architecture
- `text_to_speech_app_backup.py` - Original functional implementation

### Current Architecture (text_to_speech_app.py)

**Core Classes:**
- `AppConfig` - Configuration dataclass with application settings
- `LoggerSetup` - Handles rotating file logging setup  
- `FileHandler` - File operations (loading, sanitization, directory management, cache cleaning)
- `Utils` - Utility functions (hashing, timestamps, error display)
- `TTSService` - Core TTS functionality and OpenAI API integration
- `StreamlitApp` - Main UI controller that orchestrates all components

**Key Architecture Patterns:**
- **Dependency Injection**: `TTSService` and `FileHandler` are injected into `StreamlitApp`
- **Separation of Concerns**: UI logic separated from business logic and file operations
- **Configuration Management**: Centralized config with dataclass and post-init validation
- **Caching**: Audio files cached by hash of (mode|voice|rate|text) to avoid regeneration
- **Session State Management**: Streamlit session state handles line editing and batch processing state

### Audio Processing Flow
1. Text lines are hashed with TTS parameters to create unique cache keys
2. Audio files stored in `output_audio/` directory with hash-based filenames
3. Line-by-line generation with individual controls (generate, regenerate, edit)
4. Batch processing with progress tracking
5. ZIP download functionality for all generated audio files

### Key Technical Details
- Uses OpenAI TTS-1 model with configurable voices and speed
- Implements rotating file logging with size limits
- Supports real-time line editing with session state persistence
- Audio files cached indefinitely until manual cleanup
- Error handling with user-friendly messages and detailed logging

## File Structure
- `text_to_speech_app.py` - Main application (current)
- `text_to_speech_app_backup.py` - Legacy functional version  
- `requirements.txt` - Python dependencies
- `output_audio/` - Generated MP3 files (cached)
- `tts_app.log` - Application logs (rotating)
- `.env` - Environment variables (not tracked)