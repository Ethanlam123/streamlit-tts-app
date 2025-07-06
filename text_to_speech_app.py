"""
requirements.txt
----------------
streamlit
pyttsx3
openai
# Optional for offline fallback:
# TTS
"""

import os
import sys
import io
import hashlib
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Dependency checks
missing = []
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    missing.append('pyttsx3')
try:
    import openai
except ImportError:
    openai = None
    missing.append('openai')
try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    # TTS is optional, only warn if pyttsx3 is also missing

# --- Logging Setup ---
LOG_FILE = 'tts_app.log'
logger = logging.getLogger('tts_app')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Constants ---
SUPPORTED_EXTS = ['.txt', '.md']
DEFAULT_OUTPUT_DIR = 'output_audio'

# --- Helper Functions ---


def pip_hint(pkg):
    st.error(
        f"Missing required package: `{pkg}`.\nInstall with: `pip install {pkg}`")


def sanitize_filename(s: str) -> str:
    # Remove dangerous chars, keep only safe ones
    return ''.join(c for c in s if c.isalnum() or c in ('-', '_', '.'))


def hash_line(line: str) -> str:
    return hashlib.sha256(line.strip().encode('utf-8')).hexdigest()[:12]


def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_file(uploaded_file) -> List[str]:
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.splitlines()
        return lines
    except Exception as e:
        logger.error(f"File read error: {e}")
        st.error(f"Failed to read file: {e}")
        return []


def choose_tts_engine(mode: str, voice: Optional[str] = None, rate: Optional[float] = None):
    if mode == 'Offline (pyttsx3)':
        if not pyttsx3:
            pip_hint('pyttsx3')
            return None
        engine = pyttsx3.init()
        if rate:
            try:
                engine.setProperty('rate', int(rate))
            except Exception as e:
                logger.warning(f"Failed to set rate: {e}")
        if voice:
            try:
                voices = engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            except Exception as e:
                logger.warning(f"Failed to set voice: {e}")
        return engine
    elif mode == 'Offline (Coqui-TTS)':
        if not CoquiTTS:
            pip_hint('TTS')
            return None
        # Use a default English model
        try:
            tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
            return tts
        except Exception as e:
            logger.error(f"Coqui-TTS load error: {e}")
            st.error(f"Failed to load Coqui-TTS: {e}")
            return None
    elif mode == 'OpenAI API':
        if not openai:
            pip_hint('openai')
            return None
        return 'openai'  # Placeholder
    else:
        st.error("Unknown TTS mode selected.")
        return None


@st.cache_resource(show_spinner=False)
def get_offline_engine(mode, voice, rate):
    return choose_tts_engine(mode, voice, rate)


@st.cache_data(show_spinner=False)
def cache_audio_path(line: str, mode: str, voice: str, rate: float, output_dir: str) -> str:
    # Used for cache key only
    h = hash_line(f"{mode}|{voice}|{rate}|{line}")
    ensure_dir(output_dir)
    return str(Path(output_dir) / f"{get_timestamp()}_{h}.mp3")


def synthesize_line(line: str, mode: str, voice: str, rate: float, output_dir: str, openai_api_key: Optional[str] = None, force_regen: bool = False) -> Tuple[Optional[str], str]:
    """
    Returns (audio_path, status)
    status: 'success', 'error', 'cached'
    """
    h = hash_line(f"{mode}|{voice}|{rate}|{line}")
    ensure_dir(output_dir)
    audio_path = str(Path(output_dir) / f"{h}.mp3")
    if not force_regen and os.path.exists(audio_path):
        logger.info(f"Cache hit for line: {line}")
        return audio_path, 'cached'
    try:
        if mode == 'Offline (pyttsx3)':
            engine = pyttsx3.init()
            if rate:
                engine.setProperty('rate', int(rate))
            if voice:
                voices = engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            engine.save_to_file(line, audio_path)
            engine.runAndWait()
        elif mode == 'Offline (Coqui-TTS)':
            tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
            tts.tts_to_file(text=line, file_path=audio_path)
        elif mode == 'OpenAI API':
            if not openai_api_key:
                return None, 'error'
            openai.api_key = openai_api_key
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=line,
                response_format="mp3"
            )
            with open(audio_path, 'wb') as f:
                f.write(response.content)
        else:
            return None, 'error'
        logger.info(f"Audio generated: {audio_path}")
        return audio_path, 'success'
    except Exception as e:
        logger.error(f"TTS error for line '{line}': {e}")
        return None, 'error'


def get_available_voices(mode: str) -> List[str]:
    if mode == 'Offline (pyttsx3)':
        if not pyttsx3:
            return []
        try:
            engine = pyttsx3.init()
            return [v.name for v in engine.getProperty('voices')]
        except Exception:
            return []
    elif mode == 'Offline (Coqui-TTS)':
        # Coqui-TTS: only one voice for default model
        return ['default']
    elif mode == 'OpenAI API':
        # As of 2024, OpenAI supports 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
        return ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    return []


def get_default_voice(mode: str) -> str:
    voices = get_available_voices(mode)
    return voices[0] if voices else ''


def get_default_rate(mode: str) -> float:
    if mode == 'Offline (pyttsx3)':
        return 200
    elif mode == 'Offline (Coqui-TTS)':
        return 1.0
    elif mode == 'OpenAI API':
        return 1.0
    return 1.0


def is_openai_key_set() -> bool:
    return bool(os.environ.get('OPENAI_API_KEY', ''))


def get_openai_key() -> Optional[str]:
    return os.environ.get('OPENAI_API_KEY', None)


# --- Streamlit UI ---
st.set_page_config(page_title="Text-to-Speech App",
                   layout="wide", page_icon="🔊")
st.title("🔊 Text-to-Speech App")

# --- Sidebar ---
st.sidebar.header("Settings")

# TTS Back-End selection
tts_modes = []
if pyttsx3:
    tts_modes.append('Offline (pyttsx3)')
if CoquiTTS:
    tts_modes.append('Offline (Coqui-TTS)')
tts_modes.append('OpenAI API')

mode = st.sidebar.selectbox("TTS Back-End", tts_modes)

# Voice selection
voices = get_available_voices(mode)
def_voice = get_default_voice(mode)
voice = st.sidebar.selectbox(
    "Voice", voices, index=0 if def_voice not in voices else voices.index(def_voice))

# Speaking rate
rate = st.sidebar.slider(
    "Speaking Rate",
    min_value=50 if mode == 'Offline (pyttsx3)' else 0.5,
    max_value=300 if mode == 'Offline (pyttsx3)' else 2.0,
    value=get_default_rate(mode),
    step=1 if mode == 'Offline (pyttsx3)' else 0.05
)

# Output folder
output_dir = st.sidebar.text_input("Output Folder", value=DEFAULT_OUTPUT_DIR)

# --- File Upload ---
st.subheader("1. Upload a Text File (.txt or .md)")
uploaded_file = st.file_uploader(
    "Choose a file", type=[e[1:] for e in SUPPORTED_EXTS])

lines = []
if uploaded_file:
    lines = load_file(uploaded_file)
    if lines:
        st.success(f"Loaded {len(lines)} lines from {uploaded_file.name}")
    else:
        st.error("No lines found in file.")

# --- OpenAI API Key Check ---
openai_key_required = (mode == 'OpenAI API')
openai_key = get_openai_key() if openai_key_required else None
if openai_key_required and not openai_key:
    st.error("OPENAI_API_KEY not set in environment. Please add it to your .env or environment variables.")
    st.stop()

# --- Main UI: Text Preview & Per-Line Controls ---
if lines:
    st.subheader("2. Text Preview & Line-by-Line Generation")
    if 'editable_lines' not in st.session_state or len(st.session_state['editable_lines']) != len(lines):
        st.session_state['editable_lines'] = lines.copy()
    if 'editing_line_idx' not in st.session_state:
        st.session_state['editing_line_idx'] = None
    if 'edit_buffer' not in st.session_state:
        st.session_state['edit_buffer'] = ''
    with st.container():
        st.markdown(
            "**Preview:** (Each line has its own Generate and Edit button)")
        status_placeholders = []
        audio_placeholders = []
        for idx, line in enumerate(st.session_state['editable_lines']):
            if not line.strip():
                continue  # Skip empty lines
            col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 2, 3])
            with col1:
                st.markdown(f"`{idx+1:03}` {line}")
            with col2:
                btn_key = f"gen_{idx}"
                gen_btn = st.button("🪄", key=btn_key, disabled=False)
            with col3:
                regen_btn_key = f"regen_{idx}"
                regen_btn = st.button(
                    "♻️", key=regen_btn_key, disabled=False)
            with col4:
                edit_btn_key = f"edit_{idx}"
                edit_btn = st.button(
                    "✏️", key=edit_btn_key, disabled=st.session_state['editing_line_idx'] is not None and st.session_state['editing_line_idx'] != idx)
            with col5:
                status_placeholder = st.empty()
                audio_placeholder = st.empty()
                status_placeholders.append(status_placeholder)
                audio_placeholders.append(audio_placeholder)
            # Editing logic
            if st.session_state['editing_line_idx'] == idx:
                new_text = st.text_input(
                    f"Edit line {idx+1}", value=st.session_state['edit_buffer'], key=f"edit_input_{idx}")
                save_col, cancel_col = st.columns([1, 1])
                with save_col:
                    if st.button("💾 Save", key=f"save_{idx}"):
                        st.session_state['editable_lines'][idx] = new_text
                        st.session_state['editing_line_idx'] = None
                        st.session_state['edit_buffer'] = ''
                        st.rerun()
                with cancel_col:
                    if st.button("❌ Cancel", key=f"cancel_{idx}"):
                        st.session_state['editing_line_idx'] = None
                        st.session_state['edit_buffer'] = ''
                        st.rerun()
            elif edit_btn:
                st.session_state['editing_line_idx'] = idx
                st.session_state['edit_buffer'] = line
                st.rerun()
            if gen_btn:
                status_placeholder.info("Generating...")
                audio_path, status = synthesize_line(
                    line, mode, voice, rate, output_dir, openai_key)
                if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    if status == 'success':
                        status_placeholder.success("✅ Success")
                    else:
                        status_placeholder.info("♻️ Cached")
                    with open(audio_path, 'rb') as f:
                        audio_placeholder.audio(f.read(), format='audio/mp3')
                else:
                    status_placeholder.error(
                        "❌ Error: No audio file generated.")
                    if audio_path:
                        st.warning(
                            f"Audio file not found or empty: {audio_path}")
            if regen_btn:
                status_placeholder.info("Regenerating...")
                audio_path, status = synthesize_line(
                    line, mode, voice, rate, output_dir, openai_key, force_regen=True)
                if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    status_placeholder.success("✅ Success (Regenerated)")
                    with open(audio_path, 'rb') as f:
                        audio_placeholder.audio(f.read(), format='audio/mp3')
                else:
                    status_placeholder.error(
                        "❌ Error: No audio file generated.")
                    if audio_path:
                        st.warning(
                            f"Audio file not found or empty: {audio_path}")

    # --- Batch Generation ---
    st.subheader("3. Batch Utilities")
    if 'batch_running' not in st.session_state:
        st.session_state['batch_running'] = False

    def run_batch():
        st.session_state['batch_running'] = True
        progress = st.progress(0)
        n = len(lines)
        for idx, line in enumerate(lines):
            status_placeholder = status_placeholders[idx]
            audio_placeholder = audio_placeholders[idx]
            status_placeholder.info("Generating...")
            audio_path, status = synthesize_line(
                line, mode, voice, rate, output_dir, openai_key)
            if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                if status == 'success':
                    status_placeholder.success("✅ Success")
                else:
                    status_placeholder.info("♻️ Cached")
                with open(audio_path, 'rb') as f:
                    audio_placeholder.audio(f.read(), format='audio/mp3')
            else:
                status_placeholder.error("❌ Error: No audio file generated.")
                if audio_path:
                    st.warning(f"Audio file not found or empty: {audio_path}")
            progress.progress((idx+1)/n)
        st.session_state['batch_running'] = False
        st.toast("Batch generation complete!", icon="🎉")
    st.button("🚀 Generate All", on_click=run_batch,
              disabled=st.session_state['batch_running'])

# --- Error Handling for Missing Packages ---
if 'pyttsx3' in missing:
    pip_hint('pyttsx3')
if 'openai' in missing:
    pip_hint('openai')

# --- Logging UI ---
with st.expander("Show Logs"):
    try:
        with open(LOG_FILE, 'r') as f:
            logs = f.read()[-5000:]
        st.code(logs, language='log')
    except Exception:
        st.info("No logs yet.")

# --- Stretch Goals (TODOs) ---
# TODO: Allow voice cloning or speaker embeddings in offline mode if TTS models support it.
# TODO: Offer “download zip” of all generated audio.
# TODO: Highlight the last generated line in the UI.

# --- Usage Docstring ---
if __name__ == "__main__":
    print("""
Run this app with:
    streamlit run text_to_speech_app.py

# .env
OPENAI_API_KEY="sk-..."

Usage notes:
- Upload a .txt or .md file.
- Choose TTS back-end and settings in the sidebar.
- Generate audio per line or for all lines.
- Audio files are saved in the output folder (default: output_audio).
- Requires packages in requirements.txt (see top of file).
""")
