import os
import sys
import io
import hashlib
import time
import logging
import zipfile
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
    import openai
except ImportError:
    openai = None
    missing.append('openai')

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
    s = s.replace(' ', '_')  # Replace spaces with underscores
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
        if mode == 'OpenAI API':
            if not openai_api_key:
                return None, 'error'
            openai.api_key = openai_api_key
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=line,
                speed=rate,
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
    # As of 2024, OpenAI supports 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
    return ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']


def get_default_voice(mode: str) -> str:
    voices = get_available_voices(mode)
    return voices[0] if voices else ''


def get_default_rate(mode: str) -> float:
    return 1.2


def is_openai_key_set() -> bool:
    return bool(os.environ.get('OPENAI_API_KEY', ''))


def get_openai_key() -> Optional[str]:
    return os.environ.get('OPENAI_API_KEY', None)


def clean_cache(output_dir: str):
    """Delete all .mp3 files in the output directory."""
    dir_path = Path(output_dir)
    if not dir_path.exists():
        return 0
    count = 0
    for file in dir_path.glob('*.mp3'):
        try:
            file.unlink()
            count += 1
        except Exception as e:
            logger.error(f"Failed to delete {file}: {e}")
    return count


# --- Streamlit UI ---
st.set_page_config(page_title="Text-to-Speech App",
                   layout="wide", page_icon="🔊")
st.title("🔊 Text-to-Speech App")

# --- Sidebar ---
st.sidebar.header("Settings")

# Only OpenAI API mode
mode = 'OpenAI API'

# Voice selection
voices = get_available_voices(mode)
def_voice = get_default_voice(mode)
voice = st.sidebar.selectbox(
    "Voice", voices, index=0 if def_voice not in voices else voices.index(def_voice))

# Speaking rate
rate = st.sidebar.slider(
    "Speaking Rate",
    min_value=0.5,
    max_value=2.0,
    value=get_default_rate(mode),
    step=0.05
)

# Output folder
output_dir = st.sidebar.text_input("Output Folder", value=DEFAULT_OUTPUT_DIR)

# --- Clean Cache Button ---
if st.sidebar.button("🧹 Clean All Cache"):
    deleted = clean_cache(output_dir)
    if deleted > 0:
        st.sidebar.success(f"Deleted {deleted} audio file(s) from cache.")
    else:
        st.sidebar.info("No audio files to delete in cache.")

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
openai_key_required = True
openai_key = get_openai_key()
if not openai_key:
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
    if 'last_generated_idx' not in st.session_state:
        st.session_state['last_generated_idx'] = None

    with st.container():
        st.markdown(
            "**Preview:** (Each line has its own Generate and Edit button)")

        # Create placeholders for all lines
        placeholders = []
        for i in range(len(st.session_state['editable_lines'])):
            placeholders.append(st.empty())

        for idx, line in enumerate(st.session_state['editable_lines']):
            if not line.strip():
                continue  # Skip empty lines

            is_last_generated = st.session_state.get(
                'last_generated_idx') == idx

            with placeholders[idx].container():
                col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 2, 3])

                with col1:
                    if is_last_generated:
                        st.markdown(
                            f"<div style='background-color: #e0f7fa; padding: 10px 16px; border-radius: 8px; margin-bottom: 8px; font-size: 1.1em; color: #222; font-family: sans-serif;'>"
                            f"<b>{idx+1:03}</b> {line}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div style='padding: 10px 16px; margin-bottom: 8px; font-size: 1.1em; color: #eee; font-family: sans-serif;'>"
                            f"<b>{idx+1:03}</b> {line}</div>", unsafe_allow_html=True)

                with col2:
                    btn_key = f"gen_{idx}"
                    if st.button("🪄", key=btn_key, disabled=False):
                        st.session_state['last_generated_idx'] = idx
                        with col5:
                            st.info("Generating...")
                            audio_path, status = synthesize_line(
                                line, mode, voice, rate, output_dir, openai_key)
                            if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                if status == 'success':
                                    st.success("✅ Success")
                                else:
                                    st.info("♻️ Cached")
                                with open(audio_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/mp3')
                            else:
                                st.error("❌ Error: No audio file generated.")
                                if audio_path:
                                    st.warning(
                                        f"Audio file not found or empty: {audio_path}")

                with col3:
                    regen_btn_key = f"regen_{idx}"
                    if st.button("♻️", key=regen_btn_key, disabled=False):
                        st.session_state['last_generated_idx'] = idx
                        with col5:
                            st.info("Regenerating...")
                            audio_path, status = synthesize_line(
                                line, mode, voice, rate, output_dir, openai_key, force_regen=True)
                            if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                st.success("✅ Success (Regenerated)")
                                with open(audio_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/mp3')
                            else:
                                st.error("❌ Error: No audio file generated.")
                                if audio_path:
                                    st.warning(
                                        f"Audio file not found or empty: {audio_path}")

                with col4:
                    edit_btn_key = f"edit_{idx}"
                    if st.button("✏️", key=edit_btn_key, disabled=st.session_state['editing_line_idx'] is not None and st.session_state['editing_line_idx'] != idx):
                        st.session_state['editing_line_idx'] = idx
                        st.session_state['edit_buffer'] = line
                        st.rerun()

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

    # --- Batch Generation ---
    st.subheader("3. Batch Utilities")
    if 'batch_running' not in st.session_state:
        st.session_state['batch_running'] = False

    def run_batch(placeholders):
        st.session_state['batch_running'] = True
        st.session_state['last_generated_idx'] = None  # Reset highlight
        progress = st.progress(0)
        n = len(st.session_state['editable_lines'])

        for idx, line in enumerate(st.session_state['editable_lines']):
            if not line.strip():
                progress.progress((idx + 1) / n)
                continue

            with placeholders[idx].container():
                col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 2, 3])
                with col1:
                    st.markdown(f"`{idx+1:03}` {line}")
                with col5:
                    st.info("Generating...")
                    audio_path, status = synthesize_line(
                        line, mode, voice, rate, output_dir, openai_key)
                    if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                        if status == 'success':
                            st.success("✅ Success")
                        else:
                            st.info("♻️ Cached")
                        with open(audio_path, 'rb') as f:
                            st.audio(f.read(), format='audio/mp3')
                    else:
                        st.error("❌ Error: No audio file generated.")
                        if audio_path:
                            st.warning(
                                f"Audio file not found or empty: {audio_path}")

            progress.progress((idx + 1) / n)
            st.session_state['last_generated_idx'] = idx  # Highlight as we go

        st.session_state['batch_running'] = False
        st.toast("Batch generation complete!", icon="🎉")
        st.rerun()

    if st.button("🚀 Generate All", disabled=st.session_state['batch_running']):
        run_batch(placeholders)

    # --- Download All Audio ---
    st.subheader("4. Download All Audio")

    zip_buffer = io.BytesIO()
    script_name = Path(uploaded_file.name).stem if uploaded_file else "script"

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for idx, line in enumerate(st.session_state['editable_lines']):
            if not line.strip():
                continue
            h = hash_line(f"{mode}|{voice}|{rate}|{line}")
            audio_path = str(Path(output_dir) / f"{h}.mp3")
            if os.path.exists(audio_path):
                file_name = f"{script_name}_{idx+1:03}.mp3"
                zip_file.write(audio_path, file_name)

    zip_buffer.seek(0)

    if zip_buffer.getbuffer().nbytes > 0:
        st.download_button(
            label="📦 Download All as ZIP",
            data=zip_buffer,
            file_name=f"{script_name}_audio_{get_timestamp()}.zip",
            mime="application/zip",
        )
    else:
        st.info("No audio files have been generated yet.")

# --- Error Handling for Missing Packages ---
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
