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
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

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

# --- Configuration ---
@dataclass
class AppConfig:
    SUPPORTED_EXTS: List[str] = None
    DEFAULT_OUTPUT_DIR: str = 'output_audio'
    LOG_FILE: str = 'tts_app.log'
    MAX_LOG_SIZE: int = 1_000_000
    LOG_BACKUP_COUNT: int = 3
    OPENAI_VOICES: List[str] = None
    DEFAULT_RATE: float = 1.2
    
    def __post_init__(self):
        if self.SUPPORTED_EXTS is None:
            self.SUPPORTED_EXTS = ['.txt', '.md']
        if self.OPENAI_VOICES is None:
            self.OPENAI_VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

config = AppConfig()

# --- Logging Setup ---
class LoggerSetup:
    @staticmethod
    def setup_logger(name: str, config: AppConfig) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(
            config.LOG_FILE, 
            maxBytes=config.MAX_LOG_SIZE, 
            backupCount=config.LOG_BACKUP_COUNT
        )
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        return logger

logger = LoggerSetup.setup_logger('tts_app', config)

# --- Utility Classes ---

class FileHandler:
    @staticmethod
    def load_file(uploaded_file) -> List[str]:
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = content.splitlines()
            return lines
        except Exception as e:
            logger.error(f"File read error: {e}")
            st.error(f"Failed to read file: {e}")
            return []
    
    @staticmethod
    def sanitize_filename(s: str) -> str:
        s = s.replace(' ', '_')
        return ''.join(c for c in s if c.isalnum() or c in ('-', '_', '.'))
    
    @staticmethod
    def ensure_dir(path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def clean_cache(output_dir: str) -> int:
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

class Utils:
    @staticmethod
    def pip_hint(pkg: str):
        st.error(f"Missing required package: `{pkg}`.\nInstall with: `pip install {pkg}`")
    
    @staticmethod
    def hash_line(line: str) -> str:
        return hashlib.sha256(line.strip().encode('utf-8')).hexdigest()[:12]
    
    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

class TTSService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.file_handler = FileHandler()
    
    def get_available_voices(self, mode: str) -> List[str]:
        return self.config.OPENAI_VOICES
    
    def get_default_voice(self, mode: str) -> str:
        voices = self.get_available_voices(mode)
        return voices[0] if voices else ''
    
    def get_default_rate(self, mode: str) -> float:
        return self.config.DEFAULT_RATE
    
    def is_openai_key_set(self) -> bool:
        return bool(os.environ.get('OPENAI_API_KEY', ''))
    
    def get_openai_key(self) -> Optional[str]:
        return os.environ.get('OPENAI_API_KEY', None)
    
    def synthesize_line(self, line: str, mode: str, voice: str, rate: float, 
                      output_dir: str, openai_api_key: Optional[str] = None, 
                      force_regen: bool = False) -> Tuple[Optional[str], str]:
        h = Utils.hash_line(f"{mode}|{voice}|{rate}|{line}")
        self.file_handler.ensure_dir(output_dir)
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

class StreamlitApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tts_service = TTSService(config)
        self.file_handler = FileHandler()
        self._setup_page()
    
    def _setup_page(self):
        st.set_page_config(page_title="Text-to-Speech App", layout="wide", page_icon="🔊")
        st.title("🔊 Text-to-Speech App")
    
    def render_sidebar(self) -> Tuple[str, str, float, str]:
        st.sidebar.header("Settings")
        
        mode = 'OpenAI API'
        
        voices = self.tts_service.get_available_voices(mode)
        def_voice = self.tts_service.get_default_voice(mode)
        voice = st.sidebar.selectbox(
            "Voice", voices, 
            index=0 if def_voice not in voices else voices.index(def_voice)
        )
        
        rate = st.sidebar.slider(
            "Speaking Rate",
            min_value=0.5,
            max_value=2.0,
            value=self.tts_service.get_default_rate(mode),
            step=0.05
        )
        
        output_dir = st.sidebar.text_input("Output Folder", value=self.config.DEFAULT_OUTPUT_DIR)
        
        if st.sidebar.button("🧹 Clean All Cache"):
            deleted = self.file_handler.clean_cache(output_dir)
            if deleted > 0:
                st.sidebar.success(f"Deleted {deleted} audio file(s) from cache.")
            else:
                st.sidebar.info("No audio files to delete in cache.")
        
        return mode, voice, rate, output_dir
    
    def render_file_upload(self) -> Tuple[List[str], Optional[Any]]:
        st.subheader("1. Upload a Text File (.txt or .md)")
        uploaded_file = st.file_uploader(
            "Choose a file", type=[e[1:] for e in self.config.SUPPORTED_EXTS]
        )
        
        lines = []
        if uploaded_file:
            lines = self.file_handler.load_file(uploaded_file)
            if lines:
                st.success(f"Loaded {len(lines)} lines from {uploaded_file.name}")
            else:
                st.error("No lines found in file.")
        
        return lines, uploaded_file
    
    def check_api_key(self) -> Optional[str]:
        openai_key = self.tts_service.get_openai_key()
        if not openai_key:
            st.error("OPENAI_API_KEY not set in environment. Please add it to your .env or environment variables.")
            st.stop()
        return openai_key

    def render_line_controls(self, lines: List[str], mode: str, voice: str, 
                           rate: float, output_dir: str, openai_key: str):
        if not lines:
            return None
            
        st.subheader("2. Text Preview & Line-by-Line Generation")
        
        self._init_session_state(lines)
        
        st.markdown("**Preview:** (Each line has its own Generate and Edit button)")
        
        placeholders = [st.empty() for _ in range(len(st.session_state['editable_lines']))]
        
        for idx, line in enumerate(st.session_state['editable_lines']):
            if not line.strip():
                continue
            
            self._render_single_line(idx, line, placeholders, mode, voice, rate, output_dir, openai_key)
        
        return placeholders
    
    def _init_session_state(self, lines: List[str]):
        if 'editable_lines' not in st.session_state or len(st.session_state['editable_lines']) != len(lines):
            st.session_state['editable_lines'] = lines.copy()
        if 'editing_line_idx' not in st.session_state:
            st.session_state['editing_line_idx'] = None
        if 'edit_buffer' not in st.session_state:
            st.session_state['edit_buffer'] = ''
        if 'last_generated_idx' not in st.session_state:
            st.session_state['last_generated_idx'] = None
    
    def _render_single_line(self, idx: int, line: str, placeholders: List, 
                          mode: str, voice: str, rate: float, output_dir: str, openai_key: str):
        is_last_generated = st.session_state.get('last_generated_idx') == idx
        
        with placeholders[idx].container():
            col1, col2, col3, col4, col5 = st.columns([8, 1, 1, 2, 3])
            
            with col1:
                self._render_line_text(idx, line, is_last_generated)
            
            with col2:
                if st.button("🪄", key=f"gen_{idx}"):
                    self._handle_generate(idx, line, col5, mode, voice, rate, output_dir, openai_key)
            
            with col3:
                if st.button("♻️", key=f"regen_{idx}"):
                    self._handle_regenerate(idx, line, col5, mode, voice, rate, output_dir, openai_key)
            
            with col4:
                if st.button("✏️", key=f"edit_{idx}", 
                           disabled=st.session_state['editing_line_idx'] is not None and 
                                   st.session_state['editing_line_idx'] != idx):
                    st.session_state['editing_line_idx'] = idx
                    st.session_state['edit_buffer'] = line
                    st.rerun()
            
            if st.session_state['editing_line_idx'] == idx:
                self._render_edit_controls(idx)
    
    def _render_line_text(self, idx: int, line: str, is_last_generated: bool):
        style = (
            "background-color: #e0f7fa; padding: 10px 16px; border-radius: 8px; "
            "margin-bottom: 8px; font-size: 1.1em; color: #222; font-family: sans-serif;"
            if is_last_generated else
            "padding: 10px 16px; margin-bottom: 8px; font-size: 1.1em; "
            "color: #eee; font-family: sans-serif;"
        )
        st.markdown(
            f"<div style='{style}'><b>{idx+1:03}</b> {line}</div>", 
            unsafe_allow_html=True
        )
    
    def _handle_generate(self, idx: int, line: str, col, mode: str, 
                        voice: str, rate: float, output_dir: str, openai_key: str):
        st.session_state['last_generated_idx'] = idx
        with col:
            self._process_audio_generation(line, mode, voice, rate, output_dir, openai_key)
    
    def _handle_regenerate(self, idx: int, line: str, col, mode: str, 
                          voice: str, rate: float, output_dir: str, openai_key: str):
        st.session_state['last_generated_idx'] = idx
        with col:
            self._process_audio_generation(line, mode, voice, rate, output_dir, openai_key, force_regen=True)
    
    def _process_audio_generation(self, line: str, mode: str, voice: str, rate: float, 
                                output_dir: str, openai_key: str, force_regen: bool = False):
        status_text = "Regenerating..." if force_regen else "Generating..."
        st.info(status_text)
        
        audio_path, status = self.tts_service.synthesize_line(
            line, mode, voice, rate, output_dir, openai_key, force_regen
        )
        
        if status in ('success', 'cached') and audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            success_text = "✅ Success (Regenerated)" if force_regen and status == 'success' else (
                "✅ Success" if status == 'success' else "♻️ Cached"
            )
            st.success(success_text)
            with open(audio_path, 'rb') as f:
                st.audio(f.read(), format='audio/mp3')
        else:
            st.error("❌ Error: No audio file generated.")
            if audio_path:
                st.warning(f"Audio file not found or empty: {audio_path}")
    
    def _render_edit_controls(self, idx: int):
        new_text = st.text_input(
            f"Edit line {idx+1}", 
            value=st.session_state['edit_buffer'], 
            key=f"edit_input_{idx}"
        )
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

    def render_batch_controls(self, placeholders: List, mode: str, voice: str, 
                            rate: float, output_dir: str, openai_key: str):
        st.subheader("3. Batch Utilities")
        
        if 'batch_running' not in st.session_state:
            st.session_state['batch_running'] = False
        
        if st.button("🚀 Generate All", disabled=st.session_state['batch_running']):
            self._run_batch_generation(placeholders, mode, voice, rate, output_dir, openai_key)
    
    def _run_batch_generation(self, placeholders: List, mode: str, voice: str, 
                            rate: float, output_dir: str, openai_key: str):
        st.session_state['batch_running'] = True
        st.session_state['last_generated_idx'] = None
        
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
                    self._process_audio_generation(line, mode, voice, rate, output_dir, openai_key)
            
            progress.progress((idx + 1) / n)
            st.session_state['last_generated_idx'] = idx
        
        st.session_state['batch_running'] = False
        st.toast("Batch generation complete!", icon="🎉")
        st.rerun()

    def render_download_section(self, uploaded_file, mode: str, voice: str, 
                              rate: float, output_dir: str):
        st.subheader("4. Download All Audio")
        
        zip_buffer = io.BytesIO()
        script_name = Path(uploaded_file.name).stem if uploaded_file else "script"
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, line in enumerate(st.session_state['editable_lines']):
                if not line.strip():
                    continue
                
                h = Utils.hash_line(f"{mode}|{voice}|{rate}|{line}")
                audio_path = str(Path(output_dir) / f"{h}.mp3")
                
                if os.path.exists(audio_path):
                    file_name = f"{script_name}_{idx+1:03}.mp3"
                    zip_file.write(audio_path, file_name)
        
        zip_buffer.seek(0)
        
        if zip_buffer.getbuffer().nbytes > 0:
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_buffer,
                file_name=f"{script_name}_audio_{Utils.get_timestamp()}.zip",
                mime="application/zip",
            )
        else:
            st.info("No audio files have been generated yet.")
    
    def render_logs_section(self):
        with st.expander("Show Logs"):
            try:
                with open(self.config.LOG_FILE, 'r') as f:
                    logs = f.read()[-5000:]
                st.code(logs, language='log')
            except Exception:
                st.info("No logs yet.")
    
    def run(self):
        mode, voice, rate, output_dir = self.render_sidebar()
        lines, uploaded_file = self.render_file_upload()
        openai_key = self.check_api_key()
        
        placeholders = self.render_line_controls(lines, mode, voice, rate, output_dir, openai_key)
        
        if lines and placeholders:
            self.render_batch_controls(placeholders, mode, voice, rate, output_dir, openai_key)
            self.render_download_section(uploaded_file, mode, voice, rate, output_dir)
        
        self.render_logs_section()
        
        if 'openai' in missing:
            Utils.pip_hint('openai')

# --- Main Application ---
def main():
    app = StreamlitApp(config)
    app.run()

if __name__ == "__main__":
    main()
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