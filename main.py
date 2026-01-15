# Main Program, run this

import warnings

# 24/10 2025 All working but a few duplicated procedures
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*CUDA capability.*")
warnings.filterwarnings("ignore", message=".*overflow encountered.*")

# === SIMPLE OLLAMA CHECK & START ===
import subprocess

import time
import requests
import os


def check_ollama_running():
    """Check if Ollama API is responding."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_ollama():
    """Try to start Ollama on Windows without debug output."""
    print("[Ollama] Starting Ollama server (quiet mode)...")

    # Common Windows paths
    possible_paths = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Ollama", "ollama.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Ollama", "ollama.exe"),
        "C:\\Program Files\\Ollama\\ollama.exe",
        "C:\\Program Files (x86)\\Ollama\\ollama.exe",
        os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                # FIX: Use 'ollama serve' not just 'ollama'
                command = [path, "serve"]

                # Start minimized
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                # START: Suppress ALL Ollama debug output
                subprocess.Popen(
                    command,
                    startupinfo=startupinfo,
                    stdout=subprocess.DEVNULL,  # Suppress standard output
                    stderr=subprocess.DEVNULL,  # Suppress error output
                    creationflags=subprocess.CREATE_NO_WINDOW  # No console window
                )
                # END: Changes

                print(f"[Ollama] Started from: {path}")
                return True
            except Exception as e:
                print(f"[Ollama] Failed to start: {e}")

    print("[Ollama] Executable not found. Please install Ollama from https://ollama.com/")
    return False


def wait_for_ollama(timeout=15):  # Reduced from 30 to 15 seconds
    """Wait for Ollama to start."""
    print("[Ollama] Waiting for Ollama to start...")
    start_time = time.time()

    for i in range(timeout):
        if check_ollama_running():
            elapsed = time.time() - start_time
            print(f"[Ollama] ✓ Ollama running! ({elapsed:.1f}s)")
            return True

        # Only print every 3 seconds (less spammy)
        if i % 3 == 0 and i > 0:
            print(f"[Ollama] Still starting... ({i}s)")

        time.sleep(1)

    print(f"[Ollama] ⚠️ Timed out after {timeout}s (but continuing)")
    return False  # Still continue - sometimes API is slow






# --- MAIN CHECK ---
if __name__ == "__main__":
    # First check if Ollama is already running
    if not check_ollama_running():
        print("[Ollama] Ollama is not running")

        # Try to start it
        if start_ollama():
            # Wait for it to start
            if not wait_for_ollama():
                print("[WARNING] Ollama may not be ready. Continuing anyway...")
        else:
            print("[WARNING] Could not start Ollama. Some features may not work.")
            print("You can start Ollama manually by:")
            print("1. Opening the Ollama app")
            print("2. Or running 'ollama serve' in Command Prompt")
    else:
        print("[Ollama] ✓ Already running")

# === END OLLAMA CHECK ===



from router import CommandRouter
# Also set numpy to ignore specific warnings
import numpy as np
# ... your existing imports ...

from Avatars import CircleAvatarWindow, RectAvatarWindow, RectAvatarWindow2, RadialPulseAvatar, FaceRadialAvatar

# ... rest of your imports ...
np.seterr(over='ignore', under='ignore', invalid='ignore')
from dotenv import load_dotenv
from echo_engine import EchoEngine, EchoWindow

load_dotenv()
from tkinter import ttk
import os, json, threading, time, re, math
from collections import deque
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from io import BytesIO
from PIL import Image, ImageTk
import matplotlib
from status_light_window import StatusLightWindow

# Import the external LaTeX window
from latex_window import LatexWindow
# === Main App Class ===
from app_main import App

from web_search_window import WebSearchWindow

# This uses one model and handles images as well. It also has personalities. e/g Butler Mode,Teacher, Holly etc. You can make your own.
# Personalities are stored in Jason files in the Personalities folder. router.py has all the commands available.
# To make personalities work well I recommend using some of the Cereproc voices you can buy. Or use OneCore modern Microsoft voices.
# OneCore voices are not always available on sone windows versions. They are on Win 11 Professional.
# Look for the Json file  called config.json
#
# You will need only one vision Ollama model.
#   Minstral-3 is a good option though there are a few others like llava-llama3 or qwen2.5-vl - Alibaba's multimodal model
#  You can select the model from the main program ie main.py from the combo box. Note previous versions of this program used two models but not now.

# To search the internet you will need a key from Brave Search API https://brave.com/search/api/
# It's free for a fair number of searches but after that you need to pay
# Put that key in the file .env or the program won't work ie won't search.
# Try asking it "what is the latest news in New Zealand , and it will find that news  and summarise it.
# Warning, AI models make mistakes so check any answers

# A recent addition is that plot answers to some basic problems with polynomials and simple expressions
# Tom Moir Dec  2025


matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter.font as tkfont
from tkinter.scrolledtext import ScrolledText
import base64, tempfile, requests
import queue
import httpx
import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urljoin
import re
import winsound
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import cv2  # pip install opencv-python
except Exception:
    cv2 = None

# Optional drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # pip install tkinterdnd2
except Exception:
    DND_FILES = None
    TkinterDnD = None

try:
    import torch

    print("[GPU torch]", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except Exception:
    pass

try:
    import ctranslate2

    print("[CT2 CUDA types]", ctranslate2.get_supported_compute_types("cuda"))
except Exception:
    pass

# External modules you provide
from audio_io import list_input_devices, VADListener
from asr_whisper import ASR
from qwen_llmSearch2 import QwenLLM
from pydub import AudioSegment



# Import the math speech converter
from Speak_Maths import MathSpeechConverter

# Create a global instance
math_speech_converter = MathSpeechConverter()


# ===  ITEM DATACLASS ===
@dataclass
class Item:
    title: str
    url: str
    snippet: str = ""
    pubdate: Optional[str] = None
    summary: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)


# === END DATACLASS ===


# ---------- Config ----------
def load_cfg():
    import os, json
    env_path = os.environ.get("APP_CONFIG")
    if env_path and os.path.exists(env_path):
        path = env_path
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        c1 = os.path.join(base, "config.json")
        c2 = os.path.join(base, "config.example.json")
        c3 = "config.json" if os.path.exists("config.json") else None
        c4 = "config.example.json" if os.path.exists("config.example.json") else None
        path = next((p for p in (c1, c2, c3, c4) if p and os.path.exists(p)), None)

    if not path:
        raise FileNotFoundError("No config.json or config.example.json found")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[cfg] loaded: {os.path.abspath(path)}")
    sp = cfg.get("system_prompt", cfg.get("qwen_system_prompt", ""))
    print(f"[cfg] system_prompt present: {bool(sp)} (len={len(sp) if isinstance(sp, str) else 'n/a'})")
    return cfg


# ---------- TTS cleaner ----------
def clean_for_tts(text: str, speak_math: bool = True) -> str:
    """
    TTS cleaner that removes Markdown formatting added by AIs.
    LaTeX math is already handled by math_speech_converter.

    Args:
        text: Input text
        speak_math: If True, convert math to speech; if False, say "equation"

    Returns:
        Text ready for TTS with formatting removed
    """
    if not text:
        return ""

    # Let the math converter handle any LaTeX/math first
    cleaned_text = math_speech_converter.make_speakable_text(text, speak_math=speak_math)

    # Now remove only Markdown/formatting characters that AIs add

    # Remove ALL markdown formatting patterns:

    # Markdown headers (###, ##, #)
    cleaned_text = re.sub(r'^#{1,6}\s*', '', cleaned_text, flags=re.MULTILINE)

    # Bullet points (*, -, +, •) at line start
    cleaned_text = re.sub(r'^[\s]*[\*\-+\•]\s+', '', cleaned_text, flags=re.MULTILINE)

    # Numbered lists (1., 2., etc.) at line start
    cleaned_text = re.sub(r'^[\s]*\d+\.\s+', '', cleaned_text, flags=re.MULTILINE)

    # Bold and italic (**text**, *text*)
    # Use non-greedy matching to handle multiple in one line
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\*(.*?)\*', r'\1', cleaned_text)

    # Underline (__text__)
    cleaned_text = re.sub(r'__(.*?)__', r'\1', cleaned_text)

    # Strikethrough (~~text~~)
    cleaned_text = re.sub(r'~~(.*?)~~', r'\1', cleaned_text)

    # Inline code (`text`)
    cleaned_text = re.sub(r'`(.*?)`', r'\1', cleaned_text)

    # Remove any remaining formatting chars that surround words
    # This catches cases the patterns above might miss
    cleaned_text = re.sub(r'([#*_`])([A-Za-z0-9]+)(?=\1)', r'\2', cleaned_text)

    # Clean up: remove multiple spaces, trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()


# Add another cleaning method for odd output from Deepseek
def clean_model_output(text: str) -> str:
    """
    Clean model-specific formatting artifacts from AI responses.
    """
    if not text:
        return ""

    cleaned = text

    print(f"🔧 [CLEANER] Input: {repr(text[:100])}")

    # Remove ALL variants of DeepSeek tokens:
    # <|im_end|>
    # <|im_end>|<think>
    # <|im_start|>
    # Any other <|...|> patterns

    # Method 1: Remove everything after any end token pattern
    end_patterns = [
        '<|im_end|>',
        '<|im_end>|<think>',
        '<|end|>',
        '<|endoftext|>'
    ]

    for pattern in end_patterns:
        if pattern in cleaned:
            parts = cleaned.split(pattern)
            cleaned = parts[0].strip()
            print(f"🔧 [CLEANER] Split by pattern: {pattern}")
            break  # Stop after first match

    # Method 2: Remove any remaining individual tokens
    tokens_to_remove = [
        '<|im_start|>', '<|im_end|>', '<|end|>', '<|endoftext|>',
        '<|im_end>|<think>', '<|think|>', '<|system|>', '<|user|>', '<|assistant|>'
    ]

    for token in tokens_to_remove:
        cleaned = cleaned.replace(token, '')

    # Method 3: Aggressive regex for any <|...|> or <|...> patterns
    import re
    cleaned = re.sub(r'<\|[^>]*(?:\|>|>)', '', cleaned)

    # Method 4: Remove LaTeX document wrappers
    if '\\documentclass' in cleaned:
        if '\\begin{document}' in cleaned:
            parts = cleaned.split('\\begin{document}')
            if len(parts) > 1:
                cleaned = parts[1].strip()
        if '\\end{document}' in cleaned:
            cleaned = cleaned.split('\\end{document}')[0].strip()

    # Final cleanup
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    print(f"🔧 [CLEANER] Output: {repr(cleaned[:100])}")
    print(f"🔧 [CLEANER] Success: {not any(token in cleaned for token in tokens_to_remove)}")

    return cleaned




# -------- Run --------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.mainloop()
