# app_main.py
# CORE PYTHON IMPORTS FIRST
import os
import trafilatura
import json
import threading
import time
import queue
import re
import tempfile
import base64
import webbrowser
from collections import deque
from datetime import datetime, timedelta
from urllib.parse import urljoin
from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup
# TKINTER IMPORTS
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
#################
#Brave search

# Add these imports at the top of app_main.py
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Add debug logging
print(f"[ENV] .env loaded from: {os.path.abspath('.env') if os.path.exists('.env') else 'NOT FOUND'}")
print(f"[ENV] BRAVE_KEY exists: {'BRAVE_KEY' in os.environ}")
if 'BRAVE_KEY' in os.environ:
    key = os.environ['BRAVE_KEY']
    print(f"[ENV] BRAVE_KEY value: {key[:4]}...{key[-4:] if len(key) > 8 else ''}")
else:
    print("[ENV] WARNING: BRAVE_KEY not found in environment. Edit .env file with your Brave key")


##
# SCIENTIFIC/NUMERICAL IMPORTS
import numpy as np

# PLotter for graphs - intelligent version with auto scaling
try:
    from plotter import Plotter
except ImportError as e:
    print(f"Warning: Could not import Plotter: {e}")
    Plotter = None

# EDGE TTS IMPORT
try:
    import edge_tts
    import asyncio

    EDGE_TTS_AVAILABLE = True
    print("[tts] ✅ edge-tts available")
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("[tts] ⚠️ edge-tts not installed - run: pip install edge-tts")
# SandBox
from code_window import CodeWindow

# AUDIO/IMAGE IMPORTS
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageTk

# Add these for ImageWindow
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # pip install tkinterdnd2
except Exception:
    DND_FILES = None
    TkinterDnD = None

try:
    import cv2  # pip install opencv-python
except Exception:
    cv2 = None

# NETWORK IMPORTS
import requests
import httpx
# Import the math speech converter
from Speak_Maths import MathSpeechConverter

# Create a global instance
math_speech_converter = MathSpeechConverter()

from web_search_window import WebSearchWindow

# EXTERNAL MODULE IMPORTS
try:
    from audio_io import list_input_devices, VADListener
    from asr_whisper import ASR
    from qwen_llmSearch2 import QwenLLM
    from pydub import AudioSegment
    from Speak_Maths import MathSpeechConverter
    from router import CommandRouter
    from Avatars import CircleAvatarWindow, RectAvatarWindow, RectAvatarWindow2, RadialPulseAvatar, FaceRadialAvatar, \
        StringGridAvatar, TextureMappedSphere, Hal9000Avatar
    from latex_window import LatexWindow
    from status_light_window import StatusLightWindow
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# ECHO ENGINE IMPORT (must come after all the imports it needs)
from echo_engine import EchoEngine, EchoWindow


# === HELPER FUNCTIONS ===
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


def clean_model_output(text: str) -> str:
    """Clean model-specific formatting artifacts from AI responses."""
    if not text:
        return ""

    cleaned = text
    print(f"🔧 [CLEANER] Input: {repr(text[:100])}")

    # Remove ALL variants of DeepSeek tokens
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
            break

    # Remove any remaining individual tokens
    tokens_to_remove = [
        '<|im_start|>', '<|im_end|>', '<|end|>', '<|endoftext|>',
        '<|im_end>|<think>', '<|think|>', '<|system|>', '<|user|>', '<|assistant|>'
    ]

    for token in tokens_to_remove:
        cleaned = cleaned.replace(token, '')

    # Aggressive regex for any <|...|> or <|...> patterns
    cleaned = re.sub(r'<\|[^>]*(?:\|>|>)', '', cleaned)

    # Remove LaTeX document wrappers
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


def purge_temp_files(folder="out"):
    """
    Remove ALL temporary files (images AND audio) on startup.
    Only keeps essential/config files.
    """
    try:
        if not os.path.isdir(folder):
            return

        # Files to ALWAYS KEEP (never delete these)
        essential_files = [
            "config.json", "config.example.json", ".gitkeep",
            "beep.mp3",  # If you have a beep file
            "personality_backup.json"  # If you have this
        ]

        files_deleted = 0
        for name in os.listdir(folder):
            # Skip essential files
            if name in essential_files:
                continue

            low = name.lower()
            filepath = os.path.join(folder, name)

            # === DELETE ALL TEMPORARY FILES ===

            # 1. Delete ALL temporary images
            if low.endswith(".png"):
                try:
                    os.remove(filepath)
                    files_deleted += 1
                    print(f"[cleanup] Deleted image: {name}")
                except Exception:
                    pass

            # 2. Delete ALL WAV files (except maybe recordings you want to keep)
            elif low.endswith(".wav"):
                # Option A: Delete ALL WAV files
                try:
                    os.remove(filepath)
                    files_deleted += 1
                    print(f"[cleanup] Deleted WAV: {name}")
                except Exception:
                    pass

                # Option B: Only delete temporary-looking WAV files
                # if any(keyword in low for keyword in [
                #     "temp", "edge", "sapi", "search", "reply", "status"
                # ]):
                #     try:
                #         os.remove(filepath)
                #         files_deleted += 1
                #         print(f"[cleanup] Deleted temp WAV: {name}")
                #     except Exception:
                #         pass

            # 3. Delete ALL MP3 files
            elif low.endswith(".mp3"):
                try:
                    os.remove(filepath)
                    files_deleted += 1
                    print(f"[cleanup] Deleted MP3: {name}")
                except Exception:
                    pass

        # Create empty 'out' directory if it's now empty
        if not os.listdir(folder):
            print(f"[cleanup] {folder} is now empty")
        else:
            print(f"[cleanup] {folder} still has: {os.listdir(folder)}")

        if files_deleted > 0:
            print(f"[startup] Cleaned {files_deleted} temporary files")
        else:
            print("[startup] No temporary files to clean")

    except Exception as e:
        print(f"[startup] cleanup error: {e}")


        ###################

    # === BRAVE SEARCH METHODS ===

    # === SEARCH METHODS ===

    def brave_search(self, query: str, count: int = 6):
        brave_key = os.getenv("BRAVE_KEY")
        if not brave_key:
            raise RuntimeError("No BRAVE_KEY found in environment")
        # === Logs we are searching the Internet ===
        self.logln(f"[SEARCH] 🚀 Calling Brave API: '{query}'")

        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": brave_key, "User-Agent": "LocalAI-ResearchBot/1.0"}
        params = {"q": query, "count": count}

        with httpx.Client(timeout=25.0, headers=headers) as client:
            r = client.get(endpoint, params=params)
            r.raise_for_status()
            data = r.json()

        out = []
        for w in (data.get("web", {}) or {}).get("results", []):
            out.append(
                Item(title=w.get("title", "No title"), url=w.get("url", ""), snippet=w.get("description", "")))
            # === check what its searching  ===
            self.logln(f"[BRAVE API] ✅ Found {len(out)} results for '{query}'")

        return out

    def polite_fetch(self, url: str):
        headers = {"User-Agent": "LocalAI-ResearchBot/1.0"}
        try:
            with httpx.Client(timeout=25.0, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.text
        except Exception:
            return None

    def extract_readable(self, html: str, url: str = None):
        text = trafilatura.extract(html, url=url, include_links=False, include_formatting=False)
        return text or ""

    def guess_pubdate(self, html: str):
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return None

        metas = [
            ("property", "article:published_time"), ("property", "og:published_time"),
            ("property", "og:updated_time"), ("name", "pubdate"), ("name", "publication_date"),
            ("name", "date"), ("name", "dc.date"), ("name", "dc.date.issued"),
            ("name", "sailthru.date"), ("itemprop", "datePublished"), ("itemprop", "dateModified"),
        ]

        for key, val in metas:
            tag = soup.find("meta", attrs={key: val})
            if tag and tag.get("content"):
                return tag["content"]

        t = soup.find("time")
        if t and (t.get("datetime") or (t.text and t.text.strip())):
            return t.get("datetime") or t.text.strip()
        return None

    def summarise_for_ai_search(self, text: str, url: str, pubdate: str):
        """Enhanced summarization that preserves practical information"""
        text = text[:18000]

        # Enhanced date context
        if pubdate:
            date_context = f"PUBLICATION DATE: {pubdate}\n"
        else:
            import re
            date_matches = re.findall(
                r'\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 20\d{2})\b',
                text[:3000])
            date_context = f"MENTIONED DATES: {', '.join(date_matches[:3])}\n" if date_matches else ""

        # DETECT QUERY TYPE AND ADAPT SUMMARIZATION
        query_lower = getattr(self, '_last_search_query', '').lower()

        # Flight/travel related queries
        if any(keyword in query_lower for keyword in ['flight', 'fly', 'airline', 'airport', 'travel to']):
            summary_prompt = (
                "Extract COMPLETE flight information with these details:\n\n"
                "## FLIGHT INFORMATION\n"
                "- Airline names and flight numbers\n"
                "- Departure and arrival airports (with codes if available)\n"
                "- Departure and arrival times/dates\n"
                "- Flight duration\n"
                "- Prices and fare classes\n"
                "- Stopovers/layovers\n"
                "- Booking links or airline websites\n\n"
                "## TRAVEL DETAILS\n"
                "- Airport locations and terminals\n"
                "- Booking requirements\n"
                "- Baggage information\n"
                "- Recent deals or promotions\n\n"
                "Include ALL specific numbers, times, prices, and codes. Be very detailed about schedules and availability.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        # Business/location queries
        elif any(keyword in query_lower for keyword in
                 ['address', 'location', 'where is', 'hours', 'contact', 'phone', 'email']):
            summary_prompt = (
                "EXTRACT ONLY INFORMATION EXPLICITLY STATED IN THE TEXT. NEVER CREATE PLACEHOLDERS OR INVENT INFORMATION.\n\n"
                "CRITICAL RULES:\n"
                "1. ONLY include information that appears VERBATIM in the source text\n"
                "2. NEVER use brackets [ ], parentheses ( ), or placeholder text\n"
                "3. If a website is mentioned, copy the EXACT URL\n"
                "4. If information is missing, OMIT that line entirely\n"
                "5. Do NOT create template responses\n\n"
                "EXTRACTED INFORMATION (ONLY IF FOUND):\n"
                "- Business Name: [copy exact name if found]\n"
                "- Address: [copy exact address if found]\n"
                "- Phone: [copy exact phone number if found]\n"
                "- Email: [copy exact email if found]\n"
                "- Website: [copy exact URL if found]\n"
                "- Hours: [copy exact hours if found]\n\n"
                "EXAMPLES - WRONG:\n"
                "❌ Address: [Address may vary]\n"
                "❌ Phone: [Phone number may vary]  \n"
                "❌ Website: [Website Link]\n"
                "❌ Website: [Website URL if available]\n\n"
                "EXAMPLES - CORRECT:\n"
                "✅ Address: 456 Northshore Road, Unit 2, Glenfield 0678\n"
                "✅ Phone: +64 9 483 5555\n"
                "✅ Website: https://www.serenityspa.co.nz\n"
                "✅ Website: www.serenityspa.com\n"
                "✅ (omit Website line if no URL found)\n\n"
                "If the text contains '456 Glenfield Road, Unit 2, Glenfield 0678' and '+64 9 483 5555' but NO website, output:\n"
                "Address: 456 Glenfield Road, Unit 2, Glenfield 0678\n"
                "Phone: +64 9 483 5555\n\n"
                "DO NOT INVENT WEBSITE INFORMATION. If no website is found, omit the Website line completely.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )


        # Product/service queries
        elif any(keyword in query_lower for keyword in ['price', 'cost', 'buy', 'purchase', 'deal', 'sale']):
            summary_prompt = (
                "Extract COMPLETE product/service information:\n\n"
                "## PRICING & AVAILABILITY\n"
                "- Exact prices and currency\n"
                "- Model numbers/specifications\n"
                "- Availability status\n"
                "- Seller/retailer information\n"
                "- Shipping costs and delivery times\n"
                "- Return policies\n\n"
                "## PRODUCT DETAILS\n"
                "- Features and specifications\n"
                "- Dimensions/sizes\n"
                "- Colors/options available\n"
                "- Warranty information\n\n"
                "Include ALL pricing, specifications, and purchase details. Be very specific about numbers and options.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        # Weather queries
        elif any(keyword in query_lower for keyword in
                 ['weather', 'forecast', 'temperature', 'rain', 'snow', 'humidity']):
            summary_prompt = (
                "Extract COMPLETE weather forecast information:\n\n"
                "## CURRENT CONDITIONS\n"
                "- Temperature and feels-like temperature\n"
                "- Weather description (sunny, rainy, etc.)\n"
                "- Humidity, wind speed and direction\n"
                "- Precipitation chances\n"
                "- Air quality and UV index\n\n"
                "## FORECAST\n"
                "- Hourly and daily forecasts\n"
                "- High/low temperatures\n"
                "- Severe weather alerts\n"
                "- Sunrise/sunset times\n\n"
                "## LOCATION DETAILS\n"
                "- Specific city/region\n"
                "- Geographic details if available\n"
                "- Timezone information\n\n"
                "Include ALL numerical weather data, times, and location specifics.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        else:
            # General comprehensive summary (for news, general info, etc.)
            summary_prompt = (
                "Create a COMPREHENSIVE summary that PRESERVES practical information:\n\n"
                "## ESSENTIAL DETAILS\n"
                "- Full names of businesses, people, organizations\n"
                "- Complete addresses, phone numbers, contact information\n"
                "- Prices, costs, financial figures\n"
                "- Dates, times, schedules\n"
                "- Locations, coordinates, directions\n"
                "- Website URLs, email addresses\n\n"
                "## KEY INFORMATION\n"
                "- Main facts and findings\n"
                "- Important numbers and statistics\n"
                "- Recent developments\n"
                "- Contact methods\n\n"
                "## ADDITIONAL CONTEXT\n"
                "- Background information\n"
                "- Related services or options\n"
                "- User reviews or ratings if available\n\n"
                "CRITICAL: NEVER omit addresses, phone numbers, prices, or contact information. Include them verbatim.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        try:
            payload = {
                "model": "qwen2.5:7b-instruct",
                "prompt": summary_prompt,
                "stream": False,
                "temperature": 0.1,  # Lower temperature for more factual accuracy
                "max_tokens": 1200  # More tokens for detailed information
            }

            with httpx.Client(timeout=75.0) as client:
                r = client.post("http://localhost:11434/api/generate", json=payload)
                r.raise_for_status()
                response = r.json().get("response", "").strip()

                # Enhanced fallback for better information extraction
                if len(response) < 100 or "no information" in response.lower():
                    return self._extract_practical_information(text[:12000], query_lower)

                return response

        except Exception as e:
            return self._extract_practical_information(text[:10000], query_lower)

    def _extract_practical_information(self, text: str, query_type: str) -> str:
        """Enhanced fallback extraction focusing on practical information"""
        import re

        sections = []

        # Enhanced address extraction
        address_patterns = [
            # Standard street addresses
            r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Highway|Hwy)\.?\s*(?:#\s*\d+)?\s*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b',
            # Basic address format
            r'\b\d+\s+[\w\s]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court),\s*[\w\s]+,\s*[A-Z]{2}\b',
            # PO Boxes
            r'\b(?:P\.?O\.?\s*Box|PO Box|P O Box)\s+\d+[^.!?]*',
        ]

        addresses = []
        for pattern in address_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(found)

        # Filter out obviously fake or placeholder addresses
        real_addresses = []
        for addr in addresses:
            addr_lower = addr.lower()
            # Skip placeholder text
            if any(placeholder in addr_lower for placeholder in
                   ['address may vary', 'varies', 'please contact', 'call for', 'not available']):
                continue
            # Skip if it's just a city/state without street
            if re.match(r'^[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}', addr) and not re.search(r'\d+', addr):
                continue
            real_addresses.append(addr.strip())

        if real_addresses:
            sections.append("## ADDRESSES FOUND")
            sections.extend([f"- {addr}" for addr in set(real_addresses)[:3]])
        # Extract website URLs (more comprehensive)
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+\.[a-z]{2,}',
            r'[a-z0-9.-]+\.[a-z]{2,}/[^\s<>"{}|\\^`\[\]]*',
        ]

        urls = []
        for pattern in url_patterns:
            urls.extend(re.findall(pattern, text, re.IGNORECASE))

        # Filter and clean URLs
        clean_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?)]+$', '', url)
            # Skip common false positives
            if any(bad in url.lower() for bad in ['example.com', 'website.com', 'yourwebsite', 'domain.com']):
                continue
            # Ensure it looks like a real URL
            if '.' in url and len(url) > 8:
                # Add http:// if missing for www URLs
                if url.startswith('www.') and not url.startswith('http'):
                    url = 'https://' + url
                clean_urls.append(url)

        if clean_urls:
            sections.append("\n## WEBSITES")
            sections.extend([f"- {url}" for url in set(clean_urls)[:3]])

        # Extract phone numbers
        phone_pattern = r'(\+?\d{1,2}?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            sections.append(f"\n## PHONE NUMBERS: {', '.join(set(phones)[:3])}")

        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            sections.append(f"\n## EMAIL ADDRESSES: {', '.join(set(emails)[:3])}")

        # Extract prices and costs
        prices = re.findall(r'\$?\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|USD|€|£|¥)?', text)
        if prices:
            sections.append(f"\n## PRICES MENTIONED: {', '.join(set(prices)[:8])}")

        # Flight-specific extraction
        if 'flight' in query_type:
            flight_info = re.findall(r'[A-Z]{2}\d+\s+.*?(?:\d{1,2}:\d{2}|AM|PM)', text)
            if flight_info:
                sections.append("\n## FLIGHT DETAILS")
                sections.extend([f"- {info}" for info in flight_info[:5]])

        # Business hours
        hours = re.findall(
            r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*.*?\d{1,2}:\d{2}\s*(?:AM|PM)?.*?\d{1,2}:\d{2}\s*(?:AM|PM)?', text,
            re.IGNORECASE)
        if hours:
            sections.append("\n## BUSINESS HOURS")
            sections.extend([f"- {hour}" for hour in hours[:3]])

        # Weather data extraction
        if 'weather' in query_type:
            temps = re.findall(r'\b\d{1,3}°?F?\b', text)
            if temps:
                sections.append(f"\n## TEMPERATURES: {', '.join(set(temps)[:6])}")

        # If we found practical information, return it
        if sections:
            return "\n".join(sections)
        else:
            # Return meaningful content lines as fallback
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 30]
            return "## KEY INFORMATION EXTRACTED\n" + "\n".join([f"- {line}" for line in lines[:10]])

    def _extract_detailed_news(self, text: str) -> str:
        """Enhanced fallback extraction with more structure"""
        import re

        # Extract key information with more context
        sections = []

        # Headlines and key sentences
        sentences = re.split(r'[.!?]+', text)
        key_sentences = []

        important_indicators = [
            'announced', 'reported', 'confirmed', 'revealed', 'disclosed',
            'investigation', 'charged', 'arrested', 'settlement', 'agreement',
            'election', 'resigned', 'appointed', 'launched', 'released',
            'fire', 'accident', 'killed', 'injured', 'missing', 'found',
            'storm', 'flood', 'earthquake', 'weather', 'forecast', 'temperature',
            'budget', 'funding', 'cost', 'price', 'investment'
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 25 and
                    any(indicator in sentence.lower() for indicator in important_indicators)):
                key_sentences.append(sentence)
                if len(key_sentences) >= 12:
                    break

        if key_sentences:
            sections.append("## KEY DEVELOPMENTS")
            sections.extend([f"- {s}" for s in key_sentences[:10]])

        # Extract numbers and statistics
        numbers = re.findall(r'\b(\$?[£€]?\d+(?:,\d+)*(?:\.\d+)?[%€£$]?(?:\s*(?:million|billion|thousand))?)\b',
                             text[:5000])
        if numbers:
            sections.append(f"\n## KEY NUMBERS: {', '.join(set(numbers[:8]))}")

        # Extract locations
        locations = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text[:3000])
        unique_locs = list(
            set([loc for loc in locations if len(loc) > 3 and loc not in ['The', 'This', 'That', 'There', 'Here']]))
        if unique_locs:
            sections.append(f"\n## MENTIONED LOCATIONS: {', '.join(unique_locs[:6])}")

        if sections:
            return "\n".join(sections)
        else:
            # Last resort: return structured excerpt
            lines = text.split('\n')
            meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 40][:8]
            return "## CONTENT OVERVIEW\n" + "\n".join([f"- {line}" for line in meaningful_lines])

    def summarise_with_qwen(self, text: str, url: str, pubdate: str):
        text = text[:20000]  # Limit text length
        pd_line = f"(Publish/Update date: {pubdate})\n" if pubdate else ""

        # FIRST PASS: Extract mathematical content specifically
        math_prompt = (
            "Extract ALL mathematical equations, formulas, and technical content from the following text. "
            "Preserve them exactly in their original LaTeX format ($$...$$, \\[...\\], $...$, etc.).\n"
            "Include:\n"
            "- All equations and formulas\n"
            "- Mathematical expressions\n"
            "- Chemical formulas\n"
            "- Code snippets\n"
            "- Important technical definitions\n"
            "Output the mathematical/technical content exactly as found, without summarization.\n"
            f"{pd_line}"
            f"Source: {url}\n\nCONTENT:\n{text[:10000]}"  # Use first 10k chars for math extraction
        )

        # SECOND PASS: Create a summary that REFERENCES the preserved math
        summary_prompt = (
            "Create a comprehensive summary (10-15 bullet points) that includes:\n"
            "- Key findings and conclusions\n"
            "- Important data points and results\n"
            "- References to mathematical content (say 'see equation X' or 'the formula shows')\n"
            "- Main arguments and evidence\n"
            "- Do NOT remove technical details - include them in context\n"
            "- Preserve specific numbers, measurements, and quantitative results\n"
            "Be detailed enough to be useful for technical analysis.\n"
            f"{pd_line}"
            f"Source: {url}\n\nCONTENT:\n{text}"
        )

        try:
            # Get mathematical content
            math_content = self.qwen.generate(math_prompt)

            # Get comprehensive summary
            summary = self.qwen.generate(summary_prompt)

            # Combine both with clear separation
            combined_result = f"MATHEMATICAL CONTENT:\n{math_content}\n\nSUMMARY:\n{summary}"

            return combined_result

        except Exception:
            # Fallback: Use a more math-friendly single prompt
            fallback_prompt = (
                "Create a DETAILED technical summary (12-18 bullet points) that PRESERVES all mathematical content.\n"
                "CRITICAL: Keep ALL equations, formulas, and LaTeX expressions exactly as they appear.\n"
                "Include:\n"
                "- Complete equations in $$...$$, \\[...\\], $...$ format\n"
                "- Mathematical proofs and derivations\n"
                "- Chemical formulas and reactions\n"
                "- Code snippets and algorithms\n"
                "- Quantitative results with exact numbers\n"
                "- Do NOT simplify or remove technical details\n"
                "- Focus on preserving the mathematical richness of the content\n"
                f"{pd_line}"
                f"Source: {url}\n\nCONTENT:\n{text}"
            )
            try:
                payload = {"model": "qwen2.5:7b-instruct", "prompt": fallback_prompt, "stream": False}
                with httpx.Client(timeout=90.0) as client:
                    r = client.post("http://localhost:11434/api/generate", json=payload)
                    r.raise_for_status()
                    return r.json().get("response", "").strip()
            except Exception as e:
                return f"Summarization failed: {e}"

    def extract_images(self, html: str, base_url: str, limit: int = 3):
        urls = []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return urls

        for img in soup.find_all("img"):
            src = img.get("src") or ""
            if not src or src.startswith("data:") or re.search(r"\.svg($|\?)", src, re.I):
                continue

            alt = (img.get("alt") or "").lower()
            src_l = src.lower()
            if any(k in src_l for k in ["sprite", "icon", "logo", "ads", "advert", "pixel"]):
                continue
            if any(k in alt for k in ["icon", "logo"]):
                continue

            full = urljoin(base_url, src)
            if full not in urls:
                urls.append(full)
            if len(urls) >= limit:
                break
        return urls

    def synthesize_search_results(self, text: str):
        """Speak search results using DEDICATED search window"""

        # === STOP PROGRESS INDICATOR IMMEDIATELY ===
        self.stop_search_progress_indicator()

        def _tts_worker():
            if not text or not text.strip():
                return

            try:
                # Use math speaking for search results too
                speak_math = getattr(self, 'speak_math_var', tk.BooleanVar(value=True)).get()
                clean_tts_text = clean_for_tts(text, speak_math=speak_math)

                # === CRITICAL: Use DEDICATED search window ===
                self.preview_search_results(text)

                # Continue with TTS...
                output_path = "out/search_results.wav"

                if self.synthesize_to_wav(clean_tts_text, output_path, role="text"):
                    with self._play_lock:
                        self._play_token += 1
                        my_token = self._play_token
                        self.interrupt_flag = False
                        self.speaking_flag = True

                    self.set_light("speaking")

                    play_path = output_path
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(output_path, "out/search_results_echo.wav")
                            self.logln("[echo] processed search results -> out/search_results_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e} (playing dry)")

                    self.play_wav_with_interrupt(play_path, token=my_token)

            except Exception as e:
                self.logln(f"[search][TTS] Error: {e}")
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()

    # End syththesise_search

    def play_search_results(self, path: str, token=None):
        """Play search results audio with proper interrupt support"""
        try:
            # Use the existing playback infrastructure with token support
            with self._play_lock:
                self._play_token += 1
                my_token = self._play_token
                self.interrupt_flag = False
                self.speaking_flag = True

            self.set_light("speaking")
            self.temporary_mute_for_speech("text")  # Search uses text AI voice
            self.play_wav_with_interrupt(path, token=my_token)

        except Exception as e:
            self.logln(f"[search][playback] Error: {e}")
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")

    def normalize_query(self, q: str) -> str:
        """Add date context ONLY for specific time-related queries"""
        ql = q.lower()
        now = datetime.now()

        # Only add dates for explicit time references
        if "today" in ql:
            q += " " + now.strftime("%Y-%m-%d")
        elif "yesterday" in ql:
            q += " " + (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "this week" in ql:
            q += " " + now.strftime("week %G-W%V")
        # DON'T add dates for "latest", "recent", "current" etc.

        return q


# === END SEARCH METHODS ===


def clean_for_tts(text: str, speak_math: bool = True) -> str:
    """
    Enhanced TTS cleaner that converts LaTeX math to spoken English.
    """
    if not text:
        return ""

    # Use the math speech converter to handle LaTeX math
    cleaned_text = math_speech_converter.make_speakable_text(text, speak_math=speak_math)

    # Additional light cleanup for TTS
    cleaned_text = re.sub(r"[#*_`~>\[\]\(\)-]", "", cleaned_text)
    cleaned_text = re.sub(r":[a-z_]+:", "", cleaned_text)
    cleaned_text = re.sub(r"^[QAqa]:\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

    return cleaned_text.strip()


# === ECHO ENGINE HELPER FUNCTIONS ===
def _read_wav_mono(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def _write_wav(path, y, sr):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), sr)


# === DATACLASS ===
@dataclass
class Item:
    title: str
    url: str
    snippet: str = ""
    pubdate: Optional[str] = None
    summary: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)


# === MAIN APP CLASS ===
class App:

    def __init__(self, master):

        # Clean up temp files on startup
        purge_temp_files("out")

        self.avatar_win = None
        self.cfg = load_cfg()

        # === Initialize logln method FIRST ===
        def logln(msg):
            def _append():
                try:
                    self.log.insert("end", msg + "\n")
                    self.log.see("end")
                except Exception:
                    pass

            try:
                self.master.after(0, _append)
            except Exception:
                pass

        self.logln = logln

        # NOW we can use logln
        self.logln(f"[cfg] qwen_model_path -> {self.cfg.get('qwen_model_path')!r}")

        # Vision instructions are now part of the unified system prompt in config.json
        # No separate vision system prompt needed anymore
        self.logln("[DEBUG] Unified system prompt will be used for both text and vision")

        # Initialize command router
        self.command_router = CommandRouter(self)

        self.master = master
        master.title("Always Listening — Qwen (local)")
        master.geometry("1080x600")

        # === ADD MODEL SELECTION VARIABLES RIGHT HERE ===

        self.text_model_var = tk.StringVar()
        # self.vision_model_var = tk.StringVar()

        # === INITIALIZE ALL TKINTER VARIABLES HERE ===
        self.tts_engine = tk.StringVar(value="sapi5")
        self.speech_rate_var = tk.IntVar(value=0)
        self.sapi_voice_var = tk.StringVar()
        self.echo_enabled_var = tk.BooleanVar(value=False)
        self.ducking_enable = tk.BooleanVar(value=True)
        self.duck_db = tk.DoubleVar(value=12.0)
        self.duck_thresh = tk.DoubleVar(value=1400.0)
        self.duck_attack = tk.IntVar(value=50)
        self.duck_release = tk.IntVar(value=250)
        self.duck_var = tk.DoubleVar(value=100.0)
        self.rms_var = tk.StringVar(value="RMS: 0")
        self.state = tk.StringVar(value="idle")
        self.device_idx = tk.StringVar()
        self.out_device_idx = tk.StringVar()
        self.duplex_mode = tk.StringVar(value="Half-duplex")
        self.latex_auto = tk.BooleanVar(value=True)
        self.latex_append_mode = tk.BooleanVar(value=False)
        self.speak_math_var = tk.BooleanVar(value=True)
        self.avatar_kind = tk.StringVar(value="Rings")
#Auto code enabler
        self.auto_run_var = tk.BooleanVar(value=False)  # Default OFF
        self.logln("[code] Auto-run code: OFF (default)")

        self._last_search_query = ""
        self.search_win = None

        # External light window
        self.external_light_win = None

        # === SEARCH PROGRESS VARIABLES ===
        self._search_in_progress = False
        self._search_progress_timer = None
        self._search_progress_count = 0
        self._last_search_progress_time = 0
        # === END SEARCH PROGRESS VARIABLES ===

        # === NEW: Unified playback fencing ===
        self._play_lock = threading.Lock()
        self._play_token = 0
        # === Append Mode ====
        self.latex_append_mode = tk.BooleanVar(value=False)

        # --- UI State ---
        self.state = tk.StringVar(value="idle")
        self.running = False
        self.device_idx = tk.StringVar()
        self.out_device_idx = tk.StringVar()
        self.duplex_mode = tk.StringVar(value="Half-duplex")

        # Echo engine
        self.echo_engine = EchoEngine()
        self.echo_enabled_var = tk.BooleanVar(value=False)
        self._echo_win = None

        # Ducking controls
        self.ducking_enable = tk.BooleanVar(value=True)
        self.duck_db = tk.DoubleVar(value=12.0)
        self.duck_thresh = tk.DoubleVar(value=1400.0)
        self.duck_attack = tk.IntVar(value=50)
        self.duck_release = tk.IntVar(value=250)
        self._duck_gain = 1.0
        self._duck_active = False

        self._current_speaker = None  # Track which AI is currently speaking
        self._speaker_lock = threading.Lock()

        self._duck_log = bool(self.cfg.get("duck_log", False))
        self._chime_played = False
        self._last_chime_ts = 0.0
        self._beep_once_guard = False

        # === ADD SPEECH RATE VARIABLE HERE ===
        self.speech_rate_var = tk.IntVar(value=0)

        self._duck_gain = 1.0
        self._duck_active = False

# This bit for recording dictation and no VAD
        self._dictation_recording = False
        self._dictation_buffer = []
        self._space_held = False
        self._dictation_mode_active = False


        # Replace complex vision state with:
        self._last_image_path = None
        self._last_vision_reply = ""
        self._last_was_vision = False  # Just for tracking if last reply was vision-based

        # Track current personality for proper switching
        self._current_personality = "Default"

        # Muting control
        self.text_ai_muted = False

        self._mute_lock = threading.Lock()
        # === Initialize UI components FIRST ===
        self._setup_ui()

        # === Initialize AI engines AFTER UI ===
        self._setup_ai_engines()
        # === ADD THIS LINE ===
        self._load_personalities()  # This will populate the combo box

        # Apply config defaults
        self._apply_config_defaults()

        # Auto-refresh models after UI is ready
        self.master.after(500, self._refresh_and_select_default_model)

        # Init Plotter
        self.plotter = Plotter(master, log_fn=self.logln) if Plotter else None

    def _setup_ui(self):
        from tkinter import ttk

        # ═══════════════════════════════════════════════════════════════════════
        # ✨ MODERN UI STYLING
        # ═══════════════════════════════════════════════════════════════════════

        # Modern color scheme
        BG_DARK = "#2b2b2b"
        BG_MID = "#3a3a3a"
        BG_LIGHT = "#4a4a4a"
        FG_TEXT = "#e0e0e0"
        ACCENT_BLUE = "#4a9eff"
        ACCENT_GREEN = "#50c878"
        ACCENT_AMBER = "#ffb347"

        # Configure main window
        self.master.configure(bg=BG_DARK)

        # Configure ttk style
        style = ttk.Style()
        style.theme_use('clam')

        # Frame styling
        style.configure('TFrame', background=BG_DARK)
        style.configure('Card.TFrame', background=BG_MID, relief='raised', borderwidth=1)

        # Label styling
        style.configure('TLabel', background=BG_DARK, foreground=FG_TEXT, font=('Segoe UI', 9))
        style.configure('Title.TLabel', font=('Segoe UI', 10, 'bold'), foreground=ACCENT_BLUE)
        style.configure('Header.TLabel', font=('Segoe UI', 9, 'bold'))

        # Button styling
        style.configure('TButton',
                        background=BG_MID,
                        foreground=FG_TEXT,
                        borderwidth=1,
                        focuscolor='none',
                        font=('Segoe UI', 9),
                        padding=(8, 4))
        style.map('TButton',
                  background=[('active', BG_LIGHT), ('pressed', '#5a5a5a')],
                  foreground=[('active', '#ffffff')])

        # Success button (Start)
        style.configure('Success.TButton',
                        background=ACCENT_GREEN,
                        foreground='#ffffff',
                        font=('Segoe UI', 9, 'bold'),
                        padding=(12, 5))
        style.map('Success.TButton',
                  background=[('active', '#40b868'), ('pressed', '#30a858')])

        # Accent button (Stop, important actions)
        style.configure('Accent.TButton',
                        background=ACCENT_BLUE,
                        foreground='#ffffff',
                        font=('Segoe UI', 9, 'bold'),
                        padding=(12, 5))
        style.map('Accent.TButton',
                  background=[('active', '#3a8eef'), ('pressed', '#2a7edf')])

        # Small button style
        style.configure('Small.TButton', padding=(6, 3), font=('Segoe UI', 8))

        # Combobox styling - FIXED
        style.configure('TCombobox',
                        fieldbackground=BG_MID,
                        background=BG_MID,
                        foreground=FG_TEXT,
                        arrowcolor=FG_TEXT,
                        borderwidth=1)

        # CRITICAL: Set the actual entry colors for combobox
        style.map('TCombobox',
                  fieldbackground=[('readonly', BG_MID)],
                  foreground=[('readonly', FG_TEXT)],
                  selectbackground=[('readonly', ACCENT_BLUE)],
                  selectforeground=[('readonly', '#ffffff')])

        # Also configure the combobox option menu
        self.master.option_add('*TCombobox*Listbox.background', BG_MID)
        self.master.option_add('*TCombobox*Listbox.foreground', FG_TEXT)
        self.master.option_add('*TCombobox*Listbox.selectBackground', ACCENT_BLUE)
        self.master.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')

        # Checkbutton styling
        style.configure('TCheckbutton',
                        background=BG_DARK,
                        foreground=FG_TEXT,
                        font=('Segoe UI', 9))
        style.map('TCheckbutton',
                  background=[('active', BG_DARK)])

        # Spinbox styling
        style.configure('TSpinbox',
                        fieldbackground=BG_MID,
                        background=BG_MID,
                        foreground=FG_TEXT,
                        arrowcolor=FG_TEXT,
                        borderwidth=1)

        # Progressbar styling
        style.configure('TProgressbar',
                        background=ACCENT_GREEN,
                        troughcolor=BG_MID,
                        borderwidth=0,
                        thickness=10)

        # Scale styling
        style.configure('TScale',
                        background=BG_DARK,
                        troughcolor=BG_MID,
                        borderwidth=0,
                        sliderrelief='flat')

        # ═══════════════════════════════════════════════════════════════════════
        # UI COMPONENTS
        # ═══════════════════════════════════════════════════════════════════════

        """Initialize all UI components"""
        # Top controls frame
        top = ttk.Frame(self.master)
        top.grid(row=0, column=0, columnspan=12, sticky="we", padx=8, pady=8)

        # Status light with glow effect - PROTECTED STYLING
        self.light = tk.Canvas(top, width=54, height=54, highlightthickness=0, bg="#2b2b2b")
        self.light.create_oval(2, 2, 52, 52, fill='#1a1a1a', outline='')
        self.circle = self.light.create_oval(6, 6, 48, 48, fill="#f1c40f", outline="#f39c12", width=2)
        self.light.grid(row=0, column=0, padx=10, pady=10)

        # Force the canvas to maintain its dark background
        self.light.configure(bg="#2b2b2b", highlightbackground="#2b2b2b")
        # Main control buttons
        self.start_btn = ttk.Button(top, text="▶ Start", command=self.start, style='Success.TButton')
        self.stop_btn = ttk.Button(top, text="⏹ Stop", command=self.stop, state=tk.DISABLED, style='Accent.TButton')
        self.reset_btn = ttk.Button(top, text="🔄 Reset Chat", command=self.reset_chat)

        self.start_btn.grid(row=0, column=1, padx=6)
        self.stop_btn.grid(row=0, column=2, padx=6)
        self.reset_btn.grid(row=2, column=6, padx=6, sticky="w")

        # External light and Stop Speaking
        self.external_light_btn = ttk.Button(top, text="💡 External Light", command=self.toggle_external_light)
        self.external_light_btn.grid(row=2, column=5, padx=6)

        self.stop_speech_btn = ttk.Button(top, text="🔇 Stop Speaking", command=self.stop_speaking,
                                          style='Accent.TButton')
        self.stop_speech_btn.grid(row=0, column=3, padx=6)

        # === MUTE CONTROLS ===
        mute_frame = ttk.Frame(top)
        mute_frame.grid(row=0, column=14, padx=6, sticky="w")

        ttk.Label(mute_frame, text="Mute:", style='Header.TLabel').pack(anchor="w")

        mute_buttons_frame = ttk.Frame(mute_frame)
        mute_buttons_frame.pack(fill="x", pady=(2, 0))

        self.ai_mute_btn = ttk.Button(
            mute_buttons_frame,
            text="🔇 AI",
            width=8,
            command=self.toggle_ai_mute
        )
        self.ai_mute_btn.pack(side="left", padx=(0, 3))

        # === PLOTTING TOGGLE ===
        self.plotting_var = tk.BooleanVar(value=True)
        self.plotting_cb = ttk.Checkbutton(
            top,
            text="📊 Plotting ON",
            variable=self.plotting_var,
            command=self._on_plotting_toggle
        )
        self.plotting_cb.grid(row=4, column=5, padx=6)

        # Close Windows button
        ttk.Button(top, text="🗙 Close Windows", command=self.close_all_windows).grid(row=4, column=6, padx=9)
        # Audio file input button
        ttk.Button(top, text="🎵 Load Audio", command=self._load_audio_file).grid(row=4, column=7, padx=6)

        # === AI MODEL SELECTION (Card style) ===
        model_frame = ttk.Frame(top, style='Card.TFrame', padding=8)
        model_frame.grid(row=2, column=3, padx=6, sticky="n")

        ttk.Label(model_frame, text="🤖 AI Model (Text+Vision)", style='Title.TLabel').pack(anchor="n")
        self.text_model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.text_model_var,
            state="readonly",
            width=18
        )
        self.text_model_combo.pack(pady=(4, 4), anchor="n")

        ttk.Button(model_frame, text="🔄 Refresh Models", command=self._refresh_models,
                   width=18, style='Small.TButton').pack(anchor="n")

        self.text_model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())

        # === PERSONALITY SELECTION (Card style) ===
        personality_frame = ttk.Frame(top, style='Card.TFrame', padding=8)
        personality_frame.grid(row=2, column=1, padx=10, sticky="w")

        ttk.Label(personality_frame, text="👤 Personality", style='Title.TLabel').pack(anchor="w")

        self.personality_var = tk.StringVar(value="Default")
        self.personality_combo = ttk.Combobox(
            personality_frame,
            textvariable=self.personality_var,
            state="readonly",
            width=16,
            values=["Default"]
        )
        self.personality_combo.pack(pady=(4, 2))
        self.personality_combo.current(0)
        self.personality_combo.bind("<<ComboboxSelected>>", self._on_personality_change)

        self.personality_status = ttk.Label(personality_frame, text="✓ Default", foreground=ACCENT_GREEN)
        self.personality_status.pack()

        # Dictation controls here
        self.dictation_mode_var = tk.BooleanVar(value=False)
        self.dictation_checkbox = ttk.Checkbutton(
            top, text="📝 Dictation Mode",
            variable=self.dictation_mode_var,
            command=self._on_dictation_mode_toggle
        )

        ####
        self.dictation_checkbox.grid(row=5, column=2, padx=6, pady=6, sticky="w")

        # Prevent spacebar from toggling the checkbox
        self.dictation_checkbox.unbind_class("TCheckbutton", "<space>")

        self.dictation_btn = ttk.Button(
            top, text="🎤 Hold to Record",
            command=self._toggle_dictation_recording
        )
        self.dictation_btn.grid(row=5, column=3, padx=6, pady=6, sticky="w")
        self.dictation_btn.grid_remove()

        # Plot button - always visible
        self.plot_btn = ttk.Button(
            top, text="📈 Plot",
            command=self._plot_last_expression,
            width=8
        )
        self.plot_btn.grid(row=5, column=4, padx=6, pady=6, sticky="w")


        # Prevent spacebar from triggering the button
        self.dictation_btn.unbind_class("TButton", "<space>")

       # self.master.bind("<KeyPress-space>", self._on_space_press)
       # self.master.bind("<KeyRelease-space>", self._on_space_release)
        self.master.bind_all("<KeyPress-space>", self._on_space_press)
        self.master.bind_all("<KeyRelease-space>", self._on_space_release)

        # End Dictation controls



        # Echo controls
        ttk.Checkbutton(
            top, text="🔊 Echo ON", variable=self.echo_enabled_var,
            command=self._sync_echo_state
        ).grid(row=2, column=2, padx=(10, 4))

        ttk.Button(top, text="⚙ Show Echo", command=self._toggle_echo_window).grid(row=0, column=5, padx=(4, 10))

        # Images + Refresh buttons
        _imgbar = ttk.Frame(top)
        _imgbar.grid(row=0, column=6, padx=(6, 4))
        ttk.Button(_imgbar, text="📷 Images", command=self._toggle_image_window).pack(side="left")
        ttk.Button(_imgbar, text="🔄 Last Question", command=self._refresh_last_reply).pack(side="left", padx=(6, 0))

        # LaTeX controls
        self.latex_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="∑ Auto LaTeX", variable=self.latex_auto).grid(row=0, column=7, padx=6)
        ttk.Button(top, text="∑ Show/Hide", command=self.toggle_latex).grid(row=0, column=8, padx=6)
        ttk.Button(
            top, text="📋 Copy LaTeX",
            command=lambda: self.latex_win.copy_raw_latex() if hasattr(self, "latex_win") else None
        ).grid(row=0, column=9, padx=(0, 6))

        # LaTeX append controls
        ttk.Checkbutton(top, text="➕ Append LaTeX", variable=self.latex_append_mode).grid(row=2, column=7, padx=6)
        ttk.Button(top, text="🗑 Clear LaTeX", command=self.clear_latex).grid(row=2, column=8, padx=5)

        # Speak Math checkbox
        self.speak_math_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="🗣 Speak Math", variable=self.speak_math_var,
                        command=self.update_speak_math_setting).grid(row=0, column=12, padx=6)

        # === AVATAR SELECTION (Card style) ===
        self.avatar_win = None
        self.avatar_kind = tk.StringVar(value="Rings")
        _avatar_bar = ttk.Frame(top, style='Card.TFrame', padding=8)
        _avatar_bar.grid(row=2, column=4, padx=6, sticky="n")

        ttk.Label(_avatar_bar, text="🎭 Avatar", style='Title.TLabel').pack(anchor="n")
        self.avatar_combo = ttk.Combobox(
            _avatar_bar, textvariable=self.avatar_kind, state="readonly",
            width=14, values=["Rings", "Rectangles", "Rectangles 2", "Radial Pulse",
                              "FaceRadialAvatar", "String Grid", "Sphere", "HAL 9000"]
        )
        self.avatar_combo.current(0)
        self.avatar_combo.pack(pady=(4, 4), anchor="n")
        ttk.Button(_avatar_bar, text="Open/Close", command=self.toggle_avatar,
                   style='Small.TButton').pack(anchor="n")

        def _on_avatar_kind_change(_e=None):
            if self.avatar_win and self.avatar_win.winfo_exists():
                try:
                    self.avatar_win.destroy()
                except Exception:
                    pass
                self.avatar_win = None
                self.open_avatar()

        self.avatar_combo.bind("<<ComboboxSelected>>", _on_avatar_kind_change)

        # === MODE SELECTION (Card style) ===
        mode_bar = ttk.Frame(top, style='Card.TFrame', padding=8)
        mode_bar.grid(row=0, column=10, padx=6, sticky="n")
        ttk.Label(mode_bar, text="🎙 Mode", style='Title.TLabel').pack(anchor="n")
        self.mode_combo = ttk.Combobox(
            mode_bar, textvariable=self.duplex_mode, state="readonly", width=18,
            values=["Half-duplex", "Full-duplex (barge-in)"]
        )
        self.mode_combo.current(0)
        self.mode_combo.pack(pady=(4, 0), anchor="n")

        # === SEPARATOR ===
        ttk.Separator(self.master, orient='horizontal').grid(row=1, column=0, columnspan=12, sticky='ew', pady=8)

        # === DUCKING UI (Card style) ===
        duck = ttk.Frame(self.master, style='Card.TFrame', padding=8)
        duck.grid(row=2, column=0, columnspan=12, padx=10, pady=4, sticky="we")

        ttk.Checkbutton(duck, text="🔉 Ducking", variable=self.ducking_enable).pack(side="left", padx=(0, 8))
        ttk.Label(duck, text="↓dB").pack(side="left")
        ttk.Spinbox(duck, from_=0, to=36, width=3, textvariable=self.duck_db).pack(side="left", padx=(2, 8))
        ttk.Label(duck, text="Thr").pack(side="left")
        ttk.Spinbox(duck, from_=200, to=5000, width=5, textvariable=self.duck_thresh).pack(side="left", padx=(2, 8))
        ttk.Label(duck, text="Atk/Rel ms").pack(side="left")
        ttk.Spinbox(duck, from_=5, to=300, width=4, textvariable=self.duck_attack).pack(side="left", padx=(2, 2))
        ttk.Spinbox(duck, from_=20, to=1000, width=5, textvariable=self.duck_release).pack(side="left", padx=(2, 8))
        ttk.Label(duck, text="Gain").pack(side="left", padx=(8, 2))
        self.duck_var = tk.DoubleVar(value=100.0)
        ttk.Progressbar(duck, orient="horizontal", length=120, mode="determinate",
                        variable=self.duck_var, maximum=100.0).pack(side="left", padx=(0, 8))
        self.rms_var = tk.StringVar(value="RMS: 0")
        ttk.Label(duck, textvariable=self.rms_var, style='Header.TLabel').pack(side="left")

        # === DEVICE SELECTION ===
        # Mic device
        lbl_mic = ttk.Label(self.master, text="🎤 Mic device:", style='Header.TLabel')
        lbl_mic.grid(row=3, column=0, sticky="e", pady=6)

        self.dev_combo = ttk.Combobox(self.master, textvariable=self.device_idx, state="readonly", width=35)
        devs = list_input_devices()
        vals = [f"{i}: {n}" for i, n in devs] if devs else ["No input devices found"]
        self.dev_combo["values"] = vals
        if vals:
            self.dev_combo.current(0)

        self.dev_combo.grid(row=3, column=1, columnspan=9, sticky="w", padx=6, pady=6)

        # Output device
        lbl_speaker = ttk.Label(self.master, text="🔊 Speaker device:", style='Header.TLabel')
        lbl_speaker.grid(row=4, column=0, sticky="e", pady=6)

        out_vals = self._list_output_devices()
        self.out_combo = ttk.Combobox(self.master, textvariable=self.out_device_idx, state="readonly",
                                      width=35, values=out_vals)
        if out_vals:
            self.out_combo.current(0)
        self.out_combo.grid(row=4, column=1, columnspan=9, sticky="w", padx=6, pady=6)

        # === SAPI VOICE SELECTION ===
        self.sapi_voice_var = tk.StringVar()
        try:
            import pyttsx3
            eng = pyttsx3.init()

            # Create simple display names (NO gender/age)
            voices_display = []
            self.voice_mapping = {}

            for v in eng.getProperty("voices"):
                display_name = v.name  # ← Just use plain name
                voices_display.append(display_name)
                self.voice_mapping[display_name] = v.id

            #  voices_display.sort() #Don't need to sort at present

            # Get Text AI voice from JSON config
            config_voice = self.cfg.get("text_ai_voice")

        except Exception as e:
            voices_display = ["(no SAPI5 voices - install pyttsx3)"]
            self.voice_mapping = {}
            config_voice = None
            self.logln(f"[tts] voice enumeration error: {e}")

        lbl_sapi = ttk.Label(self.master, text="🗣️ SAPI Voice:", style='Header.TLabel')
        lbl_sapi.grid(row=5, column=0, sticky="e", pady=6)

        self.sapi_combo = ttk.Combobox(
            self.master,
            textvariable=self.sapi_voice_var,
            values=voices_display,
            width=35,
            state="readonly"
        )
        self.sapi_combo.grid(row=5, column=1, columnspan=5, sticky="w", padx=6, pady=6)

        # Set combobox to JSON config voice if available
        if voices_display and voices_display[0] != "(no SAPI5 voices - install pyttsx3)":
            if config_voice and config_voice in voices_display:
                # Use config voice
                idx = voices_display.index(config_voice)
                self.sapi_combo.current(idx)
                self.sapi_voice_var.set(config_voice)
                self.logln(f"[tts] Text AI voice from config: {config_voice}")
            else:
                # Use first available
                self.sapi_combo.current(0)
                self.sapi_voice_var.set(voices_display[0])
                if config_voice:
                    self.logln(f"[tts] Config voice '{config_voice}' not found, using first available")
        # === SPEECH SPEED CONTROL ===
        lbl_speed = ttk.Label(self.master, text="⚡ Speech Speed:", style='Header.TLabel')
        lbl_speed.grid(row=6, column=0, sticky="e", pady=6)

        self.speech_rate_var = tk.IntVar(value=5)
        rate_slider = ttk.Scale(
            self.master,
            from_=-10,
            to=10,
            variable=self.speech_rate_var,
            orient="horizontal",
            length=180
        )
        rate_slider.grid(row=6, column=1, columnspan=3, sticky="we", padx=6, pady=6)

        self.rate_value_label = ttk.Label(self.master, text="Fast", style='Header.TLabel')
        self.rate_value_label.grid(row=6, column=4, sticky="w", padx=5)

        # Preset buttons
        preset_frame = ttk.Frame(self.master)
        preset_frame.grid(row=7, column=1, columnspan=4, sticky="w", pady=2)

        ttk.Button(preset_frame, text="Slow", width=6,
                   command=lambda: self.set_speech_rate(-5), style='Small.TButton').pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Normal", width=6,
                   command=lambda: self.set_speech_rate(0), style='Small.TButton').pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Fast", width=6,
                   command=lambda: self.set_speech_rate(5), style='Small.TButton').pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Reset", width=6,
                   command=lambda: self.set_speech_rate(5), style='Small.TButton').pack(side="left", padx=2)

        self.update_rate_display()

        # === TEXT INPUT ===
        lbl_text = ttk.Label(self.master, text="✏️ Text input:", style='Header.TLabel')
        lbl_text.grid(row=8, column=0, sticky="ne", padx=(6, 0), pady=(8, 6))

        self.text_box = ScrolledText(self.master, width=70, height=10, wrap="word",
                                     bg=BG_MID, fg=FG_TEXT, insertbackground=ACCENT_BLUE,
                                     font=('Consolas', 10), relief='flat', borderwidth=2,
                                     selectbackground=ACCENT_BLUE, selectforeground='#ffffff')
        self.text_box.grid(row=8, column=1, columnspan=8, sticky="we", padx=6, pady=(8, 6))

        ttk.Button(self.master, text="➤ Send", command=self.send_text,
                   style='Accent.TButton').grid(row=8, column=9, sticky="nw", padx=6, pady=(8, 6))

        self.text_box.bind("<Control-Return>", lambda e: (self.send_text(), "break"))

        # === LOG WINDOW ===
        lbl_log = ttk.Label(self.master, text="📋 Log:", style='Header.TLabel')
        lbl_log.grid(row=9, column=0, sticky="nw", padx=6, pady=6)

        self.log = tk.Text(self.master, height=12, width=80,
                           bg=BG_DARK, fg=ACCENT_AMBER, insertbackground=ACCENT_AMBER,
                           font=('Consolas', 9), relief='sunken', borderwidth=2,
                           selectbackground=ACCENT_BLUE, selectforeground='#ffffff')
        self.log.grid(row=9, column=1, columnspan=9, sticky="nsew", padx=6, pady=6)

        # Grid configuration
        self.master.grid_rowconfigure(9, weight=1)
        self.master.grid_columnconfigure(9, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(5, weight=1)

        # Edge Speech
        # === TTS ENGINE SELECTION ===
        lbl_engine = ttk.Label(self.master, text="🔊 TTS Engine:", style='Header.TLabel')
        lbl_engine.grid(row=6, column=5, sticky="e", pady=6)

        engine_values = ["sapi5"]
        if EDGE_TTS_AVAILABLE:
            engine_values.append("edge")

        self.engine_combo = ttk.Combobox(
            self.master,
            textvariable=self.tts_engine,
            values=engine_values,
            state="readonly",
            width=10
        )
        self.engine_combo.grid(row=6, column=6, sticky="w", padx=6, pady=6)
        self.engine_combo.current(0)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)

        # === EDGE VOICE SELECTION ===
        lbl_edge = ttk.Label(self.master, text="🌐 Edge Voice:", style='Header.TLabel')
        lbl_edge.grid(row=6, column=7, sticky="e", pady=6)

        self.edge_voice_var = tk.StringVar(value="en-US-AriaNeural")
        self.edge_voice_combo = ttk.Combobox(
            self.master,
            textvariable=self.edge_voice_var,
            state="disabled",  # Start disabled since SAPI5 is default
            width=25
        )
        self.edge_voice_combo.grid(row=6, column=8, columnspan=2, sticky="w", padx=6, pady=6)
        self._populate_edge_voices()

        # End Edge

        # === LATEX WINDOWS ===
        self.latex_win_text = None
        self.latex_win_vision = None
        self.latex_win_search = None
        self.latex_win_weather = None
        self._current_latex_context = "text"

        DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
        DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
        DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

        self.latex_win_text = LatexWindow(
            self.master,
            log_fn=self.logln,
            text_family=DEFAULT_TEXT_FAMILY,
            text_size=DEFAULT_TEXT_PT,
            math_pt=DEFAULT_MATH_PT
        )
        self.latex_win_text.title("Text AI - LaTeX Preview")
        self.latex_win = self.latex_win_text

        if self.latex_auto.get():
            self.latex_win_text.show()

        # SandBox Button
        self.code_window = None  # Will be created on demand

        # Add a "Run Code" button to UI,
        self.code_btn = ttk.Button(
            top,  # Your top frame
            text="💻 Run Code",
            command=self._show_code_window,
            width=12
        )
        self.code_btn.grid(row=5, column=5, padx=6)  # Adjust column as needed

        # ADD THE CHECKBOX RIGHT AFTER IT to enable self-correcting coding
        self.auto_run_check = ttk.Checkbutton(
            top,
            text="▶ Auto-run",
            variable=self.auto_run_var,
            command=self._on_auto_run_toggle
        )
        self.auto_run_check.grid(row=5, column=6, padx=6, sticky="w")

        # === TEMPORARY DEBUG - ADD THIS AT THE VERY END ===
        def check_values():
            print(f"[DEBUG] Mic combo values: {self.dev_combo['values']}")
            print(f"[DEBUG] Mic combo current: {self.dev_combo.current()}")
            print(f"[DEBUG] Mic combo get(): {self.dev_combo.get()}")
            print(f"[DEBUG] Mic StringVar: {self.device_idx.get()}")
            print("")
            print(f"[DEBUG] Speaker combo values: {self.out_combo['values']}")
            print(f"[DEBUG] Speaker combo current: {self.out_combo.current()}")
            print(f"[DEBUG] Speaker combo get(): {self.out_combo.get()}")
            print(f"[DEBUG] Speaker StringVar: {self.out_device_idx.get()}")
            print("")
            print(f"[DEBUG] SAPI combo values: {len(self.sapi_combo['values'])} voices")
            print(f"[DEBUG] SAPI combo current: {self.sapi_combo.current()}")
            print(f"[DEBUG] SAPI combo get(): {self.sapi_combo.get()}")
            print(f"[DEBUG] SAPI StringVar: {self.sapi_voice_var.get()}")
            print("")
            print(f"[DEBUG] Model combo values: {self.text_model_combo['values']}")
            print(f"[DEBUG] Model combo current: {self.text_model_combo.current()}")
            print(f"[DEBUG] Model combo get(): {self.text_model_combo.get()}")
            print(f"[DEBUG] Model StringVar: {self.text_model_var.get()}")

        self.master.after(1000, check_values)

    def _setup_ai_engines(self):
        """Initialize single AI engine for both text and vision"""

        # Ensure we have a model selection
        if not self.text_model_var.get():
            self.logln("[ai] No model selection found - refreshing models")
            self._refresh_models()
            if not self.text_model_var.get():
                self.logln("[ai] Still no model selection - using default from config")
                # Use ministral-3 as default for single-model setup
                self.text_model_var.set("ministral-3:latest")

        # Use the selected model for everything
        selected_model = self.text_model_var.get()

        print(f"[DEBUG] _setup_ai_engines - Using single model: '{selected_model}'")
        self.logln(f"[ai] Initializing single model for text+vision: {selected_model}")

        # Check if model changed
        if hasattr(self, 'qwen') and self.qwen:
            current_model = getattr(self.qwen, 'model_path', 'unknown')
            if current_model != selected_model:
                print(f"[DEBUG] Model changed from '{current_model}' to '{selected_model}'")

        # Initialize ASR (unchanged)
        self.asr = ASR(
            self.cfg["whisper_model"],
            self.cfg["whisper_device"],
            self.cfg["whisper_compute_type"],
            self.cfg["whisper_beam_size"]
        )

        # Initialize Qwen with selected model
        self.qwen = QwenLLM(
            model_path=selected_model,  # Use same model for everything
            model=selected_model,
            temperature=self.cfg["qwen_temperature"],
            max_tokens=self.cfg["qwen_max_tokens"]
        )

        # No separate vision model needed
        # self.vl_model = vision_model  # REMOVE THIS LINE

        # === CRITICAL CONNECTION LINES ===
        # Connect main app to QwenLLM
        self.qwen.set_main_app(self)
        self.logln(
            f"[DEBUG] QwenLLM main_app connected: {hasattr(self.qwen, 'main_app') and self.qwen.main_app is not None}")

        # Connect search handler
        if hasattr(self.qwen, 'set_search_handler'):
            self.qwen.set_search_handler(self.handle_ai_search_request)
            self.logln("[DEBUG] ✅ Search handler connected to QwenLLM")
        else:
            self.logln("[DEBUG] ❌ QwenLLM missing set_search_handler method")
        # === END CRITICAL CONNECTION LINES ===

        # === SYSTEM PROMPT - ENHANCED FOR SINGLE MODEL ===
        # System prompt with REAL-TIME date awareness
        sys_prompt = (
                self.cfg.get("system_prompt")
                or self.cfg.get("qwen_system_prompt")
                or ""
        )

        # Create a dynamic system prompt for the single AI
        from datetime import datetime
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%B %d, %Y")
        current_time = current_datetime.strftime("%I:%M %p")
        current_day = current_datetime.strftime("%A")

        enhanced_system_prompt = f"""{sys_prompt}

        === CORE CAPABILITIES ===
        YOU ARE A UNIFIED AI ASSISTANT WITH TEXT, VISION, AND MATHEMATICAL CAPABILITIES.

        ABILITIES:
        1. Process text-only questions
        2. Analyze images when provided (base64-encoded in requests)
        3. Answer questions about previously seen images
        4. Provide mathematical solutions in plottable formats
        5. Support voice-driven mathematical workflows

        === MATHEMATICAL OUTPUT STANDARDS ===

        FINAL ANSWER FORMATTING:
        ✓ ALWAYS provide clean, standalone mathematical expressions
        ✓ Use LaTeX notation: $f(x) = 2x$ or $$\\frac{{x^2}}{{4}}$$
        ✓ For derivatives/integrals, show ONLY the final result when asked
        ✓ Use \\boxed{{}} for final answers: \\boxed{{f'(x) = 2x}}

        When generating LaTeX for matplotlib plots:
        - Use plain parentheses () instead of \\left( \\right)
        - Use plain brackets [] instead of \\left[ \\right]
        - Use \\{{ \\}} instead of \\left\\{{ \\right\\}}
        - Avoid \\text{{}} - use \\mathrm{{}} instead
        - Keep expressions simple - mathtext is a subset of full LaTeX

        EXAMPLES - GOOD:
        User: "What's the derivative of x³?"
        You: "The derivative is $f'(x) = 3x^2$"

        User: "Integrate x²"
        You: "The integral is $\\int x^2 dx = \\frac{{x^3}}{{3}} + C$"

        EXAMPLES - AVOID:
        ❌ "The answer is three x squared" (no LaTeX)
        ❌ Showing only intermediate steps without final answer
        ❌ Using text descriptions instead of clean expressions

        === PLOTTING SYSTEM REQUIREMENTS ===

        CRITICAL PLOTTING RULES:
        1. The plotting system works on EXPRESSIONS ONLY (not equations)
        2. Remove "y =" or "f(x) =" from plottable expressions
        3. Variables supported: x, t, theta (θ), n, s
        4. All variables will be auto-converted to 'x' for plotting
        5. Integration constants (+C) are automatically removed

        WHEN USER SAYS "GRAPH THAT":
        - The system searches your LAST response for mathematical expressions
        - It prioritizes: \\boxed{{}} > LaTeX delimiters > equations
        - It extracts the FINAL answer automatically
        - You DO NOT need to repeat the expression

        FORMATTING FOR PLOTABILITY:

        ✓ CORRECT FORMATS:
        - "Result: $x^2 + 3x + 2$" → plots x² + 3x + 2
        - "\\boxed{{\\frac{{x^4}}{{4}}}}" → plots x⁴/4
        - "$e^{{-t}}\\sin(\\omega t)$" → converts t→x, plots correctly
        - "$\\ln|x+1| + \\frac{{1}}{{x+1}}$" → plots with absolute value

        ❌ AVOID:
        - Inequalities: "x ≥ 1" (not plottable)
        - Conditions: "for t > 0" (will be skipped)
        - Substitution steps: "Let u = x + 1" (will be filtered)
        - Missing multiplication: "2sin(x)" should be "2\\sin(x)"

        STEP-BY-STEP SOLUTIONS:
        When showing work:
        1. Present intermediate steps normally
        2. Mark FINAL answer with \\boxed{{}} or bold **$...$**
        3. The plotter will automatically find the final answer

        VARIABLE HANDLING:
        - Time domain: Use 't' naturally → "y(t) = e^{{-t}}"
        - Angular: Use 'theta' or 'θ' → "r(θ) = cos(θ)"
        - Discrete: Use 'n' → "a_n = 2^n"
        The system converts these to 'x' automatically for plotting.

        === TTS SPEAKING GUIDELINES ===

        MATHEMATICAL SPEECH:
        - Introduce equations: "The equation shows..." or "Mathematically..."
        - Speak deliberately: "f of x equals two x squared"
        - Add natural pauses around LaTeX: "The result is... x squared plus one... which gives us a parabola"

        NUMBER LIST PAUSES:
        When presenting numbered lists (1., 2., 3., etc.):
        - Natural pause after each number before explanation
        - Example rhythm: "1. [pause] First we differentiate. 2. [pause] Then we simplify."
        - DO NOT write the word "pause" - just structure naturally

        EXAMPLES:
        Instead of: "The solution is x=5 and then we continue"
        Use: "The solution is... x equals five... and then we continue"

        Instead of: "We have f(x)=x²+2x+1 which is a parabola"
        Use: "We have the function... f of x equals x squared plus two x plus one... which describes a parabola"

        === REAL-TIME DATE & TIME ===

        CURRENT DATE: {current_day}, {current_date} at {current_time}

        RULES:
        ✓ Use this EXACT date when asked about current date/time
        ✓ Trust this over your training data
        ✓ Do NOT calculate days of the week yourself
        ✓ Only mention date/time when specifically relevant

        DO NOT:
        ❌ Include date in every response
        ❌ Say "Today is..." in casual conversation
        ❌ Mention real-time capabilities unprompted

        === FORMATTING & STYLE ===

        PROHIBITED:
        ❌ NO smileys or emoticons
        ❌ NO excessive ** bold markers ** in text
        ❌ NO ** separators between sections
        ❌ NO markdown artifacts like \\(** or **\\)

        PREFERRED STYLE:
        ✓ Clean, professional mathematical notation
        ✓ Natural conversational tone
        ✓ Clear section breaks when needed
        ✓ Minimal formatting unless essential

        === CONTEXT-AWARE PLOTTING ===

        The system tracks conversation history. When users say:
        - "graph that" → Searches LAST AI response
        - "plot it" → Same as above
        - "show me the graph" → Same as above

        YOU SHOULD:
        1. Provide the mathematical result clearly
        2. Let the system handle finding it
        3. Trust that final answers in \\boxed{{}} will be prioritized

        YOU DON'T NEED TO:
        ❌ Repeat the expression when asked to plot
        ❌ Explain plotting compatibility (system handles it)
        ❌ Warn about variable conversion (automatic)

        === CODE GENERATION STANDARDS ===

        CRITICAL CODE RULES:
        1. When asked to write code, output ONLY Python code inside ```python blocks
        2. NEVER include example output lines like "The result is: 3.14" inside code blocks
        3. NEVER include descriptive text or comments about output inside code blocks
        4. Make code COMPLETE and DIRECTLY RUNNABLE
        5. Include necessary import statements (math, random, statistics, etc.)
        6. Include example usage as ACTUAL CODE (print statements), not comments

        === SUMMARY OF KEY BEHAVIORS ===

        1. **Mathematics**: Always provide clean LaTeX expressions
        2. **Plotting**: Format final answers clearly, system handles extraction
        3. **TTS**: Natural pauses, clear mathematical speech
        4. **Date/Time**: Use provided current date exactly
        5. **Style**: Professional, clean, no excessive formatting
        6. **Context**: Trust the system to find previous expressions

        Respond naturally and helpfully while following these guidelines.
        """

        self.qwen.system_prompt = enhanced_system_prompt
        self.logln(f"[ai] ✅ Single model system prompt updated with unified capabilities")
        self.logln(f"[ai] 📅 Current date in system: {current_date} at {current_time}")

    def _apply_config_defaults(self):
        """Apply configuration defaults to UI"""
        try:
            # === DUPLEX MODE ===
            if bool(self.cfg.get("duplex", False)):
                self.duplex_mode.set("Full-duplex (barge-in)")
            else:
                self.duplex_mode.set("Half-duplex")

            # === DUCKING SETTINGS ===
            if "duck_enable" in self.cfg:
                self.ducking_enable.set(bool(self.cfg.get("duck_enable", True)))
                self.duck_db.set(float(self.cfg.get("duck_db", self.duck_db.get())))
                self.duck_attack.set(int(self.cfg.get("duck_attack_ms", self.duck_attack.get())))
                self.duck_release.set(int(self.cfg.get("duck_release_ms", self.duck_release.get())))
                self.duck_thresh.set(float(self.cfg.get("duck_thresh", self.duck_thresh.get())))

            # === TTS ENGINE SELECTION ===
            config_engine = self.cfg.get("tts_engine", "sapi5").lower()
            if config_engine == "edge" and EDGE_TTS_AVAILABLE:
                self.tts_engine.set("edge")
                self.logln("[cfg] TTS engine: Edge (neural)")
            else:
                self.tts_engine.set("sapi5")
                if config_engine == "edge" and not EDGE_TTS_AVAILABLE:
                    self.logln("[cfg] TTS engine: SAPI5 (Edge requested but not available)")
                else:
                    self.logln("[cfg] TTS engine: SAPI5 (local)")

            # Update UI state for engine selection
            self._on_engine_change()

            # === EDGE VOICE FROM CONFIG ===
            if EDGE_TTS_AVAILABLE:
                config_edge_voice = self.cfg.get("edge_voice")
                if config_edge_voice:
                    available_edge_voices = list(self.edge_voice_combo['values'])
                    if config_edge_voice in available_edge_voices:
                        self.edge_voice_var.set(config_edge_voice)
                        self.logln(f"[cfg] Edge voice: {config_edge_voice}")
                    else:
                        # Try partial match
                        matching = [v for v in available_edge_voices if config_edge_voice.lower() in v.lower()]
                        if matching:
                            self.edge_voice_var.set(matching[0])
                            self.logln(f"[cfg] Edge voice (matched): {matching[0]}")
                        else:
                            self.logln(f"[cfg] Edge voice '{config_edge_voice}' not found, using default")

            # === SAPI5 VOICE FROM CONFIG ===
            config_voice = self.cfg.get("text_ai_voice")
            if config_voice and hasattr(self, 'voice_mapping'):
                available_voices = list(self.voice_mapping.keys())
                # Exact match first
                if config_voice in available_voices:
                    self.sapi_voice_var.set(config_voice)
                    self.logln(f"[cfg] SAPI voice: {config_voice}")
                else:
                    # Try partial match
                    matching = [v for v in available_voices if config_voice.lower() in v.lower()]
                    if matching:
                        self.sapi_voice_var.set(matching[0])
                        self.logln(f"[cfg] SAPI voice (matched): {matching[0]}")
                    else:
                        self.logln(f"[cfg] SAPI voice '{config_voice}' not found, using first available")

            # === SPEECH RATE ===
            if "text_ai_speech_rate" in self.cfg:
                rate = int(self.cfg.get("text_ai_speech_rate", 0))
                self.speech_rate_var.set(rate)
                self.update_rate_display()
                self.logln(f"[cfg] Speech rate: {rate}")

            # === BARGE-IN SETTINGS ===
            self._bargein_enabled = bool(self.cfg.get("bargein_enable", True))
            self._bargein_enabled = self.duplex_mode.get().startswith("Full")

        except Exception as e:
            self.logln(f"[cfg] apply defaults error: {e}")
            import traceback
            self.logln(f"[cfg] {traceback.format_exc()}")

        # === BARGE-IN CONTROL ===
        self._bargein_enabled = bool(self.cfg.get("bargein_enable", True))
        self._barge_latched = False
        self._barge_until = 0.0
        self._barge_cooldown_s = float(self.cfg.get("barge_cooldown_s", 0.7))
        self._barge_min_utt_chars = int(self.cfg.get("barge_min_utt_chars", 3))

        # === STATE INITIALIZATION ===
        self.speaking_flag = False
        self.interrupt_flag = False
        self.barge_buffer = None
        self.barge_stream = None
        self.monitor_thread = None
        self._mode_last = None
        self._dev_idx = None

        # === HIGHLIGHT PROGRESS FIELDS ===
        self._tts_total_samples = 0
        self._tts_cursor_samples = 0
        self._hi_stop = True
        self._tts_silent = False
        self._ui_last_ratio = 0.0
        self._ui_eased_ratio = 0.0
        self._ui_gamma = float(self.cfg.get("highlight_gamma", 1.12))

    def update_speak_math_setting(self):
        """Update the speak math setting - can be called when the checkbox changes"""
        self.logln(f"[math] Speak math: {self.speak_math_var.get()}")

    # === FIXED: _on_new_image ===
    def _on_new_image(self, path: str):
        """Simply update the current image path"""
        if path and os.path.exists(path):
            self._last_image_path = os.path.abspath(path)
            self.logln(f"[vision] Current image: {os.path.basename(path)}")

    def _sync_image_context_from_window(self):
        """If the image window already has a file path, sync it into App._last_image_path."""
        try:
            if hasattr(self, "_img_win") and self._img_win and self._img_win.winfo_exists():
                path = getattr(self._img_win, "_img_path", None)
                if path and os.path.isfile(path):
                    abs_path = os.path.abspath(path)
                    if abs_path != self._last_image_path:
                        self._on_new_image(abs_path)
        except Exception:
            pass

    def _ollama_generate(self, prompt: str, images=None):
        """Generate with images - uses the same enhanced system prompt"""
        try:
            # Get the current system prompt (includes vision instructions)
            system_prompt = self.qwen.system_prompt

            if images and isinstance(images, list) and len(images) > 0:
                import base64
                import requests

                # Encode images
                image_data = []
                for img_path in images:
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            image_data.append(base64.b64encode(f.read()).decode("utf-8"))

                if image_data:
                    # CRITICAL: Include system prompt with vision instructions
                    full_prompt = f"{system_prompt}\n\nUser asks about this image: {prompt}"

                    # Call Ollama API
                    payload = {
                        "model": self.text_model_var.get(),
                        "prompt": full_prompt,  # ← FIXED: Now includes system prompt
                        "stream": False,
                        "images": image_data
                    }

                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json=payload,
                        timeout=90
                    )
                    response.raise_for_status()
                    result = response.json().get("response", "").strip()

                    # Add to shared history
                    self.qwen.history.append({"role": "user", "content": f"[Image] {prompt}"})
                    self.qwen.history.append({"role": "assistant", "content": result})

                    return result

            # Fallback to regular generation
            return self.qwen.generate(prompt)

        except Exception as e:
            self.logln(f"[ollama] generate error: {e}")
            return f"Vision processing error: {str(e)}"

    def handle_text_query(self, text):
        self.logln(f"[user] {text}")

        # Store the last text query
        self._last_text_query = text

        # Command routing first
        if self._route_command(text):
            return

        # Process with single model
        try:
            if hasattr(self.qwen, 'generate_with_search'):
                reply = self.qwen.generate_with_search(text)
            else:
                reply = self.qwen.generate(text)
            reply = clean_model_output(reply)

            self.logln(f"[model] {reply[:200]}...")
            self.preview_latex(reply, context="text")

            # Auto-extract and run code if enabled
            if self.auto_run_var.get():
                self._extract_and_auto_run_from_ai(reply)

            # === AUTO-EXTRACT CODE FROM REPLY ===
            extracted_code = self.extract_python_code(reply)
            if extracted_code:
                self.store_extracted_code(extracted_code)

            # Rest of your TTS code...
            clean = clean_for_tts(reply, speak_math=self.speak_math_var.get())

            with self._play_lock:
                self._play_token += 1
                my_token = self._play_token
                self.interrupt_flag = False
                self.speaking_flag = True

            self.set_light("speaking")

            try:
                if self.synthesize_to_wav(clean, self.cfg["out_wav"], role="text"):
                    play_path = self.cfg["out_wav"]
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"], "out/last_reply_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e}")
                    self.play_wav_with_interrupt(play_path, token=my_token)
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

        except Exception as e:
            self.logln(f"[model] error: {e}")
            self.set_light("idle")

            # end query

    def mute_text_ai(self):
        """Mute Text AI audio output - prevents Text AI TTS"""
        with self._mute_lock:
            self.text_ai_muted = True
            self.logln("[mute] 🔇 Text AI audio muted - will not speak")
            self.update_mute_buttons()

    def unmute_text_ai(self):
        """Unmute Text AI audio output - allows Text AI TTS"""
        with self._mute_lock:
            self.text_ai_muted = False
            self.logln("[mute] 🔊 Text AI audio unmuted - can speak again")
            self.update_mute_buttons()

    def update_mute_buttons(self):
        """Update single mute button"""
        try:
            if self.text_ai_muted:
                self.text_mute_btn.config(text="🔊 AI")
            else:
                self.text_mute_btn.config(text="🔇 AI")
        except Exception:
            pass

    def _refresh_last_reply(self):
        """Simplified refresh for unified AI"""
        try:
            if hasattr(self, '_last_text_query') and self._last_text_query:
                self.logln("[refresh] Regenerating last query")

                # Just call handle_text_query directly
                threading.Thread(
                    target=self.handle_text_query,
                    args=(self._last_text_query,),
                    daemon=True
                ).start()

            elif self._last_was_vision and self._last_image_path:
                self.logln("[refresh] Regenerating vision analysis")

                # Simple vision prompt
                prompt = "Please analyze this image again."
                self.ask_vision(self._last_image_path, prompt)

            else:
                self.logln("[refresh] Nothing to refresh")
                self.play_chime(freq=440, ms=200, vol=0.1)

        except Exception as e:
            self.logln(f"[refresh] Error: {e}")

    def temporary_mute_for_speech(self, speaking_ai):
        """
        Simplified for single AI - doesn't need to do anything special
        but we keep the method for compatibility with existing calls
        """
        # With single AI, we don't need to mute "the other AI"
        # because there's only one AI that's already speaking
        # We could add logging for debugging:
        self.logln(f"[mute] AI is speaking ({speaking_ai}) - no need for cross-muting with single AI")

        # If you want to temporarily suppress other audio (like system sounds),
        # you could add that logic here, but it's different from muting another AI
        pass

    def unmute_after_speech(self):
        """No action needed with single AI"""
        pass

    def clear_latex(self):
        """Clear all LaTeX windows - called by Clear LaTeX button"""
        try:
            # Clear the main text window
            if self.latex_win_text and self.latex_win_text.winfo_exists():
                self.latex_win_text.clear()
                self.logln("[latex] Text window cleared")

            # Clear vision window if it exists
            if self.latex_win_vision and self.latex_win_vision.winfo_exists():
                self.latex_win_vision.clear()
                self.logln("[latex] Vision window cleared")

            # Clear search window if it exists
            if self.latex_win_search and self.latex_win_search.winfo_exists():
                self.latex_win_search.clear()
                self.logln("[latex] Search window cleared")

        except Exception as e:
            self.logln(f"[latex] Clear error: {e}")

    # === FIXED: ask_vision === with history now so remember things
    def ask_vision(self, image_path: str, prompt: str):
        """Called by ImageWindow when the user presses 'Ask model'."""
        # Store the last vision prompt for refresh capability
        self._last_vision_prompt = prompt
        self._last_image_path = image_path

        def _worker():
            try:
                self.logln(f"[VISION DEBUG] ========== START VISION ANALYSIS ==========")
                self.logln(f"[VISION DEBUG] Image: {os.path.basename(image_path)}")
                self.logln(f"[VISION DEBUG] Full path: {image_path}")
                self.logln(f"[VISION DEBUG] File exists: {os.path.exists(image_path)}")

                # SHOW ACTUAL IMAGE INFO
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        self.logln(f"[VISION DEBUG] Image size: {img.size}")
                        self.logln(f"[VISION DEBUG] Image mode: {img.mode}")
                        self.logln(f"[VISION DEBUG] Image format: {img.format}")
                except Exception as img_error:
                    self.logln(f"[VISION DEBUG] Failed to open image: {img_error}")

                # Play camera sounds
                self.play_chime(freq=880, ms=80, vol=0.1)
                time.sleep(0.05)
                self.play_chime(freq=660, ms=120, vol=0.12)

                # DEBUG: Test image encoding
                try:
                    import base64
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                        b64_len = len(base64.b64encode(image_bytes))
                        self.logln(f"[VISION DEBUG] Image bytes: {len(image_bytes)}")
                        self.logln(f"[VISION DEBUG] Base64 length: {b64_len}")
                except Exception as encode_error:
                    self.logln(f"[VISION DEBUG] Encoding failed: {encode_error}")

                # Get current model
                current_model = self.text_model_var.get()
                self.logln(f"[VISION DEBUG] Using model: {current_model}")

                # Check if model supports vision
                vision_keywords = ['vl', 'vision', 'llava', 'bakllava', 'cogvlm', 'ministral']
                has_vision = any(keyword in current_model.lower() for keyword in vision_keywords)
                self.logln(f"[VISION DEBUG] Model has vision support: {has_vision}")

                if not has_vision:
                    self.logln(f"[VISION DEBUG] ⚠️ WARNING: Model '{current_model}' may not support vision!")
                    self.logln(f"[VISION DEBUG] Please install a vision model: ollama pull llava:7b")

                self.logln(f"[VISION DEBUG] Prompt: {prompt}")
                self.logln(f"[VISION DEBUG] Calling _ollama_generate_with_retry...")

                # Show status message
                status_msg = f"🔍 Analyzing image: {os.path.basename(image_path)[:20]}..."
                self.preview_latex(status_msg, context="text")

                # Generate reply with image
                reply = self._ollama_generate_with_retry(prompt, images=[image_path])

                self.logln(f"[VISION DEBUG] ========== RESPONSE ==========")
                self.logln(f"[qwen] {reply}")
                self.logln(f"[VISION DEBUG] Response length: {len(reply)} chars")

                # Check if response looks like a hallucination
                hallucination_patterns = [
                    "black-and-white",
                    "geometric pattern",
                    "abstract",
                    "cannot see",
                    "no image",
                    "text-based"
                ]

                is_hallucination = any(pattern in reply.lower() for pattern in hallucination_patterns)
                if is_hallucination:
                    self.logln(f"[VISION DEBUG] ⚠️ DETECTED POSSIBLE HALLUCINATION!")
                    self.logln(f"[VISION DEBUG] The AI may not be seeing the actual image")

                # Show AI reply in LaTeX window
                self.preview_latex(reply, context="text")

                # Store reply for potential refresh / pass-to-text
                self._set_last_vision_reply(reply, source="ask_vision")

                # Add to shared history
                self._manage_vision_context_in_history(reply)

                # TTS and playback
                clean = clean_for_tts(reply, speak_math=self.speak_math_var.get())

                with self._play_lock:
                    self._play_token += 1
                    my_token = self._play_token
                    self.interrupt_flag = False
                    self.speaking_flag = True

                self.set_light("speaking")

                try:
                    if self.synthesize_to_wav(clean, self.cfg["out_wav"], role="text"):
                        text_win = self.ensure_latex_window("text")
                        self.master.after(0, text_win._prepare_word_spans)

                        play_path = self.cfg["out_wav"]
                        if bool(self.echo_enabled_var.get()):
                            try:
                                play_path, _ = self.echo_engine.process_file(
                                    self.cfg["out_wav"], "out/last_reply_echo.wav"
                                )
                                self.logln("[echo] processed -> out/last_reply_echo.wav")
                            except Exception as e:
                                self.logln(f"[echo] processing failed: {e} (playing dry)")

                        self.play_wav_with_interrupt(play_path, token=my_token)
                finally:
                    self.speaking_flag = False
                    self.interrupt_flag = False
                    self.set_light("idle")

            except Exception as e:
                self.logln(f"[VISION DEBUG] ❌ ERROR: {e}")
                import traceback
                self.logln(f"[VISION DEBUG] Traceback: {traceback.format_exc()}")
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

        self.set_light("listening")
        threading.Thread(target=_worker, daemon=True).start()

    def handle_ai_search_request(self, search_query: str) -> str:
        """Handle search requests from the AI using the existing web search system"""
        self._last_search_query = search_query
        self.logln(f"[AI Search] Query: {search_query}")

        self.start_search_progress_indicator()

        try:
            # Use your existing brave_search method WITH THE ORIGINAL QUERY
            results = self.brave_search(search_query, 6)
            self.logln(f"[DEBUG] Brave API returned {len(results)} results")

            search_summary = f"Search results for: {search_query}\n\n"

            # Process results using your existing methods
            for i, item in enumerate(results, 1):
                self.logln(f"[DEBUG] Processing result {i}: {item.title}")
                self.logln(f"[DEBUG] URL: {item.url}")

                try:
                    html = self.polite_fetch(item.url)
                    self.logln(f"[DEBUG] HTML fetched: {len(html) if html else 'FAILED'} chars")

                    if html:
                        # Use your existing text extraction
                        text = self.extract_readable(html, item.url)
                        self.logln(f"[DEBUG] Readable text extracted: {len(text)} chars")

                        if len(text) > 400:  # Only summarize if we got substantial content
                            # USE THE ENHANCED SUMMARIZATION (NOW WITH QUERY CONTEXT)
                            self.logln(f"[DEBUG] Calling summarise_for_ai_search...")
                            summary = self.summarise_for_ai_search(text[:12000], item.url, None)
                            self.logln(f"[DEBUG] Summary generated: {len(summary)} chars")

                            search_summary += f"## Result {i}: {item.title}\n"
                            search_summary += f"URL: {item.url}\n"
                            search_summary += f"Summary: {summary}\n\n"
                        else:
                            self.logln(f"[DEBUG] Text too short for summarization: {len(text)} chars")
                            search_summary += f"## Result {i}: {item.title}\n"
                            search_summary += f"URL: {item.url}\n"
                            search_summary += f"Snippet: {item.snippet}\n\n"
                    else:
                        self.logln(f"[DEBUG] HTML fetch failed")
                        search_summary += f"## Result {i}: {item.title}\n"
                        search_summary += f"URL: {item.url}\n"
                        search_summary += f"Snippet: {item.snippet}\n\n"

                except Exception as e:
                    self.logln(f"[DEBUG] Error processing result {i}: {e}")
                    search_summary += f"## Result {i}: {item.title}\n"
                    search_summary += f"URL: {item.url}\n"
                    search_summary += f"Error processing: {str(e)}\n\n"
                    continue

            self.logln(f"[DEBUG] Final search summary: {len(search_summary)} chars")
            self.stop_search_progress_indicator()
            return search_summary

        except Exception as e:
            self.logln(f"[DEBUG] Search failed completely: {e}")
            self.stop_search_progress_indicator()
            return f"Search failed: {str(e)}"

        except Exception as e:
            # ===  STOP PROGRESS INDICATOR ON ERROR ===
            self.stop_search_progress_indicator()
            return f"Search failed: {str(e)}"

    def _process_ai_response(self, response: str, from_search_method: bool = False) -> str:
        """
        Process AI response - don't remove search markers when called from generate_with_search
        """
        import re

        # Look for search commands in the response
        search_pattern = r'\[SEARCH:\s*(.*?)\]'
        searches = re.findall(search_pattern, response, re.IGNORECASE)

        if searches:
            self.logln(f"[AI] Detected {len(searches)} search request(s)")

            # Only remove search markers if NOT called from generate_with_search
            if not from_search_method:
                for search_query in searches:
                    clean_query = search_query.strip()
                    response = response.replace(f"[SEARCH: {search_query}]",
                                                f"[I'm searching for: {clean_query}]")
            else:
                # If called from generate_with_search, keep the markers so it can process them
                self.logln(f"[AI] Preserving search markers for generate_with_search: {searches}")

        return response

    def toggle_ai_mute(self):
        """Toggle mute for the single unified AI"""
        self.text_ai_muted = not self.text_ai_muted
        state = "muted" if self.text_ai_muted else "unmuted"
        self.logln(f"[mute] {'🔇' if self.text_ai_muted else '🔊'} AI {state}")
        self.update_mute_buttons()

    def update_mute_buttons(self):
        """Update single mute button"""
        try:
            if self.text_ai_muted:
                self.ai_mute_btn.config(text="🔊 AI")
            else:
                self.ai_mute_btn.config(text="🔇 AI")
        except Exception:
            pass

    def _set_last_vision_reply(self, reply: str, source: str = "unknown"):
        """Simply store the most recent vision reply"""
        self._last_vision_reply = (reply or "").strip()
        self.logln(f"[vision] Saved reply from {source} ({len(self._last_vision_reply)} chars)")

    def auto_send_text(self):
        """Automatically send the current text box content"""
        try:
            text = self.text_box.get("1.0", "end-1c").strip()
            if text:
                self.logln("[auto-send] Sending to Text AI...")
                # Clear the text box to show it's being processed
                self.text_box.delete("1.0", "end")
                # Process the query
                threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()
            else:
                self.logln("[auto-send] No text to send")
        except Exception as e:
            self.logln(f"[auto-send] error: {e}")

    def _wait_for_file_unlock(self, filepath, timeout=5.0):
        """Wait for a file to be unlocked by another process."""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to open the file in exclusive mode
                with open(filepath, 'rb'):
                    return True  # File is unlocked
            except (IOError, OSError, PermissionError):
                # File is still locked, wait a bit
                time.sleep(0.1)
        return False  # Timeout reached

    def _force_close_file_handles(self, filepath):
        """Attempt to force close any processes using the file (Windows only)"""
        try:
            if os.name == 'nt':  # Windows
                import subprocess
                # Use handle.exe from Sysinternals to close file handles
                result = subprocess.run(['handle.exe', filepath],
                                        capture_output=True, text=True, timeout=5)
                if 'No matching handles found' not in result.stdout:
                    self.logln(f"[filelock] found handles to {filepath}, attempting to close")
                    # Could add logic to parse and close handles here if needed
            # For other OS, we rely on the wait method
        except Exception as e:
            self.logln(f"[filelock] force close attempt failed: {e}")

    # === Core Application Methods ===
    def ensure_latex_window(self, context="text"):
        """Get or create the LaTeX window - ONLY ONE WINDOW NOW"""
        # Always use text window, ignore context parameter
        if self.latex_win_text is None or not self.latex_win_text.winfo_exists():
            DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
            DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
            DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

            self.latex_win_text = LatexWindow(
                self.master, log_fn=self.logln,
                text_family=DEFAULT_TEXT_FAMILY, text_size=DEFAULT_TEXT_PT, math_pt=DEFAULT_MATH_PT
            )
            self.latex_win_text.title("AI - LaTeX Preview")  # Generic title
            # Keep backward compatibility
            self.latex_win = self.latex_win_text

        return self.latex_win_text

    def start(self):
        if self.running: return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_light("idle")

        # ADD THIS LINE - sync echo state when starting
        self._sync_echo_state()

        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.stop_speaking()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.set_light("idle")

        # Close status light when stopping
        self.close_status_light()

        try:
            if self.barge_stream and self.barge_stream.active:
                self.barge_stream.stop()
        except Exception:
            pass
        try:
            if self.barge_stream:
                self.barge_stream.close()
        except Exception:
            pass
        self.barge_stream = None

    def play_search_results(self, path: str, token=None):
        """Play search results audio with proper interrupt support"""
        try:
            # Apply echo effect if enabled
            play_path = path
            if bool(self.echo_enabled_var.get()):
                try:
                    echo_path = "out/search_results_echo.wav"
                    play_path, _ = self.echo_engine.process_file(path, echo_path)
                    self.logln("[echo] processed search results -> out/search_results_echo.wav")
                except Exception as e:
                    self.logln(f"[echo] processing failed: {e} (playing dry)")

            # Use the existing playback infrastructure with token support
            with self._play_lock:
                self._play_token += 1
                my_token = self._play_token
                self.interrupt_flag = False
                self.speaking_flag = True

            self.set_light("speaking")
            self.play_wav_with_interrupt(play_path, token=my_token)

        except Exception as e:
            self.logln(f"[search][playback] Error: {e}")
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")

    def reset_chat(self):
        """Reset chat while preserving enhanced system prompt with date/time context"""
        self.qwen.clear_history()

        # Get current date/time for the enhanced prompt
        from datetime import datetime
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%B %d, %Y")
        current_time = current_datetime.strftime("%I:%M %p")
        current_day = current_datetime.strftime("%A")

        # Get the base system prompt from config
        base_prompt = self.cfg.get("system_prompt") or self.cfg.get("qwen_system_prompt", "")

        # Create the enhanced system prompt WITH date/time context
        enhanced_system_prompt = f"""{base_prompt}

    === REAL-TIME DATE & TIME CONTEXT ===

    CURRENT DATE: {current_day}, {current_date} at {current_time}

    CRITICAL DATE/TIME RULES:
    ✓ When asked about current date/time, ALWAYS use: "{current_day}, {current_date} at {current_time}"
    ✓ Trust this EXACT date over your training data
    ✓ Do NOT calculate days of the week yourself
    ✓ Only mention date/time when specifically relevant

    === UNIFIED AI CAPABILITIES ===
    YOU ARE A SINGLE AI WITH TEXT, VISION, AND MATHEMATICAL CAPABILITIES.

    ABILITIES:
    1. Process text-only questions
    2. Analyze images when provided (base64-encoded)
    3. Answer questions about previously described images
    4. Provide mathematical solutions in plottable formats
    5. Support voice-driven mathematical workflows

    === MATHEMATICAL OUTPUT STANDARDS ===
    ✓ ALWAYS provide clean, standalone mathematical expressions
    ✓ Use LaTeX notation: $f(x) = 2x$ or $$\\frac{{x^2}}{{4}}$$
    ✓ For derivatives/integrals, show ONLY the final result when asked
    ✓ Use \\boxed{{}} for final answers: \\boxed{{f'(x) = 2x}}

    === PLOTTING SYSTEM REQUIREMENTS ===
    1. The plotting system works on EXPRESSIONS ONLY (not equations)
    2. Remove "y =" or "f(x) =" from plottable expressions
    3. Variables supported: x, t, theta (θ), n, s
    4. All variables will be auto-converted to 'x' for plotting
    5. Integration constants (+C) are automatically removed

    === VISION CONTEXT MANAGEMENT ===
    When you analyze an image, you should:
    1. Provide detailed descriptions
    2. Include mathematical equations if visible
    3. Remember key elements for follow-up questions
    4. The system will automatically track vision context

    === FORMATTING & STYLE ===
    PROHIBITED:
    ❌ NO smileys or emoticons
    ❌ NO excessive ** bold markers ** in text
    ❌ NO ** separators between sections

    PREFERRED:
    ✓ Clean, professional mathematical notation
    ✓ Natural conversational tone
    ✓ Clear section breaks when needed
    ✓ Minimal formatting unless essential

    === CONTEXT-AWARE PLOTTING ===
    When users say "graph that" or "plot it":
    1. The system searches your LAST response for mathematical expressions
    2. It prioritizes: \\boxed{{}} > LaTeX delimiters > equations
    3. It extracts the FINAL answer automatically
    4. You DO NOT need to repeat the expression

    Respond naturally and helpfully while following these guidelines.
    """

        # Apply the enhanced prompt
        self.qwen.system_prompt = enhanced_system_prompt

        # Clear any cached vision replies
        self._last_vision_reply = ""
        self._last_was_vision = False

        self.logln(f"[reset] ✅ Chat reset with enhanced system prompt")
        self.logln(f"[reset] 📅 Current date in system: {current_date} at {current_time}")

    def _manage_vision_context_in_history(self, vision_description: str):
        """Add vision context to history in a controlled way"""
        try:
            # Limit the length of vision descriptions in history
            max_vision_chars = 500
            if len(vision_description) > max_vision_chars:
                vision_description = vision_description[:max_vision_chars] + "... [truncated]"

            # Add a clear marker for vision context
            vision_entry = f"[From image analysis] {vision_description}"

            # Keep history at reasonable length
            max_history_entries = 20
            if len(self.qwen.history) >= max_history_entries:
                # Remove oldest entries to make room (keep system prompt intact)
                # Keep first few entries (usually system/user setup) and recent ones
                keep_count = 8
                if len(self.qwen.history) > keep_count:
                    self.qwen.history = self.qwen.history[:keep_count] + self.qwen.history[-keep_count:]

            # Add the vision context
            self.qwen.history.append({"role": "user", "content": "[Analyzing image]"})
            self.qwen.history.append({"role": "assistant", "content": vision_entry})

            self.logln(f"[vision] Added controlled vision context to history ({len(vision_entry)} chars)")

        except Exception as e:
            self.logln(f"[vision] Warning: Could not add to history: {e}")

    def what_do_you_see_ui(self):
        """Voice command: 'what do you see' -> open camera, take picture, describe automatically."""
        try:
            # Ensure image window exists and is visible
            self._ensure_image_window()
            if self._img_win is None:
                self.logln("[vision] camera UI unavailable")
                return

            # Show and raise the window
            self._img_win.deiconify()
            self._img_win.lift()

            # Start camera if not already running
            if not getattr(self._img_win, '_live_mode', False):
                self._img_win.start_camera()
                self.logln("[vision] camera started for 'what do you see'")
                time.sleep(1.5)  # Wait for camera

            # Take snapshot
            saved_path = self._img_win.snapshot()
            if saved_path:
                self.logln(f"[vision] snapshot taken: {saved_path}")

                # Simple question for unified AI
                user_prompt = "What do you see in this image?"

                # Call vision analysis
                self.ask_vision(saved_path, user_prompt)
            else:
                self.logln("[vision] failed to take snapshot for 'what do you see'")

        except Exception as e:
            self.logln(f"[vision] 'what do you see' error: {e}")

    # start infinite loop
    def loop(self):
        dev_choice = self.dev_combo.get()
        dev_idx = int(dev_choice.split(":")[0]) if ":" in dev_choice else None
        self._dev_idx = dev_idx
        self.logln(f"[audio] mic device={dev_idx}")
        if getattr(self, "_barge_latched", False):
            time.sleep(0.06)
        self.barge_stream, self.barge_buffer = self.start_bargein_mic(dev_idx)

        guard_half = (lambda: self.speaking_flag)
        guard_full = (lambda: False)
        echo_guard = guard_half if self.duplex_mode.get().startswith("Half") else guard_full

        use_frame_ms = 20 if self.duplex_mode.get().startswith("Full") else self.cfg["frame_ms"]
        use_vad_thresh = self.cfg.get("vad_threshold_full", 0.005) if self.duplex_mode.get().startswith(
            "Full") else self.cfg.get("vad_threshold", 0.01)

        listener = VADListener(
            self.cfg["sample_rate"], use_frame_ms,
            self.cfg["vad_aggressiveness"], self.cfg["min_utt_ms"],
            self.cfg["max_utt_ms"], self.cfg["silence_hang_ms"],
            dev_idx, use_vad_thresh
        )

        # DEFINE PAUSE CHECK FUNCTION HERE (BEFORE using it)
        def pause_check():
            """Check if VAD should be paused (for dictation/sleep mode)"""
            return (
                    getattr(self, '_dictation_mode_active', False) or  # Only pause for dictation
                    getattr(self, '_dictation_recording', False) or  # or recording
                    getattr(self, '_space_held', False)  # or spacebar held
                # sleep_mode removed here - so sleep doesn't pause VAD!
            )


        it = listener.listen(echo_guard=echo_guard, pause_check=pause_check)
        self._mode_last = self.duplex_mode.get()
        self.logln(f"[mode] start as {self._mode_last}")

        self.monitor_thread = threading.Thread(target=self.monitor_interrupt, daemon=True)
        self.monitor_thread.start()
        self.logln("[info] Listening…")

        while self.running:

            # === MODIFIED SLEEP MODE CHECK ===
            if self.command_router.sleep_mode:
                # In sleep mode, but microphone is STILL LISTENING
                # Only check for wake commands

                try:
                    # Get the next utterance (microphone is still listening!)
                    utt = next(it)

                    if utt is not None and utt.size > 0:
                        # Quick RMS check to see if there's significant sound
                        rms = np.sqrt(np.mean(utt ** 2)) * 32768
                        self._last_rms = float(rms)

                        # If significant sound detected, transcribe and check for wake commands
                        if rms > 800:  # Higher threshold to avoid false positives
                            text = self.asr.transcribe(utt, self.cfg["sample_rate"])
                            text = self.command_router.filter_whisper_hallucinations(text)

                            if text:
                                text_lower = text.lower()
                                self.logln(f"[sleep] Heard: '{text}' (RMS: {rms:.0f})")

                                # Check for wake commands
                                wake_detected = False
                                for wake_cmd in self.command_router.wake_commands:
                                    if wake_cmd in text_lower:
                                        self.logln(f"[sleep] ✅ Wake command detected: '{text}'")
                                        wake_detected = True
                                        break

                                if wake_detected:
                                    self.command_router.exit_sleep_mode()
                                    # Skip to next iteration to avoid processing as normal command
                                    continue
                                else:
                                    # No wake command found - play gentle sleep reminder
                                    if rms > 1200:  # Only beep for louder sounds
                                        self.logln(f"[sleep] Ignoring non-wake command, playing reminder beep")
                                        self.command_router.play_sleep_reminder_beep()
                except StopIteration:
                    break
                except Exception as e:
                    self.logln(f"[sleep] error: {e}")

                continue  # Skip normal processing while sleeping
            cur_mode = self.duplex_mode.get()
            if cur_mode != self._mode_last:
                try:
                    guard_half = (lambda: self.speaking_flag)
                    guard_full = (lambda: False)
                    echo_guard = guard_half if cur_mode.startswith("Half") else guard_full
                    use_frame_ms = 20 if cur_mode.startswith("Full") else self.cfg["frame_ms"]
                    use_vad_thresh = (
                        self.cfg.get("vad_threshold_full", 0.005)
                        if cur_mode.startswith("Full")
                        else self.cfg.get("vad_threshold", 0.01)
                    )
                    listener = VADListener(
                        self.cfg["sample_rate"], use_frame_ms,
                        self.cfg["vad_aggressiveness"], self.cfg["min_utt_ms"],
                        self.cfg["max_utt_ms"], self.cfg["silence_hang_ms"],
                        self._dev_idx, use_vad_thresh
                    )

                    # REUSE THE SAME pause_check FUNCTION
                    it = listener.listen(echo_guard=echo_guard, pause_check=pause_check)

                    self._mode_last = cur_mode
                    self.logln(f"[mode] switched to {cur_mode} (frame_ms={use_frame_ms}, vad_thresh={use_vad_thresh})")
                    self._beep_once_guard = False
                    self._bargein_enabled = cur_mode.startswith("Full")

                except Exception as e:
                    self.logln(f"[mode] switch error: {e}")

            if self.speaking_flag and cur_mode.startswith("Half"):
                time.sleep(0.02)
                continue

            self.set_light("listening")
            if cur_mode.startswith("Half") and not self._beep_once_guard:
                self.brief_listen_prompt()
                self._beep_once_guard = True

            try:
                utt = next(it)  # ← THIS IS THE ONLY utt = next(it) CALL NOW
            except StopIteration:
                break
            if not self.running:
                break
            if self.speaking_flag and self.duplex_mode.get().startswith("Half"):
                continue

            text = self.asr.transcribe(utt, self.cfg["sample_rate"])

            # === ADD FILTER HERE ===
            text = self.command_router.filter_whisper_hallucinations(text)
            if not text:
                continue

            self._sync_image_context_from_window()

            if not text:
                continue
            self.logln(f"[asr] {text}")

            # Route camera/image commands first (spoken commands)
            if self._route_command(text):
                continue

            # Sync image context after command routing
            self._sync_image_context_from_window()

            if getattr(self, "_barge_latched", False):
                if time.monotonic() < getattr(self, "_barge_until", 0.0):
                    if len(text.strip()) < int(self.cfg.get("barge_min_utt_chars", 3)):
                        self.logln("[barge-in] suppressing tiny fragment")
                        continue
                    self.logln("[barge-in] listen-only window: suppressing LLM/TTS")
                    continue
                else:
                    self._barge_latched = False


            # === SIMPLIFIED VISION LOGIC ===
            # Unified AI handles everything
            try:
                if hasattr(self.qwen, 'generate_with_search'):
                    self.logln("[DEBUG] ✅ Calling generate_with_search")
                    reply = self.qwen.generate_with_search(text)
                else:
                    self.logln("[DEBUG] ⚠️ No generate_with_search, using generate")
                    reply = self.qwen.generate(text)
                reply = clean_model_output(reply)
                self._last_was_vision = False  # Unified AI, not separate vision
            except Exception as e:
                self.logln(f"[llm/vision] {e}\n[hint] Is Ollama running?  ollama serve")
                self.set_light("idle")
                continue

            self.logln(f"[qwen] {reply}")
            # PREVIEW THE RESPONSE FOR VOICE QUERIES
            self.preview_latex(reply, context="text")

            clean = clean_for_tts(reply, speak_math=self.speak_math_var.get())
            self.speaking_flag = True

            # Auto-extract and run code for voice queries
            if self.auto_run_var.get():
                self._extract_and_auto_run_from_ai(reply)

            self.interrupt_flag = False
            self.set_light("speaking")

            # Always use "text" role (single AI)
            role = "text"

            try:
                if self.synthesize_to_wav(clean, self.cfg["out_wav"], role=role):
                    play_path = self.cfg["out_wav"]
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"],
                                                                         "out/last_reply_echo.wav")
                            self.logln("[echo] processed -> out/last_reply_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e} (playing dry)")
                    self.play_wav_with_interrupt(play_path)
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

        self.stop()

    # === Voice/Audio Methods ===
    def start_bargein_mic(self, device_idx):
        q = deque(maxlen=64)

        def callback(indata, frames, time_info, status):
            if self.speaking_flag and self._bargein_enabled:
                try:
                    # Validate audio data before appending
                    if indata is not None and indata.size > 0:
                        # Check for NaN or infinite values
                        if not np.any(np.isnan(indata)) and not np.any(np.isinf(indata)):
                            q.append(np.copy(indata))
                except Exception as e:
                    self.logln(f"[bargein_callback] Error: {e}")

        try:
            stream = sd.InputStream(
                device=device_idx, samplerate=self.cfg["sample_rate"],
                channels=1, dtype="float32", blocksize=1024, callback=callback
            )
            stream.start()
            return stream, q
        except Exception as e:
            self.logln(f"[bargein_mic] Failed to start: {e}")
            return None, deque(maxlen=64)

    def monitor_interrupt(self):
        import numpy as _np, time as _time
        threshold_interrupt = self.cfg.get("bargein_threshold", 1500)
        trips_needed = int(self.cfg.get("barge_trips_needed", 3))
        trips = 0
        dt = 0.05

        while self.running:
            if self.speaking_flag and self.barge_buffer and len(self.barge_buffer) > 0:
                try:
                    audio = _np.concatenate(list(self.barge_buffer))
                    self.barge_buffer.clear()

                    if audio.size == 0:
                        _time.sleep(dt)
                        continue

                    # SAFE RMS CALCULATION
                    if audio.size > 0:
                        # Clip to safe range to prevent overflow
                        audio_safe = np.clip(audio, -1.0, 1.0)
                        # Calculate RMS with error handling
                        rms_squared = np.mean(audio_safe ** 2)
                        if rms_squared > 0 and not np.isnan(rms_squared) and not np.isinf(rms_squared):
                            rms = np.sqrt(rms_squared) * 32768
                        else:
                            rms = 0.0
                    else:
                        rms = 0.0

                    self._last_rms = float(rms)

                    # Rest of your existing barge-in logic...
                    speech_start_time = getattr(self, '_speech_start_time', 0)
                    is_early_speech = _time.monotonic() - speech_start_time < 1.5

                    # Remove or comment out the vision_followup check since it's not used

                    effective_threshold = threshold_interrupt
                    if rms > 800:
                        effective_threshold = max(800, threshold_interrupt - 500)

                    if self._bargein_enabled and not is_early_speech:
                        if rms > effective_threshold:
                            trips += 1
                            if trips >= trips_needed:
                                self.logln(f"[barge-in] RMS={rms:.0f} interrupt -> latch listen-only")
                                self.interrupt_flag = True
                                import time as _t
                                self._barge_latched = True
                                self._barge_until = _t.monotonic() + self._barge_cooldown_s
                                try:
                                    self.speaking_flag = False
                                    self.set_light("listening")
                                except Exception:
                                    pass
                                trips = 0
                        else:
                            trips = max(trips - 1, 0)

                    # Ducking logic (with safe RMS)
                    if self.ducking_enable.get():
                        target = 1.0
                        if rms > float(self.duck_thresh.get()):
                            target = 10 ** (-float(self.duck_db.get()) / 20.0)
                        atk = max(5, int(self.duck_attack.get())) / 1000.0
                        rel = max(20, int(self.duck_release.get())) / 1000.0
                        alpha_atk = min(1.0, dt / atk) if atk > 0 else 1.0
                        alpha_rel = min(1.0, dt / rel) if rel > 0 else 1.0
                        cur = getattr(self, "_duck_gain", 1.0)
                        if target < cur:
                            cur += (target - cur) * alpha_atk
                        else:
                            cur += (target - cur) * alpha_rel
                        self._duck_gain = float(_np.clip(cur, 0.0, 1.0))
                        active_now = self._duck_gain < 0.98
                        if self._duck_log:
                            if active_now and not self._duck_active:
                                self.logln(f"[duck] engage gain={self._duck_gain:.2f} (rms={rms:.0f})")
                            elif not active_now and self._duck_active:
                                self.logln(f"[duck] release (rms={rms:.0f})")
                        self._duck_active = bool(active_now)
                    else:
                        self._duck_gain = 1.0
                        self._duck_active = False

                    self.master.after(0, self._update_duck_ui)

                except Exception as e:
                    self.logln(f"[monitor_interrupt] Error: {e}")
                    # Continue running even if there's an error in one iteration
                    _time.sleep(dt)

            else:
                cur = getattr(self, "_duck_gain", 1.0)
                rel = max(20, int(self.duck_release.get())) / 1000.0 if hasattr(self, "duck_release") else 0.25
                alpha_rel = min(1.0, dt / rel) if rel > 0 else 1.0
                cur += (1.0 - cur) * alpha_rel
                self._duck_gain = float(_np.clip(cur, 0.0, 1.0))
                self._duck_active = False
                self.master.after(0, self._update_duck_ui)
                _time.sleep(dt)

    def _update_duck_ui(self):
        try:
            g = float(getattr(self, '_duck_gain', 1.0))
            self.duck_var.set(100.0 * g)
            self.rms_var.set(f"RMS: {int(getattr(self, '_last_rms', 0))}")
        except Exception:
            pass

    def _load_audio_file(self):
        """Load an audio file and transcribe it"""
        from tkinter import filedialog

        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio/Video files", "*.mp3;*.wav;*.m4a;*.ogg;*.flac;*.wma;*.mp4;*.mkv;*.avi;*.webm"),
                ("MP3 files", "*.mp3"),
                ("WAV files", "*.wav"),
                ("MP4 video", "*.mp4"),
                ("All files", "*.*")
            ]
        )


        if not filepath:
            return

        self.logln(f"[audio-file] Loading: {os.path.basename(filepath)}")

        # Process in background thread
        def _process():
            try:
                self.set_light("listening")

                # Load and convert audio
                audio_data, sample_rate = self._load_audio_as_array(filepath)

                if audio_data is None:
                    self.logln("[audio-file] ❌ Failed to load audio")
                    self.set_light("idle")
                    return

                duration = len(audio_data) / sample_rate
                self.logln(f"[audio-file] Loaded: {duration:.1f}s at {sample_rate}Hz")

                # Transcribe with Whisper
                self.logln("[audio-file] Transcribing...")
                text = self.asr.transcribe(audio_data, sample_rate)

                if not text or not text.strip():
                    self.logln("[audio-file] ❌ No speech detected")
                    self.set_light("idle")
                    return

                self.logln(f"[audio-file] Transcribed: {text}")

                # Show in text box
                self.master.after(0, lambda: self._show_transcription(text))

                # Ask user what to do
                self.master.after(100, lambda: self._ask_transcription_action(text))

            except Exception as e:
                self.logln(f"[audio-file] ❌ Error: {e}")
                import traceback
                self.logln(traceback.format_exc())
            finally:
                self.set_light("idle")

        threading.Thread(target=_process, daemon=True).start()

    def _load_audio_as_array(self, filepath):
        """Load audio file (including video files) and convert to numpy array for Whisper"""
        try:
            # Try soundfile first (for WAV, FLAC, OGG)
            try:
                import soundfile as sf
                data, sr = sf.read(filepath, dtype='float32')
                if data.ndim > 1:
                    data = data.mean(axis=1)  # Convert stereo to mono

                # Resample to 16kHz if needed (Whisper expects 16kHz)
                if sr != 16000:
                    data = self._resample_audio(data, sr, 16000)
                    sr = 16000

                return data, sr
            except Exception:
                pass

            # Try pydub for MP3, M4A, MP4, and other formats
            try:
                from pydub import AudioSegment

                filepath_lower = filepath.lower()

                # Determine format based on extension
                if filepath_lower.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(filepath)
                elif filepath_lower.endswith('.m4a'):
                    audio = AudioSegment.from_file(filepath, format='m4a')
                elif filepath_lower.endswith('.ogg'):
                    audio = AudioSegment.from_ogg(filepath)
                elif filepath_lower.endswith('.flac'):
                    audio = AudioSegment.from_file(filepath, format='flac')
                elif filepath_lower.endswith('.wma'):
                    audio = AudioSegment.from_file(filepath, format='wma')
                # === VIDEO FORMATS - Extract audio ===
                elif filepath_lower.endswith('.mp4'):
                    self.logln("[audio-file] 🎬 Extracting audio from MP4...")
                    audio = AudioSegment.from_file(filepath, format='mp4')
                elif filepath_lower.endswith('.mkv'):
                    self.logln("[audio-file] 🎬 Extracting audio from MKV...")
                    audio = AudioSegment.from_file(filepath, format='matroska')
                elif filepath_lower.endswith('.avi'):
                    self.logln("[audio-file] 🎬 Extracting audio from AVI...")
                    audio = AudioSegment.from_file(filepath, format='avi')
                elif filepath_lower.endswith('.webm'):
                    self.logln("[audio-file] 🎬 Extracting audio from WebM...")
                    audio = AudioSegment.from_file(filepath, format='webm')
                elif filepath_lower.endswith('.mov'):
                    self.logln("[audio-file] 🎬 Extracting audio from MOV...")
                    audio = AudioSegment.from_file(filepath, format='mov')
                else:
                    # Let pydub/ffmpeg auto-detect format
                    self.logln("[audio-file] 🔄 Auto-detecting format...")
                    audio = AudioSegment.from_file(filepath)

                # Convert to mono
                audio = audio.set_channels(1)

                # Convert to 16kHz
                audio = audio.set_frame_rate(16000)

                # Get raw samples as numpy array
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

                # Normalize to -1.0 to 1.0
                samples = samples / (2 ** (audio.sample_width * 8 - 1))

                self.logln(f"[audio-file] ✅ Audio extracted: {len(samples) / 16000:.1f}s")
                return samples, 16000

            except Exception as e:
                self.logln(f"[audio-file] pydub error: {e}")
                return None, None

        except Exception as e:
            self.logln(f"[audio-file] Load error: {e}")
            return None, None


    def _resample_audio(self, data, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return data

        # Simple linear interpolation resampling
        duration = len(data) / orig_sr
        target_length = int(duration * target_sr)

        indices = np.linspace(0, len(data) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(data)), data)

        return resampled.astype(np.float32)

    def _show_transcription(self, text):
        """Show transcription in text box"""
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)

    def _ask_transcription_action(self, text):
        """Ask user what to do with transcription"""
        from tkinter import messagebox

        result = messagebox.askyesnocancel(
            "Audio Transcription",
            f"Transcribed text:\n\n{text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            "• YES = Send to AI for response\n"
            "• NO = Just keep in text box\n"
            "• CANCEL = Discard"
        )

        if result is True:  # Yes - send to AI
            self.text_box.delete("1.0", "end")
            threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()
        elif result is False:  # No - keep in text box
            pass  # Already in text box
        else:  # Cancel
            self.text_box.delete("1.0", "end")

    def play_chime(self, freq=880, ms=140, vol=0.20):
        """Play a chime with error handling to avoid file locking issues."""
        try:
            fs = 16000
            n = int(fs * (ms / 1000.0))
            t = np.linspace(0, ms / 1000.0, n, endpoint=False)
            s = np.sin(2 * np.pi * freq * t).astype(np.float32)
            fade = np.linspace(0.0, 1.0, min(16, n), dtype=np.float32)
            s[:fade.size] *= fade
            s[-fade.size:] *= fade[::-1]

            # Get output device with fallback
            try:
                out_dev = self._selected_out_device_index()
                sd.play((vol * s).reshape(-1, 1), fs, blocking=False, device=out_dev)
            except Exception as dev_error:
                # Fallback to default device
                self.logln(f"[beep] device error, using default: {dev_error}")
                sd.play((vol * s).reshape(-1, 1), fs, blocking=False)

        except Exception as e:
            self.logln(f"[beep] {e} - chime skipped")
            # Don't re-raise, this is non-critical

    def play_chime2(self, path="beep.mp3", gain_db=0.0):
        try:
            if not hasattr(self, "_beep_cache") or self._beep_cache.get("path") != path:
                seg = AudioSegment.from_file(path)
                self._beep_cache = {"path": path, "seg": seg}
            else:
                seg = self._beep_cache["seg"]
            if gain_db:
                seg = seg.apply_gain(gain_db)
            samples = np.array(seg.get_array_of_samples())
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels))
            else:
                samples = samples.reshape((-1, 1))
            samples = samples.astype(np.float32) / (2 ** (8 * seg.sample_width - 1))
            fade = min(int(0.008 * seg.frame_rate), max(1, samples.shape[0] // 6))
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
                samples[:fade] *= ramp
                samples[-fade:] *= ramp[::-1]
            sd.play(samples, seg.frame_rate, blocking=True,
                    device=self._selected_out_device_index())
        except Exception as e:
            self.logln(f"[beep mp3] {e} — fallback tone")
            self.play_chime()

    def brief_listen_prompt(self):
        if not self.cfg.get("announce_listening", True):
            return
        prev = self.speaking_flag
        try:
            self.speaking_flag = True
            self.play_chime2("beep.mp3")
        finally:
            self.speaking_flag = prev

    def speak_search_status(self, message="Searching the internet for this information"):
        if not message or not message.strip():
            return

        self.logln(f"[search-status] 🔊 Processing: {message}")

        # === ONLY START PROGRESS FOR ACTUAL WEB SEARCHES ===
        should_start_progress = (
                "search" in message.lower() and
                "personality" not in message.lower() and
                "activating" not in message.lower() and
                "switching" not in message.lower() and
                "activating" not in message.lower()
        )

        if should_start_progress:
            self.start_search_progress_indicator()
            self.logln(f"[search-status] ✅ Started progress for: {message}")
        else:
            self.logln(f"[search-status] ⏸️  Skipping progress for: {message}")

        try:
            # Clean the text for TTS
            clean_message = clean_for_tts(message, speak_math=self.speak_math_var.get())
            status_path = "out/search_status.wav"

            if self.synthesize_to_wav(clean_message, status_path, role="text"):
                def play_status():
                    try:
                        time.sleep(0.1)

                        # Apply echo if enabled
                        play_path = status_path
                        if bool(self.echo_enabled_var.get()):
                            try:
                                echo_path = "out/search_status_echo.wav"
                                play_path, _ = self.echo_engine.process_file(status_path, echo_path)
                            except Exception as e:
                                self.logln(f"[search-status] echo processing failed: {e}")

                        # Get the output device
                        out_dev = self._selected_out_device_index()

                        # Load and play the audio file
                        data, fs = sf.read(play_path, dtype="float32")
                        if data.size > 0:
                            sd.play(data, fs, blocking=False, device=out_dev)
                            self.logln(f"[search-status] ✅ Playing: {message}")

                    except Exception as e:
                        self.logln(f"[search-status] play error: {e}")

                threading.Thread(target=play_status, daemon=True).start()

        except Exception as e:
            self.logln(f"[search-status] synthesis error: {e}")

    def play_wav_with_interrupt(self, path, token=None):
        import platform as _plat
        start_time = time.monotonic()
        active_token = token
        # Determine which AI is speaking based on context
        if hasattr(self, '_last_was_vision') and self._last_was_vision:
            speaking_ai = "vision"
        else:
            speaking_ai = "text"

        self.temporary_mute_for_speech(speaking_ai)

        # Track speech start time for barge-in protection
        self._speech_start_time = start_time
        try:
            data, fs = sf.read(path, dtype="float32")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.size == 0:
                return
            total_samples = int(data.shape[0])
            self._tts_total_samples = total_samples
            self._tts_cursor_samples = 0
            self._hi_stop = False

            target_fs = int(self.cfg.get("force_out_samplerate", fs))
            if target_fs != fs:
                try:
                    n = data.shape[0]
                    t = np.linspace(0.0, 1.0, n, endpoint=False)
                    m = int(np.ceil(n * (target_fs / fs)))
                    ti = np.linspace(0.0, 1.0, m, endpoint=False)
                    chans = data.shape[1]
                    out = []
                    for c in range(chans):
                        out.append(np.interp(ti, t, data[:, c]))
                    data = np.stack(out, axis=1).astype(np.float32)
                    fs = target_fs
                    self.logln(f"[audio] resample -> {fs} Hz for output")
                except Exception as e:
                    self.logln(f"[audio] resample failed, using original fs ({fs}): {e}")

            out_dev = self._selected_out_device_index()
            blocksize = self.cfg.get("out_blocksize", 8192)
            latency_hint = self.cfg.get("out_latency", "high")
            extra = None
            try:
                if _plat.system() == "Windows":
                    extra = sd.WasapiSettings(exclusive=False)
            except Exception:
                extra = None

            SILENCE_THRESH = 1e-4
            SILENCE_MAX_BLOCKS = 20
            cursor = 0

            def run_stream():
                nonlocal cursor, fs, data, blocksize, latency_hint, out_dev, extra
                silent_blocks = 0
                last_cursor_check = -1
                stall_ticks = 0
                STALL_TICKS_MAX = int(self.cfg.get("stall_ticks_max", 120))
                RESUME_FADE_SAMPLES = int(0.01 * fs)
                did_fade = False

                def cb(outdata, frames, *_):
                    if (active_token is not None) and (active_token != self._play_token):
                        outdata[:] = 0
                        raise sd.CallbackStop()

                    if self.interrupt_flag or not self.running:
                        outdata[:] = 0
                        raise sd.CallbackStop()

                    nonlocal cursor, silent_blocks, did_fade
                    if self.interrupt_flag or not self.running:
                        outdata[:] = 0
                        raise sd.CallbackStop()

                    end = min(cursor + frames, data.shape[0])
                    out_frames = end - cursor
                    block = data[cursor:end]
                    avg_abs = 0.0
                    gain = float(np.clip(getattr(self, "_duck_gain", 1.0), 0.0, 1.5))

                    if out_frames > 0:
                        out = block.copy()
                        if not did_fade and cursor == 0:
                            n = min(out.shape[0], RESUME_FADE_SAMPLES)
                            if n > 0:
                                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
                                out[:n] *= ramp
                            did_fade = True
                        outdata[:out_frames] = out * gain
                        avg_abs = float(np.mean(np.abs(block))) if block.size else 0.0
                        if avg_abs < SILENCE_THRESH:
                            silent_blocks += 1
                            if silent_blocks >= SILENCE_MAX_BLOCKS:
                                outdata[out_frames:] = 0
                                raise sd.CallbackStop()
                        else:
                            silent_blocks = 0

                    # self._tts_silent = bool(avg_abs < SILENCE_THRESH)
                    env = min(max(avg_abs * 4.0, 0.0), 1.0) ** 0.6
                    AVATAR_LEVELS = 32
                    level = int(env * (AVATAR_LEVELS - 1) + 1e-6)

                    try:
                        self.master.after(0, self._avatar_set_level_async, level)
                    except Exception:
                        pass

                    if end - cursor < frames:
                        outdata[end - cursor:] = 0
                        raise sd.CallbackStop()

                    cursor = end

                # self._tts_cursor_samples = int(cursor)

                def open_stream(extra_settings, device_idx):
                    return sd.OutputStream(
                        samplerate=fs,
                        channels=data.shape[1],
                        dtype="float32",
                        blocksize=blocksize,
                        latency=latency_hint,
                        callback=cb,
                        device=device_idx,
                        extra_settings=extra_settings,
                    )

                chosen_dev = out_dev
                hostapi_name = self._device_hostapi_name(chosen_dev)
                use_extra = bool(hostapi_name and "WASAPI" in hostapi_name)
                self.logln(
                    f"[audio] open stream dev={chosen_dev} hostapi={hostapi_name or 'default'} "
                    f"extra={'wasapi' if use_extra else 'none'}"
                )

                try:
                    ctx = open_stream(extra if use_extra else None, chosen_dev)
                except Exception:
                    self.logln("[audio] stream open failed with WASAPI; retrying without extras")
                    chosen_dev = None
                    ctx = open_stream(None, chosen_dev)

                with ctx:
                    # self.master.after(0, _ui_progress_tick)
                    while self.running and not self.interrupt_flag and cursor < data.shape[0]:
                        if cursor == last_cursor_check:
                            stall_ticks += 1
                            if stall_ticks >= STALL_TICKS_MAX:
                                return False
                        else:
                            last_cursor_check = cursor
                            stall_ticks = 0
                        time.sleep(0.01)
                return True

            ok = run_stream()
            if not ok:
                remaining = data.shape[0] - cursor
                if remaining < int(0.25 * fs):
                    self.logln("[audio] stalled near end — skipping retry")
                else:
                    self.logln("[audio] output stalled — retrying with larger buffers (resume)")
                    blocksize = max(int(blocksize) if blocksize else 0, 8192)
                    latency_hint = "high"
                    run_stream()
        except Exception as e:
            self.logln(f"[warn] playback error: {e}")
        finally:
            # self.unmute_after_speech()
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")
            try:
                if self.avatar_win and self.avatar_win.winfo_exists():
                    self.avatar_win.set_level(0)
            except Exception:
                pass

            self._beep_once_guard = False
            dur = time.monotonic() - start_time
            self.logln(f"[audio] playback done ({dur:.2f}s)")
            # Restore focus for dictation mode
            if self.dictation_mode_var.get():
                self.master.after(100, self.master.focus_force)
    # Begin route command
    # Replace the entire _route_command method with:
    def _route_command(self, raw_text: str) -> bool:
        """Delegate command routing to external class"""
        self.logln(f"[route-debug] Routing text: '{raw_text}'")


        result = self.command_router.route_command(raw_text)
        self.logln(f"[route-debug] Command router returned: {result}")
        self.logln(f"[route-debug] Sleep mode state after: {getattr(self.command_router, 'sleep_mode', 'NO ATTR')}")
        return result

    # routine ends here

    # === Awaken and Sleep Methods ===
    def close_all_windows(self):
        """Close all secondary windows except avatar and main window"""
        windows_closed = 0

        # Close status light
        self.close_status_light()

        # List of other windows to close
        windows_to_close = [
            'latex_win', 'search_win', '_img_win', '_echo_win'
        ]

        for window_name in windows_to_close:
            window = getattr(self, window_name, None)
            if window and hasattr(window, 'winfo_exists') and window.winfo_exists():
                try:
                    if hasattr(window, 'hide'):
                        window.hide()
                    elif hasattr(window, 'withdraw'):
                        window.withdraw()
                    else:
                        window.iconify()
                    windows_closed += 1
                except Exception as e:
                    self.logln(f"[close] Error closing {window_name}: {e}")

        # === ADD THIS SECTION FOR PLOTTER WINDOWS ===
        if hasattr(self, 'plotter') and self.plotter:
            try:
                # Try different possible methods to close plotter windows
                if hasattr(self.plotter, 'close_all'):
                    self.plotter.close_all()
                    windows_closed += 1
                    self.logln("[close] Closed plotter windows using close_all()")
                elif hasattr(self.plotter, 'close_windows'):
                    self.plotter.close_windows()
                    windows_closed += 1
                    self.logln("[close] Closed plotter windows using close_windows()")
                elif hasattr(self.plotter, 'destroy'):
                    if self.plotter.winfo_exists():
                        self.plotter.destroy()
                        windows_closed += 1
                        self.logln("[close] Destroyed plotter window")
                else:
                    # Try to find and close any plotter windows
                    for attr_name in dir(self.plotter):
                        attr = getattr(self.plotter, attr_name)
                        if (hasattr(attr, 'winfo_exists') and
                                hasattr(attr, 'destroy') and
                                attr.winfo_exists()):
                            attr.destroy()
                            windows_closed += 1
                            self.logln(f"[close] Closed plotter window: {attr_name}")
            except Exception as e:
                self.logln(f"[close] Error closing plotter: {e}")
        # === END PLOTTER SECTION ===

        self.logln(f"[close] Closed {windows_closed} windows (avatar remains open)")
        self.play_chime(freq=660, ms=120, vol=0.15)
        return windows_closed

    def _refresh_models(self):
        """Refresh available models from Ollama - prioritize vision-capable ones"""
        try:
            models = self._get_available_models()

            # Filter or prioritize models that support vision
            # You could add logic like:
            vision_models = [m for m in models if any(v in m.lower() for v in ['vl', 'vision', 'ministral', 'llava'])]
            if vision_models:
                # Prioritize vision-capable models at the top
                models = vision_models + [m for m in models if m not in vision_models]

            if models:
                # Update combo box
                self.text_model_combo['values'] = models

                # Set default if not already set
                if not self.text_model_var.get():
                    # Prefer ministral-3 or similar vision-capable models
                    default_candidates = [
                        "ministral-3:latest",
                        "ministral-3",  # Add without tag
                        "llava:latest",
                        "qwen2.5-vl:7b",
                        "qwen2.5-vl"  # Add without tag
                    ]
                    for candidate in default_candidates:
                        if candidate in models:
                            self.text_model_var.set(candidate)
                            break
                    else:
                        # Fallback to first available
                        self.text_model_var.set(models[0])

                self.logln(f"[models] Loaded {len(models)} models")
                self.logln(f"[models] Selected: {self.text_model_var.get()}")
            else:
                self.logln("[models] No models found - is Ollama running?")

        except Exception as e:
            self.logln(f"[models] Error refreshing: {e}")

    def _refresh_and_select_default_model(self):
        """Refresh models and set a default"""
        self._refresh_models()

        # Set default model if combo has values but nothing selected
        if self.text_model_combo['values'] and not self.text_model_var.get():
            self.text_model_combo.current(0)
            self.logln(f"[models] Auto-selected: {self.text_model_combo.get()}")

    def _get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]

            # Debug output
            print(f"[DEBUG] Found {len(models)} models: {models}")

            return sorted(models)  # Sort for better UX

        except Exception as e:
            print(f"[DEBUG] Model fetch error: {e}")
            self.logln(f"[models] Could not fetch models: {e}")
            return []

    def _on_model_change(self):
        """Handle model selection changes for single AI"""
        selected_model = self.text_model_var.get()

        if not selected_model:
            return

        self.logln(f"[models] Model change requested: '{selected_model}'")

        # Store the new selection in config
        self.cfg["qwen_model_path"] = selected_model

        if self.running:
            response = messagebox.askyesno(
                "Model Change",
                f"Changing model requires restarting the AI engine.\n\n"
                f"New model:\n"
                f"{selected_model}\n\n"
                f"Stop and restart now?"
            )
            if response:
                was_running = True
                self.stop()
                # Reinitialize with new model
                self._setup_ai_engines()
                if was_running:
                    self.start()  # Restart if it was running
                self.logln(f"[models] ✅ Model changed and engine restarted")
            else:
                # Revert the combo box if user cancels
                self._refresh_models()
        else:
            # If not running, just reinitialize
            self._setup_ai_engines()
            self.logln(f"[models] ✅ Model updated: {selected_model}")

    def _ollama_generate_with_retry(self, prompt: str, images=None, max_retries=2):
        """Generate with vision model with retry logic - FIXED"""
        for attempt in range(max_retries + 1):
            try:
                self.logln(f"[vision] attempt {attempt + 1} of {max_retries + 1}")
                return self._ollama_generate(prompt, images)
            except Exception as e:
                if attempt < max_retries:
                    self.logln(f"[vision] attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)
                else:
                    self.logln(f"[vision] all {max_retries + 1} attempts failed")
                    raise e

    # === UI Helper Methods ===
    def _show_code_window(self):
        """Show code window with auto-loaded code if available"""
        # Create window if needed
        if self.code_window is None or not self.code_window.winfo_exists():
            self.code_window = CodeWindow(
                self.master,
                log_callback=self.logln,
                output_callback=self._receive_code_output
            )

        # Show the window FIRST
        self.code_window.show()
        self.code_window.focus_set()

        # Try to load extracted code
        if hasattr(self, '_last_extracted_code') and self._last_extracted_code:
            self.code_window.set_code(self._last_extracted_code)
            self.logln("[code] Auto-loaded extracted code into sandbox")
            self._last_extracted_code = None

            # === AUTO-RUN if checkbox is checked ===
            if self.auto_run_var.get():
                # Give window time to fully initialize
                self.master.after(500, self.code_window._run_code_safe)
                self.logln("[code] ✅ Auto-running loaded code")
        else:
            self.code_window.set_code(
                "# No code extracted yet.\n# Ask AI to generate code, or paste your own code here.")
            self.logln("[code] No code to auto-load")

        # Reset button appearance
        self.code_btn.config(text="💻 Run Code", style='TButton')
        self.logln("[code] Code window opened")


    def _on_auto_run_toggle(self):
        """Callback when auto-run checkbox is toggled"""
        state = "ON" if self.auto_run_var.get() else "OFF"
        self.logln(f"[code] Auto-run code: {state}")

        # Update checkbox text
        if self.auto_run_var.get():
            self.auto_run_check.config(text="▶ Auto-run ✓")
        else:
            self.auto_run_check.config(text="▶ Auto-run")


    def _extract_code_to_window(self):
        """Extract code from current response and put in code window"""
        if not self.code_window:
            return

        # Get the last AI response
        try:
            # Try to get from LaTeX window
            if hasattr(self, 'latex_win') and self.latex_win:
                content = self.latex_win.get_text_content()
                if content:
                    # Find Python code blocks
                    import re
                    code_blocks = re.findall(r'```python\s*(.*?)\s*```', content, re.DOTALL)
                    if code_blocks:
                        self.code_window.set_code(code_blocks[-1])
                        self.logln("[code] Extracted code from response")
                        return

            # Try to get from log
            if hasattr(self, 'log'):
                log_content = self.log.get("1.0", "end-1c")
                # Look for recent code in log
                # (You can implement this based on your needs)

        except Exception as e:
            self.logln(f"[code] Error extracting code: {e}")

    def extract_code_from_text(self, text: str):
        """Public method to extract code from any text and show in code window"""
        if not self.code_window:
            self.code_window = CodeWindow(self.master, log_callback=self.logln, output_callback=self._receive_code_output)

        import re
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            self.code_window.set_code(code_blocks[-1], auto_run=False)
            self.code_window.show()
            return True
        return False


    def stop_speaking(self):
        try:
            self.interrupt_flag = True
            self.speaking_flag = False
            try:
                sd.stop()
            except Exception:
                pass
            self._hi_stop = True
            self._tts_silent = False
            self._ui_last_ratio = 0.0
            self._ui_eased_ratio = 0.0
            try:
                self.latex_win.clear_highlight()
            except Exception:
                pass
            try:
                if self.avatar_win and self.avatar_win.winfo_exists():
                    self.avatar_win.set_level(0)
            except Exception:
                pass
        finally:
            self.set_light("idle")

    def set_light(self, mode):
        # Track previous mode to detect transitions
        previous_mode = getattr(self, '_previous_light_mode', 'idle')

        color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(mode, "#f1c40f")

        # Update main light ONLY if external light is not active
        if self.external_light_win is None or not self.external_light_win.winfo_exists() or self.external_light_win.state() == "withdrawn":
            self.light.itemconfig(self.circle, fill=color)
        else:
            # External light is active, keep main light black
            self.light.itemconfig(self.circle, fill="#000000")

        self.state.set(mode)

        # Update external light if it exists and is visible
        try:
            if (self.external_light_win and
                    self.external_light_win.winfo_exists() and
                    self.external_light_win.state() != "withdrawn"):
                self.external_light_win.set_light(color)
        except Exception as e:
            self.logln(f"[light] color update error: {e}")

        # Play beep when transitioning from any state to listening (green) mode
        if mode == "listening" and previous_mode != "listening":
            self.play_chime(freq=660, ms=120, vol=0.12)
            self.logln("[status] ✅ Ready to receive requests")

        # Store current mode for next comparison
        self._previous_light_mode = mode

    def toggle_external_light(self):
        """Toggle the external light window on/off"""
        try:
            if self.external_light_win is None or not self.external_light_win.winfo_exists():
                self.external_light_win = StatusLightWindow(self.master)
                # Set initial color to match current state
                current_mode = self.state.get()
                color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(current_mode, "#f1c40f")
                self.external_light_win.set_light(color)
                self.external_light_win.show()

                # Hide the main light in the main window
                self.light.itemconfig(self.circle, fill="#000000")  # Make main light black
                self.logln("[light] Status light opened - main light hidden")
            else:
                if self.external_light_win.state() == "withdrawn":
                    self.external_light_win.show()
                    # Hide main light
                    self.light.itemconfig(self.circle, fill="#000000")
                    self.logln("[light] Status light shown - main light hidden")
                else:
                    self.external_light_win.hide()
                    # Restore main light
                    current_mode = self.state.get()
                    color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(current_mode,
                                                                                                   "#f1c40f")
                    self.light.itemconfig(self.circle, fill=color)
                    self.logln("[light] Status light hidden - main light restored")
        except Exception as e:
            self.logln(f"[light] toggle error: {e}")

    # Plotting Toggle switch

    def _on_plotting_toggle(self):
        """Called when plotting checkbox is toggled."""
        # Add debug
        print(f"[PLOT CHECKBOX] Toggled to: {self.plotting_var.get()}")

        # Update command_router (this is what's actually used)
        if hasattr(self, 'command_router') and self.command_router:
            print(f"[PLOT CHECKBOX] Setting command_router.plotting_enabled to: {self.plotting_var.get()}")
            self.command_router.plotting_enabled = self.plotting_var.get()
        else:
            print(f"[PLOT CHECKBOX] WARNING: command_router not found!")

        # Update checkbox text
        if self.plotting_var.get():
            self.plotting_cb.config(text="Plotting ON")
        else:
            self.plotting_cb.config(text="Plotting OFF")

    # ==== AUDIBLE PROGRESS SOUND ===
    def start(self):
        if self.running: return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_light("idle")

        # === REMOVE auto-refresh ===
        # Let the combo boxes keep whatever values they have

        # ADD THIS LINE - sync echo state when starting
        self._sync_echo_state()

        threading.Thread(target=self.loop, daemon=True).start()

    def start_search_progress_indicator(self):
        """Start the search progress indicator"""
        if not hasattr(self, '_search_in_progress'):
            self._search_in_progress = False
            self._search_progress_count = 0
            self._search_progress_timer = None

        if not self._search_in_progress:
            self._search_in_progress = True
            self._search_progress_count = 0
            self._last_search_progress_time = time.time()
            self._schedule_next_progress_beep()
            self.logln("[search] 🔍 Starting search progress indicator")

    def stop_search_progress_indicator(self):
        """Stop the progress indicator when search completes"""
        if self._search_in_progress:
            self.logln(f"[search] Stopping progress indicator (was at {self._search_progress_count} beeps)")
        self._search_in_progress = False
        if self._search_progress_timer:
            try:
                self.master.after_cancel(self._search_progress_timer)
                self.logln("[search] Progress timer cancelled")
            except:
                pass
            self._search_progress_timer = None

    def set_external_light_color(self, color):
        """Set the color of the external light"""
        try:
            if (self.external_light_win and
                    self.external_light_win.winfo_exists() and
                    self.external_light_win.state() != "withdrawn"):
                self.external_light_win.set_color(color)
        except Exception as e:
            self.logln(f"[light] color change error: {e}")

    def close_status_light(self):
        """Close the status light window"""
        try:
            if self.external_light_win and self.external_light_win.winfo_exists():
                self.external_light_win.destroy()
                self.external_light_win = None
                self.logln("[light] Status light closed")
        except Exception as e:
            self.logln(f"[light] close error: {e}")

    def _play_search_progress_beep(self):
        """Play a progress beep with variation based on search stage"""
        if not self._search_in_progress:
            return

        # === SAFETY CHECK: Stop if too many beeps ===
        if self._search_progress_count > 50:
            self.logln("[search] Safety stop: too many progress beeps")
            self.stop_search_progress_indicator()
            return

        self._search_progress_count += 1

        # Different beep patterns based on search progress
        if self._search_progress_count == 1:
            # First beep - gentle start
            freq = 440  # A4
            duration = 0.08
            vol = 0.1
        elif self._search_progress_count <= 3:
            # Early progress - slightly higher
            freq = 523  # C5
            duration = 0.08
            vol = 0.12
        elif self._search_progress_count <= 6:
            # Middle stage - ascending pattern
            freq = 587  # D5
            duration = 0.09
            vol = 0.14
        else:
            # Extended search - more urgent but not annoying
            freq = 659  # E5
            duration = 0.1
            vol = 0.16

        try:
            fs = 16000
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # Create a pleasant beep with soft attack/decay
            beep = vol * np.sin(2 * np.pi * freq * t)

            # Smooth fade
            fade = int(0.015 * fs)
            if fade > 0:
                beep[:fade] *= np.linspace(0, 1, fade)
                beep[-fade:] *= np.linspace(1, 0, fade)

            # Play non-blocking
            out_dev = self._selected_out_device_index()
            try:
                sd.play(beep, fs, blocking=False, device=out_dev)
            except Exception:
                sd.play(beep, fs, blocking=False)

            self.logln(f"[search] Progress indicator #{self._search_progress_count}")

        except Exception as e:
            self.logln(f"[search] Progress beep error: {e}")

    def _schedule_next_progress_beep(self):
        """Schedule the next progress beep with variable timing"""
        if not self._search_in_progress:
            return

        # Variable timing: more frequent as search takes longer
        if self._search_progress_count <= 3:
            interval = 8000  # 8 seconds for first few beeps
        elif self._search_progress_count <= 6:
            interval = 6000  # 6 seconds for middle stage
        else:
            interval = 5000  # 5 seconds for extended searches

        self._search_progress_timer = self.master.after(interval, self._progress_beep_sequence)

    def _progress_beep_sequence(self):
        """The actual sequence called by the timer"""
        if self._search_in_progress:
            self._play_search_progress_beep()
            self._schedule_next_progress_beep()

    # === END SEARCH PROGRESS METHODS ===

    def _toggle_echo_window(self):
        try:
            if self._echo_win is None or not self._echo_win.winfo_exists():
                self._echo_win = EchoWindow(self.master, self.echo_engine)
            if self._echo_win.state() == "withdrawn":
                self._echo_win.deiconify()
                self._echo_win.lift()
            else:
                self._echo_win.withdraw()
        except Exception as e:
            self.logln(f"[echo] window error: {e}")

    # Add this method to ensure echo state is consistent
    def _sync_echo_state(self):
        """Sync the echo engine state with the UI checkbox"""
        self.echo_engine.enabled = bool(self.echo_enabled_var.get())
        self.logln(f"[echo] state synced: {self.echo_engine.enabled}")

    def _receive_code_output(self, message: str, auto_send: bool = True):
        """Receive output from code sandbox and put in text box"""
        try:
            # Clear existing text and insert the output
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", message)
            self.logln(f"[code] Received output ({len(message)} chars)")

            # Auto-send to AI if enabled
            if auto_send:
                self.logln("[code] Auto-sending to AI...")
                # Small delay to let user see the text
                self.master.after(500, self.send_text)
            else:
                self.logln("[code] Output in text box - edit and send manually")

        except Exception as e:
            self.logln(f"[code] Error receiving output: {e}")

    def _extract_and_auto_run_from_ai(self, ai_response: str):
        """Extract code from AI response and auto-run if enabled"""
        if not self.auto_run_var.get():
            return False

        code = self.extract_python_code(ai_response)
        if code:
            self.store_extracted_code(code)

            # Check if we should auto-open the window
            if self.auto_run_var.get():
                # Auto-open the code window
                self._show_code_window()  # This will trigger auto-run
                self.logln("[code] ✅ Auto-opened and running code from AI response")
                return True

        return False


    def _toggle_image_window(self):
        try:
            if not hasattr(self, "_img_win") or self._img_win is None or not self._img_win.winfo_exists():
                # Use the proper ImageWindow class
                self._img_win = self.ImageWindow(
                    self.master,
                    on_send=self.ask_vision,
                    on_image_change=self._on_new_image
                )

            if self._img_win.state() == "withdrawn":
                self._img_win.deiconify()
                self._img_win.lift()
            else:
                self._img_win.withdraw()
        except Exception as e:
            self.logln(f"[image] window error: {e}")

    # ImageWindow class definition
    class ImageWindow(tk.Toplevel):
        """
        Vision helper:
          - Open image (file dialog)
          - Drag & drop (if tkinterdnd2 is installed)
          - Camera preview + snapshot (if opencv is installed)
          - Send to model (calls parent App.ask_vision)
        """

        def __init__(self, master, on_send, on_image_change=None):
            super().__init__(master)
            self.title("Image / Camera")
            self.geometry("720x560")
            self.protocol("WM_DELETE_WINDOW", self.withdraw)

            self._on_send = on_send  # callback: on_send(image_path, prompt)
            self._on_image_change = on_image_change  # notify app when image changes
            self._img_path = None
            self._img_tk = None
            self._cam = None
            self._cam_timer = None
            self._live_mode = False

            # UI
            wrap = ttk.Frame(self)
            wrap.pack(fill="both", expand=True, padx=8, pady=8)
            top = ttk.Frame(wrap)
            top.pack(fill="x")

            ttk.Button(top, text="Open Image…", command=self.open_image).pack(side="left", padx=(0, 6))
            ttk.Button(top, text="Start Camera", command=self.start_camera).pack(side="left", padx=(0, 6))
            ttk.Button(top, text="Stop Camera", command=self.stop_camera).pack(side="left", padx=(0, 6))
            ttk.Button(top, text="Snapshot", command=self.snapshot).pack(side="left", padx=(0, 6))

            ttk.Label(top, text="Prompt:").pack(side="left", padx=(16, 4))
            self.prompt_var = tk.StringVar(value="Please solve/describe any equations in this image. Use LaTeX.")
            self.prompt_entry = ttk.Entry(top, textvariable=self.prompt_var, width=48)
            self.prompt_entry.pack(side="left", fill="x", expand=True)

            ttk.Button(top, text="Ask model", command=self.send_now).pack(side="left", padx=(6, 0))

            # Create canvas with explicit isolation from parent styling
            canvas_container = tk.Frame(wrap, bg='#f0f0f0')
            canvas_container.pack(fill="both", expand=True, pady=(8, 0))

            self.canvas = tk.Canvas(wrap, bg="#111", highlightthickness=0)
            self.canvas.pack(fill="both", expand=True, pady=(8, 0))
            self.canvas.bind("<Configure>", lambda e: self._redraw())

            # Drag & drop (optional)
            if DND_FILES:
                try:
                    self.drop_target_register(DND_FILES)
                    self.dnd_bind("<<Drop>>", self._on_drop)
                except Exception:
                    pass

        # ---- File ops ----
        def open_image(self):
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                title="Open image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff")]
            )
            if path:
                self.set_image(path)

        def _on_drop(self, event):
            # Windows sends a quoted path; handle multiple too
            paths = self._parse_drop_paths(event.data)
            if paths:
                self.set_image(paths[0])

        @staticmethod
        def _parse_drop_paths(data):
            # minimal parser for common DND formats
            items = []
            cur = ""
            in_quote = False
            for ch in data:
                if ch == '"':
                    in_quote = not in_quote
                elif ch in (" ", "\n") and not in_quote:
                    if cur.strip():
                        items.append(cur.strip('"'))
                    cur = ""
                else:
                    cur += ch
            if cur.strip():
                items.append(cur.strip('"'))
            return items

        def set_image(self, path):
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                messagebox.showerror("Open image", f"Could not open:\n{e}")
                return

            # Stop camera to avoid races
            self.stop_camera()
            self._img_pil = img

            # CRITICAL FIX: ALWAYS save to disk immediately
            os.makedirs("out", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            saved_path = os.path.abspath(os.path.join("out", f"uploaded_{timestamp}.png"))

            try:
                img.save(saved_path, format="PNG")
                self._img_path = saved_path  # Use the saved path, not original
                print(f"[vision] Saved uploaded image to: {saved_path}")  # CHANGED TO print
            except Exception as save_error:
                print(f"[vision] Failed to save image: {save_error}")  # CHANGED TO print
                # Fallback to original path
                self._img_path = os.path.abspath(path)

            self._redraw()

            # tell the app we have a new image file path
            if callable(self._on_image_change):
                try:
                    self._on_image_change(self._img_path)
                except Exception:
                    pass

        def _redraw(self):
            if not hasattr(self, "_img_pil"):
                self.canvas.delete("all")
                return
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            img = self._img_pil
            # fit
            scale = min((cw - 2) / img.width, (ch - 2) / img.height, 1.0)
            disp = img if scale >= 0.999 else img.resize((int(img.width * scale), int(img.height * scale)),
                                                         Image.LANCZOS)
            self._img_tk = ImageTk.PhotoImage(disp)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._img_tk)

        # ---- Camera ----
        def start_camera(self):
            if cv2 is None:
                print("[camera] OpenCV not installed. pip install opencv-python")
                messagebox.showinfo("Camera", "OpenCV not installed.\n\nRun: pip install opencv-python")
                return

            try:
                self._cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
                if not self._cam or not self._cam.isOpened():
                    raise RuntimeError("No camera found")
                # entering live mode: new frames will update _img_pil; don't keep any stale _img_path
                self._live_mode = True
                self._img_path = None
                self._update_cam()
            except Exception as e:
                print(f"[camera] {e}")
                messagebox.showerror("Camera Error", f"Could not start camera:\n{e}")
                self._cam = None
                self._live_mode = False

        def _update_cam(self):
            # Keep pushing frames to _img_pil; DO NOT touch _img_path here (prevents race)
            if self._cam is None or not self._cam.isOpened():
                return
            ok, frame = self._cam.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._img_pil = Image.fromarray(rgb)
                self._redraw()
            self._cam_timer = self.after(33, self._update_cam)

        def stop_camera(self):
            self._live_mode = False
            if self._cam_timer:
                try:
                    self.after_cancel(self._cam_timer)
                except Exception:
                    pass
                self._cam_timer = None
            if self._cam is not None:
                try:
                    self._cam.release()
                except Exception:
                    pass
                self._cam = None

        def snapshot(self):
            """
            Save current image/camera frame to ./out/snapshot_*.png.
            Returns the saved absolute path (string) on success, or None on failure.
            """
            if not hasattr(self, "_img_pil"):
                messagebox.showinfo("Snapshot", "No image/camera frame to save.")
                return None
            os.makedirs("out", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.abspath(os.path.join("out", f"snapshot_{ts}.png"))
            try:
                self._img_pil.save(path)
                self._img_path = path
                # notify app that current image changed to this snapshot
                if callable(self._on_image_change):
                    try:
                        self._on_image_change(self._img_path)
                    except Exception:
                        pass
                return path
            except Exception as e:
                print(f"[snapshot] {e}")
                return None

        # ---- Send to model ----
        def send_now(self):
            if not hasattr(self, "_img_pil"):
                print("[vision] No image/camera frame yet.")
                return

            # SIMPLE FIX: ALWAYS save image to disk
            os.makedirs("out", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.abspath(os.path.join("out", f"temp_image_{timestamp}.png"))

            try:
                # Save as PNG
                self._img_pil.save(path, format="PNG")
                self._img_path = path
            except Exception as e:
                print(f"[vision] could not save image: {e}")
                return

            # Notify app
            if callable(self._on_image_change):
                try:
                    self._on_image_change(path)
                except Exception:
                    pass

            prompt = self.prompt_var.get().strip() or "Describe and solve any equations in this image. Use LaTeX."
            self._on_send(path, prompt)

        def destroy(self):
            self.stop_camera()
            super().destroy()

    def toggle_latex(self):
        try:
            if self.latex_win.state() == "withdrawn":
                self.latex_win.show()
            else:
                self.latex_win.hide()
        except Exception:
            self.latex_win.show()

    def open_avatar(self):
        try:
            kind = self.avatar_kind.get()

            # Close any existing avatar first
            if self.avatar_win and self.avatar_win.winfo_exists():
                try:
                    self.avatar_win.destroy()
                except:
                    pass
                self.avatar_win = None

            if kind == "Rings":
                self.avatar_win = CircleAvatarWindow(self.master)
            elif kind == "Rectangles":
                self.avatar_win = RectAvatarWindow(self.master)
            elif kind == "Rectangles 2":
                self.avatar_win = RectAvatarWindow2(self.master)
            elif kind == "Radial Pulse":
                self.avatar_win = RadialPulseAvatar(self.master)
            elif kind == "FaceRadialAvatar":
                self.avatar_win = FaceRadialAvatar(self.master)
            elif kind == "String Grid":  # ← ADDED STRING GRID HERE
                self.avatar_win = StringGridAvatar(self.master)
            elif kind == "Sphere":  # ← ADDED Sphere
                self.avatar_win = TextureMappedSphere(self.master)
            elif kind == "HAL 9000":  # Supposed to be Hal 9000
                self.avatar_win = Hal9000Avatar(self.master)
            else:
                self.avatar_win = CircleAvatarWindow(self.master)

            if self.avatar_win:
                self.avatar_win.show()

        except Exception as e:
            self.logln(f"[avatar] Error opening avatar: {e}")

    def close_avatar(self):
        try:
            if self.avatar_win and self.avatar_win.winfo_exists():
                self.avatar_win.hide()
        except Exception as e:
            self.logln(f"[avatar] close error: {e}")

    def toggle_avatar(self):
        try:
            if self.avatar_win is None or not self.avatar_win.winfo_exists() or self.avatar_win.state() == "withdrawn":
                self.open_avatar()
            else:
                self.close_avatar()
        except Exception as e:
            self.logln(f"[avatar] toggle error: {e}")
            # Try to create a new one if there's an issue
            try:
                self.avatar_win = None
                self.open_avatar()
            except Exception as e2:
                self.logln(f"[avatar] recovery failed: {e2}")

    def _plot_last_expression(self):
        """Plot the last mathematical expression from context"""
        if not hasattr(self, 'plotter') or self.plotter is None:
            self.logln("[plot] Plotter not available")
            self.play_chime(freq=440, ms=200, vol=0.1)
            return

        if not self.command_router.plotting_enabled:
            self.logln("[plot] Plotting is disabled")
            self.play_chime(freq=440, ms=200, vol=0.1)
            return

        try:
            # Use command router's expression finder
            last_math = self.command_router._find_last_mathematical_expression()

            if last_math:
                self.logln(f"[plot] Plotting: {last_math}")
                plot_window = self.plotter.plot_from_text(last_math)

                if plot_window:
                    self.play_chime(freq=660, ms=150, vol=0.15)
                else:
                    self.logln("[plot] Could not create plot")
                    self.play_chime(freq=440, ms=200, vol=0.1)
            else:
                self.logln("[plot] No mathematical expression found in context")
                self.play_chime(freq=440, ms=300, vol=0.1)

        except Exception as e:
            self.logln(f"[plot] Error: {e}")
            self.play_chime(freq=440, ms=200, vol=0.1)


    def toggle_search_window(self, ensure_visible=False):
        """Toggle web search window - FIXED VERSION
        Args:
            ensure_visible: If True, always show the window (used for voice searches)
        """
        try:
            if self.search_win is None or not self.search_win.winfo_exists():
                self.search_win = WebSearchWindow(self.master, log_fn=self.logln)
                # Bind ALL the search methods to this app instance
                self.search_win.brave_search = self.brave_search
                self.search_win.polite_fetch = self.polite_fetch
                self.search_win.guess_pubdate = self.guess_pubdate
                self.search_win.extract_images = self.extract_images
                self.search_win.extract_readable = self.extract_readable
                self.search_win.summarise_with_qwen = self.summarise_with_qwen
                self.search_win.synthesize_search_results = self.synthesize_search_results
                self.search_win.normalize_query = self.normalize_query
                self.search_win.play_search_results = self.play_search_results
                # ===  CRITICAL CONNECTIONS ===
                self.search_win.main_app = self  # Give access to main app
                self.search_win.preview_latex = self.preview_latex  # Use main app's method
                self.search_win.ensure_latex_window = self.ensure_latex_window  # Use main app's method
                self.search_win.logln = self.logln  # Use main app's logging

            # Always show the window if ensure_visible is True (for voice searches)
            # or if we're toggling and it's currently withdrawn
            if ensure_visible or self.search_win.state() == "withdrawn":
                self.search_win.deiconify()
                self.search_win.lift()
                self.search_win.focus_set()
            else:
                # Only hide if not forced to be visible
                if not ensure_visible:
                    self.search_win.withdraw()

        except Exception as e:
            self.logln(f"[search] window error: {e}")

    def _avatar_set_level_async(self, lvl: int):
        try:
            if self.avatar_win and self.avatar_win.winfo_exists():
                self.avatar_win.set_level(lvl)
        except Exception:
            pass

    def send_text(self):
        if hasattr(self, "text_box"):
            text = self.text_box.get("1.0", "end-1c").strip()
            self.text_box.delete("1.0", "end")
        else:
            text = self.text_entry.get().strip()
            self.text_entry.delete(0, "end")

        if not text:
            return

        threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()

    def preview_latex(self, content: str, context="text"):
        """Preview LaTeX content with append/replace option - ALL in main window"""
        if not self.latex_auto.get():
            return

        def _go():
            try:
                # ALWAYS use the main text window, regardless of context
                latex_win = self.ensure_latex_window("text")
                latex_win.show()

                # Add context indicator for vision responses
                if context == "vision":
                    timestamp = time.strftime("%H:%M:%S")
                    content_with_header = f"🖼️ [Vision Analysis - {timestamp}]\n{content}\n\n"
                else:
                    content_with_header = content

                # === CHECK APPEND MODE ===
                if self.latex_append_mode.get():
                    # APPEND MODE - add to existing content
                    latex_win.append_document(content_with_header)
                    self.logln(f"[latex] 📝 Appended {context} content to main window")
                else:
                    # REPLACE MODE - clear and show new content
                    latex_win.show_document(content_with_header)
                    self.logln(f"[latex] 🔄 Showing {context} content in main window")

                self._current_latex_context = "text"  # Always use text context

            except Exception as e:
                self.logln(f"[latex] preview error ({context}): {e}")

        self.master.after(0, _go)

    def preview_search_results(self, content: str):
        """Special method for search results that preserves the display"""
        self.preview_latex(content, context="search")

    def _latex_theme(self, mode: str):
        """No-op since we only have one window now"""
        pass

    def synthesize_to_wav(self, text, out_wav, role="text"):
        """Synthesize speech using selected TTS engine (SAPI5 or Edge)"""

        # Check mute state
        if self.text_ai_muted:
            self.logln("[mute] 🔇 AI muted - skipping TTS")
            return False

        # Clean text for TTS
        speak_math = self.speak_math_var.get()
        clean_text = clean_for_tts(text, speak_math=speak_math)

        if not clean_text.strip():
            self.logln("[tts] Empty text, skipping")
            return False

        # Route to appropriate engine
        engine = self.tts_engine.get()

        if engine == "edge" and EDGE_TTS_AVAILABLE:
            return self._synthesize_edge(clean_text, out_wav)
        else:
            return self._synthesize_sapi5(clean_text, out_wav, role)

    def _synthesize_edge(self, text, out_wav):
        """Synthesize using Microsoft Edge TTS (neural voices, requires internet)"""
        import time
        import tempfile

        out_dir = os.path.dirname(out_wav) or "out"
        os.makedirs(out_dir, exist_ok=True)

        # Use temp file to avoid file locking issues
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp3', prefix='edge_', dir=out_dir)
        os.close(temp_fd)

        max_retries = 3

        for attempt in range(max_retries):
            try:
                voice = self.edge_voice_var.get()

                # Convert speech rate to Edge format
                # Your rate is -10 to +10, Edge wants "-50%" to "+50%"
                rate = self.speech_rate_var.get()
                rate_str = f"+{rate * 5}%" if rate >= 0 else f"{rate * 5}%"

                # Edge-tts is async, so we need to run it properly
                async def _generate():
                    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                    await communicate.save(temp_path)

                # Run the async function
                asyncio.run(_generate())

                # Check if file was created successfully
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                    # Edge outputs MP3, convert to WAV for consistency with your playback
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_mp3(temp_path)
                        audio.export(out_wav, format="wav")
                        os.remove(temp_path)  # Clean up MP3
                        self.logln(f"[tts] edge: {voice} (rate: {rate_str})")
                        return True
                    except Exception as conv_err:
                        # If pydub fails, try direct rename (some players handle MP3)
                        self.logln(f"[tts] MP3->WAV conversion failed: {conv_err}, using MP3 directly")
                        if os.path.exists(out_wav):
                            os.remove(out_wav)
                        os.rename(temp_path, out_wav.replace('.wav', '.mp3'))
                        # Update out_wav reference - but this won't help caller
                        # Better to just fail and use SAPI5
                        return False
                else:
                    self.logln(f"[tts] edge attempt {attempt + 1} failed - file not created")

            except Exception as e:
                self.logln(f"[tts] edge attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        # Clean up on failure
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

        self.logln("[tts] Edge TTS failed, falling back to SAPI5")
        return self._synthesize_sapi5(text, out_wav, "text")

    def _synthesize_sapi5(self, text, out_wav, role="text"):
        """Synthesize using Windows SAPI5 (local voices, works offline)"""
        import time
        import tempfile

        out_dir = os.path.dirname(out_wav) or "out"
        os.makedirs(out_dir, exist_ok=True)

        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='sapi_', dir=out_dir)
        os.close(temp_fd)

        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                import pyttsx3

                # Get the selected display name
                selected_display = self.sapi_voice_var.get()

                # Use voice mapping if available
                if hasattr(self, 'voice_mapping') and selected_display in self.voice_mapping:
                    voice_id = self.voice_mapping[selected_display]
                else:
                    voice_id = selected_display.split(" | ")[0] if " | " in selected_display else selected_display

                eng = pyttsx3.init()
                eng.setProperty("voice", voice_id)
                eng.setProperty("rate", 150 + self.speech_rate_var.get() * 10)

                eng.save_to_file(text, temp_path)
                eng.runAndWait()
                eng.stop()

                # Wait for file
                wait_time = 0
                while not os.path.exists(temp_path) and wait_time < 5.0:
                    time.sleep(0.1)
                    wait_time += 0.1

                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                    if os.path.exists(out_wav):
                        try:
                            os.remove(out_wav)
                        except:
                            pass

                    try:
                        os.rename(temp_path, out_wav)
                        self.logln(f"[tts] sapi5: {selected_display[:30]}")
                        return True
                    except Exception:
                        import shutil
                        shutil.copy2(temp_path, out_wav)
                        self.logln(f"[tts] sapi5 (copied): {selected_display[:30]}")
                        return True

            except Exception as e:
                self.logln(f"[tts] sapi5 attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        # Clean up
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

        return False

    def set_speech_rate(self, rate: int):
        """Set speech rate to specific value"""
        self.speech_rate_var.set(rate)
        self.update_rate_display()
        self.logln(f"[tts] Speech rate set to: {rate}")

    def update_rate_display(self):
        """Update the rate display label"""
        rate = self.speech_rate_var.get()
        if hasattr(self, 'rate_value_label'):
            if rate < -5:
                self.rate_value_label.config(text="Very Slow")
            elif rate < 0:
                self.rate_value_label.config(text="Slow")
            elif rate == 0:
                self.rate_value_label.config(text="Normal")
            elif rate <= 5:
                self.rate_value_label.config(text="Fast")
            else:
                self.rate_value_label.config(text="Very Fast")

    def _on_dictation_mode_toggle(self):
        """Toggle dictation mode - COMPLETELY stops VAD when active"""
        if self.dictation_mode_var.get():
            # ENTERING DICTATION MODE
            self.logln("[dictation] 📝 Dictation mode ON - VAD DISABLED")
            self.logln("[dictation] 💡 Hold SPACEBAR or click button to record")
            self.dictation_btn.grid()
            self.dictation_btn.config(text="🎤 Click or Hold SPACE")

            # CRITICAL: Set flag to STOP the VAD loop
            self._dictation_mode_active = True

            # Force the VAD loop to stop by breaking its iteration
            if hasattr(self, '_vad_listener'):
                try:
                    # Signal the listener to stop
                    self._vad_listener.stop_listening = True
                    self.logln("[dictation] ✅ VAD listener signaled to stop")
                except:
                    pass

            # Also stop any current listening
            self.set_light("idle")
            self.speaking_flag = False

            # Play confirmation beep
            self.play_chime(freq=660, ms=150, vol=0.15)

        else:
            # EXITING DICTATION MODE
            self.logln("[dictation] 🎙️ Dictation mode OFF - VAD ENABLED")
            self.dictation_btn.grid_remove()
            self._stop_dictation_recording()

            # CRITICAL: Re-enable VAD
            self._dictation_mode_active = False

            # Reset VAD listener flag if it exists
            if hasattr(self, '_vad_listener'):
                try:
                    self._vad_listener.stop_listening = False
                    self.logln("[dictation] ✅ VAD listener re-enabled")
                except:
                    pass

            # Play exit beep
            self.play_chime(freq=440, ms=150, vol=0.15)

            # Reset to listening state if running
            if self.running:
                self.set_light("listening")
                self.logln("[dictation] ✅ VAD mic loop should resume")


    def _on_space_press(self, event=None):
        if not self.dictation_mode_var.get():
            return

        if self._space_held:
            return  # Already recording, ignore key repeat

        # Ignore if focus is on text input widgets
        focused = self.master.focus_get()
        widget_type = type(focused).__name__

        if widget_type in ('Text', 'Entry', 'ScrolledText'):
            return  # Let spacebar type in text boxes

        self._space_held = True
        self.logln(f"[dictation] Spacebar: starting recording")
        self._start_dictation_recording()

        return "break"


        if self._space_held:
            return

        self._space_held = True
        self._start_dictation_recording()

        return "break"

    def _on_space_release(self, event=None):
        if not self._space_held:
            return

        self._space_held = False
        self.logln(f"[dictation] Spacebar: stopping recording")

        if self._dictation_recording:
            self._stop_dictation_recording()

        return "break"



    def _toggle_dictation_recording(self):
        if self._dictation_recording:
            self._stop_dictation_recording()
        else:
            self._start_dictation_recording()

    def _start_dictation_recording(self):
        if self._dictation_recording:
            return

        # CRITICAL: Set a flag to prevent VAD from processing
        self._dictation_recording = True

        # Force light to red (recording) not green (listening)
        self.set_light("speaking")  # Use red light for dictation recording

        # Make sure speaking_flag is True to block VAD
        self.speaking_flag = True

        self._dictation_buffer = []
        self.dictation_btn.config(text="⏹️ Recording... (release SPACE)")
        self.logln("[dictation] 🔴 Recording started - VAD BLOCKED")

        self.play_chime(freq=880, ms=100, vol=0.15)



        def record_continuous():
            import sounddevice as sd
            import numpy as np

            # Use same device as VADListener if available, otherwise default
            device_id = None
            if hasattr(self, 'vad') and hasattr(self.vad, 'device'):
                device_id = self.vad.device
                self.logln(f"[dictation] Using VAD device: {device_id}")

            sample_rate = 16000
            chunk_duration = 0.1
            chunk_size = int(sample_rate * chunk_duration)

            try:
                with sd.InputStream(device=device_id, channels=1,
                                    samplerate=sample_rate, dtype='float32') as stream:
                    self.logln(f"[dictation] 🎙️ Mic stream opened")

                    while self._dictation_recording:
                        audio_chunk, overflowed = stream.read(chunk_size)
                        if overflowed:
                            self.logln("[dictation] ⚠️ Audio buffer overflow")
                        self._dictation_buffer.append(audio_chunk.copy())

            except Exception as e:
                self.logln(f"[dictation] ❌ Recording error: {e}")
                self._dictation_recording = False

        self._dictation_thread = threading.Thread(target=record_continuous, daemon=True)
        self._dictation_thread.start()

    def _stop_dictation_recording(self):
        if not self._dictation_recording:
            return

        self._dictation_recording = False
        self._space_held = False  # Reset spacebar state
        self.dictation_btn.config(text="🎤 Click or Hold SPACE")

        # IMPORTANT: Different behavior based on dictation mode state
        if self.dictation_mode_var.get():
            # We're STAYING in dictation mode, so don't enable VAD
            self.set_light("processing")  # Yellow/orange for processing
            self.logln("[dictation] ⏹️ Recording stopped, transcribing...")

            # Keep speaking_flag = True to block VAD while processing
            self.speaking_flag = True
        else:
            # We're LEAVING dictation mode, re-enable VAD
            self.set_light("idle")
            self.speaking_flag = False
            self.logln("[dictation] Stopped recording and exiting dictation mode")

        self.play_chime(freq=440, ms=100, vol=0.15)

        if hasattr(self, '_dictation_thread') and self._dictation_thread.is_alive():
            self._dictation_thread.join(timeout=1.0)

        def process_dictation():
            try:
                import numpy as np

                if not self._dictation_buffer:
                    self.logln("[dictation] ❌ No audio recorded")
                    # Reset state based on dictation mode
                    if self.dictation_mode_var.get():
                        self.set_light("idle")  # Stay yellow in dictation mode
                    else:
                        self.set_light("idle")  # Back to yellow if VAD is off
                        self.set_light("listening")  # Green if VAD should be on
                    return

                audio_data = np.concatenate(self._dictation_buffer, axis=0)
                audio_data = audio_data.flatten()

                duration = len(audio_data) / 16000
                self.logln(f"[dictation] 📊 Recorded {duration:.1f}s of audio")

                if duration < 0.5:
                    self.logln("[dictation] ⚠️ Recording too short (< 0.5s)")
                    # Reset based on mode
                    if self.dictation_mode_var.get():
                        self.set_light("idle")
                        self.master.after(0, lambda: self.dictation_btn.config(text="🎤 Click or Hold SPACE"))
                    else:
                        self.set_light("listening")
                    return

                self.logln("[dictation] 🔄 Transcribing...")
                text = self.asr.transcribe(audio_data, 16000)

                if not text or not text.strip():
                    self.logln("[dictation] ❌ No speech detected")
                    # Reset based on mode
                    if self.dictation_mode_var.get():
                        self.set_light("idle")
                    else:
                        self.set_light("listening")
                    return

                self.logln(f"[dictation] ✅ Transcribed ({len(text)} chars): {text[:100]}...")

                # Show result in UI
                self.master.after(0, lambda: self._show_dictation_result(text))

                # FINAL STATE RESET - this is critical
                self.master.after(100, self._finalize_dictation_state)

            except Exception as e:
                self.logln(f"[dictation] ❌ Error: {e}")
                import traceback
                self.logln(traceback.format_exc())
                self._finalize_dictation_state()
            finally:
                self._dictation_buffer = []

        threading.Thread(target=process_dictation, daemon=True).start()

    def _finalize_dictation_state(self):
        """Final cleanup after dictation processing"""
        if self.dictation_mode_var.get():
            # Still in dictation mode
            self.set_light("idle")  # Yellow - dictation ready
            self.speaking_flag = False  # Allow spacebar to work again
            self.logln("[dictation] ✅ Ready for next dictation")
        else:
            # Exited dictation mode, VAD should be active
            self.set_light("listening")  # Green - VAD listening
            self.speaking_flag = False
            self.master.focus_force()
            self.logln("[dictation] ✅ Back to VAD mode")


    def _show_dictation_result(self, text):
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)

        self.play_chime(freq=660, ms=150, vol=0.15)

        from tkinter import messagebox

        preview = text[:300] + ('...' if len(text) > 300 else '')

        result = messagebox.askyesnocancel(
            "Dictation Complete",
            f"Transcribed ({len(text)} chars):\n\n{preview}\n\n"
            "• YES = Send to AI\n"
            "• NO = Keep in text box (edit first)\n"
            "• CANCEL = Discard"
        )

        if result is True:
            self.text_box.delete("1.0", "end")
            threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()
        elif result is False:
            self.logln("[dictation] 📝 Text kept in text box for editing")
        else:
            self.text_box.delete("1.0", "end")
            self.logln("[dictation] 🗑️ Transcription discarded")

    # === Device Methods ===
    def _device_hostapi_name(self, index):
        try:
            if index is None:
                return None
            info = sd.query_devices(index)
            hostapi_idx = info.get('hostapi', None)
            if hostapi_idx is None:
                return None
            hai = sd.query_hostapis(hostapi_idx)
            return hai.get('name')
        except Exception:
            return None

    def _list_output_devices(self):
        try:
            info = sd.query_devices()
            outs = []
            for i, d in enumerate(info):
                if d.get('max_output_channels', 0) > 0:
                    name = d.get('name', f'Device {i}')
                    outs.append(f"{i}: {name}")
            return outs if outs else ["(default output)"]
        except Exception as e:
            self.logln(f"[audio] output device query failed: {e}")
            return ["(default output)"]

    def _selected_out_device_index(self):
        try:
            choice = self.out_combo.get()
            return int(choice.split(":")[0]) if ":" in choice else None
        except Exception:
            return None

    # === Vision UI Helpers ===
    def _ensure_image_window(self):
        """Create the ImageWindow if needed (but don't show it)."""
        try:
            if not hasattr(self, "_img_win") or self._img_win is None or not self._img_win.winfo_exists():
                self._img_win = self.ImageWindow(
                    self.master,
                    on_send=self.ask_vision,
                    on_image_change=self._on_new_image
                )
        except Exception as e:
            self.logln(f"[vision] could not create image window: {e}")
            self._img_win = None

    def start_camera_ui(self):
        """Voice/typed: 'start camera' -> open window and start streaming."""
        try:
            self._ensure_image_window()
            if self._img_win is None:
                self.logln("[vision] camera UI unavailable")
                return
            self._img_win.deiconify()
            self._img_win.lift()
            self._img_win.start_camera()
            self.logln("[vision] camera started")
        except Exception as e:
            self.logln(f"[vision] start camera error: {e}")

    def stop_camera_ui(self):
        """Voice/typed: 'stop camera' -> stop streaming."""
        try:
            if hasattr(self, "_img_win") and self._img_win and self._img_win.winfo_exists():
                self._img_win.stop_camera()
                self.logln("[vision] camera stopped")
            else:
                self.logln("[vision] camera window not open")
        except Exception as e:
            self.logln(f"[vision] stop camera error: {e}")

    def take_picture_ui(self):
        """Voice/typed: 'take a picture' -> snapshot current frame."""
        try:
            self._ensure_image_window()
            if self._img_win is None:
                self.logln("[vision] camera UI unavailable")
                return
            saved = self._img_win.snapshot()
            if saved:
                self.logln(f"[vision] snapshot ready: {saved}")
            else:
                self.logln("[vision] snapshot failed")
        except Exception as e:
            self.logln(f"[vision] take picture error: {e}")

    def explain_last_image_ui(self, prompt_text=None):
        try:
            img_path = None
            if hasattr(self, "_img_win") and self._img_win and self._img_win.winfo_exists():
                img_path = getattr(self._img_win, "_img_path", None)
            if not img_path:
                img_path = self._last_image_path

            if not img_path:
                self.logln("[vision] no image available. Say 'take a picture' or open an image first.")
                return

            prompt = (prompt_text or "Describe what is in the image in clear detail.").strip()
            self.ask_vision(img_path, prompt)
        except Exception as e:
            self.logln(f"[vision] explain image error: {e}")

    def _ollama_available_models(self):
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json().get("models", [])
            return {m.get("name", "") for m in tags}
        except Exception as e:
            self.logln(f"[ollama] tag query failed: {e}")
            return set()

    def _load_personalities(self):
        """Load personality profiles from the Personalities folder"""
        personalities_dir = "Personalities"
        self.personalities = {
            "Default": {
                "name": "Default",
                "description": "Standard assistant behavior",
                "is_personality": True
            }
        }

        try:
            # Get the current working directory
            current_dir = os.getcwd()
            self.logln(f"[personality] Current directory: {current_dir}")

            # Check if Personalities directory exists
            personalities_path = os.path.join(current_dir, personalities_dir)
            self.logln(f"[personality] Looking for: {personalities_path}")

            if not os.path.exists(personalities_path):
                os.makedirs(personalities_path)
                self.logln(f"[personality] ❗ Created directory: {personalities_path}")
                return

            # List all files in the directory
            all_files = os.listdir(personalities_path)
            self.logln(f"[personality] Files in directory: {all_files}")

            # Load all JSON files from the directory
            personality_files = [f for f in all_files if f.lower().endswith('.json')]
            self.logln(f"[personality] JSON files found: {personality_files}")

            if not personality_files:
                self.logln("[personality] No personality files found, using Default only")
                return

            for filename in personality_files:
                filepath = os.path.join(personalities_path, filename)
                self.logln(f"[personality] Trying to load: {filepath}")

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        personality_data = json.load(f)

                    # Use the name from the file, or filename as fallback
                    personality_name = personality_data.get('name', os.path.splitext(filename)[0])
                    self.personalities[personality_name] = personality_data
                    self.logln(f"[personality] ✅ SUCCESS: Loaded {personality_name}")

                except Exception as e:
                    self.logln(f"[personality] ❌ ERROR loading {filename}: {e}")

            # Update the combo box values
            personality_names = list(self.personalities.keys())
            self.logln(f"[personality] Final personalities: {personality_names}")
            self.personality_combo['values'] = personality_names

            self.logln(f"[personality] Loaded {len(self.personalities)} personalities")

        except Exception as e:
            self.logln(f"[personality] ❌ CRITICAL ERROR: {e}")

            # Finished Update personalities

    def _on_personality_change(self, event=None):
        """Handle personality dropdown change"""
        selected_name = self.personality_var.get()

        if not selected_name or selected_name not in self.personalities:
            return

        # Get the PREVIOUS personality before applying new one
        previous_personality = getattr(self, '_current_personality', 'Default')

        self.logln(f"[personality] Switching from '{previous_personality}' to '{selected_name}'")

        # Apply the new personality (pass previous for comparison)
        self.apply_personality(selected_name, previous_personality)

        # Update tracking AFTER applying
        self._current_personality = selected_name

    def _enhance_system_prompt(self, personality_prompt):
        """Combine personality prompt with original system prompt"""
        original_prompt = self.cfg.get("system_prompt") or self.cfg.get("qwen_system_prompt", "")

        if original_prompt and personality_prompt:
            combined_prompt = f"""{personality_prompt}

ADDITIONAL CONTEXT:
{original_prompt}
"""
            return combined_prompt
        elif personality_prompt:
            return personality_prompt
        return original_prompt or ""

    def apply_personality(self, personality_name, previous_personality=None):
        """Apply personality settings - dual TTS engine support"""
        if personality_name not in self.personalities:
            self.logln(f"[personality] ❌ Personality '{personality_name}' not found")
            return

        personality = self.personalities[personality_name]

        try:
            # Use passed previous_personality, or try to detect
            if previous_personality is None:
                previous_personality = getattr(self, '_current_personality', 'Default')

            # Clear history if actually switching personalities
            if previous_personality != personality_name:
                self._clear_chat_context_for_personality_switch()
                self.logln(
                    f"[personality] ✅ Chat history cleared switching from {previous_personality} to {personality_name}")
                self.play_chime(freq=880, ms=150, vol=0.15)
                time.sleep(0.05)
                self.play_chime(freq=660, ms=200, vol=0.12)
            else:
                self.logln(f"[personality] Already in {personality_name} mode, no switch needed")

            # Handle Default personality
            if personality_name == "Default":
                self._reset_to_default_personality()
                self.personality_status.config(text="✓ Default", foreground="green")
                self.speak_search_status("Returning to default mode")
                self.logln("[personality] ✅ Restored default settings with fresh context")
                return

            self.logln(f"[personality] Applying {personality_name}...")

            # === VOICE SETTINGS ===
            voice_settings = personality.get('voice', {})
            engine = voice_settings.get('engine', 'sapi5').lower()

            if engine == 'edge' and EDGE_TTS_AVAILABLE:
                self.tts_engine.set('edge')
                self._on_engine_change()
                edge_voice = voice_settings.get('edge_voice')
                if edge_voice:
                    available_edge_voices = list(self.edge_voice_combo['values'])
                    if edge_voice in available_edge_voices:
                        self.edge_voice_var.set(edge_voice)
                        self.logln(f"[personality] 🌐 Edge voice: {edge_voice}")
                    else:
                        matching = [v for v in available_edge_voices if edge_voice.lower() in v.lower()]
                        if matching:
                            self.edge_voice_var.set(matching[0])
                            self.logln(f"[personality] 🌐 Edge voice (matched): {matching[0]}")
                        else:
                            self.logln(f"[personality] ⚠️ Edge voice '{edge_voice}' not found")
                else:
                    self.logln(f"[personality] 🌐 Using Edge with current voice: {self.edge_voice_var.get()}")

            elif engine == 'edge' and not EDGE_TTS_AVAILABLE:
                self.logln(f"[personality] ⚠️ Edge TTS not installed, falling back to SAPI5")
                self.tts_engine.set('sapi5')
                self._on_engine_change()
                target_voice = voice_settings.get('sapi_voice')
                if target_voice:
                    available_voices = list(self.voice_mapping.keys())
                    matching_voices = [v for v in available_voices if target_voice.lower() in v.lower()]
                    if matching_voices:
                        self.sapi_voice_var.set(matching_voices[0])
                        self.logln(f"[personality] 🗣️ Fallback SAPI voice: {matching_voices[0]}")
            else:
                self.tts_engine.set('sapi5')
                self._on_engine_change()
                target_voice = voice_settings.get('sapi_voice')
                if target_voice:
                    available_voices = list(self.voice_mapping.keys())
                    matching_voices = [v for v in available_voices if target_voice.lower() in v.lower()]
                    if matching_voices:
                        self.sapi_voice_var.set(matching_voices[0])
                        self.logln(f"[personality] 🗣️ SAPI voice: {matching_voices[0]}")
                    else:
                        self.logln(f"[personality] ⚠️ SAPI voice '{target_voice}' not found")
                else:
                    self.logln(f"[personality] 🗣️ No voice specified, keeping current")


            # Speech rate
            speech_rate = voice_settings.get('speech_rate')
            if speech_rate is not None:
                self.speech_rate_var.set(speech_rate)
                self.update_rate_display()
                self.logln(f"[personality] ⚡ Speech rate: {speech_rate}")

            # === TEMPERATURE SETTING ===
            temperature = personality.get('temperature')
            if temperature is not None:
                try:
                    temp_value = float(temperature)
                    # Clamp to reasonable range (0.0 to 2.0)
                    temp_value = max(0.0, min(2.0, temp_value))
                    if hasattr(self.qwen, 'temperature'):
                        self.qwen.temperature = temp_value
                        self.logln(f"[personality] 🌡️ Temperature: {temp_value}")
                    else:
                        self.logln(f"[personality] ⚠️ QwenLLM doesn't have temperature attribute")
                except (ValueError, TypeError) as e:
                    self.logln(f"[personality] ⚠️ Invalid temperature value: {temperature}")
            else:
                # Reset to default temperature from config if not specified in personality
                default_temp = self.cfg.get("qwen_temperature", 0.7)
                if hasattr(self.qwen, 'temperature'):
                    self.qwen.temperature = default_temp
                    self.logln(f"[personality] 🌡️ Temperature: {default_temp} (default)")


            # === SYSTEM PROMPT - AGGRESSIVE PERSONALITY RESET ===
            system_prompt = personality.get('system_prompt')
            if system_prompt:
                from datetime import datetime
                current_datetime = datetime.now()
                current_date = current_datetime.strftime("%B %d, %Y")
                current_time = current_datetime.strftime("%I:%M %p")
                current_day = current_datetime.strftime("%A")

                enhanced_prompt = f"""### CRITICAL PERSONALITY RESET ###
STOP. FORGET EVERYTHING. You are NO LONGER any previous character.
You are NO LONGER {previous_personality}.
COMPLETELY ERASE all previous character traits, accents, and behaviors.

### NEW IDENTITY ###
You are NOW and ONLY: {personality_name}
This is a FRESH conversation. No history exists.

### CURRENT DATE ###
{current_day}, {current_date} at {current_time}

### YOUR NEW CHARACTER ###
{system_prompt}

### REMEMBER ###
You are {personality_name}. Stay in this character for ALL responses.
Do NOT reference or blend with any previous personalities.
"""

                self.qwen.system_prompt = enhanced_prompt
                self.qwen.history = []  # Clear again to be sure
                self.logln(f"[personality] ✅ System prompt updated with aggressive reset")
            else:
                self._reset_to_default_personality()

            # Update status
            engine_icon = "🌐" if self.tts_engine.get() == "edge" else "🗣️"
            self.personality_status.config(text=f"✓ {personality_name} {engine_icon}", foreground="blue")
            self.logln(f"[personality] ✅ Applied: {personality_name}")
            self.logln(f"[personality] 🔊 Engine: {self.tts_engine.get().upper()}")
            self.speak_search_status(f"Activating {personality_name} personality")

        except Exception as e:
            self.logln(f"[personality] ❌ Error: {e}")
            import traceback
            self.logln(f"[personality] {traceback.format_exc()}")
            self.personality_status.config(text="❌ Error", foreground="red")

    def extract_python_code(self, text: str):
        """Extract Python code - Filter out output text and pip install"""
        import re

        pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return None

        clean_blocks = []
        for block in matches:
            lines = block.split('\n')
            python_lines = []

            for line in lines:
                stripped = line.strip()

                # === FILTER OUT OUTPUT/RESULT TEXT ===
                # Lines that start with text (not code) followed by colon and number
                if (re.match(r'^[a-zA-Z].*decimal places:', stripped) or  # "to 100 decimal places:"
                        re.match(r'^[a-zA-Z].*places:', stripped) or  # "π to 100 places:"
                        re.match(r'^[a-zA-Z].*result:', stripped) or  # "The result is:"
                        re.match(r'^[a-zA-Z].*output:', stripped) or  # "The output:"
                        'π to' in stripped or  # "π to 100 decimal places"
                        'Pi to' in stripped):  # "Pi to 100 decimal places"
                    continue  # Skip output text

                # === CRITICAL: REMOVE pip install commands ===
                if (stripped.startswith('pip install') or
                        stripped.startswith('pip3 install') or
                        stripped.startswith('!pip install') or
                        stripped.startswith('conda install') or
                        'pip install mpmath' in stripped):
                    continue

                # Also remove comments about pip installation
                if stripped.startswith('#') and 'pip install' in stripped.lower():
                    continue

                # KEEP: Valid Python code
                if (not stripped or
                        stripped.startswith('#') or
                        stripped.startswith('def ') or
                        stripped.startswith('class ') or
                        stripped.startswith('import ') or
                        stripped.startswith('from ') or
                        '=' in line or
                        '(' in stripped or
                        stripped.endswith(':') or
                        stripped.startswith('return ') or
                        stripped.startswith('print(') or
                        stripped.startswith('for ') or
                        stripped.startswith('if ') or
                        stripped.startswith('while ') or
                        stripped.startswith('try:') or
                        stripped.startswith('except ') or
                        stripped.startswith('raise ') or
                        re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[:=\(]', stripped)):
                    python_lines.append(line)

                # REMOVE: Output lines with results
                elif (re.match(r'^[A-Z][a-z].*: [-+]?[\d\.]+', stripped) or
                      re.match(r'^[A-Z][a-z].* is [-+]?[\d\.]+', stripped) or
                      'approximate square root' in stripped or
                      'Error check:' in stripped or
                      'The output will be:' in stripped):
                    continue

                # REMOVE: Standalone numbers (results)
                elif re.match(r'^[-+]?[\d\.]+(?:e[-+]?\d+)?$', stripped):
                    continue

                # Default: Keep the line
                else:
                    python_lines.append(line)

            if python_lines:
                clean_blocks.append('\n'.join(python_lines))

        if not clean_blocks:
            return None

        return "\n\n".join(clean_blocks).strip()





    def store_extracted_code(self, code: str):
        """Store extracted code and update UI"""
        if code and len(code) > 10:  # Only store substantial code
            self._last_extracted_code = code
            self.logln(f"[code] Stored {len(code)} chars of Python code")

            # Update the Run Code button to show we have code
            self.code_btn.config(text="💻 Run Code ✓", style='Success.TButton')

            # Optional: Play a subtle notification sound
            self.play_chime(freq=880, ms=100, vol=0.1)
            return True
        return False



    def _clear_chat_context_for_personality_switch(self):
        """Clear all chat context when switching personalities."""
        self.logln(f"[DEBUG] === CLEARING CHAT CONTEXT ===")

        if hasattr(self.qwen, 'clear_history'):
            try:
                self.qwen.clear_history()
                self.logln("[context] Qwen history cleared")
            except Exception as e:
                self.logln(f"[context] Error: {e}")

        if hasattr(self.qwen, 'history'):
            history_length = len(self.qwen.history)
            self.qwen.history = []
            self.logln(f"[context] History cleared ({history_length} entries)")

        for attr_name in ['conversation', '_conversation', 'chat_history', '_chat_history']:
            if hasattr(self.qwen, attr_name):
                setattr(self.qwen, attr_name, [])
                self.logln(f"[context] Cleared {attr_name}")

        self._last_vision_reply = ""
        self._last_was_vision = False
        self._last_image_path = None

        try:
            if self.latex_win_text and self.latex_win_text.winfo_exists():
                self.latex_win_text.clear()
                self.logln("[context] LaTeX window cleared")
        except Exception as e:
            self.logln(f"[context] Error clearing LaTeX: {e}")

        if hasattr(self, '_pending_graph_after_response'):
            self._pending_graph_after_response = False
            self._last_user_text = ""

        if hasattr(self, '_last_ai_response'):
            self._last_ai_response = ""

        self.logln("[context] ✅ All context cleared")

        def _clear_chat_context_for_personality_switch(self):
            """
            Clear all chat context when switching personalities.
            This ensures fresh start for each personality.
            """
            self.logln(f"[DEBUG] === CLEARING CHAT CONTEXT ===")

            # Method 1: Use Qwen's clear_history if available
            if hasattr(self.qwen, 'clear_history'):
                try:
                    self.qwen.clear_history()
                    self.logln("[context] Qwen history cleared via clear_history()")
                except Exception as e:
                    self.logln(f"[context] Error with clear_history(): {e}")

            # Method 2: Directly clear history list
            if hasattr(self.qwen, 'history'):
                history_length = len(self.qwen.history)
                self.qwen.history = []
                self.logln(f"[context] History list cleared ({history_length} entries removed)")

            # Method 3: Also clear any conversation buffers
            if hasattr(self.qwen, 'conversation') or hasattr(self.qwen, '_conversation'):
                # Some implementations use different attribute names
                for attr_name in ['conversation', '_conversation', 'chat_history', '_chat_history']:
                    if hasattr(self.qwen, attr_name):
                        setattr(self.qwen, attr_name, [])
                        self.logln(f"[context] Cleared {attr_name}")

            # Clear vision context
            self._last_vision_reply = ""
            self._last_was_vision = False
            self._last_image_path = None

            # Clear LaTeX window content
            try:
                if self.latex_win_text and self.latex_win_text.winfo_exists():
                    self.latex_win_text.clear()
                    self.logln("[context] LaTeX window cleared")
            except Exception as e:
                self.logln(f"[context] Error clearing LaTeX: {e}")

            # Clear any pending graph flags
            if hasattr(self, '_pending_graph_after_response'):
                self._pending_graph_after_response = False
                self._last_user_text = ""
                self.logln("[context] Pending graph flags cleared")

            # Clear last AI response
            if hasattr(self, '_last_ai_response'):
                self._last_ai_response = ""

            self.logln("[context] ✅ All chat context cleared for personality switch")

    def _reset_to_default_personality(self):
        """Reset to settings from main config.json WITH real-time date"""

        # === RESET TTS ENGINE TO CONFIG DEFAULT ===
        config_engine = self.cfg.get("tts_engine", "sapi5").lower()
        if config_engine == "edge" and EDGE_TTS_AVAILABLE:
            self.tts_engine.set("edge")
        else:
            self.tts_engine.set("sapi5")
        self._on_engine_change()  # Update UI state
        self.logln(f"[personality] 🔊 Reset engine to: {self.tts_engine.get()}")

        # === RESET EDGE VOICE TO CONFIG DEFAULT ===
        if EDGE_TTS_AVAILABLE:
            config_edge_voice = self.cfg.get("edge_voice", "en-US-AriaNeural")
            available_edge_voices = list(self.edge_voice_combo['values'])
            if config_edge_voice in available_edge_voices:
                self.edge_voice_var.set(config_edge_voice)
                self.logln(f"[personality] 🌐 Reset Edge voice to: {config_edge_voice}")

        # === RESET SAPI VOICE TO CONFIG DEFAULT ===
        config_voice = self.cfg.get("text_ai_voice")
        if config_voice:
            available_voices = list(self.voice_mapping.keys())
            matching_voices = [v for v in available_voices if config_voice in v]
            if matching_voices:
                self.sapi_voice_var.set(matching_voices[0])
                self.logln(f"[personality] 🗣️ Reset SAPI voice to: {matching_voices[0]}")

        # Reset speech rate
        default_rate = self.cfg.get("text_ai_speech_rate", 0)
        self.speech_rate_var.set(default_rate)
        self.update_rate_display()

        # Get current date/time for Default personality too!
        from datetime import datetime
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%B %d, %Y")
        current_time = current_datetime.strftime("%I:%M %p")
        current_day = current_datetime.strftime("%A")

        # Get the ORIGINAL system prompt from config
        original_prompt = self.cfg.get("system_prompt") or self.cfg.get("qwen_system_prompt", "")

        # Create ENHANCED Default prompt WITH date
        enhanced_default_prompt = f"""{original_prompt}

    === REAL-TIME DATE & TIME CONTEXT ===

    CURRENT DATE: {current_day}, {current_date} at {current_time}
    CRITICAL: FORGET PREVIOUS PERSONALITIES AND ASSUME THIS ONE HERE
    CRITICAL DATE/TIME RULES:
    ✓ When asked about current date/time, ALWAYS use: "{current_day}, {current_date} at {current_time}"
    ✓ Trust this EXACT date over your training data
    ✓ Do NOT calculate days of the week yourself
    ✓ Only mention date/time when specifically relevant

    === UNIFIED AI CAPABILITIES ===
    YOU ARE A SINGLE AI WITH TEXT, VISION, AND MATHEMATICAL CAPABILITIES.

    ABILITIES:
    1. Process text-only questions
    2. Analyze images when provided (base64-encoded)
    3. Answer questions about previously described images
    4. Provide mathematical solutions in plottable formats
    5. Support voice-driven mathematical workflows

    === MATHEMATICAL OUTPUT STANDARDS ===
    ✓ ALWAYS provide clean, standalone mathematical expressions
    ✓ Use LaTeX notation: $f(x) = 2x$ or $$\\frac{{x^2}}{{4}}$$
    ✓ For derivatives/integrals, show ONLY the final result when asked
    ✓ Use \\boxed{{}} for final answers: \\boxed{{f'(x) = 2x}}

    === PLOTTING SYSTEM REQUIREMENTS ===
    1. The plotting system works on EXPRESSIONS ONLY (not equations)
    2. Remove "y =" or "f(x) =" from plottable expressions
    3. Variables supported: x, t, theta (θ), n, s
    4. All variables will be auto-converted to 'x' for plotting
    5. Integration constants (+C) are automatically removed

    === VISION CONTEXT MANAGEMENT ===
    When you analyze an image, you should:
    1. Provide detailed descriptions
    2. Include mathematical equations if visible
    3. Remember key elements for follow-up questions
    4. The system will automatically track vision context

    === FORMATTING & STYLE ===
    PROHIBITED:
    ❌ NO smileys or emoticons
    ❌ NO excessive ** bold markers ** in text
    ❌ NO ** separators between sections

    PREFERRED:
    ✓ Clean, professional mathematical notation
    ✓ Natural conversational tone
    ✓ Clear section breaks when needed
    ✓ Minimal formatting unless essential

    === CONTEXT-AWARE PLOTTING ===
    When users say "graph that" or "plot it":
    1. The system searches your LAST response for mathematical expressions
    2. It prioritizes: \\boxed{{}} > LaTeX delimiters > equations
    3. It extracts the FINAL answer automatically
    4. You DO NOT need to repeat the expression

    Respond naturally and helpfully while following these guidelines.
    """


        # Reset temperature to config default
        default_temp = self.cfg.get("qwen_temperature", 0.7)
        if hasattr(self.qwen, 'temperature'):
            self.qwen.temperature = default_temp
            self.logln(f"[personality] 🌡️ Temperature reset to: {default_temp}")

        self.qwen.system_prompt = enhanced_default_prompt
        self.logln(f"[personality] ✅ Default personality restored WITH date context")
        self.logln(f"[personality] 📅 Current date in Default: {current_date} at {current_time}")

    # === SEARCH METHODS ===

    def brave_search(self, query: str, count: int = 6):
        brave_key = os.getenv("BRAVE_KEY")
        if not brave_key:
            raise RuntimeError("No BRAVE_KEY found in environment")
        # === Logs we are searching the Internet ===
        self.logln(f"[SEARCH] 🚀 Calling Brave API: '{query}'")

        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": brave_key, "User-Agent": "LocalAI-ResearchBot/1.0"}
        params = {"q": query, "count": count}

        with httpx.Client(timeout=25.0, headers=headers) as client:
            r = client.get(endpoint, params=params)
            r.raise_for_status()
            data = r.json()

        out = []
        for w in (data.get("web", {}) or {}).get("results", []):
            out.append(
                Item(title=w.get("title", "No title"), url=w.get("url", ""), snippet=w.get("description", "")))
            # === check what its searching  ===
            self.logln(f"[BRAVE API] ✅ Found {len(out)} results for '{query}'")

        return out

    def polite_fetch(self, url: str):
        headers = {"User-Agent": "LocalAI-ResearchBot/1.0"}
        try:
            with httpx.Client(timeout=25.0, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.text
        except Exception:
            return None

    def extract_readable(self, html: str, url: str = None):
        text = trafilatura.extract(html, url=url, include_links=False, include_formatting=False)
        return text or ""

    def guess_pubdate(self, html: str):
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return None

        metas = [
            ("property", "article:published_time"), ("property", "og:published_time"),
            ("property", "og:updated_time"), ("name", "pubdate"), ("name", "publication_date"),
            ("name", "date"), ("name", "dc.date"), ("name", "dc.date.issued"),
            ("name", "sailthru.date"), ("itemprop", "datePublished"), ("itemprop", "dateModified"),
        ]

        for key, val in metas:
            tag = soup.find("meta", attrs={key: val})
            if tag and tag.get("content"):
                return tag["content"]

        t = soup.find("time")
        if t and (t.get("datetime") or (t.text and t.text.strip())):
            return t.get("datetime") or t.text.strip()
        return None

    def summarise_for_ai_search(self, text: str, url: str, pubdate: str):
        """Enhanced summarization that preserves practical information"""
        text = text[:18000]

        # Enhanced date context
        if pubdate:
            date_context = f"PUBLICATION DATE: {pubdate}\n"
        else:
            import re
            date_matches = re.findall(
                r'\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 20\d{2})\b',
                text[:3000])
            date_context = f"MENTIONED DATES: {', '.join(date_matches[:3])}\n" if date_matches else ""

        # DETECT QUERY TYPE AND ADAPT SUMMARIZATION
        query_lower = getattr(self, '_last_search_query', '').lower()

        # Flight/travel related queries
        if any(keyword in query_lower for keyword in ['flight', 'fly', 'airline', 'airport', 'travel to']):
            summary_prompt = (
                "Extract COMPLETE flight information with these details:\n\n"
                "## FLIGHT INFORMATION\n"
                "- Airline names and flight numbers\n"
                "- Departure and arrival airports (with codes if available)\n"
                "- Departure and arrival times/dates\n"
                "- Flight duration\n"
                "- Prices and fare classes\n"
                "- Stopovers/layovers\n"
                "- Booking links or airline websites\n\n"
                "## TRAVEL DETAILS\n"
                "- Airport locations and terminals\n"
                "- Booking requirements\n"
                "- Baggage information\n"
                "- Recent deals or promotions\n\n"
                "Include ALL specific numbers, times, prices, and codes. Be very detailed about schedules and availability.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        # Business/location queries
        elif any(keyword in query_lower for keyword in
                 ['address', 'location', 'where is', 'hours', 'contact', 'phone', 'email']):
            summary_prompt = (
                "EXTRACT ONLY INFORMATION EXPLICITLY STATED IN THE TEXT. NEVER CREATE PLACEHOLDERS OR INVENT INFORMATION.\n\n"
                "CRITICAL RULES:\n"
                "1. ONLY include information that appears VERBATIM in the source text\n"
                "2. NEVER use brackets [ ], parentheses ( ), or placeholder text\n"
                "3. If a website is mentioned, copy the EXACT URL\n"
                "4. If information is missing, OMIT that line entirely\n"
                "5. Do NOT create template responses\n\n"
                "EXTRACTED INFORMATION (ONLY IF FOUND):\n"
                "- Business Name: [copy exact name if found]\n"
                "- Address: [copy exact address if found]\n"
                "- Phone: [copy exact phone number if found]\n"
                "- Email: [copy exact email if found]\n"
                "- Website: [copy exact URL if found]\n"
                "- Hours: [copy exact hours if found]\n\n"
                "EXAMPLES - WRONG:\n"
                "❌ Address: [Address may vary]\n"
                "❌ Phone: [Phone number may vary]  \n"
                "❌ Website: [Website Link]\n"
                "❌ Website: [Website URL if available]\n\n"
                "EXAMPLES - CORRECT:\n"
                "✅ Address: 456 Northshore Road, Unit 2, Glenfield 0678\n"
                "✅ Phone: +64 9 483 5555\n"
                "✅ Website: https://www.serenityspa.co.nz\n"
                "✅ Website: www.serenityspa.com\n"
                "✅ (omit Website line if no URL found)\n\n"
                "If the text contains '456 Glenfield Road, Unit 2, Glenfield 0678' and '+64 9 483 5555' but NO website, output:\n"
                "Address: 456 Glenfield Road, Unit 2, Glenfield 0678\n"
                "Phone: +64 9 483 5555\n\n"
                "DO NOT INVENT WEBSITE INFORMATION. If no website is found, omit the Website line completely.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )


        # Product/service queries
        elif any(keyword in query_lower for keyword in ['price', 'cost', 'buy', 'purchase', 'deal', 'sale']):
            summary_prompt = (
                "Extract COMPLETE product/service information:\n\n"
                "## PRICING & AVAILABILITY\n"
                "- Exact prices and currency\n"
                "- Model numbers/specifications\n"
                "- Availability status\n"
                "- Seller/retailer information\n"
                "- Shipping costs and delivery times\n"
                "- Return policies\n\n"
                "## PRODUCT DETAILS\n"
                "- Features and specifications\n"
                "- Dimensions/sizes\n"
                "- Colors/options available\n"
                "- Warranty information\n\n"
                "Include ALL pricing, specifications, and purchase details. Be very specific about numbers and options.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        # Weather queries
        elif any(keyword in query_lower for keyword in
                 ['weather', 'forecast', 'temperature', 'rain', 'snow', 'humidity']):
            summary_prompt = (
                "Extract COMPLETE weather forecast information:\n\n"
                "## CURRENT CONDITIONS\n"
                "- Temperature and feels-like temperature\n"
                "- Weather description (sunny, rainy, etc.)\n"
                "- Humidity, wind speed and direction\n"
                "- Precipitation chances\n"
                "- Air quality and UV index\n\n"
                "## FORECAST\n"
                "- Hourly and daily forecasts\n"
                "- High/low temperatures\n"
                "- Severe weather alerts\n"
                "- Sunrise/sunset times\n\n"
                "## LOCATION DETAILS\n"
                "- Specific city/region\n"
                "- Geographic details if available\n"
                "- Timezone information\n\n"
                "Include ALL numerical weather data, times, and location specifics.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        else:
            # General comprehensive summary (for news, general info, etc.)
            summary_prompt = (
                "Create a COMPREHENSIVE summary that PRESERVES practical information:\n\n"
                "## ESSENTIAL DETAILS\n"
                "- Full names of businesses, people, organizations\n"
                "- Complete addresses, phone numbers, contact information\n"
                "- Prices, costs, financial figures\n"
                "- Dates, times, schedules\n"
                "- Locations, coordinates, directions\n"
                "- Website URLs, email addresses\n\n"
                "## KEY INFORMATION\n"
                "- Main facts and findings\n"
                "- Important numbers and statistics\n"
                "- Recent developments\n"
                "- Contact methods\n\n"
                "## ADDITIONAL CONTEXT\n"
                "- Background information\n"
                "- Related services or options\n"
                "- User reviews or ratings if available\n\n"
                "CRITICAL: NEVER omit addresses, phone numbers, prices, or contact information. Include them verbatim.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        try:
            payload = {
                "model": "qwen2.5:7b-instruct",
                "prompt": summary_prompt,
                "stream": False,
                "temperature": 0.1,  # Lower temperature for more factual accuracy
                "max_tokens": 1200  # More tokens for detailed information
            }

            with httpx.Client(timeout=75.0) as client:
                r = client.post("http://localhost:11434/api/generate", json=payload)
                r.raise_for_status()
                response = r.json().get("response", "").strip()

                # Enhanced fallback for better information extraction
                if len(response) < 100 or "no information" in response.lower():
                    return self._extract_practical_information(text[:12000], query_lower)

                return response

        except Exception as e:
            return self._extract_practical_information(text[:10000], query_lower)

    def _extract_practical_information(self, text: str, query_type: str) -> str:
        """Enhanced fallback extraction focusing on practical information"""
        import re

        sections = []

        # Enhanced address extraction
        address_patterns = [
            # Standard street addresses
            r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Highway|Hwy)\.?\s*(?:#\s*\d+)?\s*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b',
            # Basic address format
            r'\b\d+\s+[\w\s]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court),\s*[\w\s]+,\s*[A-Z]{2}\b',
            # PO Boxes
            r'\b(?:P\.?O\.?\s*Box|PO Box|P O Box)\s+\d+[^.!?]*',
        ]

        addresses = []
        for pattern in address_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(found)

        # Filter out obviously fake or placeholder addresses
        real_addresses = []
        for addr in addresses:
            addr_lower = addr.lower()
            # Skip placeholder text
            if any(placeholder in addr_lower for placeholder in
                   ['address may vary', 'varies', 'please contact', 'call for', 'not available']):
                continue
            # Skip if it's just a city/state without street
            if re.match(r'^[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}', addr) and not re.search(r'\d+', addr):
                continue
            real_addresses.append(addr.strip())

        if real_addresses:
            sections.append("## ADDRESSES FOUND")
            sections.extend([f"- {addr}" for addr in set(real_addresses)[:3]])
        # Extract website URLs (more comprehensive)
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+\.[a-z]{2,}',
            r'[a-z0-9.-]+\.[a-z]{2,}/[^\s<>"{}|\\^`\[\]]*',
        ]

        urls = []
        for pattern in url_patterns:
            urls.extend(re.findall(pattern, text, re.IGNORECASE))

        # Filter and clean URLs
        clean_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?)]+$', '', url)
            # Skip common false positives
            if any(bad in url.lower() for bad in ['example.com', 'website.com', 'yourwebsite', 'domain.com']):
                continue
            # Ensure it looks like a real URL
            if '.' in url and len(url) > 8:
                # Add http:// if missing for www URLs
                if url.startswith('www.') and not url.startswith('http'):
                    url = 'https://' + url
                clean_urls.append(url)

        if clean_urls:
            sections.append("\n## WEBSITES")
            sections.extend([f"- {url}" for url in set(clean_urls)[:3]])

        # Extract phone numbers
        phone_pattern = r'(\+?\d{1,2}?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            sections.append(f"\n## PHONE NUMBERS: {', '.join(set(phones)[:3])}")

        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            sections.append(f"\n## EMAIL ADDRESSES: {', '.join(set(emails)[:3])}")

        # Extract prices and costs
        prices = re.findall(r'\$?\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|USD|€|£|¥)?', text)
        if prices:
            sections.append(f"\n## PRICES MENTIONED: {', '.join(set(prices)[:8])}")

        # Flight-specific extraction
        if 'flight' in query_type:
            flight_info = re.findall(r'[A-Z]{2}\d+\s+.*?(?:\d{1,2}:\d{2}|AM|PM)', text)
            if flight_info:
                sections.append("\n## FLIGHT DETAILS")
                sections.extend([f"- {info}" for info in flight_info[:5]])

        # Business hours
        hours = re.findall(
            r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*.*?\d{1,2}:\d{2}\s*(?:AM|PM)?.*?\d{1,2}:\d{2}\s*(?:AM|PM)?', text,
            re.IGNORECASE)
        if hours:
            sections.append("\n## BUSINESS HOURS")
            sections.extend([f"- {hour}" for hour in hours[:3]])

        # Weather data extraction
        if 'weather' in query_type:
            temps = re.findall(r'\b\d{1,3}°?F?\b', text)
            if temps:
                sections.append(f"\n## TEMPERATURES: {', '.join(set(temps)[:6])}")

        # If we found practical information, return it
        if sections:
            return "\n".join(sections)
        else:
            # Return meaningful content lines as fallback
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 30]
            return "## KEY INFORMATION EXTRACTED\n" + "\n".join([f"- {line}" for line in lines[:10]])

    def _extract_detailed_news(self, text: str) -> str:
        """Enhanced fallback extraction with more structure"""
        import re

        # Extract key information with more context
        sections = []

        # Headlines and key sentences
        sentences = re.split(r'[.!?]+', text)
        key_sentences = []

        important_indicators = [
            'announced', 'reported', 'confirmed', 'revealed', 'disclosed',
            'investigation', 'charged', 'arrested', 'settlement', 'agreement',
            'election', 'resigned', 'appointed', 'launched', 'released',
            'fire', 'accident', 'killed', 'injured', 'missing', 'found',
            'storm', 'flood', 'earthquake', 'weather', 'forecast', 'temperature',
            'budget', 'funding', 'cost', 'price', 'investment'
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 25 and
                    any(indicator in sentence.lower() for indicator in important_indicators)):
                key_sentences.append(sentence)
                if len(key_sentences) >= 12:
                    break

        if key_sentences:
            sections.append("## KEY DEVELOPMENTS")
            sections.extend([f"- {s}" for s in key_sentences[:10]])

        # Extract numbers and statistics
        numbers = re.findall(r'\b(\$?[£€]?\d+(?:,\d+)*(?:\.\d+)?[%€£$]?(?:\s*(?:million|billion|thousand))?)\b',
                             text[:5000])
        if numbers:
            sections.append(f"\n## KEY NUMBERS: {', '.join(set(numbers[:8]))}")

        # Extract locations
        locations = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text[:3000])
        unique_locs = list(
            set([loc for loc in locations if len(loc) > 3 and loc not in ['The', 'This', 'That', 'There', 'Here']]))
        if unique_locs:
            sections.append(f"\n## MENTIONED LOCATIONS: {', '.join(unique_locs[:6])}")

        if sections:
            return "\n".join(sections)
        else:
            # Last resort: return structured excerpt
            lines = text.split('\n')
            meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 40][:8]
            return "## CONTENT OVERVIEW\n" + "\n".join([f"- {line}" for line in meaningful_lines])

    def summarise_with_qwen(self, text: str, url: str, pubdate: str):
        text = text[:20000]  # Limit text length
        pd_line = f"(Publish/Update date: {pubdate})\n" if pubdate else ""

        # FIRST PASS: Extract mathematical content specifically
        math_prompt = (
            "Extract ALL mathematical equations, formulas, and technical content from the following text. "
            "Preserve them exactly in their original LaTeX format ($$...$$, \\[...\\], $...$, etc.).\n"
            "Include:\n"
            "- All equations and formulas\n"
            "- Mathematical expressions\n"
            "- Chemical formulas\n"
            "- Code snippets\n"
            "- Important technical definitions\n"
            "Output the mathematical/technical content exactly as found, without summarization.\n"
            f"{pd_line}"
            f"Source: {url}\n\nCONTENT:\n{text[:10000]}"  # Use first 10k chars for math extraction
        )

        # SECOND PASS: Create a summary that REFERENCES the preserved math
        summary_prompt = (
            "Create a comprehensive summary (10-15 bullet points) that includes:\n"
            "- Key findings and conclusions\n"
            "- Important data points and results\n"
            "- References to mathematical content (say 'see equation X' or 'the formula shows')\n"
            "- Main arguments and evidence\n"
            "- Do NOT remove technical details - include them in context\n"
            "- Preserve specific numbers, measurements, and quantitative results\n"
            "Be detailed enough to be useful for technical analysis.\n"
            f"{pd_line}"
            f"Source: {url}\n\nCONTENT:\n{text}"
        )

        try:
            # Get mathematical content
            math_content = self.qwen.generate(math_prompt)

            # Get comprehensive summary
            summary = self.qwen.generate(summary_prompt)

            # Combine both with clear separation
            combined_result = f"MATHEMATICAL CONTENT:\n{math_content}\n\nSUMMARY:\n{summary}"

            return combined_result

        except Exception:
            # Fallback: Use a more math-friendly single prompt
            fallback_prompt = (
                "Create a DETAILED technical summary (12-18 bullet points) that PRESERVES all mathematical content.\n"
                "CRITICAL: Keep ALL equations, formulas, and LaTeX expressions exactly as they appear.\n"
                "Include:\n"
                "- Complete equations in $$...$$, \\[...\\], $...$ format\n"
                "- Mathematical proofs and derivations\n"
                "- Chemical formulas and reactions\n"
                "- Code snippets and algorithms\n"
                "- Quantitative results with exact numbers\n"
                "- Do NOT simplify or remove technical details\n"
                "- Focus on preserving the mathematical richness of the content\n"
                f"{pd_line}"
                f"Source: {url}\n\nCONTENT:\n{text}"
            )
            try:
                payload = {"model": "qwen2.5:7b-instruct", "prompt": fallback_prompt, "stream": False}
                with httpx.Client(timeout=90.0) as client:
                    r = client.post("http://localhost:11434/api/generate", json=payload)
                    r.raise_for_status()
                    return r.json().get("response", "").strip()
            except Exception as e:
                return f"Summarization failed: {e}"

    def extract_images(self, html: str, base_url: str, limit: int = 3):
        urls = []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return urls

        for img in soup.find_all("img"):
            src = img.get("src") or ""
            if not src or src.startswith("data:") or re.search(r"\.svg($|\?)", src, re.I):
                continue

            alt = (img.get("alt") or "").lower()
            src_l = src.lower()
            if any(k in src_l for k in ["sprite", "icon", "logo", "ads", "advert", "pixel"]):
                continue
            if any(k in alt for k in ["icon", "logo"]):
                continue

            full = urljoin(base_url, src)
            if full not in urls:
                urls.append(full)
            if len(urls) >= limit:
                break
        return urls

    def synthesize_search_results(self, text: str):
        """Speak search results using DEDICATED search window"""

        # === STOP PROGRESS INDICATOR IMMEDIATELY ===
        self.stop_search_progress_indicator()

        def _tts_worker():
            if not text or not text.strip():
                return

            try:
                # Use math speaking for search results too
                speak_math = getattr(self, 'speak_math_var', tk.BooleanVar(value=True)).get()
                clean_tts_text = clean_for_tts(text, speak_math=speak_math)

                # === CRITICAL: Use DEDICATED search window ===
                self.preview_search_results(text)

                # Continue with TTS...
                output_path = "out/search_results.wav"

                if self.synthesize_to_wav(clean_tts_text, output_path, role="text"):
                    with self._play_lock:
                        self._play_token += 1
                        my_token = self._play_token
                        self.interrupt_flag = False
                        self.speaking_flag = True

                    self.set_light("speaking")

                    play_path = output_path
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(output_path, "out/search_results_echo.wav")
                            self.logln("[echo] processed search results -> out/search_results_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e} (playing dry)")

                    self.play_wav_with_interrupt(play_path, token=my_token)

            except Exception as e:
                self.logln(f"[search][TTS] Error: {e}")
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()

    # End syththesise_search
    def _populate_edge_voices(self):
        """Populate Edge voice dropdown with available voices"""
        if not EDGE_TTS_AVAILABLE:
            self.edge_voice_combo['values'] = ["(edge-tts not installed)"]
            return

        # Curated list of high-quality Edge voices
        # Format: "ShortName" which edge-tts uses directly
        # Curated list of high-quality Edge voices
        edge_voices = [
            # === ENGLISH - American ===
            "en-US-AriaNeural",
            "en-US-GuyNeural",
            "en-US-JennyNeural",
            "en-US-ChristopherNeural",
            "en-US-EricNeural",
            "en-US-MichelleNeural",
            "en-US-RogerNeural",
            "en-US-SteffanNeural",
            # === ENGLISH - British ===
            "en-GB-SoniaNeural",
            "en-GB-RyanNeural",
            "en-GB-LibbyNeural",
            "en-GB-ThomasNeural",
            # === ENGLISH - Australian ===
            "en-AU-NatashaNeural",
            # === ENGLISH - New Zealand ===
            "en-NZ-MitchellNeural",
            "en-NZ-MollyNeural",
            # === ENGLISH - Irish ===
            "en-IE-ConnorNeural",
            "en-IE-EmilyNeural",
            # === ENGLISH - Indian ===
            "en-IN-NeerjaNeural",
            "en-IN-PrabhatNeural",
            # === FRENCH ===
            "fr-FR-DeniseNeural",
            "fr-FR-HenriNeural",
            "fr-CA-SylvieNeural",
            "fr-CA-JeanNeural",
            "fr-CA-AntoineNeural",
            # === GERMAN ===
            "de-AT-IngridNeural",
            "de-AT-JonasNeural",
            "de-CH-LeniNeural",
            "de-CH-JanNeural",
            "de-DE-AmalaNeural",
            "de-DE-ConradNeural",
            "de-DE-KatjaNeural",
            "de-DE-KillianNeural",
            # === SPANISH ===
            "es-ES-AlvaroNeural",
            "es-ES-ElviraNeural",
            "es-MX-DaliaNeural",
            "es-MX-JorgeNeural",
            "es-AR-ElenaNeural",
            "es-AR-TomasNeural",
            # === RUSSIAN ===
            "ru-RU-SvetlanaNeural",
            "ru-RU-DmitryNeural",
            # === HEBREW ===
            "he-IL-HilaNeural",
            "he-IL-AvriNeural",
            # === BONUS: Other Popular Languages ===
            # Italian
            "it-IT-ElsaNeural",
            "it-IT-DiegoNeural",
            "it-IT-IsabellaNeural",
            # Portuguese
            "pt-BR-FranciscaNeural",
            "pt-BR-AntonioNeural",
            "pt-PT-RaquelNeural",
            "pt-PT-DuarteNeural",
            # Japanese
            "ja-JP-NanamiNeural",
            "ja-JP-KeitaNeural",
            # Chinese Mandarin
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-XiaoyiNeural",
            "zh-CN-YunxiNeural",
            # Korean
            "ko-KR-SunHiNeural",
            "ko-KR-InJoonNeural",
            # Dutch
            "nl-NL-ColetteNeural",
            "nl-NL-MaartenNeural",
            # Polish
            "pl-PL-MarekNeural",
            # Arabic
            "ar-SA-ZariyahNeural",
            "ar-SA-HamedNeural",
            # Hindi
            "hi-IN-SwaraNeural",
            "hi-IN-MadhurNeural",
        ]

        self.edge_voice_combo['values'] = edge_voices
        self.edge_voice_combo.current(0)

        # Auto-switch engine when voice is selected
        def _on_edge_voice_selected(event=None):
            if self.tts_engine.get() != "edge":
                self.tts_engine.set("edge")
                self._on_engine_change()
                self.logln("[tts] Auto-switched to Edge engine")

        def _on_sapi_voice_selected(event=None):
            if self.tts_engine.get() != "sapi5":
                self.tts_engine.set("sapi5")
                self._on_engine_change()
                self.logln("[tts] Auto-switched to SAPI5 engine")

        self.edge_voice_combo.bind("<<ComboboxSelected>>", _on_edge_voice_selected)
        self.sapi_combo.bind("<<ComboboxSelected>>", _on_sapi_voice_selected)

        # Store mapping for JSON loading
        self.edge_voice_list = edge_voices
        self.logln(f"[tts] Loaded {len(edge_voices)} Edge voices")

    def _on_engine_change(self, event=None):
        """Handle TTS engine change - enable/disable appropriate voice dropdowns"""
        engine = self.tts_engine.get()

        if engine == "edge":
            self.edge_voice_combo.config(state="readonly")
            self.sapi_combo.config(state="disabled")
            self.logln("[tts] 🌐 Switched to Edge TTS (neural voices)")
        else:
            self.edge_voice_combo.config(state="disabled")
            self.sapi_combo.config(state="readonly")
            self.logln("[tts] 🗣️ Switched to SAPI5 (local voices)")

    def play_search_results(self, path: str, token=None):
        """Play search results audio with proper interrupt support"""
        try:
            # Use the existing playback infrastructure with token support
            with self._play_lock:
                self._play_token += 1
                my_token = self._play_token
                self.interrupt_flag = False
                self.speaking_flag = True

            self.set_light("speaking")
            self.temporary_mute_for_speech("text")  # Search uses text AI voice
            self.play_wav_with_interrupt(path, token=my_token)

        except Exception as e:
            self.logln(f"[search][playback] Error: {e}")
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")

    def normalize_query(self, q: str) -> str:
        """Add date context ONLY for specific time-related queries"""
        ql = q.lower()
        now = datetime.now()

        # Only add dates for explicit time references
        if "today" in ql:
            q += " " + now.strftime("%Y-%m-%d")
        elif "yesterday" in ql:
            q += " " + (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "this week" in ql:
            q += " " + now.strftime("week %G-W%V")
        # DON'T add dates for "latest", "recent", "current" etc.

        return q
