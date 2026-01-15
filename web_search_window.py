# web_search_window.py
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import threading
import time
from typing import List
from PIL import Image, ImageTk
import webbrowser
from PIL import Image, ImageTk
import webbrowser
# Add these imports at the TOP of web_search_window.py
import re
import httpx
import numpy as np
import sounddevice as sd
from datetime import datetime, timedelta
from io import BytesIO
import base64
from urllib.parse import urljoin
import trafilatura
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class Item:
    title: str
    url: str
    snippet: str = ""
    pubdate: Optional[str] = None
    summary: Optional[str] = None
    image_urls: List[str] = None

    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []

class WebSearchWindow(tk.Toplevel):
    def __init__(self, master, log_fn=None):
        super().__init__(master)
        self.title("Web Search")
        self.geometry("980x740")
        self.minsize(780, 620)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self._log = log_fn or (lambda msg: None)

        self.queue = queue.Queue()
        self.in_progress = False

        # Initialize tracking variables
        self.thumb_cache = {}
        self.image_bytes_cache = {}
        self.thumb_target = {}
        self.thumb_placeholder = {}
        self._all_results = []

        # LaTeX preview window for search results
        self.latex_win = None
        self.latex_auto = tk.BooleanVar(value=True)

        self._build_ui()
        self._build_strip_lights()
        self._tick_strip_lights()
        self._poll_queue()

    def _build_ui(self):
        # Main container
        self.main_container = ttk.Frame(self, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=36, pady=36)

        # Query section
        ttk.Label(self.main_container, text="Query:").pack(anchor="w")
        self.txt_in = tk.Text(self.main_container, height=3, wrap="word")
        self.txt_in.pack(fill=tk.X, pady=(2, 8))

        # LaTeX controls for search results
        latex_controls = ttk.Frame(self.main_container)
        latex_controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(
            latex_controls, text="Auto LaTeX preview for search results",
            variable=self.latex_auto
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            latex_controls, text="Show/Hide LaTeX",
            command=self.toggle_latex
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            latex_controls, text="Copy Raw LaTeX",
            command=self.copy_raw_latex
        ).pack(side="left")

        self.btn = ttk.Button(self.main_container, text="Search & Summarise", command=self.on_go)
        self.btn.pack(pady=(0, 8), anchor="w")

        # Output section - create a container for results that we can refresh
        self.results_container = ttk.Frame(self.main_container)
        self.results_container.pack(fill=tk.BOTH, expand=True)

        # Create initial results display
        self._create_results_display()

        # Status
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.main_container, textvariable=self.status).pack(anchor="w", pady=(6, 0))

    def _create_results_display(self):
        """Create or recreate the results display area"""
        # Clear existing results display if it exists
        if hasattr(self, 'results_frame') and self.results_frame:
            try:
                self.results_frame.destroy()
            except:
                pass

        # Create new results frame
        self.results_frame = ttk.Frame(self.results_container)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Output text area
        ttk.Label(self.results_frame, text="Output:").pack(anchor="w")
        self.txt_out = tk.Text(self.results_frame, wrap="word", height=12)
        self.txt_out.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for text output
        sbr = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.txt_out.yview)
        sbr.place(relx=1.0, rely=0, relheight=0.7, anchor="ne")
        self.txt_out.config(yscrollcommand=sbr.set)

        # Images container
        self.images_container = ttk.Frame(self.results_frame)
        self.images_container.pack(fill=tk.X, pady=8)

    def _ensure_latex_window(self):
        """Create LaTeX window if it doesn't exist"""
        if self.latex_win is None or not self.latex_win.winfo_exists():
            # Use the same settings as the main app's LaTeX window
            self.latex_win = LatexWindow(
                self.master,  # Use master (main app) as parent for consistency
                log_fn=self._log,
                text_family="Segoe UI",
                text_size=12,
                math_pt=8
            )

    def toggle_latex(self):
        """Toggle LaTeX window using main app's system"""
        try:
            if hasattr(self, 'main_app') and self.main_app:
                self.main_app.toggle_latex()
            elif hasattr(self, 'master') and hasattr(self.master, 'ensure_latex_window'):
                latex_win = self.master.ensure_latex_window("search")
                if latex_win.state() == "withdrawn":
                    latex_win.show()
                else:
                    latex_win.hide()
        except Exception as e:
            error_msg = f"[search] toggle latex error: {e}"
            if hasattr(self, 'logln'):
                self.logln(error_msg)
            else:
                print(error_msg)

    def copy_raw_latex(self):
        """Copy raw LaTeX source to clipboard with proper document structure"""
        try:
            content = self._last_text or ""

            if content.strip():
                # Create a complete LaTeX document
                latex_document = f"""\\documentclass{{article}}
    \\usepackage{{amsmath}}
    \\usepackage{{amssymb}}
    \\usepackage{{graphicx}}
    \\usepackage{{hyperref}}
    \\begin{{document}}

    {content}

    \\end{{document}}"""

                self.clipboard_clear()
                self.clipboard_append(latex_document)
                self._log("[latex] Complete LaTeX document copied to clipboard")
            else:
                self.clipboard_clear()
                self.clipboard_append("")
                self._log("[latex] No content to copy")

        except Exception as e:
            self._log(f"[latex] copy raw failed: {e}")

    def preview_latex(self, content: str, context="text"):
        """Preview LaTeX content with append/replace option"""
        if not self.latex_auto.get():
            return

        def _go():
            try:
                latex_win = self.ensure_latex_window(context)
                latex_win.show()

                # === CHECK APPEND MODE ===
                if self.latex_append_mode.get():
                    # APPEND MODE - add to existing content
                    latex_win.append_document(content)
                    self.logln(f"[latex] 📝 Appended to {context} window")
                else:
                    # REPLACE MODE - clear and show new content (original behavior)
                    latex_win.show_document(content)
                    self.logln(f"[latex] 🔄 Showing in {context} window (replace mode)")

                self._current_latex_context = context

            except Exception as e:
                self.logln(f"[latex] preview error ({context}): {e}")

        self.master.after(0, _go)

    def _build_strip_lights(self):
        self.strip_canvas = tk.Canvas(self, highlightthickness=0, bg="white")
        self.strip_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        tk.Misc.lower(self.strip_canvas)

        self.strip_items = []
        self.strip_colors = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#5ac8fa", "#007aff", "#af52de"]
        self.strip_step = 0

        margin = 10
        thickness = 40
        seg = 44

        def make_rect(x1, y1, x2, y2):
            rid = self.strip_canvas.create_rectangle(x1, y1, x2, y2, width=0, fill="white")
            self.strip_items.append(rid)

        def rebuild(_evt=None):
            self.strip_canvas.delete("all")
            self.strip_items.clear()
            W = max(1, self.winfo_width())
            H = max(1, self.winfo_height())
            # bottom
            x = margin
            y = H - margin - thickness
            while x + seg <= W - margin:
                make_rect(x, y, x + seg - 3, y + thickness)
                x += seg
            # left
            x = margin
            y = H - margin - thickness
            while y - seg >= margin:
                make_rect(x, y - seg + 3, x + thickness, y)
                y -= seg
            # top
            x = margin + thickness
            y = margin
            while x + seg <= W - margin - thickness:
                make_rect(x, y, x + seg - 3, y + thickness)
                x += seg
            # right
            x = W - margin - thickness
            y = margin + thickness
            while y + seg <= H - margin - thickness:
                make_rect(x, y, x + thickness, y + seg - 3)
                y += seg

        self.bind("<Configure>", rebuild)
        self.rebuild = rebuild
        rebuild()

    def _tick_strip_lights(self):
        if self.in_progress and self.strip_items:
            n = len(self.strip_items)
            self.strip_step = (self.strip_step + 1) % (n * len(self.strip_colors))
            for i, rid in enumerate(self.strip_items):
                idx = (self.strip_step + i) % len(self.strip_colors)
                self.strip_canvas.itemconfig(rid, fill=self.strip_colors[idx])
        else:
            for rid in self.strip_items:
                self.strip_canvas.itemconfig(rid, fill="white")
        self.after(90, self._tick_strip_lights)

    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        """Hide the window (called by close button)"""
        try:
            self.withdraw()
        except Exception as e:
            print(f"Error hiding window: {e}")

    def log(self, msg: str):
        self.status.set(msg)
        self.update_idletasks()

    def on_go(self):
        if self.in_progress:
            self.log("Already running…")
            return

        q = self.txt_in.get("1.0", "end").strip()
        if not q:
            messagebox.showinfo("Info", "Please type a query.")
            return

        q = self.normalize_query(q)

        # DON'T recreate the results display - just clear the existing one
        self.txt_out.delete("1.0", "end")

        # Clear the images container
        for widget in self.images_container.winfo_children():
            widget.destroy()

        # Clear tracking variables (keep cache for performance)
        self.image_bytes_cache.clear()
        self.thumb_target.clear()
        self.thumb_placeholder.clear()
        self._all_results.clear()
        # Reset accumulated LaTeX content for new search
        self._accumulated_latex_content = ""

        self.in_progress = True
        self.btn.config(state="disabled")
        self.log("Searching…")

        # Start search in thread
        try:
            threading.Thread(target=self._thread_run, args=(q,), daemon=True).start()
        except Exception as e:
            self.log(f"Failed to start search thread: {e}")
            self.in_progress = False
            self.btn.config(state="normal")

    def _thread_run(self, query: str):
        # === SEARCH PROGRESS AUDIBLE INDICATOR ===
        if hasattr(self.master, 'start_search_progress_indicator'):
            self.master.start_search_progress_indicator()

        try:
            results = self.brave_search(query, 6)
            self.queue.put(("searched", len(results)))

            for idx, it in enumerate(results, 1):
                self.queue.put(("status", f"Processing result {idx}/{len(results)}..."))

                try:
                    html = self.polite_fetch(it.url)
                    if not html:
                        self.log(f"Failed to fetch: {it.url}")
                        self.queue.put(("article", {
                            "title": it.title, "url": it.url, "pubdate": None,
                            "summary": "Fetch failed - could not retrieve content.", "images": []
                        }))
                        continue

                    it.pubdate = self.guess_pubdate(html)
                    it.image_urls = self.extract_images(html, it.url)
                    text = self.extract_readable(html, it.url)

                    if len(text) < 400:
                        it.summary = "Not enough readable text to summarise."
                    else:
                        try:
                            it.summary = self.summarise_with_qwen(text, it.url, it.pubdate)
                            if not it.summary or "failed" in it.summary.lower():
                                it.summary = "Summarization unavailable for this content."
                        except Exception as e:
                            self.log(f"Summarization error: {e}")
                            it.summary = f"Summarization error: {str(e)}"

                    self.queue.put(("article", {
                        "title": it.title, "url": it.url, "pubdate": it.pubdate,
                        "summary": it.summary, "images": it.image_urls
                    }))
                    time.sleep(0.6)

                except Exception as e:
                    self.log(f"Error processing result {idx}: {e}")
                    continue

            self.queue.put(("done", None))
        except Exception as e:
            self.log(f"Search thread error: {e}")
            self.queue.put(("error", str(e)))
        # === STOP ON ERROR ===
        if hasattr(self.master, 'stop_search_progress_indicator'):
            self.master.stop_search_progress_indicator()

    def _create_final_summary(self) -> str:
        """Create a consolidated summary of all search results"""
        if not self._all_results:
            return "No results to summarize."

        # Use the ACCUMULATED content that already has all the equations
        if hasattr(self, '_accumulated_latex_content') and self._accumulated_latex_content:
            final_text = "COMPLETE SEARCH RESULTS WITH EQUATIONS:\n\n" + self._accumulated_latex_content
        else:
            # Fallback to the original summary method
            summary_parts = ["Search Results Summary:"]
            for i, result in enumerate(self._all_results, 1):
                title = result.get('title', 'No title')
                summary = result.get('summary', 'No summary available')

                # Clean up the summary text but preserve equations
                clean_summary = re.sub(r'=+', '', summary)
                clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
                clean_summary = re.sub(r'[\*#_-]{2,}', '', clean_summary)
                clean_summary = re.sub(r'^(summary|result|findings?):?\s*', '', clean_summary, flags=re.IGNORECASE)

                if len(clean_summary) > 250:
                    clean_summary = clean_summary[:247] + "..."

                summary_parts.append(f"Result {i}: {title}")
                if clean_summary and clean_summary != "No summary available":
                    summary_parts.append(f"{clean_summary}")
                summary_parts.append("")

            final_text = "\n".join(summary_parts)
            final_text = re.sub(r'\n{3,}', '\n\n', final_text)

        # PREVIEW LATEX FOR THE FINAL ACCUMULATED CONTENT
        self.preview_latex(final_text)

        return final_text.strip()

    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.queue.get_nowait()
                try:
                    if msg == "status":
                        self.log(payload)
                    elif msg == "searched":
                        n = payload
                        self.log(f"Found {n} results. Summarising…")
                        self.play_ding()

                    elif msg == "article":
                        if self.in_progress:  # Only process if we're still searching
                            data = payload
                            block = (
                                    f"Title: {data['title']}\n"
                                    f"URL: {data['url']}\n"
                                    + (f"Publish date: {data['pubdate']}\n" if data['pubdate'] else "")
                                    + f"\nSummary:\n{data['summary']}"
                            )
                            self.append_output(block)
                            self._show_images(data.get("images", []), data['title'], data['url'])
                            self._all_results.append(data)

                    elif msg == "done":
                        self.log("Done.")
                        self.in_progress = False
                        self.btn.config(state="normal")
                        self.play_dong()
                        # === STOP PROGRESS INDICATOR ===
                        if hasattr(self.master, 'stop_search_progress_indicator'):
                            self.master.stop_search_progress_indicator()

                        if self._all_results:
                            summary_text = self._create_final_summary()
                            if hasattr(self, 'synthesize_search_results'):
                                self.synthesize_search_results(summary_text)

                    elif msg == "imgbytes":
                        url, data = payload
                        self.image_bytes_cache[url] = data
                        target = self.thumb_target.pop(url, None)
                        if target and self.winfo_exists():
                            self._create_thumb_from_bytes(url, data, target_frame=target)

                    elif msg == "imgfail":
                        url = payload
                        ph = self.thumb_placeholder.pop(url, None)
                        if ph and ph.winfo_exists():
                            ph.config(text="Image blocked or unavailable")

                    elif msg == "error":
                        self.in_progress = False
                        self.btn.config(state="normal")
                        self.log(f"Error: {payload}")
                    # ===  STOP ON ERROR ===
                    if hasattr(self.master, 'stop_search_progress_indicator'):
                        self.master.stop_search_progress_indicator()

                except Exception as e:
                    self.log(f"Error processing queue message {msg}: {e}")

        except queue.Empty:
            pass
        except Exception as e:
            self.log(f"Queue polling error: {e}")
            self.in_progress = False
            self.btn.config(state="normal")
        # ===  STOP ON GENERAL ERROR ===
        if hasattr(self.master, 'stop_search_progress_indicator'):
            self.master.stop_search_progress_indicator()

        if self.winfo_exists():
            self.after(120, self._poll_queue)

    def append_output(self, block: str):
        self.txt_out.insert("end", block + "\n\n")
        self.txt_out.see("end")

        # ACCUMULATE all results for LaTeX preview instead of overwriting
        if not hasattr(self, '_accumulated_latex_content'):
            self._accumulated_latex_content = ""

        # Add this result to the accumulated content
        self._accumulated_latex_content += block + "\n\n"

        # Preview the ACCUMULATED content (all results so far)
        self.preview_latex(self._accumulated_latex_content)

    def _show_images(self, urls: List[str], article_title: str, article_url: str):
        try:
            section = ttk.Frame(self.images_container)
            section.pack(fill=tk.X, pady=6)

            hdr = ttk.Label(section, text=f"Images: {article_title}", cursor="hand2")
            hdr.pack(anchor="w")
            hdr.bind("<Button-1>", lambda e, u=article_url: webbrowser.open(u))

            if not urls:
                ttk.Label(section, text="(No images found.)").pack(anchor="w")
                return

            for i, u in enumerate(urls[:3]):
                try:
                    if u in self.thumb_cache:
                        self._place_thumb(u, self.thumb_cache[u], target_frame=section)
                    elif u in self.image_bytes_cache:
                        self._create_thumb_from_bytes(u, self.image_bytes_cache[u], target_frame=section)
                    else:
                        ph = ttk.Label(section, text=f"Loading image {i + 1}…")
                        ph.pack(side=tk.LEFT, padx=6)
                        self.thumb_target[u] = section
                        self.thumb_placeholder[u] = ph
                        threading.Thread(target=self._thread_fetch_image, args=(u,), daemon=True).start()
                except Exception as e:
                    self.log(f"Error displaying image {i}: {e}")
                    continue
        except Exception as e:
            self.log(f"Error in _show_images: {e}")

    def _thread_fetch_image(self, url: str):
        try:
            headers = {"User-Agent": "LocalAI-ResearchBot/1.0", "Referer": url}
            with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.content
            self.queue.put(("imgbytes", (url, data)))
        except Exception:
            self.queue.put(("imgfail", url))

    def _create_thumb_from_bytes(self, url: str, data: bytes, target_frame=None):
        try:
            img = Image.open(BytesIO(data))
            img.thumbnail((220, 140))
            tkimg = ImageTk.PhotoImage(img)
            self.thumb_cache[url] = tkimg
            ph = self.thumb_placeholder.pop(url, None)
            if ph and ph.winfo_exists():
                ph.destroy()
            self._place_thumb(url, tkimg, target_frame=target_frame)
        except Exception:
            self.queue.put(("imgfail", url))

    def _place_thumb(self, url: str, tkimg: ImageTk.PhotoImage, target_frame=None):
        parent = target_frame if target_frame is not None else self.images_container
        lbl = ttk.Label(parent, image=tkimg)
        lbl.image = tkimg
        lbl.pack(side=tk.LEFT, padx=6, pady=4)
        lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

    def play_ding(self):
        """Play search start sound"""
        try:
            fs = 16000
            duration = 0.17
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = 988
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            fade_samples = int(0.01 * fs)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[ding] error: {e}")

    def play_dong(self):
        """Play search complete sound"""
        try:
            fs = 16000
            duration = 0.26
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = 659
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            fade_samples = int(0.01 * fs)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[dong] error: {e}")

    def normalize_query(self, q: str) -> str:
        """Add date context to time-related queries"""
        ql = q.lower()
        now = datetime.now()
        if "today" in ql:
            q += " " + now.strftime("%Y-%m-%d")
        if "yesterday" in ql:
            q += " " + (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if "this week" in ql:
            q += " " + now.strftime("week %G-W%V")
        return q

    def destroy(self):
        """Proper cleanup when window is closed"""
        self.in_progress = False
        # Close LaTeX window if open
        if self.latex_win and self.latex_win.winfo_exists():
            try:
                self.latex_win.destroy()
            except:
                pass
        super().destroy()


# === END WEB SEARCH WINDOW ===