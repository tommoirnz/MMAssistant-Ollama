# latex_window.py
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from io import BytesIO
import re
import os
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0

class LatexWindow(tk.Toplevel):
    def __init__(self, master, log_fn=None, text_family="Segoe UI", text_size=12, math_pt=8):
        super().__init__(master)
        self.title("LaTeX Preview")
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.geometry("800x600")
        self._log = log_fn or (lambda msg: None)

        # === PERMANENT DARK MODE COLORS ===
        self.dark_bg = "#000000"  # Black background
        self.dark_fg = "#ffffff"  # White text
        self.dark_highlight = "#4a86e8"  # Blue highlight

        # === SET WINDOW BACKGROUND (FRAME) ===
        self.configure(bg=self.dark_bg)  # Start with black frame

        # Initialize defaults
        self.text_family = text_family
        self.text_size = int(text_size)
        self.math_pt = int(math_pt)

        self._last_text = ""
        self._img_refs = []
        self._text_font = tkfont.Font(family=self.text_family, size=self.text_size)
        self._usetex_checked = False
        self._usetex_available = False
        self.show_raw = tk.BooleanVar(value=False)

        # --- Container ---
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        # --- Top bar - SIMPLIFIED (NO DARK MODE TOGGLE) ---
        topbar = ttk.Frame(container)
        topbar.pack(fill="x", padx=6, pady=(6, 2))

        # === TOP BAR CONTROLS ===
        ttk.Checkbutton(
            topbar, text="Show raw LaTeX",
            variable=self.show_raw,
            command=lambda: self.show_document(self._last_text or "")
        ).pack(side="left")

        ttk.Button(topbar, text="Copy Raw LaTeX", command=self.copy_raw_latex).pack(side="left", padx=(8, 6))
        ttk.Button(topbar, text="LaTeX Diagnostics", command=self._run_latex_diagnostics).pack(side="left")

        # === TEXT SIZE CONTROLS ===
        ttk.Label(topbar, text="Text pt").pack(side="left", padx=(12, 2))
        self.text_pt_var = tk.IntVar(value=self.text_size)
        txt_spin = ttk.Spinbox(
            topbar, from_=8, to=48, width=4,
            textvariable=self.text_pt_var,
            command=lambda: self.set_text_font(size=self.text_pt_var.get())
        )
        txt_spin.pack(side="left")

        ttk.Label(topbar, text="Math pt").pack(side="left", padx=(12, 2))
        self.math_pt_var = tk.IntVar(value=self.math_pt)
        math_spin = ttk.Spinbox(
            topbar, from_=6, to=64, width=4,
            textvariable=self.math_pt_var,
            command=lambda: self.set_math_pt(self.math_pt_var.get())
        )
        math_spin.pack(side="left")

        # === WINDOW SIZE CONTROLS ===
        ttk.Button(topbar, text="📐 Size+", command=self._increase_size).pack(side="right", padx=(6, 2))
        ttk.Button(topbar, text="📐 Size-", command=self._decrease_size).pack(side="right", padx=(2, 6))

        txt_spin.bind("<Return>", lambda _e: self.set_text_font(size=self.text_pt_var.get()))
        math_spin.bind("<Return>", lambda _e: self.set_math_pt(self.math_pt_var.get()))

        # --- Text + Scrollbar with PERMANENT DARK THEME ---
        textwrap = ttk.Frame(container)
        textwrap.pack(fill="both", expand=True)

        # === MAIN TEXT WIDGET - ALWAYS DARK ===
        self.textview = tk.Text(
            textwrap,
            bg=self.dark_bg,  # Black background
            fg=self.dark_fg,  # White text
            wrap="word",
            undo=False,
            insertbackground=self.dark_fg,  # White cursor
            selectbackground=self.dark_highlight,  # Blue selection
            inactiveselectbackground=self.dark_highlight  # Blue selection when not focused
        )

        vbar = ttk.Scrollbar(textwrap, orient="vertical", command=self.textview.yview)
        self.textview.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.textview.pack(side="left", fill="both", expand=True)

        self.textview.configure(font=self._text_font, state="normal")

        # === TEXT BINDINGS ===
        self.textview.bind("<Key>", self._block_keys)
        self.textview.bind("<<Paste>>", lambda e: "break")
        self.textview.bind("<Control-v>", lambda e: "break")
        self.textview.bind("<Control-x>", lambda e: "break")
        self.textview.bind("<Control-c>", lambda e: None)
        self.textview.bind("<Control-a>", self._select_all)

        # --- Context menu with DARK THEME ---
        self._menu = tk.Menu(self, tearoff=0, bg=self.dark_bg, fg=self.dark_fg)
        self._menu.add_command(label="Copy", command=lambda: self.textview.event_generate("<<Copy>>"))
        self._menu.add_command(label="Select All", command=lambda: self._select_all(None))
        self._menu.add_separator()
        self._menu.add_command(label="Copy Raw LaTeX", command=self.copy_raw_latex)
        # NO DARK MODE TOGGLE IN MENU
        self.textview.bind("<Button-3>", self._popup_menu)
        self.textview.bind("<Button-2>", self._popup_menu)

        # --- Highlight tags for DARK MODE ---
        self.textview.tag_configure("speak", background=self.dark_highlight, foreground=self.dark_bg)
        self.textview.tag_configure("normal", background="")
        self.textview.tag_configure("tight", spacing1=1, spacing3=1)

        self.withdraw()

    # ==================== ESSENTIAL WINDOW METHODS ====================
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

    def show(self):
        """Show the window"""
        self.deiconify()
        self.lift()

    def hide(self):
        """Hide the window"""
        self.withdraw()

    def clear(self):
        """Clear the content"""
        self.textview.delete("1.0", "end")
        self._img_refs.clear()

    def set_text_font(self, family=None, size=None):
        """Set text font family and/or size"""
        if family is not None:
            self.text_family = family
        if size is not None:
            self.text_size = int(size)
        try:
            self._text_font.config(family=self.text_family, size=self.text_size)
            self.textview.configure(font=self._text_font)
        except Exception as e:
            self._log(f"[latex] set_text_font error: {e}")

    def set_math_pt(self, pt: int):
        """Set math point size"""
        try:
            self.math_pt = int(pt)
        except Exception as e:
            self._log(f"[latex] set_math_pt error: {e}")

    # ==================== SCHEME METHOD (VISION MODE) ====================

    def set_scheme(self, scheme: str):
        """
        Simplified scheme method - no visual changes for vision mode
        """
        try:
            # Keep everything in default dark theme regardless of mode
            self.configure(bg=self.dark_bg)
            self.textview.tag_configure("speak", background=self.dark_highlight, foreground=self.dark_bg)

            if scheme == "vision":
                self._log("[latex] Vision mode - using default dark theme")
            else:
                self._log("[latex] Default mode - dark theme")

        except Exception as e:
            self._log(f"[latex] set_scheme error: {e}")

    # ==================== WINDOW SIZE CONTROL METHODS ====================

    def _increase_size(self):
        """Increase window size by 100x80 pixels"""
        width, height = self._get_current_size()
        new_width = min(2000, width + 100)
        new_height = min(1200, height + 80)
        self.geometry(f"{new_width}x{new_height}")

    def _decrease_size(self):
        """Decrease window size by 100x80 pixels"""
        width, height = self._get_current_size()
        new_width = max(400, width - 100)
        new_height = max(300, height - 80)
        self.geometry(f"{new_width}x{new_height}")

    def _get_current_size(self):
        """Get current window size from geometry string"""
        geometry = self.geometry()
        if 'x' in geometry and '+' in geometry:
            # Format: "widthxheight+x+y"
            size_part = geometry.split('+')[0]
            width, height = map(int, size_part.split('x'))
            return width, height
        return 800, 600  # Default size

    # ==================== UI HELPER METHODS ====================
    def append_document(self, text, wrap=900, separator="\n" + "=" * 50 + "\n"):
        """Append content to the existing document instead of replacing it"""
        if not text:
            return

        # Store the combined text for raw LaTeX copying
        if self._last_text:
            self._last_text += separator + text
        else:
            self._last_text = text

        try:
            # Get current content
            current_content = self.textview.get("1.0", "end-1c")

            # Add separator if there's existing content
            if current_content.strip():
                self.textview.insert("end", separator)

            # Process and append new content
            blocks = self.split_text_math(text)
            raw_mode = bool(self.show_raw.get())

            for kind, content in blocks:
                if kind == "text":
                    self.textview.insert("end", content, ("normal", "tight"))
                    continue
                if raw_mode:
                    self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))
                    continue
                try:
                    inline = self._is_inline_math(content)
                    fsz = max(6, self.math_pt - 2) if inline else self.math_pt
                    png = self.render_png_bytes(content, fontsize=fsz)
                    img = Image.open(BytesIO(png)).convert("RGBA")
                    bbox = img.getbbox()
                    if bbox:
                        img = img.crop(bbox)
                    max_w = max(450, int(self.winfo_width() * 0.85))
                    if img.width > max_w:
                        scale = max_w / img.width
                        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self._img_refs.append(photo)
                    if inline:
                        self.textview.image_create("end", image=photo, align="baseline")
                    else:
                        self.textview.insert("end", "\n", ("tight",))
                        self.textview.image_create("end", image=photo, align="center")
                        self.textview.insert("end", "\n", ("tight",))
                except Exception as e:
                    self._log(f"[latex] render error (block): {e} — raw fallback")
                    self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))
            self.textview.insert("end", "\n")
            self._prepare_word_spans()

            # Auto-scroll to bottom
            self.textview.see("end")

        except Exception as e:
            self._log(f"[latex] append error: {e} — plain text fallback")
            self.textview.insert("end", text, ("normal", "tight"))

    def _block_keys(self, e):
        """Block most keys except copy/select all"""
        if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
            return None
        return "break"

    def _select_all(self, _):
        """Select all text"""
        self.textview.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _popup_menu(self, event):
        """Show right-click context menu"""
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    # ==================== LATEX RENDERING METHODS ====================

    def _is_inline_math(self, expr: str) -> bool:
        """Check if math expression should be rendered inline"""
        s = expr.strip()
        if "\n" in s: return False
        if re.search(r"\\begin\{.*?\}", s): return False
        if re.search(r"\\(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases)\b", s): return False
        if len(s) > 80: return False
        return True

    def _needs_latex_engine(self, s: str) -> bool:
        """Check if expression requires full LaTeX engine"""
        return bool(re.search(
            r"(\\begin\{(?:bmatrix|pmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases|smallmatrix)\})"
            r"|\\boxed\s*\(" r"|\\boxed\s*\{"
            r"|\\text\s*\{"  r"|\\overset\s*\{" r"|\\underset\s*\{",
            s, flags=re.IGNORECASE
        ))

    def _probe_usetex(self):
        """Check if LaTeX engine is available"""
        if self._usetex_checked:
            return
        self._usetex_checked = True
        try:
            _ = self._render_with_engine_dark(r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 10, 100, use_usetex=True)
            self._usetex_available = True
            self._log("[latex] usetex available")
        except Exception as e:
            self._usetex_available = False
            self._log(f"[latex] usetex not available ({e}); fallback to MathText)")

    def render_png_bytes(self, latex, fontsize=None, dpi=200):
        """Render LaTeX to PNG bytes (ALWAYS WHITE ON TRANSPARENT)"""
        fontsize = fontsize or self.math_pt
        expr = latex.strip()
        needs_tex = self._needs_latex_engine(expr)
        if needs_tex and not self._usetex_checked:
            self._probe_usetex()
        prefer_usetex = self._usetex_available and (
                needs_tex or "\\begin{pmatrix" in expr or "\\frac" in expr or "\\sqrt" in expr)
        expr = expr.replace("\n", " ")
        try:
            return self._render_with_engine_dark(expr, fontsize, dpi, use_usetex=prefer_usetex)
        except Exception:
            return self._render_with_engine_dark(expr, fontsize, dpi, use_usetex=False)

    def _render_with_engine_dark(self, latex: str, fontsize: int, dpi: int, use_usetex: bool):
        """Render LaTeX with white text on transparent background"""
        preamble = r"\usepackage{amsmath,amssymb,bm}"
        rc = {'text.usetex': bool(use_usetex)}

        # === PERMANENT DARK MODE: Always white text ===
        rc.update({
            'text.color': 'white',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        })

        if use_usetex:
            preamble += r"\usepackage{xcolor} \definecolor{textcolor}{RGB}{255,255,255} \color{textcolor}"
            rc['text.latex.preamble'] = preamble
            text_color = "white"
        else:
            text_color = "white"

        fig = plt.figure(figsize=(1, 1), dpi=dpi, facecolor='none')
        try:
            with matplotlib.rc_context(rc):
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.text(0.5, 0.5, f"${latex}$", ha="center", va="center",
                        fontsize=fontsize, color=text_color)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                            transparent=True, facecolor='none', edgecolor='none')
                return buf.getvalue()
        finally:
            plt.close(fig)
            plt.close('all')

    def split_text_math(self, text):
        """Split text into math and non-math blocks"""
        if not text:
            return []
        pattern = re.compile(
            r"""
            ```(?:math|latex)\s+(.+?)```   |   # fenced code block
            \\\[(.+?)\\\]                  |   # \[ ... \]
            \$\$(.+?)\$\$                  |   # $$ ... $$
            \\\((.+?)\\\)                  |   # \( ... \)
            \$(.+?)\$                          # $ ... $
            """,
            flags=re.DOTALL | re.IGNORECASE | re.VERBOSE
        )
        out, idx = [], 0
        for m in pattern.finditer(text):
            s, e = m.span()
            if s > idx:
                out.append(("text", text[idx:s]))
            latex_expr = next(g for g in m.groups() if g is not None)
            out.append(("math", latex_expr.strip()))
            idx = e
        if idx < len(text):
            out.append(("text", text[idx:]))
        return out

    def show_document(self, text, wrap=900):
        """Main method to display LaTeX content"""
        self._last_text = text or ""
        self.clear()
        if not text:
            return
        try:
            blocks = self.split_text_math(text)
        except Exception as e:
            self._log(f"[latex] split error: {e} — plain text")
            self.textview.insert("end", text, ("normal", "tight"))
            return

        raw_mode = bool(self.show_raw.get())
        for kind, content in blocks:
            if kind == "text":
                self.textview.insert("end", content, ("normal", "tight"))
                continue
            if raw_mode:
                self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))
                continue
            try:
                inline = self._is_inline_math(content)
                fsz = max(6, self.math_pt - 2) if inline else self.math_pt
                png = self.render_png_bytes(content, fontsize=fsz)
                img = Image.open(BytesIO(png)).convert("RGBA")
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                max_w = max(450, int(self.winfo_width() * 0.85))
                if img.width > max_w:
                    scale = max_w / img.width
                    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._img_refs.append(photo)
                if inline:
                    self.textview.image_create("end", image=photo, align="baseline")
                else:
                    self.textview.insert("end", "\n", ("tight",))
                    self.textview.image_create("end", image=photo, align="center")
                    self.textview.insert("end", "\n", ("tight",))
            except Exception as e:
                self._log(f"[latex] render error (block): {e} — raw fallback")
                self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))
        self.textview.insert("end", "\n")
        self._prepare_word_spans()

    # ==================== HIGHLIGHT METHODS ====================

    def _word_spans(self):
        """Get word positions for TTS highlighting"""
        content = self.textview.get("1.0", "end-1c")
        spans = []
        for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", content):
            s, e = m.span()
            spans.append((f"1.0+{s}c", f"1.0+{e}c"))
        return spans

    def _prepare_word_spans(self):
        """Prepare word spans for TTS highlighting"""
        try:
            self._hi_spans = self._word_spans()
            self._hi_n = len(self._hi_spans)
        except Exception:
            self._hi_spans, self._hi_n = [], 0

    def set_highlight_index(self, i: int):
        """Set highlight to specific word index"""
        if not getattr(self, "_hi_spans", None):
            return
        i = max(0, min(i, self._hi_n - 1))
        s, e = self._hi_spans[i]
        self.textview.tag_remove("speak", "1.0", "end")
        self.textview.tag_add("speak", s, e)
        self.textview.see(s)

    def set_highlight_ratio(self, r: float):
        """Set highlight based on ratio (0.0 to 1.0)"""
        if not getattr(self, "_hi_spans", None):
            return
        if r <= 0:
            idx = 0
        elif r >= 1:
            idx = self._hi_n - 1
        else:
            idx = int(r * self._hi_n)
        self.set_highlight_index(idx)

    def clear_highlight(self):
        """Clear all highlights"""
        self.textview.tag_remove("speak", "1.0", "end")

    # ==================== DIAGNOSTIC METHOD ====================

    def _run_latex_diagnostics(self):
        """Run LaTeX system diagnostics"""
        try:
            import shutil, platform, subprocess
            from matplotlib import __version__ as mpl_ver
            self._log(f"[diag] Matplotlib {mpl_ver} on {platform.system()} {platform.release()}")
            for tool in ("latex", "pdflatex", "dvipng"):
                path = shutil.which(tool)
                self._log(f"[diag] which {tool}: {path or '(not found)'}")
            gs_path = (
                    shutil.which("gswin64c") or shutil.which("gswin32c") or
                    shutil.which("gs") or shutil.which("ghostscript")
            )
            if gs_path:
                self._log(f"[diag] Ghostscript found: {gs_path}")
                try:
                    out = subprocess.check_output([gs_path, "--version"], text=True, stderr=subprocess.STDOUT)
                    self._log(f"[diag] Ghostscript version: {out.strip()}")
                except Exception as e:
                    self._log(f"[diag] (warning) Ghostscript version query failed: {e}")
            else:
                self._log("[diag] Ghostscript not found")
            self._usetex_checked = False
            self._probe_usetex()
        except Exception as e:
            self._log(f"[diag] diagnostics failed: {e}")

