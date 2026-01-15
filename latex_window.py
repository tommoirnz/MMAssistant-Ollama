# latex_window.py - IMPROVED VISUAL STYLE
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
    def __init__(self, master, log_fn=None, text_family="Segoe UI", text_size=12, math_pt=14):
        super().__init__(master)
        self.title("LaTeX Preview")
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.geometry("850x650")
        self._log = log_fn or (lambda msg: None)

        # === IMPROVED DARK THEME COLORS (matching code_window) ===
        self.colors = {
            'bg_dark': "#1e1e1e",  # Main background (softer than pure black)
            'bg_darker': "#0c0c0c",  # Text area background
            'bg_toolbar': "#2b2b2b",  # Toolbar background
            'fg_primary': "#d4d4d4",  # Primary text
            'fg_secondary': "#808080",  # Secondary/dimmed text
            'accent': "#569cd6",  # Blue accent (cursor, highlights)
            'accent_dim': "#264f78",  # Dimmed accent (selection)
            'success': "#50c878",  # Green for success
            'border': "#3c3c3c",  # Border color
            'button_bg': "#3c3c3c",  # Button background
            'button_hover': "#4a4a4a",  # Button hover
        }

        # Configure window background
        self.configure(bg=self.colors['bg_dark'])

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

        # Configure ttk styles
        self._setup_styles()

        # Build UI
        self._create_ui()

        self.withdraw()

    def _setup_styles(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style(self)

        # Try to use clam theme as base (better for custom styling)
        try:
            style.theme_use('clam')
        except:
            pass

        # Frame styles
        style.configure(
            "Dark.TFrame",
            background=self.colors['bg_dark']
        )
        style.configure(
            "Toolbar.TFrame",
            background=self.colors['bg_toolbar']
        )

        # Label styles
        style.configure(
            "Dark.TLabel",
            background=self.colors['bg_toolbar'],
            foreground=self.colors['fg_primary'],
            font=("Segoe UI", 9)
        )
        style.configure(
            "Status.TLabel",
            background=self.colors['bg_dark'],
            foreground=self.colors['success'],
            font=("Segoe UI", 9)
        )

        # Button styles
        style.configure(
            "Dark.TButton",
            background=self.colors['button_bg'],
            foreground=self.colors['fg_primary'],
            borderwidth=0,
            focuscolor=self.colors['accent'],
            padding=(10, 5),
            font=("Segoe UI", 9)
        )
        style.map(
            "Dark.TButton",
            background=[('active', self.colors['button_hover']),
                        ('pressed', self.colors['accent_dim'])],
            foreground=[('active', self.colors['fg_primary'])]
        )

        # Checkbutton styles
        style.configure(
            "Dark.TCheckbutton",
            background=self.colors['bg_toolbar'],
            foreground=self.colors['fg_primary'],
            font=("Segoe UI", 9)
        )
        style.map(
            "Dark.TCheckbutton",
            background=[('active', self.colors['bg_toolbar'])]
        )

        # Spinbox styles
        style.configure(
            "Dark.TSpinbox",
            background=self.colors['bg_darker'],
            foreground=self.colors['fg_primary'],
            fieldbackground=self.colors['bg_darker'],
            arrowcolor=self.colors['fg_primary'],
            borderwidth=1,
            padding=2
        )

        # LabelFrame styles
        style.configure(
            "Dark.TLabelframe",
            background=self.colors['bg_dark'],
            foreground=self.colors['fg_primary'],
            borderwidth=1,
            relief="solid"
        )
        style.configure(
            "Dark.TLabelframe.Label",
            background=self.colors['bg_dark'],
            foreground=self.colors['accent'],
            font=("Segoe UI", 10, "bold")
        )

        # Separator
        style.configure(
            "Dark.TSeparator",
            background=self.colors['border']
        )

    def _create_ui(self):
        """Create the improved UI"""
        # Main container
        main_frame = ttk.Frame(self, style="Dark.TFrame")
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # === TOOLBAR ===
        toolbar = ttk.Frame(main_frame, style="Toolbar.TFrame")
        toolbar.pack(fill="x", pady=(0, 8))

        # Add padding inside toolbar
        toolbar_inner = ttk.Frame(toolbar, style="Toolbar.TFrame")
        toolbar_inner.pack(fill="x", padx=8, pady=6)

        # Left side controls
        left_controls = ttk.Frame(toolbar_inner, style="Toolbar.TFrame")
        left_controls.pack(side="left", fill="x")

        # Raw LaTeX toggle
        ttk.Checkbutton(
            left_controls,
            text="Show Raw",
            variable=self.show_raw,
            style="Dark.TCheckbutton",
            command=lambda: self.show_document(self._last_text or "")
        ).pack(side="left", padx=(0, 8))

        # Copy button
        ttk.Button(
            left_controls,
            text="📋 Copy LaTeX",
            style="Dark.TButton",
            command=self.copy_raw_latex,
            width=12
        ).pack(side="left", padx=(0, 6))

        # Diagnostics button
        ttk.Button(
            left_controls,
            text="🔧 Diagnostics",
            style="Dark.TButton",
            command=self._run_latex_diagnostics,
            width=12
        ).pack(side="left", padx=(0, 6))

        # Separator
        ttk.Separator(left_controls, orient="vertical").pack(side="left", fill="y", padx=10)

        # Font size controls
        ttk.Label(left_controls, text="Text:", style="Dark.TLabel").pack(side="left", padx=(0, 4))
        self.text_pt_var = tk.IntVar(value=self.text_size)
        text_spin = ttk.Spinbox(
            left_controls,
            from_=8, to=48, width=4,
            textvariable=self.text_pt_var,
            style="Dark.TSpinbox",
            command=lambda: self.set_text_font(size=self.text_pt_var.get())
        )
        text_spin.pack(side="left", padx=(0, 10))

        ttk.Label(left_controls, text="Math:", style="Dark.TLabel").pack(side="left", padx=(0, 4))
        self.math_pt_var = tk.IntVar(value=self.math_pt)
        math_spin = ttk.Spinbox(
            left_controls,
            from_=8, to=64, width=4,
            textvariable=self.math_pt_var,
            style="Dark.TSpinbox",
            command=lambda: self.set_math_pt(self.math_pt_var.get())
        )
        math_spin.pack(side="left")

        # Right side controls
        right_controls = ttk.Frame(toolbar_inner, style="Toolbar.TFrame")
        right_controls.pack(side="right")

        # Window size buttons
        ttk.Button(
            right_controls,
            text="➖",
            style="Dark.TButton",
            command=self._decrease_size,
            width=3
        ).pack(side="left", padx=(0, 4))

        ttk.Button(
            right_controls,
            text="➕",
            style="Dark.TButton",
            command=self._increase_size,
            width=3
        ).pack(side="left", padx=(0, 8))

        # Close button
        ttk.Button(
            right_controls,
            text="✕ Close",
            style="Dark.TButton",
            command=self.hide,
            width=8
        ).pack(side="left")

        # Bind Enter key to spinboxes
        text_spin.bind("<Return>", lambda _: self.set_text_font(size=self.text_pt_var.get()))
        math_spin.bind("<Return>", lambda _: self.set_math_pt(self.math_pt_var.get()))

        # === CONTENT AREA ===
        content_frame = ttk.LabelFrame(
            main_frame,
            text=" LaTeX Content ",
            style="Dark.TLabelframe",
            padding=5
        )
        content_frame.pack(fill="both", expand=True)

        # Text widget with scrollbar
        text_container = ttk.Frame(content_frame, style="Dark.TFrame")
        text_container.pack(fill="both", expand=True)

        # Main text widget
        self.textview = tk.Text(
            text_container,
            bg=self.colors['bg_darker'],
            fg=self.colors['fg_primary'],
            wrap="word",
            undo=False,
            insertbackground=self.colors['accent'],
            selectbackground=self.colors['accent_dim'],
            inactiveselectbackground=self.colors['accent_dim'],
            borderwidth=0,
            relief="flat",
            padx=12,
            pady=10,
            spacing1=2,  # Space above lines
            spacing2=2,  # Space between wrapped lines
            spacing3=4,  # Space below lines
        )

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.textview.yview)
        self.textview.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.textview.pack(side="left", fill="both", expand=True)

        self.textview.configure(font=self._text_font, state="normal")

        # === STATUS BAR ===
        status_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        status_frame.pack(fill="x", pady=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel"
        ).pack(side="left")

        # LaTeX engine status indicator
        self.engine_var = tk.StringVar(value="")
        ttk.Label(
            status_frame,
            textvariable=self.engine_var,
            style="Dark.TLabel"
        ).pack(side="right")

        # === TEXT BINDINGS ===
        self.textview.bind("<Key>", self._block_keys)
        self.textview.bind("<<Paste>>", lambda e: "break")
        self.textview.bind("<Control-v>", lambda e: "break")
        self.textview.bind("<Control-x>", lambda e: "break")
        self.textview.bind("<Control-c>", lambda e: None)
        self.textview.bind("<Control-a>", self._select_all)

        # === CONTEXT MENU ===
        self._menu = tk.Menu(
            self,
            tearoff=0,
            bg=self.colors['bg_toolbar'],
            fg=self.colors['fg_primary'],
            activebackground=self.colors['accent_dim'],
            activeforeground=self.colors['fg_primary'],
            borderwidth=1,
            relief="solid"
        )
        self._menu.add_command(label="📋 Copy", command=lambda: self.textview.event_generate("<<Copy>>"))
        self._menu.add_command(label="📄 Select All", command=lambda: self._select_all(None))
        self._menu.add_separator()
        self._menu.add_command(label="📝 Copy Raw LaTeX", command=self.copy_raw_latex)
        self.textview.bind("<Button-3>", self._popup_menu)
        self.textview.bind("<Button-2>", self._popup_menu)

        # === HIGHLIGHT TAGS ===
        self.textview.tag_configure(
            "speak",
            background=self.colors['accent'],
            foreground=self.colors['bg_darker']
        )
        self.textview.tag_configure("normal", background="")
        self.textview.tag_configure("tight", spacing1=1, spacing3=1)

    # ==================== ESSENTIAL WINDOW METHODS ====================

    def copy_raw_latex(self):
        """Copy raw LaTeX source to clipboard with proper document structure"""
        try:
            content = self._last_text or ""
            if content.strip():
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
                self.status_var.set("LaTeX copied to clipboard")
                self._log("[latex] Complete LaTeX document copied to clipboard")
            else:
                self.status_var.set("No content to copy")
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
            self.status_var.set(f"Text size: {self.text_size}pt")
        except Exception as e:
            self._log(f"[latex] set_text_font error: {e}")

    def set_math_pt(self, pt: int):
        """Set math point size"""
        try:
            self.math_pt = int(pt)
            self.status_var.set(f"Math size: {self.math_pt}pt")
            # Re-render if we have content
            if self._last_text:
                self.show_document(self._last_text)
        except Exception as e:
            self._log(f"[latex] set_math_pt error: {e}")

    # ==================== SCHEME METHOD ====================

    def set_scheme(self, scheme: str):
        """Simplified scheme method - keeps dark theme"""
        try:
            if scheme == "vision":
                self._log("[latex] Vision mode - using dark theme")
            else:
                self._log("[latex] Default mode - dark theme")
        except Exception as e:
            self._log(f"[latex] set_scheme error: {e}")

    # ==================== WINDOW SIZE CONTROL ====================

    def _increase_size(self):
        """Increase window size"""
        width, height = self._get_current_size()
        new_width = min(2000, width + 100)
        new_height = min(1200, height + 80)
        self.geometry(f"{new_width}x{new_height}")
        self.status_var.set(f"Window: {new_width}x{new_height}")

    def _decrease_size(self):
        """Decrease window size"""
        width, height = self._get_current_size()
        new_width = max(400, width - 100)
        new_height = max(300, height - 80)
        self.geometry(f"{new_width}x{new_height}")
        self.status_var.set(f"Window: {new_width}x{new_height}")

    def _get_current_size(self):
        """Get current window size"""
        geometry = self.geometry()
        if 'x' in geometry and '+' in geometry:
            size_part = geometry.split('+')[0]
            width, height = map(int, size_part.split('x'))
            return width, height
        return 850, 650

    # ==================== UI HELPER METHODS ====================

    def append_document(self, text, wrap=900, separator="\n" + "─" * 50 + "\n"):
        """Append content to existing document"""
        if not text:
            return

        if self._last_text:
            self._last_text += separator + text
        else:
            self._last_text = text

        try:
            current_content = self.textview.get("1.0", "end-1c")
            if current_content.strip():
                self.textview.insert("end", separator)

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
                    self._log(f"[latex] render error: {e}")
                    self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))

            self.textview.insert("end", "\n")
            self._prepare_word_spans()
            self.textview.see("end")
            self.status_var.set("Content appended")

        except Exception as e:
            self._log(f"[latex] append error: {e}")
            self.textview.insert("end", text, ("normal", "tight"))

    def _block_keys(self, e):
        """Block keys except copy/select"""
        if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
            return None
        return "break"

    def _select_all(self, _):
        """Select all text"""
        self.textview.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _popup_menu(self, event):
        """Show context menu"""
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    # ==================== LATEX RENDERING ====================

    def _is_inline_math(self, expr: str) -> bool:
        """Check if expression is inline"""
        s = expr.strip()
        if "\n" in s: return False
        if re.search(r"\\begin\{.*?\}", s): return False
        if re.search(r"\\(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases)\b", s): return False
        if len(s) > 80: return False
        return True

    def _needs_latex_engine(self, s: str) -> bool:
        """Check if full LaTeX engine needed"""
        return bool(re.search(
            r"(\\begin\{(?:bmatrix|pmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases|smallmatrix)\})"
            r"|\\boxed\s*\(" r"|\\boxed\s*\{"
            r"|\\text\s*\{" r"|\\overset\s*\{" r"|\\underset\s*\{",
            s, flags=re.IGNORECASE
        ))

    def _probe_usetex(self):
        """Check LaTeX engine availability"""
        if self._usetex_checked:
            return
        self._usetex_checked = True
        try:
            _ = self._render_with_engine(r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 10, 100, use_usetex=True)
            self._usetex_available = True
            self.engine_var.set("LaTeX: ✓")
            self._log("[latex] usetex available")
        except Exception as e:
            self._usetex_available = False
            self.engine_var.set("LaTeX: MathText")
            self._log(f"[latex] usetex not available ({e})")

    def render_png_bytes(self, latex, fontsize=None, dpi=200):
        """Render LaTeX to PNG"""
        fontsize = fontsize or self.math_pt
        expr = latex.strip()
        needs_tex = self._needs_latex_engine(expr)
        if needs_tex and not self._usetex_checked:
            self._probe_usetex()
        prefer_usetex = self._usetex_available and (
                needs_tex or "\\begin{pmatrix" in expr or "\\frac" in expr or "\\sqrt" in expr)
        expr = expr.replace("\n", " ")
        try:
            return self._render_with_engine(expr, fontsize, dpi, use_usetex=prefer_usetex)
        except Exception:
            return self._render_with_engine(expr, fontsize, dpi, use_usetex=False)

    def _render_with_engine(self, latex: str, fontsize: int, dpi: int, use_usetex: bool):
        """Render LaTeX with white text on transparent background"""
        preamble = r"\usepackage{amsmath,amssymb,bm}"
        rc = {
            'text.usetex': bool(use_usetex),
            'text.color': 'white',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        }

        if use_usetex:
            preamble += r"\usepackage{xcolor} \definecolor{textcolor}{RGB}{255,255,255} \color{textcolor}"
            rc['text.latex.preamble'] = preamble

        fig = plt.figure(figsize=(1, 1), dpi=dpi, facecolor='none')
        try:
            with matplotlib.rc_context(rc):
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.text(0.5, 0.5, f"${latex}$", ha="center", va="center",
                        fontsize=fontsize, color="white")
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
            ```(?:math|latex)\s+(.+?)```   |
            \\\[(.+?)\\\]                  |
            \$\$(.+?)\$\$                  |
            \\\((.+?)\\\)                  |
            \$(.+?)\$
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
        """Display LaTeX content"""
        self._last_text = text or ""
        self.clear()
        if not text:
            return

        self.status_var.set("Rendering...")
        self.update_idletasks()

        try:
            blocks = self.split_text_math(text)
        except Exception as e:
            self._log(f"[latex] split error: {e}")
            self.textview.insert("end", text, ("normal", "tight"))
            return

        raw_mode = bool(self.show_raw.get())
        math_count = 0

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
                math_count += 1
            except Exception as e:
                self._log(f"[latex] render error: {e}")
                self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))

        self.textview.insert("end", "\n")
        self._prepare_word_spans()
        self.status_var.set(f"Rendered {math_count} equations")

    # ==================== HIGHLIGHT METHODS ====================

    def _word_spans(self):
        """Get word positions"""
        content = self.textview.get("1.0", "end-1c")
        spans = []
        for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", content):
            s, e = m.span()
            spans.append((f"1.0+{s}c", f"1.0+{e}c"))
        return spans

    def _prepare_word_spans(self):
        """Prepare word spans for highlighting"""
        try:
            self._hi_spans = self._word_spans()
            self._hi_n = len(self._hi_spans)
        except Exception:
            self._hi_spans, self._hi_n = [], 0

    def set_highlight_index(self, i: int):
        """Set highlight to word index"""
        if not getattr(self, "_hi_spans", None):
            return
        i = max(0, min(i, self._hi_n - 1))
        s, e = self._hi_spans[i]
        self.textview.tag_remove("speak", "1.0", "end")
        self.textview.tag_add("speak", s, e)
        self.textview.see(s)

    def set_highlight_ratio(self, r: float):
        """Set highlight by ratio (0.0 to 1.0)"""
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
        """Clear highlights"""
        self.textview.tag_remove("speak", "1.0", "end")

    # ==================== DIAGNOSTICS ====================

    def _run_latex_diagnostics(self):
        """Run LaTeX diagnostics"""
        self.status_var.set("Running diagnostics...")
        try:
            import shutil, platform, subprocess
            from matplotlib import __version__ as mpl_ver

            self._log(f"[diag] Matplotlib {mpl_ver} on {platform.system()} {platform.release()}")

            for tool in ("latex", "pdflatex", "dvipng"):
                path = shutil.which(tool)
                self._log(f"[diag] {tool}: {path or 'not found'}")

            gs_path = (
                    shutil.which("gswin64c") or shutil.which("gswin32c") or
                    shutil.which("gs") or shutil.which("ghostscript")
            )
            if gs_path:
                self._log(f"[diag] Ghostscript: {gs_path}")
                try:
                    out = subprocess.check_output([gs_path, "--version"], text=True, stderr=subprocess.STDOUT)
                    self._log(f"[diag] GS version: {out.strip()}")
                except Exception as e:
                    self._log(f"[diag] GS version failed: {e}")
            else:
                self._log("[diag] Ghostscript: not found")

            self._usetex_checked = False
            self._probe_usetex()
            self.status_var.set("Diagnostics complete - check log")

        except Exception as e:
            self._log(f"[diag] failed: {e}")
            self.status_var.set("Diagnostics failed")
