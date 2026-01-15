# echo_engine.py
import numpy as np
import soundfile as sf
import os
import tkinter as tk
from tkinter import ttk

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

class EchoEngine:
    """
    Ultra-light 'sci-fi echo': y[n] = dry*x[n] + wet*(x[n] + fb*y[n-D])
    Controls:
      - delay_ms (echo spacing)
      - intensity (maps to feedback & wet)
    """

    def __init__(self):
        self.enabled = False
        self.delay_ms = 144.0
        self.intensity = 0.47  # 0..1
        self.dry = 0.70

    def _coeffs(self):
        # Map intensity -> (feedback, wet) smoothly but safely stable
        fb = min(0.90, max(0.05, 0.15 + 0.8 * self.intensity))
        wet = min(0.95, max(0.05, 0.2 + 0.7 * self.intensity))
        return fb, wet

    def process_array(self, x, sr):
        if not self.enabled:
            return x.astype(np.float32)

        D = max(1, int(sr * self.delay_ms / 1000.0))
        fb, wet = self._coeffs()
        dry = float(self.dry)

        y = np.zeros_like(x, dtype=np.float32)
        # simple IIR echo; use integer delay for speed
        for n in range(len(x)):
            x_n = float(x[n])
            y_n = x_n
            if n - D >= 0:
                y_n += fb * y[n - D]
            y[n] = dry * x_n + wet * y_n

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            y = y / peak
        return y.astype(np.float32)

    def process_file(self, in_wav, out_wav):
        print(f"[ECHO DEBUG] Processing file: {in_wav} -> {out_wav}")
        print(f"[ECHO DEBUG] Echo enabled: {self.enabled}, delay: {self.delay_ms}, intensity: {self.intensity}")
        x, sr = _read_wav_mono(in_wav)
        y = self.process_array(x, sr)
        _write_wav(out_wav, y, sr)
        print(f"[ECHO DEBUG] File processing complete")
        return out_wav, sr

class EchoWindow(tk.Toplevel):
    """Tiny control panel: Enable, Delay (ms), Intensity."""

    def __init__(self, master, engine: EchoEngine):
        super().__init__(master)
        self.engine = engine
        self.title("Sci-Fi Echo")
        self.geometry("360x180")
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.build_ui()
        self.withdraw()

    def _slider(self, parent, text, vmin, vmax, var, fmt="{:.1f}"):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=16).pack(side="left")
        s = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal", variable=var)
        s.pack(side="left", fill="x", expand=True, padx=8)
        lab = ttk.Label(row, width=8)
        lab.pack(side="right")

        def update(*_):
            lab.config(text=fmt.format(var.get()))

        var.trace_add("write", lambda *_: update())
        update()
        return s

    def build_ui(self):
        e = self.engine
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        self.v_enabled = tk.BooleanVar(value=e.enabled)
        ttk.Checkbutton(wrap, text="Enable Echo", variable=self.v_enabled, command=self._apply).pack(anchor="w")

        self.v_delay = tk.DoubleVar(value=e.delay_ms)
        self._slider(wrap, "Delay (ms)", 60.0, 480.0, self.v_delay, "{:.0f}")

        self.v_inten = tk.DoubleVar(value=e.intensity)
        self._slider(wrap, "Intensity", 0.0, 1.0, self.v_inten, "{:.2f}")

        btns = ttk.Frame(wrap)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="left")
        ttk.Button(btns, text="Hide", command=self.withdraw).pack(side="right")

        for v in (self.v_delay, self.v_inten):
            v.trace_add("write", lambda *_: self._apply())

    def _apply(self):
        e = self.engine
        e.enabled = bool(self.v_enabled.get())
        e.delay_ms = float(self.v_delay.get())
        e.intensity = float(self.v_inten.get())