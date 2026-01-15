import tkinter as tk
import time
import math
import numpy as np
import os
import soundfile as sf
import sounddevice as sd
import threading
import colorsys
from PIL import Image, ImageDraw, ImageTk


# === BASE AVATAR CLASS ===
class BaseAvatarWindow(tk.Toplevel):
    """Base class for all avatars with common functionality"""

    # Common constants
    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"

    # Scale constants
    SCALE_MIN = 0.3
    SCALE_MAX = 2.0
    SCALE_STEP = 0.1

    # Window constants
    BASE_DIAMETER = 480

    def __init__(self, master, title="Avatar"):
        super().__init__(master)
        self.title(title)

        # === WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_diameter = self.BASE_DIAMETER

        # Make window circular/transparent
        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            pass

        self._update_window_size()
        self.center_on_screen()

        # === DRAG & SCALE SETUP ===
        self._drag_data = {"x": 0, "y": 0}
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._do_drag)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)
        self.bind("<Button-5>", self._on_mouse_wheel)

        # Canvas setup
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        # Common state
        self.level = 0
        self._running = True
        self.pad = 8

        # Animation timer
        self._animation_timer = None

    def _on_configure(self, e):
        """Handle canvas resize"""
        self.redraw()

    def _update_window_size(self):
        """Update window size based on current scale factor"""
        current_diameter = int(self._base_diameter * self._scale_factor)
        try:
            current_geometry = self.geometry()
            if '+' in current_geometry:
                parts = current_geometry.split('+')
                if len(parts) == 3:
                    x_pos, y_pos = int(parts[1]), int(parts[2])
                    self.geometry(f"{current_diameter}x{current_diameter}+{x_pos}+{y_pos}")
                else:
                    self.geometry(f"{current_diameter}x{current_diameter}")
            else:
                self.geometry(f"{current_diameter}x{current_diameter}")
        except Exception:
            self.geometry(f"{current_diameter}x{current_diameter}")

    def center_on_screen(self):
        """Center the window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scaling"""
        if event.delta > 0 or event.num == 4:  # Scroll up
            new_scale = min(self.SCALE_MAX, self._scale_factor + self.SCALE_STEP)
        else:  # Scroll down
            new_scale = max(self.SCALE_MIN, self._scale_factor - self.SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()

    def log_scale_change(self):
        """Log scale change - override if needed"""
        class_name = self.__class__.__name__
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")

    def _hsv_to_hex(self, h, s, v):
        """Convert HSV to hex color"""
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        r, g, b = [(v, t, p), (q, v, p), (p, v, t),
                   (p, q, v), (t, p, v), (v, p, q)][i]
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    def _circle_geom(self):
        """Get circle geometry"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2
        r = max(1, min(cw, ch) // 2 - self.pad)
        return cx, cy, r

    # === PUBLIC INTERFACE ===
    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._running = False
        try:
            if self._animation_timer is not None:
                self.after_cancel(self._animation_timer)
        except Exception:
            pass
        super().destroy()

    def set_level(self, level: int):
        """Set the audio level (0 to LEVELS-1)"""
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    def redraw(self):
        """Override this method in subclasses"""
        pass

    def set_scale(self, scale_factor: float):
        """Programmatically set scale factor"""
        self._scale_factor = max(self.SCALE_MIN, min(self.SCALE_MAX, scale_factor))
        self._update_window_size()
        self.log_scale_change()

    def get_scale(self) -> float:
        """Get current scale factor"""
        return self._scale_factor

    def reset_scale(self):
        """Reset to default scale"""
        self._scale_factor = 1.0
        self._update_window_size()
        self.log_scale_change()


# === CONCRETE AVATAR CLASSES ===

class CircleAvatarWindow(BaseAvatarWindow):
    """Circular rings avatar"""

    MAX_RINGS = 32

    def __init__(self, master):
        super().__init__(master, "Avatar - Rings")
        self._t0 = time.perf_counter()
        self.start_animation()

    def start_animation(self):
        """Start the animation timer"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(50, self.start_animation)

    def _ring_color(self, k, rings):
        t = (time.perf_counter() - self._t0) * 0.05
        x = ((k / max(1, rings - 1)) + t) % 1.0
        return self._hsv_to_hex(x, 0.9, 1.0)

    def redraw(self):
        cx, cy, r = self._circle_geom()

        # Draw circular background
        self.canvas.delete("all")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        # Draw rings based on level
        rings = max(1, int((self.level / float(self.LEVELS - 1)) * self.MAX_RINGS + 0.5))
        r_min = max(1, int(r * 0.05))
        r_max = int(r * 0.95)
        r_outer = int(r_min + (r_max - r_min) * (self.level / float(self.LEVELS - 1)))
        stroke = max(2, int(r * 0.02))

        if rings == 1:
            col = self._ring_color(0, 1)
            self.canvas.create_oval(cx - r_min, cy - r_min, cx + r_min, cy + r_min,
                                    fill=col, outline=col)
            return

        for k in range(rings):
            rk = int(r_outer * (1.0 - k / float(rings)))
            if rk <= 1:
                continue
            col = self._ring_color(k, rings)
            self.canvas.create_oval(cx - rk, cy - rk, cx + rk, cy + rk,
                                    outline=col, width=stroke)


class RectAvatarWindow(BaseAvatarWindow):
    """Horizontal rectangles avatar"""

    # Visual parameters
    MAX_PARTICLES = 450
    SPAWN_AT_MAX_LVL = 60
    RECT_MIN_LEN_F = 0.03
    RECT_MAX_LEN_F = 0.22
    RECT_THICK_F = 0.012
    RECT_LIFETIME = 0.9
    DRIFT_PIX_F = 0.01
    LEVEL_DEADZONE = 2
    SPAWN_GAMMA = 1.8
    MIN_SPAWN = 0

    def __init__(self, master):
        super().__init__(master, "Avatar - Horizontal Rectangles")
        self._last_time = time.perf_counter()
        self._particles = []
        self.start_animation()

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def _spawn_count(self):
        if self.level <= self.LEVEL_DEADZONE:
            return 0
        usable = self.LEVELS - 1 - self.LEVEL_DEADZONE
        if usable <= 0:
            return 0
        x = (self.level - self.LEVEL_DEADZONE) / float(usable)
        x = max(0.0, min(1.0, x))
        return int(0.5 + self.MIN_SPAWN +
                   (self.SPAWN_AT_MAX_LVL - self.MIN_SPAWN) * (x ** self.SPAWN_GAMMA))

    def _uniform_point_in_disc(self, cx, cy, r):
        inner_r = max(2, r * 0.85)
        u = np.random.random()
        theta = 2 * np.pi * np.random.random()
        rho = inner_r * np.sqrt(u)
        return int(cx + rho * np.cos(theta)), int(cy + rho * np.sin(theta))

    def _spawn(self, n):
        """Spawn horizontal rectangles only"""
        cx, cy, r = self._circle_geom()
        if r <= 4:
            return

        d = 2 * r
        min_len = max(6, int(d * self.RECT_MIN_LEN_F))
        max_len = max(min_len + 2, int(d * self.RECT_MAX_LEN_F))
        thick = max(3, int(r * self.RECT_THICK_F))
        drift_p = max(1, int(r * self.DRIFT_PIX_F))
        now = time.perf_counter()

        for _ in range(n):
            x0, y0 = self._uniform_point_in_disc(cx, cy, r)

            # Horizontal rectangles only
            L = np.random.randint(min_len, max_len)
            dy = abs(y0 - cy)
            chord_half = int(math.sqrt(max(0, r * r - dy * dy)) * 0.95)
            halfL = min(L // 2, chord_half)
            x1, x2 = x0 - halfL, x0 + halfL
            y1, y2 = y0 - thick // 2, y0 + thick // 2

            vx = np.random.randint(-drift_p, drift_p)
            vy = np.random.randint(-drift_p, drift_p)
            col = self._hsv_to_hex(np.random.random(), 0.95, 1.0)

            self._particles.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "vx": vx, "vy": vy, "birth": now, "life": self.RECT_LIFETIME,
                "color": col, "vertical": False
            })

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

    def redraw(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_time)
        self._last_time = now

        cx, cy, r = self._circle_geom()

        # Clear and draw circular background
        self.canvas.delete("all")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        if self.level <= 0:
            self._particles.clear()
            return

        spawn = self._spawn_count()
        if spawn > 0:
            self._spawn(spawn)

        alive = []
        for p in self._particles:
            age = now - p["birth"]
            if age > p["life"]:
                continue

            # Update position
            p["x1"] += p["vx"] * dt
            p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt
            p["y2"] += p["vy"] * dt

            # Keep within circle bounds
            mx = 0.5 * (p["x1"] + p["x2"])
            my = 0.5 * (p["y1"] + p["y2"])

            # Horizontal rectangle clipping
            dy = my - cy
            max_half_thick = max(0, int(math.sqrt(max(0, r * r - dy * dy))))
            half_thick = max(1, int((p["y2"] - p["y1"]) * 0.5))
            half_thick = min(half_thick, max_half_thick)
            p["y1"], p["y2"] = my - half_thick, my + half_thick

            chord_half = max(0, int(math.sqrt(max(0, r * r - dy * dy)) * 0.95))
            halfL = min(int((p["x2"] - p["x1"]) * 0.5), chord_half)
            p["x1"], p["x2"] = mx - halfL, mx + halfL

            # Fade effect
            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples) - 1, int(t * len(stipples)))
            stipple = stipples[idx]

            self.canvas.create_rectangle(
                int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                fill=p["color"], outline=p["color"],
                stipple=stipple if stipple else None
            )
            alive.append(p)

        self._particles = alive


class RectAvatarWindow2(RectAvatarWindow):
    """Enhanced rectangles avatar with both horizontal and vertical rectangles"""

    # Enhanced parameters
    BASE_DIAMETER = 900
    VERTICAL_PROPORTION = 0.40
    CENTER_PULL = 0.07
    EDGE_INSET_F = 1.00
    SPAWN_RADIUS_F = 0.90

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar — Rectangles 2")

    def _spawn(self, n):
        """Spawn both horizontal and vertical rectangles"""
        cx, cy, r = self._circle_geom()
        if r <= 4:
            return

        d = 2 * r
        min_len_h = max(6, int(d * self.RECT_MIN_LEN_F))
        max_len_h = max(min_len_h + 2, int(d * self.RECT_MAX_LEN_F))
        min_len_v = min_len_h
        max_len_v = max_len_h

        thick = max(3, int(r * self.RECT_THICK_F))
        thick_h = thick
        thick_v = thick

        drift_p = max(1, int(r * self.DRIFT_PIX_F))
        now = time.perf_counter()

        for _ in range(n):
            x0, y0 = self._uniform_point_in_disc(cx, cy, r)
            vertical = (np.random.random() < self.VERTICAL_PROPORTION)

            if vertical:
                L = np.random.randint(min_len_v, max_len_v)
                dx = abs(x0 - cx)
                chord_half = int(math.sqrt(max(0, r * r - dx * dx)) * self.EDGE_INSET_F)
                halfL = min(L // 2, chord_half)
                x1, x2 = x0 - thick_v // 2, x0 + thick_v // 2
                y1, y2 = y0 - halfL, y0 + halfL
            else:
                L = np.random.randint(min_len_h, max_len_h)
                dy = abs(y0 - cy)
                chord_half = int(math.sqrt(max(0, r * r - dy * dy)) * self.EDGE_INSET_F)
                halfL = min(L // 2, chord_half)
                x1, x2 = x0 - halfL, x0 + halfL
                y1, y2 = y0 - thick_h // 2, y0 + thick_h // 2

            vx = np.random.randint(-drift_p, drift_p)
            vy = np.random.randint(-drift_p, drift_p)
            col = self._hsv_to_hex(np.random.random(), 0.95, 1.0)

            self._particles.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "vx": vx, "vy": vy, "birth": now, "life": self.RECT_LIFETIME,
                "color": col, "vertical": bool(vertical)
            })

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

    def redraw(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_time)
        self._last_time = now

        cx, cy, r = self._circle_geom()

        # Clear and draw circular background
        self.canvas.delete("all")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        if self.level <= 0:
            self._particles.clear()
            return

        spawn = self._spawn_count()
        if spawn > 0:
            self._spawn(spawn)

        alive = []
        for p in self._particles:
            age = now - p["birth"]
            if age > p["life"]:
                continue

            # Integrate motion
            p["x1"] += p["vx"] * dt
            p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt
            p["y2"] += p["vy"] * dt

            # Gentle center pull
            if self.CENTER_PULL > 0.0:
                mx = 0.5 * (p["x1"] + p["x2"])
                my = 0.5 * (p["y1"] + p["y2"])
                pull = self.CENTER_PULL * dt
                dx = (cx - mx) * pull
                dy = (cy - my) * pull
                p["x1"] += dx
                p["x2"] += dx
                p["y1"] += dy
                p["y2"] += dy
                mx += dx
                my += dy
            else:
                mx = 0.5 * (p["x1"] + p["x2"])
                my = 0.5 * (p["y1"] + p["y2"])

            # Circle-aware clipping
            if p.get("vertical", False):
                dx = mx - cx
                max_half_thick = max(0, int(math.sqrt(max(0, r * r - dx * dx))))
                half_thick = max(1, int((p["x2"] - p["x1"]) * 0.5))
                half_thick = min(half_thick, max_half_thick)
                p["x1"], p["x2"] = mx - half_thick, mx + half_thick

                chord_half = max(0, int(math.sqrt(max(0, r * r - dx * dx)) * self.EDGE_INSET_F))
                halfL = min(int((p["y2"] - p["y1"]) * 0.5), chord_half)
                p["y1"], p["y2"] = my - halfL, my + halfL
            else:
                dy = my - cy
                max_half_thick = max(0, int(math.sqrt(max(0, r * r - dy * dy))))
                half_thick = max(1, int((p["y2"] - p["y1"]) * 0.5))
                half_thick = min(half_thick, max_half_thick)
                p["y1"], p["y2"] = my - half_thick, my + half_thick

                chord_half = max(0, int(math.sqrt(max(0, r * r - dy * dy)) * self.EDGE_INSET_F))
                halfL = min(int((p["x2"] - p["x1"]) * 0.5), chord_half)
                p["x1"], p["x2"] = mx - halfL, mx + halfL

            # Fade via stipple
            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples) - 1, int(t * len(stipples)))
            stipple = stipples[idx]

            self.canvas.create_rectangle(
                int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                fill=p["color"], outline=p["color"],
                stipple=stipple if stipple else None
            )
            alive.append(p)

        self._particles = alive


class RadialPulseAvatar(BaseAvatarWindow):
    """Minimalist radial pulse avatar"""

    # Visual parameters
    DOT_COLOR = "#FF0000"
    LINE_COLOR = "#FF3333"
    MAX_LINES = 24
    MAX_LINE_LENGTH = 0.8
    PULSE_SMOOTHING = 0.3

    def __init__(self, master):
        super().__init__(master, "Avatar - Radial Pulse")
        self._smoothed_level = 0.0
        self._last_time = time.perf_counter()
        self.start_animation()

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def redraw(self):
        """Draw the radial pulse avatar"""
        now = time.perf_counter()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Smooth level changes
        target_level = self.level / float(self.LEVELS - 1)
        self._smoothed_level += (target_level - self._smoothed_level) * self.PULSE_SMOOTHING

        self.canvas.delete("all")
        cx, cy, r = self._circle_geom()

        # Draw circular background
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        # Calculate pulse intensity
        base_intensity = self._smoothed_level
        pulse_intensity = base_intensity * (0.8 + 0.4 * math.sin(time.perf_counter() * 8))

        # Draw central red dot
        dot_radius = max(2, int(4 + pulse_intensity * 6))
        self.canvas.create_oval(
            cx - dot_radius, cy - dot_radius,
            cx + dot_radius, cy + dot_radius,
            fill=self.DOT_COLOR, outline=self.DOT_COLOR
        )

        # Draw radial lines
        if pulse_intensity > 0.05:
            num_lines = self.MAX_LINES
            max_line_length = min(self.canvas.winfo_width(),
                                  self.canvas.winfo_height()) * self.MAX_LINE_LENGTH * pulse_intensity

            for i in range(num_lines):
                angle = (2 * math.pi * i) / num_lines

                # Add slight randomness to line lengths
                line_variation = 0.7 + 0.6 * math.sin(angle * 3 + time.perf_counter() * 6)
                line_length = max_line_length * line_variation

                # Calculate line end point
                end_x = cx + line_length * math.cos(angle)
                end_y = cy + line_length * math.sin(angle)

                # Line width varies with intensity
                line_width = max(1, int(1 + pulse_intensity * 3))

                # Draw the radial line
                self.canvas.create_line(
                    cx, cy, end_x, end_y,
                    fill=self.LINE_COLOR,
                    width=line_width,
                    capstyle=tk.ROUND
                )


class FaceRadialAvatar(tk.Toplevel):
    """Face avatar with radial lines that respond to audio levels - Egg shaped"""

    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"

    # Scale constants
    SCALE_MIN = 0.3
    SCALE_MAX = 2.0
    SCALE_STEP = 0.1

    # Window constants - ELLIPTICAL
    BASE_HEIGHT = 600  # Vertical is longer for egg shape
    BASE_WIDTH = 480  # Horizontal is shorter

    # Radial line parameters (from RadialPulseAvatar template)
    DOT_COLOR = "#FF0000"
    LINE_COLOR = "#FF3333"
    MAX_LINES = 24
    MAX_LINE_LENGTH = 0.95
    PULSE_SMOOTHING = 0.3

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar - Face (Egg)")

        # === WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_height = self.BASE_HEIGHT
        self._base_width = self.BASE_WIDTH

        # Make window elliptical/transparent
        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            pass

        self._update_window_size()
        self.center_on_screen()

        # === DRAG & SCALE SETUP ===
        self._drag_data = {"x": 0, "y": 0}
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._do_drag)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)
        self.bind("<Button-5>", self._on_mouse_wheel)

        # Canvas setup
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        # Face image setup
        self.level = 0
        self._running = True
        self.pad = 8

        # Load face image
        self.face_image = None
        self.face_photo = None
        self._load_face_image()

        # Animation state (from RadialPulseAvatar)
        self._smoothed_level = 0.0
        self._last_time = time.perf_counter()
        self._animation_timer = None

        self.start_animation()

    def _load_face_image(self):
        """Load the face image from file"""
        try:
            self.face_image = Image.open("face.png")
            self._update_face_display()
        except Exception as e:
            print(f"Error loading face.png: {e}")
            # Create a placeholder if image fails to load
            self.face_image = Image.new('RGB', (100, 150), color='red')  # Taller placeholder

    def _on_configure(self, e):
        """Handle canvas resize"""
        self._update_face_display()
        self.redraw()

    def _update_window_size(self):
        """Update window size based on current scale factor - ELLIPTICAL"""
        current_height = int(self._base_height * self._scale_factor)
        current_width = int(self._base_width * self._scale_factor)

        try:
            current_geometry = self.geometry()
            if '+' in current_geometry:
                parts = current_geometry.split('+')
                if len(parts) == 3:
                    x_pos, y_pos = int(parts[1]), int(parts[2])
                    self.geometry(f"{current_width}x{current_height}+{x_pos}+{y_pos}")
                else:
                    self.geometry(f"{current_width}x{current_height}")
            else:
                self.geometry(f"{current_width}x{current_height}")
        except Exception:
            self.geometry(f"{current_width}x{current_height}")

    def center_on_screen(self):
        """Center the elliptical window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scaling"""
        if event.delta > 0 or event.num == 4:  # Scroll up
            new_scale = min(self.SCALE_MAX, self._scale_factor + self.SCALE_STEP)
        else:  # Scroll down
            new_scale = max(self.SCALE_MIN, self._scale_factor - self.SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()
            self._update_face_display()

    def log_scale_change(self):
        """Log scale change"""
        class_name = self.__class__.__name__
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")

    def _update_face_display(self):
        """Update the face image display based on current window size - FIT TO ELLIPSE"""
        if self.face_image is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # Calculate size for elliptical display - fit within the egg shape
        display_width = cw - (self.pad * 2)
        display_height = ch - (self.pad * 2)

        if display_width > 0 and display_height > 0:
            # Resize image to fit the elliptical bounds (taller than wide)
            resized_image = self.face_image.resize(
                (display_width, display_height),
                Image.Resampling.LANCZOS
            )
            self.face_photo = ImageTk.PhotoImage(resized_image)

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def redraw(self):
        """Draw the face avatar with radial lines - EGG SHAPED"""
        now = time.perf_counter()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Smooth level changes (from RadialPulseAvatar)
        target_level = self.level / float(self.LEVELS - 1)
        self._smoothed_level += (target_level - self._smoothed_level) * self.PULSE_SMOOTHING

        self.canvas.delete("all")

        # Get elliptical geometry instead of circular
        cx, cy, rx, ry = self._ellipse_geom()

        # Draw elliptical background (egg shape)
        self.canvas.create_oval(
            cx - rx, cy - ry,
            cx + rx, cy + ry,
            fill=self.BG, outline=self.BG
        )

        # Draw face image if loaded (fits the elliptical bounds)
        if self.face_photo:
            img_x = cx - self.face_photo.width() // 2
            img_y = cy - self.face_photo.height() // 2
            self.canvas.create_image(img_x, img_y, anchor="nw", image=self.face_photo)

        # Calculate pulse intensity (from RadialPulseAvatar)
        base_intensity = self._smoothed_level
        pulse_intensity = base_intensity * (0.8 + 0.4 * math.sin(time.perf_counter() * 8))

        # Draw central dot (optional - can remove if you don't want it)
        dot_radius = max(2, int(4 + pulse_intensity * 6))
        self.canvas.create_oval(
            cx - dot_radius, cy - dot_radius,
            cx + dot_radius, cy + dot_radius,
            fill=self.DOT_COLOR, outline=self.DOT_COLOR
        )

        # Draw radial lines (from RadialPulseAvatar template) - ADAPTED FOR ELLIPSE
        if pulse_intensity > 0.05:  # Only draw lines when there's significant audio
            num_lines = self.MAX_LINES

            # Calculate max line length based on elliptical bounds
            # Use the average of rx and ry for balanced line lengths, or adjust as needed
            avg_radius = (rx + ry) / 2
            max_line_length = avg_radius * self.MAX_LINE_LENGTH * pulse_intensity

            for i in range(num_lines):
                angle = (2 * math.pi * i) / num_lines

                # Add slight randomness to line lengths for organic feel
                line_variation = 0.7 + 0.6 * math.sin(angle * 3 + time.perf_counter() * 6)
                line_length = max_line_length * line_variation

                # Calculate line end point
                end_x = cx + line_length * math.cos(angle)
                end_y = cy + line_length * math.sin(angle)

                # Line width varies with intensity
                line_width = max(1, int(1 + pulse_intensity * 3))

                # Draw the radial line
                self.canvas.create_line(
                    cx, cy, end_x, end_y,
                    fill=self.LINE_COLOR,
                    width=line_width,
                    capstyle='round'
                )

    def _ellipse_geom(self):
        """Get ellipse geometry (egg shape - taller than wide)"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2

        # Elliptical radii - vertical is longer (egg shape)
        rx = max(1, (cw // 2) - self.pad)  # Horizontal radius
        ry = max(1, (ch // 2) - self.pad)  # Vertical radius (longer)

        return cx, cy, rx, ry

    # === PUBLIC INTERFACE ===
    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._running = False
        try:
            if self._animation_timer is not None:
                self.after_cancel(self._animation_timer)
        except Exception:
            pass
        super().destroy()

    def set_level(self, level: int):
        """Set the audio level (0 to LEVELS-1)"""
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    def set_scale(self, scale_factor: float):
        """Programmatically set scale factor"""
        self._scale_factor = max(self.SCALE_MIN, min(self.SCALE_MAX, scale_factor))
        self._update_window_size()
        self.log_scale_change()
        self._update_face_display()

    def get_scale(self) -> float:
        """Get current scale factor"""
        return self._scale_factor

    def reset_scale(self):
        """Reset to default scale"""
        self._scale_factor = 1.0
        self._update_window_size()
        self.log_scale_change()
        self._update_face_display()


class StringGridAvatar(BaseAvatarWindow):
    """50x50 grid of strings - responds to external level only (like other avatars)"""

    # Override BASE_DIAMETER for larger grid display
    BASE_DIAMETER = 900

    def __init__(self, master):
        super().__init__(master, "Avatar - String Grid")

        # Keep your intensity settings
        self.visual_intensity = 0.5

        # Grid parameters - 50x50 = 2,500 points
        self.grid_size = 50
        self.total_points = self.grid_size * self.grid_size
        self.grid_center = self.grid_size // 2

        # Visualization state (NO audio monitoring)
        self.speech_amplitude = 0
        self.draw_threshold = 0.05

        # Pre-calculate grid positions and properties
        self.grid_positions = []
        self.grid_colors = []
        self.grid_frequencies = []
        self.grid_phases = []

        np.random.seed(42)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # Calculate distance from center (0 at center, 1 at edges)
                dx = (x - self.grid_center) / self.grid_center
                dy = (y - self.grid_center) / self.grid_center
                distance = math.sqrt(dx * dx + dy * dy)

                # Store position
                self.grid_positions.append((x, y, distance))

                # Color based on vertical position (rainbow gradient)
                hue = y / self.grid_size
                saturation = 0.7 + 0.3 * (1.0 - distance)
                value = 0.5 + 0.5 * (1.0 - distance)
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
                self.grid_colors.append(color)

                # Frequency based on position (higher near center)
                freq_base = 1.0 + (1.0 - distance) * 2.0
                freq_variation = np.random.random() * 0.5
                self.grid_frequencies.append(freq_base + freq_variation)

                # Random phase
                self.grid_phases.append(np.random.random() * 2 * math.pi)

        # Start animation (NO audio monitoring)
        self.start_animation()

        print(f"[avatar] StringGrid loaded - {self.total_points} points")

    def set_level(self, level: int):
        """Override to convert level to speech_amplitude using your tuned scaling"""
        super().set_level(level)

        # Convert level (0-31) to speech_amplitude (0.0-1.0) using your scaling
        if level > 0:
            # Apply your draw threshold (0.05 = 5%)
            min_level = int(self.LEVELS * self.draw_threshold)  # ~1.5, round to 2
            if level < 2:  # Below threshold
                self.speech_amplitude = 0
            else:
                # Apply your rms * 15 scaling factor to the level
                # First convert level to a 0-1 value
                level_normalized = level / float(self.LEVELS - 1)

                # Apply your visual_intensity (0.5 = 50% reduction)
                scaled = level_normalized * self.visual_intensity

                # Apply your rms * 15 factor (but max 1.0)
                # Since level_normalized is already 0-1, we just need to adjust intensity
                # Your original: current_rms = min(rms * 15, 1.0)
                # We'll simulate that by boosting the response
                boosted = scaled * 2.0  # Simulates the *15 factor for visualization

                self.speech_amplitude = max(0.0, min(1.0, boosted))
        else:
            self.speech_amplitude = 0

    def calculate_point_height(self, x, y, distance, index):
        """Calculate height for a grid point based on speech amplitude - YOUR TUNED VERSION"""
        if self.speech_amplitude <= 0.001:
            return 0.0

        # Only activate points if amplitude is high enough for their distance
        activation_threshold = distance * 0.8
        if self.speech_amplitude < activation_threshold:
            return 0.0

        # Base height with your tuned settings
        base_height = self.speech_amplitude * (1.0 - distance * 0.7)

        # Your tuned wave patterns
        time_factor = time.time() * 2.0
        wave_x = math.sin(x * 0.3 + time_factor * self.grid_frequencies[index] + self.grid_phases[index]) * 0.2
        wave_y = math.cos(y * 0.3 + time_factor * self.grid_frequencies[index] * 1.3) * 0.15

        # Your tuned noise
        noise = np.random.normal(0, 0.03) * (1.0 - distance)

        total_height = base_height + wave_x + wave_y + noise

        return max(0.0, min(total_height, 1.0))  # Clamp to max 1.0

    def start_animation(self):
        """Start animation loop (NO audio updates)"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(33, self.start_animation)

    def redraw(self):
        """Draw the 50x50 grid - COMPLETELY EMPTY when silent"""
        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 100 or height < 100:
            return

        # Draw circular background (mask)
        cx, cy, r = self._circle_geom()
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        # Check speech amplitude - if 0, show nothing
        if self.speech_amplitude < self.draw_threshold:
            return

        # Calculate cell size
        cell_width = width / self.grid_size
        cell_height = height / self.grid_size

        # Scale number of points with amplitude (your tuned scaling)
        amplitude_ratio = (self.speech_amplitude - self.draw_threshold) / (1.0 - self.draw_threshold)
        amplitude_ratio = max(0.0, min(1.0, amplitude_ratio))
        points_ratio = amplitude_ratio ** 0.7  # Your exponential scaling
        points_to_draw = int(self.total_points * points_ratio)

        if points_to_draw == 0:
            return

        # Draw points (prioritize center points first)
        points_drawn = 0

        # Sort points by distance from center (closest first)
        sorted_indices = np.argsort([dist for _, _, dist in self.grid_positions])

        for idx in sorted_indices[:points_to_draw]:
            grid_x, grid_y, distance = self.grid_positions[idx]

            # Calculate screen position
            screen_x = grid_x * cell_width + cell_width / 2
            screen_y = grid_y * cell_height + cell_height / 2

            # Calculate height (Z position)
            height_val = self.calculate_point_height(grid_x, grid_y, distance, idx)

            if height_val > 0:
                # Calculate visual size based on height and distance from center
                base_size = 2.0  # Your base size
                size = base_size + (height_val * 4) + ((1.0 - distance) * 3)  # Your scaling
                size = max(1, int(size))

                # Calculate visual offset (makes points appear to rise)
                z_offset = height_val * cell_height * 3
                visual_y = screen_y - z_offset

                # Get color
                color = self.grid_colors[idx]

                # Draw point
                self.canvas.create_oval(
                    screen_x - size, visual_y - size,
                    screen_x + size, visual_y + size,
                    fill=color, outline=color, width=0
                )

                points_drawn += 1

        # Draw connecting lines for center region (when amplitude is high)
        if self.speech_amplitude > 0.3:
            self.draw_center_lines(width, height, cell_width, cell_height)

    def draw_center_lines(self, width, height, cell_width, cell_height):
        """Draw connecting lines in center region"""
        center_radius = int(self.grid_size * 0.3 * self.speech_amplitude)

        for y in range(self.grid_center - center_radius, self.grid_center + center_radius, 2):
            for x in range(self.grid_center - center_radius, self.grid_center + center_radius, 2):
                idx = y * self.grid_size + x
                if 0 <= idx < len(self.grid_positions):
                    grid_x, grid_y, distance = self.grid_positions[idx]

                    if distance < 0.3:
                        screen_x = grid_x * cell_width + cell_width / 2
                        screen_y = grid_y * cell_height + cell_height / 2

                        height_val = self.calculate_point_height(grid_x, grid_y, distance, idx)
                        if height_val > 0:
                            z_offset = height_val * cell_height * 3
                            visual_y = screen_y - z_offset

                            # Draw connections to neighbors
                            for dy in [-1, 1]:
                                for dx in [-1, 1]:
                                    ny, nx = y + dy, x + dx
                                    nidx = ny * self.grid_size + nx
                                    if 0 <= nidx < len(self.grid_positions):
                                        ngrid_x, ngrid_y, ndist = self.grid_positions[nidx]
                                        if ndist < 0.3:
                                            nheight = self.calculate_point_height(ngrid_x, ngrid_y, ndist, nidx)
                                            if nheight > 0:
                                                nscreen_x = ngrid_x * cell_width + cell_width / 2
                                                nscreen_y = ngrid_y * cell_height + cell_height / 2
                                                nvisual_y = nscreen_y - nheight * cell_height * 3

                                                # Draw line
                                                line_color = "#FFFFFF"

                                                self.canvas.create_line(
                                                    screen_x, visual_y,
                                                    nscreen_x, nvisual_y,
                                                    fill=line_color, width=1
                                                )

    def destroy(self):
        """Clean up before destroying"""
        # No audio threads to clean up
        super().destroy()



###########################################
# sphere_texture_proper.py - OPTIMIZED TEXTURE MAPPED SPHERE (AUTO-OPEN CONTROLS)
import tkinter as tk
import time
import math
#import numpy as np
import os
import random
import colorsys
from PIL import Image, ImageDraw, ImageEnhance


# === OPTIMIZED TEXTURE MAPPED SPHERE AVATAR ===
class TextureMappedSphere(BaseAvatarWindow):
    """Sphere with proper texture mapping - OPTIMIZED for performance"""

    BASE_DIAMETER = 600

    def __init__(self, master, image_path=None):
        super().__init__(master, "Texture Sphere")

        # === PERFORMANCE OPTIMIZATIONS ===
        # Reduce texture resolution
        self.texture_width = 256  # Reduced from 512
        self.texture_height = 256  # Reduced from 512

        # Reduce point count significantly
        self.max_points = 800  # Maximum points to draw (reduced from 1500)
        self.base_points = 300  # Base points when no audio

        # Optimize redraw frequency
        self.redraw_interval = 0.05  # ~20 FPS (was 0.033 = 30 FPS)

        # Cache frequently used values
        self._cached_geom = None
        self._cached_points = []
        self._cached_colors = []

        # Image texture
        self.texture_image = None
        self.texture_pixels = None

        if image_path and os.path.exists(image_path):
            self.load_texture(image_path)
        else:
            self.create_default_texture()

        # Create a pre-warped version of the image for sphere mapping
        self.create_sphere_texture()

        # 3D rotation - simplified
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.base_rotation_speed = 0.015  # Reduced speed

        # Rotation control
        self.current_rotation_speed = 0.0
        self.rotation_direction_x = 1
        self.rotation_direction_y = 1

        # Performance tracking
        self.last_redraw = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Sphere geometry - generate fewer points
        self.sphere_radius = 0.8
        self.sphere_points = []
        self.generate_optimized_sphere_points(density=800)  # Reduced from 1500

        # Lighting - simplified (DEFINE THESE BEFORE _precalc_lighting)
        self.light_pos = (2, 2, 3)
        self.ambient_light = 0.4  # Increased ambient to reduce calculations
        self.diffuse_strength = 0.6  # Reduced diffuse

        # Pre-calculate lighting factors (NOW AFTER light_pos is defined)
        self._lighting_factors = []
        self._precalc_lighting()

        # Control window
        self.control_window = None

        # Start animation
        self.start_animation()

        # Auto-open texture controls after short delay
        self.after(500, self.show_texture_controls)

        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] Texture Sphere loaded (OPTIMIZED)")
            else:
                print(f"[avatar] Texture Sphere loaded (OPTIMIZED)")
        except:
            print(f"[avatar] Texture Sphere loaded (OPTIMIZED)")

    def _precalc_lighting(self):
        """Pre-calculate lighting factors for sphere points"""
        self._lighting_factors = []

        # Make sure light_pos is defined
        if not hasattr(self, 'light_pos'):
            self.light_pos = (2, 2, 3)
        if not hasattr(self, 'ambient_light'):
            self.ambient_light = 0.4
        if not hasattr(self, 'diffuse_strength'):
            self.diffuse_strength = 0.6

        light_len = math.sqrt(self.light_pos[0] ** 2 + self.light_pos[1] ** 2 + self.light_pos[2] ** 2)
        lx, ly, lz = self.light_pos[0] / light_len, self.light_pos[1] / light_len, self.light_pos[2] / light_len

        for x, y, z, color in self.sphere_points:
            # Normalize
            norm = math.sqrt(x * x + y * y + z * z)
            if norm > 0:
                nx, ny, nz = x / norm, y / norm, z / norm
                # Dot product for diffuse lighting
                diffuse = max(0, nx * lx + ny * ly + nz * lz)
                brightness = self.ambient_light + diffuse * self.diffuse_strength
                self._lighting_factors.append(brightness)
            else:
                self._lighting_factors.append(self.ambient_light)

    def load_texture(self, image_path):
        """Load and prepare image texture"""
        try:
            img = Image.open(image_path)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to lower resolution for performance
            img = img.resize((self.texture_width, self.texture_height), Image.Resampling.NEAREST)  # Faster than LANCZOS

            # Reduce enhancement for performance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)  # Reduced from 1.2
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)  # Reduced from 1.3

            self.texture_image = img
            self.texture_pixels = img.load()

            try:
                if hasattr(self.master, 'logln'):
                    self.master.logln(f"[texture] Loaded (optimized): {image_path}")
                else:
                    print(f"[texture] Loaded (optimized): {image_path}")
            except:
                print(f"[texture] Loaded (optimized): {image_path}")

        except Exception as e:
            try:
                if hasattr(self.master, 'logln'):
                    self.master.logln(f"[texture] Error: {e}")
                else:
                    print(f"[texture] Error: {e}")
            except:
                print(f"[texture] Error: {e}")

            self.create_default_texture()

    def create_default_texture(self):
        """Create a colorful default texture - optimized"""
        img = Image.new('RGB', (self.texture_width, self.texture_height))
        draw = ImageDraw.Draw(img)

        # Create a vibrant rainbow gradient - optimized
        center_x = self.texture_width // 2
        center_y = self.texture_height // 2
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

        # Process in blocks for better performance
        block_size = 4
        for x in range(0, self.texture_width, block_size):
            for y in range(0, self.texture_height, block_size):
                # Calculate angle and distance from center
                dx = x - center_x
                dy = y - center_y
                angle = math.atan2(dy, dx)
                distance = math.sqrt(dx * dx + dy * dy) / max_dist

                # Create rainbow pattern
                hue = (angle / (2 * math.pi)) % 1.0
                saturation = 0.8
                value = 0.9 - distance * 0.3

                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                color = (int(r * 255), int(g * 255), int(b * 255))

                # Fill block with same color for performance
                for bx in range(x, min(x + block_size, self.texture_width)):
                    for by in range(y, min(y + block_size, self.texture_height)):
                        draw.point((bx, by), fill=color)

        self.texture_image = img
        self.texture_pixels = img.load()

    def create_sphere_texture(self):
        """Create a pre-warped version of the texture for sphere mapping - optimized"""
        sphere_img = Image.new('RGB', (self.texture_width, self.texture_height))
        sphere_pixels = sphere_img.load()

        center_x = self.texture_width // 2
        center_y = self.texture_height // 2
        radius = min(center_x, center_y) * 0.9

        # Process in blocks
        block_size = 2
        for x in range(0, self.texture_width, block_size):
            for y in range(0, self.texture_height, block_size):
                # Convert to normalized coordinates (-1 to 1)
                nx = (x - center_x) / radius
                ny = (y - center_y) / radius

                # Check if point is inside circle
                dist = math.sqrt(nx * nx + ny * ny)
                if dist <= 1.0:
                    # Sphere mapping: convert to spherical coordinates
                    theta = math.asin(dist)  # Latitude
                    phi = math.atan2(ny, nx)  # Longitude

                    # Convert to texture coordinates
                    u = (phi + math.pi) / (2 * math.pi)  # 0 to 1
                    v = (theta + math.pi / 2) / math.pi  # 0 to 1

                    # Sample from original texture
                    tx = int(u * (self.texture_width - 1))
                    ty = int(v * (self.texture_height - 1))
                    tx = max(0, min(self.texture_width - 1, tx))
                    ty = max(0, min(self.texture_height - 1, ty))

                    color = self.texture_pixels[tx, ty]

                    # Fill block
                    for bx in range(x, min(x + block_size, self.texture_width)):
                        for by in range(y, min(y + block_size, self.texture_height)):
                            sphere_pixels[bx, by] = color
                else:
                    # Outside sphere - black
                    for bx in range(x, min(x + block_size, self.texture_width)):
                        for by in range(y, min(y + block_size, self.texture_height)):
                            sphere_pixels[bx, by] = (0, 0, 0)

        self.sphere_texture = sphere_img
        self.sphere_pixels = sphere_pixels

    def generate_optimized_sphere_points(self, density=800):
        """Generate uniformly distributed points on sphere - optimized"""
        self.sphere_points = []

        # Simple uniform distribution (faster than Fibonacci)
        num_lat = int(math.sqrt(density) * 0.8)
        num_lon = int(math.sqrt(density) * 1.2)

        for i in range(num_lat):
            # Latitude from -π/2 to π/2
            theta = (i / max(1, num_lat - 1) - 0.5) * math.pi
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            for j in range(num_lon):
                # Longitude from 0 to 2π
                phi = (j / max(1, num_lon - 1)) * 2 * math.pi
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)

                x = cos_phi * cos_theta
                y = sin_theta
                z = sin_phi * cos_theta

                # Get color from texture
                u = (phi + math.pi) / (2 * math.pi)
                v = (theta + math.pi / 2) / math.pi

                tx = int(u * (self.texture_width - 1))
                ty = int(v * (self.texture_height - 1))
                tx = max(0, min(self.texture_width - 1, tx))
                ty = max(0, min(self.texture_height - 1, ty))

                color = self.texture_pixels[tx, ty]

                self.sphere_points.append((x, y, z, color))

        # Store pre-calculated lighting (after sphere_points are created)
        if hasattr(self, 'light_pos'):  # Check if attributes are defined
            self._precalc_lighting()
        else:
            # Set defaults if not already set
            self.light_pos = (2, 2, 3)
            self.ambient_light = 0.4
            self.diffuse_strength = 0.6
            self._precalc_lighting()

    def apply_lighting_optimized(self, color, idx):
        """Apply lighting using pre-calculated factors"""
        # Make sure lighting factors exist
        if not hasattr(self, '_lighting_factors') or idx >= len(self._lighting_factors):
            # Fallback to simple lighting
            brightness = 0.7 + (self.level / 124.0) if self.level > 0 else 0.7
            r = min(255, int(color[0] * brightness))
            g = min(255, int(color[1] * brightness))
            b = min(255, int(color[2] * brightness))
            return (r, g, b)

        brightness = self._lighting_factors[idx]

        # Apply audio-based brightness boost
        if self.level > 0:
            brightness *= (1.0 + self.level / 124.0)  # Reduced effect

        r = min(255, int(color[0] * brightness))
        g = min(255, int(color[1] * brightness))
        b = min(255, int(color[2] * brightness))

        return (r, g, b)

    def set_level(self, level: int):
        """Set the audio level - optimized"""
        super().set_level(level)

        if level > 2:  # Add threshold to avoid micro-rotations
            # Increase rotation speed with audio level
            self.current_rotation_speed = self.base_rotation_speed * (level / 31.0) * 1.5  # Reduced multiplier

            # Initialize random rotation directions if needed
            if not hasattr(self, 'rotation_initialized'):
                self.rotation_initialized = True
                self.rotation_direction_x = random.choice([-1, 1])
                self.rotation_direction_y = random.choice([-1, 1])
        else:
            # Gradually slow down rotation when audio stops
            self.current_rotation_speed = max(0, self.current_rotation_speed - 0.0005)  # Slower decay

    def rotate_point_optimized(self, x, y, z):
        """Optimized rotation - cache trig values"""
        # Cache trig values since they're expensive
        cos_x = math.cos(self.rotation_x)
        sin_x = math.sin(self.rotation_x)
        cos_y = math.cos(self.rotation_y)
        sin_y = math.sin(self.rotation_y)

        # Rotate around X
        y1 = y * cos_x - z * sin_x
        z1 = y * sin_x + z * cos_x

        # Rotate around Y
        x1 = x * cos_y + z1 * sin_y
        z2 = -x * sin_y + z1 * cos_y

        return x1, y1, z2

    def start_animation(self):
        """Start animation loop - optimized"""
        if not self._running:
            return

        # Update rotation if audio is active
        if self.current_rotation_speed > 0:
            self.rotation_x += self.current_rotation_speed * self.rotation_direction_x * 0.6  # Reduced
            self.rotation_y += self.current_rotation_speed * self.rotation_direction_y * 0.9  # Reduced
        else:
            # Very slow idle rotation
            self.rotation_x += 0.0005
            self.rotation_y += 0.0007

        # Redraw if enough time has passed
        current_time = time.perf_counter()
        if current_time - self.last_redraw >= self.redraw_interval:
            self.redraw_optimized()
            self.last_redraw = current_time

            # Track FPS for debugging (optional)
            self.frame_count += 1
            if current_time - self.last_fps_time > 10.0:  # Log every 10 seconds
                fps = self.frame_count / 10.0
                try:
                    if hasattr(self.master, 'logln'):
                        self.master.logln(f"[sphere] FPS: {fps:.1f}")
                    else:
                        print(f"[sphere] FPS: {fps:.1f}")
                except:
                    print(f"[sphere] FPS: {fps:.1f}")
                self.frame_count = 0
                self.last_fps_time = current_time

        self._animation_timer = self.after(40, self.start_animation)  # Reduced from 33ms

    def redraw_optimized(self):
        """Draw textured sphere - heavily optimized"""
        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 100 or height < 100:
            return

        # Get center and radius - cache if unchanged
        cx, cy, r = self._circle_geom()

        # Draw black background
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        # Scale factor
        scale_factor = min(width, height) * 0.45

        # Calculate number of points to draw based on audio level
        if self.level == 0:
            points_to_draw = self.base_points
        else:
            points_to_draw = min(len(self.sphere_points),
                                 self.base_points + int((self.level / 31.0) * (self.max_points - self.base_points)))

        # Use pre-cached point list if available
        if len(self._cached_points) == points_to_draw and self._cached_geom == (cx, cy, r, self.rotation_x,
                                                                                self.rotation_y):
            # Use cached points
            for i, (screen_x, screen_y, size, color_hex) in enumerate(self._cached_points):
                self.canvas.create_oval(
                    screen_x - size, screen_y - size,
                    screen_x + size, screen_y + size,
                    fill=color_hex, outline=color_hex
                )
            return

        # Recalculate points
        self._cached_points = []
        self._cached_geom = (cx, cy, r, self.rotation_x, self.rotation_y)

        # Projection constants
        focal_length = 4.0
        distance = 5.0

        # Use every nth point for better distribution with fewer points
        step = max(1, len(self.sphere_points) // points_to_draw)

        for i in range(0, len(self.sphere_points), step):
            if len(self._cached_points) >= points_to_draw:
                break

            x, y, z, color = self.sphere_points[i]

            # Scale
            x_scaled = x * self.sphere_radius
            y_scaled = y * self.sphere_radius
            z_scaled = z * self.sphere_radius

            # Rotate
            rot_x, rot_y, rot_z = self.rotate_point_optimized(x_scaled, y_scaled, z_scaled)

            # Project to 2D
            if rot_z + distance > 0:
                factor = focal_length / (rot_z + distance)

                # Only draw if not too distorted
                if factor > 0.3:  # Increased threshold
                    screen_x = cx + rot_x * factor * scale_factor
                    screen_y = cy + rot_y * factor * scale_factor

                    # Apply lighting (optimized)
                    lit_color = self.apply_lighting_optimized(color, i)

                    # Calculate point size - simplified
                    point_size = max(1, int(2.0 * factor * (1 + self.level / 155.0)))  # Reduced effect

                    # Store in cache
                    color_hex = f"#{lit_color[0]:02x}{lit_color[1]:02x}{lit_color[2]:02x}"
                    self._cached_points.append((screen_x, screen_y, point_size, color_hex))

                    # Draw point
                    self.canvas.create_oval(
                        screen_x - point_size, screen_y - point_size,
                        screen_x + point_size, screen_y + point_size,
                        fill=color_hex, outline=color_hex
                    )

        # Only draw grid lines at high audio levels
        if self.level > 15:  # Increased threshold
            self.draw_simple_grid(cx, cy, scale_factor)

    def draw_simple_grid(self, cx, cy, scale_factor):
        """Draw simplified grid lines"""
        # Only draw one latitude line
        lat = 0
        points = []
        for i in range(0, 36, 2):  # Draw fewer points
            angle = 2 * math.pi * i / 36
            x = math.cos(angle) * self.sphere_radius * math.sqrt(1 - lat * lat)
            y = lat * self.sphere_radius
            z = math.sin(angle) * self.sphere_radius * math.sqrt(1 - lat * lat)

            rot_x, rot_y, rot_z = self.rotate_point_optimized(x, y, z)

            focal_length = 4.0
            distance = 5.0

            if rot_z + distance > 0:
                factor = focal_length / (rot_z + distance)
                if factor > 0.4:  # Higher threshold
                    screen_x = cx + rot_x * factor * scale_factor
                    screen_y = cy + rot_y * factor * scale_factor
                    points.append((screen_x, screen_y))

        # Draw simplified line
        if len(points) > 5:
            line_color = f"#{int(100 + self.level):02x}{int(100 + self.level):02x}{255:02x}"
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                self.canvas.create_line(x1, y1, x2, y2,
                                        fill=line_color, width=1)

    def show_texture_controls(self):
        """Show texture controls in a separate window - automatically opens"""
        # Check if window already exists
        if hasattr(self, 'control_window') and self.control_window and self.control_window.winfo_exists():
            try:
                self.control_window.lift()
                self.control_window.focus_force()
                return
            except:
                pass

        self.control_window = tk.Toplevel(self)
        self.control_window.title("Texture Sphere Controls")
        self.control_window.geometry("280x180")
        self.control_window.resizable(False, False)

        # Make it stay on top
        self.control_window.transient(self)
        self.control_window.grab_set()

        # Position near the sphere
        self.update_idletasks()
        sphere_x = self.winfo_x()
        sphere_y = self.winfo_y()
        sphere_width = self.winfo_width()
        sphere_height = self.winfo_height()

        # Position to the right of the sphere
        control_x = sphere_x + sphere_width + 10
        control_y = sphere_y

        self.control_window.geometry(f"+{control_x}+{control_y}")

        # Add controls
        tk.Label(self.control_window, text="Texture Controls",
                 font=("Arial", 12, "bold")).pack(pady=10)

        # Simple button layout
        tk.Button(self.control_window, text="🎨 Random Texture",
                  command=self.create_random_texture,
                  width=20, bg="#9C27B0", fg="white",
                  font=("Arial", 10)).pack(pady=5)

        tk.Button(self.control_window, text="🔄 Default Texture",
                  command=lambda: self.update_texture(None),
                  width=20, font=("Arial", 10)).pack(pady=5)

        tk.Button(self.control_window, text="Close",
                  command=self._close_control_window,
                  width=20, font=("Arial", 10)).pack(pady=10)

        # Handle window close
        self.control_window.protocol("WM_DELETE_WINDOW", self._close_control_window)

    def _close_control_window(self):
        """Close the texture control window"""
        if hasattr(self, 'control_window') and self.control_window and self.control_window.winfo_exists():
            try:
                self.control_window.destroy()
            except:
                pass
            self.control_window = None

    def create_random_texture(self):
        """Create and apply a random texture"""
        width, height = self.texture_width, self.texture_height

        # Create random pattern
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)

        # Random pattern type
        pattern_type = random.choice(['gradient', 'stripes', 'dots', 'checkerboard', 'circular'])

        if pattern_type == 'gradient':
            # Random gradient
            color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for x in range(width):
                for y in range(height):
                    ratio = (x + y) / (width + height)
                    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                    draw.point((x, y), fill=(r, g, b))

        elif pattern_type == 'stripes':
            # Stripes
            stripe_width = random.randint(10, 50)
            for x in range(width):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if (x // stripe_width) % 2 == 0:
                    for y in range(height):
                        draw.point((x, y), fill=color)

        elif pattern_type == 'dots':
            # Polka dots
            dot_size = random.randint(5, 20)
            spacing = dot_size * 3
            base_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            dot_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            draw.rectangle([0, 0, width, height], fill=base_color)

            for x in range(0, width, spacing):
                for y in range(0, height, spacing):
                    draw.ellipse([x, y, x + dot_size, y + dot_size], fill=dot_color)

        elif pattern_type == 'circular':
            # Circular gradient
            center_x = width // 2
            center_y = height // 2
            max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

            hue_start = random.random()
            for x in range(width):
                for y in range(height):
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx * dx + dy * dy) / max_dist
                    hue = (hue_start + distance * 0.5) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9 - distance * 0.3)
                    draw.point((x, y), fill=(int(r * 255), int(g * 255), int(b * 255)))

        else:  # checkerboard
            # Checkerboard
            tile_size = random.randint(20, 40)
            color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for x in range(0, width, tile_size):
                for y in range(0, height, tile_size):
                    if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                        draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color1)
                    else:
                        draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color2)

        # Update sphere texture directly without saving file
        self.texture_image = img
        self.texture_pixels = img.load()
        self.create_sphere_texture()
        self.generate_optimized_sphere_points(density=800)
        self.redraw_optimized()

        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[texture] Created random {pattern_type} texture")
            else:
                print(f"[texture] Created random {pattern_type} texture")
        except:
            print(f"[texture] Created random {pattern_type} texture")

    def update_texture(self, image_path=None):
        """Update texture dynamically"""
        if image_path and os.path.exists(image_path):
            self.load_texture(image_path)
        else:
            self.create_default_texture()

        self.create_sphere_texture()
        self.generate_optimized_sphere_points(density=800)
        self.redraw_optimized()

    def destroy(self):
        """Clean up before destroying"""
        # Close control window
        self._close_control_window()

        # Clear caches
        self._cached_points = []
        self._cached_geom = None

        super().destroy()
#Hal 9000
class Hal9000Avatar(BaseAvatarWindow):
    """HAL 9000 inspired avatar - SIMPLE RADIAL VERSION"""

    # Colors
    SILVER = "#C0C0C0"  # Outer ring
    BLACK = "#000000"  # Background
    RED_CENTER = "#FF0000"  # Red center dot
    RED_GLOW = "#FF4444"  # Red glow
    SPIKE_COLOR = "#FF3333"  # Radial spike color
    SPIKE_HIGHLIGHT = "#FF6666"  # Bright part of spikes

    def __init__(self, master):
        super().__init__(master, "Avatar - HAL 9000")

        # Simple state
        self._smoothed_level = 0.0
        self._animation_phase = 0.0

        # Start animation
        self.start_animation()

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def set_level(self, level: int):
        """Set the audio level with 1.5x dynamic range"""
        super().set_level(level)

        # Apply 1.5x dynamic range
        target = (level / float(self.LEVELS - 1)) * 1.5
        target = min(1.0, target)  # Cap at 1.0

        # Smooth it
        self._smoothed_level += (target - self._smoothed_level) * 0.3

        # Update animation phase
        self._animation_phase += 0.1
        if self._animation_phase > math.pi * 2:
            self._animation_phase -= math.pi * 2

    def draw_radial_spikes(self, cx, cy, r, intensity):
        """Draw radial spikes like RadialPulseAvatar"""
        # Number of spikes based on intensity
        max_spikes = 24
        spikes = max(3, int(max_spikes * intensity))

        # Spike length based on intensity (with 1.5x range)
        max_length = r * 0.8
        base_length = max_length * intensity

        # Draw spikes
        for i in range(spikes):
            angle = (2 * math.pi * i) / spikes

            # Add some variation to spike lengths
            variation = 0.7 + 0.6 * math.sin(angle * 3 + self._animation_phase * 2)
            spike_length = base_length * variation

            # Calculate end point
            end_x = cx + spike_length * math.cos(angle)
            end_y = cy + spike_length * math.sin(angle)

            # Line width varies with intensity and position
            width_base = 1 + intensity * 3
            width_var = 0.8 + 0.4 * math.sin(angle * 2 + self._animation_phase)
            line_width = max(1, int(width_base * width_var))

            # Color variation (darker at base, brighter at tip)
            tip_factor = min(1.0, variation * 1.2)
            spike_color = self.SPIKE_HIGHLIGHT if tip_factor > 0.9 else self.SPIKE_COLOR

            # Draw the spike
            self.canvas.create_line(
                cx, cy, end_x, end_y,
                fill=spike_color,
                width=line_width,
                capstyle=tk.ROUND
            )

            # Add a small glow at the tip for longer spikes
            if spike_length > r * 0.4:
                tip_size = max(2, int(line_width * 0.8))
                self.canvas.create_oval(
                    end_x - tip_size, end_y - tip_size,
                    end_x + tip_size, end_y + tip_size,
                    fill=self.SPIKE_HIGHLIGHT, outline=self.SPIKE_HIGHLIGHT
                )

    def redraw(self):
        """Simple redraw that always works"""
        self.canvas.delete("all")

        # Get geometry
        cx, cy, r = self._circle_geom()

        # Draw black background
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BLACK, outline=self.BLACK)

        # === OUTER SILVER RING ===
        silver_radius = int(r * 0.95)
        silver_width = max(3, int(r * 0.07))

        self.canvas.create_oval(
            cx - silver_radius, cy - silver_radius,
            cx + silver_radius, cy + silver_radius,
            outline=self.SILVER, width=silver_width
        )

        # === INNER BLACK RING ===
        black_radius = int(r * 0.88)
        black_width = max(2, int(r * 0.05))

        self.canvas.create_oval(
            cx - black_radius, cy - black_radius,
            cx + black_radius, cy + black_radius,
            outline=self.BLACK, width=black_width
        )

        # === DRAW RADIAL SPIKES WHEN AUDIO IS ACTIVE ===
        if self._smoothed_level > 0.05:
            self.draw_radial_spikes(cx, cy, r, self._smoothed_level)

        # === RED CENTER DOT ===
        # Base center size
        center_base = max(3, int(r * 0.1))

        # Add pulse effect when active
        if self._smoothed_level > 0.1:
            pulse = 1.0 + 0.3 * math.sin(self._animation_phase * 2)
            center_radius = int(center_base * pulse)
        else:
            # Subtle idle pulse
            idle_pulse = 0.8 + 0.2 * math.sin(self._animation_phase * 0.5)
            center_radius = int(center_base * idle_pulse)

        # Outer glow for center
        glow_radius = int(center_radius * 1.5)
        glow_intensity = int(100 + 155 * self._smoothed_level)
        glow_color = f"#{glow_intensity:02x}0000"  # Red glow

        self.canvas.create_oval(
            cx - glow_radius, cy - glow_radius,
            cx + glow_radius, cy + glow_radius,
            fill=glow_color, outline=glow_color
        )

        # Main red center
        red_intensity = min(255, int(200 + 55 * self._smoothed_level))
        center_color = f"#{red_intensity:02x}0000"

        self.canvas.create_oval(
            cx - center_radius, cy - center_radius,
            cx + center_radius, cy + center_radius,
            fill=center_color, outline=center_color
        )

        # Center highlight (small white dot)
        highlight_size = max(1, int(center_radius * 0.3))
        highlight_x_offset = int(-center_radius * 0.2)
        highlight_y_offset = int(-center_radius * 0.2)

        self.canvas.create_oval(
            cx + highlight_x_offset - highlight_size,
            cy + highlight_y_offset - highlight_size,
            cx + highlight_x_offset + highlight_size,
            cy + highlight_y_offset + highlight_size,
            fill="#FFFFFF", outline="#FFFFFF"
        )





# === ALL AVATARS COLLECTION ===
ALL_AVATARS = [
    CircleAvatarWindow,
    RectAvatarWindow,
    RectAvatarWindow2,
    RadialPulseAvatar,
    FaceRadialAvatar,
    StringGridAvatar,
    TextureMappedSphere,
    Hal9000Avatar  # Add the new HAL avatar here
]
