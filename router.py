# command_router.py
import re
import time
import webbrowser
from typing import Callable, Dict, Any
import tkinter.messagebox as messagebox
import os


class CommandRouter:
    """
    Handles voice and text command routing for the AI assistant.
    Simplified for single AI model.
    """

    def __init__(self, app_instance=None):
        self.app = app_instance
        self.sleep_mode = False

        # Command categories
        self.sleep_commands = ["go to sleep", "rest mode", "stop listening", "sleep mode"]
        self.wake_commands = ["wake up", "awaken", "resume", "start listening", "listen again"]

        # =========== PLOTTING CONTROL ===========
        self.plotting_enabled = True  # default is ON

        # Personality commands mapping
        self.personality_commands = {
            "be the butler": "Jeeves the Butler",
            "butler mode": "Jeeves the Butler",
            "butler personality": "Jeeves the Butler",
            "be the teacher": "Mrs. Hardcastle",
            "teacher mode": "Mrs. Hardcastle",
            "school teacher": "Mrs. Hardcastle",
            "strict teacher": "Mrs. Hardcastle",
            "be the scientist": "Dr. Von Knowledge",
            "mad scientist": "Dr. Von Knowledge",
            "scientist mode": "Dr. Von Knowledge",
            "dr von knowledge": "Dr. Von Knowledge",
            "be default": "Default",
            "be normal": "Default",
            "normal mode": "Default",
            "be the explorer": "Space Explorer",
            "space explorer mode": "Space Explorer",
            "space explorer personality": "Space Explorer",
            "captain nova mode": "Space Explorer",
            "be yogi": "BK Yogi",
            "aum shanti": "BK Yogi",
            "be the computer": "Holly",
            "be holly": "Holly",
            "be the ship computer": "Holly",
            "holly mode": "Holly",
            "red dwarf mode": "Holly",
            "computer personality": "Holly",
            "ship computer": "Holly",
            "talk to holly": "Holly",
            "wake up holly": "Holly",
            "be rab": "Rab C. Nesbitt",
            "rab c nesbitt": "Rab C. Nesbitt",
            "glasgow mode": "Rab C. Nesbitt",
            "scottish philosopher": "Rab C. Nesbitt",
            "be the glaswegian": "Rab C. Nesbitt",
            "rab mode": "Rab C. Nesbitt",
            "talk to rab": "Rab C. Nesbitt",
            "glasgow drunk": "Rab C. Nesbitt",
            "street philosopher": "Rab C. Nesbitt",
            "nesbitt mode": "Rab C. Nesbitt",
            "be nicole": "Nicole from Paris",
            "be einstein": "Albert Einstein",
            "be shakespeare": "William Shakespeare",
            "be burns":"Robert Burns"
        }

        # SIMPLIFIED Mute commands (single AI only)
        self.mute_commands = {
            "mute audio": "toggle_mute",
            "mute the ai": "toggle_mute",
            "unmute": "toggle_mute",
            "enable audio": "toggle_mute",
            "audio on": "toggle_mute",
            "toggle mute": "toggle_mute",
            "mute toggle": "toggle_mute"
        }

        # Search commands - more specific to avoid false matches
        self.search_commands = [
            "search for", "search the web", "web search",
            "look up", "find information about", "find info on",
            "search online for", "look online for"
        ]

        # =========== PLOT COMMANDS ===========
        self.plot_commands = [
            "plot the function", "graph the function", "draw the graph",
            "show the graph", "plot the equation", "graph the equation",
            "plot the result", "graph the result", "draw the function",
            "show me a plot", "show a plot", "display the graph",
            "visualize the function", "create a plot", "make a graph",
            "draw a plot", "plot the graph", "graph that equation",
            "show me the graph", "make a plot", "create a graph",
            "show graphically", "plot this function", "graph this equation",
            "show as a graph", "draw the equation", "plot the expression",
            "graph the expression", "plot that function"
        ]

        # Short plot reference commands (handled separately)
        self.plot_reference_commands = [
            "plot that", "graph that", "plot it", "graph it",
            "draw that", "draw it", "visualize that", "visualize it"
        ]

        # Stop commands - more specific to avoid false matches
        self.stop_commands = [
            "stop speaking", "stop talking", "be quiet", "shut up",
            "that's enough", "okay stop", "ok stop",
            "stop now", "please stop", "stop the audio"
        ]

        self.close_windows_commands = [
            "close window", "close all windows", "hide windows", "minimize windows",
            "clean up windows", "tidy windows", "close extra windows", "clean desktop"
        ]

        self.debug_commands = [
            "debug mode", "system status", "show state", "debug state"
        ]

        self.test_vision_commands = [
            "test vision", "vision test", "check vision"
        ]

        # Commands that refer to STATIC images (uploaded files)
        self.static_image_commands = [
            "describe the image", "describe what's in the image", "describe what is in the image",
            "what's in the image", "what is in the image", "analyze the image",
            "explain the image", "what does the image show"
        ]

        # Commands that trigger LIVE camera view
        self.live_camera_commands = [
            "what do you see", "what can you see", "whats happening", "what is happening",
            "whats going on", "what is going on", "describe what you see", "tell me what you see",
            "show me what you see", "what are you seeing", "whats in front of you",
            "describe the scene", "whats around you", "whats in the room"
        ]

        self.camera_commands = {
            "start camera": "start_camera",
            "load camera": "start_camera",
            "turn on camera": "start_camera",
            "camera on": "start_camera",
            "start the camera": "start_camera",
            "enable camera": "start_camera",
            "stop camera": "stop_camera",
            "close camera": "stop_camera",
            "turn off camera": "stop_camera",
            "camera off": "stop_camera",
            "stop the camera": "stop_camera",
            "disable camera": "stop_camera",
            "take a picture": "take_picture",
            "take picture": "take_picture",
            "take a photo": "take_picture",
            "take photo": "take_picture",
            "snapshot": "take_picture",
            "capture photo": "take_picture",
            "capture image": "take_picture",
            "snap a picture": "take_picture"
        }

    def set_app_instance(self, app_instance):
        """Set the main app instance for callbacks"""
        self.app = app_instance

    def filter_whisper_hallucinations(self, text: str) -> str:
        """Filter out common Whisper hallucinations"""
        if not text or not text.strip():
            return ""

        hallucinations = [
            "thanks for watching",
            "I'm so sorry",
            "I am so sorry",
            "Thank you for watching!",
            "Thank you very much.",
            "Thank you very much",
            "don't forget to subscribe",
            "hit the bell icon",
            "see you next time",
            "College. Thank you.",
            "Thank you for watching!",
            "bye everyone",
            "Thank you for watching and see you next time.",
            "in this video",
            "before we start",
            "subscribe to my channel",
            "like and subscribe",
            "comment below",
            "turn on notifications",
            "hit the like button",
            "smash that like button",
            "ring the bell",
            "thanks for watching this video",
            "thanks for watching the video",
            "welcome back to",
            "hey guys",
            "what's up guys",
            "hello everyone",
            "hi everyone",
            "welcome to my channel",
            "let's get started",
            "without further ado",
            "before we begin",
            "in today's video",
            "in this tutorial",
            "question. Thank you.",
            "Thank you for listening and have a great day.",
            "That's crazy.",
            "Thank you. Thank you.",
            "Thank you",
            "Thank you.",
            "Thank you!",
            "Thanks",
            "Thanks.",
            "Thanks!",
            "Thank you thank you",
            "Thankyou",
            "Have a good day.",
            "I'm not going to do that. ",
        ]

        text_lower = text.lower().strip()
        if self.app:
            self.app.logln(f"[asr-filter] Checking: '{text}'")

        # Clean punctuation for matching
        text_clean = text_lower.replace('!', '').replace('?', '').replace('.', '').replace(',', '')
        text_clean = text_clean.strip()

        # Also check text without any cleanup for exact matches
        if text_lower in [h.lower() for h in hallucinations]:
            if self.app:
                self.app.logln(f"[asr-filter] ❌ EXACT LOWER MATCH FILTERED: '{text}'")
            return ""

        # Check cleaned versions
        for hallucination in hallucinations:
            hallucination_clean = hallucination.lower().replace('!', '').replace('?', '').replace('.', '').replace(',', '')

            if text_clean == hallucination_clean:
                if self.app:
                    self.app.logln(f"[asr-filter] ❌ EXACT MATCH FILTERED: '{text}' (matched: '{hallucination}')")
                return ""

        # Check if text starts with any hallucination
        for hallucination in hallucinations:
            hallucination_lower = hallucination.lower()
            if text_lower.startswith(hallucination_lower):
                if self.app:
                    self.app.logln(f"[asr-filter] ❌ STARTS WITH FILTERED: '{text}' (starts with: '{hallucination}')")
                return ""

        # Check for repeated short phrases
        words = text_lower.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()
        if len(words) <= 3:
            thank_you_variations = {"thank", "you", "thanks", "thankyou"}
            if all(word in thank_you_variations for word in words):
                if self.app:
                    self.app.logln(f"[asr-filter] ❌ THANK YOU VARIATION FILTERED: '{text}'")
                return ""

            if len(words) == 2 and words[0] == words[1]:
                if self.app:
                    self.app.logln(f"[asr-filter] ❌ REPETITION FILTERED: '{text}'")
                return ""

        if self.app:
            self.app.logln(f"[asr-filter] ✅ PASSED: '{text}'")
        return text

    def normalize_text(self, text: str) -> str:
        """Normalize text for command matching"""
        if not text:
            return ""

        norm_map = {
            "what's": "what is",
            "whats": "what is",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is",
        }

        normalized = text.lower()
        for k, v in norm_map.items():
            normalized = normalized.replace(k, v)

        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()

        return normalized

    def matched_phrase(self, text: str, patterns: list) -> bool:
        """Check if text matches any PHRASE pattern in list (multi-word safe)"""
        return any(p in text for p in patterns)

    def matched_word(self, text: str, word: str) -> bool:
        """Check if a single WORD matches using word boundaries"""
        return bool(re.search(rf'\b{re.escape(word)}\b', text))

    def route_command(self, raw_text: str) -> bool:
        """
        Handle voice/typed control phrases robustly.
        Returns True if a command was executed.
        """
        if not self.app:
            return False

        raw_text = self.filter_whisper_hallucinations(raw_text)
        if not raw_text:
            return False

        text = self.normalize_text(raw_text)

        # DEBUG: Add logging to see what's being checked
        if self.app:
            self.app.logln(f"[router] Checking: '{text}'")
            self.app.logln(f"[router] Sleep mode: {self.sleep_mode}")

        # Handle sleep/wake commands FIRST (before other commands)
        if self._handle_sleep_commands(text):
            return True

        # Now check other commands
        return (
                self._handle_stop_commands(text) or
                self._handle_camera_commands(text) or
                self._handle_what_see_commands(text) or
                self._handle_personality_commands(text) or
                self._handle_mute_commands(text) or
                self._handle_plot_commands(text, raw_text) or
                self._handle_search_commands(text, raw_text) or
                self._handle_test_vision(text) or
                self._handle_debug_commands(text) or
                self._handle_close_windows(text)
        )

    def _handle_sleep_commands(self, text: str) -> bool:
        """Handle sleep/wake commands"""

        # ONLY trigger on short phrases (under 10 words)
        words = text.split()
        if len(words) > 10:
            return False

        # Check sleep commands
        for cmd in self.sleep_commands:
            if cmd in text:
                if self.app:
                    self.app.logln(f"[sleep] Matched command: '{cmd}' in '{text}'")
                self.enter_sleep_mode()
                return True

        # Check wake commands (only if sleeping)
        if self.sleep_mode:
            for cmd in self.wake_commands:
                if cmd in text:
                    if self.app:
                        self.app.logln(f"[wake] Matched command: '{cmd}' in '{text}'")
                    self.exit_sleep_mode()
                    return True

        return False

    def _handle_stop_commands(self, text: str) -> bool:
        """Handle stop speaking commands"""
        if self.matched_phrase(text, self.stop_commands):
            self.app.logln("[command] Stop command detected - stopping speech")
            self.app.stop_speaking()
            self.app.set_light("idle")
            return True
        return False

    def _handle_camera_commands(self, text: str) -> bool:
        """Handle camera control commands"""
        for command, method_name in self.camera_commands.items():
            if command in text:
                method = getattr(self.app, f"{method_name}_ui", None)
                if method:
                    method()
                    self.app.set_light("idle")
                    return True
        return False

    def _handle_what_see_commands(self, text: str) -> bool:
        """Handle vision commands - distinguish between static images and live camera"""

        # Check for STATIC image commands first
        if self.matched_phrase(text, self.static_image_commands):
            self.app.logln("[vision] Static image command detected")

            has_static_image = False
            if hasattr(self.app, '_last_image_path') and self.app._last_image_path:
                if os.path.exists(self.app._last_image_path):
                    if "snapshot_" not in os.path.basename(self.app._last_image_path):
                        has_static_image = True
                        self.app.logln(f"[vision] Found static image: {os.path.basename(self.app._last_image_path)}")

            if has_static_image:
                prompt = "Describe this image in detail."
                self.app.explain_last_image_ui(prompt)
            else:
                self.app.speak_search_status("Please upload an image first")
                self.app.play_chime(freq=440, ms=300, vol=0.1)

            self.app.set_light("idle")
            return True

        # Check for LIVE CAMERA commands
        if self.matched_phrase(text, self.live_camera_commands):
            self.app.logln("[vision] Live camera command detected")
            self.app.what_do_you_see_ui()
            self.app.set_light("idle")
            return True

        return False

    def _handle_personality_commands(self, text: str) -> bool:
        """Handle personality switching commands"""
        for command, personality in self.personality_commands.items():
            if command in text:
                if hasattr(self.app, 'personalities') and personality in self.app.personalities:
                    current_personality = self.app.personality_var.get()
                    if current_personality == personality:
                        self.app.speak_search_status(f"Already in {personality} mode")
                    else:
                        self.app.personality_var.set(personality)
                        self.app.apply_personality(personality)
                        if personality == "Default":
                            self.app.speak_search_status("Returning to normal mode")
                     #   else:
                      #      self.app.speak_search_status(f"{personality} personality") Already in main app!!
                    return True
        return False

    def _handle_mute_commands(self, text: str) -> bool:
        """Handle mute/unmute commands for single AI"""
        for command, method_name in self.mute_commands.items():
            if command in text:
                method = getattr(self.app, "toggle_text_ai_mute", None)
                if method:
                    method()
                    return True
        return False

    def _handle_plot_commands(self, text: str, raw_text: str) -> bool:
        """Handle plot/graph commands."""

        text_lower = raw_text.lower().strip()

        # EXCLUDE repeat commands - they're NEVER plot commands
        repeat_patterns = [
            'repeat', 'say again', 'say that again', 'say it again',
            'could you repeat', 'can you repeat', 'please repeat',
            'repeat yourself', 'repeat what you said'
        ]

        for pattern in repeat_patterns:
            if text_lower.startswith(pattern) or text_lower == pattern:
                return False

        # Exclude standalone pronouns
        if text_lower in ['that', 'it', 'this']:
            return False

        # Check for explicit plot commands
        is_plot_command = self.matched_phrase(text, self.plot_commands)

        # Check for short reference commands like 'plot that' or 'graph it'
        is_reference_command = self.matched_phrase(text_lower, self.plot_reference_commands)

        # If not a plot-related command, return False
        if not (is_plot_command or is_reference_command):
            return False

        # Check if plotting is enabled
        if not self.plotting_enabled:
            self.app.logln(f"[plot] Plotting disabled - ignoring: '{raw_text}'")
            return False

        self.app.logln(f"[command] Plot command detected: '{raw_text}'")

        if not hasattr(self.app, 'plotter') or self.app.plotter is None:
            self.app.logln("[plot] Plotter not available")
            self.app.speak_search_status("Plotting functionality is not available")
            return True

        try:
            last_math = self._find_last_mathematical_expression()

            if is_reference_command:
                self.app.logln(f"[plot] Reference command: '{raw_text}' - looking for context")

                if last_math:
                    self.app.logln(f"[plot] Found context expression: {last_math}")
                    plot_window = self.app.plotter.plot_from_text(last_math)

                    if plot_window:
                        self.app.speak_search_status("Plotting the function")
                        self.app.play_chime(freq=660, ms=200, vol=0.15)
                    else:
                        self.app.speak_search_status("I couldn't plot the expression")
                        self.app.play_chime(freq=440, ms=300, vol=0.1)
                else:
                    self.app.speak_search_status("I'm not sure what you want me to plot. Please specify an expression.")
                    self.app.play_chime(freq=440, ms=300, vol=0.1)
                return True

            # Handle regular plot commands
            self.app.logln(f"[plot] Regular plot command: '{raw_text}'")

            if last_math:
                self.app.logln(f"[plot] Using context: {last_math}")
                plot_window = self.app.plotter.plot_from_text(last_math)

                if plot_window:
                    self.app.speak_search_status("Plotting the mathematical function")
                    self.app.play_chime(freq=660, ms=200, vol=0.15)
                else:
                    self.app.speak_search_status("I couldn't find any plottable functions")
                    self.app.play_chime(freq=440, ms=300, vol=0.1)
            else:
                self.app.speak_search_status("Please provide a function to plot")
                self.app.play_chime(freq=440, ms=300, vol=0.1)

        except Exception as e:
            self.app.logln(f"[plot] Error: {e}")
            import traceback
            self.app.logln(f"[plot] Traceback: {traceback.format_exc()}")
            self.app.speak_search_status("Error creating plot")

        return True

    def _find_last_mathematical_expression(self):
        """Find the LAST mathematical expression from conversation context - RETURNS RAW LATEX."""
        try:
            self.app.logln("[plot-context] Looking for last mathematical SOLUTION...")

            if hasattr(self.app, 'latex_win_text') and self.app.latex_win_text:
                last_text = getattr(self.app.latex_win_text, '_last_text', "")
                self.app.logln(f"[plot-context] LaTeX text ({len(last_text)} chars)")

                if last_text:
                    # Collect all matches
                    all_candidates = []

                    # Look for LaTeX math: $...$, $$...$$, \[...\], \(...\)
                    latex_patterns = [
                        r'\$\$([^\$]+)\$\$',
                        r'\$([^\$]+)\$',
                        r'\\\[([^\]]+)\\\]',
                        r'\\\(([^)]+)\\\)',
                    ]

                    for pattern in latex_patterns:
                        matches = re.findall(pattern, last_text, re.DOTALL)
                        for match in matches:
                            match_stripped = match.strip()
                            has_variable = any(
                                var in match_stripped.lower() for var in ['x', 't', 'theta', 'θ', 'n', 's'])
                            if match_stripped and has_variable:
                                cleaned = self._prepare_latex_for_plotting(match_stripped)
                                if cleaned:
                                    all_candidates.append((match_stripped, cleaned, 'latex'))

                    # Look for boxed answers: \boxed{...}
                    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
                    boxed_matches = re.findall(boxed_pattern, last_text, re.DOTALL)
                    for match in boxed_matches:
                        match_stripped = match.strip()
                        has_variable = any(var in match_stripped.lower() for var in ['x', 't', 'theta', 'θ', 'n', 's'])
                        if match_stripped and has_variable:
                            cleaned = self._prepare_latex_for_plotting(match_stripped)
                            if cleaned:
                                all_candidates.append((match_stripped, cleaned, 'boxed'))

                    # Look for equations with = sign
                    equation_patterns = [
                        r'=\s*([^=\n.,;]+)',
                        r'\\frac\{[^}]+\}\{[^}]+\}[^=\n.,;]*',
                    ]

                    for pattern in equation_patterns:
                        matches = re.findall(pattern, last_text, re.IGNORECASE)
                        for match in matches:
                            match_stripped = match.strip()
                            has_variable = any(
                                var in match_stripped.lower() for var in ['x', 't', 'theta', 'θ', 'n', 's'])
                            if match_stripped and has_variable:
                                if any(op in match_stripped for op in ['\\frac', '/', '*', '+', '-', '^']):
                                    cleaned = self._prepare_latex_for_plotting(match_stripped)
                                    if cleaned:
                                        all_candidates.append((match_stripped, cleaned, 'equation'))

                    # Prioritize results
                    if not all_candidates:
                        self.app.logln("[plot-context] ❌ No mathematical expressions found")
                        return None

                    # Priority 1: Boxed expressions
                    boxed = [c for c in all_candidates if c[2] == 'boxed']
                    if boxed:
                        match_stripped, cleaned, _ = boxed[-1]
                        self.app.logln(f"[plot-context] Found boxed: '{match_stripped}' -> '{cleaned}'")
                        return cleaned

                    # Priority 2: Regular LaTeX expressions
                    latex = [c for c in all_candidates if c[2] == 'latex']
                    if latex:
                        match_stripped, cleaned, _ = latex[-1]
                        self.app.logln(f"[plot-context] Found LaTeX: '{match_stripped}' -> '{cleaned}'")
                        return cleaned

                    # Priority 3: Equations
                    equations = [c for c in all_candidates if c[2] == 'equation']
                    if equations:
                        match_stripped, cleaned, _ = equations[-1]
                        self.app.logln(f"[plot-context] Found equation: '{match_stripped}' -> '{cleaned}'")
                        return cleaned

            self.app.logln("[plot-context] ❌ No mathematical solution found")
            return None

        except Exception as e:
            self.app.logln(f"[plot-context] Error finding last expression: {e}")
            import traceback
            self.app.logln(f"[plot-context] Traceback: {traceback.format_exc()}")
            return None

    def _handle_search_commands(self, text: str, raw_text: str) -> bool:
        """Handle search commands with query extraction"""

        # ONLY trigger search on short phrases (under 20 words)
        words = text.split()
        if len(words) > 20:
            return False

        for cmd in self.search_commands:
            if cmd in text:
                query = text.replace(cmd, "").strip()

                if not query:
                    cmd_words = cmd.split()
                    if len(words) > len(cmd_words):
                        query = " ".join(words[len(cmd_words):])

                if query:
                    self.app.logln(f"[search] Voice search: {query}")
                    self.app.speak_search_status(f"Searching for {query}")
                    self.app.toggle_search_window(ensure_visible=True)

                    def do_voice_search():
                        try:
                            if (self.app.search_win and
                                    self.app.search_win.winfo_exists() and
                                    not self.app.search_win.in_progress):
                                self.app.search_win.txt_in.delete("1.0", "end")
                                self.app.search_win.txt_in.insert("1.0", query)
                                self.app.search_win.on_go()
                            else:
                                self.app.logln("[search] Search window not ready for voice command")
                        except Exception as e:
                            self.app.logln(f"[search] Voice search error: {e}")

                    if hasattr(self.app, 'master'):
                        self.app.master.after(500, do_voice_search)
                    return True
        return False

    def _handle_test_vision(self, text: str) -> bool:
        """Handle vision test commands"""
        if self.matched_phrase(text, self.test_vision_commands):
            self.app.logln("[vision-test] Testing vision capabilities...")
            if hasattr(self.app, '_last_image_path') and self.app._last_image_path and os.path.exists(
                    self.app._last_image_path):
                self.app.logln(f"[vision-test] Current image: {os.path.basename(self.app._last_image_path)}")
                self.app.ask_vision(self.app._last_image_path, "Describe this image briefly for testing.")
            else:
                self.app.logln("[vision-test] No current image available")
            return True
        return False

    def _handle_debug_commands(self, text: str) -> bool:
        """Handle debug commands"""
        if self.matched_phrase(text, self.debug_commands):
            self.app.logln("[debug] Current state:")
            self.app.logln(f"[debug] - Has image: {bool(getattr(self.app, '_last_image_path', None))}")
            self.app.logln(f"[debug] - Last vision: {getattr(self.app, '_last_was_vision', False)}")
            self.app.set_light("idle")
            return True
        return False

    def _handle_close_windows(self, text: str) -> bool:
        """Handle close windows commands"""
        if self.matched_phrase(text, self.close_windows_commands):
            self.app.logln(f"[command] Close windows command detected: '{text}'")

            if hasattr(self.app, 'close_all_windows'):
                self.app.close_all_windows()

            if hasattr(self.app, 'plotter') and self.app.plotter:
                try:
                    self.app.plotter.close_all()
                    self.app.logln(f"[command] Also closed plot windows")
                except Exception as e:
                    self.app.logln(f"[command] Error closing plot windows: {e}")

            return True
        return False

    def _prepare_latex_for_plotting(self, latex_expr: str) -> str:
        """Prepare LaTeX expression for plotting - extract plottable part."""
        if not latex_expr:
            return ""

        # Skip substitution variables
        if re.match(r'^[uvwstpq]\s*=', latex_expr.strip(), re.IGNORECASE):
            return ""

        # Skip inequalities and conditions
        inequality_pattern = r'\\(geq|leq|neq|ge|le|gt|lt)\b'
        if re.search(inequality_pattern, latex_expr):
            return ""

        if re.search(r'[<>≥≤≠]', latex_expr):
            return ""

        # Clean markdown artifacts
        latex_expr = latex_expr.strip()
        latex_expr = re.sub(r'^\*\*\s*', '', latex_expr)
        latex_expr = re.sub(r'\s*\*\*$', '', latex_expr)
        latex_expr = re.sub(r'\\\)\s*\*\*\s*$', '', latex_expr)
        latex_expr = re.sub(r'\\\)\s*$', '', latex_expr)
        latex_expr = re.sub(r'^\\\(\s*', '', latex_expr)

        # Remove integration constants
        latex_expr = re.sub(r'\s*\+\s*C\s*$', '', latex_expr, flags=re.IGNORECASE)

        # Extract right side of equations
        if '=' in latex_expr:
            parts = latex_expr.split('=', 1)
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ""

            if '\\int' in left or 'dx' in left or 'dt' in left:
                latex_expr = right
            elif left in ['y', 'f(x)', 'g(x)', 'h(x)'] or '(' in left:
                latex_expr = right

        # Remove integral notation
        latex_expr = re.sub(r'\\int\s*', '', latex_expr)
        latex_expr = re.sub(r'\s*\\,\s*d[a-z]\s*', '', latex_expr)
        latex_expr = re.sub(r'\s*d[a-z]\s*$', '', latex_expr)

        # Wrap in $ if needed
        if not latex_expr.startswith('$') and not latex_expr.startswith('\\['):
            latex_expr = f'${latex_expr}$'

        return latex_expr.strip()

    def enter_sleep_mode(self):
        """Enter sleep mode"""
        if not self.sleep_mode:
            self.sleep_mode = True
            if self.app:
                self.app.set_light("idle")
                self.app.logln("[sleep] 💤 Sleep mode activated - ignoring voice input")
                self.play_sleep_chime()
                if hasattr(self.app, 'close_all_windows'):
                    self.app.close_all_windows()
                try:
                    self.app.master.title("Always Listening — AI (SLEEPING)")
                except:
                    pass

    def exit_sleep_mode(self):
        """Exit sleep mode"""
        if self.sleep_mode:
            self.sleep_mode = False
            if self.app:
                self.app.set_light("listening")
                self.app.logln("[sleep] 🔔 Awake mode activated - listening for voice")
                self.play_wake_chime()
                try:
                    self.app.master.title("Always Listening — AI")
                except:
                    pass

    def play_sleep_chime(self):
        """Play sleep confirmation chime"""
        try:
            import numpy as np
            import sounddevice as sd

            fs = 16000
            duration = 0.3
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = np.linspace(660, 220, len(t))
            beep = 0.2 * np.sin(2 * np.pi * freq * t)
            fade = int(0.02 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            if self.app:
                self.app.logln(f"[sleep-chime] {e}")

    def play_wake_chime(self):
        """Play wake confirmation chime"""
        try:
            import numpy as np
            import sounddevice as sd

            fs = 16000
            duration = 0.25
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = np.linspace(220, 660, len(t))
            beep = 0.2 * np.sin(2 * np.pi * freq * t)
            fade = int(0.01 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            if self.app:
                self.app.logln(f"[wake-chime] {e}")

    def play_sleep_reminder_beep(self):
        """Play noticeable beep to indicate sleeping mode"""
        try:
            import numpy as np
            import sounddevice as sd

            fs = 16000
            duration = 0.25
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            freq1 = 440
            freq2 = 330

            beep1 = 0.3 * np.sin(2 * np.pi * freq1 * t[:len(t) // 2])
            beep2 = 0.3 * np.sin(2 * np.pi * freq2 * t[len(t) // 2:])
            beep = np.concatenate([beep1, beep2])

            fade = int(0.02 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)

            sd.play(beep, fs, blocking=False)
            if self.app:
                self.app.logln("[sleep] 💤 (sleep reminder beep)")

        except Exception as e:
            if self.app:
                self.app.logln(f"[sleep-reminder] error: {e}")