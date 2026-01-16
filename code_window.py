# code_window.py - IMPROVED VERSION with plot support
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import subprocess
import tempfile
import os
import sys
import re


class CodeWindow(tk.Toplevel):
    """Popup window for code execution with run button"""

    def __init__(self, master, log_callback=None, output_callback=None):
        super().__init__(master)
        self.title("Code Sandbox")
        self.geometry("850x650")
        self.protocol("WM_DELETE_WINDOW", self.hide)

        # Logging callback to main app
        self.log_callback = log_callback or print

        # Output callback to send results to main app's text box
        self.output_callback = output_callback

        # Configure window
        self.configure(bg="#2b2b2b")

        # Auto-send option (default ON)
        self.auto_send_var = tk.BooleanVar(value=True)

        # Create UI
        self._create_ui()

        # Store current code
        self.current_code = ""

        # Track last saved plot path
        self.last_plot_path = None

        # Track last output for manual sending
        self._last_output = ""
        self._last_had_error = False

        # Hide by default
        self.withdraw()

        self.log("[sandbox] Code window initialized")

    def _create_ui(self):
        """Create the code window UI"""
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # Run button
        self.run_btn = ttk.Button(
            toolbar,
            text="Run Code",
            command=self._run_code_safe,
            width=12
        )
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Copy button
        ttk.Button(
            toolbar,
            text="Copy",
            command=self._copy_code,
            width=10
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Clear button
        ttk.Button(
            toolbar,
            text="Clear",
            command=self._clear_code,
            width=10
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Save Plot button
        self.save_plot_btn = ttk.Button(
            toolbar,
            text="Save Plot",
            command=self._save_plot,
            width=10,
            state=tk.DISABLED
        )
        self.save_plot_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Open Plot button
        self.open_plot_btn = ttk.Button(
            toolbar,
            text="Open Plot",
            command=self._open_plot,
            width=10,
            state=tk.DISABLED
        )
        self.open_plot_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            toolbar,
            textvariable=self.status_var,
            foreground="#50c878"
        ).pack(side=tk.LEFT, padx=(20, 0))

        # Separator before AI options
        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=10)

        # Auto-send to AI checkbox
        self.auto_send_cb = ttk.Checkbutton(
            toolbar,
            text="Auto-send output to AI",
            variable=self.auto_send_var
        )
        self.auto_send_cb.pack(side=tk.LEFT, padx=(0, 6))

        # Manual send to AI button
        ttk.Button(
            toolbar,
            text="Send to AI",
            command=self._send_output_to_ai,
            width=10
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Close button (right side)
        ttk.Button(
            toolbar,
            text="Close",
            command=self.hide,
            width=10
        ).pack(side=tk.RIGHT)

        # Paned window for code and output
        self.paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Code editor frame
        code_frame = ttk.LabelFrame(self.paned, text="Python Code", padding=5)

        # Code text area with monospace font
        self.code_text = scrolledtext.ScrolledText(
            code_frame,
            wrap=tk.NONE,
            font=("Consolas", 11),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#569cd6",
            selectbackground="#264f78",
            borderwidth=0,
            relief="flat",
            height=20
        )
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Output frame
        output_frame = ttk.LabelFrame(self.paned, text="Output", padding=5)

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#0c0c0c",
            fg="#cccccc",
            state="disabled",
            borderwidth=0,
            relief="flat",
            height=10
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Add frames to paned window
        self.paned.add(code_frame, weight=3)
        self.paned.add(output_frame, weight=1)

        # Bind Ctrl+Enter to run code
        self.code_text.bind("<Control-Return>", lambda e: self._run_code_safe())
        self.code_text.bind("<Control-r>", lambda e: self._run_code_safe())

    def set_code(self, code: str, auto_run=False):
        """Set code in the editor and optionally run it"""
        self.current_code = code.strip()
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", self.current_code)

        # Clear previous output
        self._clear_output()

        # Update status
        self.status_var.set("Code loaded")

        if auto_run:
            self.master.after(100, self._run_code_safe)

    def _run_code_safe(self):
        """Execute code safely with restrictions"""
        self.log("[sandbox] ⚡ AUTO-RUN STARTING")
        # Get current code
        code = self.code_text.get("1.0", tk.END).strip()
        self.log(f"[sandbox] Code length: {len(code)} chars")
        if not code:
            self._show_output("No code to execute.", is_error=True)
            return

        # Update UI
        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self.status_var.set("Executing...")
        self._clear_output()

        # Disable plot buttons until we know if there's a plot
        self.save_plot_btn.config(state=tk.DISABLED)
        self.open_plot_btn.config(state=tk.DISABLED)

        # Run in separate thread to keep UI responsive
        import threading
        threading.Thread(target=self._execute_code, args=(code,), daemon=True).start()

    def _detect_plotting(self, code: str) -> bool:
        """Detect if code uses matplotlib plotting"""
        plotting_indicators = [
            'matplotlib',
            'plt.show',
            'plt.plot',
            'plt.figure',
            'plt.subplot',
            'plt.savefig',
            '.plot(',
            'pyplot',
            'plt.bar',
            'plt.scatter',
            'plt.hist',
            'plt.imshow',
            'plt.contour',
            'ax.plot',
            'ax.bar',
            'fig,',
            'fig =',
        ]
        code_lower = code.lower()
        return any(indicator.lower() in code_lower for indicator in plotting_indicators)

    def _wrap_code_for_plotting(self, code: str) -> tuple:
        """Wrap code to save plot to temp file AND show interactively"""
        # Create a unique temp file path for the plot
        plot_path = os.path.join(tempfile.gettempdir(), f"sandbox_plot_{os.getpid()}.png")

        # Convert to forward slashes for safe embedding in Python string
        # (Python handles forward slashes fine on Windows)
        plot_path_safe = plot_path.replace('\\', '/')

        # Inject code to save figure before plt.show()
        wrapped = f'''
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt

# Plot save path
_plot_save_path = "{plot_path_safe}"

# Store original show function
_original_show = plt.show

def _patched_show(*args, **kwargs):
    # Save all figures before showing
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.savefig(_plot_save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print("[Plot auto-saved to:", _plot_save_path, "]")
    # Then show interactively (user can interact/save from matplotlib window)
    _original_show(*args, **kwargs)

plt.show = _patched_show

# === USER CODE BELOW ===
{code}
'''
        return wrapped, plot_path

    def _wrap_code_for_safety(self, code: str) -> str:
        """NO WRAPPER - just run the code directly"""
        return code.strip()

    def _execute_code(self, code: str, retry_count=0):
        """Actually execute the code (run in thread)"""
        try:
            # Check if code contains plotting
            has_plotting = self._detect_plotting(code)

            # Prepare code - inject plot saving if matplotlib is used
            if has_plotting:
                safe_code, plot_path = self._wrap_code_for_plotting(code)
            else:
                safe_code = self._wrap_code_for_safety(code)
                plot_path = None

            # Write to temp file with UTF-8 encoding
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False,
                    encoding='utf-8'
            ) as f:
                f.write(safe_code)
                temp_path = f.name

            try:
                # Use much longer timeout for plotting (user interaction), or no timeout
                # For plots: 300 seconds (5 min) to allow interaction
                # For non-plots: 30 seconds
                timeout = 300 if has_plotting else 30

                # Execute
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=timeout,
                    cwd=os.path.dirname(temp_path)
                )

                # Check for missing module error
                missing_module = self._detect_missing_module(result.stderr)
                if missing_module and retry_count < 3:
                    self.master.after(0, self._update_status, f"Installing {missing_module}...")
                    if self._install_module(missing_module):
                        # Retry execution after install
                        self.master.after(0, self._update_status, f"Installed {missing_module}, retrying...")
                        os.unlink(temp_path)
                        self._execute_code(code, retry_count + 1)
                        return

                # Show results in UI thread
                self.master.after(0, self._show_execution_results, result, plot_path)

            except subprocess.TimeoutExpired:
                self.master.after(0, self._show_output,
                                  f"Timeout: Code execution exceeded {timeout}s limit.\n"
                                  f"Tip: Close the plot window to complete execution.", True)

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
            self.master.after(0, self._show_output, error_msg, True)

        finally:
            # Reset UI in main thread
            self.master.after(0, self._reset_ui_after_execution)

    def _detect_missing_module(self, stderr: str) -> str:
        """Detect if error is due to missing module, return module name or None"""
        if not stderr:
            return None

        # Match patterns like: ModuleNotFoundError: No module named 'scipy'
        patterns = [
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
            r"ImportError: No module named ['\"]([^'\"]+)['\"]",
            r"No module named ['\"]([^'\"]+)['\"]",
        ]

        for pattern in patterns:
            match = re.search(pattern, stderr)
            if match:
                module = match.group(1)
                # Get the base package name (e.g., 'scipy.signal' -> 'scipy')
                base_module = module.split('.')[0]
                return base_module

        return None

    def _install_module(self, module_name: str) -> bool:
        """Attempt to pip install a module"""
        self.log(f"[sandbox] Attempting to install: {module_name}")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', module_name],
                capture_output=True,
                text=True,
                timeout=120  # 2 min timeout for install
            )
            if result.returncode == 0:
                self.log(f"[sandbox] Successfully installed: {module_name}")
                return True
            else:
                self.log(f"[sandbox] Failed to install {module_name}: {result.stderr}")
                return False
        except Exception as e:
            self.log(f"[sandbox] Install error: {e}")
            return False

    def _update_status(self, message: str):
        """Update status from any thread"""
        self.status_var.set(message)

    def _show_execution_results(self, result, plot_path=None):
        """Display execution results and optionally send to AI"""
        output = ""
        has_meaningful_output = False

        if result.stdout:
            stdout_clean = result.stdout.encode('ascii', 'ignore').decode('ascii').strip()
            if stdout_clean:
                output += "=== STDOUT ===\n"
                output += stdout_clean
                if not stdout_clean.endswith('\n'):
                    output += '\n'
                has_meaningful_output = True

        if result.stderr:
            stderr_clean = result.stderr.encode('ascii', 'ignore').decode('ascii').strip()
            if stderr_clean:
                output += "\n=== STDERR ===\n"
                output += stderr_clean
                has_meaningful_output = True

        # Store last output for manual sending
        self._last_output = output.strip() if has_meaningful_output else ""
        self._last_had_error = (result.returncode != 0)

        if result.returncode != 0:
            self._show_output(output, is_error=True)
            self.status_var.set(f"Failed (exit code: {result.returncode})")
        else:
            self._show_output(output, is_error=False)
            self.status_var.set("Success")

        # Check if plot was saved and enable buttons
        if plot_path and os.path.exists(plot_path):
            self.last_plot_path = plot_path
            self.save_plot_btn.config(state=tk.NORMAL)
            self.open_plot_btn.config(state=tk.NORMAL)
            self.status_var.set("Success - Plot saved!")
        else:
            self.last_plot_path = None

        # Auto-send to AI if enabled and there's meaningful output
        if has_meaningful_output and self.auto_send_var.get():
            self._send_output_to_ai()

    def _send_output_to_ai(self):
        """Send the last output to the main app's text box and optionally auto-send"""
        if not hasattr(self, '_last_output') or not self._last_output:
            self.status_var.set("No output to send")
            self.log("[sandbox] No output to send to AI")
            return

        if not self.output_callback:
            self.status_var.set("No AI connection")
            self.log("[sandbox] No output callback configured")
            return

        try:
            # Format the message for the AI
            if self._last_had_error:
                message = f" The code produced errors. Fix them and relist ENTIRE code, not partial:\n\n{self._last_output}"
            else:
                message = f"Here is the output from my code:\n\n{self._last_output}"

            # Send to main app
            self.output_callback(message, auto_send=self.auto_send_var.get())
            self.status_var.set("Sent to AI")
            self.log(f"[sandbox] Output sent to AI ({len(self._last_output)} chars)")

        except Exception as e:
            self.status_var.set("Send failed")
            self.log(f"[sandbox] Failed to send to AI: {e}")

    def _save_plot(self):
        """Save the last generated plot to a user-chosen location"""
        if not self.last_plot_path or not os.path.exists(self.last_plot_path):
            self.status_var.set("No plot to save")
            return

        # Ask user where to save
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("PDF Document", "*.pdf"),
                ("SVG Vector", "*.svg"),
                ("All Files", "*.*")
            ],
            title="Save Plot As"
        )

        if save_path:
            try:
                import shutil
                shutil.copy2(self.last_plot_path, save_path)
                self.status_var.set(f"Plot saved to {os.path.basename(save_path)}")
                self.log(f"[sandbox] Plot saved to: {save_path}")
            except Exception as e:
                self.status_var.set(f"Save failed: {e}")

    def _open_plot(self):
        """Open the last generated plot in default viewer"""
        if not self.last_plot_path or not os.path.exists(self.last_plot_path):
            self.status_var.set("No plot to open")
            return

        try:
            if sys.platform == 'win32':
                os.startfile(self.last_plot_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', self.last_plot_path])
            else:
                subprocess.run(['xdg-open', self.last_plot_path])
            self.status_var.set("Plot opened in viewer")
        except Exception as e:
            self.status_var.set(f"Could not open: {e}")

    def _show_output(self, text: str, is_error=False):
        """Show output in the output text area"""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)

        text_clean = text.encode('ascii', 'ignore').decode('ascii')

        if is_error:
            self.output_text.tag_configure("error", foreground="#ff6b6b")
            self.output_text.insert("1.0", text_clean, "error")
        else:
            self.output_text.tag_configure("success", foreground="#50c878")
            self.output_text.insert("1.0", text_clean, "success")

        self.output_text.config(state="disabled")
        self.output_text.see("1.0")

    def _clear_output(self):
        """Clear the output text area"""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state="disabled")

    def _reset_ui_after_execution(self):
        """Reset UI after code execution"""
        self.run_btn.config(state=tk.NORMAL, text="Run Code")

    def _copy_code(self):
        """Copy code to clipboard"""
        code = self.code_text.get("1.0", tk.END).strip()
        if code:
            self.clipboard_clear()
            self.clipboard_append(code)
            self.status_var.set("Copied to clipboard")
            self.log("[sandbox] Code copied to clipboard")

    def _clear_code(self):
        """Clear the code editor"""
        self.code_text.delete("1.0", tk.END)
        self._clear_output()
        self.status_var.set("Cleared")
        self.log("[sandbox] Code cleared")

    def show(self):
        """Show the code window"""
        self.deiconify()
        self.lift()
        self.focus_set()
        self.code_text.focus_set()

    def hide(self):
        """Hide the code window"""
        self.withdraw()

    def log(self, message):
        """Log messages through callback"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
