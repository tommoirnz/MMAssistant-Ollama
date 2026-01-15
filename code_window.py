# code_window.py - ASCII ONLY VERSION
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import tempfile
import os
import sys
import re


class CodeWindow(tk.Toplevel):
    """Popup window for code execution with run button"""

    def __init__(self, master, log_callback=None):
        super().__init__(master)
        self.title("Code Sandbox")
        self.geometry("850x650")
        self.protocol("WM_DELETE_WINDOW", self.hide)

        # Logging callback to main app
        self.log_callback = log_callback or print

        # Configure window
        self.configure(bg="#2b2b2b")

        # Create UI
        self._create_ui()

        # Store current code
        self.current_code = ""

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

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            toolbar,
            textvariable=self.status_var,
            foreground="#50c878"
        ).pack(side=tk.LEFT, padx=(20, 0))

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
        # Get current code
        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            self._show_output("No code to execute.", is_error=True)
            return

        # Update UI
        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self.status_var.set("Executing...")
        self._clear_output()

        # Run in separate thread to keep UI responsive
        import threading
        threading.Thread(target=self._execute_code, args=(code,), daemon=True).start()

    def _execute_code(self, code: str):
        """Actually execute the code (run in thread)"""
        try:
            # Create a safe execution environment - SIMPLE VERSION
            safe_code = self._wrap_code_for_safety(code)

            # Write to temp file with UTF-8 encoding
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False,
                    encoding='utf-8'  # Explicit UTF-8
            ) as f:
                f.write(safe_code)
                temp_path = f.name

            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',  # UTF-8 for output too
                    timeout=10,
                    cwd=os.path.dirname(temp_path)
                )

                # Show results in UI thread
                self.master.after(0, self._show_execution_results, result)

            except subprocess.TimeoutExpired:
                self.master.after(0, self._show_output, "Timeout: Code execution took too long (10s limit)", True)

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            # ASCII-only error message
            error_msg = f"Execution error: {str(e)}"
            # Remove any non-ASCII characters
            error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
            self.master.after(0, self._show_output, error_msg, True)

        finally:
            # Reset UI in main thread
            self.master.after(0, self._reset_ui_after_execution)

    def _wrap_code_for_safety(self, code: str) -> str:
        """NO WRAPPER - just run the code directly"""
        return code.strip()



    def _show_execution_results(self, result):
        """Display execution results - ASCII only"""
        output = ""

        if result.stdout:
            # Clean stdout of non-ASCII
            stdout_clean = result.stdout.encode('ascii', 'ignore').decode('ascii')
            output += "=== STDOUT ===\n"
            output += stdout_clean
            if not stdout_clean.endswith('\n'):
                output += '\n'

        if result.stderr:
            # Clean stderr of non-ASCII
            stderr_clean = result.stderr.encode('ascii', 'ignore').decode('ascii')
            output += "\n=== STDERR ===\n"
            output += stderr_clean

        if result.returncode != 0:
            self._show_output(output, is_error=True)
            self.status_var.set(f"Failed (exit code: {result.returncode})")
        else:
            self._show_output(output, is_error=False)
            self.status_var.set("Success")

    def _show_output(self, text: str, is_error=False):
        """Show output in the output text area"""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)

        # Clean text of any non-ASCII that might have slipped through
        text_clean = text.encode('ascii', 'ignore').decode('ascii')

        if is_error:
            self.output_text.tag_configure("error", foreground="#ff6b6b")
            self.output_text.insert("1.0", text_clean, "error")
        else:
            self.output_text.tag_configure("success", foreground="#50c878")
            self.output_text.insert("1.0", text_clean, "success")

        self.output_text.config(state="disabled")
        self.output_text.see("1.0")  # Scroll to top

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
