"""
latex_plot_tester.py
Test program for LaTeX expression plotting - ENHANCED VERSION
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import os

# Import your plotter and router
from plotter import Plotter
from router import CommandRouter


class LaTeXPlotTester:
    """Test harness for LaTeX plotting functionality"""

    def __init__(self, root):
        self.root = root
        self.root.title("LaTeX Plot Tester - Enhanced")
        self.root.geometry("1400x900")

        # Initialize plotter
        self.plotter = Plotter(root, log_fn=self.log)

        # Initialize router
        self.router = CommandRouter(app_instance=self)

        # Test cases
        self.test_cases = self.create_test_cases()

        self.create_ui()

    def create_test_cases(self):
        """Create comprehensive test cases for LaTeX expressions"""
        return {
            # ========== BASIC POLYNOMIAL TESTS ==========
            "Simple Polynomial": r"$x^2 + 3x + 2$",
            "Boxed Polynomial": r"$\boxed{x^2 - 4x + 3}$",
            "Fraction": r"$\frac{x^2}{2} + \frac{3x}{4}$",
            "Boxed Fraction": r"$\boxed{\frac{x^4}{4} - \frac{x^3}{3}}$",

            # ========== EXPONENTIAL & TRIG TESTS ==========
            "Exponential": r"$e^{-x}$",
            "Boxed Exponential": r"$\boxed{2e^{-0.3x}}$",
            "Decay Function": r"$\boxed{1 - e^{-0.5x}}$",
            "Trig Function": r"$\sin(x) + \cos(2x)$",
            "Boxed Trig": r"$\boxed{2\sin(x) - \cos(x)}$",

            # ========== LOGARITHM TESTS ==========
            "Logarithm": r"$\ln(x) + 2$",
            "Boxed Log": r"$\boxed{x\ln(x) - x}$",
            "Log with Abs": r"$\boxed{\log|x|}$",
            "Log Abs Complex": r"$\boxed{\ln|x+1| + \frac{1}{x+1}}$",

            # ========== TIME DOMAIN TESTS (t) ==========
            "Time Exponential Decay": r"$y(t) = e^{-t}$",
            "Time Boxed Decay": r"$\boxed{y(t) = 10 - 10e^{-t}}$",
            "Time Step Response": r"$\boxed{1 - e^{-0.5t}}$",
            "Time Damped Oscillation": r"$y(t) = e^{-t}\sin(4t)$",
            "Time Boxed Damped": r"$\boxed{e^{-3t}(4\cos(4t) - 3\sin(4t))}$",
            "Time Complex Response": r"$\boxed{10 - \frac{10}{9}e^{-t} + \frac{100}{9}e^{-10t}}$",
            "Time with Parentheses": r"$\boxed{e^{-2t}(-3\sin(5t) + 4\cos(5t))}$",

            # ========== LAMBDA TESTS (eigenvalues) ==========
            "Lambda Polynomial": r"$\lambda^2 + 3\lambda + 2$",
            "Lambda Boxed": r"$\boxed{\lambda^3 - 4\lambda^2 + 5\lambda - 2}$",
            "Lambda Characteristic": r"$\boxed{\lambda^2 - 5\lambda + 6 = 0}$",
            "Lambda Eigenvalue": r"Result: $\boxed{\lambda = -3}$ or $\boxed{\lambda = -1}$",
            "Lambda Exponential": r"$\boxed{e^{\lambda t}}$",
            "Lambda Greek": r"$\boxed{\lambda^2 + 2\lambda + 1}$",

            # ========== THETA TESTS (angular) ==========
            "Theta Sine": r"$\sin(\theta)$",
            "Theta Boxed": r"$\boxed{r(\theta) = 2\cos(\theta)}$",
            "Theta Polar": r"$\boxed{r = 1 + \cos(\theta)}$",
            "Theta Complex": r"$\boxed{\theta^2 + \sin(2\theta)}$",
            "Theta Greek": r"$\boxed{\theta^2 + 3\theta + 2}$",

            # ========== DISCRETE VARIABLE TESTS (k, n, m) ==========
            "Discrete k Exponential": r"$y[k] = 0.5^k$",
            "Discrete k Boxed": r"$\boxed{a_k = 2^k + k^2}$",
            "Discrete n Sequence": r"$\boxed{x[n] = 0.8^n}$",
            "Discrete n Polynomial": r"$\boxed{a_n = n^2 + 3n + 1}$",
            "Discrete m Series": r"$\boxed{s_m = \frac{1}{m^2}}$",
            "Discrete Mixed": r"$\boxed{y[k] = k \cdot 0.9^k}$",

            # ========== INDEX VARIABLE TESTS (i, j) ==========
            "Index i Simple": r"$\boxed{a_i = 2i + 1}$",
            "Index i Exponential": r"$\boxed{x_i = e^{-i}}$",
            "Index j Sequence": r"$\boxed{b_j = j^2 - j}$",

            # ========== INTEGRATION RESULTS ==========
            "Integration with C": r"$\boxed{\frac{x^4}{4} + C}$",
            "Integration Time": r"$\int x^3 \, dx = \frac{x^4}{4} + C$",
            "Integration Trig": r"$\boxed{\int \sin(x) \, dx = -\cos(x) + C}$",
            "Integration by Parts": r"\[\int x \ln x \, dx = \boxed{\frac{x^2}{2} \ln x - \frac{x^2}{4} + C}\]",

            # ========== DIFFERENTIATION RESULTS ==========
            "Derivative Result": r"$f'(x) = 2x$",
            "Derivative Boxed": r"$\boxed{f'(x) = 3x^2 - 6x}$",
            "Derivative Time": r"$\boxed{\frac{dy}{dt} = -e^{-t}}$",
            "Derivative Complex": r"$\boxed{f'(t) = e^{-t}(4\cos(4t) - 3\sin(4t))}$",

            # ========== MIXED VARIABLE EXPRESSIONS ==========
            "Mixed xlog": r"$\boxed{x\ln(x) - x}$",
            "Mixed xsin": r"$x \cdot \sin(x) + \cos(x)$",
            "Mixed tsin": r"$t \cdot \sin(2t)$",

            # ========== EDGE CASES ==========
            "With left/right": r"$\boxed{e^{-3x} \left( -3\cos(4x) - 4\sin(4x) \right)}$",
            "Multiple Boxed": r"Result: $\boxed{x^3}$ or $\boxed{x^2 + x}$",
            "Equation Form": r"$y = x^2 - 4$",
            "Function Notation": r"$f(x) = \sin(x) + x$",
            "With Text Before": r"The derivative is $\boxed{2x}$",
            "Nested Fractions": r"$\boxed{\frac{\frac{x^2}{2}}{x+1}}$",

            # ========== ABSOLUTE VALUE & SPECIAL FUNCTIONS ==========
            "Absolute Value": r"$\boxed{|x^2 - 4|}$",
            "Arctangent": r"$\boxed{\arctan(x)}$",
            "Square Root": r"$\boxed{\sqrt{x^2 + 1}}$",
            "Rational Function": r"$\boxed{\frac{x+1}{x-1}}$",

            # ========== SUBSTITUTION VARIABLE TESTS (should be skipped) ==========
            "Substitution u": r"Let $u = x + 1$, then $\boxed{y = u^2}$",
            "Substitution v": r"With $v = x + 1$, we get $\boxed{v^2 - 1}$",

            # ========== INEQUALITY TESTS (should be skipped) ==========
            "Inequality geq": r"For $t \geq 0$, we have $\boxed{e^{-t}}$",
            "Inequality leq": r"When $x \leq 5$, plot $\boxed{x^2}$",

            # ========== HARD TEST CASES - COMPLEX EXPRESSIONS ==========
            "Hard: Arctan with Sqrt": r"$$\boxed{x - \sqrt{5} \arctan\left(\frac{x}{\sqrt{5}}\right) + C}$$",
            "Hard: Signal Modulation": r"\[\boxed{s(t) = \left[5 + 2 \cos(2\pi \cdot 5000 \cdot t)\right] \cos(2\pi \cdot 100000 \cdot t)}\]",
            "Hard: Sine with Pi": r"\[\boxed{\sin(4\pi t)}\]",
            "Hard: Sin Squared": r"$$\boxed{\sin^2(x)}$$",
            "Hard: Sin Squared No Parens": r"$$\boxed{\sin^2 x}$$",
            "Hard: Cos Cubed": r"$$\boxed{\cos^3(2x)}$$",
            "Hard: Tan Squared": r"$$\boxed{\tan^2(\theta)}$$",

            # ========== HARD: NESTED FUNCTIONS ==========
            "Hard: Log of Sin": r"$$\boxed{\ln(\sin(x))}$$",
            "Hard: Exp of Cos": r"$$\boxed{e^{\cos(x)}}$$",
            "Hard: Sin of Log": r"$$\boxed{\sin(\ln(x))}$$",
            "Hard: Sqrt of Exp": r"$$\boxed{\sqrt{e^x + e^{-x}}}$$",

            # ========== HARD: MULTI-TERM WITH PI ==========
            "Hard: Pi Polynomial": r"$$\boxed{x^2 + \pi x + \frac{\pi^2}{4}}$$",
            "Hard: Pi Trig": r"$$\boxed{\sin\left(\frac{\pi}{2}x\right) + \cos(\pi x)}$$",
            "Hard: Pi Time": r"$$\boxed{e^{-\pi t} \sin(2\pi t)}$$",
            "Hard: E and Pi": r"$$\boxed{e^{-t} + \pi t^2}$$",

            # ========== HARD: SQUARE BRACKETS ==========
            "Hard: Brackets Simple": r"$$\boxed{[x + 1][x - 1]}$$",
            "Hard: Brackets Nested": r"$$\boxed{[5 + 2x] \cdot [3 - x]}$$",
            "Hard: Brackets Trig": r"$$\boxed{[1 + \sin(x)] \cos(x)}$$",

            # ========== HARD: COMPLEX FRACTIONS ==========
            "Hard: Frac in Frac": r"$$\boxed{\frac{1}{\frac{1}{x} + \frac{1}{x+1}}}$$",
            "Hard: Frac with Sqrt": r"$$\boxed{\frac{\sqrt{x^2 + 1}}{x}}$$",
            "Hard: Frac with Trig": r"$$\boxed{\frac{\sin(x)}{1 + \cos(x)}}$$",
            "Hard: Frac with Log": r"$$\boxed{\frac{\ln(x)}{x}}$$",

            # ========== HARD: PRODUCTS AND CHAINS ==========
            "Hard: Product Chain": r"$$\boxed{x \cdot e^x \cdot \sin(x)}$$",
            "Hard: Long Product": r"$$\boxed{(x+1)(x-1)(x+2)(x-2)}$$",
            "Hard: Trig Chain": r"$$\boxed{\sin(x) \cos(x) \tan(x)}$$",

            # ========== HARD: PIECEWISE-LOOKING (but continuous) ==========
            "Hard: Abs Polynomial": r"$$\boxed{|x^3 - 3x^2 + 2x|}$$",
            "Hard: Abs Trig": r"$$\boxed{|\sin(x)| + |\cos(x)|}$$",

            # ========== HARD: LAMBDA AND THETA COMBINATIONS ==========
            "Hard: Lambda Pi": r"$$\boxed{\lambda^2 + \pi\lambda + \frac{\pi^2}{4}}$$",
            "Hard: Theta Pi": r"$$\boxed{\sin(\pi\theta) + \cos(2\pi\theta)}$$",
            "Hard: Lambda Trig": r"$$\boxed{e^{\lambda} \sin(\theta)}$$",

            # ========== HARD: VERY LONG EXPRESSIONS ==========
            "Hard: Long Polynomial": r"$$\boxed{x^5 - 3x^4 + 2x^3 - 5x^2 + x - 7}$$",
            "Hard: Long Trig Sum": r"$$\boxed{\sin(x) + \sin(2x) + \sin(3x) + \sin(4x)}$$",
            "Hard: Long Exponential": r"$$\boxed{e^{-x} + 2e^{-2x} + 3e^{-3x} + 4e^{-4x}}$$",

            # ========== HARD: DAMPED OSCILLATIONS ==========
            "Hard: Damped Sine": r"$$\boxed{e^{-0.1t} \sin(2\pi t)}$$",
            "Hard: Damped Cosine": r"$$\boxed{e^{-\lambda t} \cos(\omega t + \phi)}$$",
            "Hard: Beat Frequency": r"$$\boxed{\sin(10t) \cdot \sin(12t)}$$",

            # ========== HARD: RATIONAL WITH COMPLEX PARTS ==========
            "Hard: Rational Complex Num": r"$$\boxed{\frac{x^3 - 3x^2 + 2}{x^2 + 1}}$$",
            "Hard: Rational Trig Num": r"$$\boxed{\frac{x \sin(x)}{x^2 + 1}}$$",
            "Hard: Rational Exp": r"$$\boxed{\frac{e^x - e^{-x}}{e^x + e^{-x}}}$$",

            # ========== HARD: PREVIOUSLY FAILED CASES ==========
            "Hard: Multi-Variable Lambda-t": r"$$\boxed{e^{\lambda t}}$$",
            "Hard: Function with Text": r"$$\boxed{f(x) = x \cdot (-0.5)^x}$$",
            "Hard: Discrete Notation": r"\[\boxed{f[n] = n (-0.5)^n \quad \text{for } n \geq 0}\]",
        }

    def run_variable_tests(self):
        """Run comprehensive variable tests and report results"""
        self.log(f"\n{'=' * 80}")
        self.log(f"[VARIABLE TEST] Starting comprehensive variable testing")
        self.log(f"{'=' * 80}\n")

        variable_tests = {
            "x": [r"$x^2 + 3x + 2$", r"$\boxed{x^3}$"],
            "t": [r"$e^{-t}$", r"$\boxed{t^2 + t}$", r"$y(t) = 1 - e^{-t}$"],
            "lambda": [r"$\lambda^2 + 2\lambda + 1$", r"$\boxed{\lambda^3 - \lambda}$"],
            "λ": [r"$\boxed{\lambda^2 + 3\lambda + 2}$"],
            "theta": [r"$\sin(\theta)$", r"$\boxed{\theta^2 + \cos(\theta)}$"],
            "θ": [r"$\boxed{\theta^2 + \theta}$"],
            "k": [r"$0.5^k$", r"$\boxed{k^2 + k}$"],
            "n": [r"$\boxed{0.8^n}$", r"$n^2 + 2n + 1$"],
            "m": [r"$\boxed{m^3 - m}$"],
            "i": [r"$\boxed{2i + 1}$"],
            "j": [r"$\boxed{j^2 - j}$"],
        }

        results = {"passed": 0, "failed": 0, "details": []}

        for var, test_cases in variable_tests.items():
            self.log(f"\n[VAR TEST] Testing variable: '{var}'")
            for latex in test_cases:
                try:
                    expressions = self.plotter.extract_math_expressions(latex)
                    if expressions:
                        self.log(f"[VAR TEST]   ✓ PASS: {latex}")
                        self.log(f"[VAR TEST]      Extracted: {expressions[0]}")
                        results["passed"] += 1
                        results["details"].append((var, latex, "PASS", expressions[0]))
                    else:
                        self.log(f"[VAR TEST]   ❌ FAIL: {latex}")
                        self.log(f"[VAR TEST]      No expressions extracted")
                        results["failed"] += 1
                        results["details"].append((var, latex, "FAIL", "No extraction"))
                except Exception as e:
                    self.log(f"[VAR TEST]   ❌ ERROR: {latex}")
                    self.log(f"[VAR TEST]      {str(e)}")
                    results["failed"] += 1
                    results["details"].append((var, latex, "ERROR", str(e)))

        # Summary
        total = results["passed"] + results["failed"]
        self.log(f"\n{'=' * 80}")
        self.log(f"[VAR TEST] SUMMARY")
        self.log(f"{'=' * 80}")
        self.log(f"[VAR TEST] Total tests: {total}")
        self.log(f"[VAR TEST] Passed: {results['passed']} ({100 * results['passed'] // total if total else 0}%)")
        self.log(f"[VAR TEST] Failed: {results['failed']} ({100 * results['failed'] // total if total else 0}%)")
        self.log(f"{'=' * 80}\n")

        return results



    def create_ui(self):
        """Create the test UI"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Top section - Test cases
        top_frame = ttk.LabelFrame(main_frame, text="Test Cases", padding="10")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(
            top_frame,
            text="🧪 Test Variables",
            command=self.run_variable_tests
        ).grid(row=0, column=5, padx=5)

        # Test case selector
        ttk.Label(top_frame, text="Select Test Case:").grid(row=0, column=0, padx=5)

        self.test_case_var = tk.StringVar()
        test_combo = ttk.Combobox(
            top_frame,
            textvariable=self.test_case_var,
            values=list(self.test_cases.keys()),
            width=40,
            state="readonly"
        )
        test_combo.grid(row=0, column=1, padx=5)
        test_combo.current(0)
        test_combo.bind("<<ComboboxSelected>>", self.on_test_case_selected)

        ttk.Button(
            top_frame,
            text="📈 Plot Selected",
            command=self.plot_selected_test
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            top_frame,
            text="🔄 Plot All",
            command=self.plot_all_tests
        ).grid(row=0, column=3, padx=5)

        ttk.Button(
            top_frame,
            text="🗑️ Clear Plots",
            command=self.clear_plots
        ).grid(row=0, column=4, padx=5)

        # Custom input section
        custom_frame = ttk.LabelFrame(main_frame, text="Custom LaTeX Input", padding="10")
        custom_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        custom_frame.columnconfigure(0, weight=1)
        custom_frame.rowconfigure(0, weight=1)

        # Input text area
        input_container = ttk.Frame(custom_frame)
        input_container.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        input_container.columnconfigure(0, weight=1)
        input_container.rowconfigure(0, weight=1)

        self.input_text = scrolledtext.ScrolledText(
            input_container,
            height=4,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.input_text.grid(row=0, column=0, sticky="nsew")

        # Buttons for custom input
        btn_frame = ttk.Frame(custom_frame)
        btn_frame.grid(row=1, column=0, sticky="ew")

        ttk.Button(
            btn_frame,
            text="📈 Plot Custom",
            command=self.plot_custom
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="🧪 Test Extraction",
            command=self.test_extraction
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="📋 Load from Clipboard",
            command=self.load_clipboard
        ).pack(side="left", padx=5)

        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="10")
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#00ff00"
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Clear log button
        ttk.Button(
            log_frame,
            text="Clear Log",
            command=self.clear_log
        ).grid(row=1, column=0, sticky="e", pady=(5, 0))

        # Load first test case
        self.on_test_case_selected(None)

    def on_test_case_selected(self, event):
        """Load selected test case into input"""
        test_name = self.test_case_var.get()
        if test_name in self.test_cases:
            latex_code = self.test_cases[test_name]
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", latex_code)
            self.log(f"[test] Loaded: {test_name}")

    def plot_selected_test(self):
        """Plot the currently selected test case"""
        test_name = self.test_case_var.get()
        latex_code = self.test_cases.get(test_name, "")

        if latex_code:
            self.log(f"\n{'=' * 60}")
            self.log(f"[test] Testing: {test_name}")
            self.log(f"[test] LaTeX: {latex_code}")
            self.log(f"{'=' * 60}")

            self.plot_latex(latex_code)

    def plot_custom(self):
        """Plot custom LaTeX input"""
        latex_code = self.input_text.get("1.0", "end-1c").strip()

        if latex_code:
            self.log(f"\n{'=' * 60}")
            self.log(f"[custom] Testing custom input")
            self.log(f"[custom] LaTeX: {latex_code}")
            self.log(f"{'=' * 60}")

            self.plot_latex(latex_code)
        else:
            self.log("[custom] ❌ No input provided")

    def test_extraction(self):
        """Test extraction without plotting"""
        latex_code = self.input_text.get("1.0", "end-1c").strip()

        if latex_code:
            self.log(f"\n{'=' * 60}")
            self.log(f"[extract] Testing extraction")
            self.log(f"[extract] Input: {latex_code}")
            self.log(f"{'=' * 60}")

            # Test plotter's extraction
            expressions = self.plotter.extract_math_expressions(latex_code)

            self.log(f"[extract] ✓ Found {len(expressions)} expressions:")
            for i, expr in enumerate(expressions, 1):
                self.log(f"[extract]   {i}. {expr}")

                # Also test parsing
                try:
                    from plotter import PlotWindow
                    window = PlotWindow(self.root, [expr], self.log)
                    func, latex_str, sympy_expr = window.parse_expression(expr)

                    if func:
                        self.log(f"[extract]      ✓ Parsed successfully")
                        self.log(f"[extract]      LaTeX: {latex_str}")
                        self.log(f"[extract]      SymPy: {sympy_expr}")
                    else:
                        self.log(f"[extract]      ❌ Failed to parse")

                    window.destroy()
                except Exception as e:
                    self.log(f"[extract]      ❌ Error: {e}")
        else:
            self.log("[extract] ❌ No input provided")

    def plot_latex(self, latex_code):
        """Plot LaTeX expressions"""
        try:
            # Extract expressions using plotter
            expressions = self.plotter.extract_math_expressions(latex_code)

            if expressions:
                self.log(f"[plot] ✓ Extracted {len(expressions)} expressions:")
                for i, expr in enumerate(expressions, 1):
                    self.log(f"[plot]   {i}. {expr}")

                # Create plot
                plot_window = self.plotter.plot_from_text(latex_code, expressions)

                if plot_window:
                    self.log(f"[plot] ✓ Plot created successfully")
                else:
                    self.log(f"[plot] ❌ Failed to create plot")
            else:
                self.log(f"[plot] ❌ No expressions found in: {latex_code}")

        except Exception as e:
            self.log(f"[plot] ❌ Error: {e}")
            import traceback
            self.log(f"[plot] Traceback:\n{traceback.format_exc()}")

    def plot_all_tests(self):
        """Plot all test cases sequentially"""
        self.log(f"\n{'=' * 60}")
        self.log(f"[batch] Plotting all {len(self.test_cases)} test cases")
        self.log(f"{'=' * 60}\n")

        for i, (name, latex_code) in enumerate(self.test_cases.items(), 1):
            self.log(f"\n[batch] Test {i}/{len(self.test_cases)}: {name}")
            self.log(f"[batch] LaTeX: {latex_code}")

            try:
                expressions = self.plotter.extract_math_expressions(latex_code)

                if expressions:
                    self.log(f"[batch] ✓ Extracted: {expressions}")
                else:
                    self.log(f"[batch] ❌ No expressions found")

            except Exception as e:
                self.log(f"[batch] ❌ Error: {e}")

            # Small delay between tests
            self.root.update()

        self.log(f"\n[batch] ✓ Batch test complete")

    def clear_plots(self):
        """Close all plot windows"""
        self.plotter.close_all()
        self.log("[ui] All plot windows closed")

    def load_clipboard(self):
        """Load LaTeX from clipboard"""
        try:
            clipboard_text = self.root.clipboard_get()
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", clipboard_text)
            self.log(f"[clipboard] Loaded {len(clipboard_text)} characters")
        except Exception as e:
            self.log(f"[clipboard] ❌ Error: {e}")

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete("1.0", "end")

    def log(self, message):
        """Log a message"""
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.root.update()
        print(message)  # Also print to console

    # Dummy methods needed by router/plotter
    def logln(self, message):
        """Compatibility method for router"""
        self.log(message)

    def speak_search_status(self, message):
        """Compatibility method for router"""
        self.log(f"[speak] {message}")

    def play_chime(self, **kwargs):
        """Compatibility method for router"""
        pass

    def set_light(self, state):
        """Compatibility method for router"""
        pass


def main():
    """Run the test application"""
    root = tk.Tk()
    app = LaTeXPlotTester(root)

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()