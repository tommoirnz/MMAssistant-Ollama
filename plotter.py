"""
Plotter class for mathematical function plotting with singularity handling.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import sympy as sp
import matplotlib

matplotlib.use('TkAgg')  # Use Tkinter backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import re
import warnings

warnings.filterwarnings('ignore')  # Suppress numpy warnings

# This version has dB plotting and a log scale option for say Bode plots. For x>0 of course only and y >0 for dB
class Plotter:
    """
    Main plotting class that can be called from anywhere in your app.
    """

    def __init__(self, master, log_fn=None):
        """
        Initialize plotter.

        Args:
            master: Parent tkinter window
            log_fn: Logging function (optional)
        """
        self.master = master
        self._log = log_fn or (lambda msg: None)
        self._plot_windows = []  # Track ALL plot windows
        self._current_expressions = []
        self._original_variables = []  # NEW: Track original variables

    def plot_from_text(self, text, expressions=None):
        """
        Extract math expressions from text and plot them.

        Args:
            text: Text to extract expressions from
            expressions: Pre-extracted expressions (optional)

        Returns:
            PlotWindow instance or None
        """
        try:
            # Extract expressions if not provided
            if expressions is None:
                expressions = self.extract_math_expressions(text)

            if not expressions:
                self._log("[plotter] No math expressions found to plot")
                return None

            self._log(f"[plotter] Found {len(expressions)} expressions: {expressions}")

            # Create plot window with original variables
            plot_window = PlotWindow(  # Don't store in self._plot_window
                self.master,
                expressions,
                self._log,
                original_vars=getattr(self, '_original_variables', None)
            )
            plot_window.show()

            # Track this window in our list
            self._plot_windows.append(plot_window)

            # Auto-remove from tracking when window is closed manually
            def on_close(w=plot_window):
                if w in self._plot_windows:
                    self._plot_windows.remove(w)
                w.destroy()

            plot_window.protocol("WM_DELETE_WINDOW", on_close)

            return plot_window


            return self._plot_window

        except Exception as e:
            self._log(f"[plotter] Error: {e}")
            import traceback
            self._log(f"[plotter] Traceback: {traceback.format_exc()}")
            return None

    def extract_math_expressions(self, text):
        """
        Extract mathematical expressions from text - LATEX ONLY VERSION (FINAL).

        Args:
            text: Text content

        Returns:
            List of math expressions
        """
        if not text:
            return []

        expressions = []
        text_remaining = text

        # ========== HELPER: Extract \boxed{} with proper brace counting ==========
        def extract_boxed_content(text):
            """Extract all \boxed{...} content handling nested braces."""
            boxed_contents = []
            search_text = text

            while '\\boxed{' in search_text:
                start = search_text.find('\\boxed{')
                if start == -1:
                    break

                # Count braces to find matching closing brace
                depth = 0
                pos = start + 7  # Skip "\boxed{"
                end = -1

                for i in range(pos, len(search_text)):
                    if search_text[i] == '{':
                        depth += 1
                    elif search_text[i] == '}':
                        if depth == 0:
                            end = i
                            break
                        else:
                            depth -= 1

                if end > start:
                    # Extract full \boxed{...} including braces
                    full_match = search_text[start:end + 1]
                    # Extract just the content inside \boxed{}
                    content = search_text[start + 7:end]
                    boxed_contents.append((full_match, content))
                    # Remove this match and continue searching
                    search_text = search_text[:start] + ' ' * len(full_match) + search_text[end + 1:]
                else:
                    # Malformed, skip it
                    break

            return boxed_contents

        # ========== STEP 1: Extract \boxed{} with nested brace support ==========
        boxed_matches = extract_boxed_content(text)
        for full_match, content in boxed_matches:
            if content.strip():
                expr = self._clean_latex_expression(content.strip())
                if expr and self._looks_like_function(expr):
                    expressions.append(expr)
                    self._log(f"[plotter] LaTeX extracted (boxed): '{content.strip()}' -> '{expr}'")
                # Remove from remaining text
                text_remaining = text_remaining.replace(full_match, ' ' * len(full_match), 1)

        # ========== STEP 2: Extract $$...$$ (display math) ==========
        double_dollar_pattern = r'\$\$(.*?)\$\$'
        for match in re.finditer(double_dollar_pattern, text_remaining):
            latex_content = match.group(1)
            if latex_content.strip():
                expr = self._clean_latex_expression(latex_content.strip())
                if expr and self._looks_like_function(expr):
                    expressions.append(expr)
                    self._log(f"[plotter] LaTeX extracted ($$): '{latex_content.strip()}' -> '{expr}'")
                # Blank out this match
                start_pos = match.start()
                end_pos = match.end()
                text_remaining = text_remaining[:start_pos] + ' ' * (end_pos - start_pos) + text_remaining[end_pos:]

        # ========== STEP 3: Extract $...$ (inline math) ==========
        single_dollar_pattern = r'\$(.*?)\$'
        for match in re.finditer(single_dollar_pattern, text_remaining):
            latex_content = match.group(1)
            if latex_content.strip():
                expr = self._clean_latex_expression(latex_content.strip())
                if expr and self._looks_like_function(expr):
                    expressions.append(expr)
                    self._log(f"[plotter] LaTeX extracted ($): '{latex_content.strip()}' -> '{expr}'")
                # Blank out this match
                start_pos = match.start()
                end_pos = match.end()
                text_remaining = text_remaining[:start_pos] + ' ' * (end_pos - start_pos) + text_remaining[end_pos:]

        # ========== STEP 4: Extract \[...\] (display math) ==========
        bracket_pattern = r'\\\[(.*?)\\\]'
        for match in re.finditer(bracket_pattern, text_remaining):
            latex_content = match.group(1)
            if latex_content.strip():
                expr = self._clean_latex_expression(latex_content.strip())
                if expr and self._looks_like_function(expr):
                    expressions.append(expr)
                    self._log(f"[plotter] LaTeX extracted (\\[\\]): '{latex_content.strip()}' -> '{expr}'")

        # ========== STEP 5: Handle equations (DO NOT normalize variables yet) ==========
        cleaned_expressions = []
        for expr in expressions:
            # CRITICAL: DO NOT call _normalize_variables() here!
            # Variable detection and conversion happens in Step 6

            # Handle equations (extract right side if it's y= or f(x)=)
            if '=' in expr:
                parts = expr.split('=', 1)
                if len(parts) == 2:
                    left, right = parts[0].strip(), parts[1].strip()
                    left_lower = left.lower()

                    # Pattern for f(x), g(t), etc.
                    func_pattern = r'^[a-zA-Z]\([a-zA-Zα-ωΑ-Ω]\)$'

                    if (left_lower in ['y'] or
                            re.match(func_pattern, left) or
                            '(' in left):
                        # Take right side for y= or f(x)=
                        cleaned_expressions.append(right)
                        self._log(f"[plotter] Equation '{expr}' -> using right side: '{right}'")
                    elif right.strip() == '0':
                        # For equations like x^2+3x+2=0, plot the left side
                        cleaned_expressions.append(left)
                        self._log(f"[plotter] Equation '{expr}' -> plotting left side: '{left}'")
                    else:
                        # Default to right side
                        cleaned_expressions.append(right)
                else:
                    cleaned_expressions.append(expr)
            else:
                cleaned_expressions.append(expr)

        # ========== STEP 6: Final cleanup + Variable Conversion ==========
        final_expressions = []
        for expr in cleaned_expressions:
            if self._log:
                self._log(f"[step6] Processing: '{expr}'")

            # Remove integration constants
            expr = re.sub(r'\s*[+\-]\s*C\s*$', '', expr, flags=re.IGNORECASE)

            # ========== DETECT ORIGINAL VARIABLE FIRST (BEFORE e** CONVERSION) ==========
            original_variable = 'x'  # Default

            # Special check for lambda/λ FIRST (because 'lambda' might appear with 'x')
            if 'lambda' in expr.lower() or 'λ' in expr:
                original_variable = 'λ'
            # Then check if 'x' is NOT present for other variables
            # Use word boundaries to avoid matching 'x' in 'exp', 'max', etc.
            elif not re.search(r'\bx\b', expr.lower()):
                if re.search(r'\bt\b', expr.lower()) and 'theta' not in expr.lower():
                    original_variable = 't'
                elif 'theta' in expr.lower():
                    original_variable = 'θ'
                elif 'θ' in expr:
                    original_variable = 'θ'
                elif re.search(r'\bk\b', expr.lower()) and not any(
                        word in expr.lower() for word in ['sin', 'sink', 'kinetic']):
                    original_variable = 'k'
                elif re.search(r'\bn\b', expr.lower()) and 'sin' not in expr.lower():
                    original_variable = 'n'
                elif re.search(r'\bm\b', expr.lower()) and not any(
                        word in expr.lower() for word in ['sin', 'sum', 'min', 'max', 'lim']):
                    original_variable = 'm'
                elif re.search(r'\bi\b', expr.lower()) and not any(
                        word in expr.lower() for word in ['sin', 'min', 'arcsin', 'asin']):
                    original_variable = 'i'
                elif re.search(r'\bj\b', expr.lower()):
                    original_variable = 'j'

                elif 'omega' in expr.lower() or 'ω' in expr:
                    original_variable = 'ω'
                elif re.search(r'\bf\b', expr.lower()):
                    original_variable = 'f'


            # ========== CONVERT VARIABLE TO X (BEFORE e** TO exp CONVERSION) ==========
            # ========== CONVERT VARIABLE TO X (BEFORE e** TO exp CONVERSION) ==========
            if original_variable != 'x':
                if original_variable == 't':
                    # Use word boundaries to only replace standalone 't', not inside function names
                    expr = re.sub(r'\bt\b', 'x', expr)
                    expr = re.sub(r'\bT\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted t→x (original: '{original_variable}')")

                elif original_variable == 'λ':
                    if 'lambda' in expr.lower():
                        expr = re.sub(r'\blambda\b', 'x', expr, flags=re.IGNORECASE)
                    elif 'λ' in expr:
                        expr = expr.replace('λ', 'x')
                    if self._log:
                        self._log(f"[plotter] Converted λ→x (original: '{original_variable}')")

                elif original_variable == 'θ':
                    if 'theta' in expr.lower():
                        expr = re.sub(r'\btheta\b', 'x', expr, flags=re.IGNORECASE)
                    elif 'θ' in expr:
                        expr = expr.replace('θ', 'x')
                    if self._log:
                        self._log(f"[plotter] Converted θ→x (original: '{original_variable}')")

                elif original_variable == 'k':
                    expr = re.sub(r'\bk\b', 'x', expr)
                    expr = re.sub(r'\bK\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted k→x (original: '{original_variable}')")

                elif original_variable == 'n':
                    expr = re.sub(r'\bn\b', 'x', expr)
                    expr = re.sub(r'\bN\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted n→x (original: '{original_variable}')")

                elif original_variable == 'm':
                    expr = re.sub(r'\bm\b', 'x', expr)
                    expr = re.sub(r'\bM\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted m→x (original: '{original_variable}')")

                elif original_variable == 'i':
                    expr = re.sub(r'\bi\b', 'x', expr)
                    expr = re.sub(r'\bI\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted i→x (original: '{original_variable}')")

                elif original_variable == 'j':
                    expr = re.sub(r'\bj\b', 'x', expr)
                    expr = re.sub(r'\bJ\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted j→x (original: '{original_variable}')")

                elif original_variable == 'ω':
                    if 'omega' in expr.lower():
                        expr = re.sub(r'\bomega\b', 'x', expr, flags=re.IGNORECASE)
                    elif 'ω' in expr:
                        expr = expr.replace('ω', 'x')
                    if self._log:
                        self._log(f"[plotter] Converted ω→x (original: '{original_variable}')")

                elif original_variable == 'f':
                    expr = re.sub(r'\bf\b', 'x', expr)
                    expr = re.sub(r'\bF\b', 'X', expr)
                    if self._log:
                        self._log(f"[plotter] Converted f→x (original: '{original_variable}')")


                # ========== SIMPLIFY REDUNDANT PARENTHESES ==========
                # Remove excessive nesting: (((x))) -> (x)
                ####   # Match ((content)) and reduce to (content)
                   # expr = re.sub(r'\(\(([^)]+)\)\)', r'(\1)', expr)
                  #  if expr == before:  # No more changes
                   #     break

            # Fix e**-EXPR/EXPR patterns - wrap full exponent in parens
            # Fix e**-EXPR/EXPR patterns - wrap full exponent in parens
            expr = re.sub(r'e\*\*-([^*/\s]+)/([^*/\s]+)', r'e**(-\1/\2)', expr)
            # Protect function names to prevent regex from breaking them
            protected_funcs = {}
            for i, func in enumerate(
                    ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'atan', 'asin', 'acos', 'pi', 'e']):
                placeholder = f'__FUNC{i}__'
                protected_funcs[placeholder] = func
                expr = expr.replace(func, placeholder)

            # NOTE: Don't convert secondary variables - keep them symbolic
            # If expression has undefined constants, plot will be blank but title shows correct expression

            # Wrap multi-character exponents: e**xx -> e**(xx)
            expr = re.sub(r'e\*\*([a-z]{2,})', r'e**(\1)', expr)

            # Insert * between adjacent letters inside parens: e**(xx) -> e**(x*x)
            expr = re.sub(r'\(([a-z])([a-z])', r'(\1*\2', expr)
            expr = re.sub(r'([a-z])([a-z])\)', r'\1*\2)', expr)

            # Fix logabs -> log(abs(...)) BEFORE other operations
            while 'logabs(' in expr:
                start = expr.find('logabs(')
                depth = 1
                pos = start + 7
                end = -1

                for i in range(pos, len(expr)):
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break

                if end > start:
                    content = expr[start + 7:end]
                    expr = expr[:start] + f'log(abs({content}))' + expr[end + 1:]
                else:
                    expr = expr.replace('logabs(', 'log(abs(', 1)
                    break



            # Fix e** patterns - convert ALL to exp()
            expr = re.sub(r'e\*\*-\(([^)]+)\)', r'exp(-\1)', expr)
            expr = re.sub(r'e\*\*-([a-zA-Z])', r'exp(-\1)', expr)  # NEW: handles e**-x -> exp(-x)
            expr = re.sub(r'e\*\*(-\d+\.?\d*)\*([a-zA-Z])', r'exp(\1*\2)', expr)
            expr = re.sub(r'e\*\*(-\d+\.?\d*)', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*(-[a-zA-Z])', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*\(([^)]+)\)', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*([a-zA-Z0-9]+)', r'exp(\1)', expr)

            # Ensure closing parens for exp
            if 'exp(' in expr:
                open_count = expr.count('(')
                close_count = expr.count(')')
                if open_count > close_count:
                    expr += ')' * (open_count - close_count)

            # Add * between )( and )function
            expr = re.sub(r'(\))(\()', r'\1*\2', expr)
            expr = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr)

            # Fix xlog -> x*log
            expr = re.sub(r'(?<![a-z])([a-zA-Z])(log|sin|cos|tan|exp|sqrt|abs)\b', r'\1*\2', expr)

            # Protect arc/inverse trig functions
            expr = re.sub(r'\ba\*tan\b', 'atan', expr)
            expr = re.sub(r'\ba\*sin\b', 'asin', expr)
            expr = re.sub(r'\ba\*cos\b', 'acos', expr)

            # Fix sqrtx -> sqrt(x)
            if 'sqrt' in expr:
                expr = re.sub(r'sqrt([a-zA-Z])', r'sqrt(\1)', expr)

            # Restore function names (moved to end to avoid regex interference)
            for placeholder, func in protected_funcs.items():
                expr = expr.replace(placeholder, func)

            # Handle func**n(x) patterns: sin**2(x) -> (sin(x))**2
            expr = re.sub(r'(sin|cos|tan|log|sqrt|atan|asin|acos|sec|csc|cot)\*\*(\d+)\(([^)]+)\)',
                          r'(\1(\3))**\2', expr)

            # Handle func**nx patterns (no parens): sin**2x -> (sin(x))**2
            expr = re.sub(r'(sin|cos|tan|log|sqrt|atan|asin|acos|sec|csc|cot)\*\*(\d+)\*?([a-zA-Z])',
                          r'(\1(\3))**\2', expr)

            # Handle funcx patterns (no parens): logx -> log(x), sinx -> sin(x)
            expr = re.sub(r'(sin|cos|tan|log|exp|sqrt|atan|asin|acos|abs)([a-zA-Z])\b', r'\1(\2)', expr)


#### 1sr
            if 'frac' in expr:
                self._log(f"[plotter] ⚠️ Skipping malformed frac expression: '{expr}'")
                continue

            # Fix logabs -> log(abs(...)) BEFORE other operations
            while 'logabs(' in expr:
                start = expr.find('logabs(')
                depth = 1
                pos = start + 7
                end = -1

                for i in range(pos, len(expr)):
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break

                if end > start:
                    content = expr[start + 7:end]
                    expr = expr[:start] + f'log(abs({content}))' + expr[end + 1:]
                else:
                    expr = expr.replace('logabs(', 'log(abs(', 1)
                    break

            # Fix e** patterns - convert ALL to exp()
            expr = re.sub(r'e\*\*(-\d+\.?\d*)\*([a-zA-Z])', r'exp(\1*\2)', expr)
            expr = re.sub(r'e\*\*(-\d+\.?\d*)', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*(-[a-zA-Z])', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*\(([^)]+)\)', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*([a-zA-Z0-9]+)', r'exp(\1)', expr)

            # Ensure closing parens for exp
            if 'exp(' in expr:
                open_count = expr.count('(')
                close_count = expr.count(')')
                if open_count > close_count:
                    expr += ')' * (open_count - close_count)

            # Add * between )( and )function
            expr = re.sub(r'(\))(\()', r'\1*\2', expr)
            expr = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr)

            # Fix xlog -> x*log
            expr = re.sub(r'(?<![a-z])([a-zA-Z])(log|sin|cos|tan|exp|sqrt|abs)\b', r'\1*\2', expr)

            # Protect arc/inverse trig functions
            expr = re.sub(r'\ba\*tan\b', 'atan', expr)
            expr = re.sub(r'\ba\*sin\b', 'asin', expr)
            expr = re.sub(r'\ba\*cos\b', 'acos', expr)

            # Fix sqrtx -> sqrt(x)
            if 'sqrt' in expr:
                expr = re.sub(r'sqrt([a-zA-Z])', r'sqrt(\1)', expr)

            if 'frac' in expr:
                self._log(f"[plotter] ⚠️ Skipping malformed frac expression: '{expr}'")
                continue

###


            if 'ERROR' in expr:
                self._log(f"[plotter] Skipping malformed expression: '{expr}'")
                continue

            if not self._looks_like_function(expr):
                self._log(f"[plotter] Skipping non-function: '{expr}'")
                continue

            # Store as tuple: (expression, original_variable)
            final_expressions.append((expr, original_variable))

        # Remove duplicates while preserving order
        seen = set()
        unique_expressions = []
        original_variables = []

        for item in final_expressions:
            if isinstance(item, tuple):
                expr, var = item
            else:
                expr, var = item, 'x'

            if expr not in seen:
                seen.add(expr)
                unique_expressions.append(expr)
                original_variables.append(var)

        self._current_expressions = unique_expressions
        self._original_variables = original_variables

        if self._log:
            self._log(f"[plotter] Found {len(unique_expressions)} unique expressions: {unique_expressions}")
            self._log(f"[plotter] Original variables: {original_variables}")

        return unique_expressions

    def _looks_like_math_text(self, text):
        """Check if text looks like it contains math."""
        has_operator = any(
            op in text for op in ['+', '-', '*', '/', '^', '**', '=', 'exp', 'sin', 'cos', 'tan', 'log', 'sqrt'])
        has_number_var = bool(re.search(r'\d+[a-zA-Z]|[a-zA-Z]\d+', text))
        has_function = bool(re.search(r'[a-zA-Z]\([^)]+\)', text))
        return has_operator or has_number_var or has_function

    def _normalize_variables(self, expr):
        """Convert all variables to x for consistent plotting - PROTECT FUNCTION NAMES."""
        if not expr:
            return expr

        function_names = ['atan', 'asin', 'acos', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'ln']
        protected = {}

        for i, func in enumerate(function_names):
            placeholder = f'__FUNC{i}__'
            expr = expr.replace(func, placeholder)
            protected[placeholder] = func

        if 'integrate' in expr:
            expr = expr.replace('integrate', '__INTEGRATE__')
            protected['__INTEGRATE__'] = 'integrate'

        replacements = {
            't': 'x', 'θ': 'x', 'theta': 'x',
            'alpha': 'x', 'beta': 'x', 'gamma': 'x',
            'λ': 'x', 'lambda': 'x', 'y': 'x',
        }

        expr_normalized = expr
        for old, new in replacements.items():
            expr_normalized = expr_normalized.replace(old, new)

        expr_normalized = re.sub(r'\b([a-df-hj-oq-z])\b', 'x', expr_normalized)

        for placeholder, func in protected.items():
            expr_normalized = expr_normalized.replace(placeholder, func)

        return expr_normalized

    def _clean_conversational_math(self, text: str) -> str:
        """Extract math from conversational text."""
        remove_phrases = [
            'mathematically that gives us', 'so the derivative of',
            'which states that', 'applying this to',
            'we have the function', 'to find the derivative',
            'the derivative is', 'is calculated as',
            'mathematically expressed as', 'gives us', 'we get',
        ]

        cleaned = text.lower()
        for phrase in remove_phrases:
            cleaned = cleaned.replace(phrase, '')

        math_patterns = [
            r'[fgy][\'′]?\(?[a-zA-Zα-ωΑ-Ω]?\)?\s*[=:]\s*[^.,;\n]+',
            r'\b\d*\.?\d*[a-zA-Zα-ωΑ-Ω]\^?\d*\b',
            r'\b(?:sin|cos|tan|exp|log|sqrt)\([^)]+\)',
            r'[a-zA-Zα-ωΑ-Ω][\^²³]',
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, cleaned)
            for match in matches:
                if match.strip():
                    match = match.replace('²', '**2').replace('³', '**3').replace('^', '**')
                    return match.strip()

        return self._clean_plain_expression(cleaned)

    def _clean_latex_expression(self, latex_str):
        """Clean LaTeX expression for plotting - CRASH-PROOF VERSION."""
        if not latex_str:
            return ""

        try:
            import re
            clean = latex_str
            # ADD THIS DEBUG LINE
            self._log(f"[clean-debug] INPUT: '{clean}' (repr: {repr(clean)})")


            # ========== STEP 0: Remove dollar signs FIRST ==========
            clean = clean.replace('$', '')

            # ========== STEP 0.5: Handle \theta and \boxed EARLY ==========
            # Handle theta specifically to avoid \t being interpreted as tab
            clean = clean.replace(r'\theta', 'theta')
            clean = clean.replace(r'\Theta', 'Theta')

            # ========== STEP 1: Handle \boxed{} - REMOVE IT COMPLETELY ==========
            max_iterations = 10
            iteration = 0

            while ('\\boxed{' in clean or 'boxed{' in clean) and iteration < max_iterations:
                iteration += 1
                found_boxed = False

                for boxed_variant in ['\\boxed{', 'boxed{']:
                    if boxed_variant in clean:
                        try:
                            start = clean.find(boxed_variant)
                            depth = 0
                            end = start + len(boxed_variant)

                            for i in range(end, len(clean)):
                                if clean[i] == '{':
                                    depth += 1
                                elif clean[i] == '}':
                                    if depth == 0:
                                        end = i
                                        break
                                    else:
                                        depth -= 1

                            if end > start and end < len(clean):
                                content = clean[start + len(boxed_variant):end]
                                clean = clean[:start] + content + clean[end + 1:]
                                found_boxed = True
                            else:
                                clean = clean.replace(boxed_variant, '')
                                break
                        except Exception as e:
                            self._log(f"[clean] ⚠️ Error removing boxed: {e}")
                            clean = clean.replace(boxed_variant, '')
                            break

                if not found_boxed:
                    break

            # ========== STEP 2: Handle \frac{a}{b} with ROBUST parsing ==========
            max_frac_iterations = 10
            frac_iteration = 0

            while ('\\frac{' in clean or '\\dfrac{' in clean) and frac_iteration < max_frac_iterations:
                frac_iteration += 1
                found_frac = False

                for frac_cmd in ['\\dfrac{', '\\frac{']:
                    if frac_cmd in clean:
                        try:
                            start = clean.find(frac_cmd)
                            depth = 0
                            num_start = start + len(frac_cmd)
                            num_end = num_start

                            for i in range(num_start, len(clean)):
                                if clean[i] == '{':
                                    depth += 1
                                elif clean[i] == '}':
                                    if depth == 0:
                                        num_end = i
                                        break
                                    else:
                                        depth -= 1

                            if num_end == num_start or num_end >= len(clean):
                                self._log(f"[clean] ⚠️ Malformed frac numerator at pos {start}")
                                clean = clean.replace(frac_cmd, '(ERROR)/(ERROR)', 1)
                                break

                            if num_end + 1 < len(clean) and clean[num_end + 1] == '{':
                                depth = 0
                                den_start = num_end + 2
                                den_end = den_start

                                for i in range(den_start, len(clean)):
                                    if clean[i] == '{':
                                        depth += 1
                                    elif clean[i] == '}':
                                        if depth == 0:
                                            den_end = i
                                            break
                                        else:
                                            depth -= 1

                                if den_end > den_start and den_end < len(clean):
                                    numerator = clean[num_start:num_end]
                                    denominator = clean[den_start:den_end]
                                    replacement = f'((({numerator})))/(({denominator}))'
                                    clean = clean[:start] + replacement + clean[den_end + 1:]
                                    found_frac = True
                                else:
                                    self._log(f"[clean] ⚠️ Malformed frac denominator at pos {start}")
                                    clean = clean.replace(frac_cmd, '(ERROR)/(ERROR)', 1)
                                    break
                            else:
                                self._log(f"[clean] ⚠️ Missing frac denominator at pos {start}")
                                clean = clean.replace(frac_cmd, '(ERROR)/(ERROR)', 1)
                                break
                        except Exception as e:
                            self._log(f"[clean] ⚠️ Error parsing frac: {e}")
                            clean = clean.replace(frac_cmd, '(ERROR)/(ERROR)', 1)
                            break

                if not found_frac:
                    break

            # ========== STEP 3: Handle \sqrt{value} with ROBUST parsing ==========
            max_sqrt_iterations = 10
            sqrt_iteration = 0

            while '\\sqrt{' in clean and sqrt_iteration < max_sqrt_iterations:
                sqrt_iteration += 1

                try:
                    start = clean.find('\\sqrt{')
                    if start == -1:
                        break

                    depth = 0
                    end = start + 6

                    for i in range(end, len(clean)):
                        if clean[i] == '{':
                            depth += 1
                        elif clean[i] == '}':
                            if depth == 0:
                                end = i
                                break
                            else:
                                depth -= 1

                    if end > start and end < len(clean):
                        content = clean[start + 6:end]
                        clean = clean[:start] + f'sqrt({content})' + clean[end + 1:]
                    else:
                        self._log(f"[clean] ⚠️ Malformed sqrt at pos {start}")
                        clean = clean.replace('\\sqrt{', 'sqrt(', 1)
                        break
                except Exception as e:
                    self._log(f"[clean] ⚠️ Error parsing sqrt: {e}")
                    clean = clean.replace('\\sqrt{', 'sqrt(', 1)
                    break

            # ========== STEP 3.5: Handle inverse trig functions ==========
            # Must convert BEFORE ^ becomes **
            clean = re.sub(r'\\tan\s*\^\s*\{?\s*-\s*1\s*\}?', 'atan', clean)
            clean = re.sub(r'\\sin\s*\^\s*\{?\s*-\s*1\s*\}?', 'asin', clean)
            clean = re.sub(r'\\cos\s*\^\s*\{?\s*-\s*1\s*\}?', 'acos', clean)
            clean = re.sub(r'\\arctan', 'atan', clean)
            clean = re.sub(r'\\arcsin', 'asin', clean)
            clean = re.sub(r'\\arccos', 'acos', clean)

            # Also handle \text{} wrapper
            clean = re.sub(r'\\text\{Phase\}', 'Phase', clean)
            clean = re.sub(r'\\text\{([^}]+)\}', r'\1', clean)  # Generic \text{...} removal

            # ========== STEP 4: Convert LaTeX commands ==========
            latex_commands = {
                '\\sin': 'sin', '\\cos': 'cos', '\\tan': 'tan',
                '\\arctan': 'atan', '\\asin': 'asin', '\\acos': 'acos',
                '\\atan': 'atan', '\\log': 'log', '\\ln': 'log',
                '\\exp': 'exp', '\\pi': 'pi', '\\e': 'E',
                '\\infty': 'oo', '\\cdot': '*', '\\times': '*',
                '\\left': '', '\\right': '', '\\,': '',
                '\\ ': '', '\\quad': '', '\\qquad': '',
                '\\lambda': 'lambda', '\\omega': 'omega', # omega support
            }

            for latex, plain in latex_commands.items():
                clean = clean.replace(latex, plain)
            # ========== STEP 4.5: Handle e^{...} BEFORE removing braces ==========
            # This preserves the exponent grouping: e^{(-4-sqrt(6))t} → exp((-4-sqrt(6))*t)
            max_exp_iterations = 20
            exp_iteration = 0

            while 'e^{' in clean and exp_iteration < max_exp_iterations:
                exp_iteration += 1

                try:
                    start = clean.find('e^{')
                    if start == -1:
                        break

                    # Find matching closing brace with depth counting
                    depth = 0
                    content_start = start + 3  # Skip "e^{"
                    end = -1

                    for i in range(content_start, len(clean)):
                        if clean[i] == '{':
                            depth += 1
                        elif clean[i] == '}':
                            if depth == 0:
                                end = i
                                break
                            else:
                                depth -= 1

                    if end > content_start:
                        content = clean[content_start:end]

                        # Remove spaces
                        content = content.replace(' ', '')

                        # Add * between )letter patterns: )t → )*t
                        content = re.sub(r'\)([a-zA-Z])', r')*\1', content)

                        # Add * between digit and letter: 2t → 2*t
                        content = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', content)

                        # Add * between consecutive single letters (but not function names)
                        for func in ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'atan', 'asin', 'acos']:
                            content = content.replace(func, f'__{func.upper()}__')
                        content = re.sub(r'([a-zA-Z])([a-zA-Z])(?![a-zA-Z_])', r'\1*\2', content)
                        for func in ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'atan', 'asin', 'acos']:
                            content = content.replace(f'__{func.upper()}__', func)

                        # Handle single letter followed by ( but NOT function names
                        content = re.sub(r'(^|[+\-*/])([a-zA-Z])\(', r'\1\2*(', content)

                        clean = clean[:start] + f'exp({content})' + clean[end + 1:]
                        self._log(f"[clean] e^{{...}} → exp(): '{content}'")
                    else:
                        break
                except Exception as e:
                    self._log(f"[clean] ⚠️ Error handling e^{{}}: {e}")
                    break


            # ========== STEP 4.3: Handle \text{sinc} ==========
            clean = re.sub(r'\\text\{sinc\}', 'sinc', clean)

            # Convert sinc(arg) to sin(arg)/(arg) or sin(pi*arg)/(pi*arg)
            while 'sinc(' in clean:
                start = clean.find('sinc(')
                depth = 0
                content_start = start + 5
                end = -1
                for i in range(content_start, len(clean)):
                    if clean[i] == '(':
                        depth += 1
                    elif clean[i] == ')':
                        if depth == 0:
                            end = i
                            break
                        else:
                            depth -= 1
                if end > content_start:
                    arg = clean[content_start:end]
                    if 'pi' in arg.lower():
                        # Already has pi - use unnormalized
                        replacement = f'(sin({arg})/({arg}))'
                        self._log(f"[clean] sinc({arg}) → sin({arg})/({arg}) [unnormalized]")
                    else:
                        # No pi - use normalized (engineering convention)
                        replacement = f'(sin(pi*({arg}))/(pi*({arg})))'
                        self._log(f"[clean] sinc({arg}) → sin(π·{arg})/(π·{arg}) [normalized]")
                    clean = clean[:start] + replacement + clean[end + 1:]
                else:
                    break


            # ========== STEP 5: Clean up ==========
            clean = clean.replace('\\', '')
            clean = clean.replace('[', '(').replace(']', ')')  # Convert brackets to parens
            clean = clean.replace('{', '').replace('}', '')
            clean = clean.replace('^', '**')
            clean = re.sub(r'\s+', '', clean)

            # ========== STEP 6: Handle absolute values ==========
            try:
                clean = re.sub(r'\|([^|]+)\|', r'abs(\1)', clean)
            except Exception as e:
                self._log(f"[clean] ⚠️ Error handling absolute values: {e}")



            # ========== STEP 7: Add multiplication where needed ==========
            try:
                clean = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', clean)
                clean = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', clean)
                clean = re.sub(r'(\))(\()', r'\1*\2', clean)
                clean = re.sub(r'(\))([a-zA-Z])', r'\1*\2', clean)
                clean = re.sub(r'(\d)(sin|cos|tan|exp|log|sqrt|abs|atan|asin|acos)\b', r'\1*\2', clean)

                # NEW: Add * before function names when preceded by SINGLE letter variable
                # Protect arc functions first to avoid matching 'a' in 'asin'
                clean = clean.replace('asin(', '__ASIN__(')
                clean = clean.replace('acos(', '__ACOS__(')
                clean = clean.replace('atan(', '__ATAN__(')

                # Now add * between single letter and function
                clean = re.sub(r'\b([a-zA-Z])(sin|cos|tan|log|sqrt|abs|exp)\(', r'\1*\2(', clean)

                # Restore protected functions
                clean = clean.replace('__ASIN__(', 'asin(')
                clean = clean.replace('__ACOS__(', 'acos(')
                clean = clean.replace('__ATAN__(', 'atan(')
            except Exception as e:
                self._log(f"[clean] ⚠️ Error adding multiplication: {e}")

            # ========== STEP 7.5: Add * between adjacent variables/constants ==========
            # Handle pi*var, var*pi, var*var patterns
            clean = re.sub(r'\bpi([a-zA-Z])', r'pi*\1', clean)  # pit → pi*t
            clean = re.sub(r'([a-zA-Z])pi\b', r'\1*pi', clean)  # tpi → t*pi
            clean = re.sub(r'\blambda([a-zA-Z])', r'lambda*\1', clean)  # lambdat → lambda*t
            clean = re.sub(r'([a-zA-Z])lambda\b', r'\1*lambda', clean)  # tlambda → t*lambda

            # ========== STEP 7.6: Fix func**n*( → func**n( ==========
            # Remove spurious * between exponent and opening paren
            clean = re.sub(r'(sin|cos|tan)\*\*(\d+)\*\(', r'\1**\2(', clean)



            # ========== STEP 8: Handle e^(...) ==========
            try:
                # NEW: Handle e**func(x) patterns first
                clean = re.sub(r'\be\*\*(sin|cos|tan|log|sqrt|abs|atan|asin|acos|exp)\(', r'exp(\1(', clean)
                # Then handle regular e**(...)
                clean = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', clean)
            except Exception as e:
                self._log(f"[clean] ⚠️ Error handling e^: {e}")

            # ========== STEP 9: Remove ERROR markers if present ==========
            if 'ERROR' in clean:
                self._log(f"[clean] ⚠️ Expression contains errors, may not plot correctly")

            result = clean.strip()

            if len(result) == 0:
                self._log(f"[clean] ⚠️ Cleaning resulted in empty string from: '{latex_str}'")
                return ""

            if len(result) > 1000:
                self._log(f"[clean] ⚠️ Result is unusually long ({len(result)} chars), may be malformed")

            return result

        except Exception as e:
            self._log(f"[clean] ❌ CRITICAL ERROR in _clean_latex_expression: {e}")
            import traceback
            self._log(f"[clean] Traceback: {traceback.format_exc()}")
            return latex_str.replace('$', '').replace('\\', '')

    def _clean_plain_expression(self, expr):
        """Clean plain text expression for plotting."""
        if not expr:
            return ""

        expr_clean = expr.replace(" ", "")
        expr_clean = re.sub(r'(\d+\.\d+)([a-zA-Z(])', r'\1*\2', expr_clean)
        expr_clean = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_clean)
        expr_clean = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr_clean)
        expr_clean = re.sub(r'(\d)(sin|cos|tan|exp|log|sqrt)', r'\1*\2', expr_clean)
        expr_clean = re.sub(r'(\))(sin|cos|tan|exp|log|sqrt)', r'\1*\2', expr_clean)

        for func in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            expr_clean = re.sub(rf'{func}([a-zA-Z])', rf'{func}(\1)', expr_clean)

        expr_clean = expr_clean.replace('^', '**')
        expr_clean = expr_clean.replace('ln', 'log')

        return expr_clean.strip()

    def _looks_like_function(self, expr):
        """Check if expression looks like a plottable function."""
        if not expr:
            return False

        expr_lower = expr.lower()

        if not re.search(r'[a-zA-Zα-ωΑ-Ω]', expr_lower):
            return False

        allowed_patterns = [
            '**', '^', '*', '/', '+', '-',
            'sin(', 'cos(', 'tan(', 'exp(', 'log(', 'sqrt(',
            '(', ')', '=', '[', ']', '{', '}'
        ]

        has_math = any(pattern in expr_lower for pattern in allowed_patterns)

        is_simple_math = (
                '**' in expr_lower or '^' in expr_lower or
                expr_lower.startswith('sin') or expr_lower.startswith('cos') or
                expr_lower.startswith('tan') or expr_lower.startswith('exp') or
                expr_lower.startswith('log') or expr_lower.startswith('sqrt') or
                '(' in expr_lower
        )

        looks_like_math = (
                ('+' in expr_lower or '-' in expr_lower or '*' in expr_lower or '/' in expr_lower) and
                re.search(r'[a-zA-Zα-ωΑ-Ω]', expr_lower)
        )

        return has_math or is_simple_math or looks_like_math

    def close_all(self):
        """Close all plot windows."""
        # Create a copy of the list to avoid modification issues during iteration
        for window in self._plot_windows[:]:
            try:
                if window and hasattr(window, 'destroy'):
                    window.destroy()
            except:
                pass

        # Clear the list
        self._plot_windows.clear()





class PlotWindow(tk.Toplevel):
    """
    Tkinter window for plotting mathematical functions.
    """

    def __init__(self, master, expressions, log_fn=None, original_vars=None):
        super().__init__(master)
        self.title("Function Plotter")
        self.geometry("1000x900")
        self._log = log_fn or (lambda msg: None)

        self.expressions = expressions
        self.original_vars = original_vars or ['x'] * len(expressions)  # NEW
        self.current_index = 0

        self._plotting = False
        self._trace_ids = {}
        self._user_set_range = False

        self._validating_ranges = False

        self.configure(bg="#1e1e1e")
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self._log_scale = False
        self._db_scale = False

        self._create_ui()

        if expressions:
            self.master.after(100, lambda: self.plot_expression(expressions[0], self.original_vars[0]))

    def _create_ui(self):
        """Create the UI components."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 10))

        if len(self.expressions) > 1:
            nav_frame = ttk.Frame(control_frame)
            nav_frame.pack(side="left", padx=(0, 20))

            ttk.Button(nav_frame, text="◀ Prev",
                       command=self.prev_expression, width=8).pack(side="left", padx=2)

            self.expr_label = ttk.Label(nav_frame, text="", width=40)
            self.expr_label.pack(side="left", padx=5)

            ttk.Button(nav_frame, text="Next ▶",
                       command=self.next_expression, width=8).pack(side="left", padx=2)


        range_frame = ttk.Frame(control_frame)
        range_frame.pack(side="left", padx=10)

        ttk.Label(range_frame, text="Domain Controls", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=6, pady=(0, 5))

        # ========== X CONTROLS (ROW 1) ==========
        ttk.Label(range_frame, text="x min:").grid(row=1, column=0, padx=2, sticky='e')
        self.xmin_var = tk.DoubleVar(value=-10.0)
        self.xmin_entry = ttk.Entry(range_frame, textvariable=self.xmin_var, width=8)
        self.xmin_entry.grid(row=1, column=1, padx=2)

        # X min slider
        self.xmin_slider = tk.Scale(range_frame, from_=-100, to=100, resolution=0.5,
                                    orient=tk.HORIZONTAL, variable=self.xmin_var,
                                    length=180, showvalue=False)
        self.xmin_slider.grid(row=1, column=2, padx=2)

        ttk.Label(range_frame, text="x max:").grid(row=1, column=3, padx=2, sticky='e')
        self.xmax_var = tk.DoubleVar(value=100.0)
        self.xmax_entry = ttk.Entry(range_frame, textvariable=self.xmax_var, width=8)
        self.xmax_entry.grid(row=1, column=4, padx=2)

        # Multiplier spinbox for xmax
        ttk.Label(range_frame, text="×").grid(row=1, column=5, padx=1)
        self.xmax_mult_var = tk.IntVar(value=1)
        self.xmax_mult_spin = ttk.Spinbox(
            range_frame,
            from_=1,
            to=10000,
            textvariable=self.xmax_mult_var,
            width=6,
            command=self._on_multiplier_change
        )
        self.xmax_mult_spin.grid(row=1, column=6, padx=2)

        # ========== Y CONTROLS (ROW 2) ==========
        ttk.Label(range_frame, text="y min:").grid(row=2, column=0, padx=2, pady=(5, 0), sticky='e')
        self.ymin_var = tk.DoubleVar(value=-10.0)
        self.ymin_entry = ttk.Entry(range_frame, textvariable=self.ymin_var, width=8)
        self.ymin_entry.grid(row=2, column=1, padx=2, pady=(5, 0))

        # Y min slider (shorter to make room for multiplier)
        self.ymin_slider = tk.Scale(range_frame, from_=-100, to=100, resolution=0.5,
                                    orient=tk.HORIZONTAL, variable=self.ymin_var,
                                    length=120, showvalue=False)
        self.ymin_slider.grid(row=2, column=2, padx=2, pady=(5, 0))

        # Multiplier spinbox for ymin (for phase plots down to -400°)
        ttk.Label(range_frame, text="×").grid(row=2, column=3, padx=1, pady=(5, 0))
        self.ymin_mult_var = tk.IntVar(value=1)
        self.ymin_mult_spin = ttk.Spinbox(
            range_frame,
            from_=1,
            to=100,
            textvariable=self.ymin_mult_var,
            width=4,
            command=self._on_ymin_multiplier_change
        )
        self.ymin_mult_spin.grid(row=2, column=4, padx=2, pady=(5, 0))

        ttk.Label(range_frame, text="y max:").grid(row=2, column=5, padx=2, pady=(5, 0), sticky='e')
        self.ymax_var = tk.DoubleVar(value=10.0)
        self.ymax_entry = ttk.Entry(range_frame, textvariable=self.ymax_var, width=8)
        self.ymax_entry.grid(row=2, column=6, padx=2, pady=(5, 0))

        # Y max slider
        self.ymax_slider = tk.Scale(range_frame, from_=-100, to=100, resolution=0.5,
                                    orient=tk.HORIZONTAL, variable=self.ymax_var,
                                    length=120, showvalue=False)
        self.ymax_slider.grid(row=2, column=7, padx=2, pady=(5, 0))


        # ========== VALIDATION TRACES ==========
        self.xmin_var.trace_add('write', lambda *args: self._validate_x_range())
        self.xmax_var.trace_add('write', lambda *args: self._validate_x_range())
        self.ymin_var.trace_add('write', lambda *args: self._validate_y_range())
        self.ymax_var.trace_add('write', lambda *args: self._validate_y_range())

        def on_range_entry_return(event):
            self._user_set_range = True
            if self.auto_scale_var.get():
                self.auto_scale_var.set(False)
                self._log("[plot] Auto-scale turned off due to manual range entry")
            self.plot_current()

        for entry in [self.xmin_entry, self.xmax_entry, self.ymin_entry, self.ymax_entry]:
            entry.bind('<Return>', on_range_entry_return)
            entry.bind('<FocusIn>', lambda e: setattr(self, '_user_set_range', True))
            entry.bind('<KeyRelease>', lambda e: self._delayed_plot())

        # Add slider change handling
        for slider in [self.xmin_slider, self.ymin_slider, self.ymax_slider]:
            slider.config(command=lambda v: self._on_slider_change())


        def on_auto_scale_toggle():
            if self.auto_scale_var.get():
                self._user_set_range = False
                self._log("[plot] Auto-scale enabled, clearing manual range flag")
                self.plot_current()
            else:
                self._user_set_range = True
                self._log("[plot] Auto-scale disabled, setting manual range flag")

        self.auto_scale_var = tk.BooleanVar(value=True)
        auto_scale_check = ttk.Checkbutton(
            range_frame,
            text="Auto-scale",
            variable=self.auto_scale_var,
            command=on_auto_scale_toggle
        )
        auto_scale_check.grid(row=3, column=0, columnspan=4, pady=(5, 0))

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side="right")

        ttk.Button(button_frame, text="📈 x ≥ 0",
                   command=self.set_positive_domain, width=10).pack(side="left", padx=2)
        ttk.Button(button_frame, text="💾 Save",
                   command=self.save_plot, width=10).pack(side="left", padx=2)
        ttk.Button(button_frame, text="🔄 Reset",
                   command=self.reset_view, width=10).pack(side="left", padx=2)

        ttk.Button(button_frame, text="Log x",
                   command=self.toggle_log_scale, width=10).pack(side="left", padx=2)
        ttk.Button(button_frame, text="dB",
                   command=self.toggle_db_scale, width=10).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side="bottom", fill="x", pady=(5, 0))

        self.figure = Figure(figsize=(10, 7), dpi=100, facecolor='#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self._setup_axes()

        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self._plot_timer = None

    def _validate_x_range(self):
        """Ensure xmin < xmax with minimum gap."""
        if self._validating_ranges:
            return

        self._validating_ranges = True

        try:
            xmin = self.xmin_var.get()
            xmax = self.xmax_var.get()

            # Ensure xmin < xmax with 1.0 minimum gap
            if xmin >= xmax - 0.5:
                self.xmax_var.set(xmin + 1.0)
                self._log("[plot] Auto-corrected: xmax must be > xmin")

        except tk.TclError:
            pass  # Ignore errors during typing
        finally:
            self._validating_ranges = False

    def _validate_y_range(self):
        """Ensure ymin < ymax with minimum gap."""
        if self._validating_ranges:
            return

        self._validating_ranges = True

        try:
            ymin = self.ymin_var.get()
            ymax = self.ymax_var.get()

            # Ensure ymin < ymax with 1.0 minimum gap
            if ymin >= ymax - 0.5:
                self.ymax_var.set(ymin + 1.0)
                self._log("[plot] Auto-corrected: ymax must be > ymin")

        except tk.TclError:
            pass  # Ignore errors during typing
        finally:
            self._validating_ranges = False


    def _setup_axes(self):
        """Setup matplotlib axes with dark theme."""
        self.ax.set_facecolor('#2b2b2b')

        for spine in self.ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1)

        self.ax.tick_params(axis='both', colors='white', labelsize=10)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

        self.ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
        self.ax.axhline(y=0, color='white', linewidth=1, alpha=0.5)
        self.ax.axvline(x=0, color='white', linewidth=1, alpha=0.5)

    def _detect_optimal_range(self, expr):
        """Detect optimal x-range based on function type."""
        expr_lower = expr.lower()

        decay_patterns = [
            r'exp\([^)]*-\d*\.?\d*\s*\*?\s*x[^)]*\)',
            r'e\^\([^)]*-\d*\.?\d*\s*\*?\s*x[^)]*\)',
            r'1\s*-\s*exp\([^)]*-\d*\.?\d*\s*\*?\s*x[^)]*\)',
            r'1\s*-\s*e\^\([^)]*-\d*\.?\d*\s*\*?\s*x[^)]*\)',
        ]

        growth_patterns = [
            r'exp\([^)]*\d*\.?\d*\s*\*?\s*x[^)]*\)',
            r'e\^\([^)]*\d*\.?\d*\s*\*?\s*x[^)]*\)',
        ]

        for pattern in decay_patterns:
            if re.search(pattern, expr_lower):
                match = re.search(r'exp\([^)]*([-\d\.]+)\s*\*?\s*x', expr_lower)
                if match:
                    try:
                        k = abs(float(match.group(1)))
                        if k == 0:
                            k = 1
                        x_max = 3 / k
                        return max(0, min(-10, x_max - 5)), x_max + 1
                    except:
                        pass
                return 0, 5

        for pattern in growth_patterns:
            if re.search(pattern, expr_lower):
                return -3, 3

        if any(term in expr_lower for term in ['x**', 'x^', '*x']):
            try:
                x = sp.symbols('x')
                sympy_expr = sp.sympify(expr, locals={'x': x})
                if sympy_expr.is_polynomial():
                    degree = sympy_expr.as_poly(x).degree()
                    return -max(3, degree), max(3, degree)
            except:
                pass

        return -10, 10

    def prev_expression(self):
        """Navigate to previous expression."""
        if len(self.expressions) > 1:
            self.current_index = (self.current_index - 1) % len(self.expressions)
            self.plot_current()

    def next_expression(self):
        """Navigate to next expression."""
        if len(self.expressions) > 1:
            self.current_index = (self.current_index + 1) % len(self.expressions)
            self.plot_current()

    def _on_slider_change(self):
        """Handle slider changes - mark as user input and trigger delayed plot."""
        self._user_set_range = True
        if self.auto_scale_var.get():
            self.auto_scale_var.set(False)
        self._delayed_plot()

    def toggle_db_scale(self):
        """Toggle dB scale (20*log10) for y-axis."""
        self._db_scale = not self._db_scale

        # Disable auto-scale for manual control
        self._user_set_range = True
        if self.auto_scale_var.get():
            self.auto_scale_var.set(False)

        if self._db_scale:
            self.status_var.set("dB scale: ON (20·log₁₀)")
        else:
            self.status_var.set("dB scale: OFF")

        self.plot_current()

    def plot_current(self):
        """Plot the current expression."""
        print(f"[DEBUG] plot_current called, _plotting={self._plotting}")  # ADD THIS

        if self._plotting:
            print("[DEBUG] Already plotting, returning")  # ADD THIS
            return

        print(f"[DEBUG] expressions={self.expressions}")  # ADD THIS
        if self.expressions:
            expr = self.expressions[self.current_index]
            var = self.original_vars[self.current_index] if self.current_index < len(self.original_vars) else 'x'
            print(f"[DEBUG] About to plot: expr='{expr}', var='{var}'")  # ADD THIS
            if len(self.expressions) > 1:
                self.expr_label.config(text=f"Expression {self.current_index + 1}/{len(self.expressions)}")
            self.plot_expression(expr, var)
        else:
            print("[DEBUG] No expressions to plot!")  # ADD THIS

    def set_positive_domain(self):
        """Set x-min to 0 for time-domain plots."""
        self.xmin_var.set(0)
        self._user_set_range = True
        if self.auto_scale_var.get():
            self.auto_scale_var.set(False)
        self.plot_current()

    def toggle_log_scale(self):
        """Toggle logarithmic x-axis (only works when x > 0)."""
        xmin = self.xmin_var.get()

        if xmin < 0:
            self.status_var.set("⚠️ Log scale requires x ≥ 0. Press 'x ≥ 0' first.")
            return

        # Ensure xmin is positive (log can't handle 0)
        if xmin <= 0:
            self.xmin_var.set(0.01)

        self._log_scale = not self._log_scale

        # Disable auto-scale to prevent it overwriting our positive xmin
        self._user_set_range = True
        if self.auto_scale_var.get():
            self.auto_scale_var.set(False)

        if self._log_scale:
            self.status_var.set("Log scale: ON")
        else:
            self.status_var.set("Log scale: OFF")

        self.plot_current()

    def _on_ymin_multiplier_change(self):
        """Handle ymin multiplier spinbox change - update slider range and value."""
        if getattr(self, '_updating_multiplier', False):
            return

        try:
            self._updating_multiplier = True
            mult = self.ymin_mult_var.get()

            # Update slider range to accommodate multiplier
            new_min = -100 * mult
            new_max = 100 * mult
            self.ymin_slider.config(from_=new_min, to=new_max)

            # Set ymin to the new minimum
            self.ymin_var.set(new_min)

            self._user_set_range = True
            if self.auto_scale_var.get():
                self.auto_scale_var.set(False)

            # Use delayed plot to debounce rapid changes
            if self._plot_timer:
                self.master.after_cancel(self._plot_timer)
            self._plot_timer = self.master.after(300, self.plot_current)
        except:
            pass
        finally:
            self._updating_multiplier = False


    def _delayed_plot(self):
        """Schedule a plot update after a short delay."""
        if self._plot_timer:
            self.master.after_cancel(self._plot_timer)
        self._plot_timer = self.master.after(500, self.plot_current)

    def parse_expression(self, expr):
        """Parse mathematical expression to sympy/numpy function."""
        try:
            x = sp.symbols('x')
            expr_clean = expr.replace('$', '')

            if '=' in expr_clean:
                parts = expr_clean.split('=', 1)
                if len(parts) == 2:
                    left, right = parts[0].strip(), parts[1].strip()
                    left_lower = left.lower()
                    if (left_lower in ['y', 'f', 'g', 'h'] or '(' in left):
                        expr_clean = right
                        self._log(f"[plot] Equation: using right side '{right}'")
                    elif right.strip() == '0':
                        expr_clean = left
                        self._log(f"[plot] Equation = 0: plotting left side '{left}'")
                    else:
                        expr_clean = right
                        self._log(f"[plot] Using right side '{right}'")

            self._log(f"[plot] Parsing: '{expr_clean}'")

            sympy_expr = sp.sympify(expr_clean, locals={'x': x})
            numpy_func = sp.lambdify(x, sympy_expr, ['numpy', 'math'])

            def vectorized_func(x_vals):
                try:
                    if np.isscalar(x_vals):
                        return numpy_func(x_vals)
                    else:
                        x_array = np.asarray(x_vals, dtype=float)
                        try:
                            return numpy_func(x_array)
                        except (TypeError, ValueError):
                            return np.array([numpy_func(val) for val in x_array])
                except Exception as e:
                    self._log(f"[plot] Function evaluation error: {e}")
                    raise

            pretty_str = sp.latex(sympy_expr)
            self._log(f"[plot] ✓ Successfully parsed: '{expr_clean}'")
            return vectorized_func, pretty_str, sympy_expr

        except Exception as e:
            self._log(f"[plot] ❌ Parse error for '{expr}': {e}")
            return None, None, None

    def plot_expression(self, expr, original_var='x'):
        """Plot a mathematical expression with original variable for axis label."""
        if self._plotting:
            return

        self._plotting = True

        try:
            self.status_var.set(f"Plotting: {expr[:50]}...")
            self._log(f"[plot] Plotting expression: {expr}")
            self._log(f"[plot] Original variable: {original_var}")

            xmin = self.xmin_var.get()
            xmax = self.xmax_var.get()
            ymin = self.ymin_var.get()
            ymax = self.ymax_var.get()

            self._log(f"[plot] Current ranges - x: [{xmin}, {xmax}], y: [{ymin}, {ymax}]")
            self._log(f"[plot] Auto-scale: {self.auto_scale_var.get()}, User set: {self._user_set_range}")

            self.ax.clear()
            self._setup_axes()

            func, latex_str, sympy_expr = self.parse_expression(expr)

            if func is None:
                self.status_var.set(f"Error: Could not parse expression")
                self.ax.text(0.5, 0.5, f"Cannot parse:\n{expr}",
                             ha='center', va='center', color='white',
                             fontsize=12, transform=self.ax.transAxes)
                self.canvas.draw()
                return

            if self.auto_scale_var.get() and not self._user_set_range:
                optimal_xmin, optimal_xmax = self._detect_optimal_range(expr)
                self._log(f"[plot] Auto-scale: optimal x-range [{optimal_xmin}, {optimal_xmax}]")

                if abs(optimal_xmin - xmin) > 0.1 or abs(optimal_xmax - xmax) > 0.1:
                    self.xmin_var.set(optimal_xmin)
                    self.xmax_var.set(optimal_xmax)
                    xmin, xmax = optimal_xmin, optimal_xmax

            x_min = xmin
            x_max = xmax

            if x_max <= x_min:
                x_max = x_min + 1

            self._log(f"[plot] Using x-range: [{x_min}, {x_max}]")

            singularities = self._find_singularities(sympy_expr, x_min, x_max)

            if singularities:
                self._log(f"[plot] Found singularities at: {singularities}")
                x_segments = self._split_domain(x_min, x_max, singularities)
            else:
                x_segments = [np.linspace(x_min, x_max, 1000)]

            colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']
            legend_handles = []
            all_y_values = []

            for i, x_segment in enumerate(x_segments):
                if len(x_segment) < 2:
                    continue

                color = colors[i % len(colors)]
                #
                try:
                    x_array = np.asarray(x_segment, dtype=float)
                    y_segment = func(x_array)


                    # Convert to dB if enabled (only positive values)
                    if self._db_scale:
                        if np.any(y_segment < 0):
                            self.status_var.set("⚠️ dB requires y > 0. Use magnitude |G(jω)| not G(jω).")
                            self._log("[plot] dB error: function has negative values")
                            self._db_scale = False
                            # Don't call plot_current() - just continue without dB
                            # The current segment will be plotted without dB conversion

                        with np.errstate(divide='ignore', invalid='ignore'):
                            y_segment = np.where(y_segment > 0, 20 * np.log10(y_segment), np.nan)

                    valid_mask = ~(np.isnan(y_segment) | np.isinf(y_segment))


                    if np.any(valid_mask):
                        x_valid = x_segment[valid_mask]
                        y_valid = y_segment[valid_mask]

                        all_y_values.extend(y_valid)

                        line, = self.ax.plot(x_valid, y_valid, color=color,
                                             linewidth=2.5, alpha=0.8,
                                             label=f"Segment {i + 1}" if len(x_segments) > 1 else None)
                        legend_handles.append(line)

                        invalid_mask = ~valid_mask
                        if np.any(invalid_mask):
                            self.ax.scatter(x_segment[invalid_mask],
                                            np.zeros_like(x_segment[invalid_mask]),
                                            color='red', alpha=0.5, s=20,
                                            marker='x', zorder=3)

                except Exception as e:
                    self._log(f"[plot] Error plotting segment {i}: {e}")
                    continue

            # Set log scale if enabled
            if self._log_scale and x_min > 0:
                self.ax.set_xscale('log')
            else:
                self.ax.set_xscale('linear')

            # ========== USE ORIGINAL VARIABLE FOR AXIS LABEL ==========
            self.ax.set_xlabel(original_var, fontsize=12, color='white')
            if self._db_scale:
                self.ax.set_ylabel('|y| (dB)', fontsize=12, color='white')
            else:
                self.ax.set_ylabel('y', fontsize=12, color='white')



         #   self.ax.set_title(f"$y = {latex_str}$", fontsize=14, color='white', pad=20)
            # Convert x back to original variable for display
            display_latex = latex_str
            if original_var != 'x':
                display_latex = latex_str.replace('x', original_var)
            self.ax.set_title(f"$y = {display_latex}$", fontsize=14, color='white', pad=20)

            if len(legend_handles) > 1:
                legend = self.ax.legend(handles=legend_handles,
                                        facecolor='#2b2b2b',
                                        edgecolor='white',
                                        labelcolor='white',
                                        loc='upper right')
                legend.set_zorder(10)

            if self.auto_scale_var.get():
                self.ax.autoscale_view()

                if all_y_values and not self._user_set_range:
                    y_min_val = np.nanmin(all_y_values)
                    y_max_val = np.nanmax(all_y_values)

                    y_range = y_max_val - y_min_val
                    if y_range == 0:
                        y_range = 1
                    padding = y_range * 0.1

                    self.ymin_var.set(y_min_val - padding)
                    self.ymax_var.set(y_max_val + padding)

                    self._log(f"[plot] Auto-scaled y-range: [{y_min_val - padding}, {y_max_val + padding}]")

                current_xlim = self.ax.get_xlim()
                if current_xlim[0] != x_min or current_xlim[1] != x_max:
                    self._log(f"[plot] Adjusting x-limits from {current_xlim} to [{x_min}, {x_max}]")
                    self.ax.set_xlim(x_min, x_max)
            else:
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(self.ymin_var.get(), self.ymax_var.get())
                self._log(f"[plot] Using manual y-limits: [{self.ymin_var.get()}, {self.ymax_var.get()}]")

            self.canvas.draw_idle()
            self.status_var.set(f"✓ Plotted: {latex_str}")
            self._log(f"[plot] Plot completed successfully")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)[:50]}...")
            self._log(f"[plot] Plot error: {e}")
            import traceback
            self._log(f"[plot] Traceback: {traceback.format_exc()}")

            self.ax.text(0.5, 0.5, f"Plot Error:\n{str(e)[:100]}",
                         ha='center', va='center', color='red',
                         fontsize=12, transform=self.ax.transAxes)
            self.canvas.draw()
        finally:
            self._plotting = False

    def _find_singularities(self, sympy_expr, x_min, x_max):
        """Find singularities in the expression within domain."""
        try:
            x = sp.symbols('x')
            singularities = []

            if sympy_expr.is_rational_function(x):
                denom = sp.denom(sympy_expr)
                solutions = sp.solve(denom, x)

                for sol in solutions:
                    if sol.is_real:
                        val = float(sol)
                        if x_min <= val <= x_max:
                            singularities.append(val)

            from sympy import log
            for sub_expr in sympy_expr.atoms(log):
                arg = sub_expr.args[0]
                zero_points = sp.solve(arg, x)
                for point in zero_points:
                    if point.is_real:
                        val = float(point)
                        if x_min <= val <= x_max:
                            singularities.append(val)

            singularities = sorted(list(set(singularities)))
            return singularities

        except:
            return []

    def _on_multiplier_change(self):
        """Handle multiplier spinbox change - directly update xmax."""
        if getattr(self, '_updating_multiplier', False):
            return

        try:
            self._updating_multiplier = True
            base_xmax = 100.0  # Base value
            mult = self.xmax_mult_var.get()
            self.xmax_var.set(base_xmax * mult)
            self._user_set_range = True
            if self.auto_scale_var.get():
                self.auto_scale_var.set(False)

            # Use delayed plot to debounce rapid changes
            if self._plot_timer:
                self.master.after_cancel(self._plot_timer)
            self._plot_timer = self.master.after(300, self.plot_current)
        except:
            pass
        finally:
            self._updating_multiplier = False




    def _split_domain(self, x_min, x_max, singularities, buffer=0.01):
        """Split domain around singularities."""
        segments = []
        points = [x_min]
        for s in singularities:
            points.extend([s - buffer, s + buffer])
        points.append(x_max)

        points = sorted(set(points))

        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            if end - start > 2 * buffer:
                seg_start = start + buffer if i > 0 else start
                seg_end = end - buffer if i < len(points) - 2 else end

                if seg_end > seg_start:
                    n_points = max(100, int(1000 * (seg_end - seg_start) / (x_max - x_min)))
                    segment = np.linspace(seg_start, seg_end, n_points)
                    segments.append(segment)

        return segments if segments else [np.linspace(x_min, x_max, 1000)]

    def save_plot(self):
        """Save the current plot to file."""
        from tkinter import filedialog
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"plot_{timestamp}.png"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg;*.jpeg"),
            ]
        )

        if filepath:
            try:
                self.figure.savefig(
                    filepath,
                    facecolor='#2b2b2b',
                    edgecolor='none',
                    bbox_inches='tight',
                    dpi=300
                )
                self.status_var.set(f"✓ Saved to: {filepath}")
                self._log(f"[plot] Saved to: {filepath}")
            except Exception as e:
                self.status_var.set(f"✗ Save failed: {e}")
                self._log(f"[plot] Save error: {e}")

    def reset_view(self):
        """Reset plot view to defaults."""
        self.xmin_var.set(-10)
        self.xmax_var.set(10)
        self.ymin_var.set(-10)
        self.ymax_var.set(10)
        self.auto_scale_var.set(True)
        self._user_set_range = False
        self.plot_current()

    def show(self):
        """Show the plot window."""
        self.deiconify()
        self.lift()
        self.focus_set()