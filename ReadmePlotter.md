# 📊 Mathematical Function Plotter - README

## Overview

A sophisticated voice-driven mathematical function plotter that extracts LaTeX expressions from AI responses and automatically plots them with intelligent variable tracking, auto-scaling, and comprehensive LaTeX parsing support.

---

## 🌟 Key Features

### **Voice-Driven Plotting**
- Integrates with voice-controlled AI assistant
- Automatic LaTeX extraction from AI responses
- Command: "graph that" or "plot that" after asking mathematical questions

### **Multi-Variable Support**
- Automatically detects and tracks original variables
- Supported variables: `x`, `t`, `λ` (lambda), `θ` (theta), `k`, `n`, `m`, `i`, `j`
- Axis labels display the original variable (e.g., `t` for time-domain functions)

### **Advanced LaTeX Parsing**
- Handles `\boxed{}`, `$$...$$`, `$...$`, `\[...\]` delimiters
- Supports nested expressions and complex notation
- Intelligent escape sequence handling (`\theta`, `\lambda`, `\pi`)

### **Symbolic Expression Handling**
- Keeps undefined constants symbolic (e.g., `exp(λt)` displays as-is)
- Warns when multi-variable expressions cannot be plotted numerically

### **Auto-Scaling**
- Intelligent domain detection based on function type
- Exponential decay: optimizes range to `[0, 5]`
- Polynomial: scales based on degree
- User can override with manual ranges

### **Professional UI**
- Dark-themed matplotlib interface
- Interactive toolbar with coordinate display
- Navigation controls for multiple expressions
- Save plots to PNG/PDF/SVG/JPEG

---

## 📐 Supported Mathematical Notation

### **LaTeX Commands**
```latex
\sin, \cos, \tan          → Trigonometric functions
\arctan, \asin, \acos     → Inverse trig functions
\sqrt{x}                  → Square root
\frac{a}{b}               → Fractions (converted to (a)/(b))
\boxed{expr}              → Boxed expressions
\lambda, \theta, \pi      → Greek letters and constants
\exp, \log, \ln           → Exponential and logarithmic
\left(...\right)          → Delimiter sizing (removed)
\cdot, \times             → Multiplication operators
```

### **Special Patterns**
```latex
\sin^2(x)                 → (sin(x))**2
e^{-t}                    → exp(-t)
x \cdot 3                 → x*3
2\sin(x)                  → 2*sin(x)
```

### **Variables**
| Variable | LaTeX | Display | Example |
|----------|-------|---------|---------|
| x | `x` | x | `$x^2 + 3x$` |
| t (time) | `t` | t | `$e^{-t}$` |
| lambda | `\lambda` or `λ` | λ | `$\lambda^2 + 1$` |
| theta | `\theta` or `θ` | θ | `$\sin(\theta)$` |
| k, n, m, i, j | `k`, `n`, etc. | k, n, etc. | `$0.5^k$` |

---

## 🎯 Usage Examples

### **Basic Polynomial**
```
User: "What is x squared plus 3x plus 2?"
AI: $\boxed{x^2 + 3x + 2}$
User: "graph that"
→ Plots: x² + 3x + 2
→ Axis: x
```

### **Exponential Decay**
```
User: "Plot the exponential decay function e to the minus t"
AI: $\boxed{y(t) = e^{-t}}$
User: "graph that"
→ Plots: exp(-t) with t converted to x
→ Axis: t (original variable displayed)
→ Auto-range: [0, 5]
```

### **Trigonometric Functions**
```
User: "Graph sine squared of x"
AI: $$\boxed{\sin^2(x)}$$
→ Plots: (sin(x))²
→ Range: [-10, 10]
```

### **Complex Expressions**
```
User: "Integrate x over x squared plus 5"
AI: $$\boxed{x - \sqrt{5} \arctan\left(\frac{x}{\sqrt{5}}\right) + C}$$
→ Plots: x - √5·arctan(x/√5)
→ Removes integration constant +C
```

### **Lambda Functions**
```
User: "What's lambda squared plus 3 lambda plus 2?"
AI: $\boxed{\lambda^2 + 3\lambda + 2}$
→ Plots: x² + 3x + 2
→ Axis: λ (displays lambda symbol)
```

---

## ⚙️ Technical Details

### **Variable Conversion Logic**
1. **Detection**: Identify primary variable (t, λ, θ, etc.) from original expression
2. **Conversion**: Convert primary variable to `x` for plotting
3. **Tracking**: Store original variable for axis labeling
4. **Secondary Variables**: Keep symbolic (don't convert constants)

**Example Flow:**
```
Input:   e^{λt}
Detect:  Primary variable = λ
Convert: e^{λt} → e^{xt} → e^{x*t} → exp(x*t)
Track:   original_variable = 'λ'
Result:  Plot exp(x*t) with x-axis labeled 'λ'
Note:    't' remains symbolic (undefined constant)
```

### **Expression Cleanup Pipeline**
```
Step 1: Extract LaTeX from delimiters ($, $$, \boxed, \[])
Step 2: Handle equations (y = ... → extract right side)
Step 3: Detect original variable
Step 4: Convert variable to x
Step 5: Protect function names (sin, cos, exp, etc.)
Step 6: Clean special patterns (e**, func^n, etc.)
Step 7: Restore function names
Step 8: Validate and plot
```

### **Auto-Scaling Algorithm**
- **Decay Functions** (`exp(-kx)`): Range [0, 3/k]
- **Growth Functions** (`exp(kx)`): Range [-3, 3]
- **Polynomials**: Range [-degree, degree]
- **Default**: [-10, 10]

### **Singularity Handling**
- Detects vertical asymptotes (e.g., `1/x` at x=0)
- Splits domain into segments around singularities
- Plots each segment separately
- Visual indicators for discontinuities

---

## 🚫 Limitations & Known Issues

### **Cannot Plot:**
1. **Discrete sequences**: `f[n] = ...` notation (use `f(x)` instead)
2. **Multi-variable functions**: `f(x,y) = ...` (needs one variable)
3. **Symbolic expressions with undefined constants**: `exp(at)` where `a` is unknown
4. **Inequalities**: `x > 5` (shows as non-function)
5. **Equation solutions**: `\lambda = -3` (constant, not function)

### **Workarounds:**
- **Discrete → Continuous**: `f[n] = n·(-0.5)^n` → `f(x) = x·(-0.5)^x`
- **Multi-variable**: Ask for specific case (e.g., "plot with a=1")
- **Symbolic**: Define constants first

---

## 🎨 UI Controls

### **Navigation** (when multiple expressions)
- **◀ Prev**: Previous expression
- **Next ▶**: Next expression

### **Range Controls**
- **x min/max**: Domain range
- **y min/max**: Range (output values)
- **Auto-scale**: Toggle automatic range detection
- **Enter**: Apply manual range (disables auto-scale)

### **Actions**
- **📈 Plot**: Refresh plot with current settings
- **💾 Save**: Export to PNG/PDF/SVG/JPEG
- **🔄 Reset**: Return to default ranges

### **Toolbar** (bottom)
- **Pan**: Drag to move view
- **Zoom**: Zoom rectangle or zoom in/out
- **Home**: Reset to original view
- **Back/Forward**: Navigate view history
- **Save**: Quick save dialog
- **Coordinates**: Hover shows (x, y) values

---

## 📝 Voice Commands

### **Plotting Commands**
```
"graph that"
"plot that"
"show me a graph"
"can you plot this"
```

### **Mathematical Questions That Auto-Plot**
```
"What is x squared plus 5?"
"Solve x squared minus 4 equals 0"
"What's the derivative of x cubed?"
"Graph sine of x"
"Plot e to the minus t"
```

---

## 🔧 Configuration

### **Window Size**
Default: 1000x900 (adjustable in `PlotWindow.__init__`)

### **Plot Colors**
- Curve: Cyan, Magenta, Yellow, Lime, Orange (cycles for segments)
- Background: Dark gray (#2b2b2b)
- Grid: White, semi-transparent
- Text: White

### **Default Ranges**
- x: [-10, 10]
- y: [-10, 10]
- Resolution: 1000 points per segment

---

## 🐛 Troubleshooting

### **"No expressions found"**
- Check if LaTeX has proper delimiters: `$`, `$$`, `\boxed{}`, `\[...\]`
- Verify expression is a function, not just text

### **"Cannot parse expression"**
- Check for unsupported notation
- Verify all braces are balanced
- Look for typos in function names

### **Blank plot with "exp(x*t)"**
- Expression has multiple variables
- One is undefined (symbolic)
- Define all constants or use single-variable form

### **"s*qrt" or "p*i" errors**
- Fixed in latest version
- Update to current plotter.py

### **Coordinates not showing**
- Stretch window vertically
- Toolbar is at bottom (may be hidden if window too short)

---

## 📊 Example Test Cases

```python
# Polynomials
"$x^2 + 3x + 2$"                    → x² + 3x + 2

# Exponentials  
"$e^{-t}$"                          → exp(-t), axis: t
"$y(t) = 1 - e^{-t}$"               → 1 - exp(-t), axis: t

# Trigonometric
"$\sin(\theta)$"                    → sin(x), axis: θ
"$\sin^2(x)$"                       → (sin(x))²

# Complex
"$\sqrt{5} \arctan\left(\frac{x}{\sqrt{5}}\right)$"  → √5·arctan(x/√5)

# Lambda
"$\lambda^2 + 3\lambda + 2$"        → x² + 3x + 2, axis: λ

# Fractions
"$\frac{x}{x^2 + 1}$"               → x/(x² + 1)
```

---

## 🚀 Future Enhancements

- [ ] Parametric plots (x(t), y(t))
- [ ] 3D surface plots
- [ ] Implicit function plotting
- [ ] Polar coordinates
- [ ] Animation support
- [ ] Multiple functions on same plot
- [ ] Custom color schemes
- [ ] LaTeX equation editor

---

## 📄 License

Part of MMAssistant2 voice-driven AI assistant system.

---

## 🙏 Credits

Built for voice-driven mathematical interaction with AI assistants. Integrates:
- **Matplotlib**: Plotting backend
- **SymPy**: Symbolic mathematics
- **NumPy**: Numerical computation
- **Tkinter**: GUI framework

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Production Ready ✅
