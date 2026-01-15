import subprocess
import shutil
# If this doesn't run it won't render maths properly
# Check what's available
for cmd in ["pdflatex", "latex", "xelatex", "lualatex"]:
    path = shutil.which(cmd)
    if path:
        print(f"{cmd}: {path}")

        # Get version
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            print(f"  Version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
        except:
            print("  Could not get version")