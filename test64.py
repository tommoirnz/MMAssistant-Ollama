import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}")
print(f"base64 module location: {base64.__file__ if 'base64' in sys.modules else 'Not loaded yet'}")