import pyttsx3
import sys
import traceback

print("Python version:", sys.version)
print("\nTesting pyttsx3 SAPI voices...")

try:
    # Method 1: Direct initialization
    print("\n1. Trying pyttsx3.init()...")
    engine = pyttsx3.init()
    print("   ✓ Engine initialized")

    # Get voices
    print("\n2. Getting voices...")
    voices = engine.getProperty('voices')
    print(f"   ✓ Found {len(voices)} voices")

    if len(voices) == 0:
        print("   ✗ No voices found! This is the problem.")
        print("\nPossible solutions:")
        print("   a) Run this as Administrator")
        print("   b) Check Windows Speech settings")
        print("   c) Try: engine = pyttsx3.init('sapi5')")

        # Try with explicit sapi5
        print("\n3. Trying explicit sapi5...")
        engine2 = pyttsx3.init('sapi5')
        voices2 = engine2.getProperty('voices')
        print(f"   Found {len(voices2)} voices with sapi5")
        for v in voices2:
            print(f"   - {v.name}")
        engine2.stop()
    else:
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"   {i}: {voice.name}")
            print(f"      ID: {voice.id}")

    engine.stop()

except Exception as e:
    print(f"\n✗ Error: {e}")
    traceback.print_exc()
    print("\nTrying fallback method...")

    # Fallback to comtypes
    try:
        import comtypes.client

        print("\n4. Trying comtypes...")
        voice_token = comtypes.client.CreateObject("SAPI.SpVoice")
        voices = voice_token.GetVoices()
        print(f"   Found {voices.Count} voices via comtypes")
        for i in range(voices.Count):
            voice = voices.Item(i)
            print(f"   {i}: {voice.GetDescription()}")
    except ImportError:
        print("   comtypes not installed. Install with: pip install comtypes")
    except Exception as e2:
        print(f"   comtypes error: {e2}")