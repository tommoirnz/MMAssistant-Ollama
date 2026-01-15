# test_edge_voices.py
import asyncio
import os

try:
    import edge_tts

    print("✅ edge-tts is installed")
except ImportError:
    print("❌ edge-tts not installed. Run: pip install edge-tts")
    exit()

# Create output folder
os.makedirs("voice_tests", exist_ok=True)

# Voices to test
voices_to_test = [
    # English
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    # French
    "fr-FR-DeniseNeural",
    "fr-FR-HenriNeural",
    "fr-FR-BrigitteNeural",
    "fr-FR-CelesteNeural",
    # German
    "de-DE-KatjaNeural",
    "de-DE-ConradNeural",
    # Spanish
    "es-ES-ElviraNeural",
    "es-ES-AlvaroNeural",
    "es-MX-DaliaNeural",
    # Russian
    "ru-RU-SvetlanaNeural",
    "ru-RU-DmitryNeural",
    # Hebrew
    "he-IL-HilaNeural",
    "he-IL-AvriNeural",
    # Italian
    "it-IT-ElsaNeural",
    "it-IT-DiegoNeural",
    # Japanese
    "ja-JP-NanamiNeural",
    "ja-JP-KeitaNeural",
    # Chinese
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
]

test_text = "Hello, this is a test of the voice synthesis system."


async def test_voice(voice_name):
    """Test a single voice"""
    output_file = f"voice_tests/{voice_name}.mp3"
    try:
        communicate = edge_tts.Communicate(test_text, voice_name)
        await communicate.save(output_file)

        # Check if file was created and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            print(f"✅ {voice_name} - OK ({os.path.getsize(output_file)} bytes)")
            return True
        else:
            print(f"❌ {voice_name} - File too small or empty")
            return False
    except Exception as e:
        print(f"❌ {voice_name} - ERROR: {e}")
        return False


async def test_all_voices():
    """Test all voices"""
    print("\n" + "=" * 60)
    print("EDGE TTS VOICE TEST")
    print("=" * 60 + "\n")

    working = []
    failed = []

    for voice in voices_to_test:
        result = await test_voice(voice)
        if result:
            working.append(voice)
        else:
            failed.append(voice)
        await asyncio.sleep(0.5)  # Small delay between tests

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n✅ WORKING VOICES ({len(working)}):")
    for v in working:
        print(f"   - {v}")

    if failed:
        print(f"\n❌ FAILED VOICES ({len(failed)}):")
        for v in failed:
            print(f"   - {v}")

    print(f"\n📁 Audio files saved to: voice_tests/")
    print("\nYou can play the .mp3 files to hear each voice.")


async def list_all_available_voices():
    """List ALL available Edge voices"""
    print("\n" + "=" * 60)
    print("ALL AVAILABLE EDGE VOICES")
    print("=" * 60 + "\n")

    voices = await edge_tts.list_voices()

    # Group by language
    by_language = {}
    for v in voices:
        locale = v['Locale']
        lang = locale.split('-')[0]
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append(v)

    for lang in sorted(by_language.keys()):
        print(f"\n=== {lang.upper()} ===")
        for v in by_language[lang]:
            gender = v.get('Gender', '?')
            name = v['ShortName']
            locale = v['Locale']
            print(f"  {name:<30} ({gender}, {locale})")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Test specific voices")
    print("2. List ALL available voices")
    print("3. Both")

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        asyncio.run(test_all_voices())
    elif choice == "2":
        asyncio.run(list_all_available_voices())
    elif choice == "3":
        asyncio.run(list_all_available_voices())
        asyncio.run(test_all_voices())
    else:
        print("Running voice test by default...")
        asyncio.run(test_all_voices())