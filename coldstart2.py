import subprocess
import time
import requests
import sys
import psutil
import os

# ================= CONFIG =================
OLLAMA_URL = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen2.5:7b-instruct"
VISION_MODEL = "qwen2.5vl:7b"

STARTUP_TIMEOUT_S = 15
CHAT_TIMEOUT_S = 10
GENERATE_TIMEOUT_S = 12
# ==========================================


def is_ollama_running():
    """Quick ping to Ollama API."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def is_ollama_process_running():
    """Detect ollama.exe process even if API not responsive."""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'ollama' in proc.info['name'].lower():
            return True
    return False


def graceful_restart_ollama():
    """Restart Ollama only if needed."""
    print("[*] Checking Ollama status...")

    if is_ollama_running():
        print("[✓] Ollama is already running and responsive")
        return True

    if is_ollama_process_running():
        print("[!] Ollama process exists but not responding – trying to stop...")
        try:
            subprocess.run(["ollama", "serve", "--stop"], timeout=5,
                           capture_output=True)
            time.sleep(2)
        except Exception:
            pass

        subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

    print("[*] Starting Ollama server in background...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    except Exception as e:
        print(f"[✗] Could not start Ollama: {e}")
        return False

    print("[*] Waiting for Ollama to become responsive...")
    for i in range(STARTUP_TIMEOUT_S):
        if is_ollama_running():
            print("[✓] Ollama started successfully")
            return True
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"[*] ... still waiting ({i + 1}/{STARTUP_TIMEOUT_S})")

    print("[✗] Ollama did not become ready in time")
    return False


def check_models_available():
    """Return list of installed models."""
    print("[*] Checking available models...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"[✓] Available models ({len(models)}): {models}")
        return models
    except Exception as e:
        print(f"[✗] Cannot check models: {e}")
        return []


def warm_model_with_retry(model, max_retries=2):
    """Warm model without hanging forever."""
    for attempt in range(1, max_retries + 1):
        start = time.time()
        print(f"[*] Warming '{model}' (attempt {attempt}/{max_retries})...")

        # --- (1) Try streaming chat ---
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Say ready."}],
                "stream": True
            }
            with requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=CHAT_TIMEOUT_S,
                stream=True
            ) as r:
                r.raise_for_status()
                got_any = False
                for chunk in r.iter_lines():
                    if chunk:
                        got_any = True
                        break
                if got_any:
                    elapsed = time.time() - start
                    print(f"[✓] {model} responded to chat stream in {elapsed:.1f}s")
                    return True
        except requests.exceptions.Timeout:
            print(f"[!] {model} chat warm-up timed out (stream).")
        except Exception as e:
            print(f"[!] {model} chat warm-up failed: {e}")

        # --- (2) Fallback small /api/generate ---
        print(f"[*] Trying fallback /api/generate for '{model}'...")
        try:
            payload = {
                "model": model,
                "prompt": "ready",
                "stream": False
            }
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=GENERATE_TIMEOUT_S
            )
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"[✓] {model} responded via generate in {elapsed:.1f}s")
                return True
            else:
                print(f"[!] {model} generate returned {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[!] {model} generate timed out.")
        except Exception as e:
            print(f"[!] {model} generate failed: {e}")

        if attempt < max_retries:
            time.sleep(2)

    return False


def load_model_to_memory(model):
    """Ping generate with keep_alive to keep model cached."""
    print(f"[*] Loading '{model}' to memory (keep_alive=30m)...")
    try:
        payload = {
            "model": model,
            "prompt": "ping",
            "stream": False,
            "keep_alive": "30m"
        }
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=GENERATE_TIMEOUT_S
        )
        if r.status_code == 200:
            print(f"[✓] {model} loaded/pinned")
            return True
        else:
            print(f"[!] {model} load returned {r.status_code}: {r.text[:200]}")
    except requests.exceptions.Timeout:
        print(f"[!] {model} load timed out – likely still loading.")
    except Exception as e:
        print(f"[!] Could not load {model}: {e}")
    return False


def main():
    print("=== Smart Ollama Manager (final version) ===")

    if not graceful_restart_ollama():
        sys.exit(1)

    models = check_models_available()
    if not models:
        print("[✗] No models reported by Ollama – is it still starting/pulling?")
        sys.exit(1)

    required = [TEXT_MODEL, VISION_MODEL]
    missing = [m for m in required if m not in models]
    if missing:
        print(f"[!] Missing models: {missing}")
        print("    Install with:")
        print("    " + " && ".join(f"ollama pull {m}" for m in missing))

    success_count = 0

    if TEXT_MODEL in models:
        if warm_model_with_retry(TEXT_MODEL):
            load_model_to_memory(TEXT_MODEL)
            success_count += 1

    if VISION_MODEL in models:
        if warm_model_with_retry(VISION_MODEL):
            load_model_to_memory(VISION_MODEL)
            success_count += 1

    if success_count == 0:
        print("[✗] Failed to warm up any models (possibly still downloading).")
        sys.exit(1)
    else:
        print(f"[✓] Warmed up {success_count} model(s). Ollama is ready.")


if __name__ == "__main__":
    main()
