"""
Wake watcher (Porcupine + sounddevice)

This helper listens for the wake phrase and signals Eddie:
- HTTP POST to WAKE_URL if available (legacy orchestrator wake server)
- Touches a local wake-flag file whose mtime acts as a 20s window that the
  mic client can honor without any HTTP server requirement.

Configure via environment:
  PORCUPINE_ACCESS_KEY   (required)
  PORCUPINE_KEYWORD_PATH (required)
  WAKE_URL               (optional; default http://127.0.0.1:6060/signal/wake)
  EDDIE_WAKE_FLAG        (optional; default ../eddie_wake.flag next to repo root)
  WAKE_WINDOW_MS         (optional; default 20000; used by mic)
"""

import os
import queue
import sys

import requests
import sounddevice as sd
import pvporcupine as pv

ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "").strip()
KEYWORD_PATH = os.getenv("PORCUPINE_KEYWORD_PATH", "./Hello-Eddie.ppn")
WAKE_URL = os.getenv("WAKE_URL", "http://127.0.0.1:6060/signal/wake")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAKE_FLAG = os.getenv("EDDIE_WAKE_FLAG", os.path.join(_ROOT, "eddie_wake.flag"))


def _post_wake():
    try:
        requests.post(WAKE_URL, timeout=0.5)
    except Exception:
        # Orchestrator HTTP wake server may be offline; that's fine.
        pass
    # Always touch/update the local wake flag so the mic can honor wake
    try:
        # Create or update mtime
        with open(WAKE_FLAG, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("wake\n")
        os.utime(WAKE_FLAG, None)
    except Exception:
        print(f"[wakewatch] failed to write wake flag: {WAKE_FLAG}", file=sys.stderr)


def main():
    if not ACCESS_KEY or not KEYWORD_PATH:
        print("[wakewatch] PORCUPINE_ACCESS_KEY or PORCUPINE_KEYWORD_PATH not set; exiting.", file=sys.stderr)
        return

    keyword_path = os.path.abspath(KEYWORD_PATH)
    if not os.path.exists(keyword_path):
        print(f"[wakewatch] Keyword file missing: {keyword_path}", file=sys.stderr)
        return

    try:
        porcupine = pv.create(access_key=ACCESS_KEY, keyword_paths=[keyword_path])
    except pv.PorcupineError as exc:
        print(f"[wakewatch] Porcupine init failed: {exc}", file=sys.stderr)
        return

    q: "queue.Queue[bytes]" = queue.Queue()

    def audio_cb(indata, frames, time_info, status):  # type: ignore[unused-argument]
        if status:
            # Non-fatal; Porcupine can tolerate sporadic glitches.
            print(f"[wakewatch] audio status: {status}", file=sys.stderr)
        q.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype="int16",
        channels=1,
        callback=audio_cb,
    )

    stream.start()
    try:
        while True:
            pcm = q.get()
            if pcm is None:
                break
            frame = memoryview(pcm).cast("h")
            if porcupine.process(frame) >= 0:
                _post_wake()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        porcupine.delete()


if __name__ == "__main__":
    main()
