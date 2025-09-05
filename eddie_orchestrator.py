# eddie_orchestrator.py
# Orchestrates: ASR(final) -> Router -> Ollama(Qwen) -> Riva TTS (Male-1)
# Fast-ACK ("Okay.") if LLM > ACK_THRESHOLD_MS, with replacement. JSONL logging.

import os, json, time, threading, queue, re, sys, wave, tempfile
from datetime import datetime
from typing import Optional, Tuple

# --- Config via env / sane defaults ---
RIVA_SPEECH_API_URL = os.getenv("RIVA_SPEECH_API_URL", "localhost:50051")
OLLAMA_URL          = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")  # default to 11434
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
VOICE_NAME          = os.getenv("VOICE_NAME", "English-US.Male-1")
LOG_PATH            = os.getenv("EDDIE_LOG", os.path.join(os.getcwd(), "eddie_turns.log.jsonl"))
ACK_THRESHOLD_MS    = int(os.getenv("ACK_THRESHOLD_MS", "950"))          # tune 900–1100ms
REPLACE_WINDOW_MS   = int(os.getenv("REPLACE_WINDOW_MS", "2000"))
MAX_REPLY_CHARS     = int(os.getenv("MAX_REPLY_CHARS", "120"))
EDDIE_DEBUG_WAV     = os.getenv("EDDIE_DEBUG_WAV", "")                   # if set, write every wav here
OPEN_WAV_APP        = os.getenv("EDDIE_OPEN_WAV", "0") != "0"            # default OFF to reduce latency

# --- Python deps you need in your venv ---
# pip install nvidia-riva-client==2.19.* requests simpleaudio

import requests
import simpleaudio as sa

# Riva TTS client (2.19.x)
try:
    from riva.client import Auth, SpeechSynthesisService, AudioEncoding
except Exception as e:
    raise RuntimeError(
        "Riva Python client not found. Install with: pip install nvidia-riva-client==2.19.*"
    ) from e


# ---------------- Utilities ----------------

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def clip_one_sentence(text: str, max_chars: int = 120) -> str:
    # Keep the first sentence-ish; fallback to hard cap.
    clean = re.sub(r'\s+', ' ', text).strip()
    # Split on common sentence enders
    parts = re.split(r'(?<=[\.!?])\s+', clean)
    first = parts[0] if parts else clean
    if len(first) <= max_chars:
        return first
    return first[:max_chars].rstrip()

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', s.casefold())


# ---------------- Router (instant responses) ----------------

ROUTER_RULES = [
    (["eddie hi", "hey eddie", "hello eddie"],              "Hey, listening."),  # ASCII only
    (["clip that", "save that clip", "clip last"],          "Clipping the last few seconds."),
    (["start the stream", "start stream"],                  "Starting your stream."),
    (["switch to talking", "talking scene"],                "Switching to Talking."),
    (["make a note", "note this", "remember this"],         "Noted."),
    (["what did i ask you about kubernetes"],               "You set a goal for October thirty-first."),
    (["hiroshi yoshimura", "play something like hiroshi"],  "Queueing something gentle and uplifting."),
    (["how hot is the gpu", "gpu temp", "gpu temperature"], "GPU is around sixty-two degrees."),
    (["thanks eddie", "thank you eddie", "thanks"],         "You got it."),
]

def router_response(text: str) -> Optional[str]:
    t = norm(text)
    for keys, reply in ROUTER_RULES:
        if any(k in t for k in keys):
            return reply
    return None


# ---------------- Riva TTS ----------------

class RivaTTS:
    def __init__(self, uri: str, voice_name: str):
        self.voice = voice_name
        self.auth = Auth(uri=uri, use_ssl=False)
        self.tts = SpeechSynthesisService(self.auth)

    def synth(self, text: str, sample_rate: int = 44100) -> bytes:
        # Prefer raw PCM for speed
        resp = self.tts.synthesize(
            text=text,
            voice_name=self.voice,
            language_code="en-US",
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hz=sample_rate,
        )
        # The client returns either bytes or an object with `.audio`
        return resp if isinstance(resp, (bytes, bytearray)) else getattr(resp, "audio", b"")


# ---------------- Audio Playback ----------------

class AudioPlayer:
    """Play PCM16LE with simpleaudio; optionally also write a WAV for debugging."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def _maybe_write_wav(self, pcm: bytes):
        if not EDDIE_DEBUG_WAV:
            return
        path = EDDIE_DEBUG_WAV
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm)
            print(f"[Audio] wrote WAV: {path}  ({len(pcm)} bytes)")
            if OPEN_WAV_APP:
                try:
                    os.startfile(path)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Audio] WAV write failed: {e}", file=sys.stderr)

    def play_pcm16le(self, audio_bytes: bytes):
        self._maybe_write_wav(audio_bytes)
        try:
            play_obj = sa.play_buffer(audio_bytes, 1, 2, self.sample_rate)
            return play_obj
        except Exception as e:
            print(f"[Audio] simpleaudio failed: {e}", file=sys.stderr)
            # Last-ditch: try Windows winsound (blocking async)
            try:
                import winsound, tempfile
                tf = tempfile.NamedTemporaryFile(prefix="eddie_", suffix=".wav", delete=False)
                path = tf.name; tf.close()
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_bytes)
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                class _WSObj:
                    def is_playing(self): return False
                    def stop(self): winsound.PlaySound(None, 0)
                return _WSObj()
            except Exception as e2:
                print(f"[Audio] winsound fallback failed: {e2}", file=sys.stderr)
                raise

    def stop(self, play_obj):
        try:
            if hasattr(play_obj, "stop"):
                play_obj.stop()
        except Exception:
            pass


# ---------------- Ollama LLM ----------------

SYSTEM_PROMPT = (
    "You are Eddie — concise, neutral, and fast. "
    "Reply in ONE sentence (<=120 characters). No emojis, no filler."
)

def call_ollama_generate(ollama_url: str, model: str, user_text: str, timeout: float = 6.0) -> str:
    """Blocking call; returns model text (may be long — we clip later)."""
    payload = {
        "model": model,
        "prompt": user_text,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 60,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "stop": ["\n"]
        }
    }
    r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Ollama /api/generate returns {"response": "...", ...}
    return data.get("response", "").strip()


# ---------------- Orchestrator ----------------

class EddieOrchestrator:
    def __init__(self):
        self.tts = RivaTTS(RIVA_SPEECH_API_URL, VOICE_NAME)
        self.player = AudioPlayer(sample_rate=44100)

    def _speak(self, text: str) -> Tuple[int, object]:
        t0 = time.perf_counter()
        audio = self.tts.synth(text, sample_rate=44100)
        t1 = time.perf_counter()
        play_obj = self.player.play_pcm16le(audio)
        l_tts_ms = int((t1 - t0) * 1000)
        return l_tts_ms, play_obj

    def handle_final_transcript(self, final_text: str, asr_latency_ms: int = 0) -> dict:
        turn_start = time.perf_counter()
        ts_iso = now_iso()
        final_text = (final_text or "").strip()

        # 1) Router — instant
        routed_reply = router_response(final_text)
        if routed_reply:
            l_tts_ms, play_obj = self._speak(routed_reply)
            first_audio_ms = int((time.perf_counter() - turn_start) * 1000)
            log = {
                "ts": ts_iso,
                "text": final_text,
                "reply": routed_reply,
                "l_asr_ms": asr_latency_ms,
                "l_llm_ms": 0,
                "l_tts_ms": l_tts_ms,
                "l_total_ms": first_audio_ms,
                "router": True,
                "ack": False,
                "ack_replaced": False,
                "voice": VOICE_NAME,
            }
            self._append_log(log)
            return log

        # 2) Else LLM with fast-ACK
        ack_sent = False
        ack_replaced = False
        ack_play = None
        l_llm_ms = 0

        # Fire LLM in a thread so we can ACK if it drags
        llm_out_q: "queue.Queue[str]" = queue.Queue(maxsize=1)
        llm_timing: dict = {}

        def llm_worker():
            t0 = time.perf_counter()
            try:
                resp = call_ollama_generate(OLLAMA_URL, OLLAMA_MODEL, final_text, timeout=6.0)
            except Exception:
                resp = "Sorry, I had a hiccup."
            t1 = time.perf_counter()
            llm_timing["ms"] = int((t1 - t0) * 1000)
            try:
                llm_out_q.put_nowait(resp)
            except queue.Full:
                pass

        thread = threading.Thread(target=llm_worker, daemon=True)
        thread.start()

        # Wait up to ACK_THRESHOLD_MS for LLM
        try:
            resp = llm_out_q.get(timeout=ACK_THRESHOLD_MS / 1000.0)
            l_llm_ms = llm_timing.get("ms", ACK_THRESHOLD_MS)
        except queue.Empty:
            # Send fast ACK
            ack_sent = True
            _, ack_play = self._speak("Okay.")
            # Now wait; replacement handled below
            resp = llm_out_q.get()
            l_llm_ms = llm_timing.get("ms", 0)

        # Prepare final reply (clip to 1 sentence/120 chars)
        final_reply = clip_one_sentence(resp, MAX_REPLY_CHARS)

        # If ACK is still playing, stop & replace
        if ack_sent and ack_play:
            try:
                self.player.stop(ack_play)
                ack_replaced = True
            except Exception:
                pass

        l_tts_ms, _ = self._speak(final_reply)
        first_audio_ms = int((time.perf_counter() - turn_start) * 1000)

        log = {
            "ts": ts_iso,
            "text": final_text,
            "reply": final_reply,
            "l_asr_ms": asr_latency_ms,
            "l_llm_ms": l_llm_ms,
            "l_tts_ms": l_tts_ms,
            "l_total_ms": first_audio_ms,   # time to first audio start
            "router": False,
            "ack": ack_sent,
            "ack_replaced": ack_replaced,
            "voice": VOICE_NAME,
        }
        self._append_log(log)
        return log

    def _append_log(self, obj: dict):
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ---------------- Public API ----------------

_orchestrator_singleton: Optional[EddieOrchestrator] = None

def get_orchestrator() -> EddieOrchestrator:
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = EddieOrchestrator()
    return _orchestrator_singleton

def handle_final_transcript(final_text: str, asr_latency_ms: int = 0) -> dict:
    """Call this from your existing PTT/ASR code on PTT release."""
    return get_orchestrator().handle_final_transcript(final_text, asr_latency_ms)


# ---------------- CLI smoke test ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Eddie orchestrator smoke test")
    ap.add_argument("--text", type=str, default="Eddie, hi.", help="Final ASR transcript")
    ap.add_argument("--asr_ms", type=int, default=150, help="ASR finalize latency (ms)")
    args = ap.parse_args()
    out = handle_final_transcript(args.text, args.asr_ms)
    print(json.dumps(out, indent=2))
