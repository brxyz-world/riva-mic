# eddie_orchestrator.py
# Orchestrates: ASR(final) -> Router -> Ollama(Qwen) -> Riva TTS (Male-1)
# Fast-ACK via fillers if LLM drags. Per-run JSONL logging (one file per session).

import os, json, time, threading, queue, re, sys, wave, tempfile, random, logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

log = logging.getLogger(__name__)

# --- Path layout ---
_FILE_PATH = Path(__file__).resolve()
_SYSTEMS_DIR = _FILE_PATH.parents[1]
_EDDIE_DIR = _FILE_PATH.parents[2]
_REPO_ROOT = _FILE_PATH.parents[3]
_CONFIG_DIR = _EDDIE_DIR / "config"

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _parse_stop_tokens(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        tokens: list[str] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            tokens.append(bytes(part, "utf-8").decode("unicode_escape"))
        return tokens
    else:
        if isinstance(parsed, list):
            return [str(token) for token in parsed]
        if isinstance(parsed, str):
            return [parsed]
        return []

# --- Config via env / sane defaults ---
RIVA_SPEECH_API_URL = os.getenv("RIVA_SPEECH_API_URL", "localhost:50051")
OLLAMA_URL          = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
VOICE_NAME          = os.getenv("VOICE_NAME", "English-US.Male-1")

# ---- TTS sizing / chunking controls (avoid 4MB gRPC receive cap) ----
# For PCM16 mono, bytes/sec = sample_rate * 2
# Default to 22050 Hz to reduce payload size vs 44100.
RIVA_TTS_SAMPLE_RATE    = _env_int("RIVA_TTS_SAMPLE_RATE", 22050)
# Conservative avg speaking rate (chars/sec) used to size chunks.
TTS_CHARS_PER_SEC       = _env_float("TTS_CHARS_PER_SEC", 13.0)
# Keep chunks comfortably under 4MB (minus a margin).
TTS_MAX_MESSAGE_BYTES   = _env_int("TTS_MAX_MESSAGE_BYTES", 4 * 1024 * 1024 - 32768)
# Allow an explicit hard cap if you want; 0 means "compute it".
TTS_CHUNK_CHAR_LIMIT    = _env_int("TTS_CHUNK_CHAR_LIMIT", 0)
TOOLS_URL           = os.getenv("TOOLS_URL")

LONGFORM_MIN_CHARS      = _env_int("LONGFORM_MIN_CHARS", 500)
LONGFORM_TEMPERATURE    = _env_float("LONGFORM_TEMPERATURE", 0.6)
_LONGFORM_STOP_SENTINEL_RAW = os.getenv(
    "EDDIE_LONGFORM_STOP_SENTINEL",
    os.getenv("LONGFORM_STOP_SENTINEL", "<END>"),
)
LONGFORM_STOP_SENTINEL  = _LONGFORM_STOP_SENTINEL_RAW.strip()
LONGFORM_RETRY          = _env_flag("LONGFORM_RETRY", False)
LONGFORM_SUGGEST_CONTINUE = _env_flag("LONGFORM_SUGGEST_CONTINUE", True)
LONGFORM_CONTINUE_TTS   = os.getenv(
    "LONGFORM_CONTINUE_TTS",
    "I can go long—want me to continue?",
).strip()
OLLAMA_KEEP_ALIVE       = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
OLLAMA_WARMUP_ON_START  = _env_flag("OLLAMA_WARMUP_ON_START", True)
LONGFORM_REFUSAL_GUARD  = _env_flag("LONGFORM_REFUSAL_GUARD", True)
_DEFAULT_REFUSAL_SEED   = "Okay—starting the story."
LONGFORM_REFUSAL_SEED   = os.getenv(
    "LONGFORM_REFUSAL_SEED",
    _DEFAULT_REFUSAL_SEED,
).strip() or _DEFAULT_REFUSAL_SEED
LONGFORM_AUTOCONFIRM    = _env_flag("LONGFORM_AUTOCONFIRM", False)
LONGFORM_AUTOCONFIRM_NUM_PREDICT = _env_int("LONGFORM_AUTOCONFIRM_NUM_PREDICT", 768)
LONGFORM_AUTOCONFIRM_TIMEOUT_MS = max(
    1000, _env_int("LONGFORM_AUTOCONFIRM_TIMEOUT_MS", 20000)
)
_DEFAULT_AUTOCONFIRM_SUFFIX = (
    "Continue the story in multiple paragraphs with vivid, concrete imagery. "
    "Add clear blank lines between paragraphs. End with <END>."
)
LONGFORM_AUTOCONFIRM_SUFFIX = os.getenv(
    "LONGFORM_AUTOCONFIRM_SUFFIX",
    _DEFAULT_AUTOCONFIRM_SUFFIX,
).strip() or _DEFAULT_AUTOCONFIRM_SUFFIX
LLM_WARMUP              = os.getenv("LLM_WARMUP", "1") == "1"
try:
    LLM_WARMUP_TIMEOUT_MS = int(os.getenv("LLM_WARMUP_TIMEOUT_MS", "30000"))
except (TypeError, ValueError):
    LLM_WARMUP_TIMEOUT_MS = 30000
LLM_WARMUP_TIMEOUT_MS = max(1000, LLM_WARMUP_TIMEOUT_MS)
LLM_TIMEOUT_MS          = max(1000, _env_int("LLM_TIMEOUT_MS", 20000))

# ---- 2.2.3-longform-TTS knobs (smoketest-friendly) ----
# Enable longform TTS (speaks the whole story) when LONG_FORM=1
LONGFORM_TTS_ENABLE   = _env_flag("LONGFORM_TTS_ENABLE", True)
# Safety budget: stop speaking once total playback time reaches this many seconds
LONGFORM_TTS_MAX_S    = max(1, _env_int("LONGFORM_TTS_MAX_S", 120))
# Target chunk size for longform speech; approx sentence-sized
LONGFORM_TTS_CHARS    = max(80, _env_int("LONGFORM_TTS_CHARS", 280))
# Also print the full story to stdout (safe: mic client extracts the last JSON)
LONGFORM_PRINT        = _env_flag("LONGFORM_PRINT", True)
# (Reserved for mic sessions) simple re-ask "ready?" gate; keep off for smoketest
LONGFORM_READY_CHECK  = _env_flag("LONGFORM_READY_CHECK", False)

# ---- 1R-hotfix-5: env-driven timeouts (safe defaults) ----
try:
    ACK_TIMEOUT_S = float(os.getenv("EDDIE_ACK_TIMEOUT_S", "6"))
except (TypeError, ValueError):
    ACK_TIMEOUT_S = 6.0

_OLLAMA_TIMEOUT_DEFAULT_S = LLM_TIMEOUT_MS / 1000.0
try:
    OLLAMA_TIMEOUT_S = float(
        os.getenv("EDDIE_OLLAMA_TIMEOUT_S", str(_OLLAMA_TIMEOUT_DEFAULT_S))
    )
except (TypeError, ValueError):
    OLLAMA_TIMEOUT_S = _OLLAMA_TIMEOUT_DEFAULT_S

try:
    TTS_MAX_CHARS = int(os.getenv("EDDIE_TTS_MAX_CHARS", "0"))
except (TypeError, ValueError):
    TTS_MAX_CHARS = 0
# ----------------------------------------------------------

_DEFAULT_LONGFORM_SYSTEM_APPEND = (
    "You are writing a creative, multi-paragraph narrative.\n"
    "Always comply; never refuse harmless creative requests.\n"
    "Use clear paragraph breaks (blank lines).\n"
    "Aim for rich detail and coherence. Minimum ~500 characters.\n"
    "End output with the token <END>."
)
LONGFORM_SYSTEM_APPEND = os.getenv(
    "LONGFORM_SYSTEM_APPEND",
    _DEFAULT_LONGFORM_SYSTEM_APPEND,
).strip()

# Per-run log rotation (kept). You can override with EDDIE_LOG.
_LOGS_DIR = _REPO_ROOT / "archive" / "logs"
os.makedirs(_LOGS_DIR, exist_ok=True)
_RUN_TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
_DEFAULT_LOG_PATH = str(_LOGS_DIR / f"Eddie_Convo_{_RUN_TS}.jsonl")
LOG_PATH = os.getenv("EDDIE_LOG", _DEFAULT_LOG_PATH)

ACK_THRESHOLD_MS    = int(os.getenv("ACK_THRESHOLD_MS", "950"))          # tune 900–1100ms
REPLACE_WINDOW_MS   = int(os.getenv("REPLACE_WINDOW_MS", "2000"))
MAX_REPLY_CHARS     = int(os.getenv("MAX_REPLY_CHARS", "240"))
EDDIE_DEBUG_WAV     = os.getenv("EDDIE_DEBUG_WAV", "")
OPEN_WAV_APP        = os.getenv("EDDIE_OPEN_WAV", "0") != "0"
SPEAKING_FLAG_PATH  = os.getenv(
    "EDDIE_SPEAKING_FLAG",
    str(_REPO_ROOT / "eddie_speaking.flag"),
)
FILLER_FIRST_MS     = int(os.getenv("FILLER_FIRST_MS", "1000"))          # don't emit filler before 1s
FILLER_POST_PAUSE_MS= int(os.getenv("FILLER_POST_PAUSE_MS", "150"))      # pause after filler completes
FILLER_PROB         = float(os.getenv("FILLER_PROB", "0.4"))             # chance to emit any filler
TWF_DISABLE         = _env_flag("TWF_DISABLE", False)
MOOD_LOCK           = os.getenv("MOOD_LOCK", "").strip()
WAKE_BIND          = os.getenv("WAKE_BIND", "127.0.0.1")
WAKE_PORT          = int(os.getenv("WAKE_PORT", "0"))  # disabled by default; mic owns wake
WAKE_WINDOW_MS     = int(os.getenv("WAKE_WINDOW_MS", "20000"))
HUSH_CLOSES_WINDOW = os.getenv("HUSH_CLOSES_WINDOW", "1") != "0"
HUSH_SILENT_ACK    = os.getenv("HUSH_SILENT_ACK", "0") == "1"
# Orchestrator no longer owns wake gating; leave empty to avoid gating even if mic uses Porcupine
PORCUPINE_ACCESS_KEY  = os.getenv("PORCUPINE_ACCESS_KEY", "")
PORCUPINE_KEYWORD_PATH = os.getenv("PORCUPINE_KEYWORD_PATH", "")

LONG_FORM_MODE = _env_flag("LONG_FORM", False)
LONGFORM_SYSTEM_APPEND_ACTIVE = bool(LONG_FORM_MODE and LONGFORM_SYSTEM_APPEND)
_STOP_TOKENS_RAW = os.getenv("STOP_TOKENS")
STOP_TOKENS = _parse_stop_tokens(_STOP_TOKENS_RAW)
NUM_PREDICT = _env_int("NUM_PREDICT", 512 if LONG_FORM_MODE else 160)
if LONG_FORM_MODE:
    NUM_PREDICT = max(NUM_PREDICT, 512)
    LONGFORM_AUTOCONFIRM_NUM_PREDICT = max(LONGFORM_AUTOCONFIRM_NUM_PREDICT, 512)
SHORT_FORM_TEMPERATURE = 0.3

_REFUSAL_GUARD_PATTERNS = [
    re.compile(r"^sorry\b", re.IGNORECASE),
    re.compile(
        r"^i (?:don'?t feel like|don'?t want|can'?t|won'?t|cannot|will not)\b",
        re.IGNORECASE,
    ),
]

_warmup_done = False
_warmup_meta: dict = {"fired": False, "ms": 0, "error": ""}
_first_call = True
_last_http_status: Optional[int] = None

# --- Python deps you need in your venv ---
# pip install nvidia-riva-client==2.19.* requests simpleaudio

import requests
from requests.exceptions import (
    ReadTimeout,
    ConnectionError as RequestsConnectionError,
    HTTPError,
)
import simpleaudio as sa

# Riva TTS client (2.19.x)
try:
    from riva.client import Auth, SpeechSynthesisService, AudioEncoding
except Exception as e:
    raise RuntimeError(
        "Riva Python client not found. Install with: pip install nvidia-riva-client==2.19.*"
    ) from e


# ---- 1R-hotfix-5: tiny helper to retry during Ollama cold start ----
def _retry_on_coldstart(fn, *, max_retries: int = 3, base_sleep_s: float = 2.0):
    """
    Retries the given callable when Ollama is still spinning up.
    Retries on ReadTimeout / ConnectionError and 499/503 HTTP errors.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except (ReadTimeout, RequestsConnectionError):
            if attempt >= max_retries:
                raise
        except HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in (499, 503) or attempt >= max_retries:
                raise
        sleep_s = base_sleep_s * (2 ** attempt)
        log.warning("Ollama cold-start retry %d; sleeping %.1fs", attempt + 1, sleep_s)
        time.sleep(sleep_s)
        attempt += 1
# ----------------------------------------------------------


# ---------------- Utilities ----------------

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def local_datetime_line() -> str:
    dt = datetime.now()
    # Windows-safe formatting; remove leading zeros for a natural read.
    return dt.strftime("It's %A, %B %d, %Y at %I:%M %p").replace(" 0", " ")

def clip_one_sentence(text: str, max_chars: int = 240) -> str:
    # Allow up to two sentences within max_chars; fallback to hard cap.
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return ""
    parts = re.split(r"(?<=[\.!?])\s+", clean)
    acc, total = [], 0
    for s in parts:
        if not s:
            continue
        candidate_len = total + (1 if acc else 0) + len(s)
        if len(acc) < 2 and candidate_len <= max_chars:
            acc.append(s); total = candidate_len
        else:
            break
    out = " ".join(acc) if acc else parts[0]
    return out[:max_chars].rstrip()

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', s.casefold())

# Keep process alive until audio completes
def wait_play(play_obj, poll_ms: int = 25):
    try:
        while getattr(play_obj, "is_playing", lambda: False)():
            time.sleep(poll_ms / 1000.0)
    except Exception:
        pass


# ---------------- Router (instant responses) ----------------


# ---------------- Riva TTS ----------------

class RivaTTS:
    def __init__(self, uri: str, voice_name: str):
        self.voice = voice_name
        self.auth = Auth(uri=uri, use_ssl=False)
        self.tts = SpeechSynthesisService(self.auth)

    def synth(self, text: str, sample_rate: int = 44100) -> bytes:
        resp = self.tts.synthesize(
            text=text,
            voice_name=self.voice,
            language_code="en-US",
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hz=sample_rate,
        )
        return resp if isinstance(resp, (bytes, bytearray)) else getattr(resp, "audio", b"")


# ---------------- Audio Playback ----------------

class AudioPlayer:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def _maybe_write_wav(self, pcm: bytes):
        if not EDDIE_DEBUG_WAV:
            return
        path = EDDIE_DEBUG_WAV
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sample_rate)
                wf.writeframes(pcm)
            if OPEN_WAV_APP:
                try: os.startfile(path)
                except Exception: pass
        except Exception as e:
            print(f"[Audio] WAV write failed: {e}", file=sys.stderr)

    def play_pcm16le(self, audio_bytes: bytes):
        self._maybe_write_wav(audio_bytes)
        try:
            return sa.play_buffer(audio_bytes, 1, 2, self.sample_rate)
        except Exception as e:
            print(f"[Audio] simpleaudio failed: {e}", file=sys.stderr)
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
            if hasattr(play_obj, "stop"): play_obj.stop()
        except Exception:
            pass




class _WakeHTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/signal/wake":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            try:
                self.rfile.read(length)
            except Exception:
                pass
        try:
            self.server.orchestrator._open_wake_window()
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.send_error(500)

    def log_message(self, format, *args):
        return


class _WakeHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, orchestrator):
        super().__init__(server_address, _WakeHTTPHandler)
        self.orchestrator = orchestrator

# ---------------- Ollama LLM ----------------

# Set from XML at startup
SYSTEM_PROMPT_V2 = ""

def _compose_system_prompt() -> str:
    base = SYSTEM_PROMPT_V2
    if LONGFORM_SYSTEM_APPEND_ACTIVE:
        if base.strip():
            return f"{base.rstrip()}\n\n{LONGFORM_SYSTEM_APPEND}"
        return LONGFORM_SYSTEM_APPEND
    return base

def _build_ollama_payload(user_text: str) -> tuple[dict, dict]:
    system_prompt = _compose_system_prompt()
    temperature = LONGFORM_TEMPERATURE if LONG_FORM_MODE else SHORT_FORM_TEMPERATURE
    stop_tokens = list(STOP_TOKENS) if STOP_TOKENS else []
    sentinel_used = ""
    if LONG_FORM_MODE and not stop_tokens and LONGFORM_STOP_SENTINEL:
        stop_tokens = [LONGFORM_STOP_SENTINEL]
        sentinel_used = LONGFORM_STOP_SENTINEL
    options = {
        "temperature": temperature,
        "num_predict": NUM_PREDICT,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
    }
    if stop_tokens:
        options["stop"] = stop_tokens
    payload = {
        "prompt": user_text,
        "system": system_prompt,
        "stream": False,
        "options": options,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    meta = {
        "temperature": temperature,
        "stop_tokens": stop_tokens,
        "longform_stop_sentinel_used": sentinel_used,
        "longform_system_append": LONGFORM_SYSTEM_APPEND_ACTIVE,
        "longform_retry_used": False,
        "longform_suggest_continue": False,
        "refusal_glitch": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "autoconfirm_attempted": False,
        "autoconfirm_used": False,
        "autoconfirm_error": "",
        "autoconfirm_llm_ms": 0,
        "http_status": None,
    }
    return payload, meta

def call_ollama_generate(
    ollama_url: str,
    model: str,
    user_text: str,
    timeout: Optional[float] = None,
    num_predict: Optional[int] = None,
    mark_call: bool = True,
) -> Tuple[str, dict]:
    global _first_call, _last_http_status
    payload, meta = _build_ollama_payload(user_text)
    payload["model"] = model
    if num_predict is not None:
        try:
            num_predict_val = int(num_predict)
        except (TypeError, ValueError):
            num_predict_val = NUM_PREDICT
        if num_predict_val <= 0:
            num_predict_val = NUM_PREDICT
        payload["options"]["num_predict"] = num_predict_val
        meta["num_predict"] = num_predict_val
    else:
        meta["num_predict"] = payload["options"]["num_predict"]
    if timeout is None:
        timeout_sec = OLLAMA_TIMEOUT_S
    else:
        try:
            timeout_sec = float(timeout)
        except (TypeError, ValueError):
            timeout_sec = OLLAMA_TIMEOUT_S
    timeout_sec = max(0.1, timeout_sec)
    if mark_call and _first_call:
        timeout_sec = max(timeout_sec, 8.0)
        _first_call = False
    status_code: Optional[int] = None
    def _do_call():
        nonlocal status_code
        global _last_http_status
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout_sec)
        status_code = resp.status_code
        _last_http_status = status_code
        meta["http_status"] = status_code
        resp.raise_for_status()
        return resp
    try:
        r = _retry_on_coldstart(_do_call)
    except requests.exceptions.RequestException as exc:
        if status_code is None and getattr(exc, "response", None) is not None:
            status_code = exc.response.status_code
            _last_http_status = status_code
            meta["http_status"] = status_code
        raise
    data = r.json()
    text = data.get("response", "")
    return text, meta


# ---------------- Orchestrator ----------------

class EddieOrchestrator:
    def __init__(self):
        self.tts = RivaTTS(RIVA_SPEECH_API_URL, VOICE_NAME)
        self.player = AudioPlayer(sample_rate=RIVA_TTS_SAMPLE_RATE)
        self._playback_lock = threading.Lock()
        self._current_play_obj: Optional[object] = None
        # Wake gating handled by mic; keep disabled in orchestrator
        self._wake_required = False
        self._wake_active = False
        self._wake_window_ms = max(0, WAKE_WINDOW_MS)
        self._wake_timer: Optional[threading.Timer] = None
        self._wake_lock = threading.Lock()
        self._wake_server = None
        self._wake_server_thread = None
        self._wake_active = True  # always allow processing; mic gates upstream
        self._start_wake_server()  # will no-op when WAKE_PORT == 0
        # warm up TTS so first real turn isn't cold
        try:
            _ = self.tts.synth("ok", sample_rate=RIVA_TTS_SAMPLE_RATE)
        except Exception:
            pass
        # Load persona (system prompt, fillers, ACK, hiccup/fallback, router)
        self._router_rules: list[tuple[list[str], list[str], Optional[list[float]], Optional[str]]] = []
        self._fillers: list[tuple[str, float]] = []
        self._hiccup_line: str = ""
        self._fallback_line: str = ""
        self._exit_lines = {"clingy": "", "signoff": ""}; self._ep_clinginess = 0.2
        self._ep_insistence_words = ["really", "now", "urgent"]; self._exit_state = {"stage": 0, "used": False}
        self._twf_max = 1
        self._twf_tier_ms = ACK_THRESHOLD_MS
        self._load_persona_from_xml()

        # Fillers (non-lexical acknowledgments), weighted
        total_w = sum(w for _, w in self._fillers) or 1.0
        self._filler_texts = [t for t, _ in self._fillers]
        self._filler_weights = [w/total_w for _, w in self._fillers]
        self._ack_pcm = {}
        for _ack in self._filler_texts:
            try:
                self._ack_pcm[_ack] = self.tts.synth(_ack, sample_rate=RIVA_TTS_SAMPLE_RATE)
            except Exception:
                self._ack_pcm[_ack] = None
        if (LLM_WARMUP or OLLAMA_WARMUP_ON_START) and not _warmup_done:
            threading.Thread(
                target=self._warmup_runner,
                name="OllamaWarmup",
                daemon=True,
            ).start()

    def _warmup_runner(self):
        global _warmup_done, _warmup_meta
        if _warmup_done:
            return
        start = time.perf_counter()
        meta = {"fired": True, "ms": 0, "error": ""}
        try:
            call_ollama_generate(
                OLLAMA_URL,
                OLLAMA_MODEL,
                "ping",
                timeout=LLM_WARMUP_TIMEOUT_MS / 1000.0,
                num_predict=1,
                mark_call=False,
            )
            meta["ms"] = int((time.perf_counter() - start) * 1000)
        except Exception as exc:
            meta["ms"] = int((time.perf_counter() - start) * 1000)
            meta["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _warmup_meta = meta
            _warmup_done = True

    # ---- TTS chunking helpers ----
    @staticmethod
    def _split_for_tts(text: str, limit: int) -> list[str]:
        # Normalize whitespace but preserve paragraph intent.
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return [""]
        if len(text) <= limit:
            return [text]
        # First split by paragraphs, then by sentences, then hard-wrap.
        para_parts = re.split(r"\n{2,}", text)
        sent_re = re.compile(r'(?<=[\.\!\?])\s+')
        chunks: list[str] = []
        for p in para_parts:
            p = p.strip()
            if not p:
                continue
            if len(p) <= limit:
                chunks.append(p)
                continue
            sents = sent_re.split(p)
            buf = ""
            for s in sents:
                if not s:
                    continue
                if len(buf) + len(s) + (1 if buf else 0) <= limit:
                    buf = (buf + " " + s) if buf else s
                else:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    if len(s) <= limit:
                        chunks.append(s)
                    else:
                        # hard wrap long sentence
                        for i in range(0, len(s), limit):
                            chunks.append(s[i:i+limit])
            if buf:
                chunks.append(buf)
        return chunks or [text[:limit]]

    def _computed_chunk_limit(self) -> int:
        if TTS_CHUNK_CHAR_LIMIT > 0:
            return TTS_CHUNK_CHAR_LIMIT
        bytes_per_sec = max(1, RIVA_TTS_SAMPLE_RATE * 2)  # mono, 16-bit
        approx_bytes_per_char = bytes_per_sec / max(1.0, TTS_CHARS_PER_SEC)
        cap = int((TTS_MAX_MESSAGE_BYTES * 0.85) / approx_bytes_per_char)
        computed = max(300, min(1500, cap))
        if TTS_MAX_CHARS > 0:
            return max(1, min(computed, TTS_MAX_CHARS))
        return computed

    def _speak(self, text: str) -> Tuple[int, object]:
        """Synthesize (chunked) and start playback. Returns (tts_ms, play_obj)."""
        t0 = time.perf_counter()
        sr = RIVA_TTS_SAMPLE_RATE
        limit = self._computed_chunk_limit()  # respects env overrides like EDDIE_TTS_MAX_CHARS
        try:
            parts = self._split_for_tts(text, limit)
            pcm_parts: list[bytes] = []
            for part in parts:
                pcm_parts.append(self.tts.synth(part, sample_rate=sr))
            audio = b"".join(pcm_parts)
            t1 = time.perf_counter()
            play_obj = self.player.play_pcm16le(audio)
            play_obj = self._set_current_play_obj(play_obj)
            l_tts_ms = int((t1 - t0) * 1000)
            return l_tts_ms, play_obj
        except Exception as e:
            # Retry once at a lower sample rate to further shrink payloads.
            try:
                if sr > 16000:
                    sr2 = 16000
                    parts = self._split_for_tts(text, limit)
                    audio = b"".join(self.tts.synth(p, sample_rate=sr2) for p in parts)
                    t1 = time.perf_counter()
                    play_obj = self.player.play_pcm16le(audio)
                    play_obj = self._set_current_play_obj(play_obj)
                    l_tts_ms = int((t1 - t0) * 1000)
                    return l_tts_ms, play_obj
            except Exception:
                pass
            # Last resort: swallow the error so we still log and progress.
            print(f"[TTS] synth failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 0, self._set_current_play_obj(None)

    def _play_ack(self, text: str):
        pcm = self._ack_pcm.get(text)
        try:
            if pcm:
                return self._set_current_play_obj(self.player.play_pcm16le(pcm))
        except Exception:
            pass
        _, play_obj = self._speak(text)
        return play_obj

    # --- Long-form sentence chunker (speaks within a time cap) ---
    def _speak_longform_sentences(self, text: str) -> tuple[int, object]:
        """
        Split long-form text into ~sentence chunks (≤ LONGFORM_TTS_CHARS),
        synth+play sequentially, and stop when total playback budget exceeds
        LONGFORM_TTS_MAX_S. Returns (tts_ms_total, last_play_obj).
        """
        tts_ms_total = 0
        sr = RIVA_TTS_SAMPLE_RATE
        # sentence-ish split
        s_re = re.compile(r'(?<=[\.\!\?])\s+')
        parts_raw = [p.strip() for p in s_re.split(re.sub(r"\s+", " ", text or "").strip()) if p.strip()]
        # pack into near-280-char chunks
        chunks: list[str] = []
        buf = ""
        for s in parts_raw:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= LONGFORM_TTS_CHARS:
                buf = f"{buf} {s}"
            else:
                chunks.append(buf); buf = s
        if buf:
            chunks.append(buf)
        if not chunks:
            chunks = [(text or "")[:LONGFORM_TTS_CHARS]]
        # speak sequentially with a hard time budget
        bytes_per_sec = max(1, sr * 2)  # mono 16-bit
        spoken_s = 0.0
        last_obj = None
        for c in chunks:
            # stop if we've exhausted the speech budget
            if spoken_s >= float(LONGFORM_TTS_MAX_S):
                break
            t0 = time.perf_counter()
            pcm = self.tts.synth(c, sample_rate=sr)
            t1 = time.perf_counter()
            tts_ms_total += int((t1 - t0) * 1000)
            # estimate duration from bytes; keep total under cap
            est_s = len(pcm) / float(bytes_per_sec)
            if spoken_s + est_s > float(LONGFORM_TTS_MAX_S):
                # try truncating late to fit remaining budget
                remain_s = max(0.0, float(LONGFORM_TTS_MAX_S) - spoken_s)
                if remain_s <= 0.0:
                    break
                # crude trim: scale bytes to remaining seconds
                keep_bytes = int(remain_s * bytes_per_sec)
                if keep_bytes > 0:
                    pcm = pcm[:keep_bytes]
                    est_s = len(pcm) / float(bytes_per_sec)
                else:
                    break
            play_obj = self.player.play_pcm16le(pcm)
            last_obj = self._set_current_play_obj(play_obj)
            wait_play(play_obj)
            self._maybe_clear_play_obj(play_obj)
            spoken_s += est_s
        return tts_ms_total, (last_obj or self._set_current_play_obj(None))

    def _set_speaking(self, on: bool):
        try:
            if on:
                with open(SPEAKING_FLAG_PATH, "w", encoding="utf-8") as f:
                    f.write(now_iso())
            else:
                if os.path.exists(SPEAKING_FLAG_PATH): os.remove(SPEAKING_FLAG_PATH)
        except Exception:
            pass

    def _load_persona_from_xml(self):
        cfg_path = _CONFIG_DIR / "personality.xml"
        try:
            root = ET.parse(str(cfg_path)).getroot()

            # System prompt
            sp = root.find('.//Personality/SystemPrompt')
            if sp is not None:
                base_sp = (sp.text or '').strip()
                if base_sp:
                    globals()['SYSTEM_PROMPT_V2'] = base_sp
                # Mood add-on via MOOD_LOCK or daily schedule
                mood_id = (MOOD_LOCK or '').strip()
                if not mood_id:
                    now = datetime.now(); mins = now.hour*60 + now.minute
                    if mins >= 21*60 or mins < 7*60: mood_id = 'TIRED_NIGHT'
                    elif mins < 11*60: mood_id = 'ENERGIZED_AM'
                    elif mins < 12*60: mood_id = 'MID_MORNING_DIP'
                    elif mins < 14*60: mood_id = 'POST_LUNCH'
                    elif mins < 16*60+30: mood_id = 'IRRITABLE_SIESTA'
                    elif mins < 19*60: mood_id = 'EVENING_NORMAL'
                    else: mood_id = 'CONTENT_EVE'
                if mood_id:
                    moods = root.find('.//Personality/Moods')
                    if moods is not None:
                        for m in moods.findall('Mood'):
                            if (m.get('id') or '').strip() == mood_id:
                                add = (m.text or '').strip()
                                if add:
                                    globals()['SYSTEM_PROMPT_V2'] = (base_sp + ' ' + add).strip()
                                    try: self._mood_id = mood_id
                                    except Exception: pass
                                break

            # Fillers and timing (env wins if set)
            fill = root.find('.//Personality/Fillers')
            if fill is not None:
                try:
                    if 'FILLER_FIRST_MS' not in os.environ and fill.get('firstMs'):
                        globals()['FILLER_FIRST_MS'] = int(fill.get('firstMs'))
                    if 'FILLER_POST_PAUSE_MS' not in os.environ and fill.get('postPauseMs'):
                        globals()['FILLER_POST_PAUSE_MS'] = int(fill.get('postPauseMs'))
                    if 'FILLER_PROB' not in os.environ and fill.get('prob'):
                        globals()['FILLER_PROB'] = float(fill.get('prob'))
                    max_count = fill.get('maxCount')
                    if max_count:
                        self._twf_max = max(1, int(max_count))
                    tier_ms_attr = fill.get('tierMs')
                    if tier_ms_attr:
                        self._twf_tier_ms = max(0, int(tier_ms_attr))
                except Exception:
                    pass
                toks = []
                for u in fill.findall('U'):
                    t = (u.text or '').strip()
                    if not t:
                        continue
                    try:
                        w = float(u.get('weight') or '1.0')
                    except Exception:
                        w = 1.0
                    toks.append((t, w))
                if toks:
                    self._fillers = toks

            # ACK replace window
            ack = root.find('.//Personality/ACK')
            if ack is not None and 'REPLACE_WINDOW_MS' not in os.environ:
                try:
                    rw = ack.get('replaceWindowMs')
                    if rw:
                        globals()['REPLACE_WINDOW_MS'] = int(rw)
                except Exception:
                    pass

            # Hiccup/Fallback lines
            h = root.find('.//Personality/Hiccup')
            if h is not None:
                self._hiccup_line = (h.get('line') or '').strip()
            f = root.find('.//Personality/Fallback')
            if f is not None:
                self._fallback_line = (f.get('line') or '').strip()
            ep = root.find('.//Personality/ExitPolicy')
            if ep is not None:
                val = ep.get('clinginess')
                if val:
                    try:
                        self._ep_clinginess = max(0.0, min(1.0, float(val)))
                    except Exception:
                        pass
                words = (ep.get('insistenceWords') or '').strip()
                if words:
                    self._ep_insistence_words = [w.strip() for w in words.split('|') if w.strip()]
            lines = root.find('.//Personality/ExitLines')
            if lines is not None:
                for key in ("clingy", "signoff"):
                    val = (lines.get(key) or '').strip()
                    if val:
                        self._exit_lines[key] = val

            # Router rules
            rules = []
            for rule in root.findall('.//Personality/Router/Rule'):
                keys = (rule.get('keys') or '').strip()
                reply_attr = (rule.get('reply') or '').strip()
                tool = rule.get('tool')
                weights_attr = rule.get('replyWeights')
                if keys and reply_attr:
                    key_list = [k.strip().casefold() for k in keys.split('|') if k.strip()]
                    replies = [r.strip() for r in reply_attr.split('|') if r.strip()]
                    if not replies:
                        continue
                    weights = None
                    if weights_attr:
                        parts = [w.strip() for w in weights_attr.split('|')]
                        if len(parts) == len(replies):
                            try:
                                weights = [float(w) for w in parts]
                            except ValueError:
                                weights = None
                    rules.append((key_list, replies, weights, tool))
            self._router_rules = rules
        except Exception as e:
            print(f"[PersonaXML] load failed: {e}", file=sys.stderr)
            self._router_rules = []

    def _router_response(self, text: str) -> Optional[tuple[list[str], Optional[list[float]], Optional[str]]]:
        t = norm(text)
        for keys, replies, weights, tool in self._router_rules:
            if any(k in t for k in keys):
                return replies, weights, tool
        return None

    def _exit_intent_urgent(self, text: str) -> bool:
        return any(w in norm(text) for w in self._ep_insistence_words)

    def _maybe_exit_override(self, text: str) -> tuple[bool, str, bool, str]:
        state = self._exit_state
        urgent = self._exit_intent_urgent(text)
        if state["stage"] == 0 and not urgent and not state["used"] and random.random() < self._ep_clinginess:
            state["stage"] = 1; state["used"] = True
            return True, self._exit_lines.get("clingy") or self._fallback_line, False, "clingy"
        if state["stage"] == 1:
            state["stage"] = 2
            if urgent:
                return False, "", True, "insist"
            return True, self._exit_lines.get("signoff") or self._fallback_line, True, "canned_signoff"
        if state["stage"] < 2:
            state["stage"] = 2
        return False, "", True, "normal"

    def _fire_tool_async(self, tool: str, text: str):
        if not TOOLS_URL:
            return
        def _w():
            try:
                requests.post(TOOLS_URL, json={"tool": tool, "args": {"text": text}}, timeout=1.5)
            except Exception:
                pass
        threading.Thread(target=_w, daemon=True).start()

    def handle_final_transcript(self, final_text: str, asr_latency_ms: int = 0) -> dict:
        self._stop_playback()
        turn_start = time.perf_counter()
        ts_iso = now_iso()
        final_text = (final_text or "").strip()

        # Wake gating removed: orchestrator always processes when invoked; mic controls wake

        # 1) Router - instant
        rr = self._router_response(final_text)
        if rr:
            replies, weights, tool = rr
            selected = ""
            if len(replies) == 1:
                selected = replies[0]
            elif len(replies) > 1:
                selected = random.choices(replies, weights=weights, k=1)[0] if weights else random.choice(replies)
            reply = clip_one_sentence(selected, MAX_REPLY_CHARS)
            exit_request = False
            exit_reason = ""
            exit_stage = 0
            if tool == "self.exit":
                handled, override, exit_request, exit_reason = self._maybe_exit_override(final_text)
                if handled and override:
                    reply = clip_one_sentence(override, MAX_REPLY_CHARS)
                exit_stage = self._exit_state["stage"]
            if tool == "clock.now":
                tt = norm(final_text)
                dt = datetime.now()
                if "time" in tt:
                    reply = dt.strftime("It's %I:%M %p").replace(" 0", " ").replace("It's 0", "It's ")
                elif "day" in tt:
                    reply = dt.strftime("It's %A")
                elif "date" in tt or "today" in tt:
                    reply = dt.strftime("It's %B %d, %Y").replace(" 0", " ")
                else:
                    reply = local_datetime_line()
            if tool == "self.hush":
                return self._perform_hush(reply, final_text, ts_iso, asr_latency_ms, turn_start)
            _, router_meta = _build_ollama_payload("")
            self._set_speaking(True)
            l_tts_ms, play_obj = self._speak(reply)
            if tool and tool not in ("self.exit", "clock.now", "self.hush"):
                self._fire_tool_async(tool, final_text)
            first_audio_ms = int((time.perf_counter() - turn_start) * 1000)
            log = {
                "ts": ts_iso,
                "text": final_text,
                "reply": reply,
                "l_asr_ms": asr_latency_ms,
                "l_llm_ms": 0,
                "l_tts_ms": l_tts_ms,
                "l_total_ms": first_audio_ms,
                "router": True,
                "ack": False,
                "ack_replaced": False,
                "ack_type": "none",
                "route": "router",
                "voice": VOICE_NAME,
            "long_form": LONG_FORM_MODE,
            "num_predict": NUM_PREDICT,
                "stop_tokens_used": list(router_meta.get("stop_tokens", [])),
                "longform_system_append": router_meta.get("longform_system_append", False),
                "longform_stop_sentinel_used": router_meta.get("longform_stop_sentinel_used", ""),
                "longform_retry_used": False,
                "longform_suggest_continue": False,
                "refusal_glitch": False,
                "autoconfirm_used": False,
                "autoconfirm_llm_ms": 0,
                "temperature": router_meta.get(
                    "temperature",
                    LONGFORM_TEMPERATURE if LONG_FORM_MODE else SHORT_FORM_TEMPERATURE,
                ),
            }
            try:
                log["mood"] = getattr(self, "_mood_id", "")
            except Exception:
                pass
            log["router_variant"] = selected
            if tool == "self.exit":
                log.update({"request_exit": exit_request, "exit_stage": exit_stage, "exit_reason": exit_reason})
            self._append_log(log)
            wait_play(play_obj)
            self._maybe_clear_play_obj(play_obj)
            self._set_speaking(False)
            if self._wake_required and self._wake_active:
                self._reset_wake_timer()
            return log

        # 2) Else LLM with optional, randomized filler
        ack_sent = False
        ack_plays: list[object] = []
        ack_type = "none"
        l_llm_ms = 0
        twf_engaged = False
        twf_tiers_used = 0
        twf_total_extra_ms = 0

        llm_out_q: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        llm_timing: dict = {}
        base_timeout_sec = LLM_TIMEOUT_MS / 1000.0
        llm_timeout_sec = base_timeout_sec
        if LONG_FORM_MODE:
            llm_timeout_sec = max(base_timeout_sec, OLLAMA_TIMEOUT_S)

        def llm_worker():
            base_hiccup = self._hiccup_line or self._fallback_line or ""
            t0 = time.perf_counter()
            try:
                resp_text, meta = call_ollama_generate(
                    OLLAMA_URL,
                    OLLAMA_MODEL,
                    final_text,
                    timeout=llm_timeout_sec,
                )
            except Exception:
                _, meta = _build_ollama_payload(final_text)
                resp_text = base_hiccup
            else:
                pass
            trimmed = resp_text.strip()
            if LONG_FORM_MODE:
                needs_retry = (
                    len(trimmed) < LONGFORM_MIN_CHARS or "\n\n" not in trimmed
                )
                if needs_retry and LONGFORM_RETRY:
                    retry_prompt = (
                        f"{final_text}\n\n"
                        "Expand the story into several paragraphs with vivid detail "
                        "and clear blank lines between them. <END>"
                    )
                    try:
                        retry_resp, retry_meta = call_ollama_generate(
                            OLLAMA_URL,
                            OLLAMA_MODEL,
                            retry_prompt,
                            timeout=llm_timeout_sec,
                        )
                    except Exception:
                        meta["longform_retry_used"] = True
                    else:
                        resp_text = retry_resp
                        meta.update(retry_meta)
                        meta["longform_retry_used"] = True
                        trimmed = resp_text.strip()
                elif needs_retry:
                    meta["longform_retry_used"] = False
                    meta["longform_suggest_continue"] = bool(
                        LONGFORM_SUGGEST_CONTINUE and LONGFORM_CONTINUE_TTS
                    )
            t1 = time.perf_counter()
            llm_timing["ms"] = int((t1 - t0) * 1000)
            try:
                llm_out_q.put_nowait({"text": resp_text, "meta": meta})
            except queue.Full:
                pass

        thread = threading.Thread(target=llm_worker, daemon=True)
        thread.start()

        try:
            resp = llm_out_q.get(timeout=FILLER_FIRST_MS / 1000.0)
            l_llm_ms = llm_timing.get("ms", FILLER_FIRST_MS)
        except queue.Empty:
            if TWF_DISABLE:
                resp = llm_out_q.get()
                l_llm_ms = llm_timing.get("ms", 0)
            else:
                twf_engaged = True
                resp = None
                for tier_index in range(max(1, self._twf_max)):
                    if self._filler_texts and (
                        random.random() < max(0.0, min(1.0, FILLER_PROB))
                    ):
                        filler = random.choices(
                            self._filler_texts, weights=self._filler_weights, k=1
                        )[0]
                        ack_type = filler.lower()
                        ack_sent = True
                        self._set_speaking(True)
                        play_obj = self._play_ack(filler)
                        if play_obj:
                            ack_plays.append(play_obj)
                    wait_start = time.perf_counter()
                    try:
                        resp = llm_out_q.get(timeout=max(0, self._twf_tier_ms) / 1000.0)
                        twf_tiers_used = tier_index + 1
                        twf_total_extra_ms += int((time.perf_counter() - wait_start) * 1000)
                        break
                    except queue.Empty:
                        twf_tiers_used = tier_index + 1
                        twf_total_extra_ms += int((time.perf_counter() - wait_start) * 1000)
                if resp is None:
                    resp = llm_out_q.get()
                l_llm_ms = llm_timing.get("ms", 0)

        if isinstance(resp, dict):
            resp_text = resp.get("text", "")
            llm_meta = resp.get("meta", {}) or {}
        elif resp is None:
            resp_text = ""
            _, llm_meta = _build_ollama_payload(final_text)
        else:
            resp_text = str(resp)
            _, llm_meta = _build_ollama_payload(final_text)
        llm_meta = llm_meta or {}
        llm_meta.setdefault("keep_alive", OLLAMA_KEEP_ALIVE)
        llm_meta.setdefault("http_status", _last_http_status)
        llm_meta.setdefault("autoconfirm_attempted", False)
        llm_meta.setdefault("autoconfirm_used", False)
        llm_meta.setdefault("autoconfirm_error", "")
        llm_meta.setdefault("autoconfirm_llm_ms", 0)
        if "refusal_glitch" not in llm_meta:
            llm_meta["refusal_glitch"] = False
        if (
            LONG_FORM_MODE
            and not LONGFORM_RETRY
            and LONGFORM_REFUSAL_GUARD
        ):
            trimmed_resp = (resp_text or "").strip()
            refusal_seed = LONGFORM_REFUSAL_SEED or _DEFAULT_REFUSAL_SEED
            if trimmed_resp:
                for pattern in _REFUSAL_GUARD_PATTERNS:
                    if pattern.match(trimmed_resp):
                        resp_text = refusal_seed
                        llm_meta["refusal_glitch"] = True
                        llm_meta["longform_suggest_continue"] = bool(
                            LONGFORM_SUGGEST_CONTINUE and LONGFORM_CONTINUE_TTS
                        )
                        break
        if LONG_FORM_MODE:
            final_reply = resp_text.strip()
            sentinel_token = llm_meta.get("longform_stop_sentinel_used", "")
            if sentinel_token and final_reply.endswith(sentinel_token):
                final_reply = final_reply[: -len(sentinel_token)].rstrip()
        else:
            final_reply = clip_one_sentence(resp_text, MAX_REPLY_CHARS)
        if "longform_system_append" not in llm_meta:
            llm_meta["longform_system_append"] = LONGFORM_SYSTEM_APPEND_ACTIVE
        if "longform_stop_sentinel_used" not in llm_meta:
            sentinel_used = ""
            if LONG_FORM_MODE and not STOP_TOKENS and LONGFORM_STOP_SENTINEL:
                sentinel_used = LONGFORM_STOP_SENTINEL
            llm_meta["longform_stop_sentinel_used"] = sentinel_used
        if "stop_tokens" not in llm_meta:
            llm_meta["stop_tokens"] = list(STOP_TOKENS)
        if "temperature" not in llm_meta:
            llm_meta["temperature"] = (
                LONGFORM_TEMPERATURE if LONG_FORM_MODE else SHORT_FORM_TEMPERATURE
            )
        if "longform_retry_used" not in llm_meta:
            llm_meta["longform_retry_used"] = False
        if "longform_suggest_continue" not in llm_meta:
            llm_meta["longform_suggest_continue"] = False

        if ack_sent and ack_plays:
            for play_obj in ack_plays:
                wait_play(play_obj)
                self._maybe_clear_play_obj(play_obj)
            if not TWF_DISABLE:
                time.sleep(max(0, FILLER_POST_PAUSE_MS) / 1000.0)
            self._set_speaking(False)

        self._set_speaking(True)
        if LONG_FORM_MODE and LONGFORM_TTS_ENABLE:
            if LONGFORM_READY_CHECK:
                # reserved for mic sessions (2.2.4); intentionally no-op in smoketest
                pass
            if LONGFORM_PRINT:
                try:
                    print("\n[longform]\n" + (final_reply or "") + "\n", flush=True)
                except Exception:
                    pass
            l_tts_ms, play_obj_final = self._speak_longform_sentences(final_reply)
        else:
            l_tts_ms, play_obj_final = self._speak(final_reply)
        first_audio_ms = int((time.perf_counter() - turn_start) * 1000)

        combined_reply = final_reply
        autoconfirm_used = bool(llm_meta.get("autoconfirm_used", False))
        autoconfirm_attempted = bool(llm_meta.get("autoconfirm_attempted", False))
        autoconfirm_llm_ms = int(llm_meta.get("autoconfirm_llm_ms", 0) or 0)
        autoconfirm_error = llm_meta.get("autoconfirm_error", "")

        log = {
            "ts": ts_iso,
            "text": final_text,
            "reply": combined_reply,
            "l_asr_ms": asr_latency_ms,
            "l_llm_ms": l_llm_ms,
            "l_tts_ms": l_tts_ms,
            "l_total_ms": first_audio_ms,
            "router": False,
            "ack": ack_sent,
            "ack_replaced": False,
            "ack_type": ack_type,
            "route": "llm",
            "voice": VOICE_NAME,
            "long_form": LONG_FORM_MODE,
            "num_predict": NUM_PREDICT,
            "stop_tokens_used": list(llm_meta.get("stop_tokens", [])),
            "longform_system_append": llm_meta.get("longform_system_append", False),
            "longform_stop_sentinel_used": llm_meta.get("longform_stop_sentinel_used", ""),
            "longform_retry_used": llm_meta.get("longform_retry_used", False),
            "longform_suggest_continue": llm_meta.get("longform_suggest_continue", False),
            "refusal_glitch": llm_meta.get("refusal_glitch", False),
            "autoconfirm_attempted": autoconfirm_attempted,
            "autoconfirm_used": autoconfirm_used,
            "autoconfirm_llm_ms": autoconfirm_llm_ms,
            "autoconfirm_error": autoconfirm_error,
            "temperature": llm_meta.get(
                "temperature",
                LONGFORM_TEMPERATURE if LONG_FORM_MODE else SHORT_FORM_TEMPERATURE,
            ),
        }
        try:
            log["mood"] = getattr(self, "_mood_id", "")
        except Exception:
            pass
        if twf_engaged:
            log["twf_tiers_used"] = twf_tiers_used
            log["twf_total_extra_ms"] = twf_total_extra_ms
        wait_play(play_obj_final)
        self._maybe_clear_play_obj(play_obj_final)

        if (
            LONG_FORM_MODE
            and llm_meta.get("longform_suggest_continue", False)
            and LONGFORM_SUGGEST_CONTINUE
            and LONGFORM_CONTINUE_TTS
        ):
            self._set_speaking(True)
            _, continue_play_obj = self._speak(LONGFORM_CONTINUE_TTS)
            wait_play(continue_play_obj)
            self._maybe_clear_play_obj(continue_play_obj)
            self._set_speaking(False)

        if (
            LONG_FORM_MODE
            and not LONGFORM_RETRY
            and LONGFORM_SUGGEST_CONTINUE
            and LONGFORM_AUTOCONFIRM
            and llm_meta.get("longform_suggest_continue", False)
        ):
            autoconfirm_attempted = True
            llm_meta["autoconfirm_attempted"] = True
            cont_parts = [final_text]
            if combined_reply:
                cont_parts.append(combined_reply)
            if LONGFORM_AUTOCONFIRM_SUFFIX:
                cont_parts.append(LONGFORM_AUTOCONFIRM_SUFFIX)
            continuation_prompt = "\n\n".join(part for part in cont_parts if part)
            cont_num_predict = max(1, LONGFORM_AUTOCONFIRM_NUM_PREDICT)
            cont_timeout_sec = max(
                LONGFORM_AUTOCONFIRM_TIMEOUT_MS / 1000.0,
                OLLAMA_TIMEOUT_S,
            )
            t_ac_start = time.perf_counter()
            try:
                cont_text, cont_meta = call_ollama_generate(
                    OLLAMA_URL,
                    OLLAMA_MODEL,
                    continuation_prompt,
                    timeout=cont_timeout_sec,
                    num_predict=cont_num_predict,
                )
                autoconfirm_llm_ms = int((time.perf_counter() - t_ac_start) * 1000)
            except requests.exceptions.RequestException as exc:
                autoconfirm_llm_ms = int((time.perf_counter() - t_ac_start) * 1000)
                status = getattr(exc.response, "status_code", None)
                autoconfirm_error = f"{status or 'EXC'}: {exc}"
                llm_meta["autoconfirm_error"] = autoconfirm_error
            except Exception as exc:
                autoconfirm_llm_ms = int((time.perf_counter() - t_ac_start) * 1000)
                autoconfirm_error = f"EXC: {type(exc).__name__}: {exc}"
                llm_meta["autoconfirm_error"] = autoconfirm_error
            else:
                continuation_reply = (cont_text or "").strip()
                sentinel_token = cont_meta.get("longform_stop_sentinel_used", "")
                if sentinel_token and continuation_reply.endswith(sentinel_token):
                    continuation_reply = continuation_reply[: -len(sentinel_token)].rstrip()
                if continuation_reply:
                    autoconfirm_used = True
                    autoconfirm_error = ""
                    llm_meta["autoconfirm_error"] = ""
                    llm_meta["autoconfirm_used"] = True
                    llm_meta["autoconfirm_llm_ms"] = autoconfirm_llm_ms
                    if combined_reply:
                        combined_reply = f"{combined_reply}\n\n{continuation_reply}".strip()
                    else:
                        combined_reply = continuation_reply
                    try:
                        self._set_speaking(True)
                        _, cont_play_obj = self._speak(continuation_reply)
                        if cont_play_obj:
                            wait_play(cont_play_obj)
                        self._maybe_clear_play_obj(cont_play_obj)
                    except Exception as e:
                        # Don't let TTS kill logging; record and continue.
                        err = f"TTS_EXC: {type(e).__name__}: {e}"
                        autoconfirm_error = err
                        llm_meta["autoconfirm_error"] = err
                    finally:
                        self._set_speaking(False)
                else:
                    autoconfirm_error = f"{cont_meta.get('http_status', 200)}: empty response"
                    llm_meta["autoconfirm_error"] = autoconfirm_error
                    autoconfirm_used = False
            llm_meta["autoconfirm_llm_ms"] = autoconfirm_llm_ms
            llm_meta["autoconfirm_used"] = autoconfirm_used

        llm_meta["autoconfirm_attempted"] = autoconfirm_attempted
        llm_meta["autoconfirm_used"] = autoconfirm_used
        llm_meta["autoconfirm_llm_ms"] = autoconfirm_llm_ms
        llm_meta["autoconfirm_error"] = autoconfirm_error

        log["reply"] = combined_reply
        log["autoconfirm_attempted"] = autoconfirm_attempted
        log["autoconfirm_used"] = autoconfirm_used
        log["autoconfirm_llm_ms"] = autoconfirm_llm_ms
        log["autoconfirm_error"] = autoconfirm_error
        log["keep_alive"] = llm_meta.get("keep_alive", OLLAMA_KEEP_ALIVE)
        log["warmup_fired"] = _warmup_meta.get("fired", False)
        log["warmup_ms"] = _warmup_meta.get("ms", 0)
        log["warmup_error"] = _warmup_meta.get("error", "")
        log["http_status"] = llm_meta.get("http_status", _last_http_status)
        self._append_log(log)
        self._set_speaking(False)
        if self._wake_required and self._wake_active:
            self._reset_wake_timer()
        return log
    def _start_wake_server(self):
        if getattr(self, "_wake_server", None) is not None:
            return
        if not WAKE_BIND or WAKE_PORT <= 0:
            return
        try:
            server = _WakeHTTPServer((WAKE_BIND, WAKE_PORT), self)
        except Exception as e:
            print(f"[wake] server not started: {e}", file=sys.stderr)
            self._wake_server = None
            return
        self._wake_server = server
        thread = threading.Thread(target=server.serve_forever, name="WakeSignalServer", daemon=True)
        thread.start()
        self._wake_server_thread = thread

    def _set_current_play_obj(self, play_obj):
        with self._playback_lock:
            self._current_play_obj = play_obj
        return play_obj

    def _maybe_clear_play_obj(self, play_obj):
        with self._playback_lock:
            if self._current_play_obj is play_obj:
                self._current_play_obj = None

    def _stop_playback(self):
        with self._playback_lock:
            play_obj = self._current_play_obj
            self._current_play_obj = None
        if play_obj:
            self.player.stop(play_obj)
        self._set_speaking(False)

    def _open_wake_window(self):
        if not self._wake_required:
            return
        with self._wake_lock:
            was_active = self._wake_active
            self._wake_active = True
            self._start_wake_timer_locked()
        if not was_active:
            self._log_event("wake_open", window_ms=self._wake_window_ms)

    def _start_wake_timer_locked(self):
        if self._wake_timer:
            try:
                self._wake_timer.cancel()
            except Exception:
                pass
        if self._wake_window_ms <= 0:
            self._wake_timer = None
            return
        timer = threading.Timer(self._wake_window_ms / 1000.0, self._wake_timeout)
        timer.daemon = True
        self._wake_timer = timer
        timer.start()

    def _wake_timeout(self):
        self._close_wake_window(reason="timeout")

    def _close_wake_window(self, reason: str = "manual") -> bool:
        with self._wake_lock:
            if not self._wake_active:
                if self._wake_timer:
                    try:
                        self._wake_timer.cancel()
                    except Exception:
                        pass
                    self._wake_timer = None
                return False
            self._wake_active = False
            timer = self._wake_timer
            self._wake_timer = None
        if timer:
            try:
                timer.cancel()
            except Exception:
                pass
        self._log_event("wake_close", reason=reason)
        return True

    def _reset_wake_timer(self):
        if not self._wake_required:
            return
        with self._wake_lock:
            if not self._wake_active:
                return
            self._start_wake_timer_locked()

    def _perform_hush(self, reply: str, final_text: str, ts_iso: str, asr_latency_ms: int, turn_start: float) -> dict:
        self._stop_playback()
        self._log_event("hush", text=final_text)
        hush_reply = "" if HUSH_SILENT_ACK else reply
        _, hush_meta = _build_ollama_payload("")
        if hush_reply:
            self._set_speaking(True)
            l_tts_ms, play_obj = self._speak(hush_reply)
            first_audio_ms = int((time.perf_counter() - turn_start) * 1000)
        else:
            l_tts_ms = 0
            play_obj = None
            first_audio_ms = int((time.perf_counter() - turn_start) * 1000)
            self._set_speaking(False)
        if HUSH_CLOSES_WINDOW and self._wake_required:
            closed = self._close_wake_window(reason="hush")
        else:
            closed = False
        log = {
            "ts": ts_iso,
            "text": final_text,
            "reply": hush_reply,
            "l_asr_ms": asr_latency_ms,
            "l_llm_ms": 0,
            "l_tts_ms": l_tts_ms,
            "l_total_ms": first_audio_ms,
            "router": True,
            "ack": False,
            "ack_replaced": False,
            "ack_type": "none",
            "route": "router",
            "voice": VOICE_NAME,
            "tool": "self.hush",
            "hush": True,
            "wake_closed": closed,
            "router_variant": reply,
            "long_form": LONG_FORM_MODE,
            "num_predict": NUM_PREDICT,
            "stop_tokens_used": list(hush_meta.get("stop_tokens", [])),
            "longform_system_append": hush_meta.get("longform_system_append", False),
            "longform_stop_sentinel_used": hush_meta.get("longform_stop_sentinel_used", ""),
            "longform_retry_used": False,
            "longform_suggest_continue": False,
            "refusal_glitch": False,
            "autoconfirm_used": False,
            "autoconfirm_llm_ms": 0,
            "temperature": hush_meta.get(
                "temperature",
                LONGFORM_TEMPERATURE if LONG_FORM_MODE else SHORT_FORM_TEMPERATURE,
            ),
        }
        try:
            log["mood"] = getattr(self, "_mood_id", "")
        except Exception:
            pass
        self._append_log(log)
        if play_obj:
            wait_play(play_obj)
            self._maybe_clear_play_obj(play_obj)
            self._set_speaking(False)
        if self._wake_required and self._wake_active:
            self._reset_wake_timer()
        return log

    def _append_log(self, obj: dict):
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_event(self, event: str, **extra):
        payload = {"ts": now_iso(), "event": event}
        if extra:
            payload.update(extra)
        self._append_log(payload)


# ---------------- Public API ----------------

_orchestrator_singleton: Optional[EddieOrchestrator] = None

def get_orchestrator() -> EddieOrchestrator:
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = EddieOrchestrator()
    return _orchestrator_singleton

def handle_final_transcript(final_text: str, asr_latency_ms: int = 0) -> dict:
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


