# eddie_orchestrator.py
# Orchestrates: ASR(final) -> Router -> Ollama(Qwen) -> Riva TTS (Male-1)
# Fast-ACK via fillers if LLM drags. Per-run JSONL logging (one file per session).

import os, json, time, threading, queue, re, sys, wave, tempfile, random
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

# --- Config via env / sane defaults ---
RIVA_SPEECH_API_URL = os.getenv("RIVA_SPEECH_API_URL", "localhost:50051")
OLLAMA_URL          = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
VOICE_NAME          = os.getenv("VOICE_NAME", "English-US.Male-1")
TOOLS_URL           = os.getenv("TOOLS_URL")

# Per-run log rotation (kept). You can override with EDDIE_LOG.
_HERE = Path(__file__).resolve().parent
_LOGS_DIR = _HERE / "logs"
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
    os.path.join(os.path.dirname(__file__), "eddie_speaking.flag"),
)
FILLER_FIRST_MS     = int(os.getenv("FILLER_FIRST_MS", "1000"))          # don't emit filler before 1s
FILLER_POST_PAUSE_MS= int(os.getenv("FILLER_POST_PAUSE_MS", "150"))      # pause after filler completes
FILLER_PROB         = float(os.getenv("FILLER_PROB", "0.4"))             # chance to emit any filler
MOOD_LOCK           = os.getenv("MOOD_LOCK", "").strip()

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


# ---------------- Ollama LLM ----------------

# Set from XML at startup
SYSTEM_PROMPT_V2 = ""

def call_ollama_generate(ollama_url: str, model: str, user_text: str, timeout: float = 6.0) -> str:
    payload = {
        "model": model,
        "prompt": user_text,
        "system": SYSTEM_PROMPT_V2,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 160,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "stop": ["\n"]
        }
    }
    r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ---------------- Orchestrator ----------------

class EddieOrchestrator:
    def __init__(self):
        self.tts = RivaTTS(RIVA_SPEECH_API_URL, VOICE_NAME)
        self.player = AudioPlayer(sample_rate=44100)
        # warm up TTS so first real turn isn't cold
        try:
            _ = self.tts.synth("ok", sample_rate=44100)
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
                self._ack_pcm[_ack] = self.tts.synth(_ack, sample_rate=44100)
            except Exception:
                self._ack_pcm[_ack] = None

    def _speak(self, text: str) -> Tuple[int, object]:
        t0 = time.perf_counter()
        audio = self.tts.synth(text, sample_rate=44100)
        t1 = time.perf_counter()
        play_obj = self.player.play_pcm16le(audio)
        l_tts_ms = int((t1 - t0) * 1000)
        return l_tts_ms, play_obj

    def _play_ack(self, text: str):
        pcm = self._ack_pcm.get(text)
        try:
            if pcm: return self.player.play_pcm16le(pcm)
        except Exception:
            pass
        _, play_obj = self._speak(text)
        return play_obj

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
        cfg_path = Path(__file__).resolve().parent / "config" / "personality.xml"
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
        turn_start = time.perf_counter()
        ts_iso = now_iso()
        final_text = (final_text or "").strip()

        # 1) Router — instant
        rr = self._router_response(final_text)
        if rr:
            replies, weights, tool = rr
            selected = ""
            if len(replies) == 1:
                selected = replies[0]
            elif len(replies) > 1:
                selected = random.choices(replies, weights=weights, k=1)[0] if weights else random.choice(replies)
            reply = clip_one_sentence(selected, MAX_REPLY_CHARS)
            exit_request = False; exit_reason = ""
            if tool == "self.exit":
                handled, override, exit_request, exit_reason = self._maybe_exit_override(final_text)
                if handled and override:
                    reply = clip_one_sentence(override, MAX_REPLY_CHARS)
                exit_stage = self._exit_state["stage"]
            # Clock tool: tailor reply before speaking
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
            self._set_speaking(True)
            l_tts_ms, play_obj = self._speak(reply)
            if tool and tool not in ("self.exit", "clock.now"):
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
            }
            try: log["mood"] = getattr(self, "_mood_id", "")
            except Exception: pass
            log["router_variant"] = selected
            if tool == "self.exit":
                log.update({"request_exit": exit_request, "exit_stage": exit_stage, "exit_reason": exit_reason})
            self._append_log(log)
            wait_play(play_obj)
            self._set_speaking(False)
            return log

        # 2) Else LLM with optional, randomized filler
        ack_sent = False
        ack_plays: list[object] = []
        ack_type = "none"
        l_llm_ms = 0
        twf_engaged = False
        twf_tiers_used = 0
        twf_total_extra_ms = 0

        llm_out_q: "queue.Queue[str]" = queue.Queue(maxsize=1)
        llm_timing: dict = {}

        def llm_worker():
            t0 = time.perf_counter()
            try:
                resp = call_ollama_generate(OLLAMA_URL, OLLAMA_MODEL, final_text, timeout=6.0)
            except Exception:
                resp = (self._hiccup_line or self._fallback_line or "")
            t1 = time.perf_counter()
            llm_timing["ms"] = int((t1 - t0) * 1000)
            try: llm_out_q.put_nowait(resp)
            except queue.Full: pass

        thread = threading.Thread(target=llm_worker, daemon=True)
        thread.start()

        try:
            resp = llm_out_q.get(timeout=FILLER_FIRST_MS / 1000.0)
            l_llm_ms = llm_timing.get("ms", FILLER_FIRST_MS)
        except queue.Empty:
            twf_engaged = True
            resp = None
            for tier_index in range(max(1, self._twf_max)):
                if self._filler_texts and (
                    random.random() < max(0.0, min(1.0, FILLER_PROB))
                ):
                    filler = random.choices(
                        self._filler_texts, weights=self._filler_weights, k=1
                    )[0]
                    ack_type = filler.lower(); ack_sent = True
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

        final_reply = clip_one_sentence(resp, MAX_REPLY_CHARS)

        if ack_sent and ack_plays:
            for play_obj in ack_plays:
                wait_play(play_obj)
            time.sleep(max(0, FILLER_POST_PAUSE_MS) / 1000.0)
            self._set_speaking(False)

        self._set_speaking(True)
        l_tts_ms, play_obj_final = self._speak(final_reply)
        first_audio_ms = int((time.perf_counter() - turn_start) * 1000)

        log = {
            "ts": ts_iso,
            "text": final_text,
            "reply": final_reply,
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
        }
        try: log["mood"] = getattr(self, "_mood_id", "")
        except Exception: pass
        if twf_engaged:
            log["twf_tiers_used"] = twf_tiers_used
            log["twf_total_extra_ms"] = twf_total_extra_ms
        self._append_log(log)
        wait_play(play_obj_final)
        self._set_speaking(False)
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
