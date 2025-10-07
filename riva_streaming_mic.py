# riva_streaming_mic.py  (revert-with-fix)
# Windows-native streaming mic → NVIDIA Riva ASR (gRPC)
# Streams audio to eddie_orchestrator.py (separate process).

import os, sys, queue, argparse, time, json, subprocess, re
from datetime import datetime
import numpy as np
import sounddevice as sd
import grpc

def _type_to_cursor(text: str):
    try:
        # Put text on clipboard, then paste via Ctrl+V at the active cursor
        p = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
        p.communicate(input=(text or '').encode('utf-16le'))
        ps = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^v')"
        subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=False)
    except Exception as e:
        print(f"[type] {e}", file=sys.stderr)

SEND_SILENCE_WHEN_MUTED = True  # keep the stream alive while muting playback
SPEAKING_FLAG_PATH = os.getenv(
    "EDDIE_SPEAKING_FLAG",
    os.path.join(os.path.dirname(__file__), "eddie_speaking.flag"),
)
ECHO_SUPPRESS_MS = int(os.getenv("ECHO_SUPPRESS_MS", "1200"))
WAKE_WINDOW_MS = int(os.getenv("WAKE_WINDOW_MS", "20000"))
WAKE_FLAG_PATH = os.getenv(
    "EDDIE_WAKE_FLAG",
    os.path.join(os.path.dirname(__file__), "eddie_wake.flag"),
)

# --- Use local generated stubs (no nvidia-riva-client import in this process) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "riva_stubs"))
import riva_asr_pb2
import riva_asr_pb2_grpc
import riva_audio_pb2


def resolve_input_device(name_or_index):
    """Return an input device index (or None for default)."""
    if name_or_index is None or str(name_or_index).strip() == "":
        return None
    try:
        return int(name_or_index)
    except ValueError:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0 and name_or_index.lower() in d["name"].lower():
            return i
    raise SystemExit(f"[err] input device not found: {name_or_index}")


def make_audio_stream(rate, block, channels, device_idx=None, is_awake=None):
    """
    Use sd.InputStream (float32) for maximum compatibility on Windows line-in.
    Downmix to mono and convert to int16 PCM bytes for Riva.
    """
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        try:
            x = indata
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            if x.ndim == 2 and x.shape[1] > 1:
                x = x.mean(axis=1, keepdims=True)

            # Pause capture while Eddie is speaking (ignore stale flags)
            try:
                if os.path.exists(SPEAKING_FLAG_PATH):
                    age_ms = (time.time() - os.path.getmtime(SPEAKING_FLAG_PATH)) * 1000.0
                    if age_ms <= ECHO_SUPPRESS_MS:
                        if SEND_SILENCE_WHEN_MUTED:
                            x = np.zeros_like(x)
                        else:
                            return
            except Exception:
                pass

            # Keep ASR/dictation always-on; gate only final handoff to Eddie

            y = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
            q.put(y.tobytes())
        except Exception as e:
            print(f"[audio-exc] {e}", file=sys.stderr)
            q.put(b"")

    stream = sd.InputStream(
        samplerate=rate,
        blocksize=block,
        dtype="float32",
        channels=channels,
        callback=cb,
        device=device_idx if device_idx is not None else None,
    )
    return stream, q


def gen_requests(q, rate, lang, punct):
    cfg = riva_asr_pb2.RecognitionConfig(
        encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=rate,
        language_code=lang,
        max_alternatives=1,
        enable_automatic_punctuation=punct,
        verbatim_transcripts=False,
        audio_channel_count=1,
    )
    yield riva_asr_pb2.StreamingRecognizeRequest(
        streaming_config=riva_asr_pb2.StreamingRecognitionConfig(
            config=cfg, interim_results=True
        )
    )
    while True:
        data = q.get()
        if data is None:
            return
        yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=data)


def _log_event(event: str, **extra):
    try:
        path = os.getenv("EDDIE_LOG", "")
        if not path:
            return
        payload = {"ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z", "event": event}
        if extra:
            payload.update(extra)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def call_orchestrator_subprocess(final_text: str, asr_latency_ms: int = 0) -> dict:
    """
    Invoke eddie_orchestrator.py in a separate process so its riva.client imports
    don't collide with our local riva_stubs. Returns the JSON dict the orchestrator prints.
    """
    py = sys.executable
    orch_path = os.path.join(os.path.dirname(__file__), "eddie_orchestrator.py")
    cmd = [py, orch_path, "--text", final_text]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = (proc.stdout or "").strip()
        payload = {}
        # Prefer full JSON parse first
        try:
            if out:
                payload = json.loads(out)
        except Exception:
            # Fallback: extract the last JSON object from mixed stdout
            try:
                start = out.find("{")
                end = out.rfind("}")
                if start != -1 and end != -1 and end > start:
                    payload = json.loads(out[start : end + 1])
            except Exception:
                payload = {}
        payload.setdefault("l_asr_ms", asr_latency_ms)
        return payload
    except subprocess.CalledProcessError as e:
        print(f"[orchestrator] subprocess failed: {e}\nSTDERR:\n{e.stderr}", file=sys.stderr)
        return {"ts": "", "text": final_text, "reply": "(orchestrator error)", "l_asr_ms": asr_latency_ms}


# -------- Turn detection --------

TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "225"))


def main():
    ap = argparse.ArgumentParser("Riva streaming mic client")
    default_server = os.getenv("RIVA_SPEECH_API", "localhost:50051")
    ap.add_argument(
        "--server",
        default=default_server,
        help=f"Riva host:port (default from RIVA_SPEECH_API or {default_server})"
    )
    ap.add_argument("--rate", type=int, default=48000)          # 48 kHz stable on many Windows inputs
    ap.add_argument("--block", type=int, default=2400)          # ~50ms @ 48k
    ap.add_argument("--device", default=None, help="Input device index or name substring")
    ap.add_argument("--channels", type=int, default=None, help="Force input channels (1 or 2). If omitted, inferred.")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--punct", action="store_true")
    ap.add_argument("--type_to_cursor", action="store_true", help="Paste final transcript at active cursor (Windows)")
    args = ap.parse_args()

    dev_idx = resolve_input_device(args.device)

    # Infer a safe channel count for the selected/default input device.
    # Some Windows host APIs (e.g., WASAPI loopback or certain drivers) require the
    # requested channel count to exactly match the device's input channel count.
    try:
        dev_info = sd.query_devices(dev_idx if dev_idx is not None else None, kind='input')
        max_in = int(dev_info.get('max_input_channels', 0) or 0)
    except Exception:
        dev_info = {}
        max_in = 0

    use_channels = None
    if args.channels in (1, 2):
        # Clamp user-requested channels to device capability (if known)
        if max_in > 0 and args.channels > max_in:
            print(f"[mic] requested channels={args.channels} > device max={max_in}; clamping to {max_in}")
            use_channels = max_in
        else:
            use_channels = args.channels
    else:
        # No explicit channels provided: prefer exact device channel count if known.
        # Fallback to mono if unknown.
        use_channels = max_in if max_in > 0 else 1

    channel = grpc.insecure_channel(args.server)
    stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

    dictation_sent = ""

    def _is_awake() -> bool:
        try:
            if not os.path.exists(WAKE_FLAG_PATH):
                return False
            age = time.time() - os.path.getmtime(WAKE_FLAG_PATH)
            return (age * 1000.0) <= WAKE_WINDOW_MS
        except Exception:
            return False

    def _extend_wake_window():
        try:
            if os.path.exists(WAKE_FLAG_PATH):
                os.utime(WAKE_FLAG_PATH, None)
                _log_event("wake_extend")
        except Exception:
            pass

    # Create stream with chosen channels; if the driver rejects the count, retry once
    # with the device's reported max channels (or 1 as a last resort).
    try:
        stream, q = make_audio_stream(
            args.rate,
            args.block,
            channels=use_channels,
            device_idx=dev_idx,
            is_awake=_is_awake,
        )
    except Exception as e:
        msg = str(e)
        if "Invalid number of channels" in msg or "PaErrorCode -9998" in msg:
            alt = max_in if max_in > 0 else 1
            if alt != use_channels:
                print(f"[mic] {msg} — retrying with channels={alt}")
                stream, q = make_audio_stream(
                    args.rate,
                    args.block,
                    channels=alt,
                    device_idx=dev_idx,
                    is_awake=_is_awake,
                )
            else:
                # Try a minimal mono fallback if even max_in failed
                if alt != 1:
                    print(f"[mic] retrying with channels=1")
                    stream, q = make_audio_stream(
                        args.rate,
                        args.block,
                        channels=1,
                        device_idx=dev_idx,
                        is_awake=_is_awake,
                    )
                else:
                    raise
        else:
            raise

    turn_buf = []
    last_final_t = None

    print(f"[mic] connecting to Riva @ {args.server}")
    print("[mic] streaming...  Ctrl+C to stop")

    last_awake = None
    try:
        with stream:
            while True:
                responses = stub.StreamingRecognize(gen_requests(q, args.rate, args.lang, args.punct))
                restart_stream = False
                for resp in responses:
                    for r in resp.results:
                        alt = r.alternatives[0].transcript if r.alternatives else ""
                        if r.is_final:
                            turn_buf.append(alt)
                            last_final_t = time.perf_counter()
                            full = " ".join(turn_buf).strip()
                            print(f"> {full} ", end="\r", flush=True)
                            if args.type_to_cursor and full and not os.path.exists(SPEAKING_FLAG_PATH):
                                new_full = full
                                if len(new_full) >= len(dictation_sent):
                                    delta = new_full[len(dictation_sent):]
                                else:
                                    delta = new_full
                                if delta:
                                    _type_to_cursor(delta)
                                    dictation_sent = new_full
                        else:
                            print(f"> {' '.join(turn_buf)}{alt}", end="\r", flush=True)

                    now_t = time.perf_counter()
                    cur_awake = _is_awake()
                    if last_awake is None:
                        last_awake = cur_awake
                    elif cur_awake != last_awake:
                        _log_event("wake_open" if cur_awake else "wake_close", window_ms=WAKE_WINDOW_MS)
                        last_awake = cur_awake
                    speaking_now = os.path.exists(SPEAKING_FLAG_PATH)
                    full = " ".join(turn_buf).strip()
                    req_ms = TURN_SILENCE_MS
                    if full and not re.search(r"[\.!?]\s*$", full):
                        req_ms += 100
                    if full and len(full.split()) < 4:
                        req_ms += 150
                    if not speaking_now and last_final_t is not None and (now_t - last_final_t) * 1000 >= req_ms and turn_buf:
                        final_text = full
                        if final_text:
                            if _is_awake():
                                finalize_ms = int((now_t - last_final_t) * 1000)
                                log = call_orchestrator_subprocess(final_text, asr_latency_ms=finalize_ms)
                                print(
                                    f"\n[Eddie] spoke: {log.get('reply')}  "
                                    f"(asr={log.get('l_asr_ms')}ms llm={log.get('l_llm_ms',0)}ms "
                                    f"tts={log.get('l_tts_ms',0)}ms total={log.get('l_total_ms',0)}ms "
                                    f"ack={log.get('ack_type','none')})"
                                )
                                if log.get("request_exit"):
                                    # Wait until speaking flag clears (or becomes stale), then exit
                                    while True:
                                        if not os.path.exists(SPEAKING_FLAG_PATH):
                                            sys.exit(0)
                                        try:
                                            age_ms = (time.time() - os.path.getmtime(SPEAKING_FLAG_PATH)) * 1000.0
                                        except Exception:
                                            age_ms = 0
                                        if age_ms >= ECHO_SUPPRESS_MS:
                                            sys.exit(0)
                                        time.sleep(0.1)
                                # Extend window after speaking ends (no stall if already cleared)
                                start_wait = time.time()
                                while time.time() - start_wait < 5.0 and os.path.exists(SPEAKING_FLAG_PATH):
                                    time.sleep(0.05)
                                _extend_wake_window()
                            else:
                                print("\n[asleep] (final ignored)")
                        turn_buf.clear()
                        last_final_t = None
                        dictation_sent = ""
                        restart_stream = True
                        break

                if not restart_stream:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[exit] done.")
        q.put(None)
        try:
            if _is_awake():
                _log_event("wake_close", reason="shutdown")
        except Exception:
            pass


if __name__ == "__main__":
    main()
