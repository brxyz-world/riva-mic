# riva_streaming_mic.py  (revert-with-fix)
# Windows-native streaming mic → NVIDIA Riva ASR (gRPC)
# On PTT release, shells out to eddie_orchestrator.py (separate process).

import os, sys, queue, argparse, time, json, subprocess, re
import numpy as np
import sounddevice as sd
import grpc
import ctypes

def _type_to_cursor(text: str):
    try:
        # Put text on clipboard, then paste via Ctrl+V at the active cursor
        p = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
        p.communicate(input=(text or '').encode('utf-16le'))
        ps = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^v')"
        subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=False)
    except Exception as e:
        print(f"[type] {e}", file=sys.stderr)

# --- Hotkeys ---
VK_F24, VK_SPACE, VK_F23 = 0x87, 0x20, 0x86  # F24=PTT, SPACE=PTT, F23=toggle always-on
_user32 = ctypes.windll.user32

def _key_down(vk) -> bool:
    return (_user32.GetAsyncKeyState(vk) & 0x8000) != 0

def _ptt_down() -> bool:
    # returns True while either key is physically down
    return _key_down(VK_F24) or _key_down(VK_SPACE)

SEND_SILENCE_WHEN_MUTED = True  # keep the stream alive when not holding the key
SPEAKING_FLAG_PATH = os.getenv(
    "EDDIE_SPEAKING_FLAG",
    os.path.join(os.path.dirname(__file__), "eddie_speaking.flag"),
)

# --- Use local generated stubs (no nvidia-riva-client import in this process) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "riva_stubs"))
import riva_asr_pb2
import riva_asr_pb2_grpc
import riva_audio_pb2


def resolve_input_device(name_or_index):
    """Return an input device index (or None for default)."""
    if name_or_index is None:
        return None
    try:
        return int(name_or_index)
    except ValueError:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0 and name_or_index.lower() in d["name"].lower():
            return i
    raise SystemExit(f"[err] input device not found: {name_or_index}")


def make_audio_stream(rate, block, channels, device_idx=None, always_on=False):
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

            # Virtual PTT: linger window & always-on; also mute while Eddie is speaking
            eddie_speaking = os.path.exists(SPEAKING_FLAG_PATH)
            now_down = (_ptt_virtual_down() or always_on) and not eddie_speaking
            if not now_down:
                if SEND_SILENCE_WHEN_MUTED:
                    x = np.zeros_like(x)
                else:
                    return

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


# -------- Virtual PTT + Always-on / Wake word --------

PTT_LINGER_MS = int(os.getenv("PTT_LINGER_MS", "220"))
PTT_VIRTUAL_DOWN = False  # module global for callback to read

def _ptt_virtual_down() -> bool:
    return PTT_VIRTUAL_DOWN

def _update_virtual_down(down_now, was_down, linger_until):
    global PTT_VIRTUAL_DOWN
    # Start linger on release
    if was_down and not down_now and linger_until[0] is None:
        linger_until[0] = time.perf_counter() + (PTT_LINGER_MS / 1000.0)
    # Cancel linger if key is pressed again
    if down_now:
        linger_until[0] = None
    # Compute virtual
    PTT_VIRTUAL_DOWN = down_now or (linger_until[0] is not None and time.perf_counter() < linger_until[0])


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
    ap.add_argument("--always_on", action="store_true", help="Start in always-on mode (wake word required).")
    ap.add_argument("--wake", default=os.getenv("EDDIE_WAKE", "eddie"), help="Wake word for always-on mode.")
    ap.add_argument("--type_to_cursor", action="store_true", help="Paste final transcript at active cursor (Windows)")
    args = ap.parse_args()

    dev_idx = resolve_input_device(args.device)
    use_channels = 1
    if dev_idx is not None:
        max_in = max(1, sd.query_devices()[dev_idx].get("max_input_channels", 1))
        use_channels = 2 if max_in >= 2 else 1
    if args.channels in (1, 2):
        use_channels = args.channels

    # gRPC stub (local stubs; keep this process free of nvidia-riva-client)
    channel = grpc.insecure_channel(args.server)
    stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

    always_on = bool(args.always_on)
    wake = args.wake.lower().strip()
    wake_armed = True
    AWAKE_WINDOW_SEC = int(os.getenv("AWAKE_WINDOW_SEC", "20"))
    WAKE_SILENCE_MS = int(os.getenv("WAKE_SILENCE_MS", "225"))
    awake_until = None
    dictation_sent = ""
    pending_awake_turn = False
    wake_cut_len = None

    stream, q = make_audio_stream(args.rate, args.block, channels=use_channels, device_idx=dev_idx, always_on=always_on)

    # Per-turn state
    turn_buf = []
    was_down = False
    last_final_t = None
    linger_until = [None]  # boxed so closures can mutate

    print(f"[mic] connecting to Riva @ {args.server}")
    print("[mic] streaming… (hold F24 or SPACE to talk; F23 toggles always-on)  Ctrl+C to stop")
    if always_on:
        print(f"[mode] ALWAYS-ON enabled (wake word: '{wake}')")

    try:
        with stream:
            f23_prev = False
            while True:
                # Start/Restart the gRPC stream bound to the current queue
                responses = stub.StreamingRecognize(gen_requests(q, args.rate, args.lang, args.punct))
                restart_stream = False
                for resp in responses:
                    # F23 toggle disabled; always-on fixed in this mode

                    # Process incoming ASR results
                    for r in resp.results:
                        alt = r.alternatives[0].transcript if r.alternatives else ""
                        if r.is_final:
                            turn_buf.append(alt)
                            last_final_t = time.perf_counter()
                            print(f"\r✅ {' '.join(turn_buf)} ", end="", flush=True)
                            if always_on:
                                full = " ".join(turn_buf).strip()
                                t = full.lower()
                                if wake_armed and wake in t:
                                    # Arm first turn after wake and cut off the wake word portion.
                                    wake_armed = False
                                    idx = t.rfind(wake)
                                    wake_cut_len = idx + len(wake)
                                    pending_awake_turn = True
                                    dictation_sent = ""
                                elif args.type_to_cursor and not (awake_until and time.perf_counter() < awake_until) and not os.path.exists(SPEAKING_FLAG_PATH):
                                    # Default dictation while sleeping: paste incremental delta only
                                    new_full = full
                                    if len(new_full) >= len(dictation_sent):
                                        delta = new_full[len(dictation_sent):]
                                    else:
                                        delta = new_full  # reset case
                                    if delta:
                                        _type_to_cursor(delta)
                                        dictation_sent = new_full
                        else:
                            print(f"\r… {' '.join(turn_buf)}{alt}", end="", flush=True)

                    # Update virtual PTT and possibly fire a turn
                    down = _ptt_down()
                    _update_virtual_down(down, was_down, linger_until)

                    # Silence-based turn boundary when awake or when a wake-triggered turn is pending
                    now_t = time.perf_counter()
                    is_awake = (awake_until is not None and now_t < awake_until) or pending_awake_turn
                    speaking_now = os.path.exists(SPEAKING_FLAG_PATH)
                    full = " ".join(turn_buf).strip()
                    req_ms = WAKE_SILENCE_MS
                    if full and not re.search(r"[\.!?]\s*$", full):
                        req_ms += 100
                    if len(full.split()) < 4:
                        req_ms += 150
                    if is_awake and not speaking_now and last_final_t is not None and (now_t - last_final_t) * 1000 >= req_ms and turn_buf:
                        if pending_awake_turn:
                            # Use text after the wake word
                            t = full.lower()
                            cut = wake_cut_len if wake_cut_len is not None else 0
                            final_text = full[cut:].strip() or "hi"
                        else:
                            final_text = full
                        if final_text:
                            finalize_ms = int((now_t - last_final_t) * 1000)
                            log = call_orchestrator_subprocess(final_text, asr_latency_ms=finalize_ms)
                            print(
                                f"\n[Eddie] spoke: {log.get('reply')}  "
                                f"(asr={log.get('l_asr_ms')}ms llm={log.get('l_llm_ms',0)}ms "
                                f"tts={log.get('l_tts_ms',0)}ms total={log.get('l_total_ms',0)}ms "
                                f"ack={log.get('ack_type','none')})"
                            )
                            awake_until = time.perf_counter() + AWAKE_WINDOW_SEC
                        # reset state for next turn
                        turn_buf.clear()
                        last_final_t = None
                        pending_awake_turn = False
                        wake_cut_len = None
                        wake_armed = True
                        # restart ASR stream to drop any queued transcripts
                        restart_stream = True
                        break

                    if not always_on:
                        if was_down and not _ptt_virtual_down():
                            final_text = " ".join(turn_buf).strip()
                            if final_text:
                                if args.type_to_cursor:
                                    _type_to_cursor(final_text)
                                finalize_ms = 0
                                if last_final_t is not None:
                                    finalize_ms = int((time.perf_counter() - last_final_t) * 1000)
                                log = call_orchestrator_subprocess(final_text, asr_latency_ms=finalize_ms)
                                print(
                                    f"\n[Eddie] spoke: {log.get('reply')}  "
                                    f"(asr={log.get('l_asr_ms')}ms llm={log.get('l_llm_ms',0)}ms "
                                    f"tts={log.get('l_tts_ms',0)}ms total={log.get('l_total_ms',0)}ms)"
                                )
                            # reset turn
                            turn_buf.clear()
                            last_final_t = None
                            linger_until[0] = None
                            restart_stream = True
                            break

                    was_down = down

                if not restart_stream:
                    # Streaming ended unexpectedly; exit outer loop
                    break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[exit] done.")
        q.put(None)


if __name__ == "__main__":
    main()
