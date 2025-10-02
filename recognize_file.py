# recognize_file.py — Riva single-file ASR (robust imports + resample)
import sys, os
import numpy as np
import soundfile as sf
import grpc

# Make riva_stubs importable (C:\riva-mic\riva_stubs\*.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "riva_stubs"))

import riva_asr_pb2
import riva_asr_pb2_grpc
import riva_audio_pb2  # <-- AudioEncoding lives here in some stub versions

RIVA_ADDR = os.getenv("RIVA_SPEECH_API", "localhost:50051")
LANG = os.getenv("RIVA_LANG", "en-US")
ASR_MODEL = os.getenv("RIVA_ASR_MODEL", "")  # empty = server default

def to_mono_16k_int16(wav_path):
    data, rate = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    target = 16000
    if rate != target:
        x = np.arange(len(mono), dtype=np.float64)
        xp = np.linspace(0, len(mono)-1, int(len(mono) * (target / rate)))
        mono = np.interp(xp, x, mono).astype(np.float32)
    pcm = np.clip(mono, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()

def main():
    if len(sys.argv) < 2:
        print("Usage: python recognize_file.py <path.wav>")
        sys.exit(2)
    wav = sys.argv[1]
    if not os.path.exists(wav):
        print(f"File not found: {wav}")
        sys.exit(2)

    audio_bytes = to_mono_16k_int16(wav)

    # Resolve enum safely across stub variants
    try:
        ENC_LINEAR_PCM = riva_audio_pb2.AudioEncoding.LINEAR_PCM
    except Exception:
        # last-resort: LINEAR_PCM is 1 in current protos
        ENC_LINEAR_PCM = 1

    req = riva_asr_pb2.RecognizeRequest(
        config=riva_asr_pb2.RecognitionConfig(
            encoding=ENC_LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code=LANG,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            model=ASR_MODEL,
        ),
        audio=audio_bytes,
    )

    try:
        with grpc.insecure_channel(RIVA_ADDR) as ch:
            stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(ch)
            resp = stub.Recognize(req, timeout=30.0)
    except grpc.RpcError as e:
        print(f"Riva ASR RPC failed: {e.code().name} - {e.details()}")
        sys.exit(1)

    printed = False
    for r in resp.results:
        if r.alternatives:
            print("FINAL:", r.alternatives[0].transcript)
            printed = True
    if not printed:
        print("No transcript")

if __name__ == "__main__":
    main()
