# recognize_file.py
import os, sys, grpc, soundfile as sf, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "riva_stubs"))
import riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2

wav = sys.argv[1] if len(sys.argv) > 1 else "test_linein.wav"
data, rate = sf.read(wav, dtype="float32")
if data.ndim == 2:
    data = data.mean(axis=1)  # mono
pcm16 = np.clip(data * 32767.0, -32768, 32767).astype(np.int16).tobytes()

ch = grpc.insecure_channel("localhost:50051")
stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(ch)

cfg = riva_asr_pb2.RecognitionConfig(
    encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=rate,        # matches audio file
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=True,
    verbatim_transcripts=False,
    audio_channel_count=1,
)

req = riva_asr_pb2.RecognizeRequest(config=cfg, audio=pcm16)  # <-- 'audio' (not audio_content)
resp = stub.StreamingRecognize(iter([req]))
print(resp)
if resp.results and resp.results[0].alternatives:
    print("TEXT:", resp.results[0].alternatives[0].transcript)
