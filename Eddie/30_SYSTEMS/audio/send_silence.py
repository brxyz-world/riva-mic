# send_silence.py
import os, sys, grpc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "riva_stubs"))
import riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2

ch = grpc.insecure_channel("localhost:50051")
stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(ch)

cfg = riva_asr_pb2.RecognitionConfig(
    encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,          # <-- note the 'hertz'
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=True,
    verbatim_transcripts=False,
    audio_channel_count=1,
)

reqs = [
    riva_asr_pb2.StreamingRecognizeRequest(
        streaming_config=riva_asr_pb2.StreamingRecognitionConfig(
            config=cfg, interim_results=True
        )
    ),
    riva_asr_pb2.StreamingRecognizeRequest(audio_content=b"\x00\x00" * 16000),
]

for resp in stub.StreamingRecognize(iter(reqs)):
    print(resp)
