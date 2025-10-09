import os, wave
from riva.client import Auth, SpeechSynthesisService, AudioEncoding

auth = Auth(uri=os.getenv("RIVA_SPEECH_API_URL","localhost:50051"), use_ssl=False)
tts = SpeechSynthesisService(auth)

resp = tts.synthesize(
    text="Test voice: Eddie should be audible now.",
    voice_name="English-US.Male-1",
    language_code="en-US",
    encoding=AudioEncoding.LINEAR_PCM,
    sample_rate_hz=22050,
)
audio = resp if isinstance(resp, (bytes, bytearray)) else getattr(resp, "audio", b"")
print("TTS bytes:", len(audio))

with wave.open("out.wav","wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
    wf.writeframes(audio)
