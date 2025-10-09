# record_3s.py
import sys, sounddevice as sd, soundfile as sf, numpy as np

idx = int(sys.argv[1]) if len(sys.argv) > 1 else None   # device index
rate = int(sys.argv[2]) if len(sys.argv) > 2 else 48000 # samplerate
ch   = int(sys.argv[3]) if len(sys.argv) > 3 else 2     # channels

print(f"[rec] device={idx} rate={rate} ch={ch} -> recording 3s …")
audio = sd.rec(int(3*rate), samplerate=rate, channels=ch, dtype="float32", device=idx)
sd.wait()
# downmix to mono for quick stats
m = audio.mean(axis=1) if audio.ndim==2 else audio
rms = float(np.sqrt((m**2).mean()))
peak = float(np.abs(m).max())
sf.write("test_linein.wav", audio, rate)
print(f"[rec] wrote test_linein.wav  rms={rms:.4f} peak={peak:.4f}")
