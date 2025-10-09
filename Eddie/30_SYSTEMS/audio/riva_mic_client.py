import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def record_wav(out_path: Path, seconds: float, rate: int = 16000):
    channels = 1
    print(f"[mic] Recording {seconds:.1f}s @ {rate} Hz …")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=channels, dtype="int16")
    sd.wait()
    sf.write(out_path, audio, rate, subtype="PCM_16")
    print(f"[mic] Saved: {out_path}  ({out_path.stat().st_size} bytes)")


def run_riva_transcribe(wav_path: Path, server_container: str = "riva-speech") -> str:
    """
    Mount the current directory into the riva-speech image and run Riva's example
    transcriber inside the container, using the server's network namespace.
    """
    # Container sees our current folder at /workspace
    workdir = wav_path.parent.resolve()
    mount_arg = f"{workdir}:/workspace"

    cmd = [
        "docker", "run", "--rm", "-i",
        "--network", f"container:{server_container}",
        "-v", mount_arg,
        "nvcr.io/nvidia/riva/riva-speech:2.19.0",
        "python3", "/opt/riva/examples/transcribe_file.py",
        "--input-file", f"/workspace/{wav_path.name}",
        "--server", "localhost:50051",
        "--print-confidence",
    ]

    print("[riva] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"[riva] docker run failed with code {proc.returncode}")

    # Riva’s script prints intermediate lines; final result usually starts with "## "
    final = None
    for line in proc.stdout.splitlines():
        if line.strip().startswith("## "):
            final = line.strip().lstrip("#").strip()
    if not final:
        # Fallback: just return the last non-empty line
        for line in reversed(proc.stdout.splitlines()):
            if line.strip():
                final = line.strip()
                break
    return final or "(no result)"


def main():
    parser = argparse.ArgumentParser(description="Riva mic → text (via docker) client")
    parser.add_argument("--seconds", type=float, default=5.0, help="Record duration in seconds")
    parser.add_argument("--server-container", default="riva-speech",
                        help="Name of the running Riva server container")
    parser.add_argument("--loop", action="store_true",
                        help="Record/transcribe repeatedly until Ctrl+C")
    args = parser.parse_args()

    # Quick sanity check: is docker available?
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True)
    except Exception:
        print("[err] Docker is not available. Start Docker Desktop and try again.")
        sys.exit(1)

    # Optional: check that server container exists
    ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    names = {n.strip() for n in ps.stdout.splitlines()}
    if args.server_container not in names:
        print(f"[warn] Container '{args.server_container}' not found in `docker ps`.")
        print("       Make sure your Riva server is running (e.g., `bash riva_start.sh` in WSL).")

    try:
        while True:
            with tempfile.TemporaryDirectory() as td:
                wav_path = Path(td) / "mic_temp.wav"
                record_wav(wav_path, args.seconds, rate=16000)
                result = run_riva_transcribe(wav_path, server_container=args.server_container)
                print(f"\n📝 Transcript: {result}\n")
            if not args.loop:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[exit] Stopped.")


if __name__ == "__main__":
    main()
