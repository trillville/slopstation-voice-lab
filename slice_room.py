"""Slice a room recording into background clips the augmenter will actually use.

    python bench\\slice_room.py room.wav <training-tree>\\data\\backgrounds\\room

Runs on the GAMING PC, and only AFTER `livekit-wakeword setup`: setup skips
the 1.1 GB MUSAN download outright, with no error, if it finds any wav already
under data/backgrounds.

The augmenter picks a background with random.choice over FILES and only then
crops, so weight is per file, not per second. Against setup's 774 MUSAN wavs
one 40-minute recording lands in ~0.1% of mixes; at 4 s a clip it becomes ~600
files and reaches ~45%. 4 s rather than 2 leaves crop headroom over the
augmenter's 2 s clip_duration.

Stdlib only: the training box has no k15 venv.
"""
import sys
import wave
from pathlib import Path

CLIP_S = 4


def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src)) as r:
        rate, chans, width = r.getframerate(), r.getnchannels(), r.getsampwidth()
        if (rate, chans) != (16000, 1):
            sys.exit(f"{src}: need 16 kHz mono, got {rate} Hz / {chans} ch")
        n = rate * CLIP_S
        i = 0
        while True:
            frames = r.readframes(n)
            if len(frames) < n * width:
                break               # drop the short tail rather than pad it
            with wave.open(str(dest / f"room_{i:05d}.wav"), "wb") as o:
                o.setnchannels(chans)
                o.setsampwidth(width)
                o.setframerate(rate)
                o.writeframes(frames)
            i += 1
    print(f"{i} clips of {CLIP_S}s -> {dest}")


if __name__ == "__main__":
    main()
