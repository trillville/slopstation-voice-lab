"""Slice a room recording into background clips the augmenter will actually use.

    python bench\\slice_room.py room.wav <training-tree>\\data\\backgrounds\\room

Runs on the GAMING PC, and only AFTER `livekit-wakeword setup`: setup skips the
1.1 GB MUSAN download outright if it finds any wav already under
data/backgrounds, so clips dropped in early cost you the noise set with no
error to notice.

WHY slice at all, when handing the augmenter one long recording is the obvious
move. It picks a background with random.choice over FILES and only then crops
it, so weight is per file, not per second. setup fetches 774 MUSAN wavs; one 40-minute
recording is one candidate in 775 and would land in ~0.1% of the mixes. At 4 s
a clip, 40 minutes becomes ~600 files and the room reaches ~45% of them, which
is the entire reason for recording it. 4 s rather than 2 leaves the augmenter
crop headroom over its 2 s clip_duration.

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
