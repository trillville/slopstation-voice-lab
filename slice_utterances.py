"""Split a recording of repeated wake phrases into one-utterance clips.

    .venv\\Scripts\\python bench\\slice_utterances.py positives.wav out\\dir

Feeds two things that both want the same shape - one utterance per wav:
wake-training/bench_real.py (real-voice recall) and bench/train_verifier.py
(the verifier's positive examples).

Say the phrase, pause about 3 seconds, repeat - 30 times, varying it the way
real use varies: from both ends of the couch, quietly, over the TV, mid-sentence
("hey alfred play hades"). A bench built only from careful clear utterances
measures a room nobody lives in.

WHY THE LEAD-IN MATTERS, and it is the whole reason this is not a two-line
silence splitter: openWakeWord scores a ~2 s window ENDING at the current hop,
so a clip trimmed tight to the phrase gives the model a window that is mostly
empty and it scores low for reasons that have nothing to do with the model.
Every clip therefore carries LEAD_S of whatever preceded it, clamped so it can
never reach back into the previous utterance. Clips that could not get a full
lead-in are reported rather than silently kept, because they will read as false
rejects and drag a model's score down unfairly.
"""
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 320                     # 20 ms RMS frames
LEAD_S = 2.0                    # oWW's window - the clip must fill it
# THE SCORE PEAKS AFTER THE TALKER STOPS, and this is the second time that has
# cost a measurement. "alfred" ends on a low-energy /d/, so the energy gate
# closes while the score is still climbing - it crests 0.7-1.1 s later. Measured
# 2026-08-16 on room_test.wav against the same 20 utterances streamed
# continuously (median peak 0.892):
#
#     tail 0.5 s -> median 0.396      tail 1.5 s -> median 0.888
#     tail 1.0 s -> median 0.888      tail 2.0 s -> median 0.888
#
# A 0.5 s tail therefore understates every model by more than half and would
# have been read as "the retrain destroyed the model". 2.0 s is past the point
# it converges; the extra second costs nothing but a few hops of inference.
TAIL_S = 2.0
BRIDGE_S = 0.35                 # "hey ... alfred" is one utterance, not two
MIN_UTTERANCE_S = 0.25
MAX_UTTERANCE_S = 3.0
MIN_LEAD_S = 1.5                # below this the window is short - warn


def voiced_frames(pcm):
    """Frames whose RMS clears a floor taken from the recording itself.

    The floor is the 20th percentile rather than the minimum: a room is never
    silent, and anchoring to the quietest single frame would put the gate under
    the noise and call the whole file voiced."""
    n = len(pcm) // FRAME
    rms = np.sqrt(np.mean(np.square(
        pcm[:n * FRAME].astype(np.float64).reshape(n, FRAME)), axis=1))
    floor = np.percentile(rms, 20)
    return rms > max(floor * 3.0, 60.0), rms


def runs(mask):
    """Contiguous True regions, with short gaps bridged."""
    bridge = int(BRIDGE_S * RATE / FRAME)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > bridge:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1].strip())
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    with wave.open(str(src)) as w:
        if (w.getframerate(), w.getnchannels()) != (RATE, 1):
            sys.exit(f"{src}: need 16 kHz mono, got {w.getframerate()} Hz "
                     f"/ {w.getnchannels()} ch")
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    dest.mkdir(parents=True, exist_ok=True)

    mask, _ = voiced_frames(pcm)
    keep = [(a, b) for a, b in runs(mask)
            if MIN_UTTERANCE_S <= (b - a + 1) * FRAME / RATE <= MAX_UTTERANCE_S]

    short, prev_end = [], 0
    for i, (a, b) in enumerate(keep):
        start_s, end_s = a * FRAME, (b + 1) * FRAME
        # Never reach back into the previous utterance - a clip holding two
        # phrases is one detection, not two, and quietly deflates recall.
        lead = min(int(LEAD_S * RATE), start_s - prev_end)
        if lead < MIN_LEAD_S * RATE:
            short.append(i)
        clip = pcm[max(0, start_s - lead):min(len(pcm), end_s + int(TAIL_S * RATE))]
        prev_end = end_s
        with wave.open(str(dest / f"utt_{i:03d}.wav"), "wb") as o:
            o.setnchannels(1)
            o.setsampwidth(2)
            o.setframerate(RATE)
            o.writeframes(clip.tobytes())

    print(f"{len(keep)} utterances -> {dest}")
    print("Check that count against how many times you actually said it. Too "
          "few means\nthe gate missed the quiet ones; too many means the room "
          "moved between takes.")
    if short:
        print(f"\nWARNING: {len(short)} clip(s) got under {MIN_LEAD_S}s of "
              f"lead-in ({short[:8]}).\nopenWakeWord scores a ~2 s trailing "
              f"window, so those start part-filled and will\nread as false "
              f"rejects. Leave ~3 s between repetitions and re-record, or "
              f"delete them.")


if __name__ == "__main__":
    main()
