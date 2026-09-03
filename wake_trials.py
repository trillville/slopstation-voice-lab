"""Measure a wake model on the K15's own microphone: recall over deliberate
attempts, and false accepts per hour while nobody is talking to it.

    .venv\\Scripts\\python wake_trials.py trials      say the wake word, repeatedly
    .venv\\Scripts\\python wake_trials.py soak        leave it running for hours

Both report the PEAK score, not the crossing score. openWakeWord's score is
still rising when it first crosses the threshold, so a threshold set from a
crossing is set from a number the next attempt will not reproduce. The peak
scan reads on for PEAK_HOPS after the crossing and keeps the high-water mark;
it costs those hops of latency, which is why the production lane does not do
it and this script does.

Runs against the SHIPPING listener - `pip install` the slopstation package,
and the model, thresholds and device name all come from its config.json - so
the numbers apply to the lane that will run the model. A session is never
started and nothing is spoken back.
"""

import argparse
import time

from slopstation import config
from slopstation.agent.speech import audio

PEAK_HOPS = 15  # 1.2 s of peak search past the crossing
REFRACTORY_S = 1.0  # one hit per attempt


def listen(listener, stream, threshold):
    """Block until the score crosses `threshold`; return (score, peak)."""
    import numpy as np

    while True:
        chunk = np.frombuffer(
            stream.read(listener.CHUNK, exception_on_overflow=False), np.int16
        )
        score = listener.score_chunk(chunk)
        if score < threshold:
            continue
        peak = score
        for _ in range(PEAK_HOPS):
            more = np.frombuffer(
                stream.read(listener.CHUNK, exception_on_overflow=False), np.int16
            )
            peak = max(peak, listener.score_chunk(more))
        listener.model.reset()
        return score, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("trials", "soak"))
    ap.add_argument(
        "--threshold",
        type=float,
        help="override config.json's wakeThreshold for this run",
    )
    args = ap.parse_args()

    cfg = config.current()
    voice = cfg["voice"]
    threshold = args.threshold if args.threshold is not None else voice["wakeThreshold"]
    pa, input_idx, _ = audio.open_audio(voice)
    listener = audio.WakeListener(pa, voice, input_idx)
    stream = listener._open_stream()
    print(f"model {listener.model_name}  threshold {threshold}")
    print("say the wake word" if args.mode == "trials" else "leave it running")

    t0, n = time.time(), 0
    try:
        while True:
            score, peak = listen(listener, stream, threshold)
            n += 1
            hours = (time.time() - t0) / 3600
            if args.mode == "trials":
                print(f"  {n:3}  score {score:.2f}  peak {peak:.3f}")
            else:
                print(
                    f"  {n:3}  peak {peak:.3f}  after {hours:.2f} h  "
                    f"({n / max(hours, 0.01):.1f}/hour)"
                )
            time.sleep(REFRACTORY_S)
    except KeyboardInterrupt:
        hours = (time.time() - t0) / 3600
        print(f"\n{n} hits in {hours:.2f} h")
    finally:
        audio.close_stream_quietly(stream)
        pa.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
