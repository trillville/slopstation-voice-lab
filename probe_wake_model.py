"""Does a wake model load, cost what it should, and mean the same thing here
as it did on the machine that trained it?

    .venv\\Scripts\\python bench\\probe_wake_model.py models\\hey_x_v1.0.onnx
    .venv\\Scripts\\python bench\\probe_wake_model.py m.onnx --dump k15.json
    python bench\\probe_wake_model.py --compare gamepc.json k15.json

Not the blind suite: this takes a model path, so it is a bench tool for the
day a new wake model arrives. See docs/custom-wakeword-design.md.

WHY: the models are trained by livekit-wakeword and run by openWakeWord, and
that interop is NOT documented upstream - it was verified by measurement
(2026-08-13) and can only be re-verified the same way. Two failure modes it
catches, neither of which is visible from a score:

  1. The export stops transferring (a custom op, a changed input signature).
     The agent would crash-loop every 10 s behind the supervisor.
  2. The runtimes stop agreeing numerically, which silently voids every
     FPPH/recall number the training machine reported.

The first ~5 hops are EXPECTED to differ: openWakeWord returns 0.0 while its
embedding buffer primes, where livekit scores zero-padded audio. Steady-state
agreement is the assertion; a trained model diverging after warm-up is the
thing that must stop a deploy.

Exit code is non-zero on a failed compare, so this can gate a model swap.
"""
import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

CHUNK = 1280                    # audio.py's oWW-native 80 ms hop
WINDOW = 32000                  # livekit's stateless ~2 s window
WARMUP_HOPS = 5                 # oWW returns 0.0 until its buffer fills
PEAK_TOLERANCE = 0.02           # peak-score agreement between runtimes


def load_wav(path):
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            sys.exit(f"{path}: need 16 kHz mono, got {w.getframerate()} Hz "
                     f"/ {w.getnchannels()} ch")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def score_oww(model_path, pcm):
    """Streaming: one 80 ms chunk per hop, the model keeps its own buffer."""
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    m.reset()
    out, times = [], []
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        t0 = time.perf_counter()
        s = m.predict(pcm[i:i + CHUNK])
        times.append((time.perf_counter() - t0) * 1000)
        out.append(float(max(s.values())))
    return out, times


def score_livekit(model_path, pcm):
    """Stateless: the caller owns the window, so hand it the whole 2 s ring."""
    from livekit.wakeword import WakeWordModel
    m = WakeWordModel(models=[str(model_path)])
    ring, out, times = np.zeros(WINDOW, np.int16), [], []
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        ring = np.concatenate([ring[CHUNK:], pcm[i:i + CHUNK]])
        t0 = time.perf_counter()
        s = m.predict(ring)
        times.append((time.perf_counter() - t0) * 1000)
        out.append(float(max(s.values())))
    return out, times


def compare(a_path, b_path):
    """Per-wav, never concatenated: EVERY wav starts a fresh model, so a
    flattened series hides a warm-up in the middle of it and reads as a
    steady-state divergence. That false FAIL is easier to hit than the real
    one it would be masking."""
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    if a.keys() != b.keys():
        sys.exit(f"FAIL - different wavs ({sorted(a)} vs {sorted(b)}): "
                 "both runs must use the SAME wav set")
    failed = []
    for name in sorted(a):
        if len(a[name]) != len(b[name]):
            sys.exit(f"FAIL - {name}: hop count differs "
                     f"({len(a[name])} vs {len(b[name])})")
        x, y = np.array(a[name]), np.array(b[name])
        # PEAK, not per-hop. The runtimes are equivalent only up to a sub-hop
        # mel alignment, so on a rising edge - exactly where a wake phrase
        # lives - the same model legitimately differs by ~0.1 between them
        # (measured on the shipped hey_jarvis). Asserting per-hop rejects good
        # models. The wake loop compares a PEAK against a threshold and cares
        # WHEN it crossed, so those are the two things that have to transfer.
        dpeak = abs(x.max() - y.max())
        # Firing hop is read from AFTER warm-up on both sides: openWakeWord
        # cannot fire during priming by construction, so including it would
        # score that structural difference as a disagreement.
        fx, fy = (next((i for i, v in enumerate(s)
                        if i >= WARMUP_HOPS and v >= 0.5), None) for s in (x, y))
        drift = None if fx is None or fy is None else abs(fx - fy)
        bad = dpeak > PEAK_TOLERANCE or (fx is None) != (fy is None) or \
            (drift is not None and drift > 1)
        print(f"  {name:16} peak {x.max():.4f} vs {y.max():.4f} "
              f"(d={dpeak:.4f})   first>=0.5 {fx} vs {fy}"
              f"   per-hop mean d={np.abs(x - y)[WARMUP_HOPS:].mean():.4f}"
              + ("   <-- FAIL" if bad else ""))
        if bad:
            failed.append(name)
    if not failed:
        print(f"  PASS - peaks within {PEAK_TOLERANCE}, firing hop within 1")
        return 0
    print(f"  FAIL on {failed}. The export is not transferring; "
          "training-machine metrics do not apply here.")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="path to the wake model .onnx")
    ap.add_argument("--wav", action="append", required=False,
                    help="16 kHz mono wav; repeatable. Same wav on both sides "
                         "of a --compare.")
    ap.add_argument("--engine", choices=("oww", "livekit"), default="oww",
                    help="oww = the production runtime (default); livekit = "
                         "the training machine's runtime")
    ap.add_argument("--dump", help="write the per-hop score series as JSON")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not (args.model and args.wav):
        ap.error("need a model and at least one --wav (or --compare)")

    score = score_oww if args.engine == "oww" else score_livekit
    print(f"=== {Path(args.model).name} on {args.engine} ===")
    series, all_times = {}, []
    for wav in args.wav:
        pcm = load_wav(wav)
        s, t = score(args.model, pcm)
        series[Path(wav).stem] = s
        all_times += t
        fired = [i for i, v in enumerate(s) if v >= 0.5]
        print(f"  {Path(wav).stem:16} hops={len(s):>3}  max={max(s):.4f}"
              + (f"  first>=0.5 at hop {fired[0]}" if fired else "  never fired"))

    mean = statistics.mean(all_times)
    # The budget that matters: one hop is 80 ms of wall clock, and the wake
    # loop must consume the mic stream faster than it fills or PortAudio drops
    # audio - a missed wake, not a late one.
    print(f"\n  per-hop ms      mean={mean:.2f}  "
          f"p95={sorted(all_times)[int(len(all_times) * .95)]:.2f}")
    print(f"  always-on cost  {mean / 80 * 100:.1f}% of one core"
          + ("" if mean < 8 else "   <-- HIGH, expected ~2-3% on oww"))

    if args.dump:
        Path(args.dump).write_text(json.dumps(series))
        print(f"  wrote {args.dump} ({len(series)} wavs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
