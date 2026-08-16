"""Rank candidate models on REAL audio through the runtime that will run them.

    Bench.bat                            every .onnx in artifacts\\
    Bench.bat --models a.onnx b.onnx     just these
    Bench.bat --target-fa 0.5            allow 0.5 false accepts/hour

THIS IS THE ONLY EVAL THAT HAS EVER RANKED THESE MODELS CORRECTLY.

livekit's eval cannot. On 2026-08-15 it scored small, medium and large
identically - 3 false positives in 17.85 h, ~99.3% recall, AUT 0.0000 - and on
2026-08-16 it did it again, both surviving sizes landing on threshold 0.18 with
~99.8% recall and DET curves pinned flat against both axes. That is a saturated
test: both candidates ace it, so it has no resolving power left. Meanwhile the
same three models measured on 20 seconds of real couch audio had median peaks
of 0.083, 0.892 and 0.585 - a ranking the synthetic eval never saw.

Three things make this different, and all three are necessary:

  * REAL VOICE, not TTS. The positives are the phrase as this household says
    it, in this room, at couch distance.
  * REAL ROOM, not MUSAN. The negatives are the actual TV and the actual games.
  * openWakeWord's STREAMING runtime, not livekit's stateless one. The parity
    gate (2026-08-13) established that the two disagree by 0.021-0.075 per hop
    and that the gap scales with head size - so a livekit score is not a
    prediction about the K15, and ranking on it ranks the wrong thing.

The output is the number that actually matters: with the threshold set as
tightly as the false-accept budget allows, what fraction of real wake words
still fire.

TWO WAYS TO GET A LYING ANSWER OUT OF THIS, both hit on 2026-08-16:

  * NEGATIVES ONE MODEL TRAINED ON. If model A saw these clips during training
    and model B did not, A's noise ceiling is partly memorisation and the
    comparison is rigged in its favour. Bench negatives must be held out from
    EVERY candidate, which in practice means recording them after the last
    model was trained, not carving them out of the background pool.
  * CLIPS TRIMMED TOO TIGHT. The score crests ~1 s AFTER the talker stops
    (see slice_utterances.TAIL_S). A short tail understated every model by
    more than half and read as "the retrain destroyed it".

Both produce a confident, precise, wrong table. Sanity-check any surprising
result by streaming the whole recording continuously and comparing peaks -
that path has no clipping and no reset, so it is the arbiter.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

CHUNK = 1280                    # oWW's native 80 ms hop
RATE = 16000
# After a detection the live agent calls model.reset(), so one loud moment can
# only ever cost ONE false accept. Streaming here without reset (50x cheaper -
# one pass serves every threshold) and applying the same refractory offline is
# the same count for far less compute. 1.5 s covers oWW's score decay.
REFRACTORY_HOPS = 19
THRESHOLDS = np.round(np.arange(0.02, 1.00, 0.02), 2)


def load_wav(path):
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels()) != (RATE, 1):
            sys.exit(f"{path}: need 16 kHz mono, got {w.getframerate()} Hz "
                     f"/ {w.getnchannels()} ch")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def trace(model, pcm):
    """Score every 80 ms hop, exactly as audio.py's wake loop does."""
    model.reset()
    return np.array([float(max(model.predict(pcm[i:i + CHUNK]).values()))
                     for i in range(0, len(pcm) - CHUNK + 1, CHUNK)])


def count_crossings(scores, threshold):
    """Distinct detections above `threshold`, one per refractory window.

    Per-hop counting would be wrong by an order of magnitude: oWW holds a high
    score for a second or more after an event, so a single door slam reads as
    fifteen false accepts and every model looks equally hopeless."""
    n, i = 0, 0
    while i < len(scores):
        if scores[i] >= threshold:
            n += 1
            i += REFRACTORY_HOPS
        else:
            i += 1
    return n


def bench(model_path, positives, negatives):
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    # One peak per positive clip: each clip is ONE utterance, so its peak is
    # the model's best answer for that utterance, and recall at threshold T is
    # simply the fraction of clips whose peak clears T.
    peaks = np.array([trace(m, load_wav(p)).max() for p in positives])

    neg_traces, hours = [], 0.0
    for p in negatives:
        pcm = load_wav(p)
        neg_traces.append(trace(m, pcm))
        hours += len(pcm) / RATE / 3600
    return peaks, neg_traces, hours


def operating_point(peaks, neg_traces, hours, target_fa):
    """Lowest threshold whose false-accept rate fits the budget, and the recall
    it buys. Lowest rather than highest because recall falls monotonically as
    the threshold rises - so the cheapest threshold that satisfies the budget
    is also the best one that does."""
    rows = []
    for t in THRESHOLDS:
        fa = sum(count_crossings(s, t) for s in neg_traces)
        rows.append((float(t), float((peaks >= t).mean()),
                     fa / hours if hours else float("nan")))
    ok = [r for r in rows if r[2] <= target_fa]
    return (ok[0] if ok else None), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\tillm\wake"))
    ap.add_argument("--models", type=Path, nargs="*", default=None,
                    help="default: every .onnx under <root>/artifacts")
    ap.add_argument("--positives", type=Path, default=None,
                    help="default: <root>/bench/positives - ONE utterance per wav")
    ap.add_argument("--negatives", type=Path, default=None,
                    help="default: <root>/data/heldout - room/game, no wake word")
    ap.add_argument("--target-fa", type=float, default=1.0,
                    help="false accepts per hour the threshold must fit inside")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    pos_dir = args.positives or root / "bench" / "positives"
    neg_dir = args.negatives or root / "data" / "heldout"
    positives = sorted(Path(pos_dir).glob("**/*.wav"))
    negatives = sorted(Path(neg_dir).glob("**/*.wav"))
    models = args.models or sorted((root / "artifacts").glob("*.onnx"))

    if not positives:
        sys.exit(f"no positives under {pos_dir}\n"
                 f"Record yourself saying the wake phrase and split it with "
                 f"k15/voice/bench/slice_utterances.py - ONE utterance per wav.")
    if not negatives:
        sys.exit(f"no negatives under {neg_dir}")
    if not models:
        sys.exit(f"no .onnx under {root / 'artifacts'}")

    print(f"positives  {len(positives)} utterances")
    print(f"negatives  {len(negatives)} files from {neg_dir}")
    print(f"budget     <= {args.target_fa} false accepts/hour\n")

    results = {}
    for mp in models:
        peaks, neg_traces, hours = bench(mp, positives, negatives)
        best, rows = operating_point(peaks, neg_traces, hours, args.target_fa)
        ceiling = max((s.max() for s in neg_traces), default=0.0)
        results[Path(mp).stem] = {
            "peak_median": round(float(np.median(peaks)), 3),
            "peak_min": round(float(peaks.min()), 3),
            "noise_ceiling": round(float(ceiling), 3),
            "negative_hours": round(hours, 2),
            "threshold": best[0] if best else None,
            "recall": round(best[1], 3) if best else None,
            "fa_per_hour": round(best[2], 2) if best else None,
            "curve": [{"t": t, "recall": round(r, 3), "fa_hr": round(f, 2)}
                      for t, r, f in rows],
        }
        print(f"  scored {Path(mp).stem}", flush=True)

    print(f"\n{'model':34}{'peak med':>10}{'peak min':>10}{'noise':>8}"
          f"{'thresh':>8}{'recall':>9}{'FA/hr':>8}")
    print("-" * 87)
    # Best first: recall at the operating point IS the ranking. A model with no
    # operating point cannot hit the budget at any threshold and sorts last.
    for name, r in sorted(results.items(),
                          key=lambda kv: (kv[1]["recall"] is not None,
                                          kv[1]["recall"] or 0), reverse=True):
        if r["recall"] is None:
            print(f"{name:34}{r['peak_median']:>10.3f}{r['peak_min']:>10.3f}"
                  f"{r['noise_ceiling']:>8.3f}     -- never fits the budget --")
            continue
        print(f"{name:34}{r['peak_median']:>10.3f}{r['peak_min']:>10.3f}"
              f"{r['noise_ceiling']:>8.3f}{r['threshold']:>8.2f}"
              f"{r['recall']:>8.0%}{r['fa_per_hour']:>8.2f}")

    print(f"\n'noise' is the highest score any negative audio produced - the "
          f"floor a\nthreshold has to clear. 'peak min' is the WORST real "
          f"utterance: if that sits\nbelow the threshold, some way of saying "
          f"it never works, whatever the median says.")
    print("\nThe threshold column IS deployable - same runtime, same room, "
          "same voice.\nStart there, then confirm with --wake-trials on the K15.")

    out = args.json or root / "bench_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfull curves: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
