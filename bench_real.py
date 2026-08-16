"""Rank candidate models on REAL audio through the runtime that will run them.

    Bench.bat                    artifacts\\*.onnx PLUS the vendored models in
                                 k15\\voice\\models - "does it beat the model the
                                 K15 already runs" is the default question, so
                                 the incumbent is in the lineup by default
    Bench.bat --models a.onnx b.onnx     just these
    Bench.bat --target-fa 0.5            allow 0.5 false accepts/hour
    Bench.bat --noise-only               negatives only, ranked by noise
                                 ceiling. The one comparison that works ACROSS
                                 phrases: recall cannot be compared between
                                 hey_jarvis and hey_alfred (different words),
                                 but how hard the same room audio pushes each
                                 model can - which is the experiment for "is
                                 the phrase itself the problem"

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
import math
import sys
import wave
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent          # .../wake-training, in the repo

CHUNK = 1280                    # oWW's native 80 ms hop
RATE = 16000
# After a detection the live agent calls model.reset(), so one loud moment can
# only ever cost ONE false accept. Streaming here without reset (50x cheaper -
# one pass serves every threshold) and applying the same refractory offline is
# the same count for far less compute. 1.5 s covers oWW's score decay.
REFRACTORY_HOPS = 19
THRESHOLDS = np.round(np.arange(0.02, 1.00, 0.02), 2)


def wilson(k, n, z=1.96):
    """95% interval on a proportion, so the table can say how much of a recall
    difference is real. With 20 positives the half-width around 0.9 is ~0.13 -
    i.e. a 5-point gap between two models is NOISE at this sample size, and the
    2026-08-16 bench nearly shipped a decision on exactly such a gap. Wilson
    rather than normal approximation because n is small and p is near 1, which
    is where the normal interval is at its worst."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


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
    # simply the fraction of clips whose peak clears T. Empty in --noise-only.
    peaks = (np.array([trace(m, load_wav(p)).max() for p in positives])
             if positives else np.zeros(0))

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
    is also the best one that does. Rows carry the raw EVENT COUNT alongside
    the rate: with thin negatives the count is the honest number (0.22 h means
    one event IS 4.5/hr, and a rate printed alone hides that resolution)."""
    rows = []
    for t in THRESHOLDS:
        fa = sum(count_crossings(s, t) for s in neg_traces)
        rec = float((peaks >= t).mean()) if peaks.size else None
        rows.append((float(t), rec, int(fa),
                     fa / hours if hours else float("nan")))
    ok = [r for r in rows if r[3] <= target_fa]
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
    ap.add_argument("--noise-only", action="store_true",
                    help="skip positives; rank by noise ceiling (cross-phrase)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    pos_dir = args.positives or root / "bench" / "positives"
    neg_dir = args.negatives or root / "data" / "heldout"
    positives = [] if args.noise_only else sorted(Path(pos_dir).glob("**/*.wav"))
    negatives = sorted(Path(neg_dir).glob("**/*.wav"))

    models = args.models
    if not models:
        # The vendored models are the incumbents. The default question is
        # "does anything in artifacts beat what the K15 already runs", so they
        # are in the lineup unless --models says otherwise - the 2026-08-16
        # comparison only happened because v1.0 was copied in by hand, and a
        # bench that only sees the new candidates can only ever crown one.
        models = (sorted((root / "artifacts").glob("*.onnx"))
                  + sorted((HERE.parent / "k15" / "voice" / "models")
                           .glob("*.onnx")))
        seen = set()
        models = [m for m in models
                  if not (m.resolve() in seen or seen.add(m.resolve()))]

    if not positives and not args.noise_only:
        sys.exit(f"no positives under {pos_dir}\n"
                 f"Record yourself saying the wake phrase and split it with "
                 f"k15/voice/bench/slice_utterances.py - ONE utterance per wav "
                 f"(or pass --noise-only for a negatives-only comparison).")
    if not negatives:
        sys.exit(f"no negatives under {neg_dir}")
    if not models:
        sys.exit(f"no .onnx under {root / 'artifacts'} or the repo models dir")

    print(f"positives  {len(positives)} utterances"
          + (" (--noise-only)" if args.noise_only else ""))
    print(f"negatives  {len(negatives)} files from {neg_dir}")
    print(f"budget     <= {args.target_fa} false accepts/hour\n")

    results = {}
    for mp in models:
        peaks, neg_traces, hours = bench(mp, positives, negatives)
        best, rows = operating_point(peaks, neg_traces, hours, args.target_fa)
        ceiling = max((float(s.max()) for s in neg_traces), default=0.0)
        entry = {
            "n_positives": int(peaks.size),
            "peak_median": round(float(np.median(peaks)), 3) if peaks.size else None,
            "peak_min": round(float(peaks.min()), 3) if peaks.size else None,
            "noise_ceiling": round(ceiling, 3),
            "negative_hours": round(hours, 2),
            "threshold": None, "recall": None, "recall_ci": None,
            "fa_events": None, "fa_per_hour": None,
            "curve": [{"t": t, "recall": None if r is None else round(r, 3),
                       "fa_events": e, "fa_hr": round(f, 2)}
                      for t, r, e, f in rows],
        }
        if best and peaks.size:
            t = best[0]
            k = int((peaks >= t).sum())
            lo, hi = wilson(k, int(peaks.size))
            entry.update(threshold=t, recall=round(k / peaks.size, 3),
                         recall_ci=[round(lo, 3), round(hi, 3)],
                         fa_events=int(best[2]), fa_per_hour=round(best[3], 2))
        results[Path(mp).stem] = entry
        print(f"  scored {Path(mp).stem}", flush=True)

    hours = max(r["negative_hours"] for r in results.values())
    if args.noise_only:
        print(f"\n{'model':36}{'noise ceiling':>14}{'neg hours':>10}")
        print("-" * 60)
        for name, r in sorted(results.items(),
                              key=lambda kv: kv[1]["noise_ceiling"]):
            print(f"{name:36}{r['noise_ceiling']:>14.3f}"
                  f"{r['negative_hours']:>10.2f}")
        print("\nLower is better: the ceiling is where the room can push a "
              "model WITHOUT the\nphrase being said. It is the one number "
              "comparable across phrases - recall\nnever is - so jarvis-vs-"
              "alfred rows here are a read on the phrase itself.")
    else:
        print(f"\n{'model':36}{'peak med':>9}{'peak min':>9}{'noise':>7}"
              f"{'thresh':>7}{'recall':>8}{'95% CI':>12}{'FA':>4}{'/hr':>6}")
        print("-" * 98)
        # Best first: recall at the operating point IS the ranking. A model
        # with no operating point never fits the budget and sorts last.
        for name, r in sorted(results.items(),
                              key=lambda kv: (kv[1]["recall"] is not None,
                                              kv[1]["recall"] or 0),
                              reverse=True):
            if r["recall"] is None:
                print(f"{name:36}{r['peak_median']:>9.3f}{r['peak_min']:>9.3f}"
                      f"{r['noise_ceiling']:>7.3f}"
                      f"   -- never fits the budget --")
                continue
            lo, hi = r["recall_ci"]
            print(f"{name:36}{r['peak_median']:>9.3f}{r['peak_min']:>9.3f}"
                  f"{r['noise_ceiling']:>7.3f}{r['threshold']:>7.2f}"
                  f"{r['recall']:>7.0%}{f'{lo:.0%}-{hi:.0%}':>12}"
                  f"{r['fa_events']:>4}{r['fa_per_hour']:>6.2f}")

        n = max(r["n_positives"] for r in results.values())
        print(f"\nRead the CI before the ranking: with {n} positives, recall "
              f"is only known to\nthe printed interval (Wilson 95%). Two "
              f"models whose intervals overlap are NOT\nranked by this table, "
              f"whatever the point estimates say.")
        print(f"\n'noise' is the highest score any negative audio produced - "
              f"the floor a\nthreshold has to clear. 'peak min' is the WORST "
              f"real utterance: if that sits\nbelow the threshold, some way of "
              f"saying it never works, whatever the median says.")
        print("\nThe threshold column IS deployable - same runtime, same "
              "room, same voice.\nStart there, then confirm with "
              "--wake-trials on the K15.")
    if hours:
        print(f"\nRate resolution: {hours:.2f} h of negatives means one event "
              f"= {1 / hours:.1f} FA/hr. A budget\nbelow that can only be met "
              f"by ZERO events at the chosen threshold.")

    # --noise-only writes ELSEWHERE by default. Its rows carry recall=None, so
    # sharing the filename silently destroyed a full bench and left a results
    # file that looks complete and answers no recall question - which cost a
    # re-measurement on 2026-08-16 when a threshold had to be read back out.
    out = args.json or root / ("bench_results.noise.json" if args.noise_only
                               else "bench_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfull curves: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
