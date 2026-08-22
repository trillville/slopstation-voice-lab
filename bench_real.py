"""Rank candidate models on REAL audio through the runtime that will run them.

    Bench.bat                    artifacts\\*.onnx PLUS the vendored models in
                                 k15\\voice\\models - the incumbent is in the
                                 lineup by default
    Bench.bat --models a.onnx b.onnx     just these
    Bench.bat --target-fa 0.5            allow 0.5 false accepts/hour
    Bench.bat --noise-only               negatives only, ranked by noise
                                 ceiling - the one comparison valid ACROSS
                                 phrases (recall is not: different words)

livekit's eval cannot rank these models: on 2026-08-15 it scored small, medium
and large identically (3 false positives in 17.85 h, ~99.3% recall, AUT 0.0000)
and on 2026-08-16 both surviving sizes landed on threshold 0.18 with ~99.8%
recall. The same three models on 20 s of real couch audio had median peaks of
0.083, 0.892 and 0.585.

Real voice, real room, and openWakeWord's STREAMING runtime rather than
livekit's stateless one - the parity gate (2026-08-13) measured the two
disagreeing by 0.021-0.075 per hop, the gap scaling with head size.

Two ways to get a lying answer, both hit 2026-08-16: negatives one model
trained on rig the comparison in its favour, so bench negatives must be
recorded after the last model was trained; and clips trimmed too tight
understate every model by more than half, because the score crests ~1 s AFTER
the talker stops (slice_utterances.TAIL_S). Sanity-check a surprise by
streaming the whole recording continuously and comparing peaks - no clipping,
no reset.
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
# After a detection the live agent calls model.reset(), so one loud moment costs
# at most ONE false accept. Streaming here without reset (50x cheaper - one pass
# serves every threshold) and applying the refractory offline gives the same
# count. 1.5 s covers oWW's score decay.
REFRACTORY_HOPS = 19
THRESHOLDS = np.round(np.arange(0.02, 1.00, 0.02), 2)
DRAWS = 3                       # noise draws averaged per SNR cell


def wilson(k, n, z=1.96):
    """95% interval on a proportion. With 20 positives the half-width around
    0.9 is ~0.13, so a 5-point gap between two models is NOISE. Wilson because
    n is small and p is near 1."""
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
    """Distinct detections above `threshold`, one per refractory window. oWW
    holds a high score for a second or more after an event, so per-hop counting
    reads a single door slam as fifteen false accepts."""
    n, i = 0, 0
    while i < len(scores):
        if scores[i] >= threshold:
            n += 1
            i += REFRACTORY_HOPS
        else:
            i += 1
    return n


def active_rms(x):
    """RMS of the frames that carry the utterance. Every positive clip is ~2 s
    of quiet lead-in plus the phrase plus a 2 s tail, so whole-clip RMS would
    overstate the SNR of a mix by 10 dB or more."""
    n = len(x) // 320
    if not n:
        return 0.0
    frames = x[:n * 320].astype(np.float64).reshape(n, 320)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    loud = energy > 0.2 * energy.max() if energy.max() > 0 else None
    return float(energy[loud].mean()) if loud is not None and loud.any() \
        else float(energy.mean())


def mix_at_snr(speech, noise_pool, snr_db, rng):
    """Speech plus a random slice of real room audio at a stated SNR.
    Synthesised rather than recorded: slice_utterances segments by energy and
    cannot find speech at or below the background, so recorded noisy positives
    would be biased toward the loud takes."""
    pool = noise_pool
    while len(pool) < len(speech):
        pool = np.concatenate([pool, noise_pool])
    start = int(rng.integers(0, len(pool) - len(speech) + 1))
    seg = pool[start:start + len(speech)].astype(np.float64)

    s_rms, n_rms = active_rms(speech), float(np.sqrt((seg ** 2).mean()))
    if s_rms <= 0 or n_rms <= 0:
        return speech
    seg *= (s_rms / (10 ** (snr_db / 20))) / n_rms
    out = speech.astype(np.float64) + seg
    # Scale the WHOLE mix if it would clip: scaling both parts equally keeps
    # the SNR where it was asked to be.
    peak = np.abs(out).max()
    if peak > 32767:
        out *= 32767 / peak
    return out.astype(np.int16)


def snr_sweep(model_path, positives, noise_pool, snrs, threshold, seed=0):
    """Recall vs SNR at ONE fixed threshold - the deployed operating point.
    Clean positives measure a quiet room and negatives measure false accepts
    under noise; neither measures the phrase being SAID under noise."""
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    out = {}
    for snr in snrs:
        # DRAWS repeats: the room audio is non-stationary and the measured
        # spread across seeds at a single draw was 6-8 points. Seeded per
        # (model, snr) so every model meets identical mixes.
        recalls, medians = [], []
        for d in range(DRAWS):
            rng = np.random.default_rng(seed + 1000 * d)
            peaks = np.array([
                trace(m, mix_at_snr(load_wav(p), noise_pool, snr, rng)).max()
                for p in positives])
            recalls.append(float((peaks >= threshold).mean()))
            medians.append(float(np.median(peaks)))
        out[snr] = {"recall": sum(recalls) / len(recalls),
                    "recall_spread": round(max(recalls) - min(recalls), 3),
                    "peak_median": round(sum(medians) / len(medians), 3)}
    return out


def bench(model_path, positives, negatives):
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    # One peak per positive clip: each clip is ONE utterance. Empty under
    # --noise-only.
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
    it buys - lowest because recall falls monotonically with the threshold.
    Rows carry the raw EVENT COUNT too: with 0.22 h of negatives one event IS
    4.5/hr, which a rate alone hides."""
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
    ap.add_argument("--snr-sweep", nargs="*", type=float, default=None,
                    metavar="DB", help="also measure recall with the phrase "
                    "MIXED INTO room audio at these SNRs (default: 20 15 10 "
                    "5 0 -5). The couch condition the clean bench misses.")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    pos_dir = args.positives or root / "bench" / "positives"
    neg_dir = args.negatives or root / "data" / "heldout"
    positives = [] if args.noise_only else sorted(Path(pos_dir).glob("**/*.wav"))
    negatives = sorted(Path(neg_dir).glob("**/*.wav"))

    models = args.models
    if not models:
        # The vendored models are the incumbents: the default question is "does
        # anything in artifacts beat what the K15 already runs".
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
    # openWakeWord raises ValueError on a missing file, so one model whose
    # training run failed would take the whole bench down with it.
    missing = [m for m in models if not Path(m).is_file()]
    models = [m for m in models if Path(m).is_file()]
    for m in missing:
        print(f"[skip] not on disk: {Path(m).name}")
    if not models:
        sys.exit(f"no .onnx found (checked {len(missing)} explicit path(s), "
                 f"else {root / 'artifacts'} and the repo models dir)")

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
        # Best first; a model with no operating point sorts last.
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

    if args.snr_sweep is not None and positives:
        snrs = args.snr_sweep or [20, 15, 10, 5, 0, -5]
        pool = np.concatenate([load_wav(p) for p in negatives])
        print(f"\n\n=== recall vs SNR: the phrase SAID OVER room audio ===")
        print(f"each model at its OWN threshold from the table above; "
              f"{len(positives)} utterances per cell\n")
        print(f"{'model':34}{'thresh':>7}" +
              "".join(f"{s:>+7.0f}dB" for s in snrs))
        print("-" * (41 + 9 * len(snrs)))
        for name, r in sorted(results.items(),
                              key=lambda kv: -(kv[1]["recall"] or 0)):
            if r["threshold"] is None:
                continue
            sw = snr_sweep(next(m for m in models
                                if Path(m).stem == name),
                           positives, pool, snrs, r["threshold"])
            results[name]["snr_sweep"] = {str(k): v for k, v in sw.items()}
            print(f"{name:34}{r['threshold']:>7.2f}" +
                  "".join(f"{sw[s]['recall']:>8.0%}" for s in snrs))
        print("\n+20 dB is a quiet room; 0 dB is the talker and the TV equally "
              "loud at the mic;\nnegative is the TV winning. A model whose "
              "column collapses between +10 and 0\nis the 'fine in a quiet "
              "room, useless with a game on' complaint, measured.")

    # --noise-only writes ELSEWHERE by default: its rows carry recall=None, so
    # a shared filename would overwrite a full bench with a recall-free one.
    out = args.json or root / ("bench_results.noise.json" if args.noise_only
                               else "bench_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfull curves: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
