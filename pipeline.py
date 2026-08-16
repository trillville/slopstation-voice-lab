"""One command from a config to a vendored wake model. Gaming PC only.

    Train.bat                          every size in SIZES, from scratch
    Train.bat medium                   just medium
    Train.bat medium --from train      reuse the clips already on disk
    Train.bat --list                   what is on disk, and what would run

Runs generate -> augment -> features -> train -> export -> eval by calling
livekit-wakeword's own run_* functions, not its CLI. Two reasons, both learned
the hard way:

  * The CLI cannot express the SNR fix. livekit hardcodes background mixing at
    +5..+15 dB (augment.py, `snr_db_range` default) and exposes no way to
    change it, so every positive the 2026-08-15 model ever saw had the speaker
    5-15 dB LOUDER than the interference. The couch is the opposite case - a
    talker reaches the mic 10-20 dB BELOW TV dialogue - so the deployment
    condition was literally absent from training. patch_augmentation() below is
    the whole point of this file.
  * run_eval RETURNS its metrics dict, so the numbers come from the source
    instead of being scraped out of log lines a formatting change would break.

WHY one file and not the old six hand-run CLI steps plus sweep.py: the data
stages and the model stages have to agree about one config, and running them
from two places is exactly how a model got trained at the wrong SNR while
nobody noticed. sweep.py covered train/export/eval only; this supersedes it.

The heavy inputs - ACAV100M's 16.5 GB of features, MUSAN, the RIRs, the venv -
stay OUT of the repo under --root (default C:\\Users\\tillm\\wake). Only the
code is version-controlled; a git pull must never move 16 GB.
"""
import argparse
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.data.augment import run_augment
from livekit.wakeword.data.features import run_extraction
from livekit.wakeword.data.generate import run_generate
from livekit.wakeword.eval.evaluate import run_eval
from livekit.wakeword.export.onnx import run_export
from livekit.wakeword.training.trainer import run_train

# torch.onnx's exporter prints a U+2705 on success and Windows' console is
# cp1252, so `export` dies with UnicodeEncodeError AFTER the training run has
# already finished - the most expensive possible place to lose a job. Measured
# 2026-08-15; it kills the plain `livekit-wakeword export` CLI too.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "alfred.yaml"
# v1.0 is the model in the repo now, so the retrain is v1.1 - two files called
# v1.0 with different weights is the one confusion that costs a whole evening.
VERSION = "v1.1"

SIZES = ["medium", "large"]             # small scored 0.083 on real voice - out
STAGES = ["generate", "augment", "features", "train"]


def patch_augmentation(lo, hi, clean_p):
    """Widen the background-mix SNR and let some clips through clean.

    livekit's default is `snr_db_range=(5.0, 15.0)` applied to 100% of clips,
    with no probability gate and no config key. openWakeWord's own recipe, for
    comparison, is min_snr_in_db=-10 / max_snr_in_db=15 at p=0.75 - a 25 dB
    span with a quarter left clean, against livekit's 10 dB span with none.
    That gap is the single best explanation for why the custom model collapses
    in a loud room while stock hey_jarvis does not.

    Monkeypatched rather than edited into site-packages on purpose: a venv
    rebuild silently reverts a patched file and the next model would train at
    the old SNR with nothing in the logs to say so. Here it travels with the
    repo and prints itself on every run.

    Do NOT widen this AND raise augmentation.rounds together: rounds compound
    (round N mixes fresh noise into round N-1's output), so three rounds at
    -5 dB land near -18 dB and teach the model to fire on noise."""
    from livekit.wakeword.data import augment as _aug
    original = _aug.AudioAugmentor.mix_with_background

    def widened(self, audio, snr_db_range=(lo, hi)):
        if clean_p and random.random() < clean_p:
            return audio                # the clean quarter, oWW-style
        return original(self, audio, snr_db_range)

    _aug.AudioAugmentor.mix_with_background = widened
    print(f"[patch] background SNR {lo:+g}..{hi:+g} dB, {clean_p:.0%} left clean "
          f"(livekit default: +5..+15 dB, 0% clean)", flush=True)


def load_config(root, size=None):
    """A config with every path resolved against --root and an optional size
    override, built from the raw YAML so the user's file is never rewritten -
    a run that mutates its own input makes itself unreproducible.

    The path rewriting is not cosmetic. livekit resolves background_paths and
    rir_paths relative to the CURRENT DIRECTORY, which was harmless while the
    config and the data shared a folder and is a silent failure now that the
    code lives in the repo and the data does not: a missing background folder
    does not raise, it just augments against nothing."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["data_dir"] = str(root / "data")
    raw["output_dir"] = str(root / "output")
    aug = raw.setdefault("augmentation", {})
    for key in ("background_paths", "rir_paths"):
        aug[key] = [p if Path(p).is_absolute() else str((root / p).resolve())
                    for p in aug.get(key, [])]
    if size:
        raw.setdefault("model", {})["model_size"] = size
    cfg = WakeWordConfig(**raw)
    for d in cfg.augmentation.background_paths + cfg.augmentation.rir_paths:
        if not Path(d).is_dir():
            sys.exit(f"augmentation path does not exist: {d}\n"
                     f"Paths in alfred.yaml are relative to --root ({root}).")
    return cfg


def provenance(cfg, snr, clean_p):
    """What the numbers were measured against. Without this a results file is
    a few rows of metrics with no way to tell which data produced them - and
    with the SNR now patched, the augmentation settings are the single most
    important thing to be able to read back."""
    bg = [p for d in cfg.augmentation.background_paths
          for p in Path(d).glob("**/*.wav")]
    return {
        "target_phrases": cfg.target_phrases,
        "custom_negative_phrases": cfg.custom_negative_phrases,
        "n_samples": cfg.n_samples,
        "steps": cfg.steps,
        "n_background_samples": cfg.n_background_samples,
        "rounds": cfg.augmentation.rounds,
        "snr_db_range": list(snr),
        "clean_fraction": clean_p,
        "background_wavs": len(bg),
        "by_folder": {d: sum(1 for p in bg if p.parent.name == d)
                      for d in sorted({p.parent.name for p in bg})},
    }


def param_count(pt_path):
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    return sum(t.numel() for t in sd.values() if hasattr(t, "numel"))


def keep(src, dest):
    if Path(src).exists():
        shutil.copy2(src, dest)
        return Path(dest).name
    return None


def run_data_stages(cfg, first):
    """generate/augment/features, skipping everything before `first`.

    These are shared by every size - generate and augment key their output on
    model_name, not on model_size - so they run ONCE and all sizes train
    against identical data. That is also what makes the size comparison mean
    anything.

    Slowest first: generating 25k phrases is tens of minutes of TTS and is the
    stage worth resuming into. run_generate counts existing clips and picks up
    where it stopped, so --from is for skipping deliberately, not for crash
    recovery."""
    order = STAGES.index(first)
    if order <= STAGES.index("generate"):
        print("\n=== generate: TTS positives + adversarial negatives ===", flush=True)
        run_generate(cfg)
    if order <= STAGES.index("augment"):
        print("\n=== augment: RIR + background at the patched SNR ===", flush=True)
        run_augment(cfg)
    if order <= STAGES.index("features"):
        print("\n=== features: mel -> speech_embedding -> .npy ===", flush=True)
        run_extraction(cfg)


def run_size(size, root, results, artifacts):
    cfg = load_config(root, size)
    stem = f"{cfg.model_name}_{size}_{VERSION}"
    print(f"\n=== {size}: train {cfg.steps} steps ===", flush=True)

    t0 = time.time()
    pt_path = run_train(cfg)
    onnx_path = run_export(cfg)
    metrics = run_eval(cfg, onnx_path)
    minutes = round((time.time() - t0) / 60, 1)

    # TWO operating points, and confusing them is a deployment bug.
    # evaluate.py hardcodes threshold=0.5 "for consistent comparison", so the
    # *_at_half fields rank the sizes fairly against each other but are NOT
    # what you would ship. find_best_threshold separately maximises recall
    # subject to fpph <= target_fp_per_hour. Neither is a K15 threshold: the
    # parity gate (2026-08-13) showed livekit's scores do not transfer to
    # openWakeWord's streaming runtime, so the DEPLOYED number comes from
    # --wake-trials peaks on the K15 and nowhere else.
    row = {
        "params": param_count(pt_path),
        "onnx_kb": round(onnx_path.stat().st_size / 1024),
        "aut": round(metrics["aut"], 4),
        "fpph_at_half": round(metrics["fpph"], 3),
        "recall_at_half": round(metrics["recall"], 4),
        "accuracy_at_half": round(metrics["accuracy"], 4),
        "optimal_threshold": round(metrics["optimal_threshold"], 3),
        "optimal_recall": round(metrics["optimal_recall"], 4),
        "optimal_fpph": round(metrics["optimal_fpph"], 3),
        "validation_hours": metrics["validation_hours"],
        "train_min": minutes,
        # export always writes output/<model_name>/<model_name>.onnx and eval
        # always overwrites <model_name>_det.png, so a second size silently
        # destroys the first one's artifacts. Copy out before the next run.
        "onnx": keep(onnx_path, artifacts / f"{stem}.onnx"),
        "pt": keep(pt_path, artifacts / f"{stem}.pt"),
        "det_png": keep(cfg.model_output_dir / f"{cfg.model_name}_det.png",
                        artifacts / f"{stem}_det.png"),
    }
    results[size] = row
    print(f"  {size}: {row['optimal_recall']:.1%} recall at "
          f"{row['optimal_fpph']} FP/hr over {row['validation_hours']:.1f} h "
          f"({minutes} min)", flush=True)


def table(results, root):
    print(f"\n{'':29}{'--- at fixed 0.5 ---':>26}{'--- tuned ---':>26}")
    print(f"{'size':10}{'params':>9}{'onnx':>10}{'AUT':>9}{'FPPH':>8}"
          f"{'recall':>9}{'thresh':>9}{'recall':>9}{'FPPH':>8}{'train':>8}")
    print("-" * 89)
    for size, r in results.items():
        if size.startswith("_"):
            continue
        if "error" in r:
            print(f"{size:10}FAILED - {r['error'][:60]}")
            continue
        print(f"{size:10}{r['params']/1000:>8.1f}k{r['onnx_kb']:>7} KB"
              f"{r['aut']:>9.4f}{r['fpph_at_half']:>8.2f}"
              f"{r['recall_at_half']:>9.1%}{r['optimal_threshold']:>9.2f}"
              f"{r['optimal_recall']:>9.1%}{r['optimal_fpph']:>8.2f}"
              f"{r['train_min']:>7.1f}m")
    hrs = next((r.get("validation_hours") for r in results.values()
                if isinstance(r, dict) and "validation_hours" in r), None)
    print(f"\nFPPH is measured over {hrs} h of validation negatives. If "
          f"make_validation.py\nhas run, that is YOUR room and the number "
          f"means something; if it has not, it is\nlivekit's synthetic set, "
          f"which could not tell three sizes apart in 2026-08-15's\nsweep "
          f"(all three: 3 FPs, ~99.3% recall, AUT 0.0000).")
    print("\nNEITHER threshold column ships. The parity gate showed livekit's "
          "scores do\nnot transfer to openWakeWord's runtime - set "
          "voice.wakeThreshold from the peak\nvalues that --wake-trials logs "
          "on the K15.")
    print(f"\nartifacts: {root / 'artifacts'}")
    # The size is in the artifact name so several can sit side by side; the
    # DEPLOYED name must not carry it. audio.py derives the spoken phrase with
    # rsplit("_v", 1)[0].replace("_", " "), so hey_alfred_medium_v1.1.onnx
    # would have the agent listening for "hey alfred medium".
    print("\nTo deploy one, copy it to k15/voice/models/ RENAMED - the size "
          "must come out\nof the filename or it becomes part of the wake "
          "phrase:")
    for size in results:
        if size.startswith("_") or "error" in results[size]:
            continue
        print(f"    {results[size]['onnx']}  ->  hey_alfred_{VERSION}.onnx")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sizes", nargs="*", default=None, help=f"default: {SIZES}")
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\tillm\wake"),
                    help="where data/, output/ and the venv live (not the repo)")
    ap.add_argument("--from", dest="first", choices=STAGES, default="generate",
                    help="skip stages before this one")
    ap.add_argument("--snr", nargs=2, type=float, default=(0.0, 20.0),
                    metavar=("LO", "HI"), help="background mix range in dB")
    ap.add_argument("--clean", type=float, default=0.25,
                    help="fraction of clips left un-mixed")
    ap.add_argument("--list", action="store_true", help="show state and exit")
    args = ap.parse_args()

    sizes = args.sizes or SIZES
    root = args.root.resolve()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    results_path = root / "pipeline_results.json"

    cfg0 = load_config(root)
    val = Path(cfg0.data_path) / "features" / "validation_set_features.npy"
    print(f"root       {root}")
    print(f"config     {CONFIG}")
    print(f"validation {val.name}: "
          f"{'MISSING' if not val.exists() else f'{val.stat().st_size/1e6:.0f} MB'}"
          f"  <- make_validation.py replaces this with your room")
    print(f"sizes      {sizes}   stages from '{args.first}'")
    if args.list:
        return 0

    patch_augmentation(args.snr[0], args.snr[1], args.clean)
    run_data_stages(cfg0, args.first)

    # Appended after EACH size and a finished size is skipped on a re-run: a
    # multi-size sweep is an hour or more, and a crash in the last one must not
    # cost the ones that already succeeded. Delete the file to force a retrain.
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    for size in sizes:
        if size in results and "error" not in results[size]:
            print(f"=== {size}: already in {results_path.name}, skipping ===")
            continue
        try:
            run_size(size, root, results, artifacts)
        except Exception:
            # A failed size is recorded and the sweep continues - an unattended
            # run should come back with results and one error, not one error.
            traceback.print_exc()
            results[size] = {"error": traceback.format_exc(limit=1).strip()}
            print(f"  {size} FAILED - continuing", flush=True)
        results["_run"] = provenance(load_config(root, size), args.snr, args.clean)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table(results, root)
    print(f"results:   {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
