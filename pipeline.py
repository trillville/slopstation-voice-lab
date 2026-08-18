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
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
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
# Bump this for every retrain whose DATA changed - two files with the same
# version and different weights is the confusion that costs an evening, and
# stale same-version rows in pipeline_results.json get skipped as already
# done. v1.0 = the original sweep; v1.1 = SNR fix, no game audio; v1.2 = game
# backgrounds + fresh held-out negatives (recorded 2026-08-16); v1.3 = the
# clean fraction stops being digital silence (patch_pad_then_mix's third
# bullet), which is a DIFFERENT pad-then-mix recipe from the four v1.2 models
# left on disk by 2026-08-17's aborted sweep - two recipes sharing one version
# is exactly the confusion this constant exists to prevent.
VERSION = "v1.3"

STAGES = ["generate", "augment", "features", "train"]

# What a VARIANT may override. Everything here affects only training, so every
# variant trains against byte-identical data and a comparison between them
# means something.
#
# Anything NOT in this set changes what generate/augment/features produce - and
# since those stages run ONCE and are shared, a variant touching them would be
# silently trained on the previous variant's data. That is the same class of
# mistake as the +5..+15 dB SNR: not wrong loudly, wrong quietly. To sweep one
# of those, edit alfred.yaml and run the whole pipeline again.
TRAINING_ONLY = {"model", "steps", "learning_rate", "weight_decay",
                 "label_smoothing", "max_negative_weight", "target_fp_per_hour",
                 "batch_n_per_class"}


def check_not_compounding(lo, rounds):
    """Refuse a widened SNR stacked on multiple augmentation rounds.

    Round N mixes fresh noise into round N-1's OUTPUT, so the effective SNR
    compounds: three rounds drawing from a range whose floor is 0 dB lands
    somewhere near -18 dB, which is not "robust to noise", it is a model that
    has learned to fire on noise. livekit's own prod.yaml uses rounds: 3 and
    gets away with it only because its floor is +5 dB.

    A guard rather than a comment because the comment already existed and the
    failure costs a three-hour run to discover."""
    if rounds > 1 and lo < 5.0:
        sys.exit(
            f"refusing: snr floor {lo:+g} dB with augmentation.rounds={rounds}.\n"
            f"Rounds COMPOUND - each one re-mixes the previous round's output, "
            f"so {rounds} rounds from a {lo:+g} dB floor lands far below it and "
            f"teaches the model to fire on noise.\n"
            f"Pick one: keep rounds at 1 (what alfred.yaml ships), or raise "
            f"--snr's floor to +5 or above.")


def patch_rir(p):
    """Override the reverberation probability livekit hardcodes at its call
    site. apply_rir(audio) is invoked with no p, so the only way to change it
    is here. 270 impulse responses are on disk; an empty rir_files list would
    make this a silent no-op, so the count is printed rather than assumed."""
    from livekit.wakeword.data import augment as _aug
    original = _aug.AudioAugmentor.apply_rir

    def with_p(self, audio, p_override=p):
        return original(self, audio, p_override)

    _aug.AudioAugmentor.apply_rir = with_p


def patch_seed(seed):
    """Pin every RNG the training run touches.

    livekit exposes no seed - not in config.py, not in trainer.py - so every
    run so far has been a random draw. That is not a detail: two models trained
    on BYTE-IDENTICAL features with the same recipe and different seeds scored
    clean 64% vs 86%, noise ceiling 0.678 vs 0.220, and +10 dB recall 14% vs
    37% (medium@snr-neg10 vs ab_mixupON_s1234, 2026-08-17). Seed variance is
    larger than every effect this project has chased, so every single-run
    comparison in the results file is unresolved until it is re-run across
    seeds. Sweep >=3 per arm and compare distributions, never point estimates."""
    import random as _random

    import numpy as _np
    import torch
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_pad_then_mix(lo, hi, clean_p, target_len, jitter=3200):
    """Build the 2 s window FROM BACKGROUND, then write the clip into it.

    THE DEFECT THIS FIXES. Upstream mixes background into the ~0.9 s clip and
    only then drops it into a buffer of np.zeros (align_clip_to_end for
    positives, centre-pad for negatives). Measured on the shipped data:
    positives carry 912 ms of leading DIGITAL SILENCE (51.8% exact zeros),
    adversarial negatives 412 ms / 41%, background negatives 0 ms / 0%, and
    ACAV100M - 87% of every batch - is precomputed continuous features. So
    "quiet lead-in, then energy at the end of the window" is very nearly a free
    discriminant for POSITIVE, and openWakeWord's 80 ms-hop stream never
    provides it: the buffer always holds real room audio.

    Causally confirmed on the deployed model, phrase audio byte-identical and
    only the lead-in zeroed: +20 dB 74%->90%, +10 dB 38%->82%, 0 dB 2%->42%.
    (Zeroing also raises false accepts ~5.8x, so FA-matched the artifact is
    worth ~40-60 points rather than the raw delta - still the largest single
    term found.)

    It explains why four generations barely moved: every one changed what the
    noise SOUNDED like, none changed the geometry of the window it occupied.

    Three details that make or break the fix:
      * SNR is referenced to the CLIP's power, not the padded window's.
        Upstream's np.mean(audio**2) over a half-zero window understates speech
        power and mixes noise ~3 dB hot; measuring on the clip keeps the
        requested SNR honest.
      * Negatives get the same treatment. Fixing only positives would invert
        the shortcut into "zeros mean NEGATIVE", which is the same bug wearing
        a different sign.
      * The clean fraction is background pushed 35 dB DOWN, not removed.
        Returning the bare padded window for clean_p of the clips reinstates
        the shortcut on that fraction, and it is not a small residue: measured
        on 2026-08-17's arm-B data, 20.7% of positives and 24.7% of adversarial
        negatives still carried a ~0.9 s digital-silence lead-in against 0% of
        background negatives - and ACAV100M, 87% of every batch, is continuous
        features. So zeros still marked a window as TTS-derived. A quiet room
        is room TONE, not zeros."""
    import soundfile as sf
    from livekit.wakeword.data import augment as _aug

    CLEAN_SNR_DB = 35.0

    def bg_window(self, n):
        bg, _ = sf.read(str(random.choice(self.background_files)))
        if bg.ndim > 1:
            bg = bg[:, 0]
        bg = bg.astype(np.float32)
        if len(bg) < n:
            bg = np.tile(bg, (n // len(bg)) + 1)
        s = random.randint(0, max(0, len(bg) - n))
        return bg[s:s + n]

    def padded_mix(self, audio, snr_db_range=(lo, hi)):
        if not self.background_files:
            return audio
        # End-aligned with the same 0-200 ms jitter upstream used, so the
        # phrase still lands at varying offsets - only the FILL changes.
        win = np.zeros(target_len, dtype=np.float32)
        end = target_len - random.randint(0, jitter)
        start = max(0, end - len(audio))
        win[start:end] = audio[max(0, len(audio) - (end - start)):][:end - start]
        # The clean fraction is a HIGH SNR, not a bypass - see the docstring's
        # third bullet. Returning `win` here is the shortcut, rebuilt.
        snr_db = (CLEAN_SNR_DB if clean_p and random.random() < clean_p
                  else random.uniform(*snr_db_range))
        bg = bg_window(self, target_len)
        clip_power = float(np.mean(audio ** 2)) + 1e-8      # NOT the window
        bg_power = float(np.mean(bg ** 2)) + 1e-8
        scale = np.sqrt(clip_power / (bg_power * 10 ** (snr_db / 10)))
        return (win + scale * bg).astype(np.float32)

    # Everything downstream already has the right length, so the zero-padding
    # steps must become no-ops or they would re-introduce silence at the tail
    # (align_clip_to_end crops by the jitter and zero-fills the remainder).
    original_align = _aug.align_clip_to_end

    def align_noop(audio, target_length, **kw):
        return audio if len(audio) == target_length else original_align(
            audio, target_length, **kw)

    _aug.AudioAugmentor.mix_with_background = padded_mix
    _aug.align_clip_to_end = align_noop
    print(f"[patch] pad-then-mix: {target_len / 16000:.1f}s window built from "
          f"background, clip written into it (no zero pad), "
          f"{clean_p:.0%} at +{CLEAN_SNR_DB:g} dB instead of clean", flush=True)


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
    import inspect

    from livekit.wakeword.data import augment as _aug
    original = _aug.AudioAugmentor.mix_with_background
    # A monkeypatch rots silently: if upstream renames the parameter, the
    # override below would still apply and quietly mix at whatever the new
    # default is - the exact class of failure this file exists to prevent.
    # Verified against 0.2.1, the latest on PyPI as of 2026-08-16.
    if "snr_db_range" not in inspect.signature(original).parameters:
        sys.exit("livekit's mix_with_background no longer takes snr_db_range - "
                 "the SNR patch would silently do nothing. Re-read "
                 "data/augment.py and update patch_augmentation before "
                 "training anything.")

    def widened(self, audio, snr_db_range=(lo, hi)):
        if clean_p and random.random() < clean_p:
            return audio                # the clean quarter, oWW-style
        return original(self, audio, snr_db_range)

    _aug.AudioAugmentor.mix_with_background = widened
    print(f"[patch] background SNR {lo:+g}..{hi:+g} dB, {clean_p:.0%} left clean "
          f"(livekit default: +5..+15 dB, 0% clean)", flush=True)


def merge(base, over):
    """Recursive dict overlay, so a variant can set model.model_size without
    also having to restate model_type."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(
            out.get(k), dict) else v
    return out


def variant_specs():
    """{name: overrides} from alfred.yaml, refusing any that reach past
    training into the shared data stages."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    specs = raw.get("variants") or {}
    for name, over in specs.items():
        bad = set(over) - TRAINING_ONLY
        if bad:
            sys.exit(
                f"variant '{name}' overrides {sorted(bad)}, which change what "
                f"generate/augment/features produce.\nThose stages run ONCE "
                f"and are shared by every variant, so this one would train on "
                f"another\nvariant's data and the comparison would be a lie. "
                f"Move it into the base config\nand re-run the whole pipeline "
                f"instead. Overridable: {sorted(TRAINING_ONLY)}")
    return specs


def load_config(root, size=None, over=None):
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
    raw.pop("variants", None)               # ours, not livekit's schema
    if over:
        raw = merge(raw, over)
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


def tool_versions():
    """Versions and the code commit, stamped into every result row. The
    monkeypatch makes results a function of THIS code, not just the config -
    a row that cannot say which pipeline produced it is unreproducible."""
    import importlib.metadata as im
    out = {}
    for pkg in ("livekit-wakeword", "openwakeword", "torch"):
        try:
            out[pkg] = im.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short",
                            "HEAD"], capture_output=True, text=True, timeout=10)
        out["pipeline_commit"] = r.stdout.strip() or None
    except Exception:
        out["pipeline_commit"] = None
    return out


def param_count(pt_path):
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    return sum(t.numel() for t in sd.values() if hasattr(t, "numel"))


def keep(src, dest):
    if Path(src).exists():
        shutil.copy2(src, dest)
        return Path(dest).name
    return None


def data_stamp_path(cfg):
    return Path(cfg.model_output_dir) / "data_settings.json"


def write_data_stamp(cfg, prov):
    """Record, NEXT TO THE FEATURES, the settings that actually produced them.

    Provenance used to stamp args.snr - what was ASKED for on this invocation,
    not what the .npy files on disk contain. Those differ the moment --from
    train skips augment, and on 2026-08-16 that silently trained medium-400k
    and dnn-medium on a -10..+15 dataset left behind by an earlier run while
    the results file recorded [0, 20]. Two void runs, no signal in the logs.
    The stamp travels with the data, so a later run reads the truth instead of
    re-asserting its own arguments."""
    try:
        p = data_stamp_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[warn] could not write {p}: {e}")


def read_data_stamp(cfg):
    """What the features on disk were actually built with, or None."""
    try:
        return json.loads(data_stamp_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
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


def artifact_stem(model_name, name, tag, seed):
    """What this run will call its model - and therefore the identity a re-run
    has to check before believing a finished row.

    The seed is IN THE FILENAME. It was not on 2026-08-17, and three seeds per
    cell silently overwrote one another - nine hours of GPU left one model per
    cell. patch_seed was defined but never called in the same botched edit, so
    those runs were not even reproducible draws. Both failures came from a grep
    that matched 2 of 3 patterns being read as success; verify each edit
    separately, and compile the file.

    One home because main() must predict this name before run_variant builds
    it: the skip guard compares it against what is actually on disk."""
    return f"{model_name}_{name}{f'_{tag}' if tag else ''}_s{seed}_{VERSION}"


def run_variant(name, over, root, results, artifacts, key, stem, prov, seed):
    # A name with no entry in `variants` is taken as a bare model_size, so
    # `Train.bat large` still works without the yaml having to list it.
    cfg = (load_config(root, over=over) if over is not None
           else load_config(root, size=name))
    print(f"\n=== {key}: {cfg.model.model_type}/{cfg.model.model_size}, "
          f"{cfg.steps} steps, seed {seed} ===", flush=True)

    # Per VARIANT, not once per process: run_train advances the global RNG, so
    # seeding only at startup would leave the second variant in a sweep drawing
    # from a different state, and the arms would not be paired.
    patch_seed(seed)
    t0 = time.time()
    pt_path = run_train(cfg)
    onnx_path = run_export(cfg)
    # The eval is a SMOKE TEST (the saturation note in table has the history):
    # a crashed smoke test must not discard the training run it follows.
    try:
        metrics, eval_err = run_eval(cfg, onnx_path), None
    except Exception as e:
        metrics, eval_err = None, f"{type(e).__name__}: {e}"
    minutes = round((time.time() - t0) / 60, 1)

    row = {
        "params": param_count(pt_path),
        "onnx_kb": round(onnx_path.stat().st_size / 1024),
        "train_min": minutes,
        # export always writes output/<model_name>/<model_name>.onnx and eval
        # always overwrites <model_name>_det.png, so a second size silently
        # destroys the first one's artifacts. Copy out before the next run.
        "onnx": keep(onnx_path, artifacts / f"{stem}.onnx"),
        "pt": keep(pt_path, artifacts / f"{stem}.pt"),
        "det_png": keep(cfg.model_output_dir / f"{cfg.model_name}_det.png",
                        artifacts / f"{stem}_det.png"),
        # The settings THIS row was measured under. The global _run blob goes
        # stale the moment a --tag re-run mixes two SNR settings into one
        # results file; the row is the only record that stays true.
        "data": dict(prov),
    }
    if metrics is None:
        row["eval_error"] = eval_err
        results[key] = row
        print(f"  {key}: trained + exported, EVAL FAILED - artifacts kept "
              f"({eval_err})", flush=True)
        return

    # TWO operating points, and confusing them is a deployment bug.
    # evaluate.py hardcodes threshold=0.5 "for consistent comparison", so the
    # *_at_half fields rank the sizes fairly against each other but are NOT
    # what you would ship. find_best_threshold separately maximises recall
    # subject to fpph <= target_fp_per_hour. Neither is a K15 threshold: the
    # parity gate (2026-08-13) showed livekit's scores do not transfer to
    # openWakeWord's streaming runtime, so the DEPLOYED number comes from
    # --wake-trials peaks on the K15 and nowhere else.
    row.update(
        aut=round(metrics["aut"], 4),
        fpph_at_half=round(metrics["fpph"], 3),
        recall_at_half=round(metrics["recall"], 4),
        accuracy_at_half=round(metrics["accuracy"], 4),
        optimal_threshold=round(metrics["optimal_threshold"], 3),
        optimal_recall=round(metrics["optimal_recall"], 4),
        optimal_fpph=round(metrics["optimal_fpph"], 3),
        validation_hours=metrics["validation_hours"],
    )
    results[key] = row
    print(f"  {key}: {row['optimal_recall']:.1%} recall at "
          f"{row['optimal_fpph']} FP/hr over {row['validation_hours']:.1f} h "
          f"({minutes} min)", flush=True)


def table(results, root, target_fpph):
    print(f"\n{'':37}{'--- at fixed 0.5 ---':>26}{'--- tuned ---':>26}")
    print(f"{'variant':18}{'params':>9}{'onnx':>10}{'AUT':>9}{'FPPH':>8}"
          f"{'recall':>9}{'thresh':>9}{'recall':>9}{'FPPH':>8}{'train':>8}")
    print("-" * 97)
    for key, r in results.items():
        if key.startswith("_"):
            continue
        if "error" in r:
            print(f"{key:18}FAILED - {r['error'][:60]}")
            continue
        if "eval_error" in r:
            print(f"{key:18}trained OK, eval FAILED (artifacts kept) - "
                  f"{r['eval_error'][:40]}")
            continue
        print(f"{key:18}{r['params']/1000:>8.1f}k{r['onnx_kb']:>7} KB"
              f"{r['aut']:>9.4f}{r['fpph_at_half']:>8.2f}"
              f"{r['recall_at_half']:>9.1%}{r['optimal_threshold']:>9.2f}"
              f"{r['optimal_recall']:>9.1%}{r['optimal_fpph']:>8.2f}"
              f"{r['train_min']:>7.1f}m")
    # The banner that should have been here from the start: on both sweeps so
    # far EVERY candidate landed on the same tuned numbers with a DET curve
    # pinned flat against the axes, and each time the table was read as a
    # result instead of as a saturated test. Detect it and say it.
    scored = {k: r for k, r in results.items() if not k.startswith("_")
              and "error" not in r and "eval_error" not in r}
    live = list(scored.values())
    # find_best_threshold falls back to MAX BALANCED ACCURACY when no threshold
    # fits target_fp_per_hour, and says nothing about having done so - it just
    # prints a threshold and a recall like any other row. 2026-08-17's
    # pad-then-mix arm was the first family ever to miss the budget (16.3 and
    # 13.4 FP/hr against a 0.2 target) and its "98.4% recall" sat in the table
    # one line under a real 99.8%, inviting exactly the wrong read.
    missed = [k for k, r in scored.items() if r["optimal_fpph"] > target_fpph]
    if missed:
        print(f"\n*** {', '.join(missed)}\n    NO threshold met the "
              f"{target_fpph} FP/hr budget, so those tuned columns are "
              f"livekit's\n    fallback (max balanced accuracy), not an "
              f"operating point. Read them as\n    'did not fit', not as "
              f"recall.")
    # Rows are only comparable if they were graded on the same test set, and
    # positive_features_test comes from the SAME augmentation as training. So a
    # pad-then-mix row is scored on realistic windows while a row beside it is
    # scored on windows that still carry the silence shortcut. Their recalls
    # are two different measurements printed in one column.
    if len({bool((r.get("data") or {}).get("pad_then_mix"))
            for r in scored.values()}) > 1:
        print("\n*** MIXED TEST SETS: pad-then-mix rows are graded on "
              "realistic windows, the\n    others on windows carrying the "
              "silence shortcut - positive_features_test is\n    built by "
              "whichever augmentation trained the model. Recall is NOT "
              "comparable\n    across that line. Bench.bat is; it scores every "
              "model on the same real audio.")
    if len(live) >= 2 and (
            max(r["optimal_recall"] for r in live)
            - min(r["optimal_recall"] for r in live) < 0.005
            and max(r["optimal_fpph"] for r in live)
            - min(r["optimal_fpph"] for r in live) < 0.02):
        print("\nSATURATED: every variant landed within 0.5% recall and 0.02 "
              "FPPH of the rest.\nThis table cannot rank them - it is a smoke "
              "test that they all passed. The\nranking, if there is one, "
              "comes from Bench.bat.")
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
    # This table cannot pick a winner and never could. On 2026-08-16 both
    # surviving sizes landed on threshold 0.18 with ~99.8% recall and DET
    # curves pinned flat against both axes - a saturated test, which ranks
    # nothing. Bench.bat is the eval that does, on real voice through the
    # runtime that will actually run the model.
    print("\nNEXT: Bench.bat. Nothing above ranks these candidates - the "
          "synthetic eval is\nsaturated (both sizes ace it). bench_real.py "
          "scores them on your voice, in your\nroom, under openWakeWord, and "
          "that ranking is the one that has ever been right.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("variants", nargs="*", default=None,
                    help="names from alfred.yaml's `variants:` (default: all). "
                         "A name with no entry is taken as a bare model_size.")
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\tillm\wake"),
                    help="where data/, output/ and the venv live (not the repo)")
    ap.add_argument("--from", dest="first", choices=STAGES, default="generate",
                    help="skip stages before this one")
    # (0, 20). NOT openWakeWord's (-10, 15), and that is a MEASURED choice -
    # do not "fix" it back without re-reading this.
    #
    # Copying their floor was tried on 2026-08-16 as a clean single-variable
    # A/B (medium@snr-neg10 vs medium, identical data). It was refuted, badly:
    #
    #     noise ceiling  0.103 -> 0.678   (6x worse)
    #     separation     5.34x -> 1.07x   (essentially none left)
    #     threshold       0.10 -> 0.62    (forced up to clear the noise)
    #     recall @+10 dB   23% -> 14%     (worse on the axis it targeted)
    #
    # Why it works for them and not for us: openWakeWord pairs that floor with
    # ~31,000 h of negatives; we have ~2,000 h. Buried under a -10 dB mix the
    # phrase is barely there, so what the model can still learn is "noisy =
    # maybe wake word" - and only a negative corpus an order of magnitude
    # bigger teaches it otherwise. The floor is not portable without the
    # corpus that makes it safe.
    ap.add_argument("--snr", nargs=2, type=float, default=(0.0, 20.0),
                    metavar=("LO", "HI"), help="background mix range in dB")
    ap.add_argument("--clean", type=float, default=0.25,
                    help="fraction of clips left un-mixed")
    # Far-field IS the deployment condition - the talker is metres from the
    # mic across a reverberant room - so reverberation is not a corner case to
    # sprinkle on. livekit's default is 0.5; a knob so it can be swept rather
    # than guessed at, since nothing has measured its effect yet.
    ap.add_argument("--rir", type=float, default=0.5,
                    help="probability a clip is convolved with a room impulse "
                         "response (livekit default 0.5)")
    ap.add_argument("--tag", default="",
                    help="suffix for result keys and artifact names. Use when "
                         "re-running with DIFFERENT data settings (an --snr "
                         "A/B): without it the second run's artifacts "
                         "overwrite the first's and its finished rows are "
                         "skipped as already done")
    ap.add_argument("--seed", type=int, default=0,
                    help="training RNG seed. Seed variance is LARGER than "
                         "every effect measured here (64%% vs 86%% clean on "
                         "identical data), so sweep >=3 per arm.")
    ap.add_argument("--pad-then-mix", action="store_true",
                    help="build the training window from background instead of "
                         "zeros - removes the silence shortcut (see "
                         "patch_pad_then_mix). Changes the DATA: needs "
                         "--from augment.")
    ap.add_argument("--no-bench", action="store_true",
                    help="skip the real-audio bench that normally closes a run")
    ap.add_argument("--list", action="store_true", help="show state and exit")
    args = ap.parse_args()
    if args.tag and not all(c.isalnum() or c in "._-" for c in args.tag):
        sys.exit("--tag must be filename-safe: letters, digits, . _ -")

    specs = variant_specs()
    names = args.variants or list(specs) or ["medium"]
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
    print(f"variants   {names}   stages from '{args.first}'")
    print(f"augment    snr {args.snr[0]:+g}..{args.snr[1]:+g} dB, "
          f"{args.clean:.0%} clean, rir p={args.rir}, "
          f"rounds {cfg0.augmentation.rounds}")
    # What is ALREADY trained, and under what. An A/B is only an A/B if the
    # two rows differ, and on 2026-08-16 a `--tag snr-neg10` run on a checkout
    # three commits stale trained at the OLD (0, 20) range and produced a
    # perfect duplicate of `medium` wearing a label that said otherwise. The
    # per-row provenance caught it afterwards; showing it here catches it
    # before the hour is spent.
    if results_path.exists():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        rows = [(k, v.get("data") or {}) for k, v in prior.items()
                if not k.startswith("_")]
        if rows:
            print("\nalready trained:")
            for k, dat in rows:
                snr, pad = dat.get("snr_db_range"), dat.get("pad_then_mix")
                # "SAME" must mean the same DATA, not the same SNR. Every
                # pad-then-mix row carries an identical snr to its control, so
                # on 2026-08-17 this line marked the treatment arm SAME as a
                # run asking for the opposite window geometry.
                same = (" <- SAME as this run" if
                        (snr, bool(pad)) == (list(args.snr),
                                             bool(args.pad_then_mix)) else "")
                print(f"  {k:22} snr={snr} clean={dat.get('clean_fraction')} "
                      f"rir={dat.get('rir_p')} pad={bool(pad)} "
                      f"commit={dat.get('pipeline_commit')}{same}")
    # What the models will ACTUALLY train on. Before the --list return, because
    # this is the line that would have saved two runs: --from train reuses
    # whatever augment last wrote, which on 2026-08-16 was a -10..+15 dataset
    # from an earlier experiment, while the results file said [0, 20].
    asked = {"snr_db_range": list(args.snr), "clean_fraction": args.clean,
             "rir_p": args.rir, "rounds": cfg0.augmentation.rounds,
             "pad_then_mix": bool(args.pad_then_mix)}
    rebuilding = STAGES.index(args.first) <= STAGES.index("augment")
    on_disk = read_data_stamp(cfg0)
    prov = dict(asked) if rebuilding else {**(on_disk or asked),
                                           "data_stamp": bool(on_disk)}
    prov.update(tool_versions())
    # EVERY data setting, not just the SNR. Comparing snr alone was the check
    # until 2026-08-17, and it would have waved through the one that matters
    # most in the experiment running that day: `--pad-then-mix --from train`
    # over features built WITHOUT it prints "matches this run" and trains the
    # treatment arm on the control arm's data. Keys absent from an older stamp
    # are left alone rather than reported as mismatches.
    differs = {k: (on_disk[k], v) for k, v in asked.items()
               if on_disk and k in on_disk and on_disk[k] != v}
    if not rebuilding:
        if differs:
            print(f"\n*** STALE DATA: the features on disk were NOT built the "
                  f"way this run asks.\n    --from {args.first} does not "
                  f"rebuild them, so that is what these models\n    will train "
                  f"on; the rows are stamped with the real values. Use\n"
                  f"    --from augment to change it.", flush=True)
            for k, (was, want) in differs.items():
                print(f"      {k}: on disk {was!r}, this run asked {want!r}",
                      flush=True)
            print(flush=True)
        elif not on_disk:
            print(f"\n[warn] no data_settings.json beside the features: they "
                  f"predate this stamping,\n       so provenance falls back "
                  f"to the arguments and may be wrong. --from augment\n"
                  f"       rebuilds and records the truth.\n", flush=True)
        else:
            print(f"data       features built with snr="
                  f"{on_disk['snr_db_range']}, pad_then_mix="
                  f"{on_disk.get('pad_then_mix')} (matches this run)")
    if args.list:
        return 0

    check_not_compounding(args.snr[0], cfg0.augmentation.rounds)
    if args.pad_then_mix:
        patch_pad_then_mix(args.snr[0], args.snr[1], args.clean,
                           int(cfg0.augmentation.clip_duration * 16000))
    else:
        patch_augmentation(args.snr[0], args.snr[1], args.clean)
    n_rir = sum(len(list(Path(d).glob("**/*.wav")))
                for d in cfg0.augmentation.rir_paths)
    if not n_rir:
        sys.exit(f"no impulse responses under {cfg0.augmentation.rir_paths} - "
                 f"apply_rir would silently no-op and every positive would "
                 f"train anechoic, which is the opposite of far-field.")
    patch_rir(args.rir)
    print(f"[patch] reverberation p={args.rir} over {n_rir} impulse responses",
          flush=True)
    run_data_stages(cfg0, args.first)

    # Appended after EACH size and a finished size is skipped on a re-run: a
    # multi-size sweep is an hour or more, and a crash in the last one must not
    # cost the ones that already succeeded. Delete the file to force a retrain.
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    if rebuilding:
        # The data stages ran, so the arguments ARE the truth now.
        write_data_stamp(cfg0, asked)
    for name in names:
        # The seed is ALWAYS part of the identity. Two rows differing only by
        # seed are two samples of the same arm, not two arms - and with seed
        # variance this large, a key that hides it invites exactly the
        # single-run conclusions that have had to be retracted twice.
        suffix = f"{args.tag}-" if args.tag else ""
        key = f"{name}@{suffix}s{args.seed}"
        stem = artifact_stem(cfg0.model_name, name, args.tag, args.seed)
        # A row counts as done only if its MODEL is on disk under the name this
        # run would write. The 2026-08-17 sweep left twelve finished-looking
        # rows whose artifacts had all overwritten one another; re-running it
        # would have skipped every one as already done and then benched the
        # files that were never written - nine hours of no-op that reads
        # exactly like success. Believe the artifact, not the row.
        done = results.get(key)
        if done and "error" not in done:
            if done.get("onnx") == f"{stem}.onnx" and (
                    artifacts / f"{stem}.onnx").is_file():
                print(f"=== {key}: already in {results_path.name}, skipping ===")
                continue
            print(f"=== {key}: row exists but {stem}.onnx does not - "
                  f"retraining ===")
        try:
            run_variant(name, specs.get(name), root, results, artifacts,
                        key, stem, prov, args.seed)
        except Exception:
            # A failed variant is recorded and the sweep continues - an
            # unattended run should come back with results and one error, not
            # one error.
            traceback.print_exc()
            results[key] = {"error": traceback.format_exc(limit=1).strip()}
            print(f"  {key} FAILED - continuing", flush=True)
        results["_run"] = provenance(load_config(root), args.snr, args.clean)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table(results, root, cfg0.target_fp_per_hour)
    print(f"results:   {results_path}")

    # The real eval, run automatically, so the LAST thing on screen is the
    # number that predicts the couch rather than the one that flatters it.
    #
    # livekit's eval cannot be repaired into a ranking. positive_train and
    # positive_test come from ONE tts.synthesize_clips call with identical
    # arguments - same phrases, same voice pool, no speaker holdout - and then
    # through the same augmentation at the same SNR. It is an i.i.d. resample
    # of the training distribution, so it measures "did training converge",
    # which every candidate passes with a DET curve flat against the axes.
    # Four generations were judged on it before anyone measured real audio.
    if not args.no_bench:
        print("\n" + "=" * 79)
        print("REAL-AUDIO BENCH - your voice, your room, openWakeWord's runtime")
        print("=" * 79, flush=True)
        r = subprocess.run([sys.executable, str(HERE / "bench_real.py"),
                            "--root", str(root), "--snr-sweep"])
        if r.returncode:
            # Missing positives or negatives is the usual cause and is not a
            # training failure - the artifacts are on disk either way.
            print("\n[bench] did not run. The models are still in "
                  f"{artifacts}; fix the bench inputs and run Bench.bat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
