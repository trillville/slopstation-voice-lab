"""One command from a config to a vendored wake model. Gaming PC only.

    Train.bat                          every size in SIZES, from scratch
    Train.bat medium                   just medium
    Train.bat medium --from train      reuse the clips already on disk
    Train.bat --list                   what is on disk, and what would run

Runs generate -> augment -> features -> train -> export -> eval through
livekit-wakeword's run_* functions, not its CLI: the CLI cannot express the SNR
fix (livekit hardcodes background mixing at +5..+15 dB via augment.py's
`snr_db_range` default), and run_eval returns its metrics dict. The couch is
the opposite case - a talker reaches the mic 10-20 dB BELOW TV dialogue.

Data and venv - 16.5 GB of ACAV100M features, MUSAN, the RIRs - live under
--root (default C:\\Users\\tillm\\wake), out of the repo.
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

# torch.onnx's exporter prints U+2705 and the Windows console is cp1252, so
# export dies with UnicodeEncodeError AFTER training finishes (2026-08-15).
# Kills the plain `livekit-wakeword export` CLI too.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "alfred.yaml"
# Bump for every retrain whose DATA changed: same-version rows in
# pipeline_results.json are skipped as done. v1.0 = original sweep; v1.1 = SNR
# fix, no game audio; v1.2 = game backgrounds + held-out negatives (2026-08-16);
# v1.3 = clean fraction is no longer digital silence (patch_pad_then_mix), a
# DIFFERENT pad-then-mix recipe from the four v1.2 models left on disk by
# 2026-08-17's aborted sweep.
VERSION = "v1.3"

STAGES = ["generate", "augment", "features", "train"]

# Set in main() from --continuations; every load_config in a run must agree.
CONTINUATIONS = False

# What a VARIANT may override. Anything else changes what
# generate/augment/features produce, and those run ONCE and are shared - a
# variant touching them would train on the previous variant's data.
TRAINING_ONLY = {
    "model",
    "steps",
    "learning_rate",
    "weight_decay",
    "label_smoothing",
    "max_negative_weight",
    "target_fp_per_hour",
    "batch_n_per_class",
}


def check_not_compounding(lo, rounds):
    """Refuse a widened SNR stacked on multiple augmentation rounds: round N
    mixes fresh noise into round N-1's OUTPUT, so three rounds from a 0 dB floor
    land near -18 dB. livekit's prod.yaml runs rounds: 3 at a +5 dB floor."""
    if rounds > 1 and lo < 5.0:
        sys.exit(
            f"refusing: snr floor {lo:+g} dB with augmentation.rounds={rounds}.\n"
            f"Rounds COMPOUND - each one re-mixes the previous round's output, "
            f"so {rounds} rounds from a {lo:+g} dB floor lands far below it and "
            f"teaches the model to fire on noise.\n"
            f"Pick one: keep rounds at 1 (what alfred.yaml ships), or raise "
            f"--snr's floor to +5 or above."
        )


def patch_rir(p):
    """Override the reverberation probability livekit hardcodes at its call
    site - apply_rir(audio) is invoked with no p. 270 impulse responses are on
    disk; an empty rir_files list makes apply_rir a silent no-op."""
    from livekit.wakeword.data import augment as _aug

    original = _aug.AudioAugmentor.apply_rir

    def with_p(self, audio, p_override=p):
        return original(self, audio, p_override)

    _aug.AudioAugmentor.apply_rir = with_p


def patch_seed(seed):
    """Pin every RNG the training run touches; livekit exposes no seed.

    Two models on BYTE-IDENTICAL features, same recipe, different seeds: clean
    64% vs 86%, noise ceiling 0.678 vs 0.220, +10 dB recall 14% vs 37%
    (2026-08-17). Seed variance beats every effect chased here - sweep >=3 per
    arm and compare distributions."""
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

    Upstream mixes background into the ~0.9 s clip and only then drops it into
    np.zeros (align_clip_to_end for positives, centre-pad for negatives), so
    positives carry 912 ms of leading DIGITAL SILENCE (51.8% exact zeros),
    adversarial negatives 412 ms / 41%, background negatives 0 ms / 0% - a
    near-free discriminant that openWakeWord's 80 ms-hop stream never provides
    (ACAV100M - 87% of every batch - is precomputed continuous features, so a
    digital-silence lead-in marks a window as TTS-derived).
    Zeroing only the lead-in of the deployed model's inputs moved +20 dB
    74%->90%, +10 dB 38%->82%, 0 dB 2%->42% (and false accepts ~5.8x, so
    FA-matched the artifact is worth ~40-60 points).

    Three details that make or break it:
      * SNR references the CLIP's power, not the padded window's - upstream's
        np.mean(audio**2) over a half-zero window mixes noise ~3 dB hot.
      * Negatives get the same treatment, or the shortcut inverts into "zeros
        mean NEGATIVE".
      * The clean fraction is background 35 dB DOWN, not removed. Returning the
        bare window reinstates the shortcut on 20.7% of positives and 24.7% of
        adversarial negatives (2026-08-17, arm B)."""
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
        return bg[s : s + n]

    def padded_mix(self, audio, snr_db_range=(lo, hi)):
        if not self.background_files:
            return audio
        # End-aligned with upstream's 0-200 ms jitter; only the FILL changes.
        win = np.zeros(target_len, dtype=np.float32)
        end = target_len - random.randint(0, jitter)
        start = max(0, end - len(audio))
        win[start:end] = audio[max(0, len(audio) - (end - start)) :][: end - start]
        # The clean fraction is a HIGH SNR, not a bypass (docstring, third
        # bullet): returning `win` here rebuilds the shortcut.
        snr_db = (
            CLEAN_SNR_DB
            if clean_p and random.random() < clean_p
            else random.uniform(*snr_db_range)
        )
        bg = bg_window(self, target_len)
        clip_power = float(np.mean(audio**2)) + 1e-8  # NOT the window
        bg_power = float(np.mean(bg**2)) + 1e-8
        scale = np.sqrt(clip_power / (bg_power * 10 ** (snr_db / 10)))
        return (win + scale * bg).astype(np.float32)

    # Clips already have the target length, so align_clip_to_end must no-op or
    # it re-crops by the jitter and zero-fills the tail.
    original_align = _aug.align_clip_to_end

    def align_noop(audio, target_length, **kw):
        return (
            audio
            if len(audio) == target_length
            else original_align(audio, target_length, **kw)
        )

    _aug.AudioAugmentor.mix_with_background = padded_mix
    _aug.align_clip_to_end = align_noop
    print(
        f"[patch] pad-then-mix: {target_len / 16000:.1f}s window built from "
        f"background, clip written into it (no zero pad), "
        f"{clean_p:.0%} at +{CLEAN_SNR_DB:g} dB instead of clean",
        flush=True,
    )


def patch_augmentation(lo, hi, clean_p):
    """Widen the background-mix SNR and let some clips through clean.

    livekit's default is `snr_db_range=(5.0, 15.0)` on 100% of clips, with no
    probability gate and no config key; openWakeWord's is -10..15 dB at p=0.75.
    Monkeypatched, not edited into site-packages: a venv rebuild would revert a
    patched file and the next model would train at the old SNR silently. Do NOT
    widen this AND raise augmentation.rounds - rounds compound."""
    import inspect

    from livekit.wakeword.data import augment as _aug

    original = _aug.AudioAugmentor.mix_with_background
    # If upstream renames the parameter the override still applies and quietly
    # mixes at the new default. Verified against 0.2.1 (2026-08-16).
    if "snr_db_range" not in inspect.signature(original).parameters:
        sys.exit(
            "livekit's mix_with_background no longer takes snr_db_range - "
            "the SNR patch would silently do nothing. Re-read "
            "data/augment.py and update patch_augmentation before "
            "training anything."
        )

    def widened(self, audio, snr_db_range=(lo, hi)):
        if clean_p and random.random() < clean_p:
            return audio  # the clean quarter, oWW-style
        return original(self, audio, snr_db_range)

    _aug.AudioAugmentor.mix_with_background = widened
    print(
        f"[patch] background SNR {lo:+g}..{hi:+g} dB, {clean_p:.0%} left clean "
        f"(livekit default: +5..+15 dB, 0% clean)",
        flush=True,
    )


def patch_adversarial_from_bare(bare):
    """Derive the adversarial negatives from the BARE phrases only.

    livekit substitutes one word of each target phrase for a near-rhyme, so
    "hey alfred play" yields "hey alfred clay", "hey alfred nintendo" - 39,098
    of 133,713 phrases, 29% of the negative corpus, each containing the COMPLETE
    wake phrase and labelled NOT A WAKE WORD. It is PREFIXES that are lethal;
    the bare two-word list alone is nearly safe (48 of 17,124 are genuinely
    different words). The patch keeps the substitution generator on
    "hey alfred" / "hey al fred" only, so continuation forms exist as positives
    and nothing else."""
    from livekit.wakeword.data import generate as _gen

    original = _gen.generate_adversarial_phrases

    def bare_only(target_phrases=None, **kw):
        return original(target_phrases=list(bare), **kw)

    _gen.generate_adversarial_phrases = bare_only
    print(
        f"[patch] adversarial negatives derived from {list(bare)} only - "
        f"continuation forms are positives and never negatives",
        flush=True,
    )


def merge(base, over):
    """Recursive dict overlay, so a variant can set model.model_size without
    restating model_type."""
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
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
                f"instead. Overridable: {sorted(TRAINING_ONLY)}"
            )
    return specs


def load_config(root, size=None, over=None):
    """A config with every path resolved against --root and an optional size
    override, built from the raw YAML so the user's file is never rewritten.

    livekit resolves background_paths and rir_paths against the CURRENT
    DIRECTORY, and a missing background folder does not raise - it augments
    against nothing."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["data_dir"] = str(root / "data")
    raw["output_dir"] = str(root / "output")
    aug = raw.setdefault("augmentation", {})
    for key in ("background_paths", "rir_paths"):
        aug[key] = [
            p if Path(p).is_absolute() else str((root / p).resolve())
            for p in aug.get(key, [])
        ]
    raw.pop("variants", None)  # ours, not livekit's schema
    cont = raw.pop("continuation_phrases", None) or []
    if CONTINUATIONS and cont:
        # Weighted by repetition: piper's sampler is
        # `itertools.cycle(phrases)`, so every entry gets an equal share of
        # n_samples and 2 bare + 12 continuations would put the isolated
        # delivery at 14%. Repeating the bare forms makes the split 50/50.
        bare = raw["target_phrases"]
        raw["target_phrases"] = bare * max(1, round(len(cont) / len(bare))) + cont
    if over:
        raw = merge(raw, over)
    if size:
        raw.setdefault("model", {})["model_size"] = size
    cfg = WakeWordConfig(**raw)
    for d in cfg.augmentation.background_paths + cfg.augmentation.rir_paths:
        if not Path(d).is_dir():
            sys.exit(
                f"augmentation path does not exist: {d}\n"
                f"Paths in alfred.yaml are relative to --root ({root})."
            )
    return cfg


def provenance(cfg, snr, clean_p):
    """Which data produced the numbers - above all the augmentation settings,
    now that the SNR is patched."""
    bg = [
        p for d in cfg.augmentation.background_paths for p in Path(d).glob("**/*.wav")
    ]
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
        "by_folder": {
            d: sum(1 for p in bg if p.parent.name == d)
            for d in sorted({p.parent.name for p in bg})
        },
    }


def tool_versions():
    """Versions and the code commit, stamped into every result row: the
    monkeypatches make results a function of THIS code, not just the config."""
    import importlib.metadata as im

    out = {}
    for pkg in ("livekit-wakeword", "openwakeword", "torch"):
        try:
            out[pkg] = im.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        r = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
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


def _short(v):
    """One-line diffs; target_phrases is 24 entries once continuations are
    in."""
    return f"{len(v)} phrases" if isinstance(v, list) and len(v) > 4 else repr(v)


def data_stamp_path(cfg):
    return Path(cfg.model_output_dir) / "data_settings.json"


def write_data_stamp(cfg, prov):
    """Record, NEXT TO THE FEATURES, the settings that produced them - not the
    ones this invocation asked for. The two diverge the moment --from train
    skips augment; on 2026-08-16 that trained medium-400k and dnn-medium on a
    leftover -10..+15 dataset while the results file recorded [0, 20]."""
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


TTS_SPLITS = ("positive_train", "positive_test", "negative_train", "negative_test")


def clear_tts_splits(cfg):
    """Delete the synthesised clips so run_generate actually regenerates them:
    it counts existing clip_######.wav files and SKIPS a split that already has
    n_samples of them, whatever phrases they were made from."""
    for split in TTS_SPLITS:
        d = Path(cfg.model_output_dir) / split
        n = len(list(d.glob("*.wav"))) if d.is_dir() else 0
        if n:
            shutil.rmtree(d)
            print(f"    cleared {split} ({n} wavs)", flush=True)


def check_positive_lengths(cfg, sample=2000):
    """Warn if a positive is too long to survive end-alignment: the clip is
    written into a clip_duration window ending at target_len minus 0-200 ms of
    jitter and cropped from the FRONT if it does not fit, so a long
    continuation loses "hey alfred" and carries no wake word."""
    import soundfile as sf

    budget = cfg.augmentation.clip_duration - 0.2
    files = sorted((Path(cfg.model_output_dir) / "positive_train").glob("*.wav"))
    if not files:
        return
    step = max(1, len(files) // sample)
    durs = np.array([sf.info(str(p)).duration for p in files[::step]])
    over = float((durs > budget).mean())
    print(
        f"positives   {len(files)} clips, median {np.median(durs):.2f}s, "
        f"p99 {np.percentile(durs, 99):.2f}s, {over:.1%} over the "
        f"{budget:.1f}s budget",
        flush=True,
    )
    if over > 0.01:
        print(
            f"\n*** {over:.1%} of positives are longer than {budget:.1f}s. "
            f"Those get cropped from the FRONT\n    when they are written "
            f"into the window, which removes the wake phrase and\n    leaves "
            f"a positive that does not contain one. Shorten the "
            f"continuation\n    forms in alfred.yaml before training on "
            f"this.\n",
            flush=True,
        )


def run_data_stages(cfg, first):
    """generate/augment/features, skipping everything before `first`.

    Shared by every size - generate and augment key their output on model_name,
    not model_size - so they run ONCE and all sizes train on identical data.
    run_generate resumes from existing clips, so --from is for skipping
    deliberately, not for crash recovery."""
    order = STAGES.index(first)
    if order <= STAGES.index("generate"):
        print("\n=== generate: TTS positives + adversarial negatives ===", flush=True)
        # If the stamp cannot prove the clips came from this phrase list,
        # rebuild them (~40 min of TTS).
        stamped = (read_data_stamp(cfg) or {}).get("target_phrases")
        if stamped != list(cfg.target_phrases):
            print(
                f"    phrase list changed (on disk: "
                f"{'unrecorded' if stamped is None else len(stamped)} "
                f"phrases, this run: {len(cfg.target_phrases)}) - "
                f"regenerating",
                flush=True,
            )
            clear_tts_splits(cfg)
        run_generate(cfg)
        check_positive_lengths(cfg)
    if order <= STAGES.index("augment"):
        print("\n=== augment: RIR + background at the patched SNR ===", flush=True)
        run_augment(cfg)
    if order <= STAGES.index("features"):
        print("\n=== features: mel -> speech_embedding -> .npy ===", flush=True)
        run_extraction(cfg)


def artifact_stem(model_name, name, tag, seed):
    """The model name this run will write - the identity a re-run checks before
    believing a finished row. The seed is IN THE FILENAME: without it, three
    seeds per cell overwrite one another (2026-08-17). One home because main()
    must predict the name before run_variant builds it."""
    return f"{model_name}_{name}{f'_{tag}' if tag else ''}_s{seed}_{VERSION}"


def run_variant(name, over, root, results, artifacts, key, stem, prov, seed):
    # A name with no entry in `variants` is taken as a bare model_size.
    cfg = (
        load_config(root, over=over)
        if over is not None
        else load_config(root, size=name)
    )
    print(
        f"\n=== {key}: {cfg.model.model_type}/{cfg.model.model_size}, "
        f"{cfg.steps} steps, seed {seed} ===",
        flush=True,
    )

    # Per VARIANT, not once per process: run_train advances the global RNG, so
    # seeding only at startup would leave the arms unpaired.
    patch_seed(seed)
    t0 = time.time()
    pt_path = run_train(cfg)
    onnx_path = run_export(cfg)
    # The eval is a smoke test; a crash in it must not discard the training run.
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
        # always overwrites <model_name>_det.png, so copy out before the next
        # size destroys them.
        "onnx": keep(onnx_path, artifacts / f"{stem}.onnx"),
        "pt": keep(pt_path, artifacts / f"{stem}.pt"),
        "det_png": keep(
            cfg.model_output_dir / f"{cfg.model_name}_det.png",
            artifacts / f"{stem}_det.png",
        ),
        # The settings THIS row was measured under; the global _run blob goes
        # stale once a --tag re-run mixes two SNR settings into one file.
        "data": dict(prov),
    }
    if metrics is None:
        row["eval_error"] = eval_err
        results[key] = row
        print(
            f"  {key}: trained + exported, EVAL FAILED - artifacts kept ({eval_err})",
            flush=True,
        )
        return

    # Two operating points: evaluate.py hardcodes threshold=0.5 (*_at_half),
    # find_best_threshold maximises recall subject to fpph <=
    # target_fp_per_hour. Neither is a K15 threshold - the parity gate
    # (2026-08-13) showed livekit's scores do not transfer to openWakeWord's
    # streaming runtime, so the deployed number comes from --wake-trials peaks.
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
    print(
        f"  {key}: {row['optimal_recall']:.1%} recall at "
        f"{row['optimal_fpph']} FP/hr over {row['validation_hours']:.1f} h "
        f"({minutes} min)",
        flush=True,
    )


def table(results, root, target_fpph):
    print(f"\n{'':37}{'--- at fixed 0.5 ---':>26}{'--- tuned ---':>26}")
    print(
        f"{'variant':18}{'params':>9}{'onnx':>10}{'AUT':>9}{'FPPH':>8}"
        f"{'recall':>9}{'thresh':>9}{'recall':>9}{'FPPH':>8}{'train':>8}"
    )
    print("-" * 97)
    for key, r in results.items():
        if key.startswith("_"):
            continue
        if "error" in r:
            print(f"{key:18}FAILED - {r['error'][:60]}")
            continue
        if "eval_error" in r:
            print(
                f"{key:18}trained OK, eval FAILED (artifacts kept) - "
                f"{r['eval_error'][:40]}"
            )
            continue
        print(
            f"{key:18}{r['params'] / 1000:>8.1f}k{r['onnx_kb']:>7} KB"
            f"{r['aut']:>9.4f}{r['fpph_at_half']:>8.2f}"
            f"{r['recall_at_half']:>9.1%}{r['optimal_threshold']:>9.2f}"
            f"{r['optimal_recall']:>9.1%}{r['optimal_fpph']:>8.2f}"
            f"{r['train_min']:>7.1f}m"
        )
    scored = {
        k: r
        for k, r in results.items()
        if not k.startswith("_") and "error" not in r and "eval_error" not in r
    }
    live = list(scored.values())
    # find_best_threshold silently falls back to MAX BALANCED ACCURACY when no
    # threshold fits target_fp_per_hour (2026-08-17: 16.3 and 13.4 FP/hr against
    # a 0.2 target, still printed as "98.4% recall").
    missed = [k for k, r in scored.items() if r["optimal_fpph"] > target_fpph]
    if missed:
        print(
            f"\n*** {', '.join(missed)}\n    NO threshold met the "
            f"{target_fpph} FP/hr budget, so those tuned columns are "
            f"livekit's\n    fallback (max balanced accuracy), not an "
            f"operating point. Read them as\n    'did not fit', not as "
            f"recall."
        )
    # positive_features_test comes from the SAME augmentation as training, so a
    # pad-then-mix row and the row beside it are graded on different test sets
    # and their recalls are not comparable.
    if (
        len({bool((r.get("data") or {}).get("pad_then_mix")) for r in scored.values()})
        > 1
    ):
        print(
            "\n*** MIXED TEST SETS: pad-then-mix rows are graded on "
            "realistic windows, the\n    others on windows carrying the "
            "silence shortcut - positive_features_test is\n    built by "
            "whichever augmentation trained the model. Recall is NOT "
            "comparable\n    across that line. Bench.bat is; it scores every "
            "model on the same real audio."
        )
    if len(live) >= 2 and (
        max(r["optimal_recall"] for r in live) - min(r["optimal_recall"] for r in live)
        < 0.005
        and max(r["optimal_fpph"] for r in live) - min(r["optimal_fpph"] for r in live)
        < 0.02
    ):
        print(
            "\nSATURATED: every variant landed within 0.5% recall and 0.02 "
            "FPPH of the rest.\nThis table cannot rank them - it is a smoke "
            "test that they all passed. The\nranking, if there is one, "
            "comes from Bench.bat."
        )
    hrs = next(
        (
            r.get("validation_hours")
            for r in results.values()
            if isinstance(r, dict) and "validation_hours" in r
        ),
        None,
    )
    print(
        f"\nFPPH is measured over {hrs} h of validation negatives. If "
        f"make_validation.py\nhas run, that is YOUR room and the number "
        f"means something; if it has not, it is\nlivekit's synthetic set, "
        f"which could not tell three sizes apart in 2026-08-15's\nsweep "
        f"(all three: 3 FPs, ~99.3% recall, AUT 0.0000)."
    )
    print(
        "\nNEITHER threshold column ships. The parity gate showed livekit's "
        "scores do\nnot transfer to openWakeWord's runtime - set "
        "voice.wakeThreshold from the peak\nvalues that --wake-trials logs "
        "on the K15."
    )
    print(f"\nartifacts: {root / 'artifacts'}")
    print(
        "\nNEXT: Bench.bat. Nothing above ranks these candidates - the "
        "synthetic eval is\nsaturated (both sizes ace it). bench_real.py "
        "scores them on your voice, in your\nroom, under openWakeWord, and "
        "that ranking is the one that has ever been right."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "variants",
        nargs="*",
        default=None,
        help="names from alfred.yaml's `variants:` (default: all). "
        "A name with no entry is taken as a bare model_size.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\tillm\wake"),
        help="where data/, output/ and the venv live (not the repo)",
    )
    ap.add_argument(
        "--from",
        dest="first",
        choices=STAGES,
        default="generate",
        help="skip stages before this one",
    )
    # (0, 20), NOT openWakeWord's (-10, 15): A/B'd 2026-08-16 on identical data
    # and refuted - noise ceiling 0.103 -> 0.678, separation 5.34x -> 1.07x,
    # threshold 0.10 -> 0.62, recall @+10 dB 23% -> 14%. oWW pairs that floor
    # with ~31,000 h of negatives; we have ~2,000 h.
    ap.add_argument(
        "--snr",
        nargs=2,
        type=float,
        default=(0.0, 20.0),
        metavar=("LO", "HI"),
        help="background mix range in dB",
    )
    ap.add_argument(
        "--clean", type=float, default=0.25, help="fraction of clips left un-mixed"
    )
    # Far-field is the deployment condition. livekit's default is 0.5; a knob
    # because nothing has measured its effect yet.
    ap.add_argument(
        "--rir",
        type=float,
        default=0.5,
        help="probability a clip is convolved with a room impulse "
        "response (livekit default 0.5)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for result keys and artifact names. Use when "
        "re-running with DIFFERENT data settings (an --snr "
        "A/B): without it the second run's artifacts "
        "overwrite the first's and its finished rows are "
        "skipped as already done",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="training RNG seed. Seed variance is LARGER than "
        "every effect measured here (64%% vs 86%% clean on "
        "identical data), so sweep >=3 per arm.",
    )
    ap.add_argument(
        "--pad-then-mix",
        action="store_true",
        help="build the training window from background instead of "
        "zeros - removes the silence shortcut (see "
        "patch_pad_then_mix). Changes the DATA: needs "
        "--from augment.",
    )
    ap.add_argument(
        "--continuations",
        action="store_true",
        help="add alfred.yaml's continuation_phrases to the "
        "positives - the run-on delivery ('hey alfred play "
        "hades') the TTS corpus has never contained. Changes "
        "the DATA: needs --from generate.",
    )
    ap.add_argument(
        "--no-bench",
        action="store_true",
        help="skip the real-audio bench that normally closes a run",
    )
    ap.add_argument("--list", action="store_true", help="show state and exit")
    args = ap.parse_args()
    if args.tag and not all(c.isalnum() or c in "._-" for c in args.tag):
        sys.exit("--tag must be filename-safe: letters, digits, . _ -")

    # Before the first load_config, because it changes what a config IS.
    global CONTINUATIONS
    CONTINUATIONS = args.continuations

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
    print(
        f"validation {val.name}: "
        f"{'MISSING' if not val.exists() else f'{val.stat().st_size / 1e6:.0f} MB'}"
        f"  <- make_validation.py replaces this with your room"
    )
    print(f"variants   {names}   stages from '{args.first}'")
    print(
        f"augment    snr {args.snr[0]:+g}..{args.snr[1]:+g} dB, "
        f"{args.clean:.0%} clean, rir p={args.rir}, "
        f"rounds {cfg0.augmentation.rounds}"
    )
    # What is ALREADY trained, and under what: a `--tag snr-neg10` run on a
    # stale checkout once trained at (0, 20) and produced a duplicate of
    # `medium` under a label that said otherwise (2026-08-16).
    if results_path.exists():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        rows = [
            (k, v.get("data") or {}) for k, v in prior.items() if not k.startswith("_")
        ]
        if rows:
            print("\nalready trained:")
            for k, dat in rows:
                snr, pad = dat.get("snr_db_range"), dat.get("pad_then_mix")
                # "SAME" must mean the same DATA, not the same SNR: every
                # pad-then-mix row carries an identical snr to its control.
                same = (
                    " <- SAME as this run"
                    if (snr, bool(pad)) == (list(args.snr), bool(args.pad_then_mix))
                    else ""
                )
                print(
                    f"  {k:22} snr={snr} clean={dat.get('clean_fraction')} "
                    f"rir={dat.get('rir_p')} pad={bool(pad)} "
                    f"commit={dat.get('pipeline_commit')}{same}"
                )
    # What the models will ACTUALLY train on, printed before the --list return:
    # --from train reuses whatever augment last wrote.
    asked = {
        "snr_db_range": list(args.snr),
        "clean_fraction": args.clean,
        "rir_p": args.rir,
        "rounds": cfg0.augmentation.rounds,
        "pad_then_mix": bool(args.pad_then_mix),
        # What the CLIPS were made from: a resume vs a silent reuse.
        "target_phrases": list(cfg0.target_phrases),
    }
    rebuilding = STAGES.index(args.first) <= STAGES.index("augment")
    on_disk = read_data_stamp(cfg0)
    prov = (
        dict(asked)
        if rebuilding
        else {**(on_disk or asked), "data_stamp": bool(on_disk)}
    )
    prov.update(tool_versions())
    # EVERY data setting, not just the SNR: `--pad-then-mix --from train` over
    # features built without it would otherwise pass. Keys absent from an older
    # stamp are left alone rather than reported as mismatches.
    differs = {
        k: (on_disk[k], v)
        for k, v in asked.items()
        if on_disk and k in on_disk and on_disk[k] != v
    }
    if not rebuilding:
        if differs:
            print(
                f"\n*** STALE DATA: the features on disk were NOT built the "
                f"way this run asks.\n    --from {args.first} does not "
                f"rebuild them, so that is what these models\n    will train "
                f"on; the rows are stamped with the real values. Use\n"
                f"    --from augment to change it.",
                flush=True,
            )
            for k, (was, want) in differs.items():
                print(
                    f"      {k}: on disk {_short(was)}, this run asked {_short(want)}",
                    flush=True,
                )
            print(flush=True)
        elif not on_disk:
            print(
                "\n[warn] no data_settings.json beside the features: they "
                "predate this stamping,\n       so provenance falls back "
                "to the arguments and may be wrong. --from augment\n"
                "       rebuilds and records the truth.\n",
                flush=True,
            )
        else:
            phrases = on_disk.get("target_phrases")
            print(
                f"data       features built with snr="
                f"{on_disk['snr_db_range']}, pad_then_mix="
                f"{on_disk.get('pad_then_mix')}, "
                f"{len(phrases) if phrases else 'unrecorded'} phrases "
                f"(matches this run)"
            )
    if args.list:
        return 0

    check_not_compounding(args.snr[0], cfg0.augmentation.rounds)
    if args.continuations:
        # The BARE list: load_config already appended the continuations.
        patch_adversarial_from_bare(
            yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["target_phrases"]
        )
    if args.pad_then_mix:
        patch_pad_then_mix(
            args.snr[0],
            args.snr[1],
            args.clean,
            int(cfg0.augmentation.clip_duration * 16000),
        )
    else:
        patch_augmentation(args.snr[0], args.snr[1], args.clean)
    n_rir = sum(
        len(list(Path(d).glob("**/*.wav"))) for d in cfg0.augmentation.rir_paths
    )
    if not n_rir:
        sys.exit(
            f"no impulse responses under {cfg0.augmentation.rir_paths} - "
            f"apply_rir would silently no-op and every positive would "
            f"train anechoic, which is the opposite of far-field."
        )
    patch_rir(args.rir)
    print(
        f"[patch] reverberation p={args.rir} over {n_rir} impulse responses", flush=True
    )
    run_data_stages(cfg0, args.first)

    # Appended after EACH size; a finished size is skipped on a re-run. Delete
    # the file to force a retrain.
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    if rebuilding:
        # The data stages ran, so the arguments ARE the truth now.
        write_data_stamp(cfg0, asked)
    for name in names:
        # The seed is ALWAYS part of the identity: two rows differing only by
        # seed are two samples of one arm, not two arms.
        suffix = f"{args.tag}-" if args.tag else ""
        key = f"{name}@{suffix}s{args.seed}"
        stem = artifact_stem(cfg0.model_name, name, args.tag, args.seed)
        # Done only if the MODEL is on disk under the name this run would
        # write: 2026-08-17's sweep left twelve finished-looking rows whose
        # artifacts had all overwritten one another.
        done = results.get(key)
        if done and "error" not in done:
            if (
                done.get("onnx") == f"{stem}.onnx"
                and (artifacts / f"{stem}.onnx").is_file()
            ):
                print(f"=== {key}: already in {results_path.name}, skipping ===")
                continue
            print(f"=== {key}: row exists but {stem}.onnx does not - retraining ===")
        try:
            run_variant(
                name,
                specs.get(name),
                root,
                results,
                artifacts,
                key,
                stem,
                prov,
                args.seed,
            )
        except Exception:
            # A failed variant is recorded and the sweep continues.
            traceback.print_exc()
            results[key] = {"error": traceback.format_exc(limit=1).strip()}
            print(f"  {key} FAILED - continuing", flush=True)
        results["_run"] = provenance(load_config(root), args.snr, args.clean)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table(results, root, cfg0.target_fp_per_hour)
    print(f"results:   {results_path}")

    # livekit's eval cannot rank: positive_train and positive_test come from ONE
    # tts.synthesize_clips call - same phrases, same voice pool, no speaker
    # holdout - through the same augmentation at the same SNR.
    if not args.no_bench:
        print("\n" + "=" * 79)
        print("REAL-AUDIO BENCH - your voice, your room, openWakeWord's runtime")
        print("=" * 79, flush=True)
        r = subprocess.run(
            [
                sys.executable,
                str(HERE / "bench_real.py"),
                "--root",
                str(root),
                "--snr-sweep",
            ]
        )
        if r.returncode:
            # Missing positives or negatives is the usual cause, and is not a
            # training failure.
            print(
                "\n[bench] did not run. The models are still in "
                f"{artifacts}; fix the bench inputs and run Bench.bat."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
