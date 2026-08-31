"""Point the threshold optimiser at YOUR living room instead of a synthetic set.

    Validate.bat                      featurize data\\heldout\\**.wav
    Validate.bat --restore            put livekit's stock set back

Writes data/features/validation_set_features.npy, which trainer.py's
_load_validation_data() concatenates into the validation negatives, so both the
reported FPPH and find_best_threshold's tuned threshold are measured against
it. Feed it audio the model has NEVER trained on.

Non-overlapping windows: validation_hours is rows * clip_duration / 3600, so
overlap would inflate the denominator and understate FPPH.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.data.features import _pad_or_truncate
from livekit.wakeword.models.feature_extractor import (MelSpectrogramFrontend,
                                                       SpeechEmbedding)
from livekit.wakeword.resources import (get_embedding_model_path,
                                        get_mel_model_path)

for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "alfred.yaml"
RATE = 16000
# The stock set survives under this name so --restore never needs the 176 MB
# re-download.
STOCK_BACKUP = "validation_set_features.livekit-stock.npy"


def clips(wav_paths, window):
    """Every wav cut into non-overlapping `window`-sample chunks, mono."""
    for path in wav_paths:
        audio, sr = sf.read(str(path), dtype="float32")
        if sr != RATE:
            sys.exit(f"{path.name}: {sr} Hz, need {RATE}. Record with "
                     f"k15/agent/bench/record_room.py, which writes 16 kHz mono.")
        if audio.ndim > 1:
            audio = audio[:, 0]
        for start in range(0, len(audio) - window + 1, window):
            yield audio[start:start + window]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\tillm\wake"))
    ap.add_argument("--heldout", type=Path, default=None,
                    help="default: <root>/data/heldout")
    ap.add_argument("--restore", action="store_true",
                    help="put livekit's stock validation set back")
    ap.add_argument("--replace", action="store_true",
                    help="use ONLY the held-out audio (default: append to stock)")
    args = ap.parse_args()

    root = args.root.resolve()
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["data_dir"], raw["output_dir"] = str(root / "data"), str(root / "output")
    cfg = WakeWordConfig(**raw)

    feat_dir = Path(cfg.data_path) / "features"
    target = feat_dir / "validation_set_features.npy"
    backup = feat_dir / STOCK_BACKUP

    if args.restore:
        if not backup.exists():
            sys.exit(f"no backup at {backup} - nothing to restore")
        shutil.copy2(backup, target)
        print(f"restored livekit's stock set -> {target}")
        return 0

    heldout = (args.heldout or root / "data" / "heldout").resolve()
    wavs = sorted(heldout.glob("**/*.wav"))
    if not wavs:
        sys.exit(f"no wavs under {heldout}\n"
                 f"Put HELD-OUT room and game recordings there - audio the "
                 f"model has never trained on.")

    window = int(cfg.augmentation.clip_duration * RATE)
    print(f"heldout  {heldout}  ({len(wavs)} wav)")
    print(f"window   {cfg.augmentation.clip_duration}s = {window} samples, non-overlapping")

    mel = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
    emb = SpeechEmbedding(onnx_path=get_embedding_model_path())

    rows = []
    for chunk in clips(wavs, window):
        rows.append(_pad_or_truncate(emb.extract_embeddings(mel(chunk))[0]))
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} clips ({len(rows) * cfg.augmentation.clip_duration / 3600:.2f} h)",
                  flush=True)
    if not rows:
        sys.exit("every wav was shorter than one window - nothing to write")

    room = np.stack(rows, axis=0).astype(np.float32)
    room_h = room.shape[0] * cfg.augmentation.clip_duration / 3600

    # First run only: keep livekit's stock set before overwriting it.
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
        print(f"stock set backed up -> {backup.name}")

    # APPEND by default, rebasing from the BACKUP not from `target` - otherwise
    # a re-run stacks the room audio onto a file that already has it. Replacing
    # the stock set's 16.7 h with minutes of room audio throws away the axis
    # that catches a model firing on ordinary speech.
    stock = np.zeros((0, 16, 96), dtype=np.float32)
    if backup.exists() and not args.replace:
        stock = np.load(str(backup))
        if stock.ndim == 2:                 # (N, 96) -> (N//16, 16, 96)
            stock = stock[:(stock.shape[0] // 16) * 16].reshape(-1, 16, 96)
    combined = np.concatenate([stock, room], axis=0) if stock.shape[0] else room
    stock_h = stock.shape[0] * cfg.augmentation.clip_duration / 3600
    total_h = combined.shape[0] * cfg.augmentation.clip_duration / 3600

    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(target), combined)
    print(f"\nwrote {combined.shape} -> {target}")
    print(f"  stock negatives  {stock_h:6.2f} h")
    print(f"  your room        {room_h:6.2f} h")
    print(f"  total            {total_h:6.2f} h")

    # find_best_threshold maximises recall subject to fpph <=
    # target_fp_per_hour, so the whole set gets a budget of (target x hours)
    # false positives, shared between the stock and room blocks.
    budget = cfg.target_fp_per_hour * total_h
    print(f"\nAt target_fp_per_hour {cfg.target_fp_per_hour}, the tuned "
          f"threshold gets a budget of\n{budget:.1f} false positives across "
          f"all {total_h:.2f} h. The 2026-08-15 sweep spent 3 of those on the "
          f"stock\nset alone, so a candidate that fires even once on your room "
          f"is now pushed to a\nhigher threshold than one that never does - "
          f"which is the discrimination the\neval was missing entirely.")
    if room_h < 0.5:
        print(f"\nNOTE: {room_h:.2f} h of room audio is a thin gate - it can "
              f"pass a model simply\nbecause those minutes held nothing "
              f"confusable. Adding more later is cheap:\nre-run this, delete "
              f"pipeline_results.json, then Train.bat --from train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
