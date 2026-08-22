"""Train the second-stage verifier from clips this rig actually fired on.

    .venv\\Scripts\\python bench\\train_verifier.py

Runs on the K15, where the clips already are: audio.py writes every wake
fire's pre-roll to logs/wake/*.wav, and labelling is sorting those by ear.

    logs\\wake\\yes\\   fires where you really said it
    logs\\wake\\no\\    fires where the TV said something and it woke anyway

A second stage rather than a better threshold: on 2026-08-15 the three false
accepts scored 0.25 / 0.26 / 0.28 against a median genuine wake of 0.255, so
no threshold separates them. openWakeWord's verifier is a logistic regression
over the same embeddings the wake model already computed, and can see whose
voice it is.

Before enabling it: it REPLACES the score rather than gating it, so every
threshold measured so far is void and wakeThreshold must be re-derived from
--wake-trials. And it is speaker-specific by design, so train it on everyone
who uses the room or guests get worse service.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # .../slopstation
sys.path.insert(0, str(HERE.parent))        # k15/voice
sys.path.insert(0, str(ROOT / "k15"))

import audio          # noqa: E402
import cglib          # noqa: E402

MIN_POSITIVE = 3                            # openWakeWord's documented floor


def check_clips(folder, what):
    """openWakeWord requires single-channel 16 kHz 16-bit WAV but does not
    validate it - it just trains something useless. audio.py's clips are
    already correct; hand-recorded ones may not be."""
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        sys.exit(f"no wavs in {folder}\n{what}")
    for p in wavs:
        with wave.open(str(p), "rb") as w:
            got = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if got != (1, 2, 16000):
            sys.exit(f"{p.name}: channels/width/rate {got}, need (1, 2, 16000)")
    return wavs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    clips = cglib.BASE / "logs" / "wake"
    ap.add_argument("--yes", type=Path, default=clips / "yes")
    ap.add_argument("--no", dest="no_dir", type=Path, default=clips / "no")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    voice = json.loads((cglib.BASE / "config.json")
                       .read_text(encoding="utf-8-sig"))["voice"]

    pos = check_clips(args.yes, "Sort real wakes from logs/wake into here.")
    neg = check_clips(args.no_dir,
                      "Sort false fires from logs/wake into here. Adding ~10 s "
                      "of your ordinary speech (real commands) helps too.")
    if len(pos) < MIN_POSITIVE:
        sys.exit(f"{len(pos)} positives, openWakeWord wants at least {MIN_POSITIVE}")

    # Resolved through the agent's OWN resolver: the verifier is keyed to one
    # specific model, and training against a different file than the one that
    # runs is a silent mismatch. pa/device are unused until a stream opens.
    listener = audio.WakeListener(None, voice, None)
    out = args.out or listener.model_path.with_name(
        f"{listener.model_path.stem}_verifier.pkl")

    print(f"model      {listener.model_path.name} ({listener.model_source})")
    print(f"positives  {len(pos)} in {args.yes}")
    print(f"negatives  {len(neg)} in {args.no_dir}")

    import openwakeword
    openwakeword.train_custom_verifier(
        positive_reference_clips=str(args.yes),
        negative_reference_clips=str(args.no_dir),
        output_path=str(out),
        model_name=str(listener.model_path),
        inference_framework="onnx",
    )

    print(f"\nwrote {out}")
    print(f"\nTo enable, in k15\\config.json under \"voice\":\n"
          f'    "wakeVerifier": "{out.name}",\n'
          f'    "wakeVerifierThreshold": 0.1\n'
          f"Then RE-RUN --wake-trials. The verifier replaces the score, so "
          f"every\nthreshold measured before this is void.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
