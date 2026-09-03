# slopstation-voice-lab

Wake-word training and voice-recognition benchmarking for
[slopstation](https://github.com/trillville/slopstation), a couch-gaming
control plane whose voice lane listens for a wake word on a mini PC and
streams what follows to a speech-to-text service.

This is research code. It produces one artifact the product ships - a wake
model - and a set of measurements that decide the settings around it. Nothing
here runs in production, which is why it lives apart: the application repo
holds only what its two lanes execute.

## The problem this exists for

An off-the-shelf wake word ("hey jarvis") fires on the television and misses
the person on the sofa. Training a custom one is easy; knowing whether it is
*better* is not, and the honest answer needs two things this repo carries.

**The training evaluator cannot rank models.** livekit-wakeword scored small,
medium and large identically on its synthetic set (3 false positives, ~99.3%
recall, AUT 0.0000 for all three). The same three models on 20 seconds of real
couch audio had median peak scores of 0.083, 0.892 and 0.585. The evaluator is
not wrong, it is answering a different question.

**Scores do not transfer between runtimes.** Models are trained by
livekit-wakeword and run by openWakeWord. That interop is not documented
upstream. The two disagree by 0.021-0.075 per hop, the gap scaling with head
size, because one is stateless and the other keeps a streaming buffer. A
threshold chosen on the training machine does not mean the same thing on the
device.

So the deployed threshold comes from `wake_trials.py`, measured on the
device's own microphone through the runtime that will run the model, and the
training metrics are used only to decide which candidates are worth measuring.

## What is here

| | |
|---|---|
| `pipeline.py` | The training run: generate, augment, extract features, train, export. Sweeps variants and records every result. |
| `alfred.yaml` | The training configuration the pipeline reads. |
| `bench_real.py` | Ranks candidate models on real recorded audio through openWakeWord, the production runtime. |
| `make_validation.py` | Builds a validation set from room recordings, so false-positive rates are measured against the actual room. |
| `record_room.py` | Records the room on the device, for backgrounds and negatives. |
| `slice_room.py` | Cuts a long room recording into background clips. |
| `slice_utterances.py` | Cuts a recording of spoken wake words into one clip per utterance. |
| `wake_trials.py` | The device measurement: recall over deliberate attempts, and false accepts per hour. This sets the shipped threshold. |
| `train_verifier.py` | Trains the optional second-stage verifier from clips the lane saved. |
| `probe_stt.py` | Speech-to-text quality: synthesized speech through the live recognizer, does the right game come out. |
| `models/` | Exported candidates. The application repo vendors whichever one it ships. |
| `Train.bat`, `Bench.bat`, `Validate.bat` | The three entry points, so a run is one command. |

## Two machines, two environments

The split is not tidiness, it is hardware.

**Training, on a machine with a GPU.** `pipeline.py`, `bench_real.py`,
`make_validation.py`, `slice_room.py`, `slice_utterances.py`. Its own virtual
environment:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Measurement, on the device with the microphone.** `wake_trials.py`,
`record_room.py`, `train_verifier.py`, `probe_stt.py`. These import the
application package, deliberately: they measure the shipping listener, its
model resolution and its device binding, not a copy that might drift from it.
Run them from a checkout of the application, in its virtual environment, or
install it alongside:

```
pip install git+https://github.com/trillville/slopstation
```

They read that application's `config.json` for the model name, thresholds and
device names.

## The loop

1. Record the room (`record_room.py`), slice it (`slice_room.py`), and build a
   validation set from it (`make_validation.py`). Without this, false-positive
   rates are measured against someone else's living room.
2. Train candidates (`Train.bat`, at least three seeds per arm: seed variance
   beat every effect measured here, so a single run compares nothing).
3. Rank them on real audio (`Bench.bat`), not on the training evaluator.
4. Measure the survivor on the device (`wake_trials.py`), set the threshold
   from the peak distribution, and record false accepts per hour over a long
   idle run.
5. Copy the model into the application repo and point its `wakeModel` at it.

Recordings, datasets and training checkpoints are gitignored. They are large,
they are specific to one room, and they are not reproducible from this repo.

## History

This code lived in the application repository until September 2026 and was
split out with its history intact. Commits before the split may reference
paths that no longer exist there.
