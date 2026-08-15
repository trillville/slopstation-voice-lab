"""Capture the living room as training data for a custom wake word.

    .venv\\Scripts\\python bench\\record_room.py 2400        (40 minutes)

Runs on the K15, because the point is THIS mic in THIS room: the recording
becomes background audio a custom wake model is trained against, and the
model's real failure mode is a TV talking through the couch mic. Play a
dialogue-heavy film (not music) at normal listening volume and leave the room.

Close the voice supervisor first - it holds the same mic, and two readers on
one endpoint is a fight neither wins.

WHY it resolves the device rather than taking PortAudio's default: the agent
binds the mic NAMED in config.json (audio.resolve_device), which is not
necessarily the Windows default. Recording the default would capture a
different microphone than the one the model has to survive, and nothing
downstream would say so - the negatives would just be quietly wrong. Resolved
with log=None: a bench tool has no business emitting audio_device into prod
telemetry, and the name is printed here instead.

Writes room.wav (16 kHz mono, ~77 MB for 40 min) to the current directory,
gitignored. It is NOT usable as-is: slice it with bench/slice_room.py before
it reaches the augmenter - docs/custom-wakeword-design.md § 3 has the why.
"""
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

RATE = 16000
CHUNK = 1280                                # audio.py's native 80 ms hop
# Multiply before dividing: a hop is 12.5 chunks per second, so RATE // CHUNK
# truncates to 12 and quietly makes every duration 4% short.
MINUTE = 60 * RATE // CHUNK                 # chunks per minute of wall clock


def main():
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    voice = json.loads((cglib.BASE / "config.json")
                       .read_text(encoding="utf-8-sig"))["voice"]

    import pyaudio
    pa = pyaudio.PyAudio()
    idx = audio.resolve_device(pa, voice["inputDeviceName"],
                               want_input=True, log=None)
    print(f"  mic: {pa.get_device_info_by_index(idx)['name'] if idx is not None else 'system default'}")

    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK,
                     input_device_index=idx)
    w = wave.open("room.wav", "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(RATE)
    # A silent 40-minute run is indistinguishable from a hung stream, and
    # finding out at the end costs another 40 minutes and another film.
    for i in range(secs * RATE // CHUNK):
        w.writeframes(stream.read(CHUNK, exception_on_overflow=False))
        if i % MINUTE == 0:
            print(f"  {i // MINUTE} / {secs // 60} min", flush=True)
    w.close()
    audio.close_stream_quietly(stream)
    pa.terminate()
    print(f"done -> room.wav ({secs // 60} min)")


if __name__ == "__main__":
    main()
