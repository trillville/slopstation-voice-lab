"""Capture the living room as training data for a custom wake word.

    .venv\\Scripts\\python -m slopstation.agent.bench.record_room 2400        (40 minutes)

Runs on the K15 - the point is THIS mic in THIS room. Close the voice
supervisor first; it holds the same mic. Play a dialogue-heavy film or a game
at normal listening volume; talking over it is GOOD data, since live voices
are the negative class MUSAN (no speech by design) and TV dialogue cover
worst. One rule: nobody says the wake phrase - in a training background it
becomes a negative, in a held-out recording it punishes correct fires.

Binds the mic NAMED in config.json (audio.resolve_device), not PortAudio's
default, which may be a different microphone. log=None keeps audio_device out
of prod telemetry.

Writes room.wav (16 kHz mono, ~77 MB for 40 min) to the current directory,
gitignored. Slice it with bench/slice_room.py before it reaches the augmenter.
"""

import json
import sys
import wave

from slopstation import paths
from slopstation.agent.speech import audio

RATE = 16000
CHUNK = 1280  # audio.py's native 80 ms hop
# Multiply before dividing: RATE // CHUNK truncates 12.5 to 12 and makes every
# duration 4% short.
MINUTE = 60 * RATE // CHUNK  # chunks per minute of wall clock


def main():
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    voice = json.loads((paths.HOME / "config.json").read_text(encoding="utf-8-sig"))[
        "voice"
    ]

    import pyaudio

    pa = pyaudio.PyAudio()
    idx = audio.resolve_device(pa, voice["inputDeviceName"], want_input=True, log=None)
    print(
        f"  mic: {pa.get_device_info_by_index(idx)['name'] if idx is not None else 'system default'}"
    )

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=idx,
    )
    w = wave.open("room.wav", "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(RATE)
    # A silent 40-minute run is indistinguishable from a hung stream.
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
