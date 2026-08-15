"""Does Flux hear a game title well enough to LAUNCH it?

    .venv\\Scripts\\python bench\\probe_stt.py
    .venv\\Scripts\\python bench\\probe_stt.py --raw-names     (the old keyterms)
    .venv\\Scripts\\python bench\\probe_stt.py --no-keyterms   (no boost at all)

Live Deepgram, real money, non-deterministic - see harness.py. Reads nothing
from the room: Windows SAPI speaks each utterance, so this measures the STT
config and not tonight's couch. It cannot tell you the array is aimed; it CAN
tell you a keyterm list change moved transcription, which is the one thing
nobody could answer before.

WHY (2026-08-14, read out of the logs): keyterms were Steam's own strings, so
Flux was taught ARMORED CORE VI FIRES OF RUBICON while the couch says
"armored core six". The boost half-fired - ARMORED CORSICS, ARMORED COSTS,
ARMOR CORE 6, "thyroid core 6" - and 11 of 12 launches missed. The bar is not
a pretty transcript, it is whether titles.build_resolver gets an appid out of
it, because that is what decides whether the game starts.

Exit code is the number of failing probes, so this can gate a keyterm change.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# Path setup inline rather than via harness: that module is the LLM probes'
# plumbing (fake job store, trial loops, a live backend per trial) and this
# probe drives the STT socket instead, so importing it would buy a sys.path
# and a misleading dependency.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # k15/voice
sys.path.insert(0, str(HERE.parents[2] / "k15"))

import cglib                                    # noqa: E402
import library                                  # noqa: E402
import session_runtime                          # noqa: E402
import titles                                   # noqa: E402
from grammar_gate import GrammarMatcher         # noqa: E402

URL = "wss://api.deepgram.com/v2/listen"
RATE = 16000
CHUNK = 3200                                    # 100 ms of s16 mono

# (spoken utterance, expected canonical title substring). The utterances are
# how a person asks, not how Steam writes it - that mismatch IS the bug.
# Expected is matched case-insensitively against the resolved title, so a
# title that gets renamed on Steam fails loudly rather than silently passing.
CASES = [
    ("play armored core six", "ARMORED CORE"),
    ("i want to play armored core six", "ARMORED CORE"),
    ("open armored core six", "ARMORED CORE"),
    ("play hades two", "Hades II"),
    ("play baldurs gate three", "Baldur's Gate 3"),
    ("play slay the spire two", "Slay the Spire"),
    ("play deadlock", "Deadlock"),
    ("play warhammer forty thousand darktide", "Darktide"),
    ("play the elder scrolls four oblivion remastered", "Oblivion"),
    ("play risk of rain two", "Risk of Rain"),
]

VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]


def synth(voice, text, path):
    """Windows SAPI -> 16 kHz mono wav. Same trick tests/test_wake.py uses."""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); "
        f"$s.SetOutputToWaveFile('{path}', $f); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   capture_output=True)
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1
        return w.readframes(w.getnframes())


async def transcribe(pcm, api_key, keyterms):
    """One utterance through Flux; returns the EndOfTurn transcript.

    Trailing silence is what ENDS the turn - Flux is a turn model, not a
    stream-to-text - so without it the socket just waits and the probe hangs.
    """
    import websockets
    params = ["model=flux-general-en", f"sample_rate={RATE}",
              "encoding=linear16", "numerals=true", "mip_opt_out=true"]
    from urllib.parse import urlencode
    for k in keyterms:
        params.append(urlencode({"keyterm": k}))
    url = f"{URL}?{'&'.join(params)}"

    async with websockets.connect(
            url, additional_headers={"Authorization": f"Token {api_key}"}) as ws:
        async def feed():
            for i in range(0, len(pcm), CHUNK):
                await ws.send(pcm[i:i + CHUNK])
                await asyncio.sleep(0.01)       # pace it like a live mic
            # Keep the silence COMING. Flux ends a turn on the audio it
            # receives, not on a gap in delivery, so a single trailing buffer
            # and then nothing leaves the turn open until the socket times
            # out - which read as a probe hang.
            while True:
                await ws.send(b"\x00" * CHUNK)
                await asyncio.sleep(0.05)
        send_task = asyncio.create_task(feed())
        try:
            async with asyncio.timeout(45):
                async for msg in ws:
                    if not isinstance(msg, str):
                        continue
                    data = json.loads(msg)
                    if data.get("event") == "EndOfTurn":
                        return data.get("transcript", "")
        finally:
            send_task.cancel()
    return ""


def keyterm_set(mode, voice):
    """The three lists worth comparing: what ships, what used to ship, none."""
    if mode == "none":
        return []
    if mode == "raw":                           # pre-fix: Steam's own strings
        return (["hey jarvis"] + session_runtime.load_titles(voice["keytermCount"])
                + library.query_terms())
    return session_runtime.stt_keyterms(voice, "hey jarvis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-names", action="store_true",
                    help="use the pre-fix keyterms (Steam's strings)")
    ap.add_argument("--no-keyterms", action="store_true")
    ap.add_argument("--voice", default=None, help="one SAPI voice instead of all")
    args = ap.parse_args()

    mode = "raw" if args.raw_names else "none" if args.no_keyterms else "spoken"
    cfg = cglib.load_config()
    secrets = cglib.load_secrets()
    key = secrets.get("deepgramApiKey")
    if not cglib.real_key(key):
        print("no deepgramApiKey - nothing to probe")
        return 1

    keyterms = keyterm_set(mode, cfg["voice"])
    resolve = titles.build_resolver(cfg["voice"]["fuzzyTitleThreshold"])
    if resolve is None:
        print("no installed titles indexed - run library.py sync")
        return 1
    # The gate matches the grammar FIRST and resolves only the {game} slot, so
    # the resolver never sees "play " or the trailing period. Handing it the
    # whole transcript measures a pipeline that does not exist.
    matcher = GrammarMatcher(cfg["voice"])

    voices = [args.voice] if args.voice else VOICES
    print(f"keyterms: {mode} ({len(keyterms)} terms) | voices: {len(voices)} "
          f"| cases: {len(CASES)}\n")

    tmp = Path(tempfile.mkdtemp())
    fails = 0
    for voice in voices:
        print(f"[{voice.split()[1]}]")
        for i, (spoken, expect) in enumerate(CASES):
            wav = tmp / f"{voice.split()[1]}-{i}.wav"
            pcm = synth(voice, spoken, wav)
            heard = asyncio.run(transcribe(pcm, key, keyterms))
            # Exactly the gate's path: grammar match -> {game} slot -> resolve.
            m = matcher.match(heard)
            slot = m[1].get("game") if m and m[0] == "PlayGame" else None
            title = resolve(str(slot))[1] if slot else None
            ok = title is not None and expect.lower() in title.lower()
            fails += not ok
            mark = "ok  " if ok else "MISS"
            print(f"  {mark} {spoken!r}")
            print(f"       heard {heard!r}")
            print(f"       slot {slot!r} -> {title!r}")
    print(f"\n{len(CASES) * len(voices) - fails}/{len(CASES) * len(voices)} "
          f"resolved ({mode} keyterms)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
