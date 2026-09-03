"""Does Flux hear a command well enough to ACT on it?

    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --no-keyterms   (no boost at all)
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --sweep         (fuzzyTitleThreshold)

Live Deepgram, real money, non-deterministic. Windows SAPI speaks each
utterance, so this measures the STT config, not tonight's room.
Exit code is the number of failing probes.

Keyterms are the spoken forms the couch says ("armored core 6"), not Steam's
own strings (ARMORED CORE VI FIRES OF RUBICON): Flux is taught what it will
hear.

The bar is the gate's own path - grammar, slot, resolver - producing an appid,
collection id or nav kind; a transcript that reads fine and resolves to
nothing is a failure. Negatives are titles the user OWNS but has not
installed: a loose threshold launches the wrong game.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from slopstation import config
from slopstation.agent.speech import audio
from slopstation.agent.speech.grammar_gate import GrammarMatcher
from slopstation.agent.speech.keyterms import stt_keyterms
from slopstation.agent.tools import titles

URL = "wss://api.deepgram.com/v2/listen"
RATE = 16000
CHUNK = 3200  # 100 ms of s16 mono

# (utterance, kind, expected). kind picks which resolution path must succeed;
# expected is matched case-insensitively, or None for a negative that must
# resolve to nothing.
CASES = [
    ("play armored core six", "game", "ARMORED CORE"),
    ("i want to play armored core six", "game", "ARMORED CORE"),
    ("open armored core six", "game", "ARMORED CORE"),
    ("play hades two", "game", "Hades II"),
    ("play baldurs gate three", "game", "Baldur's Gate 3"),
    ("play slay the spire two", "game", "Slay the Spire"),
    ("play deadlock", "game", "Deadlock"),
    ("play warhammer forty thousand darktide", "game", "Darktide"),
    ("play the elder scrolls four oblivion remastered", "game", "Oblivion"),
    ("play risk of rain two", "game", "Risk of Rain"),
    ("show me my mech collection", "collection", "mech"),
    ("show me my rpg collection", "collection", "RPG"),
    ("open my shooter collection", "collection", "shooter"),
    ("show me my downloads", "nav", "downloads"),
    ("show me the store", "nav", "store"),
    ("go to my library", "nav", "library"),
    # Negatives: owned but NOT installed, so the only answer is "nothing".
    ("play counter strike", "game", None),
    ("play half life two", "game", None),
    ("play team fortress two", "game", None),
    ("play portal two", "game", None),
]

SWEEP_THRESHOLDS = [78, 82, 85, 87, 90]
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
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True
    )
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1
        return w.readframes(w.getnframes())


async def transcribe(pcm, api_key, keyterms):
    """One utterance through Flux; returns the EndOfTurn transcript. Trailing
    silence is what ENDS the turn - Flux is a turn model, not stream-to-text -
    so without it the socket just waits."""
    from urllib.parse import urlencode

    import websockets

    params = [
        "model=flux-general-en",
        f"sample_rate={RATE}",
        "encoding=linear16",
        "numerals=true",
        "mip_opt_out=true",
    ]
    for k in keyterms:
        params.append(urlencode({"keyterm": k}))
    url = f"{URL}?{'&'.join(params)}"

    async with websockets.connect(
        url, additional_headers={"Authorization": f"Token {api_key}"}
    ) as ws:

        async def feed():
            for i in range(0, len(pcm), CHUNK):
                await ws.send(pcm[i : i + CHUNK])
                await asyncio.sleep(0.01)  # pace it like a live mic
            # Keep the silence COMING: Flux ends a turn on the audio it
            # receives, not on a gap in delivery.
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


def act(matcher, resolve_game, resolve_coll, kind, heard):
    """Grammar match, then slot, then resolver - what would reach dispatch."""
    m = matcher.match(heard)
    if m is None:
        return None
    intent, slots = m
    if kind == "game":
        return (
            resolve_game(str(slots["game"]))[1]
            if intent == "PlayGame" and resolve_game
            else None
        )
    if kind == "collection":
        return (
            resolve_coll(str(slots["collection"]))[1]
            if intent == "ShowCollection" and resolve_coll
            else None
        )
    return str(slots["target"]) if intent == "Nav" else None


def hear_all(cases, voices, key, keyterms, tmp):
    """Transcribe every case once; the sweep reuses these, so a threshold
    comparison changes exactly one variable."""
    out = []
    for voice in voices:
        for i, (spoken, kind, expect) in enumerate(cases):
            wav = tmp / f"{voice.split()[1]}-{i}.wav"
            pcm = synth(voice, spoken, wav)
            heard = asyncio.run(transcribe(pcm, key, keyterms))
            out.append((voice, spoken, kind, expect, heard))
    return out


def report(heard_rows, matcher, resolve_game, resolve_coll):
    fails = 0
    by_kind: dict = {}
    for voice, spoken, kind, expect, heard in heard_rows:
        got = act(matcher, resolve_game, resolve_coll, kind, heard)
        ok = (
            (got is None)
            if expect is None
            else (got is not None and expect.lower() in str(got).lower())
        )
        fails += not ok
        tally = by_kind.setdefault("negative" if expect is None else kind, [0, 0])
        tally[0] += ok
        tally[1] += 1
        print(f"  {'ok  ' if ok else 'MISS'} [{voice.split()[1]:5}] {spoken!r}")
        print(f"        heard {heard!r} -> {got!r}")
    print()
    for k, (good, total) in sorted(by_kind.items()):
        print(f"  {k:11} {good}/{total}")
    return fails


def sweep(heard_rows, matcher, voice_cfg):
    """Same transcripts, resolver threshold varied. Flags a threshold loose
    enough to match a game the user owns but never installed."""
    print(f"\n{'thresh':>7}  {'resolved':>9}  {'FALSE MATCH':>12}   detail")
    for th in SWEEP_THRESHOLDS:
        rg = titles.build_resolver(th)
        rc = titles.build_collection_resolver(th)
        hits = total = false = 0
        bad = []
        for _, spoken, kind, expect, heard in heard_rows:
            got = act(matcher, rg, rc, kind, heard)
            if expect is None:
                if got is not None:
                    false += 1
                    bad.append(f"{spoken!r}->{got!r}")
            else:
                total += 1
                hits += got is not None and expect.lower() in str(got).lower()
        mark = "  <-- unsafe" if false else ""
        print(
            f"{th:>7}  {hits:>4}/{total:<4}  {false:>12}   {', '.join(bad)[:60]}{mark}"
        )
    print(f"\nshipping fuzzyTitleThreshold = {voice_cfg['fuzzyTitleThreshold']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-keyterms", action="store_true")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="vary fuzzyTitleThreshold over the same transcripts",
    )
    ap.add_argument("--voice", default=None, help="one SAPI voice instead of all")
    args = ap.parse_args()

    cfg = config.load()
    key = config.secrets().get("deepgramApiKey")
    if not config.real_key(key):
        print("no deepgramApiKey - nothing to probe")
        return 1

    voice_cfg = cfg["voice"]
    mode = "no" if args.no_keyterms else "shipping"
    keyterms = (
        []
        if args.no_keyterms
        else stt_keyterms(voice_cfg, audio.wake_phrase(voice_cfg["wakeModel"]))
    )
    resolve_game = titles.build_resolver(voice_cfg["fuzzyTitleThreshold"])
    resolve_coll = titles.build_collection_resolver(voice_cfg["fuzzyTitleThreshold"])
    if resolve_game is None:
        print("no installed titles indexed - run library sync")
        return 1
    # The gate matches the grammar FIRST and resolves only the slot, so the
    # resolver never sees "play " or the trailing period.
    matcher = GrammarMatcher(voice_cfg)

    voices = [args.voice] if args.voice else VOICES
    cases = [c for c in CASES if c[1] == "game"] if args.sweep else CASES
    print(
        f"keyterms: {mode} ({len(keyterms)} terms) | voices: {len(voices)} "
        f"| cases: {len(cases)}\n"
    )

    tmp = Path(tempfile.mkdtemp())
    rows = hear_all(cases, voices, key, keyterms, tmp)
    if args.sweep:
        sweep(rows, matcher, voice_cfg)
        return 0
    fails = report(rows, matcher, resolve_game, resolve_coll)
    print(f"\n{len(rows) - fails}/{len(rows)} correct ({mode} keyterms)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
