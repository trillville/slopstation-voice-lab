"""Does Flux hear a command well enough to ACT on it?

    .venv\\Scripts\\python bench\\probe_stt.py
    .venv\\Scripts\\python bench\\probe_stt.py --raw-names     (the old keyterms)
    .venv\\Scripts\\python bench\\probe_stt.py --no-keyterms   (no boost at all)
    .venv\\Scripts\\python bench\\probe_stt.py --sweep         (fuzzyTitleThreshold)

Live Deepgram, real money, non-deterministic - see harness.py. Reads nothing
from the room: Windows SAPI speaks each utterance, so this measures the STT
config and not tonight's couch. It cannot tell you the array is aimed; it CAN
tell you a config change moved things, which is what nobody could answer.

WHY (2026-08-14, read out of the logs): keyterms were Steam's own strings, so
Flux was taught ARMORED CORE VI FIRES OF RUBICON while the couch says
"armored core six". The boost half-fired - ARMORED CORSICS, ARMORED COSTS,
ARMOR CORE 6, "thyroid core 6" - and 11 of 12 launches missed.

The bar is never a pretty transcript. It is whether the gate's own path -
grammar match, then slot, then resolver - produces the thing that makes the
room change: an appid, a collection id, a nav kind. A transcript that reads
fine and resolves to nothing is a failure here, which is the point.

NEGATIVES matter as much as the hits, and more when sweeping a threshold.
They are titles the user OWNS but has not installed: the resolver must refuse
them, because the failure mode of a loose threshold is not a miss, it is
launching the wrong game.

Exit code is the number of failing probes, so this can gate a config change.
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

# (utterance, kind, expected). kind picks which of the gate's resolution paths
# has to succeed; expected is matched case-insensitively against what came
# back, or is None for a negative that must resolve to nothing.
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
    # Negatives: owned but NOT installed, so the only correct answer is
    # "nothing". A threshold loose enough to match one of these launches the
    # wrong game, which is worse than any miss.
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
    from urllib.parse import urlencode
    params = ["model=flux-general-en", f"sample_rate={RATE}",
              "encoding=linear16", "numerals=true", "mip_opt_out=true"]
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


def act(matcher, resolve_game, resolve_coll, kind, heard):
    """The gate's own path: grammar match, then slot, then resolver. Returns
    what would actually reach dispatch, or None."""
    m = matcher.match(heard)
    if m is None:
        return None
    intent, slots = m
    if kind == "game":
        return (resolve_game(str(slots["game"]))[1]
                if intent == "PlayGame" and resolve_game else None)
    if kind == "collection":
        return (resolve_coll(str(slots["collection"]))[1]
                if intent == "ShowCollection" and resolve_coll else None)
    return str(slots["target"]) if intent == "Nav" else None


def keyterm_set(mode, voice):
    """The three lists worth comparing: what ships, what used to ship, none."""
    if mode == "none":
        return []
    if mode == "raw":                           # pre-fix: Steam's own strings
        return (["hey jarvis"] + session_runtime.load_titles(voice["keytermCount"])
                + library.query_terms())
    return session_runtime.stt_keyterms(voice, "hey jarvis")


def hear_all(cases, voices, key, keyterms, tmp):
    """Transcribe every case once. The sweep reuses these, so a threshold
    comparison changes exactly one variable and costs nothing extra."""
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
    by_kind = {}
    for voice, spoken, kind, expect, heard in heard_rows:
        got = act(matcher, resolve_game, resolve_coll, kind, heard)
        ok = (got is None) if expect is None else (
            got is not None and expect.lower() in str(got).lower())
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
    """Same transcripts, resolver threshold varied. Prints hits and, more
    importantly, whether a looser threshold starts matching a game the user
    owns but never installed - that is a wrong launch, not a miss."""
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
        print(f"{th:>7}  {hits:>4}/{total:<4}  {false:>12}   "
              f"{', '.join(bad)[:60]}{mark}")
    print(f"\nshipping fuzzyTitleThreshold = {voice_cfg['fuzzyTitleThreshold']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-names", action="store_true",
                    help="use the pre-fix keyterms (Steam's strings)")
    ap.add_argument("--no-keyterms", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="vary fuzzyTitleThreshold over the same transcripts")
    ap.add_argument("--voice", default=None, help="one SAPI voice instead of all")
    args = ap.parse_args()

    mode = "raw" if args.raw_names else "none" if args.no_keyterms else "spoken"
    cfg = cglib.load_config()
    key = cglib.load_secrets().get("deepgramApiKey")
    if not cglib.real_key(key):
        print("no deepgramApiKey - nothing to probe")
        return 1

    voice_cfg = cfg["voice"]
    keyterms = keyterm_set(mode, voice_cfg)
    resolve_game = titles.build_resolver(voice_cfg["fuzzyTitleThreshold"])
    resolve_coll = titles.build_collection_resolver(voice_cfg["fuzzyTitleThreshold"])
    if resolve_game is None:
        print("no installed titles indexed - run library.py sync")
        return 1
    # The gate matches the grammar FIRST and resolves only the slot, so the
    # resolver never sees "play " or the trailing period. Handing it the whole
    # transcript measures a pipeline that does not exist.
    matcher = GrammarMatcher(voice_cfg)

    voices = [args.voice] if args.voice else VOICES
    cases = [c for c in CASES if c[1] == "game"] if args.sweep else CASES
    print(f"keyterms: {mode} ({len(keyterms)} terms) | voices: {len(voices)} "
          f"| cases: {len(cases)}\n")

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
