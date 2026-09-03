"""Does Flux hear a command well enough to ACT on it?

    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --raw-names     (the old keyterms)
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --no-keyterms   (no boost at all)
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --raw-tags     (SteamSpy's tag strings)
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --capitalized  (proper-noun case)
    .venv\\Scripts\\python -m slopstation.agent.bench.probe_stt --sweep         (fuzzyTitleThreshold)

Live Deepgram, real money, non-deterministic. Windows SAPI speaks each
utterance, so this measures the STT config, not tonight's room.
Exit code is the number of failing probes.

2026-08-14: keyterms were Steam's own strings, so Flux was taught ARMORED CORE
VI FIRES OF RUBICON while the couch says "armored core six" - ARMORED CORSICS,
ARMOR CORE 6, "thyroid core 6", and 11 of 12 launches missed.

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
from slopstation.agent.speech import session_runtime
from slopstation.agent.speech.grammar_gate import GrammarMatcher
from slopstation.agent.tools import library, titles

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


def _capitalized(term):
    """Deepgram documents proper-noun capitalization and says keyterm case
    influences the transcript; keyterm_forms lowercases instead. The
    2026-08-14 measurement that settled that moved case, numerals AND the
    subtitle at once, so case alone was never tested. Free to try - the gate
    and both resolvers run spoken_form first, so nothing downstream sees it."""
    return " ".join(w[:1].upper() + w[1:] for w in term.split())


def _pre_branch_tags(voice):
    """The vocabulary before spoken_form and GENERIC_TERMS reached the tag
    words: SteamSpy's own strings, 'Rogue-like' and 'Action' included."""
    catalog = library.Catalog.load()
    terms = ["hey jarvis"]
    for name in session_runtime.load_titles(voice["keytermCount"], catalog.installed):
        terms += titles.keyterm_forms(name)
    terms += [
        titles.spoken_form(c["name"]) for c in catalog.collections if c.get("name")
    ]
    terms += library.query_terms(session_runtime.QUERY_TERM_SLOTS)
    return list(dict.fromkeys(terms))[: session_runtime.MAX_KEYTERMS]


def keyterm_set(mode, voice):
    """The lists worth comparing: what ships, what shipped before it, the two
    open questions, none."""
    if mode == "none":
        return []
    if mode == "raw":  # pre-fix: Steam's own strings
        return (
            ["hey jarvis"]
            + session_runtime.load_titles(voice["keytermCount"])
            + library.query_terms(session_runtime.QUERY_TERM_SLOTS)
        )
    if mode == "raw-tags":
        return _pre_branch_tags(voice)
    if mode == "capitalized":
        return [
            _capitalized(t) for t in session_runtime.stt_keyterms(voice, "hey jarvis")
        ]
    return session_runtime.stt_keyterms(voice, "hey jarvis")


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
    ap.add_argument(
        "--raw-names",
        action="store_true",
        help="use the pre-fix keyterms (Steam's strings)",
    )
    ap.add_argument("--no-keyterms", action="store_true")
    ap.add_argument(
        "--raw-tags",
        action="store_true",
        help="tag words as SteamSpy writes them (pre-spoken_form)",
    )
    ap.add_argument(
        "--capitalized",
        action="store_true",
        help="the shipping forms, proper-noun capitalization",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="vary fuzzyTitleThreshold over the same transcripts",
    )
    ap.add_argument("--voice", default=None, help="one SAPI voice instead of all")
    args = ap.parse_args()

    mode = (
        "raw"
        if args.raw_names
        else "none"
        if args.no_keyterms
        else "raw-tags"
        if args.raw_tags
        else "capitalized"
        if args.capitalized
        else "spoken"
    )
    cfg = config.load()
    key = config.secrets().get("deepgramApiKey")
    if not config.real_key(key):
        print("no deepgramApiKey - nothing to probe")
        return 1

    voice_cfg = cfg["voice"]
    keyterms = keyterm_set(mode, voice_cfg)
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
