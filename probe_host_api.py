"""Does the -9999 stream death follow the MME host API, or the machine?

    .venv\\Scripts\\python bench\\probe_host_api.py 7200      (2 hours)
    .venv\\Scripts\\python bench\\probe_host_api.py 600 --dump run1.json
    .venv\\Scripts\\python bench\\probe_host_api.py 60 --channels 6

Runs on the K15. STOP THE VOICE SUPERVISOR FIRST - it holds the same mic, and
the point is to measure the endpoint rather than race the agent for it.

WHY this exists: audio.resolve_device matches on NAME only and takes the first
hit, and MME sorts first - so the wake listener has always bound the array
through MME (index 1). couch.log carries 69 wake_stream_died on 2026-08-14 and
30 on 2026-08-15, and across every one of them the DEVICE STAYS ENUMERATED: the
rebuild rebinds the same name within 5 s and open_audio never once logs
audio_device_wait. That makes it a stream failure on a device that is present,
not a USB disconnect, which puts the host API in the frame - and this is the
cheapest way to indict or clear it.

Both outcomes are useful, which is why it is worth the wall clock:
  - flaps reproduce with the agent down => environmental or host-API, and the
    per-candidate counts say which;
  - flaps do NOT reproduce => the agent's own stream churn is implicated and
    MME is exonerated. Do not read that as "fixed".

WHY concurrent rather than one candidate and then the other: the deaths cluster
(ten inside three minutes on 2026-08-15). Run sequentially and whichever
candidate happens to draw the burst loses - the result would be a time-of-day
artifact wearing a host-API label. Both streams stay open across the same
seconds or the comparison means nothing.

WHY 6 ch by default, when the agent runs mono today. Measured on the array
2026-08-15, PortAudio accepts:

    MME              1 / 2 / 4 / 6 ch      DirectSound   1 / 2 / 4 / 6 ch
    WASAPI           6 ch ONLY (-9998)     WDM-KS        nothing (-9999)

WASAPI shared mode demands the engine's exact format, and the endpoint's
shared-mode format IS the device native one (6 ch / 16 kHz / 16-bit, discrete
mask). So mono is not a setting WASAPI can be soaked at, and mono-vs-6ch would
confound host API with channel count. 6 ch is the ONLY value every candidate
accepts, which makes it the one place host API is the sole variable - and it is
also the only shape a WASAPI production binding could ever take, so the fair
comparison and the real candidate happen to coincide.

--channels 1 is still worth a run: it drops WASAPI and asks the narrower
question of whether channel count alone moves MME. WDM-KS is excluded outright;
it is exclusive-mode and will not open at any width while the audio engine
holds the endpoint, so it has no production form to compare.

WHY one PyAudio instance, never terminated mid-run: audio.rebuild_audio tears
the whole world down on recovery for reasons its docstring records, and this
deliberately does not - a terminate would take the other candidate's stream
with it and destroy the pairing. A probe counts deaths; it does not test
recovery. The cost is that a stale-index reopen is possible here, and it lands
on both candidates equally.

TWO failure modes are counted, because the incident history has both: a read
that RAISES, and a stream that keeps returning bytes that are all zero -
rebuild_audio's "'succeeds' onto a dead endpoint and goes deaf". A real mic has
a noise floor (this array idles around 1.3 RMS), so exact digital silence for
ZOMBIE_CHUNKS is not a quiet room, it is a corpse.

No events are emitted. A bench tool has no business writing prod telemetry
(record_room.py's rule) and a multi-hour soak would flood it; deaths print with
a timestamp instead, so they can be lined up against couch.log by eye.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # .../slopstation
sys.path.insert(0, str(HERE.parent))        # k15/voice
sys.path.insert(0, str(ROOT / "k15"))

import audio          # noqa: E402
import cglib          # noqa: E402

RATE = 16000
CHUNK = 1280                                # audio.py's native 80 ms hop
REOPEN_S = audio.RETRY_S                    # settle exactly as the agent does
CHUNKS_PER_S = RATE / CHUNK                 # 12.5: what a real-time stream owes
ZOMBIE_CHUNKS = 125                         # ~10 s of exact silence
WARMUP_CHUNKS = 5                           # probe_wake_model's rule: a stream
#                                             may hand back zeros while its
#                                             buffer primes, and counting those
#                                             would call every open a corpse.
PACE_SAMPLE = 50                            # ~4 s of audio before timing is
#                                             judged - long enough that a burst
#                                             off a warm buffer cannot convict.
PROGRESS_S = 300
# WDM-KS is deliberately absent - see the header's format matrix.
CANDIDATES = ("MME", "Windows DirectSound", "Windows WASAPI")


def resolve_on_api(pa, fragment, api_name):
    """audio.resolve_device's name-fragment rule, narrowed to ONE host API.

    Its own copy on purpose. resolve_device cannot select a host API - that is
    the production change this probe exists to justify - and a probe that
    presupposed the fix could not be evidence for it."""
    api_idx = next((i for i in range(pa.get_host_api_count())
                    if pa.get_host_api_info_by_index(i)["name"] == api_name),
                   None)
    if api_idx is None:
        return None, None
    frag = fragment.lower()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if (d["hostApi"] == api_idx and d["maxInputChannels"]
                and frag in d["name"].lower()):
            return i, d["name"]
    return None, None


class Candidate:
    """One host API's view of the same physical endpoint, plus its tally."""

    def __init__(self, api, index, name):
        self.api, self.index, self.name = api, index, name
        self.stream = None
        self.chunks = 0
        self.deaths = []                    # [{at, err}]
        self.zombies = []                   # [{at, chunks}]
        self.reopen_fails = 0
        self.deaf_s = 0.0
        self.silent = 0
        self.retired = None                 # reason, once it stops being soaked
        self.opened_at = None
        self.since_open = 0

    def open(self, pa, channels):
        import pyaudio
        self.stream = pa.open(format=pyaudio.paInt16, channels=channels,
                              rate=RATE, input=True, frames_per_buffer=CHUNK,
                              input_device_index=self.index)
        self.opened_at, self.since_open = time.monotonic(), 0

    def free_running(self):
        """Delivering faster than real time = not capturing, just handing back
        buffers on demand. Measured since the last OPEN, not since the run
        started, so time spent deaf between reopens cannot drag the rate down
        and hide a free-runner. The 2x margin and PACE_SAMPLE together are what
        keep a merely fast machine from being convicted on timing."""
        span = time.monotonic() - (self.opened_at or time.monotonic())
        return self.since_open / max(span, 1e-9) > 2 * CHUNKS_PER_S

    def summary(self, elapsed):
        heard = max(elapsed - self.deaf_s, 1e-9)
        # A 16 kHz stream on an 80 ms hop owes 12.5 chunks/s and blocks to pace
        # itself. A rate far above that is a stream that is not capturing at
        # all, just handing back buffers as fast as it is asked - the same
        # "'succeeds' onto a dead endpoint" corpse rebuild_audio describes,
        # caught by its timing rather than by its contents. DirectSound at 6 ch
        # free-ran at ~125 chunks/s on 2026-08-15; MME and WASAPI paced.
        return {
            "host_api": self.api, "index": self.index, "device": self.name,
            "chunks": self.chunks,
            "chunks_per_s": round(self.chunks / heard, 1),
            "paces": abs(self.chunks / heard - CHUNKS_PER_S) < 2.0,
            "retired": self.retired,
            "deaths": len(self.deaths), "zombies": len(self.zombies),
            "reopen_fails": self.reopen_fails,
            "deaf_s": round(self.deaf_s, 1),
            "uptime_pct": round(100.0 * heard / elapsed, 2) if elapsed else 0.0,
            "per_hour": round(len(self.deaths) / max(elapsed / 3600, 1e-9), 2),
            "death_log": self.deaths, "zombie_log": self.zombies,
        }


def stamp():
    return time.strftime("%H:%M:%S")


def soak(cand, pa, channels, stop):
    """Read until stop, reopening on death the way the agent would. Never
    raises: a probe thread that dies takes its own measurement with it, and
    the surviving candidate would look artificially good."""
    while not stop.is_set():
        if cand.stream is None:
            t0 = time.monotonic()
            stop.wait(REOPEN_S)
            if stop.is_set():
                return
            try:
                cand.open(pa, channels)
                print(f"  [{stamp()}] {cand.api}: reopened "
                      f"after {time.monotonic() - t0:.1f}s deaf", flush=True)
            except Exception as e:
                cand.reopen_fails += 1
                print(f"  [{stamp()}] {cand.api}: REOPEN FAILED - {e}",
                      flush=True)
            finally:
                cand.deaf_s += time.monotonic() - t0
            continue
        try:
            data = cand.stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            cand.deaths.append({"at": stamp(), "err": str(e)})
            print(f"  [{stamp()}] {cand.api}: DIED - {e}", flush=True)
            audio.close_stream_quietly(cand.stream)
            cand.stream = None
            continue
        cand.chunks += 1
        cand.since_open += 1
        # TIMING FIRST - it is the reliable tell, and content is not. Soaking
        # DirectSound at 6 ch on 2026-08-15 returned exact zeros on one run and
        # non-zero garbage on the next, so a content test alone cleared it the
        # second time; both runs delivered ~3 million chunks/s against the 12.5
        # a real-time stream owes. Retire rather than reopen: a free-runner
        # mints a fresh event every second of wall clock and would bury the
        # real candidates under thousands of its own over a 2 h soak. The
        # finding is made; repeating it is not more evidence.
        if cand.since_open > PACE_SAMPLE and cand.free_running():
            cand.retired = ("free-running - never paced to real time, so it "
                            "was never capturing")
            print(f"  [{stamp()}] {cand.api}: RETIRED - {cand.retired}",
                  flush=True)
            audio.close_stream_quietly(cand.stream)
            cand.stream = None
            return
        if cand.since_open <= WARMUP_CHUNKS:
            continue
        # A stream that DOES pace and still returns nothing but zeros is the
        # other corpse - rebuild_audio's "'succeeds' onto a dead endpoint".
        # any() short-circuits on the first non-zero byte, so this costs
        # nothing on a live mic and only walks the buffer on a dead one.
        cand.silent = 0 if any(data) else cand.silent + 1
        if cand.silent < ZOMBIE_CHUNKS:
            continue
        cand.zombies.append({"at": stamp(), "chunks": cand.silent})
        print(f"  [{stamp()}] {cand.api}: ZOMBIE - {cand.silent} chunks "
              f"of exact silence, forcing a reopen", flush=True)
        cand.silent = 0
        audio.close_stream_quietly(cand.stream)
        cand.stream = None


def main():
    ap = argparse.ArgumentParser(
        description="Soak MME and WASAPI against the same mic, concurrently.")
    ap.add_argument("seconds", nargs="?", type=int, default=7200)
    ap.add_argument("--channels", type=int, default=6,
                    help="6 = the array's native format and the only width "
                         "every candidate accepts (default); 1 = the width "
                         "the agent runs today, MME/DirectSound only")
    ap.add_argument("--dump", metavar="PATH",
                    help="write the tally as JSON for comparison across runs")
    args = ap.parse_args()

    voice = json.loads((cglib.BASE / "config.json")
                       .read_text(encoding="utf-8-sig"))["voice"]
    frag = voice["inputDeviceName"]

    import pyaudio
    pa = pyaudio.PyAudio()
    print(f"soaking {frag!r} for {args.seconds}s at {args.channels} ch\n")

    cands = []
    for api in CANDIDATES:
        idx, name = resolve_on_api(pa, frag, api)
        if idx is None:
            print(f"  {api}: no input device matching {frag!r} - skipped")
            continue
        c = Candidate(api, idx, name)
        try:
            c.open(pa, args.channels)
        except Exception as e:
            # An ANSWER, not a fault: at --channels 1 this is WASAPI declining
            # anything but the engine format, which is the documented shape of
            # the thing and not a problem with the run.
            print(f"  {api}: index {idx} would not open at "
                  f"{args.channels} ch - {e}")
            continue
        print(f"  {api}: index {idx} open ({name})")
        cands.append(c)

    if len(cands) < 2:
        print("\nfewer than two candidates opened - there is nothing to "
              "compare, and a one-sided count is not evidence.")
        pa.terminate()
        return 1

    stop = threading.Event()
    threads = [threading.Thread(target=soak, args=(c, pa, args.channels, stop),
                                daemon=True, name=c.api) for c in cands]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    print("\nrunning - deaths print as they happen\n")
    try:
        while True:
            # Bounded by what is LEFT, not by PROGRESS_S: waiting the full
            # interval and only then re-checking overshoots every run shorter
            # than the interval, which is every smoke test.
            left = args.seconds - (time.monotonic() - t0)
            if left <= 0:
                break
            stop.wait(min(PROGRESS_S, left))
            if stop.is_set():
                break
            mins = (time.monotonic() - t0) / 60
            tally = " | ".join(f"{c.api} {len(c.deaths)}d/{len(c.zombies)}z"
                               for c in cands)
            print(f"  [{stamp()}] {mins:.0f} / {args.seconds / 60:.0f} min - "
                  f"{tally}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - reporting what was measured so far")
    stop.set()
    for t in threads:
        t.join(timeout=REOPEN_S + 2)
    elapsed = time.monotonic() - t0
    for c in cands:
        # Guarded here rather than teaching close_stream_quietly to accept
        # None: a retired candidate has already closed its own stream, and
        # loosening a production helper for a bench caller's convenience is
        # how the resolver picked up its required= flag.
        if c.stream is not None:
            audio.close_stream_quietly(c.stream)
    pa.terminate()

    out = {"fragment": frag, "channels": args.channels,
           "elapsed_s": round(elapsed, 1),
           "candidates": [c.summary(elapsed) for c in cands]}
    print(f"\n{'host API':<22}{'deaths':>7}{'zombies':>8}{'reopen!':>8}"
          f"{'deaf s':>8}{'uptime':>8}{'per hr':>8}{'chunk/s':>14}")
    for s in out["candidates"]:
        print(f"{s['host_api']:<22}{s['deaths']:>7}{s['zombies']:>8}"
              f"{s['reopen_fails']:>8}{s['deaf_s']:>8}"
              f"{s['uptime_pct']:>7}%{s['per_hour']:>8}"
              f"{s['chunks_per_s']:>14,.1f}")
        if s["retired"]:
            print(f"{'':<22}retired - {s['retired']}")
        elif not s["paces"]:
            print(f"{'':<22}NOT REAL TIME ({CHUNKS_PER_S:.1f} owed) - "
                  f"this stream was not capturing")

    live = [s for s in out["candidates"] if not s["retired"]]
    total = sum(s["deaths"] + s["zombies"] for s in live)
    hours = elapsed / 3600
    if len(live) < 2:
        print(f"\nOnly {len(live)} candidate(s) survived the run - there is no "
              f"comparison left to draw, whatever the counts above say.")
    elif not total:
        # The logged rate is ~2/h. Silence over a short run is the expected
        # outcome even when MME is guilty, and reading it as a clean bill of
        # health is how this probe would produce a wrong answer.
        print(f"\nNo events in {hours:.1f}h. At the ~2/h logged on 2026-08-15 "
              f"that is unsurprising below ~2h and settles nothing - run it "
              f"longer, or run it while the agent is up to test the other "
              f"hypothesis.")
    else:
        print(f"\n{total} event(s) in {hours:.1f}h. A verdict needs the counts "
              f"to be lopsided AND the run long enough to have caught a "
              f"cluster - one death each is a coin flip, not a finding.")

    if args.dump:
        Path(args.dump).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
