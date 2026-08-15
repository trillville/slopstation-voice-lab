"""Does the assistant answer game questions with a FACT tool in the same
breath, and reserve the background worker for judgment?

    .venv\\Scripts\\python bench\\probe_tool_select.py
    .venv\\Scripts\\python bench\\probe_tool_select.py --provider openai --trials 8

Live model, real money, non-deterministic - see harness.py; dry_run=True, so
nothing reaches the gaming PC.

WHY: the data lane (search_store / list_games / get_game_details facets) only
pays off if the model REACHES for it instead of either answering from memory or
queueing a background task for a fact one call settles. The two failure modes
this guards: a filterable "find me a ..." over-escalating to background_task,
and a feed question ("anything on sale?") that no tool touched. It also guards
the other direction - a genuine multi-source judgment SHOULD still escalate.

Exit code is the number of failing probes, so it can gate a prompt or a tool
description change.
"""
import argparse
import sys

import harness

DATA = ("search_store", "list_games", "get_game_details")

# (utterance, want, why). want="data": a DATA tool must fire and background_task
# must NOT. want="research": background_task is the correct call.
PROBES = [
    ("Anything on my wishlist that's on sale right now?", "data",
     "wishlist-on-sale is a feed read, not research"),
    ("What's on sale on Steam at the moment?", "data", "specials feed"),
    ("What have I been playing the last couple of weeks?", "data",
     "recently-played feed"),
    ("Find me a co-op roguelike under twenty bucks.", "data",
     "Steam's own filters answer this - search_store, not the worker"),
    ("How long does Hades take to beat?", "data",
     "a how-long-to-beat fact - get_game_details facet"),
    ("Can you go do a deep dive comparing what critics and players are saying "
     "about Silksong across a bunch of sites, and tell me if it's worth it?",
     "research", "explicit multi-source judgment - background_task is right"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(harness.assistant.BACKENDS))
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int, default=8)
    a = ap.parse_args()

    cfg, secrets = harness.load()
    provider, model = harness.resolve(cfg, a.provider, a.model)

    print(f"probe_tool_select: {provider}/{model}, {a.trials} trials, dry_run=True")
    print("facts answered in-breath with a data tool; judgment escalates\n")
    failures = 0
    for utterance, want, why in PROBES:
        good_n, samples = 0, []
        for _ in range(a.trials):
            try:
                calls, reply, _ = harness.run_one(
                    cfg, secrets, provider, model, utterance)
                names = [name for name, _ in calls]
            except Exception as e:                  # an API blip is not a verdict
                names, reply = ["ERROR"], repr(e)[:80]
            used_data = any(n in DATA for n in names)
            escalated = "background_task" in names
            good = escalated if want == "research" else (used_data and not escalated)
            good_n += good
            samples.append((names, reply))
        ok = good_n == a.trials
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {why}")
        print(f"       \"{utterance[:76]}\"")
        print(f"       want {want.upper()}; correct in {good_n}/{a.trials}")
        if not ok:
            for names, reply in samples:
                print(f"         tools={names or '[]'} reply={reply[:80]!r}")
        print()
    print(f"{len(PROBES) - failures}/{len(PROBES)} probes correct")
    return failures


if __name__ == "__main__":
    sys.exit(main())
