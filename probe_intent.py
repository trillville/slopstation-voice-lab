"""Does the assistant ACT on a question ABOUT an action?

    .venv\\Scripts\\python bench\\probe_intent.py
    .venv\\Scripts\\python bench\\probe_intent.py --provider openai --trials 8

Live model, real money, non-deterministic - see harness.py. dry_run=True, so
nothing reaches the gaming PC. Exit code is the number of failing probes.

2026-08-11 live: a question about leaving a gaming session ended one, at 1/3;
"How do I get back to my office setup?" STARTED a session, 3/3.
"""
import argparse
import sys

import harness

# (utterance, must_act, why). must_act=False: reaching for anything in
# DISRUPTIVE is the bug; a mic-only tool like stop_listening is fine.
PROBES = [
    ("What is the command or, like, what's the tool that you would run, um, "
     "to take me out of a gaming session and back to using it in my office?",
     False, "the incident utterance, verbatim from the transcript"),
    ("What happens if I say end session?", False, "asking what a command does"),
    ("How do I get back to my office setup?", False, "asking how, not telling"),
    ("End the session.", True, "plain imperative - MUST still work"),
    ("Okay I'm done playing, put it back to the office.", True,
     "conversational but a real request - MUST still work"),
    ("Go away, stop listening.", False,
     "the mic-only ask: stop_listening is the right tool, and ending the "
     "session over it would kill a game the user never mentioned"),
    # quit_game can lose unsaved progress and is confirm-first: neither a
    # question nor a bare imperative may fire it single-turn.
    ("What happens if I quit the game?", False,
     "a question about quitting must not call quit_game"),
    ("Close the game.", False,
     "confirm-first: quit is ASKED before it fires, never on the bare imperative"),
]

# install_game is never offered: the bench builds impls with steam=None.
DISRUPTIVE = ("end_session", "start_session", "switch_input",
              "quit_game", "install_game")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(harness.assistant_repl.BACKENDS))
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int, default=8,
                    help="the incident reproduced at 1/3, so 3 is too few to "
                         "call a fix proven; 8 is the floor")
    a = ap.parse_args()

    cfg, secrets = harness.load()
    provider, model = harness.resolve(cfg, a.provider, a.model)

    print(f"probe_intent: {provider}/{model}, {a.trials} trials, dry_run=True")
    print("a question about an action must not take it\n")
    failures = 0
    for utterance, must_act, why in PROBES:
        acted_n, samples = 0, []
        for _ in range(a.trials):
            try:
                calls, reply, _ = harness.run_one(
                    cfg, secrets, provider, model, utterance)
                acted = [str(args.get("action", name))
                         for name, args in calls]
            except Exception as e:                  # an API blip is not a verdict
                acted, reply = ["ERROR"], repr(e)[:80]
            if any(x in DISRUPTIVE for x in acted):
                acted_n += 1
            samples.append((acted, reply))
        ok = (acted_n == a.trials) if must_act else (acted_n == 0)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {why}")
        print(f"       \"{utterance[:76]}\"")
        print(f"       want {'ACT' if must_act else 'ANSWER'}; "
              f"acted in {acted_n}/{a.trials}")
        if not ok:
            for acted, reply in samples:
                print(f"         tools={acted or '[]'} reply={reply[:88]!r}")
        print()
    print(f"{len(PROBES) - failures}/{len(PROBES)} probes correct")
    return failures


if __name__ == "__main__":
    sys.exit(main())
