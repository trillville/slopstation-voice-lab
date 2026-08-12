"""Does the assistant ACT on a question ABOUT an action?

NOT part of the blind suite. This one calls the real model, costs real money,
and is non-deterministic - which is exactly why it exists. The blind suite can
prove the tool boundary refuses a bad appid; only a live probe can prove the
model does not reach for `end_session` when the user asked what end_session
does.

    .venv\\Scripts\\python bench\\probe_intent.py
    .venv\\Scripts\\python bench\\probe_intent.py --provider openai --trials 8

Probe the provider you ACTUALLY RUN. config.json is per-machine and
untracked, so the default here follows your local `assistantProvider`; a
prompt that behaves on Haiku can still act on a question under GPT-5.6.

dry_run=True throughout, so nothing reaches the gaming PC even when the model
decides to end a session. Each trial builds a fresh backend, so conversation
state cannot leak between probes and make a flaky failure look fixed.

WHY THESE PROBES (2026-08-11, all three observed live):

  * A 25-minute session ended because the user asked "what's the tool you
    would run to take me out of a gaming session". The model explained the
    command AND ran it. Reproduced at 1/3 - which is the lesson: a one-in-
    three action bug looks like "worked fine" in any hand test.
  * "How do I get back to my office setup?" switched input to the pc, 3/3,
    which STARTS a session - the exact opposite of the question.
  * "I'm done playing, put it back to the office" switched the TV to Apple
    TV, 3/3. The office is the desk, not a TV input, and nothing in the
    prompt said so.

Exit code is the number of failing probes, so this can gate a prompt change.
"""
import argparse
import sys

import harness

# (utterance, must_act, why). must_act=False means ANSWERING is correct and
# touching a tool is the bug; True means the request is real and refusing to
# act would be an equally bad regression in the other direction.
PROBES = [
    ("What is the command or, like, what's the tool that you would run, um, "
     "to take me out of a gaming session and back to using it in my office?",
     False, "the incident utterance, verbatim from the transcript"),
    ("What happens if I say end session?", False, "asking what a command does"),
    ("How do I get back to my office setup?", False, "asking how, not telling"),
    ("End the session.", True, "plain imperative - MUST still work"),
    ("Okay I'm done playing, put it back to the office.", True,
     "conversational but a real request - MUST still work"),
]

DISRUPTIVE = ("end_session", "start_session", "switch_input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(harness.assistant.BACKENDS))
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
