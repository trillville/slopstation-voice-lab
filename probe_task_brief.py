"""Does the assistant write the background agent an IMPOSSIBLE brief?

    .venv\\Scripts\\python bench\\probe_task_brief.py
    .venv\\Scripts\\python bench\\probe_task_brief.py --provider openai

Live model, real money, non-deterministic - see harness.py. dry_run=True and
a FakeJobs, so nothing is queued and nothing reaches the gaming PC.

Asked to research couch co-op games the user does NOT own, the assistant
queued a brief limiting the worker to the provided catalog - which IS the
library. RULES binds the ASSISTANT to the catalog; the worker is not. The
assertion is about the BRIEF, not the answer.
"""
import argparse
import re
import sys

import harness

# Phrasings that hand the worker the assistant's own leash. Deliberately
# narrow: "check/compare against/exclude what's in the catalog" must not trip.
RESTRICTIONS = [
    # A limiting word anywhere in the ~45 chars before catalog/library; match
    # the SHAPE, not fixed phrasings.
    r"(?:only|solely|exclusively|strictly|restrict\w*|limit\w*|confined"
    r"|must\s+be\s+in)\b[^.]{0,45}(?:catalog|librar)",
    # The same leash backwards: "do not recommend games outside the catalog".
    r"(?:do\s+not|don'?t|never|avoid)\s+\w*\s*(?:recommend|suggest|include"
    r"|consider|go|look)[^.]{0,45}(?:outside|beyond|not\s+in)[^.]{0,25}"
    r"(?:catalog|librar)",
    r"(?:catalog|library)\s+only",
]

# Intent that must survive into the brief, asserted only where the user
# expressed an exclusion. Matches a negation near own/library/catalog.
OWNS = (r"(?:not|n'?t|exclud\w*|without|outside|beyond)\b[^.]{0,45}"
        r"(?:own|librar|catalog)"
        r"|avoid\s+(?:repeat|duplicat|anything\s+they)"
        r"|skip\s+\w+\s+they\s+(?:own|have)"
        r"|new\s+to\s+(?:them|the\s+user)")

PROBES = [
    ("Can you research some good couch co-op games for Steam that I don't "
     "currently own, and get back to me?", OWNS,
     "the incident shape - open-ended, explicitly about unowned games"),
    ("Look into what I should buy next for couch multiplayer and get back "
     "to me later.", OWNS,
     "what to buy next - the library is what they ALREADY bought"),
    ("Do some digging on whether there are good local co-op games coming out "
     "soon, and let me know when you've got something.", None,
     "future releases - cannot possibly be in the catalog"),
]


def check(brief, wanted):
    """-> (ok, reason). Bad if it leashes the worker to the catalog, weak if
    it drops an exclusion the user stated."""
    for pat in RESTRICTIONS:
        m = re.search(pat, brief, re.I)
        if m:
            return False, f"catalog leash: ...{m.group(0)!r}..."
    if wanted and not re.search(wanted, brief, re.I):
        # Heuristic: a leash failure is a verdict, this one means go look.
        return False, "lost the user's 'games I don't own' intent (heuristic)"
    return True, "open brief, intent preserved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(harness.assistant_repl.BACKENDS))
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()

    cfg, secrets = harness.load()
    provider, model = harness.resolve(cfg, a.provider, a.model)
    print(f"probe_task_brief: {provider}/{model}, {a.trials} trials")
    print("an open-ended research request must not reach the worker leashed "
          "to the catalog\n")

    failures = 0
    for utterance, wanted, why in PROBES:
        results = []
        for _ in range(a.trials):
            try:
                calls, reply, jobs = harness.run_one(
                    cfg, secrets, provider, model, utterance)
            except Exception as e:                  # an API blip is not a verdict
                results.append((None, f"ERROR {e!r}"[:90]))
                continue
            if not jobs.briefs:
                # All three asks need research the catalog cannot answer, so
                # an immediate reply is a different failure.
                results.append((None, f"no task queued; said {reply[:70]!r}"))
                continue
            results.append(check(jobs.briefs[-1], wanted) + (jobs.briefs[-1],))
        good = sum(1 for r in results if r[0] is True)
        ok = good == a.trials
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {why}")
        print(f"       \"{utterance[:74]}\"")
        print(f"       usable briefs: {good}/{a.trials}")
        if not ok:
            for r in results:
                print(f"         {r[1]}")
                if len(r) > 2:
                    print(f"           brief: {r[2][:150]!r}")
        print()
    print(f"{len(PROBES) - failures}/{len(PROBES)} probes correct")
    return failures


if __name__ == "__main__":
    sys.exit(main())
