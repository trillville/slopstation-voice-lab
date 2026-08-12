"""Does the assistant write the background agent an IMPOSSIBLE brief?

    .venv\\Scripts\\python bench\\probe_task_brief.py
    .venv\\Scripts\\python bench\\probe_task_brief.py --provider openai

Live model, real money, non-deterministic - see harness.py. dry_run=True and
a FakeJobs, so nothing is queued and nothing reaches the gaming PC.

WHY (2026-08-11, trace 4fb8776a): the user asked for couch co-op games they
do NOT own, was given five invented titles, called it out, and said "why
don't you research it and get back to me". The assistant queued this:

    "Research Steam couch co-op games the user does not currently own,
     USING ONLY GAMES IN THE PROVIDED CATALOG. ..."

The catalog IS the user's library, so that brief asks for games they do not
own drawn exclusively from the games they own. Empty set, by construction.

The cause is a rule leak, not a model whim. RULES binds the ASSISTANT to the
catalog; the background agent is deliberately not bound (AGENTS.md hands it
the library as a resource and it has web access). The tool asked for "every
constraint the user said" and the model exported its own.

So the assertion is about the BRIEF, not the answer: an open-ended research
request must reach the worker without the assistant's catalog leash on it.
"""
import argparse
import re
import sys

import harness

# Phrasings that hand the worker the assistant's own leash. Deliberately
# narrow: "check the catalog", "compare against the catalog" and "exclude
# what's in the catalog" are all FINE and must not trip this.
RESTRICTIONS = [
    # A limiting word anywhere in the ~45 chars before catalog/library. Match
    # the SHAPE, because listing phrasings loses: an earlier version of this
    # file enumerated "restricted to", "limited to", "only games in" - and
    # then "limited STRICTLY to games in the provided catalog" walked
    # straight through the gap, from a real pre-fix brief.
    r"(?:only|solely|exclusively|strictly|restrict\w*|limit\w*|confined"
    r"|must\s+be\s+in)\b[^.]{0,45}(?:catalog|librar)",
    # The same leash worn backwards: "do not recommend games outside the
    # catalog". Reads like a negation, means a restriction - and it slipped
    # past the ownership check for exactly that reason.
    r"(?:do\s+not|don'?t|never|avoid)\s+\w*\s*(?:recommend|suggest|include"
    r"|consider|go|look)[^.]{0,45}(?:outside|beyond|not\s+in)[^.]{0,25}"
    r"(?:catalog|librar)",
    r"(?:catalog|library)\s+only",
]

# Intent that must survive into the brief, per probe. Only asserted where the
# user actually expressed an exclusion - demanding it everywhere was this
# file's own first bug: "what's coming out soon" needs no ownership clause,
# and scoring open, correct briefs as failures would have sent me editing a
# prompt that was already right.
# Match the SHAPE - a negation followed closely by owning/library/catalog -
# not a fixed phrasing. Listing literal phrases was this file's second bug:
# "not already in their library" is a perfect brief and an early version of
# this pattern scored it a failure, which would have had me tightening a
# prompt that was already doing the right thing.
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
    """-> (ok, reason). A brief is bad if it leashes the worker to the
    catalog, and weak if it drops an exclusion the user actually stated."""
    for pat in RESTRICTIONS:
        m = re.search(pat, brief, re.I)
        if m:
            return False, f"catalog leash: ...{m.group(0)!r}..."
    if wanted and not re.search(wanted, brief, re.I):
        # HEURISTIC, unlike the leash check above: no regex really models
        # "expresses an exclusion". Read the printed brief before believing
        # this one - a leash failure is a verdict, this is a prompt to look.
        return False, "lost the user's 'games I don't own' intent (heuristic)"
    return True, "open brief, intent preserved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(harness.assistant.BACKENDS))
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
                # Answering directly is a different failure: these all ask for
                # research explicitly, and the catalog cannot answer any of
                # them, so an immediate reply means it invented or refused.
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
