"""Shared plumbing for the live behavioural probes in this directory.

These are NOT the blind suite: they call the real model, cost real money and
are non-deterministic. That is the point - the blind suite can prove a tool
boundary refuses a bad appid, but only a live probe can prove the model does
not reach for a destructive tool, or does not write itself an impossible
brief.

Every trial builds a FRESH backend. Reusing one carries conversation state
between probes, which makes a flaky failure look fixed.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # .../slopstation
sys.path.insert(0, str(HERE.parent))        # k15/voice
sys.path.insert(0, str(ROOT / "k15"))

import cglib          # noqa: E402
import assistant      # noqa: E402
import dispatch as dp  # noqa: E402


class FakeJobs:
    """Stands in for the Tier-3 JobStore and keeps the briefs.

    Passing jobs=None would make background_task refuse, and the model would
    fall back to answering from memory - so a probe about brief QUALITY would
    silently measure nothing.
    """

    def __init__(self):
        self.briefs = []

    def enqueue(self, task):
        self.briefs.append(task)
        return True, "queued - the result will be announced"


def load():
    cfg = json.loads((ROOT / "k15" / "config.json").read_text(encoding="utf-8"))
    secrets = json.loads(
        (ROOT / "k15" / "secrets.json").read_text(encoding="utf-8"))
    return cfg, secrets


def resolve(cfg, provider=None, model=None):
    """Probe the provider you ACTUALLY RUN. config.json is per-machine and
    untracked, so this follows the local setting unless overridden - a prompt
    that behaves on Haiku can still misfire under GPT-5.6."""
    provider = provider or cfg["voice"]["assistantProvider"]
    return provider, (model or assistant.default_model(cfg, provider))


def run_convo(cfg, secrets, provider, model, utterances):
    """One trial of N turns on ONE backend, so later turns see the earlier
    ones. Returns (calls, replies, jobs); calls is [(tool_name, args), ...]
    in the order the model made them, across all turns.

    NOTE for anything probing what the model remembers: the REPL backends and
    the voice pipeline keep conversation state DIFFERENTLY. OpenAIBackend
    threads it server-side via previous_response_id, so a second turn can see
    the first turn's server-executed searches for free. The voice lane
    rebuilds an LLMContext client-side, where those items are dropped - which
    is the whole reason llm_audit.py exists. So a memory probe here is not a
    faithful proxy for production; it tests the PROMPT rule, not the plumbing.
    """
    log = cglib.CapturingLog("probe")
    jobs = FakeJobs()
    impls = assistant.tool_impls(
        dp.Dispatch(cfg, log, dry_run=True), log, jobs=jobs)
    calls = []
    for name in list(impls):
        inner = impls[name]

        def wrap(args, _n=name, _i=inner):
            calls.append((_n, args))
            return _i(args)
        impls[name] = wrap
    backend = assistant.BACKENDS[provider](
        secrets, model, effort=cfg["voice"]["assistantReasoningEffort"],
        voice=cfg["voice"])
    system = assistant.system_instruction(cfg)
    replies = [(backend.turn(system, u, impls) or "").strip()
               for u in utterances]
    return calls, replies, jobs


def run_one(cfg, secrets, provider, model, utterance):
    """One single-turn trial. Returns (calls, reply, jobs)."""
    calls, replies, jobs = run_convo(cfg, secrets, provider, model,
                                     [utterance])
    return calls, replies[0], jobs
