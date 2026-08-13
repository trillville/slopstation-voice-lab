"""Shared plumbing for the live behavioural probes in this directory.

Not the blind suite: these call the real model, cost real money and are
non-deterministic - only a live probe catches the model reaching for a
destructive tool. Every trial builds a FRESH backend; reusing one carries
conversation state between probes and makes a flaky failure look fixed.
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
    """Stands in for the Tier-3 JobStore and keeps the briefs. jobs=None makes
    background_task refuse, so a probe about brief QUALITY measures nothing.
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
    """Follows the local, untracked config.json unless overridden - a prompt
    that behaves on Haiku can still misfire under GPT-5.6."""
    provider = provider or cfg["voice"]["assistantProvider"]
    return provider, (model or assistant.default_model(cfg, provider))


def run_convo(cfg, secrets, provider, model, utterances):
    """One trial of N turns on ONE backend, so later turns see the earlier
    ones. Returns (calls, replies, jobs); calls is [(tool_name, args), ...]
    in the order the model made them, across all turns.

    Memory probes here test the PROMPT rule, not production: OpenAIBackend
    threads state server-side via previous_response_id, while the voice lane
    rebuilds an LLMContext client-side and drops those items.
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
