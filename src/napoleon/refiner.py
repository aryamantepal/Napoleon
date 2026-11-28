"""The continual harness's refiner: a text-only pass of the same model over recent play.

On a schedule it reads the last ~50 trajectory steps, a tool-failure section and stall
heuristics, then runs four independent passes (strategy prompt, sub-agents, skills, memory),
each in its own try/except so one bad reply costs only that pass. The prompt pass returns a
full markdown rewrite. The others return JSON ``{analysis, add/create, update, retire}``,
parsed leniently. Every edit goes through ``napoleon.harness`` for validation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from napoleon import harness as hx
from napoleon.llm import LLMClient, clean_reply, extract_json
from napoleon.skills import verify_skill
from napoleon.usage import Usage

PASSES = ("prompt", "subagents", "skills", "memory")
CLIP = 300


def clip(text: str | None, n: int = CLIP) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


class Schedule(BaseModel):
    """Refine at ``warmup``, then every ``interval`` steps until ``until``, then every
    ``late_interval`` steps. ``step`` counts actions taken so far in the game."""

    warmup: int = 25
    interval: int = 25
    until: int = 200
    late_interval: int = 100

    def due(self, step: int) -> bool:
        if step <= 0 or step < self.warmup:
            return False
        if step <= self.until:
            return (step - self.warmup) % max(1, self.interval) == 0
        return (step - self.until) % max(1, self.late_interval) == 0


@dataclass
class StepLog:
    """One step as the refiner sees it."""

    step: int
    turn: int
    score: int
    state: str
    reasoning: str = ""
    action: str = ""
    ok: bool = True
    result: str = ""
    tool_failures: list[str] = field(default_factory=list)


def render_window(window: list[StepLog]) -> str:
    lines = []
    for s in window:
        status = "ok" if s.ok else f"REJECTED {clip(s.result, 120)}"
        lines.append(
            f"[{s.step}] turn {s.turn} | state: {clip(s.state)} | why: {clip(s.reasoning)} | "
            f"action: {clip(s.action)} | result: {status}"
        )
    return "\n".join(lines) or "(no steps yet)"


def tool_failure_section(window: list[StepLog], limit: int = 12) -> str:
    rows = [f"[{s.step}] {clip(f, 160)}" for s in window for f in s.tool_failures]
    if not rows:
        return "No tool failures in this window."
    extra = f"\n... and {len(rows) - limit} more" if len(rows) > limit else ""
    return "\n".join(rows[:limit]) + extra


def stall_flags(window: list[StepLog]) -> list[str]:
    """Plain-language flags for turn stalls, repeated actions and illegal-action streaks."""
    flags: list[str] = []
    if len(window) >= 8:
        advanced = window[-1].turn - window[0].turn
        if advanced == 0:
            flags.append(f"the turn did not advance in {len(window)} steps")
        elif len(window) / advanced > 15:
            flags.append(f"only {advanced} turns passed in {len(window)} steps")
    run = best = 0
    prev = None
    for s in window:
        run = run + 1 if s.action == prev else 1
        prev = s.action
        best = max(best, run)
    if best >= 4:
        flags.append(f"the same action was repeated {best} times in a row")
    streak = worst = 0
    for s in window:
        streak = 0 if s.ok else streak + 1
        worst = max(worst, streak)
    if worst >= 3:
        flags.append(f"{worst} illegal or rejected actions in a row")
    return flags


@dataclass(frozen=True)
class WindowStats:
    steps: int
    illegal_rate: float
    score_trend: float  # score gained per turn over the window


def window_stats(window: list[StepLog]) -> WindowStats:
    if not window:
        return WindowStats(0, 0.0, 0.0)
    bad = sum(1 for s in window if not s.ok)
    turns = max(1, window[-1].turn - window[0].turn)
    return WindowStats(len(window), bad / len(window), (window[-1].score - window[0].score) / turns)


class RollbackPolicy(BaseModel):
    enabled: bool = True
    min_steps: int = 10  # ignore windows shorter than this
    illegal_delta: float = 0.15  # illegal-action rate may rise by this much
    score_margin: float = 0.5  # score trend may fall by this fraction of the old trend

    def worse(self, before: WindowStats, after: WindowStats) -> str | None:
        """Why ``after`` is clearly worse than ``before``, or ``None``."""
        if not self.enabled or min(before.steps, after.steps) < self.min_steps:
            return None
        if after.illegal_rate > before.illegal_rate + self.illegal_delta:
            return f"illegal-action rate {before.illegal_rate:.0%} -> {after.illegal_rate:.0%}"
        if before.score_trend > 0 and after.score_trend < before.score_trend * (
            1 - self.score_margin
        ):
            return f"score trend {before.score_trend:.2f} -> {after.score_trend:.2f} per turn"
        return None


# Lenient parsing of the JSON passes ------------------------------------------------------

_ADD = ("add", "create", "new", "added", "created")
_RETIRE = ("retire", "delete", "remove", "retired", "removed")
_UPDATE = ("update", "updates", "modify", "edit")


def _items(obj: dict, keys: tuple[str, ...]) -> list:
    for k in keys:
        if k in obj and obj[k]:
            v = obj[k]
            return v if isinstance(v, list) else [v]
    return []


def parse_edits(text: str) -> tuple[str, list, list, list]:
    """``(analysis, adds, updates, retires)`` from a reply. Raises ``ValueError`` if no JSON."""
    obj = extract_json(text)
    if obj is None:
        raise ValueError("no JSON object in the reply")
    analysis = obj.get("analysis")
    retires = []
    for r in _items(obj, _RETIRE):
        retires.append(r.get("id", r.get("name")) if isinstance(r, dict) else r)
    return (
        analysis if isinstance(analysis, str) else "",
        [x for x in _items(obj, _ADD) if isinstance(x, dict)],
        [x for x in _items(obj, _UPDATE) if isinstance(x, dict)],
        retires,
    )


def parse_markdown(text: str) -> str:
    """The rewritten strategy: the reply minus think tags and a wrapping code fence."""
    text = clean_reply(text)
    m = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*?)\n?```", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip()


# The refiner -----------------------------------------------------------------------------

REFINER_SYSTEM = """\
You improve the harness of an agent that plays a turn-based strategy game, while it plays.
You only read text: you do not play, and you cannot edit the fixed system prompt below, which
holds the game rules, the action schema and the output format. You can edit four things: the
strategy prompt (markdown the agent reads each step), sub-agents, skills and memory.
Base every change on evidence in the recent steps. Make small, specific improvements and keep
what works. Reply in exactly the format the pass asks for.

=== The agent's fixed system prompt (read-only) ===
"""


@dataclass
class PassResult:
    status: str  # ok | unchanged | error
    outcomes: list[str] = field(default_factory=list)
    analysis: str = ""
    error: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class Refinement:
    harness: hx.Harness
    passes: dict[str, PassResult]
    usage: Usage
    changed: bool


class Refiner:
    def __init__(
        self,
        client: LLMClient,
        fixed_prompt: str,
        limits: hx.Limits | None = None,
        *,
        max_tokens: int = 1500,
        verifier: hx.Verifier | None = verify_skill,
    ) -> None:
        self.client = client
        self.system = REFINER_SYSTEM + fixed_prompt
        self.limits = limits or hx.Limits()
        self.max_tokens = max_tokens
        self.verifier = verifier
        self._spent = Usage()

    def refine(self, h: hx.Harness, window: list[StepLog]) -> Refinement:
        evidence = (
            f"Recent steps (oldest first):\n{render_window(window)}\n\n"
            f"Tool failures:\n{tool_failure_section(window)}\n\n"
            "Stall checks: " + ("; ".join(stall_flags(window)) or "none") + "."
        )
        current = h.copy()
        self._spent = Usage()
        results: dict[str, PassResult] = {}
        passes: dict[str, Callable[[hx.Harness, str], PassResult]] = {
            "prompt": self._prompt_pass,
            "subagents": self._subagent_pass,
            "skills": self._skill_pass,
            "memory": self._memory_pass,
        }
        for name, run in passes.items():
            trial = current.copy()
            try:
                result = run(trial, evidence)
                if result.status == "ok":
                    current = trial
            except Exception as exc:  # one pass failing must not stop the others
                result = PassResult("error", error=f"{type(exc).__name__}: {exc}")
            results[name] = result
        changed = any(r.status == "ok" for r in results.values())
        if changed:
            current.generation = h.generation + 1
        return Refinement(current, results, self._spent, changed)

    # Passes ---------------------------------------------------------------------------

    def _ask(self, user: str) -> str:
        """One text-only call. Its cost is counted even if the reply turns out unusable."""
        reply = self.client.complete(
            [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
            max_tokens=self.max_tokens,
        )
        self._spent = self._spent + reply.usage
        return reply.text

    def _finish(self, analysis: str, outcomes: list[str]) -> PassResult:
        ok = any(o.startswith("ok") for o in outcomes)
        return PassResult("ok" if ok else "unchanged", outcomes, analysis)

    def _prompt_pass(self, h: hx.Harness, evidence: str):
        user = (
            f"PASS: strategy prompt.\nCurrent strategy prompt:\n---\n{h.prompt}\n---\n\n"
            f"{evidence}\n\n"
            "Rewrite the whole strategy prompt in markdown, fixing what the steps show going "
            f"wrong and keeping what works. At most {self.limits.max_prompt_chars} characters. "
            "Reply with the markdown only."
        )
        text = self._ask(user)
        outcomes = hx.set_prompt(h, parse_markdown(text), self.limits)
        return self._finish("", outcomes)

    def _subagent_pass(self, h: hx.Harness, evidence: str):
        have = (
            "\n".join(
                f"- id {g.id} {g.name}: {g.description} (tools {g.allowed_tools}, "
                f"{g.max_turns} turns)"
                for g in h.subagents
            )
            or "(none)"
        )
        user = (
            f"PASS: sub-agents. Current sub-agents:\n{have}\n\n{evidence}\n\n"
            "A sub-agent is a focused handler the main agent can delegate a step to. It follows "
            "its directive with the allowed tools and returns one action or a short note. "
            f"Allowed tools: {list(hx.TOOL_WHITELIST)}. max_turns is the number of model calls "
            f"it may use, 1 to {hx.MAX_SUBAGENT_TURNS}; 1 means a one-step handler.\n"
            'Reply with JSON: {"analysis": "...", "add": [{"name": "snake_case", "description": '
            '"...", "directive": "...", "allowed_tools": [...], "max_turns": 1}], "update": '
            '[{"id": 1, "directive": "..."}], "retire": [id, ...]}. Use empty lists for no change.'
        )
        text = self._ask(user)
        analysis, adds, updates, retires = parse_edits(text)
        out = [hx.add_subagent(h, a, self.limits) for a in adds]
        out += [hx.update_subagent(h, u, self.limits) for u in updates]
        out += [hx.retire_subagent(h, r) for r in retires]
        return self._finish(analysis, out)

    def _skill_pass(self, h: hx.Harness, evidence: str):
        have = (
            "\n".join(
                f"- id {s.id} {s.name}: {s.description} (effectiveness {s.effectiveness:.2f}, "
                f"used {s.uses}, rejected {s.failures})"
                for s in h.skills
            )
            or "(none)"
        )
        user = (
            f"PASS: skills. Current skills:\n{have}\n\n{evidence}\n\n"
            "A skill is a data-only routine in a small DSL that plays several legal actions "
            "without a model call. Skill format: "
            '{"name": "snake_case", "description": "what it does and when", '
            '"params": {"q": "int"}, '
            '"rules": [{"do": {"type": "unit_command", "command": "found_city"}, '
            '"when": {"has_unit": "settler"}, "unit_type": "settler", "blocked_only": false, '
            '"choose": "first"}], "until": {"no_blockers": true}, "max_steps": 20, '
            '"max_turns": 1}. '
            "Conditions: has_unit, cities_below, cities_at_least, turn_at_least, turn_below, "
            "enemy_within, blocker, no_blockers, legal, not, all, any. Choosers: first, "
            "nearest_to, farthest_from_cities, nearest_enemy. A parameter is written $name. "
            "Skills with parameters need an example_args object for testing. Skills are tested "
            "on the fake game and discarded if they fail.\n"
            'Reply with JSON: {"analysis": "...", "create": [<skill with example_args>], '
            '"update": [{"id": 1, "description": "...", "effectiveness": 0.7}], '
            '"retire": [id, ...]}. Use empty lists for no change.'
        )
        text = self._ask(user)
        analysis, adds, updates, retires = parse_edits(text)
        out = [hx.add_skill(h, a, self.limits, self.verifier) for a in adds]
        out += [hx.update_skill(h, u) for u in updates]
        out += [hx.retire_skill(h, r) for r in retires]
        return self._finish(analysis, out)

    def _memory_pass(self, h: hx.Harness, evidence: str):
        have = (
            "\n".join(
                f"- id {m.id} [{m.importance}] {m.title}: {clip(m.content, 120)}" for m in h.memory
            )
            or "(none)"
        )
        user = (
            f"PASS: memory. Current entries:\n{have}\n\n{evidence}\n\n"
            "Memory holds durable facts and lessons the agent can read later (an overview of "
            f"titles is always shown). Keep each entry under {self.limits.max_memory_chars} "
            "characters. Importance is 1 (minor) to 5 (critical).\n"
            'Reply with JSON: {"analysis": "...", "add": [{"title": "...", "content": "...", '
            '"importance": 3}], "update": [{"id": 1, "content": "..."}], "retire": [id, ...]}. '
            "Use empty lists for no change."
        )
        text = self._ask(user)
        analysis, adds, updates, retires = parse_edits(text)
        out = [hx.add_memory(h, a, self.limits) for a in adds]
        out += [hx.update_memory(h, u, self.limits) for u in updates]
        out += [hx.retire_memory(h, r) for r in retires]
        return self._finish(analysis, out)


def summarize(r: Refinement) -> dict[str, Any]:
    return {name: res.as_dict() for name, res in r.passes.items()}
