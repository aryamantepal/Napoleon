"""A small, safe skill language for macro-plans over the six actions.

A skill is data, not code. It has a name, a description, typed parameters and an ordered
list of rules. Running a skill never executes anything the model wrote: each step looks at
the legal actions in the current snapshot and picks the first rule whose pattern matches
one of them. That makes every skill action legal by construction, bounded in length, and
safe to load from a file an LLM helped write.

A rule reads as: *when* the condition holds, take a legal action matching ``do`` (fields of
an action, with ``"$name"`` standing for a parameter), optionally only for units of
``unit_type`` or only for units and cities that have a blocker, picking among the matches
with ``choose``.

``verify_skill`` plays the skill in fake-game rollouts before a library accepts it.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from napoleon.hexgrid import distance
from napoleon.schema import Action as ActionT
from napoleon.schema import (
    Coord,
    EndTurn,
    MoveUnit,
    SetCivic,
    SetProduction,
    SetResearch,
    Snapshot,
    UnitCommand,
)

ACTION_CLASSES: dict[str, type] = {
    cls.model_fields["type"].default: cls
    for cls in (MoveUnit, UnitCommand, SetProduction, SetResearch, SetCivic, EndTurn)
}

MAX_RULES = 12
MAX_DEPTH = 4
MAX_STEPS_LIMIT = 40
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
PARAM_TYPES = {"int": int, "str": str}
CHOOSERS = ("nearest_to", "farthest_from_cities", "nearest_enemy")
CONDITIONS = (
    "has_unit",
    "cities_below",
    "cities_at_least",
    "turn_at_least",
    "turn_below",
    "enemy_within",
    "blocker",
    "no_blockers",
    "legal",
    "not",
    "all",
    "any",
)
BLOCKER_FOR = {"unit_id": "unit_needs_orders", "city_id": "city_needs_production"}


class SkillError(ValueError):
    """A skill is malformed or could not be used. The message is written for the model."""


# The language -----------------------------------------------------------------------


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    do: dict[str, Any]
    when: dict[str, Any] | None = None
    unit_type: str | None = None
    blocked_only: bool = False
    choose: Literal["first"] | dict[str, Any] = "first"


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    params: dict[str, Literal["int", "str"]] = {}
    rules: list[Rule]
    until: dict[str, Any] | None = None
    max_steps: int = 20
    max_turns: int = 1
    uses: int = 0
    verification: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError("name must be snake_case, 3 to 41 characters, starting with a letter")
        return v

    @field_validator("description")
    @classmethod
    def _description(cls, v: str) -> str:
        if not 10 <= len(v) <= 300:
            raise ValueError("description must be 10 to 300 characters")
        return v


def _refs(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value[1:]} if value.startswith("$") else set()
    if isinstance(value, dict):
        return set().union(*(_refs(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_refs(v) for v in value)) if value else set()
    return set()


def _check_condition(cond: Any, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise SkillError(f"conditions nest more than {MAX_DEPTH} deep")
    if not isinstance(cond, dict) or len(cond) != 1:
        raise SkillError(f"a condition is an object with one key from {list(CONDITIONS)}")
    (key, arg), *_ = cond.items()
    if key not in CONDITIONS:
        raise SkillError(f"unknown condition {key!r}; use one of {list(CONDITIONS)}")
    if key in ("all", "any"):
        if not isinstance(arg, list) or not arg:
            raise SkillError(f"{key} takes a non-empty list of conditions")
        for sub in arg:
            _check_condition(sub, depth + 1)
    elif key == "not":
        _check_condition(arg, depth + 1)
    elif key == "legal":
        _check_pattern(arg)


def _check_pattern(pattern: Any) -> None:
    if not isinstance(pattern, dict) or pattern.get("type") not in ACTION_CLASSES:
        raise SkillError(f"an action pattern needs a 'type' from {sorted(ACTION_CLASSES)}")
    fields = ACTION_CLASSES[pattern["type"]].model_fields
    for key in pattern:
        if key not in fields:
            raise SkillError(f"{pattern['type']} has no field {key!r}; fields are {sorted(fields)}")


def validate_skill(skill: Skill) -> Skill:
    """Check everything pydantic cannot. Raises ``SkillError``."""
    if not skill.rules:
        raise SkillError("a skill needs at least one rule")
    if len(skill.rules) > MAX_RULES:
        raise SkillError(f"a skill may have at most {MAX_RULES} rules")
    if not 1 <= skill.max_steps <= MAX_STEPS_LIMIT:
        raise SkillError(f"max_steps must be 1 to {MAX_STEPS_LIMIT}")
    if not 1 <= skill.max_turns <= 5:
        raise SkillError("max_turns must be 1 to 5")
    used: set[str] = set()
    for i, rule in enumerate(skill.rules):
        try:
            _check_pattern(rule.do)
            if rule.when is not None:
                _check_condition(rule.when)
            if isinstance(rule.choose, dict):
                if len(rule.choose) != 1 or next(iter(rule.choose)) not in CHOOSERS:
                    raise SkillError(f"choose is 'first' or an object with one key of {CHOOSERS}")
                if "nearest_to" in rule.choose:
                    target = rule.choose["nearest_to"]
                    if not isinstance(target, dict) or set(target) != {"q", "r"}:
                        raise SkillError("nearest_to takes {'q': .., 'r': ..}")
        except SkillError as exc:
            raise SkillError(f"rule {i}: {exc}") from None
        used |= _refs(rule.do) | _refs(rule.when) | _refs(rule.choose)
    if skill.until is not None:
        _check_condition(skill.until)
        used |= _refs(skill.until)
    missing = used - set(skill.params)
    if missing:
        raise SkillError(f"parameters used but not declared in params: {sorted(missing)}")
    return skill


def parse_skill(data: Any) -> Skill:
    """Build and validate a skill from model output. Raises ``SkillError`` with the reason."""
    if not isinstance(data, dict):
        raise SkillError("a skill must be a JSON object")
    try:
        skill = Skill.model_validate(
            {k: v for k, v in data.items() if k not in ("uses", "verification")}
        )
    except ValidationError as exc:
        err = exc.errors()[0]
        where = ".".join(str(p) for p in err["loc"])
        raise SkillError(f"invalid skill ({where}: {err['msg']})") from None
    return validate_skill(skill)


def bind_args(skill: Skill, args: dict | None) -> dict[str, Any]:
    """Check and coerce call arguments against the skill's declared parameters."""
    args = dict(args or {})
    unknown = set(args) - set(skill.params)
    if unknown:
        raise SkillError(
            f"unknown arguments {sorted(unknown)}; parameters are {list(skill.params)}"
        )
    out: dict[str, Any] = {}
    for name, kind in skill.params.items():
        if name not in args:
            raise SkillError(f"missing argument {name!r} ({kind})")
        value = args[name]
        try:
            if kind == "int" and (isinstance(value, bool | float)):
                raise ValueError
            out[name] = PARAM_TYPES[kind](value)
        except (TypeError, ValueError):
            raise SkillError(f"argument {name!r} must be {kind}, got {value!r}") from None
    return out


def _subst(value: Any, args: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return args[value[1:]]
    if isinstance(value, dict):
        return {k: _subst(v, args) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v, args) for v in value]
    return value


# Running a skill --------------------------------------------------------------------


def _subset(pattern: Any, actual: Any) -> bool:
    if isinstance(pattern, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(v, actual[k]) for k, v in pattern.items()
        )
    return pattern == actual


def _mine(snap: Snapshot):
    return [u for u in snap.units if u.owner == snap.player.id]


def _foreign(snap: Snapshot):
    return [u for u in snap.units if u.owner != snap.player.id]


def holds(cond: dict, snap: Snapshot) -> bool:
    (key, arg), *_ = cond.items()
    if key == "has_unit":
        return any(u.type == arg for u in _mine(snap))
    if key == "cities_below":
        return len(snap.cities) < arg
    if key == "cities_at_least":
        return len(snap.cities) >= arg
    if key == "turn_at_least":
        return snap.turn >= arg
    if key == "turn_below":
        return snap.turn < arg
    if key == "enemy_within":
        return any(
            distance(e.coord, c.coord) <= arg for e in _foreign(snap) for c in snap.cities
        ) or any(distance(e.coord, u.coord) <= arg for e in _foreign(snap) for u in _mine(snap))
    if key == "blocker":
        return any(b.kind == arg for b in snap.blockers)
    if key == "no_blockers":
        return not any(b.kind != "other" for b in snap.blockers)
    if key == "legal":
        return any(_subset(arg, a.model_dump(mode="json")) for a in snap.legal_actions)
    if key == "not":
        return not holds(arg, snap)
    if key == "all":
        return all(holds(c, snap) for c in arg)
    return any(holds(c, snap) for c in arg)  # any


def _action_key(a: ActionT) -> str:
    return a.model_dump_json()


def _pick(rule: Rule, cands: list[ActionT], snap: Snapshot, args: dict) -> ActionT:
    if rule.choose == "first":
        return cands[0]
    (kind, arg), *_ = rule.choose.items()
    moves = [(a, a.to) for a in cands if isinstance(a, MoveUnit)]
    if not moves:
        return cands[0]
    if kind == "nearest_to":
        target = _subst(arg, args)
        goal = Coord(q=int(target["q"]), r=int(target["r"]))
        return min(moves, key=lambda m: (distance(m[1], goal), m[1].q, m[1].r))[0]
    if kind == "farthest_from_cities":
        cities = [c.coord for c in snap.cities]
        return max(
            moves,
            key=lambda m: (min((distance(m[1], c) for c in cities), default=0), -m[1].q, -m[1].r),
        )[0]
    enemies = [e.coord for e in _foreign(snap)]  # nearest_enemy
    if not enemies:
        return cands[0]
    return min(moves, key=lambda m: (min(distance(m[1], e) for e in enemies), m[1].q, m[1].r))[0]


@dataclass
class SkillRun:
    """One invocation of a skill. Call ``next_action`` once per agent step until it is done."""

    skill: Skill
    args: dict[str, Any]
    start_turn: int | None = None
    steps: int = 0
    done: bool = False
    executed: set[str] = field(default_factory=set)
    last_rule: int | None = None

    def next_action(self, snap: Snapshot) -> ActionT | None:
        """The next action, or ``None`` when the skill has nothing more to do (it is then done).

        An identical action is never issued twice in one run, which stops skills that keep
        re-issuing something that stays legal, such as skipping the same unit.
        """
        if self.done:
            return None
        if self.start_turn is None:
            self.start_turn = snap.turn
        if (
            self.steps >= self.skill.max_steps
            or snap.turn - self.start_turn >= self.skill.max_turns
            or (self.skill.until is not None and holds(_subst(self.skill.until, self.args), snap))
        ):
            self.done = True
            return None
        for i, rule in enumerate(self.skill.rules):
            action = self._resolve(rule, snap)
            if action is not None:
                self.steps += 1
                self.last_rule = i
                if not isinstance(action, EndTurn):
                    self.executed.add(_action_key(action))
                return action
        self.done = True
        return None

    def _resolve(self, rule: Rule, snap: Snapshot) -> ActionT | None:
        if rule.when is not None and not holds(_subst(rule.when, self.args), snap):
            return None
        pattern = _subst(rule.do, self.args)
        cands = [
            a
            for a in snap.legal_actions
            if _subset(pattern, a.model_dump(mode="json")) and _action_key(a) not in self.executed
        ]
        if rule.unit_type is not None:
            types = {u.id: u.type for u in _mine(snap)}
            cands = [a for a in cands if types.get(getattr(a, "unit_id", None)) == rule.unit_type]
        if rule.blocked_only:
            blocked = {(b.kind, b.ref_id) for b in snap.blockers if b.ref_id is not None}
            cands = [
                a
                for a in cands
                if any(
                    (kind, getattr(a, field_name, None)) in blocked
                    for field_name, kind in BLOCKER_FOR.items()
                )
            ]
        if not cands:
            return None
        return _pick(rule, cands, snap, self.args)


# The library ------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be", "it"}
)


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower().replace("_", " ")) if w not in _STOP]


class SkillLibrary:
    """Skills stored in one JSON file. Retrieval is TF-IDF over names and descriptions."""

    def __init__(self, path: str | Path | None = None, max_skills: int = 50) -> None:
        self.path = Path(path) if path is not None else None
        self.max_skills = max_skills
        self.skills: dict[str, Skill] = {}
        if self.path is not None and self.path.exists():
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                skill = Skill.model_validate(raw)
                self.skills[skill.name] = skill

    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, name: str) -> bool:
        return name in self.skills

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def add(self, skill: Skill) -> None:
        if skill.name in self.skills:
            raise SkillError(f"a skill named {skill.name!r} already exists; pick another name")
        if len(self.skills) >= self.max_skills:
            raise SkillError(f"the library is full ({self.max_skills} skills)")
        self.skills[skill.name] = skill
        self.save()

    def record_use(self, name: str) -> None:
        skill = self.skills[name]
        self.skills[name] = skill.model_copy(update={"uses": skill.uses + 1})
        self.save()

    def clear(self) -> None:
        self.skills.clear()
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps([s.model_dump(mode="json") for s in self.skills.values()], indent=1)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, self.path)

    def retrieve(self, query: str, k: int = 4) -> list[Skill]:
        """The ``k`` skills whose name and description best match ``query``, best first."""
        docs = {n: tokens(f"{n} {s.description}") for n, s in self.skills.items()}
        if not docs:
            return []
        df = Counter(w for words in docs.values() for w in set(words))
        n_docs = len(docs)

        def vec(words: list[str]) -> dict[str, float]:
            tf = Counter(words)
            return {w: c * (math.log((1 + n_docs) / (1 + df.get(w, 0))) + 1) for w, c in tf.items()}

        q = vec(tokens(query))
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored = []
        for name, words in docs.items():
            d = vec(words)
            dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
            sim = sum(q[w] * d.get(w, 0.0) for w in q) / (qn * dn)
            scored.append((sim, self.skills[name].uses, name))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return [self.skills[name] for sim, _, name in scored[:k] if sim > 0]


# Verification -----------------------------------------------------------------------


@dataclass(frozen=True)
class Verification:
    ok: bool
    reason: str
    fired: int = 0
    scores: dict[str, int] = field(default_factory=dict)
    baselines: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "fired": self.fired,
            "scores": self.scores,
            "baselines": self.baselines,
            "reason": self.reason,
        }


class _SkillThenGreedy:
    """Plays the skill whenever it applies and the greedy sanity agent otherwise."""

    name = "skill-rollout"

    def __init__(self, skill: Skill, args: dict) -> None:
        from napoleon.agents.greedy import GreedyAgent

        self.skill, self.args = skill, args
        self.greedy = GreedyAgent()
        self.run: SkillRun | None = None
        self.fired = 0

    def reset(self, seed: int) -> None:
        pass

    def act(self, snap: Snapshot):
        from napoleon.agents.base import Decision

        for _ in range(2):
            if self.run is None:
                self.run = SkillRun(self.skill, self.args)
            action = self.run.next_action(snap)
            if action is not None:
                self.fired += 1
                return Decision(action, f"skill {self.skill.name}")
            self.run = None
        return self.greedy.act(snap)


def _rollout(agent, scenario: str, turns: int, seed: int) -> tuple[int, int]:
    """Final score and rejected-step count of one fake-game rollout."""
    from napoleon.env import CivEnv
    from napoleon.fake import FakeCivGame

    env = CivEnv(FakeCivGame(max_turns=turns))
    snap = env.reset(scenario, seed)
    agent.reset(seed)
    failed = steps = 0
    limit = turns * 80
    while not snap.done and steps < limit:
        snap, result = env.step(agent.act(snap).action)
        failed += not result.ok
        steps += 1
    if not snap.done:
        raise SkillError("the skill stalls the game: the rollout did not finish")
    return snap.player.score, failed


@lru_cache(maxsize=32)
def _baseline(scenario: str, turns: int, seed: int) -> int:
    from napoleon.agents.greedy import GreedyAgent

    return _rollout(GreedyAgent(), scenario, turns, seed)[0]


def verify_skill(
    skill: Skill,
    args: dict | None = None,
    *,
    scenarios: tuple[str, ...] = ("default", "bad_opening"),
    turns: int = 20,
    seed: int = 0,
    min_ratio: float = 0.75,
) -> Verification:
    """Play ``skill`` in fake-game rollouts, falling back to the greedy agent when it has
    nothing to do. It passes when it fires at least once, the game never rejects its actions,
    it never stalls the game, and its final score stays within ``min_ratio`` of the greedy
    baseline in every scenario. The score check is a heuristic that catches skills which
    wreck a game, not a proof that they help."""
    try:
        bound = bind_args(skill, args)
    except SkillError as exc:
        return Verification(False, str(exc))
    fired, scores, baselines = 0, {}, {}
    for scenario in scenarios:
        agent = _SkillThenGreedy(skill, bound)
        try:
            score, failed = _rollout(agent, scenario, turns, seed)
        except SkillError as exc:
            return Verification(False, f"{scenario}: {exc}", fired, scores, baselines)
        base = _baseline(scenario, turns, seed)
        fired += agent.fired
        scores[scenario], baselines[scenario] = score, base
        if failed:
            return Verification(
                False, f"{scenario}: {failed} actions were rejected", fired, scores, baselines
            )
        if score < min_ratio * base:
            return Verification(
                False,
                f"{scenario}: final score {score} is below {min_ratio:.0%} of the baseline {base}",
                fired,
                scores,
                baselines,
            )
    if fired == 0:
        return Verification(
            False,
            f"the skill never applied in {len(scenarios)} rollouts of {turns} turns",
            0,
            scores,
            baselines,
        )
    return Verification(True, "ok", fired, scores, baselines)
