"""Voyager-style agent: a growing library of verified, reusable skills.

The model either plays one primitive action, runs a skill it already has, or defines a new
skill. A new skill is parsed into the skill language (``napoleon.skills``), played in
fake-game rollouts, and stored only if it passes. Once a skill is running, its following
actions are taken without calling the model, so a good library cuts cost as well as
mistakes. The library is a JSON file, so it carries across games and processes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from napoleon.agents.base import Decision
from napoleon.agents.prompting import (
    GAME_RULES,
    NO_COT_NOTE,
    TurnLog,
    parse_reply,
    render_snapshot,
    retry_note,
)
from napoleon.fewshot import make_fewshot
from napoleon.llm import LLMClient, clean_reply, extract_json, make_client
from napoleon.schema import Action, Snapshot
from napoleon.skills import (
    Skill,
    SkillError,
    SkillLibrary,
    SkillRun,
    Verification,
    bind_args,
    parse_skill,
    verify_skill,
)
from napoleon.usage import Usage

DEFAULT_LIBRARY = Path("runs/skills/library.json")

SKILL_GUIDE = """\
You keep a library of skills: named, parameterized routines that play several actions for
you without asking the model again, which is faster and cheaper. Reuse a skill when one fits.
When you notice a routine you will need again, define a skill. New skills are tested in
practice games and saved only if they pass.

Each reply is one JSON object, one of:
1. {"thought": "...", "action": <one entry from the legal actions>}
2. {"thought": "...", "use_skill": "<name>", "args": {...}}
   Runs an existing skill. Its first action happens now; the rest follow on later steps.
3. {"thought": "...", "new_skill": <skill>, "args": {...}}   defines, tests, saves and runs it.

A skill is {"name": "snake_case", "description": "what it does and when to use it",
"params": {"q": "int"}, "rules": [<rule>, ...], "until": <condition>?, "max_steps": 20?,
"max_turns": 1?}. Parameter types are "int" or "str"; "$q" in a rule stands for the argument.
A rule is {"do": <action fields>, "when": <condition>?, "unit_type": "settler"?,
"blocked_only": true?, "choose": <chooser>?}:
- "do" holds fields of an action, e.g. {"type": "unit_command", "command": "found_city"} or
  {"type": "move_unit"} or {"type": "set_production", "item": "warrior"}. Any legal action that
  contains those fields can match.
- "unit_type" keeps only actions by units of that type. "blocked_only" keeps only actions for
  units that need orders or cities that need production.
- "choose" picks among several matches: "first" (default), {"nearest_to": {"q": "$q", "r": "$r"}},
  {"farthest_from_cities": true} or {"nearest_enemy": true}; the last three apply to moves.
Every step, rules are tried in order and the first one that matches a legal action runs. The
skill ends when no rule matches, after max_steps actions, after max_turns turns, or when
"until" holds. The same action is never repeated within one run.
Conditions are objects with one key: {"has_unit": "settler"}, {"cities_below": 3},
{"cities_at_least": 2}, {"turn_at_least": 10}, {"turn_below": 10}, {"enemy_within": 4},
{"blocker": "city_needs_production"}, {"no_blockers": true}, {"legal": <action fields>},
{"not": c}, {"all": [c, ...]}, {"any": [c, ...]}.

Example skill:
{"name": "settle_at", "description": "Found a city with a settler, walking to the tile first",
 "params": {"q": "int", "r": "int"},
 "rules": [{"do": {"type": "unit_command", "command": "found_city"}, "unit_type": "settler"},
           {"do": {"type": "move_unit"}, "unit_type": "settler",
            "choose": {"nearest_to": {"q": "$q", "r": "$r"}}}]}
"""

SYSTEM_PROMPT = GAME_RULES + "\n" + SKILL_GUIDE


def retrieval_query(snap: Snapshot) -> str:
    """Words describing the situation, matched against skill descriptions."""
    words = [b.kind for b in snap.blockers]
    words += [u.type for u in snap.units if u.owner == snap.player.id]
    if any(u.owner != snap.player.id for u in snap.units):
        words.append("enemy attack defend")
    if not snap.cities:
        words.append("found city settle")
    if not any(b.kind != "other" for b in snap.blockers):
        words.append("end turn")
    return " ".join(words)


def _skill_view(skill: Skill) -> str:
    return json.dumps(
        skill.model_dump(mode="json", exclude={"uses", "verification"}, exclude_defaults=True),
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class _Chosen:
    action: Action
    why: str


@dataclass(frozen=True)
class _Rejected:
    error: str
    raw: dict | str


class VoyagerAgent:
    name = "voyager"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        skills_path: str | Path | None = DEFAULT_LIBRARY,
        max_retries: int = 2,
        max_tokens: int = 2048,
        max_skills: int = 50,
        retrieve_k: int = 4,
        verify: bool = True,
        verifier: Callable[[Skill, dict], Verification] | None = None,
        shots: int = 0,
        demos: str | None = None,
        cot: bool = True,
        turn_log: bool = False,
    ) -> None:
        self._client = client
        self.cot = cot
        self.turn_log = TurnLog() if turn_log else None
        self.fewshot = make_fewshot(shots, demos, cot)
        self.library = SkillLibrary(skills_path, max_skills=max_skills)
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.retrieve_k = retrieve_k
        self.verify = verify
        self.verifier = verifier or (lambda skill, args: verify_skill(skill, args))
        self.active: tuple[str, SkillRun] | None = None
        self.created = 0
        self.reused = 0
        self.free_steps = 0
        self.loaded = 0

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = make_client()
        return self._client

    def config(self) -> dict:
        return {
            "model": self.client.model,
            "max_retries": self.max_retries,
            "skills_path": str(self.library.path) if self.library.path else None,
            "skills_loaded": self.loaded,
            "verify": self.verify,
            "cot": self.cot,
            "turn_log": self.turn_log is not None,
            **(self.fewshot.config() if self.fewshot else {"shots": 0}),
        }

    def reset(self, seed: int) -> None:
        self.active = None
        self.created = self.reused = self.free_steps = 0
        self.loaded = len(self.library)
        if self.turn_log:
            self.turn_log.reset()
        if self.fewshot:
            self.fewshot.reset(seed)

    # Deciding -----------------------------------------------------------------------

    def act(self, snapshot: Snapshot) -> Decision:
        decision = self._decide(snapshot)
        if self.turn_log:
            self.turn_log.note(snapshot, decision.action)
        return decision

    def _decide(self, snapshot: Snapshot) -> Decision:
        if self.active is not None:
            name, run = self.active
            action = run.next_action(snapshot)
            if action is not None:
                self.free_steps += 1
                return Decision(action, f"[skill {name}, rule {run.last_rule}] no model call")
            self.active = None

        shots = self.fewshot.messages(snapshot, "thought") if self.fewshot else []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + ("" if self.cot else "\n" + NO_COT_NOTE)},
            *shots,
            {"role": "user", "content": self._prompt(snapshot)},
        ]
        usage = Usage()
        for attempt in range(self.max_retries + 1):
            reply = self.client.complete(messages, max_tokens=self.max_tokens)
            usage = usage + reply.usage
            outcome = self._interpret(reply.text, snapshot)
            if isinstance(outcome, _Chosen):
                note = f"[after {attempt} retries] " if attempt else ""
                return Decision(outcome.action, note + outcome.why, usage)
            messages += [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": retry_note(outcome.error, snapshot)},
            ]
        why = f"unusable output after {self.max_retries + 1} attempts: {outcome.error}"
        return Decision(outcome.raw, why, usage)

    def _prompt(self, snap: Snapshot) -> str:
        text = render_snapshot(snap)
        shown = self.library.retrieve(retrieval_query(snap), self.retrieve_k)
        if shown:
            text += "\n\nRelevant skills in your library:\n" + "\n".join(
                f"- {_skill_view(s)}" for s in shown
            )
        others = [n for n in self.library.skills if n not in {s.name for s in shown}]
        if others:
            text += "\nOther skills you can call by name: " + ", ".join(others)
        if not self.library.skills:
            text += "\n\nYour skill library is empty."
        if self.turn_log:
            text += self.turn_log.render(snap)
        return text

    def _interpret(self, text: str, snap: Snapshot) -> _Chosen | _Rejected:
        obj = extract_json(text)
        if obj is None:
            return _Rejected("no JSON object found in the reply.", clean_reply(text)[:500])
        thought = obj.get("thought") or obj.get("reasoning")
        thought = thought if isinstance(thought, str) else ""
        try:
            if "new_skill" in obj:
                return _Chosen(*self._new_skill(obj, thought, snap))
            if "use_skill" in obj:
                return _Chosen(*self._use_skill(obj, thought, snap))
        except SkillError as exc:
            return _Rejected(f"{exc}", obj)
        parsed = parse_reply(text)
        if parsed.action is not None:
            return _Chosen(parsed.action, parsed.reasoning or thought)
        return _Rejected(f"{parsed.error}.", parsed.raw)

    def _start(self, skill: Skill, args: dict, snap: Snapshot):
        run = SkillRun(skill, args)
        action = run.next_action(snap)
        if action is None:
            raise SkillError(
                f"skill {skill.name!r} has nothing to do right now: no rule matches a legal "
                "action. Pick an action or another skill."
            )
        self.active = (skill.name, run)
        return action

    def _use_skill(self, obj: dict, thought: str, snap: Snapshot):
        name = obj["use_skill"]
        skill = self.library.get(name) if isinstance(name, str) else None
        if skill is None:
            raise SkillError(f"no skill named {name!r}; skills are {sorted(self.library.skills)}.")
        args = bind_args(skill, obj.get("args"))
        action = self._start(skill, args, snap)
        self.library.record_use(skill.name)
        self.reused += 1
        return action, f"[skill {skill.name}] {thought}"

    def _new_skill(self, obj: dict, thought: str, snap: Snapshot):
        skill = parse_skill(obj["new_skill"])
        if skill.name in self.library:
            raise SkillError(f"a skill named {skill.name!r} already exists; pick another name.")
        args = bind_args(skill, obj.get("args"))
        result = self.verifier(skill, args) if self.verify else None
        if result is not None and not result.ok:
            raise SkillError(f"skill {skill.name!r} failed testing: {result.reason}.")
        skill = skill.model_copy(
            update={"verification": result.summary() if result is not None else None}
        )
        self.library.add(skill)
        self.created += 1
        action = self._start(skill, args, snap)
        self.library.record_use(skill.name)
        return action, f"[new skill {skill.name}, tested ok] {thought}"
