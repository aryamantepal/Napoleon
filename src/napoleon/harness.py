"""Harness state for the continual agent: H = (p, G, K, M), with guardrails on every edit.

p is an editable markdown strategy prompt, G a set of sub-agents, K a set of skills in the
data-only DSL of ``napoleon.skills`` and M a set of memory entries. Every edit goes through
a function here that validates it and returns what it did, so a refiner can never put the
harness into a state the acting agent cannot use: the strategy prompt is capped in size,
skills must parse and pass fake-game verification, and sub-agents may only use whitelisted
tools with a turn budget clamped to ``MAX_SUBAGENT_TURNS``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from napoleon.skills import Skill, SkillError, Verification, bind_args, parse_skill, verify_skill

MAX_SUBAGENT_TURNS = 10
# Tools a sub-agent may be given. The lookups are read-only; memory tools touch only M.
SNAPSHOT_TOOLS = (
    "inspect_tile",
    "inspect_unit",
    "inspect_city",
    "list_options",
    "find_city_sites",
    "threats",
)
TOOL_WHITELIST = (*SNAPSHOT_TOOLS, "read_memory", "write_memory")

DEFAULT_PROMPT = """\
# Strategy

- Clear blockers first (idle units, idle cities, research, civics), then end the turn.
- Found cities early and keep each city building something useful.
- Do not repeat an action that was just rejected; pick a different legal action.
"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryEntry(_Model):
    id: int
    title: str
    content: str
    importance: int = Field(3, ge=1, le=5)


class SkillEntry(_Model):
    id: int
    name: str
    description: str
    effectiveness: float = 0.5  # 0..1, moved toward 1 by accepted actions and toward 0 by rejects
    uses: int = 0
    failures: int = 0
    body: Skill


class SubAgent(_Model):
    id: int
    name: str
    description: str
    directive: str
    allowed_tools: list[str] = []
    max_turns: int = 1  # model calls; 1 is a one-step handler, more is a looping one


class Harness(_Model):
    generation: int = 0
    prompt: str = DEFAULT_PROMPT
    skills: list[SkillEntry] = []
    memory: list[MemoryEntry] = []
    subagents: list[SubAgent] = []
    next_id: int = 1

    def copy(self) -> Harness:
        return self.model_copy(deep=True)

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def skill(self, key: Any) -> SkillEntry | None:
        return next((s for s in self.skills if key in (s.id, s.name)), None)

    def subagent(self, key: Any) -> SubAgent | None:
        return next((g for g in self.subagents if key in (g.id, g.name)), None)

    def memory_entry(self, key: Any) -> MemoryEntry | None:
        return next((m for m in self.memory if m.id == key), None)


class Limits(_Model):
    """Sizes the refiner and the agent cannot exceed."""

    max_prompt_chars: int = 4000
    max_skills: int = 20
    max_subagents: int = 8
    max_memory: int = 40
    max_memory_chars: int = 400
    max_directive_chars: int = 600


# Persistence ------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def save_harness(h: Harness, directory: Path) -> None:
    """Write the latest state and this generation's strategy prompt."""
    write_atomic(directory / "harness.json", h.model_dump_json(indent=1))
    write_atomic(directory / "prompt_evolution" / f"gen-{h.generation:03d}.md", h.prompt)


def load_harness(directory: str | Path) -> Harness:
    path = Path(directory)
    path = path / "harness.json" if path.is_dir() else path
    return Harness.model_validate_json(path.read_text(encoding="utf-8"))


def append_log(directory: Path, entry: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "evolution_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# Edits. Each returns a list of human-readable outcomes, "ok: ..." or "rejected: ...". -----


def set_prompt(h: Harness, text: str, limits: Limits) -> list[str]:
    text = text.strip()
    if not text:
        return ["rejected: empty strategy prompt"]
    if len(text) > limits.max_prompt_chars:
        return [f"rejected: strategy is {len(text)} chars, over the cap {limits.max_prompt_chars}"]
    if text == h.prompt.strip():
        return ["unchanged: strategy prompt"]
    h.prompt = text
    return [f"ok: strategy rewritten ({len(text)} chars)"]


def _str(item: dict, key: str, default: str = "") -> str:
    value = item.get(key, default)
    return value.strip() if isinstance(value, str) else default


def add_memory(h: Harness, item: Any, limits: Limits) -> str:
    if not isinstance(item, dict) or not _str(item, "title") or not _str(item, "content"):
        return "rejected: memory needs a title and content"
    importance = item.get("importance", 3)
    importance = (
        importance if isinstance(importance, int) and not isinstance(importance, bool) else 3
    )
    entry = MemoryEntry(
        id=h.new_id(),
        title=_str(item, "title")[:80],
        content=_str(item, "content")[: limits.max_memory_chars],
        importance=min(5, max(1, importance)),
    )
    if len(h.memory) >= limits.max_memory:
        weakest = min(h.memory, key=lambda m: (m.importance, m.id))
        if weakest.importance > entry.importance:
            return f"rejected: memory is full of more important entries ({limits.max_memory})"
        h.memory.remove(weakest)
    h.memory.append(entry)
    return f"ok: memory {entry.id} {entry.title!r}"


def update_memory(h: Harness, item: Any, limits: Limits) -> str:
    entry = h.memory_entry(item.get("id")) if isinstance(item, dict) else None
    if entry is None:
        return "rejected: no such memory id"
    if _str(item, "title"):
        entry.title = _str(item, "title")[:80]
    if _str(item, "content"):
        entry.content = _str(item, "content")[: limits.max_memory_chars]
    imp = item.get("importance")
    if isinstance(imp, int) and not isinstance(imp, bool):
        entry.importance = min(5, max(1, imp))
    return f"ok: memory {entry.id} updated"


def retire_memory(h: Harness, key: Any) -> str:
    entry = h.memory_entry(key)
    if entry is None:
        return f"rejected: no memory {key!r}"
    h.memory.remove(entry)
    return f"ok: memory {entry.id} retired"


def clamp_tools(tools: Any) -> tuple[list[str], list[str]]:
    """Keep only whitelisted tool names. Returns (kept, dropped)."""
    names = [t for t in tools if isinstance(t, str)] if isinstance(tools, list) else []
    kept = list(dict.fromkeys(t for t in names if t in TOOL_WHITELIST))
    return kept, [t for t in names if t not in TOOL_WHITELIST]


def _subagent_fields(item: dict, limits: Limits) -> dict:
    fields: dict[str, Any] = {}
    for key in ("name", "description"):
        if _str(item, key):
            fields[key] = _str(item, key)[:200]
    if _str(item, "directive"):
        fields["directive"] = _str(item, "directive")[: limits.max_directive_chars]
    if "allowed_tools" in item:
        fields["allowed_tools"], _ = clamp_tools(item["allowed_tools"])
    turns = item.get("max_turns")
    if isinstance(turns, int) and not isinstance(turns, bool):
        fields["max_turns"] = min(MAX_SUBAGENT_TURNS, max(1, turns))
    return fields


def add_subagent(h: Harness, item: Any, limits: Limits) -> str:
    if not isinstance(item, dict):
        return "rejected: sub-agent must be an object"
    fields = _subagent_fields(item, limits)
    missing = [k for k in ("name", "description", "directive") if k not in fields]
    if missing:
        return f"rejected: sub-agent is missing {missing}"
    if h.subagent(fields["name"]) is not None:
        return f"rejected: sub-agent {fields['name']!r} already exists"
    if len(h.subagents) >= limits.max_subagents:
        return f"rejected: at most {limits.max_subagents} sub-agents"
    sub = SubAgent(id=h.new_id(), **fields)
    h.subagents.append(sub)
    return f"ok: sub-agent {sub.id} {sub.name!r} ({sub.max_turns} turns, tools {sub.allowed_tools})"


def update_subagent(h: Harness, item: Any, limits: Limits) -> str:
    sub = h.subagent(item.get("id")) if isinstance(item, dict) else None
    if sub is None:
        return "rejected: no such sub-agent id"
    fields = _subagent_fields(item, limits)
    fields.pop("name", None)
    for key, value in fields.items():
        setattr(sub, key, value)
    return f"ok: sub-agent {sub.id} updated"


def retire_subagent(h: Harness, key: Any) -> str:
    sub = h.subagent(key)
    if sub is None:
        return f"rejected: no sub-agent {key!r}"
    h.subagents.remove(sub)
    return f"ok: sub-agent {sub.id} retired"


Verifier = Callable[[Skill, dict], Verification]


def add_skill(
    h: Harness, item: Any, limits: Limits, verifier: Verifier | None = verify_skill
) -> str:
    """Parse the skill into the DSL, then verify it in fake-game rollouts before saving.

    A skill with parameters is verified with ``example_args``; without them only the schema
    is checked, because there is nothing to play it with.
    """
    if not isinstance(item, dict):
        return "rejected: skill must be an object"
    if len(h.skills) >= limits.max_skills:
        return f"rejected: at most {limits.max_skills} skills"
    data = {k: v for k, v in item.items() if k not in ("id", "effectiveness", "example_args")}
    try:
        skill = parse_skill(data)
        if h.skill(skill.name) is not None:
            raise SkillError(f"a skill named {skill.name!r} already exists")
        example = item.get("example_args")
        note = "schema checked"
        if verifier is not None and (not skill.params or isinstance(example, dict)):
            args = bind_args(skill, example if isinstance(example, dict) else {})
            result = verifier(skill, args)
            if not result.ok:
                raise SkillError(f"failed fake-game verification: {result.reason}")
            skill = skill.model_copy(update={"verification": result.summary()})
            note = "verified on the fake game"
    except (SkillError, ValidationError) as exc:
        return f"rejected: skill {item.get('name')!r}: {exc}"
    entry = SkillEntry(id=h.new_id(), name=skill.name, description=skill.description, body=skill)
    h.skills.append(entry)
    return f"ok: skill {entry.id} {entry.name!r} ({note})"


def update_skill(h: Harness, item: Any) -> str:
    entry = h.skill(item.get("id")) if isinstance(item, dict) else None
    if entry is None:
        return "rejected: no such skill id"
    desc = _str(item, "description")
    if desc:
        if not 10 <= len(desc) <= 300:
            return "rejected: description must be 10 to 300 characters"
        entry.description = desc
        entry.body = entry.body.model_copy(update={"description": desc})
    eff = item.get("effectiveness")
    if isinstance(eff, int | float) and not isinstance(eff, bool):
        entry.effectiveness = min(1.0, max(0.0, float(eff)))
    return f"ok: skill {entry.id} updated"


def retire_skill(h: Harness, key: Any) -> str:
    entry = h.skill(key)
    if entry is None:
        return f"rejected: no skill {key!r}"
    h.skills.remove(entry)
    return f"ok: skill {entry.id} retired"


def record_skill_outcome(h: Harness, name: str, ok: bool) -> None:
    entry = h.skill(name)
    if entry is None:
        return
    entry.uses += 1
    entry.failures += 0 if ok else 1
    entry.effectiveness = round(0.8 * entry.effectiveness + 0.2 * (1.0 if ok else 0.0), 3)
