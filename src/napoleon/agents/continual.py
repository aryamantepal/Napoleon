"""Continual Harness: an agent whose harness improves while it plays, with no resets.

See ``docs/continual-harness.md``. The harness H = (p, G, K, M) lives in
``napoleon.harness``; the scheduled text-only refiner lives in ``napoleon.refiner``. This
module is the acting side: an orchestrator that, per step, either plays a legal action,
runs a skill, delegates to a sub-agent, or uses memory, and that keeps the trajectory the
refiner reads. Every refinement is a generation, saved to the harness directory and
reported as an agent event so a replay can show it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from napoleon import harness as hx
from napoleon.agents.base import Decision
from napoleon.agents.prompting import GAME_RULES, action_json, render_snapshot
from napoleon.agents.react import ToolCall, parse_step
from napoleon.agents.tools import TOOL_DOCS, SnapshotTools, ToolError
from napoleon.llm import LLMClient, clean_reply, extract_json, make_client
from napoleon.refiner import Refiner, RollbackPolicy, Schedule, StepLog, clip, window_stats
from napoleon.refiner import summarize as summarize_refinement
from napoleon.schema import Action, Snapshot
from napoleon.skills import SkillError, SkillRun, bind_args, verify_skill
from napoleon.usage import Usage

FORMAT = """\
Each reply is one JSON object and nothing else, one of:
{"thought": "<why>", "action": <one entry copied from the legal actions>}
{"thought": "<why>", "tool": "run_skill", "args": {"name": "<skill>", "args": {...}}}
    runs a skill from your skill list; its first action happens now, the rest follow on later
    steps without asking you again.
{"thought": "<why>", "tool": "delegate", "args": {"name": "<sub-agent>", "task": "<optional>"}}
    hands this step to a sub-agent, which returns an action or a note.
{"thought": "<why>", "tool": "read_memory", "args": {"id": <int>}}
{"thought": "<why>", "tool": "write_memory",
 "args": {"title": "...", "content": "...", "importance": 1-5}}
    adds a memory entry; pass "id" to overwrite an entry.
Lookups and memory calls return an observation and you may continue; run_skill, delegate
(when the sub-agent returns an action) and an action end the step.
"""

FIXED_SYSTEM = GAME_RULES + "\n" + TOOL_DOCS + "\n" + FORMAT

SUBAGENT_FORMAT = """\
You are a sub-agent: a focused handler the main agent delegated this step to. Follow your
directive. Reply with one JSON object and nothing else, one of:
{"thought": "<why>", "action": <one entry copied from the legal actions>}
{"thought": "<why>", "note": "<short message back to the main agent>"}
{"thought": "<why>", "tool": "<one of your allowed tools>", "args": {...}}
"""

ORCHESTRATOR_TOOLS = frozenset(
    {"run_skill", "delegate", "read_memory", "write_memory"}
) | frozenset(hx.SNAPSHOT_TOOLS)


@dataclass
class _Out:
    """How one decision loop ended: an action (maybe starting a skill), a note, or nothing."""

    action: Action | None = None
    raw: Any = None
    why: str = ""
    note: str | None = None
    usage: Usage = field(default_factory=Usage)
    skill: str | None = None


class ContinualAgent:
    name = "continual"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        max_retries: int = 2,
        max_tokens: int = 2048,
        max_tool_calls: int = 4,
        window: int = 50,
        schedule: Schedule | None = None,
        rollback: RollbackPolicy | None = None,
        limits: hx.Limits | None = None,
        bootstrap_from: str | Path | None = None,
        bootstrap_update: bool = False,
        harness_dir: str | Path | None = None,
        verify: bool = True,
        verifier: hx.Verifier | None = None,
        refine_tokens: int = 1500,
    ) -> None:
        self._client = client
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.max_tool_calls = max_tool_calls
        self.window = window
        self.schedule = schedule or Schedule()
        self.rollback = rollback or RollbackPolicy()
        self.limits = limits or hx.Limits()
        self.bootstrap_from = Path(bootstrap_from) if bootstrap_from else None
        # From scratch the harness always updates; a bootstrapped one is frozen unless asked.
        self.refine = self.bootstrap_from is None or bootstrap_update
        self.verifier = (verifier or verify_skill) if verify else None
        self.refine_tokens = refine_tokens
        self._dir_option = Path(harness_dir) if harness_dir else None
        self.dir = self._dir_option
        self._refiner: Refiner | None = None
        self.h = hx.Harness()
        self.trajectory: list[StepLog] = []
        self.step = 0
        self.active: tuple[str, SkillRun] | None = None
        self._failures: list[str] = []
        self._last_skill: str | None = None
        self._checkpoint: tuple[hx.Harness, int, Any] | None = None
        self.refinements = self.rollbacks = self.free_steps = self.delegations = 0

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = make_client()
        return self._client

    @property
    def refiner(self) -> Refiner:
        if self._refiner is None:
            self._refiner = Refiner(
                self.client,
                FIXED_SYSTEM,
                self.limits,
                max_tokens=self.refine_tokens,
                verifier=self.verifier,
            )
        return self._refiner

    def config(self) -> dict:
        return {
            "model": self.client.model,
            "max_retries": self.max_retries,
            "max_tool_calls": self.max_tool_calls,
            "bootstrap_from": str(self.bootstrap_from) if self.bootstrap_from else None,
            "refining": self.refine,
            "schedule": self.schedule.model_dump(),
            "rollback": self.rollback.model_dump(),
            "limits": self.limits.model_dump(),
            "harness_dir": str(self.dir) if self.dir else None,
        }

    # Lifecycle ----------------------------------------------------------------------

    def reset(self, seed: int) -> None:
        self.h = hx.load_harness(self.bootstrap_from) if self.bootstrap_from else hx.Harness()
        self.trajectory, self.step, self.active = [], 0, None
        self._checkpoint = None
        self._failures, self._last_skill = [], None
        self.refinements = self.rollbacks = self.free_steps = self.delegations = 0
        self.dir = self._dir_option
        if self.dir is not None:
            self._persist("start")

    def attach_run(self, path: Path) -> None:
        """Keep the harness generations next to the run file unless a directory was given."""
        if self._dir_option is None:
            self.dir = path.parent / f"{path.stem}.harness"
            self._persist("start")

    def end_game(self, run) -> Usage:
        if self.dir is not None:
            hx.save_harness(self.h, self.dir)
        return Usage()

    def observe_result(self, decision: Decision, result, snapshot: Snapshot) -> None:
        """The game's verdict on the last action, for the refiner and skill statistics."""
        if self.trajectory:
            last = self.trajectory[-1]
            last.ok = result.ok
            last.result = result.error or "; ".join(result.events)
        if self._last_skill is not None:
            hx.record_skill_outcome(self.h, self._last_skill, result.ok)
            self._last_skill = None

    # Deciding -----------------------------------------------------------------------

    def act(self, snapshot: Snapshot) -> Decision:
        events: list[dict] = []
        usage = Usage()
        if self.refine and self.schedule.due(self.step):
            events, usage = self._refine(snapshot)
        self._failures = []
        out = self._step(snapshot)
        usage = usage + out.usage
        self._log(snapshot, out)
        self.step += 1
        return Decision(
            out.action if out.action is not None else out.raw,
            out.why,
            usage if usage.calls else None,
            tuple(events),
        )

    def _step(self, snap: Snapshot) -> _Out:
        if self.active is not None:
            name, run = self.active
            action = run.next_action(snap)
            if action is not None:
                self.free_steps += 1
                self._last_skill = name
                return _Out(action, why=f"[skill {name}, rule {run.last_rule}] no model call")
            self.active = None
        messages = [
            {
                "role": "system",
                "content": f"{FIXED_SYSTEM}\n# Strategy (editable)\n{self.h.prompt}",
            },
            {"role": "user", "content": self._prompt(snap)},
        ]
        return self._loop(
            messages,
            snap,
            ORCHESTRATOR_TOOLS,
            budget=self.max_tool_calls + 1 + self.max_retries,
            lookups=self.max_tool_calls,
        )

    def _prompt(self, snap: Snapshot) -> str:
        text = render_snapshot(snap)
        if self.h.memory:
            top = sorted(self.h.memory, key=lambda m: (-m.importance, m.id))[:30]
            text += "\n\nMemory (read_memory {id} opens one):\n" + "\n".join(
                f"- {m.id} [{m.importance}] {m.title}" for m in top
            )
        if self.h.skills:
            text += "\n\nSkills (run_skill {name, args}):\n" + "\n".join(
                f"- {s.name}({', '.join(f'{k}: {v}' for k, v in s.body.params.items())}) "
                f"[eff {s.effectiveness:.2f}, used {s.uses}]: {s.description}"
                for s in sorted(self.h.skills, key=lambda s: -s.effectiveness)[:12]
            )
        if self.h.subagents:
            text += "\n\nSub-agents (delegate {name, task}):\n" + "\n".join(
                f"- {g.name}: {g.description}" for g in self.h.subagents
            )
        recent = [f"turn {s.turn}: {s.action}" for s in self.trajectory[-5:]]
        if recent:
            text += "\n\nYour recent actions, oldest first:\n" + "\n".join(recent)
        return text

    def _fail(self, message: str) -> None:
        self._failures.append(message)

    def _loop(
        self,
        messages: list[dict],
        snap: Snapshot,
        tools: frozenset[str] | set[str],
        *,
        budget: int,
        lookups: int,
        sub: hx.SubAgent | None = None,
    ) -> _Out:
        """One decision: up to ``budget`` model calls, of which ``lookups`` may be tool calls."""
        usage = Usage()
        trace: list[str] = []
        snaptools = SnapshotTools(snap)
        used = 0
        last: Any = "no usable reply"
        for _ in range(budget):
            reply = self.client.complete(messages, max_tokens=self.max_tokens)
            usage = usage + reply.usage
            messages.append({"role": "assistant", "content": reply.text})
            obj = extract_json(reply.text)
            if sub is not None and obj is not None and isinstance(obj.get("note"), str):
                return _Out(note=obj["note"], why="\n".join(trace), usage=usage)
            step = parse_step(reply.text)
            if isinstance(step, ToolCall):
                thought = step.thought or "-"
                feedback, terminal = self._tool(step, snap, snaptools, tools, used >= lookups)
                if terminal is not None:
                    if terminal.action is not None:
                        terminal.usage = terminal.usage + usage
                        terminal.why = "\n".join([*trace, f"Thought: {thought}", terminal.why])
                        return terminal
                    feedback = terminal.note or feedback
                    usage = usage + terminal.usage
                if not feedback.startswith("error"):
                    used += 1
                trace += [f"Thought: {thought}", f"Tool: {step.tool} {clip(json.dumps(step.args))}"]
                trace.append(f"Observation: {clip(feedback, 400)}")
                messages.append({"role": "user", "content": f"Observation:\n{feedback}"})
                continue
            last = step.raw
            if step.action is not None:
                trace += [
                    f"Thought: {step.reasoning or '-'}",
                    f"Action: {action_json(step.action)}",
                ]
                return _Out(step.action, why="\n".join(trace), usage=usage)
            self._fail(f"unusable reply: {step.error}")
            trace.append(f"Rejected reply: {step.error}")
            messages.append(
                {"role": "user", "content": f"That reply was rejected: {step.error}. Reply again."}
            )
        trace.append(f"Gave up after {budget} model calls without a usable action")
        return _Out(raw=last, why="\n".join(trace), usage=usage)

    # Tools --------------------------------------------------------------------------

    def _tool(self, call: ToolCall, snap, snaptools, allowed, exhausted) -> tuple[str, _Out | None]:
        """Run one tool call. Returns (observation text, a finished decision or note or None)."""
        name, args = call.tool, call.args if isinstance(call.args, dict) else {}
        if name not in allowed:
            self._fail(f"tool {name!r} is not available")
            return f"error: tool {name!r} is not available; tools are {sorted(allowed)}", None
        try:
            if name == "run_skill":
                return "", self._run_skill(args, snap)
            if name == "delegate":
                return self._delegate(args, snap)
            if exhausted:
                return "No lookups left. Commit an action now.", None
            if name == "read_memory":
                return self._read_memory(args), None
            if name == "write_memory":
                return self._write_memory(args), None
            return snaptools.run(name, args), None
        except (SkillError, ToolError) as exc:
            self._fail(f"{name}: {exc}")
            return f"error: {exc}", None

    def _read_memory(self, args: dict) -> str:
        if "id" not in args:
            return "\n".join(f"{m.id} [{m.importance}] {m.title}" for m in self.h.memory) or "empty"
        entry = self.h.memory_entry(args["id"])
        if entry is None:
            raise ToolError(f"no memory with id {args['id']!r}")
        return f"{entry.title}\n{entry.content}"

    def _write_memory(self, args: dict) -> str:
        if "id" in args:
            result = hx.update_memory(self.h, args, self.limits)
        else:
            result = hx.add_memory(self.h, args, self.limits)
        if result.startswith("rejected"):
            raise ToolError(result)
        return result

    def _run_skill(self, args: dict, snap: Snapshot) -> _Out:
        key = args.get("name", args.get("id"))
        entry = self.h.skill(key)
        if entry is None:
            names = [s.name for s in self.h.skills]
            raise SkillError(f"no skill named {key!r}; skills are {names}")
        bound = bind_args(entry.body, args.get("args"))
        run = SkillRun(entry.body, bound)
        action = run.next_action(snap)
        if action is None:
            raise SkillError(f"skill {entry.name!r} has nothing to do right now")
        self.active = (entry.name, run)
        self._last_skill = entry.name
        return _Out(action, why=f"[skill {entry.name}]")

    def _delegate(self, args: dict, snap: Snapshot) -> tuple[str, _Out | None]:
        sub = self.h.subagent(args.get("name", args.get("id")))
        if sub is None:
            names = [g.name for g in self.h.subagents]
            raise ToolError(f"no sub-agent named {args.get('name')!r}; sub-agents are {names}")
        self.delegations += 1
        allowed, _ = hx.clamp_tools(sub.allowed_tools)  # re-checked: harness.json may be edited
        turns = min(hx.MAX_SUBAGENT_TURNS, max(1, sub.max_turns))
        task = args.get("task")
        tools_line = f"Allowed tools: {allowed or 'none'}."
        messages = [
            {
                "role": "system",
                "content": f"{GAME_RULES}\n{TOOL_DOCS}\n{SUBAGENT_FORMAT}\n{tools_line}\n"
                f"Directive: {sub.directive}",
            },
            {
                "role": "user",
                "content": render_snapshot(snap)
                + (f"\n\nTask from the main agent: {task}" if isinstance(task, str) else ""),
            },
        ]
        out = self._loop(
            messages, snap, frozenset(allowed), budget=turns, lookups=turns - 1, sub=sub
        )
        out.why = f"[sub-agent {sub.name}] {out.why}"
        if out.action is not None:
            return "", out
        note = out.note if out.note is not None else "returned no usable action"
        if out.note is None:
            self._fail(f"sub-agent {sub.name} returned no usable action")
        out.note = f"Sub-agent {sub.name} note: {clean_reply(note)}"
        return out.note, out

    # Trajectory and refinement ----------------------------------------------------------

    def _log(self, snap: Snapshot, out: _Out) -> None:
        me = snap.player.id
        state = (
            f"score {snap.player.score}, cities {sum(c.owner == me for c in snap.cities)}, "
            f"units {sum(u.owner == me for u in snap.units)}, gold {snap.player.gold:g}, "
            f"blockers {[b.kind for b in snap.blockers] or 'none'}"
        )
        action = action_json(out.action) if out.action is not None else str(out.raw)
        self.trajectory.append(
            StepLog(
                self.step,
                snap.turn,
                snap.player.score,
                state,
                out.why,
                action,
                tool_failures=list(self._failures),
            )
        )

    def _persist(self, kind: str, entry: dict | None = None) -> None:
        if self.dir is None:
            return
        hx.save_harness(self.h, self.dir)
        if entry is not None:
            hx.append_log(self.dir, {"kind": kind, **entry})

    def _refine(self, snap: Snapshot) -> tuple[list[dict], Usage]:
        events: list[dict] = []
        if self._checkpoint is not None:
            before_h, at, before = self._checkpoint
            self._checkpoint = None
            after = window_stats(self.trajectory[at:])
            why = self.rollback.worse(before, after)
            if why is not None:
                failed_gen = self.h.generation
                self.h = before_h
                self.h.generation = failed_gen + 1
                self.rollbacks += 1
                event = {
                    "type": "harness_rollback",
                    "step": self.step,
                    "turn": snap.turn,
                    "generation": self.h.generation,
                    "reason": why,
                    "restored": "the harness from before generation " + str(failed_gen),
                }
                events.append(event)
                self._persist("rollback", event)
        window = self.trajectory[-self.window :]
        previous = self.h.copy()
        n = max(self.rollback.min_steps, self.schedule.interval)
        before_stats = window_stats(self.trajectory[-n:])
        result = self.refiner.refine(self.h, window)
        self.refinements += 1
        event = {
            "type": "harness_refined",
            "step": self.step,
            "turn": snap.turn,
            "generation": result.harness.generation,
            "changed": result.changed,
            "passes": summarize_refinement(result),
            "model_calls": result.usage.calls,
            "skills": len(result.harness.skills),
            "subagents": len(result.harness.subagents),
            "memory": len(result.harness.memory),
            "prompt_chars": len(result.harness.prompt),
        }
        if result.changed:
            self.h = result.harness
            self._checkpoint = (previous, self.step, before_stats)
        events.append(event)
        self._persist("refine", event)
        return events, result.usage
