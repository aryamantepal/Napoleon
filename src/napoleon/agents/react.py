"""ReAct: look things up with read-only tools, then commit one action."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

from napoleon.agents.base import Decision
from napoleon.agents.prompting import (
    GAME_RULES,
    Reply,
    action_json,
    parse_reply,
    render_snapshot,
    retry_note,
)
from napoleon.agents.tools import TOOL_DOCS, SnapshotTools, ToolError
from napoleon.llm import LLMClient, extract_json, make_client
from napoleon.schema import Snapshot
from napoleon.usage import Usage

REACT_FORMAT = """\
Each reply is one JSON object and nothing else. Either look something up:
{"thought": "<why>", "tool": "<tool name>", "args": {...}}
or commit your one action for this step:
{"thought": "<why>", "action": <an action exactly as returned by list_options>}
After a lookup you get an Observation and may look up more, up to the budget stated in each
observation. Then commit. Only legal actions can succeed, so check list_options before you
commit to something you have not seen listed.
"""

SYSTEM_PROMPT = GAME_RULES + "\n" + TOOL_DOCS + "\n" + REACT_FORMAT


@dataclass(frozen=True)
class ToolCall:
    thought: str | None
    tool: str
    args: dict


def parse_step(text: str) -> ToolCall | Reply:
    """A lookup request if the reply names a tool, otherwise an attempt at an action."""
    obj = extract_json(text)
    if obj is not None and isinstance(obj.get("tool"), str):
        thought = obj.get("thought")
        args = obj.get("args", obj.get("arguments", {}))
        return ToolCall(thought if isinstance(thought, str) else None, obj["tool"], args)
    return parse_reply(text)


class ReActAgent:
    """Thought, lookup, observation, repeat, then one action.

    ``max_inner_steps`` bounds the lookups per decision. Unusable replies and lookups past
    the budget cost a model call but not a lookup; ``max_retries`` bounds those. The agent
    keeps a short log of its own recent actions, but the game state is always re-read.
    """

    name = "react"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        max_inner_steps: int = 5,
        max_retries: int = 2,
        max_tokens: int = 1024,
        memory: int = 5,
    ) -> None:
        self._client = client
        self.max_inner_steps = max_inner_steps
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.recent: deque[str] = deque(maxlen=memory)

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = make_client()
        return self._client

    def config(self) -> dict:
        return {
            "model": self.client.model,
            "max_inner_steps": self.max_inner_steps,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
        }

    def reset(self, seed: int) -> None:
        self.recent.clear()

    def act(self, snapshot: Snapshot) -> Decision:
        tools = SnapshotTools(snapshot)
        prompt = render_snapshot(snapshot, tiles=False, legal=False)
        if self.recent:
            prompt += "\nYour recent actions, oldest first:\n" + "\n".join(self.recent)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        usage = Usage()
        trace: list[str] = []
        lookups = 0
        reply: Reply | None = None
        for _ in range(self.max_inner_steps + 1 + self.max_retries):
            response = self.client.complete(messages, max_tokens=self.max_tokens)
            usage = usage + response.usage
            messages.append({"role": "assistant", "content": response.text})
            step = parse_step(response.text)
            if isinstance(step, ToolCall):
                if lookups >= self.max_inner_steps:
                    feedback = "No lookups left. Commit an action now."
                else:
                    lookups += 1
                    left = self.max_inner_steps - lookups
                    observation = self._lookup(tools, step)
                    trace += [
                        f"Thought: {step.thought or '-'}",
                        f"Lookup: {step.tool} {json.dumps(step.args)}",
                        f"Observation: {observation[:400]}",
                    ]
                    feedback = f"Observation:\n{observation}\n(lookups left: {left})"
                messages.append({"role": "user", "content": feedback})
                continue
            reply = step
            if step.action is not None:
                trace.append(f"Thought: {step.reasoning or '-'}")
                trace.append(f"Action: {action_json(step.action)}")
                self.recent.append(f"turn {snapshot.turn}: {action_json(step.action)}")
                return Decision(step.action, "\n".join(trace), usage)
            trace.append(f"Rejected reply: {step.error}")
            messages.append({"role": "user", "content": retry_note(step.error, snapshot)})
        raw: dict | str = reply.raw if reply is not None else "no usable reply"
        trace.append(f"Gave up after {len(trace)} trace lines without a usable action")
        self.recent.append(f"turn {snapshot.turn}: (no usable action)")
        return Decision(raw, "\n".join(trace), usage)

    @staticmethod
    def _lookup(tools: SnapshotTools, call: ToolCall) -> str:
        try:
            return tools.run(call.tool, call.args)
        except ToolError as exc:
            return f"error: {exc}"
