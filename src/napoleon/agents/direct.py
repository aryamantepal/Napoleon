"""Stateless direct policy: one prompt per decision, no memory, no tools."""

from __future__ import annotations

from napoleon.agents.base import Decision
from napoleon.agents.prompting import direct_messages, parse_reply, retry_note
from napoleon.fewshot import make_fewshot
from napoleon.llm import LLMClient, make_client
from napoleon.schema import Snapshot
from napoleon.usage import Usage


class DirectAgent:
    """Shows the model the snapshot and asks for one action as JSON.

    Output that is not valid JSON or not a valid action is retried up to ``max_retries``
    times with the problem pointed out. After that the raw output is passed to the game
    as the action, where it is rejected and counted in the illegal-action rate. A
    well-formed but illegal action is *not* retried: that is the agent's mistake to own.
    """

    name = "direct"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        max_retries: int = 2,
        max_tokens: int = 2048,
        shots: int = 0,
        demos: str | None = None,
        cot: bool = True,
    ) -> None:
        self._client = client
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.cot = cot
        self.fewshot = make_fewshot(shots, demos, cot)
        self.lessons = ""  # extra system text, set by agents that learn across games

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = make_client()
        return self._client

    def config(self) -> dict:
        return {
            "model": self.client.model,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "cot": self.cot,
            **(self.fewshot.config() if self.fewshot else {"shots": 0}),
        }

    def reset(self, seed: int) -> None:
        if self.fewshot:
            self.fewshot.reset(seed)

    def act(self, snapshot: Snapshot) -> Decision:
        messages = direct_messages(snapshot, self.cot, self.lessons)
        if self.fewshot:
            messages[1:1] = self.fewshot.messages(snapshot)
        usage = Usage()
        for attempt in range(self.max_retries + 1):
            reply = self.client.complete(messages, max_tokens=self.max_tokens)
            usage = usage + reply.usage
            parsed = parse_reply(reply.text)
            if parsed.action is not None:
                note = f"[after {attempt} retries] " if attempt else ""
                return Decision(parsed.action, note + (parsed.reasoning or ""), usage)
            messages += [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": retry_note(parsed.error, snapshot)},
            ]
        why = f"unusable output after {self.max_retries + 1} attempts: {parsed.error}"
        return Decision(parsed.raw, why, usage)
