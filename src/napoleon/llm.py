"""A thin LLM interface and clients for it.

``LLMClient`` is one method: chat messages in, text and token usage out. Agents ask for
JSON and parse it themselves, because small local models are unreliable at native tool
calling. ``OpenAICompatClient`` speaks the OpenAI chat-completions protocol, so one client
covers Ollama, LM Studio, vLLM, llama.cpp's server and hosted endpoints such as Qwen on
DashScope. ``ScriptedClient`` is the deterministic stand-in for tests.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from napoleon.usage import Pricing, Usage

Message = dict[str, str]

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama
DEFAULT_MODEL = "qwen3:8b"
API_KEY_ENV = "NAPOLEON_LLM_API_KEY"


class LLMError(RuntimeError):
    """The model server could not be reached or answered with an error."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> LLMResponse: ...


# Parsing model output ---------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def clean_reply(text: str) -> str:
    """Drop reasoning-model ``<think>`` sections. A cut-off section is dropped too."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():  # an opening tag the server already stripped
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


def extract_json(text: str) -> dict | None:
    """The first JSON object in a reply, tolerating think tags, code fences and chatter."""
    text = clean_reply(text)
    decoder = json.JSONDecoder()
    for chunk in [*_FENCE.findall(text), text]:
        for i, ch in enumerate(chunk):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(chunk[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Clients ----------------------------------------------------------------------------


class OpenAICompatClient:
    """Chat completions over HTTP with the standard library, so it needs no extra install."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        *,
        temperature: float = 0.2,
        timeout: float = 300.0,
        pricing: Pricing | None = None,
        json_mode: bool = False,
        no_think: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.pricing = pricing or Pricing()
        self.json_mode = json_mode
        self.no_think = no_think

    def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> LLMResponse:
        if self.no_think and messages:  # Qwen3's soft switch for skipping its thinking phase
            messages = [
                *messages[:-1],
                {**messages[-1], "content": messages[-1]["content"] + " /no_think"},
            ]
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise LLMError(f"{self.base_url} answered {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            raise LLMError(f"could not get a reply from {self.base_url}: {exc}") from exc
        return self._parse(data, messages)

    def _parse(self, data: dict, messages: list[Message]) -> LLMResponse:
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from exc
        raw = data.get("usage") or {}
        tokens_in = raw.get("prompt_tokens")
        tokens_out = raw.get("completion_tokens")
        if tokens_in is None:
            tokens_in = estimate_tokens("".join(m["content"] for m in messages))
        if tokens_out is None:
            tokens_out = estimate_tokens(text)
        usage = Usage(
            calls=1,
            input_tokens=int(tokens_in),
            output_tokens=int(tokens_out),
            cost_usd=self.pricing.cost(int(tokens_in), int(tokens_out)),
        )
        return LLMResponse(text, usage)


Reply = str | LLMResponse | Callable[[list[Message]], "str | LLMResponse"]


class ScriptedClient:
    """Replays canned replies in order, or calls a function with the messages.

    Usage is estimated from text length, so cost tracking has something to count. Once the
    script runs out the last reply repeats, which suits "always answer end_turn" tests.
    """

    def __init__(
        self,
        replies: Sequence[Reply] | Reply,
        model: str = "scripted",
        pricing: Pricing | None = None,
    ) -> None:
        self.replies: list[Reply] = (
            list(replies) if isinstance(replies, list | tuple) else [replies]
        )
        self.model = model
        self.pricing = pricing or Pricing()
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        reply = self.replies[idx]
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, LLMResponse):
            return reply
        tokens_in = estimate_tokens("".join(m["content"] for m in messages))
        tokens_out = estimate_tokens(reply)
        usage = Usage(
            calls=1,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=self.pricing.cost(tokens_in, tokens_out),
        )
        return LLMResponse(reply, usage)


def make_client(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = API_KEY_ENV,
    *,
    temperature: float = 0.2,
    pricing: Pricing | None = None,
    no_think: bool = False,
    json_mode: bool = False,
) -> OpenAICompatClient:
    """Build the HTTP client from flags, falling back to ``NAPOLEON_LLM_*`` env vars."""
    return OpenAICompatClient(
        model or os.environ.get("NAPOLEON_LLM_MODEL") or DEFAULT_MODEL,
        base_url or os.environ.get("NAPOLEON_LLM_BASE_URL") or DEFAULT_BASE_URL,
        os.environ.get(api_key_env) or None,
        temperature=temperature,
        pricing=pricing,
        no_think=no_think,
        json_mode=json_mode,
    )
