"""Build an agent by name from loosely typed options, for the CLI and the sweep."""

from __future__ import annotations

from napoleon.agents import AGENTS, LLM_AGENTS
from napoleon.llm import LLMClient

# Options each LLM agent understands. Anything else in ``options`` is ignored, and ``None``
# means "use the agent's default", so one options dict can serve every agent.
OPTIONS: dict[str, tuple[str, ...]] = {
    "direct": ("max_retries", "max_tokens"),
    "react": ("max_retries", "max_tokens", "max_inner_steps"),
    "reflexion": (
        "max_retries",
        "max_tokens",
        "max_inner_steps",
        "base",
        "memory_path",
        "max_reflections",
    ),
    "voyager": ("max_retries", "max_tokens", "skills_path", "verify"),
}


def make_agent(name: str, client: LLMClient | None = None, **options):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    if name not in LLM_AGENTS:
        return AGENTS[name]()
    allowed = OPTIONS[name]
    kwargs = {k: v for k, v in options.items() if k in allowed and v is not None}
    if name == "reflexion" and kwargs.get("base") == "direct":
        kwargs.pop("max_inner_steps", None)
    return AGENTS[name](client=client, **kwargs)
