"""Build an agent by name from loosely typed options, for the CLI and the sweep."""

from __future__ import annotations

from napoleon.agents import AGENTS, LLM_AGENTS
from napoleon.llm import LLMClient

# The prompting condition: few-shot demonstrations and whether to ask for reasoning.
CONDITION = ("shots", "demos", "cot")
# Agents without memory can be shown their own moves this turn (off by default).
STATELESS = ("turn_log",)

# Options each LLM agent understands. Anything else in ``options`` is ignored, and ``None``
# means "use the agent's default", so one options dict can serve every agent.
OPTIONS: dict[str, tuple[str, ...]] = {
    "direct": ("max_retries", "max_tokens", *CONDITION, *STATELESS),
    "react": ("max_retries", "max_tokens", "max_inner_steps", *CONDITION),
    "reflexion": (
        "max_retries",
        "max_tokens",
        "max_inner_steps",
        *CONDITION,
        "base",
        "memory_path",
        "max_reflections",
    ),
    "voyager": ("max_retries", "max_tokens", "skills_path", "verify", *CONDITION, *STATELESS),
    "continual": (
        "max_retries",
        "max_tokens",
        "max_tool_calls",
        "schedule",
        "rollback",
        "limits",
        "bootstrap_from",
        "bootstrap_update",
        "harness_dir",
        "verify",
        *CONDITION,
    ),
}


def make_agent(name: str, client: LLMClient | None = None, **options):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    if name == "rl":
        if options.get("checkpoint") is None:
            raise ValueError("the rl agent needs a checkpoint: pass --checkpoint models/x.npz")
        agent = AGENTS[name](options["checkpoint"], sample=options.get("sample"))
        agent._load()  # fail now, on a bad file, not at the first move
        return agent
    if name not in LLM_AGENTS:
        return AGENTS[name]()
    allowed = OPTIONS[name]
    kwargs = {k: v for k, v in options.items() if k in allowed and v is not None}
    if name == "reflexion" and kwargs.get("base") == "direct":
        kwargs.pop("max_inner_steps", None)
    return AGENTS[name](client=client, **kwargs)
