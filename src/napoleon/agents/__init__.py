"""Agents. The sanity agents prove the pipeline end to end; the LLM agents are the experiment."""

from napoleon.agents.base import Agent, Decision
from napoleon.agents.continual import ContinualAgent
from napoleon.agents.direct import DirectAgent
from napoleon.agents.greedy import GreedyAgent
from napoleon.agents.random_agent import RandomAgent
from napoleon.agents.react import ReActAgent
from napoleon.agents.reflexion import ReflexionAgent
from napoleon.agents.voyager import VoyagerAgent

AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "direct": DirectAgent,
    "react": ReActAgent,
    "reflexion": ReflexionAgent,
    "voyager": VoyagerAgent,
    "continual": ContinualAgent,
}

# Agents built around an LLM client; the rest take no arguments.
LLM_AGENTS = frozenset({"direct", "react", "reflexion", "voyager", "continual"})

__all__ = [
    "AGENTS",
    "LLM_AGENTS",
    "Agent",
    "ContinualAgent",
    "Decision",
    "DirectAgent",
    "GreedyAgent",
    "RandomAgent",
    "ReActAgent",
    "ReflexionAgent",
    "VoyagerAgent",
]
