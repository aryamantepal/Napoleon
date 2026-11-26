"""Agents. The sanity agents here exist to prove the pipeline end to end."""

from napoleon.agents.base import Agent, Decision
from napoleon.agents.greedy import GreedyAgent
from napoleon.agents.random_agent import RandomAgent

AGENTS: dict[str, type] = {"random": RandomAgent, "greedy": GreedyAgent}

__all__ = ["AGENTS", "Agent", "Decision", "GreedyAgent", "RandomAgent"]
