"""Uniform random over legal actions, ending the turn half the time it can."""

from __future__ import annotations

import random

from napoleon.agents.base import Decision
from napoleon.schema import EndTurn, Snapshot


class RandomAgent:
    name = "random"

    def __init__(self, end_turn_prob: float = 0.5) -> None:
        self.end_turn_prob = end_turn_prob
        self.rng = random.Random(0)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, snapshot: Snapshot) -> Decision:
        legal = list(snapshot.legal_actions)
        if not legal:
            return Decision(EndTurn(), "no legal actions")
        if EndTurn() in legal and self.rng.random() < self.end_turn_prob:
            return Decision(EndTurn(), "random end turn")
        return Decision(self.rng.choice(legal), "random legal action")
