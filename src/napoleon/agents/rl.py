"""The ``rl`` agent: plays the fake game with a policy trained by ``napoleon train``."""

from __future__ import annotations

from pathlib import Path

from napoleon.agents.base import Decision
from napoleon.schema import EndTurn, Snapshot


class RLAgent:
    """Scores every legal action with the trained policy and samples from it (PPO) or plays
    the best (Q-learning).

    A PPO policy is stochastic by training, and its argmax can loop forever on moves that are
    equally good, so it samples by default (seeded by the game seed, so runs repeat).
    ``sample=False`` forces argmax.

    Needs numpy (``pip install -e '.[rl]'``), imported only when an agent is built.
    """

    name = "rl"

    def __init__(
        self, checkpoint: str | Path | None = None, *, policy=None, sample: bool | None = None
    ) -> None:
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.sample = sample
        self.policy = policy
        self.meta: dict = {"kind": "in-memory"} if policy is not None else {}
        self._rng = None

    def _load(self) -> None:
        if self.policy is not None:
            return
        if self.checkpoint is None:
            raise ValueError("the rl agent needs a checkpoint: pass --checkpoint models/x.npz")
        from napoleon.rl.policy import Policy

        self.policy, self.meta = Policy.load(self.checkpoint)
        if self.sample is None:
            self.sample = self.meta.get("kind") == "ppo"

    def config(self) -> dict:
        keys = ("kind", "episodes", "hidden", "seed")
        return {
            "checkpoint": self.checkpoint,
            "sample": self.sample,
            **{k: self.meta[k] for k in keys if k in self.meta},
        }

    def reset(self, seed: int) -> None:
        import numpy as np

        self._load()
        if self.sample is None:
            self.sample = False
        self._rng = np.random.default_rng(seed)

    def act(self, snapshot: Snapshot) -> Decision:
        from napoleon.rl.features import step_features
        from napoleon.rl.policy import segment_softmax

        if self.policy is None:
            self._load()
        if self._rng is None:
            self.reset(0)
        if not snapshot.legal_actions:
            return Decision(EndTurn(), "no legal actions")
        import numpy as np

        phi, _ = step_features(snapshot)
        _, p = segment_softmax(
            self.policy.logits(phi), np.array([0]), np.zeros(len(phi), dtype=int)
        )
        i = int(self._rng.choice(len(p), p=p)) if self.sample else int(np.argmax(p))
        return Decision(snapshot.legal_actions[i], f"policy p={p[i]:.2f}")
