"""A softmax policy over a variable list of legal actions, in numpy, with a value baseline.

The scorer maps each action's feature row to one logit: a linear map (``hidden=0``) or a
one-hidden-layer tanh MLP. The value baseline is a small MLP on the state vector. Gradients
are written out by hand, over flattened batches: every step contributes as many rows as it
had legal actions, and ``starts`` marks where each step's rows begin, so the softmax over a
step's legal actions is a segment operation and action masking needs no padding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from napoleon.rl import features as F


class Policy:
    def __init__(
        self, hidden: int = 32, value_hidden: int = 16, rng: np.random.Generator | None = None
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.hidden = hidden
        self.value_hidden = value_hidden
        p: dict[str, np.ndarray] = {}
        if hidden:
            p["W1"] = rng.normal(0, 1 / np.sqrt(F.DIM), (F.DIM, hidden))
            p["b1"] = np.zeros(hidden)
            p["w2"] = rng.normal(0, 0.1, hidden)
        else:
            p["w"] = np.zeros(F.DIM)
        p["V1"] = rng.normal(0, 1 / np.sqrt(F.STATE_DIM), (F.STATE_DIM, value_hidden))
        p["c1"] = np.zeros(value_hidden)
        p["v2"] = rng.normal(0, 0.1, value_hidden)
        self.params = p

    # Forward and backward ------------------------------------------------------------

    def logits(self, phi: np.ndarray) -> np.ndarray:
        return self._forward(phi)[0]

    def _forward(self, phi: np.ndarray):
        p = self.params
        if not self.hidden:
            return phi @ p["w"], None
        h = np.tanh(phi @ p["W1"] + p["b1"])
        return h @ p["w2"], h

    def backward_logits(self, phi: np.ndarray, h, dz: np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        if not self.hidden:
            return {"w": phi.T @ dz}
        dpre = (dz[:, None] * p["w2"][None, :]) * (1 - h * h)
        return {"W1": phi.T @ dpre, "b1": dpre.sum(0), "w2": h.T @ dz}

    def value(self, s: np.ndarray) -> np.ndarray:
        return self._value_forward(s)[0]

    def _value_forward(self, s: np.ndarray):
        p = self.params
        h = np.tanh(s @ p["V1"] + p["c1"])
        return h @ p["v2"], h

    def backward_value(self, s: np.ndarray, h, dv: np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        dpre = (dv[:, None] * p["v2"][None, :]) * (1 - h * h)
        return {"V1": s.T @ dpre, "c1": dpre.sum(0), "v2": h.T @ dv}

    # Checkpoints ---------------------------------------------------------------------

    def save(self, path: str | Path, meta: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta = {**meta, "hidden": self.hidden, "value_hidden": self.value_hidden}
        meta |= {"feature_version": F.VERSION, "feature_dim": F.DIM}
        np.savez(path, meta=np.array(json.dumps(meta)), **self.params)

    @classmethod
    def load(cls, path: str | Path) -> tuple[Policy, dict]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("feature_version") != F.VERSION or meta.get("feature_dim") != F.DIM:
                raise ValueError(f"{path}: trained with different features than this version")
            policy = cls(meta["hidden"], meta["value_hidden"])
            policy.params = {k: data[k] for k in policy.params}
        return policy, meta


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-3) -> None:
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def segment_softmax(z: np.ndarray, starts: np.ndarray, seg: np.ndarray):
    """Per-step log-probabilities and probabilities over flattened logits."""
    zs = z - np.maximum.reduceat(z, starts)[seg]
    e = np.exp(zs)
    denom = np.add.reduceat(e, starts)
    return zs - np.log(denom)[seg], e / denom[seg]
