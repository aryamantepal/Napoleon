"""Train policies on the fake game: PPO and a linear Q-learning baseline.

Reward (per agent step, documented in docs/rl.md):

    r_t = score_scale * (score_{t+1} - score_t) - step_cost
    plus, on the last step, terminal_per_point * final_score if the game finished,
    or -truncation_penalty if the step cap ended the episode first.

Score only moves when the game moves (a city is founded, a turn ends), so the score term is
"score delta per turn" spread over the steps that caused it. The step cost keeps the policy
from shuffling units without ending the turn, and the terminal term puts extra weight on
how the game ends.

Training episode ``i`` uses seed ``TRAIN_SEED_BASE + i``, never repeated and far from the
seeds anyone evaluates on (``0`` up), so evaluation seeds are never trained on.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.rl import features as F
from napoleon.rl.policy import Adam, Policy, segment_softmax

TRAIN_SEED_BASE = 10_000


class Reward(BaseModel):
    score_scale: float = 0.1
    step_cost: float = 0.01
    terminal_per_point: float = 0.05
    truncation_penalty: float = 2.0


@dataclass
class Episode:
    phis: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    chosen: list[int] = field(default_factory=list)
    logps: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    final_score: int = 0
    truncated: bool = False

    @property
    def ret(self) -> float:
        return float(sum(self.rewards))


def play(
    policy: Policy,
    scenario: str,
    seed: int,
    turns: int,
    rng: np.random.Generator,
    reward: Reward,
    *,
    max_steps: int,
    epsilon: float | None = None,
) -> Episode:
    """One game. Actions are sampled from the policy; with ``epsilon`` set (Q-learning) they
    are epsilon-greedy on the scores instead."""
    env = CivEnv(FakeCivGame(max_turns=turns))
    snap = env.reset(scenario, seed)
    ep = Episode()
    for _ in range(max_steps):
        if snap.done:
            break
        phi, state = F.step_features(snap)
        logp, p = segment_softmax(policy.logits(phi), np.array([0]), np.zeros(len(phi), dtype=int))
        if epsilon is None:
            a = int(rng.choice(len(p), p=p))
        else:
            a = int(rng.integers(len(phi))) if rng.random() < epsilon else int(np.argmax(p))
        nxt, _ = env.step(snap.legal_actions[a])
        r = reward.score_scale * (nxt.player.score - snap.player.score) - reward.step_cost
        if nxt.done:
            r += reward.terminal_per_point * nxt.player.score
        ep.phis.append(phi)
        ep.states.append(state)
        ep.chosen.append(a)
        ep.logps.append(float(logp[a]))
        ep.rewards.append(r)
        snap = nxt
    if not snap.done:
        ep.truncated = True
        ep.rewards[-1] -= reward.truncation_penalty
    ep.final_score = snap.player.score
    return ep


# Evaluation ------------------------------------------------------------------------------


def evaluate(
    make_agent: Callable[[], object], scenarios: Sequence[str], seeds: Sequence[int], turns: int
) -> dict[str, float]:
    """Mean final score per scenario for an agent, one fresh agent per game."""
    out: dict[str, float] = {}
    for scenario in scenarios:
        scores = []
        for seed in seeds:
            env, agent = CivEnv(FakeCivGame(max_turns=turns)), make_agent()
            agent.reset(seed)
            snap, steps = env.reset(scenario, seed), 0
            while not snap.done and steps < turns * 15:
                snap, _ = env.step(agent.act(snap).action)
                steps += 1
            scores.append(snap.player.score)
        out[scenario] = float(np.mean(scores))
    return out


# PPO -------------------------------------------------------------------------------------


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float):
    adv, last = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


class PPOConfig(BaseModel):
    hidden: int = 32
    lr: float = 3e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    batch_episodes: int = 16
    target_kl: float = 0.03


def ppo_update(policy: Policy, opt: Adam, episodes: list[Episode], cfg: PPOConfig) -> dict:
    advs, rets = [], []
    for ep in episodes:
        values = policy.value(np.array(ep.states))
        a, r = gae(np.array(ep.rewards), values, cfg.gamma, cfg.lam)
        advs.append(a)
        rets.append(r)
    adv, ret = np.concatenate(advs), np.concatenate(rets)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    phi = np.concatenate([p for ep in episodes for p in ep.phis])
    counts = np.array([len(p) for ep in episodes for p in ep.phis])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    seg = np.repeat(np.arange(len(counts)), counts)
    chosen = starts + np.array([c for ep in episodes for c in ep.chosen])
    old = np.array([lp for ep in episodes for lp in ep.logps])
    states = np.concatenate([ep.states for ep in episodes])
    n = len(counts)
    onehot = np.zeros(len(phi))
    onehot[chosen] = 1.0
    info = {}
    for _ in range(cfg.epochs):
        z, h = policy._forward(phi)
        logp_all, p = segment_softmax(z, starts, seg)
        logp = logp_all[chosen]
        ratio = np.exp(logp - old)
        active = ((adv >= 0) & (ratio < 1 + cfg.clip)) | ((adv < 0) & (ratio > 1 - cfg.clip))
        ent = -np.add.reduceat(p * logp_all, starts)
        g_lp = -adv * ratio * active / n
        dz = g_lp[seg] * (onehot - p) + (cfg.entropy / n) * p * (logp_all + ent[seg])
        grads = policy.backward_logits(phi, h, dz)
        v, vh = policy._value_forward(states)
        grads |= policy.backward_value(states, vh, cfg.value_coef * (v - ret) / n)
        opt.step(policy.params, grads)
        kl = float(np.mean(old - logp))
        info = {
            "kl": kl,
            "entropy": float(ent.mean()),
            "value_loss": float(np.mean((v - ret) ** 2)),
        }
        if kl > cfg.target_kl:
            break
    return info


# Training loops --------------------------------------------------------------------------


@dataclass
class TrainResult:
    policy: Policy
    curve: list[dict]
    seconds: float


def _log_row(done, eps, scenarios, evals, extra):
    row = {
        "episodes": done,
        "train_return": round(float(np.mean([e.ret for e in eps])), 4),
        "train_score": round(float(np.mean([e.final_score for e in eps])), 2),
        "truncated": round(float(np.mean([e.truncated for e in eps])), 3),
        "eval_mean": round(float(np.mean(list(evals.values()))), 2),
        **{f"eval_{s}": round(v, 2) for s, v in evals.items()},
        **extra,
    }
    return row


def train(
    agent: str,
    episodes: int,
    scenarios: Sequence[str],
    *,
    turns: int = 30,
    eval_turns: int = 50,
    eval_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    eval_every: int = 64,
    seed: int = 0,
    cfg: PPOConfig | None = None,
    reward: Reward | None = None,
    max_steps_per_turn: int = 12,
    alpha: float = 0.003,
    log: Callable[[str], None] | None = None,
) -> TrainResult:
    """Train ``agent`` ('ppo' or 'q') for ``episodes`` games. Returns the policy and a curve."""
    from napoleon.agents.rl import RLAgent

    if max(eval_seeds) >= TRAIN_SEED_BASE:
        raise ValueError(f"eval seeds must stay below the training seeds ({TRAIN_SEED_BASE})")
    cfg, reward = cfg or PPOConfig(), reward or Reward()
    rng = np.random.default_rng(seed)
    policy = Policy(cfg.hidden if agent == "ppo" else 0, rng=rng)
    opt = Adam(policy.params, cfg.lr)
    start, curve, done = time.time(), [], 0
    batch = cfg.batch_episodes if agent == "ppo" else 1
    next_eval = eval_every
    while done < episodes:
        eps = []
        for _ in range(min(batch, episodes - done)):
            scenario = scenarios[(done + len(eps)) % len(scenarios)]
            eps_seed = TRAIN_SEED_BASE + done + len(eps)
            epsilon = max(0.02, 0.3 * (1 - done / max(1, episodes))) if agent == "q" else None
            ep = play(policy, scenario, eps_seed, turns, rng, reward,
                      max_steps=turns * max_steps_per_turn, epsilon=epsilon)  # fmt: skip
            if agent == "q":
                q_update(policy, ep, cfg.gamma, alpha)
            eps.append(ep)
        extra = ppo_update(policy, opt, eps, cfg) if agent == "ppo" else {}
        done += len(eps)
        if done >= next_eval or done >= episodes:
            next_eval = done + eval_every
            make = lambda: RLAgent(policy=policy, sample=agent == "ppo")  # noqa: E731
            evals = evaluate(make, scenarios, eval_seeds, eval_turns)
            row = _log_row(done, eps, scenarios, evals, {k: round(v, 4) for k, v in extra.items()})
            row["seconds"] = round(time.time() - start, 1)
            curve.append(row)
            if log:
                log(" ".join(f"{k}={v}" for k, v in row.items()))
    return TrainResult(policy, curve, time.time() - start)


def q_update(policy: Policy, ep: Episode, gamma: float, alpha: float) -> None:
    """One semi-gradient Q-learning sweep over an episode (Q is linear in the features)."""
    w = policy.params["w"]
    for t, (phi, a, r) in enumerate(zip(ep.phis, ep.chosen, ep.rewards, strict=True)):
        last = t + 1 == len(ep.phis)
        target = r if last else r + gamma * float(np.max(ep.phis[t + 1] @ w))
        w += alpha * (target - float(phi[a] @ w)) * phi[a]


def write_curve(curve: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in curve for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(curve)
