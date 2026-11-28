"""Few-shot demonstrations for the LLM agents: (situation summary -> action JSON) pairs.

Demonstrations come from greedy play on the fake game (rolled out here, read back from a
recorded run, or loaded from a curated JSONL file). They are chosen by similarity to the
current situation, not from a fixed set. Why: what the right action is depends on the
situation (an idle city, a settler on a good site, a barbarian next to a city), and a fixed
set of three examples would mostly show situations that have nothing to do with the current
one. Similarity here is Jaccard overlap of a few discrete features (blocker kinds, unit types,
legal action types, city count, enemy nearby), which is cheap, deterministic and needs no
embedding model. The picks are also diversified so the K examples do not all repeat one move.

Demonstrations never come from the game being evaluated: the default pool uses seeds from
``DEMO_SEEDS`` (1000 and up, away from the evaluation seeds in this repo), and ``reset(seed)``
drops any demonstration recorded on the seed about to be played.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from napoleon.agents.prompting import action_json
from napoleon.hexgrid import distance
from napoleon.schema import Action, Snapshot

DEMO_SEEDS = tuple(range(1000, 1006))
DEMO_SCENARIOS = ("default", "bad_opening")


class Demo(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: str
    seed: int
    snapshot: Snapshot
    action: Action
    reasoning: str | None = None


# Building demonstrations ----------------------------------------------------------------


def greedy_demos(
    seeds=DEMO_SEEDS, scenarios=DEMO_SCENARIOS, turns: int = 40, *, keep_end_turn: int = 1
) -> list[Demo]:
    """Roll the greedy agent out on the fake game and keep every step it took.

    End-of-turn steps dominate a game, so only the first ``keep_end_turn`` per game is kept.
    """
    from napoleon.agents.greedy import GreedyAgent
    from napoleon.env import CivEnv
    from napoleon.fake import FakeCivGame

    out: list[Demo] = []
    for scenario in scenarios:
        for seed in seeds:
            env, agent = CivEnv(FakeCivGame(max_turns=turns)), GreedyAgent()
            snap, ends = env.reset(scenario, seed), 0
            while not snap.done:
                d = agent.act(snap)
                if d.action.type != "end_turn" or ends < keep_end_turn:
                    out.append(
                        Demo(
                            scenario=scenario,
                            seed=seed,
                            snapshot=snap,
                            action=d.action,
                            reasoning=d.reasoning,
                        )
                    )
                ends += d.action.type == "end_turn"
                snap, _ = env.step(d.action)
    return out


def demos_from_run(path: str | Path) -> list[Demo]:
    """Demonstrations from a recorded fake-game run, by replaying its actions.

    Run files store a full snapshot only at the start of each turn, so the game is replayed
    from the header's scenario and seed. The replay must reproduce every recorded snapshot
    hash, otherwise the run is not from this game and a ``ValueError`` is raised. Rejected
    steps and steps without a typed action are skipped.
    """
    from napoleon.env import CivEnv
    from napoleon.fake import FakeCivGame
    from napoleon.recording import load_run, snapshot_hash

    run = load_run(path)
    h = run.header
    if h.backend != "fake" or run.footer is None:
        raise ValueError(f"{path}: only finished fake-game runs can be replayed")
    env = CivEnv(FakeCivGame(max_turns=run.footer.final_snapshot.max_turns))
    snap = env.reset(h.scenario, h.seed)
    out = []
    for step in run.steps:
        if snapshot_hash(snap) != step.snapshot_hash:
            raise ValueError(f"{path}: step {step.step} does not replay; wrong game version?")
        if step.result.ok and step.action is not None:
            out.append(
                Demo(
                    scenario=h.scenario,
                    seed=h.seed,
                    snapshot=snap,
                    action=step.action,
                    reasoning=step.reasoning,
                )
            )
        snap, _ = env.step(step.action if step.action is not None else step.raw_action)
    return out


def save_demos(demos: list[Demo], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(d.model_dump_json() + "\n" for d in demos), encoding="utf-8")


def load_demos(path: str | Path) -> list[Demo]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Demo.model_validate_json(ln) for ln in lines if ln.strip()]


@lru_cache(maxsize=1)
def default_pool() -> tuple[Demo, ...]:
    return tuple(greedy_demos())


# Choosing and rendering -----------------------------------------------------------------


def situation_features(snap: Snapshot) -> frozenset[str]:
    me = snap.player.id
    feats = {f"blocker:{b.kind}" for b in snap.blockers}
    feats |= {f"unit:{u.type}" for u in snap.units if u.owner == me}
    feats |= {f"legal:{a.type}" for a in snap.legal_actions}
    if any(getattr(a, "command", "") == "found_city" for a in snap.legal_actions):
        feats.add("can_found")
    feats.add(f"cities:{min(len(snap.cities), 4)}")
    mine = [c.coord for c in snap.cities] + [u.coord for u in snap.units if u.owner == me]
    if any(distance(u.coord, c) <= 4 for u in snap.units if u.owner != me for c in mine):
        feats.add("enemy_near")
    if not any(b.kind != "other" for b in snap.blockers):
        feats.add("clear")
    return frozenset(feats)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def render_summary(snap: Snapshot) -> str:
    """A compact stand-in for the full prompt: state in a few lines, legal actions by count."""
    me = snap.player.id
    cities = [
        f"{c.name} {c.id} at ({c.coord.q},{c.coord.r}) pop {c.population} "
        f"building {c.production or 'nothing'}"
        for c in snap.cities
        if c.owner == me
    ]
    units = [
        f"{u.type} {u.id} at ({u.coord.q},{u.coord.r}) moves {u.moves_left}"
        for u in snap.units
        if u.owner == me
    ]
    others = [f"{u.type} at ({u.coord.q},{u.coord.r})" for u in snap.units if u.owner != me]
    blockers = [b.kind + (f" {b.ref_id}" if b.ref_id is not None else "") for b in snap.blockers]
    counts = Counter(a.type for a in snap.legal_actions)
    by_type = ", ".join(f"{k} {v}" for k, v in counts.items())
    lines = [
        f"Turn {snap.turn} of {snap.max_turns}. Score {snap.player.score}, "
        f"gold {snap.player.gold:g}.",
        "Cities: " + ("; ".join(cities) or "none"),
        "Units: " + ("; ".join(units) or "none"),
    ]
    if others:
        lines.append("Other units in sight: " + "; ".join(others))
    lines += [
        f"Research: {snap.tech.current or 'none'}. Civics: {snap.civics.current or 'none'}.",
        "Blockers: " + (", ".join(blockers) or "none"),
        f"Legal actions: {len(snap.legal_actions)} ({by_type})",
    ]
    return "\n".join(lines)


class FewShot:
    """Picks K demonstrations for a snapshot and renders them as chat messages."""

    def __init__(
        self,
        k: int,
        demos: list[Demo] | None = None,
        source: str | Path | None = None,
        cot: bool = True,
    ) -> None:
        self.k = k
        self.cot = cot
        self.source = str(source) if source else "greedy"
        pool = (
            demos if demos is not None else (load_demos(source) if source else list(default_pool()))
        )
        seen: set[tuple] = set()
        self.pool: list[tuple[frozenset[str], Demo]] = []
        for d in pool:
            feats = situation_features(d.snapshot)
            key = (feats, action_json(d.action), d.seed)
            if key not in seen:
                seen.add(key)
                self.pool.append((feats, d))
        self.exclude_seed: int | None = None

    def reset(self, seed: int) -> None:
        self.exclude_seed = seed

    def config(self) -> dict:
        usable = sum(1 for _, d in self.pool if d.seed != self.exclude_seed)
        return {"shots": self.k, "cot": self.cot, "demos": self.source, "demo_pool": usable}

    def pick(self, snap: Snapshot) -> list[Demo]:
        """The K most similar demonstrations, preferring ones with a different action type."""
        target = situation_features(snap)
        ranked = sorted(
            ((_jaccard(target, f), d) for f, d in self.pool if d.seed != self.exclude_seed),
            key=lambda t: (-t[0], t[1].seed, t[1].snapshot.turn, action_json(t[1].action)),
        )
        chosen: list[Demo] = []
        for strict in (True, False):  # first want new action types, then fill any gap
            for _, d in ranked:
                if len(chosen) >= self.k:
                    break
                if any(d is c for c in chosen):
                    continue
                if not strict or all(d.action.type != c.action.type for c in chosen):
                    chosen.append(d)
        return chosen

    def messages(self, snap: Snapshot, key: str = "reasoning") -> list[dict[str, str]]:
        """Alternating user and assistant messages, the demonstration most similar last."""
        picked = self.pick(snap)[::-1]
        out: list[dict[str, str]] = []
        for i, d in enumerate(picked, 1):
            reply = {"action": d.action.model_dump(mode="json")}
            if self.cot and d.reasoning:
                reply = {key: d.reasoning, **reply}
            out.append(
                {
                    "role": "user",
                    "content": f"Example {i} of {len(picked)}.\n{render_summary(d.snapshot)}",
                }
            )
            out.append({"role": "assistant", "content": json.dumps(reply, separators=(",", ":"))})
        return out


def make_fewshot(shots: int, demos: str | Path | None = None, cot: bool = True) -> FewShot | None:
    return FewShot(shots, source=demos, cot=cot) if shots > 0 else None
