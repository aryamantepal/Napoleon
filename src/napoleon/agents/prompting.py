"""Turn a snapshot into prompt text, and a model reply into an action.

Shared by every LLM-backed agent, so they all see the game the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from napoleon.hexgrid import distance
from napoleon.llm import clean_reply, extract_json
from napoleon.schema import Action, Snapshot, Yields, parse_action

GAME_RULES = """\
You are playing a turn-based 4X strategy game (a small Civilization-like game) as player 0.
Each turn you issue actions one at a time. Between actions the game state is shown again.
Goals: grow your score and victory progress by expanding, building, and researching, and keep
your cities and units safe from barbarians and the rival.

Rules of the game:
- Coordinates are axial hex coordinates (q, r). Moving a unit onto an enemy unit's tile attacks it.
- A settler can found a city with the unit command found_city on a legal site.
- Blockers must be resolved before end_turn is allowed: give idle units orders (fortify or
  skip), pick production for idle cities, and pick research and civics when asked.
- Only the actions in the "Legal actions" list can succeed. Anything else is rejected and
  wastes a step.
- end_turn finishes your turn. Unfinished business is not carried out for you.
"""

ACTION_FORMAT = """\
Choose exactly one action. Reply with one JSON object and nothing else:
{"reasoning": "<one or two sentences>", "action": <one entry copied from the legal actions>}
"""

SYSTEM_PROMPT = GAME_RULES + "\n" + ACTION_FORMAT


def _yields(y: Yields) -> str:
    parts = [f"{k} {v:g}" for k, v in y.model_dump().items() if v]
    return ", ".join(parts) or "none"


def action_json(a: BaseModel) -> str:
    return json.dumps(a.model_dump(mode="json"), separators=(",", ":"))


def render_snapshot(snap: Snapshot, *, max_legal: int = 300, tile_radius: int = 2) -> str:
    me = snap.player.id
    lines = [
        f"Turn {snap.turn} of {snap.max_turns}.",
        f"You: player {me} ({snap.player.civ}), score {snap.player.score}, "
        f"gold {snap.player.gold:g}, yields per turn: {_yields(snap.player.yields)}.",
    ]
    lines.append("Your cities:" if snap.cities else "Your cities: none yet.")
    for c in snap.cities:
        if c.owner != me:
            continue
        build = "nothing"
        if c.production:
            cost = f"/{c.production_cost:g}" if c.production_cost else ""
            build = f"{c.production} {c.production_progress:g}{cost}"
        lines.append(
            f"- city {c.id} {c.name} at ({c.coord.q},{c.coord.r}), pop {c.population}, "
            f"hp {c.hp}, building {build}, yields: {_yields(c.yields)}"
            + (", capital" if c.is_capital else "")
        )
    mine = [u for u in snap.units if u.owner == me]
    other = [u for u in snap.units if u.owner != me]
    lines.append("Your units:" if mine else "Your units: none.")
    for u in mine:
        lines.append(
            f"- unit {u.id} {u.type} at ({u.coord.q},{u.coord.r}), hp {u.hp}, "
            f"moves {u.moves_left}/{u.max_moves}" + (", fortified" if u.fortified else "")
        )
    if other:
        lines.append("Other units in sight:")
        for u in other:
            lines.append(f"- {u.type} of player {u.owner} at ({u.coord.q},{u.coord.r}), hp {u.hp}")
    for label, tree in (("Research", snap.tech), ("Civics", snap.civics)):
        cur = "none"
        if tree.current:
            cost = f"/{tree.cost:g}" if tree.cost else ""
            cur = f"{tree.current} {tree.progress:g}{cost}"
        lines.append(
            f"{label}: current {cur}; done: {', '.join(tree.completed) or 'none'}; "
            f"available: {', '.join(tree.available) or 'none'}"
        )
    for ct in snap.contacts:
        lines.append(
            f"Contact: player {ct.player_id} ({ct.civ}), {'at war' if ct.at_war else 'at peace'}"
            + (f", score {ct.score}" if ct.score is not None else "")
        )
    if snap.victory_progress:
        prog = ", ".join(f"{k} {v:.0%}" for k, v in snap.victory_progress.items())
        lines.append(f"Victory progress: {prog}")
    if snap.blockers:
        lines.append("Blockers (resolve before end_turn):")
        for b in snap.blockers:
            lines.append(
                f"- {b.kind}"
                + (f" ref {b.ref_id}" if b.ref_id is not None else "")
                + (f": {b.message}" if b.message else "")
            )
    else:
        lines.append("Blockers: none, you may end the turn.")

    anchors = [c.coord for c in snap.cities if c.owner == me] + [u.coord for u in mine]
    near = [t for t in snap.tiles if any(distance(t.coord, a) <= tile_radius for a in anchors)]
    if near:
        lines.append(f"Known tiles within {tile_radius} of your cities and units:")
        for t in near[:80]:
            extra = "".join(
                [
                    f" {t.feature}" if t.feature else "",
                    f" {t.resource}" if t.resource else "",
                    " river" if t.river else "",
                    f" owner {t.owner}" if t.owner is not None else "",
                    "" if t.visible else " (fogged)",
                ]
            )
            lines.append(f"- ({t.coord.q},{t.coord.r}) {t.terrain}{extra}")

    legal = snap.legal_actions[:max_legal]
    lines.append(f"Legal actions ({len(snap.legal_actions)}):")
    lines += [action_json(a) for a in legal]
    if len(legal) < len(snap.legal_actions):
        lines.append(f"... {len(snap.legal_actions) - len(legal)} more not shown")
    return "\n".join(lines)


@dataclass(frozen=True)
class Reply:
    """A parsed model reply. ``action`` is ``None`` when the reply could not be used."""

    action: Action | None
    raw: dict | str
    reasoning: str | None = None
    error: str | None = None


def parse_reply(text: str) -> Reply:
    """Read ``{"reasoning": ..., "action": {...}}``, or a bare action object, from a reply."""
    obj = extract_json(text)
    if obj is None:
        return Reply(None, clean_reply(text)[:500], error="no JSON object found in the reply")
    reasoning = None
    if isinstance(obj.get("action"), dict):
        payload = obj["action"]
        reasoning = obj.get("reasoning") or obj.get("thought")
    else:
        payload = {k: v for k, v in obj.items() if k not in ("reasoning", "thought")}
        reasoning = obj.get("reasoning") or obj.get("thought")
    reasoning = reasoning if isinstance(reasoning, str) else None
    try:
        return Reply(parse_action(payload), payload, reasoning)
    except ValidationError as exc:
        err = exc.errors()[0]
        where = ".".join(str(p) for p in err["loc"])
        return Reply(None, payload, reasoning, f"invalid action ({where}: {err['msg']})")
