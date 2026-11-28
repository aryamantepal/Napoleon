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
- end_turn finishes your turn, and the game does not advance until you end it. Once the
  blockers are clear, do any clearly useful action, then end the turn. Switching production
  or research back and forth, or moving units around without a purpose, wastes steps.

There are exactly six actions, with exactly these fields:
{"type":"move_unit","unit_id":<int>,"to":{"q":<int>,"r":<int>}}
{"type":"unit_command","unit_id":<int>,"command":"fortify" | "skip" | "found_city"}
{"type":"set_production","city_id":<int>,"item":"<item name>"}
{"type":"set_research","tech":"<tech name>"}
{"type":"set_civic","civic":"<civic name>"}
{"type":"end_turn"}
"""

ACTION_FORMAT = """\
Choose exactly one action. Reply with one JSON object and nothing else:
{"reasoning": "<one or two sentences>", "action": <one entry copied from the legal actions>}
"""

ACTION_FORMAT_NO_COT = """\
Choose exactly one action. Reply with one JSON object and nothing else:
{"action": <one entry copied from the legal actions>}
"""

# For agents whose reply format has a thought or reasoning field: switch it off.
NO_COT_NOTE = (
    'Do not explain your choice. Leave out the "thought" and "reasoning" fields and reply '
    "with the action (or tool call) only."
)

SYSTEM_PROMPT = GAME_RULES + "\n" + ACTION_FORMAT
SYSTEM_PROMPT_NO_COT = GAME_RULES + "\n" + ACTION_FORMAT_NO_COT


def _yields(y: Yields) -> str:
    parts = [f"{k} {v:g}" for k, v in y.model_dump().items() if v]
    return ", ".join(parts) or "none"


def action_json(a: BaseModel) -> str:
    return json.dumps(a.model_dump(mode="json"), separators=(",", ":"))


def render_snapshot(
    snap: Snapshot,
    *,
    max_legal: int = 300,
    tile_radius: int = 2,
    tiles: bool = True,
    legal: bool = True,
) -> str:
    """The snapshot as text. ``tiles=False`` and ``legal=False`` leave those out, for agents
    that look them up through tools instead (the legal list is then summarized by count)."""
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
    if near and tiles:
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

    if not legal:
        counts: dict[str, int] = {}
        for a in snap.legal_actions:
            counts[a.type] = counts.get(a.type, 0) + 1
        summary = ", ".join(f"{k} {v}" for k, v in counts.items())
        lines.append(
            f"Legal actions: {len(snap.legal_actions)} ({summary}). Use list_options to see them."
        )
        return "\n".join(lines)
    shown = snap.legal_actions[:max_legal]
    lines.append(f"Legal actions ({len(snap.legal_actions)}):")
    lines += [action_json(a) for a in shown]
    if len(shown) < len(snap.legal_actions):
        lines.append(f"... {len(snap.legal_actions) - len(shown)} more not shown")
    return "\n".join(lines)


class TurnLog:
    """The actions an agent has chosen so far this turn, for agents without memory.

    A stateless agent cannot tell that it already set research this turn, and will flip
    between two legal choices forever. Showing it its own moves since the turn began is
    the least memory that breaks that loop; it is off by default so the pure stateless
    baseline stays available.
    """

    def __init__(self, limit: int = 20) -> None:
        self.limit = limit
        self.turn: int | None = None
        self.actions: list[str] = []

    def reset(self) -> None:
        self.turn, self.actions = None, []

    def note(self, snap: Snapshot, action: object) -> None:
        if snap.turn != self.turn:
            self.turn, self.actions = snap.turn, []
        text = action_json(action) if isinstance(action, BaseModel) else str(action)[:200]
        self.actions.append(text)

    def render(self, snap: Snapshot) -> str:
        if snap.turn != self.turn or not self.actions:
            return "\n\nYour actions so far this turn: none."
        shown = self.actions[-self.limit :]
        more = len(self.actions) - len(shown)
        head = f"\n\nYour actions so far this turn ({len(self.actions)}"
        head += f", last {len(shown)} shown):" if more else "):"
        return head + "\n" + "\n".join(f"- {a}" for a in shown)


def direct_messages(snap: Snapshot, cot: bool = True, lessons: str = "") -> list[dict[str, str]]:
    """The exact system and user messages the direct agent sends for ``snap``."""
    system = (SYSTEM_PROMPT if cot else SYSTEM_PROMPT_NO_COT) + lessons
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_snapshot(snap)},
    ]


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
        return Reply(None, payload, reasoning, f"invalid action ({_explain(exc)})")


def _explain(exc: ValidationError) -> str:
    """Pydantic's errors, put in words a small model can act on."""
    parts = []
    for err in exc.errors()[:3]:
        where = ".".join(str(p) for p in err["loc"][1:])
        if err["type"] == "union_tag_not_found":
            parts.append("the action has no 'type' field")
        elif err["type"] == "union_tag_invalid":
            parts.append("unknown action type")
        elif err["type"] == "missing":
            parts.append(f"missing field '{where}'")
        elif err["type"] == "extra_forbidden":
            parts.append(f"unexpected field '{where}'")
        else:
            parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def retry_note(error: str, snap: Snapshot) -> str:
    """What to tell a model whose reply was unusable, with a real action as a format example."""
    note = f"That reply was rejected: {error.rstrip('.')}. Reply again with only the JSON object."
    if snap.legal_actions:
        note += f" Actions look like this: {action_json(snap.legal_actions[0])}"
    return note
