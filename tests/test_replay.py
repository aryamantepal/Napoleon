import json
import re
import shutil
import subprocess

import pytest
from stubs import PickLegal

from napoleon.agents import DirectAgent, GreedyAgent
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.replay import FORMAT, build_replay, render_html
from napoleon.runner import run_game
from napoleon.usage import Pricing


def make_run(tmp_path, turns=20):
    path = tmp_path / "g.jsonl"
    run_game(CivEnv(FakeCivGame(max_turns=turns)), GreedyAgent(), path, backend="fake")
    return path


def test_replay_has_a_frame_per_turn_and_every_step(tmp_path):
    path = make_run(tmp_path)
    run = load_run(path)
    replay = build_replay(run)
    assert replay["format"] == FORMAT
    turns = [f["turn"] for f in replay["frames"]]
    assert turns == sorted(set(turns)) and turns[0] == 1
    assert len(replay["steps"]) == len(run.steps)
    assert replay["series"]["score"][-1][1] == replay["frames"][-1]["score"]
    fields = replay["frame_fields"]
    frame = replay["frames"][-1]
    for key in ("tiles", "units", "cities"):
        assert all(len(row) == len(fields[key]) for row in frame[key])


def test_steps_keep_actions_and_reasoning(tmp_path):
    replay = build_replay(load_run(make_run(tmp_path)))
    first = replay["steps"][0]
    assert first["action"]["type"] == "unit_command"
    assert first["reasoning"]
    assert any(s["action"]["type"] == "end_turn" for s in replay["steps"])


def test_html_embeds_the_replay(tmp_path):
    replay = build_replay(load_run(make_run(tmp_path, turns=5)))
    page = render_html(replay)
    assert "/*REPLAY_DATA*/null" not in page
    start = page.index("const REPLAY = ") + len("const REPLAY = ")
    end = page.index(";\nconst NS")
    assert json.loads(page[start:end])["run"]["run_id"] == replay["run"]["run_id"]


def test_cli_writes_both_files(tmp_path, capsys):
    path = make_run(tmp_path, turns=5)
    assert main(["replay", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (tmp_path / "g.replay.json").exists() and (tmp_path / "g.replay.html").exists()
    assert out["turns"] >= 5


# Usage, cost and the viewer ---------------------------------------------------------


@pytest.fixture
def llm_run(tmp_path):
    path = tmp_path / "llm.jsonl"
    client = ScriptedClient(
        PickLegal(), model="stub-1", pricing=Pricing(input_per_mtok=1e6, output_per_mtok=1e6)
    )
    run_game(CivEnv(FakeCivGame(max_turns=12)), DirectAgent(client), path, backend="fake")
    return path


def test_replay_carries_usage_and_cost(llm_run):
    replay = build_replay(load_run(llm_run))
    assert replay["run"]["config"]["model"] == "stub-1"
    first = replay["steps"][0]
    assert first["usage"]["calls"] == 1 and first["usage"]["in"] > 0 and first["usage"]["cost"] > 0
    totals = replay["totals"]
    assert totals["llm_calls"] == len(replay["steps"]) and totals["cost_usd"] > 0
    cost = replay["series"]["cost"]
    values = [v for _, v in cost]
    assert values == sorted(values) and values[-1] == pytest.approx(totals["cost_usd"], abs=1e-5)
    assert replay["series"]["calls"][-1][1] == totals["llm_calls"]
    assert [t for t, _ in cost] == [t for t, _ in replay["series"]["score"]]


def test_scripted_agents_have_no_usage(tmp_path):
    replay = build_replay(load_run(make_run(tmp_path, turns=6)))
    assert all(s["usage"] is None for s in replay["steps"])
    assert replay["totals"]["llm_calls"] == 0 and replay["series"]["cost"][-1][1] == 0


def test_rival_score_series_and_contacts(tmp_path):
    from stubs import ReactPlayer

    from napoleon.agents import ReActAgent

    path = tmp_path / "rival.jsonl"
    run_game(
        CivEnv(FakeCivGame(max_turns=30)),
        ReActAgent(ScriptedClient(ReactPlayer())),
        path,
        backend="fake",
        scenario="bad_opening",
        seed=2,
    )
    replay = build_replay(load_run(path))
    rival = replay["series"]["rival"]
    assert rival and all(len(p) == 2 for p in rival)
    with_contact = [f for f in replay["frames"] if f["contacts"]]
    assert with_contact and len(with_contact[0]["contacts"][0]) == 4


def test_model_text_cannot_break_out_of_the_page(tmp_path):
    path = tmp_path / "x.jsonl"
    evil = "</script><img src=x onerror=alert(1)><!-- <script>"
    client = ScriptedClient(
        lambda msgs: '{"reasoning": ' + json.dumps(evil) + ', "action": {"type": "end_turn"}}'
    )
    run_game(
        CivEnv(FakeCivGame(max_turns=2)), DirectAgent(client), path, backend="fake", max_steps=3
    )
    page = render_html(build_replay(load_run(path)))
    assert "</script><img" not in page and "<!-- <script>" not in page
    start = page.index("const REPLAY = ") + len("const REPLAY = ")
    end = page.index(";\nconst NS")
    assert json.loads(page[start:end])["steps"][0]["reasoning"] == evil


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_viewer_replays_moves_within_a_turn():
    from importlib import resources

    page = resources.files("napoleon").joinpath("replay_page.html").read_text("utf-8")
    pure = re.search(r"// <pure>(.*?)// </pure>", page, re.S).group(1)
    harness = (
        pure
        + """
const frame = {
  units: [[1, 0, "settler", 0, 0, 100], [2, 0, "warrior", 1, 0, 100], [9, 2, "warrior", 2, 0, 80]],
  cities: [[5, 0, "Lutetia", -3, 0, 2, "monument"]],
};
const mv = (id, q, r, ok = true) =>
  ({ ok, action: { type: "move_unit", unit_id: id, to: { q, r } } });
const steps = [
  mv(1, 0, 1),
  mv(2, 2, 0),                                   // onto an enemy: an attack, the unit stays
  mv(1, 5, 5, false),                            // refused: ignored
  { ok: true, action: { type: "unit_command", unit_id: 1, command: "found_city" } },
  { ok: true, action: { type: "set_production", city_id: 5, item: "granary" } },
  { ok: false, action: "garbage" },
];
const at = (k) => stateAt(frame, steps, k);
const out = {
  start: at(null).units.map((u) => [u.id, u.q, u.r]),
  moved: at(0).units.find((u) => u.id === 1),
  attack: [at(1).units.find((u) => u.id === 2), at(1).focus],
  refused: at(2).units.find((u) => u.id === 1),
  founded: [at(3).units.map((u) => u.id), at(3).cities.map((c) => [c.name, c.q, c.r, c.fresh])],
  prod: at(4).cities.find((c) => c.id === 5).prod,
  junk: at(5).focus !== null,
  describe: [describe(steps[0].action), describe("garbage"),
    describe({ type: "end_turn" })],
  usage: [usageText(null), usageText({ calls: 2, in: 10, out: 3, cost: 0.0042 }),
    money(0), money(1.5)],
  original: frame.units[0][3],
};
console.log(JSON.stringify(out));
"""
    )
    result = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, check=True)
    out = json.loads(result.stdout)
    assert out["start"] == [[1, 0, 0], [2, 1, 0], [9, 2, 0]]
    assert (out["moved"]["q"], out["moved"]["r"]) == (0, 1)
    unit, focus = out["attack"]
    assert (unit["q"], unit["r"]) == (1, 0) and focus["attack"] is True
    assert (out["refused"]["q"], out["refused"]["r"]) == (0, 1)
    assert out["founded"][0] == [2, 9] and out["founded"][1][1] == ["new city", 0, 1, True]
    assert out["prod"] == "granary" and out["junk"] is True
    assert out["describe"] == ["move unit 1 to (0, 1)", "garbage", "end turn"]
    assert out["usage"] == [None, "2 calls · 10 in / 3 out · $0.0042", "$0", "$1.50"]
    assert out["original"] == 0


def test_compare_page_overlays_runs(tmp_path, llm_run, capsys):
    greedy = make_run(tmp_path, turns=12)
    out = tmp_path / "cmp.html"
    assert main(["compare", str(greedy), str(llm_run), "--html", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {"runs": 2, "html": str(out)}
    page = out.read_text()
    assert "/*COMPARE_DATA*/null" not in page
    start = page.index("const COMPARE = ") + len("const COMPARE = ")
    data = json.loads(page[start : page.index(";\nconst NS", start)])
    labels = [r["label"] for r in data["runs"]]
    assert labels == ["greedy · default seed 0", "direct (stub-1) · default seed 0"]
    assert data["runs"][1]["totals"]["cost_usd"] > 0 and data["runs"][0]["totals"]["llm_calls"] == 0
