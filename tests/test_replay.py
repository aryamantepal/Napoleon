import json

from napoleon.agents import GreedyAgent
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.recording import load_run
from napoleon.replay import FORMAT, build_replay, render_html
from napoleon.runner import run_game


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
    assert (
        json.loads(page[start:end].replace("<\\/", "</"))["run"]["run_id"]
        == replay["run"]["run_id"]
    )


def test_cli_writes_both_files(tmp_path, capsys):
    path = make_run(tmp_path, turns=5)
    assert main(["replay", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (tmp_path / "g.replay.json").exists() and (tmp_path / "g.replay.html").exists()
    assert out["turns"] >= 5
