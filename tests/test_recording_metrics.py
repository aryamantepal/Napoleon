import json

import pytest

from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.metrics import compute_metrics, median_trajectory, recovery_turn
from napoleon.recording import Recorder, RunHeader, load_run
from napoleon.schema import EndTurn, SetCivic, SetProduction, SetResearch, UnitCommand


def play(env: CivEnv, rec: Recorder) -> None:
    """Found a city, then keep clearing blockers and ending turns until the game ends."""
    snap = env.observe()
    while not snap.done:
        action = choose(snap)
        nxt, result = env.step(action)
        rec.record(snap, action, result, reasoning=f"step for {action.type}")
        snap = nxt
    rec.close(snap)


def choose(snap):
    for u in snap.units:
        if (
            u.type == "settler"
            and UnitCommand(unit_id=u.id, command="found_city") in snap.legal_actions
        ):
            return UnitCommand(unit_id=u.id, command="found_city")
    for b in snap.blockers:
        if b.kind == "unit_needs_orders":
            return UnitCommand(unit_id=b.ref_id, command="skip")
        if b.kind == "city_needs_production":
            return SetProduction(city_id=b.ref_id, item="warrior")
        if b.kind == "research_needed":
            return SetResearch(tech=snap.tech.available[0])
        if b.kind == "civic_needed":
            return SetCivic(civic=snap.civics.available[0])
    return EndTurn()


@pytest.fixture
def run_file(tmp_path):
    env = CivEnv(FakeCivGame(max_turns=12))
    env.reset(seed=1)
    path = tmp_path / "run.jsonl"
    with Recorder(path, RunHeader(backend="fake", scenario="default", seed=1, agent="test")) as rec:
        play(env, rec)
    return path


def test_run_file_has_header_steps_and_footer(run_file):
    lines = [json.loads(line) for line in run_file.read_text().splitlines()]
    assert lines[0]["kind"] == "header"
    assert lines[-1]["kind"] == "footer"
    assert all(line["kind"] == "step" for line in lines[1:-1])


def test_full_snapshot_once_per_turn(run_file):
    run = load_run(run_file)
    full = [s for s in run.steps if s.snapshot is not None]
    assert [s.turn for s in full] == sorted({s.turn for s in run.steps})
    assert all(s.reasoning for s in run.steps)


def test_metrics_from_run(run_file):
    m = compute_metrics(load_run(run_file))
    assert m.turns_played == 12
    assert m.failed_steps == 0
    assert m.final_score > 0
    scores = [v for _, v in m.score_by_turn]
    assert scores == sorted(scores)
    assert m.cities_by_turn[-1][1] >= 1


def test_truncated_run_still_loads(run_file, tmp_path):
    lines = run_file.read_text().splitlines()
    cut = tmp_path / "cut.jsonl"
    cut.write_text("\n".join(lines[:-1]) + "\n")
    run = load_run(cut)
    assert run.footer is None
    assert compute_metrics(run).steps == len(lines) - 2


def test_illegal_actions_are_recorded_and_counted(tmp_path):
    env = CivEnv(FakeCivGame(max_turns=3))
    snap = env.reset()
    with Recorder(
        tmp_path / "r.jsonl", RunHeader(backend="fake", scenario="default", seed=0, agent="t")
    ) as rec:
        _, result = env.step({"type": "launch_nukes"})
        rec.record(snap, {"type": "launch_nukes"}, result)
        rec.close(env.observe(), reason="stopped")
    run = load_run(tmp_path / "r.jsonl")
    assert run.steps[0].raw_action == {"type": "launch_nukes"}
    assert compute_metrics(run).illegal_action_rate == 1.0
    assert run.footer.reason == "stopped"


def test_median_trajectory_uses_shared_turns():
    a = [(1, 1), (2, 4), (3, 9)]
    b = [(1, 3), (2, 2)]
    c = [(1, 2), (2, 6)]
    assert median_trajectory([a, b, c]) == [(1, 2), (2, 4)]


def test_recovery_turn():
    ref = [(1, 5), (2, 6), (3, 7), (4, 8)]
    assert recovery_turn([(1, 1), (2, 6), (3, 6), (4, 9)], ref) == 4
    assert recovery_turn([(1, 1), (2, 2), (3, 3), (4, 4)], ref) is None
    assert recovery_turn([(1, 5), (2, 7), (3, 7), (4, 8)], ref) == 1
