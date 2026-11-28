import json

import pytest
from stubs import PickLegal

from napoleon.agents import DirectAgent, GreedyAgent
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.finetune import data as D
from napoleon.finetune import mlx as M
from napoleon.finetune.dagger import DaggerConfig, GameState, dagger
from napoleon.finetune.evaluate import evaluate_student
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.runner import run_game


@pytest.fixture
def snap():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


def greedy_run(tmp_path, seed=5, turns=8):
    path = tmp_path / f"g{seed}.jsonl"
    run_game(CivEnv(FakeCivGame(max_turns=turns)), GreedyAgent(), path, backend="fake", seed=seed)
    return path


def test_chat_record_is_the_agents_exact_prompt_plus_the_target(snap):
    action, why = D.GreedyTeacher().label(snap)
    record = D.to_chat(D.Example(snapshot=snap, action=action, reasoning=why))
    seen = ScriptedClient(PickLegal())
    DirectAgent(seen).act(snap)
    assert record["messages"][:2] == seen.calls[0]  # same system and user text, byte for byte
    reply = json.loads(record["messages"][2]["content"])
    assert reply == {"reasoning": why, "action": action.model_dump(mode="json")}
    bare = D.to_chat(D.Example(snapshot=snap, action=action, reasoning=why), cot=False)
    assert list(json.loads(bare["messages"][2]["content"])) == ["action"]
    no_cot_agent = ScriptedClient(PickLegal())
    DirectAgent(no_cot_agent, cot=False).act(snap)
    assert bare["messages"][:2] == no_cot_agent.calls[0]


def move(snap, nxt, action, ok=True, parsed=True):
    from napoleon.agents.base import Decision
    from napoleon.schema import StepResult

    decision = Decision(action if parsed else {"junk": 1}, "because")
    return D.Move(snap, decision, StepResult(ok=ok, error=None if ok else "illegal"), nxt)


def play(n_moves, agent=None):
    """Real moves from the fake game, with the agent's choices."""
    env, agent = CivEnv(FakeCivGame(max_turns=30)), agent or GreedyAgent()
    s = env.reset(seed=0)
    out = []
    for _ in range(n_moves):
        d = agent.act(s)
        nxt, r = env.step(d.action)
        out.append(D.Move(s, d, r, nxt))
        s = nxt
    return out


def test_process_reward_weights_legality_format_and_score():
    moves = play(8)
    good = D.window_reward(moves)
    assert good.legal_rate == 1.0 and good.format_rate == 1.0 and good.score_gain > 0
    assert good.reward == pytest.approx(1.0)
    from napoleon.schema import StepResult

    bad = [
        D.Move(m.snapshot, m.decision, StepResult(ok=False, error="x"), moves[0].snapshot)
        for m in moves
    ]
    assert D.window_reward(bad).reward == pytest.approx(0.3)  # legal 0, format 1, no gain
    junk = [
        D.Move(m.snapshot, m.decision.__class__({"x": 1}, "r"), m.result, moves[0].snapshot)
        for m in moves
    ]
    assert D.window_reward(junk).format_rate == 0.0
    assert [len(w) for w in D.split_windows(moves, 3)] == [3, 3, 2]


class Marker:
    name = "marker"

    def __init__(self):
        self.seen = []

    def label(self, snapshot):
        from napoleon.schema import EndTurn

        self.seen.append(snapshot.turn)
        return EndTurn(), "teacher"


def test_only_bad_windows_are_relabeled_and_good_ones_keep_the_students_action():
    moves = play(16)
    from napoleon.schema import StepResult

    # make the second window of 8 all rejected
    broken = [
        *moves[:8],
        *(
            D.Move(m.snapshot, m.decision, StepResult(ok=False, error="x"), m.snapshot)
            for m in moves[8:]
        ),
    ]  # noqa: E501
    teacher = Marker()
    examples, scores = D.relabel(broken, teacher, window=8, bad_below=0.75)
    assert [s.reward >= 0.75 for s in scores] == [True, False]
    relabeled = [e for e in examples if e.source == "relabel:marker"]
    assert len(relabeled) == 8 and all(e.action.type == "end_turn" for e in relabeled)
    assert len(teacher.seen) == 8
    kept = [e for e in examples if e.source == "student"]
    assert len(kept) == 8 and [e.action for e in kept] == [m.decision.action for m in broken[:8]]
    none, _ = D.relabel(broken, teacher, window=8, keep_good=False)
    assert all(e.source.startswith("relabel") for e in none)


def test_llm_teacher_falls_back_to_greedy_on_an_illegal_answer(snap):
    illegal = json.dumps({"action": {"type": "set_research", "tech": "nope"}})
    teacher = D.LLMTeacher(ScriptedClient(illegal), cot=True)
    action, _ = teacher.label(snap)
    assert action in snap.legal_actions and teacher.fallbacks == 1
    legal = json.dumps({"action": json.loads(snap.legal_actions[0].model_dump_json())})
    ok = D.LLMTeacher(ScriptedClient(legal))
    assert ok.label(snap)[0] == snap.legal_actions[0] and ok.fallbacks == 0


def test_examples_from_runs_and_a_teacher_relabeling(tmp_path):
    path = greedy_run(tmp_path)
    recorded = D.examples_from_runs([path])
    assert len(recorded) == len(load_run(path).steps) and recorded[0].source == "run:g5.jsonl"
    relabeled = D.examples_from_runs([path], Marker())
    assert {e.source for e in relabeled} == {"marker"} and len(relabeled) == len(recorded)
    assert all(e.action in e.snapshot.legal_actions for e in recorded)


def test_chat_data_files_have_the_layout_mlx_lm_reads(tmp_path):
    examples = D.greedy_examples(seeds=(2000,), scenarios=("default",), turns=6)
    counts = D.write_chat_data(examples, tmp_path / "d", valid_fraction=0.2)
    assert counts["train"] + counts["valid"] == len(examples) and counts["valid"] >= 1
    for name in ("train", "valid"):
        rows = [
            json.loads(ln) for ln in (tmp_path / "d" / f"{name}.jsonl").read_text().splitlines()
        ]
        assert rows and all(
            [m["role"] for m in r["messages"]] == ["system", "user", "assistant"] for r in rows
        )  # noqa: E501
    assert D.TRAIN_SEEDS[0] >= 2000  # away from evaluation seeds and the few-shot pool


def test_lora_command_is_a_plain_mlx_lm_invocation(tmp_path):
    cfg = M.LoraConfig(model="m", iters=7, learning_rate=2e-5)
    cmd = M.lora_command(cfg, tmp_path / "data", tmp_path / "ad")
    assert cmd[1:4] == ["-m", "mlx_lm", "lora"] and "--train" in cmd and "--mask-prompt" in cmd
    assert cmd[cmd.index("--iters") + 1] == "7" and cmd[cmd.index("--model") + 1] == "m"
    assert "--resume-adapter-file" not in cmd
    again = M.lora_command(cfg, tmp_path / "data", tmp_path / "ad", resume=True)
    assert again[again.index("--resume-adapter-file") + 1].endswith(M.ADAPTER_FILE)


def test_missing_mlx_lm_is_a_clear_error_not_a_crash(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(M.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(M.MLXUnavailable, match=r"\.\[mlx\]"):
        M.MLXClient("m")
    code = main(["finetune", "eval", "--out", str(tmp_path)])
    assert code == 2 and "pip install" in capsys.readouterr().err


def test_cli_data_and_sft_dry_run(tmp_path, capsys):
    out = tmp_path / "d"
    assert main(["finetune", "data", "--out", str(out), "--seeds", "2000", "--turns", "6"]) == 0
    assert json.loads(capsys.readouterr().out)["train"] > 0
    code = main(["finetune", "sft", "--out", str(tmp_path / "ad"), "--dry-run", "--iters", "9",
                 "--from", str(greedy_run(tmp_path))])  # fmt: skip
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert "--train" in printed["command"] and printed["train"] > 0


def test_game_state_round_trips_through_a_file(tmp_path):
    state = GameState.start(("default",), (2000,), 0, 10)
    for _ in range(5):
        state.snapshot, _ = state.env.step(GreedyAgent().act(state.snapshot).action)
    state.save(tmp_path / "s.pkl")
    back = GameState.load(tmp_path / "s.pkl")
    assert back.snapshot == state.snapshot and (back.scenario, back.seed) == ("default", 2000)
    a, b = GreedyAgent().act(state.snapshot).action, GreedyAgent().act(back.snapshot).action
    assert state.env.step(a)[0] == back.env.step(b)[0]


class Student:
    """Plays legal actions but is illegal on every 4th call, like a model that sometimes slips."""

    def __init__(self):
        self.n = 0
        self.pick = PickLegal()

    def __call__(self, messages):
        self.n += 1
        if self.n % 4 == 0:
            return json.dumps({"action": {"type": "set_research", "tech": "nope"}})
        return self.pick(messages)


def test_dagger_is_reset_free_aggregates_data_and_trains_softly(tmp_path):
    adapters_seen, trained = [], []

    def client_factory(adapter):
        adapters_seen.append(adapter)
        return ScriptedClient(Student())

    def trainer(data, adapter, resume):
        trained.append((resume, sum(1 for _ in (data / "train.jsonl").open())))
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / M.ADAPTER_FILE).write_bytes(b"x")

    cfg = DaggerConfig(iterations=2, steps_per_iter=16, window=4, turns=20)
    rows = dagger(tmp_path / "run", D.GreedyTeacher(), cfg, model="m",
                  client_factory=client_factory, trainer=trainer)  # fmt: skip
    assert [r["iteration"] for r in rows] == [1, 2]
    # reset-free: round 2 continues the game round 1 left, it does not start a new one
    assert rows[1]["game"]["from_turn"] >= rows[0]["game"]["from_turn"]
    assert rows[1]["game"]["seed"] == rows[0]["game"]["seed"] and not rows[0]["game_ended"]
    # the first round trains the base model (no adapter yet), the second continues the adapter
    assert adapters_seen[0] is None and adapters_seen[1] == tmp_path / "run" / "adapters"
    assert [t[0] for t in trained] == [False, True]
    assert trained[1][1] > trained[0][1] and rows[1]["pool"] > rows[0]["pool"]  # DAgger aggregation
    assert rows[0]["bad_windows"] > 0 and rows[0]["illegal_rate"] > 0 and rows[0]["trained"]
    assert (tmp_path / "run" / "iter-01.jsonl").exists() and (
        tmp_path / "run" / "state.pkl"
    ).exists()
    # a new process picks up where this one stopped
    again_cfg = DaggerConfig(iterations=1, steps_per_iter=8, turns=20)
    more = dagger(tmp_path / "run", D.GreedyTeacher(), again_cfg, model="m",
                  client_factory=client_factory, trainer=trainer)  # fmt: skip
    assert more[0]["iteration"] == 3 and more[0]["game"]["seed"] == rows[0]["game"]["seed"]
    log = (tmp_path / "run" / "dagger_log.jsonl").read_text().splitlines()
    assert len(log) == 3


def test_dagger_starts_the_next_game_when_one_ends(tmp_path):
    def trainer(data, adapter, resume):
        adapter.mkdir(parents=True, exist_ok=True)

    cfg = DaggerConfig(iterations=3, steps_per_iter=60, turns=2, window=6)
    rows = dagger(tmp_path / "r", D.GreedyTeacher(), cfg, model="m", trainer=trainer,
                  client_factory=lambda a: ScriptedClient(PickLegal()))  # fmt: skip
    assert rows[0]["game_ended"]
    assert (
        rows[1]["game"]["from_turn"] == 1
        and rows[1]["game"]["scenario"] != rows[0]["game"]["scenario"]
    )  # noqa: E501


def test_evaluate_student_reports_illegal_and_format_rates(tmp_path):
    replies = iter(["not json", json.dumps({"action": {"type": "set_research", "tech": "x"}}), "x"])

    def reply(messages):
        return next(replies, json.dumps({"action": {"type": "end_turn"}}))

    res = evaluate_student(lambda: ScriptedClient(reply), tmp_path, seeds=(0,), turns=3,
                           max_steps=6, label="t")  # fmt: skip
    assert res["games"] == 1 and res["format_invalid_rate"] == pytest.approx(2 / 6, abs=1e-3)
    assert res["illegal_rate"] == 1.0  # the end_turn replies are blocked by open orders
    assert (tmp_path / "t-default-seed0.jsonl").exists()
