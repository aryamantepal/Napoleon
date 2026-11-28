import json

import pytest

np = pytest.importorskip("numpy")  # the rl extra; the rest of the suite never needs it

from napoleon.agents import AGENTS, RLAgent  # noqa: E402
from napoleon.agents.factory import make_agent  # noqa: E402
from napoleon.cli import main  # noqa: E402
from napoleon.env import CivEnv  # noqa: E402
from napoleon.fake import FakeCivGame  # noqa: E402
from napoleon.recording import load_run  # noqa: E402
from napoleon.rl import features as F  # noqa: E402
from napoleon.rl import train as T  # noqa: E402
from napoleon.rl.policy import Adam, Policy, segment_softmax  # noqa: E402
from napoleon.runner import run_game  # noqa: E402


@pytest.fixture
def snap():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


def test_features_have_one_row_per_legal_action(snap):
    phi, state = F.step_features(snap)
    assert phi.shape == (len(snap.legal_actions), F.DIM) and state.shape == (F.STATE_DIM,)
    assert np.isfinite(phi).all()
    found = [
        i for i, a in enumerate(snap.legal_actions) if getattr(a, "command", "") == "found_city"
    ]
    assert found and phi[found[0], F.TYPES.index("unit_command")] == 1.0
    # the same action in two situations differs in its crossed features, not its own
    assert phi[found[0], F.N_ACTION + F.N_GLOBAL :].any()


def test_features_follow_a_changing_legal_list():
    env = CivEnv(FakeCivGame(max_turns=20))
    s = env.reset(seed=0)
    sizes = set()
    for _ in range(12):
        sizes.add(F.step_features(s)[0].shape[0])
        s, _ = env.step(s.legal_actions[0])
    assert len(sizes) > 1  # no fixed action vocabulary


def test_segment_softmax_is_a_softmax_per_step():
    z = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    starts = np.array([0, 3])
    seg = np.array([0, 0, 0, 1, 1])
    logp, p = segment_softmax(z, starts, seg)
    assert np.allclose([p[:3].sum(), p[3:].sum()], 1.0) and np.allclose(np.exp(logp), p)
    assert np.allclose(p[3:], 0.5)


@pytest.mark.parametrize("hidden", [0, 8])
def test_hand_written_gradients_match_finite_differences(hidden):
    rng = np.random.default_rng(1)
    pol = Policy(hidden, rng=rng)
    pol.params = {k: v + rng.normal(0, 0.1, v.shape) for k, v in pol.params.items()}
    phi, s = rng.normal(size=(7, F.DIM)), rng.normal(size=(5, F.STATE_DIM))
    wz, wv = rng.normal(size=7), rng.normal(size=5)

    def loss():
        return float(wz @ pol.logits(phi) + wv @ pol.value(s))

    z, h = pol._forward(phi)
    v, vh = pol._value_forward(s)
    grads = pol.backward_logits(phi, h, wz) | pol.backward_value(s, vh, wv)
    for name, g in grads.items():
        flat = pol.params[name].reshape(-1)
        for i in rng.choice(flat.size, size=min(4, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            up = loss()
            flat[i] = old - 1e-6
            down = loss()
            flat[i] = old
            assert abs((up - down) / 2e-6 - g.reshape(-1)[i]) < 1e-5, name


def test_gae_matches_a_hand_computation():
    adv, ret = T.gae(np.array([1.0, 0.0, 2.0]), np.array([0.5, 0.5, 0.5]), 0.9, 1.0)
    # lambda=1: advantage is the discounted return minus the value
    g2 = 2.0
    g1 = 0.0 + 0.9 * g2
    g0 = 1.0 + 0.9 * g1
    assert np.allclose(adv, [g0 - 0.5, g1 - 0.5, g2 - 0.5]) and np.allclose(ret, adv + 0.5)


def test_ppo_raises_the_probability_of_the_rewarded_action():
    rng = np.random.default_rng(0)
    pol = Policy(8, rng=rng)
    phi = rng.normal(size=(3, F.DIM))
    state = rng.normal(size=F.STATE_DIM)
    opt = Adam(pol.params, 1e-2)
    cfg = T.PPOConfig(batch_episodes=8, epochs=4, entropy=0.0)

    def prob():
        return segment_softmax(pol.logits(phi), np.array([0]), np.zeros(3, dtype=int))[1]

    before = prob()
    for _ in range(30):
        eps = []
        for _ in range(16):
            _, p = segment_softmax(pol.logits(phi), np.array([0]), np.zeros(3, dtype=int))
            a = int(rng.choice(3, p=p))
            ep = T.Episode()
            ep.phis, ep.states, ep.chosen = [phi], [state], [a]
            ep.logps, ep.rewards = [float(np.log(p[a]))], [1.0 if a == 2 else 0.0]
            eps.append(ep)
        T.ppo_update(pol, opt, eps, cfg)
    assert prob()[2] > before[2] + 0.3 and prob()[2] > 0.6


def test_reward_is_score_delta_plus_terminal_bonus():
    pol = Policy(0, rng=np.random.default_rng(0))
    r = T.Reward(score_scale=1.0, step_cost=0.0, terminal_per_point=0.5, truncation_penalty=7.0)
    ep = T.play(pol, "default", 3, 3, np.random.default_rng(0), r, max_steps=2000)
    assert not ep.truncated and ep.final_score > 0
    assert sum(ep.rewards) == pytest.approx(1.5 * ep.final_score)  # score deltas sum to the score
    cut = T.play(pol, "default", 3, 30, np.random.default_rng(0), r, max_steps=3)
    assert cut.truncated and cut.rewards[-1] < -6


def test_training_seeds_are_disjoint_from_evaluation_seeds(monkeypatch):
    with pytest.raises(ValueError, match="eval seeds"):
        T.train("ppo", 2, ["default"], eval_seeds=[T.TRAIN_SEED_BASE])
    seen = []
    real = T.play
    monkeypatch.setattr(T, "play", lambda *a, **k: (seen.append(a[2]), real(*a, **k))[1])
    T.train("ppo", 4, ["default"], turns=3, eval_turns=3, eval_seeds=[0], eval_every=99,
            cfg=T.PPOConfig(batch_episodes=2))  # fmt: skip
    assert seen == [T.TRAIN_SEED_BASE + i for i in range(4)]


@pytest.mark.parametrize("kind", ["ppo", "q"])
def test_train_save_load_and_play(tmp_path, kind):
    result = T.train(kind, 6, ["default", "bad_opening"], turns=4, eval_turns=5, eval_seeds=[0, 1],
                     eval_every=3, cfg=T.PPOConfig(batch_episodes=3, hidden=8))  # fmt: skip
    assert [row["episodes"] for row in result.curve] == [3, 6]
    assert {"eval_mean", "eval_default", "eval_bad_opening", "train_return"} <= set(result.curve[0])
    path = tmp_path / "m.npz"
    result.policy.save(path, {"kind": kind, "episodes": 6})
    agent = make_agent("rl", checkpoint=path)
    assert isinstance(agent, RLAgent) and AGENTS["rl"] is RLAgent
    m = run_game(CivEnv(FakeCivGame(max_turns=5)), agent, tmp_path / "r.jsonl", backend="fake",
                 max_steps=400)  # fmt: skip
    assert m.failed_steps == 0  # a masked policy only ever picks legal actions
    assert load_run(tmp_path / "r.jsonl").header.config["kind"] == kind
    again = make_agent("rl", checkpoint=path)
    run_game(CivEnv(FakeCivGame(max_turns=5)), again, tmp_path / "r2.jsonl", backend="fake",
             max_steps=400)  # fmt: skip
    first = [s.action for s in load_run(tmp_path / "r.jsonl").steps]
    assert first == [s.action for s in load_run(tmp_path / "r2.jsonl").steps]  # deterministic


def test_checkpoint_from_other_features_is_refused(tmp_path):
    pol = Policy(4)
    pol.save(tmp_path / "x.npz", {})
    with np.load(tmp_path / "x.npz") as data:
        arrays = dict(data)
    meta = json.loads(str(arrays["meta"]))
    meta["feature_dim"] = 5
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "y.npz", **arrays)
    with pytest.raises(ValueError, match="different features"):
        Policy.load(tmp_path / "y.npz")
    with pytest.raises(ValueError, match="needs a checkpoint"):
        make_agent("rl")


def test_cli_train_then_run_and_sweep(tmp_path, capsys):
    out = tmp_path / "models" / "ppo.npz"
    train_args = [
        "train", "--agent", "ppo", "--episodes", "4", "--turns", "3", "--eval-turns", "3",
        "--eval-seeds", "0-1", "--eval-every", "2", "--batch-episodes", "2", "--hidden", "4",
        "--out", str(out),
    ]  # fmt: skip
    code = main(train_args)
    assert code == 0 and out.exists()
    assert json.loads(capsys.readouterr().out)["final"]["episodes"] == 4
    curve = out.with_suffix(".curve.csv").read_text().splitlines()
    assert curve[0].startswith("episodes,") and len(curve) == 3
    code = main(["run", "--agent", "rl", "--checkpoint", str(out), "--turns", "4",
                 "--max-steps", "300", "--out", str(tmp_path / "r.jsonl")])  # fmt: skip
    assert code == 0 and json.loads(capsys.readouterr().out)["failed_steps"] == 0
    code = main(["sweep", "--agents", "rl", "greedy", "--scenarios", "default", "--seeds", "0",
                 "--turns", "4", "--max-steps", "300", "--checkpoint", str(out),
                 "--out", str(tmp_path / "sw")])  # fmt: skip
    assert code == 0 and "| rl |" in capsys.readouterr().out
