# RL agents trained on the fake game

The fake game is pure Python and fast (about 2 ms per step), so policies are trained there with
numpy only (`pip install -e '.[rl]'`, which installs numpy; there is no torch dependency and no
test needs more than numpy).

## Policy over a variable action list

`legal_actions` changes every step, so there is no action vocabulary. Each legal action becomes
one feature row (`napoleon/rl/features.py`, 90 numbers): the action's type, command, unit kind,
move features (attack, closer to the nearest enemy, closer to a city, closer to a city site,
unexplored tile, terrain, unit health), the item, tech or civic name, an 8-number global context
(turn fraction, cities, units, gold, enemy near, blockers, score, settlers), and the action type
crossed with that context. A scorer maps each row to one logit and the policy is a softmax over
the rows of that step. Masking is therefore built in: an illegal action has no row, so a trained
policy cannot play one (`failed_steps` is 0 in every evaluation).

The scorer is a one-hidden-layer tanh MLP (`--hidden`, default 32), or linear with `--hidden 0`.

## Algorithms

- `ppo` (`napoleon train --agent ppo`): PPO-clip with GAE (gamma 0.99, lambda 0.95), a small MLP value baseline on a state vector, an entropy bonus, Adam, 16 games per update, 4 epochs, early stop at KL 0.03. Gradients are written by hand over flattened batches (a finite-difference test checks them), with the softmax as a segment operation.
- `q` (`--agent q`): linear Q-learning, epsilon-greedy with epsilon decaying from 0.3 to 0.02, one semi-gradient TD sweep per game. A cheap baseline that shares the features.

## Reward

Per agent step: `0.1 * (score_after - score_before) - 0.01`. On the last step add `0.05 * final_score` if the game finished, or subtract 2 if the step cap (12 steps per turn) ended the episode first. The score term is "score delta per turn" spread over the steps that caused it, because score only moves when a city is founded or a turn ends. The step cost stops the policy from moving units around without ending the turn, and the terminal term weights how the game ends.

## Seeds

Training game `i` uses seed `10000 + i`, never repeated. Evaluation seeds default to 0 to 4, and `train` refuses evaluation seeds at or above 10000, so a training seed is never an evaluation seed.

## Use

```
napoleon train --agent ppo --episodes 1600 --out models/ppo.npz     # also writes models/ppo.curve.csv
napoleon run --agent rl --checkpoint models/ppo.npz --turns 50
napoleon sweep --agents greedy rl --checkpoint models/ppo.npz --seeds 0-4 --turns 50
```

The curve CSV has, per evaluation, the games trained so far, mean training return and score, the share of truncated games, and the greedy-style evaluation score per scenario on the held-out seeds (deterministic argmax policy, 50 turns).

## Results

`models/ppo.npz` (2,500 games of 50 turns, batch 32, about 4.5 minutes on an M-series laptop) and `models/q.npz` (2,000 games) are the checkpoints these numbers come from, with learning curves in `models/ppo.curve.csv` and `models/q.curve.csv`. Mean final score over 50-turn games, `napoleon sweep --turns 50`, training seeds 10000 and up so none of the games below were trained on:

| agent | default | bad_opening | seeds |
|---|---|---|---|
| greedy | 36.6 (sd 4.7) | 21.8 (0.6) | 0-9 |
| random | 28.4 (4.7) | 23.4 (4.3) | 0-9 |
| ppo | 39.8 (5.9) | 22.0 (1.3) | 0-9 |
| greedy | 35.4 (3.5) | 21.9 (0.5) | 10-29 |
| ppo | 41.6 (6.6) | 22.4 (1.0) | 10-29 |
| q (linear) | 21.4 (5.7) | 10.0 (0.0) | 0-9 |

PPO beats greedy on the default scenario (about +6 points on 20 fresh seeds, a gap of roughly 3.7 standard errors) and ties it on `bad_opening`, with the same number of steps per turn. Random scores nearly as well on `bad_opening` because the score there barely depends on play. The linear Q-learner is unstable: its argmax policy loops on moves that look equally good and hits the step cap, so it is a baseline that fails, not a competitor. Seeds 0-9 were also the evaluation seeds in the training curve, but the final checkpoint was taken as is, not chosen by that curve, and seeds 10-29 are untouched.

PPO is evaluated by sampling from the policy, as it was trained. Its argmax policy scores 19.6 (default) and 16.8 (bad_opening) on seeds 0-9 because it repeats moves; `--no-sample` shows this.

## Not built here

Fine-tuning the LLM itself with RL (GRPO or LoRA, as in the paper's co-learning loop) is a follow-up, see the fine-tuning notes.
