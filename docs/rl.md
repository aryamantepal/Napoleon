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

## Not built here

Fine-tuning the LLM itself with RL (GRPO or LoRA, as in the paper's co-learning loop) is a follow-up, see the fine-tuning notes.
