# Fine-tuning a small student with MLX

`napoleon finetune` LoRA-tunes a small local model on fake-game play with
[mlx-lm](https://github.com/ml-explore/mlx-lm), on Apple silicon. It is modeled on the
Continual Harness paper's co-learning loop (roll out, relabel bad windows with a stronger
teacher, soft SFT, repeat), run locally. Install the extra with `pip install -e '.[mlx]'`; nothing
else in the repo, and no test, needs it.

The default student is `mlx-community/Qwen3-1.7B-4bit` (`--model` to change it, for example to
`mlx-community/Qwen3-4B-4bit`). Model downloads come from Hugging Face. If one stalls, set
`HF_HUB_DISABLE_XET=1` and retry (it fixed a stalled download here).

## Data

A training pair is the direct agent's exact prompt (`agents/prompting.direct_messages`, a test
checks it byte for byte) and a target reply, `{"reasoning": ..., "action": ...}` (or the action
alone with `--no-cot`), in mlx-lm's chat JSONL format (`train.jsonl`, `valid.jsonl` with
`{"messages": [system, user, assistant]}`). `--mask-prompt` trains on the reply only.

Teachers:

- `greedy` (default): the rule-based greedy agent's action. This is behavior cloning, free, and always legal. Data is either fresh greedy games on seeds 2000 and up (away from the evaluation seeds and the few-shot pool) or the states of recorded fake-game runs (`--from runs/x.jsonl`, replayed with hash checks).
- `llm` (`--teacher llm --teacher-model qwen3.8:27b-mlx`): a stronger model served by Ollama labels the states. Optional, since an 18 GB model is tight on 24 GB next to the student. Its answer must be one of the legal actions, otherwise greedy answers instead (the count is `LLMTeacher.fallbacks`).

## Process reward

`finetune.data.window_reward` scores a window of steps (default 8) in [0, 1]:
0.4 x the share of actions the game accepted, 0.3 x the share of replies that parsed as an action at all, 0.3 x score gained over the window (full marks from one point). Windows below `--bad-below` (0.75) are "bad".

## Commands

```
napoleon finetune data  --out data/ --seeds 2000-2005            # just the JSONL
napoleon finetune sft   --out adapters/sft --seeds 2000-2005 --turns 30 --iters 300
napoleon finetune dagger --adapter adapters/sft --out adapters/dagger --iterations 3 --steps-per-iter 48
napoleon finetune eval  --adapter adapters/dagger --seeds 0-2 --turns 10     # omit --adapter for the base model
```

`sft` builds the data, writes it to `<out>/data` and runs `mlx_lm lora` (`--dry-run` prints the
command instead). `dagger` per iteration:

1. plays `--steps-per-iter` steps of the fake game with the current student (the direct agent, no retries, so format errors show), continuing the same game where the last iteration stopped. A new game starts only when one ends, on the next training seed. The game is pickled to `<out>/state.pkl`, so a stopped run resumes;
2. scores the play in windows and has the teacher relabel the bad ones. Good windows keep the student's own accepted actions (`keep_good`);
3. adds the examples to everything gathered so far (DAgger aggregation, newest 1,500 kept);
4. soft SFT: continues the same adapter (`--resume-adapter-file`) at a low learning rate (2e-5, 60 iterations).

Each iteration writes `iter-NN.jsonl` (a normal run file, so `napoleon metrics` and `replay` work) and a line in `dagger_log.jsonl` with the window reward, illegal and format-invalid rates, bad windows and example counts.

## Serving the adapter

`mlx_lm.server` is OpenAI-compatible, so the existing client and every LLM agent use the tuned model through `--base-url`:

```
mlx_lm.server --model mlx-community/Qwen3-1.7B-4bit --adapter-path adapters/dagger --port 8080
napoleon run --agent direct --base-url http://localhost:8080/v1 --model mlx-community/Qwen3-1.7B-4bit --turns 20
napoleon run --agent continual --base-url http://localhost:8080/v1 --model mlx-community/Qwen3-1.7B-4bit
```

The tuned model answers with bare JSON, which `direct` parses. `react` and `continual` use other reply formats that this SFT did not teach, so expect `direct` to be the agent that benefits.

## Not built

GRPO (offline, G=4 candidates, reward 0.6 legality and correctness plus 0.4 format) is not built here. The process reward and the data pipeline are what it would reuse.
