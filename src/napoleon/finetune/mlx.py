"""mlx-lm glue: an in-process LLM client, and the LoRA trainer run as a subprocess.

mlx-lm is imported only when a client is built or a trainer is run, so the rest of the package
and every test work without it (``pip install -e '.[mlx]'`` adds it, Apple silicon only).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from napoleon.llm import LLMResponse, estimate_tokens
from napoleon.usage import Usage

DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-4bit"
ADAPTER_FILE = "adapters.safetensors"


class MLXUnavailable(RuntimeError):
    pass


def require_mlx() -> None:
    if importlib.util.find_spec("mlx_lm") is None:
        raise MLXUnavailable(
            "mlx-lm is not installed. Install it on Apple silicon with: pip install -e '.[mlx]'"
        )


class MLXClient:
    """``LLMClient`` over a model (and optional LoRA adapter) loaded in this process.

    Thinking is switched off in the chat template (Qwen3's ``enable_thinking=False``), which
    matches the bare-JSON targets the student is trained on.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        adapter_path: str | Path | None = None,
        *,
        temperature: float = 0.0,
    ) -> None:
        require_mlx()
        from mlx_lm import load

        self.model = model
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.temperature = temperature
        self._model, self._tokenizer = load(model, adapter_path=self.adapter_path)

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 1024) -> LLMResponse:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:  # a template that takes no such flag
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )
        usage = Usage(
            calls=1,
            input_tokens=len(self._tokenizer.encode(prompt)) or estimate_tokens(prompt),
            output_tokens=len(self._tokenizer.encode(text)) or estimate_tokens(text),
        )
        return LLMResponse(text, usage)


@dataclass(frozen=True)
class LoraConfig:
    """Arguments for ``mlx_lm lora``. The defaults suit a 1.7B 4-bit student on 24 GB."""

    model: str = DEFAULT_MODEL
    iters: int = 200
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_layers: int = 8
    max_seq_length: int = 4096
    steps_per_report: int = 10
    steps_per_eval: int = 100
    val_batches: int = 5
    mask_prompt: bool = True
    grad_checkpoint: bool = True


def lora_command(
    cfg: LoraConfig, data_dir: str | Path, adapter_dir: str | Path, *, resume: bool = False
) -> list[str]:
    """The command that trains (or, with ``resume``, continues) an adapter on ``data_dir``."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", cfg.model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(cfg.iters),
        "--batch-size", str(cfg.batch_size),
        "--learning-rate", str(cfg.learning_rate),
        "--num-layers", str(cfg.num_layers),
        "--max-seq-length", str(cfg.max_seq_length),
        "--steps-per-report", str(cfg.steps_per_report),
        "--steps-per-eval", str(cfg.steps_per_eval),
        "--val-batches", str(cfg.val_batches),
        "--save-every", str(cfg.iters),
    ]  # fmt: skip
    if cfg.mask_prompt:
        cmd.append("--mask-prompt")
    if cfg.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if resume:
        cmd += ["--resume-adapter-file", str(Path(adapter_dir) / ADAPTER_FILE)]
    return cmd


def run_lora(
    cfg: LoraConfig,
    data_dir: str | Path,
    adapter_dir: str | Path,
    *,
    resume: bool = False,
    log_path: str | Path | None = None,
) -> None:
    """Run the trainer, streaming its output to ``log_path`` (or this process's stderr)."""
    require_mlx()
    cmd = lora_command(cfg, data_dir, adapter_dir, resume=resume)
    if log_path is None:
        done = subprocess.run(cmd, check=False)
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a", encoding="utf-8") as fh:
            done = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"mlx_lm lora failed with exit code {done.returncode}: {' '.join(cmd)}")
