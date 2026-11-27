"""Token and cost accounting for LLM-backed agents."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Pricing(BaseModel):
    """Dollars per million tokens. Local models cost nothing, so the default is zero."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class Usage(BaseModel):
    """What one or more LLM calls consumed. Adds with ``+``."""

    model_config = ConfigDict(frozen=True)

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


def total_usage(items: list[Usage | None]) -> Usage:
    total = Usage()
    for u in items:
        if u is not None:
            total = total + u
    return total
