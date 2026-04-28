#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence


BYTE_ALPHABET = 256
NORMALIZATION_TOL = 1e-6


class TokenLevelPPM:
    """Small correctness-first token-normalized PPM scorer.

    This is intentionally simple: it proves the normalization and score-before-
    update semantics before we optimize or move the scorer into the GPU eval path.
    """

    def __init__(self, *, order: int, vocab_bytes: Sequence[bytes], alpha: float = 1.0) -> None:
        if order < 0:
            raise ValueError("order must be non-negative")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not vocab_bytes:
            raise ValueError("vocab_bytes must not be empty")
        self.order = order
        self.vocab_bytes = list(vocab_bytes)
        self.alpha = float(alpha)
        self._context = deque(maxlen=order)
        self._counts: defaultdict[bytes, Counter[int]] = defaultdict(Counter)

    def _context_key(self, context: deque[int] | list[int]) -> bytes:
        if self.order == 0:
            return b""
        return bytes(context)[-self.order :]

    def _byte_probability(self, byte: int, context: deque[int] | list[int]) -> float:
        counts = self._counts[self._context_key(context)]
        total = sum(counts.values())
        return (counts[byte] + self.alpha) / (total + self.alpha * BYTE_ALPHABET)

    def token_log_probability(self, token_id: int) -> float:
        token = self.vocab_bytes[token_id]
        context = list(self._context)
        logp = 0.0
        for byte in token:
            logp += math.log(self._byte_probability(byte, context))
            context.append(byte)
            if self.order:
                context = context[-self.order :]
        return logp

    def token_distribution(self) -> list[float]:
        logps = [self.token_log_probability(token_id) for token_id in range(len(self.vocab_bytes))]
        max_logp = max(logps)
        weights = [math.exp(logp - max_logp) for logp in logps]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("invalid PPM token distribution")
        return [weight / total for weight in weights]

    def observe_token(self, token_id: int) -> None:
        token = self.vocab_bytes[token_id]
        for byte in token:
            key = self._context_key(self._context)
            self._counts[key][byte] += 1
            self._context.append(byte)


def _validate_distribution(values: Sequence[float], *, name: str) -> None:
    if not values:
        raise ValueError(f"{name} distribution must not be empty")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} distribution contains invalid probabilities")
    if abs(sum(values) - 1.0) > NORMALIZATION_TOL:
        raise ValueError(f"{name} distribution must be normalized")


def mix_token_distributions(nn_probs: Sequence[float], ppm_probs: Sequence[float], *, lam: float) -> list[float]:
    if len(nn_probs) != len(ppm_probs):
        raise ValueError("distributions must have the same length")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    _validate_distribution(nn_probs, name="neural")
    _validate_distribution(ppm_probs, name="PPM")
    mixed = [lam * nn + (1.0 - lam) * ppm for nn, ppm in zip(nn_probs, ppm_probs, strict=True)]
    total = sum(mixed)
    return [value / total for value in mixed]


def observe_tokens(scorer: TokenLevelPPM, token_ids: Iterable[int]) -> None:
    for token_id in token_ids:
        scorer.observe_token(token_id)
