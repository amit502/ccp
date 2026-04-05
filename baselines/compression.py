"""
baselines/compression.py

Baseline context compression methods for comparison with CCP.
All baselines implement the same interface as CCPContextManager so they
can be dropped into the LangGraph agent unchanged.

Baselines:
  1. NoCompression  — full context, no pruning (upper-bound accuracy)
  2. FIFO           — discard oldest elements when threshold exceeded
  3. TokenPerplexity — lightweight token-level compression (LLMLingua-style)
  4. RetrievalBased  — keep top-k most similar to current goal query
"""

from __future__ import annotations

import math
from typing import Any, List

from ..llm_client import call_llm
from ..models import AgentContext, CCPStats, ContextElement


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class BaseContextManager:
    """Common interface: matches CCPContextManager's public API."""

    def __init__(self, token_threshold: int = 4000):
        self.token_threshold = token_threshold
        self._context: AgentContext = AgentContext(goal="")
        self._step: int = 0
        self._stats_log: List[CCPStats] = []

    def set_goal(self, goal: str) -> None:
        self._context.goal = goal

    def add_observation(
        self,
        tool_name:   str,
        tool_input:  dict,
        tool_output: str,
        status:      str = "ok",
    ) -> ContextElement:
        self._step += 1
        element = ContextElement(
            step=self._step,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            status=status,
        )
        self._context.add(element)
        if self._context.total_tokens() > self.token_threshold:
            self._compress()
        return element

    def get_compressed_context(self) -> AgentContext:
        return self._context

    def get_stats_log(self) -> List[CCPStats]:
        return self._stats_log

    def reset(self, goal: str = "") -> None:
        self._context = AgentContext(goal=goal)
        self._step = 0

    def _compress(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. No Compression
# ---------------------------------------------------------------------------

class NoCompression(BaseContextManager):
    """Full context — no pruning. Upper-bound on accuracy, lower-bound on cost."""

    def _compress(self) -> None:
        pass  # Never compress


# ---------------------------------------------------------------------------
# 2. FIFO — discard oldest elements
# ---------------------------------------------------------------------------

class FIFOManager(BaseContextManager):
    """
    Sliding window: discard the oldest elements when threshold exceeded.
    Simple but causes catastrophic information loss on tasks with long-range
    dependencies (e.g., an API key retrieved in step 2 needed in step 20).
    """

    def __init__(self, token_threshold: int = 4000, keep_ratio: float = 0.5):
        super().__init__(token_threshold)
        self.keep_ratio = keep_ratio  # Fraction of threshold to keep after compression

    def _compress(self) -> None:
        tokens_before = self._context.total_tokens()
        target = int(self.token_threshold * self.keep_ratio)

        # Drop oldest elements until we're under target
        while self._context.total_tokens() > target and len(self._context.elements) > 1:
            dropped = self._context.elements.pop(0)

        tokens_after = self._context.total_tokens()
        n = len(self._context.elements)

        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=n,
            active_count=n,
            relevant_count=0,
            inert_count=0,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=0,
        ))


# ---------------------------------------------------------------------------
# 3. Token-level perplexity compression (LLMLingua-style)
# ---------------------------------------------------------------------------

class TokenPerplexityManager(BaseContextManager):
    """
    Approximation of LLMLingua's approach: score tokens by a proxy for
    perplexity and drop the lowest-scoring ones.

    True LLMLingua uses a language model to compute token-level perplexity.
    Here we approximate with a rule: tokens that appear frequently across
    elements (high frequency → low information → lower priority) are dropped.

    This is element-level rather than token-level for tractability in the
    agentic loop, consistent with how LLMLingua is adapted for agent use.
    """

    def __init__(self, token_threshold: int = 4000, compression_ratio: float = 0.5):
        super().__init__(token_threshold)
        self.compression_ratio = compression_ratio  # Target: keep this fraction of tokens

    def _perplexity_score(self, element: ContextElement) -> float:
        """
        Proxy perplexity score: elements with longer, more unique outputs
        score higher (less compressible). Errors and short outputs score lower.
        """
        length_score = min(1.0, len(element.tool_output) / 500)
        error_penalty = 0.3 if element.status == "error" else 1.0
        return length_score * error_penalty

    def _compress(self) -> None:
        tokens_before = self._context.total_tokens()
        target_tokens = int(tokens_before * self.compression_ratio)

        # Score all elements
        scored = [(e, self._perplexity_score(e)) for e in self._context.elements]
        # Sort by score descending (keep high-score elements)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep elements greedily until we'd exceed the target
        kept: List[ContextElement] = []
        token_sum = 0
        for e, _ in scored:
            if token_sum + e.token_count() <= target_tokens:
                kept.append(e)
                token_sum += e.token_count()

        # Restore original order
        kept_steps = {e.step for e in kept}
        self._context.elements = [
            e for e in self._context.elements if e.step in kept_steps
        ]

        tokens_after = self._context.total_tokens()
        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=len(self._context.elements),
            active_count=len(kept),
            relevant_count=0,
            inert_count=0,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=0,
        ))


# ---------------------------------------------------------------------------
# 4. Retrieval-Based — keep top-k most similar to current query
# ---------------------------------------------------------------------------

class RetrievalBasedManager(BaseContextManager):
    """
    Embed past observations and keep those most similar to the current goal.
    Retrieves by topical similarity, NOT causal necessity — a core limitation
    identified in the proposal (a topically similar observation from step 3 may
    be causally irrelevant at step 30).

    Embedding is approximated with TF-IDF-style word overlap (no external
    embedding model required, keeping this training-free and cheap).
    """

    def __init__(self, token_threshold: int = 4000, top_k: int = 10):
        super().__init__(token_threshold)
        self.top_k = top_k

    def _word_overlap_score(self, query: str, element: ContextElement) -> float:
        """Compute word-overlap similarity between query and element text."""
        q_words = set(query.lower().split())
        e_words = set((element.tool_name + " " + element.tool_output).lower().split())
        if not q_words or not e_words:
            return 0.0
        return len(q_words & e_words) / (len(q_words | e_words) + 1e-9)

    def _compress(self) -> None:
        tokens_before = self._context.total_tokens()
        query = self._context.goal

        # Score all elements by similarity to goal
        scored = [
            (e, self._word_overlap_score(query, e))
            for e in self._context.elements
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top-k most similar + always keep the most recent element
        top_steps = {e.step for e, _ in scored[: self.top_k]}
        if self._context.elements:
            top_steps.add(self._context.elements[-1].step)  # Always keep latest

        self._context.elements = [
            e for e in self._context.elements if e.step in top_steps
        ]

        tokens_after = self._context.total_tokens()
        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=len(self._context.elements),
            active_count=len(self._context.elements),
            relevant_count=0,
            inert_count=0,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=0,
        ))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "no_compression":    NoCompression,
    "fifo":              FIFOManager,
    "token_perplexity":  TokenPerplexityManager,
    "retrieval":         RetrievalBasedManager,
}

def get_baseline(name: str, **kwargs) -> BaseContextManager:
    cls = BASELINE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown baseline '{name}'. Choose from: {list(BASELINE_REGISTRY)}")
    return cls(**kwargs)
