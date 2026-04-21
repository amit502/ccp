"""
context_manager.py

Causal Context Pruning (CCP) — Selective Value Preservation.

Core idea: an agent only needs two things from its history —
  1. The VALUES it will reuse (auth tokens, entity IDs, task outputs)
  2. What just happened (immediate context)

Everything else is noise. This manager keeps exactly that:

  ACTIVE    steps: output contained values reused in later inputs → kept,
                   but compacted to just the reused key-value pairs.
  RECENT    steps: last RECENT_WINDOW steps → kept verbatim.
  All other steps: dropped completely (no digests, no summaries).

No LLM calls. Fully deterministic. Compression fires when total tokens
exceed token_threshold.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .causal_scorer import ValueRegistry
from .models import AgentContext, CCPStats, CompressionTier, ContextElement


DEFAULT_TOKEN_THRESHOLD = 500
RECENT_WINDOW = 2   # last N steps always kept verbatim


# ---------------------------------------------------------------------------
# Compact extractor — keeps only the values the agent will actually reuse
# ---------------------------------------------------------------------------

def _compact(element: ContextElement, referenced_values: set) -> str:
    """
    From a tool output, extract only the fields whose values were
    referenced in a later tool input.  Always retains fields named
    id / status / *_id / *_token / access_token regardless of reference,
    because those are universally needed for follow-up API calls.
    Falls back to a 200-char truncation for non-JSON outputs.
    """
    output = element.tool_output
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            kept: Dict[str, Any] = {}
            for k, v in data.items():
                if v is None:
                    continue
                if (str(v) in referenced_values
                        or k in ("id", "status", "access_token", "token")
                        or k.endswith("_id")
                        or k.endswith("_token")):
                    kept[k] = v
            if kept:
                return json.dumps(kept)
        elif isinstance(data, list) and data:
            # Keep items that contain at least one referenced value
            hits = [it for it in data if any(str(v) in str(it) for v in referenced_values)]
            shown = (hits or data)[:3]
            suffix = f" …({len(data)} total)" if len(data) > 3 else ""
            return json.dumps(shown) + suffix
    except (json.JSONDecodeError, TypeError):
        pass
    return output[:200] + ("…" if len(output) > 200 else "")


# ---------------------------------------------------------------------------
# CCPContextManager
# ---------------------------------------------------------------------------

class CCPContextManager:
    """
    Selective Value Preservation — deterministic, zero LLM calls.

    After each compression event the context contains only:
      • Causally active steps  (φ=0.92): output compacted to reused values
      • Recent steps (last 2): verbatim — agent needs immediate context
      • Everything else: dropped

    Because the compacted outputs are small (just the IDs and tokens the
    agent will actually use), peak tokens stay well below methods that keep
    full history or LLM-generated summaries.
    """

    def __init__(
        self,
        token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
        recent_window:   int = RECENT_WINDOW,
    ):
        self.token_threshold = token_threshold
        self.recent_window   = recent_window

        self._context:   AgentContext   = AgentContext(goal="")
        self._step:      int            = 0
        self._stats_log: List[CCPStats] = []
        self._registry   = ValueRegistry()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

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
        self._registry.register_input(self._step, str(tool_input))
        self._registry.register_output(self._step, tool_output)

        element = ContextElement(
            step=self._step,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            status=status,
        )
        self._context.add(element)

        if self._context.total_tokens() > self.token_threshold:
            self._run_compression()

        return element

    def get_compressed_context(self) -> AgentContext:
        return self._context

    def get_stats_log(self) -> List[CCPStats]:
        return self._stats_log

    def reset(self, goal: str = "") -> None:
        self._context  = AgentContext(goal=goal)
        self._step     = 0
        self._registry = ValueRegistry()

    # ------------------------------------------------------------------ #
    # Compression pipeline                                                 #
    # ------------------------------------------------------------------ #

    def _run_compression(self) -> None:
        elements = [e for e in self._context.elements if e.step > 0]
        if not elements:
            return

        n_before      = len(elements)
        tokens_before = self._context.total_tokens()

        recent_steps = {e.step for e in elements[-self.recent_window:]}

        kept: List[ContextElement] = []
        n_active = n_inert = 0

        for e in elements:
            phi = self._registry.phi(e.step)
            e.phi = phi

            if e.step in recent_steps:
                # Recent: keep verbatim, mark active
                e.tier = CompressionTier.ACTIVE
                kept.append(e)
                n_active += 1
            elif phi >= 0.7:
                # Causally referenced: compact to just the reused values
                e.tier = CompressionTier.ACTIVE
                if e.compressed_output is None:
                    values = self._registry.output_values(e.step)
                    e.compressed_output = _compact(e, values)
                kept.append(e)
                n_active += 1
            else:
                # Not referenced, not recent: drop completely
                e.tier = CompressionTier.INERT
                n_inert += 1

        self._context.elements = kept

        tokens_after = self._context.total_tokens()
        delta = (1 - tokens_after / max(tokens_before, 1)) * 100

        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=n_before,
            active_count=n_active,
            relevant_count=0,
            inert_count=n_inert,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=0,
        ))

        print(
            f"[CCP] Step {self._step}: kept {n_active}/{n_before} "
            f"(+{n_active - self.recent_window} referenced, "
            f"dropped {n_inert}) | "
            f"tokens {tokens_before}→{tokens_after} (-{delta:.1f}%)"
        )
