"""
context_manager.py

Causal Context Pruning (CCP) — Dead Branch Elimination (DBE).

Core idea: treat the agent trajectory as a directed dependency graph.
Each step S depends on the steps whose output values it used in its input.
Dead branches — steps that are ancestors of abandoned sub-paths, not of
the current live path — are completely dropped.

Algorithm:
  1. Identify RECENT steps (last N) as the "live frontier".
  2. BFS backward from the live frontier, following parent edges
     (step S's parents = steps whose values S's input referenced).
  3. LIVE ANCESTORS: kept, but compacted to just the referenced key-values.
  4. RECENT steps: kept verbatim (agent needs immediate context).
  5. Everything else (dead branches): dropped completely.

No LLM calls. Fully deterministic. Compression fires when total tokens
exceed token_threshold.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .causal_scorer import ValueRegistry, _heuristic_phi
from .models import AgentContext, CCPStats, CompressionTier, ContextElement


DEFAULT_TOKEN_THRESHOLD = 500
RECENT_WINDOW = 4   # last N steps always kept verbatim


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
    Dead Branch Elimination — deterministic, zero LLM calls.

    After each compression event the context contains only:
      • Live ancestors (BFS from recent steps): output compacted to reused values
      • Recent steps (last N): verbatim — agent needs immediate context
      • Dead branches: dropped completely

    "Dead branch" = a step referenced only within an abandoned sub-path,
    not on the dependency chain leading to the current live step.
    This eliminates the token overhead of keeping every-ever-referenced step,
    keeping only the minimal ancestor set needed for the current trajectory.
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

    def _live_ancestors(self, recent_steps: set) -> set:
        """
        BFS backward from recent_steps, following parent edges.
        Returns the set of all ancestor step numbers reachable from
        recent steps via the dependency graph — the live ancestry.
        """
        live: set = set(recent_steps)
        queue = list(recent_steps)
        while queue:
            step = queue.pop()
            for parent in self._registry.get_parents(step):
                if parent not in live:
                    live.add(parent)
                    queue.append(parent)
        return live

    def _run_compression(self) -> None:
        elements = [e for e in self._context.elements if e.step > 0]
        if not elements:
            return

        n_before      = len(elements)
        tokens_before = self._context.total_tokens()

        recent_steps  = {e.step for e in elements[-self.recent_window:]}

        # Always-live seeds: high-phi tool steps (task-spec, auth, credentials)
        # survive the full trajectory regardless of exact string matching.
        anchor_steps  = {e.step for e in elements if (_heuristic_phi(e) or 0) >= 0.9}
        live_set      = self._live_ancestors(recent_steps | anchor_steps)

        kept: List[ContextElement] = []
        n_active = n_inert = 0

        for e in elements:
            e.phi = self._registry.phi(e.step)

            if e.step in recent_steps:
                # Live frontier: keep verbatim
                e.tier = CompressionTier.ACTIVE
                kept.append(e)
                n_active += 1
            elif e.step in live_set:
                # Live ancestor: compact to just the referenced values
                e.tier = CompressionTier.ACTIVE
                if e.compressed_output is None:
                    values = self._registry.output_values(e.step)
                    e.compressed_output = _compact(e, values)
                kept.append(e)
                n_active += 1
            else:
                # Dead branch or unreferenced: drop completely
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
            f"[CCP-DBE] Step {self._step}: kept {n_active}/{n_before} "
            f"({len(recent_steps)} recent + {n_active - len(recent_steps)} ancestors, "
            f"dropped {n_inert} dead) | "
            f"tokens {tokens_before}→{tokens_after} (-{delta:.1f}%)"
        )
