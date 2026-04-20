"""
context_manager.py

Causal Context Pruning (CCP) — compression policy.

Implements the three-tier partition from the proposal:

    C_active   = {(a_i, o_i) : ϕ ≥ τ_H}   → preserve at full resolution
    C_relevant = {(a_i, o_i) : τ_L ≤ ϕ < τ_H} → compress to summary
    C_inert    = {(a_i, o_i) : ϕ < τ_L}   → discard / one-line digest

The compressed context becomes:
    C*_t = C_active ∪ compress(C_relevant) ∪ digest(C_inert)

Compression is triggered whenever total token count exceeds threshold T.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .causal_scorer import score_context, ValueRegistry
from .llm_client import call_llm
from .models import AgentContext, CCPStats, CompressionTier, ContextElement


# ---------------------------------------------------------------------------
# Default threshold hyperparameters (Ablation A1 varies these)
# ---------------------------------------------------------------------------

DEFAULT_TAU_HIGH  = 0.6   # τ_H: above this → active (preserve)
DEFAULT_TAU_LOW   = 0.3   # τ_L: below this → inert  (discard/digest)
DEFAULT_TOKEN_THRESHOLD = 500   # T: trigger compression when context > T tokens


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_tiers(
    elements:        List[ContextElement],
    tau_high:        float          = DEFAULT_TAU_HIGH,
    tau_low:         float          = DEFAULT_TAU_LOW,
    retention_ratio: Optional[float] = None,
) -> Tuple[List[ContextElement], List[ContextElement], List[ContextElement]]:
    """
    Partition scored elements into (active, relevant, inert) lists.
    Elements without a ϕ score default to RELEVANT (conservative).

    If retention_ratio is set (e.g. 0.65), elements are promoted from INERT
    to RELEVANT (by φ score descending) until at least that fraction of all
    elements are in ACTIVE or RELEVANT tier.
    """
    active, relevant, inert = [], [], []

    for e in elements:
        phi = e.phi if e.phi is not None else 0.5  # Conservative default

        if phi >= tau_high:
            e.tier = CompressionTier.ACTIVE
            active.append(e)
        elif phi >= tau_low:
            e.tier = CompressionTier.RELEVANT
            relevant.append(e)
        else:
            e.tier = CompressionTier.INERT
            inert.append(e)

    # Enforce minimum retention floor: promote lowest-φ INERT → RELEVANT
    if retention_ratio is not None and elements:
        target = int(len(elements) * retention_ratio)
        shortfall = target - (len(active) + len(relevant))
        if shortfall > 0:
            inert.sort(key=lambda e: e.phi or 0.0, reverse=True)
            for e in inert[:shortfall]:
                e.tier = CompressionTier.RELEVANT
                relevant.append(e)
            inert = inert[shortfall:]

    return active, relevant, inert


# ---------------------------------------------------------------------------
# Compressors for RELEVANT and INERT tiers
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = """\
You are a lossless context compressor for an AI agent.
Your job is to summarise a tool-call result into a concise, information-dense
summary that preserves every fact the agent might need for future actions.
Be specific: keep IDs, names, counts, status codes, and key values.
Drop verbose formatting, repetition, and decorative text.
Respond with the summary only — no preamble.
"""

def _compress_to_summary(element: ContextElement) -> str:
    """Compress a RELEVANT element to a dense summary (1–3 sentences)."""
    user = (
        f"Tool: {element.tool_name}\n"
        f"Input: {element.tool_input}\n"
        f"Output: {element.tool_output}\n\n"
        "Summarise the output, preserving any IDs, values, or constraints "
        "the agent might reference later."
    )
    return call_llm(system_prompt=_SUMMARY_SYSTEM, user_prompt=user)


def _compress_to_digest(element: ContextElement) -> str:
    """Compress an INERT element to a single-line digest."""
    # No LLM call needed — deterministic one-liner
    truncated = element.tool_output[:80].replace("\n", " ")
    if len(element.tool_output) > 80:
        truncated += "…"
    return f"[digest] {element.tool_name} → {truncated}"


# ---------------------------------------------------------------------------
# Main CCP compression trigger
# ---------------------------------------------------------------------------

class CCPContextManager:
    """
    Sits between the MCP tool layer and the agent's context window.

    Usage (in a LangGraph node):
        manager = CCPContextManager()
        manager.add_observation(tool_name, tool_input, tool_output, status, goal)
        compressed_context = manager.get_compressed_context()
    """

    def __init__(
        self,
        tau_high:         float          = DEFAULT_TAU_HIGH,
        tau_low:          float          = DEFAULT_TAU_LOW,
        token_threshold:  int            = DEFAULT_TOKEN_THRESHOLD,
        use_heuristics:   bool           = True,
        compress_relevant: bool          = True,
        retention_ratio:  Optional[float] = None,
    ):
        self.tau_high          = tau_high
        self.tau_low           = tau_low
        self.token_threshold   = token_threshold
        self.use_heuristics    = use_heuristics
        self.compress_relevant = compress_relevant
        self.retention_ratio   = retention_ratio

        self._context: AgentContext = AgentContext(goal="")
        self._step: int = 0
        self._stats_log: List[CCPStats] = []

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
        """
        Called after every MCP tool response.
        Adds the new (action, observation) pair to the context.
        Triggers compression if the token threshold is exceeded.
        """
        self._step += 1
        element = ContextElement(
            step=self._step,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            status=status,
        )
        self._context.add(element)

        # Trigger CCP if context is getting large
        if self._context.total_tokens() > self.token_threshold:
            self._run_compression()

        return element

    def get_compressed_context(self) -> AgentContext:
        """Return the current (possibly compressed) context."""
        return self._context

    def get_stats_log(self) -> List[CCPStats]:
        return self._stats_log

    def reset(self, goal: str = "") -> None:
        """Start a new task trajectory."""
        self._context = AgentContext(goal=goal)
        self._step = 0

    # ------------------------------------------------------------------ #
    # Internal compression pipeline                                        #
    # ------------------------------------------------------------------ #

    def _run_compression(self) -> None:
        """
        Execute the full CCP compression pipeline:
          1. Score all unscored elements (heuristics → LLM scorer)
          2. Assign tiers (active / relevant / inert)
          3. Compress relevant elements to summaries
          4. Replace inert elements with one-line digests
          5. Record stats
        """
        tokens_before = self._context.total_tokens()
        n_before = len(self._context.elements)

        # Step 1: Score
        _, scorer_calls = score_context(
            self._context,
            use_heuristics=self.use_heuristics,
        )

        # Step 2: Assign tiers
        active, relevant, inert = assign_tiers(
            self._context.elements,
            tau_high=self.tau_high,
            tau_low=self.tau_low,
            retention_ratio=self.retention_ratio,
        )

        # Step 3 & 4: Compress
        if self.compress_relevant:
            for e in relevant:
                if e.compressed_output is None:  # Don't re-compress
                    e.compressed_output = _compress_to_summary(e)

        for e in inert:
            if e.compressed_output is None:
                e.compressed_output = _compress_to_digest(e)

        # Step 5: Stats
        tokens_after = self._context.total_tokens()
        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=n_before,
            active_count=len(active),
            relevant_count=len(relevant),
            inert_count=len(inert),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=scorer_calls,
        ))

        delta_pct = (1 - tokens_after / max(tokens_before, 1)) * 100
        if delta_pct > 0.05:
            direction = f"-{delta_pct:.1f}% reduction"
        elif delta_pct < -0.05:
            direction = f"+{abs(delta_pct):.1f}% growth"
        else:
            direction = "no change"
        print(
            f"[CCP] Step {self._step}: {n_before} elements → "
            f"active={len(active)}, relevant={len(relevant)}, inert={len(inert)} | "
            f"tokens {tokens_before}→{tokens_after} ({direction})"
        )


# ===========================================================================
# CCP-v2: Value-Reference Scoring + LLM Summaries + Working Memory
# ===========================================================================


class WorkingMemory:
    """
    Structured state extracted from every tool output.
    Always prepended to the agent context — survives all compression.
    Guarantees access tokens and IDs are never silently dropped.
    """

    def __init__(self) -> None:
        self.access_tokens: Dict[str, str] = {}
        self.known_ids:     Dict[str, Any] = {}
        self._n_steps:      int            = 0

    def update(self, tool_name: str, tool_input: Dict, tool_output: str) -> None:
        self._n_steps += 1
        app = tool_name.split("__")[0] if "__" in tool_name else tool_name
        try:
            data = json.loads(tool_output)
            if isinstance(data, dict):
                for k in ("access_token", "token", "api_key", "auth_token"):
                    if k in data and isinstance(data[k], str) and data[k]:
                        self.access_tokens[app] = data[k]
                for k, v in data.items():
                    if v is None:
                        continue
                    if k == "id":
                        self.known_ids[f"{app}.id"] = v
                    elif k.endswith("_id"):
                        self.known_ids[k] = v
        except (json.JSONDecodeError, TypeError):
            stripped = tool_output.strip().strip('"')
            if 10 <= len(stripped) <= 100 and " " not in stripped:
                if any(kw in tool_name for kw in ("auth", "token", "login", "credential")):
                    self.access_tokens[app] = stripped

    def to_block(self) -> str:
        lines = ["=== WORKING MEMORY ==="]
        if self.access_tokens:
            lines.append(f"Access Tokens : {json.dumps(self.access_tokens)}")
        if self.known_ids:
            lines.append(f"Known IDs     : {json.dumps(self.known_ids)}")
        lines.append(f"Steps taken   : {self._n_steps}")
        return "\n".join(lines)

    def token_count(self) -> int:
        return max(1, len(self.to_block()) // 4)


class CCPv2ContextManager:
    """
    CCP v2 — three targeted improvements over CCP v1:

    1. Value-Reference Scoring  (replaces LLM φ scorer)
       φ = 0.92 when any output value appears in a later tool input — exact,
       deterministic, zero LLM overhead.  φ = 0.10 otherwise.
       retention_ratio floor then promotes unreferenced elements to RELEVANT
       so at least that fraction receives an LLM summary.

    2. LLM Summaries  (same as CCP v1 for RELEVANT tier)
       Unreferenced-but-retained elements get the same dense LLM summary as
       original CCP, preserving narrative state the agent needs.

    3. Working Memory
       A structured {access_tokens, known_ids} dict is always prepended.
       Critical credentials and IDs survive even the most aggressive compression.

    Context layout after compression:
        [WORKING MEMORY]
        [ACTIVE  — directly referenced steps, verbatim]
        [RELEVANT — retention-floor steps, LLM summary]
        [INERT   — remaining steps, one-line digest]
    """

    def __init__(
        self,
        tau_high:          float          = DEFAULT_TAU_HIGH,
        tau_low:           float          = DEFAULT_TAU_LOW,
        token_threshold:   int            = DEFAULT_TOKEN_THRESHOLD,
        compress_relevant: bool           = True,
        retention_ratio:   Optional[float] = None,
    ) -> None:
        self.tau_high          = tau_high
        self.tau_low           = tau_low
        self.token_threshold   = token_threshold
        self.compress_relevant = compress_relevant
        self.retention_ratio   = retention_ratio

        self._context   = AgentContext(goal="")
        self._step      = 0
        self._stats_log: List[CCPStats] = []
        self._registry  = ValueRegistry()
        self._memory    = WorkingMemory()

    # ------------------------------------------------------------------ #
    # Public interface  (same as CCPContextManager)                       #
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
        self._memory.update(tool_name, tool_input, tool_output)

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
        self._memory   = WorkingMemory()

    # ------------------------------------------------------------------ #
    # Internal compression pipeline                                        #
    # ------------------------------------------------------------------ #

    def _make_wm_element(self) -> ContextElement:
        wm_text = self._memory.to_block()
        e = ContextElement(
            step=0, tool_name="__working_memory__",
            tool_input={}, tool_output=wm_text, status="ok",
        )
        e.compressed_output = wm_text
        return e

    def _run_compression(self) -> None:
        elements = [e for e in self._context.elements if e.step > 0]
        n_before = len(elements)
        if not elements:
            return

        tokens_before = self._context.total_tokens()

        # Score using ValueRegistry — deterministic, zero LLM scorer calls
        for e in elements:
            e.phi = self._registry.phi(e.step)

        # Partition into tiers; retention_ratio floor promotes unreferenced
        # elements from INERT → RELEVANT so they receive LLM summaries
        active, relevant, inert = assign_tiers(
            elements,
            tau_high=self.tau_high,
            tau_low=self.tau_low,
            retention_ratio=self.retention_ratio,
        )

        # RELEVANT → LLM summary (same quality as original CCP)
        if self.compress_relevant:
            for e in relevant:
                if e.compressed_output is None:
                    try:
                        e.compressed_output = _compress_to_summary(e)
                    except Exception:
                        e.compressed_output = _compress_to_digest(e)

        # INERT → one-line digest
        for e in inert:
            if e.compressed_output is None:
                e.compressed_output = _compress_to_digest(e)

        # Working memory prepended — always survives
        self._context.elements = [self._make_wm_element()] + active + relevant + inert

        tokens_after = self._context.total_tokens()
        delta        = (1 - tokens_after / max(tokens_before, 1)) * 100
        direction    = f"-{delta:.1f}%" if delta > 0.05 else f"+{abs(delta):.1f}%"

        print(
            f"[CCP-v2] Step {self._step}: {n_before} elements → "
            f"active={len(active)}, relevant={len(relevant)}, inert={len(inert)} | "
            f"tokens {tokens_before}→{tokens_after} ({direction})"
        )

        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=n_before,
            active_count=len(active),
            relevant_count=len(relevant),
            inert_count=len(inert),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=0,
        ))
