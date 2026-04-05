"""
types.py
Shared data structures used across the CCP codebase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Compression tier (maps to the three-tier policy in the proposal)
# ---------------------------------------------------------------------------

class CompressionTier(str, Enum):
    ACTIVE   = "active"    # ϕ ≥ τ_H  →  preserve at full resolution
    RELEVANT = "relevant"  # τ_L ≤ ϕ < τ_H  →  compress to summary
    INERT    = "inert"     # ϕ < τ_L  →  discard / one-line digest


# ---------------------------------------------------------------------------
# A single tool-call / observation pair — the atomic context element
# ---------------------------------------------------------------------------

@dataclass
class ContextElement:
    """
    Represents one (action, observation) pair as defined in the proposal:
        (a_i, o_i) = (tool_name, tool_input | tool_output, status)

    MCP-native: the structured fields enable more efficient causal scoring
    than free-text context (see §MCP Integration in the proposal).
    """
    step: int                          # Position in the trajectory
    tool_name: str                     # MCP tool identifier
    tool_input: Dict[str, Any]         # Parameters passed to the tool
    tool_output: str                   # Raw tool response / observation
    status: str = "ok"                 # "ok" | "error" | "timeout"
    timestamp: float = field(default_factory=time.time)

    # Fields populated by CCP after scoring
    phi: Optional[float] = None        # Causal necessity score ϕ ∈ [0, 1]
    tier: Optional[CompressionTier] = None
    compressed_output: Optional[str] = None  # Set when tier == RELEVANT/INERT

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def action_str(self) -> str:
        """Human-readable action summary for prompts."""
        return f"{self.tool_name}({self.tool_input})"

    def observation_str(self) -> str:
        """Returns compressed output if available, otherwise raw output."""
        if self.compressed_output is not None:
            return self.compressed_output
        return self.tool_output

    def token_count(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        text = self.action_str() + self.observation_str()
        return max(1, len(text) // 4)

    def to_context_block(self) -> str:
        """Format for injection into the agent's context window."""
        return (
            f"[Step {self.step}] Tool: {self.tool_name}\n"
            f"  Input:  {self.tool_input}\n"
            f"  Output: {self.observation_str()}\n"
            f"  Status: {self.status}"
        )


# ---------------------------------------------------------------------------
# The agent's full context at step t
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """
    Represents C_t = {(a_i, o_i)}_{i=1}^{t}  plus the current goal g.
    This is the object CCP operates on.
    """
    goal: str
    elements: List[ContextElement] = field(default_factory=list)

    def add(self, element: ContextElement) -> None:
        self.elements.append(element)

    def total_tokens(self) -> int:
        return sum(e.token_count() for e in self.elements)

    def to_prompt_str(self) -> str:
        """Render full context as a string for the agent LLM."""
        blocks = [f"Goal: {self.goal}\n"]
        for e in self.elements:
            blocks.append(e.to_context_block())
        return "\n\n".join(blocks)

    def __len__(self) -> int:
        return len(self.elements)


# ---------------------------------------------------------------------------
# Statistics emitted by CCP for metrics collection
# ---------------------------------------------------------------------------

@dataclass
class CCPStats:
    step: int
    total_elements: int
    active_count: int
    relevant_count: int
    inert_count: int
    tokens_before: int
    tokens_after: int
    scorer_calls: int          # Number of LLM calls made by the scorer

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def token_reduction_pct(self) -> float:
        return (1 - self.compression_ratio) * 100
