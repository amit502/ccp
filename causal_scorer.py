"""
causal_scorer.py

Causal Necessity Scorer for CCP.

Implements the ϕ(a_i, o_i | C_t, g) score defined in the proposal:

    ϕ(a_i, o_i | C_t, g) = P(â_{t+1} ≠ â^{-i}_{t+1} | g)

where â_{t+1} is the agent's next action given full context and
â^{-i}_{t+1} is the next action with element i removed.

Exact computation requires two full forward passes of the agent LLM —
expensive at inference time. CCP uses a lightweight binary-classification
approximation: a small prompted LLM that answers:

    "If this tool response were removed, would the agent's next action change?"

This is Ablation A2 in the proposal; the faithfulness gap between the
approximation and the two-pass ground truth is measured experimentally.

Additionally, CCP exploits MCP structured metadata (tool_name, status)
to apply fast domain-specific heuristics BEFORE calling the LLM scorer,
reducing the number of expensive scorer calls (Ablation A3).
"""

from __future__ import annotations

import json
import re
from typing import Dict, FrozenSet, List, Optional, Set

from .llm_client import call_llm
from .models import AgentContext, CCPStats, CompressionTier, ContextElement


# ---------------------------------------------------------------------------
# ValueRegistry — deterministic causal scorer (CCP-v2, replaces LLM scorer)
# ---------------------------------------------------------------------------

class ValueRegistry:
    """
    Tracks values extracted from tool outputs and marks steps as causally
    active when their output values appear in any later tool input.

    Replaces the expensive LLM φ approximation with exact string matching:
        φ(step) = 0.92  if any output value was used in a later input
        φ(step) = 0.10  otherwise (unreferenced — safe to digest/drop)

    Zero LLM calls. O(n·k) per input registration where k = values per step.
    """

    # Values to extract: numeric IDs, long tokens/keys, quoted identifiers
    _PATTERNS: List[str] = [
        r'\b(\d{3,12})\b',                    # numeric IDs (3-12 digits)
        r'([a-zA-Z0-9_\-]{20,80})',            # tokens / API keys
        r'"([a-zA-Z0-9_.@+\-]{5,60})"',        # quoted short identifiers
    ]

    # Common noise values that should never be treated as causal anchors
    _STOP: FrozenSet[str] = frozenset({
        "true", "false", "null", "none", "ok", "error", "success",
        "failed", "pending", "200", "201", "400", "401", "403", "404", "500",
        # JSON response envelope keys — present in every tool output, not content
        "observation", "status", "state_changed", "data", "files", "value",
        "workbook_id", "doc_id", "pptx_id", "wb_id", "document_id",
    })

    # Resource handle detector: hex string containing at least one a-f letter.
    # Matches UUIDs/doc-ids like "bb5b2771" but NOT pure decimal numbers like "50000".
    _HANDLE_RE: re.Pattern = re.compile(r'^[0-9a-f]{4,16}$', re.IGNORECASE)

    def __init__(self) -> None:
        self._outputs:    Dict[int, FrozenSet[str]] = {}   # step → values
        self._referenced: Set[int] = set()                 # steps marked active
        self._parents:    Dict[int, Set[int]] = {}         # step → steps it depends on

    # -- registration --------------------------------------------------------

    def register_output(self, step: int, text: str) -> None:
        """Extract candidate anchor values from a tool output."""
        values: Set[str] = set()
        for pat in self._PATTERNS:
            values.update(re.findall(pat, text))
        self._outputs[step] = frozenset(
            v for v in values
            if len(v) >= 3
            and v.lower() not in self._STOP
            # Exclude hex handles (UUID fragments like "bb5b2771") but keep
            # pure decimal numbers (like cell values "50000") — require ≥1 a-f letter.
            and not (self._HANDLE_RE.match(v) and any(c in 'abcdefABCDEF' for c in v))
        )

    def register_input(self, current_step: int, input_text: str) -> None:
        """Scan a tool input and mark past steps whose values appear in it."""
        parents: Set[int] = set()
        for past_step, values in self._outputs.items():
            if past_step >= current_step:
                continue
            if any(v in input_text for v in values):
                self._referenced.add(past_step)
                parents.add(past_step)
        self._parents[current_step] = parents

    # -- scoring -------------------------------------------------------------

    def phi(self, step: int) -> float:
        return 0.92 if step in self._referenced else 0.10

    def output_values(self, step: int) -> FrozenSet[str]:
        return self._outputs.get(step, frozenset())

    def get_parents(self, step: int) -> Set[int]:
        """Return the set of steps whose output values were used by this step."""
        return self._parents.get(step, set())

# ---------------------------------------------------------------------------
# MCP-aware heuristics (fast path — no LLM call needed)
# ---------------------------------------------------------------------------

# Tools whose outputs almost always carry long-range causal weight
_HIGH_PHI_TOOLS = {
    "login",
    "authenticate",
    "get_token",
    "get_api_key",
    "get_credentials",
    "create_session",
    "register",
    "get_user_id",
    "get_account_id",
    "set_config",
    "initialize",
}

# Tools whose outputs are typically ephemeral / informational
_LOW_PHI_TOOLS = {
    "list_items",
    "search",
    "browse",
    "get_recommendations",
    "get_trending",
    "ping",
    "health_check",
}

# Short outputs (identifiers, tokens, URLs) tend to be high-ϕ
_SHORT_OUTPUT_THRESHOLD = 120   # characters — likely an ID/token/key
_LONG_OUTPUT_THRESHOLD  = 2000  # characters — likely a verbose list


def _heuristic_phi(element: ContextElement) -> Optional[float]:
    """
    Return a ϕ estimate [0, 1] using MCP structure alone, without an LLM call.
    Returns None if no heuristic applies (→ fall through to LLM scorer).

    MCP structure exploited (Ablation A3):
      - tool_name category membership
      - output length (identifier vs. verbose list)
      - status (error responses rarely causally necessary)
    """
    tool = element.tool_name.lower()
    output = element.tool_output
    status = element.status.lower()

    # Errors are almost never referenced in later steps
    if status == "error":
        return 0.05

    # Identity / credential tools → always high-ϕ
    if any(tool.endswith(h) or tool == h for h in _HIGH_PHI_TOOLS):
        return 0.95

    # Verbose list / search results → usually low-ϕ
    if any(tool.endswith(l) or tool == l for l in _LOW_PHI_TOOLS):
        return 0.15

    # Short output → likely an identifier / token → high-ϕ
    if len(output) <= _SHORT_OUTPUT_THRESHOLD:
        return 0.80

    # Very long output → verbose list → low-ϕ
    if len(output) >= _LONG_OUTPUT_THRESHOLD:
        return 0.20

    return None  # No heuristic matched — use LLM scorer


# ---------------------------------------------------------------------------
# LLM-based binary causal scorer
# ---------------------------------------------------------------------------

_SCORER_SYSTEM = """\
You are a causal necessity evaluator for an AI agent's context window.

Your task is to estimate whether a specific tool-call result (action + observation)
is causally necessary for the agent's NEXT action, given the current task goal.

A context element is causally necessary if removing it would change what the
agent does next — e.g., it contains an ID/token used in the next step, a
constraint that shapes the decision, or a key intermediate result.

A context element is causally inert if the agent's next action would be identical
even if this element were absent — e.g., it is a verbose list that was already
acted upon, a ping/health-check response, or an old error the agent has moved past.

You must respond with a JSON object with exactly these fields:
{
  "phi": <float between 0.0 and 1.0>,
  "tier": <"active" | "relevant" | "inert">,
  "reason": <one sentence explanation>
}

phi = 0.0 means certainly causally inert (safe to discard).
phi = 1.0 means certainly causally necessary (must be preserved).
"""

def _build_scorer_prompt(
    element: ContextElement,
    context: AgentContext,
    recent_window: int = 5,
) -> str:
    """
    Build the user prompt for the LLM scorer.

    To keep the scorer call cheap, we give it:
    - The task goal
    - The element under evaluation
    - The last `recent_window` elements (recent trajectory context)
    Rather than the full history (which could be very long).
    """
    recent = context.elements[-recent_window:] if len(context.elements) > recent_window else context.elements
    recent_str = "\n".join(e.to_context_block() for e in recent if e.step != element.step)

    return f"""\
TASK GOAL:
{context.goal}

RECENT TRAJECTORY (last {recent_window} steps, for context):
{recent_str or "(no prior steps)"}

ELEMENT UNDER EVALUATION:
{element.to_context_block()}

Question: If this element were removed from the agent's context, would the agent's
next action change? Respond with JSON only.
"""


def score_element(
    element: ContextElement,
    context: AgentContext,
    use_heuristics: bool = True,
) -> tuple[float, int]:
    """
    Score a single context element for causal necessity.

    Returns:
        (phi, scorer_calls) where scorer_calls is 0 if heuristic was used,
        1 if the LLM was called.
    """
    # Fast path: MCP-aware heuristics
    if use_heuristics:
        phi_h = _heuristic_phi(element)
        if phi_h is not None:
            element.phi = phi_h
            return phi_h, 0

    # Slow path: LLM binary scorer
    system  = _SCORER_SYSTEM
    user    = _build_scorer_prompt(element, context)

    raw = call_llm(system_prompt=system, user_prompt=user, temperature=0.0)

    # Parse JSON response
    phi = 0.5  # Safe default if parsing fails
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        phi = float(parsed.get("phi", 0.5))
        phi = max(0.0, min(1.0, phi))  # Clamp to [0, 1]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"[CausalScorer] JSON parse error for step {element.step}: {exc}")
        print(f"  Raw response: {raw[:200]}")

    element.phi = phi
    return phi, 1


# ---------------------------------------------------------------------------
# Batch scorer — scores all unscored elements in a context
# ---------------------------------------------------------------------------

def score_context(
    context: AgentContext,
    use_heuristics: bool = True,
) -> tuple[List[ContextElement], int]:
    """
    Score every element in the context that doesn't yet have a ϕ score.

    Returns:
        (elements, total_scorer_calls)
    """
    total_calls = 0
    for element in context.elements:
        if element.phi is None:
            _, calls = score_element(element, context, use_heuristics=use_heuristics)
            total_calls += calls
    return context.elements, total_calls
