"""
context_manager.py

Causal Context Pruning (CCP) — Dead Branch Elimination (DBE) + Causal State Synthesis (CSS).

Algorithm:
  1. Identify RECENT steps (last N) as the "live frontier" — kept verbatim.
  2. BFS backward from the live frontier, following parent edges.
  3. LIVE ANCESTORS: facts extracted into the CSS world-state block; shown
     as "[→ KNOWN STATE]" in message history (4 tokens vs 20-100 compacted).
  4. DEAD BRANCHES: dropped completely; action recorded in CSS "done" list.

CSS (Causal State Synthesis):
  Instead of keeping live-ancestor tool outputs verbatim/compacted, extract
  their key-value facts into a single compact "KNOWN STATE" block injected
  into the system prompt.  This replaces N×30-100 token compacted outputs
  with one ~40-token structured block, giving 4-8× token reduction on
  credential-heavy benchmarks (AppWorld).

No LLM calls. Fully deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .causal_scorer import ValueRegistry, _heuristic_phi
from .models import AgentContext, CCPStats, CompressionTier, ContextElement


DEFAULT_TOKEN_THRESHOLD = 500
RECENT_WINDOW  = int(os.environ.get("CCP_RECENT_WINDOW", "2"))
MAX_ANCESTORS  = int(os.environ.get("CCP_MAX_ANCESTORS", "4"))

# Tools where EVERY call must be kept as an anchor (no deduplication).
# These produce independent results per call — deduping would lose earlier answers.
_NO_DEDUP_TOOLS: frozenset = frozenset({
    "search", "lookup_fact", "web_search",      # MultiQA — each query is a separate answer
    "read_cell", "read_range", "read_content",  # OfficeBench — may read different ranges
    "read_slide", "read_email", "read_file",
    "list_inbox", "list_events", "list_directory",
})


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
            # No ID/token fields — look for a "results"/"items" list containing
            # an answer/snippet field (search-result format from NQ MCP server).
            for list_key in ("results", "items", "data"):
                results_list = data.get(list_key)
                if isinstance(results_list, list) and results_list:
                    first = results_list[0]
                    if isinstance(first, dict):
                        for ans_key in ("answer", "snippet", "result", "text", "value"):
                            if first.get(ans_key):
                                return json.dumps({ans_key: str(first[ans_key])[:150]})
                    break
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
# CSS helpers — Causal State Synthesis
# ---------------------------------------------------------------------------

_CSS_SKIP_KEYS = frozenset({
    "status", "state_changed", "observation", "error", "message",
})
_CSS_SKIP_INPUT_KEYS = frozenset({
    "access_token", "token", "authorization", "workbook_id",
    "doc_id", "pptx_id", "wb_id",
})


def _extract_css_facts(element: ContextElement) -> Tuple[Dict[str, str], str]:
    """
    Extract (facts_dict, action_summary) from a context element for CSS.

    facts_dict: key→value pairs to merge into the CSS "known" dict.
    action_summary: brief one-liner for the CSS "done" list.
    Returns ({}, action) if no useful facts can be extracted.
    """
    tool = element.tool_name
    inp  = element.tool_input

    # Build a compact action summary — omit credential/handle args
    key_args = [
        f"{k}={str(v)[:25]}"
        for k, v in inp.items()
        if k not in _CSS_SKIP_INPUT_KEYS and str(v).strip()
    ]
    action = f"{tool}({', '.join(key_args[:2])})" if key_args else tool

    facts: Dict[str, str] = {}

    try:
        data = json.loads(element.tool_output)
    except (json.JSONDecodeError, TypeError):
        text = element.tool_output.strip()[:80]
        if text:
            facts["_result"] = text
        return facts, action

    # Unwrap common {"status": "ok", "data": <payload>} envelope
    if isinstance(data, dict) and "data" in data:
        inner = data["data"]
        if isinstance(inner, (dict, list)):
            data = inner

    # Credential/account list: [{"account_name": "venmo", "password": "abc123"}, ...]
    if isinstance(data, list):
        for item in data[:20]:
            if not isinstance(item, dict):
                continue
            name = (item.get("account_name") or item.get("account")
                    or item.get("name") or item.get("app") or "")
            pw   = item.get("password", "")
            if name and pw:
                facts[f"{name}_pw"] = str(pw)[:40]
            elif name:
                for k, v in item.items():
                    if v is not None and not isinstance(v, (dict, list)):
                        facts[f"{name}_{k}"] = str(v)[:40]
        return facts, action

    if isinstance(data, dict):
        for k, v in data.items():
            if v is None or k in _CSS_SKIP_KEYS:
                continue
            if isinstance(v, list) and v and isinstance(v[0], dict):
                # Nested credential list inside a dict response
                for item in v[:20]:
                    name = (item.get("account_name") or item.get("account")
                            or item.get("name") or "")
                    pw   = item.get("password", "")
                    if name and pw:
                        facts[f"{name}_pw"] = str(pw)[:40]
                continue
            if isinstance(v, (dict, list)):
                continue
            sv = str(v)
            if len(sv) > 100:
                sv = sv[:100]
            facts[k] = sv

    return facts, action


def _css_summary(css_state: dict) -> str:
    """
    Format the accumulated CSS state as a compact block for system-prompt injection.
    Returns empty string if the state is empty.
    """
    known = css_state.get("known", {})
    done  = css_state.get("done",  [])
    if not known and not done:
        return ""

    creds   = {k: v for k, v in known.items() if k.endswith("_pw")}
    handles = {k: v for k, v in known.items()
               if k.endswith("_id") or k in ("workbook_id", "doc_id", "pptx_id")}
    facts   = {k: v for k, v in known.items()
               if k not in creds and k not in handles}

    parts = []
    if facts:
        parts.append("FACTS: " + "; ".join(
            f"{k}={v}" for k, v in list(facts.items())[:12]
        ))
    if creds:
        parts.append("CREDS: " + "; ".join(
            f"{k}={v}" for k, v in creds.items()
        ))
    if handles:
        parts.append("HANDLES: " + "; ".join(
            f"{k}={v}" for k, v in handles.items()
        ))
    if done:
        parts.append("DONE: " + " | ".join(done[-8:]))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CCPContextManager
# ---------------------------------------------------------------------------

class CCPContextManager:
    """
    Dead Branch Elimination + Causal State Synthesis — deterministic, zero LLM calls.

    After each compression event the context contains:
      • CSS world-state block (step=0): synthesized facts from all compressed ancestors
      • Recent steps (last N): verbatim — agent needs immediate context
      • Dead branches: dropped completely (action recorded in CSS "done")

    The CSS block replaces N compacted ancestor elements with one compact
    structured block injected into the system prompt, achieving 4-8× further
    token reduction beyond standard DBE compaction.
    """

    def __init__(
        self,
        token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
        recent_window:   int = RECENT_WINDOW,
        max_ancestors:   int = MAX_ANCESTORS,
    ):
        self.token_threshold = token_threshold
        self.recent_window   = recent_window
        self.max_ancestors   = max_ancestors

        self._context:   AgentContext   = AgentContext(goal="")
        self._step:      int            = 0
        self._stats_log: List[CCPStats] = []
        self._registry   = ValueRegistry()
        self._css_state: Dict[str, Any] = {"known": {}, "done": []}

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

    def get_css_summary(self) -> str:
        """Return the current CSS world-state string for system-prompt injection."""
        return _css_summary(self._css_state)

    def reset(self, goal: str = "") -> None:
        self._context   = AgentContext(goal=goal)
        self._step      = 0
        self._registry  = ValueRegistry()
        self._css_state = {"known": {}, "done": []}

    # ------------------------------------------------------------------ #
    # Compression pipeline                                                 #
    # ------------------------------------------------------------------ #

    def _live_ancestors(self, recent_steps: set) -> set:
        """BFS backward from recent_steps, following parent edges."""
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
        # step=0 elements are CSS blocks from previous compressions — skip them here
        elements = [e for e in self._context.elements if e.step > 0]
        if not elements:
            return

        n_before      = len(elements)
        tokens_before = self._context.total_tokens()

        recent_steps  = {e.step for e in elements[-self.recent_window:]}

        # Anchor seeds: high-phi tools that survive the full trajectory.
        # _NO_DEDUP_TOOLS keep ALL instances (each call is independent).
        # All others: deduplicate to most-recent call per tool_name.
        _anchor_by_tool: Dict[str, int] = {}
        anchor_keep_all: set = set()
        for e in elements:
            if (_heuristic_phi(e) or 0) >= 0.9:
                tool = e.tool_name
                is_no_dedup = any(
                    tool == nd or tool.endswith(f"__{nd}") or tool.endswith(nd)
                    for nd in _NO_DEDUP_TOOLS
                )
                if is_no_dedup:
                    anchor_keep_all.add(e.step)
                else:
                    _anchor_by_tool[tool] = e.step

        anchor_steps = set(_anchor_by_tool.values()) | anchor_keep_all
        live_set     = self._live_ancestors(recent_steps | anchor_steps)

        # Cap regular (non-anchor, non-recent) ancestors
        regular_ancestors = sorted(live_set - recent_steps - anchor_steps)
        if len(regular_ancestors) > self.max_ancestors:
            live_set -= set(regular_ancestors[:len(regular_ancestors) - self.max_ancestors])

        kept: List[ContextElement] = []
        n_active = n_inert = 0

        for e in elements:
            e.phi = self._registry.phi(e.step)

            if e.step in recent_steps:
                # Live frontier: keep verbatim
                e.tier = CompressionTier.ACTIVE
                kept.append(e)
                n_active += 1

            elif e.step in anchor_keep_all:
                # No-dedup anchor (search results, read results): keep verbatim
                # so the agent can see every individual answer/value directly.
                e.tier = CompressionTier.ACTIVE
                kept.append(e)
                n_active += 1

            elif e.step in live_set:
                # Live ancestor (regular anchor or BFS-reachable step):
                # synthesise into CSS on first compression, replace with
                # compact "[→ KNOWN STATE]" marker in message history.
                e.tier = CompressionTier.ACTIVE
                if e.compressed_output is None:
                    facts, action = _extract_css_facts(e)
                    if facts:
                        self._css_state["known"].update(facts)
                        done_list = self._css_state["done"]
                        if action not in done_list:
                            done_list.append(action)
                        e.compressed_output = "[→ KNOWN STATE]"
                    else:
                        # CSS extraction yielded nothing — fall back to _compact
                        values = self._registry.output_values(e.step)
                        e.compressed_output = _compact(e, values)
                kept.append(e)
                n_active += 1

            else:
                # Dead branch: record action in CSS done-list, drop element
                e.tier = CompressionTier.INERT
                _, action = _extract_css_facts(e)
                done_list = self._css_state["done"]
                if action not in done_list:
                    done_list.append(f"{action}[dropped]")
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
            f"[CCP-CSS] Step {self._step}: kept {n_active}/{n_before} "
            f"({len(recent_steps)} recent + {n_active - len(recent_steps)} ancestors→CSS, "
            f"dropped {n_inert} dead) | "
            f"tokens {tokens_before}→{tokens_after} (-{delta:.1f}%)"
        )
