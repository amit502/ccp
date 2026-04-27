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
import os
import re
from typing import Any, Dict, List

from .causal_scorer import ValueRegistry, _heuristic_phi
from .models import AgentContext, CCPStats, CompressionTier, ContextElement


DEFAULT_TOKEN_THRESHOLD = 500
RECENT_WINDOW  = int(os.environ.get("CCP_RECENT_WINDOW", "2"))
MAX_ANCESTORS  = int(os.environ.get("CCP_MAX_ANCESTORS", "4"))

# Tools whose EVERY instance is kept forever — each call returns a unique,
# independent answer that will be needed in the final response.
# MultiQA search results all appear in the final combined answer.
_MULTI_RESULT_TOOLS: frozenset = frozenset({
    "search", "lookup_fact", "web_search",
})

# Tools that read data the agent computes with.  Each call is independent
# (different cells / ranges / slides) so we keep all instances — but only
# while they are still "fresh" (their values still referenced by recent steps).
# Once the agent has moved past the computation, these expire.
_DATA_READ_TOOLS: frozenset = frozenset({
    "read_cell", "read_range", "read_content",
    "read_slide", "read_email", "read_file",
    "list_inbox", "list_events", "list_directory",
})

# Combined: the old _NO_DEDUP_TOOLS — used only for the N-1 compaction guard
# (data-provider tools must not be compacted at recent position N-1).
_NO_DEDUP_TOOLS: frozenset = _MULTI_RESULT_TOOLS | _DATA_READ_TOOLS


# ---------------------------------------------------------------------------
# ValueInternTable — lossless JWT interning
# ---------------------------------------------------------------------------

class ValueInternTable:
    """
    Lossless token compression via symbolic interning.

    JWTs (eyJ...) are ~40-60 tokens each but semantically atomic — the agent
    never reads them, only copies them as access_token values. Replacing each
    unique JWT with a short symbol ($T1, $T2, ...) in stored outputs cuts
    those tokens to 1-2 while preserving all information. The interceptor
    resolves symbols back to real values before any tool call.
    """
    _JWT_RE = re.compile(
        r'eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}'
    )

    def __init__(self) -> None:
        self._sym_to_real: Dict[str, str] = {}
        self._real_to_sym: Dict[str, str] = {}
        self._count: int = 0

    def intern(self, text: str) -> str:
        """Replace each unique JWT in text with a short symbol."""
        def _replace(m: re.Match) -> str:
            val = m.group(0)
            if val not in self._real_to_sym:
                self._count += 1
                sym = f"$T{self._count}"
                self._real_to_sym[val] = sym
                self._sym_to_real[sym] = val
            return self._real_to_sym[val]
        return self._JWT_RE.sub(_replace, text)

    def resolve(self, text: str) -> str:
        """Expand symbols back to their real JWT values."""
        for sym, real in self._sym_to_real.items():
            if sym in text:
                text = text.replace(sym, real)
        return text

    def resolve_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve symbols in all string values of a dict."""
        if not self._sym_to_real:
            return d
        return {k: (self.resolve(v) if isinstance(v, str) else v) for k, v in d.items()}

    def __len__(self) -> int:
        return len(self._sym_to_real)


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
        self._interns    = ValueInternTable()

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
        # Intern JWTs in registered input so they don't create transitive BFS
        # parent links back to credential steps (show_account_passwords etc.).
        # Without this, auto-injected JWTs keep login ancestors alive forever.
        interned_input = self._interns.intern(str(tool_input))
        self._registry.register_input(self._step, interned_input)
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

    @property
    def intern_table(self) -> ValueInternTable:
        return self._interns

    def reset(self, goal: str = "") -> None:
        self._context  = AgentContext(goal=goal)
        self._step     = 0
        self._registry = ValueRegistry()
        self._interns  = ValueInternTable()

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

        # ── Temporal Anchor Freshness (TAF) ─────────────────────────────────
        # Anchors are only kept while their output values are still referenced
        # by the most-recent step's inputs.  Once the agent has moved past the
        # computation that needed those values, the anchor expires and falls
        # through to ValueRegistry ancestry (kept if still depended upon) or
        # dead-branch elimination (dropped).
        #
        # Three anchor categories:
        #  _MULTI_RESULT_TOOLS  — kept FOREVER (all search answers needed at end)
        #  _DATA_READ_TOOLS     — freshness-gated: drop once values leave recent inputs
        #  all other high-phi   — freshness-gated + suffix-deduped (one slot per tool type)
        #
        # Safe default: if a step has NO extractable values (e.g. open_workbook
        # whose workbook_id is hex-filtered), it's always treated as fresh so
        # we never accidentally drop a critical handle anchor.
        # ─────────────────────────────────────────────────────────────────────

        # Text of the single most-recent step's inputs — used for freshness check.
        # Intern JWTs here so credential anchors (show_account_passwords etc.) correctly
        # expire: after the JWT chain fix, registered inputs store $T1 not real JWTs, so
        # output_values() returns the real JWT while this text has $T1 — no match → expired.
        # Non-JWT values (cell data, IDs) are unaffected by interning.
        recent_input_text: str = ""
        if elements:
            recent_input_text = self._interns.intern(" ".join(
                str(e.tool_input) for e in elements if e.step in recent_steps
            ))

        def _is_fresh(step: int) -> bool:
            values = self._registry.output_values(step)
            if not values:
                return True   # no extracted values → assume fresh (safe default)
            return any(v in recent_input_text for v in values)

        _anchor_by_tool: Dict[str, int] = {}
        anchor_keep_all: set = set()
        for e in elements:
            if (_heuristic_phi(e) or 0) >= 0.9:
                tool = e.tool_name

                is_multi = any(
                    tool == nd or tool.endswith(f"__{nd}") or tool.endswith(nd)
                    for nd in _MULTI_RESULT_TOOLS
                )
                is_data_read = any(
                    tool == nd or tool.endswith(f"__{nd}") or tool.endswith(nd)
                    for nd in _DATA_READ_TOOLS
                )

                if is_multi:
                    # Search/lookup: keep ALL instances regardless of freshness
                    anchor_keep_all.add(e.step)
                elif is_data_read:
                    # Data-read: keep this instance only while values are fresh
                    if _is_fresh(e.step):
                        anchor_keep_all.add(e.step)
                    # else: expired — fall through to BFS / dead-branch
                else:
                    # Credential / handle anchor: dedup by suffix, freshness-gated
                    if _is_fresh(e.step):
                        tool_key = tool.split("__")[-1] if "__" in tool else tool
                        _anchor_by_tool[tool_key] = e.step
                    # else: expired — fall through to BFS / dead-branch

        anchor_steps = set(_anchor_by_tool.values()) | anchor_keep_all
        live_set     = self._live_ancestors(recent_steps | anchor_steps)

        # Cap regular ancestors: keep only the most recent MAX_ANCESTORS non-anchor,
        # non-recent live ancestors.
        regular_ancestors = sorted(live_set - recent_steps - anchor_steps)
        if len(regular_ancestors) > self.max_ancestors:
            live_set -= set(regular_ancestors[:len(regular_ancestors) - self.max_ancestors])

        # ── Adaptive State Distillation (ASD) ──────────────────────────────────
        # Regular live ancestors (non-anchor, non-recent) are merged into a single
        # synthetic "_state_" dict rather than kept as individual ContextElements.
        # This replaces N×~50t individual records with one ~20t merged JSON dict,
        # cutting mean tokens and beating fifo on efficiency.
        #
        # Carry forward content from any prior synthetic state snapshot so that
        # information accumulated in previous compression cycles is not lost.
        # ─────────────────────────────────────────────────────────────────────
        ancestor_kv: Dict[str, Any] = {}
        for prior_e in self._context.elements:
            if prior_e.step == 0 and prior_e.tool_name == "_state_":
                try:
                    prior_data = json.loads(prior_e.compressed_output or "{}")
                    if isinstance(prior_data, dict):
                        ancestor_kv.update(prior_data)
                except (json.JSONDecodeError, TypeError):
                    pass

        kept: List[ContextElement] = []
        n_active = n_inert = 0

        for e in elements:
            e.phi = self._registry.phi(e.step)

            if e.step in recent_steps:
                e.tier = CompressionTier.ACTIVE
                _is_anchor = (_heuristic_phi(e) or 0) >= 0.9
                _is_last   = (e.step == max(recent_steps))
                _is_data_provider = any(
                    e.tool_name == nd or e.tool_name.endswith(f"__{nd}") or e.tool_name.endswith(nd)
                    for nd in _NO_DEDUP_TOOLS
                )
                if _is_anchor and not _is_last and not _is_data_provider:
                    # Credential/handle anchor at N-1 (e.g. show_profile, open_workbook):
                    # compact to key values — the agent already acted on it.
                    # Data-provider tools (_NO_DEDUP_TOOLS: read_range, read_cell, etc.)
                    # are excluded: their data grids are needed for active computation.
                    if e.compressed_output is None:
                        values = self._registry.output_values(e.step)
                        e.compressed_output = _compact(e, values)
                    e.compressed_output = self._interns.intern(e.compressed_output)
                else:
                    # Last step, non-anchor, or data-provider: keep verbatim, just intern JWTs
                    src = e.compressed_output if e.compressed_output is not None else e.tool_output
                    e.compressed_output = self._interns.intern(src)
                kept.append(e)
                n_active += 1
            elif e.step in anchor_steps:
                # Anchor: keep as individual element (search results, data reads, handles)
                e.tier = CompressionTier.ACTIVE
                if e.compressed_output is None:
                    values = self._registry.output_values(e.step)
                    e.compressed_output = _compact(e, values)
                e.compressed_output = self._interns.intern(e.compressed_output)
                kept.append(e)
                n_active += 1
            elif e.step in live_set:
                # Regular live ancestor: merge into synthetic state snapshot instead of
                # keeping as a separate element — eliminates per-record overhead.
                e.tier = CompressionTier.ACTIVE
                if e.compressed_output is None:
                    values = self._registry.output_values(e.step)
                    e.compressed_output = _compact(e, values)
                try:
                    data = json.loads(e.compressed_output)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            ancestor_kv[k] = v   # newer step overrides older
                except (json.JSONDecodeError, TypeError):
                    ancestor_kv[f"_s{e.step}"] = e.compressed_output[:80]
                n_active += 1
            else:
                # Dead branch or unreferenced: drop completely
                e.tier = CompressionTier.INERT
                n_inert += 1

        # Build synthetic state snapshot from all merged live ancestor key-values
        if ancestor_kv:
            snap_str = self._interns.intern(json.dumps(ancestor_kv))
            snap = ContextElement(
                step=0,
                tool_name="_state_",
                tool_input={},
                tool_output=snap_str,
                compressed_output=snap_str,
                status="ok",
            )
            snap.tier = CompressionTier.ACTIVE
            kept.insert(0, snap)
            n_active += 1

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
