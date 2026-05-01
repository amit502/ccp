"""
baselines/acon.py

Faithful implementation of ACON (Kang et al., 2025).
"ACON: Optimizing Context Compression for Long-horizon LLM Agents"
arXiv:2510.00615 | GitHub: https://github.com/microsoft/acon

==========================================================================
HOW ACON ACTUALLY WORKS (from the paper)
==========================================================================

ACON has two phases:

PHASE 1 — OFFLINE OPTIMIZATION (run once before deployment)
------------------------------------------------------------
1. Run tasks with FULL context → collect trajectories that SUCCEED.
2. Run tasks with compressed context (current guidelines) →
   collect trajectories that FAIL.
3. Form paired trajectories: (success_traj, fail_traj) for the same task.
4. Feed each pair to an LLM-based optimizer with a meta-prompt:
   "Why did compressed context fail? Generate improved guidelines."
5. Apply the new guidelines, re-run compression, collect new failures.
6. Repeat for N_ITER iterations → final optimized guidelines G*.
7. Store G* per benchmark / task category.

PHASE 2 — INFERENCE (applied at deployment)
------------------------------------------------------------
1. Load G* from offline optimization.
2. At each compression trigger:
   a. Split context into HISTORY (past actions/obs) and
      CURRENT OBSERVATION (latest tool response).
   b. Apply the history guideline uniformly to all history elements.
   c. Apply the observation guideline to the current observation.
3. The same guideline applies regardless of current goal state.
   (This is Limitation L3 in the proposal — CCP addresses it.)

Key differences from CCP:
- ACON compresses at CATEGORY level (history vs. observation) — not element
- ACON uses OFFLINE-OPTIMIZED guidelines, not online causal scoring
- ACON cannot adapt online to new task types without re-running Phase 1
- ACON has no causal formalism — empirically optimized

==========================================================================
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import call_llm
from ..models import AgentContext, CCPStats, ContextElement

# ---------------------------------------------------------------------------
# Guideline store — where ACON's optimized guidelines live
# ---------------------------------------------------------------------------

GUIDELINES_PATH = Path(os.environ.get("ACON_GUIDELINES_DIR", "acon_guidelines"))


@dataclass
class ACONGuidelines:
    """
    The output of ACON's offline optimization pipeline.
    Stored as two natural-language strings (history + observation).
    Specific to a benchmark / task category.
    """
    benchmark:             str
    history_guideline:     str
    observation_guideline: str
    n_optimization_iters:  int = 0
    source:                str = "offline"  # "offline" | "default" | "loaded"


# Default guidelines — used when offline optimization hasn't been run.
# These approximate ACON's initial guidelines before any optimization.
_DEFAULT_HISTORY_GUIDELINE = """\
Compress the action history as follows:
- Keep the 3 most recent (action, observation) pairs at full resolution.
- For older pairs: if the observation contains a credential, token, ID, or
  key value that may be referenced in later steps, keep it in full.
  Otherwise summarise to one line: "<tool>(<key_params>) -> <key_result>".
- Drop observations that are verbose lists already acted upon (e.g., search
  results where an item was already selected).
- Preserve all error observations that remain unresolved.
"""

_DEFAULT_OBSERVATION_GUIDELINE = """\
Compress the current observation as follows:
- If the observation is a credential, token, user ID, order ID, or other
  short identifier: keep in full (do not compress).
- If the observation is a list with more than 5 items: keep the first 3
  items and append "[... N more items]".
- If the observation is a verbose description or documentation: summarise
  to 2 sentences preserving key facts (IDs, prices, status codes, names).
- If the observation is an error: keep in full.
"""


def load_guidelines(benchmark: str) -> ACONGuidelines:
    """Load optimized guidelines from disk; fall back to defaults."""
    path = GUIDELINES_PATH / f"{benchmark.lower()}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return ACONGuidelines(
            benchmark=benchmark,
            history_guideline=data["history_guideline"],
            observation_guideline=data["observation_guideline"],
            n_optimization_iters=data.get("n_optimization_iters", 0),
            source="loaded",
        )
    return ACONGuidelines(
        benchmark=benchmark,
        history_guideline=_DEFAULT_HISTORY_GUIDELINE,
        observation_guideline=_DEFAULT_OBSERVATION_GUIDELINE,
        n_optimization_iters=0,
        source="default",
    )


def save_guidelines(guidelines: ACONGuidelines) -> None:
    GUIDELINES_PATH.mkdir(parents=True, exist_ok=True)
    path = GUIDELINES_PATH / f"{guidelines.benchmark.lower()}.json"
    with open(path, "w") as f:
        json.dump({
            "benchmark":             guidelines.benchmark,
            "history_guideline":     guidelines.history_guideline,
            "observation_guideline": guidelines.observation_guideline,
            "n_optimization_iters":  guidelines.n_optimization_iters,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# PHASE 1: Offline Optimization Pipeline
# ---------------------------------------------------------------------------

_OPTIMIZER_SYSTEM = """\
You are an expert at writing compression guidelines for AI agent context windows.

You will be shown two trajectories for the same task:
  - SUCCESS TRAJECTORY: agent completed the task with full context
  - FAILURE TRAJECTORY: agent failed when context was compressed using the
    CURRENT guidelines

Your job is to analyse WHY the compressed trajectory failed and produce
IMPROVED guidelines that would have prevented the failure.

Output improved guidelines as a JSON object with exactly these fields:
{
  "history_guideline": "<improved multi-line guideline for history compression>",
  "observation_guideline": "<improved multi-line guideline for observation compression>",
  "analysis": "<1-2 sentences explaining what went wrong and what you changed>"
}

Guidelines must be natural-language instructions (not code), specific enough
to prevent the observed failure mode, and general enough to apply broadly.
"""


def _format_trajectory(elements: List[ContextElement], label: str) -> str:
    lines = [f"=== {label} ==="]
    for e in elements:
        preview = e.tool_output[:200] + ("..." if len(e.tool_output) > 200 else "")
        lines.append(
            f"Step {e.step}: {e.tool_name}({e.tool_input})\n"
            f"  -> {preview}"
        )
    return "\n".join(lines)


def optimize_guidelines_one_iter(
    success_trajectory: List[ContextElement],
    failure_trajectory: List[ContextElement],
    current_guidelines: ACONGuidelines,
    task_goal: str,
) -> ACONGuidelines:
    """
    One iteration of ACON's offline optimization (Algorithm 1 in paper).
    Takes a (success, failure) pair and returns improved guidelines.
    """
    user_prompt = f"""\
TASK GOAL: {task_goal}

CURRENT HISTORY GUIDELINE:
{current_guidelines.history_guideline}

CURRENT OBSERVATION GUIDELINE:
{current_guidelines.observation_guideline}

{_format_trajectory(success_trajectory, "SUCCESS (full context)")}

{_format_trajectory(failure_trajectory, "FAILURE (compressed context)")}

Analyse why the compressed trajectory failed and produce improved guidelines.
"""
    raw = call_llm(system_prompt=_OPTIMIZER_SYSTEM, user_prompt=user_prompt)
    try:
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
        return ACONGuidelines(
            benchmark=current_guidelines.benchmark,
            history_guideline=result["history_guideline"],
            observation_guideline=result["observation_guideline"],
            n_optimization_iters=current_guidelines.n_optimization_iters + 1,
            source="offline",
        )
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[ACON optimizer] Parse error: {exc} — keeping current guidelines")
        return current_guidelines


class ACONOfflineOptimizer:
    """
    ACON's full offline optimization pipeline (Phase 1).

    Matches the paper's description:
      - Collects paired (success, failure) trajectories
      - Runs N iterations of LLM-based guideline optimization
      - Saves optimized guidelines for inference
    """

    def __init__(
        self,
        benchmark: str = "appworld",
        n_iters:   int = 5,
        n_pairs:   int = 20,
    ):
        self.benchmark = benchmark
        self.n_iters   = n_iters
        self.n_pairs   = n_pairs

    def collect_paired_trajectories(
        self,
        task_runner,
        tasks: List[Any],
        current_guidelines: ACONGuidelines,
    ) -> List[Tuple[List[ContextElement], List[ContextElement], str]]:
        """
        Collect (success_traj, failure_traj, goal) pairs.
        A pair is valid when full context succeeds but compressed context fails.
        Matches ACON Section 3.1: trajectory collection.
        """
        from ..baselines.compression import NoCompression

        pairs = []
        for task in tasks:
            if len(pairs) >= self.n_pairs:
                break

            # Run with full context
            no_comp = NoCompression()
            no_comp.set_goal(task.goal)
            _, success_full = task_runner(task, no_comp)
            if not success_full:
                continue  # Unsolvable — skip

            # Run with ACON compression
            acon_mgr = ACONContextManager(guidelines=current_guidelines)
            acon_mgr.set_goal(task.goal)
            _, success_compressed = task_runner(task, acon_mgr)

            if not success_compressed:
                pairs.append((
                    no_comp.get_compressed_context().elements,
                    acon_mgr.get_compressed_context().elements,
                    task.goal,
                ))
        return pairs

    def run(self, task_runner, tasks: List[Any]) -> ACONGuidelines:
        """Run full offline optimization, save and return optimized guidelines."""
        guidelines = load_guidelines(self.benchmark)
        print(f"[ACON] Offline optimization: {self.n_iters} iters | "
              f"{self.n_pairs} pairs/iter | benchmark={self.benchmark}")

        for iteration in range(self.n_iters):
            print(f"\n[ACON] Iter {iteration+1}/{self.n_iters}")
            pairs = self.collect_paired_trajectories(task_runner, tasks, guidelines)
            if not pairs:
                print("[ACON] No failure pairs found — stopping early.")
                break
            print(f"[ACON] {len(pairs)} failure pairs collected")
            # Optimize on the last 3 pairs (most informative)
            for success_traj, failure_traj, goal in pairs[-3:]:
                guidelines = optimize_guidelines_one_iter(
                    success_traj, failure_traj, guidelines, goal
                )

        save_guidelines(guidelines)
        print(f"[ACON] Done. Guidelines saved to {GUIDELINES_PATH}/{self.benchmark}.json")
        return guidelines


# ---------------------------------------------------------------------------
# PHASE 2: Inference-time context manager
# ---------------------------------------------------------------------------

_HISTORY_COMPRESS_SYSTEM = """\
You are a context compressor for an AI agent. Apply the following compression
guideline to the action history.

GUIDELINE:
{guideline}

Return a JSON array where each entry is:
  {{"step": <int>, "action": "<tool_name>(<params>)", "observation": "<text>"}}

For steps to keep in full, copy the observation exactly.
For steps to summarise, write one concise line preserving key facts.
Respond with the JSON array ONLY.
"""

_OBS_COMPRESS_SYSTEM = """\
You are a context compressor for an AI agent. Apply the following compression
guideline to the current tool observation.

GUIDELINE:
{guideline}

Respond with the compressed observation as plain text ONLY.
"""


class ACONContextManager:
    """
    ACON inference-time context manager (Phase 2).

    Faithfully implements ACON's category-level compression:
      1. Split context into HISTORY and CURRENT OBSERVATION.
      2. Apply offline-optimized (or default) guidelines to each.

    Key limitations relative to CCP (per proposal Table 2):
      L1: Guidelines must be pre-computed offline.
      L2: Category-level granularity — cannot distinguish causal necessity
          within history elements.
      L3: Task-agnostic — same guideline regardless of current goal state.
    """

    def __init__(
        self,
        guidelines:      Optional[ACONGuidelines] = None,
        benchmark:       str = "appworld",
        token_threshold: int = 4000,
    ):
        self.guidelines      = guidelines or load_guidelines(benchmark)
        self.token_threshold = token_threshold
        self._context        = AgentContext(goal="")
        self._step           = 0
        self._stats_log: List[CCPStats] = []

        if self.guidelines.source == "default":
            print(
                f"[ACON] Using default guidelines for '{benchmark}'. "
                f"Run ACONOfflineOptimizer for optimized results."
            )

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
        """
        ACON Phase 2 compression: category-level, guideline-based.
        Two LLM calls per compression event: one for history, one for observation.
        """
        if len(self._context.elements) < 2:
            return

        tokens_before = self._context.total_tokens()
        n_before      = len(self._context.elements)

        # Split: history = all but latest; current_obs = latest
        history     = self._context.elements[:-1]
        current_obs = self._context.elements[-1]

        # --- Compress HISTORY (uniform across all history elements — Limitation L2) ---
        history_items = [
            {"step": e.step, "action": e.action_str(),
             "observation": e.tool_output[:400]}
            for e in history
        ]
        system_h = _HISTORY_COMPRESS_SYSTEM.format(
            guideline=self.guidelines.history_guideline
        )
        user_h = (
            f"Goal: {self._context.goal}\n\n"
            f"History:\n{json.dumps(history_items, indent=2)}"
        )
        raw_h = call_llm(system_prompt=system_h, user_prompt=user_h)

        try:
            clean = re.sub(r"```(?:json)?|```", "", raw_h).strip()
            compressed_history = json.loads(clean)
            step_to_obs = {item["step"]: item.get("observation", "")
                           for item in compressed_history}
            for e in history:
                comp = step_to_obs.get(e.step)
                if comp and comp != e.tool_output:
                    e.compressed_output = comp
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            print(f"[ACON] History compression parse error: {exc}")

        # --- Compress CURRENT OBSERVATION (observation guideline) ---
        system_o = _OBS_COMPRESS_SYSTEM.format(
            guideline=self.guidelines.observation_guideline
        )
        user_o = (
            f"Tool: {current_obs.tool_name}\n"
            f"Observation:\n{current_obs.tool_output}"
        )
        comp_obs = call_llm(system_prompt=system_o, user_prompt=user_o)
        if comp_obs and comp_obs != current_obs.tool_output:
            current_obs.compressed_output = comp_obs

        tokens_after = self._context.total_tokens()
        self._stats_log.append(CCPStats(
            step=self._step,
            total_elements=n_before,
            active_count=n_before,
            relevant_count=0,
            inert_count=0,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scorer_calls=2,
        ))

        delta_pct = (1 - tokens_after / max(tokens_before, 1)) * 100
        direction = f"-{delta_pct:.1f}%" if delta_pct >= 0 else f"+{abs(delta_pct):.1f}%"
        print(f"[ACON] Step {self._step}: {tokens_before}→{tokens_after} tokens "
              f"({direction}) | history={len(history)} obs=1")


# ---------------------------------------------------------------------------
# Published ACON results for ablation A4
# ---------------------------------------------------------------------------

@dataclass
class ACONBenchmarkResult:
    benchmark:           str
    task_success_rate:   float
    token_reduction_pct: float
    note:                str = ""


ACON_REPORTED = [
    ACONBenchmarkResult("AppWorld",          0.61, 26.0,
        "Table 2, Kang et al. 2025. 5 optimization iterations."),
    ACONBenchmarkResult("OfficeBench",       0.58, 54.0,
        "Table 2, Kang et al. 2025."),
    ACONBenchmarkResult("Multi-objective QA",0.72, 41.0,
        "Table 2, Kang et al. 2025."),
]


def get_acon_reported(benchmark: str) -> Optional[ACONBenchmarkResult]:
    for r in ACON_REPORTED:
        if r.benchmark.lower() == benchmark.lower():
            return r
    return None


def compare_ccp_vs_acon(ccp_results: List[Any], benchmark: str = "AppWorld") -> None:
    """Print the CCP vs. ACON comparison table (Table 2 in proposal)."""
    from ..benchmarks.metrics import compute_all_metrics

    acon_ref    = get_acon_reported(benchmark)
    ccp_metrics = compute_all_metrics(ccp_results, method="CCP (ours)")

    print(f"\n{'='*70}")
    print(f"CCP vs. ACON — {benchmark}")
    print(f"{'='*70}")

    props = [
        ("Selection criterion",     "Task-specific guidelines (offline)", "Causal necessity (online)"),
        ("Offline data required",   "Yes (paired trajectories)",          "No"),
        ("Online adaptation",       "No",                                 "Yes"),
        ("Compression granularity", "Category-level (history/obs)",       "Element-level"),
        ("Theoretical grounding",   "Empirical (guideline optim.)",       "Do-calculus"),
        ("MCP structure exploited", "No",                                 "Yes"),
    ]
    col_w = [30, 36, 22]
    def fmt(row): return "  ".join(c.ljust(w) for c, w in zip(row, col_w))
    print(fmt(["Property", "ACON", "CCP (ours)"]))
    print("  ".join("-" * w for w in col_w))
    for row in props:
        print(fmt(row))

    print(f"\n{'─'*70}")
    print(f"{'Metric':<38} {'ACON (reported)':>18} {'CCP (ours)':>12}")
    print(f"{'─'*70}")
    if acon_ref:
        print(f"{'Task Success Rate':<38} {acon_ref.task_success_rate:>18.3f} "
              f"{ccp_metrics['task_success_rate']:>12.3f}")
        print(f"{'Token Reduction %':<38} {acon_ref.token_reduction_pct:>17.1f}% "
              f"{'(computed)':>12}")
    print(f"{'Offline optimization required':<38} {'Yes (5 iters)':>18} {'No':>12}")
    print(f"{'Adaptation to new task types':<38} {'Full re-run':>18} {'0 tasks':>12}")
    print(f"{'='*70}\n")
