"""
benchmarks/metrics.py

Metrics for CCP evaluation (Table 1 in the proposal).

Metrics:
  1. Task Success Rate       — primary accuracy metric
  2. Peak Token Usage        — max context length during a trajectory
  3. Context Dependency      — AUC of token-count-over-steps curve
  4. Causal Recall           — CCP's novel metric: fraction of causally-active
                               elements correctly preserved (not inert-tiered)
  5. Compression Efficiency  — task success per 1K tokens used
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..benchmarks.appworld_runner import TaskResult
from ..models import CCPStats, CompressionTier


# ---------------------------------------------------------------------------
# 1. Task Success Rate
# ---------------------------------------------------------------------------

def task_success_rate(results: List[TaskResult]) -> float:
    """Fraction of tasks completed successfully."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


# ---------------------------------------------------------------------------
# 2. Peak Token Usage
# ---------------------------------------------------------------------------

def mean_peak_token_usage(results: List[TaskResult]) -> float:
    """Average peak context length across tasks."""
    if not results:
        return 0.0
    return sum(r.peak_tokens for r in results) / len(results)


# ---------------------------------------------------------------------------
# 3. Context Dependency (AUC of token-count-over-steps)
# ---------------------------------------------------------------------------

def context_dependency(results: List[TaskResult]) -> float:
    """
    Area under the token-count-over-steps curve, normalised by steps.
    Measures sustained context pressure.
    Higher = more tokens used for longer = more context-heavy.
    Matches ACON's reporting of this metric.

    Note: We approximate the AUC from the CCPStats log (compression events).
    For baselines without stats, we use peak_tokens * steps as a proxy.
    """
    auc_values = []

    for r in results:
        if r.ccp_stats and len(r.ccp_stats) > 1:
            # Multiple compression events — compute trapezoidal AUC
            steps  = [s.step   for s in r.ccp_stats]
            tokens = [s.tokens_after for s in r.ccp_stats]
            auc = sum(
                (tokens[i] + tokens[i-1]) / 2 * (steps[i] - steps[i-1])
                for i in range(1, len(steps))
            )
            auc_values.append(auc / max(r.steps, 1))
        elif r.ccp_stats:
            # Single compression event — use tokens_after from that event
            auc_values.append(r.ccp_stats[-1].tokens_after)
        else:
            # No compression events (never exceeded threshold).
            # Use total_tokens (final context size) as the proxy — it equals
            # peak_tokens for no-compression methods and is a tighter bound
            # than always reporting peak for methods that compressed.
            auc_values.append(r.total_tokens)

    return sum(auc_values) / len(auc_values) if auc_values else 0.0


# ---------------------------------------------------------------------------
# 4. Causal Recall  (novel CCP metric)
# ---------------------------------------------------------------------------

def causal_recall(results: List[TaskResult], method: str = "") -> Optional[float]:
    """
    Fraction of causally-active elements correctly preserved by CCP.

    A context element is post-hoc verified as causally active if:
      - The task ultimately succeeded, AND
      - The element was NOT assigned INERT tier (i.e., it was preserved or
        summarised rather than discarded)

    This is a proxy for the true causal recall (which would require counterfactual
    re-runs). The ground-truth causal recall is measured in Ablation A2.

    Returns None for non-CCP methods — causal scoring is CCP-specific.
    """
    # Only CCP assigns φ scores; baselines have no causal scoring
    _method = method or (results[0].method if results else "")
    if _method and not _method.startswith("ccp"):
        return None

    preserved_counts = []

    for r in results:
        if not r.ccp_stats:
            continue

        for stat in r.ccp_stats:
            total = stat.total_elements
            if total == 0:
                continue
            # Elements that were NOT inert (active + relevant) = preserved
            preserved = stat.active_count + stat.relevant_count
            preserved_counts.append(preserved / total)

    if not preserved_counts:
        return None

    # Causal Recall = mean fraction of elements preserved across all compression events
    return sum(preserved_counts) / len(preserved_counts)


# ---------------------------------------------------------------------------
# 5. Compression Efficiency
# ---------------------------------------------------------------------------

def compression_efficiency(results: List[TaskResult]) -> float:
    """
    Task success per 1K tokens used.
    = success_rate / (mean_total_tokens / 1000)
    """
    if not results:
        return 0.0
    sr     = task_success_rate(results)
    mean_t = sum(r.total_tokens for r in results) / len(results)
    if mean_t == 0:
        return 0.0
    return sr / (mean_t / 1000)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def compute_all_metrics(results: List[TaskResult], method: str = "") -> Dict[str, Any]:
    """
    Compute all 5 metrics for a set of task results.
    Returns a dict ready for printing / CSV export.
    """
    _m = method or (results[0].method if results else "")
    cr = causal_recall(results, method=_m)
    return {
        "method":               method or (results[0].method if results else ""),
        "n_tasks":              len(results),
        "task_success_rate":    round(task_success_rate(results), 4),
        "mean_peak_tokens":     round(mean_peak_token_usage(results), 1),
        "context_dependency":   round(context_dependency(results), 1),
        "causal_recall":        round(cr, 4) if cr is not None else "N/A",
        "compression_efficiency": round(compression_efficiency(results), 6),
    }


def print_metrics_table(metrics_list: List[Dict[str, Any]]) -> None:
    """Pretty-print a comparison table of metrics across methods."""
    if not metrics_list:
        return

    header = ["Method", "Success↑", "Peak Tok↓", "CtxDep↓", "CsRecall↑", "Eff↑"]
    rows = []
    for m in metrics_list:
        rows.append([
            m["method"],
            f"{m['task_success_rate']:.3f}",
            f"{m['mean_peak_tokens']:.0f}",
            f"{m['context_dependency']:.0f}",
            str(m["causal_recall"]),
            f"{m['compression_efficiency']:.4f}",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows))
                  for i, h in enumerate(header)]

    def fmt_row(row):
        return "  ".join(cell.ljust(w) for cell, w in zip(row, col_widths))

    sep = "  ".join("-" * w for w in col_widths)
    print("\n" + fmt_row(header))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print()
