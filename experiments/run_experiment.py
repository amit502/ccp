"""
experiments/run_experiment.py

Run CCP experiments. Results saved as CSV to RESULTS_PATH env var.

Individual experiment examples:
    python -m experiments.run_experiment --experiment appworld_all
    python -m experiments.run_experiment --experiment appworld_ccp
    python -m experiments.run_experiment --experiment multiqa_all
    python -m experiments.run_experiment --experiment ablation_threshold
    python -m experiments.run_experiment --experiment acon_optimize

--experiment values:
    all                 — all benchmarks, all methods (default)
    appworld_all        — AppWorld, all 6 methods
    appworld_ccp        — AppWorld, CCP only
    appworld_fifo       — AppWorld, FIFO only
    appworld_acon       — AppWorld, ACON only
    appworld_no_compression
    appworld_retrieval
    appworld_token_perplexity
    multiqa_all         — Multi-objective QA, all 6 methods
    multiqa_ccp / multiqa_fifo / ...
    officebench_all     — OfficeBench (requires server at OFFICEBENCH_URL)
    officebench_ccp / officebench_fifo / ...
    ablation_threshold  — A1: vary τ_H, τ_L
    ablation_faithfulness — A2: binary scorer vs LLM scorer
    ablation_mcp_struct — A3: MCP structure benefit
    ablation_online     — A4: CCP online vs ACON offline
    acon_optimize       — Run ACON offline optimization, save guidelines
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

# Results written directly to RESULTS_PATH (set by Kubernetes job YAML)
RESULTS_DIR = Path(os.environ.get("RESULTS_PATH", "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Point ACON guidelines to RESULTS_DIR so they persist on PVC
# and are loaded correctly during both optimization AND evaluation.
import ccp.baselines.acon as _acon_module
_acon_module.GUIDELINES_PATH = RESULTS_DIR / "acon_guidelines"
(RESULTS_DIR / "acon_guidelines").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy imports (so missing optional deps don't crash at import time)
# ---------------------------------------------------------------------------

def _compression_methods(token_threshold: int) -> List[tuple]:
    """Return list of (method_name, factory) for all methods."""
    from ..baselines.compression import (
        FIFOManager, NoCompression, RetrievalBasedManager, TokenPerplexityManager,
    )
    from ..baselines.acon import ACONContextManager
    from ..context_manager import CCPContextManager

    return [
        ("no_compression",   lambda t=token_threshold: NoCompression(token_threshold=t)),
        ("fifo",             lambda t=token_threshold: FIFOManager(token_threshold=t)),
        ("token_perplexity", lambda t=token_threshold: TokenPerplexityManager(token_threshold=t)),
        ("retrieval",        lambda t=token_threshold: RetrievalBasedManager(token_threshold=t)),
        ("acon",             lambda t=token_threshold: ACONContextManager(token_threshold=t)),
        ("ccp",              lambda t=token_threshold: CCPContextManager(token_threshold=t)),
    ]


def _single_method_factory(method: str, token_threshold: int) -> Callable:
    """Return factory for one specific method."""
    methods = dict(_compression_methods(token_threshold))
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods)}")
    return methods[method]


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_results(rows: List[Dict], filename: str) -> Path:
    path = RESULTS_DIR / filename
    if not rows:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Runner construction — graceful: warns and returns None if unavailable
# ---------------------------------------------------------------------------

def _make_appworld_runner(max_tasks: int, max_steps: int, split: str = "test"):
    try:
        from ..benchmarks.mcp_runner import AppWorldMCPRunner
        return AppWorldMCPRunner(max_tasks=max_tasks, max_steps=max_steps, split=split)
    except Exception as e:
        print(f"[WARN] AppWorld unavailable: {e}")
        print("       Run: appworld download all && appworld server start")
        return None


def _make_officebench_runner(max_tasks: int, max_steps: int):
    try:
        from ..benchmarks.mcp_runner import OfficeBenchMCPRunner
        return OfficeBenchMCPRunner(max_tasks=max_tasks, max_steps=max_steps)
    except Exception as e:
        print(f"[WARN] OfficeBench unavailable: {e}")
        print(f"       Set OFFICEBENCH_URL and run: cd OfficeBench && python server.py")
        return None


def _make_multiqa_runner(max_tasks: int, max_steps: int):
    try:
        from ..benchmarks.mcp_runner import MultiObjQAMCPRunner
        return MultiObjQAMCPRunner(max_tasks=max_tasks, max_steps=max_steps)
    except Exception as e:
        print(f"[WARN] MultiObjQA unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Any], method: str, benchmark: str) -> Dict:
    from ..benchmarks.metrics import compute_all_metrics
    m = compute_all_metrics(results, method=method)
    m["benchmark"] = benchmark
    return m


def print_table(rows: List[Dict]) -> None:
    from ..benchmarks.metrics import print_metrics_table
    print_metrics_table(rows)


# ---------------------------------------------------------------------------
# Per-benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    runner:    Any,
    bench_name: str,
    methods:   List[tuple],    # [(name, factory), ...]
    filename:  str,
    verbose:   bool,
) -> List[Dict]:
    """Run all given methods on one benchmark, save CSV, return metrics rows."""
    all_metrics = []
    for method_name, factory in methods:
        print(f"\n  [{bench_name}] {method_name.upper()}")
        try:
            results = runner.evaluate(factory, method_name=method_name, verbose=verbose)
            metrics = compute_metrics(results, method=method_name, benchmark=bench_name)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [ERROR] {bench_name}/{method_name}: {e}")
        # Save after every method — partial results survive if job is killed
        if all_metrics:
            save_results(all_metrics, filename)

    if all_metrics:
        print_table(all_metrics)
    return all_metrics


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def run_appworld(method: str, max_tasks: int, max_steps: int, verbose: bool):
    TOKEN_THRESHOLD = int(os.environ.get("APPWORLD_TOKEN_THRESHOLD",
                                         os.environ.get("TOKEN_THRESHOLD", "1500")))
    runner = _make_appworld_runner(max_tasks, max_steps)
    if runner is None:
        return

    methods = (
        _compression_methods(TOKEN_THRESHOLD)
        if method == "all"
        else [(method, _single_method_factory(method, TOKEN_THRESHOLD))]
    )
    run_benchmark(runner, "AppWorld", methods, f"appworld_{method}.csv", verbose)


def run_officebench(method: str, max_tasks: int, max_steps: int, verbose: bool):
    TOKEN_THRESHOLD = int(os.environ.get("OFFICEBENCH_TOKEN_THRESHOLD",
                                         os.environ.get("TOKEN_THRESHOLD", "4000")))
    runner = _make_officebench_runner(max_tasks, max_steps)
    if runner is None:
        return

    methods = (
        _compression_methods(TOKEN_THRESHOLD)
        if method == "all"
        else [(method, _single_method_factory(method, TOKEN_THRESHOLD))]
    )
    run_benchmark(runner, "OfficeBench", methods, f"officebench_{method}.csv", verbose)


def run_multiqa(method: str, max_tasks: int, max_steps: int, verbose: bool):
    TOKEN_THRESHOLD = int(os.environ.get("MULTIQA_TOKEN_THRESHOLD",
                                         os.environ.get("TOKEN_THRESHOLD", "1500")))
    runner = _make_multiqa_runner(max_tasks, max_steps)
    if runner is None:
        return

    methods = (
        _compression_methods(TOKEN_THRESHOLD)
        if method == "all"
        else [(method, _single_method_factory(method, TOKEN_THRESHOLD))]
    )
    run_benchmark(runner, "MultiObjQA", methods, f"multiqa_{method}.csv", verbose)


def run_all(max_tasks: int, max_steps: int, verbose: bool):
    """Run all methods on all available benchmarks."""
    run_appworld("all", max_tasks, max_steps, verbose)
    run_officebench("all", max_tasks, max_steps, verbose)
    run_multiqa("all", max_tasks, max_steps, verbose)


# ---------------------------------------------------------------------------
# Ablations — all on AppWorld via real MCP
# ---------------------------------------------------------------------------

def run_ablation_threshold(max_tasks: int, verbose: bool):
    """A1: vary τ_H ∈ {0.2,0.4,0.6,0.8} × τ_L < τ_H."""
    from ..context_manager import CCPContextManager
    runner = _make_appworld_runner(max_tasks, max_steps=40)
    if runner is None:
        return

    TOKEN_THRESHOLD = int(os.environ.get("TOKEN_THRESHOLD", "500"))
    tau_values = [0.2, 0.4, 0.6, 0.8]
    all_metrics = []

    for tau_h in tau_values:
        for tau_l in tau_values:
            if tau_l >= tau_h:
                continue
            name    = f"ccp_tH{tau_h}_tL{tau_l}"
            factory = (lambda h=tau_h, l=tau_l, t=TOKEN_THRESHOLD:
                       CCPContextManager(tau_high=h, tau_low=l, token_threshold=t,
                                        use_heuristics=True))
            print(f"\n[A1] {name}")
            try:
                results = runner.evaluate(factory, method_name=name, verbose=verbose)
                m = compute_metrics(results, name, "AppWorld")
                m["tau_high"] = tau_h
                m["tau_low"]  = tau_l
                all_metrics.append(m)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

    if all_metrics:
        print_table(all_metrics)
        save_results(all_metrics, "ablation_a1_threshold.csv")


def run_ablation_faithfulness(max_tasks: int, verbose: bool):
    """A2: CCP with heuristics (binary scorer) vs CCP without (LLM scorer only)."""
    from ..context_manager import CCPContextManager
    runner = _make_appworld_runner(max_tasks, max_steps=40)
    if runner is None:
        return

    TOKEN_THRESHOLD = int(os.environ.get("TOKEN_THRESHOLD", "500"))
    all_metrics = []

    for use_h, name in [(True, "ccp_binary_scorer"), (False, "ccp_llm_scorer_only")]:
        factory = (lambda h=use_h, t=TOKEN_THRESHOLD:
                   CCPContextManager(tau_high=0.6, tau_low=0.3, token_threshold=t,
                                    use_heuristics=h))
        print(f"\n[A2] {name}")
        try:
            results = runner.evaluate(factory, method_name=name, verbose=verbose)
            all_metrics.append(compute_metrics(results, name, "AppWorld"))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if all_metrics:
        print_table(all_metrics)
        save_results(all_metrics, "ablation_a2_faithfulness.csv")


def run_ablation_mcp_struct(max_tasks: int, verbose: bool):
    """A3: CCP with MCP heuristics vs without (all through LLM scorer)."""
    from ..context_manager import CCPContextManager
    runner = _make_appworld_runner(max_tasks, max_steps=40)
    if runner is None:
        return

    TOKEN_THRESHOLD = int(os.environ.get("TOKEN_THRESHOLD", "500"))
    all_metrics = []

    for use_h, name in [(True, "ccp_mcp_heuristics"), (False, "ccp_no_heuristics")]:
        factory = (lambda h=use_h, t=TOKEN_THRESHOLD:
                   CCPContextManager(tau_high=0.6, tau_low=0.3, token_threshold=t,
                                    use_heuristics=h))
        print(f"\n[A3] {name}")
        try:
            results = runner.evaluate(factory, method_name=name, verbose=verbose)
            all_metrics.append(compute_metrics(results, name, "AppWorld"))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if all_metrics:
        print_table(all_metrics)
        save_results(all_metrics, "ablation_a3_mcp_struct.csv")


def run_ablation_online(max_tasks: int, verbose: bool):
    """A4: CCP (online, no pre-training) vs ACON (offline-optimized guidelines)."""
    from ..baselines.acon import ACONContextManager, get_acon_reported
    from ..context_manager import CCPContextManager
    runner = _make_appworld_runner(max_tasks, max_steps=40)
    if runner is None:
        return

    TOKEN_THRESHOLD = int(os.environ.get("TOKEN_THRESHOLD", "500"))
    all_metrics = []

    for factory, name in [
        (lambda t=TOKEN_THRESHOLD: ACONContextManager(token_threshold=t), "acon"),
        (lambda t=TOKEN_THRESHOLD: CCPContextManager(tau_high=0.6, tau_low=0.3,
                                                     token_threshold=t,
                                                     use_heuristics=True), "ccp_online"),
    ]:
        print(f"\n[A4] {name}")
        try:
            results = runner.evaluate(factory, method_name=name, verbose=verbose)
            all_metrics.append(compute_metrics(results, name, "AppWorld"))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # Append ACON published numbers for reference
    ref = get_acon_reported("AppWorld")
    if ref:
        all_metrics.append({
            "method": "acon_reported", "benchmark": "AppWorld",
            "task_success_rate": ref.task_success_rate,
            "mean_peak_tokens": "N/A",
            "context_dependency": "N/A",
            "compression_efficiency": "N/A",
            "note": f"token_reduction={ref.token_reduction_pct}% (paper)",
        })

    if all_metrics:
        print_table(all_metrics)
        save_results(all_metrics, "ablation_a4_online.csv")


def run_acon_optimize(max_tasks: int, n_iters: int, benchmark: str, verbose: bool, max_steps: int = 20):
    """
    Run ACON offline optimization on the TRAIN split.
    Saves guidelines to RESULTS_PATH/acon_guidelines/<benchmark>.json.
    Main evaluation uses TEST split — correct separation.

    OfficeBench is skipped: its MCP server is task-specific (requires task_id + app
    per call) and is incompatible with the task-agnostic server_configs assumption
    in the optimizer's task_runner. ACON will run on OfficeBench without guidelines.
    """
    import asyncio
    from ..baselines.acon import ACONOfflineOptimizer
    from ..benchmarks.mcp_runner import _run_one_task

    if benchmark == "officebench":
        print("[ACON] Skipping optimization for OfficeBench — task-specific server "
              "configs are incompatible with the offline optimizer.")
        return

    # Select the correct train runner for the benchmark
    if benchmark == "multiqa":
        train_runner = _make_multiqa_runner(max_tasks=max_tasks, max_steps=max_steps)
    else:
        # AppWorld: use TRAIN split so test tasks remain unseen during optimization
        train_runner = _make_appworld_runner(max_tasks=max_tasks, max_steps=max_steps, split="train")

    if train_runner is None:
        return

    tasks = train_runner._tasks

    optimizer = ACONOfflineOptimizer(
        benchmark=benchmark,
        n_iters=n_iters,
        n_pairs=min(20, max_tasks // 2),
    )

    def task_runner(task, manager):
        """Run one task through real MCP for trajectory collection."""
        from ..benchmarks.appworld_runner import (
            _seed_task, _reset_task, APPWORLD_ROOT, APPWORLD_URL,
        )
        if APPWORLD_ROOT and APPWORLD_URL:
            if not _seed_task(task, APPWORLD_ROOT, APPWORLD_URL):
                return [], False

        result = asyncio.run(_run_one_task(
            task_id=task.id,
            goal=task.goal,
            manager=manager,
            server_configs=train_runner._server_configs(),
            max_steps=max_steps,
            score_fn=lambda _, fs: 1.0 if fs.get("done") else 0.0,
            verbose=True,
            cached_tools=train_runner._tools,
            interceptor=train_runner._interceptor,
        ))

        if APPWORLD_ROOT and APPWORLD_URL:
            _reset_task(task.id, APPWORLD_URL)

        return manager.get_compressed_context().elements, result.success

    optimizer.run(task_runner, tasks)
    print(f"\n[ACON] Optimization complete. Guidelines at: {_acon_module.GUIDELINES_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXPERIMENT_CHOICES = [
    "all",
    "appworld_all", "appworld_ccp", "appworld_fifo", "appworld_acon",
    "appworld_no_compression", "appworld_retrieval", "appworld_token_perplexity",
    "multiqa_all", "multiqa_ccp", "multiqa_fifo", "multiqa_acon",
    "multiqa_no_compression", "multiqa_retrieval", "multiqa_token_perplexity",
    "officebench_all", "officebench_ccp", "officebench_fifo", "officebench_acon",
    "officebench_no_compression", "officebench_retrieval", "officebench_token_perplexity",
    "ablation_threshold", "ablation_faithfulness", "ablation_mcp_struct", "ablation_online",
    "acon_optimize",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CCP Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {c}" for c in EXPERIMENT_CHOICES),
    )
    parser.add_argument(
        "--experiment", "-e",
        default=os.environ.get("EXPERIMENT", "all"),
        choices=EXPERIMENT_CHOICES,
        help="Which experiment to run (also set via EXPERIMENT env var)",
    )
    parser.add_argument("--tasks",     type=int, default=int(os.environ.get("MAX_TASKS", "50")))
    parser.add_argument("--steps",     type=int, default=int(os.environ.get("MAX_STEPS", "40")))
    parser.add_argument("--acon-iters",type=int, default=int(os.environ.get("ACON_OPT_ITERS", "5")))
    parser.add_argument("--benchmark", default="appworld",
                        choices=["appworld", "multiqa", "officebench"],
                        help="Benchmark for acon_optimize")
    parser.add_argument("--verbose",   action="store_true", default=True)
    args = parser.parse_args()

    exp = args.experiment
    print(f"\n{'='*60}")
    print(f"  Experiment : {exp}")
    print(f"  Tasks      : {args.tasks}")
    print(f"  Steps      : {args.steps}")
    print(f"  Results    : {RESULTS_DIR}")
    print(f"{'='*60}\n")

    # Route to the right function
    if exp == "all":
        run_all(args.tasks, args.steps, args.verbose)

    elif exp.startswith("appworld_"):
        method = exp.removeprefix("appworld_")
        run_appworld(method, args.tasks, args.steps, args.verbose)

    elif exp.startswith("multiqa_"):
        method = exp.removeprefix("multiqa_")
        run_multiqa(method, args.tasks, args.steps, args.verbose)

    elif exp.startswith("officebench_"):
        method = exp.removeprefix("officebench_")
        run_officebench(method, args.tasks, args.steps, args.verbose)

    elif exp == "ablation_threshold":
        run_ablation_threshold(args.tasks, args.verbose)

    elif exp == "ablation_faithfulness":
        run_ablation_faithfulness(args.tasks, args.verbose)

    elif exp == "ablation_mcp_struct":
        run_ablation_mcp_struct(args.tasks, args.verbose)

    elif exp == "ablation_online":
        run_ablation_online(args.tasks, args.verbose)

    elif exp == "acon_optimize":
        run_acon_optimize(args.tasks, args.acon_iters, args.benchmark, args.verbose, max_steps=args.steps)

    print(f"\nDone. Results in: {RESULTS_DIR}")
