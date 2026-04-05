"""
plotting/plot_results.py

Generate all comparison plots from saved experiment CSV results.

Plots produced:
  1. Per-benchmark bar charts — all methods, all 5 metrics
  2. Cross-benchmark heatmap — method × benchmark × metric
  3. Ablation curves (A1 threshold, A2 faithfulness, A3 MCP struct)
  4. Token reduction vs. task success scatter (efficiency frontier)
  5. Compression event timeline per task (step-level detail)

Usage:
    python plotting/plot_results.py --results-dir /results/ccp --output-dir /results/ccp/plots

Arguments:
    --results-dir   Directory containing experiment CSV files (default: ./results)
    --output-dir    Where to save plots               (default: ./results/plots)
    --format        Image format: pdf | png | svg     (default: pdf)
    --dpi           Resolution for raster formats     (default: 300)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless K8s)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colour palette — consistent across all plots
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "no_compression":    "#9e9e9e",   # grey
    "fifo":              "#e57373",   # red
    "token_perplexity":  "#ffb74d",   # orange
    "retrieval":         "#81c784",   # green
    "acon":              "#64b5f6",   # blue
    "acon_optimized":    "#1565c0",   # dark blue
    "ccp":               "#8e24aa",   # purple (our method)
    "ccp_binary_scorer": "#6a1b9a",
    "ccp_no_mcp_structure": "#ab47bc",
    "ccp_with_mcp_structure": "#8e24aa",
    "ccp_online":        "#8e24aa",
}

METHOD_LABELS = {
    "no_compression":       "No Compression",
    "fifo":                 "FIFO",
    "token_perplexity":     "LLMLingua-style",
    "retrieval":            "Retrieval-Based",
    "acon":                 "ACON (default)",
    "acon_optimized":       "ACON (optimised)",
    "ccp":                  "CCP (ours)",
    "ccp_binary_scorer":    "CCP (binary scorer)",
    "ccp_llm_scorer_only":  "CCP (LLM scorer only)",
    "ccp_no_mcp_structure": "CCP (no MCP struct)",
    "ccp_with_mcp_structure":"CCP (with MCP struct)",
    "ccp_online":           "CCP (online)",
}

BENCHMARK_ORDER = ["AppWorld", "OfficeBench", "Multi-QA"]

METRIC_LABELS = {
    "task_success_rate":      "Task Success Rate ↑",
    "mean_peak_tokens":       "Peak Token Usage ↓",
    "context_dependency":     "Context Dependency ↓",
    "causal_recall":          "Causal Recall ↑",
    "compression_efficiency": "Compression Efficiency ↑",
}

HIGHER_IS_BETTER = {
    "task_success_rate":      True,
    "mean_peak_tokens":       False,
    "context_dependency":     False,
    "causal_recall":          True,
    "compression_efficiency": True,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV result files from results_dir.
    Returns a dict: filename_stem → DataFrame.
    """
    dfs = {}
    for csv_path in sorted(results_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            # Coerce numeric columns
            for col in METRIC_LABELS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            dfs[csv_path.stem] = df
            print(f"  Loaded: {csv_path.name}  ({len(df)} rows)")
        except Exception as e:
            print(f"  Skipped {csv_path.name}: {e}")
    return dfs


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#607d8b")


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Plot 1: Per-benchmark bar charts (one chart per benchmark × metric)
# ---------------------------------------------------------------------------

def plot_per_benchmark_bars(dfs: Dict[str, pd.DataFrame], output_dir: Path, fmt: str, dpi: int):
    """
    For each benchmark, produce a grouped bar chart showing all methods × all metrics.
    Also produces one subplot figure with all 5 metrics as subplots.
    """
    benchmark_map = {
        "appworld_comparison":  "AppWorld",
        "main_comparison":      "AppWorld",
        "officebench_comparison": "OfficeBench",
        "multiqa_comparison":   "Multi-QA",
        "multi-qa_comparison":  "Multi-QA",
    }

    for csv_stem, bench_name in benchmark_map.items():
        if csv_stem not in dfs:
            continue
        df = dfs[csv_stem]
        if "method" not in df.columns:
            continue

        metrics = [m for m in METRIC_LABELS if m in df.columns and m != "causal_recall"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
        fig.suptitle(f"{bench_name}: Method Comparison", fontsize=14, fontweight="bold")

        for ax, metric in zip(axes, metrics):
            methods = df["method"].tolist()
            values  = df[metric].tolist()
            colors  = [_method_color(m) for m in methods]
            labels  = [_method_label(m) for m in methods]

            bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white", linewidth=0.5)

            # Highlight CCP bar
            for i, m in enumerate(methods):
                if "ccp" in m.lower() and "acon" not in m.lower():
                    bars[i].set_edgecolor("#212121")
                    bars[i].set_linewidth(2)

            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(METRIC_LABELS[metric], fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

            # Arrow indicating better direction
            better = "↑ better" if HIGHER_IS_BETTER[metric] else "↓ better"
            ax.set_xlabel(better, fontsize=7, color="grey")

        plt.tight_layout()
        out_path = output_dir / f"bar_{bench_name.lower().replace('-','_')}_all_metrics.{fmt}"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Cross-benchmark heatmap  (method × benchmark, coloured by success rate)
# ---------------------------------------------------------------------------

def plot_cross_benchmark_heatmap(dfs: Dict[str, pd.DataFrame], output_dir: Path, fmt: str, dpi: int):
    """
    Heatmap: rows = methods, cols = benchmarks, cell = task_success_rate.
    Annotated with token_reduction % where available.
    """
    bench_csv = {
        "AppWorld":   ["appworld_comparison", "main_comparison"],
        "OfficeBench":["officebench_comparison"],
        "Multi-QA":   ["multiqa_comparison", "multi-qa_comparison"],
    }

    # Collect success rates
    data: Dict[str, Dict[str, float]] = {}   # method → bench → success_rate
    for bench, stems in bench_csv.items():
        for stem in stems:
            if stem in dfs:
                df = dfs[stem]
                if "method" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    m = row["method"]
                    if m not in data:
                        data[m] = {}
                    data[m][bench] = float(row.get("task_success_rate", float("nan")))
                break

    if not data:
        print("  [heatmap] No cross-benchmark data found — skipping.")
        return

    methods    = sorted(data.keys(), key=lambda m: ("ccp" not in m, m))
    benchmarks = [b for b in BENCHMARK_ORDER if any(b in d for d in data.values())]

    matrix = np.full((len(methods), len(benchmarks)), np.nan)
    for i, m in enumerate(methods):
        for j, b in enumerate(benchmarks):
            matrix[i, j] = data[m].get(b, np.nan)

    fig, ax = plt.subplots(figsize=(2.5 * len(benchmarks) + 1, 0.6 * len(methods) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_method_label(m) for m in methods], fontsize=9)

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(benchmarks)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Task Success Rate")
    ax.set_title("Task Success Rate: Method × Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / f"heatmap_cross_benchmark.{fmt}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 3: Ablation A1 — threshold sensitivity curves
# ---------------------------------------------------------------------------

def plot_ablation_threshold(dfs: Dict[str, pd.DataFrame], output_dir: Path, fmt: str, dpi: int):
    stem = "ablation_a1_threshold"
    if stem not in dfs:
        return
    df = dfs[stem]
    if not {"tau_high", "task_success_rate", "mean_peak_tokens"}.issubset(df.columns):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Ablation A1: Threshold Sensitivity (τ_H, τ_L)", fontsize=13, fontweight="bold")

    for tau_l, group in df.groupby("tau_low"):
        group = group.sort_values("tau_high")
        label = f"τ_L={tau_l}"
        ax1.plot(group["tau_high"], group["task_success_rate"],
                 marker="o", label=label)
        ax2.plot(group["tau_high"], group["mean_peak_tokens"],
                 marker="s", label=label)

    ax1.set_xlabel("τ_H (active threshold)")
    ax1.set_ylabel("Task Success Rate ↑")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("τ_H (active threshold)")
    ax2.set_ylabel("Peak Token Usage ↓")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"ablation_a1_threshold.{fmt}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 4: Efficiency frontier (success rate vs. token reduction)
# ---------------------------------------------------------------------------

def plot_efficiency_frontier(dfs: Dict[str, pd.DataFrame], output_dir: Path, fmt: str, dpi: int):
    """
    Scatter plot: x = peak token usage, y = task success rate.
    Methods closer to top-left dominate (high success, low tokens).
    """
    bench_stems = ["appworld_comparison", "main_comparison",
                   "officebench_comparison", "multiqa_comparison"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    bench_names = ["AppWorld", "OfficeBench", "Multi-QA"]
    stem_lists  = [
        ["appworld_comparison", "main_comparison"],
        ["officebench_comparison"],
        ["multiqa_comparison", "multi-qa_comparison"],
    ]

    any_plotted = False

    for ax, bench_name, stems in zip(axes, bench_names, stem_lists):
        ax.set_title(bench_name, fontsize=11)
        ax.set_xlabel("Peak Token Usage ↓", fontsize=9)
        ax.set_ylabel("Task Success Rate ↑", fontsize=9)
        ax.grid(alpha=0.3)

        for stem in stems:
            if stem not in dfs:
                continue
            df = dfs[stem]
            if not {"method", "task_success_rate", "mean_peak_tokens"}.issubset(df.columns):
                continue

            for _, row in df.iterrows():
                m    = row["method"]
                x    = float(row.get("mean_peak_tokens", float("nan")))
                y    = float(row.get("task_success_rate", float("nan")))
                size = 120 if "ccp" in m else 60
                ec   = "#212121" if "ccp" in m else "white"
                ax.scatter(x, y, color=_method_color(m), s=size, zorder=5,
                           edgecolors=ec, linewidths=1.5, label=_method_label(m))
                ax.annotate(_method_label(m), (x, y),
                            textcoords="offset points", xytext=(6, 3), fontsize=7)
            any_plotted = True
            break

        if not any_plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")

    # Build shared legend from last axes
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.15))

    fig.suptitle("Efficiency Frontier: Success Rate vs. Token Usage", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / f"efficiency_frontier.{fmt}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 5: Causal Recall bar (CCP-only metric)
# ---------------------------------------------------------------------------

def plot_causal_recall(dfs: Dict[str, pd.DataFrame], output_dir: Path, fmt: str, dpi: int):
    """Bar chart showing Causal Recall for CCP variants across benchmarks."""
    rows = []
    bench_stems = {
        "AppWorld":    ["appworld_comparison", "main_comparison"],
        "OfficeBench": ["officebench_comparison"],
        "Multi-QA":    ["multiqa_comparison"],
    }

    for bench, stems in bench_stems.items():
        for stem in stems:
            if stem not in dfs:
                continue
            df = dfs[stem]
            if "causal_recall" not in df.columns:
                break
            for _, row in df.iterrows():
                val = row.get("causal_recall", float("nan"))
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                rows.append({"method": row["method"], "benchmark": bench, "causal_recall": val})
            break

    if not rows:
        print("  [causal_recall] No data — skipping.")
        return

    plot_df = pd.DataFrame(rows)
    ccp_df  = plot_df[plot_df["method"].str.contains("ccp", case=False, na=False)]
    if ccp_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    benchmarks = ccp_df["benchmark"].unique()
    x = np.arange(len(benchmarks))
    methods  = ccp_df["method"].unique()
    width    = 0.8 / max(len(methods), 1)

    for i, m in enumerate(methods):
        vals = [ccp_df[(ccp_df["method"] == m) & (ccp_df["benchmark"] == b)]["causal_recall"].mean()
                for b in benchmarks]
        ax.bar(x + i * width, vals, width, label=_method_label(m),
               color=_method_color(m), edgecolor="white")

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Causal Recall ↑")
    ax.set_title("Causal Recall: CCP Novel Metric\n(fraction of causally-active elements preserved)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    out_path = output_dir / f"causal_recall.{fmt}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 6: Ablation A3/A4 — binary comparison bars
# ---------------------------------------------------------------------------

def plot_ablation_comparison(dfs: Dict[str, pd.DataFrame], output_dir: Path, fmt: str, dpi: int):
    ablation_pairs = {
        "ablation_a2_faithfulness": ("Ablation A2: Scorer Faithfulness",
                                     ["task_success_rate", "mean_peak_tokens"]),
        "ablation_a3_mcp_structure": ("Ablation A3: MCP Structure Benefit",
                                      ["task_success_rate", "compression_efficiency"]),
        "ablation_a4_online_offline": ("Ablation A4: Online vs. Offline (CCP vs. ACON)",
                                       ["task_success_rate", "mean_peak_tokens"]),
    }

    for stem, (title, metrics) in ablation_pairs.items():
        if stem not in dfs:
            continue
        df = dfs[stem]
        if "method" not in df.columns:
            continue

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric not in df.columns:
                continue
            methods = df["method"].tolist()
            values  = pd.to_numeric(df[metric], errors="coerce").tolist()
            colors  = [_method_color(m) for m in methods]
            labels  = [_method_label(m) for m in methods]

            bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white")
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
            better = "↑ better" if HIGHER_IS_BETTER.get(metric, True) else "↓ better"
            ax.set_xlabel(better, fontsize=7, color="grey")
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"{stem}.{fmt}"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CCP Experiment Plotting")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing experiment CSV files")
    parser.add_argument("--output-dir",  type=Path, default=None,
                        help="Where to save plots (default: results-dir/plots)")
    parser.add_argument("--format",      default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output image format")
    parser.add_argument("--dpi",         type=int, default=300,
                        help="DPI for raster formats (png)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir  = args.output_dir or (results_dir / "plots")

    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from: {results_dir}")
    print(f"Saving plots to:      {output_dir}")
    print(f"Format: {args.format} | DPI: {args.dpi}\n")

    dfs = load_results(results_dir)
    if not dfs:
        print("No CSV files found. Run experiments first.")
        sys.exit(1)

    print("\nGenerating plots...")
    plot_per_benchmark_bars(dfs, output_dir, args.format, args.dpi)
    plot_cross_benchmark_heatmap(dfs, output_dir, args.format, args.dpi)
    plot_ablation_threshold(dfs, output_dir, args.format, args.dpi)
    plot_efficiency_frontier(dfs, output_dir, args.format, args.dpi)
    plot_causal_recall(dfs, output_dir, args.format, args.dpi)
    plot_ablation_comparison(dfs, output_dir, args.format, args.dpi)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
