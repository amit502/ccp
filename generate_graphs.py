#!/usr/bin/env python3
"""
generate_graphs.py — Paper-quality plots from experiment run folders.

Usage:
    python generate_graphs.py <results_folder> [--out DIR] [--format pdf|png|svg] [--dpi 300]

Same folder structure as generate_tables.py:
    results_folder/
        run_1/   (one CSV with all 3 benchmarks)
        run_2/
        ...

Single run → plain bars.  Multiple runs → mean + std error bars.

Figures produced:
    eff_comparison.{fmt}   Main result: Compression Efficiency per method, per benchmark
    frontier.{fmt}         Efficiency frontier: Success Rate vs Context Dependency scatter
    metrics_grid.{fmt}     4-metric overview (Success, Peak Tok, CtxDep, Eff) per benchmark
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

METHOD_ORDER = ["no_compression", "fifo", "token_perplexity", "retrieval", "acon", "ccp"]

METHOD_LABELS = {
    "no_compression":   "No Comp.",
    "fifo":             "FIFO",
    "token_perplexity": "Tok-PPL",
    "retrieval":        "Retrieval",
    "acon":             "ACON",
    "ccp":              "CCP (ours)",
}

METHOD_COLORS = {
    "no_compression":   "#9e9e9e",
    "fifo":             "#e57373",
    "token_perplexity": "#ffb74d",
    "retrieval":        "#81c784",
    "acon":             "#64b5f6",
    "ccp":              "#8e24aa",
}

BENCH_ORDER   = ["AppWorld", "MultiObjQA", "OfficeBench"]
BENCH_DISPLAY = {"AppWorld": "AppWorld", "MultiObjQA": "MultiQA", "OfficeBench": "OfficeBench"}

METRICS: List[Tuple[str, str, bool, str]] = [
    ("task_success_rate",     "Success Rate ↑",          True,  ".3f"),
    ("mean_peak_tokens",      "Peak Token Usage ↓",       False, ".0f"),
    ("context_dependency",    "Context Dependency ↓",     False, ".0f"),
    ("compression_efficiency","Compression Efficiency ↑", True,  ".4f"),
]

# ---------------------------------------------------------------------------
# Data loading  (mirrors generate_tables.py)
# ---------------------------------------------------------------------------

RunData = Dict[Tuple[str, str], Dict[str, List[float]]]


def collect_runs(root: Path) -> Tuple[RunData, int]:
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"[ERROR] No subdirectories in {root}", file=sys.stderr)
        sys.exit(1)

    data: RunData = defaultdict(lambda: defaultdict(list))
    valid = 0

    for run_dir in run_dirs:
        csvs = sorted(run_dir.glob("*.csv"))
        if not csvs:
            print(f"[WARN] Skipping {run_dir.name}: no CSV", file=sys.stderr)
            continue
        valid += 1
        for csv_path in csvs:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    bench  = row.get("benchmark", "").strip()
                    method = row.get("method", "").strip()
                    if not bench or not method:
                        continue
                    for col, _, _, _ in METRICS:
                        try:
                            data[(bench, method)][col].append(float(row[col]))
                        except (KeyError, ValueError):
                            pass

    if valid == 0:
        print("[ERROR] No valid runs found.", file=sys.stderr)
        sys.exit(1)

    return data, valid


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return m, s


def get(data: RunData, bench: str, method: str, col: str) -> Tuple[float, float]:
    vals = data.get((bench, method), {}).get(col, [])
    return mean_std(vals)


# ---------------------------------------------------------------------------
# Figure 1: Compression Efficiency comparison  (main result)
# ---------------------------------------------------------------------------

def fig_eff_comparison(data: RunData, run_count: int, out_path: Path) -> None:
    multi = run_count > 1
    n_bench = len(BENCH_ORDER)
    fig, axes = plt.subplots(1, n_bench, figsize=(4.5 * n_bench, 4.2), sharey=False)
    fig.suptitle("Compression Efficiency  (↑ higher is better)", fontsize=12, fontweight="bold", y=1.01)

    for ax, bench in zip(axes, BENCH_ORDER):
        display = BENCH_DISPLAY.get(bench, bench)
        methods = [m for m in METHOD_ORDER if (bench, m) in data]
        means   = []
        stds    = []

        for m in methods:
            mv, sv = get(data, bench, m, "compression_efficiency")
            means.append(mv)
            stds.append(sv)

        x      = np.arange(len(methods))
        colors = [METHOD_COLORS.get(m, "#607d8b") for m in methods]
        edges  = ["#212121" if m == "ccp" else "white" for m in methods]
        lws    = [2.0 if m == "ccp" else 0.5 for m in methods]

        bars = ax.bar(x, means, color=colors, edgecolor=edges, linewidth=lws, zorder=3)

        if multi:
            ax.errorbar(x, means, yerr=stds, fmt="none", color="#333333",
                        capsize=3, capthick=1, elinewidth=1, zorder=4)

        # Value labels on top of bars
        for bar, val in zip(bars, means):
            if not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(display, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=35, ha="right")
        ax.set_ylabel("Eff = Success / (Tokens / 1K)" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2: Efficiency frontier  (Success Rate vs Context Dependency scatter)
# ---------------------------------------------------------------------------

def fig_frontier(data: RunData, run_count: int, out_path: Path) -> None:
    multi = run_count > 1
    n_bench = len(BENCH_ORDER)
    fig, axes = plt.subplots(1, n_bench, figsize=(4.5 * n_bench, 4.2), sharey=False)
    fig.suptitle("Efficiency Frontier  (top-left = better: low tokens, high success)",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, bench in zip(axes, BENCH_ORDER):
        display = BENCH_DISPLAY.get(bench, bench)
        ax.set_title(display, fontweight="bold")
        ax.set_xlabel("Context Dependency ↓  (mean tokens)")
        ax.set_ylabel("Task Success Rate ↑" if ax == axes[0] else "")

        # Draw Eff iso-curves (dashed)
        all_x = [get(data, bench, m, "context_dependency")[0]
                 for m in METHOD_ORDER if (bench, m) in data]
        all_x = [v for v in all_x if not math.isnan(v)]
        if all_x:
            x_range = np.linspace(max(1, min(all_x) * 0.5), max(all_x) * 1.2, 200)
            for eff_level in [1, 3, 6, 10]:
                y_iso = eff_level * x_range / 1000
                mask  = (y_iso >= 0) & (y_iso <= 1.05)
                if mask.any():
                    ax.plot(x_range[mask], y_iso[mask], "--", color="#cccccc",
                            linewidth=0.8, zorder=1)
                    xi = x_range[mask][-1]
                    yi = y_iso[mask][-1]
                    ax.text(xi, yi, f"Eff={eff_level}", fontsize=6, color="#aaaaaa",
                            ha="left", va="center")

        for m in METHOD_ORDER:
            if (bench, m) not in data:
                continue
            mx, sx = get(data, bench, m, "context_dependency")
            my, sy = get(data, bench, m, "task_success_rate")
            if math.isnan(mx) or math.isnan(my):
                continue

            is_ccp = (m == "ccp")
            size   = 120 if is_ccp else 70
            ec     = "#212121" if is_ccp else "white"
            lw     = 2.0 if is_ccp else 0.8
            zorder = 5 if is_ccp else 4

            ax.scatter(mx, my, s=size, color=METHOD_COLORS.get(m, "#607d8b"),
                       edgecolors=ec, linewidths=lw, zorder=zorder)

            if multi:
                ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="none",
                            color=METHOD_COLORS.get(m, "#607d8b"),
                            capsize=2, capthick=0.8, elinewidth=0.8, zorder=3)

            offset_x = mx * 0.03 + 2
            ax.annotate(METHOD_LABELS.get(m, m), (mx, my),
                        xytext=(offset_x, 0), textcoords="offset points",
                        fontsize=7, va="center",
                        color=METHOD_COLORS.get(m, "#607d8b"),
                        fontweight="bold" if is_ccp else "normal")

        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

    # Shared legend
    handles = [mpatches.Patch(color=METHOD_COLORS.get(m, "#607d8b"),
                               label=METHOD_LABELS.get(m, m))
               for m in METHOD_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHOD_ORDER),
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3: 4-metric overview grid  (2 rows × 3 cols per metric)
# ---------------------------------------------------------------------------

def fig_metrics_grid(data: RunData, run_count: int, out_path: Path) -> None:
    multi   = run_count > 1
    n_bench = len(BENCH_ORDER)
    n_met   = len(METRICS)

    fig, axes = plt.subplots(n_met, n_bench,
                             figsize=(4.0 * n_bench, 3.2 * n_met),
                             squeeze=False)
    fig.suptitle("All Metrics by Benchmark and Method",
                 fontsize=13, fontweight="bold", y=1.01)

    for r, (col, label, higher, fmt) in enumerate(METRICS):
        for c, bench in enumerate(BENCH_ORDER):
            ax = axes[r][c]
            display = BENCH_DISPLAY.get(bench, bench)

            if r == 0:
                ax.set_title(display, fontweight="bold", fontsize=10)
            if c == 0:
                short_label = label.split(" ")[0] + " " + label.split(" ")[1]
                ax.set_ylabel(short_label, fontsize=8)

            methods = [m for m in METHOD_ORDER if (bench, m) in data]
            means, stds = [], []
            for m in methods:
                mv, sv = get(data, bench, m, col)
                means.append(mv)
                stds.append(sv)

            x      = np.arange(len(methods))
            colors = [METHOD_COLORS.get(m, "#607d8b") for m in methods]
            edges  = ["#212121" if m == "ccp" else "white" for m in methods]
            lws    = [2.0 if m == "ccp" else 0.5 for m in methods]

            ax.bar(x, means, color=colors, edgecolor=edges, linewidth=lws, zorder=3)

            if multi:
                ax.errorbar(x, means, yerr=stds, fmt="none", color="#333333",
                            capsize=2, capthick=0.8, elinewidth=0.8, zorder=4)

            # Mark best
            valid = [(i, v) for i, v in enumerate(means) if not math.isnan(v)]
            if valid:
                best_i = max(valid, key=lambda t: t[1] if higher else -t[1])[0]
                ax.get_children()[best_i].set_linewidth(2.5)

            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                               rotation=40, ha="right", fontsize=7)
            better_arrow = "↑" if higher else "↓"
            ax.set_title(f"{display}\n{better_arrow}", fontsize=8) if r > 0 else None

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality plots from experiment run folders."
    )
    parser.add_argument("folder", type=Path,
                        help="Root folder containing one subfolder per run.")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="Output directory (default: <folder>/plots/)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                        help="Image format (default: pdf)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster formats (default: 300)")
    args = parser.parse_args()

    root = args.folder.resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out or root / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.dpi"] = args.dpi

    data, run_count = collect_runs(root)
    print(f"\nLoaded {run_count} run(s) from {root}")
    print(f"Saving plots ({args.format}) to {out_dir}\n")

    fmt = args.format
    fig_eff_comparison(data, run_count, out_dir / f"eff_comparison.{fmt}")
    fig_frontier      (data, run_count, out_dir / f"frontier.{fmt}")
    fig_metrics_grid  (data, run_count, out_dir / f"metrics_grid.{fmt}")

    print(f"\nDone. {out_dir}/")


if __name__ == "__main__":
    main()
