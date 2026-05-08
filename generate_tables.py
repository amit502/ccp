#!/usr/bin/env python3
"""
generate_tables.py — Aggregate experiment CSVs and produce summary tables.

Usage:
    python generate_tables.py <results_folder> [--out <output_folder>]

Expected structure:
    results_folder/
        run_1/   (one CSV file containing rows for all 3 benchmarks)
        run_2/
        ...

Single run  → display results directly.
Multi run   → display mean ± std across runs.

Outputs (written to <output_folder>, default: <results_folder>/tables/):
    summary.csv   flat CSV with mean (± std) per method / benchmark / metric
    table.tex     LaTeX booktabs table, one panel per benchmark
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRICS: List[Tuple[str, str, bool, str]] = [
    # (csv_column,          header,      higher_is_better, fmt)
    ("task_success_rate",    "Success↑",  True,             ".3f"),
    ("mean_peak_tokens",     "Peak Tok↓", False,            ".1f"),
    ("context_dependency",   "CtxDep↓",   False,            ".1f"),
    ("compression_efficiency","Eff↑",      True,             ".4f"),
]

BENCH_ORDER = ["AppWorld", "MultiObjQA", "OfficeBench"]
BENCH_DISPLAY = {
    "AppWorld":   "AppWorld",
    "MultiObjQA": "MultiQA",
    "OfficeBench":"OfficeBench",
}

METHOD_ORDER = ["no_compression", "fifo", "token_perplexity", "retrieval", "acon", "ccp"]
METHOD_DISPLAY = {
    "no_compression":   "No Compression",
    "fifo":             "FIFO",
    "token_perplexity": "Token Perplexity",
    "retrieval":        "Retrieval",
    "acon":             "ACON",
    "ccp":              "CCP (ours)",
}
# LaTeX version of method names (for table.tex)
METHOD_LATEX = {
    "no_compression":   "No Compression",
    "fifo":             "FIFO",
    "token_perplexity": "Token Perplexity",
    "retrieval":        "Retrieval",
    "acon":             "ACON",
    "ccp":              r"\textbf{CCP (ours)}",
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_csvs(run_dir: Path) -> List[Path]:
    """Return all CSV files in a run directory."""
    return sorted(run_dir.glob("*.csv"))


def load_rows(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

# Key: (benchmark, method)  Value: {metric_col: [val_run0, val_run1, ...]}
RunData = Dict[Tuple[str, str], Dict[str, List[float]]]


def collect_runs(root: Path) -> Tuple[RunData, int]:
    """
    Walk each run subfolder, load CSV(s), and accumulate metric values.
    Returns (data_dict, number_of_valid_runs).
    """
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"[ERROR] No subdirectories found in {root}", file=sys.stderr)
        sys.exit(1)

    data: RunData = defaultdict(lambda: defaultdict(list))
    valid_runs = 0

    for run_dir in run_dirs:
        csvs = find_csvs(run_dir)
        if not csvs:
            print(f"[WARN] Skipping {run_dir.name}: no CSV file found", file=sys.stderr)
            continue

        valid_runs += 1
        for csv_path in csvs:
            for row in load_rows(csv_path):
                bench  = row.get("benchmark", "").strip()
                method = row.get("method", "").strip()
                if not bench or not method:
                    continue
                key = (bench, method)
                for col, _, _, _ in METRICS:
                    raw = row.get(col, "").strip()
                    try:
                        data[key][col].append(float(raw))
                    except ValueError:
                        pass

    if valid_runs == 0:
        print("[ERROR] No valid run directories with CSV files found.", file=sys.stderr)
        sys.exit(1)

    return data, valid_runs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def best_per_metric(data: RunData, bench: str) -> Dict[str, float]:
    """Return the best mean value per metric across all methods for a benchmark."""
    best: Dict[str, float] = {}
    for col, _, higher, _ in METRICS:
        all_means = []
        for method in METHOD_ORDER:
            vals = data.get((bench, method), {}).get(col, [])
            if vals:
                m, _ = mean_std(vals)
                if not math.isnan(m):
                    all_means.append(m)
        if all_means:
            best[col] = max(all_means) if higher else min(all_means)
    return best


def is_best(val: float, best_val: Optional[float]) -> bool:
    if best_val is None or math.isnan(val):
        return False
    return math.isclose(val, best_val, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def fmt_cell(mean: float, std: float, fmt: str, multi_run: bool) -> str:
    if math.isnan(mean):
        return "—"
    s = f"{mean:{fmt}}"
    if multi_run and std > 0:
        s += f" ± {std:{fmt}}"
    return s


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_console(data: RunData, run_count: int) -> None:
    multi = run_count > 1
    col_w = [18] + [16 if multi else 10] * len(METRICS)

    for bench in BENCH_ORDER:
        display = BENCH_DISPLAY.get(bench, bench)
        width = sum(col_w)
        print(f"\n{'─' * width}")
        print(f"  {display}  (n={run_count} run{'s' if run_count > 1 else ''})")
        print("─" * width)

        headers = ["Method"] + [h for _, h, _, _ in METRICS]
        print("".join(h.ljust(w) for h, w in zip(headers, col_w)))
        print("".join("─" * w for w in col_w))

        best = best_per_metric(data, bench)

        for method in METHOD_ORDER:
            key = (bench, method)
            if key not in data:
                continue
            cells = [METHOD_DISPLAY.get(method, method)]
            for col, _, _, fmt in METRICS:
                vals = data[key].get(col, [])
                m, s = mean_std(vals)
                cell = fmt_cell(m, s, fmt, multi)
                if is_best(m, best.get(col)):
                    cell = f"*{cell}"   # mark best with asterisk in console
                cells.append(cell)
            print("".join(c.ljust(w) for c, w in zip(cells, col_w)))
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(data: RunData, run_count: int, out_path: Path) -> None:
    multi = run_count > 1
    fieldnames = ["benchmark", "method"]
    for col, _, _, fmt in METRICS:
        fieldnames.append(f"{col}_mean")
        if multi:
            fieldnames.append(f"{col}_std")

    rows = []
    for bench in BENCH_ORDER:
        for method in METHOD_ORDER:
            key = (bench, method)
            if key not in data:
                continue
            row: Dict[str, str] = {"benchmark": bench, "method": method}
            for col, _, _, fmt in METRICS:
                vals = data[key].get(col, [])
                m, s = mean_std(vals)
                row[f"{col}_mean"] = f"{m:{fmt}}" if not math.isnan(m) else ""
                if multi:
                    row[f"{col}_std"] = f"{s:{fmt}}" if not math.isnan(s) else ""
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV  → {out_path}")


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def write_latex(data: RunData, run_count: int, out_path: Path) -> None:
    multi = run_count > 1
    n_metrics = len(METRICS)
    n_cols = 1 + n_metrics
    col_spec = "l" + "r" * n_metrics

    caption_suffix = (
        f", mean $\\pm$ std over {run_count} runs" if multi else ""
    )

    L: List[str] = []

    def line(s: str = "") -> None:
        L.append(s)

    line(r"\begin{table*}[t]")
    line(r"  \centering")
    line(r"  \small")
    line(
        r"  \caption{Main results across three benchmarks"
        + caption_suffix
        + r". "
        r"$\uparrow$ higher is better, $\downarrow$ lower is better. "
        r"Best per column in \textbf{bold}.}"
    )
    line(r"  \label{tab:main_results}")
    line(r"  \begin{tabular}{" + col_spec + r"}")
    line(r"    \toprule")

    # Column headers
    header_cells = ["Method"] + [h.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
                                  for _, h, _, _ in METRICS]
    line("    " + " & ".join(header_cells) + r" \\")

    for b_idx, bench in enumerate(BENCH_ORDER):
        line(r"    \midrule")
        display = BENCH_DISPLAY.get(bench, bench)
        line(
            r"    \multicolumn{" + str(n_cols) + r"}{l}"
            r"{\textit{" + display + r"}} \\"
        )

        best = best_per_metric(data, bench)

        for method in METHOD_ORDER:
            key = (bench, method)
            if key not in data:
                continue
            method_tex = METHOD_LATEX.get(method, method)
            cells = [method_tex]
            for col, _, _, fmt in METRICS:
                vals = data[key].get(col, [])
                m, s = mean_std(vals)
                cell = fmt_cell(m, s, fmt, multi)
                if is_best(m, best.get(col)):
                    cell = r"\textbf{" + cell + r"}"
                cells.append(cell)
            line("    " + " & ".join(cells) + r" \\")

    line(r"    \bottomrule")
    line(r"  \end{tabular}")
    line(r"\end{table*}")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  Saved LaTeX → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary tables (CSV + LaTeX) from experiment run folders."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder containing one subfolder per run, each with a CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: <folder>/tables/)",
    )
    args = parser.parse_args()

    root = args.folder.resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out or root / "tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data, run_count = collect_runs(root)
    print(f"\nLoaded {run_count} run(s) from {root}")

    print_console(data, run_count)

    write_csv(data, run_count, out_dir / "summary.csv")
    write_latex(data, run_count, out_dir / "table.tex")


if __name__ == "__main__":
    main()
