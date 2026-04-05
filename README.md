<!-- # Causal Context Pruning (CCP)
**Training-Free Causal Necessity Scoring for Context Management in Long-Horizon Agentic Systems**

> University of South Dakota — CSC-792-U18 Agentic AI | Spring 2026
> Amit Kumar Patel | Instructor: Dr. Rodrigue Rizk

---

## Overview

Long-horizon agentic systems face a fundamental problem: as an agent executes
a multi-step task by iteratively calling tools and observing results, its context
grows without bound. This causes token cost explosion, context drift, and
irreversible information loss.

**CCP** addresses this by asking the right question:

> *If I removed this specific context element, would the agent's next action change?*

This is the **causal necessity** of a context element — the information-theoretically
correct criterion for compression decisions. CCP scores every tool-call/observation
pair with a lightweight causal necessity score ϕ ∈ [0,1] and partitions the
context into three tiers:

| Tier | Condition | Action |
|------|-----------|--------|
| **Active** | ϕ ≥ τ_H | Preserve at full resolution |
| **Relevant** | τ_L ≤ ϕ < τ_H | Compress to dense summary |
| **Inert** | ϕ < τ_L | Discard / one-line digest |

CCP is **training-free**, **online** (no offline trajectory collection), and
**MCP-native** — it exploits structured tool-call metadata for efficient scoring.

---

## Project Structure

```
ccp/
├── models.py               # Core data structures (ContextElement, AgentContext, CCPStats)
├── causal_scorer.py        # Causal necessity scorer — MCP heuristics + LLM binary classifier
├── context_manager.py      # Three-tier compression policy + trigger logic
├── agent.py                # LangGraph agent with CCP wired as a node
├── llm_client.py           # Nautilus gpt-oss client (OpenAI-compatible)
├── demo.py                 # End-to-end demo (runs without AppWorld/Docker)
│
├── baselines/
│   ├── compression.py      # NoCompression, FIFO, TokenPerplexity, RetrievalBased
│   └── acon.py             # ACON integration + published number comparison
│
├── benchmarks/
│   ├── appworld_runner.py  # AppWorld task runner + mock tools for local dev
│   └── metrics.py          # 5 metrics including novel Causal Recall
│
├── experiments/
│   └── run_experiment.py   # Main experiment runner + ablations A1–A4
│
└── tests/
    └── test_ccp.py         # 30 unit tests (all passing)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or
pip install -e ".[benchmark,dev]"
```

### 2. Set environment variables

```bash
export NAUTILUS_API_KEY="your-api-key"
export NAUTILUS_BASE_URL="https://your-nautilus-endpoint/v1"
```

### 3. Run the demo (no AppWorld/Docker needed)

```bash
PYTHONPATH=. python demo.py --method all
```

This runs a mock 10-step task (Amazon order + Gmail notification) through
all 4 methods and prints the comparison table.

### 4. Run unit tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
# 30/30 tests passing
```

### 5. Set up AppWorld (full benchmark)

```bash
# Requires Docker
appworld download all
appworld server start        # Starts API server at localhost:8000

# Run main comparison (CCP vs all baselines)
PYTHONPATH=. python -m experiments.run_experiment --tasks 50

# Run ablations
PYTHONPATH=. python -m experiments.run_experiment --ablation threshold --tasks 30
PYTHONPATH=. python -m experiments.run_experiment --ablation faithfulness --tasks 20
PYTHONPATH=. python -m experiments.run_experiment --ablation mcp_struct --tasks 30
PYTHONPATH=. python -m experiments.run_experiment --ablation online --tasks 30
```

---

## Causal Necessity Score

The ϕ score is defined as:

```
ϕ(a_i, o_i | C_t, g) = P(â_{t+1} ≠ â^{-i}_{t+1} | g)
```

Exact computation requires two full LLM forward passes. CCP approximates
it with a lightweight binary-classification call:

> *"Given the current task goal and context, if this specific tool response
> were removed, would the agent's next action change?"*

**Fast path (MCP heuristics — no LLM call):**
- `authenticate`, `get_token`, `get_api_key` → ϕ = 0.95 (always high)
- `list_items`, `search`, `ping` → ϕ = 0.15 (usually low)
- Short outputs (< 120 chars, likely IDs/tokens) → ϕ = 0.80
- Very long outputs (> 2000 chars, likely verbose lists) → ϕ = 0.20
- Error responses → ϕ = 0.05

**Slow path (LLM binary scorer):** Used when no heuristic applies.

---

## Metrics

| Metric | Description | Novel? |
|--------|-------------|--------|
| Task Success Rate | Fraction of tasks completed | No |
| Peak Token Usage | Max context length during trajectory | No |
| Context Dependency | AUC of token-count-over-steps | No |
| **Causal Recall** | Fraction of causally-active elements preserved | **Yes — CCP** |
| Compression Efficiency | Success per 1K tokens used | No |

---

## CCP vs. ACON (State of the Art)

| Property | ACON | CCP (ours) |
|----------|------|-----------|
| Selection criterion | Task-specific guidelines (offline) | Causal necessity (online) |
| Offline data required | Yes (paired trajectories) | **No** |
| Online adaptation | No | **Yes** |
| Compression granularity | Category-level | **Element-level** |
| Theoretical grounding | Empirical | **Do-calculus** |
| MCP structure exploited | No | **Yes** |

---

## Ablation Studies

| ID | Study | What it measures |
|----|-------|-----------------|
| A1 | Threshold sensitivity | Vary τ_H, τ_L ∈ {0.2, 0.4, 0.6, 0.8} — accuracy vs. token reduction |
| A2 | Scorer faithfulness | Binary scorer vs. two-pass ground truth ϕ |
| A3 | MCP structure benefit | CCP with vs. without MCP metadata heuristics |
| A4 | Online vs. offline | CCP (0 tasks adaptation) vs. ACON (full pipeline re-run) |

---

## Key Files for the Paper

- **Causal scorer** (`causal_scorer.py`) — the core theoretical contribution
- **Compression policy** (`context_manager.py`) — the three-tier algorithm
- **Causal Recall metric** (`benchmarks/metrics.py`) — the novel evaluation metric
- **ACON comparison** (`baselines/acon.py`) — Table 2 in the paper
- **Experiment runner** (`experiments/run_experiment.py`) — reproduces all results

---

## References

1. Kang et al. (2025). ACON: Optimizing Context Compression for Long-horizon LLM Agents. arXiv:2510.00615.
2. Trivedi et al. (2024). AppWorld: A Controllable World of Apps. ACL 2024.
3. Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference. EMNLP 2023.
4. Pearl, J. (2009). Causality: Models, Reasoning and Inference (2nd ed.).
5. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. -->

# Causal Context Pruning (CCP)

**Training-Free Causal Necessity Scoring for Context Management in Long-Horizon Agentic Systems**

> University of South Dakota — CSC-792-U18 Agentic AI | Spring 2026
> Amit Kumar Patel | Instructor: Dr. Rodrigue Rizk

---

## What is CCP?

Long-horizon agentic systems accumulate context as they call tools across many steps. Every API response, error message, and intermediate result piles up in the context window, causing token cost explosion, context drift, and information loss.

CCP addresses this with one question:

> _If I removed this specific context element right now, would the agent's next action change?_

This is the **causal necessity** of a context element — the theoretically correct criterion for deciding what to compress. CCP scores every tool-call/observation pair with a causal necessity score ϕ ∈ [0, 1] and partitions context into three tiers:

| Tier         | Condition     | Action                      |
| ------------ | ------------- | --------------------------- |
| **Active**   | ϕ ≥ τ_H       | Preserve at full resolution |
| **Relevant** | τ_L ≤ ϕ < τ_H | Compress to dense summary   |
| **Inert**    | ϕ < τ_L       | Discard / one-line digest   |

CCP is **training-free** (no fine-tuning, no offline data collection), **online** (adapts to new task types instantly), and **MCP-native** (exploits structured tool-call metadata for efficient scoring).

---

## Project Structure

```
ccp/
├── models.py                   # Core data structures
├── causal_scorer.py            # MCP heuristics + LLM binary scorer
├── context_manager.py          # Three-tier compression policy
├── llm_client.py               # OpenAI-compatible LLM client
├── mcp_server.py               # AppWorld MCP server (stdio transport)
├── officebench_mcp_server.py   # OfficeBench MCP server
├── nq_mcp_server.py            # Natural Questions MCP server
├── mcp_agent.py                # LangGraph agent with MCP + CCP interceptor
├── agent.py                    # Standalone agent (no MCP, for ablations)
│
├── baselines/
│   ├── compression.py          # NoCompression, FIFO, TokenPerplexity, Retrieval
│   └── acon.py                 # ACON (faithful implementation, Kang et al. 2025)
│
├── benchmarks/
│   ├── appworld_runner.py      # AppWorld task runner
│   ├── officebench_runner.py   # OfficeBench task runner
│   ├── multiobjqa_runner.py    # Multi-objective QA task runner
│   ├── mcp_runner.py           # Unified MCP runner (all benchmarks)
│   └── metrics.py              # 5 metrics including novel Causal Recall
│
├── experiments/
│   └── run_experiment.py       # Main experiment runner (all experiments)
│
├── plotting/
│   └── plot_results.py         # Generate all plots from result CSVs
│
├── k8s/                        # Kubernetes job manifests
│   ├── ccp_job_appworld.yaml
│   ├── ccp_job_multiqa.yaml
│   ├── ccp_job_officebench.yaml
│   └── ccp_job_plots.yaml
│
└── tests/
    ├── test_ccp.py
    └── test_new_components.py
```

---

## Baselines

CCP is compared against 5 baselines:

| Method               | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **No Compression**   | Full context, no pruning — upper bound on accuracy                                          |
| **FIFO**             | Discard oldest elements when threshold exceeded                                             |
| **Token Perplexity** | LLMLingua-style, keeps elements by information score                                        |
| **Retrieval-Based**  | Keeps top-k elements most similar to current goal                                           |
| **ACON**             | Current state of the art — offline-optimized guideline-based compression (Kang et al. 2025) |

---

## Benchmarks

| Benchmark              | Tasks | Avg Steps | Role                            |
| ---------------------- | ----- | --------- | ------------------------------- |
| **AppWorld**           | 750   | 15–40     | Primary — 9 apps, 457 APIs      |
| **OfficeBench**        | 500   | 10–25     | Secondary — office productivity |
| **Multi-objective QA** | 200   | 15+       | Tertiary — multi-hop retrieval  |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/amit502/ccp.git
cd ccp
pip install -r requirements.txt
pip install -e . --no-deps
```

### 2. Install AppWorld separately

AppWorld pins `pydantic<2` which conflicts with LangGraph. Install it with `--no-deps`:

```bash
pip install appworld --no-deps
pip install "pydantic==1.10.21" "sqlmodel==0.0.10"
```

### 3. Set your API key

```bash
export OPENAI_API_KEY="your-api-key"

# Optional: if your LLM endpoint is not the default
export OPENAI_BASE_URL="https://your-llm-endpoint/v1"
```

### 4. Set up benchmarks

**AppWorld:**

```bash
appworld download all
appworld server start          # runs on localhost:8000
```

**OfficeBench:**

```bash
git clone https://github.com/zlwangx/OfficeBench.git
cd OfficeBench && pip install -r requirements.txt
python server.py --port 8001   # runs on localhost:8001
export OFFICEBENCH_URL=http://localhost:8001
export OFFICEBENCH_TASKS_DIR=/path/to/OfficeBench/tasks
```

**Multi-objective QA (Natural Questions):**

```bash
# Option A: download once locally (fastest)
wget https://storage.googleapis.com/natural_questions/v1.0/dev/nq-dev-00.jsonl.gz
export MULTIQA_DATA_FILE=/path/to/nq-dev-00.jsonl.gz

# Option B: auto-download from HuggingFace (needs internet at run time)
pip install datasets
```

---

## Running Experiments

All experiments write CSV results to `RESULTS_PATH` (defaults to `./results`):

```bash
export RESULTS_PATH=./results
mkdir -p results
```

### Run everything

```bash
python -m experiments.run_experiment --experiment all
```

### Run one benchmark at a time

```bash
# All 6 methods on AppWorld
python -m experiments.run_experiment --experiment appworld_all

# All 6 methods on Multi-QA
python -m experiments.run_experiment --experiment multiqa_all

# All 6 methods on OfficeBench (requires server running)
python -m experiments.run_experiment --experiment officebench_all
```

### Run one method on one benchmark

```bash
python -m experiments.run_experiment --experiment appworld_ccp
python -m experiments.run_experiment --experiment appworld_fifo
python -m experiments.run_experiment --experiment appworld_acon
python -m experiments.run_experiment --experiment appworld_no_compression
python -m experiments.run_experiment --experiment appworld_retrieval
python -m experiments.run_experiment --experiment appworld_token_perplexity

# same pattern for multiqa_* and officebench_*
```

### Run ablations

```bash
# A1: Threshold sensitivity (vary τ_H, τ_L)
python -m experiments.run_experiment --experiment ablation_threshold

# A2: Scorer faithfulness (binary scorer vs LLM scorer)
python -m experiments.run_experiment --experiment ablation_faithfulness

# A3: MCP structure benefit (heuristics on vs off)
python -m experiments.run_experiment --experiment ablation_mcp_struct

# A4: Online vs offline (CCP vs ACON)
python -m experiments.run_experiment --experiment ablation_online
```

### Run ACON offline optimization

Run this once before the main comparison to get optimized ACON guidelines:

```bash
python -m experiments.run_experiment --experiment acon_optimize --benchmark appworld
```

### Control task count and steps

```bash
python -m experiments.run_experiment --experiment appworld_all --tasks 50 --steps 40
```

---

## Generating Plots

Once you have result CSVs, run the plotting script pointing at the results folder:

```bash
python plotting/plot_results.py \
    --results-dir ./results \
    --output-dir  ./results/plots \
    --format pdf \
    --dpi 300
```

Plots generated:

- Per-benchmark bar charts (all methods × all metrics)
- Cross-benchmark heatmap (method × benchmark)
- Ablation A1 threshold sensitivity curves
- Efficiency frontier scatter (success rate vs token usage)
- Causal Recall bars (CCP novel metric)
- Ablation A2/A3/A4 comparison bars

---

## Running on a Cluster

Four job manifests are provided in `k8s/`. Each is self-contained — it installs all dependencies, downloads data, runs the experiment, and saves CSVs to a persistent volume.

```bash
# Submit each benchmark as a separate job
kubectl apply -f k8s/ccp_job_appworld.yaml
kubectl apply -f k8s/ccp_job_multiqa.yaml
kubectl apply -f k8s/ccp_job_officebench.yaml

# After all three complete, generate plots
kubectl apply -f k8s/ccp_job_plots.yaml
```

To run a single specific experiment on the cluster, edit `k8s/ccp_job_appworld.yaml` and change the `EXPERIMENT` env var to any value from the list above (e.g. `appworld_ccp`), then apply.

Results are saved to `/results/ccp/` on the persistent volume.

---

## Running Tests

```bash
cd ccp
python -m pytest tests/ -v
```

62 tests covering all components.

---

## Environment Variables Reference

| Variable                | Required | Default                        | Description                           |
| ----------------------- | -------- | ------------------------------ | ------------------------------------- |
| `OPENAI_API_KEY`        | Yes      | —                              | API key for LLM endpoint              |
| `OPENAI_BASE_URL`       | No       | `https://llm.nrp-nautilus.io/` | LLM endpoint base URL                 |
| `RESULTS_PATH`          | No       | `./results`                    | Where CSVs are written                |
| `APPWORLD_URL`          | No       | `http://localhost:8000`        | AppWorld server URL                   |
| `OFFICEBENCH_URL`       | No       | `http://localhost:8001`        | OfficeBench server URL                |
| `OFFICEBENCH_TASKS_DIR` | No       | —                              | Path to OfficeBench task JSON files   |
| `MULTIQA_DATA_FILE`     | No       | —                              | Path to NQ JSONL(.gz) file            |
| `MAX_TASKS`             | No       | `50`                           | Tasks per benchmark                   |
| `MAX_STEPS`             | No       | `40`                           | Max agent steps per task              |
| `TAU_HIGH`              | No       | `0.6`                          | CCP active tier threshold             |
| `TAU_LOW`               | No       | `0.3`                          | CCP inert tier threshold              |
| `TOKEN_THRESHOLD`       | No       | `4000`                         | Token count that triggers compression |

---

## References

1. Kang et al. (2025). ACON: Optimizing Context Compression for Long-horizon LLM Agents. arXiv:2510.00615.
2. Trivedi et al. (2024). AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. ACL 2024.
3. Wang et al. (2024). OfficeBench: Benchmarking Language Agents Across Multiple Applications for Office Automation. arXiv:2407.19056.
4. Kwiatkowski et al. (2019). Natural Questions: A Benchmark for Question Answering Research. TACL.
5. Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023.
6. Pearl, J. (2009). Causality: Models, Reasoning and Inference (2nd ed.). Cambridge University Press.
