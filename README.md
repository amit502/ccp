# Causal Context Pruning (CCP)

**Causal Context Pruning with Adaptive State Distillation for Long-Horizon Agents**

> University of South Dakota — CSC-792-U18 Agentic AI | Spring 2026
> Amit Kumar Patel | Instructor: Dr. Rodrigue Rizk

---

## What is CCP?

Long-horizon agentic systems accumulate tool call history as they execute multi-step tasks. Every API response, error, and intermediate result piles up in the context window, causing token cost explosion and information loss.

CCP addresses this by modelling the agent trajectory as a **causal dependency graph** and removing steps that are causally disconnected from the current decision frontier — no LLM calls, no offline training.

> *A step is dead if nothing downstream uses its output.*

### How it works

**Dead Branch Elimination (DBE):** Runs a backward BFS from the most recent W steps (the live frontier). Any step unreachable from the frontier is marked INERT and dropped entirely.

**Adaptive State Distillation (ASD):** Live ancestor steps (reachable but outside the recency window) are merged into a single compact `_state_` key-value snapshot, replacing N verbose tool records with one message.

**ValueRegistry:** Tracks exact value propagation — when a tool result value appears verbatim as an argument in a later call, a causal edge is created. This is what makes the dependency graph exact rather than heuristic.

CCP is **training-free**, **online**, and **MCP-native** — it runs as middleware between the LangGraph agent and MCP tool servers.

---

## Results (75 tasks per benchmark)

| Method | AppWorld Eff↑ | OfficeBench Eff↑ | MultiQA Eff↑ |
|--------|--------------|-----------------|-------------|
| No Compression | 0.034 | 1.396 | 5.772 |
| FIFO | 0.282 | 6.460 | 3.541 |
| Token Perplexity | 0.174 | **10.977** | 5.457 |
| Retrieval | 0.094 | 1.659 | 5.400 |
| ACON | 0.103 | 1.456 | 1.135 |
| **CCP (ours)** | **0.322** | 7.404 | **5.570** |

Efficiency = Task Success Rate / (Mean Peak Tokens / 1000). CCP achieves the best compression efficiency on AppWorld and MultiQA, and second-best on OfficeBench.

---

## Project Structure

```
ccp/
├── models.py                   # Core data structures
├── context_manager.py          # DBE + ASD compression pipeline
├── llm_client.py               # OpenAI-compatible LLM client
├── mcp_server.py               # AppWorld MCP server (stdio transport)
├── officebench_server.py       # OfficeBench MCP server
├── nq_mcp_server.py            # Natural Questions MCP server
├── mcp_agent.py                # LangGraph agent with MCP + CCP interceptor
│
├── baselines/
│   ├── compression.py          # NoCompression, FIFO, TokenPerplexity, Retrieval
│   └── acon.py                 # ACON (faithful implementation, Kang et al. 2025)
│
├── benchmarks/
│   ├── mcp_runner.py           # Unified MCP runner (all benchmarks)
│   └── metrics.py              # Task success, peak tokens, context dependency, efficiency
│
├── experiments/
│   └── run_experiment.py       # Main experiment runner
│
├── generate_tables.py          # Generate summary CSV + LaTeX table from result folders
├── generate_graphs.py          # Generate paper figures from result folders
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

| Method | Criterion | Offline training? |
|--------|-----------|------------------|
| No Compression | None | No |
| FIFO | Age (recency) | No |
| Token Perplexity | LM surprise score | No |
| Retrieval | Embedding similarity to goal | No |
| ACON | LLM-optimized task guidelines | **Yes** |
| **CCP (ours)** | Causal value-flow reachability | **No** |

---

## Benchmarks

| Benchmark | Tasks run | Avg tokens (no comp.) | Role |
|-----------|-----------|----------------------|------|
| **AppWorld** | 75 | ~2,330 | Primary — API task execution, 9 apps |
| **OfficeBench** | 75 | ~296 | Secondary — document manipulation |
| **MultiQA** | 75 | ~90 | Retrieval-dense QA |

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

AppWorld pins `pydantic<2` which conflicts with LangGraph. Install with `--no-deps`:

```bash
pip install appworld --no-deps
pip install "pydantic==1.10.21" "sqlmodel==0.0.10"
```

### 3. Set your API key

```bash
export OPENAI_API_KEY="your-api-key"

# Optional: custom LLM endpoint
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
git clone https://github.com/zlwang-cs/OfficeBench.git
cd OfficeBench && pip install -r requirements.txt
# Server is started automatically by the experiment runner
export OFFICEBENCH_URL=http://localhost:8001
export OFFICEBENCH_TASKS_DIR=/path/to/OfficeBench/tasks
```

**MultiQA (Natural Questions):**
```bash
# Option A: download once locally
wget https://storage.googleapis.com/natural_questions/v1.0/dev/nq-dev-00.jsonl.gz
export MULTIQA_DATA_FILE=/path/to/nq-dev-00.jsonl.gz

# Option B: auto-download from HuggingFace at run time
pip install datasets
```

---

## Running Experiments

Results are written as CSVs to `RESULTS_PATH` (default: `./results`):

```bash
export RESULTS_PATH=./results
mkdir -p results
```

### Run all benchmarks

```bash
python -m ccp.experiments.run_experiment --experiment all --tasks 75 --steps 25
```

### Run one benchmark

```bash
python -m ccp.experiments.run_experiment --experiment appworld_all  --tasks 75 --steps 25
python -m ccp.experiments.run_experiment --experiment multiqa_all   --tasks 75 --steps 25
python -m ccp.experiments.run_experiment --experiment officebench_all --tasks 75 --steps 25
```

### Run one method on one benchmark

```bash
python -m ccp.experiments.run_experiment --experiment appworld_ccp
python -m ccp.experiments.run_experiment --experiment appworld_fifo
python -m ccp.experiments.run_experiment --experiment appworld_acon
python -m ccp.experiments.run_experiment --experiment appworld_no_compression
python -m ccp.experiments.run_experiment --experiment appworld_retrieval
python -m ccp.experiments.run_experiment --experiment appworld_token_perplexity
# same pattern for multiqa_* and officebench_*
```

---

## Generating Tables and Graphs

Both scripts expect a root folder containing one subfolder per run, each with a CSV:

```
results/
    run_1/
        appworld_results.csv
    run_2/
        ...
```

### Summary table (CSV + LaTeX)

```bash
python generate_tables.py ./results --out ./results/tables
```

Outputs:
- `tables/summary.csv` — flat CSV with mean (± std for multi-run)
- `tables/table.tex` — LaTeX booktabs table, one panel per benchmark

### Paper figures

```bash
python generate_graphs.py ./results --out ./results/plots --format pdf --dpi 300
```

Outputs:
- `plots/eff_comparison.pdf` — Compression Efficiency per method per benchmark
- `plots/frontier.pdf` — Success Rate vs Context Dependency scatter with iso-Eff curves
- `plots/metrics_grid.pdf` — 4-metric overview grid (all methods × all benchmarks)

---

## Running on a Cluster (Kubernetes)

Four self-contained job manifests are provided in `k8s/`. Each installs all dependencies, runs the experiment, and saves CSVs to a persistent volume.

```bash
kubectl apply -f k8s/ccp_job_appworld.yaml
kubectl apply -f k8s/ccp_job_multiqa.yaml
kubectl apply -f k8s/ccp_job_officebench.yaml

# After all three complete, generate plots
kubectl apply -f k8s/ccp_job_plots.yaml
```

Results are saved to `/results/ccp/` on the persistent volume (`reflexion-data-pvc`).

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key for LLM endpoint |
| `OPENAI_BASE_URL` | No | `https://llm.nrp-nautilus.io/` | LLM endpoint base URL |
| `RESULTS_PATH` | No | `./results` | Where CSVs are written |
| `APPWORLD_URL` | No | `http://localhost:8000` | AppWorld server URL |
| `OFFICEBENCH_URL` | No | `http://localhost:8001` | OfficeBench server URL |
| `OFFICEBENCH_TASKS_DIR` | No | — | Path to OfficeBench task JSON files |
| `OFFICEBENCH_TASK_IDS_FILE` | No | — | Path to pinned task ID file (for reproducibility) |
| `MULTIQA_DATA_FILE` | No | — | Path to NQ JSONL(.gz) file |
| `MAX_TASKS` | No | `50` | Tasks per benchmark run |
| `MAX_STEPS` | No | `25` | Max agent steps per task |
| `TOKEN_THRESHOLD` | No | `500` | Token count that triggers CCP on AppWorld |
| `OFFICEBENCH_TOKEN_THRESHOLD` | No | `80` | Token threshold for OfficeBench |
| `MULTIQA_TOKEN_THRESHOLD` | No | `100` | Token threshold for MultiQA |
| `CCP_RECENT_WINDOW` | No | `1` | Number of recent steps kept verbatim (W) |
| `CCP_MAX_ANCESTORS` | No | `1` | Max live ancestors distilled into state (A_max) |

---

## Ablation (AppWorld, 25 tasks)

| Stage | Eff↑ |
|-------|------|
| Base (no CCP components) | 0.074 |
| + Dead Branch Elimination | 0.125 |
| + Temporal Anchor Freshness | 0.149 |
| + Adaptive State Distillation | **0.505** |

Each component contributes independently; ASD is the largest single contributor.

---

## References

1. Kang et al. (2025). ACON: Optimizing Context Compression for Long-horizon LLM Agents. arXiv:2510.00615.
2. Trivedi et al. (2024). AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. ACL 2024.
3. Wang et al. (2024). OfficeBench: Benchmarking Language Agents Across Multiple Applications for Office Automation. arXiv:2407.19056.
4. Kwiatkowski et al. (2019). Natural Questions: A Benchmark for Question Answering Research. TACL.
5. Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023.
6. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
