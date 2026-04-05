# Causal Context Pruning (CCP)
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
5. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
