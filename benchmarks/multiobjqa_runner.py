"""
benchmarks/multiobjqa_runner.py

Multi-objective QA benchmark integration for CCP evaluation.

Based on Natural Questions (Kwiatkowski et al., 2019) adapted for multi-hop
agentic evaluation, as used in the ACON paper (tertiary benchmark).

The multi-objective QA benchmark tests agents on tasks that require:
- Retrieving multiple pieces of information from different sources
- Combining them to answer compound questions
- Maintaining context across 15+ retrieval steps

This format specifically stresses context compression methods because:
  1. Each retrieval step adds new (often verbose) observations
  2. Early retrieved facts are needed to answer the final compound question
  3. Methods that discard old context (FIFO) fail catastrophically

==========================================================================
DATA LOADING — THREE MODES (in priority order)
==========================================================================

MODE 1 — Local NQ file (REAL, fastest, no internet needed at run time):
    # Download once from Google:
    wget https://storage.googleapis.com/natural_questions/v1.0/dev/nq-dev-00.jsonl.gz
    export MULTIQA_DATA_FILE=/path/to/nq-dev-00.jsonl.gz
    # Then run as normal — file is read directly, no HF Hub needed.

MODE 2 — HuggingFace Hub (REAL, requires internet + pip install datasets):
    pip install datasets
    # No env var needed — automatic if MULTIQA_DATA_FILE is not set.
    # Uses google-research-datasets/natural_questions from HF Hub.

MODE 3 — Built-in mock questions (always works, no setup):
    # Used automatically when modes 1 and 2 are unavailable.
    # 40 hand-crafted questions covering a range of factual topics.
    # Sufficient for testing the compression pipeline end-to-end.

==========================================================================
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..benchmarks.appworld_runner import TaskResult

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[MultiObjQA] HuggingFace datasets not installed — using mock tasks.")


# ---------------------------------------------------------------------------
# Multi-hop question builder
# ---------------------------------------------------------------------------

def _build_multihop_goal(questions: List[str]) -> str:
    """
    Combine N single-hop questions into one compound multi-objective task.
    The agent must retrieve answers to all sub-questions and combine them.
    """
    numbered = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))
    return (
        f"Answer all of the following questions. Use search tools to retrieve "
        f"the required information. Provide a final combined answer.\n\n"
        f"Questions:\n{numbered}"
    )


_TASK_CACHE_FILE = os.environ.get("MULTIQA_TASK_CACHE_FILE", "")


def _load_nq_tasks(max_tasks: int, hops: int = 3) -> List[Any]:
    """
    Load Natural Questions and group them into multi-hop tasks.
    Each task = `hops` NQ questions combined into one compound question.

    Loading priority:
    0. Task cache file at MULTIQA_TASK_CACHE_FILE — reuses exact tasks from
       a prior run, guaranteeing consistency across methods and ACON opt.
    1. Local JSONL file at MULTIQA_DATA_FILE env var (fastest, no network)
    2. HuggingFace Hub (requires `pip install datasets` + internet)
    3. Built-in mock questions (always works)
    """
    from types import SimpleNamespace

    # --- Priority 0: task cache (same tasks every run) ---
    if _TASK_CACHE_FILE and Path(_TASK_CACHE_FILE).exists():
        try:
            cached = json.loads(Path(_TASK_CACHE_FILE).read_text())
            tasks = [
                SimpleNamespace(
                    id=t["id"], goal=t["goal"],
                    questions=t["questions"], answers=t["answers"],
                )
                for t in cached[:max_tasks]
            ]
            # Validate: if ALL answers are empty the cache was built before the
            # NQ answer-extraction fix was applied — discard and rebuild.
            all_answers = [a for t in tasks for a in t.answers]
            has_answers = any(a and str(a).strip() for a in all_answers)
            if tasks and has_answers:
                print(f"[MultiObjQA] Loaded {len(tasks)} tasks from cache {_TASK_CACHE_FILE}")
                return tasks
            print(f"[MultiObjQA] Cache has empty answers — rebuilding from source")
            Path(_TASK_CACHE_FILE).unlink(missing_ok=True)
        except Exception as e:
            print(f"[MultiObjQA] Cache load failed ({e}) — re-loading from source")

    # qa_pairs: list of (question, answer_or_empty_string)
    qa_pairs: List[tuple] = []

    # --- Priority 1: local file ---
    local_path = os.environ.get("MULTIQA_DATA_FILE", "")
    if local_path and Path(local_path).exists():
        qa_pairs = _load_nq_qa_from_file(local_path, max_tasks * hops)

    # --- Priority 2: HuggingFace Hub ---
    if not qa_pairs and HF_AVAILABLE:
        try:
            ds = load_dataset(
                "google-research-datasets/natural_questions",
                split="validation",
                streaming=True,
            )
            for item in ds:
                q = (item.get("question") or {}).get("text") or item.get("question", "")
                if not (isinstance(q, str) and q.strip()):
                    continue
                # Try to extract short answer — NQ HF schema varies by version
                ans = ""
                try:
                    ann = (item.get("annotations") or {})
                    sa_list = ann.get("short_answers", [])
                    if sa_list and sa_list[0]:
                        first = sa_list[0]
                        if isinstance(first, list):
                            ans = first[0] if first else ""
                        elif isinstance(first, dict):
                            # HF NQ format: {"start_token": [...], "text": ["answer"]}
                            texts = first.get("text", [])
                            ans = texts[0] if texts else ""
                        else:
                            ans = str(first) if first else ""
                except Exception:
                    pass
                qa_pairs.append((q.strip(), ans))
                if len(qa_pairs) >= max_tasks * hops:
                    break
            if qa_pairs:
                print(f"[MultiObjQA] Loaded {len(qa_pairs)} QA pairs from HuggingFace NQ.")
        except Exception as e:
            print(f"[MultiObjQA] HuggingFace load failed: {e}")

    # --- Priority 3: built-in mock ---
    if not qa_pairs:
        print("[MultiObjQA] Using built-in question bank (no external data needed).")
        return _mock_tasks(max_tasks)

    # Group into multi-hop compound tasks
    tasks = []
    for i in range(0, len(qa_pairs) - hops + 1, hops):
        group   = qa_pairs[i: i + hops]
        qlist   = [q for q, _ in group]
        alist   = [a for _, a in group]  # may be "" for items without short answers
        goal    = _build_multihop_goal(qlist)
        tasks.append(SimpleNamespace(
            id=f"moqa_{i // hops:04d}",
            goal=goal,
            questions=qlist,
            answers=alist,   # store expected answers alongside questions
        ))
        if len(tasks) >= max_tasks:
            break

    # Save to cache so subsequent runs (including per-method runs and ACON opt)
    # use exactly the same tasks for consistency.
    if _TASK_CACHE_FILE and tasks:
        try:
            cache_path = Path(_TASK_CACHE_FILE)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                cache_path.write_text(json.dumps([
                    {"id": t.id, "goal": t.goal,
                     "questions": t.questions, "answers": t.answers}
                    for t in tasks
                ]))
                print(f"[MultiObjQA] Saved {len(tasks)} tasks to cache {_TASK_CACHE_FILE}")
        except Exception as e:
            print(f"[MultiObjQA] Cache save failed: {e}")

    return tasks


def _load_nq_from_file(path: str, max_questions: int) -> List[str]:
    """Load questions only (backwards compat). Delegates to _load_nq_qa_from_file."""
    return [q for q, _ in _load_nq_qa_from_file(path, max_questions)]


def _load_nq_qa_from_file(path: str, max_items: int) -> List[tuple]:
    """
    Load (question, answer) pairs from a local NQ JSONL or JSONL.GZ file.

    NQ dev set format (each line is a JSON object):
    {"question_text": "who sang ...", "annotations": [...], ...}

    Download from:
    https://storage.googleapis.com/natural_questions/v1.0/dev/nq-dev-00.jsonl.gz
    """
    import gzip

    pairs = []
    opener = gzip.open if path.endswith(".gz") else open

    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q = item.get("question_text") or item.get("question", "")
                    if not q:
                        continue
                    # Extract first short answer from NQ annotations
                    ans = ""
                    for ann in item.get("annotations", []):
                        sa = ann.get("short_answers", [])
                        if sa:
                            start  = sa[0].get("start_token", 0)
                            end    = sa[0].get("end_token", 0)
                            tokens = item.get("document", {}).get("tokens", [])
                            if tokens and end > start:
                                ans = " ".join(
                                    t["token"] for t in tokens[start:end]
                                    if not t.get("is_html", False)
                                )
                            break
                    pairs.append((q.strip(), ans))
                    if len(pairs) >= max_items:
                        break
                except json.JSONDecodeError:
                    continue
        print(f"[MultiObjQA] Loaded {len(pairs)} QA pairs from {path}")
    except Exception as e:
        print(f"[MultiObjQA] File load error ({path}): {e}")

    return pairs


def _mock_tasks(max_tasks: int) -> List[Any]:
    from types import SimpleNamespace

    mock_questions = [
        [
            "Who was the first president of the United States?",
            "In what year was the Eiffel Tower completed?",
            "What is the capital city of Australia?",
        ],
        [
            "Who wrote the novel '1984'?",
            "What is the chemical symbol for gold?",
            "In what country was the printing press invented?",
        ],
        [
            "What is the largest planet in our solar system?",
            "Who painted the Mona Lisa?",
            "What year did World War II end?",
        ],
        [
            "What is the speed of light in km/s?",
            "Who discovered penicillin?",
            "What is the longest river in the world?",
        ],
        [
            "What is the currency of Japan?",
            "How many bones are in the adult human body?",
            "What is the smallest country in the world?",
        ],
    ]

    # Pre-computed answers matching the mock KB
    mock_answers = [
        ["George Washington", "1889", "Canberra"],
        ["George Orwell", "Au", "Germany (Johannes Gutenberg)"],
        ["Jupiter", "Leonardo da Vinci", "1945"],
        ["299,792 km/s", "Alexander Fleming (1928)", "The Nile River"],
        ["Japanese Yen (JPY)", "206", "Vatican City"],
    ]

    tasks = []
    for i, (qs, ans) in enumerate(
        zip(
            mock_questions * (max_tasks // len(mock_questions) + 1),
            mock_answers   * (max_tasks // len(mock_answers)   + 1),
        )
    ):
        goal = _build_multihop_goal(qs)
        tasks.append(SimpleNamespace(id=f"moqa_{i:04d}", goal=goal,
                                     questions=qs, answers=ans))
        if len(tasks) >= max_tasks:
            break
    return tasks


# ---------------------------------------------------------------------------
# Mock retrieval tools
# ---------------------------------------------------------------------------

_MOCK_KB: Dict[str, str] = {
    "first president united states": "George Washington",
    "eiffel tower completed":         "1889",
    "capital australia":              "Canberra",
    "1984 novel":                     "George Orwell",
    "chemical symbol gold":           "Au",
    "printing press invented":        "Germany (Johannes Gutenberg)",
    "largest planet solar system":    "Jupiter",
    "mona lisa painted":              "Leonardo da Vinci",
    "world war ii end":               "1945",
    "speed of light km":              "299,792 km/s",
    "penicillin discovered":          "Alexander Fleming (1928)",
    "longest river world":            "The Nile River",
    "currency japan":                 "Japanese Yen (JPY)",
    "bones adult human body":         "206",
    "smallest country world":         "Vatican City",
}


def _fuzzy_lookup(query: str) -> str:
    """Simple keyword-overlap retrieval from mock KB."""
    q_words = set(query.lower().split())
    best_score = 0
    best_ans   = "Information not found."
    for key, val in _MOCK_KB.items():
        k_words = set(key.split())
        overlap  = len(q_words & k_words)
        if overlap > best_score:
            best_score = overlap
            best_ans   = val
    return best_ans


_MOQA_TOOLS_HINT = (
    "\n\nAvailable tools — use EXACTLY these names:\n"
    "  search__query(query)            — search for information about any topic\n"
    "  kb__lookup(entity, attribute)   — look up a specific fact about an entity\n"
    "  web__fetch(url)                 — fetch content from a URL\n"
    "Call tools to retrieve each answer, then respond with action=finish."
)


def _register_moqa_tools(registry: Dict[str, Any]) -> None:
    """Register retrieval tools for the multi-objective QA setting."""

    def search(query: str = "", **kw) -> Dict:
        # Return a plausible-looking search result page
        answer = _fuzzy_lookup(query)
        return {
            "query":   query,
            "results": [
                {"title": f"Search result for: {query}", "snippet": answer},
                {"title": "Wikipedia",
                 "snippet": f"According to sources, the answer is: {answer}"},
                {"title": "Encyclopedia entry",
                 "snippet": f"Historical records indicate: {answer}"},
            ],
        }

    def fetch_page(url: str = "", **kw) -> str:
        # Simulate fetching a web page
        return (
            f"Page content for {url}\n\n"
            "This page contains factual information about the requested topic. "
            "Based on available records, the relevant information is as follows. "
            "The answer can be found in the third paragraph of this document."
        )

    def lookup_fact(entity: str = "", attribute: str = "", **kw) -> Dict:
        query = f"{entity} {attribute}".lower()
        answer = _fuzzy_lookup(query)
        return {"entity": entity, "attribute": attribute, "value": answer}

    registry.update({
        "search__query":  search,
        "web__fetch":     fetch_page,
        "kb__lookup":     lookup_fact,
    })


def _score_moqa_answer(final_answer: str, task: Any) -> float:
    """
    Score the agent's final answer by checking how many sub-questions
    were answered (approximate — checks for known answer strings).
    Returns a score in [0, 1].

    If task.answers contains explicit expected answers (from NQ data), use those.
    Otherwise fall back to fuzzy lookup against mock KB.
    """
    if not final_answer:
        return 0.0

    questions = getattr(task, "questions", [])
    answers   = getattr(task, "answers",   [])
    total     = len(questions) or 1
    answered  = 0

    for i, q in enumerate(questions):
        # Use explicit answer if available and non-empty
        explicit = answers[i] if i < len(answers) else ""
        if explicit:
            expected = explicit
        else:
            # Fall back: derive expected answer from mock KB using last 3 words
            key_words = set(q.lower().replace("?", "").split()[-3:])
            expected  = _fuzzy_lookup(" ".join(key_words))

        if expected.lower() in final_answer.lower():
            answered += 1

    return answered / max(total, 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MultiObjQARunner:
    """
    Runs CCP (or a baseline) against Multi-objective QA tasks.
    Same interface as AppWorldRunner for drop-in use in run_experiment.py.
    """

    def __init__(
        self,
        max_tasks: int = 50,
        max_steps: int = 20,
        n_hops:    int = 3,    # Questions per compound task
    ):
        self.max_tasks = max_tasks
        self.max_steps = max_steps
        self.n_hops    = n_hops

    def evaluate(
        self,
        manager_factory: Callable,
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        from ..agent import _TOOL_REGISTRY, agent_think, execute_tool

        tasks   = _load_nq_tasks(self.max_tasks, hops=self.n_hops)
        results = []

        for i, task in enumerate(tasks):
            if verbose:
                print(f"\n[MultiObjQA/{method_name}] Task {i+1}/{len(tasks)}: "
                      f"{task.goal[:60]}...")

            manager = manager_factory()
            # Append tool names to goal so the agent knows exactly which tools
            # to call — without this, agent.py's prompt never lists them and the
            # LLM guesses wrong names, getting "not registered" errors every step.
            manager.set_goal(task.goal + _MOQA_TOOLS_HINT)
            _TOOL_REGISTRY.clear()
            _register_moqa_tools(_TOOL_REGISTRY)

            state: Dict[str, Any] = {
                "goal":         task.goal,
                "step":         0,
                "max_steps":    self.max_steps,
                "done":         False,
                "final_answer": None,
                "ccp_manager":  manager,
            }

            peak_tokens  = 0
            total_tokens = 0
            t0 = time.time()

            while not state["done"] and state["step"] < self.max_steps:
                state = agent_think(state)
                state = execute_tool(state)
                tok   = manager.get_compressed_context().total_tokens()
                peak_tokens   = max(peak_tokens, tok)
                total_tokens += tok

            score   = _score_moqa_answer(state.get("final_answer", ""), task)
            success = score >= 0.67  # At least 2/3 sub-questions answered

            result = TaskResult(
                task_id=task.id,
                goal=task.goal,
                success=success,
                steps=state["step"],
                final_answer=state.get("final_answer"),
                peak_tokens=peak_tokens,
                total_tokens=total_tokens,
                time_elapsed=time.time() - t0,
                ccp_stats=manager.get_stats_log(),
                method=method_name,
            )
            results.append(result)

            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Score={score:.2f} Steps={state['step']} "
                      f"PeakTok={peak_tokens}")

        return results
