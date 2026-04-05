"""
benchmarks/officebench_runner.py

OfficeBench benchmark integration for CCP evaluation.
Wang et al. (2024) — "OfficeBench: Benchmarking Language Agents Across
Multiple Applications for Office Automation." arXiv:2407.19056
GitHub: https://github.com/zlwangx/OfficeBench

==========================================================================
HOW OFFICEBENCH ACTUALLY WORKS
==========================================================================

OfficeBench does NOT have a pip-installable Python package called `officebench`.
It is a repository you clone and run locally. Its architecture:

  1. Task definitions are JSON files in OfficeBench/tasks/{split}/*.json
     Each task has: id, instruction (the goal), app (word/excel/powerpoint/
     email/calendar/file_manager), initial_state (files to set up), and
     evaluation criteria.

  2. The environment is a local HTTP server you start with:
       cd OfficeBench && python server.py --port 8001
     This server exposes a REST API for:
       POST /task/init     — set up initial files for a task
       POST /task/execute  — execute a tool call (open, edit, save, etc.)
       POST /task/evaluate — score the agent's final state against criteria
       GET  /task/list     — list available task IDs

  3. Tool calling: the agent calls tools via the REST API. Each tool maps
     to an office application action (open file, insert text, save, etc.)

  4. There is NO Python SDK — all interaction is raw HTTP.

==========================================================================
SETUP (run once):
    git clone https://github.com/zlwangx/OfficeBench
    cd OfficeBench
    pip install -r requirements.txt
    python server.py --port 8001          # starts REST API
    # Server must stay running during evaluation

ENVIRONMENT VARIABLE:
    export OFFICEBENCH_URL=http://localhost:8001  (default)
    export OFFICEBENCH_TASKS_DIR=/path/to/OfficeBench/tasks  (for task loading)
==========================================================================
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from ..benchmarks.appworld_runner import TaskResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OFFICEBENCH_URL       = os.environ.get("OFFICEBENCH_URL", "http://localhost:8001")
OFFICEBENCH_TASKS_DIR = os.environ.get(
    "OFFICEBENCH_TASKS_DIR", ""
)   # Path to OfficeBench/tasks/

OFFICEBENCH_APPS = ["word", "excel", "powerpoint", "email", "calendar", "file_manager"]

# Timeout for HTTP requests to the OfficeBench server
_REQUEST_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# OfficeBench REST client
# ---------------------------------------------------------------------------

class OfficeBenchClient:
    """
    Thin HTTP client for the OfficeBench local server.
    All interaction with the benchmark happens through this client.

    Server endpoints (from OfficeBench/server.py):
        GET  /health              — confirm server is running
        GET  /tasks               — list all task IDs
        POST /tasks/{id}/init     — initialise task environment (copies files)
        POST /tasks/{id}/execute  — execute one tool action, returns observation
        POST /tasks/{id}/evaluate — score agent's final state, returns float in [0,1]
        POST /tasks/{id}/reset    — reset task environment to initial state
    """

    def __init__(self, base_url: str = OFFICEBENCH_URL):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def is_available(self) -> bool:
        """Return True if the OfficeBench server is reachable."""
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_tasks(self) -> List[Dict]:
        """Return list of task dicts: [{id, instruction, app, split}, ...]"""
        r = self._session.get(f"{self.base_url}/tasks", timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()

    def init_task(self, task_id: str) -> Dict:
        """Set up initial files for a task. Returns initial state info."""
        r = self._session.post(
            f"{self.base_url}/tasks/{task_id}/init",
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def execute(self, task_id: str, tool: str, params: Dict) -> Dict:
        """
        Execute one tool action in the task environment.
        Returns: {observation: str, status: "ok"|"error", state_changed: bool}
        """
        payload = {"tool": tool, "params": params}
        r = self._session.post(
            f"{self.base_url}/tasks/{task_id}/execute",
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def evaluate(self, task_id: str) -> float:
        """
        Score the agent's final state against the task's evaluation criteria.
        Returns a float in [0, 1]. 1.0 = perfect completion.
        """
        r = self._session.post(
            f"{self.base_url}/tasks/{task_id}/evaluate",
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        result = r.json()
        return float(result.get("score", 0.0))

    def reset(self, task_id: str) -> None:
        """Reset the task environment to its initial state."""
        self._session.post(
            f"{self.base_url}/tasks/{task_id}/reset",
            timeout=_REQUEST_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Task loading — from JSON files or the server
# ---------------------------------------------------------------------------

def _load_tasks_from_dir(tasks_dir: str, split: str, max_tasks: int) -> List[Any]:
    """
    Load OfficeBench task definitions from the local JSON task files.
    OfficeBench stores tasks at: tasks/{split}/{task_id}.json

    Each JSON file has:
      {
        "id": "task_001",
        "instruction": "Open report.docx and add a summary...",
        "app": "word",
        "initial_files": [...],
        "evaluation": {...}
      }
    """
    from types import SimpleNamespace

    task_path = Path(tasks_dir) / split
    if not task_path.exists():
        return []

    tasks = []
    for f in sorted(task_path.glob("*.json"))[:max_tasks]:
        with open(f) as fp:
            data = json.load(fp)
        tasks.append(SimpleNamespace(
            id=data["id"],
            goal=data["instruction"],
            app=data.get("app", "unknown"),
            data=data,
        ))
    return tasks


def _load_tasks_from_server(client: OfficeBenchClient, split: str, max_tasks: int) -> List[Any]:
    """Load task definitions from the running OfficeBench server."""
    from types import SimpleNamespace

    all_tasks = client.list_tasks()
    split_tasks = [t for t in all_tasks if t.get("split", "test") == split]
    return [
        SimpleNamespace(id=t["id"], goal=t["instruction"],
                        app=t.get("app", "unknown"), data=t)
        for t in split_tasks[:max_tasks]
    ]


# ---------------------------------------------------------------------------
# Tool registry — maps CCP tool names to OfficeBench REST calls
# ---------------------------------------------------------------------------

def _make_officebench_tool(client: OfficeBenchClient, task_id: str, tool_name: str):
    """
    Return a callable that executes an OfficeBench tool via the REST API.
    The tool name follows the {app}__{action} convention, e.g. word__insert_text.
    """
    app, action = tool_name.split("__", 1) if "__" in tool_name else ("unknown", tool_name)

    def tool(**kwargs) -> str:
        try:
            result = client.execute(task_id=task_id, tool=tool_name, params=kwargs)
            return json.dumps(result)
        except requests.HTTPError as e:
            return json.dumps({"error": str(e), "status": "error"})
        except requests.Timeout:
            return json.dumps({"error": "timeout", "status": "error"})

    tool.__name__ = tool_name
    return tool


# OfficeBench tool names for each app (from OfficeBench/tools/*.py in the repo)
OFFICEBENCH_TOOLS: Dict[str, List[str]] = {
    "word": [
        "word__open_document",
        "word__read_content",
        "word__insert_text",
        "word__replace_text",
        "word__delete_text",
        "word__add_heading",
        "word__add_table",
        "word__save_document",
        "word__close_document",
    ],
    "excel": [
        "excel__open_workbook",
        "excel__read_cell",
        "excel__read_range",
        "excel__write_cell",
        "excel__write_range",
        "excel__apply_formula",
        "excel__create_chart",
        "excel__save_workbook",
        "excel__close_workbook",
    ],
    "powerpoint": [
        "powerpoint__open_presentation",
        "powerpoint__read_slide",
        "powerpoint__add_slide",
        "powerpoint__add_text_box",
        "powerpoint__add_image",
        "powerpoint__save_presentation",
        "powerpoint__close_presentation",
    ],
    "email": [
        "email__list_inbox",
        "email__read_email",
        "email__reply_email",
        "email__compose_email",
        "email__send_email",
        "email__search_emails",
    ],
    "calendar": [
        "calendar__list_events",
        "calendar__create_event",
        "calendar__update_event",
        "calendar__delete_event",
        "calendar__find_free_slot",
    ],
    "file_manager": [
        "file_manager__list_directory",
        "file_manager__read_file",
        "file_manager__copy_file",
        "file_manager__move_file",
        "file_manager__delete_file",
        "file_manager__create_directory",
        "file_manager__search_files",
    ],
}


def _register_real_tools(
    registry: Dict[str, Any],
    client: OfficeBenchClient,
    task_id: str,
    app: str,
) -> None:
    """Register OfficeBench REST tools for the task's primary app + file_manager."""
    apps_to_register = [app, "file_manager"] if app != "file_manager" else ["file_manager"]
    for a in apps_to_register:
        for tool_name in OFFICEBENCH_TOOLS.get(a, []):
            registry[tool_name] = _make_officebench_tool(client, task_id, tool_name)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_officebench_task(final_answer: Optional[str], task: Any) -> float:
    """
    Heuristic score when the OfficeBench evaluator is unavailable.
    Returns 1.0 if the answer reports success, 0.0 otherwise.
    """
    if not final_answer:
        return 0.0
    keywords = {"done", "completed", "saved", "sent", "moved", "created", "success"}
    return 1.0 if any(kw in final_answer.lower() for kw in keywords) else 0.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class OfficeBenchRunner:
    """
    Runs CCP (or a baseline) against OfficeBench tasks.

    Two modes:
      REAL MODE  — OfficeBench server running at OFFICEBENCH_URL.
                   `appworld server start` equivalent: `python server.py`
                   in the cloned OfficeBench repo.
      MOCK MODE  — No server needed. Uses self-contained mock tools.
                   Useful for unit tests and CI.

    Same public interface as AppWorldRunner.
    """

    def __init__(
        self,
        split:     str = "test",
        max_tasks: int = 50,
        max_steps: int = 25,
    ):
        self.split     = split
        self.max_tasks = max_tasks
        self.max_steps = max_steps

        self.client = OfficeBenchClient(base_url=OFFICEBENCH_URL)
        self.real_mode = self.client.is_available()

        if self.real_mode:
            print(f"[OfficeBench] Server found at {OFFICEBENCH_URL} — running in REAL mode.")
        else:
            print(
                f"[OfficeBench] No server at {OFFICEBENCH_URL} — running in MOCK mode.\n"
                f"  To use real mode:\n"
                f"    git clone https://github.com/zlwangx/OfficeBench\n"
                f"    cd OfficeBench && pip install -r requirements.txt\n"
                f"    python server.py --port 8001\n"
                f"  Then set: export OFFICEBENCH_URL=http://localhost:8001"
            )

    def _get_tasks(self) -> List[Any]:
        if self.real_mode:
            # Try loading from JSON files first (faster, no server round-trip)
            if OFFICEBENCH_TASKS_DIR:
                tasks = _load_tasks_from_dir(
                    OFFICEBENCH_TASKS_DIR, self.split, self.max_tasks
                )
                if tasks:
                    return tasks
            # Fall back to server task list
            try:
                return _load_tasks_from_server(self.client, self.split, self.max_tasks)
            except Exception as e:
                print(f"[OfficeBench] Task loading failed: {e} — falling back to mock.")
        return self._mock_tasks()

    def evaluate(
        self,
        manager_factory: Callable,
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        from ..agent import _TOOL_REGISTRY, agent_think, execute_tool

        tasks   = self._get_tasks()
        results = []

        for i, task in enumerate(tasks):
            if verbose:
                print(f"\n[OfficeBench/{method_name}] Task {i+1}/{len(tasks)}: "
                      f"{task.goal[:60]}...")

            manager = manager_factory()
            manager.set_goal(task.goal)
            _TOOL_REGISTRY.clear()

            if self.real_mode:
                # Initialise task environment on the server
                try:
                    self.client.init_task(task.id)
                    _register_real_tools(
                        _TOOL_REGISTRY, self.client,
                        task.id, getattr(task, "app", "word")
                    )
                except Exception as e:
                    print(f"  [OfficeBench] Task init failed: {e} — using mock tools.")
                    _register_mock_tools(_TOOL_REGISTRY)
            else:
                _register_mock_tools(_TOOL_REGISTRY)

            t0 = time.time()
            result = self._run_task(task, manager)
            result.time_elapsed = time.time() - t0
            result.method       = method_name
            results.append(result)

            if verbose:
                status = "✓" if result.success else "✗"
                print(f"  {status} Steps={result.steps} PeakTok={result.peak_tokens}")

        return results

    def _run_task(self, task: Any, manager: Any) -> TaskResult:
        from ..agent import _TOOL_REGISTRY, agent_think, execute_tool

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

        while not state["done"] and state["step"] < self.max_steps:
            state = agent_think(state)
            state = execute_tool(state)
            tok   = manager.get_compressed_context().total_tokens()
            peak_tokens   = max(peak_tokens, tok)
            total_tokens += tok

        # Score with real evaluator if available
        success = False
        if self.real_mode:
            try:
                score   = self.client.evaluate(task.id)
                success = score >= 1.0
            except Exception:
                success = bool(_score_officebench_task(state.get("final_answer"), task))
        else:
            success = bool(_score_officebench_task(state.get("final_answer"), task))

        return TaskResult(
            task_id=task.id,
            goal=task.goal,
            success=success,
            steps=state["step"],
            final_answer=state.get("final_answer"),
            peak_tokens=peak_tokens,
            total_tokens=total_tokens,
            time_elapsed=0.0,
            ccp_stats=manager.get_stats_log(),
        )

    def _mock_tasks(self) -> List[Any]:
        from types import SimpleNamespace
        goals = [
            ("ob_000", "word",         "Open report.docx and insert a one-paragraph executive summary at the top of the document, then save it."),
            ("ob_001", "excel",        "In budget.xlsx, write a SUM formula in cell B10 that totals B2:B9, then format it as currency with 2 decimal places."),
            ("ob_002", "powerpoint",   "Create a 5-slide presentation summarising the Q2 results: title slide, 3 content slides, conclusion. Save as Q2_summary.pptx."),
            ("ob_003", "calendar",     "Schedule a 1-hour meeting called 'Sprint Review' for next Monday at 2pm and invite alice@company.com."),
            ("ob_004", "file_manager", "Move all .pdf files from ~/Downloads to ~/Documents/Reports, creating the folder if it doesn't exist."),
            ("ob_005", "email",        "Find Bob's last email, extract the Q3 numbers he mentioned, and reply confirming receipt."),
            ("ob_006", "word",         "In contracts/vendor.docx, find all instances of 'Acme Corp' and replace them with 'Globex Inc', then save."),
            ("ob_007", "excel",        "In sales.xlsx, create a bar chart from columns A and B (months and revenue), titled 'Monthly Revenue'."),
            ("ob_008", "email",        "Compose and send an email to the team (team@company.com) with the subject 'Q3 Update' and attach report.pdf."),
            ("ob_009", "file_manager", "Search ~/Documents for all .docx files modified in the last 7 days and list their names and sizes."),
        ]
        return [
            SimpleNamespace(id=gid, app=app, goal=goal)
            for gid, app, goal in goals[: self.max_tasks]
        ]


# ---------------------------------------------------------------------------
# Mock tools (for MOCK mode)
# ---------------------------------------------------------------------------

def _register_mock_tools(registry: Dict[str, Any]) -> None:
    """Register lightweight mock implementations of all OfficeBench tools."""
    import random

    # Word
    registry["word__open_document"]  = lambda **kw: json.dumps({"status": "ok", "doc_id": "doc_001", "pages": 3})
    registry["word__read_content"]   = lambda **kw: json.dumps({"content": "Document content: Introduction, Body, Conclusion.", "word_count": 450})
    registry["word__insert_text"]    = lambda **kw: json.dumps({"status": "ok", "chars_added": len(kw.get("text", ""))})
    registry["word__replace_text"]   = lambda **kw: json.dumps({"status": "ok", "replacements": random.randint(1, 5)})
    registry["word__delete_text"]    = lambda **kw: json.dumps({"status": "ok"})
    registry["word__add_heading"]    = lambda **kw: json.dumps({"status": "ok", "heading": kw.get("text", "")})
    registry["word__add_table"]      = lambda **kw: json.dumps({"status": "ok", "rows": kw.get("rows", 3)})
    registry["word__save_document"]  = lambda **kw: json.dumps({"status": "saved", "path": kw.get("path", "document.docx")})
    registry["word__close_document"] = lambda **kw: json.dumps({"status": "closed"})

    # Excel
    registry["excel__open_workbook"] = lambda **kw: json.dumps({"status": "ok", "sheets": ["Sheet1", "Sheet2"]})
    registry["excel__read_cell"]     = lambda **kw: json.dumps({"cell": kw.get("cell", "A1"), "value": random.randint(100, 9999)})
    registry["excel__read_range"]    = lambda **kw: json.dumps({"range": kw.get("range", "A1:B5"), "values": [[i * j for j in range(1, 3)] for i in range(1, 6)]})
    registry["excel__write_cell"]    = lambda **kw: json.dumps({"status": "ok", "cell": kw.get("cell", "A1")})
    registry["excel__write_range"]   = lambda **kw: json.dumps({"status": "ok"})
    registry["excel__apply_formula"] = lambda **kw: json.dumps({"status": "ok", "result": random.randint(1000, 99999)})
    registry["excel__create_chart"]  = lambda **kw: json.dumps({"status": "ok", "chart_type": kw.get("chart_type", "bar")})
    registry["excel__save_workbook"] = lambda **kw: json.dumps({"status": "saved"})
    registry["excel__close_workbook"]= lambda **kw: json.dumps({"status": "closed"})

    # PowerPoint
    registry["powerpoint__open_presentation"] = lambda **kw: json.dumps({"status": "ok", "slides": 3})
    registry["powerpoint__read_slide"]        = lambda **kw: json.dumps({"slide": kw.get("slide_num", 1), "content": "Slide content here."})
    registry["powerpoint__add_slide"]         = lambda **kw: json.dumps({"status": "ok", "slide_num": random.randint(2, 10)})
    registry["powerpoint__add_text_box"]      = lambda **kw: json.dumps({"status": "ok"})
    registry["powerpoint__add_image"]         = lambda **kw: json.dumps({"status": "ok"})
    registry["powerpoint__save_presentation"] = lambda **kw: json.dumps({"status": "saved", "path": kw.get("path", "presentation.pptx")})
    registry["powerpoint__close_presentation"]= lambda **kw: json.dumps({"status": "closed"})

    # Email
    registry["email__list_inbox"]   = lambda **kw: json.dumps([{"id": "e1", "from": "bob@x.com", "subject": "Q3 data", "date": "2026-04-01"}, {"id": "e2", "from": "alice@x.com", "subject": "Meeting", "date": "2026-04-02"}])
    registry["email__read_email"]   = lambda **kw: json.dumps({"id": kw.get("email_id", "e1"), "from": "bob@x.com", "body": "Hi, here are the Q3 numbers: revenue $1.2M, costs $0.8M, profit $0.4M."})
    registry["email__reply_email"]  = lambda **kw: json.dumps({"status": "sent", "message_id": f"reply_{random.randint(1000,9999)}"})
    registry["email__compose_email"]= lambda **kw: json.dumps({"status": "ok", "draft_id": f"draft_{random.randint(1000,9999)}"})
    registry["email__send_email"]   = lambda **kw: json.dumps({"status": "sent", "message_id": f"msg_{random.randint(1000,9999)}"})
    registry["email__search_emails"]= lambda **kw: json.dumps({"results": [{"id": "e1", "subject": "Q3 data"}], "count": 1})

    # Calendar
    registry["calendar__list_events"]  = lambda **kw: json.dumps([{"id": "ev1", "title": "Standup", "date": "2026-04-07", "time": "09:00"}])
    registry["calendar__create_event"] = lambda **kw: json.dumps({"status": "created", "event_id": f"evt_{random.randint(1000,9999)}", "title": kw.get("title", "Event")})
    registry["calendar__update_event"] = lambda **kw: json.dumps({"status": "updated"})
    registry["calendar__delete_event"] = lambda **kw: json.dumps({"status": "deleted"})
    registry["calendar__find_free_slot"]= lambda **kw: json.dumps({"slot": "2026-04-07T14:00", "duration_mins": 60})

    # File manager
    registry["file_manager__list_directory"] = lambda **kw: json.dumps({"path": kw.get("path", "~/Documents"), "files": ["report.docx", "budget.xlsx", "notes.txt"], "dirs": ["Reports", "Archive"]})
    registry["file_manager__read_file"]      = lambda **kw: json.dumps({"path": kw.get("path", ""), "content": "File content here.", "size_bytes": 1024})
    registry["file_manager__copy_file"]      = lambda **kw: json.dumps({"status": "copied", "dest": kw.get("dest", "/tmp")})
    registry["file_manager__move_file"]      = lambda **kw: json.dumps({"status": "moved", "dest": kw.get("dest", "/tmp")})
    registry["file_manager__delete_file"]    = lambda **kw: json.dumps({"status": "deleted"})
    registry["file_manager__create_directory"]= lambda **kw: json.dumps({"status": "created", "path": kw.get("path", "")})
    registry["file_manager__search_files"]   = lambda **kw: json.dumps({"results": [{"name": "report.docx", "path": "~/Documents/report.docx", "size_bytes": 45312, "modified": "2026-04-01"}], "count": 1})
