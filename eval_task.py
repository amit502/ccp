#!/usr/bin/env python3
"""
Standalone AppWorld task evaluator — runs inside the appworld venv.
Usage: python eval_task.py <task_id> <appworld_root>
Outputs ONLY JSON to stdout: {"success": true/false, "pass_count": N, "num_tests": N}
All other output goes to stderr.
"""
import sys
import json
import os

# Redirect all prints/warnings to stderr so stdout stays clean JSON
import warnings
warnings.filterwarnings("ignore")

task_id       = sys.argv[1]
appworld_root = sys.argv[2]

try:
    # from appworld.common.path_store import path_store
    # # path_store.set_root(appworld_root)
    # path_store.root = appworld_root
    from appworld.common.path_store import path_store, PathStore

    if hasattr(path_store, "set_root"):
        path_store.set_root(appworld_root)
    elif hasattr(path_store, "root"):
        path_store.root = appworld_root
    else:
        path_store = PathStore(appworld_root)

    # Suppress any stdout from evaluator
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    from appworld.evaluator import evaluate_task
    result = evaluate_task(task_id=task_id, suppress_errors=True, save_report=False)

    sys.stdout = old_stdout

    print(json.dumps({
        "success":    bool(result.success),
        "pass_count": int(result.pass_count),
        "num_tests":  int(result.num_tests),
    }))

except Exception as e:
    sys.stdout = sys.__stdout__  # restore in case of error
    print(json.dumps({"success": False, "error": str(e)}))
    print(f"[eval_task] error: {e}", file=sys.stderr)
