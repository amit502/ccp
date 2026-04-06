#!/usr/bin/env python3
"""
Standalone AppWorld task evaluator.
Run with the appworld venv Python.
Usage: python eval_task.py <task_id> <appworld_root>
Prints JSON: {"success": true/false, "pass_count": N, "num_tests": N}
"""
import sys, json, os

task_id      = sys.argv[1]
appworld_root = sys.argv[2]

# Set root so path_store finds data
from appworld.common.path_store import path_store
path_store.set_root(appworld_root)

from appworld.evaluator import evaluate_task
try:
    result = evaluate_task(task_id=task_id, suppress_errors=True, save_report=False)
    print(json.dumps({
        "success":    result.success,
        "pass_count": result.pass_count,
        "num_tests":  result.num_tests,
    }))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
