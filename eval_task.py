#!/usr/bin/env python3
"""
Standalone AppWorld task evaluator — runs inside the appworld venv.
Usage: python eval_task.py <task_id> <appworld_root> [experiment_name]
Reads task output DBs from:
  {appworld_root}/experiments/outputs/{experiment_name}/tasks/{task_id}/dbs/
Outputs ONLY JSON to stdout.
"""
import sys, json, os, io

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
experiment_name = sys.argv[3] if len(sys.argv) > 3 else "ccp"

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    # Suppress stdout from evaluator internals
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    from appworld.evaluator import evaluate_task
    result = evaluate_task(
        task_id=task_id,
        experiment_name=experiment_name,
        suppress_errors=True,
        save_report=False,
    )

    sys.stdout = old_stdout

    print(json.dumps({
        "success":    bool(result.success),
        "pass_count": int(result.pass_count),
        "num_tests":  int(result.num_tests),
    }))

except Exception as e:
    try:
        sys.stdout = sys.__stdout__
    except Exception:
        pass
    print(json.dumps({"success": False, "error": str(e)}))
    print(f"[eval_task] error: {e}", file=sys.stderr)
