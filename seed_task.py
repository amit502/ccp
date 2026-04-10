#!/usr/bin/env python3
"""
Seed a task using AppWorld's Python library.
Run in the appworld venv — properly initializes task DBs and JWT validation.

Usage: python seed_task.py <task_id> <appworld_root> <apis_url>
Prints JSON: {"success": true/false}
"""
import sys, json, os

task_id      = sys.argv[1]
appworld_root = sys.argv[2]
apis_url     = sys.argv[3]  # http://localhost:8000

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld

    # Initialize with remote APIs server — this properly seeds the task DBs
    # and sets up JWT validation against the correct frozen datetime
    world = AppWorld(
        task_id=task_id,
        remote_apis_url=apis_url,
    )

    print(json.dumps({"success": True, "task_id": task_id}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
    print(f"[seed_task] error: {e}", file=sys.stderr)
