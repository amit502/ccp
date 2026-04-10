#!/usr/bin/env python3
"""
Seed a task using AppWorld's Python library.
Run in the appworld venv — properly initializes task DBs and JWT validation.

IMPORTANT: This script must NOT call AppWorld.close() or let the AppWorld
object be garbage collected before the agent finishes. AppWorld.close()
calls clear_remote_dbs_cache() which erases the task state from the
APIs server memory.

Usage: python seed_task.py <task_id> <appworld_root> <apis_url>
Prints JSON: {"success": true}  then waits for stdin to close (agent done signal).
"""
import sys, json, os, atexit

task_id       = sys.argv[1]
appworld_root = sys.argv[2]
apis_url      = sys.argv[3]

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld

    # Initialize — seeds task DBs into remote APIs server memory
    # and sets frozen datetime for correct JWT validation
    world = AppWorld(
        task_id=task_id,
        remote_apis_url=apis_url,
        load_ground_truth=False,  # faster, we evaluate separately
    )

    print(json.dumps({"success": True, "task_id": task_id}), flush=True)

    # Keep process alive so AppWorld doesn't close and clear the DB
    # Caller closes stdin when task is done
    sys.stdin.read()

    # Don't call world.close() — let the process exit naturally
    # The APIs server will retain the DB state for save

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
