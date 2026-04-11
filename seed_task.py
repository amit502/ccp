#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.
Protocol:
  - Prints {"success": true}  when seeding is done
  - Waits on stdin for either:
      "save <output_dir>\n"  → saves task state to disk, prints {"saved": true}
      EOF                    → exits
"""
import sys, json, os

task_id       = sys.argv[1]
appworld_root = sys.argv[2]
apis_url      = sys.argv[3]

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld

    world = AppWorld(
        task_id=task_id,
        remote_apis_url=apis_url,
        load_ground_truth=False,
    )

    print(json.dumps({"success": True, "task_id": task_id}), flush=True)

    # Wait for commands
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("save "):
            output_dir = line[5:].strip()
            try:
                # Use AppWorld's native save — correct DB path guaranteed
                world._save_state(output_dir)
                print(json.dumps({"saved": True, "dir": output_dir}), flush=True)
            except Exception as e:
                print(json.dumps({"saved": False, "error": str(e)}), flush=True)
        elif line == "exit":
            break

    # Exit without calling world.close() — keeps DB state intact until save
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
