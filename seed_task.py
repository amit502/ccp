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
    from appworld.apps.api_lib import save_remote_dbs

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
            out_dir = line[5:].strip()
            try:
                os.makedirs(out_dir, exist_ok=True)
                app_names = list(world.task.allowed_apps) + ["admin", "supervisor"]
                # Save with format="full" so evaluator gets complete DB state
                save_remote_dbs(
                    remote_apis_url=apis_url,
                    from_db_home_path=world.output_db_home_path_in_memory,
                    to_db_home_path=out_dir,
                    format="full",
                    app_names=app_names,
                    delete_if_exists=True,
                    skip_mandatory_apps=False,
                )
                print(json.dumps({"saved": True, "dir": out_dir,
                                  "path": world.output_db_home_path_in_memory,
                                  "apps": app_names}), flush=True)
            except Exception as e:
                print(json.dumps({"saved": False, "error": str(e)}), flush=True)
        elif line == "exit":
            break

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
