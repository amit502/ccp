#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, "db_path": ":memory:task_output-{task_id}"}  when ready
  Reads "save <output_dir>" from stdin → saves current server state → prints {"saved":true}
  Reads EOF → exits without calling world.close()
"""
import sys, json, os, requests as _req

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

    # Confirm what DB path server is now using
    try:
        dbs = _req.get(f"{apis_url}/dbs", timeout=5).json()
        server_venmo_path = dbs.get("venmo", world.output_db_home_path_in_memory)
    except Exception:
        server_venmo_path = world.output_db_home_path_in_memory

    print(json.dumps({
        "success": True,
        "task_id": task_id,
        "db_path": world.output_db_home_path_in_memory,
        "server_path": server_venmo_path,
    }), flush=True)

    # Process commands
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("save "):
            if line == "exit":
                break
            continue

        out_dir = line[5:].strip()
        try:
            os.makedirs(out_dir, exist_ok=True)

            # Always use the task-specific in-memory path set at seeding time.
            # Do NOT rely on GET /dbs (server state can drift); world knows
            # the canonical path for this task.
            active_path = world.output_db_home_path_in_memory
            print(f"[seed] saving from task DB path: {active_path}", file=sys.stderr, flush=True)

            app_names = list(world.task.allowed_apps) + ["admin", "supervisor"]

            # format="changes" is what AppWorld's own _save_state() uses and
            # what evaluate_task() / ModelCollection.load() expects.
            # format="full" saves from the on-disk base DB (no agent changes).
            save_remote_dbs(
                remote_apis_url=apis_url,
                from_db_home_path=active_path,
                to_db_home_path=out_dir,
                format="changes",
                app_names=app_names,
                delete_if_exists=True,
                skip_mandatory_apps=False,
                save_model_hashes=True,
            )
            print(json.dumps({
                "saved": True,
                "dir": out_dir,
                "from": active_path,
                "apps": len(app_names),
            }), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"saved": False, "error": str(e)}), flush=True)

    # Exit without world.close() — keeps server state for other purposes
    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
