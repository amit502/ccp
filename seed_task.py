#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, ...}  when ready
  Reads "save" from stdin → calls world.close() → prints {"saved":true}
  Reads EOF → exits

Args: task_id appworld_root apis_url [experiment_name]
"""
import sys, json, os, io

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
apis_url        = sys.argv[3]
experiment_name = sys.argv[4] if len(sys.argv) > 4 else "ccp"

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld

    # Pass experiment_name so world.close() saves to the correct folder.
    # NOTE: do NOT access world.output_db_home_path — that attr may not exist
    # in this version of AppWorld; close() computes the path internally.
    world = AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        remote_apis_url=apis_url,
        load_ground_truth=False,
    )

    # Expose available attrs for debugging (goes to runner log)
    _all_attrs = [a for a in dir(world) if not a.startswith("__")]
    _save_attrs = [a for a in _all_attrs
                   if any(k in a.lower() for k in ("save","close","output","path","db"))]

    print(json.dumps({
        "success":    True,
        "task_id":    task_id,
        "db_path":    world.output_db_home_path_in_memory,
        "save_attrs": _save_attrs,
    }), flush=True)

    # Wait for "save" command
    for line in sys.stdin:
        cmd = line.strip()
        if cmd != "save":
            if cmd == "exit":
                break
            continue

        try:
            # Suppress any stdout AppWorld might emit during close()
            _buf = io.StringIO()
            _old = sys.stdout
            sys.stdout = _buf
            try:
                world.close()
            finally:
                sys.stdout = _old

            print(json.dumps({
                "saved":  True,
                "method": "world.close()",
                "exp":    experiment_name,
            }), flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"saved": False, "error": str(e)}), flush=True)

        # After close() the server DB is reset; exit so a fresh subprocess
        # can seed the next task cleanly.
        break

    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
