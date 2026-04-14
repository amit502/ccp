#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, ...}  when ready
  Reads "save" from stdin → calls save_remote_dbs() → prints {"saved":true}
  Reads EOF → exits

Args: task_id appworld_root apis_url [experiment_name]
"""
import sys, json, os, io
from pathlib import Path

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
apis_url        = sys.argv[3]
experiment_name = sys.argv[4] if len(sys.argv) > 4 else "ccp"

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld

    world = AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        remote_apis_url=apis_url,
        load_ground_truth=False,
    )

    # Expose available attrs for debugging (goes to runner log via stderr)
    _all_attrs = [a for a in dir(world) if not a.startswith("__")]
    _save_attrs = [a for a in _all_attrs
                   if any(k in a.lower() for k in ("save","close","output","path","db"))]
    print(f"[seed_task] save_attrs: {_save_attrs}", file=sys.stderr, flush=True)

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
            # The agent made changes via REST calls to the APIs server.
            # save_remote_dbs diffs the CURRENT in-memory server state against
            # the task's INITIAL on-disk DB → gives exactly what the agent changed,
            # in the JSON model-record format that evaluate_task expects.
            initial_dbs = str(Path(appworld_root) / "data" / "tasks" / task_id / "dbs")
            out_dbs = str(
                Path(appworld_root) / "experiments" / "outputs"
                / experiment_name / "tasks" / task_id / "dbs"
            )
            Path(out_dbs).mkdir(parents=True, exist_ok=True)

            print(f"[seed_task] saving: from={initial_dbs} to={out_dbs}", file=sys.stderr, flush=True)

            world.save_remote_dbs(
                out_dbs,
                format="changes",
                from_db_home_path=initial_dbs,
            )

            # Report sizes of saved files for diagnostics
            saved_files = list(Path(out_dbs).glob("*.jsonl"))
            sizes = {f.name: f.stat().st_size for f in saved_files}
            venmo_bytes = sizes.get("venmo.jsonl", 0)
            nonzero = {k: v for k, v in sizes.items() if v > 0}
            print(f"[seed_task] saved {len(saved_files)} files, "
                  f"nonzero={list(nonzero.keys())}, venmo={venmo_bytes}B",
                  file=sys.stderr, flush=True)

            print(json.dumps({
                "saved":       True,
                "method":      "save_remote_dbs(changes,initial_dbs)",
                "exp":         experiment_name,
                "venmo_bytes": venmo_bytes,
                "nonzero":     len(nonzero),
            }), flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Fallback: try world.close() (works if world tracked the changes)
            try:
                _buf = io.StringIO()
                _old = sys.stdout
                sys.stdout = _buf
                try:
                    world.close()
                finally:
                    sys.stdout = _old
                print(json.dumps({
                    "saved":  True,
                    "method": "world.close() [fallback]",
                    "exp":    experiment_name,
                }), flush=True)
            except Exception as e2:
                print(json.dumps({"saved": False, "error": str(e), "error2": str(e2)}), flush=True)

        # After saving, exit so a fresh subprocess can seed the next task.
        break

    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
