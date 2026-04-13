#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, "db_path": ":memory:task_output-{task_id}"}  when ready
  Reads "save" from stdin → calls world.close() (AppWorld's own save) → prints {"saved":true}
  Reads EOF → exits

Args: task_id appworld_root apis_url [experiment_name]
  experiment_name defaults to "ccp"
"""
import sys, json, os, io, glob

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
apis_url        = sys.argv[3]
experiment_name = sys.argv[4] if len(sys.argv) > 4 else "ccp"

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld

    # Pass experiment_name so world.close() saves to the right experiment folder.
    world = AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        remote_apis_url=apis_url,
        load_ground_truth=False,
    )

    print(json.dumps({
        "success":    True,
        "task_id":    task_id,
        "db_path":    world.output_db_home_path_in_memory,
        "output_dir": str(world.output_db_home_path),
    }), flush=True)

    # Wait for "save" command
    for line in sys.stdin:
        line = line.strip()
        if line != "save":
            if line == "exit":
                break
            continue

        try:
            # Use AppWorld's own save mechanism — world.close() is the
            # canonical way to persist task state and is guaranteed to match
            # what evaluate_task() expects.
            # Suppress any incidental stdout from close().
            _buf = io.StringIO()
            _old = sys.stdout
            sys.stdout = _buf
            try:
                world.close()
            finally:
                sys.stdout = _old

            # Verify what was saved
            out_dir = str(world.output_db_home_path)
            saved = {
                os.path.basename(f): os.path.getsize(f)
                for f in glob.glob(os.path.join(out_dir, "*"))
            }
            print(f"[seed] close() saved to: {out_dir}", file=sys.stderr, flush=True)
            print(f"[seed] files: {saved}", file=sys.stderr, flush=True)

            # Peek at venmo.jsonl content for diagnosis
            venmo_path = os.path.join(out_dir, "venmo.jsonl")
            if os.path.exists(venmo_path):
                with open(venmo_path) as _vf:
                    _preview = _vf.read(300)
                print(f"[seed] venmo.jsonl preview: {_preview!r}", file=sys.stderr, flush=True)
            else:
                print("[seed] venmo.jsonl NOT FOUND after close()", file=sys.stderr, flush=True)

            print(json.dumps({
                "saved":  True,
                "dir":    out_dir,
                "method": "world.close()",
                "files":  len(saved),
            }), flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"saved": False, "error": str(e)}), flush=True)

        # After close() the server is reset; exit so a fresh subprocess handles
        # the next task.
        break

    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
