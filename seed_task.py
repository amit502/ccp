#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, ...}  when ready
  Reads "save <output_dir>" from stdin → saves → prints {"saved":true}
  Reads EOF → exits without calling world.close()

Args: task_id appworld_root apis_url [experiment_name]
"""
import sys, json, os, glob

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
apis_url        = sys.argv[3]
experiment_name = sys.argv[4] if len(sys.argv) > 4 else "ccp"

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    from appworld import AppWorld
    from appworld.apps.api_lib import save_remote_dbs

    # NOTE: do NOT pass experiment_name here — it triggers an auto-save of the
    # initial state inside __init__ which (a) takes time and (b) may interfere
    # with our own save.  We compute the output path ourselves.
    world = AppWorld(
        task_id=task_id,
        remote_apis_url=apis_url,
        load_ground_truth=False,
    )

    # Probe what attributes are available — helps debug save options
    _save_attrs = [a for a in dir(world)
                   if ("save" in a.lower() or "output" in a.lower() or "path" in a.lower())
                   and not a.startswith("__")]
    print(f"[seed] world save/path attrs: {_save_attrs}", file=sys.stderr, flush=True)

    print(json.dumps({
        "success": True,
        "task_id": task_id,
        "db_path": world.output_db_home_path_in_memory,
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

            # from_db_home_path is the REFERENCE BASE for the diff, not the source.
            # diff = (current active task DB) − (from_db_home_path)
            # Passing the task path gives diff = 0 (same DB compared against itself).
            # Passing ":memory:base" gives diff = task_setup + agent_changes,
            # which is what evaluate_task() expects (it subtracts models_start itself).
            ref_path = ":memory:base"
            print(f"[seed] saving changes vs {ref_path} → {out_dir}", file=sys.stderr, flush=True)

            app_names = list(world.task.allowed_apps) + ["admin", "supervisor"]

            save_remote_dbs(
                remote_apis_url=apis_url,
                from_db_home_path=ref_path,
                to_db_home_path=out_dir,
                format="changes",
                app_names=app_names,
                delete_if_exists=True,
                skip_mandatory_apps=False,
                save_model_hashes=True,
            )

            # Verify what was saved
            saved = {
                os.path.basename(f): os.path.getsize(f)
                for f in glob.glob(os.path.join(out_dir, "*"))
            }
            print(f"[seed] saved files: {saved}", file=sys.stderr, flush=True)

            # Peek at venmo.jsonl
            venmo_path = os.path.join(out_dir, "venmo.jsonl")
            if os.path.exists(venmo_path):
                with open(venmo_path) as _f:
                    _preview = _f.read(400)
                print(f"[seed] venmo.jsonl ({os.path.getsize(venmo_path)}B): {_preview!r}",
                      file=sys.stderr, flush=True)
            else:
                print("[seed] venmo.jsonl NOT FOUND", file=sys.stderr, flush=True)

            print(json.dumps({
                "saved": True,
                "dir":   out_dir,
                "from":  active_path,
                "files": len(saved),
            }), flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"saved": False, "error": str(e)}), flush=True)

    # Exit without world.close() — keeps server state alive
    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
