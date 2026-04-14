#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, ...}  when ready
  Reads "save" from stdin → saves task output → prints {"saved":true}
  Reads EOF → exits

Save strategy (in order of preference):
  1. Native world method: save_output_dbs / dump_output_dbs / export_output_dbs
  2. Supervisor REST API export endpoint
  3. SQLite diff: initial disk DBs vs current server state via supervisor REST

Args: task_id appworld_root apis_url [experiment_name]
"""
import sys, json, os, io, sqlite3
import requests
from pathlib import Path

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
apis_url        = sys.argv[3].rstrip("/")
experiment_name = sys.argv[4] if len(sys.argv) > 4 else "ccp"


# ---------------------------------------------------------------------------
# Strategy 3: SQLite diff helper (defined before main try so it's in scope)
# ---------------------------------------------------------------------------

def _sqlite_diff_save(task_id, appworld_root, apis_url, out_dbs_str):
    """
    Compare each app's on-disk initial SQLite DB against current server state
    (queried via supervisor REST API), write changed records to out_dbs/*.jsonl.

    Returns (success: bool, method_description: str).
    """
    initial_dbs = Path(appworld_root) / "data" / "tasks" / task_id / "dbs"
    out_dbs     = Path(out_dbs_str)

    if not initial_dbs.exists():
        raise FileNotFoundError(f"initial dbs not found: {initial_dbs}")

    # Fetch supervisor OpenAPI to discover record-listing endpoints
    try:
        spec     = requests.get(f"{apis_url}/supervisor/openapi.json", timeout=5).json()
        sv_paths = list(spec.get("paths", {}).keys())
    except Exception:
        sv_paths = []

    print(f"[seed_task] sqldiff: supervisor paths={sv_paths[:30]}", file=sys.stderr, flush=True)

    written = 0
    for db_file in sorted(initial_dbs.glob("*.db")):
        app_name = db_file.stem   # e.g. "venmo"
        out_file = out_dbs / f"{app_name}.jsonl"

        # Read initial record IDs from disk SQLite
        initial_ids_by_table: dict = {}
        try:
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
            for tbl in tables:
                try:
                    rows = conn.execute(f"SELECT id FROM {tbl}").fetchall()
                    initial_ids_by_table[tbl] = {row[0] for row in rows}
                except Exception:
                    pass
            conn.close()
        except Exception as e_db:
            print(f"[seed_task] sqldiff: cannot read {db_file}: {e_db}", file=sys.stderr)
            continue

        # Try to get current records via supervisor list endpoints
        new_records = []
        for tbl in initial_ids_by_table:
            # Try common URL patterns
            for url_pat in (
                f"{apis_url}/supervisor/{app_name}/{tbl}",
                f"{apis_url}/supervisor/{app_name}/{tbl}/list",
                f"{apis_url}/{app_name}/supervisor/{tbl}",
            ):
                try:
                    r = requests.get(url_pat, timeout=10)
                    if r.status_code == 200:
                        data    = r.json()
                        records = data if isinstance(data, list) else data.get("data", [])
                        init_ids = initial_ids_by_table[tbl]
                        for rec in records:
                            rid = rec.get("id")
                            if rid is not None and rid not in init_ids:
                                new_records.append(rec)
                        break
                except Exception:
                    pass

        if new_records:
            with open(out_file, "w") as f:
                for rec in new_records:
                    f.write(json.dumps(rec) + "\n")
            written += len(new_records)
            print(f"[seed_task] sqldiff: wrote {len(new_records)} records to {out_file.name}",
                  file=sys.stderr, flush=True)

    return written > 0, f"sqldiff ({written} new records)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Collect all non-dunder attributes for debugging
    _all_attrs  = [a for a in dir(world) if not a.startswith("__")]
    _save_attrs = [a for a in _all_attrs
                   if any(k in a.lower()
                          for k in ("save","close","output","path","db","dump","export"))]
    print(f"[seed_task] save_attrs: {_save_attrs}", file=sys.stderr, flush=True)

    # Fetch supervisor OpenAPI — include paths in ready signal (stdout) so they show in logs
    _sv_paths = []
    try:
        r = requests.get(f"{apis_url}/supervisor/openapi.json", timeout=5)
        if r.status_code == 200:
            _sv_paths = list(r.json().get("paths", {}).keys())
    except Exception as e_spec:
        print(f"[seed_task] supervisor spec error: {e_spec}", file=sys.stderr, flush=True)

    # Also expose world.apis attribute names (one level deep)
    _apis_attrs = []
    try:
        _apis_attrs = [a for a in dir(world.apis) if not a.startswith("_")]
    except Exception:
        pass

    print(json.dumps({
        "success":    True,
        "task_id":    task_id,
        "db_path":    world.output_db_home_path_in_memory,
        "save_attrs": _save_attrs,
        "sv_paths":   _sv_paths,
        "apis_attrs": _apis_attrs,
    }), flush=True)

    # -----------------------------------------------------------------------
    # Wait for "save" command
    # -----------------------------------------------------------------------
    for line in sys.stdin:
        cmd = line.strip()
        if cmd != "save":
            if cmd == "exit":
                break
            continue

        out_dbs = Path(appworld_root) / "experiments" / "outputs" \
                  / experiment_name / "tasks" / task_id / "dbs"
        out_dbs.mkdir(parents=True, exist_ok=True)

        saved_ok    = False
        save_method = "none"
        save_error  = ""

        # ---- Strategy 1: look for a native save method on world ----
        for method_name in ("save_output_dbs", "dump_output_dbs",
                            "export_output_dbs", "write_output_dbs"):
            fn = getattr(world, method_name, None)
            if fn is not None:
                try:
                    fn(str(out_dbs))
                    saved_ok    = True
                    save_method = method_name
                    break
                except Exception as e_m:
                    save_error = f"{method_name}: {e_m}"
                    print(f"[seed_task] {save_error}", file=sys.stderr, flush=True)

        # ---- Strategy 2: supervisor REST export endpoint ----
        if not saved_ok:
            try:
                for sv_path in (
                    f"/supervisor/export_task_output/{task_id}",
                    f"/supervisor/tasks/{task_id}/export",
                    f"/supervisor/save/{task_id}",
                ):
                    for method, kw in (
                        ("post", {"json": {"output_dir": str(out_dbs),
                                           "experiment_name": experiment_name}}),
                        ("get",  {"params": {"output_dir": str(out_dbs),
                                             "experiment_name": experiment_name}}),
                    ):
                        r = requests.request(method, f"{apis_url}{sv_path}", timeout=30, **kw)
                        if r.status_code == 200:
                            saved_ok    = True
                            save_method = f"supervisor {method.upper()} {sv_path}"
                            break
                    if saved_ok:
                        break
            except Exception as e_sv:
                save_error += f" | supervisor: {e_sv}"
                print(f"[seed_task] supervisor export error: {e_sv}", file=sys.stderr, flush=True)

        # ---- Strategy 3: SQLite diff ----
        if not saved_ok:
            try:
                saved_ok, save_method = _sqlite_diff_save(
                    task_id, appworld_root, apis_url, str(out_dbs)
                )
            except Exception as e_diff:
                save_error += f" | sqldiff: {e_diff}"
                print(f"[seed_task] sqldiff error: {e_diff}", file=sys.stderr, flush=True)
                import traceback; traceback.print_exc(file=sys.stderr)

        # ---- Fallback: world.close() ----
        if not saved_ok:
            try:
                _buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf
                try:
                    world.close()
                finally:
                    sys.stdout = _old
                save_method = "world.close() [0-byte fallback]"
                saved_ok    = True
            except Exception as e_c:
                save_error += f" | close: {e_c}"

        # Report file sizes
        saved_files = list(out_dbs.glob("*.jsonl"))
        sizes       = {f.name: f.stat().st_size for f in saved_files}
        venmo_bytes = sizes.get("venmo.jsonl", 0)
        nonzero     = {k: v for k, v in sizes.items() if v > 0}
        print(f"[seed_task] result '{save_method}': "
              f"venmo={venmo_bytes}B nonzero={list(nonzero.keys())}",
              file=sys.stderr, flush=True)

        print(json.dumps({
            "saved":       saved_ok,
            "method":      save_method,
            "save_error":  save_error[:300] if save_error else "",
            "venmo_bytes": venmo_bytes,
            "nonzero":     len(nonzero),
        }), flush=True)

        break   # one save per subprocess lifetime

    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
