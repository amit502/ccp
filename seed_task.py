#!/usr/bin/env python3
"""
Seed + Save task using AppWorld Python library in the appworld venv.

Protocol:
  Prints {"success": true, ...}  when ready
  Reads "save" from stdin → saves task output → prints {"saved":true}
  Reads EOF → exits

Save strategy:
  1. rest_diff: get credentials via /supervisor/account_passwords,
     login to each app, query current records, diff against initial
     SQLite, write SQL INSERT format (same format world.close() uses)
  2. Always also run world.close() after — it handles supervisor.jsonl +
     DELETE /dbs/cache + DELETE /date_time cleanup.

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
# REST diff helpers — must be defined before main try block
# ---------------------------------------------------------------------------

def _get_app_token(apis_url, app_name, cred):
    """Try common OAuth + JSON login patterns to get an access token."""
    username = (cred.get("account_name") or cred.get("username")
                or cred.get("email") or "")
    password = cred.get("password") or ""
    if not username or not password:
        return None

    for login_path in (
        f"/{app_name}/auth/token",
        f"/{app_name}/accounts/login",
        f"/{app_name}/login",
    ):
        url = f"{apis_url}{login_path}"
        # OAuth form-data (AppWorld standard)
        try:
            r = requests.post(url, data={"username": username, "password": password}, timeout=10)
            if r.status_code == 200:
                tok = r.json().get("access_token") or r.json().get("token")
                if tok:
                    return tok
        except Exception:
            pass
        # JSON body fallback
        for body in ({"email": username, "password": password},
                     {"username": username, "password": password}):
            try:
                r2 = requests.post(url, json=body, timeout=10)
                if r2.status_code == 200:
                    tok = r2.json().get("access_token") or r2.json().get("token")
                    if tok:
                        return tok
            except Exception:
                pass
    return None


def _query_table(apis_url, app_name, table_name, token):
    """
    Fetch all records for a table via the app's REST API.
    Tries several common URL patterns.
    Returns a list of dicts or None if not found.
    """
    params = {"access_token": token}
    for url in (
        f"{apis_url}/{app_name}/{table_name}",
        f"{apis_url}/{app_name}/{table_name}/list",
        f"{apis_url}/{app_name}/{table_name}s",
    ):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            return v
        except Exception:
            pass
    return None


def _rest_diff_save(task_id, appworld_root, apis_url, out_dbs_path):
    """
    For each app that the supervisor has credentials for:
      1. Login to the app
      2. Enumerate all records for all tables
      3. Diff against the task's initial SQLite state (on disk)
      4. Write new records to out_dbs_path/{app}.jsonl in SQL INSERT format

    Returns (records_written: int, skipped_apps: list).
    """
    initial_dbs = Path(appworld_root) / "data" / "tasks" / task_id / "dbs"
    out_dbs     = Path(out_dbs_path)

    # ---- Get app credentials from supervisor ----
    creds_r = requests.get(f"{apis_url}/supervisor/account_passwords", timeout=10)
    creds_r.raise_for_status()
    creds_raw = creds_r.json()
    print(f"[seed_task] account_passwords ({len(json.dumps(creds_raw))}B): "
          f"{json.dumps(creds_raw)[:600]}", file=sys.stderr, flush=True)

    # Normalise to {app_name: cred_dict}
    app_creds: dict = {}
    if isinstance(creds_raw, list):
        for item in creds_raw:
            app = item.get("app_name") or item.get("app") or ""
            if app:
                app_creds[app] = item
    elif isinstance(creds_raw, dict):
        app_creds = creds_raw

    print(f"[seed_task] apps with credentials: {list(app_creds.keys())}",
          file=sys.stderr, flush=True)

    written   = 0
    skipped   = []

    for db_file in sorted(initial_dbs.glob("*.db")):
        app_name = db_file.stem
        if app_name == "supervisor":
            continue  # world.close() handles supervisor

        if app_name not in app_creds:
            skipped.append(f"{app_name}:no-creds")
            continue

        # ---- Login ----
        token = _get_app_token(apis_url, app_name, app_creds[app_name])
        if not token:
            skipped.append(f"{app_name}:login-failed")
            print(f"[seed_task] login failed for {app_name}", file=sys.stderr, flush=True)
            continue

        # ---- Read initial state + schema from disk SQLite ----
        initial_by_table: dict = {}
        schema_by_table:  dict = {}
        try:
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
            for tbl in tables:
                try:
                    rows = conn.execute(f"SELECT * FROM {tbl}").fetchall()
                    initial_by_table[tbl] = {row["id"] for row in rows}
                    schema_by_table[tbl]  = [
                        r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")
                    ]
                except Exception:
                    pass
            conn.close()
        except Exception as e_db:
            skipped.append(f"{app_name}:db-read-error")
            print(f"[seed_task] cannot read {db_file}: {e_db}", file=sys.stderr)
            continue

        # ---- Query current records and find new ones ----
        new_by_table: dict = {}
        for table, init_ids in initial_by_table.items():
            current = _query_table(apis_url, app_name, table, token)
            if current is None:
                continue
            new_recs = [r for r in current if r.get("id") not in init_ids]
            if new_recs:
                new_by_table[table] = new_recs

        if not new_by_table:
            continue

        # ---- Write SQL INSERT format (same as world.close() produces) ----
        out_file = out_dbs / f"{app_name}.jsonl"
        with open(out_file, "w") as f:
            for table, records in new_by_table.items():
                cols = schema_by_table.get(table, [])
                if not cols:
                    continue
                sql = (f"INSERT INTO {table} "
                       f"({', '.join(cols)}) "
                       f"VALUES ({', '.join(['?'] * len(cols))})")
                for rec in records:
                    vals = [rec.get(col) for col in cols]
                    f.write(json.dumps([sql, vals]) + "\n")
                    written += 1

        n_new = sum(len(v) for v in new_by_table.items())
        print(f"[seed_task] {app_name}: {len(new_by_table)} tables, "
              f"{written} records written", file=sys.stderr, flush=True)

    return written, skipped


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

    _all_attrs  = [a for a in dir(world) if not a.startswith("__")]
    _save_attrs = [a for a in _all_attrs
                   if any(k in a.lower()
                          for k in ("save","close","output","path","db","dump","export"))]

    _sv_paths = []
    try:
        _sv_paths = list(
            requests.get(f"{apis_url}/supervisor/openapi.json", timeout=5)
            .json().get("paths", {}).keys()
        )
    except Exception:
        pass

    print(json.dumps({
        "success":    True,
        "task_id":    task_id,
        "db_path":    world.output_db_home_path_in_memory,
        "save_attrs": _save_attrs,
        "sv_paths":   _sv_paths,
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

        rest_written = 0
        rest_skipped = []
        rest_error   = ""

        # ---- Step 1: REST diff (writes app-specific .jsonl files) ----
        try:
            rest_written, rest_skipped = _rest_diff_save(
                task_id, appworld_root, apis_url, str(out_dbs)
            )
        except Exception as e_diff:
            rest_error = str(e_diff)[:300]
            print(f"[seed_task] rest_diff error: {e_diff}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc(file=sys.stderr)

        # ---- Step 2: world.close() — writes supervisor.jsonl + cleanup ----
        close_error = ""
        try:
            _buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf
            try:
                world.close()
            finally:
                sys.stdout = _old
        except Exception as e_c:
            close_error = str(e_c)[:200]
            print(f"[seed_task] world.close() error: {e_c}", file=sys.stderr, flush=True)

        # ---- Report ----
        saved_files = list(out_dbs.glob("*.jsonl"))
        sizes       = {f.name: f.stat().st_size for f in saved_files}
        venmo_bytes = sizes.get("venmo.jsonl", 0)
        nonzero     = [k for k, v in sizes.items() if v > 0]

        print(f"[seed_task] save done: rest_written={rest_written} "
              f"skipped={rest_skipped} venmo={venmo_bytes}B nonzero={nonzero}",
              file=sys.stderr, flush=True)

        print(json.dumps({
            "saved":        True,
            "method":       f"rest_diff+close ({rest_written} new records)",
            "rest_error":   rest_error,
            "close_error":  close_error,
            "skipped":      rest_skipped,
            "venmo_bytes":  venmo_bytes,
            "nonzero":      len(nonzero),
        }), flush=True)

        break

    os._exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}), flush=True)
    print(f"[seed_task] error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
