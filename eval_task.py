#!/usr/bin/env python3
"""
Standalone AppWorld task evaluator — runs inside the appworld venv.
Usage: python eval_task.py <task_id> <appworld_root> [experiment_name]
"""
import sys, json, os, io

task_id         = sys.argv[1]
appworld_root   = sys.argv[2]
experiment_name = sys.argv[3] if len(sys.argv) > 3 else "ccp"

try:
    os.environ["APPWORLD_ROOT"] = appworld_root

    from appworld.common.path_store import path_store
    path_store.update_root(appworld_root)

    # Check the output dir exists and has files
    out_dbs = os.path.join(appworld_root, "experiments", "outputs",
                           experiment_name, "tasks", task_id, "dbs")
    if os.path.exists(out_dbs):
        files = os.listdir(out_dbs)
        print(f"[eval_task] output dbs dir has {len(files)} files: {files[:5]}", file=sys.stderr)
        # Show venmo.jsonl first (the key file), then a sample of others
        import glob as _glob
        all_jl = sorted(_glob.glob(os.path.join(out_dbs, "*.jsonl")))
        # Always print venmo first, then up to 3 others for context
        # Print all non-zero files + venmo + supervisor
        priority = [j for j in all_jl if "venmo" in j or "supervisor" in j]
        nonzero  = [j for j in all_jl if os.path.getsize(j) > 0 and j not in priority]
        for jl in priority + nonzero:
            try:
                sz = os.path.getsize(jl)
                limit = 1500 if "supervisor" in jl else 600
                with open(jl) as _f:
                    _preview = _f.read(limit).strip()
                print(f"[eval_task] {os.path.basename(jl)} ({sz}B): {_preview!r}", file=sys.stderr)
            except Exception as _e:
                print(f"[eval_task] {os.path.basename(jl)}: read error {_e}", file=sys.stderr)
    else:
        print(f"[eval_task] output dbs dir NOT FOUND: {out_dbs}", file=sys.stderr)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    from appworld.evaluator import evaluate_task
    result = evaluate_task(
        task_id=task_id,
        experiment_name=experiment_name,
        suppress_errors=True,
        save_report=True,
    )

    sys.stdout = old_stdout

    for t in (result.passes or []):
        print(f"  PASS: {t.get('label','?')}", file=sys.stderr)
    for t in (result.failures or []):
        req = t.get('requirement', '')[:150]
        print(f"  FAIL: {t.get('label','?')} — {req}", file=sys.stderr)

    print(json.dumps({
        "success":    bool(result.success),
        "pass_count": int(result.pass_count),
        "num_tests":  int(result.num_tests),
    }))

except Exception as e:
    try:
        sys.stdout = sys.__stdout__
    except Exception:
        pass
    print(json.dumps({"success": False, "error": str(e)}))
    print(f"[eval_task] error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
