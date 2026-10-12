#!/usr/bin/env python3
"""Use an isolated-process exit after a fully persisted cache-trace workload.

The frozen engine can fault in scan-manager destruction after worker_finished.
This wrapper preserves all query/cache settings and bypasses only that final
destructor. It requires every planned query to succeed, synchronizes CUDA and
flushes logs before exiting; failures do not take the successful-exit path.
"""

import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

import run_full_cache_pin_comparison as runner


def argument(name):
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve()


def main():
    manifest = json.loads(argument("--execute-plan").read_text())
    wrapper = manifest["post_workload_exit_wrapper"]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == wrapper["sha256"]
    assert wrapper["policy"] == "synchronize_flush_then_process_exit_after_all_queries"
    if "--worker" in sys.argv:
        import duckdb

        metadata = argument("--metadata")
        meta = json.loads(metadata.read_text())
        assert meta["trace_mode"] == "hit"
        original_connect = duckdb.connect

        class Connection:
            def __init__(self, connection):
                self.connection = connection

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def close(self):
                rows = list(csv.DictReader(Path(meta["query_csv"]).open()))
                expected = {(ex, q) for ex in (1, 2, 3) for q in meta["order"]}
                complete = (
                    len(rows) == len(expected)
                    and {(int(r["execution"]), int(r["query"])) for r in rows}
                    == expected
                    and all(r["status"] == "ok" for r in rows)
                )
                journal = Path(meta["journal"])
                events = [json.loads(line) for line in journal.read_text().splitlines()]
                complete = complete and events[-1]["stage"] == "worker_finished"
                if not complete:
                    self.connection.close()
                    return
                runtime = ctypes.CDLL("libcudart.so.13")
                assert runtime.cudaDeviceSynchronize() == 0
                runner.event(
                    journal,
                    stage="post_workload_process_exit",
                    reason="known cache-page destructor fault",
                    all_queries_completed=True,
                    cuda_synchronized=True,
                )
                print(
                    "All queries persisted; CUDA synchronized; bypassing only final engine destruction.",
                    flush=True,
                )
                sys.stdout.flush()
                sys.stderr.flush()
                libc = ctypes.CDLL(None)
                libc.fflush.argtypes = [ctypes.c_void_p]
                assert libc.fflush(None) == 0
                os._exit(0)

        duckdb.connect = lambda *args, **kwargs: Connection(
            original_connect(*args, **kwargs)
        )
    # execute_plan uses this path when launching worker subprocesses. Its original
    # immutable manifest/source checks and query worker stay unchanged.
    runner.__file__ = __file__
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
