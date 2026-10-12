#!/usr/bin/env python3
"""Queue the full cache/pin comparison, verify telemetry, and generate its figures."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMOKE = r'''
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
from run_full_cache_pin_comparison import config_text, case_environment
case = Path(tempfile.mkdtemp(prefix='sirius_cache_trace_smoke_', dir='/tmp'))
for child in ('log_dir', 'telemetry_data', 'disk_downgrade'):
    (case / child).mkdir()
(case / 'sirius.yaml').write_text(config_text(case))
os.environ.update(case_environment(case, 'pin_gpu', 6 * 2**30, 'hit'))
import duckdb
con=duckdb.connect(':memory:', config={'allow_unsigned_extensions':'true'})
parquet=case / 'sample.parquet'
con.execute(f"COPY (SELECT i::BIGINT AS id, ('value_' || i::VARCHAR) AS text FROM range(64) t(i)) TO '{parquet}' (FORMAT PARQUET)")
con.execute(f"CREATE VIEW sample AS SELECT * FROM read_parquet('{parquet}')")
con.execute(f"LOAD '{Path('build/release/extension/sirius/sirius.duckdb_extension').resolve()}'")
query="SELECT SUM(id), MIN(text), MAX(text) FROM sample WHERE id >= 0"
con.execute('SET gpu_execution=false')
expected=con.execute(query).fetchall()
con.execute('SET gpu_execution=true')
con.execute("CALL sirius_set_query_label('smoke_q1_exec1')")
assert con.execute(query).fetchall()==expected
con.execute("CALL sirius_set_query_label('preparation')")
con.execute(f"CALL pin_table('{parquet}', name='sample', tier='gpu', cols=['id', 'text'])")
con.execute("CALL sirius_set_query_label('smoke_q1_exec2')")
assert con.execute(query).fetchall()==expected
con.execute("CALL unpin_table('sample')")
con.close()
logs='\n'.join(p.read_text() for p in (case/'log_dir').glob('*.log'))
assert 'path=parquet' in logs and 'path=cached_provider' in logs and 'metric=value_bytes_v2' in logs
expected_bytes=64*8 + sum(len(f'value_{i}') for i in range(64))
assert f'total_bytes={expected_bytes} read_bytes=0 kind=full path=cached_provider' in logs
print(f'PASS: CPU/reader/pin results match; cached-provider value bytes={expected_bytes}; artifacts={case}')
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wait-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'background_status.json'
    if status_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing supervisor status: {status_path}')
    status = {'pid': os.getpid(), 'output': str(output), 'wait_pids': args.wait_pid,
              'queries': {'ClickBench': list(range(1, 44)), 'TPC-H SF50': list(range(1, 23))}}

    def update(stage: str, **fields) -> None:
        status.update(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(status), flush=True)

    parent_commands = {}
    for pid in args.wait_pid:
        p = Path(f'/proc/{pid}/cmdline')
        if p.exists():
            parent_commands[pid] = p.read_bytes()
    update('queued')
    idle_checks = 0
    while idle_checks < 2:
        parents = [pid for pid, command in parent_commands.items()
                   if Path(f'/proc/{pid}/cmdline').exists()
                   and Path(f'/proc/{pid}/cmdline').read_bytes() == command]
        gpu = subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True)
        busy = int(gpu.stdout.strip()) > 512
        # Avoid sharing the NVMe with another benchmark even on a different GPU.
        processes = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True, check=True).stdout
        storage_jobs = [line.strip() for line in processes.splitlines()
                        if any(token in line for token in (
                            'run_clickbench_breakdown.py', 'run_random22_breakdown_var.py',
                            'performance_test.py', 'check_clickbench_correctness.py',
                            'check_tpch_correctness.py', 'run_cpu_baseline.py'))]
        if parents or busy or storage_jobs:
            idle_checks = 0
            if status.get('stage') != 'waiting_for_existing_experiment':
                update('waiting_for_existing_experiment', active_parents=parents,
                       gpu_memory_mib=int(gpu.stdout.strip()), active_jobs=storage_jobs)
        else:
            idle_checks += 1
        if idle_checks < 2:
            time.sleep(10)
    update('verifying_metrics', active_parents=[], active_jobs=[])
    with (output / 'metrics_smoke.log').open('w') as log:
        verification = subprocess.run([sys.executable, '-c', SMOKE], cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT)
    if verification.returncode:
        update('verification_failed', returncode=verification.returncode)
        return verification.returncode
    update('running_full_comparison')
    command = [sys.executable, str(ROOT / 'scripts/run_full_cache_pin_comparison.py'),
               '--output', str(output / 'runs'), '--trace-mode', 'both', '--run']
    with (output / 'experiment_driver.log').open('w') as log:
        experiment = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    update('generating_figures', experiment_returncode=experiment.returncode)
    plot = [sys.executable, str(ROOT / 'result/plot_full_cache_pin_comparison.py'),
            '--input', str(output / 'runs'), '--output', str(output / 'figures')]
    with (output / 'plot.log').open('w') as log:
        figures = subprocess.run(plot, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    stage = ('finished' if experiment.returncode == 0 else 'finished_with_experiment_failures')
    if figures.returncode:
        stage = 'figure_generation_failed'
    update(stage, plot_returncode=figures.returncode, figures=str(output / 'figures'))
    return experiment.returncode or figures.returncode


if __name__ == '__main__':
    raise SystemExit(main())
