from pathlib import Path
import pandas as pd
import time
from dateutil.parser import parse
import argparse
import shutil
import sys
import os
import importlib

try:
    sys.path.append("/home/wz2/mythril/Auto-Steer")
    os.chdir("/home/wz2/mythril/Auto-Steer")
    from main import run
except Exception as e:
    print(e)

sys.path.remove("/home/wz2/mythril/Auto-Steer")
sys.path.append("/home/wz2/mythril")
os.chdir("/home/wz2/mythril")

# Unfortunate hack.
mod = importlib.import_module("utils")
importlib.reload(mod)
from envs.spec import Spec
from envs.pg_env import PostgresEnv
from utils.dotdict import DotDict
from scripts.parse_hyrise import parse_hyrise_configs


def hyrise_load(args):
    abench = None
    if "dsb" in args.benchmark:
        abench = "dsb"
    elif "tpch" in args.benchmark:
        abench = "tpch"
    else:
        assert "job" in args.benchmark
        abench = "job_full"

    benchmark_knobs = {
        "dsb": [
            "shared_buffers = 8GB",
            "effective_cache_size = 24GB",
            "maintenance_work_mem = 2GB",
            "checkpoint_completion_target = 0.9",
            "wal_buffers = 16MB",
            "default_statistics_target = 500",
            "random_page_cost = 1.1",
            "effective_io_concurrency = 200",
            "work_mem = 10485kB",
            "min_wal_size = 4GB",
            "max_wal_size = 16GB",
            "max_worker_processes = 20",
            "max_parallel_workers_per_gather = 10",
            "max_parallel_workers = 20",
            "max_parallel_maintenance_workers = 4",
        ],
        "job_full": [
            "max_connections = 40",
            "shared_buffers = 8GB",
            "effective_cache_size = 24GB",
            "maintenance_work_mem = 2GB",
            "checkpoint_completion_target = 0.9",
            "wal_buffers = 16MB",
            "default_statistics_target = 500",
            "random_page_cost = 1.1",
            "effective_io_concurrency = 200",
            "work_mem = 10485kB",
            "min_wal_size = 4GB",
            "max_wal_size = 16GB",
            "max_worker_processes = 20",
            "max_parallel_workers_per_gather = 10",
            "max_parallel_workers = 20",
            "max_parallel_maintenance_workers = 4",
        ],
        "tpch": [
            "max_connections = 40",
            "shared_buffers = 8GB",
            "effective_cache_size = 24GB",
            "maintenance_work_mem = 2GB",
            "checkpoint_completion_target = 0.9",
            "wal_buffers = 16MB",
            "default_statistics_target = 500",
            "random_page_cost = 1.1",
            "effective_io_concurrency = 200",
            "work_mem = 10485kB",
            "min_wal_size = 4GB",
            "max_wal_size = 16GB",
            "max_worker_processes = 20",
            "max_parallel_workers_per_gather = 10",
            "max_parallel_workers = 20",
            "max_parallel_maintenance_workers = 4",
        ],
    }[abench]

    if "no_knob" in args and args.no_knob:
        benchmark_knobs = []

    spec = Spec(
        agent_type=None,
        seed=0,
        config_path=args.config_file,
        benchmark_config_path=args.benchmark_config_file,
        horizon=0,
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=0,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    nindexdefs, _ = parse_hyrise_configs(args)
    if abench == "tpch":
        [indexdefs.append("""
create view revenue0_PID (supplier_no, total_revenue) as
	select
		l_suppkey,
		sum(l_extendedprice * (1 - l_discount))
	from
		lineitem
	where
		l_shipdate >= date '1994-09-01'
		and l_shipdate < date '1994-09-01' + interval '3' month
	group by
		l_suppkey;
        """) for indexdefs in nindexdefs]

    if args.execute:
        env.restore_pristine_snapshot()
        assert len(nindexdefs) > 0
        indexdefs = nindexdefs[-1]
        env.shift_state(benchmark_knobs, indexdefs, ignore_error=True, dump_page_cache=True)
        with open("out.txt", "w") as f:
            for _ in range(3):
                time = spec.workload._execute_workload(connection=env.connection, workload_timeout=300)
                f.write(f"{time}\n")

    if args.execute_as:
        import logging
        assert len(nindexdefs) > 0
        benchmark = args.benchmark
        asc = args.as_config
        rdir = args.output_dir
        Path(rdir).mkdir(parents=True, exist_ok=True)

        for i, indexdefs in enumerate(nindexdefs):
            assert len(indexdefs) > 0
            env.restore_pristine_snapshot()
            env.shift_state(benchmark_knobs, indexdefs, ignore_error=True)

            import time
            start = time.time()
            aargs = DotDict({
                "training": True,
                "inference": False,
                "database": "postgres",
                "benchmark": benchmark,
                "config": asc,
                "output_dir": rdir,
                "output_name": f"{args.output_name}_{i}",
            })
            run(aargs)

            import logging
            import time
            logging.warn(f"Finished {i} auto-steer sweep: {time.time() - start}")

    if args.check_mem:
        data = []
        for i, indexdefs in enumerate(nindexdefs):
            assert len(indexdefs) > 0
            env.restore_pristine_snapshot()
            env.shift_state(benchmark_knobs, indexdefs, ignore_error=True)
            _, mem_gb = env.workload.compute_used_mem(env.connection, False, None, None)
            data.append({
                "indexdefidx": i,
                "mem_gb": mem_gb,
                "mem_mb": mem_gb*1024,
            })
        pd.DataFrame(data).to_csv(args.output_name, index=False)

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--benchmark-config-file", default=None)
    parser.add_argument("--hyrise-log", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--no-knob", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-as", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--check-mem", action="store_true")
    parser.add_argument("--as-config", default="Auto-Steer/configs/postgres.cfg")
    args = parser.parse_args()
    hyrise_load(args)
