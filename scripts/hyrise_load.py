import argparse

import sys
sys.path.append("/home/wz2/mythril")
from envs.spec import Spec
from envs.pg_env import PostgresEnv

def scan_index_str(index_str):
    indexes = set()
    current_index = None
    current_column = None
    for i in range(len(index_str)):
        if index_str[i] == "I":
            current_index = []
        elif index_str[i] == "C":
            current_column = ""
        elif index_str[i] == ",":
            if current_index is not None:
                current_index.append(current_column)
                current_column = ""
        elif index_str[i] == ")":
            if current_column is not None and len(current_column) > 0:
                current_index.append(current_column)
            indexes.add(tuple(current_index))
            current_index = None
        elif current_column is not None and index_str[i] != " ":
            current_column += index_str[i]

    indexdefs = []
    for index_cols in indexes:
        index_tbl = None
        index_colnames = []
        for ic in index_cols:
            tbl, col = ic.split(".")
            if index_tbl is None:
                index_tbl = tbl
            index_colnames.append(col)

        cnames = ",".join(index_colnames)
        indexdefs.append(f"CREATE INDEX index{len(indexdefs)} ON {index_tbl} ({cnames})")
    return indexdefs


def hyrise_load(args):
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
        "dsb_s2": [
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
        "dsb_s3": [
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
        "job_a": [
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
        "job_ab": [
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
    }[args.benchmark]

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
    env.restore_pristine_snapshot()

    indexdefs = None
    with open(args.hyrise_log) as f:
        for line in f:
            if "Indexes found: " in line:
                indexstr = line.split("Indexes found: ")[-1]
                indexdefs = scan_index_str(indexstr)
                break

    if indexdefs is None:
        with open(args.hyrise_log) as f:
            last_dta = None
            addtl_indexes = []
            for line in f:
                if "AnytimeDTA found new best:" in line:
                    last_dta = line.strip()

                elif "Additional best index found: " in line:
                    addtl_indexes.append(line)

            if last_dta is not None:
                last_dta = last_dta.split("AnytimeDTA found new best: ")[-1]
                indexdefs = scan_index_str(last_dta)

            else:
                # Attempt to read the stream of additional best indexes found.
                indexdefs = []
                for addtl_index in addtl_indexes:
                    indexdef = addtl_index.split("(I(")[-1].split("), ")[0]
                    cols = indexdef.split(",")
                    columns = []
                    table = None
                    for col in cols:
                        assert col.startswith("C ")
                        coldef = col[2:].split(".")
                        if table is None:
                            table = coldef[0]
                        else:
                            assert table == coldef[0]
                        columns.append(coldef[1])

                    cnames = ",".join(columns)
                    indexdefs.append(f"CREATE INDEX index{len(indexdefs)} ON {table} ({cnames})")

    if args.benchmark == "tpch":
        indexdefs.append("""
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
        """)

    assert len(indexdefs) > 0
    env.shift_state(benchmark_knobs, indexdefs, ignore_error=True, dump_page_cache=True)

    if "execute" in args and args.execute:
        with open("out.txt", "w") as f:
            for _ in range(3):
                time = spec.workload._execute_workload(connection=env.connection, workload_timeout=300)
                print(time)
                f.write(f"{time}\n")

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--benchmark-config-file", default=None)
    parser.add_argument("--hyrise-log", default=None)
    parser.add_argument("--no-knob", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    hyrise_load(args)
