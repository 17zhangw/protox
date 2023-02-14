import argparse
import os
import shutil
import psycopg
from psycopg.rows import dict_row
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict
from plumbum import local
from plumbum.commands.processes import ProcessTimedOut
from pathlib import Path
import logging

import sys
sys.path.append("/home/wz2/mythril")

from envs.spec import Spec
from envs.repository import Repository
from envs.reward import RewardUtility
from envs.workload import Workload
from envs.spec import Spec
from envs.pg_env import PostgresEnv
from scripts.replay_bao_utils import run_bao

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--bao-file", default=None)
    parser.add_argument("--bao-config-file", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--workload-timeout", type=int, default=600)
    parser.add_argument("--sample-interval", type=int, default=900)
    parser.add_argument("--skip-knobs", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    benchmark_config_path = "configs/benchmark/job_full.yaml"
    assert args.bao_file is not None
    if args.bao_file is not None:
        benchmark_config_path = args.bao_config_file

    spec = Spec(
        agent_type=None,
        seed=0,
        config_path="configs/job_full.yaml",
        benchmark_config_path=benchmark_config_path,
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

    ccs = [
        "max_connections = 40",
        "shared_buffers = 32GB",
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
    ]
    indexes = [
        "CREATE INDEX cast_info_mi on cast_info (movie_id)",
        "CREATE INDEX cast_info_pi on cast_info (person_id)",
        "CREATE INDEX cast_info_ri on cast_info (role_id)",
        "CREATE INDEX movie_companies_mi on movie_companies (movie_id)",
        "CREATE INDEX movie_info_mi on movie_info (movie_id)",
        "CREATE INDEX movie_keyword_ki on movie_keyword (keyword_id)",
    ]
    ccs = [] if args.skip_knobs else ccs
    indexes = [] if args.skip_index else indexes
    env.shift_state(ccs, indexes, dump_page_cache=True)

    if args.bao_file is not None:
        # quarter hour interval.
        run_bao(env, args.num_samples, args.workload_timeout, args.bao_file, benchmark_config_path, args.sample_interval)
    else:
        with open("out.txt", "w") as f:
            for _ in range(3):
                time = spec.workload._execute_workload(connection=env.connection, workload_timeout=600)
                print(time)
                f.write(f"{time}\n")
