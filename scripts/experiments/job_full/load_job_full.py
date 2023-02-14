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
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--bao-file", default=None)
    parser.add_argument("--omit-index", action="store_true")
    args = parser.parse_args()

    benchmark_config_path = "configs/benchmark/job_full.yaml"
    if args.bao_file is not None:
        benchmark_config_path = "configs/benchmark/job_full.yaml.bao"

    spec = Spec(
        agent_type=None,
        seed=0,
        config_path=args.config_file,
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

    idxs = [] if args.omit_index else [
        "CREATE INDEX cast_info_mi on cast_info (movie_id)",
        "CREATE INDEX cast_info_pi on cast_info (person_id)",
        "CREATE INDEX cast_info_ri on cast_info (role_id)",
        "CREATE INDEX movie_companies_mi on movie_companies (movie_id)",
        "CREATE INDEX movie_info_mi on movie_info (movie_id)",
        "CREATE INDEX movie_keyword_ki on movie_keyword (keyword_id)",
    ]

    env.shift_state(
        [
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
        idxs,
        dump_page_cache=True)

    #with open("/mnt/nvme0n1/wz2/noisepage/pgdata/postgresql.auto.conf", "r") as f:
    #    lines = []
    #    for line in f:
    #        if "shared_preload" not in line:
    #            lines.append(line.strip())

    #with open("/mnt/nvme0n1/wz2/noisepage/pgdata/postgresql.auto.conf", "w") as f:
    #    for line in lines:
    #        f.write(line + "\n")

    #    f.write("shared_preload_libraries = 'pg_bao'")
