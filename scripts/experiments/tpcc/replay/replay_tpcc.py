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

if __name__ == "__main__":
    spec = Spec(
        agent_type=None,
        seed=0,
        config_path="configs/tpcc.yaml",
        benchmark_config_path="configs/benchmark/tpcc.yaml",
        horizon=0,
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=0,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    reward = RewardUtility("tps", "multiplier", 1.)

    with open("out.txt", "w") as f:
        for i in range(3):
            env.restore_pristine_snapshot()

            env.shift_state(
                [
                    "shared_buffers = 8GB",
                    "effective_cache_size = 24GB",
                    "maintenance_work_mem = 2GB",
                    "checkpoint_completion_target = 0.9",
                    "wal_buffers = 16MB",
                    "default_statistics_target = 100",
                    "random_page_cost = 1.1",
                    "effective_io_concurrency = 200",
                    "work_mem = 52428kB",
                    "huge_pages = try",
                    "min_wal_size = 2GB",
                    "max_wal_size = 8GB",
                    "max_worker_processes = 20",
                    "max_parallel_workers_per_gather = 4",
                    "max_parallel_workers = 20",
                    "max_parallel_maintenance_workers = 4",
                ],
                [
                    "CREATE INDEX idx_customer_name ON customer (c_last, c_first);",
                    "CREATE INDEX idx_oorder_ocid_oid  ON oorder (o_c_id, o_id);",
                    "CREATE INDEX idx_stock_siid on stock (s_i_id);",
                    "CREATE INDEX idx_stock_sq on stock (s_quantity);",
                ], dump_page_cache=True)

            results = Path(f"/home/wz2/mythril/exps_tpcc/validations/base/result{i}")
            if results.exists():
                shutil.rmtree(results)
            results.mkdir(parents=True, exist_ok=True)

            assert spec.workload._execute_benchbase(spec, results)
            tps = reward(results)
            print(tps)
            f.write(f"{tps}\n")
