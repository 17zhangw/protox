from plumbum import local
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TPCC-Replay")
    parser.add_argument("--index-tuner", default="dexter")
    args = parser.parse_args()

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

    indexes = {
        "dexter": [
            "CREATE INDEX idx_customer_name ON customer (c_last, c_first);",
            "CREATE INDEX idx_oorder_ocid_oid  ON oorder (o_c_id, o_id);",
            "CREATE INDEX idx_stock_siid on stock (s_i_id);",
            "CREATE INDEX idx_stock_sq on stock (s_quantity);",
        ],
        "dta_2_width": [
            "CREATE INDEX index0 ON warehouse (w_id, w_name);",
            "CREATE INDEX index1 ON warehouse (w_id, w_tax);",
            "CREATE INDEX index2 ON district (d_id, d_w_id);",
            "CREATE INDEX index3 ON oorder (o_w_id, o_c_id);",
            "CREATE INDEX index4 ON item (i_id);",
            "CREATE INDEX index5 ON stock (s_i_id, s_w_id);",
            "CREATE INDEX index6 ON customer (c_w_id, c_last);",
            "CREATE INDEX index7 ON stock (s_i_id);",
        ],
        "dta_3_width": [
            "CREATE INDEX index0 ON customer (c_last, c_d_id, c_w_id);",
            "CREATE INDEX index1 ON district (d_w_id, d_id, d_next_o_id);",
            "CREATE INDEX index2 ON warehouse (w_id, w_tax);",
            "CREATE INDEX index3 ON customer (c_w_id, c_d_id, c_id);",
            "CREATE INDEX index4 ON oorder (o_w_id, o_d_id, o_id);",
            "CREATE INDEX index5 ON new_order (no_d_id, no_w_id, no_o_id);",
            "CREATE INDEX index6 ON order_line (ol_d_id, ol_w_id, ol_o_id);",
            "CREATE INDEX index7 ON stock (s_i_id, s_w_id, s_quantity);",
            "CREATE INDEX index8 ON stock (s_i_id);",
            "CREATE INDEX index9 ON oorder (o_d_id, o_c_id, o_w_id);",
            "CREATE INDEX index10 ON item (i_id, i_data, i_price);",
        ],
        "dta_4_width": [
            "CREATE INDEX index0 ON customer (c_w_id, c_d_id, c_last, c_first);",
            "CREATE INDEX index1 ON order_line (ol_d_id, ol_w_id, ol_o_id, ol_i_id);",
            "CREATE INDEX index2 ON customer (c_id, c_w_id, c_d_id, c_payment_cnt);",
            "CREATE INDEX index3 ON oorder (o_c_id, o_d_id, o_w_id, o_id);",
            "CREATE INDEX index4 ON oorder (o_w_id, o_id, o_d_id, o_c_id);",
            "CREATE INDEX index5 ON order_line (ol_w_id, ol_o_id, ol_d_id, ol_amount);",
            "CREATE INDEX index6 ON stock (s_i_id, s_w_id, s_quantity);",
            "CREATE INDEX index7 ON item (i_id, i_price, i_data, i_name);",
            "CREATE INDEX index8 ON warehouse (w_id, w_tax, w_state, w_city);",
            "CREATE INDEX index9 ON district (d_id, d_w_id, d_next_o_id, d_state);",
            "CREATE INDEX index10 ON stock (s_i_id);",
            "CREATE INDEX index11 ON new_order (no_d_id, no_w_id, no_o_id);",
            "CREATE INDEX index12 ON customer (c_id, c_w_id, c_d_id, c_data);",
        ],
        "human": [
            "CREATE INDEX index0 ON customer (c_w_id, c_d_id, c_last);",
        ]
    }[args.index_tuner]

    indexes += [f"VACUUM ANALYZE {tbl}" for tbl in spec.workload.tables]
    indexes += ["CHECKPOINT"]

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
            "min_wal_size = 2GB",
            "max_wal_size = 8GB",
            "max_worker_processes = 20",
            "max_parallel_workers_per_gather = 4",
            "max_parallel_workers = 20",
            "max_parallel_maintenance_workers = 4",
        ],
        indexes,
    )

    env._shutdown_postgres()
    local["tar"]["cf", f"{spec.postgres_data}.tgz", "-C", spec.postgres_path, spec.postgres_data_folder].run()

    with open("out.txt", "w") as f:
        for i in range(3):
            # Restore and dump.
            env._restore_last_snapshot()
            env.shift_state(None, [], dump_page_cache=True)

            results = Path(f"/home/wz2/mythril/exps_tpcc/validations/base/result{i}")
            if results.exists():
                shutil.rmtree(results)
            results.mkdir(parents=True, exist_ok=True)

            assert spec.workload._execute_benchbase(spec, results)
            tps = reward(results)
            print(tps)
            f.write(f"{tps}\n")
