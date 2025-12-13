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
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--stream", type=int, default=1)
    args = parser.parse_args()

    spec = Spec(
        agent_type=None,
        seed=0,
        config_path=args.config_file,
        benchmark_config_path="configs/benchmark/dsb_s1.yaml",
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

    if args.sf == 1:
        idxs = [
            "CREATE INDEX index0 ON catalog_returns (cr_returned_date_sk)",
            "CREATE INDEX index1 ON catalog_sales (cs_bill_customer_sk)",
            "CREATE INDEX index3 ON catalog_sales (cs_ship_date_sk)",
            "CREATE INDEX index4 ON catalog_sales (cs_sold_date_sk)",
            "CREATE INDEX index5 ON inventory (inv_item_sk)",
            "CREATE INDEX index6 ON store_returns (sr_cdemo_sk)",
            "CREATE INDEX index7 ON store_returns (sr_returned_date_sk)",
            "CREATE INDEX index8 ON store_sales (ss_customer_sk)",
            "CREATE INDEX index9 ON store_sales (ss_sold_date_sk)",
            "CREATE INDEX index10 ON web_sales (ws_sold_date_sk)",
            "CREATE INDEX index11 ON web_sales (ws_ship_date_sk)",
        ]
    elif args.sf == 10:
        if args.stream == 1 or args.stream == 2:
            idxs = [
                "CREATE INDEX index0 ON catalog_returns (cr_returned_date_sk)",
                "CREATE INDEX index1 ON catalog_sales (cs_bill_customer_sk)",
                "CREATE INDEX index2 ON catalog_sales (cs_ship_customer_sk)",
                "CREATE INDEX index3 ON catalog_sales (cs_ship_date_sk)",
                "CREATE INDEX index4 ON catalog_sales (cs_sold_date_sk)",
                "CREATE INDEX index5 ON inventory (inv_item_sk)",
                "CREATE INDEX index6 ON store_returns (sr_cdemo_sk)",
                "CREATE INDEX index7 ON store_returns (sr_returned_date_sk)",
                "CREATE INDEX index8 ON store_sales (ss_customer_sk)",
                "CREATE INDEX index9 ON store_sales (ss_sold_date_sk)",
                "CREATE INDEX index10 ON web_returns (wr_returned_date_sk)",
                "CREATE INDEX index11 ON web_sales (ws_sold_date_sk)",
            ]
        elif args.stream == 3:
            idxs = [
                "CREATE INDEX index0 ON catalog_returns (cr_returned_date_sk)",
                "CREATE INDEX index1 ON catalog_sales (cs_bill_customer_sk)",
                "CREATE INDEX index2 ON catalog_sales (cs_ship_customer_sk)",
                "CREATE INDEX index3 ON catalog_sales (cs_sold_date_sk)",
                "CREATE INDEX index4 ON inventory (inv_item_sk)",
                "CREATE INDEX index5 ON store_returns (sr_cdemo_sk)",
                "CREATE INDEX index6 ON store_returns (sr_returned_date_sk)",
                "CREATE INDEX index7 ON store_sales (ss_customer_sk)",
                "CREATE INDEX index8 ON store_sales (ss_sold_date_sk)",
                "CREATE INDEX index9 ON web_returns (wr_returned_date_sk)",
                "CREATE INDEX index10 ON web_sales (ws_bill_customer_sk)",
                "CREATE INDEX index11 ON web_sales (ws_sold_date_sk)",
            ]
        elif args.stream == 5:
            idxs = [
                "CREATE INDEX index0 ON catalog_returns (cr_returned_date_sk)",
                "CREATE INDEX index1 ON catalog_sales (cs_bill_customer_sk)",
                "CREATE INDEX index2 ON catalog_sales (cs_ship_customer_sk)",
                "CREATE INDEX index3 ON catalog_sales (cs_ship_date_sk)",
                "CREATE INDEX index4 ON catalog_sales (cs_sold_date_sk)",
                "CREATE INDEX index5 ON inventory (inv_item_sk)",
                "CREATE INDEX index6 ON store_returns (sr_cdemo_sk)",
                "CREATE INDEX index7 ON store_returns (sr_returned_date_sk)",
                "CREATE INDEX index8 ON store_sales (ss_customer_sk)",
                "CREATE INDEX index9 ON store_sales (ss_sold_date_sk)",
                "CREATE INDEX index10 ON store_sales (ss_ticket_number)",
                "CREATE INDEX index11 ON web_returns (wr_returned_date_sk)",
                "CREATE INDEX index12 ON web_sales (ws_sold_date_sk)",
            ]
    elif args.sf == 20:
        idxs = [
            "CREATE INDEX index0 ON catalog_returns (cr_returned_date_sk)",
            "CREATE INDEX index1 ON catalog_sales (cs_bill_customer_sk)",
            "CREATE INDEX index2 ON catalog_sales (cs_ship_customer_sk)",
            "CREATE INDEX index3 ON catalog_sales (cs_ship_date_sk)",
            "CREATE INDEX index4 ON catalog_sales (cs_sold_date_sk)",
            "CREATE INDEX index6 ON store_returns (sr_cdemo_sk)",
            "CREATE INDEX index7 ON store_returns (sr_returned_date_sk)",
            "CREATE INDEX index8 ON store_sales (ss_customer_sk)",
            "CREATE INDEX index9 ON store_sales (ss_sold_date_sk)",
            "CREATE INDEX index10 ON web_returns (wr_returned_date_sk)",
            "CREATE INDEX index11 ON web_sales (ws_ship_date_sk)",
        ]
    else:
        assert False

    env.shift_state(
        [
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
        ], idxs)

    with open(f"/mnt/nvme0n1/wz2/noisepage/{spec.postgres_data_folder}/postgresql.auto.conf", "r") as f:
        lines = []
        for line in f:
            if "shared_preload" not in line:
                lines.append(line.strip())
