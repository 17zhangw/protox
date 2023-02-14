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
from scripts.replay_bao_utils import run_bao, run_autosteer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--dta", action="store_true")
    parser.add_argument("--bao-file", default=None)
    parser.add_argument("--autosteer-file", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--benchmark-config-file", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--workload-timeout", type=int, default=600)
    parser.add_argument("--sample-interval", type=int, default=900)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--stream", type=int, default=1)
    args = parser.parse_args()

    benchmark_config_path = args.benchmark_config_file

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

    if args.dta:
        idxs = [
            "create index index0 on store_sales ( ss_sold_date_sk asc, ss_net_profit asc, ss_sales_price asc, ss_hdemo_sk asc, ss_store_sk asc, ss_cdemo_sk asc, ss_addr_sk asc) include(ss_item_sk,ss_customer_sk,ss_promo_sk,ss_ticket_number,ss_quantity,ss_wholesale_cost,ss_list_price,ss_ext_sales_price,ss_ext_wholesale_cost,ss_coupon_amt) ;",
            "create index index1 on store_sales ( ss_sold_date_sk asc, ss_cdemo_sk asc, ss_store_sk asc, ss_item_sk asc) include(ss_quantity,ss_list_price,ss_sales_price,ss_coupon_amt) ;",
            "create index index2 on store_sales ( ss_sold_date_sk asc, ss_item_sk asc, ss_ticket_number asc, ss_customer_sk asc, ss_store_sk asc) include(ss_promo_sk,ss_ext_sales_price,ss_net_profit) ;",
            "create index index3 on store_sales ( ss_customer_sk asc) include(ss_sold_date_sk,ss_item_sk,ss_ticket_number,ss_quantity,ss_sales_price) ;",
            "create index index4 on store_sales ( ss_sold_date_sk asc, ss_item_sk asc, ss_ticket_number asc, ss_customer_sk asc, ss_store_sk asc) include(ss_net_profit);",
            "create index index5 on store_sales ( ss_sold_date_sk asc, ss_store_sk asc) include(ss_item_sk,ss_sales_price,ss_ext_sales_price) ;",
            "create index index6 on store_sales ( ss_item_sk asc, ss_ticket_number asc, ss_customer_sk asc, ss_sold_date_sk asc, ss_store_sk asc) ;",
            "create index index7 on store_sales ( ss_sold_date_sk asc, ss_item_sk asc, ss_addr_sk asc) include(ss_ext_sales_price) ;",
            "create index index8 on store_sales ( ss_sold_date_sk asc, ss_addr_sk asc, ss_item_sk asc) include(ss_ext_sales_price) ;",
            "create index index9 on store_sales ( ss_sold_date_sk asc, ss_item_sk asc) include(ss_quantity,ss_list_price) ;",
            "create index index10 on store_sales ( ss_sold_date_sk asc, ss_customer_sk asc, ss_item_sk asc, ss_ticket_number asc) ;",
            "create index index11 on store_sales ( ss_customer_sk asc, ss_sold_date_sk asc) include(ss_ext_sales_price) ;",
            "create index index12 on store_sales ( ss_item_sk asc, ss_ticket_number asc, ss_customer_sk asc) ;",
            "create index index13 on store_sales ( ss_ticket_number asc, ss_item_sk asc) ;",
            "create index index14 on catalog_sales ( cs_sold_date_sk asc, cs_item_sk asc, cs_bill_cdemo_sk asc, cs_bill_customer_sk asc) include(cs_ship_date_sk,cs_bill_hdemo_sk,cs_order_number,cs_quantity,cs_list_price,cs_sales_price,cs_coupon_amt,cs_net_profit) ;",
            "create index index15 on catalog_sales ( cs_promo_sk asc, cs_bill_hdemo_sk asc, cs_ship_date_sk asc, cs_bill_cdemo_sk asc, cs_sold_date_sk asc, cs_item_sk asc) include(cs_call_center_sk,cs_ship_mode_sk,cs_warehouse_sk,cs_order_number,cs_quantity) ;",
            "create index index16 on catalog_sales ( cs_sold_date_sk asc) include(cs_bill_customer_sk,cs_item_sk,cs_order_number,cs_quantity,cs_list_price,cs_ext_sales_price) ;",
            "create index index17 on catalog_sales ( cs_ship_date_sk asc, cs_call_center_sk asc, cs_ship_mode_sk asc, cs_warehouse_sk asc) include(cs_item_sk,cs_order_number) ;",
            "create index index18 on catalog_sales ( cs_sold_date_sk asc, cs_item_sk asc, cs_bill_customer_sk asc) include(cs_order_number,cs_net_profit) ;",
            "create index index19 on catalog_sales ( cs_ship_date_sk asc) include(cs_sold_date_sk,cs_call_center_sk,cs_ship_mode_sk,cs_warehouse_sk) ;",
            "create index index20 on catalog_sales ( cs_item_sk asc, cs_bill_customer_sk asc, cs_sold_date_sk asc) include(cs_net_profit) ;",
            "create index index21 on catalog_sales ( cs_item_sk asc, cs_order_number asc) include(cs_ext_list_price) ;",
            "create index index22 on catalog_sales ( cs_sold_date_sk asc, cs_bill_customer_sk asc) ;",
            "create index index23 on catalog_sales ( cs_sold_date_sk asc) include(cs_bill_customer_sk) ;",
            "create index index24 on web_sales ( ws_ship_date_sk asc, ws_order_number asc, ws_ship_addr_sk asc, ws_web_site_sk asc) include(ws_warehouse_sk,ws_ext_ship_cost,ws_net_profit) ;",
            "create index index25 on web_sales ( ws_sold_date_sk asc, ws_item_sk asc, ws_bill_customer_sk asc) include(ws_order_number) ;",
            "create index index26 on web_sales ( ws_sold_date_sk asc) include(ws_bill_addr_sk,ws_ext_sales_price) ;",
            "create index index27 on web_sales ( ws_order_number asc) include(ws_warehouse_sk) ;",
            "create index index28 on web_sales ( ws_sold_date_sk asc, ws_bill_customer_sk asc) ;",
            "create index index29 on store_returns ( sr_returned_date_sk asc) include(sr_item_sk,sr_customer_sk,sr_ticket_number,sr_net_loss) ;",
            "create index index30 on store_returns ( sr_cdemo_sk asc) include(sr_returned_date_sk,sr_item_sk,sr_ticket_number,sr_return_quantity) ;",
            "create index index31 on store_returns ( sr_returned_date_sk asc) include(sr_item_sk,sr_customer_sk,sr_ticket_number) ;",
            "create index index32 on customer ( c_first_name asc, c_last_name asc) ;",
            "create index index33 on customer ( c_customer_sk asc, c_current_addr_sk asc) ;",
            "create index index34 on item ( i_item_sk asc) include(i_item_id,i_item_desc) ;",
            "create index index35 on item ( i_category asc, i_class asc, i_item_sk asc) ;",
            "create index index36 on item ( i_color asc) ;",
            "create index index37 on item ( i_item_id asc, i_item_sk asc) ;",
            "create index index38 on date_dim ( d_year asc, d_month_seq asc, d_moy asc, d_date_sk asc) ;",
            "create index index39 on date_dim ( d_year asc, d_moy asc, d_date_sk asc) ;",
            "create index index40 on date_dim ( d_date_sk asc, d_year asc, d_moy asc) ;",
            "create index index41 on date_dim ( d_year asc, d_qoy asc, d_date_sk asc) ;",
            "create index index42 on date_dim ( d_moy asc, d_year asc, d_date_sk asc) ;",
            "create index index43 on date_dim ( d_month_seq asc) include(d_date_sk) ;",
            "create index index44 on date_dim ( d_year asc, d_date_sk asc) ;",
            "create index index45 on date_dim ( d_year asc, d_moy asc) ;",
            "create index index46 on date_dim ( d_month_seq asc, d_date asc) ;",
            "create index index47 on store ( s_store_sk asc) include(s_store_id,s_store_name) ;",
            "create index index48 on store ( s_state asc, s_store_sk asc) ;",
            "create index index49 on store_sales ( ss_customer_sk asc, ss_store_sk asc, ss_item_sk asc, ss_sold_date_sk asc) include(ss_ext_sales_price) ;",
            "create index index50 on customer ( c_birth_month asc, c_current_addr_sk asc) ;",
        ]
    else:
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
        ],
        idxs,
        dump_page_cache=True)

    if args.autosteer_file is not None:
        samples = run_autosteer(env, args.num_samples, args.workload_timeout, args.autosteer_file, benchmark_config_path)
        print(samples)
        with open("out.txt", "w") as f:
            for s in samples:
                f.write(f"{s}\n")
    elif args.bao_file is not None:
        # quarter hour interval.
        run_bao(env, args.num_samples, args.workload_timeout, args.bao_file, benchmark_config_path, args.sample_interval, blocklist=["query081"])
    else:
        with open("out.txt", "w") as f:
            for _ in range(3):
                time = spec.workload._execute_workload(connection=env.connection, workload_timeout=600)
                print(time)
                f.write(f"{time}\n")
