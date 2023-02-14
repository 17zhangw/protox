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
    parser.add_argument("--bao-file", default=None)
    args = parser.parse_args()

    benchmark_config_path = "configs/benchmark/job_full.yaml"
    if args.bao_file is not None:
        benchmark_config_path = "configs/benchmark/job_full.yaml.bao"

    spec = Spec(
        agent_type=None,
        seed=0,
        config_path="configs/job.yaml",
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

    env.shift_state(
        [
        ],
        [
        ],
        dump_page_cache=True)

    with open("/mnt/nvme0n1/wz2/noisepage/pgdata/postgresql.auto.conf", "r") as f:
        lines = []
        for line in f:
            if "shared_preload" not in line:
                lines.append(line.strip())

    with open("/mnt/nvme0n1/wz2/noisepage/pgdata/postgresql.auto.conf", "w") as f:
        for line in lines:
            f.write(line + "\n")

        f.write("shared_preload_libraries = 'pg_bao'")
