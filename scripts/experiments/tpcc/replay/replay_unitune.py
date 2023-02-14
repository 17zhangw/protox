import shutil
import logging
import time
import yaml
import os
import json
import pandas as pd
import tqdm
import argparse
import gymnasium as gym
from plumbum import local
from pathlib import Path
from dateutil.parser import parse

import sys
sys.path.append("/home/wz2/mythril")

from envs.spec import Spec
from envs.pg_env import PostgresEnv
from envs.reward import RewardUtility

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def gogo(args):
    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=5,
        config_path="configs/tpcc.yaml",
        benchmark_config_path="configs/benchmark/tpcc.yaml",
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=5,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    env.restore_pristine_snapshot()
    env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
    spec.workload.reset()

    num_lines = 0
    with open(f"{args.input}/tpch_test.log", "r") as f:
        for line in f:
            if "Init record" in line:
                num_lines += 1
            elif "timestamp" in line:
                num_lines += 1
    pbar = tqdm.tqdm(total=num_lines)

    run_data = []
    with open(f"{args.input}/tpch_test.log") as f:
        start_found = False
        start_time = None
        current_step = 0
        cur_reward_max = 0
        reward_util = RewardUtility("tps", "multiplier", 1.)

        for line in f:
            line = line.strip()

            # Keep going until we've found the start.
            if not start_found:
                if "Init record" in line:
                    start_found = True
                    start_time = parse(line.split("[tpch_test][")[-1].split("]:")[0])
                    pbar.update(1)
                continue

            if "timestamp" in line:
                lat_mean = line.split("time_cost ")[-1].split("space_cost")[0]
                lat_mean = -float(lat_mean.strip())
                cur_time = parse(line.split("[tpch_test][")[-1].split("]:")[0])

                if lat_mean > cur_reward_max:
                    ts = line.split("timestamp ")[-1]

                    with open(f"{args.input}/results/{ts}.auto.conf", "r") as ccfile:
                        ccs = [l.strip() for l in ccfile.readlines()]
                        ccs = [cc for cc in ccs if len(cc) > 0]

                    with open(f"{args.input}/results/{ts}.indexes.txt", "r") as idxs:
                        idxs = [l.strip() for l in idxs.readlines()]
                        idxs = [idx for idx in idxs if len(idx) > 0]
                        idxs = [(idx.split(".")[0], idx.split(".")[-1].split(" =")[0]) for idx in idxs]
                        idxs = [f"CREATE INDEX advisor_{t}_{c} ON {t} ({c})" for (t, c) in idxs]

                    sqls = []
                    checkpoint = False
                    with open(f"{args.input}/results/{ts}.pg_class.csv", "r") as pgc:
                        for pgc_record in pgc:
                            tbl = pgc_record.split(",")[2]
                            if "fillfactor" in pgc_record:
                                ff = int(pgc_record.split("fillfactor=")[-1].split("']")[0])
                                sqls.append(f"ALTER TABLE {tbl} SET (fillfactor = {ff})")
                                sqls.append(f"VACUUM FULL {tbl}")
                                checkpoint = True
                            elif tbl in spec.tables:
                                sqls.append(f"ALTER TABLE {tbl} SET (fillfactor = 100)")
                                sqls.append(f"VACUUM FULL {tbl}")
                                checkpoint = True

                    if checkpoint:
                        for tbl in spec.tables:
                            sqls.append(f"VACUUM ANALYZE {tbl}")
                        sqls.append(f"CHECKPOINT")
                    sqls = sqls + idxs

                    env.restore_pristine_snapshot()
                    env.shift_state(ccs, sqls)
                    env._shutdown_postgres()
                    local["tar"]["cf", f"{spec.postgres_data}.tgz", "-C", spec.postgres_path, spec.postgres_data_folder].run()

                    samples = []
                    for _ in range(3):
                        env._restore_last_snapshot()
                        env.shift_state(None, [], dump_page_cache=True)

                        results = Path(f"/tmp/results")
                        if results.exists():
                            shutil.rmtree(results)
                        results.mkdir(parents=True, exist_ok=True)

                        assert spec.workload._execute_benchbase(spec, results)
                        tps, p99, avg = reward_util.parse_tps_avg_p99_for_metric(results)
                        samples.append((tps, p99, avg))

                    logging.info(f"Original Runtime: {lat_mean}. New Samples: {samples}")

                    data = {
                        "step": current_step,
                        "orig_cost": lat_mean,
                        "time_since_start": (cur_time - start_time).total_seconds(),
                    }
                    data.update({f"runtime{i}": s[0] for i, s in enumerate(samples)})
                    data.update({f"p99{i}": s[1] for i, s in enumerate(samples)})
                    data.update({f"avg{i}": s[2] for i, s in enumerate(samples)})
                    run_data.append(data)
                    current_step += 1

                    cur_reward_max = lat_mean

                pbar.update(1)

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UniTune Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()

    while True:
        pargs = DotDict(vars(args))
        output_path = "out.csv"

        runs = Path(pargs.input).rglob("tpch_test.log")
        runs = sorted([f for f in runs if not (f.parent / output_path).exists()])
        for run in tqdm.tqdm([f for f in runs], leave=False):
            adjust_output = run.parent / "out.csv"
            if adjust_output.exists():
                continue

            print(f"Parsing {run.parent}")
            new_args = pargs
            new_args.input = run.parent
            new_args.output = adjust_output
            gogo(new_args)

        break
