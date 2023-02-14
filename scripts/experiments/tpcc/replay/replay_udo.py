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

    run_data = []
    past_configs = {}

    # Figure out how many runs.
    num_lines = 0
    start_time = None
    last_interval_check = None
    best_met1 = False
    best_met1_sc = []
    best_met1_cc = []
    past_configs = {}

    valid_configs = []
    reward_util = RewardUtility("tps", "multiplier", 1.)
    with open(args.input, "r") as f:
        for line in f:
            if "start to tuning your database" in line:
                start_found = True
                start_time = parse(line.split(" [")[0])
                last_interval_check = start_time
            elif "new_best_config" in line or "best_eps_so_far" in line or "interval_check" in line:
                assert "frozenset" not in line
                current_time = parse(line.split(" [")[0])

                if "interval_check" in line and (current_time - last_interval_check).total_seconds() < 900:
                    # Don't perform the check if it's not within 15 minutes.
                    continue
                elif "interval_check" in line:
                    last_interval_check = current_time

                time_since_start = (current_time - start_time).total_seconds()

                if "new_best_config" in line:
                    data = eval(line.split("new_best_config: ")[-1])
                elif "interval_check" in line:
                    data = eval(line.split("interval_check: ")[-1])
                else:
                    data = eval(line.split("best_eps_so_far: ")[-1])

                sql_commands = data["indices"]
                cc = [c.split("set ")[-1].split(";")[0] for c in data["sys"]]
                key = (";".join(sorted(sql_commands)), ";".join(sorted(cc)))
                valid_configs.append((time_since_start, sql_commands, cc, key))

            elif "M1: " in line:
                best_met1 = True

            elif best_met1:
                if "CREATE INDEX" in line:
                    best_met1_sc.append(line.split(" INFO : ")[-1].strip())
                elif "set" in line:
                    best_met1_cc.append(line.split(" INFO : ")[-1].strip().split("set ")[-1].split(";")[0])

    key = (";".join(sorted(best_met1_sc)), ";".join(sorted(best_met1_cc)))
    valid_configs.append((28800., best_met1_sc, best_met1_cc, key))

    for i, config in tqdm.tqdm(enumerate(valid_configs), total=len(valid_configs)):
        time_since_start, sql_commands, cc, key = config
        data = {
            "step": i,
            "idxlist": ",".join(sql_commands),
            "knobs": ",".join(cc),
            "time_since_start": time_since_start,
        }

        if key in past_configs:
            samples = past_configs[key]
        else:
            # Shift into state.
            env.restore_pristine_snapshot()

            actual_sql_commands = []
            checkpoint = False
            for c in cc:
                if "fillfactor" in c:
                    comps = c.split(" = ")
                    tbl = comps[0].split("_fillfactor")[0]
                    actual_sql_commands.append(f"ALTER TABLE {tbl} SET (fillfactor = {comps[1]})")
                    actual_sql_commands.append(f"VACUUM FULL {tbl}")
                    checkpoint = True

            if checkpoint:
                for tbl in spec.tables:
                    actual_sql_commands.append(f"VACUUM ANALYZE {tbl}")
                actual_sql_commands.append("CHECKPOINT")
            actual_sql_commands = actual_sql_commands + sql_commands
            cc = [c for c in cc if "fillfactor" not in c]
            env.shift_state(cc, actual_sql_commands)

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
                print(tps)

            assert key not in past_configs
            past_configs[key] = samples
            samples = past_configs[key]
            print(samples)

        data.update({f"runtime{i}": s[0] for i, s in enumerate(samples)})
        data.update({f"p99{i}": s[1] for i, s in enumerate(samples)})
        data.update({f"avg{i}": s[2] for i, s in enumerate(samples)})
        run_data.append(data)

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UDO Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()

    while True:
        pargs = DotDict(vars(args))
        output_path = "out.csv"

        runs = sorted([f for f in Path(pargs.input).rglob("*.log")])
        idx = 0
        for run in tqdm.tqdm([f for f in runs], leave=False):
            adjust_output = run.parent / f"out{idx}.csv"
            if adjust_output.exists():
                continue

            print(f"Parsing {run}")
            new_args = pargs
            new_args.input = run
            new_args.output = adjust_output
            gogo(new_args)
            idx += 1

        break
