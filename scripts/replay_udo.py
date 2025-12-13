import logging
import os
import json
import pandas as pd
import tqdm
import argparse
import gymnasium as gym
from pathlib import Path
from dateutil.parser import parse

import sys
sys.path.append("/home/wz2/mythril")

from envs.spec import Spec
from envs.pg_env import PostgresEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UDO Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=str)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--workload-timeout", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--blocklist", default="")
    args = parser.parse_args()

    spec = Spec(
        agent_type=None,
        seed=0,
        config_path=args.config,
        benchmark_config_path=args.benchmark_config,
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

    def run_sample(knobs, idxlist, timeout):
        samples = []
        for _ in range(args.samples):
            runtime = spec.workload._execute_workload(
                connection=env.connection,
                workload_timeout=timeout,
                blocklist=args.blocklist.split(","))
            samples.append(runtime)

            if runtime == args.workload_timeout:
                # Give up if we've hit the workload timeout.
                break

            if args.samples == 2 and runtime >= timeout:
                break
            elif args.samples > 2 and len(samples) >= 2 and runtime >= timeout:
                break
        return samples

    timeout = args.workload_timeout
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
            env.shift_state(cc, sql_commands, dump_page_cache=True)

            past_configs[key] = run_sample(sql_commands, cc, timeout)
            samples = past_configs[key]

            if max(samples) < timeout:
                timeout = max(samples)

        samples = {f"runtime{i}": s for i, s in enumerate(samples)}
        print(samples)
        data.update(samples)
        run_data.append(data)

    # Shift into state.
    print("Evaluating first best state.")
    env.restore_pristine_snapshot()
    env.shift_state(best_met1_cc, best_met1_sc, dump_page_cache=True)
    samples_1 = run_sample(best_met1_cc, best_met1_sc, timeout)
    print("And we are done!")

    data = {
        "step": len(run_data),
        "idxlist": ",".join(best_met1_sc),
        "knobs": ",".join(best_met1_cc),
        "time_since_start": 28800.,
    }
    data.update({f"runtime{i}": s for i, s in enumerate(samples_1)})
    run_data.append(data)

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
