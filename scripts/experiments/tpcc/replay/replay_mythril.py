import shutil
import math
import logging
import time
import yaml
import os
import json
import pandas as pd
import tqdm
import argparse
import gymnasium as gym
from pathlib import Path
from dateutil.parser import parse
from plumbum import local

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
    with open(f"{args.input}/config.yaml") as f:
        mythril = yaml.safe_load(f)
        mythril["mythril"]["benchbase_config_path"] = f"/home/wz2/mythril/{args.input}/benchmark.xml"
        mythril["mythril"]["verbose"] = True

    with open(f"{args.input}/config.yaml2", "w") as f:
        yaml.dump(mythril, stream=f, default_flow_style=False)

    with open(f"{args.input}/stdout", "r") as f:
        config = f.readlines()[0]
        config = eval(config.split("HPO Configuration: ")[-1])
        horizon = config["horizon"]

    with open(f"{args.input}/stdout", "r") as f:
        for line in f:
            if "HPO Configuration: " in line:
                hpo = eval(line.split("HPO Configuration: ")[-1].strip())
                per_query_timeout = hpo["mythril_args"]["timeout"]

    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=horizon,
        config_path=f"{args.input}/config.yaml2",
        benchmark_config_path=f"{args.input}/{args.benchmark}.yaml",
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=horizon,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    env.restore_pristine_snapshot()
    env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
    spec.workload.reset()

    num_lines = 0
    max_reward = 0
    with open(f"{args.input}/stderr", "r") as f:
        for line in f:
            if "Baseilne Metric" in line:
                num_lines += 1
            elif "mv" in line and "repository" in line:
                num_lines += 1
            elif "Benchmark iteration with metric" in line:
                metric = line.split("with metric ")[-1].split(" (reward")[0]
                if float(metric) > max_reward:
                    max_reward = math.floor(float(metric))

    run_data = []
    reward_util = RewardUtility("tps", "multiplier", 1.)
    pbar = tqdm.tqdm(total=num_lines)
    with open(f"{args.input}/stderr", "r") as f:
        current_step = 0

        start_found = False
        start_time = None
        cur_reward_max = 0
        selected_action_knobs = None
        noop_index = False

        for line in f:
            # Keep going until we've found the start.
            if not start_found:
                if "Baseilne Metric" in line:
                    start_found = True
                    start_time = parse(line.split("INFO:")[-1].split(" Baseilne Metric")[0])
                    pbar.update(1)
                continue

            elif "Selected action: " in line:
                act = eval(line.split("Selected action: ")[-1])
                selected_action_knobs = env.action_space.get_knob_space().from_jsonable(act[0])[0]
                noop_index = "NOOP" in act[1][0]

            elif "mv" in line and "repository" in line:
                repo = eval(line.split("Running ")[-1])[-1]
                time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0])

                summary = [r for r in Path(f"{args.input}/{repo}").rglob("*.summary.json")][0]
                with open(summary, "r") as f:
                    reward = json.load(f)["Throughput (requests/second)"]

                if cur_reward_max < reward:
                    index_sqls = []
                    knobs = {}
                    insert_knobs = False

                    with open(f"{args.input}/{repo}/act_sql.txt", "r") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) == 0:
                                insert_knobs = True
                            elif not insert_knobs:
                                index_sqls.append(line)
                            else:
                                k, v = line.split(" = ")
                                knobs[k] = float(v)

                    assert len(index_sqls) > 0
                    assert len(knobs) > 0
                    with open(f"{args.input}/{repo}/prior_state.txt", "r") as f:
                        prior_states = eval(f.read())
                        all_sc = prior_states[1]
                        if not noop_index:
                            all_sc.extend(index_sqls)
                        index_sqls = all_sc

                    # Reset snapshot.
                    env.restore_pristine_snapshot()
                    env.action_space.reset(connection=env.connection, workload=env.workload)
                    cc, cmds = env.action_space.get_knob_space().generate_plan(selected_action_knobs if selected_action_knobs else {}, force=True)
                    cmds = cmds + index_sqls
                    env.shift_state(cc, cmds)

                    # Make a copy.
                    env._shutdown_postgres()
                    local["tar"]["cf", f"{spec.postgres_data}.tgz", "-C", spec.postgres_path, spec.postgres_data_folder].run()

                    samples = []
                    for i in range(3):
                        # Restore and dump.
                        env._restore_last_snapshot()
                        env.shift_state(None, [], dump_page_cache=True)

                        # Then execute.
                        results = Path(f"/tmp/results")
                        if results.exists():
                            shutil.rmtree(results)
                        results.mkdir(parents=True, exist_ok=True)

                        assert spec.workload._execute_benchbase(spec, results)
                        tps, p99, avg = reward_util.parse_tps_avg_p99_for_metric(results)
                        samples.append((tps, p99, avg))

                    logging.info(f"Original Runtime: {reward}. New Samples: {samples}")

                    data = {
                        "step": current_step,
                        "orig_cost": reward,
                        "time_since_start": (time_since_start - start_time).total_seconds(),
                    }
                    data.update({f"runtime{i}": s[0] for i, s in enumerate(samples)})
                    data.update({f"p99{i}": s[1] for i, s in enumerate(samples)})
                    data.update({f"avg{i}": s[2] for i, s in enumerate(samples)})
                    run_data.append(data)

                    current_step += 1

                    # Apply a tolerance..
                    cur_reward_max = reward
                    if args.tolerance is not None:
                        cur_reward_max += args.tolerance
                        cur_reward_max = min(cur_reward_max, max_reward)
            pbar.update(1)

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UDO Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--tolerance", type=int)
    args = parser.parse_args()

    while True:
        pargs = DotDict(vars(args))
        output_path = "out.csv"

        runs = Path(pargs.input).rglob("config.yaml")
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
