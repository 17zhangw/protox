import pandas as pd
import json
import tqdm
import argparse
import logging
from pathlib import Path
from dateutil.parser import parse

import sys
sys.path.append("/home/wz2/mythril")

from envs.spec import Spec
from envs.pg_env import PostgresEnv


def apply_best(env, config):
    config_changes = []
    sql_commands = []
    per_query_knobs = {}
    for k, v in config.items():
        if not k.startswith("knob"):
            continue

        if "." in k:
            k = k.split(".")[-1]

        if k.startswith("Q"):
            if "scanmethod" in k:
                per_query_knobs[k] = int("Index" in v)
            elif "parallel_rel" in k:
                if v.lower() == "sentinel":
                    per_query_knobs[k] = 0
                else:
                    per_query_knobs[k] = env.action_space.get_knob_space().knobs[k].values.index(v)
            elif isinstance(v, str):
                per_query_knobs[k] = int(v == "on")
            else:
                per_query_knobs[k] = v

            continue

        set_str = f"{k} = {v}"
        config_changes.append(set_str)

    for k, v in config.items():
        if not k.startswith("index"):
            continue

        table, col = k.split("index.")[-1].split(".")
        if v == "off":
            sql_commands.append(f"DROP INDEX IF EXISTS {table}_{col}")
        else:
            sql_commands.append(f"CREATE INDEX IF NOT EXISTS {table}_{col} ON {table} ({col})")

    workload_qdir = None
    workload_qfiles = None
    if "query.file_id" in config:
        file_id = config["query.file_id"]
        if int(file_id) != 0:
            workload_qdir = args.qdir / str(file_id)
            workload_qfiles = args.qdir / f"{file_id}.txt"

    # Shift state.
    env.shift_state(config_changes, sql_commands, dump_page_cache=True, ignore_error=True)
    return (workload_qdir, workload_qfiles), per_query_knobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UniTune Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--res", type=Path)
    parser.add_argument("--qdir", type=Path)

    parser.add_argument("--output", type=str)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--workload-timeout", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--blocklist", default="")
    args = parser.parse_args()

    spec = Spec(
        agent_type=None,
        horizon=5,
        seed=0,
        config_path=args.config,
        benchmark_config_path=args.benchmark_config,
        workload_timeout=0)

    knobs = spec.action_space.get_knob_space().knobs

    env = PostgresEnv(
        spec,
        horizon=0,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)
    env.restore_pristine_snapshot()

    with open(args.input, "r") as f:
        loglines = f.readlines()

    for line in loglines:
        if "Init record" in line:
            # Get the start record.
            s = line.split("[tpch_test][")[-1].split("]:")[0]
            time_start = parse(s)

    timeout = args.workload_timeout
    run_cost = args.workload_timeout
    best_result = {"all": args.workload_timeout}
    with open(args.res, "r") as f:
        lines = f.readlines()[1:]

    lines_best = [line[5:] for line in lines if "best|" in line]
    current_step = 0
    global_config = {}
    run_data = []
    for i, line in enumerate(tqdm.tqdm(lines_best)):
        tmp = eval(line.strip())
        config = tmp['configuration']
        time_cost = tmp['time_cost']

        if time_cost[0] < run_cost:
            cur_time = None
            while True:
                log = loglines[0]
                if "INFO, Iteration " in log and "time_cost" in log:
                    log_time_cost = float(log.split("time_cost ")[-1].split("space_cost")[0].strip())
                    log_latmean = float(log.split("lat_mean ")[-1].split("timestamp")[0].strip())
                    if time_cost[0] == log_time_cost and time_cost[1] == log_latmean:
                        cur_time = parse(log.split("[tpch_test][")[-1].split("]:")[0])
                        loglines = loglines[1:]
                        break
                loglines = loglines[1:]
            assert cur_time is not None

            global_config.update(config)
            workload_qdir, per_query_knobs = apply_best(env, global_config)
            # This assert checks that the knob exists.
            per_query_knobs = {k: (knobs[k], v) for k, v in per_query_knobs.items()}

            # Obtain the samples.
            samples = []
            for _ in range(args.samples):
                runtime = spec.workload._execute_workload(
                    connection=env.connection,
                    ql_knobs=per_query_knobs,
                    workload_timeout=timeout,
                    workload_qdir=workload_qdir,
                    blocklist=args.blocklist.split(","))
                samples.append(runtime)

                if runtime >= args.workload_timeout:
                    break

                if args.samples == 2 and runtime >= timeout:
                    break
                elif args.samples > 2 and len(samples) >= 2 and runtime >= timeout:
                    break

            if max(samples) < timeout:
                timeout = max(samples)
            run_cost = time_cost[0]

            data = {
                "step": current_step,
                "orig_time_cost": time_cost[0],
                "time_since_start": (cur_time - time_start).total_seconds(),
            }
            data.update({f"runtime{i}": s for i, s in enumerate(samples)})
            run_data.append(data)

            current_step += 1
            logging.info(f"Step {current_step} ({time_cost[0]}): {samples}")

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
