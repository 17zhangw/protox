import ast
import numpy as np
import psycopg
import shutil
import datetime
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

import sys
sys.path.append("/home/wz2/mythril")

from envs.spec import Spec
from envs.pg_env import PostgresEnv
from envs.spaces.knob import CategoricalKnob
from utils.derive_repo_config import derive_repo_config

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _compute_index_memory(conn, indexes, planfile):
    used_indexes = set()
    def _extract(p):
        if "Index Name" in p:
            used_indexes.add(p["Index Name"])
        if "Plan" in p:
            _extract(p["Plan"])
        if "Plans" in p:
            for pp in p["Plans"]:
                _extract(pp)

    # Get the used indexes.
    assert planfile.exists()
    with open(planfile, "r") as f:
        plans = [json.loads(p.strip()) for p in f.readlines() if p.startswith("{")]
    [_extract(p) for p in plans]

    indexsizes = {}
    for index in indexes:
        idxname = index.split(" ON ")[0].split(" on ")[0].split("INDEX ")[-1].split("index ")[-1].strip()
        idxsize = [r for r in conn.execute("SELECT pg_relation_size('{}', 'main')".format(idxname))][0][0]
        indexsizes[idxname] = (idxsize, idxname in used_indexes)
        indexsizes[index] = (idxsize, idxname in used_indexes)
    return indexsizes



def _run_plan_indexes(run_plans):
    with open(run_plans) as f:
        data = [json.loads(l) for l in f.readlines() if l.startswith("[") or l.startswith("{")]

    gidxes = set()
    def _recurse(p):
        nonlocal gidxes
        if "Plan" in p:
            _recurse(p["Plan"])
        if "Plans" in p:
            for pp in p["Plans"]:
                _recurse(pp)
        if "Index Name" in p:
            gidxes.add(p["Index Name"])

    [_recurse(d) for d in data]
    return gidxes


def baseline_run(args):
    with open(f"{args.input}/config.yaml") as f:
        mythril = yaml.safe_load(f)
        mythril["mythril"]["benchbase_config_path"] = f"{args.input}/benchmark.xml"
        mythril["mythril"]["verbose"] = True
        mythril["mythril"]["postgres_path"] = args.pg_path

        if "constraints" not in mythril["mythril"]:
            mythril["mythril"]["constraints"] = {
                "memory_constraint": False,
                "partial_account": True,
                "memory_budgetb": "1GB",
                "query_constraint": False,
                "query_tolerance": "10%",
                "reject": False,
            }

    with open(f"{args.input}/config.yaml2", "w") as f:
        yaml.dump(mythril, stream=f, default_flow_style=False)

    with open(f"{args.input}/stdout", "r") as f:
        config = f.readlines()[0]
        config = eval(config.split("HPO Configuration: ")[-1])
        horizon = config["horizon"]
        per_query_timeout = config["mythril_args"]["timeout"]
    timeout = args.workload_timeout

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

    def run_sample(action, timeout):
        # This should reliably check that we are loading the correct knobs...
        ql_knobs = spec.action_space.get_knob_space().get_query_level_knobs(action) if action is not None else {}

        samples = []
        for i in range(args.samples):
            runtime = spec.workload._execute_workload(
                connection=env.connection,
                workload_timeout=timeout,
                ql_knobs=ql_knobs,
                env_spec=spec,
                blocklist=[l for l in args.blocklist.split(",") if len(l) > 0],
            )
            samples.append(runtime)
            logging.info(f"Runtime: {runtime}")

            if runtime >= args.workload_timeout:
                break

            if runtime >= timeout:
                break

        return samples

    with open(args.input / "params.json", "r") as f:
        params = json.load(f)

    bc = params["baseline_config"]
    assert bc is not None and bc != ""

    knobs, index_sqls = derive_repo_config(env, bc, False, args.benchmark)

    # Reset snapshot.
    env.action_space.reset(connection=env.connection, workload=env.workload)
    cc, _ = env.action_space.get_knob_space().generate_plan(knobs, no_check=True)
    env.shift_state(cc, index_sqls, ignore_error=args.ignore_error, dump_page_cache=True)

    # Get samples.
    samples = run_sample(knobs, timeout)
    reward = samples[0]
    logging.info(f"Original Runtime: {reward}. New Samples: {samples}")

    data = {
        "step": 0,
        "orig_cost": reward,
        "time_since_start": 0,
    }
    samples = {f"runtime{i}": s for i, s in enumerate(samples)}
    data.update(samples)
    run_data = [data]

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
    env.close()


def gogo(args):
    maximal = args.maximal
    maximal_only = args.maximal_only
    threshold = args.threshold
    threshold_percent = args.threshold_percent
    if args.load_baseline:
        baseline_run(args)
        return

    assert not (threshold > 0 and threshold_percent > 0)
    shadow_drop = False
    with open(f"{args.input}/config.yaml") as f:
        mythril = yaml.safe_load(f)
        mythril["mythril"]["benchbase_config_path"] = f"{args.input}/benchmark.xml"
        mythril["mythril"]["verbose"] = True
        mythril["mythril"]["postgres_path"] = args.pg_path

        if "constraints" not in mythril["mythril"]:
            mythril["mythril"]["constraints"] = {
                "memory_constraint": False,
                "partial_account": True,
                "memory_budgetb": "1GB",
                "query_constraint": False,
                "query_tolerance": "10%",
                "reject": False,
            }

        if mythril["mythril"]["constraints"]["memory_constraint"]:
            if mythril["mythril"]["constraints"].get("partial_account", True):
                shadow_drop = True

    with open(f"{args.input}/config.yaml2", "w") as f:
        yaml.dump(mythril, stream=f, default_flow_style=False)

    bcf = f"{args.input}/{args.benchmark}.yaml"
    with open(bcf) as f:
        mythril = yaml.safe_load(f)
        if args.sample_specialization is not None:
            special = str(args.sample_specialization)
            mythril["mythril"]["query_spec"]["execute_query_directory"] = special
            mythril["mythril"]["query_spec"]["execute_query_order"] = f"{special}/d_order.txt"
            with open("/tmp/benchmark.yaml", "w") as f:
                yaml.dump(mythril, f)
            bcf = "/tmp/benchmark.yaml"

    if args.alternate:
        horizon = args.horizon
        per_query_timeout = args.pqt
    else:
        with open(f"{args.input}/stdout", "r") as f:
            config = f.readlines()[0]
            config = eval(config.split("HPO Configuration: ")[-1])
            horizon = config["horizon"]

        with open(f"{args.input}/stdout", "r") as f:
            for line in f:
                if "HPO Configuration: " in line:
                    hpo = eval(line.split("HPO Configuration: ")[-1].strip())
                    per_query_timeout = hpo["mythril_args"]["timeout"]

    folders = []
    filename = "output.log" if args.alternate else "stderr"
    last_evaluation = None
    prior_eval_state = None
    prior_reward_calc = None
    start_time = None
    with open(f"{args.input}/{filename}", "r") as f:
        for line in f:
            if "Baseilne Metric" in line or "Baseline Benchmark" in line:
                start_time = parse(line.split("INFO:")[-1].split(" Baseilne Metric")[0].split("Baseline Benchmark")[0])
            elif "Benchmark iteration with metric" in line:
                prior_eval_state = line
            elif "[reward_calc]" in line:
                prior_reward_calc = line
            else:
                if "mv" in line and "repository" in line and ("baseline" not in line):
                    assert prior_eval_state is not None
                    assert prior_reward_calc is not None

                    illegal = ast.literal_eval(prior_reward_calc.split(" ")[-1])
                    assert isinstance(illegal, bool)
                    metric = float(prior_eval_state.split("Benchmark iteration with metric ")[-1].split(" (")[0])
                    qtimeout = eval(prior_eval_state.split("q_timeout: ")[-1].split(")")[0])

                    repo = eval(line.split("Running ")[-1])[-1]
                    last_folder = repo.split("/")[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0])
                    last_evaluation = time_since_start
                    if start_time is not None:
                        if (time_since_start - start_time).total_seconds() < args.cutoff * 3600 or args.cutoff == 0:
                            folders.append((last_folder, metric, qtimeout, illegal))
                    else:
                        folders.append((last_folder, metric, qtimeout, illegal))
                    prior_eval_state = None
                    prior_reward_calc = None

    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=horizon,
        config_path=f"{args.input}/config.yaml2",
        benchmark_config_path=bcf,
        workload_timeout=0)

    # Only apply threshold if time is less than.
    threshold_limit = last_evaluation - datetime.timedelta(seconds=int(args.threshold_limit * 3600))

    # Get the minimum reward.
    # folder: <last_folder, metric, qtimeout, illegal>
    nonillegals = [f[1] for f in folders if not f[-1]]
    min_reward = min(nonillegals) if len(nonillegals) > 0 else None
    if maximal:
        target = [(r[2] or r[3]) for r in folders if r[1] == min_reward]
        if len(target) == 0 or target[0]:
            # Don't use maximal if the min maximal is timed out.
            # Don't threshold either.
            threshold = 0
            threshold_percent = 0.0
            maximal = False
            # Reject maximal only.
            maximal_only = False
            logging.warn("Maximal disabled.")

        else:
            logging.info(f"Maximal found: {min_reward}")

    env = PostgresEnv(
        spec,
        horizon=horizon,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    if not args.simulated:
        env.restore_pristine_snapshot()
        env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
        spec.workload.reset()

    def run_sample(action, timeout, step_counter, out_metrics_dir, sdindexes=None):
        # This should reliably check that we are loading the correct knobs...
        ql_knobs = spec.action_space.get_knob_space().get_query_level_knobs(action) if action is not None else {}

        output_file = None
        if new_args.output_artifacts is not None:
            output_file = open(new_args.output_artifacts / f"step{step_counter}.plans.new", "w")

        # Get the existing indexes...
        dbindexes = [r[0] for r in env.connection.execute("""
            SELECT indexname from pg_index, pg_class, pg_indexes
            where pg_index.indexrelid = pg_class.oid and pg_class.relnamespace = 2200
              and pg_indexes.indexname = pg_class.relname
              and not pg_index.indisprimary
              and not pg_index.indisunique
        """)]
        env.connection.execute("CREATE EXTENSION IF NOT EXISTS hypopg")

        samples = []
        qmetric_runtime = 1e6
        qmetric_data = None
        for i in range(args.samples):
            if sdindexes is not None:
                # Log out the hiding.
                hide_sqls = [f"SELECT hypopg_hide_index('{idx}'::regclass);" for idx in dbindexes if idx not in sdindexes]
                for hsql in hide_sqls:
                    logging.debug(f"Hiding: {hsql}")
                [env.connection.execute(sql) for sql in hide_sqls]

            runtime, qmetric_data = spec.workload._execute_workload(
                connection=env.connection,
                workload_timeout=timeout,
                ql_knobs=ql_knobs,
                env_spec=spec,
                blocklist=[l for l in args.blocklist.split(",") if len(l) > 0],
                metrics=True)
            samples.append(runtime)
            logging.info(f"Runtime: {runtime}")

            if runtime < qmetric_runtime:
                qmetric_runtime = runtime
                qmetric_data = qmetric_data

            if runtime >= args.workload_timeout:
                break

            if runtime >= timeout:
                break

        if sdindexes is not None:
            env.connection.execute("SELECT hypopg_reset()")

        if qmetric_data is not None and out_metrics_dir is not None:
            (Path(out_metrics_dir) / "qdata").mkdir(parents=True, exist_ok=True)
            for qid, metric in qmetric_data.items():
                def flatten(d):
                    flat = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            flat[k] = flatten(v)
                        elif isinstance(v, np.ndarray):
                            flat[k] = float(v[0])
                        elif isinstance(v, np.ScalarType):
                            if isinstance(v, str):
                                flat[k] = v
                            else:
                                flat[k] = float(v)
                        else:
                            flat[k] = v
                    return flat

                output = flatten(metric)
                output["flattened"] = True
                with open(f"{out_metrics_dir}/qdata/{qid}.metrics.json", "w") as f:
                    f.write(json.dumps(output, indent=4))

        if output_file is not None:
            output_file.close()

        return samples

    fnames = set([f[0] for f in folders])
    run_data = []
    pbar = tqdm.tqdm(total=len(fnames) + 1, leave=False, dynamic_ncols=True, miniters=1)
    with open(f"{args.input}/{filename}", "r") as f:
        current_step = 0

        start_time = None
        timeout = args.workload_timeout
        cur_reward_max = timeout
        selected_action_knobs = None
        noop_index = False
        maximal_repo = None
        existing_indexes = []

        if threshold_percent > 0:
            # Gate the reward max and evaluate everything from there.
            cur_reward_max = min_reward * (1. + threshold_percent)

        for line in f:
            # Keep going until we've found the start.
            if "Baseilne Metric" in line or "Baseline Benchmark" in line:
                start_time = parse(line.split("INFO:")[-1].split(" Baseilne Metric")[0].split(" Baseline Benchmark")[0])
                pbar.update(1)

            elif "Selected action: " in line:
                act = eval(line.split("Selected action: ")[-1])
                kact = act[0]
                if not args.sample_specialization:
                    selected_action_knobs = env.action_space.get_knob_space().from_jsonable(kact)[0]
                noop_index = "NOOP" in act[1][0]

            elif (maximal and ("mv" in line and "repository" in line)):
                maximal_repo = line

            elif (maximal and "Found new maximal state " in line) or (not maximal and ("mv" in line and "repository" in line and "baseline" not in line)):
                if "mv" in line and "repository" in line:
                    repo = eval(line.split("Running ")[-1])[-1]
                    time_since_start = parse(line.split("DEBUG:")[-1].split(" Running")[0])
                    pbar.update(1)
                elif "Found new maximal state " in line:
                    repo = eval(maximal_repo.split("Running ")[-1])[-1]
                    time_since_start = parse(maximal_repo.split("DEBUG:")[-1].split(" Running")[0])
                    maximal_repo = None
                    pbar.update(1)

                if args.parquet_scan is not None:
                    paths = [p for p in Path(f"{args.input}/{repo}").rglob(f"{args.parquet_scan}.parquet")]
                    if len(paths) == 0:
                        pbar.update(1)
                        continue

                run_folder = repo.split("/")[-1]
                assert run_folder in fnames, print(run_folder, fnames)
                _, fmetric, ftimeout, fillegal = [f for f in folders if f[0] == run_folder][0]

                # Get the evaluation reward.
                reward = pd.read_csv(f"{args.input}/{repo}/run.raw.csv")
                assert len(reward.columns) == 6
                has_timeout = (reward["Latency (microseconds)"].max() / 1e6) == per_query_timeout
                reward = reward["Latency (microseconds)"].sum() / 1e6
                is_illegal = (not np.isclose(fmetric, reward)) and fillegal
                assert (not is_illegal) or (fmetric > reward), print(is_illegal, fmetric, reward)
                assert reward > 0

                if ((not maximal_only and reward < cur_reward_max) or reward == min_reward) and (not maximal or not has_timeout) and (not is_illegal):
                    index_sqls = []
                    knobs = None
                    insert_knobs = False

                    sdindexes = None
                    if shadow_drop:
                        # Load the shadow drop indexes we want.
                        assert Path(f"{args.input}/{repo}/run.plans").exists()
                        sdindexes = _run_plan_indexes(f"{args.input}/{repo}/run.plans")

                    if Path(f"{args.input}/{repo}/act_sql.txt").exists():
                        knobs, index_sqls = derive_repo_config(
                            env,
                            f"{args.input}/{repo}/",
                            False,
                            args.benchmark,
                            noop_index=noop_index,
                        )
                        if not args.sample_specialization:
                            # Assert the system knobs are fully loaded.
                            for k in selected_action_knobs:
                                if not k.startswith("Q"):
                                    assert k in knobs

                        assert len(index_sqls) > 0
                        assert len(knobs) > 0

                    execute_sqls = []
                    for index_sql in index_sqls:
                        if index_sql in existing_indexes:
                            continue
                        execute_sqls.append(index_sql)

                    for index_sql in existing_indexes:
                        if index_sql not in index_sqls:
                            # Only allow dropping if we are able to back-track or if OLTP.
                            indexname = index_sql.split("CREATE INDEX")[-1].split(" ON ")[0]
                            execute_sqls.append(f"DROP INDEX IF EXISTS {indexname}")

                    cc = []
                    if (not args.simulated) and knobs is not None:
                        # Reset snapshot.
                        env.action_space.reset(connection=env.connection, workload=env.workload)
                        if args.sample_specialization:
                            cc, _ = env.action_space.get_knob_space().generate_plan(knobs, no_check=True)
                        else:
                            cc, _ = env.action_space.get_knob_space().generate_plan(selected_action_knobs)
                        env.shift_state(cc, execute_sqls, ignore_error=args.ignore_error, dump_page_cache=True)
                    existing_indexes = index_sqls

                    # Make sure the actual index exists.
                    if sdindexes is not None:
                        for sdindex in sdindexes:
                            assert "_pkey" in sdindex or any([f" {sdindex} " in ei for ei in existing_indexes]), print(repo, sdindex, existing_indexes)

                    old_plans = Path(f"{args.input}/{repo}/run.plans")
                    out_dir = f"{args.input}/{repo}/" if args.parquet_scan is not None else None
                    indexsizes = _compute_index_memory(env.connection, existing_indexes, old_plans) if not args.simulated else {}
                    if args.memory_only:
                        with open(Path(out_dir) / "idxsize.json", "w") as f:
                            f.write(json.dumps(indexsizes, indent=4))

                    if not args.simulated and (not args.memory_only or (args.parquet_scan and not Path(f"{args.input}/{repo}/qdata").exists())):
                        # Get samples.
                        out_dir = f"{args.input}/{repo}/" if args.parquet_scan is not None else None
                        run_samples = samples = run_sample(knobs, timeout, current_step, out_dir, sdindexes=sdindexes)
                        logging.info(f"Original Runtime: {reward} (timeout {has_timeout}). New Samples: {samples}")
                    else:
                        run_samples = samples = [reward, reward]

                    if args.output_artifacts is not None:
                        if old_plans.exists():
                            shutil.copy(old_plans, new_args.output_artifacts / f"step{current_step}.plans.old")

                    data = {
                        "step": current_step,
                        "orig_cost": reward,
                        "nopkidxmem_gb": sum([v[0] for k, v in indexsizes.items() if ("CREATE INDEX" in k and "pkey" not in k and "UNIQUE" not in k)]) / 1024. / 1024. / 1024.,
                        "time_since_start": (time_since_start - start_time).total_seconds() if start_time is not None else 0,
                    }
                    samples = {f"runtime{i}": s for i, s in enumerate(samples)}
                    data.update(samples)
                    run_data.append(data)

                    current_step += 1

                    if maximal:
                        if (not has_timeout) or (max(run_samples) < timeout):
                            # Apply a tolerance..
                            # If we've timed out, only apply threshold only if we've found a strictly better config.
                            tdegree = threshold if threshold > 0 else 0
                            apply_threshold = tdegree if time_since_start < threshold_limit else 0.2 # Make sure it's not just noise... brr
                            cur_reward_max = reward - apply_threshold
                    elif args.force_threshold:
                            cur_reward_max = reward

                    if len(run_samples) > 0:
                        if max(run_samples) < timeout:
                            # Provide a little wiggle room...
                            timeout = max(run_samples) * 1.2

                if run_folder in folders and run_folder == folders[-1]:
                    break
                elif maximal_only and reward == min_reward:
                    break

    if not args.sample_specialization:
        if not Path(args.output).exists():
            # Output.
            pd.DataFrame(run_data).to_csv(args.output, index=False)
    else:
        pd.DataFrame(run_data).to_csv(args.sample_out, index=False)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UDO Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--workload-timeout", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-limit", type=float, default=0)
    parser.add_argument("--threshold-percent", type=float, default=0.0)
    parser.add_argument("--force-threshold", action="store_true")
    parser.add_argument("--maximal", action="store_true")
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--maximal-only", action="store_true")
    parser.add_argument("--alternate", action="store_true", default=False)
    parser.add_argument("--pqt", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--cutoff", type=float, default=0)
    parser.add_argument("--blocklist", default="")
    parser.add_argument("--pg-path", type=str, default="/mnt/nvme0n1/wz2/noisepage")
    parser.add_argument("--oltp", action="store_true")

    parser.add_argument("--parquet-scan", default=None)
    parser.add_argument("--memory-only", action="store_true")

    parser.add_argument("--output-artifacts", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="out.csv")
    parser.add_argument("--ignore-error", action="store_true")

    parser.add_argument("--sample-specialization", default=None)
    parser.add_argument("--sample-out", default=None)
    parser.add_argument("--load-baseline", action="store_true")
    args = parser.parse_args()

    while True:
        pargs = DotDict(vars(args))
        output_path = args.output_path

        runs = Path(pargs.input).rglob("config.yaml")
        runs = sorted([f for f in runs if not (f.parent / output_path).exists()])
        for run in tqdm.tqdm([f for f in runs], leave=False):
            if args.simulated:
                adjust_output = run.parent / "out_simulated.csv"
            else:
                adjust_output = run.parent / args.output_path

            if adjust_output.exists():
                continue

            print(f"Parsing {run.parent}")
            new_args = pargs
            new_args.input = run.parent
            new_args.output = adjust_output
            if args.output_artifacts is not None:
                new_args.output_artifacts = run.parent / args.output_artifacts
                if Path(new_args.output_artifacts).exists():
                    shutil.rmtree(new_args.output_artifacts)
                Path(new_args.output_artifacts).mkdir(parents=True, exist_ok=True)

            gogo(new_args)

        break
