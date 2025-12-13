from sklearn.preprocessing import StandardScaler
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


def euclidean_dist(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
        vector1 (array-like): First vector.
        vector2 (array-like): Second vector.

    Returns:
        float: Cosine similarity value.
    """
    # Convert inputs to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Compute the dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Return cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Handle zero-vector case
    return dot_product / (magnitude1 * magnitude2)


def load_existing(output):
    epart = Path(output).parts[-1]
    exist_map = {}
    for i in range(4):
        wlpart = epart.split("_wl_")
        sbaseline = wlpart[1].split("_baseline")[1]
        exist = Path(output).parent / (wlpart[0] + "_wl_" + f"{i}" + "_baseline" + sbaseline)
        if exist.exists():
            df = pd.read_csv(exist)
            df["dt"] = np.concatenate([[df.time_since_start[0]], (df.time_since_start[1:].values - df.time_since_start[0:-1].values)])
            for t in df.itertuples():
                exist_map[t.base_input] = (
                    t.dt,
                    t.runtime0,
                    t.runtime1 if "runtime1" in df else np.nan,
                    t.runtime2 if "runtime2" in df else np.nan,
                )
    return exist_map


def gogo(args):
    if "dsb_" in str(args.input):
        bcf = "configs/benchmark/dsb_revise.yaml"
        specialization = "queries/specializations/" + str(args.input).split("_sf")[0].split("/")[-1]
        benchmark = "dsb"
    elif "job_" in str(args.input):
        bcf = "configs/benchmark/job_full.yaml"
        specialization = None
        benchmark = "job"
    elif "tpch_" in str(args.input):
        bcf = "configs/benchmark/tpch.yaml"
        specialization = None
        benchmark = "tpch"

    # Try to yoink existing repo datas...
    exist_map = load_existing(args.output)

    blocklist = Path(args.input).parent.parent / "blocklist.txt"
    bsqls = set()
    if blocklist.exists():
        with open(blocklist) as f:
            bsqls = set([l.strip() for l in f.readlines() if len(l.strip()) > 0])

    start_time = time.time()
    for workload_input in args.workload_input.split(","):
        inputs = [q for q in Path(workload_input).rglob("qdata")]
        print(f"Parsing {workload_input} (#{len(inputs)})")
        repo_metrics = {}
        for iqdata in inputs:
            with open(iqdata.parent.parent.parent / f"{benchmark}.yaml") as f:
                data = yaml.safe_load(f)["mythril"]["query_spec"]
                qorder = data.get("execute_query_order", data["query_order"])
            with open(qorder) as f:
                orig_qmap = {k.split(",")[0]: k.split(",")[1].strip() for k in f.readlines()}

            metrics = {}
            for qd in iqdata.rglob("*.metrics.json"):
                if "/baselines/dsb/" in str(qd) and ("/dsb_baseline/baseline/" not in str(qd)):
                    # Make sure we only look at <12 hours...
                    if "2024-12-31" not in str(qd):
                        continue

                    hr = int(str(qd).split("_")[-1].split("-")[0])
                    if hr >= 19:
                        continue
                elif "/job_30hr_baseline/" in str(qd) and ("/job_30hr_baseline/baseline/" not in str(qd)):
                    if "2025-01-01" not in str(qd):
                        continue
                elif "/tpch_30hr_baseline/" in str(qd) and ("/tpch_30hr_baseline/baseline/" not in str(qd)):
                    if ("2025-01-02" not in str(qd) and "2025-01-03" not in str(qd)):
                        continue

                    hr = int(str(qd).split("_")[-1].split("-")[0])
                    if "2025-01-03" in str(qd) and hr >= 7:
                        continue

                qid = qd.parts[-1].split(".metrics.json")[0]
                assert qid in orig_qmap, print(qd, orig_qmap.keys())
                if orig_qmap[qid] in bsqls:
                    # Do not add anything in the blocklist.
                    continue

                with open(qd) as f:
                    qdm = json.load(f)
                    for k, v in qdm.items():
                        if k in ["lsc", "flattened"]:
                            continue
                        if k in metrics:
                            metrics[k] += v
                        else:
                            metrics[k] = v
            repo_metrics[str(iqdata.parent)] = metrics

        for i, wbase in enumerate(args.workload_baseline.split(",")):
            print(f"Parsing {wbase}")
            wbasem = Path(wbase) / "baseline" / "workload.jsonl"
            metrics = {}
            with open(wbasem) as f:
                for l in f.readlines():
                    wbase = json.loads(l)
                    qpart = Path(wbase["qfile"]).parts[-1]
                    if qpart in bsqls:
                        # Do not add anything in the blocklist.
                        continue

                    wbasej = wbase["metrics"]
                    del wbasej["lsc"]
                    for k, v in wbasej.items():
                        if k in ["lsc", "flattened"]:
                            continue
                        if k in metrics:
                            metrics[k] += v[0]
                        else:
                            metrics[k] = v[0]
            repo_metrics[f"baseline{i}"] = metrics

    # Get all keys.
    rmkeys = set()
    for v in repo_metrics.values():
        rmkeys.update(v.keys())
    # Zero-fill.
    for _, v in repo_metrics.items():
        for rmk in rmkeys:
            if rmk not in v:
                v[rmk] = 0.

    input_metrics = {rmk: 0.0 for rmk in rmkeys}
    with open(args.input) as f:
        for l in f.readlines():
            qdm = json.loads(l)["metrics"]
            for k, v in qdm.items():
                if k in ["lsc", "flattened"]:
                    continue
                if k in input_metrics:
                    input_metrics[k] += v[0]
                else:
                    input_metrics[k] = v[0]

    # Get sorted keys...
    rmkeys = list(sorted(rmkeys))
    repo_metrics = {k: [v[k] for k in rmkeys] for k, v in repo_metrics.items()}
    input_metrics = [input_metrics[k] for k in rmkeys]

    if args.normalize:
        # Normalize the metrics.
        scaler = StandardScaler()
        scaler.fit([v for v in repo_metrics.values()])
        repo_metrics = {k: scaler.transform([v])[0] for k, v in repo_metrics.items()}
        input_metrics = scaler.transform([input_metrics])[0]

    if args.dist_fn == "euclidean":
        rsims = [(k, euclidean_dist(v, input_metrics)) for k, v in repo_metrics.items()]
        rsims = sorted(rsims, key=lambda x: x[1])[:args.top_k]

    else:
        rsims = [(k, cosine_similarity(v, input_metrics)) for k, v in repo_metrics.items()]
        # Put highest cosine similarity first..
        rsims = sorted(rsims, key=lambda x: x[1], reverse=True)[:args.top_k]

    # Write the config yaml file.
    with open(f"configs/config.yaml") as f:
        mythril = yaml.safe_load(f)
        mythril["mythril"]["benchbase_config_path"] = f"benchmark.xml"
        mythril["mythril"]["verbose"] = True
        mythril["mythril"]["postgres_path"] = args.pg_path
        mythril["mythril"]["index_vae_metadata"]["index_vae"] = False

        if args.snapshot is not None:
            mythril["mythril"]["data_snapshot_path"] = args.snapshot
        elif "dsb_" in str(args.input):
            mythril["mythril"]["data_snapshot_path"] = "data/dsb_sf10.tgz"
        elif "job_" in str(args.input):
            mythril["mythril"]["data_snapshot_path"] = "data/job.tgz"
        elif "tpch_" in str(args.input):
            mythril["mythril"]["data_snapshot_path"] = "data/tpch_sf10.tgz"

        if "constraints" not in mythril["mythril"]:
            mythril["mythril"]["constraints"] = {
                "memory_constraint": False,
                "partial_account": True,
                "memory_budgetb": "1GB",
                "query_constraint": False,
                "query_tolerance": "10%",
                "reject": False,
            }
    with open(f"configs/config.yaml2", "w") as f:
        yaml.dump(mythril, stream=f, default_flow_style=False)

    if specialization:
        with open(bcf, "r") as f:
            data = yaml.safe_load(f)
            data["mythril"]["query_spec"]["execute_query_directory"] = str(specialization)
            data["mythril"]["query_spec"]["execute_query_order"] = f"{specialization}/d_order.txt"
        with open("/tmp/benchmark.yaml", "w") as f:
            yaml.dump(data, f)
        bcf = "/tmp/benchmark.yaml"

    # Generate the database.
    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=5,
        config_path="configs/config.yaml2",
        benchmark_config_path=bcf,
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=5,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    def run_sample(action, timeout):
        # This should reliably check that we are loading the correct knobs...
        ql_knobs = spec.action_space.get_knob_space().get_query_level_knobs(action) if action is not None else {}

        samples = []
        for i in range(args.samples):
            runtime  = spec.workload._execute_workload(
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

    run_data = []
    timeout = args.workload_timeout
    pbar = tqdm.tqdm(total=len(rsims), leave=False, dynamic_ncols=True, miniters=1)
    current_elapsed = time.time() - start_time
    for current_step, (repo, _) in enumerate(rsims):
        if repo in exist_map:
            run_samples = samples = exist_map[repo][1:]
            logging.info(f"Existing Samples: {samples}")

            data = {
                "step": current_step,
                "base_input": repo,
                "time_since_start": current_elapsed + exist_map[repo][0],
            }
            samples = {f"runtime{i}": s for i, s in enumerate(samples)}
            data.update(samples)
            run_data.append(data)

            if len(run_samples) > 0:
                if max(run_samples) < timeout:
                    # Provide a little wiggle room...
                    timeout = max(run_samples) * 1.2
            pbar.update(1)
            current_elapsed += exist_map[repo][0]
            continue

        index_sqls = []
        knobs = {}
        current_start = time.time()

        if not repo.startswith("baseline"):
            knobs, index_sqls = derive_repo_config(env, repo, args.generalize_template, benchmark)

        # Always reset to clean state.
        env.restore_pristine_snapshot()
        env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
        spec.workload.reset()

        if len([k for k in knobs.keys() if not k.startswith("Q")]):
            cc, _ = env.action_space.get_knob_space().generate_plan(knobs, no_check=True)
        else:
            cc = []
        env.shift_state(cc, index_sqls, ignore_error=args.ignore_error, dump_page_cache=True)

        # Run...
        run_samples = samples = run_sample(knobs, timeout)
        logging.info(f"New Samples: {samples}")
        round_dt = time.time() - current_start

        data = {
            "step": current_step,
            "base_input": repo,
            "time_since_start": current_elapsed + round_dt,
        }
        samples = {f"runtime{i}": s for i, s in enumerate(samples)}
        data.update(samples)
        run_data.append(data)

        if len(run_samples) > 0:
            if max(run_samples) < timeout:
                # Provide a little wiggle room...
                timeout = max(run_samples) * 1.2
        pbar.update(1)
        current_elapsed += round_dt

    # Output.
    pd.DataFrame(run_data).to_csv(args.output, index=False)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="WorkloadMapping")
    parser.add_argument("--workload-input", type=str)
    parser.add_argument("--workload-baseline", type=str)

    parser.add_argument("--input", type=Path)
    parser.add_argument("--workload-timeout", type=int)
    parser.add_argument("--samples", type=int)

    parser.add_argument("--output", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--generalize-template", action="store_true")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--dist-fn", default="euclidean")

    parser.add_argument("--blocklist", default="")
    parser.add_argument("--pg-path", type=str, default="/mnt/nvme0n1/wz2/noisepage")
    parser.add_argument("--snapshot", default=None)
    args = parser.parse_args()

    while True:
        pargs = DotDict(vars(args))
        Path(args.output).mkdir(parents=True, exist_ok=True)

        runs = Path(pargs.input).rglob("workload.jsonl")
        for run in tqdm.tqdm([f for f in runs], leave=False):
            out_name = "{}_{}".format(run.parts[-3], run.parts[-2])
            if args.normalize:
                out_name += "_norm"
            if args.generalize_template:
                out_name += "_gentemplate"
            out_name += f"_k{args.top_k}"

            if Path(f"{args.output}/{out_name}.csv").exists():
                continue

            print(f"Parsing {run}")
            new_args = pargs
            new_args.input = run
            new_args.output = f"{pargs.output}/{out_name}.csv"
            gogo(new_args)

        break
