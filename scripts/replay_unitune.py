import psycopg
import numpy as np
import shutil
import yaml
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
from scripts.parse_unituneconfig import parse_unituneconfig


def run(args):
    bcf = args.benchmark_config
    if args.specialization or args.sample_specialization is not None:
        special = args.specialization
        if special is None:
            special = str(args.sample_specialization)
        elif "." in special:
            special = special.split("_0.")[0]

        with open(args.benchmark_config, "r") as f:
            data = yaml.safe_load(f)
            data["mythril"]["query_spec"]["execute_query_directory"] = special
            data["mythril"]["query_spec"]["execute_query_order"] = f"{special}/d_order.txt"
        with open("/tmp/benchmark.yaml", "w") as f:
            yaml.dump(data, f)
        bcf = "/tmp/benchmark.yaml"

        if args.sample_specialization:
            tdir = Path("/tmp") / Path(special).stem
            if tdir.exists():
                shutil.rmtree(tdir)

            shutil.copytree(special, tdir)
            with open(f"{special}/d_order.txt") as f:
                dorder = f.read()
            with open(f"/tmp/dord.txt", "w") as f:
                f.write(dorder)

    with open(bcf) as f:
        data = yaml.safe_load(f)
        eqord = data["mythril"]["query_spec"]["execute_query_order"]

    spec = Spec(
        agent_type=None,
        horizon=5,
        seed=0,
        config_path=args.config,
        benchmark_config_path=bcf,
        workload_timeout=0)

    knobs = spec.action_space.get_knob_space().knobs

    env = PostgresEnv(
        spec,
        horizon=5,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)
    env.restore_pristine_snapshot()
    #env.connection = psycopg.connect("host=localhost port=5432 dbname=benchbase", autocommit=True, prepare_threshold=None)

    configs = parse_unituneconfig(args.input, args.workload_timeout, env, wordfile=eqord)
    timeout = args.workload_timeout
    run_data = []
    for current_step, config in enumerate(configs):
        config_changes = config[0]
        sql_commands = config[1]
        workload_qdir = config[2]
        per_query_knobs = config[3]

        # Shift state.
        env.shift_state(config_changes, sql_commands, dump_page_cache=True, ignore_error=True)
        exist_mem_gb, used_mem_gb = env.workload.compute_used_mem(env.connection, False, None, None)

        if args.sample_specialization:
            # This assert checks that the knob exists.
            per_query_knobs = {k: (knobs[k], v) for k, v in per_query_knobs.items() if k in knobs}
            if workload_qdir is not None and workload_qdir[0] is not None:
                for wsql in Path(workload_qdir[0]).glob("*.sql"):
                    shutil.copy(wsql, f"/tmp/{Path(special).stem}")
                workload_qir = (Path(f"/tmp/{Path(special).stem}"), Path(f"/tmp/dord.txt"))
        else:
            per_query_knobs = {k: (knobs[k], v) for k, v in per_query_knobs.items()}

        # Obtain the samples.
        samples = []
        plans = None
        if args.plans:
            plans = {}
            spec.workload._execute_workload(
                connection=env.connection,
                ql_knobs=per_query_knobs,
                workload_timeout=timeout,
                workload_qdir=workload_qdir,
                blocklist=args.blocklist.split(",") if len(args.blocklist) > 0 else [],
                plans=plans,
            )

        qid_runtimes = None
        qmetric_runtime = 1e6
        qmetric_data = None
        for _ in range(args.samples):
            qid_runtime_l = {}
            runtime, qmetric_datum = spec.workload._execute_workload(
                connection=env.connection,
                ql_knobs=per_query_knobs,
                workload_timeout=timeout,
                workload_qdir=workload_qdir,
                blocklist=args.blocklist.split(",") if len(args.blocklist) > 0 else [],
                qid_runtimes=qid_runtime_l,
                env_spec=spec,
                metrics=True,
            )
            samples.append(runtime)

            if runtime < qmetric_runtime:
                qmetric_runtime = runtime
                qmetric_data = qmetric_datum

            if runtime >= args.workload_timeout:
                break

            if args.samples == 2 and runtime >= timeout:
                break
            elif args.samples > 2 and len(samples) >= 2 and runtime >= timeout:
                break

            qid_runtimes = qid_runtime_l

        if args.metrics and qmetric_data is not None:
            mout = Path(args.input) / "qdata" / f"{current_step}"
            mout.mkdir(parents=True, exist_ok=True)
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
                with open(mout / f"{qid}.metrics.json", "w") as f:
                    f.write(json.dumps(output, indent=4))

        if plans is not None:
            pfolder = args.input / f"plans" / f"{current_step}"
            pfolder.mkdir(parents=True, exist_ok=True)
            with open(pfolder / f"{current_step}plans.json", "w") as f:
                f.write(json.dumps(plans, indent=2))

            with open(pfolder / f"{current_step}rt.json", "w") as f:
                f.write(json.dumps(qid_runtimes, indent=2))
        elif not args.metrics and max(samples) < timeout:
            timeout = min(timeout, max(samples) * 1.5)

        data = {
            "step": current_step,
            "orig_time_cost": config[4],
            "time_since_start": config[5].total_seconds(),
            "exist_mem_mb": exist_mem_gb * 1024,
            "used_mem_mb": used_mem_gb * 1024,
        }
        data.update({f"runtime{i}": s for i, s in enumerate(samples)})
        run_data.append(data)
        logging.info(f"Step {current_step} ({config[4]}): {samples}")

    if not args.plans and not args.metrics:
        if not args.sample_specialization:
            # Output.
            pd.DataFrame(run_data).to_csv(args.output, index=False)
        else:
            pd.DataFrame(run_data).to_csv(args.sample_out, index=False)
    else:
        pd.DataFrame(run_data).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="UniTune Replay")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=str)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--specialization")
    parser.add_argument("--workload-timeout", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--blocklist", default="")
    parser.add_argument("--plans", action="store_true")
    parser.add_argument("--metrics", action="store_true")

    parser.add_argument("--sample-specialization", default=None)
    parser.add_argument("--sample-out", default=None)
    args = parser.parse_args()

    if not Path(args.output).exists():
        run(args)
