import importlib
import json
import logging
import psycopg
import yaml
import copy
import os
import sys
import argparse

try:
    sys.path.append("/home/wz2/mythril/Auto-Steer")
    os.chdir("/home/wz2/mythril/Auto-Steer")
    from main import run
except:
    pass

sys.path.remove("/home/wz2/mythril/Auto-Steer")
sys.path.append("/home/wz2/mythril")
os.chdir("/home/wz2/mythril")
# Unfortunate hack.
mod = importlib.import_module("utils")
importlib.reload(mod)

from envs.spec import Spec
from envs.pg_env import PostgresEnv
from scripts.parse_hyrise import parse_hyrise_configs
from scripts.parse_lambdaconfig import parse_lambdaconfig
from sqlalchemy import create_engine, event
from dateutil.parser import parse
from pathlib import Path
import pandas as pd


class DotDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ReplayLambda")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--benchmark-config-file", required=True)
    parser.add_argument("--specialization")
    parser.add_argument("--benchmark")

    parser.add_argument("--llm-log", required=True)
    parser.add_argument("--llm-out", required=True)
    parser.add_argument("--as-config")

    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--workload-timeout", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--blocklist", type=str, default="")
    args = parser.parse_args()

    bcf = args.benchmark_config_file
    if args.specialization:
        with open(args.benchmark_config_file, "r") as f:
            data = yaml.safe_load(f)
            data["mythril"]["query_spec"]["execute_query_directory"] = str(args.specialization)
            data["mythril"]["query_spec"]["execute_query_order"] = f"{args.specialization}/d_order.txt"
        with open("/tmp/benchmark.yaml", "w") as f:
            yaml.dump(data, f)
        bcf = "/tmp/benchmark.yaml"

    if args.as_config is None:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        formatter = "%(levelname)s:%(asctime)s %(message)s"
        file_logger = logging.FileHandler(f"{args.output}/output.log", mode="w")
        file_logger.setFormatter(logging.Formatter(formatter))
        file_logger.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(file_logger)

    spec = Spec(
        agent_type="wolp",
        seed=0,
        config_path=args.config_file,
        benchmark_config_path=bcf,
        horizon=5,
        workload_timeout=args.workload_timeout,
        logger=None)

    env = PostgresEnv(
        spec,
        horizon=0,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    sknobs, nindexes, start, last = parse_lambdaconfig(args)

    env.restore_pristine_snapshot()
    # Dump if evaluating.
    env.shift_state(sknobs, nindexes, ignore_error=True, dump_page_cache=(args.as_config is None))

    if args.as_config is not None:
        if args.benchmark == "tpch":
            env.connection.execute("""
create view revenue0_PID (supplier_no, total_revenue) as
	select
		l_suppkey,
		sum(l_extendedprice * (1 - l_discount))
	from
		lineitem
	where
		l_shipdate >= date '1994-09-01'
		and l_shipdate < date '1994-09-01' + interval '3' month
	group by
		l_suppkey;
            """)

        import time
        start = time.time()
        aargs = DotDict({
            "training": True,
            "inference": False,
            "database": "postgres",
            "benchmark": args.benchmark,
            "config": args.as_config,
            "output_dir": args.output_dir,
            "output_name": f"{args.output}",
        })
        run(aargs)

        import logging
        import time
        logging.warn(f"Finished auto-steer sweep: {time.time() - start}")

    else:
        collect_samples = []
        for sample_idx in range(args.num_samples):
            qid_runs = []
            qid_runtimes = {}
            collect_samples.append(spec.workload._execute_workload(
                connection=env.connection,
                workload_timeout=args.workload_timeout,
                ql_knobs={},
                env_spec=spec,
                blocklist=args.blocklist.split(",") if len(args.blocklist) > 0 else [],
                qid_runtimes=qid_runtimes))

            for qid, val in qid_runtimes.items():
                qid_runs.append({
                    "sample": sample_idx,
                    "qid": qid,
                    "runtime": val,
                })
            qid_runs.append({
                "sample": sample_idx,
                "qid": "total",
                "runtime": collect_samples[-1],
            })

            logging.info(f"Workload execution time: {collect_samples[-1]}")
            if collect_samples[-1] == args.workload_timeout:
                break

            data = pd.DataFrame(qid_runs)
            data.to_csv(Path(args.output) / f"qid_raw_{sample_idx}.csv", index=False)

        final_out = [{
            "step": 0,
            "orig_cost": 0,
            "time_since_start": (last - start).total_seconds(),
        }]
        final_out[-1].update({
            f"runtime{r}": collect_samples[r] for r in range(len(collect_samples))
        })
        pd.DataFrame(final_out).to_csv(Path(args.output) / f"out.csv", index=False)
        env.close()
