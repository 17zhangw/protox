import logging
import psycopg
import yaml
import copy
import os
import sys
import argparse
sys.path.append(".")
from envs.spec import Spec
from scripts.hyrise_load import hyrise_load
from sqlalchemy import create_engine, event
from pathlib import Path
import pandas as pd


BEST_ALTERNATIVE_PLANS = """
WITH default_plans (query_path, walltime) AS
  (SELECT q.query_path,
          median(walltime)
   FROM queries q,
        query_optimizer_configs qoc,
        measurements m
   WHERE q.id = qoc.query_id
     AND qoc.id = m.query_optimizer_config_id
     AND qoc.num_disabled_rules = 0
     AND qoc.disabled_rules = 'None'
   GROUP BY q.query_path,
            qoc.num_disabled_rules,
            qoc.disabled_rules), -- default for queries that timed out
     results(query_path, num_disabled_rules, runtime, runtime_baseline, savings, disabled_rules, rank) AS
  (SELECT q.query_path,
          qoc.num_disabled_rules,
          median(m.walltime),
          dp.walltime,
          (dp.walltime * 1.0 - median(m.walltime)) / dp.walltime  AS savings,
          qoc.disabled_rules,
          dense_rank() OVER (PARTITION BY q.query_path
                             ORDER BY (dp.walltime - median(m.walltime)) / dp.walltime DESC) AS ranki
   FROM queries q,
        query_optimizer_configs qoc,
        measurements m,
        default_plans dp
   WHERE q.id = qoc.query_id
     AND qoc.id = m.query_optimizer_config_id
     AND dp.query_path = q.query_path
     AND qoc.num_disabled_rules > 0
   GROUP BY q.query_path,
            qoc.num_disabled_rules,
            qoc.disabled_rules,
            dp.walltime
   ORDER BY savings DESC)
SELECT *
FROM results
WHERE rank = 1
ORDER BY savings DESC;
"""


def get_configurations(db_file):
    class OptimizerConfigResult:
        def __init__(self, path, num_disabled_rules, runtime, runtime_baseline, savings, disabled_rules, rank):
            self.path = path
            self.num_disabled_rules = num_disabled_rules
            self.runtime = runtime
            self.runtime_baseline = runtime_baseline
            self.savings = savings
            self.disabled_rules = disabled_rules
            self.rank = rank

    def _db():
        global ENGINE
        url = f'sqlite:///{db_file}'
        logging.info(f"Connect to database: {url}")
        ENGINE = create_engine(url)

        @event.listens_for(ENGINE, 'connect')
        def connect(dbapi_conn, _):
            """Load SQLite extension for median calculation"""
            extension_path = './sqlean-extensions/stats.so'

            if not os.path.isfile(extension_path):
                logger.fatal('Please, first download the required sqlite3 extension using sqlean-extensions/download.sh')
                sys.exit(1)

            dbapi_conn.enable_load_extension(True)
            dbapi_conn.load_extension(extension_path)
            dbapi_conn.enable_load_extension(False)

        return ENGINE.connect()

    with _db() as conn:
        cursor = conn.execute(BEST_ALTERNATIVE_PLANS)
        return [OptimizerConfigResult(*row) for row in cursor.fetchall()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--benchmark", type=str, default="")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--benchmark-config-file", required=True)
    parser.add_argument("--hyrise-log", default=None)

    parser.add_argument("--load", action="store_true")
    parser.add_argument("--conn-str", type=str, default=None)

    parser.add_argument("--db-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workload-timeout", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--blocklist", type=str, default="")
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    spec = Spec(
        agent_type="wolp",
        seed=0,
        config_path=args.config_file,
        benchmark_config_path=args.benchmark_config_file,
        horizon=5,
        workload_timeout=args.workload_timeout,
        logger=None)

    formatter = "%(levelname)s:%(asctime)s %(message)s"
    file_logger = logging.FileHandler(f"{args.output}/output.log", mode="w")
    file_logger.setFormatter(logging.Formatter(formatter))
    file_logger.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_logger)

    # Get configurations.
    configurations = get_configurations(args.db_file)

    sqlfile_qid_map = {}
    with open(args.benchmark_config_file) as f:
        query_order = yaml.safe_load(f)["mythril"]["query_spec"]["query_order"]
    with open(query_order) as f:
        for line in f:
            parts = line.strip().split(",")
            sqlfile_qid_map[parts[1]] = parts[0]

    if args.load:
        # Load the DB from the hyrise.
        env = hyrise_load(args)
        connection = env.connection

    else:
        assert args.conn_str is not None
        connection = psycopg.connect(args.conn_str, prepare_threshold=None, autocommit=True)

    knob_space = spec.action_space.get_knob_space()
    knob_space.reset(connection=connection, workload=spec.workload)
    ql_knobs = knob_space.get_query_level_knobs(knob_space.get_state(None))

    forcescans = {}
    blocklist =args.blocklist.split(",") if len(args.blocklist) > 0 else []
    for configuration in configurations:
        filename = Path(configuration.path).parts[-1]
        if len(blocklist) > 0 and any([b in filename for b in blocklist]):
            continue
        assert filename in sqlfile_qid_map, print(filename)

        qid = sqlfile_qid_map[filename]
        for rule in configuration.disabled_rules.split(","):
            if rule == "no_forceseq":
                # Disable no force seq => force seq
                forcescans[qid] = "seq"
            elif rule == "no_forceidx":
                if qid not in forcescans:
                    # Defer seq if both disabled for some reason.
                    forcescans[qid] = "index"
            elif rule == "no_forceidxbase":
                if qid not in forcescans:
                    # Defer seq if both disabled for some reason.
                    forcescans[qid] = "indexbase"
            elif not rule.startswith("tbl_"):
                # Disable rule.
                assert f"{qid}_{rule}" in ql_knobs
                ql_knobs[f"{qid}_{rule}"] = (ql_knobs[f"{qid}_{rule}"][0], 0.)
            else:
                tblname = rule.split("tbl_")[-1]
                key = f"{qid}_{tblname}_{scanmethod}"
                assert key in ql_knobs
                ql_knobs[key] = (ql_knobs[key], 1.)

        if args.save_dir is not None:
            knobs = [k.split(f"{qid}_")[-1] for k in ql_knobs.keys() if k.startswith(f"{qid}_")]
            options = [f"SET {k} = ON" for k in knobs]
            options += [f"SET {k} = OFF" for k in knobs if ql_knobs[f"{qid}_{k}"][1] == 0.]
            with open(args.save_dir / filename, "w") as f:
                queries = ";\n".join([q[1] for q in spec.workload.queries[qid]])
                queries = ";\n".join(options) + ";\n" + queries
                f.write(queries)

    # Dump the OS page cache.
    os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

    qid_runs = []
    collect_samples = []
    for sample_idx in range(args.num_samples):
        qid_runtimes = {}
        collect_samples.append(spec.workload._execute_workload(
            connection=connection,
            workload_timeout=args.workload_timeout,
            ql_knobs=ql_knobs,
            env_spec=spec,
            forcescans=forcescans,
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
    data.to_csv(Path(args.output) / "qid_raw.csv", index=False)
