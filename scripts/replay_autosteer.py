import logging
import psycopg
import yaml
import copy
import os
import sys
import argparse
sys.path.append("/home/wz2/mythril")
from envs.spec import Spec
from envs.pg_env import PostgresEnv
from scripts.parse_hyrise import parse_hyrise_configs
from scripts.parse_lambdaconfig import parse_lambdaconfig
from sqlalchemy import create_engine, event
from dateutil.parser import parse
from pathlib import Path
import pandas as pd


BEST_ALTERNATIVE_PLANS = """
WITH default_plans (query_path, walltime) AS
  (SELECT q.query_path,
          COALESCE(median(walltime), 1e6) AS walltime
   FROM queries q
   JOIN query_optimizer_configs qoc
      ON q.id = qoc.query_id
     AND qoc.num_disabled_rules = 0
     AND qoc.disabled_rules = 'None'
LEFT JOIN measurements m ON qoc.id = m.query_optimizer_config_id
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
        configs = [OptimizerConfigResult(*row) for row in cursor.fetchall()]
    return sorted(configs, key=lambda x: x.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Replay")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--benchmark-config-file", required=True)
    parser.add_argument("--specialization")
    parser.add_argument("--benchmark")

    parser.add_argument("--hyrise-log")
    parser.add_argument("--stream-log")
    parser.add_argument("--db-root")

    parser.add_argument("--llm-log")
    parser.add_argument("--llm-out")

    parser.add_argument("--output", required=True)
    parser.add_argument("--workload-timeout", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--blocklist", type=str, default="")

    parser.add_argument("--sample-specialization", default=None)
    parser.add_argument("--sample-out", default=None)
    args = parser.parse_args()

    bcf = args.benchmark_config_file
    if args.specialization or args.sample_specialization is not None:
        with open(args.benchmark_config_file, "r") as f:
            data = yaml.safe_load(f)
            if args.specialization:
                data["mythril"]["query_spec"]["execute_query_directory"] = str(args.specialization)
                data["mythril"]["query_spec"]["execute_query_order"] = f"{args.specialization}/d_order.txt"
            else:
                data["mythril"]["query_spec"]["execute_query_directory"] = str(args.sample_specialization)
                data["mythril"]["query_spec"]["execute_query_order"] = f"{args.sample_specialization}/d_order.txt"
        with open("/tmp/benchmark.yaml", "w") as f:
            yaml.dump(data, f)
        bcf = "/tmp/benchmark.yaml"

    Path(args.output).mkdir(parents=True, exist_ok=True)
    if not args.sample_specialization:
        if (Path(args.output) / f"out.csv").exists():
            import sys
            sys.exit()
    else:
        if Path(args.sample_out).exists():
            import sys
            sys.exit()

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

    formatter = "%(levelname)s:%(asctime)s %(message)s"
    file_logger = logging.FileHandler(f"{args.output}/output.log", mode="w")
    file_logger.setFormatter(logging.Formatter(formatter))
    file_logger.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_logger)

    if args.hyrise_log:
        # Get the configurations.
        nindexdefs, ntimes = parse_hyrise_configs(args)
        nknobs = [[
            "shared_buffers = 8GB",
            "effective_cache_size = 24GB",
            "maintenance_work_mem = 2GB",
            "checkpoint_completion_target = 0.9",
            "wal_buffers = 16MB",
            "default_statistics_target = 500",
            "random_page_cost = 1.1",
            "effective_io_concurrency = 200",
            "work_mem = 10485kB",
            "min_wal_size = 4GB",
            "max_wal_size = 16GB",
            "max_worker_processes = 20",
            "max_parallel_workers_per_gather = 10",
            "max_parallel_workers = 20",
            "max_parallel_maintenance_workers = 4",
        ]] * len(nindexdefs)

    else:
        sknobs, nindexes, start, last = parse_lambdaconfig(args)
        nknobs = [sknobs]
        nindexdefs = [nindexes]
        ntimes = [(last - start).total_seconds()]

    if args.benchmark == "tpch":
        [indexdefs.append("""
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
            """) for indexdefs in nindexdefs]

    spairs = []
    with open(args.stream_log) as f:
        cstart = None
        for line in f:
            if " INFO" in line and "Run AutoSteer's training mode" in line:
                cstart = parse(line.split(" INFO ")[0])
            elif cstart and "WARNING" in line and "auto-steer sweep: " in line:
                spairs.append((cstart, parse(line.split("WARNING:")[1].split("Finished")[0])))

    if args.hyrise_log:
        for i, indexdefs in enumerate(nindexdefs):
            db_file = args.db_root + f"_{i}.sqlite"
            assert Path(db_file).exists(), print(db_file)
            assert i < len(spairs)

    if args.sample_specialization:
        # Only evaluate the last one.
        nindexdefs = [nindexdefs[-1]]
        ntimes = [ntimes[-1]]
        spairs = [spairs[-1]]

    final_out = []
    for i, indexdefs in enumerate(nindexdefs):
        assert len(indexdefs) > 0
        env.restore_pristine_snapshot()
        env.shift_state(nknobs[i], indexdefs, ignore_error=True, dump_page_cache=True)
        connection = env.connection

        # Get configurations.
        db_file = args.db_root + (f"_{i}.sqlite" if args.hyrise_log else ".sqlite")
        configurations = get_configurations(db_file)

        sqlfile_qid_map = {}
        with open(bcf) as f:
            query_order = yaml.safe_load(f)["mythril"]["query_spec"]["execute_query_order"]
        with open(query_order) as f:
            for line in f:
                parts = line.strip().split(",")
                sqlfile_qid_map[parts[1]] = parts[0]

        knob_space = spec.action_space.get_knob_space()
        knob_space.reset(connection=connection, workload=spec.workload)
        ql_knobs = knob_space.get_query_level_knobs(knob_space.get_state(None))

        forcescans = {}
        blocklist = args.blocklist.split(",") if len(args.blocklist) > 0 else []
        for configuration in configurations:
            filename = Path(configuration.path).parts[-1]
            if len(blocklist) > 0 and any([b in filename for b in blocklist]):
                continue

            if "dsb_pd25" in str(db_file):
                if filename == "query040s5_spj.sql":
                    continue

            if filename == "15.sql" and args.benchmark == "tpch":
                filename = "15_bao.sql"

            if args.sample_specialization:
                if filename not in sqlfile_qid_map:
                    continue
                else:
                    qids = [sqlfile_qid_map[filename]]

            else:
                assert filename in sqlfile_qid_map, print(filename)
                qids = [sqlfile_qid_map[filename]]

            for qid in qids:
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
                        assert f"{qid}_{rule}" in ql_knobs, print(f"{qid}_{rule}")
                        ql_knobs[f"{qid}_{rule}"] = (ql_knobs[f"{qid}_{rule}"][0], 0.)
                    else:
                        tblname = rule.split("tbl_")[-1]
                        key = f"{qid}_{tblname}_{scanmethod}"
                        assert key in ql_knobs
                        ql_knobs[key] = (ql_knobs[key], 1.)

        if (not args.sample_specialization):
            # Only gather the index-only if hyrise.
            collect_samples = []
            for sample_idx in range(args.num_samples):
                qid_runtimes = {}
                collect_samples.append(spec.workload._execute_workload(
                    connection=connection,
                    workload_timeout=args.workload_timeout,
                    ql_knobs={},
                    env_spec=spec,
                    forcescans=forcescans,
                    blocklist=args.blocklist.split(",") if len(args.blocklist) > 0 else [],
                    qid_runtimes=qid_runtimes))

                logging.info(f"Workload execution time: {collect_samples[-1]}")
                if collect_samples[-1] == args.workload_timeout:
                    break
            final_out.append({
                "step": len(final_out),
                "orig_cost": 0,
                "time_since_start": ntimes[i],
            })
            final_out[-1].update({
                f"runtime{r}": collect_samples[r] for r in range(len(collect_samples))
            })

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

        if not args.sample_specialization:
            data = pd.DataFrame(qid_runs)
            data.to_csv(Path(args.output) / f"qid_raw_{i}.csv", index=False)

        final_out.append({
            "step": len(final_out),
            "orig_cost": 0,
            "time_since_start": ntimes[i] + ((spairs[i][1] - spairs[i][0]).total_seconds()),
        })
        final_out[-1].update({
            f"runtime{r}": collect_samples[r] for r in range(len(collect_samples))
        })

    if not args.sample_specialization:
        pd.DataFrame(final_out).to_csv(Path(args.output) / f"out.csv", index=False)
    else:
        pd.DataFrame(final_out).to_csv(args.sample_out, index=False)
    env.close()
