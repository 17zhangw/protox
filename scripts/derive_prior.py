import os
import time
import tqdm
from pathlib import Path
import argparse
import itertools
from itertools import chain
from itertools import combinations
import psycopg

import sys
sys.path.append("/home/wz2/mythril")
from envs.spec import Spec
from envs.pg_env import PostgresEnv
from envs.workload_utils import force_statement_timeout
from envs.workload import QueryType


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Hintset Maximizer")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--per-query-timeout", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed-config", type=Path, default=None)
    args = parser.parse_args()

    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=5,
        config_path=args.config,
        benchmark_config_path=args.benchmark_config,
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=5,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    env.restore_pristine_snapshot()

    if args.seed_config is not None:
        with open(args.seed_config, "r") as f:
            config = f.read()
        config = eval(config)
        env.shift_state(config[0], config[1])

    start_time = time.time()

    conn = env.connection
    workload = spec.workload

    # Prewarm.
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
    for tbl in workload.tables:
        sql = f"SELECT pg_prewarm('{tbl}')"
        conn.execute(sql)
        print(f"Executing {sql}")

    with open(args.output, "w") as f:
        f.write("qid,forceidx,idxall,forceseq,seqall\n")
        f.flush()

        for q in tqdm.tqdm(workload.order):
            timeout = args.per_query_timeout
            qlist = workload.queries[q]

            force_statement_timeout(conn, 0)

            for sql_type, query in qlist:
                force_statement_timeout(conn, 0)
                if sql_type != QueryType.SELECT:
                    assert sql_type != QueryType.INS_UPD_DEL
                    conn.execute(query)
                    continue

                qaliases = []
                for _, als in workload.query_aliases[q].items():
                    qaliases.extend(als)

                idxall = " ".join([f"IndexOnlyScan({t})" for t in qaliases])
                seqall = " ".join([f"SeqScan({t})" for t in qaliases])

                force_statement_timeout(conn, timeout * 1000)
                def run(conn, sql):
                    try:
                        runtime = 1e6
                        for _ in range(2):
                            s = time.time()
                            _ = conn.execute(sql)
                            end = time.time() - s
                            if end < runtime:
                                runtime = end
                    except:
                        runtime = timeout
                    return runtime

                idxall_runtime = run(conn, "/*+ " + idxall + " */ " + query)
                seqall_runtime = run(conn, "/*+ " + seqall + " */ " + query)
                force_statement_timeout(conn, 0)

            f.write(f"{q},{idxall_runtime},{idxall},{seqall_runtime},{seqall}\n")
            f.flush()
        delta = time.time() - start_time
        f.write(f"{delta},0,,0,\n")
        f.flush()
