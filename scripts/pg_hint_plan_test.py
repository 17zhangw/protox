import pickle
import time
import psycopg
from pathlib import Path
import pglast
from pglast import stream
import argparse
import yaml
import numpy as np


def force_statement_timeout(connection, timeout_ms):
    retry = True
    while retry:
        retry = False
        try:
            connection.execute(f"SET statement_timeout = {timeout_ms}")
        except QueryCanceled:
            retry = True


if __name__ == "__main__":
    parser = argparser.ArgumentParser(prog="pg_hint_explore")
    # connection to postgres
    parser.add_argument("--conn", type=str, required=True)
    # per-table mutually exclusive options
    parser.add_argument("--table-options", type=str, required=True)
    # which queries to test
    parser.add_argument("--queries", type=str)
    # benchmark config
    parser.add_argument("--benchmark-config", type=Path)
    # num trials
    parser.add_argument("--num-trials", type=int)
    # output directory
    parser.add_argument("--outdir", type=str)
    # per query timeout
    parser.add_argument("--pqt", type=int)
    args = parser.parse_args()

    with open(args.benchmark_config, "r") as f:
        config = yaml.safe_load(f)["mythril"]["query_spec"]
        query_dir = Path(config["query_directory"])
        query_order = Path(config["query_order"])

    query_mapping = {}
    with open(query_order, "r") as f:
        for l in f:
            comps = l.strip().split(",")

            with open(query_dir / comps[1], "r") as qfile:
                sql = qfile.read()

            root = pglast.Node(pglast.parse_sql(sql))
            ref_tbls = set()
            for node in root.traverse():
                if isinstance(node, pglast.node.Node):
                    if isinstance(node.ast_node, pglast.ast.SelectStmt):
                        select_stmt = node.ast_node
                        if select_stmt.fromClause is not None:
                            for ft in select_stmt.fromClause:
                                if isinstance(ft, pglast.ast.RangeVar):
                                    relname = ft.relname
                                    alias = ft.relname if (ft.alias is None or ft.alias.aliasname is None or ft.alias.aliasname == "") else ft.alias.aliasname
                                    ref_tbls.add(alias)
            query_mapping[comps[0]] = (list(ref_tbls), sql)

    global_flags = args.options.split(",")
    outdir = Path(args.outdir).mkdir(parents=True, exist_ok=True)

    breakout_limit = args.num_trials * 12
    with psycopg.connect(args.conn, autocommit=True, prepare_threshold=None) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS 'pg_hint_plan'")
        query_list = args.queries.split(",")
        for q in query_list:
            assert q in query_mapping
            options, sql = query_mapping[q]

            trials = {}
            num_iters = 0
            while len(trials) < args.num_trials and num_iters < breakout_limit:
                rand_opts = np.random.randint(low=0, high=len(global_flags), size=len(options))
                if rand_opts not in trials:
                    opts = "/*+ " + " ".join([f"{global_flags[opt]}({options[opt]})" for i, opt in enumerate(rand_opts)]) + "*/ "
                    query = opts + sql

                    force_statement_timeout(connection, args.pqt * 1000)
                    _ = connection.execute(query)

                    start_time = time.time()
                    _ = connection.execute(query)
                    end = time.time() - start_time
                    trials[rand_opts] = (opts, end)

                    force_statement_timeout(connection, 0)
                num_iters += 1

            with open(f"{outdir}/{q}.pickle", "wb") as f:
                pickle.dump(trials, f)

