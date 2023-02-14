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

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Hintset Maximizer")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--knob-list", type=str)
    parser.add_argument("--per-query-timeout", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed-config", type=Path, default=None)
    parser.add_argument("--vary-qalias", action="store_true")
    parser.add_argument("--vary-parallel", action="store_true")
    parser.add_argument("--check-timeout", action="store_true")
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
    max_parallel = [r for r in conn.execute("SHOW max_parallel_workers")][0][0]

    knobs = args.knob_list.split(",")
    plan_data = open("plan_data", "w")

    with open(args.output, "w") as f:
        f.write("qid,hintset,idxs,cost\n")
        f.flush()

        for q in tqdm.tqdm(workload.order):
            timeout = args.per_query_timeout
            qlist = workload.queries[q]
            min_comb = None
            min_idxs = set()
            min_cost = 1e6

            num_combs = 0
            for _ in powerset(knobs):
                num_combs += 1

            sigmap = {}

            force_statement_timeout(conn, 0)
            exec_select = False
            for sql_type, query in qlist:
                if sql_type != QueryType.SELECT:
                    conn.execute(query)
                    continue

                assert not exec_select
                exec_select = True
                force_statement_timeout(conn, timeout * 1000)
                try:
                    s = time.time()
                    _ = conn.execute(query)
                    runtime = time.time() - s
                except:
                    runtime = timeout
                force_statement_timeout(conn, 0)
                timeout = min(args.per_query_timeout, runtime * 1.5)

            for knob_comb in tqdm.tqdm(powerset(knobs), leave=False, total=num_combs):
                missing_knobs = [k for k in knobs if k not in knob_comb]
                comb = {k: "ON" for k in knob_comb}
                comb.update({k: "OFF" for k in missing_knobs})
                set_str = " ".join([f"Set ({k} {v})" for (k, v) in comb.items()])

                for sql_type, query in qlist:
                    force_statement_timeout(conn, 0)
                    if sql_type != QueryType.SELECT:
                        assert sql_type != QueryType.INS_UPD_DEL
                        conn.execute(query)
                        continue

                    qaliases = []
                    num_qalias = 0
                    if args.vary_qalias:
                        for _, als in workload.query_aliases[q].items():
                            qaliases.extend(als)

                    for _ in powerset(qaliases):
                        num_qalias += 1

                    for qalias in tqdm.tqdm(powerset(qaliases), leave=False, total=num_qalias):
                        alias_str = ""
                        if args.vary_qalias:
                            missing_qalias = [q for q in qaliases if q not in qalias]
                            alias_str = " ".join([f"SeqScan({t})" for t in qalias] + [f"IndexScan({t})" for t in missing_qalias])

                        parallel = [None]
                        if args.vary_parallel:
                            parallel += qaliases

                        for parallel in tqdm.tqdm(parallel, leave=False, total=len(parallel)):
                            parallel_str = "" if parallel is None else f"Parallel({parallel} {max_parallel})"

                            query_str = "EXPLAIN (FORMAT JSON) /*+ " + set_str + " " + alias_str + " " + parallel_str + " */ " + query
                            plan = [r for r in conn.execute(query_str)][0][0][0]["Plan"]

                            indexes = set()
                            def sign(plan):
                                node_sig = plan["Node Type"]
                                if "Workers Launched" in plan:
                                    wl = plan["Workers Launched"]
                                    node_sig += " " + f"Workers {wl}"
                                if "Relation Name" in plan:
                                    node_sig += " " + plan["Relation Name"]
                                if "Alias" in plan:
                                    node_sig += " " + plan["Alias"]
                                if "Index Name" in plan:
                                    node_sig += " " + plan["Index Name"]
                                    indexes.add(plan["Index Name"])

                                if "Plan" in plan:
                                    node_sig += " (" + sign(plan["Plan"]) + ")"
                                elif "Plans" in plan:
                                    pps = [sign(p) for p in plan["Plans"]]
                                    node_sig += " (" + ",".join(pps) + ")"
                                return node_sig
                            sig = sign(plan)
                            plan_data.write(f"{set_str}: {sig}\n")
                            plan_data.flush()
                            if sig in sigmap:
                                continue

                            sigmap[sig] = True

                            force_statement_timeout(conn, timeout * 1000)
                            try:
                                pqk_query = "/*+ " + set_str + alias_str + " */ " + query
                                s = time.time()
                                _ = conn.execute(pqk_query)
                                runtime = time.time() - s
                            except:
                                runtime = timeout

                            if runtime < timeout:
                                timeout = runtime

                            if runtime < min_cost:
                                min_cost = runtime
                                min_comb = set_str + alias_str + parallel_str
                                min_idxs = indexes

                            force_statement_timeout(conn, 0)
                            if args.check_timeout and runtime < args.per_query_timeout:
                                break

                        if args.check_timeout and timeout < args.per_query_timeout:
                            break

                if args.check_timeout and min_cost < args.per_query_timeout:
                    break

            min_idxs = ",".join([l for l in min_idxs])
            f.write(f"{q},\"{min_comb}\",\"{min_idxs}\",{min_cost}\n")
            f.flush()

        f.write(f"-1,,{time.time() - start_time}")
    plan_data.close()
