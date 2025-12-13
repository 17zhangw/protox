import sys
import time
import math
import json
import psycopg
import argparse

import sys
sys.path.append("/home/wz2/mythril")
from envs.spec import Spec
from envs.pg_env import PostgresEnv
from utils.derive_repo_config import _derive_booster_config
from psycopg.errors import QueryCanceled


def _gen_sig(p, full=False, shape=False):
    sig = ""
    assert not isinstance(p, list)
    if "Node Type" in p:
        if p.get("Parallel Aware", False):
            sig += " P-Aware "

        if p.get("Async Capable", False):
            sig += " Async "

        if "Join Type" in p:
            sig += " Join({}) ".format(p["Join Type"])

        sig += p["Node Type"]

        if "Subplan Name" in p:
            sig += " SP[{}] ".format(p["Subplan Name"])

        if "Strategy" in p:
            sig += " Strategy({}) ".format(p["Strategy"])

        if "Command" in p:
            sig += " Command({}) ".format(p["Command"])

        if "Partial Mode" in p:
            sig += " PartialMode({}) ".format(p["Partial Mode"])

        if "Scan" in p["Node Type"]:
            alias = None
            if p["Node Type"] in ["Index Scan", "Index Only Scan"]:
                if shape:
                    alias = p.get("Relation Name", p.get("Alias", None))

                elif not full:
                    alias = p["Index Name"]

                else:
                    alias = ""
                    if "Index Cond" in p:
                        alias = "Cond({})".format(p["Index Cond"])

                    if "Filter" in p:
                        alias += " Filter({})".format(p["Filter"])

                    if alias == "":
                        alias = "CondFilter(NULL)"
            elif "CTE Name" in p:
                alias = p["CTE Name"]
            elif "Bitmap Index Scan" in p["Node Type"]:
                if shape:
                    # There is no alias... alias is on the Bitmap Heap...
                    alias = ""
                elif not full:
                    alias = p["Index Name"]
                elif "Index Cond" in p:
                    alias = "Cond({})".format(p["Index Cond"])
                    assert "Filter" not in p, print(p)
                else:
                    alias = "Cond(NULL)"
                    assert "Filter" not in p, print(p)
            else:
                # TODO: Always use the relation name.
                # Semantically, the "alias" is just internal mapping.
                # Duplicating the alias exactly is extremely hard...
                alias = p.get("Relation Name", p.get("Alias", None))
            assert alias is not None
            if not full:
                sig += " {}".format(alias)
            else:
                if "Alias" in p:
                    sig += " {} ({})".format(alias, p["Alias"])
                else:
                    sig += " {}".format(alias)

        # if  p.get("Workers Planned", None):
        #     sig += " P-Workers({}) ".format(p["Workers Planned"])

    if "Plans" in p:
        subplans = []
        for pp in p["Plans"]:
            subplans.append(_gen_sig(pp, full=full, shape=shape))
        sig += "(" + ",".join(subplans) + ")"
    elif "Plan" in p:
        sig += _gen_sig(p["Plan"], full=full, shape=shape)

    return sig


def _get_used_indexes(conn, queries):
    def _rplan(plan):
        if "Plan" in plan:
            return _rplan(plan["Plan"])

        elif "Plans" in plan:
            iset = set()
            for p in plan["Plans"]:
                iset.update(_rplan(p))
            return iset

        elif "Index Name" in plan:
            return set([plan["Index Name"]])

        return set()

    totindexes = set()
    sigs = []
    for sql in queries:
        eplan = [r for r in conn.execute("EXPLAIN (FORMAT JSON) " + sql)][0][0][0]
        totindexes.update(_rplan(eplan))
        sigs.append(_gen_sig(eplan, full=True, shape=True))
    return totindexes, sigs


def _build_workload(input_path):
    with open(input_path) as f:
        data = json.load(f)

    sqls = []
    qcmap = { q["qidx"]: q["hint"] for q in data["cardinals"][0]["qconfigs"] }
    for iqt, qt in enumerate(data["targets"]):
        with open(qt["qfile"]) as f:
            sql = f.read()

        hint = qcmap[qt["qidx"]]
        sqls.append(hint + sql)

    return sqls, data


def force_statement_timeout(connection, timeout_ms):
    retry = True
    while retry:
        retry = False
        try:
            connection.execute(f"SET statement_timeout = {timeout_ms}")
        except QueryCanceled:
            retry = True


def run_query(conn, btime, sql):
    diags = []
    def _diag_handler(diag):
        nonlocal diags
        if diag is not None and (diag.message_primary or diag.message_hint or diag.message_detail):
            diags.append((diag.message_primary, diag.message_hint, diag.message_detail))
    conn.add_notice_handler(_diag_handler)
    sql = "EXPLAIN (FORMAT JSON, ANALYZE, TIMING OFF) " + sql

    # Run with timeout.
    timeout = math.ceil(btime) * 1e3
    hit_timeout = False
    force_statement_timeout(conn, timeout)
    try:
        eplan = [r for r in conn.execute(sql)][0][0][0]
    except KeyboardInterrupt:
        raise
    except QueryCanceled:
        hit_timeout = True
        eplan = {"Execution Time": timeout}

    if not hit_timeout:
        # Make sure we aren't getting any pg_hint_plan diagnostics.
        assert len(diags) == 0, print(sql, diags)
        conn.remove_notice_handler(_diag_handler)
    force_statement_timeout(conn, 0)
    return eplan["Execution Time"] / 1e3


def run_workload(conn, wtime, queries):
    actual_wtime = 0
    for idx, sql in enumerate(queries):
        qtime = run_query(conn, max(1, math.ceil(wtime) + 2), sql)
        actual_wtime += qtime
        wtime -= qtime
        if wtime < 0:
            break
        print(f"Query {idx+1}: {qtime}")
    print(f"Total Workload Runtime: {actual_wtime}")
    return actual_wtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sanitizer")
    parser.add_argument("--input")
    parser.add_argument("--benchmark-config")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--connection")
    parser.add_argument("--ceiling", type=float)
    parser.add_argument("--sweep-size", type=int, default=1)
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()

    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=5,
        config_path="configs/config.yaml",
        benchmark_config_path=args.benchmark_config,
        workload_timeout=0)

    env = PostgresEnv(
        spec,
        horizon=5,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    if args.restore:
        env.restore_pristine_snapshot()
        env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
        spec.workload.reset()

        env.action_space.reset(connection=env.connection, workload=env.workload)

        knobs, indexes = _derive_booster_config(env, args.input)
        cc, _ = env.action_space.get_knob_space().generate_plan(knobs, no_check=True)
        env.shift_state(cc, indexes, dump_page_cache=False)
        connection = env.connection

    else:
        assert args.connection
        connection = psycopg.connect(args.connection, autocommit=True, prepare_threshold=None)

    # Make sure all live.
    connection.execute("UPDATE pg_index SET indisvalid = true;")

    all_idxes = {
        r[0]: r[1]
        for r in connection.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'")
    }

    def _enable_index(name, d):
        #connection.execute(d)
        #return

        connection.execute(f"""
        UPDATE pg_index
        SET indisvalid = TRUE
        WHERE indexrelid = (
          SELECT oid
            FROM pg_class
              WHERE relname = '{name}'
              );
        """)

    def _disable_index(name):
        #connection.execute(f"DROP INDEX {name}")
        #return

        connection.execute(f"""
        UPDATE pg_index
        SET indisvalid = FALSE
        WHERE indexrelid = (
          SELECT oid
            FROM pg_class
              WHERE relname = '{name}'
              );
        """)

    start = time.time()
    eindexes = [r[0] for r in connection.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'") if "_pkey" not in r[0]]
    wl, data = _build_workload(args.input)
    assert run_workload(connection, 60, wl) < args.ceiling
    if args.seed_only:
        sys.exit(0)

    used_idxes, sigs = _get_used_indexes(connection, wl)
    hides = []
    for idx in eindexes:
        if idx not in used_idxes:
            _disable_index(idx)
            hides.append(idx)
    run_workload(connection, 60, wl)

    idxtups = [(idx, all_idxes[idx].split(" ON ")[-1].split(" (")[0].strip(),  all_idxes[idx]) for idx in used_idxes if "_pkey" not in idx]
    idxtups = sorted(idxtups, key=lambda x: x[1])
    for istart in range(0, len(idxtups), args.sweep_size):
        slots = idxtups[istart:istart+args.sweep_size]
        for iname, _, d in slots:
            _disable_index(iname)

        testm = run_workload(connection, args.ceiling ,wl)
        if testm < args.ceiling:
            for iname, _, _ in slots:
                hides.append(iname)
                print(f"Hiding {iname}")
        else:
            for iname, _, _ in slots:
                _enable_index(iname, d)
                print(f"Preserving {iname}")

    for hide in hides:
        print(hide)
    end = time.time()
    print(end - start)

    print("====== FINAL VALIDATION ======= ")
    #for hide in hides:
    #    connection.execute(f"DROP INDEX {hide}")
    #run_workload(connection, 60, wl)

    rtups = [
        d
        for iname, _, d in idxtups
        if iname not in hides
    ]
    rtups = sorted(rtups, key=lambda x: int(x.split("eindex")[-1].split(" ON ")[0].strip()))
    data["cardinals"][0]["indexes"] = rtups
    with open("sanitized.json", "w") as f:
        json.dump(data, f, indent=2)
