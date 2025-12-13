from pathlib import Path
import shutil
import json
from dateutil.parser import parse


def _load_ut_boost(file, wordfile):
    with open(file.split("boost-")[1]) as f:
        cc = json.load(f)

    targets = {Path(c["qfile"]).parts[-1]: c["qidx"] for c in cc["targets"]}
    bestc = sorted(cc["cardinals"], key=lambda x: x["total_rc"])[0]
    mworkers = bestc["sysknobs"]["max_worker_processes"]

    bestcq = {}
    for qc in bestc["qconfigs"]:
        bestcq.update({k: v for k, v in qc.items() if "qknobs" in k})

    with open(wordfile) as f:
        sfiles = [l.strip() for l in f.readlines()]
        sfiles = [s.split(",")[1] if "," in s else s for s in sfiles]

    config = {
        f"knob.{k}": v for k, v in bestc["sysknobs"].items()
    }

    for idx in bestc["indexes"]:
        eindname = idx.split(" INDEX ")[1].split(" ON ")[0].strip()
        idx = idx.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS") if "IF NOT EXISTS" not in idx else idx
        config[eindname] = idx

    config["query.file_id"] = -1
    if Path("/tmp/baseline").exists():
        shutil.rmtree("/tmp/baseline")
    Path("/tmp/baseline").mkdir(parents=True, exist_ok=True)
    for sf in sfiles:
        with open(Path(wordfile).parent / sf) as f:
            sql = f.read()

        hset = []
        qidx = targets[sf]
        qknobs = bestcq[f"qknobs{qidx}"]
        for qk, qv in qknobs.items():
            if "Access Method " in qk:
                if qv == "Default":
                    continue

                qv = {
                    "Index Scan": "IndexOnlyScan",
                    "Bitmap Scan": "BitmapScan",
                    "Seq Scan": "SeqScan",
                }[qv]

                tbl = qk.split(" for ")[1].split(" (")[0].strip()
                hset.append(f"{qv}({tbl})")

            elif "Force Parallel" in qk:
                if qv != "None" and qv != "Default":
                    hset.append(f"Parallel({qv} {mworkers})")

            elif "Materialize or Inline" in qk:
                if qv != "Default":
                    ctename = qk.split(" Inline CTE ")[1].strip()
                    mtype = "MATERIALIZE" if qv.lower() == "materialize" else "INLINE"
                    hset.append(f"Materialize({ctename} {mtype})")

            elif "enable_" in qk and qv != "Default":
                hset.append(f"Set({qk} {1 if qv else 0})")

            elif qk in ["dkid"]:
                continue

            elif qv != "Default":
                hset.append(f"Set({qk} {qv})")

        sql = "/*+ " + " ".join(hset) + " */ " + sql
        with open(f"/tmp/baseline/{sf}", "w") as f:
            f.write(sql)

    with open(f"/tmp/base.txt", "w") as f:
        f.write("\n".join(sfiles).strip())
    return config


def _parse_config(config, qdir, env=None):
    config_changes = []
    sql_commands = []
    per_query_knobs = {}
    for k, v in config.items():
        if not k.startswith("knob"):
            continue

        if "." in k:
            k = k.split(".")[-1]

        if k.startswith("Q"):
            assert env is not None
            if "scanmethod" in k:
                per_query_knobs[k] = int("Index" in v)
            elif "parallel_rel" in k:
                if v.lower() == "sentinel":
                    per_query_knobs[k] = 0
                else:
                    per_query_knobs[k] = env.action_space.get_knob_space().knobs[k].values.index(v)
            elif isinstance(v, str):
                per_query_knobs[k] = int(v == "on")
            else:
                per_query_knobs[k] = v

            continue

        set_str = f"{k} = {v}"
        config_changes.append(set_str)

    for k, v in config.items():
        if k.startswith("eindex"):
            sql_commands.append(v)
            continue

        if not k.startswith("index"):
            continue

        table, col = k.split("index.")[-1].split(".")
        if v == "off":
            sql_commands.append(f"DROP INDEX IF EXISTS {table}_{col}")
        else:
            sql_commands.append(f"CREATE INDEX IF NOT EXISTS {table}_{col} ON {table} ({col})")

    workload_qdir = None
    workload_qfiles = None
    if "query.file_id" in config:
        file_id = config["query.file_id"]
        if int(file_id) > 0:
            workload_qdir = qdir / str(file_id)
            workload_qfiles = qdir / f"{file_id}.txt"
        elif int(file_id) == -1:
            workload_qdir = Path("/tmp/baseline")
            workload_qfiles = Path("/tmp/base.txt")

    return config_changes, sql_commands, (workload_qdir, workload_qfiles), per_query_knobs


def parse_unituneconfig(input_dir, workload_timeout, env=None, wordfile=None):
    with open(input_dir / "tpch_test.log") as f:
        loglines = f.readlines()

    global_config = {}
    base_config = False
    for line in loglines:
        if "Init record" in line:
            # Get the start record.
            s = line.split("[tpch_test][")[-1].split("]:")[0]
            time_start = parse(s)

        if "Loaded from " in line and "boost-" in line:
            global_config.update(_load_ut_boost(line.split(" from ")[1].strip(), wordfile))

        elif "Loaded from " in line:
            base_config = True

        elif base_config and " Config: {" in line:
            assert "Knob Config" in line or "Index Config" in line
            pfx = "knob." if "Knob Config" in line else "index."
            config = eval(line.split(" Config: ")[-1])
            config = {k if k.startswith(pfx) else (pfx+k): v for k, v in config.items()}
            global_config.update(config)

        elif "[Initialize] default cost: " in line:
            base_config = False

    run_cost = workload_timeout
    with open(input_dir / "tpch_test.res") as f:
        lines = f.readlines()[1:]

    lines_best = [line[5:] for line in lines if "best|" in line]
    mrcost = 1e6
    for i, line in enumerate(lines_best):
        tmp = eval(line.strip())
        time_cost = tmp['time_cost']
        if time_cost[0] < mrcost:
            mrcost = time_cost[0]

    configs = []
    for line in lines_best:
        tmp = eval(line.strip())
        config = tmp['configuration']
        time_cost = tmp['time_cost']

        if time_cost[0] <= run_cost:
            cur_time = None
            while True:
                log = loglines[0]
                if "INFO, Iteration " in log and "time_cost" in log:
                    log_time_cost = float(log.split("time_cost ")[-1].split("space_cost")[0].strip())
                    log_latmean = float(log.split("lat_mean ")[-1].split("timestamp")[0].strip())
                    if time_cost[0] == log_time_cost and time_cost[1] == log_latmean:
                        cur_time = parse(log.split("[tpch_test][")[-1].split("]:")[0])
                        loglines = loglines[1:]
                        break
                loglines = loglines[1:]
            assert cur_time is not None

            global_config.update(config)
            outputs = _parse_config(global_config, input_dir / "workload_qdirs", env)
            configs.append((*outputs, time_cost, cur_time - time_start))
            run_cost = time_cost[0]
    return configs
