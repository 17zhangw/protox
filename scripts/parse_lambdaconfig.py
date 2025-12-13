import json
from dateutil.parser import parse


def parse_lambdaconfig(args):
    with open(f"{args.llm_out}/reports.json", "r") as f:
        reports = json.load(f)
    met = min([r["best_execution_time"] for r in reports])
    bconfig = [r for r in reports if r["best_execution_time"] == met][0]

    indexes = {}
    with open(f"{args.llm_log}", "r") as f:
        start = None
        last = None
        for line in f.readlines():
            if "INFO:" in line:
                if start is None:
                    #print("Start", repr(line.split(" [")[0].split("INFO:")[-1]))
                    start = parse(line.split(" [")[0].split("INFO:")[1])
                else:
                    last = parse(line.split(" [")[0].split("INFO:")[-1])

            if "configuration_selector" in line and "Creating index: " in line:
                idxdef = line.split("Creating index: Index")[-1]
                idxdef = idxdef.split("(")[-1].split(")")[0]
                iparts = [i.strip() for i in idxdef.split(",")]
                indexes[iparts[0]] = "CREATE INDEX {} ON {} ({})".format(iparts[0], iparts[1], ",".join(iparts[2:]))
                indexes[iparts[0].lower()] = "CREATE INDEX {} ON {} ({})".format(iparts[0], iparts[1], ",".join(iparts[2:]))

    sknobs = [c.split(" SET ")[1] for c in bconfig["lambda_tune_config"] if " SET " in c]
    sknobs = [c.split(";")[0] for c in sknobs]
    sknobs = [c.replace(" TO ", " = ").replace(" to ", " = ") for c in sknobs]
    sknobs = [c.replace("LOCAL", "") for c in sknobs]
    sanitized_knobs = []
    for sk in sknobs:
        if "_cost" in sk and "ms" in sk:
            continue
        elif "." in sk:
            continue
        elif sk in [
            "cpu_index_cost_adj",
        ]:
            continue
        else:
            sanitized_knobs.append(sk)

    nindexes = []
    for idx in bconfig["created_indexes"]:
        assert idx in indexes, print(idx)
        nindexes.append(indexes[idx])

    return sanitized_knobs, nindexes, start, last
