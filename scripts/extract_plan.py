import time
import json
import psycopg
import argparse
from pathlib import Path


def construct_pqkk(query_flags):
    flags = []
    for method, val in query_flags:
        if method.endswith("_scanmethod"):
            alias = method.split("_", 1)[-1].split("_scanmethod")[0]
            access = "IndexScan" if val == 1. else "SeqScan"
            flags.append(f"{access}({alias})")
        else:
            name = method.split("_", 1)[-1]
            flag = "ON" if val == 1 else "OFF"
            flags.append(f"Set ({name} {flag})")
    return " ".join(flags)


if __name__ == "__main__":
    QUERY_PATH = Path("/home/wz2/mythril/queries/job_full")
    parser = argparse.ArgumentParser(prog="Extract")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    output = open(args.output, "w")
    if args.run:
        output.write("query_id,run_secs\n")

    with psycopg.connect("host=localhost port=5432 dbname=benchbase", autocommit=True, prepare_threshold=None) as conn:
        query_order = []

        # Query Order
        with open(QUERY_PATH / "order.txt", "r") as f:
            for line in f:
                line = line.strip()
                query_order.append(line.split(","))

        query_flags = {}
        with open(args.input, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Q"):
                    query = line.split("_")[0]
                    method, val = line.split(" = ")
                    val = eval(val)

                    if query not in query_flags:
                        query_flags[query] = []
                    query_flags[query].append((method, val))

        for qid, query in query_order:
            with open(QUERY_PATH / query, "r") as f:
                sql = f.read()

            pqkk = construct_pqkk(query_flags[qid])
            if args.run:
                sql = "/*+ " + pqkk + " */ " + sql

                start_time = time.time()
                _ = conn.execute(sql)
                runtime = time.time() - start_time
                print(f"{qid}: {runtime}")
                output.write(f"{qid},{runtime}\n")
            else:
                sql = "EXPLAIN /*+ " + pqkk + " */" + sql
                c = [r for r in conn.execute(sql)]
                c = [cc[0] for cc in c] 

                output.write(qid + "\n")
                output.write(f"PerQuery: {query_flags[qid]}\n")
                output.write("\n".join(c))
                output.write("\n\n")
