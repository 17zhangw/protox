from pathlib import Path
import psycopg
import pandas as pd
import time


if __name__ == "__main__":
    conn = psycopg.connect("host=localhost port=5432 dbname=benchbase", autocommit=True, prepare_threshold=None)
    data = pd.read_csv("tpch_out.csv")

    with open("queries/tpch/qorder_bao.txt", "r") as f:
        qorder = f.readlines()
        qorder = [l.strip() for l in qorder if len(l.strip()) > 0]

    for _ in range(3):
        workload_time = 0
        for i, tup in enumerate(data.itertuples()):
            if tup.qid == "-1" or tup.qid == -1:
                continue

            hintset = eval(tup.hintset)

            with open(f"queries/tpch/{qorder[i]}", "r") as f:
                query = f.read()

            pqk_query = "/*+ " + " ".join([f"Set ({k} {v})" for k, v in hintset.items()]) + " */ " + query
            t = time.time()
            _ = conn.execute(pqk_query)
            duration = time.time() - t
            workload_time += duration
            print(qorder[i], duration)

        print(workload_time)

