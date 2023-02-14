import shutil
from pathlib import Path

if __name__ == "__main__":
    sqls = [str(s) for s in Path("old").rglob("*.sql")]
    key_fn = lambda x: (int(x.split("old/")[-1].split("-")[-1].split(".sql")[0]), int(x.split("old/")[-1].split("query")[-1].split("-")[0].split("_")[0]))
    sqls = sorted(sqls, key=key_fn)

    d_order = open("d_order.txt", "w")
    d_qorder = open("d_qorder.txt", "w")
    d_qid = 1

    s_order = open("s_order.txt", "w")
    s_qorder = open("s_qorder.txt", "w")
    s_qid = 1

    query_0 = open("d_query0.sql", "w")

    for sql in sqls:
        query_tag = sql.split("old/")[-1].split("-")[0].split("_")[0]
        spj = "_spj" if "spj" in sql else ""
        stream_tag = int(sql.split("-")[-1].split(".sql")[0])
        with open(sql, "r") as f:
            data = f.read()

        queries = [s.strip() for s in data.split(";")]
        queries = [q for q in queries if len(q) > 0]
        queries = [q + ";" for q in queries]
        if len(queries) == 1:
            with open(query_tag + f"s{stream_tag}{spj}" + ".sql", "w") as f:
                f.write(queries[0].strip())
                query_0.write(queries[0].strip())

            d_order.write(f"Q{d_qid},{query_tag}s{stream_tag}{spj}.sql\n")
            d_qorder.write(f"{query_tag}s{stream_tag}{spj}.sql\n")
            d_qid += 1

            if stream_tag == 0:
                s_order.write(f"Q{s_qid},{query_tag}s{stream_tag}{spj}.sql\n")
                s_qorder.write(f"{query_tag}s{stream_tag}{spj}.sql\n")
                s_qid += 1
        else:
            opts = ["a", "b", "c", "d", "e"]
            assert len(queries) <= len(opts)
            for i, query in enumerate(queries):
                with open(query_tag + opts[i] + f"s{stream_tag}{spj}" + ".sql", "w") as f:
                    f.write(query.strip())
                    query_0.write(query.strip())

                d_order.write(f"Q{d_qid},{query_tag}{opts[i]}s{stream_tag}{spj}.sql\n")
                d_qorder.write(f"{query_tag}{opts[i]}s{stream_tag}{spj}.sql\n")
                d_qid += 1

                if stream_tag == 0:
                    s_order.write(f"Q{s_qid},{query_tag}{opts[i]}s{stream_tag}{spj}.sql\n")
                    s_qorder.write(f"{query_tag}{opts[i]}s{stream_tag}{spj}.sql\n")
                    s_qid += 1

    d_order.close()
    d_qorder.close()
    s_order.close()
    s_qorder.close()
    query_0.close()
