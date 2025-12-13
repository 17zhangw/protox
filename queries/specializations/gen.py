from pathlib import Path


if __name__ == "__main__":
    dirs = [d for d in Path(".").glob("dsb_*")]
    for d in dirs:
        sqls = [s for s in d.glob("*.sql")]
        def _key(x):
            x = str(x.stem)
            x = x.split(".sql")[0]
            spj = "spj" in x
            it = int(x.split("_spj")[0].split("s")[-1])
            name = x.split("_spj")[0].split("s")[0] + ("_spj" if spj else "")
            return (name, it)

        sqls = sorted(sqls, key=_key)
        with open(d / "d_order.txt", "w") as f:
            for i, s in enumerate(sqls):
                f.write(f"Q{i},{s.parts[-1]}\n")

        with open(d / "d_qorder.txt", "w") as f:
            for i, s in enumerate(sqls):
                f.write(f"{s.parts[-1]}\n")
