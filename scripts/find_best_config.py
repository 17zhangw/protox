import pandas as pd
import glob
from pathlib import Path
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Analyze")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-best", type=int, default=3)

    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    inputs = args.input.split(",")
    paths = []
    for inp in inputs:
        paths.extend([p for p in Path(inp).rglob("result.json")])

    adata = []
    for path in paths:
        with open(path) as f:
            mythril_hpo = Path(path).parts[-3]
            assert "MythrilHPO" in mythril_hpo

            baseline = None
            best = None
            best_seen = None
            best_step = 0
            for line in f:
                data = json.loads(line)

                if baseline is None:
                    hostname = data["hostname"]
                    p = list(Path(path).parts)
                    p = [pp if pp != "ray_results" else hostname for pp in p]
                    p[-1] = "stderr"
                    with open("/".join(p), "r") as f:
                        for line in f:
                            if "Benchmark iteration with metric" in line:
                                baseline = float(line.split(" metric ")[1].split(" ")[0])
                                break

                if best is not None and data["Best Metric"] < best:
                    best_step = data["Timesteps"]

                best = data["Best Metric"]
                best_seen = data["Best Seen Metric"]

            if baseline is not None and best is not None:
                adata.append({
                    "hostname": hostname,
                    "baseline": baseline,
                    "best": best,
                    "path": str(path.parent),
                    "params": str(path.parent / "params.json"),
                })

    df = pd.DataFrame(adata)
    df = df.sort_values(by=["best"]).iloc[:args.num_best]

    params = []
    for tup in df.itertuples():
        with open(tup.params, "r") as f:
            param = json.load(f)
            if args.timeout is not None:
                assert "mythril_args" in param
                assert "timeout" in param["mythril_args"]
                param["mythril_args"]["timeout"] = args.timeout

            params.append(param)

    with open(args.output, "w") as f:
        f.write(json.dumps(params, indent=4))
