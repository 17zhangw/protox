import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import json
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plot")
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base-timeout", required=True, type=float)
    args = parser.parse_args()

    with open(args.agents, "r") as f:
        agents = json.load(f)

    datum = []
    for agent, files in agents.items():
        min_runtime = 1e9
        for file in files:
            if "pgtune" in agent.lower() and "bao" not in agent.lower():
                with open(file, "r") as f:
                    lines = [l.strip() for l in f.readlines()]
                    lines = [float(l) for l in lines if len(l) > 0]
                    min_runtime = min(lines)
            else:
                try:
                    data = pd.read_csv(file)
                except:
                    min_runtime = args.base_timeout
                    data = None

                if data is not None:
                    assert len(data) > 0
                    runtime_cols = [c for c in data if "runtime" in c]
                    for tup in data.itertuples():
                        runtime = min([getattr(tup, c) for c in runtime_cols])
                        if runtime < min_runtime:
                            min_runtime = runtime

        datum.append({"Agent": agent, "Benchmark": args.benchmark, "Runtime": min_runtime})

    datum = pd.DataFrame(datum)

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1., rc={"figure.figsize": (12, 12)})
    sns_plot = sns.barplot(
        data=datum,
        x="Benchmark",
        y="Runtime",
        hue="Agent",
    )

    sns_plot.set_xlabel("")
    #sns_plot.set_xlabel("Time (hr)")
    #sns_plot.set_ylabel("Time (seconds)")

    plt.tight_layout()
    plt.show()
