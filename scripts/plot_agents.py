import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import json
from pathlib import Path
import pandas as pd


def parse_file(agent, file, base_timeout, duration, populate_interval):
    new_data = []
    try:
        data = pd.read_csv(file)
    except:
        data = None
        new_data = [
            {"time_since_start": 0., "runtime": base_timeout},
            {"time_since_start": duration, "runtime": base_timeout},
        ]

    tracking_runtime = base_timeout
    if data is not None:
        runtime_cols = [c for c in data if "runtime" in c]
        for tup in data.itertuples():
            new_data.append({
                "time_since_start": getattr(tup, "time_since_start") / 3600.0,
                "runtime": min([getattr(tup, c) for c in runtime_cols] + [tracking_runtime]),
            })

            if new_data[-1]["runtime"] < tracking_runtime:
                tracking_runtime = new_data[-1]["runtime"]

    if new_data[0]["time_since_start"] != 0:
        new_data = [{"time_since_start": 0., "runtime": base_timeout}] + new_data

    if new_data[-1]["time_since_start"] < duration:
        new_data.append({
            "time_since_start": duration,
            "runtime": tracking_runtime,
        })

    if populate_interval > 0:
        actual_data = []
        current_interval = 0
        while current_interval <= duration:
            if len(new_data) == 0:
                assert len(actual_data) > 0
                actual_data.append({
                    "time_since_start": current_interval,
                    "runtime": actual_data[-1]["runtime"],
                })
                current_interval += populate_interval
            elif current_interval < new_data[0]["time_since_start"]:
                actual_data.append({
                    "time_since_start": current_interval,
                    "runtime": actual_data[-1]["runtime"],
                })
                current_interval += populate_interval
            else:
                while len(new_data) > 0 and current_interval >= new_data[0]["time_since_start"]:
                    current_data = new_data[0]
                    new_data = new_data[1:]

                actual_data.append({
                    "time_since_start": current_interval,
                    "runtime": current_data["runtime"],
                })
                current_interval += populate_interval
        return pd.DataFrame(actual_data)
    else:
        return pd.DataFrame(new_data)


def consolidate_datum(args, datums):
    assert len(datums) > 0
    agent_type = datums[0].iloc[0].agent_type
    if agent_type == "Bao":
        initial_baseline = max([d.runtime.max() for d in datums])
    else:
        initial_baseline = args.base_timeout

    times = pd.concat(datums, ignore_index=True).sort_values(by=["time_since_start"]).time_since_start
    new_data = []
    for datum in datums:
        for i, time in enumerate(times):
            if i == 0:
                if time != 0:
                    new_data.append({
                        "agent_type": datums[0].iloc[0].agent_type,
                        "agent": datums[0].iloc[0].agent,
                        "time_since_start": 0.,
                        "runtime": initial_baseline,
                    })
                else:
                    new_data.append({
                        "agent_type": datums[0].iloc[0].agent_type,
                        "agent": datums[0].iloc[0].agent,
                        "time_since_start": 0.,
                        "runtime": initial_baseline,
                    })
            else:
                match_tup = datum[datum.time_since_start == time]
                assert match_tup.shape[0] <= 1
                if match_tup.shape[0] == 0:
                    last_data = new_data[-1].copy()
                    last_data["time_since_start"] = time
                    new_data.append(last_data)
                else:
                    last_data = dict(match_tup.iloc[0])
                    last_data["runtime"] = min(last_data["runtime"], new_data[-1]["runtime"])
                    new_data.append(last_data)

    return pd.DataFrame(new_data)


def error_fn(x):
    return (min(x), max(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plot")
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--base-timeout", type=int, required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--populate-interval", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pd.options.display.max_rows = 100

    with open(args.agents, "r") as f:
        agents = json.load(f)

    holistic_datum = []
    bao_baseline = args.base_timeout
    for agent, files in agents.items():
        datum = []
        for file in files:
            if "pgtune" in agent.lower() and "bao" not in agent.lower():
                agent_type = "PGTune+Dexter"
            elif "bao" in agent.lower():
                agent_type = "Bao"
            elif "udo" in agent.lower():
                agent_type = "UDO"
            elif "unitune" in agent.lower():
                agent_type = "UniTune"
            elif "us" in agent.lower():
                agent_type = "Us"
            else:
                print(agent)
                assert False

            if agent_type == "PGTune+Dexter":
                with open(file, "r") as f:
                    lines = f.readlines()
                    lines = [l.strip() for l in lines]
                    lines = [l for l in lines if len(l) > 0]
                    runtimes = [float(l) for l in lines]
                    bao_baseline = min(runtimes)
                    data = pd.DataFrame([
                        {
                            "time_since_start": 0.,
                            "runtime": bao_baseline,
                        }
                    ])

            elif agent_type != "Bao":
                data = parse_file(agent, file, args.base_timeout, args.duration, args.populate_interval)
            else:
                data = parse_file(agent, file, bao_baseline, args.duration, args.populate_interval)

            data["agent"] = agent
            data["agent_type"] = agent_type
            datum.append(data)

        holistic_datum.append(consolidate_datum(args, datum))

    all_data = pd.concat(holistic_datum, ignore_index=True)
    all_data["Agent"] = all_data["agent"]

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=2., rc={"figure.figsize": (18, 10)})
    sns_plot = sns.lineplot(
        data=all_data,
        x="time_since_start",
        y="runtime",
        hue="Agent",
        style="Agent",
        markers=True,
        markersize=12,
        dashes=True,
        err_style="band",
        errorbar=error_fn,
    )

    sns_plot.set_xlim((0, args.duration))
    sns_plot.set_ylim((0, args.base_timeout))
    sns_plot.set_xlabel("Time (hr)")
    sns_plot.set_ylabel("Runtime (seconds)")

    plt.tight_layout()
    plt.show()
    sns_plot.figure.savefig(args.output)
