import shutil
import os
import pandas as pd
import json
import numpy as np
import torch
import argparse
from pathlib import Path
import tqdm

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def _load_data(args):
    data = []
    stats = [s for s in args.input.rglob("stats.txt")]
    for stat in stats:
        if "curated" in str(stat):
            continue

        info = {}
        with open(stat, "r") as f:
            stat_dict = json.load(f)
            info["recon"] = stat_dict["recon_accum"]
            info["metric"] = stat_dict["metric_accum"]
            info["elbo"] = info["recon"]
            info["elbo_metric"] = info["recon"] + info["metric"]
            info["all_loss"] = info["recon"] + info["metric"]

            if args.recon is not None and args.recon < info["recon"]:
                # Did not pass reconstruction threshold.
                continue

            info["path"] = str(stat.parent)
            info["root"] = str(stat.parent.parent.parent)

        with open(stat.parent.parent.parent / "config", "r") as f:
            config = json.load(f)
            def recurse_set(source, target):
                for k, v in source.items():
                    if isinstance(v, dict):
                        recurse_set(v, target)
                    else:
                        target[k] = v
            recurse_set(config, info)
            if args.latent_dim is not None:
                if info["latent_dim"] != args.latent_dim:
                    continue

            if not info["weak_bias"]:
                continue

            output_scale = config["metric_loss_md"]["output_scale"]
            bias_sep = config["metric_loss_md"]["bias_separation"]

            if args.bias_sep is not None:
                if args.bias_sep != bias_sep:
                    continue

            info["analyze_file"] = str(Path(stat).parent / "analyze.txt")

        data.append(info)

    data = pd.DataFrame(data)
    data = data.loc[:, ~(data == data.iloc[0]).all()]
    if "output_scale" not in data:
        data["output_scale"] = output_scale

    if "bias_separation" not in data:
        data["bias_separation"] = bias_sep
    return data


def _attach(data, raw_data, num_limit=0):
    # As the group index goes up, the perf should go up (i.e., bounds should tighten)
    filtered_data = {}
    new_data = []
    for tup in tqdm.tqdm(data.itertuples(), total=data.shape[0]):
        tup = DotDict({k: getattr(tup, k) for k in data.columns})
        if raw_data is not None and Path(tup.analyze_file).exists():
            def compute_dist_score(current_dists, base, upper):
                nonlocal filtered_data
                key = (base, upper)
                if key not in filtered_data:
                    data_range = raw_data[(raw_data.quant_mult_cost_improvement >= base) & (raw_data.quant_mult_cost_improvement < upper)]
                    filtered_data[key] = data_range
                    if data_range.shape[0] == 0:
                        return 0
                else:
                    data_range = filtered_data[key]

                error = 0
                if "real_idx_class" in data_range:
                    data_dists = data_range.real_idx_class.value_counts() / data_range.shape[0]
                else:
                    data_dists = data_range.idx_class.value_counts() / data_range.shape[0]

                for (key, dist) in zip(data_dists.index, data_dists):
                    if str(key) not in current_dists:
                        error += dist
                    else:
                        error += abs(current_dists[str(key)] - dist)
                return error

            with open(tup.analyze_file, "r") as f:
                errors = []
                drange = (None, None)
                current_dists = {}

                for line in f:
                    if "Generating range" in line:
                        if len(current_dists) > 0:
                            assert drange[0] is not None
                            errors.append(compute_dist_score(current_dists, drange[0], drange[1]))
                            if num_limit > 0 and len(errors) >= num_limit:
                                current_dists = {}
                                break

                        if drange[0] is None:
                            drange = (1. - tup.bias_separation, 1.01)
                        else:
                            drange = (drange[0] - tup.bias_separation, drange[0])
                        current_dists = {}

                    else:
                        ci = line.split(": ")[0]
                        dist = float(line.strip().split(": ")[-1])
                        current_dists[ci] = dist

                if len(current_dists) > 0:
                    # Put the error in.
                    errors.append(compute_dist_score(current_dists, 0., tup.bias_separation))

                tup["idx_class_errors"] = ",".join([str(np.round(e, 2)) for e in errors])
                for i, e in enumerate(errors):
                    tup[f"idx_class_error{i}"] = np.round(e, 2)

                if len(errors) > 0:
                    tup["idx_class_mean_error"] = np.mean(errors)
                    tup["idx_class_total_error"] = np.sum(errors)
                    tup["idx_class_min_error"] = np.min(errors)
                    tup["idx_class_max_error"] = np.max(errors)
        new_data.append(dict(tup))
    return pd.DataFrame(new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="EmbedFilter")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data", type=Path, required=False)
    parser.add_argument("--idx-limit", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--bias-sep", type=float, default=None)
    parser.add_argument("--recon", type=float, default=None)

    parser.add_argument("--curate", action="store_true")
    parser.add_argument("--num-curate", type=int, default=10)
    parser.add_argument("--allow-all", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--flatten-idx", type=int, default=0)
    args = parser.parse_args()

    data = _load_data(args)

    if args.data is not None and args.data.exists():
        raw_data = pd.read_parquet(args.data)
        data = _attach(data, raw_data, args.idx_limit)

    data.to_csv(args.out, index=False)

    if args.curate:
        if (args.input / "curated").exists():
            shutil.rmtree(args.input / "curated")
        os.mkdir(args.input / "curated")

        if "idx_class_total_error" in data:
            data["elbo"] = data.elbo + data.idx_class_total_error

        if args.allow_all:
            df = data.sort_values(by=["elbo"]).iloc[:args.num_curate]
        else:
            df = data.sort_values(by=["elbo"]).groupby(by=["root"]).head(1).iloc[:args.num_curate]

        if not args.flatten:
            for tup in df.itertuples():
                shutil.copytree(tup.path, f"{args.input}/curated/{tup.path}", dirs_exist_ok=True)
                shutil.copy(Path(tup.root) / "config", f"{args.input}/curated/{tup.root}/config")
        else:
            idx = args.flatten_idx
            Path(f"{args.input}/curated").mkdir(parents=True, exist_ok=True)
            info_txt = open(f"{args.input}/curated/info.txt", "w")

            for tup in df.itertuples():
                epoch = int(str(tup.path).split("epoch")[-1])
                shutil.copytree(tup.path, f"{args.input}/curated/model{idx}")
                shutil.copy(Path(tup.root) / "config", f"{args.input}/curated/model{idx}/config")

                info_txt.write(f"model{idx}/embedder_{epoch}.pth\n")
                idx += 1

            info_txt.close()
