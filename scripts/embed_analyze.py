import tqdm
import math
import numpy as np
import yaml
import torch
from pathlib import Path
import json
import sys

sys.path.append("/home/wz2/mythril")

from embeddings.train import _create_vae_model, _fetch_index_parameters
from envs.spaces.index_space import IndexAction, IndexRepr, IndexSpace
import argparse


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def analyze(args):
    INPUT = f"{args.base}/models/epoch{args.epoch}/embedder_{args.epoch}.pth"
    STATS = f"{args.base}/models/epoch{args.epoch}/stats.txt"
    CONFIG = f"{args.base}/config"
    print(INPUT, CONFIG)

    if args.recon_threshold > 0 and Path(STATS).exists():
        with open(STATS, "r") as f:
            stats = json.load(f)
            if stats["recon_accum"] > args.recon_threshold:
                # Exceeded the threshold.
                return
    elif args.recon_threshold > 0:
        print(f"{STATS} does not exist.")
        assert False

    # Load the benchmark configuration.
    with open(args.benchmark_config, "r") as f:
        data = yaml.safe_load(f)
        tables = data["mythril"]["tables"]
        max_attrs, max_cat_features, att_usage, class_mapping = _fetch_index_parameters(data)

    with open(CONFIG, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _create_vae_model(config, max_attrs, max_cat_features)
    # Load the specific epoch model.
    vae.load_state_dict(torch.load(INPUT, map_location=device))
    vae.to(device=device).eval()

    idxs = IndexSpace(
        agent_type="wolp",
        tables=tables,
        max_num_columns=0,
        index_repr=IndexRepr.ONE_HOT_DETERMINISTIC.name,
        seed=np.random.randint(1, 1e10),
        latent_dim=config["latent_dim"],
        index_vae_model=vae,
        index_output_scale=1.,
        attributes_overwrite=att_usage)
    idxs.rel_metadata = att_usage
    idxs._build_mapping(att_usage)

    def decode_to_classes(rand_points):
        with torch.no_grad():
            rand_decoded = idxs._decode(act=rand_points)
            classes = {}
            for r in range(rand_points.shape[0]):
                act = idxs.index_repr_policy.sample_action(idxs.np_random, rand_decoded[r], att_usage, False, True)
                idx_class = idxs.get_index_class(act)
                if idx_class not in classes:
                    classes[idx_class] = 0
                classes[idx_class] += 1
        return sorted([(k, v) for k, v in classes.items()], key=lambda x: x[1], reverse=True)

    output_scale = config["metric_loss_md"]["output_scale"]
    bias_separation = config["metric_loss_md"]["bias_separation"]
    num_segments = min(args.max_segments, math.ceil(1.0 / bias_separation))

    base = 0
    with open(f"{args.base}/models/epoch{args.epoch}/analyze.txt", "w") as f:
        for _ in tqdm.tqdm(range(num_segments), total=num_segments, leave=False):
            classes = decode_to_classes(torch.rand(args.num_points, config["latent_dim"]) * output_scale + base)
            if args.top != 0:
                classes = classes[:args.top]

            f.write(f"Generating range {base} - {base + output_scale}\n")
            f.write("\n".join([f"{k}: {v / args.num_points}" for (k, v) in classes]))
            f.write("\n")
            if not args.glob:
                print(f"Generating range {base} - {base + output_scale}")
                print("\n".join([f"{k}: {v / args.num_points}" for (k, v) in classes]))
            base += output_scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="AnalyzeEmbed")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--num-points", type=int, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--recon-threshold", type=float, default=0)
    parser.add_argument("--max-segments", type=int, default=20)
    parser.add_argument("--start-epoch", type=int, default=10)
    args = parser.parse_args()

    vargs = DotDict(vars(args))
    paths = sorted([f for f in args.base.rglob("embedder_*.pth") if "optimizer" not in str(f)])
    for p in tqdm.tqdm(paths):
        epoch = int(str(p).split("embedder_")[-1].split(".pth")[0])
        if epoch < args.start_epoch:
            continue

        vargs["base"] = p.parent.parent.parent
        vargs["epoch"] = epoch
        analyze(vargs)
