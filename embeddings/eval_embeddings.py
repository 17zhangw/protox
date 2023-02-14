import pandas as pd
import numpy as np
import shutil
import yaml
import itertools
from torch.utils.data import TensorDataset
import gc
import random
import tqdm
from pathlib import Path
import json
import torch
from torch import nn

from envs.spaces.index_space import IndexSpace, IndexRepr
from embeddings.loss import COST_COLUMNS, CostLoss, get_bias_fn
from embeddings.train import _fetch_index_parameters, _load_input_data, _create_vae_model
from embeddings.vae import VAELoss, gen_vae_collate
from embeddings.trainer import StratifiedRandomSampler
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance


def create_eval_parser(subparser):
    parser = subparser.add_parser("eval")
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--intermediate-step", type=int, default=1)

    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--start-epoch", type=int, default=10)
    parser.add_argument("--all-batches", action="store_true")
    parser.set_defaults(func=eval_embedding)


def eval_embedding(args):
    # Load the benchmark configuration.
    with open(args.benchmark_config, "r") as f:
        data = yaml.safe_load(f)
        tables = data["mythril"]["tables"]
        max_attrs, max_cat_features, att_usage, _ = _fetch_index_parameters(data)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models = itertools.chain(*[Path(m).rglob("config") for m in args.models.split(",")])
    models = [m for m in models]
    for model_config in tqdm.tqdm(models):
        if ((Path(model_config).parent) / "FAILED").exists():
            print("Detected failure in: ", model_config)
            continue

        with open(model_config, "r") as f:
            config = json.load(f)

        # Create them here since these are constant for a given "model" configuration.
        groups, dataset, original_y, idx_class, num_classes = None, None, None, None, None
        class_mapping = None
        miner, metric_loss_fn, vae_loss = None, None, None
        vae = _create_vae_model(config, max_attrs, max_cat_features)
        require_cost = config["metric_loss_md"].get("require_cost", False)

        submodules = [f for f in (Path(model_config).parent / "models").glob("*")]
        submodules = sorted(submodules, key=lambda x: int(str(x).split("epoch")[-1]))
        # This is done for semantic sense since the "first" is actually at no epoch.
        modules = [submodules[r] for r in range(-1, len(submodules), args.intermediate_step) if r >= 0]
        if modules[0] != submodules[0]:
            modules = [submodules[0]] + modules

        if modules[-1] != submodules[-1]:
            modules.append(submodules[-1])

        modules = [m for m in modules if int(str(m).split("epoch")[-1]) >= args.start_epoch]

        for i, module in tqdm.tqdm(enumerate(modules), total=len(modules), leave=False):
            epoch = int(str(module).split("epoch")[-1])
            module_path = f"{module}/embedder_{epoch}.pth"

            if Path(f"{module}/stats.txt").exists():
                continue

            # Load the specific epoch model.
            vae.load_state_dict(torch.load(module_path, map_location=device))
            vae.to(device=device).eval()
            collate_fn = gen_vae_collate(max_cat_features)

            if dataset is None:
                # Get the dataset if we need to.
                dataset, original_y, idx_class, _, num_classes = _load_input_data(
                    args.dataset,
                    1.,
                    max_attrs,
                    require_cost,
                    seed=0)

                class_mapping = []
                for c in range(num_classes):
                    if idx_class[idx_class == c].shape[0] > 0:
                        class_mapping.append(c)

                # Use a common loss function.
                metric_loss_fn = CostLoss(config["metric_loss_md"])
                vae_loss = VAELoss(config["loss_fn"], max_attrs, max_cat_features)

            # Construct the accumulator.
            accumulated_stats = {}
            for class_idx in class_mapping:
                accumulated_stats[f"recon_{class_idx}"] = []

            if args.num_batches > 0 or args.all_batches:
                accumulated_stats.update({
                    "recon_accum": [],
                    "metric_accum": [],
                })

                # Setup the dataloader.
                if args.all_batches:
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
                    total = len(dataloader)
                else:
                    sampler = StratifiedRandomSampler(idx_class, max_class=num_classes, batch_size=args.batch_size, allow_repeats=False)
                    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
                    total = min(len(sampler), args.num_batches)
                error = False
                with torch.no_grad():
                    with tqdm.tqdm(total=total, leave=False) as pbar:
                        for (x, y) in dataloader:
                            x = x.to(device=device)

                            if config["use_bias"]:
                                bias_fn = get_bias_fn(config)
                                bias = bias_fn(x, y)
                                if isinstance(bias, torch.Tensor):
                                    bias = bias.to(device=device)
                                else:
                                    lbias = bias[0].to(device=device)
                                    hbias = bias[1].to(device=device)
                                    bias = (lbias, hbias)
                            else:
                                bias = None

                            # Pass it through the VAE with the settings.
                            z, decoded, error = vae(x, bias=bias)
                            if error:
                                # If we've encountered an error, abort early.
                                # Don't use a model that can produce errors.
                                break

                            # Flatten.
                            classes = y[:, -1].flatten()

                            assert metric_loss_fn is not None
                            loss_dict = vae_loss.compute_loss(
                                preds=decoded,
                                unused0=None,
                                unused1=None,
                                data=(x, y),
                                is_eval=True)

                            assert vae_loss.loss_fn is not None
                            for class_idx in class_mapping:
                                y_mask = classes == class_idx
                                x_extract = x[y_mask.bool()]
                                if x_extract.shape[0] > 0:
                                    decoded_extract = decoded[y_mask.bool()]
                                    loss = vae_loss.loss_fn(decoded_extract, x_extract, y[y_mask.bool()])
                                    accumulated_stats[f"recon_{class_idx}"].append(loss.mean().item())

                            input_y = y
                            if y.shape[1] == 1:
                                input_y = y.flatten()

                            metric_loss = metric_loss_fn(z, input_y, None).item()
                            accumulated_stats["recon_accum"].append(loss_dict["recon_loss"]["losses"].item())
                            accumulated_stats["metric_accum"].append(metric_loss)

                            del z
                            del x
                            del y

                            # Break out if we are done.
                            pbar.update(1)
                            total -= 1
                            if total == 0:
                                break

                # Output the evaluated stats.
                with open(f"{module}/stats.txt", "w") as f:
                    stats = {
                        stat_key: (stats if isinstance(stats, np.ScalarType) else (np.mean(stats) if len(stats) > 0 else 0))
                        for stat_key, stats in accumulated_stats.items()
                    }
                    stats["error"] = error.item()
                    f.write(json.dumps(stats, indent=4))

                del dataloader
                gc.collect()
                gc.collect()
