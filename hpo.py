import json
import glob
import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import time

import ray
from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial
from ray.tune import with_parameters, Tuner, TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import Trainable, SyncConfig
from ray.air import session, RunConfig, FailureConfig
from parse_args import parse_cmdline_args
from hyperopt import hp

from agents.hpo import construct_wolp_config


METRIC = "Best Metric"

class TuneOpt(Trainable):
    def f_unpack_dict(self, dct):
        """
        Unpacks all sub-dictionaries in given dictionary recursively.
        There should be no duplicated keys across all nested
        subdictionaries, or some instances will be lost without warning

        Source: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

        Parameters:
        ----------------
        dct : dictionary to unpack

        Returns:
        ----------------
        : unpacked dictionary
        """
        res = {}
        for (k, v) in dct.items():
            if "mythril_" in k:
                res[k] = v
            elif isinstance(v, dict):
                res = {**res, **self.f_unpack_dict(v)}
            else:
                res[k] = v
        return res


    def setup(self, hpo_config):
        print("HPO Configuration: ", hpo_config)
        assert "mythril_args" in hpo_config
        mythril_args = hpo_config["mythril_args"]
        mythril_dir = os.path.expanduser(mythril_args["mythril_dir"])
        sys.path.append(mythril_dir)

        from tune import TuneTrial, TimeoutChecker
        from utils.dotdict import DotDict
        from agents.hpo import mutate_wolp_config
        hpo_config = DotDict(self.f_unpack_dict(hpo_config))
        mythril_args = DotDict(self.f_unpack_dict(mythril_args))

        # Compute the limit.
        self.early_kill = mythril_args["early_kill"]
        self.stabilize_kill = 0
        if "stabilize_kill" in mythril_args:
            self.stabilize_kill = mythril_args["stabilize_kill"]

        self.last_best_time = None
        self.last_best_metric = None

        self.duration = mythril_args["duration"] * 3600
        self.workload_timeout = mythril_args["workload_timeout"]
        self.timeout = TimeoutChecker(mythril_args["duration"])
        if mythril_args.agent == "wolp":
            benchmark, pg_path, port = mutate_wolp_config(self.logdir, mythril_dir, hpo_config, mythril_args)
        else:
            assert False, f"Unspecified agent {mythril_args.agent}"

        self.pg_path = pg_path
        self.port = port

        # We will now overwrite the config files.
        mythril_args["config"] = str(Path(self.logdir) / "config.yaml")
        mythril_args["model_config"] = str(Path(self.logdir) / "model_params.yaml")
        mythril_args["benchmark_config"] = str(Path(self.logdir) / f"{benchmark}.yaml")
        mythril_args["reward"] = hpo_config.reward
        mythril_args["horizon"] = hpo_config.horizon
        self.trial = TuneTrial()
        self.trial.setup(mythril_args, self.timeout)
        self.start_time = time.time()

    def step(self):
        self.timeout.resume()
        data = self.trial.step()

        # Decrement remaining time.
        self.timeout.pause()
        if self.timeout():
            self.cleanup()
            data[ray.tune.result.DONE] = True

        if self.early_kill:
            if (time.time() - self.start_time) >= 10800:
                if "Best Metric" in data and data["Best Metric"] >= 190:
                    self.cleanup()
                    data[ray.tune.result.DONE] = True
            elif (time.time() - self.start_time) >= 7200:
                if "Best Metric" in data and data["Best Metric"] >= 250:
                    self.cleanup()
                    data[ray.tune.result.DONE] = True

        if self.stabilize_kill > 0 and "Best Metric" in data:
            if self.last_best_metric is None or data["Best Metric"] < self.last_best_metric:
                self.last_best_metric = data["Best Metric"]
                self.last_best_time = time.time()

            if self.last_best_time is not None and (time.time() - self.last_best_time) > self.stabilize_kill * 3600:
                self.trial.logger.info("Killing due to run stabilizing.")
                self.cleanup()
                data[ray.tune.result.DONE] = True

        return data

    def cleanup(self):
        self.trial.cleanup()
        if Path(f"{self.pg_path}/{self.port}.signal").exists():
            os.remove(f"{self.pg_path}/{self.port}.signal")

    def save_checkpoint(self, checkpoint_dir):
        # We can't actually do anything about this right now.
        pass

    def load_checkpoint(self, checkpoint_dir):
        # We can't actually do anything about this right now.
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Mythril-HPO")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent jobs.")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials.")
    parser.add_argument("--mythril-dir", type=str, default="~/mythril", help="Location to mythril.")
    parser.add_argument("--initial-configs", type=Path, default=None)
    parser.add_argument("--initial-repeats", type=int, default=1)
    parser.add_argument("--early-kill", action="store_true")
    args = parse_cmdline_args(parser, path_type=str)
    obj = "max" if args.target == "tps" else "min"

    # Get the system knobs.
    with open(args.hpo_config, "r") as f:
        config = yaml.safe_load(f)["mythril"]
        system_knobs = config["system_knobs"]

    # Per query knobs.
    benchmark_config = args.hpo_benchmark_config if args.hpo_benchmark_config else args.benchmark_config
    with open(benchmark_config, "r") as f:
        bb_config = yaml.safe_load(f)["mythril"]
        per_query_scan_method = bb_config["per_query_scan_method"]
        per_query_select_parallel = bb_config["per_query_select_parallel"]
        index_space_aux_type = bb_config["index_space_aux_type"]
        index_space_aux_include = bb_config["index_space_aux_include"]
        per_query_knobs = bb_config["per_query_knobs"]
        per_query_knob_gen = bb_config["per_query_knob_gen"]
        query_spec = bb_config["query_spec"]

    # Connect to cluster or die.
    ray.init(address="localhost:6379", log_to_driver=False)

    # Config.
    if args.agent == "wolp":
        config = construct_wolp_config(dict(args))
    else:
        assert False, f"Unspecified agent {args.agent}"

    # Pass the knobs through.
    config["mythril_system_knobs"] = system_knobs
    config["mythril_per_query_knobs"] = per_query_knobs
    config["mythril_per_query_scan_method"] = per_query_scan_method
    config["mythril_per_query_select_parallel"] = per_query_select_parallel
    config["mythril_index_space_aux_type"] = index_space_aux_type
    config["mythril_index_space_aux_include"] = index_space_aux_include
    config["mythril_per_query_knob_gen"] = per_query_knob_gen
    config["mythril_query_spec"] = query_spec

    # Scheduler.
    scheduler = FIFOScheduler()

    initial_configs = None
    if args.initial_configs is not None and args.initial_configs.exists():
        initial_tmp_configs = []
        with open(args.initial_configs, "r") as f:
            initial_configs = json.load(f)

            for config in initial_configs:
                if "mythril_per_query_knobs" not in config:
                    config["mythril_per_query_knobs"] = per_query_knobs
                if "mythril_per_query_scan_method" not in config:
                    config["mythril_per_query_scan_method"] = per_query_scan_method
                if "mythril_per_query_select_parallel" not in config:
                    config["mythril_per_query_select_parallel"] = per_query_select_parallel
                if "mythril_index_space_aux_type" not in config:
                    config["mythril_index_space_aux_type"] = index_space_aux_type
                if "mythril_index_space_aux_include" not in config:
                    config["mythril_index_space_aux_include"] = index_space_aux_include
                if "mythril_per_query_knob_gen" not in config:
                    config["mythril_per_query_knob_gen"] = per_query_knob_gen
                if "mythril_system_knobs" not in config:
                    config["mythril_system_knobs"] = system_knobs

                assert "mythril_args" in config
                config["mythril_args"]["early_kill"] = args.early_kill

                for _ in range(args.initial_repeats):
                    initial_tmp_configs.append(config)
        initial_configs = initial_tmp_configs

    # Search.
    search = BasicVariantGenerator(points_to_evaluate=initial_configs, max_concurrent=args.max_concurrent)

    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=args.num_trials,
        max_concurrent_trials=args.max_concurrent,
        chdir_to_trial_dir=True,
        metric=METRIC,
        mode=obj,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"MythrilHPO_{dtime}",
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(upload_dir=None, syncer=None),
        verbose=2,
        log_to_file=True,
    )

    tuner = ray.tune.Tuner(
        TuneOpt,
        tune_config=tune_config,
        run_config=run_config,
        param_space=config,
    )

    results = tuner.fit()
    if results.num_errors > 0:
        print("Encountered exceptions!")
        for i in range(len(results)):
            if results[i].error:
                print(f"Trial {results[i]} FAILED")
        assert False
    print("Best hyperparameters found were: ", results.get_best_result(metric=METRIC, mode="max").config)
