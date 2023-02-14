import copy
import os
import logging
import time
import json
import numpy as np
import argparse
import yaml
import pandas as pd
from pathlib import Path
from scipy.stats import qmc
import torch
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, FlattenObservation
import random

from agents import setup_agent
from agents.common.env_util import unwrap_wrapper, unwrap_to_base_env
from envs.spec import Spec
from envs.reward import RewardUtility
from utils.target_state_reset import TargetStateResetWrapper
from utils.mythril_vec_env_wrapper import MythrilVecEnvWrapper
from utils.alternate_env import AlternateEnv
from utils.logger import Logger
from agents.common.vec_env import VecCheckNan
from utils.dotdict import DotDict
import pickle
from agents.hpo import mutate_wolp_config


class TimeoutChecker(object):
    running = False
    limit = False
    remain = 0
    start = 0

    def __init__(self, duration):
        self.limit = (duration * 3600) > 0
        self.remain = int(duration * 3600)

    def resume(self):
        self.start = time.time()
        self.running = True

    def pause(self):
        if self.limit and self.running:
            self.remain -= int(time.time() - self.start)
        self.running = False

    def __call__(self):
        if not self.limit:
            return False

        if self.remain <= 0:
            return True

        if self.running:
            return int(time.time() - self.start) >= self.remain

        return False


class AlternateTrial(object):
    def setup(self, output_dir, mythril_args, hpo_config, benchmark, timeout_checker):
        torch.set_default_dtype(torch.float32)
        self.args = args = DotDict(mythril_args)

        # Setup the seed
        seed = self.args.seed if self.args.seed != -1 else np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create environment from the "holistic" one.
        self.spec = Spec(self.args.agent, seed, config_path=self.args.config, benchmark_config_path=self.args.benchmark_config, horizon=self.args.horizon, workload_timeout=self.args.workload_timeout)
        self.logger = self.spec.logger
        # Create the reward utility.
        self.reward_utility = RewardUtility(self.args.target, self.args.reward, self.spec.reward_scaler)
        # Create the environment.
        self.base_env = self.env = gym.make("Postgres-v0",
            spec=self.spec,
            horizon=self.args.horizon,
            timeout=self.args.timeout,
            reward_utility=self.reward_utility,
            logger=self.logger)
        self.actual_base_env = unwrap_to_base_env(self.base_env)
        # Attach a target state reset wrapper around it.
        self.target_state_wrapper = self.env = TargetStateResetWrapper(self.env, self.spec.maximize_state, self.reward_utility, maximize_knobs_only=False, start_reset=False)
        # Now attach a flatten observation wrapper.
        self.env = FlattenObservation(self.env)
        self.normalize_env = None
        self.normalize_reward = None
        if self.spec.normalize_state:
            # Normalize state if we are asked to.
            self.normalize_env = self.env = NormalizeObservation(self.env)
        if self.spec.normalize_reward:
            # Normalize reward if we are asked to.
            self.normalize_reward = self.env = NormalizeReward(self.env, gamma=self.spec.gamma)
        self.vec_env = self.env = MythrilVecEnvWrapper([lambda: self.env])
        self.env = VecCheckNan(self.env, raise_exception=True, warn_once=False)
        self.env = AlternateEnv(self.env, self.base_env)

        old_as = self.env.action_space

        ko_spec = Spec("wolp", seed, f"{output_dir}/ko_config.yaml", f"{output_dir}/ko_{benchmark}.yaml", horizon=self.args.horizon, workload_timeout=self.args.workload_timeout, logger=self.logger)
        self.env.action_space = ko_spec.action_space
        ko_agent = setup_agent(self.env, ko_spec, seed, self.args.agent, self.args.model_config)
        ko_agent.set_logger(self.logger)
        ko_agent.set_timeout_checker(timeout_checker)
        ko_agent._ignore_index = True
        ko_agent.name = "Knob-only Agent"

        io_spec = Spec("wolp", seed, f"{output_dir}/io_config.yaml", f"{output_dir}/io_{benchmark}.yaml", horizon=self.args.horizon, workload_timeout=self.args.workload_timeout, logger=self.logger)
        io_spec.action_space.spaces = (self.spec.action_space.get_index_space(),)

        self.env.action_space = io_spec.action_space
        io_agent = setup_agent(self.env, io_spec, seed, self.args.agent, self.args.model_config)
        io_agent.set_logger(self.logger)
        io_agent.set_timeout_checker(timeout_checker)
        io_agent.name = "Index-only Agent"

        self.qo_spec = qo_spec = Spec("wolp", seed, f"{output_dir}/pqo_config.yaml", f"{output_dir}/{benchmark}.yaml", horizon=self.args.horizon, workload_timeout=self.args.workload_timeout, logger=self.logger)
        self.env.action_space = qo_spec.action_space
        qo_agent = setup_agent(self.env, qo_spec, seed, self.args.agent, self.args.model_config)
        qo_agent.set_logger(self.logger)
        qo_agent.set_timeout_checker(timeout_checker)
        qo_agent._ignore_index = True
        qo_agent.name = "Query-only Agent"

        self.env.action_space = old_as

        self.current_agent = (ko_agent, ko_spec)
        self.agents = [(io_agent, io_spec), (qo_agent, qo_spec)]

        # Hijack the workload correctly.
        self.actual_base_env.workload = self.current_agent[1].workload
        # Have the LSC track the index space LSC.
        self.actual_base_env.action_space.get_index_space().lsc = io_spec.action_space.get_index_space().lsc
        # Freeze the index.
        io_spec.action_space.get_index_space().lsc.freeze()

        self.env.set_active(self.current_agent)

        self.env_init = False


    def step(self):
        if not self.env_init:
            # Create the baseline repository data.
            self.state, infos = self.env.reset()
            info = infos[0]
            baseline_reward = info["baseline_reward"]
            self.baseline_metric = info["baseline_metric"]
            logging.info(f"Baseilne Metric: {self.baseline_metric}. Baseline Reward: {baseline_reward}")
            self.env_init = True

            self.current_agent[0]._last_obs = self.state

        episode = self.current_agent[0]._episode_num
        it = self.current_agent[0].num_timesteps
        logging.info(f"[{self.current_agent[0].name}] Starting episode: {episode+1}, iteration: {it+1}")

        self.current_agent[0].learn(
            total_timesteps=1,
            log_interval=None,
            reset_num_timesteps=False,
            progress_bar=False)

        self.logger.advance()


    def switch(self):
        leaving_index = self.current_agent[0].name == "Index-only Agent"

        # Swap the current.
        self.agents.append(self.current_agent)
        self.current_agent = self.agents[0]
        self.agents = self.agents[1:]

        # Hijack.
        self.actual_base_env.workload = self.current_agent[1].workload
        self.env.set_active(self.current_agent)

        obs, infos = self.env.reset()

        if leaving_index:
            # Freeze.
            self.agents[-1][1].action_space.get_index_space().lsc.freeze()

        if self.current_agent[0].name == "Index-only Agent":
            # Unfreeze if we are going to index.
            self.current_agent[0].action_space.get_index_space().lsc.unfreeze()

        if len(infos) > 0:
            logging.info(infos)

        self.current_agent[0]._last_obs = obs


    def cleanup(self):
        self.logger.flush()
        self.env.close()


def f_unpack_dict(dct):
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
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Mythril-Alternative")
    parser.add_argument("--params-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--switch-time", default=0, type=int, help="Switch time in seconds")
    parser.add_argument("--switch-step", default=False, action="store_true")
    args = parser.parse_args()

    duration_secs = args.duration * 3600

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.params_file) as f:
        data = json.load(f)
        hpo_config = data[0]

    # Change working directory.
    os.chdir(args.output_dir)

    mythril_args = hpo_config["mythril_args"]
    mythril_dir = os.path.expanduser(mythril_args["mythril_dir"])
    hpo_config = DotDict(f_unpack_dict(hpo_config))
    mythril_args = DotDict(f_unpack_dict(mythril_args))

    benchmark, pg_path, port = mutate_wolp_config(args.output_dir, mythril_dir, hpo_config, mythril_args)

    with open(f"{args.output_dir}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(f"{args.output_dir}/{benchmark}.yaml", "r") as f:
        benchmark_config = yaml.safe_load(f)

    # Setup knob-only.
    with open(f"{args.output_dir}/ko_config.yaml", "w") as f:
        c = copy.deepcopy(config)
        c["mythril"]["index_space"] = False
        c["mythril"]["output_log_path"] = str(args.output_dir)
        c["mythril"]["tensorboard_path"] = str(args.output_dir / "tensorboard")
        c["mythril"]["workload_eval_mode"] = "pq"
        c["mythril"]["workload_eval_inverse"] = False
        c["mythril"]["workload_eval_reset"] = False
        yaml.dump(c, stream=f, default_flow_style=False)
    with open(f"{args.output_dir}/ko_{benchmark}.yaml", "w") as f:
        c = copy.deepcopy(benchmark_config)
        c["mythril"]["per_query_knobs"] = {}
        c["mythril"]["per_query_knob_gen"] = {}
        c["mythril"]["per_query_scan_method"] = False
        c["mythril"]["per_query_select_parallel"] = False
        yaml.dump(c, stream=f, default_flow_style=False)

    # Setup index-only.
    with open(f"{args.output_dir}/io_config.yaml", "w") as f:
        c = copy.deepcopy(config)
        c["mythril"]["knob_space"] = False
        c["mythril"]["workload_eval_mode"] = "pq"
        c["mythril"]["workload_eval_inverse"] = False
        c["mythril"]["workload_eval_reset"] = False
        yaml.dump(c, stream=f, default_flow_style=False)
    with open(f"{args.output_dir}/io_{benchmark}.yaml", "w") as f:
        c = copy.deepcopy(benchmark_config)
        c["mythril"]["per_query_knobs"] = {}
        c["mythril"]["per_query_knob_gen"] = {}
        c["mythril"]["per_query_scan_method"] = False
        c["mythril"]["per_query_select_parallel"] = False
        yaml.dump(c, stream=f, default_flow_style=False)

    # Setup per-query only.
    with open(f"{args.output_dir}/pqo_config.yaml", "w") as f:
        c = copy.deepcopy(config)
        c["mythril"]["index_space"] = False
        c["mythril"]["system_knobs"] = {}
        yaml.dump(c, stream=f, default_flow_style=False)

    mythril_args["config"] = str(Path(args.output_dir) / "config.yaml")
    mythril_args["model_config"] = str(Path(args.output_dir) / "model_params.yaml")
    mythril_args["benchmark_config"] = str(Path(args.output_dir) / f"{benchmark}.yaml")
    mythril_args["reward"] = hpo_config.reward
    mythril_args["horizon"] = hpo_config.horizon

    checker = TimeoutChecker(args.duration)

    trial = AlternateTrial()
    trial.setup(args.output_dir, mythril_args, hpo_config, benchmark, checker)
    start_time = time.time()
    while (time.time() - start_time) <= duration_secs:
        part_start = time.time()
        if args.switch_step:
            trial.step()
            trial.switch()
        else:
            while (time.time() - part_start) <= args.switch_time:
                if (time.time() - start_time) > duration_secs:
                    break
                trial.step()

            trial.switch()
    trial.cleanup()
