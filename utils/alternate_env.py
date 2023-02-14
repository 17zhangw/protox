import copy
import warnings
from typing import List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class AlternateEnv(VecEnvWrapper):
    def __init__(self, venv: VecEnv, base_env):
        super().__init__(venv)
        self.base_env = base_env

        start_q = None
        for i, k in enumerate(self.base_env.action_space.get_knob_space().keys()):
            if k.startswith("Q") and start_q is None:
                start_q = i

            if start_q is not None:
                assert k.startswith("Q")
        self.start_q = start_q

    def set_active(self, active):
        self.active = active

    def step_async(self, actions: np.ndarray) -> None:
        new_actions = []
        for act in actions:
            parts = []
            for i, act_part in enumerate(act):
                if isinstance(act_part, dict) and i == 0:
                    cmp = {}
                    for k, v in self.tracked_knobs.items():
                        if k not in act_part:
                            cmp[k] = v
                        else:
                            cmp[k] = act_part[k]
                    parts.append(cmp)
                elif isinstance(act_part, tuple) and i == 0:
                    # We are on an index, so dump the knobs.
                    parts.append(copy.deepcopy(self.tracked_knobs))
                    parts.append(act_part)

            if len(parts) == 1:
                # Get the null index action.
                parts.append(self.base_env.action_space.get_index_space().null_action())
            new_actions.append(tuple(parts))
        self.venv.step_async(new_actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        if self.active[0].name in ["Knob-only Agent", "Index-only Agent"]:
            for info in infos:
                if "mutilated_embed" in info:
                    del info["mutilated_embed"]
        else:
            for info in infos:
                if "mutilated_embed" in info:
                    embed = info["mutilated_embed"]
                    if len(embed.shape) == 2:
                        embed = embed[:, self.start_q:-self.base_env.action_space.get_index_space().get_critic_dim()]
                    else:
                        embed = embed[self.start_q:-self.base_env.action_space.get_index_space().get_critic_dim()]
                    info["mutilated_embed"] = embed
        return obs, rews, dones, infos

    def reset(self, **kwargs) -> VecEnvObs:
        observations, info = self.venv.reset(**kwargs)
        self.tracked_knobs = self.base_env.action_space.get_knob_space().get_state(None)
        return observations, info
