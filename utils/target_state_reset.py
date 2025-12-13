import random
from enum import unique, Enum
import math
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gymnasium as gym
import numpy as np
from agents.common.env_util import unwrap_to_base_env
from envs.pg_env import PostgresEnv
import random


@unique
class ResetPurity(Enum):
    SHADOW = 0
    PURE = 1
    LOCAL_SHADOW = 2


class Node(object):
    node_counter = 1

    def __init__(self, metric, obs, accum_metric, state, reset_lsc):
        # Count node Ids.
        self.node_id = Node.node_counter
        Node.node_counter += 1

        self.metric = metric
        self.shadow_metric = metric
        self.obs = obs
        self.accum_metric = accum_metric
        self.state = state
        self.reset_lsc = reset_lsc

        # Number of visits through this node.
        self.visits = 0
        # Number of attempts *from* this node.
        self.shadow_visits = 0
        self.children = []
        self.rollout_lscs = set()


def print_to_fh(node, fh):
    node_info = "Node({}, {}, {}, {})\n".format(
        node.node_id,
        node.metric,
        node.visits,
        node.shadow_visits,
    )
    fh.write(node_info)

    c = "-> [" + ",".join([str(n.node_id) for n in node.children]) + "]\n\n"
    fh.write(c)

    for c in node.children:
        print_to_fh(c, fh)


class MaximalPolicy(object):
    def __init__(self, minimize=True):
        self.minimize = minimize

    def rollout(self, root):
        if len(root.children) == 0:
            return [root]

        mcs = [root.metric] + [c.metric for c in root.children]
        bscore = min(mcs) if self.minimize else max(mcs)
        nc = [c for c in [root] + root.children if c.metric == bscore]
        node = random.choices(nc, weights=[c.visits for c in nc], k=1)[0] if len(nc) > 1 else nc[0]
        if node == root:
            return [root]

        return [root] + self.rollout(node)

    def backprop(self, traj):
        # Propagate upwards.
        rtraj = list(reversed(traj))
        for n in rtraj:
            n.visits += 1


class UCBPolicy(object):
    def __init__(self, minimize=True, c=math.sqrt(2)):
        self.minimize = minimize
        self.c = c

    def rollout(self, root):
        if len(root.children) == 0:
            # We've reached a leaf.
            root.shadow_visits += 1
            return [root]

        # Allow re-pulling the current node always.
        consider_nodes = [(root, root.shadow_metric, root.shadow_visits)]
        consider_nodes += [(c, c.metric, c.visits) for c in root.children]
        # This properly normalizes scores into [0, 1]
        qscores = [(n, root.metric / m if self.minimize else m / root.metric, v) for (n, m, v) in consider_nodes]

        scores = [
            (node, score + self.c * (math.log(root.visits) / (node_visits + 1)), node_visits)
            for (node, score, node_visits) in qscores
        ]

        # Take the argmax of scores.
        mscore = max(scores, key=lambda x: x[1])
        cands = [n for n in scores if n[1] == mscore[1]]
        nchild_cand = random.choices(cands, weights=[c[2] for c in cands], k=1)[0] if len(cands) > 1 else cands[0]
        nchild = nchild_cand[0]
        if nchild == root:
            # This is a "self-pull"
            # We will update the true pull in backprop.
            nchild.shadow_visits += 1
            return [nchild]

        # Recurse.
        return [root] + self.rollout(nchild)

    def backprop(self, traj):
        leaf_metric = traj[-1].metric

        # Propagate upwards.
        rtraj = list(reversed(traj))
        for n in rtraj:
            n.metric = min(n.metric, leaf_metric) if self.minimize else max(n.metric, leaf_metric)
            n.visits += 1


class TargetStateResetWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reset_policy,
        reset_purity,
        reward_utility = None,
        debug_log_path = None,
        local_shadow_interval: int = 2,
    ):
        gym.Wrapper.__init__(self, env)

        minimize = not reward_utility.maximize
        # Create the reset policy.
        self.reset_policy = {
            "MAXIMAL": MaximalPolicy(minimize),
            "UCB": UCBPolicy(minimize),
        }[reset_policy]
        self.reset_purity = ResetPurity[reset_purity]
        self.local_shadow_interval = local_shadow_interval

        self.active_traj = None
        self.root = None
        self.debug_log_path = debug_log_path
        self.debug_cnt = 0

        self.reward_utility = reward_utility
        self.best_metric = None
        self.real_best_metric = None
        self.pg_env = unwrap_to_base_env(env)

    def save_state(self):
        return {
            "tracked_states": self.tracked_states,
            "best_metric": self.best_metric,
            "real_best_metric": self.real_best_metric,
        }

    def load_state(self, d):
        self.tracked_states = d["tracked_states"]
        self.best_metric = d["best_metric"]
        self.real_best_metric = d["real_best_metric"]

    def _build_node(self, metric, obs, accum_metric):
        reset_lsc = None
        state = self.pg_env.env_spec.action_space.get_state(self.pg_env)

        if self.reset_purity != ResetPurity.SHADOW:
            # Save an estimation for what the reset-to LSC should look like.
            # LOCAL_SHADOW needs to save; we will skew the LSC based on shadow pulls.
            reset_lsc = self.pg_env.action_space.get_lsc_for_reset()

        return Node(metric, obs, accum_metric, state, reset_lsc)

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        assert self.active_traj is not None
        rollout_leaf = self.active_traj[-1]

        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        accum_metric = infos.get("accum_metric", None)
        assert self.best_metric is not None
        q_timeout = infos.get("q_timeout", False)
        violate = infos.get("violate", False)
        repo_dir = infos.get("repo_dir", None)

        if not violate:
            # Update the "global" what is better for reporting.
            metric = infos["metric"]
            if self.reward_utility.is_perf_better(metric, self.best_metric):
                self.best_metric = metric
                if not q_timeout:
                    self.real_best_metric = self.best_metric

            # Now consider the metric in terms of the "trajectory"
            if self.reward_utility.is_perf_better(metric, rollout_leaf.metric):
                node = self._build_node(
                    metric,
                    obs,
                    accum_metric,
                )

                logging.info("[maximal]: Found new maximal state from {} to {} with {} at repo ({})".format(
                    rollout_leaf.node_id,
                    node.node_id,
                    node.metric,
                    repo_dir if repo_dir else "UNKNOWN"
                ))

                # Add this as a child to the rollout trajectory. Backprop on reset will handle counts..
                rollout_leaf.children.append(node)
                self.active_traj.append(node)
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        if self.root is None:
            # First time.
            state, info = self.env.reset(**kwargs)
            self.best_metric = info["baseline_metric"]
            self.real_best_metric = self.best_metric
            self.root = self._build_node(
                self.best_metric,
                state.copy(),
                info.get("accum_metric", None),
            )
        else:
            debug_fh = None
            if self.debug_log_path is not None:
                debug_fh = open(f"{self.debug_log_path}/policy{self.debug_cnt}.txt", "w")
                print_to_fh(self.root, debug_fh)

            if self.active_traj is not None:
                self.reset_policy.backprop(self.active_traj)

                if debug_fh:
                    # Log the backpropagation.
                    debug_fh.write("\n\nPost-Backpropagation:\n")
                    print_to_fh(self.root, debug_fh)

            self.active_traj = self.reset_policy.rollout(self.root)
            rollout_leaf = self.active_traj[-1]
            # If the rollout has children, but we are not executing it,
            # Then this can be thought of as re-pulling an inner node.
            is_shadow_pull = len(rollout_leaf.children) > 0
            target_reset_lsc = rollout_leaf.reset_lsc
            if is_shadow_pull and self.reset_purity == ResetPurity.LOCAL_SHADOW:
                if (rollout_leaf.shadow_visits % self.local_shadow_interval) == 0:
                    k = (rollout_leaf.shadow_visits / self.local_shadow_interval)
                    shift_lsc = self.pg_env.get_lsc_shift_k(k)
                    rollout_leaf.rollout_lscs.add(tuple(shift_lsc.tolist()))

                # Now, just randomly draw.
                rlsc = random.choice(list(rollout_leaf.rollout_lscs))
                target_reset_lsc = np.array(rlsc, dtype=np.float32)

            if debug_fh:
                nids = "Traj: " + ",".join([str(n.node_id) for n in self.active_traj])
                debug_fh.write("\n\nSelection:\n")
                debug_fh.write(nids + "\n")
                debug_fh.write(f"Reset LSC: {rollout_leaf.reset_lsc}\n")
                debug_fh.close()

            kwargs = kwargs if kwargs else {}
            kwargs["options"] = kwargs["options"] if kwargs.get("options", {}) else {}
            kwargs["options"]["metric"] = rollout_leaf.metric
            kwargs["options"]["state"] = rollout_leaf.obs
            kwargs["options"]["accum_metric"] = rollout_leaf.accum_metric
            kwargs["options"]["config"] = rollout_leaf.state
            kwargs["options"]["reset_lsc"] = rollout_leaf.reset_lsc
            state, info = self.env.reset(**kwargs)
        self.debug_cnt += 1
        return state, info
