import os

import numpy as np

from agents.wolp import Wolp
from agents.wolp.config import setup_wolp_agent

# Small monkey patch so gym 0.21 is compatible with numpy >= 1.24
# TODO: remove when upgrading to gym 0.26
np.bool = bool  # type: ignore[attr-defined]

def setup_agent(env, spec, seed, agent_type, agent_config):
    if agent_type == "wolp":
        return setup_wolp_agent(env, spec, seed, agent_config)
    else:
        assert False, f"Unsupported agent {agent_type}"


__all__ = [
    "Wolp",
]
