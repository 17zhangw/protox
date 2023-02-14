import typing
from copy import deepcopy
from typing import Optional, Type, Union

from agents.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from agents.common.vec_env.vec_check_nan import VecCheckNan

# Avoid circular import
if typing.TYPE_CHECKING:
    from agents.common.type_aliases import GymEnv


def unwrap_vec_wrapper(env: Union["GymEnv", VecEnv], vec_wrapper_class: Type[VecEnvWrapper]) -> Optional[VecEnvWrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env:
    :param vec_wrapper_class:
    :return:
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


def is_vecenv_wrapped(env: Union["GymEnv", VecEnv], vec_wrapper_class: Type[VecEnvWrapper]) -> bool:
    """
    Check if an environment is already wrapped by a given ``VecEnvWrapper``.

    :param env:
    :param vec_wrapper_class:
    :return:
    """
    return unwrap_vec_wrapper(env, vec_wrapper_class) is not None


__all__ = [
    "VecEnv",
    "VecEnvWrapper",
    "VecCheckNan",
    "unwrap_vec_wrapper",
    "is_vecenv_wrapped",
]
