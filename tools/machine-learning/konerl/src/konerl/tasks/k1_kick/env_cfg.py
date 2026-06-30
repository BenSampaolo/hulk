from mjlab.envs import ManagerBasedRlEnvCfg

from .simulation import make_kick_env_cfg


def k1_kick_env_cfg(*, play: bool = False, control_arms: bool = False) -> ManagerBasedRlEnvCfg:
    return make_kick_env_cfg(play, control_arms=control_arms)
