from mjlab.tasks.registry import register_mjlab_task

from konerl.scripts.runner import MjlabOnPolicyRunner

from .env_cfg import k1_kick_env_cfg
from .rl_cfg import k1_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Kick-K1",
    env_cfg=k1_kick_env_cfg(),
    play_env_cfg=k1_kick_env_cfg(play=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
