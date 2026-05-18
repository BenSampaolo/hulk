from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import k1_rough_env_cfg
from .rl_cfg import k1_ppo_runner_cfg
from konerl.scripts.runner import MjlabAMPOnPolicyRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-K1",
    env_cfg=k1_rough_env_cfg(),
    play_env_cfg=k1_rough_env_cfg(play=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=MjlabAMPOnPolicyRunner,
)
