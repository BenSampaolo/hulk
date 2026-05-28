from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import k1_rough_env_cfg
from .rl_cfg import k1_equivariant_ppo_runner_cfg, k1_ppo_runner_cfg
from konerl.scripts.runner import MjlabAMPOnPolicyRunner
from konerl.scripts.runner import MjlabOnPolicyRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-K1",
    env_cfg=k1_rough_env_cfg(),
    play_env_cfg=k1_rough_env_cfg(play=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-K1-AMP",
    env_cfg=k1_rough_env_cfg(amp=True),
    play_env_cfg=k1_rough_env_cfg(play=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=MjlabAMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-K1-AMP-Arms",
    env_cfg=k1_rough_env_cfg(amp=True, control_arms=True),
    play_env_cfg=k1_rough_env_cfg(play=True, control_arms=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=MjlabAMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-K1-AMP-Arms-EQ",
    env_cfg=k1_rough_env_cfg(amp=True, control_arms=True),
    play_env_cfg=k1_rough_env_cfg(play=True, control_arms=True),
    rl_cfg=k1_equivariant_ppo_runner_cfg(),
    runner_cls=MjlabAMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-K1-AMP-Arms-EQ",
    env_cfg=k1_rough_env_cfg(amp=True, control_arms=True),
    play_env_cfg=k1_rough_env_cfg(play=True, control_arms=True),
    rl_cfg=k1_equivariant_ppo_runner_cfg(),
    runner_cls=MjlabAMPOnPolicyRunner,
)
