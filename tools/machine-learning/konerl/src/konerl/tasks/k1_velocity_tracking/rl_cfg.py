from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)
from dataclasses import dataclass, field

from konerl.symmetry.k1_specs import k1_velocity_actor_model_kwargs, k1_velocity_critic_model_kwargs

def k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            obs_normalization=True,
            hidden_dims=(1024, 512),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            obs_normalization=True,
            hidden_dims=(1024, 512),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "log",
            },
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.004,
            num_learning_epochs=4,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="k1_velocity_tracking",
        save_interval=100,
        num_steps_per_env=48,
        max_iterations=5_000,
        clip_actions=5.0
    )


def k1_equivariant_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Opt-in runner config using the initial equivariant actor/critic port.

    This is deliberately not used by the registered default tasks yet, so normal
    velocity and AMP training remain unchanged. Use this only for explicit
    symmetry experiments after validating the observation/action specs.
    """
    cfg = k1_ppo_runner_cfg()
    actor_kwargs = k1_velocity_actor_model_kwargs()
    cfg.actor.class_name = str(actor_kwargs["class_name"])
    cfg.actor.activation = str(actor_kwargs["activation"])

    critic_kwargs = k1_velocity_critic_model_kwargs()
    cfg.critic.class_name = str(critic_kwargs["class_name"])
    cfg.critic.activation = str(critic_kwargs["activation"])
    return cfg
