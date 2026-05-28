# Symmetry port risk register

Higher-risk commits in the initial symmetry port:

## 04449c6c9 — Add opt-in K1 velocity symmetry specs

Superseded/extended by:

- `f247603c4` — adds K1 arm joints to `model/K1.xml` and extends the symmetry specs to support both leg-only and full-body actor/critic dimensions.

Risk level: **high**

Why this is risky:

- `K1_VELOCITY_ACTOR_SPEC`, `K1_VELOCITY_CRITIC_SPEC`, `K1_FULL_BODY_VELOCITY_ACTOR_SPEC`, and `K1_FULL_BODY_VELOCITY_CRITIC_SPEC` assume exact concatenated observation ordering.
- `K1_ACTION_SPEC` and `K1_FULL_BODY_ACTION_SPEC` assume exact K1 action/joint order.
- The critic spec covers many privileged/randomization observations, so any term ordering/shape drift can silently make the invariant critic mathematically wrong.
- Foot/contact vector blocks assume left xyz followed by right xyz, with left/right groups kept together.

Mitigations before enabling for training:

- For leg-only AMP, validate `actor == 52`, `critic == 148`, and `env.num_actions == 12`.
- For full-body AMP arms, validate `actor == 76`, `critic == 220`, and `env.num_actions == 20`.
- Validate the compiled MuJoCo model exposes the expected arm joints before enabling arm control.
- Run a live-env mirror smoke test before switching registered AMP/velocity tasks to `k1_equivariant_ppo_runner_cfg()`.

Current status:

- The equivariant config is opt-in only.
- Existing registered velocity and AMP tasks still use `k1_ppo_runner_cfg()`.
- `tests/test_symmetry.py` now checks the K1 actor/action/critic dimensions and catches left/right gain-block mirroring.

## Low-risk review notes resolved after EMLP comparison

These started as low-risk notes while symmetry was still opt-in. They have since been implemented and covered by `tests/test_symmetry.py`.

### `ReflectionInvariantify` assumes diagonal hidden representations

Risk level: **low**

Original concern: `ReflectionInvariantify` worked for `ReflectionSpec.hidden(...)`, where channels are already separated into sign-only even and odd coordinates, but it was not a general invariant map for swapped left/right representations.

Resolution:

- `ReflectionInvariantify` now builds invariant features per reflection orbit.
- Fixed even channels pass through.
- Fixed odd channels are mapped through `square`/`abs`.
- Swapped pairs produce an invariant pair: even combination plus transformed odd combination.
- `tests/test_symmetry.py` verifies the swapped-representation case.

### `ReflectionEquivariantLinear` projects parameters outside autograd

Risk level: **low**

Original concern: layers projected parameters in `forward()` under `torch.no_grad()`, so forward equivariance held but gradients were not projected through the equivariant weight map.

Resolution:

- `ReflectionEquivariantLinear.forward()` now computes projected effective `weight`/`bias` inside the autograd graph and does not mutate parameters.
- `project_parameters_()` remains available as explicit no-grad cleanup/export preparation.
- `tests/test_symmetry.py` verifies projected-gradient behavior.
