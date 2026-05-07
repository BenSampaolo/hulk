from copy import deepcopy
from collections import deque
from typing import TypeAlias, Literal
from weakref import WeakKeyDictionary

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity import mdp

class EpisodeLengthMeanTracker:
    """Track a pure rolling average episode length from completed episodes.

    This term is evaluated when environments are reset. At that point,
    `env.episode_length_buf[env_ids]` still contains the just-completed episode
    lengths, so we can aggregate a rolling window estimate.

    The tracker updates at most once per environment step (`common_step_counter`).
    Multiple calls during the same step return the cached value.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size: int = window_size
        self._window: deque[float] = deque()
        self._last_step: int = -1
        self._cached_mean: float = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        step = int(env.common_step_counter)
        if step != self._last_step:
            lengths = env.episode_length_buf[env_ids].float()
            completed = lengths[lengths > 0]

            if completed.numel() > 0:
                # Add one sample per completed episode to avoid reset-batch-size bias.
                self._window.extend(float(v) for v in completed.tolist())
                while len(self._window) > max(1, int(self.window_size)):
                    self._window.popleft()

            self._cached_mean = sum(self._window) / len(self._window) if self._window else 0.0
            self._last_step = step

        return torch.tensor(self._cached_mean)


_MEAN_EPISODE_LENGTH_TRACKERS: "WeakKeyDictionary[ManagerBasedRlEnv, dict[int, EpisodeLengthMeanTracker]]" = (
    WeakKeyDictionary()
)


def mean_ep_len(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    window_size: int = 100,
) -> torch.Tensor:
    """Get rolling mean episode length, updating at most once per step.

    Use this in multiple curriculum functions to share one tracker state and
    avoid duplicate updates within the same training iteration.
    """
    per_env = _MEAN_EPISODE_LENGTH_TRACKERS.setdefault(env, {})
    size = max(1, int(window_size))
    tracker = per_env.get(size)
    if tracker is None:
        tracker = EpisodeLengthMeanTracker(window_size=size)
        per_env[size] = tracker
    return tracker(env, env_ids)


class MeanLenThresholdCounter:
    """Count how often mean episode length exceeds a fixed threshold.

    The threshold is provided at initialization. The counter is updated at most
    once per environment step and can be reused across multiple curriculum
    functions in the same step.
    """

    def __init__(self, threshold: float, window_size: int = 100) -> None:
        self.threshold: float = float(threshold)
        self.window_size: int = max(1, int(window_size))
        self._last_step: int = -1
        self._count: int = 0

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
        step = int(env.common_step_counter)
        if step != self._last_step:
            mean_episode_length = float(
                mean_ep_len(env, env_ids, window_size=self.window_size).item()
            )
            if mean_episode_length > self.threshold:
                self._count += 1
            self._last_step = step
        return torch.tensor(float(self._count))


_MEAN_LEN_THRESHOLD_COUNTERS: "WeakKeyDictionary[ManagerBasedRlEnv, dict[tuple[int, float], MeanLenThresholdCounter]]" = (
    WeakKeyDictionary()
)


def mean_ep_len_hit_count(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    threshold: float,
    window_size: int = 100,
) -> torch.Tensor:
    """Get count of steps where rolling mean episode length exceeded threshold."""
    per_env = _MEAN_LEN_THRESHOLD_COUNTERS.setdefault(env, {})
    size = max(1, int(window_size))
    key = (size, float(threshold))
    counter = per_env.get(key)
    if counter is None:
        counter = MeanLenThresholdCounter(
            threshold=float(threshold), window_size=size
        )
        per_env[key] = counter
    return counter(env, env_ids)


def alpha_from_hits(
    hit_count: int | float | torch.Tensor,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
) -> torch.Tensor:
    """Map threshold hit count to alpha in [alpha_start, alpha_end].

    Each hit moves alpha by `alpha_step` toward `alpha_end`, starting from
    `alpha_start`. The result is clamped to the inclusive range bounded by
    (`alpha_start`, `alpha_end`).
    """
    if alpha_step <= 0:
        raise ValueError("alpha_step must be > 0")

    hits = int(float(hit_count.item() if isinstance(hit_count, torch.Tensor) else hit_count))
    hits = max(0, hits)

    start = float(alpha_start)
    end = float(alpha_end)
    direction = 1.0 if end >= start else -1.0
    alpha = start + direction * float(alpha_step) * hits
    low, high = (start, end) if start <= end else (end, start)
    alpha = max(low, min(high, alpha))
    return torch.tensor(alpha)


def alpha_from_mean_len_hits(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    threshold: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    window_size: int = 100,
) -> torch.Tensor:
    """Convenience wrapper: derive alpha from mean-length threshold hits."""
    hit_count = mean_ep_len_hit_count(
        env=env,
        env_ids=env_ids,
        threshold=threshold,
        window_size=window_size,
    )
    return alpha_from_hits(
        hit_count=hit_count,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
    )


InterpValue: TypeAlias = float | tuple[float, float] | dict[str, "InterpValue"]


def lerp_value(
    alpha: float | torch.Tensor,
    value_start: InterpValue,
    value_end: InterpValue,
    clamp_alpha: bool = True,
) -> InterpValue:
    """Linearly interpolate between scalar/tuple/dict values using alpha.

    This is useful for both reward weights (scalar) and randomization ranges
    (2-tuples), and entire parameter dictionaries.
    """
    a = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if clamp_alpha:
        a = max(0.0, min(1.0, a))

    if isinstance(value_start, dict) and isinstance(value_end, dict):
        keys_start = set(value_start.keys())
        keys_end = set(value_end.keys())
        if keys_start != keys_end:
            missing_in_end = keys_start - keys_end
            missing_in_start = keys_end - keys_start
            raise KeyError(
                "value_start and value_end dict keys must match. "
                f"Missing in end: {sorted(missing_in_end)}; missing in start: {sorted(missing_in_start)}"
            )

        return {
            key: lerp_value(a, value_start[key], value_end[key], clamp_alpha=False)
            for key in value_start
        }

    if isinstance(value_start, tuple) and isinstance(value_end, tuple):
        if len(value_start) != 2 or len(value_end) != 2:
            raise TypeError("Only 2-tuples are supported for tuple interpolation")
        return (
            float(value_start[0] + a * (value_end[0] - value_start[0])),
            float(value_start[1] + a * (value_end[1] - value_start[1])),
        )

    if isinstance(value_start, (int, float)) and isinstance(value_end, (int, float)):
        return float(value_start + a * (value_end - value_start))

    raise TypeError("value_start and value_end must have the same type")


def interp_from_mean_len_hits(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    threshold: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    value_start: InterpValue,
    value_end: InterpValue,
    window_size: int = 100,
    clamp_alpha: bool = True,
) -> InterpValue:
    """Compute alpha from mean-length hits, then interpolate between values."""
    alpha = alpha_from_mean_len_hits(
        env=env,
        env_ids=env_ids,
        threshold=threshold,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
        window_size=window_size,
    )
    return lerp_value(
        alpha=alpha,
        value_start=value_start,
        value_end=value_end,
        clamp_alpha=clamp_alpha,
    )


def _zero_like(value: InterpValue) -> InterpValue:
    if isinstance(value, dict):
        return {k: _zero_like(v) for k, v in value.items()}
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError("Only 2-tuples are supported")
        return (0.0, 0.0)
    if isinstance(value, (int, float)):
        return 0.0
    raise TypeError(f"Unsupported interpolation value type: {type(value)}")


_EVENT_PARAM_TARGETS: "WeakKeyDictionary[ManagerBasedRlEnv, dict[tuple[str, tuple[str, ...]], dict[str, InterpValue]]]" = (
    WeakKeyDictionary()
)


def ramp_event_params_from_mean_len_hits(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    param_keys: tuple[str, ...],
    threshold: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    start_values: dict[str, InterpValue],
    window_size: int = 100,
) -> torch.Tensor:
    """Ramp selected event params from configured start values to targets using alpha."""
    per_env_targets = _EVENT_PARAM_TARGETS.setdefault(env, {})
    cache_key = (event_name, tuple(param_keys))
    targets = per_env_targets.get(cache_key)
    if targets is None:
        term_cfg = env.event_manager.get_term_cfg(event_name)
        targets = {k: deepcopy(term_cfg.params[k]) for k in param_keys}
        per_env_targets[cache_key] = targets

    alpha = alpha_from_mean_len_hits(
        env=env,
        env_ids=env_ids,
        threshold=threshold,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
        window_size=window_size,
    )

    term_cfg = env.event_manager.get_term_cfg(event_name)
    for key in param_keys:
        term_cfg.params[key] = lerp_value(
            alpha=alpha,
            value_start=start_values[key],
            value_end=targets[key],
        )

    return alpha

def make_curriculum_cfg(terrain_type: Literal["flat", "rough", "bumpy"]) -> dict[str, CurriculumTermCfg]:
    # Shared settings
    thresh = 2950.0
    w_size = 1000
    step = 0.02 / 48  # 50 hits to complete a stage (1.0 / 50)

    curriculum = {
        # "push_robot": CurriculumTermCfg(
        #     func=ramp_event_params_from_mean_len_hits,
        #     params={
        #         "event_name": "push_robot", "param_keys": ("force_range", "torque_range"),
        #         "threshold": thresh, "window_size": w_size,
        #         "alpha_start": 0, "alpha_end": 1.0, "alpha_step": step,
        #         "start_values": {"force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)},
        #     },
        # ),
        # "impulse": CurriculumTermCfg(
        #     func=ramp_event_params_from_mean_len_hits,
        #     params={
        #         "event_name": "impulse", "param_keys": ("force_range", "torque_range"),
        #         "threshold": thresh, "window_size": w_size,
        #         "alpha_start": 0, "alpha_end": 1.0, "alpha_step": step,
        #         "start_values": {"force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)},
        #     },
        # ),
    }
    if terrain_type == "rough":
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        )

    return curriculum