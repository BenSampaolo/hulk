#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from konerl.tasks.k1_velocity_tracking.randomization import reset_ball_relative_to_robot
from konerl.tasks.k1_velocity_tracking.simulation import make_velocity_env_cfg


DEFAULT_SOURCES = ("reset", "teleport_after_reset", "contact_reset_after_reset", "teleport_sequence")
FIELDNAMES = (
    "sample",
    "source",
    "env_id",
    "valid",
    "invalid_reason",
    "rel_x_b",
    "rel_y_b",
    "rel_z_b",
    "rel_x_w",
    "rel_y_w",
    "rel_z_w",
    "robot_x_w",
    "robot_y_w",
    "robot_z_w",
    "ball_x_w",
    "ball_y_w",
    "ball_z_w",
)


Row = dict[str, float | int | str | bool]


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def validate_row(row: Row, args: argparse.Namespace) -> tuple[bool, str]:
    reasons: list[str] = []
    rel_x_b = float(row["rel_x_b"])
    rel_y_b = float(row["rel_y_b"])
    ball_z_w = float(row["ball_z_w"])

    if not (args.min_rel_x <= rel_x_b <= args.max_rel_x):
        reasons.append(f"rel_x_b={rel_x_b:.4f} outside [{args.min_rel_x:.4f},{args.max_rel_x:.4f}]")
    if abs(rel_y_b) > args.max_abs_rel_y:
        reasons.append(f"abs(rel_y_b)={abs(rel_y_b):.4f} > {args.max_abs_rel_y:.4f}")
    if not (args.min_ball_z <= ball_z_w <= args.max_ball_z):
        reasons.append(f"ball_z_w={ball_z_w:.4f} outside [{args.min_ball_z:.4f},{args.max_ball_z:.4f}]")

    return not reasons, "; ".join(reasons)


def relative_ball_positions(env: ManagerBasedRlEnv, sample: int, source: str, args: argparse.Namespace) -> list[Row]:
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    robot_pos_w = robot.data.root_link_pos_w
    ball_pos_w = ball.data.root_link_pos_w
    rel_w = ball_pos_w - robot_pos_w
    yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rel_x_b = cos_yaw * rel_w[:, 0] + sin_yaw * rel_w[:, 1]
    rel_y_b = -sin_yaw * rel_w[:, 0] + cos_yaw * rel_w[:, 1]

    rows: list[Row] = []
    for env_id in range(env.num_envs):
        row: Row = {
            "sample": sample,
            "source": source,
            "env_id": env_id,
            "valid": True,
            "invalid_reason": "",
            "rel_x_b": float(rel_x_b[env_id].item()),
            "rel_y_b": float(rel_y_b[env_id].item()),
            "rel_z_b": float(rel_w[env_id, 2].item()),
            "rel_x_w": float(rel_w[env_id, 0].item()),
            "rel_y_w": float(rel_w[env_id, 1].item()),
            "rel_z_w": float(rel_w[env_id, 2].item()),
            "robot_x_w": float(robot_pos_w[env_id, 0].item()),
            "robot_y_w": float(robot_pos_w[env_id, 1].item()),
            "robot_z_w": float(robot_pos_w[env_id, 2].item()),
            "ball_x_w": float(ball_pos_w[env_id, 0].item()),
            "ball_y_w": float(ball_pos_w[env_id, 1].item()),
            "ball_z_w": float(ball_pos_w[env_id, 2].item()),
        }
        valid, reason = validate_row(row, args)
        row["valid"] = valid
        row["invalid_reason"] = reason
        rows.append(row)
    return rows


def all_env_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.arange(env.num_envs, dtype=torch.int64, device=env.device)


def apply_ball_reset(env: ManagerBasedRlEnv) -> None:
    params = dict(env.cfg.events["reset_ball"].params)
    reset_ball_relative_to_robot(env, all_env_ids(env), **params)
    env.scene.write_data_to_sim()
    env.sim.forward()


def apply_ball_offset_event(env: ManagerBasedRlEnv, event_name: str) -> None:
    params = dict(env.cfg.events[event_name].params)
    params.pop("sensor_name", None)
    params.pop("delay_frames", None)
    reset_ball_relative_to_robot(env, all_env_ids(env), **params)
    env.scene.write_data_to_sim()
    env.sim.forward()


def run_samples(env: ManagerBasedRlEnv, samples: int, sources: Iterable[str], args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    env_ids = all_env_ids(env)
    for sample in range(samples):
        for source in sources:
            if source == "reset":
                env.reset(env_ids=env_ids)
            elif source == "reset_ball":
                env.reset(env_ids=env_ids)
                apply_ball_reset(env)
            elif source == "teleport_after_reset":
                env.reset(env_ids=env_ids)
                apply_ball_offset_event(env, "teleport_ball")
            elif source == "teleport_sequence":
                if sample == 0:
                    env.reset(env_ids=env_ids)
                apply_ball_offset_event(env, "teleport_ball")
            elif source == "contact_reset_after_reset":
                env.reset(env_ids=env_ids)
                apply_ball_offset_event(env, "reset_ball_on_contact")
            else:
                raise ValueError(f"unknown source: {source}")
            rows.extend(relative_ball_positions(env, sample, source, args))
    return rows


def print_summary(rows: list[Row], limit: int) -> None:
    total = len(rows)
    invalid_rows = [row for row in rows if not bool(row["valid"])]
    print(f"sampled rows: {total}", file=sys.stderr)
    print(f"valid rows:   {total - len(invalid_rows)}", file=sys.stderr)
    print(f"invalid rows: {len(invalid_rows)}", file=sys.stderr)

    by_source: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)

    for source, source_rows in sorted(by_source.items()):
        invalid = sum(not bool(row["valid"]) for row in source_rows)
        rel_x = [float(row["rel_x_b"]) for row in source_rows]
        rel_y_abs = [abs(float(row["rel_y_b"])) for row in source_rows]
        ball_z = [float(row["ball_z_w"]) for row in source_rows]
        print(
            f"{source}: invalid={invalid}/{len(source_rows)} "
            f"rel_x_b=[{min(rel_x):.3f},{max(rel_x):.3f}] "
            f"abs(rel_y_b)=[{min(rel_y_abs):.3f},{max(rel_y_abs):.3f}] "
            f"ball_z_w=[{min(ball_z):.3f},{max(ball_z):.3f}]",
            file=sys.stderr,
        )

    if invalid_rows:
        reason_counts = Counter(str(row["invalid_reason"]) for row in invalid_rows)
        print("invalid reasons:", file=sys.stderr)
        for reason, count in reason_counts.most_common(10):
            print(f"  {count}: {reason}", file=sys.stderr)
        print(f"first {min(limit, len(invalid_rows))} invalid rows:", file=sys.stderr)
        for row in invalid_rows[:limit]:
            print(row, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and validate ball positions relative to the robot after reset/teleport randomizations."
    )
    parser.add_argument("--samples", type=int, default=200, help="Number of samples per source.")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel envs per sample.")
    parser.add_argument("--control-arms", action="store_true", help="Use the full-body K1 config.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP rewards in the constructed env config.")
    parser.add_argument(
        "--source",
        action="append",
        choices=("reset", "reset_ball", "teleport_after_reset", "teleport_sequence", "contact_reset_after_reset"),
        help=f"Source to sample. Can be passed multiple times. Defaults to {', '.join(DEFAULT_SOURCES)}.",
    )
    parser.add_argument("--min-rel-x", type=float, default=0.0, help="Minimum valid ball x in robot yaw frame.")
    parser.add_argument("--max-rel-x", type=float, default=2.0, help="Maximum valid ball x in robot yaw frame.")
    parser.add_argument("--max-abs-rel-y", type=float, default=1.0, help="Maximum valid absolute ball y in robot yaw frame.")
    parser.add_argument("--min-ball-z", type=float, default=0.15, help="Minimum valid world-frame ball z.")
    parser.add_argument("--max-ball-z", type=float, default=0.35, help="Maximum valid world-frame ball z.")
    parser.add_argument("--max-invalid", type=int, default=0, help="Allowed invalid rows before returning failure.")
    parser.add_argument("--summary-limit", type=int, default=10, help="Number of invalid rows to print in the summary.")
    parser.add_argument("--output", type=Path, help="CSV output path. Defaults to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = args.source or DEFAULT_SOURCES

    cfg = make_velocity_env_cfg(play=False, amp=args.amp, control_arms=args.control_arms)
    cfg.scene.num_envs = args.num_envs

    with contextlib.redirect_stdout(sys.stderr):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        rows = run_samples(env, args.samples, sources, args)

    if args.output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.output}", file=sys.stderr)

    print_summary(rows, args.summary_limit)
    invalid_count = sum(not bool(row["valid"]) for row in rows)
    return 1 if invalid_count > args.max_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
