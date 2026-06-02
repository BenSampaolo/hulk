#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


VELOCITY_KEYS = (
    "qvel",
    "dof_vel",
    "root_vel",
    "root_ang_vel",
    "root_lin_vel_local",
    "root_ang_vel_local",
)

# LaFAN was retargeted at 100 fps instead of 30 fps, then frame-stretched by
# about 333 / 100. Position-like arrays were interpolated, but velocity arrays
# remained in the original, too-fast time scale. Empirically, current LaFAN2 has
# median |qvel| / |finite_diff(qpos) * fps| ~= 3.3, while CMU is ~= 1.0.
DEFAULT_SCALE = 100.0 / 333.0


def load_motion(path: Path) -> dict[str, Any]:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).item()
    return pickle.loads(path.read_bytes())


def save_motion(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        with path.open("wb") as file:
            np.save(file, data, allow_pickle=True)
    else:
        with path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def motion_files(input_dir: Path, *, include_qpos_pkl: bool) -> list[Path]:
    files = [path for path in input_dir.glob("**/*") if path.suffix in (".pkl", ".npy")]
    if not include_qpos_pkl:
        files = [path for path in files if not path.name.endswith(".qpos.pkl")]
    return sorted(files)


def scalar_fps(raw: dict[str, Any], default: float = 100.0) -> float:
    fps = raw.get("fps", default)
    array = np.asarray(fps)
    if array.shape == ():
        return float(array)
    return float(array.reshape(-1)[0])


def velocity_consistency_ratio(raw: dict[str, Any]) -> float | None:
    """Return median |stored velocity| / |finite-difference velocity|.

    A self-consistent file should be near 1.0. Current broken LaFAN2 files are
    around 3.3. Uses only DoF velocities to avoid root-frame convention issues.
    """
    fps = scalar_fps(raw)
    if "qpos" in raw and "qvel" in raw:
        pos = np.asarray(raw["qpos"], dtype=np.float64)[:, 7:]
        vel = np.asarray(raw["qvel"], dtype=np.float64)[:, 6:]
    elif "dof_pos" in raw and "dof_vel" in raw:
        pos = np.asarray(raw["dof_pos"], dtype=np.float64)
        vel = np.asarray(raw["dof_vel"], dtype=np.float64)
    else:
        return None

    if pos.ndim != 2 or vel.ndim != 2 or pos.shape[0] < 3 or vel.shape[0] != pos.shape[0]:
        return None

    channels = min(pos.shape[1], vel.shape[1])
    if channels <= 0:
        return None
    pos = pos[:, :channels]
    vel = vel[:, :channels]

    finite_diff_vel = np.gradient(pos, axis=0) * fps
    valid = np.abs(finite_diff_vel) > 1e-3
    if not np.any(valid):
        return None
    return float(np.median(np.abs(vel[valid]) / (np.abs(finite_diff_vel[valid]) + 1e-9)))


def scale_velocity_fields(raw: dict[str, Any], scale: float) -> tuple[dict[str, Any], list[str]]:
    fixed = dict(raw)
    scaled_keys: list[str] = []
    for key in VELOCITY_KEYS:
        value = fixed.get(key)
        if isinstance(value, np.ndarray):
            fixed[key] = value.astype(value.dtype, copy=True) * scale
            scaled_keys.append(key)
    return fixed, scaled_keys


def copy_non_motion_files(input_dir: Path, output_dir: Path, *, overwrite: bool) -> int:
    copied = 0
    for src in input_dir.glob("**/*"):
        if not src.is_file() or src.suffix in (".pkl", ".npy"):
            continue
        dst = output_dir / src.relative_to(input_dir)
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def repair_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    scale: float,
    overwrite: bool,
    dry_run: bool,
    include_qpos_pkl: bool,
    copy_extras: bool,
) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite and not dry_run:
        raise FileExistsError(f"Output directory is non-empty; pass --overwrite to write into {output_dir}")

    files = motion_files(input_dir, include_qpos_pkl=include_qpos_pkl)
    if not files:
        raise FileNotFoundError(f"No .pkl/.npy motion files found in {input_dir}")

    ratios_before: list[float] = []
    ratios_after: list[float] = []
    processed = 0
    copied_unchanged = 0
    skipped = 0

    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Scale:     {scale:.9f}")
    print(f"Dry run:   {dry_run}")
    print(f"Files:     {len(files)}")
    print()
    print("file,before_ratio,after_ratio,scaled_keys")

    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        try:
            raw = load_motion(src)
            before = velocity_consistency_ratio(raw)
            fixed, scaled_keys = scale_velocity_fields(raw, scale)
            after = velocity_consistency_ratio(fixed)

            if before is not None:
                ratios_before.append(before)
            if after is not None:
                ratios_after.append(after)

            if not dry_run:
                if scaled_keys:
                    save_motion(dst, fixed)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_unchanged += 1

            processed += 1
            before_text = "nan" if before is None else f"{before:.4f}"
            after_text = "nan" if after is None else f"{after:.4f}"
            print(f"{rel},{before_text},{after_text},{'+'.join(scaled_keys) if scaled_keys else 'unchanged'}")
        except Exception as exc:  # noqa: BLE001 - batch repair should report and continue.
            skipped += 1
            print(f"{rel},ERROR,ERROR,{exc}")

    extras_copied = 0
    if copy_extras and not dry_run:
        extras_copied = copy_non_motion_files(input_dir, output_dir, overwrite=overwrite)

    print()
    print("Summary")
    print(f"  processed:        {processed}")
    print(f"  skipped:          {skipped}")
    print(f"  copied unchanged: {copied_unchanged}")
    print(f"  copied extras:    {extras_copied}")
    if ratios_before:
        print(f"  median ratio before: {np.median(ratios_before):.4f}")
    if ratios_after:
        print(f"  median ratio after:  {np.median(ratios_after):.4f}")

    if ratios_after and not (0.8 <= float(np.median(ratios_after)) <= 1.25):
        print()
        print("WARNING: after-ratio is not close to 1.0. Check --scale and fps metadata.")

    return 1 if skipped else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale broken LaFAN2 velocity channels after frame-stretching qpos/position arrays."
    )
    parser.add_argument("--input", type=Path, default=Path("motions/lafan2"), help="Input motion directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("motions/lafan2_velocity_fixed"),
        help="Output directory for repaired files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Velocity multiplier. Default is 100/333 for the current LaFAN2 stretch.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only print diagnostics; do not write files.")
    parser.add_argument(
        "--include-qpos-pkl",
        action="store_true",
        help="Also process *.qpos.pkl files. They usually have no velocity fields and are copied unchanged.",
    )
    parser.add_argument(
        "--copy-extras",
        action="store_true",
        help="Copy non-.pkl/.npy files such as .DS_Store or notes. Usually unnecessary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return repair_directory(
        args.input,
        args.output,
        scale=args.scale,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        include_qpos_pkl=args.include_qpos_pkl,
        copy_extras=args.copy_extras,
    )


if __name__ == "__main__":
    raise SystemExit(main())
