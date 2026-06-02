import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import math
    import pickle
    from dataclasses import dataclass
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from konerl.scripts.AMP.features import (
        K1_AMP_FULL_BODY_JOINT_NAMES,
        K1_AMP_LEG_JOINT_NAMES,
        MOCAP_TO_K1,
    )

    return (
        K1_AMP_FULL_BODY_JOINT_NAMES,
        K1_AMP_LEG_JOINT_NAMES,
        MOCAP_TO_K1,
        Path,
        dataclass,
        mo,
        np,
        pickle,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # LaFAN2 vs CMU_Certified_Speed AMP mocap comparison

    This notebook loads the exact AMP feature channels used by training
    (`joint_pos`, `joint_vel`, root linear velocity, root angular velocity),
    then compares clip lengths, speeds, joint distributions, and selected
    clips over time.

    Run with:

    ```bash
    uv run --with marimo marimo edit scripts/compare_lafan2_cmu_mocap.py
    ```
    """)
    return


@app.cell
def _(dataclass, np):
    # qpos/qvel exports use free-root qpos=[xyz, quat, dofs] and
    # qvel=[root_lin, root_ang, dof_vel]. The DoF order is not MOCAP_TO_K1
    # dict order; keep this explicit to match SimpleAMPBuilder.
    MOCAP_QPOS_DOF_NAMES = (
        "Head1",
        "Head2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    )

    @dataclass(frozen=True)
    class MotionClip:
        dataset: str
        path: object
        fps: float
        joint_names: tuple[str, ...]
        features: np.ndarray
        qpos: np.ndarray | None
        root_pos: np.ndarray | None
        root_up_dirs: np.ndarray | None
        left_foot_positions: np.ndarray | None
        right_foot_positions: np.ndarray | None

        @property
        def frames(self) -> int:
            return int(self.features.shape[0])

        @property
        def duration_s(self) -> float:
            return self.frames / self.fps if self.fps > 0 else float("nan")

        @property
        def time_s(self) -> np.ndarray:
            return np.arange(self.frames, dtype=np.float32) / self.fps

    return MOCAP_QPOS_DOF_NAMES, MotionClip


@app.cell
def _(MOCAP_QPOS_DOF_NAMES, MOCAP_TO_K1, MotionClip, Path, np, pickle):
    def _as_float(value, default: float = 100.0) -> float:
        if value is None:
            return default
        array = np.asarray(value)
        if array.shape == ():
            return float(array)
        return float(array.reshape(-1)[0])

    def _load_raw(path: Path) -> dict:
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=True).item()
        return pickle.loads(path.read_bytes())

    def _mocap_indices(raw: dict, joint_names: tuple[str, ...]) -> tuple[list[int], str]:
        k1_to_mocap = {v: k for k, v in MOCAP_TO_K1.items()}
        if "link_body_list" in raw:
            source_names = tuple(raw["link_body_list"])
            root_offset = 1
            schema = "link_body_list"
        elif "qpos" in raw and "qvel" in raw:
            source_names = tuple(
                raw.get("dof_names")
                or raw.get("joint_names")
                or raw.get("qpos_dof_names")
                or MOCAP_QPOS_DOF_NAMES
            )
            root_offset = 0
            schema = "qpos/qvel"
        else:
            raise ValueError("unsupported schema: expected link_body_list or qpos+qvel")

        indices: list[int] = []
        for joint_name in joint_names:
            mocap_name = k1_to_mocap.get(joint_name)
            if mocap_name not in source_names:
                raise KeyError(f"cannot map K1 joint {joint_name!r} to mocap source names")
            indices.append(source_names.index(mocap_name) - root_offset)
        return indices, schema

    def _features_from_raw(raw: dict, joint_names: tuple[str, ...]) -> np.ndarray:
        indices, schema = _mocap_indices(raw, joint_names)
        if schema == "link_body_list":
            dof_pos = np.asarray(raw["dof_pos"], dtype=np.float32)
            dof_vel = np.asarray(raw["dof_vel"], dtype=np.float32)
            root_vel = np.asarray(raw["root_vel"], dtype=np.float32)
            root_ang_vel = np.asarray(raw["root_ang_vel"], dtype=np.float32)
        else:
            qpos = np.asarray(raw["qpos"], dtype=np.float32)
            qvel = np.asarray(raw["qvel"], dtype=np.float32)
            dof_pos = qpos[:, 7:]
            dof_vel = qvel[:, 6:]
            root_vel = np.asarray(raw.get("root_lin_vel_local", qvel[:, :3]), dtype=np.float32)
            root_ang_vel = np.asarray(raw.get("root_ang_vel_local", qvel[:, 3:6]), dtype=np.float32)

        return np.concatenate(
            [dof_pos[:, indices], dof_vel[:, indices], root_vel, root_ang_vel],
            axis=-1,
        ).astype(np.float32)

    def load_clip(path: Path, dataset: str, joint_names: tuple[str, ...]) -> MotionClip:
        raw = _load_raw(path)
        features = _features_from_raw(raw, joint_names)
        return MotionClip(
            dataset=dataset,
            path=path,
            fps=_as_float(raw.get("fps")),
            joint_names=joint_names,
            features=features,
            qpos=np.asarray(raw["qpos"], dtype=np.float32) if "qpos" in raw else None,
            root_pos=np.asarray(raw["root_pos"], dtype=np.float32) if "root_pos" in raw else None,
            root_up_dirs=np.asarray(raw["root_up_dirs"], dtype=np.float32) if "root_up_dirs" in raw else None,
            left_foot_positions=np.asarray(raw["left_foot_positions"], dtype=np.float32)
            if "left_foot_positions" in raw
            else None,
            right_foot_positions=np.asarray(raw["right_foot_positions"], dtype=np.float32)
            if "right_foot_positions" in raw
            else None,
        )

    def discover_motion_files(data_dir: Path) -> list[Path]:
        # Training MocapBuffer intentionally ignores *.qpos.pkl files; do the
        # same by default so this compares training material, not auxiliary qpos
        # exports without qvel.
        files = [
            path
            for path in data_dir.glob("**/*")
            if path.suffix in (".pkl", ".npy") and not path.name.endswith(".qpos.pkl")
        ]
        return sorted(files)

    def load_dataset(
        data_dir: Path,
        dataset: str,
        joint_names: tuple[str, ...],
        max_clips: int,
    ) -> tuple[list[MotionClip], list[str]]:
        clips: list[MotionClip] = []
        errors: list[str] = []
        for path in discover_motion_files(data_dir)[:max_clips]:
            try:
                clips.append(load_clip(path, dataset, joint_names))
            except Exception as exc:  # noqa: BLE001 - notebook should report bad files and continue.
                errors.append(f"{path.name}: {exc}")
        return clips, errors

    return (load_dataset,)


@app.cell
def _(mo):
    lafan_dir = mo.ui.text(value="motions/lafan2_velocity_fixed/", label="LaFAN2 directory")
    cmu_dir = mo.ui.text(value="motions/CMU_Certified_Speed", label="CMU directory")
    joint_set = mo.ui.dropdown(options=["legs", "full_body"], value="legs", label="AMP joint set")
    max_clips = mo.ui.slider(start=1, stop=250, value=80, step=1, label="Max clips per dataset")

    mo.vstack([lafan_dir, cmu_dir, joint_set, max_clips])
    return cmu_dir, joint_set, lafan_dir, max_clips


@app.cell
def _(
    K1_AMP_FULL_BODY_JOINT_NAMES,
    K1_AMP_LEG_JOINT_NAMES,
    Path,
    cmu_dir,
    joint_set,
    lafan_dir,
    load_dataset,
    max_clips,
    mo,
):
    selected_joint_names = (
        K1_AMP_LEG_JOINT_NAMES if joint_set.value == "legs" else K1_AMP_FULL_BODY_JOINT_NAMES
    )
    lafan_clips, lafan_errors = load_dataset(
        Path(lafan_dir.value), "LaFAN2", selected_joint_names, int(max_clips.value)
    )
    cmu_clips, cmu_errors = load_dataset(
        Path(cmu_dir.value), "CMU_Certified_Speed", selected_joint_names, int(max_clips.value)
    )

    mo.md(
        f"""
        Loaded **{len(lafan_clips)}** LaFAN2 clips and **{len(cmu_clips)}** CMU clips
        with **{len(selected_joint_names)}** active joints.

        Bad/skipped files: LaFAN2={len(lafan_errors)}, CMU={len(cmu_errors)}.
        """
    )
    return (
        cmu_clips,
        cmu_errors,
        lafan_clips,
        lafan_errors,
        selected_joint_names,
    )


@app.cell
def _(cmu_errors, lafan_errors, mo):
    error_lines = [*lafan_errors[:20], *cmu_errors[:20]]
    error_output = (
        mo.accordion({"First skipped files/errors": mo.md("\n".join(f"- `{line}`" for line in error_lines))})
        if error_lines
        else mo.md("No skipped files.")
    )
    error_output
    return


@app.cell
def _(np):
    def split_features(features: np.ndarray, joint_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        joint_pos = features[:, :joint_count]
        joint_vel = features[:, joint_count : 2 * joint_count]
        root_vel = features[:, 2 * joint_count : 2 * joint_count + 3]
        root_ang_vel = features[:, 2 * joint_count + 3 : 2 * joint_count + 6]
        return joint_pos, joint_vel, root_vel, root_ang_vel

    def root_speed(clip, horizontal: bool = False) -> np.ndarray:
        _, _, root_vel, _ = split_features(clip.features, len(clip.joint_names))
        cols = slice(0, 2) if horizontal else slice(0, 3)
        return np.linalg.norm(root_vel[:, cols], axis=1)

    def root_ang_speed(clip) -> np.ndarray:
        _, _, _, root_ang_vel = split_features(clip.features, len(clip.joint_names))
        return np.linalg.norm(root_ang_vel, axis=1)

    def concat_or_empty(arrays: list[np.ndarray]) -> np.ndarray:
        arrays = [array for array in arrays if array.size > 0]
        if not arrays:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    return concat_or_empty, root_ang_speed, root_speed, split_features


@app.cell
def _(cmu_clips, lafan_clips, np, root_ang_speed, root_speed):
    def summary_rows(clips) -> list[dict[str, object]]:
        rows = []
        for clip in clips:
            speed = root_speed(clip, horizontal=True)
            angular = root_ang_speed(clip)
            rows.append(
                {
                    "dataset": clip.dataset,
                    "clip": clip.path.name,
                    "fps": round(clip.fps, 3),
                    "frames": clip.frames,
                    "duration_s": round(clip.duration_s, 2),
                    "mean_xy_speed": round(float(np.mean(speed)), 3),
                    "p95_xy_speed": round(float(np.percentile(speed, 95)), 3),
                    "mean_ang_speed": round(float(np.mean(angular)), 3),
                }
            )
        return rows

    clip_summary_rows = summary_rows(lafan_clips) + summary_rows(cmu_clips)
    return (clip_summary_rows,)


@app.cell
def _(clip_summary_rows, mo):
    mo.ui.table(clip_summary_rows, label="Per-clip summary")
    return


@app.cell
def _(cmu_clips, concat_or_empty, lafan_clips, np, root_ang_speed, root_speed):
    def dataset_stats(clips) -> dict[str, float | int]:
        durations = np.asarray([clip.duration_s for clip in clips], dtype=np.float32)
        frames = np.asarray([clip.frames for clip in clips], dtype=np.float32)
        xy_speed = concat_or_empty([root_speed(clip, horizontal=True) for clip in clips])
        ang_speed = concat_or_empty([root_ang_speed(clip) for clip in clips])
        features = concat_or_empty([clip.features for clip in clips])
        return {
            "clips": len(clips),
            "frames": int(frames.sum()) if frames.size else 0,
            "duration_min": round(float(durations.sum() / 60.0), 2) if durations.size else 0.0,
            "median_clip_s": round(float(np.median(durations)), 2) if durations.size else 0.0,
            "mean_xy_speed": round(float(np.mean(xy_speed)), 3) if xy_speed.size else 0.0,
            "p95_xy_speed": round(float(np.percentile(xy_speed, 95)), 3) if xy_speed.size else 0.0,
            "mean_ang_speed": round(float(np.mean(ang_speed)), 3) if ang_speed.size else 0.0,
            "feature_abs_mean": round(float(np.mean(np.abs(features))), 3) if features.size else 0.0,
            "feature_std_mean": round(float(np.mean(np.std(features, axis=0))), 3) if features.size else 0.0,
        }

    aggregate_rows = [
        {"dataset": "LaFAN2", **dataset_stats(lafan_clips)},
        {"dataset": "CMU_Certified_Speed", **dataset_stats(cmu_clips)},
    ]
    return (aggregate_rows,)


@app.cell
def _(aggregate_rows, mo):
    mo.ui.table(aggregate_rows, label="Dataset aggregate summary")
    return


@app.cell
def _(cmu_clips, lafan_clips, mo):
    lafan_labels = [f"{i:03d} | {clip.path.name} | {clip.duration_s:.1f}s" for i, clip in enumerate(lafan_clips)]
    cmu_labels = [f"{i:03d} | {clip.path.name} | {clip.duration_s:.1f}s" for i, clip in enumerate(cmu_clips)]
    lafan_select = mo.ui.dropdown(options=lafan_labels, value=lafan_labels[0] if lafan_labels else None, label="LaFAN2 clip")
    cmu_select = mo.ui.dropdown(options=cmu_labels, value=cmu_labels[0] if cmu_labels else None, label="CMU clip")

    mo.hstack([lafan_select, cmu_select])
    return cmu_labels, cmu_select, lafan_labels, lafan_select


@app.cell
def _(
    cmu_clips,
    cmu_labels,
    cmu_select,
    lafan_clips,
    lafan_labels,
    lafan_select,
):
    selected_lafan_clip = lafan_clips[lafan_labels.index(lafan_select.value)] if lafan_labels else None
    selected_cmu_clip = cmu_clips[cmu_labels.index(cmu_select.value)] if cmu_labels else None
    return selected_cmu_clip, selected_lafan_clip


@app.cell
def _(mo, selected_joint_names):
    joint_options = list(selected_joint_names)
    joint_choice = mo.ui.dropdown(options=joint_options, value=joint_options[0], label="Joint trace")
    smoothing_window = mo.ui.slider(start=1, stop=101, value=11, step=2, label="Speed smoothing window")
    trajectory_defaults = joint_options[: min(4, len(joint_options))]
    trajectory_joints = mo.ui.multiselect(
        options=joint_options,
        value=trajectory_defaults,
        label="LaFAN joint trajectory channels",
    )

    mo.vstack([mo.hstack([joint_choice, smoothing_window]), trajectory_joints])
    return joint_choice, smoothing_window, trajectory_joints


@app.cell
def _(np):
    def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
        window = max(1, int(window))
        if window <= 1 or values.size < window:
            return values
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(values, kernel, mode="same")

    def frame_window(clip, trim_start_s: float, trim_end_s: float) -> tuple[int, int]:
        if clip is None:
            return 0, 0
        start = int(np.clip(round(trim_start_s * clip.fps), 0, clip.frames))
        end_trim = int(np.clip(round(trim_end_s * clip.fps), 0, clip.frames - start))
        end = clip.frames - end_trim
        return start, max(start, end)

    def max_frame_jump(values: np.ndarray) -> tuple[float, int]:
        if values.shape[0] < 2:
            return 0.0, 0
        jumps = np.linalg.norm(np.diff(values, axis=0), axis=1)
        frame = int(np.argmax(jumps)) + 1
        return float(jumps[frame - 1]), frame

    def suggested_tpose_trim_frame(clip, threshold_rad: float, hold_frames: int, split_features) -> int:
        if clip is None or clip.frames < 2:
            return 0
        joint_pos, _, _, _ = split_features(clip.features, len(clip.joint_names))
        departure = np.linalg.norm(joint_pos - joint_pos[0:1], axis=1) / np.sqrt(joint_pos.shape[1])
        moving = departure > threshold_rad
        hold = max(1, int(hold_frames))
        if moving.size < hold:
            return 0
        for frame in range(0, moving.size - hold + 1):
            if bool(np.all(moving[frame : frame + hold])):
                return frame
        return 0

    return frame_window, max_frame_jump, smooth_1d, suggested_tpose_trim_frame


@app.cell
def _(mo, np, selected_lafan_clip):
    _duration = float(np.ceil(selected_lafan_clip.duration_s * 100.0) / 100.0) if selected_lafan_clip else 0.0
    lafan_trim_start_s = mo.ui.slider(
        start=0.0,
        stop=max(0.01, _duration),
        value=0.0,
        step=0.01,
        label="LaFAN trim start [s]",
    )
    lafan_trim_end_s = mo.ui.slider(
        start=0.0,
        stop=max(0.01, _duration),
        value=0.0,
        step=0.01,
        label="LaFAN trim end [s]",
    )
    jump_threshold_rad = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.20,
        step=0.01,
        label="Max allowed selected-joint jump [rad/frame]",
    )
    tpose_departure_threshold_rad = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.05,
        step=0.01,
        label="T-pose departure threshold [RMS rad]",
    )
    tpose_hold_frames = mo.ui.slider(
        start=1,
        stop=200,
        value=15,
        step=1,
        label="T-pose departure hold [frames]",
    )
    use_suggested_tpose_trim = mo.ui.checkbox(
        value=False,
        label="Use suggested T-pose trim start in LaFAN trajectory view",
    )

    mo.vstack(
        [
            mo.md("## LaFAN joint trajectory and trimming controls"),
            mo.hstack([lafan_trim_start_s, lafan_trim_end_s]),
            mo.hstack([jump_threshold_rad, tpose_departure_threshold_rad, tpose_hold_frames]),
            use_suggested_tpose_trim,
        ]
    )
    return (
        jump_threshold_rad,
        lafan_trim_end_s,
        lafan_trim_start_s,
        tpose_departure_threshold_rad,
        tpose_hold_frames,
        use_suggested_tpose_trim,
    )


@app.cell
def _(
    mo,
    selected_lafan_clip,
    split_features,
    suggested_tpose_trim_frame,
    tpose_departure_threshold_rad,
    tpose_hold_frames,
):
    suggested_lafan_tpose_trim_frame = suggested_tpose_trim_frame(
        selected_lafan_clip,
        float(tpose_departure_threshold_rad.value),
        int(tpose_hold_frames.value),
        split_features,
    )
    suggested_lafan_tpose_trim_s = (
        suggested_lafan_tpose_trim_frame / selected_lafan_clip.fps if selected_lafan_clip else 0.0
    )
    mo.md(
        f"Suggested T-pose trim start: **frame {suggested_lafan_tpose_trim_frame}** "
        f"(**{suggested_lafan_tpose_trim_s:.3f} s**)"
    )
    return suggested_lafan_tpose_trim_frame, suggested_lafan_tpose_trim_s


@app.cell
def _(
    frame_window,
    jump_threshold_rad,
    lafan_trim_end_s,
    lafan_trim_start_s,
    max_frame_jump,
    mo,
    np,
    plt,
    selected_lafan_clip,
    suggested_lafan_tpose_trim_s,
    trajectory_joints,
    use_suggested_tpose_trim,
    split_features,
):
    _selected_names = list(trajectory_joints.value)
    _effective_trim_start_s = (
        float(suggested_lafan_tpose_trim_s)
        if bool(use_suggested_tpose_trim.value)
        else float(lafan_trim_start_s.value)
    )
    _trim_start_frame, _trim_end_frame = frame_window(
        selected_lafan_clip,
        _effective_trim_start_s,
        float(lafan_trim_end_s.value),
    )
    _jump_threshold = float(jump_threshold_rad.value)

    _rows = []
    _fig_lafan_traj, _axes_lafan_traj = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    if selected_lafan_clip is None or not _selected_names or _trim_end_frame <= _trim_start_frame:
        _axes_lafan_traj[0].text(0.5, 0.5, "Select a LaFAN clip and at least one joint", ha="center", va="center")
    else:
        _joint_pos, _joint_vel, _root_vel, _root_ang_vel = split_features(
            selected_lafan_clip.features,
            len(selected_lafan_clip.joint_names),
        )
        _time = selected_lafan_clip.time_s
        _window = slice(_trim_start_frame, _trim_end_frame)
        _time_window = _time[_window]

        for _joint_name in _selected_names:
            _joint_idx = selected_lafan_clip.joint_names.index(_joint_name)
            _pos_full = _joint_pos[:, _joint_idx]
            _vel_full = _joint_vel[:, _joint_idx]
            _pos_window = _pos_full[_window]
            _vel_window = _vel_full[_window]
            _abs_pos_jump = np.abs(np.diff(_pos_window))
            _abs_vel_jump = np.abs(np.diff(_vel_window))

            _axes_lafan_traj[0].plot(_time_window, _pos_window, label=_joint_name)
            _axes_lafan_traj[1].plot(_time_window, _vel_window, label=_joint_name)
            if _abs_pos_jump.size:
                _jump_time = _time_window[1:]
                _axes_lafan_traj[2].plot(_jump_time, _abs_pos_jump, label=_joint_name)
                _max_pos_jump_idx = int(np.argmax(_abs_pos_jump)) + 1
                _max_vel_jump_idx = int(np.argmax(_abs_vel_jump)) + 1 if _abs_vel_jump.size else 0
                _max_pos_jump = float(_abs_pos_jump[_max_pos_jump_idx - 1])
                _max_vel_jump = float(_abs_vel_jump[_max_vel_jump_idx - 1]) if _abs_vel_jump.size else 0.0
            else:
                _max_pos_jump_idx = 0
                _max_vel_jump_idx = 0
                _max_pos_jump = 0.0
                _max_vel_jump = 0.0

            _rows.append(
                {
                    "joint": _joint_name,
                    "max_pos_jump_rad": round(_max_pos_jump, 5),
                    "max_pos_jump_frame": int(_trim_start_frame + _max_pos_jump_idx),
                    "max_pos_jump_s": round(float(_time[min(_trim_start_frame + _max_pos_jump_idx, len(_time) - 1)]), 3),
                    "exceeds_threshold": bool(_max_pos_jump > _jump_threshold),
                    "max_vel_jump_rad_s": round(_max_vel_jump, 5),
                    "max_vel_jump_frame": int(_trim_start_frame + _max_vel_jump_idx),
                }
            )

        _axes_lafan_traj[2].axhline(_jump_threshold, color="red", linestyle="--", linewidth=1.2, label="jump threshold")

        if selected_lafan_clip.qpos is not None:
            _root_xy = selected_lafan_clip.qpos[_window, :2]
            if _root_xy.shape[0] > 1:
                _axes_lafan_traj[3].plot(_root_xy[:, 0], _root_xy[:, 1], color="black", linewidth=1.4)
                _axes_lafan_traj[3].scatter(_root_xy[0, 0], _root_xy[0, 1], color="green", label="trimmed start")
                _axes_lafan_traj[3].scatter(_root_xy[-1, 0], _root_xy[-1, 1], color="red", label="trimmed end")
                _root_jump, _root_jump_frame_local = max_frame_jump(_root_xy)
                _rows.append(
                    {
                        "joint": "root_xy_path",
                        "max_pos_jump_rad": round(_root_jump, 5),
                        "max_pos_jump_frame": int(_trim_start_frame + _root_jump_frame_local),
                        "max_pos_jump_s": round(float(_time[min(_trim_start_frame + _root_jump_frame_local, len(_time) - 1)]), 3),
                        "exceeds_threshold": False,
                        "max_vel_jump_rad_s": "",
                        "max_vel_jump_frame": "",
                    }
                )
        else:
            _axes_lafan_traj[3].text(0.5, 0.5, "No qpos root XY trajectory available", ha="center", va="center")

    _axes_lafan_traj[0].set_title("LaFAN selected joint position trajectories after trim")
    _axes_lafan_traj[0].set_ylabel("rad")
    _axes_lafan_traj[1].set_title("LaFAN selected joint velocity trajectories after trim")
    _axes_lafan_traj[1].set_ylabel("rad/s")
    _axes_lafan_traj[2].set_title("LaFAN selected joint frame-to-frame position jumps after trim")
    _axes_lafan_traj[2].set_ylabel("abs Δrad/frame")
    _axes_lafan_traj[2].set_xlabel("time [s]")
    _axes_lafan_traj[3].set_title("LaFAN root XY trajectory after trim")
    _axes_lafan_traj[3].set_xlabel("x [m]")
    _axes_lafan_traj[3].set_ylabel("y [m]")
    _axes_lafan_traj[3].axis("equal")
    for _axis in _axes_lafan_traj:
        _axis.grid(True, alpha=0.25)
        _axis.legend(loc="best")

    mo.vstack(
        [
            mo.md(
                f"Trimmed LaFAN window: **frames {_trim_start_frame}:{_trim_end_frame}** "
                f"({_trim_end_frame - _trim_start_frame} frames)."
            ),
            mo.ui.table(_rows, label="Max frame-to-frame jumps in trimmed LaFAN window"),
            _fig_lafan_traj,
        ]
    )
    return


@app.cell
def _(cmu_clips, lafan_clips, np, plt, root_ang_speed, root_speed):
    fig_dataset, axes_dataset = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for _clips, _label, _color in (
        (lafan_clips, "LaFAN2", "tab:blue"),
        (cmu_clips, "CMU", "tab:orange"),
    ):
        _durations = [_clip.duration_s for _clip in _clips]
        _mean_speeds = [float(np.mean(root_speed(_clip, horizontal=True))) for _clip in _clips]
        _p95_speeds = [float(np.percentile(root_speed(_clip, horizontal=True), 95)) for _clip in _clips]
        _mean_ang = [float(np.mean(root_ang_speed(_clip))) for _clip in _clips]

        axes_dataset[0, 0].hist(_durations, bins=30, alpha=0.55, label=_label, color=_color)
        axes_dataset[0, 1].hist(_mean_speeds, bins=30, alpha=0.55, label=_label, color=_color)
        axes_dataset[1, 0].scatter(_durations, _mean_speeds, s=18, alpha=0.7, label=_label, color=_color)
        axes_dataset[1, 1].scatter(_p95_speeds, _mean_ang, s=18, alpha=0.7, label=_label, color=_color)

    axes_dataset[0, 0].set_title("Clip duration distribution")
    axes_dataset[0, 0].set_xlabel("duration [s]")
    axes_dataset[0, 0].set_ylabel("clips")
    axes_dataset[0, 1].set_title("Mean horizontal root speed per clip")
    axes_dataset[0, 1].set_xlabel("mean speed [m/s]")
    axes_dataset[0, 1].set_ylabel("clips")
    axes_dataset[1, 0].set_title("Duration vs mean horizontal speed")
    axes_dataset[1, 0].set_xlabel("duration [s]")
    axes_dataset[1, 0].set_ylabel("mean speed [m/s]")
    axes_dataset[1, 1].set_title("Fast clips vs turning/rotation")
    axes_dataset[1, 1].set_xlabel("p95 horizontal speed [m/s]")
    axes_dataset[1, 1].set_ylabel("mean angular speed [rad/s]")
    for _axis in axes_dataset.ravel():
        _axis.grid(True, alpha=0.25)
        _axis.legend()

    fig_dataset
    return


@app.cell
def _(
    joint_choice,
    plt,
    root_ang_speed,
    root_speed,
    selected_cmu_clip,
    selected_lafan_clip,
    smooth_1d,
    smoothing_window,
    split_features,
):
    fig_time, axes_time = plt.subplots(4, 1, figsize=(14, 10), sharex=False, constrained_layout=True)

    for _clip, _label, _color in (
        (selected_lafan_clip, "LaFAN2", "tab:blue"),
        (selected_cmu_clip, "CMU", "tab:orange"),
    ):
        if _clip is None:
            continue
        _joint_idx = _clip.joint_names.index(joint_choice.value)
        _joint_pos, _joint_vel, _root_vel, _root_ang_vel = split_features(_clip.features, len(_clip.joint_names))
        _t = _clip.time_s

        axes_time[0].plot(_t, smooth_1d(root_speed(_clip, horizontal=True), smoothing_window.value), label=_label, color=_color)
        axes_time[1].plot(_t, smooth_1d(root_ang_speed(_clip), smoothing_window.value), label=_label, color=_color)
        axes_time[2].plot(_t, _joint_pos[:, _joint_idx], label=_label, color=_color)
        axes_time[3].plot(_t, _joint_vel[:, _joint_idx], label=_label, color=_color)

    axes_time[0].set_title("Selected clip: horizontal root speed over time")
    axes_time[0].set_ylabel("m/s")
    axes_time[1].set_title("Selected clip: root angular speed over time")
    axes_time[1].set_ylabel("rad/s")
    axes_time[2].set_title(f"Selected clip: {joint_choice.value} position")
    axes_time[2].set_ylabel("rad")
    axes_time[3].set_title(f"Selected clip: {joint_choice.value} velocity")
    axes_time[3].set_ylabel("rad/s")
    axes_time[3].set_xlabel("time [s]")
    for _axis in axes_time:
        _axis.grid(True, alpha=0.25)
        _axis.legend()

    fig_time
    return


@app.cell
def _(
    cmu_clips,
    concat_or_empty,
    lafan_clips,
    np,
    plt,
    root_ang_speed,
    root_speed,
    split_features,
):
    def all_feature_parts(clips):
        if not clips:
            return tuple(np.empty((0, 0), dtype=np.float32) for _ in range(4))
        joint_count = len(clips[0].joint_names)
        positions = concat_or_empty([split_features(clip.features, joint_count)[0] for clip in clips])
        velocities = concat_or_empty([split_features(clip.features, joint_count)[1] for clip in clips])
        lin = concat_or_empty([split_features(clip.features, joint_count)[2] for clip in clips])
        ang = concat_or_empty([split_features(clip.features, joint_count)[3] for clip in clips])
        return positions, velocities, lin, ang

    lafan_pos, lafan_vel, lafan_lin, lafan_ang = all_feature_parts(lafan_clips)
    cmu_pos, cmu_vel, cmu_lin, cmu_ang = all_feature_parts(cmu_clips)

    fig_dist, axes_dist = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    plots = [
        (np.ravel(lafan_pos), np.ravel(cmu_pos), "joint positions", "rad"),
        (np.ravel(lafan_vel), np.ravel(cmu_vel), "joint velocities", "rad/s"),
        (np.ravel(lafan_lin), np.ravel(cmu_lin), "root linear velocity components", "m/s"),
        (np.ravel(lafan_ang), np.ravel(cmu_ang), "root angular velocity components", "rad/s"),
        (
            concat_or_empty([root_speed(clip, horizontal=True) for clip in lafan_clips]),
            concat_or_empty([root_speed(clip, horizontal=True) for clip in cmu_clips]),
            "horizontal root speed magnitude",
            "m/s",
        ),
        (
            concat_or_empty([root_ang_speed(clip) for clip in lafan_clips]),
            concat_or_empty([root_ang_speed(clip) for clip in cmu_clips]),
            "root angular speed magnitude",
            "rad/s",
        ),
    ]
    for _axis, (_lafan_values, _cmu_values, _title, _xlabel) in zip(axes_dist.ravel(), plots, strict=True):
        if _lafan_values.size:
            _axis.hist(_lafan_values, bins=80, density=True, alpha=0.5, label="LaFAN2", color="tab:blue")
        if _cmu_values.size:
            _axis.hist(_cmu_values, bins=80, density=True, alpha=0.5, label="CMU", color="tab:orange")
        _axis.set_title(_title)
        _axis.set_xlabel(_xlabel)
        _axis.set_ylabel("density")
        _axis.grid(True, alpha=0.25)
        _axis.legend()

    fig_dist
    return


@app.cell
def _(cmu_clips, lafan_clips, np, plt, selected_joint_names, split_features):
    def channel_stats(clips):
        if not clips:
            zeros = np.zeros(len(selected_joint_names), dtype=np.float32)
            return zeros, zeros, zeros, zeros
        joint_count = len(clips[0].joint_names)
        pos = np.concatenate([split_features(clip.features, joint_count)[0] for clip in clips], axis=0)
        vel = np.concatenate([split_features(clip.features, joint_count)[1] for clip in clips], axis=0)
        return np.mean(pos, axis=0), np.std(pos, axis=0), np.mean(vel, axis=0), np.std(vel, axis=0)

    lafan_pos_mean, lafan_pos_std, lafan_vel_mean, lafan_vel_std = channel_stats(lafan_clips)
    cmu_pos_mean, cmu_pos_std, cmu_vel_mean, cmu_vel_std = channel_stats(cmu_clips)

    x = np.arange(len(selected_joint_names))
    fig_joint, axes_joint = plt.subplots(2, 1, figsize=(max(12, 0.55 * len(selected_joint_names)), 8), constrained_layout=True)
    width = 0.38
    axes_joint[0].bar(x - width / 2, lafan_pos_std, width, label="LaFAN2", color="tab:blue", alpha=0.75)
    axes_joint[0].bar(x + width / 2, cmu_pos_std, width, label="CMU", color="tab:orange", alpha=0.75)
    axes_joint[0].set_title("Per-joint position standard deviation")
    axes_joint[0].set_ylabel("rad")
    axes_joint[1].bar(x - width / 2, lafan_vel_std, width, label="LaFAN2", color="tab:blue", alpha=0.75)
    axes_joint[1].bar(x + width / 2, cmu_vel_std, width, label="CMU", color="tab:orange", alpha=0.75)
    axes_joint[1].set_title("Per-joint velocity standard deviation")
    axes_joint[1].set_ylabel("rad/s")
    for _axis in axes_joint:
        _axis.set_xticks(x)
        _axis.set_xticklabels(selected_joint_names, rotation=45, ha="right")
        _axis.grid(True, axis="y", alpha=0.25)
        _axis.legend()

    fig_joint
    return


@app.cell
def _(np, plt, selected_cmu_clip, selected_lafan_clip):
    fig_extra, axes_extra = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    any_extra = False

    for _clip, _label, _color in (
        (selected_lafan_clip, "LaFAN2", "tab:blue"),
        (selected_cmu_clip, "CMU", "tab:orange"),
    ):
        if _clip is None:
            continue
        _t = _clip.time_s
        if _clip.qpos is not None:
            axes_extra[0].plot(_t, _clip.qpos[:, 2], label=f"{_label} qpos root z", color=_color)
            any_extra = True
        elif _clip.root_pos is not None:
            axes_extra[0].plot(_t, _clip.root_pos[:, 2], label=f"{_label} root z", color=_color)
            any_extra = True

        if _clip.left_foot_positions is not None and _clip.right_foot_positions is not None:
            axes_extra[1].plot(_t, _clip.left_foot_positions[:, 2], label=f"{_label} left foot z", color=_color, alpha=0.7)
            axes_extra[1].plot(_t, _clip.right_foot_positions[:, 2], label=f"{_label} right foot z", color=_color, linestyle="--", alpha=0.7)
            any_extra = True
        elif _clip.root_up_dirs is not None:
            _up_norm = np.linalg.norm(_clip.root_up_dirs, axis=1).clip(min=1e-6)
            _up_z = _clip.root_up_dirs[:, 2] / _up_norm
            axes_extra[1].plot(_t, _up_z, label=f"{_label} root up dot world z", color=_color)
            any_extra = True

    axes_extra[0].set_title("Selected clip: root height if available")
    axes_extra[0].set_ylabel("m")
    axes_extra[1].set_title("Selected clip: foot heights or root uprightness if available")
    axes_extra[1].set_xlabel("time [s]")
    axes_extra[1].set_ylabel("m or unitless")
    for _axis in axes_extra:
        _axis.grid(True, alpha=0.25)
        _axis.legend()
    if not any_extra:
        axes_extra[0].text(0.5, 0.5, "No qpos/root/foot/up extra channels available", ha="center", va="center")

    fig_extra
    return


@app.cell
def _(mo):
    mo.md("""
    ## What to look for

    - **Duration imbalance**: frame-weighted AMP sampling makes long clips dominate.
    - **Speed mismatch**: CMU_Certified_Speed should usually have higher root-speed mass than walking-heavy LaFAN2.
    - **Joint std mismatch**: large per-joint std differences mean the discriminator sees dataset identity easily.
    - **Root angular speed spikes**: often reveal turns, falls, or noisy conversions.
    - **Root/foot traces**: useful for spotting crouches, falls, foot-skate, and height scale mismatches.
    """)
    return


if __name__ == "__main__":
    app.run()
