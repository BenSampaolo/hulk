import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math

    import matplotlib.pyplot as plt

    from pathlib import Path
    import pickle

    from copy import deepcopy
    from typing import Any

    from tqdm import tqdm
    return Any, Path, deepcopy, math, np, pickle, tqdm


@app.cell
def _(Path, e, np, pickle, tqdm):
    def load_data(data_dir: Path):
        data = {}
        files = [f for f in data_dir.glob("**/*") if f.suffix in (".pkl", ".npy") and not f.name.endswith(".qpos.pkl")]
        if not files:
            raise FileNotFoundError(f"No motion files (.pkl/.npy) in {data_dir}")
    
        print(f"# --- Loading {len(files)} motion files... --- #")
    
        for file in tqdm(files):
            try:
                if file.suffix == ".pkl":
                    raw_data = pickle.loads(file.read_bytes())
                else: # .npy
                    raw_data = np.load(file, allow_pickle=True).item()
                data[file.name] = raw_data
            except e:
                print(f"Skipping {file.name}: {e}")
        print(f"\n# --- Finished loading --- #\nFiles loaded sucessfully: {len(data.keys())}\nFiles skipped: {len(files) - len(data.keys())}\n")
        return data
    return (load_data,)


@app.cell
def _(Any, Path, np, pickle, tqdm):
    def save_data(data_dir: Path, data: dict[str, Any]) -> None:
        files: list[str] = list(data.keys())
    
        print(f"# --- Saving {len(files)} motion files... --- #")

        count = 0
    
        for file in tqdm(files):
            file_path = data_dir / file

            file_path.parent.mkdir(parents=True, exist_ok=True)
        
            try:
                if file.endswith(".pkl"):
                    with open(file_path, "wb") as f:
                        pickle.dump(data[file], f)
                else:
                    with open(file_path, "wb") as f:
                        np.save(f, data[file], allow_pickle=True)
                count += 1
            except Exception as e:
                print(f"Skipping {file}: {e}")

        print(f"\n# --- Finished saving --- #\nFiles saved successfully: {count}\nFiles skipped: {len(files) - count}\n")
    return (save_data,)


@app.cell
def _(Any, math, np):
    def change_fps_linear(
        data: dict[str, Any], 
        keys: list[str], 
        old_fps: float,
        new_fps: float,
    ) -> dict[str, Any]:
        for key in keys:
            new_data = []
            try:
                raw_data = data[key]
            except KeyError as e:
                print(f"Skipped key {key}: {e}")
                continue
            
            num_frames = len(raw_data)
            new_num_frames = math.floor((num_frames / old_fps) * new_fps)
        
            for i in range(new_num_frames):
                frame_idx = i * (old_fps / new_fps)
            
                if frame_idx.is_integer():
                    new_data.append(raw_data[int(frame_idx)])
                    continue
            
                frame_idx_rounded = int(math.floor(frame_idx))
                frame_idx_decimal = frame_idx - frame_idx_rounded
            
                next_idx = min(frame_idx_rounded + 1, num_frames - 1)
            
                interpolated_value = (
                    raw_data[frame_idx_rounded] * (1 - frame_idx_decimal) + 
                    raw_data[next_idx] * frame_idx_decimal
                )
                new_data.append(interpolated_value)

            data[key] = np.array(new_data)

        data["fps"] = np.array(new_fps)
        return data
    return (change_fps_linear,)


@app.cell
def _(Any, change_fps_linear, deepcopy, np, tqdm):
    def change_fps(
        data: dict[str, Any], 
        old_fps: float | None = None,
        new_fps: float = 100,
        interpolation: str = "linear",
    ) -> dict[str, Any]:
        interpolation_types = ["linear"]
        assert interpolation in interpolation_types, f"Interpolation type {interpolation} not in {interpolation_types}"
    
        data = deepcopy(data)
        files = list(data.keys())
        count = 0

        print(f"# --- Setting new speed for {len(files)} files --- #")

        for file in tqdm(files):
            current_fps = old_fps if old_fps is not None else data[file].get("fps")
        
            if current_fps is None:
                print(f"Skipping {file}: No fps provided")
                continue
    
            array_keys = [
                key for key, val in data[file].items() 
                if isinstance(val, np.ndarray)
            ]

            match interpolation:
                case "linear": 
                    data[file] = change_fps_linear(
                        data=data[file], 
                        keys=array_keys, 
                        old_fps=current_fps, 
                        new_fps=new_fps
                    )

            count += 1

        print(f"\n# --- Finished respeeding --- #\nFiles processed successfully: {count}\nFiles skipped: {len(files) - count}\n")
    
        return data
    return (change_fps,)


@app.cell
def _(Path):
    data_dir = Path("./motions/CMU_Certified")
    data_dir
    save_dir = Path("./motions/CMU_Certified_Speed")
    return data_dir, save_dir


@app.cell
def _(change_fps, data_dir, load_data, save_data, save_dir):
    data = load_data(data_dir)
    data = change_fps(data)
    save_data(data_dir=save_dir, data=data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
