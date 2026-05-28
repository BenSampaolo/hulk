import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from pathlib import Path
    import pickle
    from tqdm import tqdm
    return Path, np, pickle, tqdm


@app.cell
def _(np):
    sauergurke = np.load("./motions/unused/85/85_02_stageii.pkl", allow_pickle=True)
    qpos_sauergurke = np.load("./motions/unused/85/85_02_stageii.qpos.pkl", allow_pickle=True)
    return qpos_sauergurke, sauergurke


@app.cell
def _(qpos_sauergurke):
    qpos_sauergurke
    return


@app.cell
def _(sauergurke):
    sauergurke
    return


@app.cell
def _(np):
    def convert_to_qpos_format(data):
        # Concatenate root position, root rotation (quaternion), and joint positions
        qpos = np.concatenate([
            np.array(data["root_pos"]), 
            np.array(data["root_rot"]), 
            np.array(data["dof_pos"])
        ], axis=1)
    
        return {
            "fps": data["fps"],
            "qpos": qpos
        }
    return (convert_to_qpos_format,)


@app.cell
def _(convert_to_qpos_format, qpos_sauergurke, sauergurke):
    (qpos_sauergurke["qpos"] == convert_to_qpos_format(sauergurke)["qpos"]).all()
    return


@app.cell
def _(Path, convert_to_qpos_format, pickle, tqdm):
    def convert_nested_folders(base_directory):
        for pkl_path in tqdm(Path(base_directory).rglob("*.pkl")):
            # Skip files that are already converted
            if pkl_path.name.endswith(".qpos.pkl"):
                continue
            
            out_path = pkl_path.with_name(f"{pkl_path.stem}.qpos.pkl")
        
            # Skip if the target converted file already exists
            if out_path.exists():
                continue
            
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            with open(out_path, "wb") as f:
                pickle.dump(convert_to_qpos_format(data), f)
    return (convert_nested_folders,)


@app.cell
def _(convert_nested_folders):
    convert_nested_folders("/Users/ben/hulks/hulk-konerl/tools/machine-learning/konerl/motions/CMU_Certified_Speed/")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
