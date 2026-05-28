# W&B log fetching instructions

This project logs training runs to Weights & Biases using the `mjlab` project.
Future agents should use these steps to fetch the latest training data and checkpoints.

## Defaults

- Entity: use the logged-in W&B default entity from `wandb.Api().default_entity`.
- Project: `mjlab`
- Local training logs: `logs/rsl_rl/k1_velocity_tracking/<run-name>/`
- Local fetched W&B metadata: `logs/wandb_runs/<run-name>_<run-id>/`

Check login first:

```sh
WANDB_SILENT=true uv run python - <<'PY'
import wandb
api = wandb.Api(timeout=30)
print(api.default_entity)
PY
```

## List recent runs

```sh
WANDB_SILENT=true uv run python - <<'PY'
import wandb
api = wandb.Api(timeout=30)
path = f"{api.default_entity}/mjlab"
for i, run in zip(range(15), api.runs(path, order="-created_at")):
    print(i, run.created_at, run.id, repr(run.name), run.state, run.url)
PY
```

## Fetch latest run metadata, config, summary, history, and logs

This creates:

- `run_meta.json`
- `config.json`
- `summary.json`
- `history.jsonl`
- `history.csv`
- downloaded W&B files like `config.yaml`, `wandb-summary.json`, `requirements.txt`, `wandb-metadata.json`, `output.log`

```sh
mkdir -p logs/wandb_runs
WANDB_SILENT=true uv run python - <<'PY'
from pathlib import Path
import csv
import json
import wandb

api = wandb.Api(timeout=60)
run = next(iter(api.runs(f"{api.default_entity}/mjlab", order="-created_at")))
out = Path("logs/wandb_runs") / f"{run.name}_{run.id}"
out.mkdir(parents=True, exist_ok=True)

meta = {
    "entity": run.entity,
    "project": run.project,
    "id": run.id,
    "name": run.name,
    "state": run.state,
    "created_at": run.created_at,
    "url": run.url,
    "path": run.path,
}
(out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
(out / "config.json").write_text(json.dumps(run.config, indent=2, default=str))
(out / "summary.json").write_text(json.dumps(dict(run.summary), indent=2, default=str))

rows = list(run.scan_history(page_size=1000))
(out / "history.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
keys = sorted({k for r in rows for k in r.keys()})
with (out / "history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

for name in ["config.yaml", "wandb-summary.json", "requirements.txt", "wandb-metadata.json", "output.log"]:
    try:
        run.file(name).download(root=str(out), replace=True)
        print("downloaded", name)
    except Exception as exc:
        print("skip", name, type(exc).__name__, exc)

print("saved", out)
print("history rows", len(rows), "keys", len(keys))
print("run url", run.url)
PY
```

## Parse `output.log` into complete per-iteration metrics

`run.scan_history()` may sample or duplicate rows. For this project, the W&B `output.log` contains one block per training iteration and is the most reliable source for every printed scalar.

Run this after fetching `output.log`:

```sh
uv run python - <<'PY'
from pathlib import Path
import csv
import json
import re

# Change this to the fetched run directory.
out = max(Path("logs/wandb_runs").glob("*"), key=lambda p: p.stat().st_mtime)
text = (out / "output.log").read_text(errors="replace")
text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # strip ANSI colors
blocks = re.split(r"Learning iteration ", text)[1:]
rows = []

for block in blocks:
    match = re.match(r"(\d+)/(\d+)", block)
    if not match:
        continue
    row = {"iteration": int(match.group(1)), "max_iterations": int(match.group(2))}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, val = line.strip().rsplit(":", 1)
        key = key.strip()
        val = val.strip()
        key = {
            "Total steps": "Train/total_steps",
            "Steps per second": "Perf/total_fps",
            "Collection time": "Perf/collection_time",
            "Learning time": "Perf/learning_time",
            "Mean value loss": "Loss/value",
            "Mean surrogate loss": "Loss/surrogate",
            "Mean entropy loss": "Loss/entropy",
            "Mean reward": "Train/mean_reward",
            "Mean episode length": "Train/mean_episode_length",
            "Mean action std": "Policy/mean_std",
            "Iteration time": "Perf/iteration_time",
            "Time elapsed": "Perf/time_elapsed",
            "ETA": "Perf/eta",
        }.get(key, key)
        if key.startswith("Mean amp/"):
            key = "Loss/amp/" + key[len("Mean amp/"):].replace(" loss", "")
        try:
            val = float(val[:-1]) if val.endswith("s") else float(val)
        except ValueError:
            pass
        row[key] = val
    rows.append(row)

csv_path = out / "output_metrics.csv"
keys = sorted({k for r in rows for k in r})
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
(csv_path.with_suffix(".csv.json")).write_text(json.dumps(rows, indent=2))
print("parsed", len(rows), "iterations")
print("wrote", csv_path)
PY
```

## Download checkpoints from a run

Use the run id from the listing or `run_meta.json`.

```sh
WANDB_SILENT=true uv run python - <<'PY'
import wandb
from pathlib import Path

RUN_ID = "83mthf63"  # change this
CHECKPOINTS = ["model_4300.pt", "model_4400.pt", "model_4999.pt"]  # change as needed

api = wandb.Api(timeout=60)
run = api.run(f"{api.default_entity}/mjlab/{RUN_ID}")
out = Path("logs/rsl_rl/k1_velocity_tracking") / run.name
out.mkdir(parents=True, exist_ok=True)

for name in CHECKPOINTS:
    path = out / name
    if path.exists():
        print("exists", path)
        continue
    run.file(name).download(root=str(out), replace=True)
    print("downloaded", path)
PY
```

To find available checkpoint files:

```sh
WANDB_SILENT=true uv run python - <<'PY'
import re
import wandb
api = wandb.Api(timeout=30)
run = next(iter(api.runs(f"{api.default_entity}/mjlab", order="-created_at")))
files = [f.name for f in run.files() if re.match(r"^model_\d+\.pt$", f.name)]
files = sorted(files, key=lambda n: int(n.split("_")[1].split(".")[0]))
print("\n".join(files))
PY
```

## Current known latest fetched run

As of 2026-05-28, the latest fetched run was:

- Run path: `bensampaolo-hamburg-university-of-technology/mjlab/83mthf63`
- Name: `2026-05-28_01-21-04`
- URL: <https://wandb.ai/bensampaolo-hamburg-university-of-technology/mjlab/runs/83mthf63>
- Local fetched metadata: `logs/wandb_runs/2026-05-28_01-21-04_83mthf63/`
- Local checkpoints: `logs/rsl_rl/k1_velocity_tracking/2026-05-28_01-21-04/`

Useful checkpoints from that run:

- `model_4300.pt`: strong reward / low bad-base-height candidate
- `model_4400.pt`: longest episode-length candidate
- `model_4999.pt`: final checkpoint
