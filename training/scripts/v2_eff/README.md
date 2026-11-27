# v2 GPU cost measurements

Short timing jobs that measure what a training step costs and where its time goes. Their results are in [`training/docs/v2-experiment-results.md`](../../docs/v2-experiment-results.md) ("GPU cost"), and the plan they feed is [`training/docs/v2-experiment-plan.md`](../../docs/v2-experiment-plan.md). No training code is changed for them.

| File | What it does |
|---|---|
| `step_probe.py` | Runs `training/scripts/train.py` in its own process and times what `TABSIL_RANK_TIMING=1` leaves out: the optimizer step, the gradient clipping, the per-group norms, each GPU's data wait and every pause of Python's garbage collector. The training loop's own collection and freeze at each log step (`TrainConfig.gc_freeze`) are timed apart from those pauses, with the number of frozen objects after each freeze. `TABSIL_EFF_GC=freeze:<step>` turns the loop's freeze off and makes the probe call `gc.freeze()` once after that many steps: the measurement that led to the loop's own freeze |
| `timing.yaml` | The AI Runtime job: `train.py` under `step_probe.py`, stage 1 at v1's size for 300 steps on 8 H100 by default. Size, micro batch, stage and GPU count are overrides (listed in the file) |
| `step_breakdown.py` | Reads one run from MLflow (`sec_per_step`, the rank timing records and the probe's records) and prints the step time by part, the GPU-seconds per table and the H100-h per 1K steps |

A run made with `training/scripts/train.yaml` and `TABSIL_RANK_TIMING=1` has no probe records. `step_breakdown.py` then reports forward, backward, the waiting for the slowest GPU and the rest as one number.

```bash
# stage 1, v1's size, 8 H100, the training loop as it is
databricks air run -p e2-demo-field-eng -f training/scripts/v2_eff/timing.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override mlflow_run_name=tabsil-v2-eff-s1-m-mb4-probe

# the same without the loop's freeze, and one freeze by the probe after step 200
databricks air run -p e2-demo-field-eng -f training/scripts/v2_eff/timing.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override mlflow_run_name=tabsil-v2-eff-s1-m-mb4-probe \
  --override env_variables.TABSIL_EFF_GC=freeze:200

# the breakdown of that run
DATABRICKS_CONFIG_PROFILE=e2-demo-field-eng MLFLOW_TRACKING_URI=databricks \
  uv run --package tabsil-train python training/scripts/v2_eff/step_breakdown.py <run id> --steps 100:300
```

Run them on AI Runtime, not locally. The first 100 steps of a stage-1 run hold the compilation (the first 20 of a stage-2 run at batch 256 on 8 H100, the first 40 at batch 8 on one GPU), so the breakdown starts after them (`--steps`). The loop's first collection and freeze walks every object once (about a second on every GPU at the same time) and falls into the record of the step after the first log step, which is step 100 at the default `log_every`: `--steps 101:300` leaves it out.
