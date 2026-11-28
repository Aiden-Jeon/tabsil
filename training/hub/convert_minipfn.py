"""Convert a TabSil checkpoint written by minipfn into a `save_pretrained` folder.

The v1 stage-1 runs (MLflow experiment `tabpfn-v3-reproduction`) were written by minipfn. Since
PR #3 (`9ebae5d`, "Drop minipfn checkpoint compatibility") TabSil reads no minipfn checkpoint, and
the 150K-step one reaches training only as the Hub branch `stage1-150k`. Their weights have
TabSil's names and shapes; only the config differs. minipfn wrote the fields that TabSil now fixes,
with the marker field `task`, and called `gate_hidden` `qassmax_hidden`.

This script applies the mapping PR #3 removed (`FIXED_FIELDS`, `LEGACY_MARKER` and
`RENAMED_FIELDS` of `tabsil.models.column_row_transformer` at `9ebae5d^`, copied below): each fixed
field must hold TabSil's value (a missing one had minipfn's default) and is dropped, the renamed
field is renamed, and the weights are loaded strictly into a model built by today's code. The
folder is written with `tabsil.hub.save_pretrained` and read back with `tabsil.hub.from_pretrained`;
every tensor and the regression borders must equal the source bit for bit.

With `--compare <spec>` the converted weights must have the weights digest
(`tabsil_train.hub.weights_digest`) of another checkpoint, e.g. the 150K run against the Hub branch
it was published as. With `--log-run-name` the folder is logged to MLflow as `model/` of a new run
in `--experiment`, with the source and the weights digest as params, so that a training run can
start from it (`train.py --init-hub runs:/<run_id>/model`) and `eval.py` can read it
(`--classifier-path runs:/<run_id>/model`) without the weights going to the Hub.

From the repository root:
  DATABRICKS_CONFIG_PROFILE=e2-demo-field-eng MLFLOW_TRACKING_URI=databricks \\
    uv run --frozen --package tabsil-train python training/hub/convert_minipfn.py \\
    --checkpoint runs:/<run_id>/model/final.pt --out-dir /tmp/convert-150k \\
    --compare hf://aiden-jeon/tabsil@stage1-150k
  ... --log-run-name tabsil-stage1-250k --experiment /Users/<user>/pfn-cookbook/tabsil
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tabsil import hub
from tabsil.models import DEFAULT_MODEL_TYPE, build_model
from tabsil_train.hub import RUNS_PREFIX, download_artifact, load_model, weights_digest

# minipfn's config fields that TabSil fixes: name -> (minipfn's default, the only value TabSil
# accepts). Copied from `tabsil.models.column_row_transformer` at 9ebae5d^ (PR #3 removed it)
FIXED_FIELDS: dict[str, tuple[Any, Any]] = {
    "task": ("classification", "multitask"),
    "qk_norm": (False, True),
    "encoder": ("linear", "fourier"),
    "target_norm": (False, True),
    "head_mlp": (False, True),
    "n_buckets": (5000, 5000),
    "logit_softcap": (None, None),
    "qk_norm_affine": (True, False),
    "column_summary_temperature": ("qassmax", "logscale"),
    "column_spread_temperature": ("none", "none"),
    "row_temperature": ("none", "gated_logscale"),
    "icl_temperature": ("qassmax", "logscale"),
    "decoder_temperature": ("qassmax", "logscale"),
    "decoder_qk_norm": (False, False),
    "stage_layout": ("sequential", "interleaved"),
    "regression_head": ("bar", "quantile"),
    "categorical_encoder": (False, False),
}
# minipfn always wrote this field and TabSil never does: its presence marks a minipfn config
LEGACY_MARKER = "task"
# fields renamed since minipfn: old name -> current name
RENAMED_FIELDS = {"qassmax_hidden": "gate_hidden"}


def tabsil_config(values: dict) -> dict:
    """The TabSil model config of a minipfn one: fixed fields checked and dropped, renamed fields
    renamed. SystemExit if `values` is not a minipfn config or a fixed field differs."""
    if LEGACY_MARKER not in values:
        raise SystemExit(f"not a minipfn config (no `{LEGACY_MARKER}` field)")
    values = dict(values)
    wrong = {}
    for name, (old_default, fixed) in FIXED_FIELDS.items():
        value = values.pop(name, old_default)
        if value != fixed:
            wrong[name] = (value, fixed)
    if wrong:
        raise SystemExit(
            "this minipfn model is not the TabSil structure; TabSil fixes these fields"
            f" (saved value, TabSil value): {wrong}"
        )
    for old, new in RENAMED_FIELDS.items():
        if old in values:
            values[new] = values.pop(old)
    return values


def convert(state: dict) -> tuple[torch.nn.Module, torch.Tensor | None]:
    """(model, regression borders) of a minipfn checkpoint dict (`model_type`, `config`,
    `state_dict`, `borders`), the weights loaded strictly into a model built by today's code."""
    model = build_model(state.get("model_type", DEFAULT_MODEL_TYPE), tabsil_config(state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    return model.eval(), state.get("borders")


def check_same(source: dict, model: torch.nn.Module, borders: torch.Tensor | None) -> None:
    """SystemExit unless `model` holds exactly the source's tensors and borders."""
    weights = model.state_dict()
    if set(weights) != set(source["state_dict"]):
        raise SystemExit("the converted model's tensors are not the source's")
    for name, tensor in source["state_dict"].items():
        other = weights[name]
        if other.dtype != tensor.dtype or not torch.equal(other, tensor):
            raise SystemExit(f"tensor {name} differs from the source")
    source_borders = source.get("borders")
    if (source_borders is None) != (borders is None) or (
        borders is not None and not torch.equal(borders, source_borders)
    ):
        raise SystemExit("the regression borders differ from the source")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="a minipfn .pt file or runs:/ path")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compare", help="a checkpoint whose weights digest must match")
    parser.add_argument("--log-run-name", help="log the folder to MLflow under this run name")
    parser.add_argument("--experiment", help="the MLflow experiment of --log-run-name")
    args = parser.parse_args(argv)
    if args.log_run_name and not args.experiment:
        parser.error("--log-run-name needs --experiment")
    return args


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.checkpoint)
    if args.checkpoint.startswith(RUNS_PREFIX):
        path = download_artifact(args.checkpoint, args.out_dir / "source")
    source = torch.load(path, map_location="cpu", weights_only=True)
    model, borders = convert(source)
    folder = hub.save_pretrained(model, borders, args.out_dir / "model")
    reloaded, reloaded_borders = hub.from_pretrained(folder)
    check_same(source, reloaded, reloaded_borders)
    digest = weights_digest(reloaded, reloaded_borders)
    print(f"converted {args.checkpoint} -> {folder} (weights digest {digest})")
    if args.compare:
        other, other_borders = load_model(args.compare, "regression", args.out_dir / "compare")
        other_digest = weights_digest(other, other_borders)
        if other_digest != digest:
            raise SystemExit(
                f"weights digest {digest} differs from {args.compare}'s {other_digest}"
            )
        print(f"same weights as {args.compare}")
    if args.log_run_name:
        import mlflow

        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=args.log_run_name) as run:
            mlflow.log_params(
                {
                    "source_checkpoint": args.checkpoint,
                    "weights_digest": digest,
                    "converter": "convert_minipfn",
                }
            )
            mlflow.log_artifacts(str(folder), artifact_path="model")
            print(f"logged to MLflow run {run.info.run_id}: runs:/{run.info.run_id}/model")
    return digest


if __name__ == "__main__":
    main()
