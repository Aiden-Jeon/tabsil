"""Phase 0.4 of the v2 experiment plan: the learned attention temperatures of saved weights.

Every attention that reads a variable number of keys multiplies q by a_h * log n
(`tabsil.models.column_row_transformer.LogScale`): a_h is a learned scalar per head, drawn from
U(0, 1) at initialization, and q and k carry an RMSNorm without gain, so a_h bounds the logits
(sqrt(head dim) * a_h * log n). Kumo-Tabular reports that with BF16 the temperature of its first
in-context layer grew about 30-fold, and trains its regressors in FP16 for that reason. This
script reads a_h from the weight files only (no forward pass) and prints, per layer and per
checkpoint, the mean, the largest |a_h| and the largest |a_h| over the mean of the U(0, 1) draw
(0.5). The gated row-stage attention also has a gate (1 + tanh) that can at most double a_h; its
weights are not read here.

    uv run --package tabsil-train python training/scripts/v2_prior_diag/attention_temperatures.py \
        aiden-jeon/tabsil@stage1-150k aiden-jeon/tabsil@stage2-10k aiden-jeon/tabsil@v1.1

An argument is a Hub repo id (`owner/name@revision`) or a local `.safetensors` / `.pt` file
(a `tabsil.models.save_checkpoint` file). `--csv` writes the per-head values.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

# U(0, 1) initialization of head_scale: the expected value of a freshly drawn a_h
INIT_MEAN = 0.5
SUFFIX = ".log_scale.head_scale"
# Display order of the attention groups, with the meaning of each
GROUPS = {
    "column_blocks": "column summary (inducing points -> train rows)",
    "row_blocks": "row stage (cells of a row, gated)",
    "icl_layers": "row ICL (rows -> train rows)",
    "heads": "classification retrieval decoder",
}


def load_state(source: str) -> dict[str, torch.Tensor]:
    """The head_scale tensors of a Hub repo (`owner/name@revision`) or a local weight file."""
    path = Path(source)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        repo, _, revision = source.partition("@")
        path = Path(hf_hub_download(repo, "model.safetensors", revision=revision or None))
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            return {k: f.get_tensor(k) for k in f.keys() if k.endswith(SUFFIX)}
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state.get("state_dict", state))
    return {k.removeprefix("module."): v for k, v in state.items() if k.endswith(SUFFIX)}


def layer_key(name: str) -> tuple[int, int, str]:
    """Sort key and label of a parameter: (group order, layer index, label)."""
    group = name.split(".")[0]
    match = re.match(r"[a-z_]+\.(\d+)\.", name)
    index = int(match.group(1)) if match else 0
    label = f"{group}.{index}" if match else name.removesuffix(SUFFIX)
    return list(GROUPS).index(group) if group in GROUPS else len(GROUPS), index, label


def temperatures(sources: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-layer table, per-head table). Per layer: mean a_h, max |a_h| and max |a_h| / 0.5,
    one column group per source."""
    heads, layers = [], []
    for source in sources:
        for name, value in load_state(source).items():
            order, index, label = layer_key(name)
            a = value.float()
            for h, v in enumerate(a.tolist()):
                heads.append({"source": source, "layer": label, "head": h, "a": v})
            layers.append(
                {
                    "source": source,
                    "order": order,
                    "index": index,
                    "layer": label,
                    "mean": a.mean().item(),
                    "max_abs": a.abs().max().item(),
                    "max_over_init": a.abs().max().item() / INIT_MEAN,
                }
            )
    if not layers:
        raise SystemExit(f"no *{SUFFIX} parameters found in {sources}")
    table = (
        pd.DataFrame(layers)
        .pivot_table(
            index=["order", "index", "layer"], columns="source", values=["mean", "max_abs"]
        )
        .sort_index()
        .droplevel([0, 1])
    )
    table.columns = [f"{src} {stat}" for stat, src in table.columns]
    table = table[[f"{s} {stat}" for s in sources for stat in ("mean", "max_abs")]]
    return table, pd.DataFrame(heads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--csv", help="write the per-head values to this file")
    args = parser.parse_args()
    table, heads = temperatures(args.sources)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.round(3).to_string())
    if args.csv:
        heads.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
