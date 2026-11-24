"""Upload trained TabSil weights to the Hugging Face Hub.

Example, from the repo root (in a shell that has loaded .env.local):
  MLFLOW_TRACKING_URI=databricks uv run --package tabsil-train python training/hub/push_to_hub.py \
    --checkpoint runs:/<stage 3 run_id>/checkpoints/step_0005000.pt --dry-run

The checkpoint is a TabSil multitask checkpoint (`column_row_transformer`, one set of weights for
classification and regression), anything `tabsil_train.hub.load_model` reads. Both tasks are
uploaded together: the regression borders go into the same `model.safetensors`, and the inference
settings of the two tasks must be equal so that `config.json` holds one `inference` entry
(`tabsil.recipe.default_inference`: 8 `standard` members, at most 768 columns per member,
`balanced`). Other model settings are in the `model` entry of `config.json`.

The checkpoint is converted to `model.safetensors` + `config.json` and uploaded with a model card.
The card is given by `--model-card`, or else found at `training/hub/<name>/README.md` from the repo
name. Only images the card links as `](images/...)` are copied from next to the card, and files in
the Hub's `images/` that the card no longer uses are deleted. `--dry-run` builds the folder without
uploading.

The repo is TabSil's published repo (`HUB_REPO`, `tabsil.recipe.HUB_REPO`) unless `--repo-id`
gives another one. So that the card's figures and numbers cannot belong to another checkpoint, the
card must name the weights id of the converted `model.safetensors`
(`` `sha256:<first 12 hex digits of the SHA-256>` ``) before anything is uploaded. The id is only
known after conversion, so run with `--dry-run` first and write the printed id into the card. A
`runs:/` checkpoint is downloaded into `--out-dir` and reused by later runs, so rerunning after
fixing the card does not download it again. Ids in other branches' rows of the intermediate
checkpoint table do not count for main. The card, Hub files and Hub commit messages never contain
the training run id (MLflow). The visibility of an existing repo is not changed, so with
`--private` an upload to a public repo is refused.

Intermediate training checkpoints go to a branch of the same repo, e.g. `--revision step-100k`.
`main` (the final weights the model card describes) must be uploaded first so the repo exists. The
branch is created from the repo's first commit, so it does not inherit `main`'s card and images;
it holds the weights, config and a short README describing that step. The branch and its weights
id must then appear together on one line of the `main` card (a row of the intermediate checkpoint
table), so that weights of a different step cannot be uploaded under the branch name.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi

# The script folder (training/hub/) is on sys.path, so the card sync script next to it is imported
# directly.
from sync_hub_card import (
    WEIGHTS_ID_HEX,
    WEIGHTS_NAME,
    branch_rows,
    card_images,
    file_sha256,
    names_branch_weights,
    names_main_weights,
    weights_id,
)

from tabsil.recipe import HUB_REPO

REPO_ROOT = Path(__file__).resolve().parents[2]
HUB_DIR = Path(__file__).resolve().parent
TITLE = "TabSil"
# The tasks a TabSil multitask checkpoint must serve
MULTITASK = ("classification", "regression")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="checkpoint (file or runs:/ URI)")
    parser.add_argument(
        "--repo-id", default=HUB_REPO, help=f"Hub repo id (owner/name, default {HUB_REPO})"
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        help="model card uploaded as the Hub README (default: training/hub/<repo name>/README.md)",
    )
    parser.add_argument(
        "--revision", help="branch for an intermediate checkpoint (e.g. step-100k); default main"
    )
    parser.add_argument("--out-dir", default="hub_upload")
    parser.add_argument("--private", action="store_true", help="create a new repo as private")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.revision == "main":
        args.revision = None
    return args


def resolve_model_card(repo_id: str, model_card: Path | None) -> Path:
    """Pick the model card. By default only the card named after the repo is used, so that another
    model's card is never attached."""
    card = model_card or HUB_DIR / repo_id.split("/")[-1] / "README.md"
    if not card.is_file():
        raise SystemExit(
            f"Model card not found: {card}. Create training/hub/<repo name>/README.md"
            " or pass --model-card."
        )
    return card


def check_card_weights(card: Path, sha256: str) -> None:
    """Check that the card's figures and numbers belong to the weights being uploaded. Ids in
    other branches' rows of the intermediate checkpoint table do not count (so that weights of an
    intermediate step are not uploaded as main)."""
    if not names_main_weights(card.read_text(), sha256):
        raise SystemExit(
            f"Model card does not name `{weights_id(sha256)}` as the main weights (rows for other"
            " branches do not count). If the card describes this checkpoint, add that id to it."
            " If it describes another checkpoint, regenerate the figures and update the card's"
            " numbers and id."
        )


def check_visibility(api: HfApi, repo_id: str, private: bool) -> None:
    """`create_repo(exist_ok=True)` does not change the visibility of an existing repo.

    Prevents a `--private` upload from landing in a public repo."""
    if private and api.repo_exists(repo_id) and not api.model_info(repo_id).private:
        raise SystemExit(
            f"{repo_id} is already a public repo, so a --private upload would be public. Pass"
            " another repo id or make it private on the Hub first."
        )


def _save_and_reload(model, borders, tasks: tuple[str, ...], args, upload_dir: Path):
    """Convert with the inference settings of `tasks` and read the result back. Checks that the
    regression borders and inference settings are unchanged. With several tasks (multitask) the
    inference settings must be equal across tasks (config.json holds one)."""
    from tabsil.hub import from_pretrained, save_pretrained
    from tabsil.recipe import default_inference

    def by_task(m):
        # inference settings per task
        out = {}
        for task in tasks:
            inference = default_inference(m, task)
            out[task] = {**inference, "member_kinds": tuple(inference["member_kinds"])}
        return out

    expected = by_task(model)
    inference = expected[tasks[0]]
    if any(v != inference for v in expected.values()):
        raise SystemExit(
            f"Inference settings differ across tasks, so config.json cannot hold one: {expected}"
        )
    inference = {**inference, "member_kinds": list(inference["member_kinds"])}
    print("inference:", inference)
    save_pretrained(model, borders, upload_dir, inference=inference)
    reloaded, reloaded_borders = from_pretrained(upload_dir)
    if (borders is None) != (reloaded_borders is None) or (
        borders is not None and not torch.equal(borders.cpu(), reloaded_borders)
    ):
        raise RuntimeError("Converted regression borders differ from the original")
    # The reloaded model must yield the same settings from config.json's `inference` alone
    if by_task(reloaded) != expected:
        raise RuntimeError("Reloaded inference settings differ from the ones written")
    return reloaded


def prepare(args: argparse.Namespace, out_dir: Path, upload_dir: Path):
    """TabSil multitask: convert a checkpoint serving both classification and regression, and
    return the original and reloaded models. Checks the regression borders (same file) and the
    inference settings of both tasks."""
    from tabsil.hub import served_tasks
    from tabsil_train.hub import load_model

    try:
        model, borders = load_model(args.checkpoint, MULTITASK[0], out_dir / "checkpoint")
    except SystemExit as e:  # e.g. a regression-only model: also say why it is rejected
        raise SystemExit(
            f"{e} Only TabSil multitask checkpoints (classification and regression) are uploaded."
        ) from e
    if (
        getattr(model, "model_type", None) != "column_row_transformer"
        or served_tasks(model, borders) != MULTITASK
    ):
        raise SystemExit(
            f"{args.checkpoint} is not a TabSil multitask model serving classification and"
            f" regression (model {getattr(model, 'model_type', type(model).__name__)},"
            f" task {'·'.join(served_tasks(model, borders))})."
        )
    return model, _save_and_reload(model, borders, MULTITASK, args, upload_dir)


FRONT_MATTER = re.compile(r"\A---\r?\n.*?^---[ \t]*(?:\r?\n|\Z)", re.S | re.M)


def check_card_revision(card: Path, revision: str, sha256: str | None = None) -> None:
    """Check that the branch row (a line starting with `` | `step-100k` | ``) of the main card's
    intermediate checkpoint table names its weights id. A body line naming the branch together with
    other weights does not count, so that another step's weights are not uploaded to the branch.

    Without `sha256` (before the checkpoint is downloaded) only the presence of the row is
    checked."""
    text = card.read_text()
    if revision not in branch_rows(text):
        raise SystemExit(
            f"Model card's intermediate checkpoint table has no `{revision}` row. Add the row"
            f" (| `{revision}` | `sha256:...` | ...) or check the branch name."
        )
    if sha256 is not None and not names_branch_weights(text, revision, sha256):
        raise SystemExit(
            f"Model card's `{revision}` row does not name weights id `{weights_id(sha256)}`. Add"
            " this id to the row or check the branch name and checkpoint."
        )


def revision_card(model_card: Path, repo_id: str, revision: str, sha256: str) -> str:
    """README of an intermediate checkpoint branch: the main card's front matter and a short
    body."""
    match = FRONT_MATTER.match(model_card.read_text())
    front = match.group(0).rstrip() + "\n\n" if match else ""
    url = f"https://huggingface.co/{repo_id}"
    return (
        f"{front}# {repo_id} @ `{revision}`\n\n"
        f"An intermediate training checkpoint, kept for comparison. The model card, benchmark"
        f" numbers for every checkpoint, and usage are on the [main branch]({url}).\n\n"
        f"Weights: `{weights_id(sha256)}` (first {WEIGHTS_ID_HEX} hex digits of the SHA-256 of"
        f" `{WEIGHTS_NAME}`).\n\n"
        "```python\n"
        "from tabsil import from_pretrained\n\n"
        f'model, borders = from_pretrained("{repo_id}", revision="{revision}")\n'
        "```\n"
    )


def create_revision(api: HfApi, repo_id: str, revision: str) -> None:
    """Create the branch from the repo's first commit (kept if it exists). main must be uploaded
    first."""
    if not api.repo_exists(repo_id):
        raise SystemExit(f"{repo_id} does not exist. Upload main first, without --revision.")
    first = api.list_repo_commits(repo_id)[-1].commit_id
    api.create_branch(repo_id, branch=revision, revision=first, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Check the card before downloading a large artifact; the weights id is checked after
    # conversion.
    model_card = resolve_model_card(args.repo_id, args.model_card)
    images = card_images(model_card)
    if args.revision is not None:
        check_card_revision(model_card, args.revision)
    api = None if args.dry_run else HfApi()
    if api is not None:
        check_visibility(api, args.repo_id, args.private)
        if args.revision is not None and not api.repo_exists(args.repo_id):
            raise SystemExit(
                f"{args.repo_id} does not exist. Upload main first, without --revision."
            )

    out_dir = Path(args.out_dir)
    upload_dir = out_dir / "upload"
    # Start from an empty folder every time so that files from a previous run are not uploaded.
    shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True)
    model, reloaded = prepare(args, out_dir, upload_dir)
    # Before uploading, check that the converted files alone reproduce the same weights.
    for (name, a), b in zip(
        model.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        if not torch.equal(a.cpu(), b.cpu()):
            raise RuntimeError(f"Converted weights differ from the original: {name}")
    sha256 = file_sha256(upload_dir / WEIGHTS_NAME)
    print("weights:", weights_id(sha256))
    if args.revision is None:
        check_card_weights(model_card, sha256)
    else:
        check_card_revision(model_card, args.revision, sha256)
    if args.revision is None:
        shutil.copy(model_card, upload_dir / "README.md")
        for rel in images:
            (upload_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(model_card.parent / rel, upload_dir / rel)
    else:
        card = revision_card(model_card, args.repo_id, args.revision, sha256)
        (upload_dir / "README.md").write_text(card)
    prepared = sorted(str(p.relative_to(upload_dir)) for p in upload_dir.rglob("*") if p.is_file())
    print("prepared:", prepared)
    if api is None:
        return

    if args.revision is None:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    else:
        create_revision(api, args.repo_id, args.revision)
    # Files re-uploaded in the same commit are not deleted, so only images the card no longer
    # uses are removed.
    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=upload_dir,
        revision=args.revision,
        commit_message=f"Upload {TITLE} weights {weights_id(sha256)}",
        delete_patterns=["images/*"],
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
