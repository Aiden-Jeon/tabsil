"""Upload only the model card (README and the images it links) to a Hugging Face Hub repo.

Example, from the repo root:
  uv run --no-project --with huggingface_hub python training/hub/sync_hub_card.py \
    --card-dir training/hub/tabsil --repo-id aiden-jeon/tabsil

Weights are not uploaded (after new training, use `training/hub/push_to_hub.py`), so if the repo
does not exist yet the card is skipped rather than uploaded first. The card must name Hub `main`'s
`model.safetensors` by its weights id (`` `sha256:<first 12 hex digits of the SHA-256>` ``), so that
the card's numbers cannot belong to a checkpoint other than the uploaded weights. Ids in branch rows
of the intermediate checkpoint table (lines starting with `` | `step-100k` | ``) belong to that
branch: they are not used for the main check, and are matched against that branch's weights if the
branch exists on the Hub. Hashes come from the SHA-256 the Hub records for each LFS file, so no
weights are downloaded. Neither the card nor the Hub contains the training run id (MLflow). Files in
the Hub's `images/` that the card no longer uses are deleted. Runs with huggingface_hub only,
without torch. `push_to_hub.py` also uses this file's card checks.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

IMAGE_LINK = re.compile(r"\]\((images/[^)\s]+)\)")
WEIGHTS_NAME = "model.safetensors"
# Number of leading SHA-256 hex digits written on the card; enough to catch a wrong checkpoint
WEIGHTS_ID_HEX = 12
# A row of the intermediate checkpoint table: the first cell is `` `branch` ``
# (the `main` row is `` | `main` (500k) | ``)
BRANCH_ROW = re.compile(r"^\|\s*`([^`]+)`")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--card-dir", type=Path, required=True, help="training/hub/<name> containing README.md"
    )
    parser.add_argument("--repo-id", required=True, help="Hub repo id (owner/name)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def card_images(card: Path) -> list[str]:
    """Relative paths of the images under `images/` that the card links. Missing files and paths
    outside the folder are rejected."""
    images_dir = (card.parent / "images").resolve()
    paths = sorted(set(IMAGE_LINK.findall(card.read_text())))
    for rel in paths:
        path = (card.parent / rel).resolve()
        if not path.is_relative_to(images_dir) or not path.is_file():
            raise SystemExit(f"Image linked by the card is not in {images_dir}: {rel}")
    return paths


def weights_id(sha256: str) -> str:
    """Weights id written on the card: `sha256:` followed by the first `WEIGHTS_ID_HEX` hex digits
    of the SHA-256 of `model.safetensors`."""
    return f"sha256:{sha256[:WEIGHTS_ID_HEX]}"


def file_sha256(path: Path) -> str:
    """SHA-256 of a file (the same value the Hub records for an LFS file)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def names_weights(text: str, sha256: str) -> bool:
    """Whether `text` (the card or one of its lines) names the weights id as `` `...` ``."""
    return f"`{weights_id(sha256)}`" in text


def branch_rows(text: str) -> dict[str, list[str]]:
    """Rows of the card's intermediate checkpoint table, per branch."""
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if match := BRANCH_ROW.match(line):
            rows.setdefault(match.group(1), []).append(line)
    return rows


def names_main_weights(text: str, sha256: str) -> bool:
    """Whether the card names the id as the main weights. Ids in other branches' rows do not count
    (the main card lists every branch's id in its table, so a plain search would let an
    intermediate step's weights pass)."""
    return any(
        names_weights(line, sha256)
        for line in text.splitlines()
        if (match := BRANCH_ROW.match(line)) is None or match.group(1) == "main"
    )


def names_branch_weights(text: str, branch: str, sha256: str) -> bool:
    """Whether the card's `branch` row names the id. A body line naming the branch together with
    other weights (e.g. "main continues training from `step-500k` (`sha256:...`)") does not
    count."""
    return any(names_weights(line, sha256) for line in branch_rows(text).get(branch, []))


def hub_weights_sha256(api: HfApi, repo_id: str, revision: str | None = None) -> str | None:
    """SHA-256 of `model.safetensors` on the Hub, or None if there are no weights.

    Stops if the file exists without LFS info, since it cannot be checked (so nothing is uploaded
    unchecked)."""
    for info in api.get_paths_info(repo_id, [WEIGHTS_NAME], revision=revision):
        if info.path != WEIGHTS_NAME:
            continue
        lfs = getattr(info, "lfs", None)
        if lfs is None or not getattr(lfs, "sha256", None):
            where = f"{repo_id}@{revision}" if revision else repo_id
            raise SystemExit(
                f"{WEIGHTS_NAME} in {where} has no LFS SHA-256, so it cannot be matched against"
                " the card (it may have been uploaded as a regular, non-LFS file)."
            )
        return lfs.sha256
    return None


def check_branch_rows(api: HfApi, repo_id: str, text: str) -> list[str]:
    """Check that the ids in the card table's branch rows equal the Hub weights of those branches.
    Returns the branches checked.

    Branches not yet on the Hub (rows are added to the card before upload) are skipped."""
    hub_branches = {ref.name for ref in api.list_repo_refs(repo_id).branches}
    checked = []
    for branch, lines in branch_rows(text).items():
        if branch == "main" or not any("`sha256:" in line for line in lines):
            continue
        if branch not in hub_branches:
            print(f"{repo_id} has no {branch} branch yet; its row was not checked")
            continue
        sha256 = hub_weights_sha256(api, repo_id, branch)
        if sha256 is None or not names_branch_weights(text, branch, sha256):
            found = "has no weights" if sha256 is None else f"id is `{weights_id(sha256)}`"
            raise SystemExit(
                f"Card's `{branch}` row does not match the weights of Hub {branch} branch"
                f" (Hub {found})."
            )
        checked.append(branch)
    return checked


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    card = args.card_dir / "README.md"
    if not card.is_file():
        raise SystemExit(f"Model card not found: {card}")
    images = card_images(card)
    api = HfApi()
    if not api.repo_exists(args.repo_id):
        # Log a warning in GitHub Actions without failing the job
        print(f"::warning::{args.repo_id} does not exist yet; card not uploaded (weights first)")
        return
    text = card.read_text()
    sha256 = hub_weights_sha256(api, args.repo_id)
    if sha256 is not None and not names_main_weights(text, sha256):
        raise SystemExit(
            f"Card does not name the Hub weights id `{weights_id(sha256)}` (rows for other"
            " branches do not count). If the card describes another checkpoint, re-upload the"
            " weights with it."
        )
    if sha256 is not None:
        print("branches checked:", check_branch_rows(api, args.repo_id, text))
    with tempfile.TemporaryDirectory() as tmp:
        upload = Path(tmp)
        shutil.copy(card, upload / "README.md")
        for rel in images:
            (upload / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(card.parent / rel, upload / rel)
        print("prepared:", ["README.md", *images])
        if args.dry_run:
            return
        commit = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=upload,
            commit_message=f"Sync model card from {args.card_dir}",
            delete_patterns=["images/*"],
        )
        print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
