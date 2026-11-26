"""Tag the published weights with a tabsil release's inference settings, without new weights.

`TabSilClassifier()` and `TabSilRegressor()` download `tabsil.recipe.HUB_REPO` at the tag
`HUB_REVISION`, and the `inference` entry of that revision's `config.json` comes before the
package defaults (`tabsil.recipe.default_inference`). When a release changes the inference defaults
for the same weights, this script makes the tag the release downloads (tabsil 0.1.1: `v1.1`):

1. read `config.json` at `--source` (the tag of the weights, `v1`), and replace its `inference`
   entry with `tabsil.recipe`'s settings (the two tasks must agree; config.json holds one entry)
2. create the branch `--branch` from `--source`, commit only the new `config.json` to it, and
   check that `model.safetensors` there is the source's (by the SHA-256 the Hub records)
3. tag that commit `--tag`

`main` and the source tag do not change, so older releases (which download `v1`) and users of
`main` keep their settings: tabsil 0.1.0 rejects member kinds it does not know. An existing tag is
never moved. `--dry-run` prints the new `inference` entry and changes nothing.

    uv run python training/hub/tag_inference.py --dry-run
    uv run python training/hub/tag_inference.py   # after the dry run looks right
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from tabsil.recipe import HUB_REPO, HUB_REVISION, INFERENCE, default_inference

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=HUB_REPO)
    parser.add_argument("--source", default="v1", help="the tag (or commit) of the weights")
    parser.add_argument("--tag", default=HUB_REVISION, help="the tag to create")
    parser.add_argument("--branch", default=None, help="default: inference-<tag>")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def recipe_inference() -> dict:
    """The `inference` entry for config.json: `default_inference` of a model without saved
    settings, which must be the same for both tasks (JSON types: lists, not tuples)."""

    class Unsaved:
        inference_config = None

    per_task = {task: default_inference(Unsaved(), task) for task in INFERENCE["n_estimators"]}
    first, *rest = per_task.values()
    if any(other != first for other in rest):
        raise RuntimeError(f"the inference settings differ between the tasks: {per_task}")
    return {**first, "member_kinds": list(first["member_kinds"])}


def tagged_config(config: dict, inference: dict) -> dict:
    """`config` with its `inference` entry replaced (everything else unchanged)."""
    return {**config, "inference": inference}


def weights_sha256(api: HfApi, repo_id: str, revision: str) -> str:
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    for sibling in info.siblings or []:
        if sibling.rfilename == WEIGHTS_NAME and sibling.lfs is not None:
            return sibling.lfs.sha256
    raise RuntimeError(f"{repo_id}@{revision} has no LFS {WEIGHTS_NAME}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    branch = args.branch or f"inference-{args.tag}"
    api = HfApi()
    refs = api.list_repo_refs(args.repo_id)
    if any(t.name == args.tag for t in refs.tags):
        raise SystemExit(f"the tag {args.tag} exists already; tags are not moved")
    config = json.loads(
        Path(hf_hub_download(args.repo_id, CONFIG_NAME, revision=args.source)).read_text()
    )
    new = tagged_config(config, recipe_inference())
    print(f"{args.repo_id}@{args.source} inference:", json.dumps(config.get("inference")))
    print(f"{args.repo_id}@{args.tag} inference:", json.dumps(new["inference"]))
    source_weights = weights_sha256(api, args.repo_id, args.source)
    print(f"weights sha256 at {args.source}: {source_weights[:12]}")
    if args.dry_run:
        print(
            f"dry run: would create branch {branch} from {args.source}, commit {CONFIG_NAME},"
            f" tag {args.tag}"
        )
        return
    if any(b.name == branch for b in refs.branches):
        raise SystemExit(f"the branch {branch} exists already")
    api.create_branch(args.repo_id, branch=branch, revision=args.source)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / CONFIG_NAME
        path.write_text(json.dumps(new, indent=2) + "\n")
        commit = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=CONFIG_NAME,
            repo_id=args.repo_id,
            revision=branch,
            commit_message=f"Inference settings for {args.tag} (weights of {args.source})",
        )
    if weights_sha256(api, args.repo_id, branch) != source_weights:
        raise SystemExit(f"{WEIGHTS_NAME} on {branch} differs from {args.source}; not tagged")
    api.create_tag(args.repo_id, tag=args.tag, revision=commit.oid)
    print(f"tagged {args.tag} at {commit.oid} ({commit.commit_url})")


if __name__ == "__main__":
    main()
