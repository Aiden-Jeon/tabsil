"""Tests for hub/push_to_hub.py and hub/sync_hub_card.py (no MLflow or Hub server)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

HUB = Path(__file__).resolve().parents[1] / "hub"
# Put hub/ on sys.path as when run as a script, so push_to_hub can import sync_hub_card.
sys.path.insert(0, str(HUB))
spec = importlib.util.spec_from_file_location("push_to_hub", HUB / "push_to_hub.py")
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)
import sync_hub_card  # noqa: E402

INFERENCE = {
    "n_estimators": 8,
    "member_kinds": ["standard", "power", "robust", "quantile"],
    "max_features_per_member": 768,
    "feature_subsampling": "balanced",
    "categories": "counts",
    "softmax_temperature": 1.0,
}


@pytest.fixture
def no_download(monkeypatch):
    # Must not download large artifacts before checking the card.
    def fail(**kwargs):
        raise AssertionError("artifact downloaded before the card check")

    import mlflow

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", fail)


def _card(tmp_path, sha256="0" * 64):
    card = tmp_path / "card" / "README.md"
    card.parent.mkdir(exist_ok=True)
    card.write_text(f"Figures come from weights `{sync_hub_card.weights_id(sha256)}`.\n")
    return card


def _weights(tmp_path, model, borders=None):
    """SHA-256 of the model.safetensors that push_to_hub would produce from this model."""
    from tabsil.hub import save_pretrained

    out = save_pretrained(model, borders, tmp_path / "expected")
    return sync_hub_card.file_sha256(out / "model.safetensors")


def _checkpoint(tmp_path, with_borders=True):
    """A small TabSil model (smoke size) with its regression borders and the path of its
    checkpoint. Without the borders the checkpoint serves classification only."""
    from train_helpers import tabsil_smoke

    from tabsil.models import randomize_zero_weights, save_checkpoint

    torch.manual_seed(0)
    model = randomize_zero_weights(tabsil_smoke().build_model()).eval()
    borders = torch.linspace(-3, 3, 5001) if with_borders else None
    checkpoint = tmp_path / "step_0005000.pt"
    save_checkpoint(checkpoint, model, borders)
    return model, borders, str(checkpoint)


def test_default_model_card_follows_repo_name(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "HUB_DIR", tmp_path)
    (tmp_path / "tabsil").mkdir()
    (tmp_path / "tabsil" / "README.md").write_text("card\n")
    card = push.resolve_model_card("aiden-jeon/tabsil", None)
    assert card == tmp_path / "tabsil" / "README.md"
    with pytest.raises(SystemExit, match="Model card not found"):
        push.resolve_model_card("aiden-jeon/other", None)


def test_branch_rows_do_not_pass_for_main_and_body_lines_do_not_pass_for_branches():
    # the main card's table lists every branch id; intermediate-step weights must not pass as main
    main, step = "a" * 64, "b" * 64
    card = (
        f"Main weights `{sync_hub_card.weights_id(main)}`, continued from `stage2-20k`.\n"
        f"| `stage2-20k` | `{sync_hub_card.weights_id(step)}` |\n"
        f"| `main` (stage 3) | `{sync_hub_card.weights_id(main)}` |\n"
    )
    assert sync_hub_card.names_main_weights(card, main)
    assert not sync_hub_card.names_main_weights(card, step)
    assert sync_hub_card.names_branch_weights(card, "stage2-20k", step)
    # main weights must not pass as a branch even if a body line lists the branch and main ids
    body = f"`stage2-20k` came before `{sync_hub_card.weights_id(main)}`.\n"
    assert not sync_hub_card.names_branch_weights(body, "stage2-20k", main)
    assert sync_hub_card.names_branch_weights(card, "main", main)


def test_weights_id_is_the_backticked_sha256_prefix(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"weights")
    sha = sync_hub_card.file_sha256(path)
    assert sha == hashlib.sha256(b"weights").hexdigest()
    assert sync_hub_card.weights_id(sha) == f"sha256:{sha[:12]}"
    assert sync_hub_card.names_weights(f"from `sha256:{sha[:12]}`.", sha)
    # an id without backticks or with too few leading digits does not count
    assert not sync_hub_card.names_weights(f"from sha256:{sha[:12]}.", sha)
    assert not sync_hub_card.names_weights(f"from `sha256:{sha[:11]}`.", sha)


def test_repo_without_card_fails_before_download(tmp_path, no_download):
    with pytest.raises(SystemExit, match="Model card not found"):
        push.main(["--checkpoint", "x.pt", "--repo-id", "a/missing"])
    with pytest.raises(SystemExit, match="Model card not found"):
        push.main(
            ["--checkpoint", "x.pt", "--repo-id", "a/b"]
            + ["--model-card", str(tmp_path / "typo.md")]
        )


def test_weights_not_on_card_fail_with_the_id_to_write(tmp_path):
    # the card describes another checkpoint's figures and numbers; the error reports this weights id
    model, borders, checkpoint = _checkpoint(tmp_path)
    expected = sync_hub_card.weights_id(_weights(tmp_path, model, borders))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=f"does not name `{expected}` as the main weights"):
        push.main(
            ["--checkpoint", checkpoint, "--repo-id", "a/b"]
            + ["--model-card", str(_card(tmp_path)), "--out-dir", str(out), "--dry-run"]
        )
    assert not (out / "upload" / "README.md").exists()


def test_only_tabsil_flags():
    # one multitask path: no --paper, no --task
    for extra in (["--paper", "tabpfn_v2"], ["--task", "classification"]):
        with pytest.raises(SystemExit):
            push.parse_args(["--checkpoint", "x.pt", "--repo-id", "a/b", *extra])
    assert push.TITLE == "TabSil"


def test_dry_run_keeps_both_tasks_and_predicts_the_same(tmp_path):
    # one multitask checkpoint serves classification and regression: borders and inference settings
    # travel with it, and the reloaded model must predict like the original
    import numpy as np
    from sklearn.datasets import load_breast_cancer, load_diabetes

    from tabsil import TabSilClassifier, TabSilRegressor
    from tabsil.hub import from_pretrained
    from tabsil.models import config_dict

    model, borders, checkpoint = _checkpoint(tmp_path)
    card = _card(tmp_path, _weights(tmp_path, model, borders))
    out = tmp_path / "out"
    push.main(
        ["--checkpoint", checkpoint, "--repo-id", "a/b"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    assert {p.name for p in upload.iterdir()} == {"model.safetensors", "config.json", "README.md"}
    config = json.loads((upload / "config.json").read_text())
    assert "source" not in config
    assert config["task"] == "multitask" and config["model_type"] == "column_row_transformer"
    assert config["model"] == config_dict(model.config)  # the sizes; the structure is fixed
    assert config["inference"] == INFERENCE
    loaded, _ = from_pretrained(upload)
    assert loaded.inference_config == config["inference"]

    X, y = load_breast_cancer(return_X_y=True)
    a = TabSilClassifier(model).fit(X[:60], y[:60]).predict_proba(X[60:80])
    b = TabSilClassifier(str(upload)).fit(X[:60], y[:60]).predict_proba(X[60:80])
    assert np.array_equal(a, b)
    Xr, yr = load_diabetes(return_X_y=True)
    a = TabSilRegressor(model, borders).fit(Xr[:60], yr[:60]).predict(Xr[60:80])
    reg = TabSilRegressor(str(upload))
    assert torch.equal(reg.borders, borders)
    assert np.array_equal(a, reg.fit(Xr[:60], yr[:60]).predict(Xr[60:80]))


def test_repo_id_defaults_to_the_tabsil_repo():
    from tabsil.recipe import HUB_REPO

    assert push.parse_args(["--checkpoint", "x.pt"]).repo_id == HUB_REPO == "aiden-jeon/tabsil"
    with pytest.raises(SystemExit):
        push.parse_args(["--checkpoint", "x.pt", "--member-kinds", "standard"])  # the flag is gone


def test_needs_a_multitask_checkpoint(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "r").mkdir()
    _, _, single = _checkpoint(tmp_path / "s", with_borders=False)
    # a training state saved before checkpoints recorded the model config
    unreadable = tmp_path / "r" / "step_0000100.pt"
    torch.save({"model": {}, "step": 100, "borders": None}, unreadable)
    common = ["--repo-id", "a/b", "--model-card", str(_card(tmp_path)), "--dry-run"]
    with pytest.raises(SystemExit, match="is not a TabSil multitask model"):
        push.main(["--checkpoint", single, "--out-dir", str(tmp_path / "o1"), *common])
    with pytest.raises(SystemExit, match="Only TabSil multitask checkpoints"):
        push.main(["--checkpoint", str(unreadable), "--out-dir", str(tmp_path / "o2"), *common])


def test_revision_dry_run_writes_a_short_card(tmp_path, monkeypatch):
    import tabsil.hub
    from tabsil.models.column_row_transformer import ColumnRowTransformer

    model, borders, checkpoint = _checkpoint(tmp_path)
    wid = sync_hub_card.weights_id(_weights(tmp_path, model, borders))
    card = _card(tmp_path)
    card.write_text(
        "---\nlicense: apache-2.0\n---\n\n# Card\n\n![x](images/x.png)\n"
        + f"| `stage1-250k` | `{wid}` |\n| `main` (stage 3) | `sha256:{'0' * 12}` |\n"
    )
    (card.parent / "images").mkdir()
    (card.parent / "images" / "x.png").write_bytes(b"png")
    out = tmp_path / "out"
    push.main(
        ["--checkpoint", checkpoint, "--repo-id", "a/b", "--revision", "stage1-250k"]
        + ["--model-card", str(card), "--out-dir", str(out), "--dry-run"]
    )
    upload = out / "upload"
    # branches do not get the main card's images
    assert {p.name for p in upload.iterdir()} == {"model.safetensors", "config.json", "README.md"}
    readme = (upload / "README.md").read_text()
    assert readme.startswith("---\nlicense: apache-2.0\n---\n\n# a/b @ `stage1-250k`")
    assert f"`{wid}`" in readme and 'revision="stage1-250k"' in readme
    assert "from tabsil import from_pretrained" in readme
    assert checkpoint not in readme

    # run the card's example code as is, reading branch files from the upload folder
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(upload / filename)

    monkeypatch.setattr(tabsil.hub, "hf_hub_download", download)
    snippet = readme.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(snippet, namespace)
    assert set(calls) == {("a/b", "stage1-250k")}
    loaded = namespace["model"]
    assert isinstance(loaded, ColumnRowTransformer)
    assert torch.equal(namespace["borders"], borders)
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b), name


def test_revision_needs_its_weights_on_its_table_row(tmp_path, no_download):
    main, step, step_200k = "a" * 64, "b" * 64, "e" * 64
    card = tmp_path / "README.md"
    card.write_text(
        f"Main `{sync_hub_card.weights_id(main)}` continued `step-200k`"
        f" (`{sync_hub_card.weights_id(step_200k)}`).\n"
        f"| `step-100k` | `{sync_hub_card.weights_id(step)}` |\n"
        f"| `main` (500k) | `{sync_hub_card.weights_id(main)}` |\n"
    )
    # a branch with no table row is blocked before downloading (incl. one only in the body text)
    base = ["--repo-id", "a/b", "--model-card", str(card), "--dry-run"]
    base += ["--checkpoint", "runs:/a/x.pt"]
    for revision in ("step-200k", "step-300k"):
        with pytest.raises(SystemExit, match=f"has no `{revision}` row"):
            push.main(base + ["--revision", revision])
    push.check_card_revision(card, "step-100k")
    push.check_card_revision(card, "step-100k", step)
    with pytest.raises(SystemExit, match="row does not name weights id"):
        push.check_card_revision(card, "step-100k", main)
    # the main check does not match other branches' rows
    push.check_card_weights(card, main)
    with pytest.raises(SystemExit, match="as the main weights"):
        push.check_card_weights(card, step)


def test_front_matter_without_trailing_newline():
    match = push.FRONT_MATTER.match("---\nlicense: x\n---")
    assert match and match.group(0).rstrip() == "---\nlicense: x\n---"
    assert push.FRONT_MATTER.match("# no front matter\n") is None


class _RevisionApi:
    def __init__(self, exists):
        self.exists, self.branches = exists, []

    def repo_exists(self, repo_id):
        return self.exists

    def list_repo_commits(self, repo_id):
        class C:
            def __init__(self, commit_id):
                self.commit_id = commit_id

        return [C("latest"), C("first")]

    def create_branch(self, repo_id, branch, revision, exist_ok):
        self.branches.append((branch, revision, exist_ok))


def test_revision_branches_from_the_first_commit_and_needs_main():
    api = _RevisionApi(exists=True)
    push.create_revision(api, "a/b", "step-100k")
    assert api.branches == [("step-100k", "first", True)]
    with pytest.raises(SystemExit, match="Upload main first"):
        push.create_revision(_RevisionApi(exists=False), "a/b", "step-100k")


class _Info:
    def __init__(self, private):
        self.private = private


class _Api:
    def __init__(self, exists, private=False):
        self.exists, self.private = exists, private

    def repo_exists(self, repo_id):
        return self.exists

    def model_info(self, repo_id):
        return _Info(self.private)


def test_private_push_to_an_existing_public_repo_is_refused():
    with pytest.raises(SystemExit, match="is already a public repo"):
        push.check_visibility(_Api(exists=True, private=False), "a/b", private=True)
    push.check_visibility(_Api(exists=True, private=True), "a/b", private=True)
    push.check_visibility(_Api(exists=False), "a/b", private=True)
    push.check_visibility(_Api(exists=True, private=False), "a/b", private=False)


class _PathsApi:
    """HfApi faking only `get_paths_info` and `list_repo_refs`. `sha256` is the main weights (None:
    absent), `branches` maps branch to weights SHA-256. `lfs=False`: a regular non-LFS file."""

    def __init__(self, sha256, exists=True, branches=None, lfs=True):
        self.sha256, self.exists, self.uploads = sha256, exists, []
        self.branches, self.lfs = dict(branches or {}), lfs

    def repo_exists(self, repo_id):
        return self.exists

    def list_repo_refs(self, repo_id):
        class Ref:
            def __init__(self, name):
                self.name = name

        class Refs:
            branches = [Ref(name) for name in ["main", *self.branches]]

        return Refs()

    def get_paths_info(self, repo_id, paths, revision=None):
        sha256 = self.sha256 if revision in (None, "main") else self.branches[revision]

        class Lfs:
            pass

        Lfs.sha256 = sha256

        class Info:
            path = "model.safetensors"
            lfs = Lfs() if self.lfs else None

        return [] if sha256 is None else [Info()]

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

        class Commit:
            commit_url = "https://example/commit"

        return Commit()


def test_hub_weights_sha256_reads_the_lfs_hash():
    assert sync_hub_card.hub_weights_sha256(_PathsApi("c" * 64), "a/b") == "c" * 64
    assert sync_hub_card.hub_weights_sha256(_PathsApi(None), "a/b") is None
    api = _PathsApi("c" * 64, branches={"step-100k": "d" * 64})
    assert sync_hub_card.hub_weights_sha256(api, "a/b", "step-100k") == "d" * 64
    # weights without LFS info must not pass unchecked
    with pytest.raises(SystemExit, match="has no LFS SHA-256"):
        sync_hub_card.hub_weights_sha256(_PathsApi("c" * 64, lfs=False), "a/b")


def test_sync_needs_the_hub_weights_on_the_card(tmp_path, monkeypatch):
    card = _card(tmp_path, "c" * 64)
    argv = ["--card-dir", str(card.parent), "--repo-id", "a/b"]
    api = _PathsApi("d" * 64)
    monkeypatch.setattr(sync_hub_card, "HfApi", lambda: api)
    with pytest.raises(SystemExit, match="sha256:dddddddddddd"):
        sync_hub_card.main(argv)
    api.sha256 = "c" * 64
    sync_hub_card.main(argv)
    assert len(api.uploads) == 1
    # a repo with no weights yet has nothing to check
    api.sha256 = None
    sync_hub_card.main(argv)
    assert len(api.uploads) == 2


def test_sync_checks_branch_rows_against_hub_branches(tmp_path, monkeypatch):
    main, step = "c" * 64, "d" * 64
    card = _card(tmp_path, main)
    card.write_text(
        card.read_text()
        + f"| `step-100k` | `{sync_hub_card.weights_id(step)}` | 0.9 |\n"
        + f"| `step-200k` | `{sync_hub_card.weights_id('e' * 64)}` | 0.9 |\n"
        + f"| `main` (500k) | `{sync_hub_card.weights_id(main)}` | 0.9 |\n"
    )
    argv = ["--card-dir", str(card.parent), "--repo-id", "a/b", "--dry-run"]
    # step-200k is not on the Hub yet, so it is skipped (row added to the card before upload)
    api = _PathsApi(main, branches={"step-100k": step})
    monkeypatch.setattr(sync_hub_card, "HfApi", lambda: api)
    sync_hub_card.main(argv)
    assert sync_hub_card.check_branch_rows(api, "a/b", card.read_text()) == ["step-100k"]
    # the card's step-100k row differs from that Hub branch (typo, another step's weights)
    api.branches["step-100k"] = "f" * 64
    with pytest.raises(
        SystemExit, match="`step-100k` row does not match the weights of Hub step-100k"
    ):
        sync_hub_card.main(argv)


def test_runs_checkpoint_is_downloaded_once_per_out_dir(tmp_path, monkeypatch):
    # after stopping on a missing card id, a rerun with a fixed card does not re-download
    import mlflow

    (tmp_path / "remote").mkdir()
    model, borders, source = _checkpoint(tmp_path / "remote")
    source = Path(source)
    calls = []

    def download(artifact_uri, dst_path):
        calls.append(artifact_uri)
        dst = Path(dst_path) / source.name
        dst.write_bytes(source.read_bytes())
        return str(dst)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    card = _card(tmp_path)
    uri = "runs:/r/checkpoints/step_0005000.pt"
    argv = ["--checkpoint", uri]
    argv += ["--repo-id", "a/b", "--model-card", str(card), "--out-dir", str(tmp_path / "out")]
    argv += ["--dry-run"]
    with pytest.raises(SystemExit, match="as the main weights"):
        push.main(argv)
    card.write_text(f"weights `{sync_hub_card.weights_id(_weights(tmp_path, model, borders))}`\n")
    push.main(argv)
    assert calls == [uri]
    # another checkpoint is downloaded anew (this fake download returns the same file, so it passes)
    other = "runs:/other/checkpoints/step_0005000.pt"
    push.main([argv[0], other, *argv[2:]])
    assert calls == [uri, other]
