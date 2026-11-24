"""Training and evaluation run helpers: MLflow run logging, the learning rate schedule and the
torchrun process setup."""

import contextlib
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import tabsil


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def log_run_source(script: str | Path, env_prefix: str, artifact_path: str = "code") -> None:
    """Log the script, the `tabsil` and `tabsil_train` sources and the command line to the active
    MLflow run.

    Job snapshots without `.git` (e.g. Databricks AI Runtime) have no commit, so the code that
    actually ran is uploaded to `artifact_path/` and can be diffed against the repository. Inside
    a git checkout (local runs) the commit and whether the tree is dirty are tagged as well.
    """
    import mlflow

    script = Path(script).resolve()
    mlflow.log_artifact(str(script), artifact_path)
    # the inference package and this one (`code/tabsil/...`, `code/tabsil_train/...`)
    for package in (Path(tabsil.__file__).resolve().parent, Path(__file__).resolve().parent):
        for path in sorted(package.rglob("*.py")):
            rel = path.relative_to(package.parent).parent.as_posix()
            mlflow.log_artifact(str(path), f"{artifact_path}/{rel}")

    tags = {"command": shlex.join([script.name, *sys.argv[1:]])}
    tags.update({f"env.{k}": v for k, v in sorted(os.environ.items()) if k.startswith(env_prefix)})
    commit = _git("rev-parse", "HEAD", cwd=script.parent)
    if commit:
        tags["git_commit"] = commit
        tags["git_dirty"] = str(bool(_git("status", "--porcelain", cwd=script.parent)))
    mlflow.set_tags(tags)


@contextlib.contextmanager
def mlflow_eval_run(enabled: bool, script: str | Path, env_prefix: str, default_experiment: str):
    """One MLflow run for an evaluation script: the platform's run (`MLFLOW_RUN_ID`) on AI
    Runtime, otherwise a new run in `<env_prefix>EXPERIMENT` (default `default_experiment`). The
    code is logged with `log_run_source`; an exception ends the run as FAILED. Does nothing when
    `enabled` is False."""
    if not enabled:
        yield
        return
    import mlflow

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(os.environ.get(f"{env_prefix}EXPERIMENT", default_experiment))
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        log_run_source(script, env_prefix=env_prefix)
        yield


def lr_factor(step: int, warmup_steps: int, schedule_steps: int) -> float:
    """Learning rate multiplier: linear warmup from 0, then cosine decay to 0."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, schedule_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


@dataclass
class DistContext:
    """The position of one torchrun process."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> tuple[DistContext, torch.device]:
    """Open an NCCL process group under torchrun (`WORLD_SIZE` > 1); otherwise one GPU or CPU."""
    import torch.distributed as dist

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        d = DistContext(
            int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        )
        torch.cuda.set_device(d.local_rank)
        dist.init_process_group("nccl")
        return d, torch.device("cuda", d.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistContext(), device
