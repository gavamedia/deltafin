#!/usr/bin/env python3
"""Safely update an existing Deltafin checkout in one command.

Run this with the Python interpreter from the environment that already runs
Deltafin:

    ./venv/bin/python tools/upgrade.py

The updater is intentionally conservative. It only operates on a clean normal
branch with a configured remote upstream, fetches that remote, and permits only
a fast-forward. It never runs setup, reset, clean, model conversion, or model
download commands. Ignored model weights and caches therefore stay in place.

Python dependencies are installed into the current virtual environment. The
host-specific PyTorch build is constrained to its exact installed version and
checked again afterward. Native libraries are rebuilt by build_native.py, whose
validated multi-artifact installation is transactional.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent


class UpgradeError(RuntimeError):
    """The requested update could not be completed safely."""


@dataclass(frozen=True)
class GitState:
    branch: str
    upstream: str
    remote: str
    head: str
    upstream_head: str
    relation: str


CommandResult = subprocess.CompletedProcess[str]
Runner = Callable[[Sequence[str], Path], CommandResult]
VersionGetter = Callable[[str], str]
Printer = Callable[[str], object]


def _default_runner(command: Sequence[str], cwd: Path) -> CommandResult:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise UpgradeError(f"could not start {command[0]!r}: {exc}") from exc


def _run(
    runner: Runner,
    command: Sequence[str],
    cwd: Path,
    *,
    action: str,
    allowed: Iterable[int] = (0,),
) -> CommandResult:
    result = runner(tuple(str(part) for part in command), cwd)
    allowed_codes = frozenset(allowed)
    if result.returncode in allowed_codes:
        return result
    diagnostics = (result.stderr or result.stdout or "").strip()
    if not diagnostics:
        diagnostics = "(the command produced no diagnostics)"
    raise UpgradeError(
        f"{action} failed with exit code {result.returncode}:\n{diagnostics}"
    )


def _git(
    runner: Runner,
    repo_root: Path,
    arguments: Sequence[str],
    *,
    action: str,
    allowed: Iterable[int] = (0,),
) -> CommandResult:
    return _run(
        runner,
        ("git", *arguments),
        repo_root,
        action=action,
        allowed=allowed,
    )


def _git_text(
    runner: Runner,
    repo_root: Path,
    arguments: Sequence[str],
    *,
    action: str,
) -> str:
    return _git(
        runner, repo_root, arguments, action=action
    ).stdout.strip()


def _is_virtual_environment(
    prefix: Path,
    base_prefix: Path,
    *,
    real_prefix: str | None = None,
) -> bool:
    return prefix.resolve() != base_prefix.resolve() or real_prefix is not None


def _require_virtual_environment(
    prefix: Path,
    base_prefix: Path,
    *,
    real_prefix: str | None = None,
) -> None:
    if _is_virtual_environment(
        prefix, base_prefix, real_prefix=real_prefix
    ):
        return
    raise UpgradeError(
        "refusing to install packages into the system Python. Run the updater "
        "with Deltafin's existing environment, for example:\n"
        "  ./venv/bin/python tools/upgrade.py"
    )


def _require_project_files(repo_root: Path) -> tuple[Path, Path]:
    requirements = repo_root / "requirements.txt"
    builder = repo_root / "tools" / "build_native.py"
    missing = [
        str(path.relative_to(repo_root))
        for path in (requirements, builder)
        if not path.is_file()
    ]
    if missing:
        raise UpgradeError(
            "the checkout is missing required upgrade files: "
            + ", ".join(missing)
        )
    return requirements, builder


_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _require_no_torch_requirement(requirements: Path) -> None:
    """Keep platform-specific Torch out of the generic dependency update."""
    try:
        lines = requirements.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UpgradeError(f"could not read {requirements}: {exc}") from exc
    for line_number, original in enumerate(lines, 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            raise UpgradeError(
                f"{requirements.name}:{line_number} uses the unsupported "
                f"requirements directive {line!r}. Keep the upgrade manifest "
                "self-contained"
            )
        match = _REQUIREMENT_NAME.match(line)
        if match and match.group(1).lower().replace("_", "-") == "torch":
            raise UpgradeError(
                f"{requirements.name}:{line_number} attempts to manage Torch. "
                "Torch must remain host-selected, so the updater stopped"
            )


def _worktree_changes(
    runner: Runner,
    repo_root: Path,
) -> str:
    return _git_text(
        runner,
        repo_root,
        ("status", "--porcelain=v1", "--untracked-files=normal"),
        action="checking the Git worktree",
    )


def _require_clean_worktree(runner: Runner, repo_root: Path) -> None:
    changes = _worktree_changes(runner, repo_root)
    if not changes:
        return
    shown = changes.splitlines()
    if len(shown) > 20:
        shown = [*shown[:20], f"... and {len(shown) - 20} more"]
    raise UpgradeError(
        "the checkout has tracked, staged, or non-ignored untracked changes. "
        "The updater will not stash, discard, or overwrite them:\n"
        + "\n".join(f"  {line}" for line in shown)
    )


_PRESERVED_ROOTS = {
    ".cache",
    ".venv",
    "checkpoints",
    "k3-cache",
    "k3-experts",
    "k3-meta",
    "k3-model",
    "models",
    "venv",
    "weights",
}
_PRESERVED_SUFFIXES = {
    ".bf16",
    ".bin",
    ".blob",
    ".ckpt",
    ".f16",
    ".f32",
    ".fp16",
    ".fp32",
    ".gguf",
    ".i4",
    ".i8",
    ".mlmodelc",
    ".mlmodel",
    ".mlpackage",
    ".mxfp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".s4",
    ".safetensors",
    ".sc",
}


def _is_preserved_data_path(raw_path: str) -> bool:
    path = PurePosixPath(raw_path)
    if not path.parts:
        return False
    root = path.parts[0]
    if (
        root in _PRESERVED_ROOTS
        or root.startswith("k3-resident")
        or root.startswith("k3-draft")
        or root.startswith("k3-cache")
        or root.startswith("k3-experts")
        or root.startswith("k3-model")
    ):
        return True
    if raw_path == "tiktoken.model":
        return True
    if raw_path == "tools/expert_index.meta.json":
        return True
    if (
        len(path.parts) >= 3
        and path.parts[:2] == ("tools", "k3pkg")
        and path.name.startswith(("modeling_", "configuration_"))
    ):
        return True
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return bool(
        suffixes & _PRESERVED_SUFFIXES
        or any(re.fullmatch(r"\.[is][0-9]+", suffix) for suffix in suffixes)
    )


def _require_update_preserves_data(
    runner: Runner,
    repo_root: Path,
    head: str,
    upstream_head: str,
) -> None:
    result = _git(
        runner,
        repo_root,
        (
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            head,
            upstream_head,
            "--",
        ),
        action="checking the update for model/cache path collisions",
    )
    collisions = sorted(
        path
        for path in result.stdout.split("\0")
        if path and _is_preserved_data_path(path)
    )
    if collisions:
        raise UpgradeError(
            "the incoming update touches paths reserved for downloaded models "
            "or caches, so it was not applied:\n"
            + "\n".join(f"  {path}" for path in collisions)
        )


def _relationship(
    runner: Runner,
    repo_root: Path,
    head: str,
    upstream_head: str,
) -> str:
    if head == upstream_head:
        return "equal"
    head_is_ancestor = _git(
        runner,
        repo_root,
        ("merge-base", "--is-ancestor", head, upstream_head),
        action="comparing the local branch with its upstream",
        allowed=(0, 1),
    ).returncode == 0
    upstream_is_ancestor = _git(
        runner,
        repo_root,
        ("merge-base", "--is-ancestor", upstream_head, head),
        action="comparing the upstream with the local branch",
        allowed=(0, 1),
    ).returncode == 0
    if head_is_ancestor and not upstream_is_ancestor:
        return "behind"
    if upstream_is_ancestor and not head_is_ancestor:
        return "ahead"
    return "diverged"


def fetch_state(runner: Runner, repo_root: Path) -> GitState:
    """Validate, fetch, and describe the update without changing HEAD."""
    top_level = _git_text(
        runner,
        repo_root,
        ("rev-parse", "--show-toplevel"),
        action="locating the Git checkout",
    )
    if Path(top_level).resolve() != repo_root.resolve():
        raise UpgradeError(
            f"tools/upgrade.py belongs to {repo_root}, but Git reported "
            f"{top_level}; refusing to update the wrong checkout"
        )

    _require_clean_worktree(runner, repo_root)

    branch_result = _git(
        runner,
        repo_root,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        action="reading the current branch",
        allowed=(0, 1),
    )
    branch = branch_result.stdout.strip()
    if branch_result.returncode != 0 or not branch:
        raise UpgradeError(
            "the checkout is in detached-HEAD state. Switch to a normal branch "
            "with an upstream before upgrading"
        )

    upstream_result = _git(
        runner,
        repo_root,
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
        action="reading the branch upstream",
        allowed=(0, 1, 128),
    )
    upstream = upstream_result.stdout.strip()
    if upstream_result.returncode != 0 or not upstream:
        raise UpgradeError(
            f"branch {branch!r} has no upstream. Configure one explicitly "
            "(normally origin/main or origin/master) before upgrading"
        )

    remote = _git_text(
        runner,
        repo_root,
        ("config", "--get", f"branch.{branch}.remote"),
        action="reading the upstream remote",
    )
    if not remote or remote == ".":
        raise UpgradeError(
            f"branch {branch!r} does not track a normal remote upstream"
        )

    _git(
        runner,
        repo_root,
        ("fetch", "--no-tags", "--", remote),
        action=f"fetching {remote}",
    )
    # A fetch should not alter the worktree. This second check also closes the
    # obvious race between the first status check and the fast-forward.
    _require_clean_worktree(runner, repo_root)

    head = _git_text(
        runner,
        repo_root,
        ("rev-parse", "--verify", "HEAD"),
        action="reading the local revision",
    )
    upstream_head = _git_text(
        runner,
        repo_root,
        ("rev-parse", "--verify", "@{upstream}"),
        action="reading the fetched upstream revision",
    )
    relation = _relationship(
        runner, repo_root, head, upstream_head
    )
    if relation == "diverged":
        raise UpgradeError(
            f"{branch!r} and {upstream!r} have diverged. The updater only "
            "permits fast-forwards and will not merge, rebase, or reset"
        )
    if relation == "behind":
        _require_update_preserves_data(
            runner, repo_root, head, upstream_head
        )
    return GitState(
        branch=branch,
        upstream=upstream,
        remote=remote,
        head=head,
        upstream_head=upstream_head,
        relation=relation,
    )


def apply_fast_forward(
    runner: Runner,
    repo_root: Path,
    state: GitState,
) -> bool:
    if state.relation != "behind":
        return False
    _git(
        runner,
        repo_root,
        ("merge", "--ff-only", state.upstream_head),
        action=f"fast-forwarding {state.branch}",
    )
    observed = _git_text(
        runner,
        repo_root,
        ("rev-parse", "--verify", "HEAD"),
        action="verifying the fast-forward",
    )
    if observed != state.upstream_head:
        raise UpgradeError(
            "Git reported success but HEAD does not match the fetched upstream; "
            "dependency installation and native rebuilding were not attempted"
        )
    _require_clean_worktree(runner, repo_root)
    return True


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise UpgradeError(
            f"{distribution} is not installed in {sys.executable}. Install the "
            "correct host-specific PyTorch build before running the updater"
        ) from exc


def install_dependencies(
    runner: Runner,
    repo_root: Path,
    python_executable: Path,
    requirements: Path,
    *,
    version_getter: VersionGetter = _installed_version,
) -> str:
    """Install non-Torch dependencies while pinning the existing Torch build."""
    _require_no_torch_requirement(requirements)
    torch_before = version_getter("torch")
    with tempfile.TemporaryDirectory(prefix="deltafin-upgrade-") as directory:
        constraint = Path(directory) / "preserve-torch.txt"
        constraint.write_text(
            f"torch=={torch_before}\n",
            encoding="utf-8",
        )
        _run(
            runner,
            (
                str(python_executable),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--upgrade-strategy",
                "only-if-needed",
                "--constraint",
                str(constraint),
                "--requirement",
                str(requirements),
            ),
            repo_root,
            action="installing Deltafin dependencies",
        )
    torch_after = version_getter("torch")
    if torch_after != torch_before:
        raise UpgradeError(
            "the installed Torch version changed unexpectedly "
            f"({torch_before} -> {torch_after}). Stop before running Deltafin "
            "and restore the correct platform-specific build"
        )
    return torch_before


def require_pip(
    runner: Runner,
    repo_root: Path,
    python_executable: Path,
) -> None:
    if not python_executable.is_file():
        raise UpgradeError(
            f"the current Python executable does not exist: {python_executable}"
        )
    _run(
        runner,
        (str(python_executable), "-m", "pip", "--version"),
        repo_root,
        action="checking pip in the current virtual environment",
    )


def rebuild_native(
    runner: Runner,
    repo_root: Path,
    python_executable: Path,
    builder: Path,
) -> None:
    _run(
        runner,
        (str(python_executable), str(builder)),
        repo_root,
        action="building and validating native libraries",
    )


def upgrade(
    *,
    repo_root: Path = REPO_ROOT,
    runner: Runner = _default_runner,
    python_executable: Path = Path(sys.executable),
    prefix: Path = Path(sys.prefix),
    base_prefix: Path = Path(sys.base_prefix),
    real_prefix: str | None = getattr(sys, "real_prefix", None),
    version_getter: VersionGetter = _installed_version,
    printer: Printer = print,
) -> GitState:
    repo_root = repo_root.resolve()
    _require_virtual_environment(
        prefix, base_prefix, real_prefix=real_prefix
    )
    requirements, builder = _require_project_files(repo_root)
    _require_no_torch_requirement(requirements)
    # Fail before fetching if this environment cannot preserve its Torch build.
    torch_version = version_getter("torch")
    require_pip(runner, repo_root, python_executable)

    printer("Deltafin safe upgrade")
    printer(f"  repository: {repo_root}")
    printer(f"  environment: {python_executable}")
    printer(f"  preserving host-selected Torch {torch_version}")
    printer("  model weights and caches are not touched")

    state = fetch_state(runner, repo_root)
    if state.relation == "behind":
        printer(
            f"Fast-forwarding {state.branch} to {state.upstream_head[:12]}..."
        )
        apply_fast_forward(runner, repo_root, state)
        # Use the dependency manifest delivered by the new revision.
        requirements, builder = _require_project_files(repo_root)
    elif state.relation == "ahead":
        printer(
            f"{state.branch} is clean and locally ahead of {state.upstream}; "
            "leaving commits unchanged."
        )
    else:
        printer(f"{state.branch} is already current with {state.upstream}.")

    printer("Checking Deltafin's non-Torch Python dependencies...")
    preserved_torch = install_dependencies(
        runner,
        repo_root,
        python_executable,
        requirements,
        version_getter=version_getter,
    )
    printer(f"Rebuilding native libraries (Torch remains {preserved_torch})...")
    rebuild_native(
        runner, repo_root, python_executable, builder
    )
    printer("Upgrade complete. Downloaded models and caches remain in place.")
    return state


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Safely fast-forward Deltafin, refresh non-Torch dependencies, "
            "and transactionally rebuild native libraries"
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        upgrade(printer=lambda message: print(message, flush=True))
    except UpgradeError as exc:
        print(f"upgrade stopped safely: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
