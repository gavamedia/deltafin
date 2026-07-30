#!/usr/bin/env python3
"""Offline tests for the conservative Deltafin updater."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import upgrade


def completed(
    command: tuple[str, ...],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command, returncode, stdout=stdout, stderr=stderr
    )


class FakeRunner:
    def __init__(
        self,
        repo_root: Path,
        *,
        relation: str = "behind",
        dirty: str = "",
        incoming_paths: tuple[str, ...] = (),
        detached: bool = False,
        has_upstream: bool = True,
    ) -> None:
        self.repo_root = repo_root
        self.relation = relation
        self.dirty = dirty
        self.incoming_paths = incoming_paths
        self.detached = detached
        self.has_upstream = has_upstream
        self.commands: list[tuple[str, ...]] = []
        self.merged = False
        self.torch_constraint: str | None = None

    def __call__(
        self, command: tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:2] == ("git", "rev-parse"):
            arguments = command[2:]
            if arguments == ("--show-toplevel",):
                return completed(command, stdout=f"{self.repo_root}\n")
            if arguments == (
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ):
                if not self.has_upstream:
                    return completed(
                        command, returncode=128, stderr="no upstream"
                    )
                return completed(command, stdout="origin/main\n")
            if arguments == ("--verify", "HEAD"):
                if self.merged or self.relation == "ahead":
                    value = "new"
                elif self.relation == "diverged":
                    value = "local"
                else:
                    value = "old"
                return completed(command, stdout=f"{value}\n")
            if arguments == ("--verify", "@{upstream}"):
                if self.relation == "behind":
                    upstream = "new"
                elif self.relation == "diverged":
                    upstream = "remote"
                else:
                    upstream = "old"
                return completed(command, stdout=f"{upstream}\n")
        if command[:2] == ("git", "status"):
            return completed(command, stdout=self.dirty)
        if command[:2] == ("git", "symbolic-ref"):
            if self.detached:
                return completed(command, returncode=1)
            return completed(command, stdout="main\n")
        if command[:3] == ("git", "config", "--get"):
            return completed(command, stdout="origin\n")
        if command[:2] == ("git", "fetch"):
            return completed(command)
        if command[:3] == ("git", "merge-base", "--is-ancestor"):
            left, right = command[-2:]
            if self.relation == "behind":
                is_ancestor = (left, right) == ("old", "new")
                return completed(command, returncode=0 if is_ancestor else 1)
            if self.relation == "ahead":
                is_ancestor = (left, right) == ("old", "new")
                return completed(command, returncode=0 if is_ancestor else 1)
            if self.relation == "diverged":
                return completed(command, returncode=1)
            return completed(command)
        if command[:3] == ("git", "diff", "--name-only"):
            encoded = "\0".join(self.incoming_paths)
            if encoded:
                encoded += "\0"
            return completed(command, stdout=encoded)
        if command[:3] == ("git", "merge", "--ff-only"):
            self.merged = True
            return completed(command)
        if len(command) >= 4 and command[1:4] == ("-m", "pip", "install"):
            constraint_index = command.index("--constraint") + 1
            self.torch_constraint = Path(
                command[constraint_index]
            ).read_text(encoding="utf-8")
            return completed(command)
        if len(command) == 4 and command[1:] == ("-m", "pip", "--version"):
            return completed(command, stdout="pip fixture\n")
        if len(command) == 2 and command[1].endswith("build_native.py"):
            return completed(command)
        raise AssertionError(f"unexpected command: {command}")


class VersionSequence:
    def __init__(self, *versions: str) -> None:
        self.versions = list(versions)

    def __call__(self, distribution: str) -> str:
        if distribution != "torch":
            raise AssertionError(distribution)
        if not self.versions:
            raise AssertionError("too many Torch version checks")
        return self.versions.pop(0)


class UpgradeTests(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "tools").mkdir()
        (root / "tools" / "build_native.py").write_text(
            "# fixture\n", encoding="utf-8"
        )
        (root / "requirements.txt").write_text(
            "numpy\ntransformers==4.56.2\n", encoding="utf-8"
        )
        return temporary, root

    def run_upgrade(
        self,
        root: Path,
        runner: FakeRunner,
        versions: VersionSequence,
    ) -> upgrade.GitState:
        output: list[str] = []
        return upgrade.upgrade(
            repo_root=root,
            runner=runner,
            python_executable=Path(sys.executable),
            prefix=root / "venv",
            base_prefix=root / "base-python",
            real_prefix=None,
            version_getter=versions,
            printer=output.append,
        )

    def test_clean_behind_fast_forwards_then_maintains_environment(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        sentinel = root / "k3-resident-int8" / "tensors" / "weight.i8"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"downloaded-model-data")
        runner = FakeRunner(root, relation="behind")

        state = self.run_upgrade(
            root,
            runner,
            VersionSequence("2.7.1+cu128", "2.7.1+cu128", "2.7.1+cu128"),
        )

        self.assertEqual(state.relation, "behind")
        self.assertTrue(runner.merged)
        self.assertEqual(sentinel.read_bytes(), b"downloaded-model-data")
        commands = runner.commands
        merge_index = next(
            i for i, command in enumerate(commands)
            if command[:3] == ("git", "merge", "--ff-only")
        )
        pip_index = next(
            i for i, command in enumerate(commands)
            if len(command) >= 4 and command[1:4] == ("-m", "pip", "install")
        )
        build_index = next(
            i for i, command in enumerate(commands)
            if len(command) == 2 and command[1].endswith("build_native.py")
        )
        self.assertLess(merge_index, pip_index)
        self.assertLess(pip_index, build_index)
        self.assertEqual(runner.torch_constraint, "torch==2.7.1+cu128\n")
        flattened = " ".join(" ".join(command) for command in commands)
        self.assertNotIn(" reset ", f" {flattened} ")
        self.assertNotIn(" clean ", f" {flattened} ")
        self.assertNotIn("setup_k3.py", flattened)

    def test_dirty_worktree_stops_before_fetch_or_install(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root, dirty=" M tools/kimi_run.py\n")

        with self.assertRaisesRegex(
            upgrade.UpgradeError, "will not stash, discard, or overwrite"
        ):
            self.run_upgrade(root, runner, VersionSequence("2.7.1"))

        self.assertFalse(
            any(command[:2] == ("git", "fetch") for command in runner.commands)
        )
        self.assertFalse(
            any(
                len(command) >= 4
                and command[1:4] == ("-m", "pip", "install")
                for command in runner.commands
            )
        )

    def test_diverged_branch_is_refused_without_mutation(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root, relation="diverged")

        with self.assertRaisesRegex(upgrade.UpgradeError, "have diverged"):
            self.run_upgrade(root, runner, VersionSequence("2.7.1"))

        self.assertFalse(runner.merged)
        self.assertFalse(
            any(
                len(command) >= 4
                and command[1:4] == ("-m", "pip", "install")
                for command in runner.commands
            )
        )

    def test_detached_head_is_refused_before_fetch(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root, detached=True)

        with self.assertRaisesRegex(upgrade.UpgradeError, "detached-HEAD"):
            self.run_upgrade(root, runner, VersionSequence("2.7.1"))

        self.assertFalse(
            any(command[:2] == ("git", "fetch") for command in runner.commands)
        )

    def test_branch_without_upstream_is_refused_before_fetch(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root, has_upstream=False)

        with self.assertRaisesRegex(upgrade.UpgradeError, "has no upstream"):
            self.run_upgrade(root, runner, VersionSequence("2.7.1"))

        self.assertFalse(
            any(command[:2] == ("git", "fetch") for command in runner.commands)
        )

    def test_clean_local_ahead_is_left_unchanged_but_rebuilt(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root, relation="ahead")

        state = self.run_upgrade(
            root,
            runner,
            VersionSequence("2.7.1", "2.7.1", "2.7.1"),
        )

        self.assertEqual(state.relation, "ahead")
        self.assertFalse(runner.merged)
        self.assertTrue(
            any(
                len(command) == 2
                and command[1].endswith("build_native.py")
                for command in runner.commands
            )
        )

    def test_incoming_model_or_cache_collision_is_refused(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(
            root,
            relation="behind",
            incoming_paths=("tools/kimi_run.py", "k3-resident-int8/weights.i8"),
        )

        with self.assertRaisesRegex(
            upgrade.UpgradeError, "reserved for downloaded models or caches"
        ):
            self.run_upgrade(root, runner, VersionSequence("2.7.1"))

        self.assertFalse(runner.merged)

    def test_torch_in_requirements_is_rejected_before_fetch(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        (root / "requirements.txt").write_text(
            "numpy\ntorch>=2\n", encoding="utf-8"
        )
        runner = FakeRunner(root)

        with self.assertRaisesRegex(
            upgrade.UpgradeError, "attempts to manage Torch"
        ):
            self.run_upgrade(root, runner, VersionSequence("2.7.1"))

        self.assertEqual(runner.commands, [])

    def test_system_python_is_refused_before_git(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root)

        with self.assertRaisesRegex(
            upgrade.UpgradeError, "system Python"
        ):
            upgrade.upgrade(
                repo_root=root,
                runner=runner,
                python_executable=Path(sys.executable),
                prefix=root / "same",
                base_prefix=root / "same",
                real_prefix=None,
                version_getter=VersionSequence("2.7.1"),
                printer=lambda _: None,
            )

        self.assertEqual(runner.commands, [])

    def test_changed_torch_version_is_detected(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        runner = FakeRunner(root, relation="equal")

        with self.assertRaisesRegex(
            upgrade.UpgradeError, "changed unexpectedly"
        ):
            self.run_upgrade(
                root,
                runner,
                VersionSequence("2.7.1+cu128", "2.7.1+cu128", "2.8.0"),
            )

        self.assertFalse(
            any(
                len(command) == 2
                and command[1].endswith("build_native.py")
                for command in runner.commands
            )
        )

    def test_preserved_path_classifier_covers_weights_and_caches(self) -> None:
        protected = (
            "k3-resident-mix46/tensors/embed.weight.i6",
            "k3-draft-qwen3-0.6b-base/model.safetensors",
            "k3-cache-raw/expert.npz",
            "models/Kimi-K3/file",
            "nested/expert.safetensors",
            "nested/expert.weight.i6",
            "tools/k3pkg/modeling_kimi_k3.py",
            "venv/bin/python",
            "tiktoken.model",
        )
        for path in protected:
            with self.subTest(path=path):
                self.assertTrue(upgrade._is_preserved_data_path(path))
        self.assertFalse(upgrade._is_preserved_data_path("tools/kimi_run.py"))


if __name__ == "__main__":
    unittest.main()
