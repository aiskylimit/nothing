"""Cheap, CPU-only structural checks for project scripts and output paths."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = sorted(REPO_ROOT.glob("scripts/**/*.sh"))


def test_project_commands_sh_exists_at_repo_root():
    assert (REPO_ROOT / "project_commands.sh").is_file()


def test_project_commands_only_runs_project_scripts():
    text = (REPO_ROOT / "project_commands.sh").read_text(encoding="utf-8")
    commands = [line.strip() for line in text.splitlines() if "bash \"${BASE_PATH}/scripts/" in line]
    assert commands
    assert all("/scripts/" in command for command in commands)


def test_project_commands_sh_is_valid_bash():
    subprocess.run(["bash", "-n", str(REPO_ROOT / "project_commands.sh")], check=True)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_no_output_path_inside_the_code_folder(script: Path):
    """`--output_dir outputs/...` would write checkpoints into the repo, which the
    runner forbids; every default must be rooted at ${PVSD_*_ROOT} or ~/."""
    text = script.read_text(encoding="utf-8")
    for match in re.finditer(r"--(?:output_dir|output|train-output|eval-output)\s+(\S+)", text):
        value = match.group(1).strip('"')
        assert value.startswith(("${", "~/", "$HOME", "/")), f"{script.name}: {value}"


def test_scripts_source_the_shared_paths_file():
    """Anything that trains or evaluates needs PVSD_*_ROOT and HF_HOME defined."""
    for name in (
        "scripts/math/train_pvsd_qwen3_4b.sh",
        "scripts/math/eval_math.sh",
        "scripts/math/probe_pvsd.sh",
        "scripts/math/train_avsd_qwen3_8b.sh",
    ):
        assert "scripts/remote/paths.sh" in (REPO_ROOT / name).read_text(encoding="utf-8"), name


def test_no_stale_avsd_module_imports():
    """The package is `pvsd`; `from avsd.` / `-m avsd.` would fail at import time."""
    stale = []
    for path in list(REPO_ROOT.glob("src/**/*.py")) + list(REPO_ROOT.glob("tests/*.py")) + SHELL_SCRIPTS + list(
        REPO_ROOT.glob("scripts/**/*.py")
    ):
        if path.resolve() == Path(__file__).resolve():
            continue  # this file contains the pattern it searches for
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bfrom avsd\.|\bimport avsd\b|-m avsd\.", text):
            stale.append(str(path.relative_to(REPO_ROOT)))
    assert not stale, stale


def test_gitignore_blocks_the_runner_instructions_and_large_outputs():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for entry in ("GUIDE.md", "outputs/", "results/", "_run_log_/", "*.safetensors"):
        assert entry in ignored, entry
