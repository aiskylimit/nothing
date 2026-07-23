from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSet:
    selector_system: str
    selector_user: str
    decomposer_system: str
    decomposer_user: str
    refiner_system: str
    refiner_user: str


PROMPT_FILES = {
    "selector_system": "selector_system.txt",
    "selector_user": "selector_user.txt",
    "decomposer_system": "decomposer_system.txt",
    "decomposer_user": "decomposer_user.txt",
    "refiner_system": "refiner_system.txt",
    "refiner_user": "refiner_user.txt",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_prompt_set(prompt_dir: Path) -> PromptSet:
    missing = [name for name in PROMPT_FILES.values() if not (prompt_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prompt files under {prompt_dir}: {', '.join(missing)}")
    values = {
        key: load_text(prompt_dir / filename)
        for key, filename in PROMPT_FILES.items()
    }
    return PromptSet(**values)

