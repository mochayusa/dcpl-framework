from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "configs" / "schemas" / "llm_pilot.yaml"


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    targets: List[str]
    blocks: Dict[str, List[str]]
    aliases: Dict[str, List[str]]
    categorical_levels: Dict[str, List[str]]

    def block_columns(self, block_name: str) -> List[str]:
        try:
            return list(self.blocks[block_name])
        except KeyError as exc:
            raise KeyError(f"Unknown block '{block_name}' in schema '{self.name}'") from exc

    def alias_map(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self.aliases.items()}

    def categories_for(self, column: str) -> List[str]:
        return list(self.categorical_levels.get(column, []))


def resolve_schema_path(schema: str | Path | None = None) -> Path:
    path = DEFAULT_SCHEMA_PATH if schema is None else Path(schema)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@lru_cache(maxsize=None)
def _load_schema_cached(schema_path: str) -> DatasetSchema:
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    blocks = {
        str(name): [str(col) for col in cols or []]
        for name, cols in (raw.get("blocks", {}) or {}).items()
    }
    aliases = {
        str(canonical): (
            [str(v) for v in values]
            if isinstance(values, list)
            else [str(values)]
        )
        for canonical, values in (raw.get("aliases", {}) or {}).items()
    }
    categorical_levels = {
        str(name): [str(v) for v in values or []]
        for name, values in (raw.get("categorical_levels", {}) or {}).items()
    }

    return DatasetSchema(
        name=str(raw.get("name", path.stem)),
        targets=[str(v) for v in raw.get("targets", [])],
        blocks=blocks,
        aliases=aliases,
        categorical_levels=categorical_levels,
    )


def load_schema(schema: str | Path | None = None) -> DatasetSchema:
    return _load_schema_cached(str(resolve_schema_path(schema)))
