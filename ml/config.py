"""Paths and YAML configuration."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def resolve_path(value: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def artifacts_dir() -> Path:
    return resolve_path(os.environ.get("ARTIFACTS_DIR", "artifacts"))


def get_data_root() -> Path:
    value = os.environ.get("DATA_ROOT")
    if not value:
        raise SystemExit(
            "DATA_ROOT is not set. Set it in wafer-inspection/.env or the shell, e.g.\n"
            '  $env:DATA_ROOT = "D:\\...\\data-carinthia-s\\extracted\\data"\n'
            "It must contain images/, masks/ and carinthia-s.csv."
        )
    root = Path(value)
    if not root.is_dir():
        raise SystemExit(f"DATA_ROOT does not exist or is not a directory: {root}")
    return root


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*names: str | os.PathLike) -> dict:
    """Merge configs/common.yaml with the given configs (names or paths), in order."""
    cfg: dict = {}
    for name in ("common.yaml", *map(str, names)):
        p = Path(name)
        if not p.is_file():
            p = CONFIG_DIR / name
        if not p.is_file():
            raise SystemExit(f"Config file not found: {name}")
        cfg = _deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg


def split_dir(cfg: dict) -> Path:
    return artifacts_dir() / "splits" / cfg["split"]["id"]


def run_dir(cfg: dict, smoke: bool = False) -> Path:
    """Logs, histories, summaries and evaluation results (no weights)."""
    base = artifacts_dir() / ("smoke" if smoke else "runs")
    return base / cfg["name"]


def weights_root() -> Path:
    return resolve_path(os.environ.get("WEIGHTS_DIR", "weight"))


def weights_dir(cfg: dict) -> Path:
    """Where trained checkpoints live: weight/<model>/{best.pt, latest.pt, resume_state.pt}."""
    return weights_root() / cfg["name"]
