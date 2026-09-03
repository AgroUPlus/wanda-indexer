"""Minimal .env loading, so credentials never live in a committed launcher.

Deliberately dependency-free (no python-dotenv): the indexer has to run on a
stock Python install with nothing but numpy available.
"""
import os
from typing import List

ENV_FILENAME = ".env"


def candidate_paths(explicit: str = "") -> List[str]:
    if explicit:
        return [explicit]
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return [
        os.path.join(os.getcwd(), ENV_FILENAME),
        os.path.join(project_root, ENV_FILENAME),
    ]


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(explicit: str = "", override: bool = False) -> str:
    """Loads KEY=VALUE lines into os.environ. Returns the file used, or ""."""
    for path in candidate_paths(explicit):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            continue

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = _strip_quotes(value.strip())
        return path
    return ""
