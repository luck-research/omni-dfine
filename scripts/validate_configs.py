"""Validate all Omni-DFINE YAML configs parse cleanly and that __include__
files resolve relative to each config.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _resolve_includes(cfg_path: Path, seen: set | None = None) -> None:
    """Recursively load __include__ entries (relative to the config file)."""
    if seen is None:
        seen = set()
    real = cfg_path.resolve()
    if real in seen:
        return
    seen.add(real)
    text = cfg_path.read_text()
    cfg = yaml.safe_load(text)
    assert isinstance(cfg, dict), f"{cfg_path} top-level is not a dict"
    includes = cfg.get("__include__")
    if includes is None:
        return
    if isinstance(includes, str):
        includes = [includes]
    for inc in includes:
        inc_path = (cfg_path.parent / inc).resolve()
        print(f"  includes: {cfg_path.name} -> {inc}")
        assert inc_path.exists(), f"missing include target: {inc_path}"
        _resolve_includes(inc_path, seen)


def main() -> int:
    print(f"Validating configs under {CONFIG_ROOT}\n")
    cfg_files = sorted(CONFIG_ROOT.rglob("*.yml"))
    if not cfg_files:
        print("No YAML configs found.")
        return 1

    errors = 0
    for cfg_path in cfg_files:
        try:
            text = cfg_path.read_text()
            cfg = yaml.safe_load(text)
            assert isinstance(cfg, dict), "top-level YAML must be a dict"
            print(f"OK   {cfg_path.relative_to(CONFIG_ROOT)}  ({len(text)} bytes)")
            # Resolve and validate include chain.
            _resolve_includes(cfg_path)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAIL {cfg_path.relative_to(CONFIG_ROOT)} -- {exc}")
            errors += 1

    print()
    if errors == 0:
        print(f"All {len(cfg_files)} configs are valid.")
        return 0
    print(f"{errors} config(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
