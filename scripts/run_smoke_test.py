#!/usr/bin/env python
"""Convenience runner for the Omni-DFINE smoke tests.

Usage
-----
    python scripts/run_smoke_test.py
    # or, from the project root:
    python -m tests.test_smoke
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Make the `omni_dfine` package importable regardless of the cwd.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from tests.test_smoke import main as _main

    return _main()


if __name__ == "__main__":
    sys.exit(main())
