"""Shared pytest config for the regression suite.

Adds ree-v3 repo root to sys.path so `from ree_core import ...` works
regardless of pytest invocation cwd.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
