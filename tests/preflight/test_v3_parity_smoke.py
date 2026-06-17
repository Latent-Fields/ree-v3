"""Preflight: V3-parity smoke.

Extends the runner's regression-suite preflight layer with the default-path
select_action smoke. If a higher-version (V4/V5) change has leaked into V3
default behaviour -- or a cross-checkout skew leaves a V4 call-site passing a
kwarg an older shared module lacks -- this fails the preflight and the runner
exits non-zero before claiming any experiment (honoured via --skip-preflight /
REE_SKIP_PREFLIGHT=1).

Motivating incident: 2026-06-17 V3-EXQ-654e (DR-12 unconditional call-site
skew crash-burned a V3 critical-path experiment). See ree_core/version_layering.py
and REE_assembly/docs/architecture/version_layering_doctrine.md.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.v3_parity_smoke import run_v3_parity_smoke  # noqa: E402


def test_v3_default_path_select_action_runs():
    # Raises on any failure (TypeError on a skewed call-site, AssertionError on a
    # version-layering violation, construction crash, etc.).
    run_v3_parity_smoke(steps=3)
