"""Contracts for the inert `salience_apply_to_dacc_bias` lint (MECH-244).

Surfaces under test:
  (1) validate_experiments.inert_salience_dacc_bias_lint -- flags a config construction
      that passes a literal `salience_apply_to_dacc_bias=True` while setting no positive
      `dacc_weight`, so the dACC->E3 behavioural channel is arithmetically inert.
  (2) validate_experiments.py --checks inert_salience_dacc_bias -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).
  (3) The SUBSTRATE FACT the gate rests on, asserted against ree_core rather than trusted:
      `dacc_weight` defaults to 0.0 and `DACCtoE3Adapter.forward` multiplies the whole
      bias by it, so with the flag on and no weight the write-gate scales the zero vector.

WHY THIS GATE EXISTS. `REEConfig.salience_apply_to_dacc_bias=True` makes agent.py scale
the dACC->E3 score bias by `write_gate("e3_policy")` -- it reads as "the MECH-261
write-gate now modulates action selection". But the bias comes from
`DACCtoE3Adapter.forward`, which multiplies the entire bias by `self.config.dacc_weight`,
and that (with every per-candidate sub-weight) defaults to 0.0. So the flag WITHOUT a
positive `dacc_weight` multiplies the ZERO vector: no error, and any arm resting on that
channel is a guaranteed null that looks measured. Same family as the
from_dims-swallows-unknown-kwargs hazard -- a flag necessary but not sufficient.

CONFIRMED NEAR-MISS (V3-EXQ-799, 2026-07). Authored with the flag True and no
dacc_weight; a P0 probe measured `dacc_bias` cross-candidate range = 0.0 in all four arms.
The driver added `dacc_weight=DACC_WEIGHT` + `dacc_interaction_weight` to make the channel
live. That fixed shape -- flag True WITH a positive dacc_weight -- is the reference and
must NOT fire; `test_confirmed_799_correct_shape_is_quiet` pins it against the real driver.

AST-CALL-KEYWORD DETECTION is the discriminator, and it is deliberately blind to a flag
that only appears in a docstring. V3-EXQ-455a lists `salience_apply_to_dacc_bias=True` in
its Arms docstring but is a gated `NotImplementedError` stub that constructs no config, so
it correctly does not fire (its inertness is stronger than this gate targets).

SCOPE. This gates NEW scripts, and stays WARN-only: a landed carrier's run is complete and
its pre-registered emission is not rewritten to chase a lint, and a `dacc_weight` assembled
at runtime is invisible to the static scan (acceptable under-fire for an advisory net).
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# --------------------------------------------------------------------------------------
# (1) the substrate fact the gate rests on -- asserted, not trusted
# --------------------------------------------------------------------------------------

def test_dacc_weight_defaults_zero_and_scales_the_whole_bias():
    """`dacc_weight` defaults to 0.0 and multiplies the entire dACC->E3 bias.

    If either fact ever changes -- a non-zero default, or the bias no longer gated on
    dacc_weight -- the flag would no longer be inert without a weight, and this gate would
    be flagging a non-problem.
    """
    dacc = (REPO_ROOT / "ree_core" / "cingulate" / "dacc.py").read_text(encoding="utf-8")
    tree = ast.parse(dacc)

    # dacc_weight is a dataclass-style field defaulting to 0.0.
    defaults = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            if isinstance(n.value, ast.Constant):
                defaults[n.target.id] = n.value.value
    assert defaults.get("dacc_weight") == 0.0, (
        "dacc_weight no longer defaults to 0.0 -- the inert-channel premise has changed.")
    for sub in ("dacc_interaction_weight", "dacc_foraging_weight", "dacc_suppression_weight"):
        assert defaults.get(sub) == 0.0, f"{sub} no longer defaults to 0.0"

    # forward() multiplies the bias by dacc_weight -- so weight 0 => zero bias.
    assert "self.config.dacc_weight * float(bundle" in dacc, (
        "DACCtoE3Adapter.forward no longer scales the bias by dacc_weight -- re-derive "
        "the gate's premise.")


def test_flag_exists_on_reeconfig():
    """The knob the lint keys on is a real REEConfig field, not a phantom."""
    cfg = (REPO_ROOT / "ree_core" / "utils" / "config.py").read_text(encoding="utf-8")
    assert "salience_apply_to_dacc_bias" in cfg, (
        "salience_apply_to_dacc_bias is no longer a REEConfig field -- the lint keys on a "
        "knob that no longer exists.")


# --------------------------------------------------------------------------------------
# (2) positive / negative shapes
# --------------------------------------------------------------------------------------

_INERT = """
    from ree_core.utils.config import REEConfig

    def make(env):
        return REEConfig.from_dims(
            env.dims,
            use_dacc=True,
            salience_apply_to_dacc_bias=True,
        )
"""


def test_fires_on_flag_true_without_dacc_weight(tmp_path):
    """The defect shape: flag on, no dacc_weight, so the channel scales the zero vector."""
    msg = V.inert_salience_dacc_bias_lint(_write(tmp_path, _INERT))
    assert msg is not None
    assert "salience_apply_to_dacc_bias=True" in msg
    assert "dacc_weight" in msg


def test_quiet_when_dacc_weight_positive_same_call(tmp_path):
    """The V3-EXQ-799 fix: a positive dacc_weight in the same call clears it."""
    fixed = _INERT.replace(
        "salience_apply_to_dacc_bias=True,",
        "dacc_weight=1.0,\n            dacc_interaction_weight=0.3,\n"
        "            salience_apply_to_dacc_bias=True,")
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, fixed)) is None


def test_quiet_when_dacc_weight_is_a_module_constant(tmp_path):
    """799 uses `dacc_weight=DACC_WEIGHT` -- a non-literal is assumed set-and-positive."""
    fixed = _INERT.replace(
        "salience_apply_to_dacc_bias=True,",
        "dacc_weight=DACC_WEIGHT,\n            salience_apply_to_dacc_bias=True,")
    fixed = "DACC_WEIGHT = 0.6\n" + fixed
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, fixed)) is None


def test_fires_on_explicit_dacc_weight_zero(tmp_path):
    """An explicit `dacc_weight=0.0` is still inert -- the literal-zero case must fire."""
    zeroed = _INERT.replace(
        "salience_apply_to_dacc_bias=True,",
        "dacc_weight=0.0,\n            salience_apply_to_dacc_bias=True,")
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, zeroed)) is not None


def test_quiet_when_flag_false(tmp_path):
    """Observer-mode arms set the flag False -- never a fire (446/453/455 shape)."""
    off = _INERT.replace("salience_apply_to_dacc_bias=True,",
                         "salience_apply_to_dacc_bias=False,")
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, off)) is None


def test_quiet_when_dacc_weight_set_separately(tmp_path):
    """The weight-set-separately escape: `cfg.dacc_weight = 0.5` after construction."""
    src = _INERT + """
    def wire(cfg):
        cfg.dacc_weight = 0.5
        return cfg
"""
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, src)) is None


def test_quiet_on_docstring_only_mention(tmp_path):
    """A flag named ONLY in a docstring is not a config call -- the V3-EXQ-455a stub shape.

    455a lists `salience_apply_to_dacc_bias=True` in its Arms docstring but raises
    NotImplementedError and constructs no config. AST-keyword detection must not fire on
    prose.
    """
    src = '''
    """Arms: COORD_ON uses salience_apply_to_dacc_bias=True, no dacc_weight."""
    def main():
        raise NotImplementedError("gated")
'''
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, src)) is None


def test_exempt_marker_suppresses(tmp_path):
    src = 'INERT_SALIENCE_DACC_BIAS_EXEMPT = "bias driven from another head"\n' + _INERT
    assert V.inert_salience_dacc_bias_lint(_write(tmp_path, src)) is None


def test_unparseable_file_is_not_an_error(tmp_path):
    p = tmp_path / "v3_exq_999_broken.py"
    p.write_text("def (: oops\n", encoding="utf-8")
    assert V.inert_salience_dacc_bias_lint(p) is None


# --------------------------------------------------------------------------------------
# (3) the confirmed near-miss, as a differential against the real driver
# --------------------------------------------------------------------------------------

def test_confirmed_799_correct_shape_is_quiet():
    """V3-EXQ-799 enables the flag WITH a positive dacc_weight -- it must NOT fire.

    This is the driver that caught the defect live and then fixed it correctly; it is the
    reference shape. If the lint fires on it, the dacc_weight-clears-it path is broken.
    """
    good = EXPERIMENTS_DIR / "v3_exq_799_mech048_stability_temperature_behavioural_did.py"
    if not good.exists():
        pytest.skip("V3-EXQ-799 not present in this checkout")
    assert V.inert_salience_dacc_bias_lint(good) is None, (
        "V3-EXQ-799 sets dacc_weight=DACC_WEIGHT alongside the flag -- the lint must treat "
        "the channel as live and stay quiet.")


def test_gated_stub_455a_is_quiet():
    """V3-EXQ-455a lists the flag in its docstring but constructs no config -- quiet."""
    stub = EXPERIMENTS_DIR / "v3_exq_455a_sd032a_salience_with_vs.py"
    if not stub.exists():
        pytest.skip("V3-EXQ-455a not present in this checkout")
    assert V.inert_salience_dacc_bias_lint(stub) is None


# --------------------------------------------------------------------------------------
# (4) selector + WARN-only invariant
# --------------------------------------------------------------------------------------

def test_check_is_registered():
    assert "inert_salience_dacc_bias" in V.CHECK_NAMES


def test_gate_is_warn_only_even_under_strict(tmp_path):
    """Never hardens: not under --strict, not under --paths."""
    script = _write(tmp_path, _INERT)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"),
         "--strict", "--checks", "inert_salience_dacc_bias", "--paths", str(script)],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, (
        "inert_salience_dacc_bias must be WARN-only in BOTH modes; it changed the exit "
        f"code.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "INERT-SALIENCE-DACC_BIAS WARNINGS" in proc.stdout
    assert "salience_apply_to_dacc_bias=True" in proc.stdout


# --------------------------------------------------------------------------------------
# (5) pinned corpus fire count
# --------------------------------------------------------------------------------------
# 0 as of the 2026-07-28 audit that introduced this gate: the only corpus drivers passing
# salience_apply_to_dacc_bias=True are V3-EXQ-799 (which sets dacc_weight, correct shape)
# and V3-EXQ-455a (a docstring-only mention in a gated NotImplementedError stub). Every
# other carrier sets the flag False. This is a PREVENTIVE gate for future authoring -- a
# fire means a NEW driver enabled the flag without a weight and should be fixed at
# authoring time (set dacc_weight > 0, or add INERT_SALIENCE_DACC_BIAS_EXEMPT), not
# re-pinned. If the count rises because a genuine inert carrier LANDED, investigate that
# carrier before re-pinning.
_PINNED_CORPUS_FIRE_COUNT = 0


def test_inert_dacc_bias_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`).

    Per that module's standing pattern, a new corpus-wide lint goes in `path_lints` and
    its corpus test takes `corpus_scan` rather than enumerating `experiments/` itself.
    """
    fired = corpus_scan["inert_salience_dacc_bias_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"inert-salience-dacc_bias fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(set dacc_weight > 0 alongside the flag -- V3-EXQ-799 is the canonical shape -- "
        f"or add INERT_SALIENCE_DACC_BIAS_EXEMPT) rather than re-pinning.\nfired:\n  "
        + "\n  ".join(p.name for p in fired))
