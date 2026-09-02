"""MECH-530: E3's decision-outcome output surface is typed, not category-labelled.

WHAT THIS PINS. MECH-530 (claims.yaml, claim_type: design_decision, subject
e3.non_oracular_output_typing) asserts that every field of E3's
decision-outcome output surface (SelectionResult, harm_eval/benefit_eval,
last_score_diagnostics) is a continuous, boolean, or count-typed value, and
that NONE of them is a semantic category-label string or enum naming a
content/object/context class. This is what keeps E3 from becoming an oracle
that hands a downstream consumer (agent.py today; a future MECH-529-style
rebucketing mechanism tomorrow) a ready-made semantic verdict instead of a
typed magnitude it must still interpret.

This test was NOT YET WRITTEN when MECH-530 was registered (2026-09-01) --
the claim held only by manual inspection. Writing it required reconciling
the claim's own text against the live surface:

RECONCILIATION FINDING (surfaced by this test, not assumed away). MECH-530's
verification note claims "a repo-wide grep for category_label/oracular ...
returned zero hits. No category-label-shaped field exists anywhere on E3's
current output surface." That is not quite true: `last_score_diagnostics`
carries THREE STRING-valued fields today --
  - "commit_gate_mode": "harm_score_variance" | "world_variance"
    (e3_selector.py ~3706, added 2026-07-18, predates MECH-530's registration)
  - "modulatory_shortlist_mode": "margin" | "top_k" | "f_demotion"
    (e3_selector.py ~3980 / pre-seeded ~3380, added 2026-06-16)
  - "modulatory_authority_normalize_basis": "range" | "std"
    (e3_selector.py ~3378, added 2026-06-15)
None of the three NAMES A CONTENT CLASS (none ever describes a candidate's
semantic category -- "predator", "food", "safe route"). All three are
INTERNAL-MODE SELECTORS: which of E3's own commit-gate formulas,
shortlist-construction strategies, or authority-rescaling bases was used on
this tick, and two of the three are direct echoes of an operator-set config
knob rather than a runtime perceptual judgement. That is architecturally the
same kind of thing as `loop_named_channel_routed_ranges`' dict KEYS
(architectural loop names like "associative"/"limbic") -- already implicitly
permitted by the claim's own "keyed by descriptive strings" language -- just
surfaced as a VALUE instead of a KEY. So this test treats all three as
legitimate under the claim's actual intent (no semantic content-class leak)
rather than papering over the discrepancy or failing on landing for a
property that predates the claim. All three are captured in a closed CITED
allowlist below, not waved through open-endedly: a change to any field's
value set, or any FOURTH string-valued field appearing anywhere on the
surface, fails this test by name. This finding is also raised as a
governance_flag (evidence_discrepancy) against MECH-530 so /governance can
decide whether to correct the claim's verification note.

METHOD.
  (1) RUNTIME: drive a real E3TrajectorySelector.select() call through a
      scenario built to exercise the surface widely (loop segregation +
      finer-channel routing + modulatory shortlist + score decomposition
      diagnostics -- the same combination the sibling
      test_e3_last_scores_post_arbitration.py loop-segregation scenario
      uses), then walk EVERY key of the resulting last_score_diagnostics
      dict and EVERY field of the returned SelectionResult, asserting each
      leaf value's type against the permitted-type policy. Because this
      walks the dict as populated (not a fixed expected-key list), it
      transparently covers whatever the ~77 write sites in e3_selector.py
      happen to produce on this path, and would catch a NEW field added to
      an already-exercised code path without needing this test edited.
  (2) STATIC COMPLEMENT: an AST scan of e3_selector.py's source for
      `self.last_score_diagnostics[<key>] = <literal-or-ternary-of-string-
      literals>` assignments, checked against the same allowlist. This
      catches a hardcoded category-label literal added on a branch this
      test's scenario does not happen to exercise -- exactly the gap a
      pure runtime check has. It is a complement, not a substitute (per
      CLAUDE.md guidance): it cannot see a label built by string
      concatenation, f-string, or a helper call, only a direct literal or
      ternary-of-literals.
  (3) NEGATIVE CONTROL: unit tests against the type-checker helper itself,
      confirming it actually rejects an unallowlisted string and an
      allowlisted field holding an out-of-set value, so the runtime/static
      passes above are not vacuous.

BOUNDARY. This is ONLY the v3 typing contract on E3's EXISTING output
surface. MECH-529's split/merge/reweight rebucketing loop is v4 and gated
on the ARC-134 P0 grain operator; nothing here builds, stubs, or tests any
rebucketing consumer.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector

_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "ree_core" / "predictors" / "e3_selector.py"
)

# --------------------------------------------------------------------------- #
# The type policy: MECH-530 itself, expressed as code.                        #
# --------------------------------------------------------------------------- #

# Leaf types E3's decision-outcome surface may hold. bool is intentionally
# listed even though it is a subclass of int -- documents both are permitted
# rather than relying on that subclassing fact.
_PERMITTED_LEAF_TYPES = (bool, int, float, torch.Tensor, type(None))

# CLOSED allowlist of string-valued fields, cited to the exact reason each is
# not a semantic category label (see RECONCILIATION FINDING above). Adding a
# THIRD field here is a real decision -- it must be justified the same way,
# not simply appended to make a red test green.
ALLOWED_STRING_FIELDS: Dict[str, frozenset] = {
    # e3_selector.py ~3706: which formula the ARC-016 commit gate compared
    # commit_variance against on this tick -- an internal-mode selector, not
    # a description of the candidates/environment.
    "commit_gate_mode": frozenset({"harm_score_variance", "world_variance"}),
    # e3_selector.py ~3980 (pre-seeded ~3380): which shortlist-construction
    # strategy (MECH-448/439/ARC-107) built the F-eligible near-tie set --
    # again an internal algorithm-mode selector, not a content-class label.
    "modulatory_shortlist_mode": frozenset({"margin", "top_k", "f_demotion"}),
    # e3_selector.py ~3378 (E3Config.modulatory_authority_normalize_basis):
    # which formula the additive modulatory authority anchors its target
    # range against ("range" = raw_score_range, outlier-sensitive legacy;
    # "std" = raw_score_std, robust). An echo of an operator-set config knob
    # naming an internal normalisation formula, not a content class.
    "modulatory_authority_normalize_basis": frozenset({"range", "std"}),
}


def _assert_typed_diagnostic_value(field_name: str, value: Any, *, path: str = "") -> None:
    """Recursively assert `value` (found at `field_name`, nested at `path`) obeys
    MECH-530's type policy. Raises AssertionError naming the offending field
    otherwise."""
    full_path = f"{field_name}{path}"
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), (
                f"MECH-530 violation: {full_path} has a non-string dict key "
                f"{k!r} ({type(k).__name__}) -- dict keys on E3's output "
                f"surface must be descriptive string labels (e.g. loop/channel "
                f"names), never a typed value smuggled into key position."
            )
            _assert_typed_diagnostic_value(field_name, v, path=f"{path}[{k!r}]")
        return
    if isinstance(value, str):
        allowed = ALLOWED_STRING_FIELDS.get(field_name)
        assert allowed is not None, (
            f"MECH-530 violation: {full_path} = {value!r} is a string value on "
            f"E3's decision-outcome output surface, but {field_name!r} is not in "
            f"this test's ALLOWED_STRING_FIELDS allowlist. E3's output must be "
            f"continuous/boolean/count-typed, never a semantic category label "
            f"naming a content class (MECH-530, claims.yaml). If this is "
            f"genuinely an internal-mode selector (which formula/strategy fired), "
            f"not a description of a candidate/environment content class, add it "
            f"to ALLOWED_STRING_FIELDS with a citation to the exact write site "
            f"and why it is not oracular. If it names a content/object/context "
            f"class, remove it -- it breaks MECH-530."
        )
        assert value in allowed, (
            f"MECH-530 violation: {full_path} = {value!r} is not one of the "
            f"allowlisted values {sorted(allowed)!r} for this known internal-mode "
            f"selector. A value outside the closed set is either a bug or a new "
            f"mode that needs its own allowlist review (MECH-530)."
        )
        return
    assert isinstance(value, _PERMITTED_LEAF_TYPES), (
        f"MECH-530 violation: {full_path} has type {type(value).__name__}, which "
        f"is none of the permitted continuous/boolean/count/tensor types "
        f"({', '.join(t.__name__ for t in _PERMITTED_LEAF_TYPES)}). E3's "
        f"decision-outcome output surface must never carry an untyped or "
        f"unrecognised-shape value."
    )


def _assert_surface_typed(last_score_diagnostics: Dict[str, Any]) -> None:
    for key, value in last_score_diagnostics.items():
        _assert_typed_diagnostic_value(key, value)


def _assert_selection_result_typed(result) -> None:
    # selected_trajectory is the chosen Trajectory object itself (states/
    # actions/world_states), not a diagnostic signal about it -- out of
    # scope for the "is this a category label" question this claim is about.
    _SKIP_FIELDS = {"selected_trajectory"}
    for f in dataclasses.fields(result):
        if f.name in _SKIP_FIELDS:
            continue
        _assert_typed_diagnostic_value(f.name, getattr(result, f.name))


# --------------------------------------------------------------------------- #
# Static complement: AST scan for hardcoded string-literal diagnostics.       #
# --------------------------------------------------------------------------- #


def _string_literals_in(expr: ast.expr) -> List[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, ast.IfExp):
        return _string_literals_in(expr.body) + _string_literals_in(expr.orelse)
    return []


def _last_score_diagnostics_key(target: ast.expr) -> str | None:
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "last_score_diagnostics"
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    ):
        return target.slice.value
    return None


def _scan_string_literal_diagnostics(source: str) -> Dict[str, List[str]]:
    """Static scan: for every `self.last_score_diagnostics[<key>] = <expr>`
    assignment in `source`, collect the string literal(s) `<expr>` can
    evaluate to (direct literal, or a ternary between literals). Returns
    {key: [literal, ...]} only for keys with at least one literal RHS."""
    tree = ast.parse(source)
    found: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            key = _last_score_diagnostics_key(target)
            if key is None:
                continue
            literals = _string_literals_in(node.value)
            if literals:
                found.setdefault(key, []).extend(literals)
    return found


# --------------------------------------------------------------------------- #
# Scenario: drive a real select() call to populate the surface widely.        #
# --------------------------------------------------------------------------- #


def _candidate(action_class: int, action_dim: int = 8) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class % action_dim] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _patch_raw(selector, candidates, raw_costs):
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


def _populated_selector_and_result():
    """Reproduces test_e3_last_scores_post_arbitration.py's loop-segregation
    scenario (shortlist-then-modulate + loop segregation + finer-channel
    routing, so loop_named_channel_routed_ranges is non-empty) and also turns
    on e3_score_decomp_enabled so the commit-gate diagnostics (including
    commit_gate_mode) are populated -- between the two, most of the ~77
    last_score_diagnostics write sites in a single tick are reachable."""
    sel = E3TrajectorySelector(
        E3Config(
            world_dim=6,
            hidden_dim=8,
            use_modulatory_shortlist_then_modulate=True,
            modulatory_shortlist_margin=1.0,
            use_loop_segregation=True,
            use_finer_channel_gating=True,
            loop_segregation_spiral_gain_limbic=3.0,
            # ARC-110 C2: also route named-channel per-candidate range through
            # to the loop arbitration, so loop_named_channel_routed_ranges
            # (Dict[str, float], the dict-keyed-by-descriptive-string case
            # MECH-530 explicitly permits) is populated on this scenario.
            use_named_channel_routing=True,
        )
    )
    sel._running_variance = 0.0  # force committed path (deterministic argmin)
    sel.e3_score_decomp_enabled = True

    raw = [-5.0, 0.5, 0.5, 0.5]
    limbic_val = torch.tensor([0.5, 0.5, 0.5, -0.01])
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, raw)
    result = sel.select(
        candidates,
        temperature=1.0,
        score_bias=limbic_val.clone() * 3,
        score_bias_channels={
            "ofc": limbic_val.clone(),
            "liking": limbic_val.clone(),
            "vigour": limbic_val.clone(),
        },
        score_bias_channel_routed={
            "ofc": limbic_val.clone(),
            "liking": limbic_val.clone(),
            "vigour": limbic_val.clone(),
        },
    )
    return sel, result


# --------------------------------------------------------------------------- #
# (1) runtime: the live populated surface is fully typed                      #
# --------------------------------------------------------------------------- #


def test_live_last_score_diagnostics_surface_is_typed():
    sel, _ = _populated_selector_and_result()

    # Non-degeneracy: this scenario must actually populate a wide surface, or
    # the walk below would pass vacuously over an empty/tiny dict.
    assert len(sel.last_score_diagnostics) >= 20, (
        f"scenario only populated {len(sel.last_score_diagnostics)} "
        f"last_score_diagnostics keys -- too few to be a meaningful walk; the "
        f"scenario may have stopped exercising the intended code paths"
    )
    # The three known string fields must actually be present and on-allowlist
    # -- confirms the walk below is exercising the real reconciliation
    # finding, not silently missing the code paths that produce them.
    assert sel.last_score_diagnostics.get("commit_gate_mode") == "world_variance"
    assert sel.last_score_diagnostics.get("modulatory_shortlist_mode") == "margin"
    assert (
        sel.last_score_diagnostics.get("modulatory_authority_normalize_basis")
        == "range"
    )
    assert sel.last_score_diagnostics.get("loop_named_channel_routed_ranges"), (
        "expected a non-empty loop_named_channel_routed_ranges (Dict[str, float]) "
        "-- the dict-keyed-by-descriptive-string case MECH-530 explicitly permits"
    )

    _assert_surface_typed(sel.last_score_diagnostics)

    string_fields = {
        k: v for k, v in sel.last_score_diagnostics.items() if isinstance(v, str)
    }
    assert set(string_fields) <= set(ALLOWED_STRING_FIELDS), (
        f"unexpected string-valued field(s) on the live surface: "
        f"{sorted(set(string_fields) - set(ALLOWED_STRING_FIELDS))} -- "
        f"MECH-530 violation not caught by the per-field check above"
    )


def test_live_selection_result_is_typed():
    _, result = _populated_selector_and_result()
    _assert_selection_result_typed(result)


def test_harm_eval_and_benefit_eval_are_typed():
    sel, _ = _populated_selector_and_result()
    z_world = torch.zeros(1, 6)
    harm = sel.harm_eval(z_world)
    benefit = sel.benefit_eval(z_world)
    assert isinstance(harm, torch.Tensor)
    assert isinstance(benefit, torch.Tensor)
    assert harm.dtype.is_floating_point
    assert benefit.dtype.is_floating_point


# --------------------------------------------------------------------------- #
# (2) static complement: no unreviewed hardcoded category-label literal       #
# --------------------------------------------------------------------------- #


def test_static_scan_finds_only_allowlisted_string_literals():
    source = _SOURCE_PATH.read_text()
    found = _scan_string_literal_diagnostics(source)

    # Non-degeneracy: the scan itself must find the two known cases, or the
    # scan machinery (not the property) has silently broken.
    assert "commit_gate_mode" in found, (
        "static scan found no string-literal assignment to "
        "last_score_diagnostics['commit_gate_mode'] -- either the scan is "
        "broken or the source changed shape; investigate before trusting a "
        "green result here"
    )

    for key, literals in found.items():
        allowed = ALLOWED_STRING_FIELDS.get(key)
        assert allowed is not None, (
            f"MECH-530 violation: e3_selector.py hardcodes string literal(s) "
            f"{literals!r} into last_score_diagnostics['{key}'], but {key!r} is "
            f"not in ALLOWED_STRING_FIELDS. This is a category-label leak unless "
            f"reviewed and added to the allowlist with a citation (see this "
            f"file's module docstring)."
        )
        for literal in literals:
            assert literal in allowed, (
                f"MECH-530 violation: e3_selector.py hardcodes "
                f"last_score_diagnostics['{key}'] = {literal!r}, which is outside "
                f"the reviewed allowlist {sorted(allowed)!r} for that field."
            )


# --------------------------------------------------------------------------- #
# (3) negative control: the checker itself is not vacuous                     #
# --------------------------------------------------------------------------- #


def test_checker_accepts_permitted_leaf_types():
    _assert_typed_diagnostic_value("some_field", True)
    _assert_typed_diagnostic_value("some_field", 3)
    _assert_typed_diagnostic_value("some_field", 3.14)
    _assert_typed_diagnostic_value("some_field", torch.tensor([1.0, 2.0]))
    _assert_typed_diagnostic_value("some_field", None)
    _assert_typed_diagnostic_value(
        "loop_named_channel_routed_ranges", {"ofc": 0.3, "liking": 0.1}
    )
    _assert_typed_diagnostic_value("commit_gate_mode", "world_variance")
    _assert_typed_diagnostic_value("modulatory_shortlist_mode", "top_k")


def test_checker_rejects_unallowlisted_category_label():
    with pytest.raises(AssertionError, match="MECH-530"):
        _assert_typed_diagnostic_value("perceived_object_class", "predator")


def test_checker_rejects_allowlisted_field_with_out_of_set_value():
    with pytest.raises(AssertionError, match="MECH-530"):
        _assert_typed_diagnostic_value("commit_gate_mode", "not_a_known_mode")


def test_checker_rejects_non_string_dict_key():
    with pytest.raises(AssertionError, match="MECH-530"):
        _assert_typed_diagnostic_value("some_dict_field", {1: 0.5})


def test_checker_rejects_unrecognised_type():
    class _NotAPermittedType:
        pass

    with pytest.raises(AssertionError, match="MECH-530"):
        _assert_typed_diagnostic_value("some_field", _NotAPermittedType())


def test_static_scan_rejects_unallowlisted_literal():
    source = (
        "class X:\n"
        "    def f(self):\n"
        "        self.last_score_diagnostics['new_field'] = (\n"
        "            'predator' if flag else 'no_predator'\n"
        "        )\n"
    )
    found = _scan_string_literal_diagnostics(source)
    assert found == {"new_field": ["predator", "no_predator"]}
    allowed = ALLOWED_STRING_FIELDS.get("new_field")
    assert allowed is None  # would fail the real test's allowlist assertion
