"""Pin which config knobs `REEConfig.from_dims()` actually plumbs through.

WHAT THIS GUARDS. `from_dims` ends in `**kwargs` (config.py), and the body pops
only a handful of them. Every other unrecognised kwarg is **silently dropped --
no error, no warning**. So `from_dims(some_flag=False)` can be a complete no-op
while reading exactly like a working ablation. That is
[memory] `reference-reeconfig-from-dims-silent-kwargs`, and it is not
hypothetical: the MECH-307 repair (2026-08-07) found twelve dataclass fields with
no signature entry, into which **84 drivers** had been passing
`use_mech307_conjunction=True` and running with all four gaps OFF.

THE SIGNATURE IS NOT THE ANSWER, AND THAT IS THE POINT OF SWEEPING RATHER THAN
PARSING. Membership of the `from_dims` signature and actually landing on the live
config tree are two different properties, and they came apart in BOTH directions
when this was first measured (2026-08-22):

  * six names land WITHOUT a signature entry -- `use_iterative_inference`,
    `inference_settle_iters`, `inference_convergence_rel_tol`,
    `use_self_recurrence`, `self_recurrence_e1_coupling`,
    `use_cross_module_consolidation` (+ its three companions) are explicit
    `kwargs.pop(...)` reads in the body;
  * `sd016_diversification_weight` has a signature entry (added 2026-06-05) and
    the body never assigns it, so it is accepted and dropped on the floor --
    a SHARPER failure than the `**kwargs` swallow, because a signature entry
    actively advertises reachability. Ten drivers pass a non-zero weight and
    run with the diversification loss term OFF. STILL OPEN: the one-line repair
    was written and verified on 2026-08-22 and deliberately NOT landed, because
    `ree_core/utils/config.py` was held by a concurrent claim
    (`elated-jackson-f12eae-docs`, ARC-065 GAP-A head training) whose own
    uncommitted edit was sitting in the shared checkout at the time -- committing
    that file would have swept 18 of their lines under this change. See
    `test_sd016_diversification_weight_is_an_open_defect` for the exact repair.

So this contract EXECUTES `from_dims` and reads the values back off a live config
tree. It never parses config.py.

TWO NAMING CONVENTIONS ARE DELIBERATELY NOT USED AS THE FILTER.
`tests/test_flag_inertness.py::test_flag_registry_is_current` -- good prior art,
and the model for the registry shape here -- scans for `use_*` / `*_enabled`
names. Thirteen of the 37 unreachable bool flags found here match NEITHER,
including `beta_gate_bistable`, which is the flag whose silent swallow was
measured live on 2026-08-22 (session `igw-232-mech091-driver-successor`:
`from_dims(beta_gate_bistable=True)` did not land, so MECH-091's completion
trigger was structurally unreachable and that session's first probe measured a
dead completion path). This file therefore sweeps EVERY bool on the tree.

THE WALK IS RECURSIVE ON PURPOSE. `scripts/authority_trace_probe._config_objects`
-- whose `flag_map` contract this reuses -- descends exactly one level from
`REEConfig`. Six live config objects sit two deep under `hippocampal`
(`anchor_set`, `event_segmenter`, `ghost_goal_bank_config`,
`invalidation_trigger`, `persistence_appraisal_compute`,
`staleness_accumulator`), and `use_composite_cue_outshining` /
`use_persistence_efficacy_gate` / `subscribe_to_boundary_events` are only
visible from there. Keys are dotted PATHS, not class names, so two instances of
one class are not collapsed.

WHY THIS FILE DOES NOT RIDE `conftest.corpus_scan`. That fixture's docstring
states the standing pattern for a new corpus-wide lint (add to `path_lints`,
take the `corpus_scan` fixture), and this deviates from it for two specific
reasons rather than by oversight. (1) FILE SET: the `path_lints` share is
driver-major over `glob("v3_exq_*.py")`, and one of the confirmed drop sites is
`experiments/_lib/baselines/exq610_inv074_crystallization_baseline.py`, which
that glob does not contain -- riding the share would mean widening its file set,
which is exactly what its exact-count pins exist to prevent. (2) SHAPE: the
share is driver-major and hands each lint one already-parsed tree, whereas this
check is a whole-call-graph question about kwarg names, so it would need a new
rglob-scoped entry beside `prereg_share_feasibility_lint` rather than a
`path_lints` line.

BE HONEST ABOUT WHAT THAT COSTS: this IS a seventh independent parse of the
corpus, which is precisely what the shared scan was built to delete, and the
prefilter does not rescue it -- only ~280 of 1466 files lack a forwarder token,
so ~1184 are really parsed. Folding it into `conftest.scan_corpus` is worth
doing and is tracked as follow-on; it is deliberately NOT done in the same
change as the audit that motivated the file, because that fixture carries
exact-count pins whose whole purpose is to fail when a file set moves.

TIMING NUMBERS BELONG ON AN IDLE BOX, NOT HERE. Every figure quoted in this
docstring was taken on the CONTENDED Mac during an interactive session and
moved by 3x between runs of the identical code (17.3s vs 58.5s to parse the
same 1184 files). Treat them as orders of magnitude, and re-measure per
CLAUDE.md "Running the test suite" before reasoning from them.

SCOPE, STATED SO IT IS NOT MISREAD. This file says nothing about whether a flag
DOES anything once it lands -- that is `test_flag_inertness.py`'s question. It
asks only whether the canonical constructor can set it at all.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Structural dimensions, not knobs: overriding these changes the shape of the
# tree rather than a setting on it, so they are not swept.
DIMS = dict(body_obs_dim=10, world_obs_dim=20, action_dim=4, self_dim=16, world_dim=16)

# Classmethods that forward `**kwargs` into `from_dims` (config.py ~9026/9075/9117).
# A name dropped by `from_dims` is dropped identically through any of these.
FROM_DIMS_FORWARDERS = frozenset(
    {"from_dims", "for_grid_world", "for_causal_grid_world", "minimal", "standard"}
)


# ---------------------------------------------------------------------------
# registries -- adding a flag forces a decision, exactly as PROBED /
# KNOWN_UNPROBED do in tests/test_flag_inertness.py
# ---------------------------------------------------------------------------

# Unreachable through `from_dims`, but reachable by an idiom this repo ACCEPTS:
# direct sub-config construction (`HippocampalConfig(...)`), attribute assignment
# (`cfg.latent.use_resource_encoder = True`), or a profile method
# (`enable_goal_stream`). These are DOCUMENTATION gaps, not defects -- the value
# is the count of call sites already using the working idiom, which is the
# evidence that it works.
REACHABLE_BY_ALTERNATIVE_IDIOM = {
    "benefit_terrain_enabled": "residue; 16 drivers + 5 tests set it directly",
    "benefit_terrain_live_producer": "residue; 2 drivers + 1 test",
    "beta_gate_bistable": "heartbeat; 160 drivers + 4 lib + 11 tests set it directly",
    "descending_attenuation_factor": (
        "REEConfig; companion float knob of harm_descending_mod_enabled, set "
        "directly alongside it by the same 7 call sites (see that entry)."
    ),
    "e3_include_untrained_fallback_scorers": "e3; 2 drivers + 4 tests",
    "ewc_enabled": "residue; 5 tests + 1 ree_core site",
    "gaba_harm_state_recurrence": "latent; ree_core-internal only",
    "harm_descending_mod_enabled": (
        "REEConfig; SD-021 descending pain modulation. REPAIRED 2026-08-22 "
        "(chip-20260822-fromdims-exq325-dead-ablation-axis) -- all 6 "
        "v3_exq_325-family drivers now set it by attribute assignment on the "
        "returned config, the same idiom v3_exq_610_inv074_crystallization_"
        "necessity.py already used. Formerly a live KNOWN_FROM_DIMS_DROP_SITES "
        "entry; moved here once no call site passed it into from_dims anymore."
    ),
    "include_active": "hippocampal.ghost_goal_bank_config; 2 drivers via ctor",
    "include_inactive": "hippocampal.ghost_goal_bank_config; 2 drivers via ctor",
    "mode_conditioning_enabled": "hippocampal; 11 drivers + 2 tests",
    "mode_partitioned_cem": "hippocampal; 1 driver",
    "safety_terrain_enabled": "residue; 3 drivers + 2 tests + 1 ree_core site",
    "unified_latent_mode": "latent; 52 drivers set it directly",
    "use_commit_readiness_gate": "heartbeat; 14 drivers + 2 tests",
    "use_composite_cue_outshining": "ghost_goal_bank_config; 3 drivers + 2 tests",
    "use_conditional_precision_gate": "e3; 1 test",
    "use_curiosity_familiarity": (
        "hippocampal; HippocampalConfig(...) in test_sd025_curiosity_drive.py C5 "
        "and drivers v3_exq_767 / v3_exq_768. NEGATIVE CONTROL: this is the flag "
        "chip-20260822-curiosity-familiarity-unablatable wrongly called "
        "'cannot be ablated'; direct sub-config construction is an accepted idiom "
        "here, so it must NOT be reported as a defect."
    ),
    "use_da_modulated_rbf_density": "residue; 11 drivers + 3 tests",
    "use_e2_world_uncertainty": "latent; 2 drivers + 6 tests + 1 ree_core site",
    "use_harm_un": "latent; 12 drivers + 1 lib + 2 tests",
    "use_hierarchical_goal_credit": "goal; 2 drivers + 2 tests",
    "use_identity_classifier": "latent; 17 drivers + 1 ree_core site",
    "use_persistence_efficacy_gate": "ghost_goal_bank_config; 2 drivers + 4 tests",
    "use_progress_velocity_effort_modulation": "goal; 1 lib + 1 test",
    "use_waking_confidence_inflation": "e3; 7 drivers + 1 test",
    "valence_bounding_enabled": "residue; 1 test",
    "valence_enabled": "residue; 27 drivers + 2 lib + 2 tests",
}

# Unreachable through `from_dims` AND with no call site anywhere in this repo --
# not drivers, not tests, not ree_core. Dataclass-only fields that nothing has
# ever set. Listed separately because "nobody can reach it" and "nobody has
# tried" are different findings, and only the first is a defect waiting to
# happen.
NO_CONFIRMED_CALLER = {
    "action_loop_gate_enabled": "REEConfig",
    "commit_readiness_strict_single_candidate": "heartbeat",
    "preserve_on_life_end": "REEConfig",
    "preserve_on_life_end_strict": "REEConfig",
    "subscribe_to_boundary_events": "hippocampal.anchor_set",
    "use_arbitration_aware_decisiveness_margin": "REEConfig",
}

# OPEN DEFECTS: a name really passed into `from_dims` (or a forwarder) at these
# paths and really dropped. Deliberately keyed by PATH, not (path, lineno) --
# line numbers churn on every unrelated edit and would make this a maintenance
# tax rather than a guard.
#
# NOT FIXED BY MASS-ADDING SIGNATURE ENTRIES. Adding a `from_dims` parameter for
# every dataclass field is a convention change, not a repair, and this repo has
# a working alternative idiom for all of them (see the registry above). Each
# entry below records the measured blast radius so the decision can be taken per
# flag, by whoever owns the claim.
KNOWN_FROM_DIMS_DROP_SITES = {
    "sd016_diversification_weight": {
        "reason": (
            "THE ONE WITH A SIGNATURE ENTRY, which is why it is listed first: "
            "from_dims ACCEPTS it (no **kwargs swallow, no TypeError) and the "
            "body simply never assigns it, so a caller has no signal of any kind. "
            "Consumer is agent.py's compute_prediction_loss, gated "
            "`_div_w > 0.0 and sd016_enabled` -- so ten of these eleven drivers "
            "passed SD016_DIVERSIFICATION_WEIGHT=0.5 and trained with the "
            "diversification loss term OFF (418f passes 0.0 and is unaffected). "
            "REPAIR, verified 2026-08-22 but NOT landed -- see the module "
            "docstring for why -- one line in from_dims, beside the other SD-016 "
            "assignments: `config.sd016_diversification_weight = "
            "sd016_diversification_weight`. Default 0.0 == the dataclass default, "
            "so an unaffected driver stays bit-identical."
        ),
        "paths": {
            "experiments/v3_exq_265a_sd017_sleep_phase_methods_validation_phase2.py",
            "experiments/v3_exq_418f_sd016_attention_uniformity_probe.py",
            "experiments/v3_exq_418l_sd017_action_bias_div_phase2.py",
            "experiments/v3_exq_436a_sd017_arc045_mech166_context_harm_phase2.py",
            "experiments/v3_exq_436b_sd017_mech166_repr_confirmer.py",
            "experiments/v3_exq_436c_sd017_mech166_repr_confirmer.py",
            "experiments/v3_exq_436d_sd017_mech166_writepath_retest.py",
            "experiments/v3_exq_436e_sd017_mech166_occupied_slot_retest.py",
            "experiments/v3_exq_436f_sd017_mech166_sd016_armed_retest.py",
            "experiments/v3_exq_500a_sd017_sleep_phase_readiness_phase2.py",
            "experiments/v3_exq_503a_sd017_sleep_phase_discriminative_phase2.py",
        },
    },
    # harm_descending_mod_enabled / descending_attenuation_factor: REPAIRED
    # 2026-08-22 (chip-20260822-fromdims-exq325-dead-ablation-axis) -- all six
    # v3_exq_325-family call sites now set both by attribute assignment on the
    # returned config, the same idiom v3_exq_610 already used. Removed from
    # this registry rather than left as a stale phantom-path entry.
    "gated_policy_use_differential_heads": {
        "reason": (
            "Passed True in all ten, commented '# ARC-062 fix.' -- so the fix is "
            "not applied, in BOTH arms (it sits in the shared builder; the arm "
            "difference is **xtal_kwargs). Not a dead axis, an unarmed mechanism. "
            "Consumer: agent.py:1128. The same builder sets "
            "config.heartbeat.beta_gate_bistable / config.harm_descending_mod_enabled "
            "by attribute assignment three lines later, which DO land -- the "
            "working idiom is already in the file, immediately below the broken one."
        ),
        "paths": {
            "experiments/_lib/baselines/exq610_inv074_crystallization_baseline.py",
            "experiments/v3_exq_610_inv074_crystallization_necessity.py",
            "experiments/v3_exq_610a_inv074_crystallization_necessity.py",
            "experiments/v3_exq_610b_inv074_crystallization_necessity.py",
            "experiments/v3_exq_610c_inv074_crystallization_necessity.py",
            "experiments/v3_exq_610d_inv074_crystallization_necessity.py",
            "experiments/v3_exq_610e_inv074_crystallization_necessity.py",
            "experiments/v3_exq_610f_inv074_crystallization_necessity.py",
            "experiments/v3_exq_655_inv074_crystallization_necessity_taskshift.py",
            "experiments/v3_exq_656_inv074_crystallization_necessity_taskshift.py",
        },
    },
    "use_resource_encoder": {
        "reason": (
            "v3_exq_527 passes True on the GOAL_PRESENT arm and False on "
            "GOAL_ABSENT; both are dropped, so no ResourceEncoder is built and "
            "the arm's own probe condition (`latent.z_resource is not None`, "
            "line 313) can never hold. The two arms still differ on "
            "use_identity_classifier / z_goal_enabled, so the experiment is not "
            "wholly dead -- but the z_resource pathway it names is absent. "
            "Reachable via enable_goal_stream (config.py:6273) and by attribute "
            "assignment, which this same driver uses two lines later for "
            "use_identity_classifier."
        ),
        "paths": {"experiments/v3_exq_527_mech112_identity_goal_reef.py"},
    },
    "harm_surprise_pe_enabled": {
        "reason": (
            "DECORATIVE, NOT DEAD -- and the distinction is the whole reason this "
            "registry carries prose. v3_exq_324's arms look identically at risk to "
            "v3_exq_325's, but its driver DOES realise the arm itself "
            "(`if pe_enabled:` in its own training loop), so the experiment "
            "remains valid; only the config flag is inert. Do not fold this entry "
            "into the harm_descending one on the strength of the shape."
        ),
        "paths": {"experiments/v3_exq_324_sd020_harm_surprise_pe.py"},
    },
    "harm_obs_ema_alpha": {
        "reason": (
            "Companion knob in the same v3_exq_324 call. Harmless twice over: the "
            "value passed (0.1) equals the default, and the driver computes its "
            "own EMA with the module constant."
        ),
        "paths": {"experiments/v3_exq_324_sd020_harm_surprise_pe.py"},
    },
    "harm_nonredundancy_weight": {
        "reason": (
            "Same shape as harm_surprise_pe_enabled: v3_exq_323's NONREDUNDANT "
            "(1.0) vs BASELINE (0.0) axis goes through this kwarg and is dropped, "
            "but the driver applies the penalty itself "
            "(`if nonredundancy_weight > 0.0:`), so the arms do differ. "
            "Decorative flag, valid experiment."
        ),
        "paths": {
            "experiments/v3_exq_323_sd019_harm_nonredundancy.py",
            "experiments/v3_exq_323a_sd019_harm_nonredundancy.py",
        },
    },
    "wanting_weight": {
        "reason": (
            "NEGATIVE CONTROL, AND NOT A DEFECT. This is a NAME COLLISION between "
            "two independent knobs: HippocampalConfig.wanting_weight (CEM scoring, "
            "default 0.0) and GhostGoalBankConfig.wanting_weight (ghost-priority, "
            "default 1.0). from_dims targets the hippocampal one and lands on it "
            "correctly; the sweep reports PARTIAL only because a same-named field "
            "on an unrelated config does not move. Registered so that a later "
            "session does not 'repair' it by plumbing from_dims into the ghost "
            "bank, which would silently couple two unrelated knobs."
        ),
        "paths": {
            "experiments/v3_exq_259_wanting_gradient_navigation.py",
            "experiments/v3_exq_263_mech216_e1_predictive_wanting.py",
            "experiments/v3_exq_263a_mech216_e1_predictive_wanting.py",
            "experiments/v3_exq_263b_sd023_mech216_landmark_wanting.py",
            "experiments/v3_exq_326_wanting_gradient_nav_fix.py",
            "experiments/v3_exq_326a_wanting_gradient_nav_fix.py",
            "experiments/v3_exq_327_mech163_goal_conditioned_nav.py",
            "experiments/v3_exq_331_arc030_approach_avoidance_balance.py",
            "experiments/v3_exq_332_mech216_predictive_wanting.py",
            "experiments/v3_exq_332a_mech216_predictive_wanting_dense.py",
            "experiments/v3_exq_495_mech163_planned_system_gate.py",
            "experiments/v3_exq_495a_mech163_planned_system_gate.py",
            "experiments/v3_exq_559a_goal_stream_canonical.py",
            "experiments/v3_exq_560_goal_stream_selectivity_score_decomp.py",
            "experiments/v3_exq_914_mech236_hippocampal_zgoal_channel_ablation.py",
            "experiments/v3_exq_931_cem_wanting_weight_selection_authority.py",
        },
    },
}


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

SCALARS = (bool, int, float, str)


def _config_objects(cfg, seen=None, path=""):
    """Every live `*Config` object reachable from `cfg`, keyed by dotted path.

    Same contract as `scripts/authority_trace_probe._config_objects`, widened to
    recurse -- see the module docstring for the six two-deep objects that
    one-level version cannot see.
    """
    if seen is None:
        seen = {}
    key = path or type(cfg).__name__
    if key in seen:
        return seen
    seen[key] = cfg
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if type(value).__name__.endswith("Config") and dataclasses.is_dataclass(value):
            _config_objects(value, seen, (path + "." if path else "") + name)
    return seen


def _scalar_map(cfg):
    """field name -> {config path: value} over a LIVE config tree."""
    out = {}
    for path, obj in _config_objects(cfg).items():
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if type(value) in SCALARS:
                out.setdefault(name, {})[path] = value
    return out


def _read_at(cfg, paths, name):
    """`{dotted config path: getattr(obj, name)}` without re-walking the tree."""
    out = {}
    for path in paths:
        obj = cfg
        if path != type(cfg).__name__:
            for part in path.split("."):
                obj = getattr(obj, part)
        try:
            out[path] = getattr(obj, name)
        except AttributeError:
            pass
    return out


def _build(**overrides):
    return REEConfig.from_dims(**DIMS, **overrides)


def _probe_values(value):
    """Two values of `value`'s own type, neither equal to it. None = do not sweep."""
    if type(value) is bool:
        return (True, False)
    if type(value) is int:
        return (value + 7, value + 13)
    if type(value) is float:
        return (value + 0.3721, value + 0.8419)
    return None  # str fields are enum-validated; an arbitrary probe raises


@pytest.fixture(scope="module")
def sweep():
    """name -> ('LANDS' | 'PARTIAL' | 'UNREACHABLE', landed paths, missed paths)."""
    base_map = _scalar_map(_build())
    result = {}
    for name, base_values in base_map.items():
        if name in DIMS:
            continue
        if len({type(v) for v in base_values.values()}) != 1:
            continue
        targets = _probe_values(next(iter(base_values.values())))
        if targets is None:
            continue
        paths = sorted(base_values)
        try:
            # Read back only the paths this name lives on, rather than rebuilding
            # the whole `_scalar_map`. Same answer; it turns two full 17-object
            # tree walks per field into two getattr chains, which is what keeps
            # the sweep seconds rather than minutes over ~1000 fields.
            per_target = {t: _read_at(_build(**{name: t}), paths, name) for t in targets}
        except Exception:
            continue  # a validated knob that refuses an arbitrary probe
        lands = [p for p in paths if all(per_target[t].get(p) == t for t in targets)]
        misses = [p for p in paths if p not in lands]
        status = "LANDS" if not misses else ("UNREACHABLE" if not lands else "PARTIAL")
        result[name] = (status, lands, misses)
    return result


def _bool_names(sweep_result):
    base_map = _scalar_map(_build())
    return {
        n for n in sweep_result
        if all(type(v) is bool for v in base_map.get(n, {}).values()) and base_map.get(n)
    }


# ---------------------------------------------------------------------------
# vacuity guards -- a broken walk must fail, not pass quietly
# ---------------------------------------------------------------------------

def test_sweep_covers_the_whole_config_tree(sweep):
    paths = set(_config_objects(_build()))
    assert len(paths) >= 17, f"config tree walk collapsed to {sorted(paths)}"
    # the six objects a one-level walk cannot see
    for deep in (
        "hippocampal.anchor_set",
        "hippocampal.event_segmenter",
        "hippocampal.ghost_goal_bank_config",
        "hippocampal.invalidation_trigger",
        "hippocampal.persistence_appraisal_compute",
        "hippocampal.staleness_accumulator",
    ):
        assert deep in paths, f"recursive walk lost {deep}; see module docstring"
    assert len(sweep) >= 900, f"sweep collapsed to {len(sweep)} fields"


def test_sweep_detects_a_synthetic_swallow(sweep):
    """POSITIVE CONTROL: a name from_dims has never heard of must read UNREACHABLE.

    Without this, a sweep that silently stopped overriding anything would report
    every flag as landing and every test in this file would pass vacuously.
    """
    cfg = _build(**{"ree_nonexistent_probe_flag_xyz": True})
    assert not hasattr(cfg, "ree_nonexistent_probe_flag_xyz"), (
        "from_dims grew a **kwargs passthrough onto the config object; the "
        "silent-swallow hazard this file guards may no longer exist as described"
    )


# ---------------------------------------------------------------------------
# the registry contract
# ---------------------------------------------------------------------------

def test_every_unreachable_bool_flag_is_registered(sweep):
    """A new flag `from_dims` cannot set forces a decision, as PROBED does."""
    registered = (
        set(REACHABLE_BY_ALTERNATIVE_IDIOM)
        | set(NO_CONFIRMED_CALLER)
        | set(KNOWN_FROM_DIMS_DROP_SITES)
    )
    unreachable = {
        n for n in _bool_names(sweep) if sweep[n][0] == "UNREACHABLE"
    }
    unregistered = sorted(unreachable - registered)
    assert not unregistered, (
        "Config flag(s) that REEConfig.from_dims() SILENTLY DROPS and that no "
        f"registry in this file accounts for: {unregistered}. This is the "
        "[memory] reference-reeconfig-from-dims-silent-kwargs hazard. Decide, "
        "per flag: (a) it is reachable by direct sub-config construction / "
        "attribute assignment / a profile method -> add it to "
        "REACHABLE_BY_ALTERNATIVE_IDIOM with the call sites that prove it; "
        "(b) nothing anywhere sets it -> NO_CONFIRMED_CALLER; (c) something "
        "passes it into from_dims and loses it -> KNOWN_FROM_DIMS_DROP_SITES "
        "with the measured blast radius. Do NOT reflexively add a from_dims "
        "signature entry -- that is a convention change, not a repair."
    )


def test_registry_has_no_phantom_entries(sweep):
    """A registered flag that now LANDS (or is gone) must be pruned."""
    registered = (
        set(REACHABLE_BY_ALTERNATIVE_IDIOM)
        | set(NO_CONFIRMED_CALLER)
        | set(KNOWN_FROM_DIMS_DROP_SITES)
    )
    now_landing = sorted(
        n for n in registered if n in sweep and sweep[n][0] == "LANDS"
    )
    assert not now_landing, (
        f"Registry lists flag(s) that from_dims now plumbs through: {now_landing}. "
        "Remove them -- a stale entry makes a repaired flag look broken and "
        "hides the next real one."
    )
    gone = sorted(n for n in registered if n not in sweep)
    assert not gone, (
        f"Registry lists flag(s) no longer on any live config object: {gone}. "
        "Remove them."
    )


def test_sd016_diversification_weight_is_an_open_defect():
    """Pins the OPEN defect, and fails loudly the moment it is repaired.

    This is the in-signature-but-never-assigned case (see module docstring).
    It is pinned in its BROKEN state rather than fixed here because
    `ree_core/utils/config.py` was under a concurrent claim; the repair is one
    line in `from_dims`, beside the other SD-016 assignments:

        config.sd016_diversification_weight = sd016_diversification_weight

    Verified 2026-08-22: with that line, `from_dims(sd016_diversification_weight
    =0.5)` yields 0.5 and the default stays 0.0 (so every unaffected driver is
    bit-identical). WHEN YOU LAND IT, this test fails on purpose -- delete it
    and drop `sd016_diversification_weight` from KNOWN_FROM_DIMS_DROP_SITES.
    """
    assert _build().sd016_diversification_weight == 0.0, (
        "the dataclass default moved; the repair is no longer bit-identical"
    )
    assert _build(sd016_diversification_weight=0.5).sd016_diversification_weight == 0.0, (
        "sd016_diversification_weight now lands -- the defect is REPAIRED. "
        "Delete this test and remove the flag from KNOWN_FROM_DIMS_DROP_SITES."
    )


# ---------------------------------------------------------------------------
# the call-site contract -- who is actually LOSING a value right now
# ---------------------------------------------------------------------------

def _drop_sites(sweep_result):
    """{name: {relpath}} for names passed into a from_dims forwarder and dropped.

    Type-agnostic on purpose: restricting this to bools would have missed
    sd016_diversification_weight (float) and harm_nonredundancy_weight (float).
    PARTIAL counts as well as UNREACHABLE -- see the wanting_weight entry.
    """
    at_risk = {n for n, (st, _l, _m) in sweep_result.items() if st in ("UNREACHABLE", "PARTIAL")}
    found = {}
    for path in sorted(EXPERIMENTS_DIR.rglob("*.py")):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Prefilter on TEXT before parsing. A SECOND prefilter on the at-risk
        # names themselves was written, measured and REMOVED: it cost ~9s of
        # `re.findall` and eliminated only 38 of the 1184 surviving files,
        # because the at-risk set contains generic names (`hidden_dim`,
        # `learning_rate`, `size`) present in nearly every driver. The forwarder
        # substring is the whole prefilter.
        if not any(w in src for w in FROM_DIMS_FORWARDERS):
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = (
                node.func.attr if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name)
                else None
            )
            if fname not in FROM_DIMS_FORWARDERS:
                continue
            for kw in node.keywords:
                if kw.arg in at_risk:
                    found.setdefault(kw.arg, set()).add(rel)
    return found


@pytest.fixture(scope="module")
def drop_sites(sweep):
    return _drop_sites(sweep)


def test_no_unregistered_from_dims_drop_site(drop_sites):
    """No NEW call site may pass a value into from_dims that from_dims loses."""
    unregistered = {
        name: sorted(paths - KNOWN_FROM_DIMS_DROP_SITES.get(name, {}).get("paths", set()))
        for name, paths in drop_sites.items()
    }
    unregistered = {k: v for k, v in unregistered.items() if v}
    assert not unregistered, (
        "These call sites pass a value into REEConfig.from_dims() (or a "
        f"forwarder) that from_dims SILENTLY DROPS: {unregistered}. The value "
        "never reaches the config, with no error -- so an arm that looks ablated "
        "may be identical to its control. Fix the call site to use the idiom "
        "that works for that flag (attribute assignment / direct sub-config "
        "construction / a profile method), or register it in "
        "KNOWN_FROM_DIMS_DROP_SITES with the measured blast radius."
    )


def test_drop_site_registry_has_no_phantom_paths(drop_sites):
    """A registered path that no longer passes the flag must be pruned."""
    stale = {
        name: sorted(entry["paths"] - drop_sites.get(name, set()))
        for name, entry in KNOWN_FROM_DIMS_DROP_SITES.items()
    }
    stale = {k: v for k, v in stale.items() if v}
    assert not stale, (
        f"KNOWN_FROM_DIMS_DROP_SITES lists path(s) that no longer pass the "
        f"flag into from_dims: {stale}. Prune them -- if a flag's whole path "
        "set is empty, the defect is fixed and the entry should go."
    )


# ---------------------------------------------------------------------------
# negative controls -- the assertions that stop this widening into noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag",
    [
        "use_harm_stream",              # SD-010, explicit signature entry
        "use_affective_harm_stream",    # SD-011
        "sd016_enabled",                # SD-016
        "use_mech307_conjunction",      # the 2026-08-07 repair
        "use_dualsystem_arbitration",   # SD-081, signature entry is load-bearing
        "use_iterative_inference",      # NOT in the signature: an explicit kwargs.pop
    ],
)
def test_known_good_flags_still_land(sweep, flag):
    """A flag from_dims really does plumb must never be reported unreachable."""
    assert flag in sweep, f"{flag} vanished from the config tree"
    status, lands, misses = sweep[flag]
    assert status == "LANDS", f"{flag} regressed to {status}; missed {misses}"


def test_alternative_idiom_flags_are_not_reported_as_defects(drop_sites):
    """A flag reachable ONLY by direct sub-config construction is NOT a defect.

    This is the assertion that pins the correction in
    chip-20260822-fromdims-swallowed-flag-audit's own scoping paragraph: the
    withdrawn chip called `use_curiosity_familiarity` unablatable, when it is
    ablated by `HippocampalConfig(...)`, which this repo accepts. Being
    unreachable through from_dims is a documentation gap; only a live call site
    LOSING a value is a defect.
    """
    for flag in REACHABLE_BY_ALTERNATIVE_IDIOM:
        assert flag not in KNOWN_FROM_DIMS_DROP_SITES, (
            f"{flag} is registered as both reachable-by-alternative and a live "
            "drop site; those are different findings, pick one"
        )
        assert flag not in drop_sites, (
            f"{flag} is now passed into from_dims somewhere and dropped -- move "
            "it to KNOWN_FROM_DIMS_DROP_SITES with the call sites"
        )


def test_wanting_weight_lands_on_its_own_target(sweep):
    """NEGATIVE CONTROL for the PARTIAL verdict: a name collision is not a defect.

    `wanting_weight` exists independently on HippocampalConfig (CEM scoring,
    default 0.0) and GhostGoalBankConfig (ghost priority, default 1.0).
    from_dims targets the hippocampal one and MUST keep landing there; the ghost
    bank's field MUST keep not moving. 'Repairing' the PARTIAL by plumbing
    from_dims into both would silently couple two unrelated knobs.
    """
    status, lands, misses = sweep["wanting_weight"]
    assert status == "PARTIAL", f"wanting_weight is now {status}"
    assert lands == ["hippocampal"], lands
    assert misses == ["hippocampal.ghost_goal_bank_config"], misses
    assert _build(wanting_weight=3.5).hippocampal.wanting_weight == 3.5
    assert _build(wanting_weight=3.5).hippocampal.ghost_goal_bank_config.wanting_weight == 1.0
