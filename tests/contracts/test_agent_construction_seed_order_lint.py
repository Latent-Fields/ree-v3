"""Contracts for the agent-construction-before-seed lint.

Surface under test: validate_experiments.agent_construction_before_seed_lint --
flags a module-level function that constructs `REEAgent(...)` before ever seeding
torch's global RNG in its own execution flow, so the agent's initial weights are
NOT a function of `seed`.

WHY THIS GATE EXISTS. `torch.nn.Module` weight init draws from torch's own global
RNG (`torch.manual_seed`), never from numpy's or Python's `random` -- seeding
either of those alone does nothing for weight reproducibility. A driver that
builds its agent (directly, or via a `make_agent(env)`-style wrapper) before ever
seeding torch gets weights that depend on process-level torch RNG history --
import order, prior random draws in the same process -- not on `seed`.

FOUND 2026-08-01 while building a positive-control regression test for
`experiments/_lib/q081_pair_reach_check.py::run_pair_specific_reach_probe`: that
function had exactly this bug (fixed, `02c155c658` -- `reset_all_rng(seed)` moved
before agent-template construction). A source-level corpus audit then found the
same shape in 18 driver scripts, most concentratedly the Q-081 family
(`v3_exq_824`/`824a`/`838`) and the INV-091 family (`v3_exq_827`/`827a`/`828`/
`828a`): each builds ONE shared P0 agent template per seed via `make_agent(...)`
BEFORE any `with arm_cell(seed, ...)` in the same `run_seed`/`_run_seed`. Audited
immaterial to each carrier's OWN reported finding -- every arm within a seed
shares that one template via `copy.deepcopy`, so the reported comparisons are
arm-matched regardless of what the shared (seed-uncontrolled) weights are; it
only breaks exact seed-to-seed reproducibility across separate runs. See
`REE_assembly/evidence/planning/q081_landmark_removal_arm_design.md` section 8
for the full audit.

SCOPE. Like every other WARN-only lint in this family, this gates NEW scripts.
Landed carriers' runs are complete and are NOT retro-edited, so the fire-rate
pin is a BACKLOG SIZE, not a target of zero.

TIER 1 ONLY -- see agent_construction_before_seed_lint()'s own docstring. This
does NOT attempt "is this script's agent seed-reproducible" in general (that
needs interprocedural analysis this lint deliberately does not do); it fires
only on the unambiguous, high-confidence shape: a real seed call exists in the
SAME function, but too late. A clean result here is not proof of
reproducibility -- only proof this specific defect is absent.

PRECISION PASS (2026-08-01, same day as the original 18-file audit). Triaging
all 18 for materiality (`REE_assembly/evidence/planning/
agent_seed_order_lint_backlog_triage.md`) surfaced two genuine FALSE-POSITIVE
shapes in the original Tier-1 design, both fixed here rather than merely
documented, because a lint whose fires require manual disambiguation before
they can be trusted is exactly the kind of noise that stops getting read:

  1. BRANCH-UNAWARENESS. The original walker merged an `if args.dry_run: ...;
     return` guard-clause branch's calls into the same flat timeline as the
     function's real path, even though the two are mutually exclusive
     execution paths (SD-016 418j/418k: the unseeded construction was
     confined to the dry-run smoke-test branch; the real run path was already
     correctly seeded). Fixed by isolating any guard-clause `if` -- one whose
     body unconditionally exits -- into its own scope, checked independently
     (`_DirectFlowWalker.visit_If` / `_is_guard_clause`).
  2. ONE-HOP-ONLY RESOLUTION vs. a "probe, discard, real-build-two-hops-away"
     idiom. A recurring author pattern (confirmed in 5 of the original 18:
     MECH-463 785/785a/787, ARC-003 804, ARC-016 805) builds a throwaway probe
     via a one-hop-visible ctor call (`probe_slice = _build(seed)[4]`, used
     only to harvest a static config dict) BEFORE `arm_cell`, while the REAL,
     scored agent is built by a second call two hops away
     (`_collect_cell` -> `_build`), invisible to one-hop resolution, INSIDE
     the (correctly resetting) `arm_cell` block. Fixed not by extending hop
     resolution (which would just make the lint blind in a different way) but
     by recognising the discard itself: a ctor call whose result is
     immediately subscripted for a POSITIVELY CONFIRMED non-agent tuple
     position doesn't count as an agent-construction event at all
     (`_is_discarded_agent_subscript` / `_find_agent_tuple_index`).

Both fixes are DELIBERATELY CONSERVATIVE in the false-negative direction: when
either cannot positively verify its precondition (an `if` that doesn't
unconditionally exit; a subscript whose extracted index or the ctor's return
shape can't be resolved), they do NOT suppress. A missed fire (silently
letting a genuine 3rd-shape bug through) would re-introduce exactly the
evidence-integrity risk this whole effort exists to close off -- worse than
occasionally still flagging a benign construction, which is what led to (1)
and (2) being noticed and fixed in the first place. See
`test_guard_clause_with_genuine_bug_inside_still_fires` and
`test_subscript_extracting_the_agent_itself_still_fires` below for the
negative controls proving the conservative direction actually holds, and
`test_real_688_construction_bug_still_fires` for the real-corpus negative
control (688's flagged construction is a genuine unseeded build, NOT a
discarded probe, and must keep firing).

Backlog count after the fix: 11 (was 18; 418j, 418k, 785, 785a, 787, 804, 805
cleared -- all seven were already independently triaged IMMATERIAL before this
fix, for reasons unrelated to whether the lint itself should have fired on
them; see the triage doc for the full per-script record, now annotated with
which precision gap explains each cleared fire).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so relative scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.agent_construction_before_seed_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) the confirmed shape: shared-P0-template family (824/824a/838/827x) --------

_SHARED_TEMPLATE_DEFECTIVE = '''
"""Synthetic replica of the Q-081/INV-091 shared-P0-template shape."""
import copy
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments._lib.arm_fingerprint import arm_cell


def make_agent(env):
    cfg = REEConfig.from_dims(world_dim=32)
    return REEAgent(cfg)


def run_seed(seed):
    env_p0 = make_env(seed)
    agent_p0 = make_agent(env_p0)          # <-- constructed before any seed call
    for arm_label in ("intact", "iei_permute"):
        with arm_cell(seed, config_slice={"arm": arm_label}) as cell:
            agent = copy.deepcopy(agent_p0)
            row = run_one_cell(agent, seed, arm_label)
            cell.stamp(row)
'''


def test_fires_on_shared_p0_template_shape():
    r = _lint_src(_SHARED_TEMPLATE_DEFECTIVE)
    assert r is not None
    assert "run_seed" in r
    assert "not seed-reproducible" in r


def test_fires_on_direct_reeagent_construction_before_manual_seed():
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run(seed):
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    torch.manual_seed(seed)
    return agent
'''
    r = _lint_src(src)
    assert r is not None
    assert "run" in r


# ---- (2) safe orderings: must NOT fire -------------------------------------------

def test_clean_when_reset_all_rng_precedes_direct_construction():
    src = '''
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments._lib.arm_fingerprint import reset_all_rng


def run(seed):
    reset_all_rng(seed)
    cfg = REEConfig.from_dims(world_dim=32)
    return REEAgent(cfg)
'''
    assert _lint_src(src) is None


def test_clean_when_torch_manual_seed_precedes_wrapper_call():
    """The false positive this lint's design specifically had to avoid: a helper
    (make_agent) is DEFINED earlier in the file (ordinary style: helpers first,
    driver below) than the function that correctly seeds before calling it.
    Confirmed real instance: v3_exq_453a_mech261_write_gate_landing.py."""
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def make_agent(env, condition):
    cfg = REEConfig.from_dims(world_dim=32)
    return REEAgent(cfg)


def run_condition(seed, condition):
    torch.manual_seed(seed)
    env = make_env(seed)
    agent = make_agent(env, condition)
    return agent
'''
    assert _lint_src(src) is None


def test_clean_when_arm_cell_precedes_construction():
    src = '''
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments._lib.arm_fingerprint import arm_cell


def run(seed):
    with arm_cell(seed, config_slice={}) as cell:
        cfg = REEConfig.from_dims(world_dim=32)
        agent = REEAgent(cfg)
        cell.stamp({"agent": "built"})
    return agent
'''
    assert _lint_src(src) is None


def test_clean_when_seeded_construct_helper_is_used():
    src = '''
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments._lib.arm_fingerprint import seeded_construct


def make_agent(env):
    cfg = REEConfig.from_dims(world_dim=32)
    return REEAgent(cfg)


def run(seed, env):
    agent = seeded_construct(seed, lambda: make_agent(env))
    return agent
'''
    assert _lint_src(src) is None


def test_dual_purpose_helper_correctly_ordered_does_not_cause_caller_false_positive():
    """Confirmed real instance: v3_exq_519a_sd051_conditioned_safety_store_readiness.py
    ::run_integration_arm seeds THEN constructs, correctly, within one function. A
    naive one-hop name resolution would classify the caller's single call to it as
    BOTH an agent event and a seed event at the same line -- an unresolvable tie
    that this lint must not read as a violation."""
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run_integration_arm(seed, arm_name):
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    return {"agent": agent}


def main():
    results = []
    for seed in [0, 1, 2]:
        r = run_integration_arm(seed, "arm_a")
        results.append(r)
    return results
'''
    assert _lint_src(src) is None


def test_dual_purpose_helper_itself_still_checked_when_internally_wrong():
    """Companion to the above: if the SAME dual-purpose helper gets the order
    wrong internally, it must still fire on its own account (independently
    scanned as a module-level function) even though a caller's call site is
    unresolvable."""
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run_integration_arm(seed, arm_name):
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    torch.manual_seed(seed)
    return {"agent": agent}
'''
    r = _lint_src(src)
    assert r is not None
    assert "run_integration_arm" in r


# ---- (2b) precision fix 1: guard-clause branch isolation ---------------------------

def test_clean_when_unseeded_construction_is_confined_to_a_dry_run_guard_clause():
    """Confirmed real shape: v3_exq_418k_sd016_context_memory_reef.py::main.
    An `if args.dry_run: ...; return` block builds an agent with no local seed
    call, but the REAL run path (reached only when the branch is NOT taken) is
    a separate, correctly-seeded call -- the two never execute together, so
    merging their events into one timeline was the false-positive source."""
    src = '''
import argparse
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def _make_agent(env):
    cfg = REEConfig.from_dims(world_dim=32)
    return REEAgent(cfg)


def _run_one_arm_seed(seed):
    torch.manual_seed(seed)
    env = make_env(seed)
    agent = _make_agent(env)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        env = make_env(42)
        agent = _make_agent(env)
        print("dry run", agent)
        return
    for seed in [0, 1, 2]:
        _run_one_arm_seed(seed)
'''
    assert _lint_src(src) is None


def test_guard_clause_with_genuine_bug_inside_still_fires():
    """Conservative-direction negative control: a guard-clause branch that
    itself has BOTH an agent construction and a too-late seed call is a
    genuine violation CONFINED to that branch, and must still fire -- guard-
    clause isolation checks each scope independently, it does not exempt
    branches from Tier 1 wholesale."""
    src = '''
import argparse
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        cfg = REEConfig.from_dims(world_dim=32)
        agent = REEAgent(cfg)
        torch.manual_seed(42)  # too late, inside this same isolated branch
        print("dry run", agent)
        return
    print("real run")
'''
    r = _lint_src(src)
    assert r is not None
    assert "main" in r


def test_non_exiting_if_branches_are_still_merged_as_before():
    """An `if`/`else` that does NOT unconditionally exit is not a guard clause
    (no isolation) -- both branches still merge into the same flow, matching
    pre-fix behaviour. Real branch-sensitive analysis for arbitrary if/else is
    out of scope; only the confirmed guard-clause shape is isolated."""
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run(seed, flag):
    if flag:
        cfg = REEConfig.from_dims(world_dim=32)
        agent = REEAgent(cfg)
    else:
        cfg = REEConfig.from_dims(world_dim=16)
        agent = REEAgent(cfg)
    torch.manual_seed(seed)  # too late for either branch's construction
    return agent
'''
    r = _lint_src(src)
    assert r is not None


# ---- (2c) precision fix 2: discarded-probe subscript detection --------------------

def test_clean_when_flagged_construction_is_a_discarded_probe_subscript():
    """Confirmed real shape (785/785a/787/804/805): a throwaway probe call
    extracts only a non-agent tuple element via `[N]` before `arm_cell`; the
    constructed REEAgent itself is immediately garbage. The REAL, scored agent
    is built two call-hops away (`_collect_cell` -> `_build`) inside the
    (correctly resetting) `arm_cell` block -- invisible to one-hop resolution,
    but the discard detection means there is no remaining agent event to
    wrongly pin on the probe."""
    src = '''
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments._lib.arm_fingerprint import arm_cell


def _build(seed):
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    env = make_env(seed)
    kw = {"world_dim": 32}
    return agent, env, {}, None, kw


def _collect_cell(seed):
    agent, env, obs, decoder, kw = _build(seed)
    return run_rollout(agent, env)


def run_experiment(seeds):
    for seed in seeds:
        probe_slice = _build(seed)[4]
        with arm_cell(seed, config_slice=probe_slice) as cell:
            rows = _collect_cell(seed)
            cell.stamp({"rows": rows})
'''
    assert _lint_src(src) is None


def test_subscript_extracting_the_agent_itself_still_fires():
    """Conservative-direction negative control: when the subscript's constant
    index DOES match the agent's real position in the returned tuple (proven
    via _find_agent_tuple_index), it is a genuine use, not a discard, and must
    still fire. Guards against over-generalising "subscript = discard". (A
    torch.manual_seed call after the extraction is included so this actually
    exercises the discard heuristic -- without ANY seed evidence at all the
    case would stay clean regardless, per Tier 1's separate "no local seed
    evidence" scope decision, and would not test what this test is for.)"""
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def _build(seed):
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    env = make_env(seed)
    return env, agent, {}


def run_experiment(seeds):
    for seed in seeds:
        agent = _build(seed)[1]  # index 1 IS the agent -- must not be discarded
        torch.manual_seed(seed)  # too late
        agent.eval()
'''
    r = _lint_src(src)
    assert r is not None


def test_subscript_with_unresolvable_ctor_shape_still_fires():
    """Conservative-direction negative control: when the ctor function's
    return shape can't be analysed (not a simple tuple literal), the subscript
    discard heuristic must NOT apply -- default to firing rather than
    silently trusting an unverifiable discard. (Seed call included for the
    same reason as the sibling test above -- otherwise "no local seed
    evidence" would explain a clean result regardless of the discard logic.)"""
    src = '''
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def _build(seed):
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    bundle = {"agent": agent, "env": make_env(seed)}
    return bundle  # not a tuple literal -- _find_agent_tuple_index can't resolve this


def run_experiment(seeds):
    for seed in seeds:
        probe = _build(seed)["agent"]
        torch.manual_seed(seed)  # too late
'''
    r = _lint_src(src)
    assert r is not None


def test_clean_when_no_agent_constructed():
    src = '''
def run(seed):
    return seed * 2
'''
    assert _lint_src(src) is None


def test_clean_when_agent_constructed_with_no_local_seed_evidence_at_all():
    """Scoped OUT on purpose (Tier 1 only, see module docstring) -- a function
    with no local seed call at all is common and often fine (many diagnostics
    never claim seed-driven weight reproducibility; the seeding may legitimately
    happen in a caller this single-function scan cannot see)."""
    src = '''
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def make_agent(env):
    cfg = REEConfig.from_dims(world_dim=32)
    return REEAgent(cfg)


def run(env):
    return make_agent(env)
'''
    assert _lint_src(src) is None


def test_arm_cell_with_do_reset_false_does_not_count_as_a_seed_event():
    src = '''
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments._lib.arm_fingerprint import arm_cell


def run(seed):
    with arm_cell(seed, config_slice={}, do_reset=False) as cell:
        cfg = REEConfig.from_dims(world_dim=32)
        agent = REEAgent(cfg)
        cell.stamp({})
    return agent
'''
    # arm_cell(do_reset=False) does not seed -- so this is the "no local seed
    # evidence at all" case, which Tier 1 deliberately does not fire on.
    assert _lint_src(src) is None


def test_np_random_seed_before_agent_does_not_satisfy_the_order_check():
    """The trap this lint exists to close precisely: np.random.seed/random.seed
    touch neither torch's global RNG nor agent weight init, so an EARLIER
    np.random.seed/random.seed call must NOT be read as discharging the
    ordering requirement when the only genuine torch seed call comes LATER --
    this must still fire, not be fooled into 'well, something ran seed() first'."""
    src = '''
import numpy as np
import random
import torch
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run(seed):
    np.random.seed(seed)
    random.seed(seed)
    cfg = REEConfig.from_dims(world_dim=32)
    agent = REEAgent(cfg)
    torch.manual_seed(seed)  # too late -- and the np/random calls above don't count
    return agent
'''
    r = _lint_src(src)
    assert r is not None, (
        "np.random.seed/random.seed appearing before the agent must NOT be read "
        "as covering torch's global RNG -- only the (too-late) torch.manual_seed "
        "call is a real seed event here, so this must still fire"
    )


# ---- (3) opt-out ------------------------------------------------------------------

def test_explicit_opt_out_is_honoured():
    src = (
        'AGENT_SEED_ORDER_EXEMPT = "deliberately shared, order-independent template"\n'
        + _SHARED_TEMPLATE_DEFECTIVE
    )
    assert _lint_src(src) is None


# ---- (4) message content ------------------------------------------------------------

def test_message_names_the_fix():
    r = _lint_src(_SHARED_TEMPLATE_DEFECTIVE)
    assert "seeded_construct" in r
    assert "AGENT_SEED_ORDER_EXEMPT" in r


# ---- (5) real corpus witnesses ------------------------------------------------------

def test_real_q081_838_is_the_detection_witness():
    p = EXPERIMENTS_DIR / "v3_exq_838_q081_cross_stream_recording.py"
    assert p.exists()
    r = V.agent_construction_before_seed_lint(p)
    assert r is not None
    assert "run_seed" in r


def test_real_q081_824_and_824a_fire():
    for name in (
        "v3_exq_824_q081_shared_organisation_landmark_removal.py",
        "v3_exq_824a_q081_shared_organisation_landmark_removal.py",
    ):
        p = EXPERIMENTS_DIR / name
        assert p.exists(), name
        assert V.agent_construction_before_seed_lint(p) is not None, name


def test_real_inv091_family_fires():
    for name in (
        "v3_exq_827_inv091_cross_stream_similarity_band.py",
        "v3_exq_827a_inv091_cross_stream_similarity_band_phase_sync.py",
        "v3_exq_828_inv091_cross_stream_similarity_band_remaining_ablations.py",
        "v3_exq_828a_inv091_cross_stream_similarity_band_null_validated.py",
    ):
        p = EXPERIMENTS_DIR / name
        assert p.exists(), name
        assert V.agent_construction_before_seed_lint(p) is not None, name


def test_real_453a_false_positive_case_stays_clean():
    """v3_exq_453a_mech261_write_gate_landing.py::_run_condition seeds correctly
    before calling its make_agent wrapper -- this is the case that motivated
    restricting name resolution to module-level functions only."""
    p = EXPERIMENTS_DIR / "v3_exq_453a_mech261_write_gate_landing.py"
    assert p.exists()
    assert V.agent_construction_before_seed_lint(p) is None


def test_real_519a_dual_purpose_helper_case_stays_clean():
    """v3_exq_519a_sd051_conditioned_safety_store_readiness.py::run_integration_arm
    seeds then constructs, correctly, in one function -- the case that motivated
    excluding dual-purpose helper names from one-hop resolution."""
    p = EXPERIMENTS_DIR / "v3_exq_519a_sd051_conditioned_safety_store_readiness.py"
    assert p.exists()
    assert V.agent_construction_before_seed_lint(p) is None


def test_real_q081_pair_reach_check_lib_module_is_not_scanned():
    """The lint is scoped (like its siblings) to the top-level v3_exq_*.py glob,
    not to _lib/ helper modules -- q081_pair_reach_check.py's OWN fix
    (02c155c658) lives in _lib/ and is covered by its own regression test
    (tests/contracts/test_q081_pair_reach_check.py
    ::test_pair_specific_reach_probe_seed_is_fully_deterministic_across_calls),
    not by this corpus lint. This test documents that scope boundary rather
    than asserting anything about the file's content."""
    p = EXPERIMENTS_DIR / "_lib" / "q081_pair_reach_check.py"
    assert p.exists()
    # v3_exq_849, the one caller of that function, correctly has no LOCAL
    # evidence of the bug (it calls an imported function) -- Tier 1 scope, not
    # a false negative this lint claims to catch.
    p849 = EXPERIMENTS_DIR / "v3_exq_849_q081_reach_preflight_scan.py"
    assert p849.exists()
    assert V.agent_construction_before_seed_lint(p849) is None


# ---- (5b) real corpus witnesses for the two precision fixes ------------------------

def test_real_418j_and_418k_no_longer_false_fire_on_the_dry_run_branch():
    """Precision fix 1 (guard-clause isolation): these were in the original
    18-file backlog ONLY because main()'s --dry-run branch was merged into the
    same flow as the real run path. Both scripts' real path
    (_run_one_arm_seed) was already correctly seeded (confirmed during the
    2026-08-01 triage) -- clearing the fire is a strict precision
    improvement, not a loosening of the rule."""
    for name in (
        "v3_exq_418j_sd016_context_memory_reef.py",
        "v3_exq_418k_sd016_context_memory_reef.py",
    ):
        p = EXPERIMENTS_DIR / name
        assert p.exists(), name
        assert V.agent_construction_before_seed_lint(p) is None, name


def test_real_mech463_and_arc_family_no_longer_false_fire_on_discarded_probes():
    """Precision fix 2 (discarded-probe subscript detection): all five were in
    the original 18-file backlog ONLY because a throwaway
    `probe_slice = _build(seed)[N]` before `arm_cell` was misread as the
    load-bearing construction. Each script's REAL, scored agent (built via
    _collect_cell inside arm_cell) was already correctly seeded (confirmed
    during the 2026-08-01 triage)."""
    for name in (
        "v3_exq_785_mech463_arousal_variance_amplifier_decomp.py",
        "v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py",
        "v3_exq_787_mech463_hazard_geometry_exogenous_proximity.py",
        "v3_exq_804_arc003_e3_selection_authority.py",
        "v3_exq_805_arc016_eval_derived_commit_threshold.py",
    ):
        p = EXPERIMENTS_DIR / name
        assert p.exists(), name
        assert V.agent_construction_before_seed_lint(p) is None, name


def test_real_688_construction_bug_still_fires():
    """Negative control for precision fix 2: 688's flagged construction
    (`test_agent = _build_agent(...)`, a P0-only readiness probe) is a GENUINE
    unseeded build directly assigned to a name and used
    (`_p0_readiness_checks(test_agent)`) -- not a discarded, immediately-
    subscripted probe. It must keep firing; the fix must not over-suppress
    real bugs that merely happen to also be immaterial for other reasons
    (688's arm comparison is separately, correctly seeded and never reached in
    the landed runs -- see the triage doc -- but that is a materiality
    finding, not something this lint should silently assume)."""
    p = EXPERIMENTS_DIR / "v3_exq_688_mech044_hippocampal_relational_binding.py"
    assert p.exists()
    r = V.agent_construction_before_seed_lint(p)
    assert r is not None
    assert "run_experiment" in r


# ---- (6) invariants: WARN-only, never blocks ---------------------------------------

def test_is_warn_only_under_strict_and_paths():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_SHARED_TEMPLATE_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "agent_seed_order", "--quiet", "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "AGENT-SEED-ORDER" in r.stdout
    finally:
        os.unlink(name)


def test_selector_runs_only_this_check():
    """--checks agent_seed_order must not also run unrelated corpus lints."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_SHARED_TEMPLATE_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "agent_seed_order", "--quiet", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "AGENT-SEED-ORDER" in r.stdout
        assert "DEAD-Z_GOAL" not in r.stdout.upper() or "dead-z_goal-stream-warning(s)" in r.stdout
    finally:
        os.unlink(name)


# ---- (7) corpus-wide pin -----------------------------------------------------------

# RE-PINNED 18 -> 11 on 2026-08-01, same day, deliberately: fixing the two
# precision gaps the materiality triage found (branch-unawareness; one-hop-only
# resolution missing the probe-then-real-build-two-hops-away idiom -- see the
# module docstring's PRECISION PASS section) cleared 7 of the original 18:
#   - 418j, 418k (SD-016): unseeded construction was confined to main()'s
#     --dry-run guard-clause branch; the real run path was already correct.
#   - 785, 785a, 787 (MECH-463), 804 (ARC-003), 805 (ARC-016): the flagged
#     construction was a discarded probe (`probe_slice = _build(seed)[N]`);
#     the real, scored agent (built inside arm_cell) was already correct.
# All seven were independently triaged IMMATERIAL before this fix (they never
# invalidated anything) -- the fix is about the LINT's own precision (fewer
# fires needing manual disambiguation going forward), not about un-flagging a
# risk. Original pin + full per-script triage (all 18, including these 7's
# materiality reasoning) preserved in
# REE_assembly/evidence/planning/agent_seed_order_lint_backlog_triage.md.
#
# The remaining 11: v3_exq_108, 615, 635, 688, 824, 824a, 827, 827a, 828, 828a,
# 838. All still TRUE positives (a real unseeded construction exists) and all
# still individually triaged IMMATERIAL -- via shared-object/deepcopy design
# (824/824a/827/827a/828/828a/838/635/108) or effect margin swamping plausible
# init variance (615), or (688) a genuine unseeded P0-only probe that never
# fed the separately-correct arm comparison. This is a BACKLOG SIZE, not a
# target -- all 11 have run and are deliberately NOT retro-edited (a completed
# run's pre-registered emission is not rewritten).
_PINNED_CORPUS_FIRE_COUNT = 11


def test_corpus_fire_rate_is_pinned(corpus_scan):
    # Shared corpus walk -- same file set, lint and order as an inline
    # comprehension would give; see tests/contracts/conftest.py.
    fired = corpus_scan["agent_construction_before_seed_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"agent-construction-before-seed fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix it (seed "
        f"with experiments/_lib/arm_fingerprint.seeded_construct, or move an existing "
        f"reset_all_rng/torch.manual_seed/arm_cell call earlier) rather than re-pinning. "
        f"If you deliberately widened or narrowed the rule, re-pin and say so in the "
        f"commit message. Fired: {[p.name for p in fired]}")
