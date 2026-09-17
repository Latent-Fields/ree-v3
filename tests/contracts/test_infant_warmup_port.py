"""
infant_substrate:GAP-14 contract -- the ported 603n-style warm-up
(`experiments/_lib/infant_warmup.py`).

WHAT THIS PINS, AND WHY IT IS THE RIGHT THING TO PIN

GAP-14's prerequisite (b) ("the goal-pipeline training regime produces non-trivial
z_goal") was cleared on V3-EXQ-603n's harness, `scaffolded_sd054_onboarding`, and NEVER on
`InfantCurriculumScheduler` -- the object EXQ-ISEF-005 actually needs to exercise. These
tests assert (b) holds on the SCHEDULER'S OWN TERMS: not "z_goal is a biggish number", but
"the scheduler's Phase 1 -> 2 gate ADMITS the warmed agent's z_goal and REFUSES a cold
agent's", which is the decision the vacuous-null risk actually turns on.

PORTABILITY -- every assertion here is UPSTREAM of the discrete quantizer. Nothing asserts
a committed action, an action sequence, or any argmax/sample outcome: `torch.multinomial`
returns different categories on linux-x86_64 vs darwin-arm64 from a bit-identical
probability tensor at the same seed (CLAUDE.md "Running the test suite"), so a
committed-action pin would be a cross-machine flake, not a contract. The quantities pinned
are z_goal norms, observation widths, config flags and the scheduler's own phase integer.

COST -- C2/C3 run the z_goal-FORMING stages only (`include_hazard_stage=False`). Stage-H
trains harm/survival and writes no z_goal (it runs `seed_goal=False`), so it is not
load-bearing for (b), and including it would put hours into the contract suite. The full
Stage-0 -> Stage-0b -> P0 -> Stage-H path is exercised by the consuming experiment driver.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments._lib.infant_warmup import (  # noqa: E402
    INFANT_STRUCT_ENV_KWARGS,
    Z_GOAL_HANDOFF_FLOOR,
    build_warmed_agent,
    infant_env,
    run_warmup,
    warmup_config_slice,
    warmup_scaffold_config,
)
from infant_curriculum import (  # noqa: E402
    InfantCurriculumScheduler,
    PHASE_EP_MIN,
    Z_GOAL_THRESHOLD,
)
from scaffolded_sd054_onboarding import _build_env  # noqa: E402


def _fast_cfg():
    """Smallest config that still FORMS z_goal (Stage-0 + Stage-0b only)."""
    return warmup_scaffold_config(
        stage0_budget=2, stage0b_budget=2, p0_budget=1, hazard_budget=1,
        steps_per_episode=30, env_seed=1234,
    )


@pytest.fixture(scope="module")
def warmed():
    """One warmed agent, shared across the tests that only READ it."""
    torch.manual_seed(42)
    cfg = _fast_cfg()
    agent = build_warmed_agent(cfg, sleep_loop_episodes_K=10_000)
    result = run_warmup(agent, cfg, include_hazard_stage=False)
    return agent, result, cfg


# ------------------------------------------------------------------
# C1: the warm-up's envs and the infant curriculum's envs are dimension-
#     compatible -- in EVERY phase, with no zero-padding anywhere.
# ------------------------------------------------------------------

def test_c1_infant_env_matches_warmup_dims_in_every_phase():
    cfg = _fast_cfg()
    ref = _build_env(cfg, "p0", seed=0)
    ref.reset()
    ref_dims = (ref.body_obs_dim, ref.world_obs_dim, ref.action_dim)

    sched = InfantCurriculumScheduler(grid_size=12)
    for phase in range(4):
        env = infant_env(sched, phase=phase, seed=7)
        env.reset()
        assert (env.body_obs_dim, env.world_obs_dim, env.action_dim) == ref_dims, (
            "infant phase %d env is %s but the warm-up trains at %s; a warmed agent's "
            "encoders are fixed-width, so this mismatch would force zero-padding and "
            "move the agent off its training distribution"
            % (phase, (env.body_obs_dim, env.world_obs_dim, env.action_dim), ref_dims)
        )


def test_c1b_struct_kwargs_are_what_close_the_dim_gap():
    """The bare 591-lineage env does NOT match -- the structural kwargs are load-bearing.

    Guards against someone 'simplifying' INFANT_STRUCT_ENV_KWARGS away: without them the
    infant env is narrower, which is precisely the padding hazard C1 exists to forbid.
    """
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    sched = InfantCurriculumScheduler(grid_size=12)
    bare = CausalGridWorldV2(
        size=12, seed=7, resource_respawn_on_consume=True,
        pos_telemetry_enabled=True, traj_telemetry_enabled=True,
        **sched.env_kwargs(phase=0))
    bare.reset()
    rich = infant_env(sched, phase=0, seed=7)
    rich.reset()
    assert bare.world_obs_dim < rich.world_obs_dim
    assert bare.body_obs_dim < rich.body_obs_dim
    for key in ("limb_damage_enabled", "reef_enabled", "multi_resource_heterogeneity_enabled"):
        assert key in INFANT_STRUCT_ENV_KWARGS


# ------------------------------------------------------------------
# C2: prerequisite (b) -- the warm-up forms a z_goal that CLEARS the
#     scheduler's own Phase 1 exit threshold.
# ------------------------------------------------------------------

def test_c2_warmup_clears_the_phase1_z_goal_threshold(warmed):
    _agent, result, _cfg = warmed
    assert result.aborted_at is None, (
        "warm-up aborted at %s: %s" % (result.aborted_at, result.abort_reason))
    assert result.stage0_z_goal_formed is True
    assert result.z_goal_norm_at_exit >= Z_GOAL_THRESHOLD, (
        "z_goal at warm-up exit %.4f is below InfantCurriculumScheduler's Phase 1 exit "
        "gate %.2f -- prerequisite (b) is NOT cleared on this harness and any "
        "curriculum-vs-flat run would return a vacuous z_goal ~ 0 null"
        % (result.z_goal_norm_at_exit, Z_GOAL_THRESHOLD))
    # The module's own floor must not drift below the gate it exists to clear.
    assert Z_GOAL_HANDOFF_FLOOR >= Z_GOAL_THRESHOLD
    assert result.z_goal_cleared is True


def test_c2b_goal_pipeline_is_unfrozen_at_handoff(warmed):
    """The scaffold's run_p1 normally unfreezes; the curriculum replaces run_p1.

    A warmed agent handed over still-frozen would have its MECH-295 / MECH-307 goal
    writes short-circuited for the whole curriculum run -- z_goal would decay and never
    refresh, reproducing the vacuous null by a different route.
    """
    agent, _result, _cfg = warmed
    assert agent.config.use_mech295_liking_bridge is True
    assert agent.config.use_mech307_conjunction is True


# ------------------------------------------------------------------
# C3: the DECISION the warm-up exists to change -- the scheduler admits
#     the warmed z_goal and refuses a cold one.
# ------------------------------------------------------------------

def _drive_to_phase1(sched):
    """Advance 0 -> 1 on the hard episode count (h_pos=None path), no telemetry needed."""
    sched.update(PHASE_EP_MIN[1], h_pos=None)
    assert sched.current_phase == 1
    return sched


def test_c3_scheduler_admits_warmed_zgoal_and_refuses_cold(warmed):
    _agent, result, _cfg = warmed

    warm = _drive_to_phase1(InfantCurriculumScheduler(grid_size=12))
    warm.update(PHASE_EP_MIN[2], z_goal_norm=result.z_goal_norm_at_exit,
                benefit_contacts=10)
    assert warm.current_phase == 2, (
        "the scheduler refused a warmed z_goal of %.4f at the Phase 1 -> 2 gate"
        % result.z_goal_norm_at_exit)

    cold = _drive_to_phase1(InfantCurriculumScheduler(grid_size=12))
    cold.update(PHASE_EP_MIN[2], z_goal_norm=0.0, benefit_contacts=10)
    assert cold.current_phase == 1, (
        "a COLD agent's z_goal of 0.0 advanced past the Phase 1 -> 2 gate; if this ever "
        "passes, the gate is not the thing the warm-up is needed for and this whole "
        "contract is measuring nothing")


# ------------------------------------------------------------------
# C4: a cold agent really does sit at ~0 -- the warm-up is not decorative.
# ------------------------------------------------------------------

def test_c4_cold_agent_z_goal_is_trivial():
    torch.manual_seed(42)
    cfg = _fast_cfg()
    agent = build_warmed_agent(cfg, sleep_loop_episodes_K=10_000)  # built, NOT warmed
    cold_norm = float(agent.goal_state.goal_norm())
    assert cold_norm < Z_GOAL_THRESHOLD, (
        "a freshly-built agent already carries z_goal %.4f >= %.2f, so the warm-up would "
        "be unnecessary -- re-derive whether prerequisite (b) is still a real gap"
        % (cold_norm, Z_GOAL_THRESHOLD))


# ------------------------------------------------------------------
# C5: the warm-up is additive -- it changes no existing behaviour.
# ------------------------------------------------------------------

def test_c5_warmup_is_additive_not_a_scheduler_change():
    """InfantCurriculumScheduler must stay a pure, torch-free phase helper.

    The port was originally specified as a knob ON the scheduler wired through
    REEConfig.from_dims. That is not implementable -- the scheduler holds no agent,
    optimizer or trainable state -- and this test pins that it stays that way, so a later
    session does not re-attempt the same shape.
    """
    import inspect

    import infant_curriculum

    src = inspect.getsource(infant_curriculum)
    for forbidden in ("import torch", "optim.", ".backward(", "nn.Module"):
        assert forbidden not in src, (
            "infant_curriculum.py gained %r -- it is a pure phase-advancement helper; "
            "training belongs in the warm-up module, not here" % forbidden)

    sig = inspect.signature(InfantCurriculumScheduler.__init__)
    assert "agent" not in sig.parameters


def test_c6_config_slice_declares_only_what_the_warmup_reads():
    """Arm-reuse safety: the slice must not carry acceptance thresholds or arm labels.

    Checked over KEY NAMES, not a repr substring scan: 'arm' is a substring of
    'harm_pathway_lr' and 'z_harm_a_dim', both of which the warm-up genuinely reads and
    must declare. A substring scan false-positives on exactly the fields that belong here.
    """
    slice_ = warmup_config_slice(_fast_cfg())
    assert slice_["warmup"] == "infant_warmup/v1"
    assert slice_["env_struct"] == INFANT_STRUCT_ENV_KWARGS

    def _keys(obj, acc):
        if isinstance(obj, dict):
            for k, v in obj.items():
                acc.add(str(k).lower())
                _keys(v, acc)
        return acc

    keys = _keys(slice_, set())
    forbidden = ("arm_id", "arm_label", "arms", "acceptance", "pass_rule",
                 "criterion", "criteria", "verdict", "threshold_min", "min_fraction")
    leaked = sorted(k for k in keys if k in forbidden)
    assert not leaked, (
        "config slice leaked acceptance/arm-labelling key(s) %s -- the slice must declare "
        "only what the warm-up COMPUTATION reads, or a consumer's fingerprint stops "
        "matching the mint for reasons unrelated to the computation" % leaked)
    # The fields the warm-up really does read must be present and declared.
    for required in ("harm_pathway_lr", "steps_per_episode", "stage0_budget"):
        assert required in keys
