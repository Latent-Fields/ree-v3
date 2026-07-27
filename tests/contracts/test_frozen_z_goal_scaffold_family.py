"""Contracts for the FROZEN (not dead) z_goal condition in the scaffold-warmed family.

WHAT THE CONDITION IS. ~28 experiment scripts drive their warmup through
`experiments/scaffolded_sd054_onboarding.py`'s `ScaffoldedSD054OnboardingScheduler`,
which calls `agent.update_z_goal(...)` on Stage-0 / P1 / P2 steps, and then hand the
warmed agent to a HAND-ROLLED measurement loop that never calls it again. Two substrate
facts make that consequential:

  - `REEAgent.reset()` resets ~47 subsystems but NOT `goal_state` (its only documented
    exception is residue). So the per-episode reset in the measurement loop does not
    clear the goal.
  - `GoalState`'s decay lives INSIDE `GoalState.update()`, reachable only through
    `update_z_goal`. No call -> no decay.

So z_goal FREEZES at its post-warmup value for the whole measurement phase.
`is_active()` stays True and every goal consumer keeps firing against a goal that no
longer tracks the episode. This is a DIFFERENT and milder defect than the dead-zero
stream gated by `validate_experiments.dead_z_goal_stream_lint`, which discharges this
family on purpose via `_uses_a_z_goal_driving_helper` -- read that docstring first.

WHY THE FREEZE IS NOT THE SCAFFOLD'S OWN "FROZEN GOAL PIPELINE". The scheduler has a
purpose-built primitive for holding the goal still, `_set_goal_pipeline_frozen(agent,
frozen)`, used by Stage-0b consolidation, P0 and Stage-H. It is a PAIR: skip
`update_z_goal` AND short-circuit the MECH-295 liking bridge + MECH-307 conjunction, so
the held goal cannot drive MECH-295/307. (CORRECTION, 2026-07-27 follow-on triage: this
file originally said the pair means "the held goal cannot drive behaviour". It does not.
The pair silences the goal WRITE paths only; the two goal READ paths -- the E3
`goal_weight * goal_proximity` term and E1 goal-conditioning -- are gated independently
and stay LIVE inside every frozen stage. Measured on the 460c dry-run config over 3
seeds: goal_active_frac 1.000 in Stage-0b / P0 / Stage-H, and removing the E3 goal term
counterfactually moves the cost-argmin candidate on 39% / 19% / 38% of those stages'
ticks. See `scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md` and the
`..._does_not_touch_the_read_paths` pin below. FOLLOW-ON, same day: the strict form the
triage chipped is now BUILT as an opt-in, default-OFF knob --
`ScaffoldedSD054OnboardingConfig.scaffold_strict_goal_isolation`, which makes the frozen
stages additionally zero `e3.goal_weight` and clear `e1_goal_conditioned`, restoring the
saved priors on unfreeze. Default False keeps every landed run bit-identical; the
`..._strict_goal_isolation_...` tests below pin both halves.) `run_p1` sets `frozen=False`
and nothing sets it
back, so the agent reaches the measurement phase with the goal held still but the
consumers LIVE -- a combination the scaffold itself never constructs. The measurement
phase inherits exactly half of the primitive, silently: no script in the family
documents its measurement-phase goal state at all.

TRIAGE OUTCOME (2026-07-27). Not a retro-fix. All 28 build the curriculum ONCE per seed
and evaluate every arm from a copy of that one build, so the frozen z_goal is
bit-identical across arms and cannot produce a between-arm difference. Of the 28 landed
manifests, 22 are `non_contributory`, 3 `superseded`, 1 `mixed` (diagnostic), 3 carry no
direction (diagnostic), and exactly ONE -- V3-EXQ-466e, SD-034 -- is `supports`. 466e's
criteria are existence thresholds on the ON arm (n_closures >= 1, discharge_events >= 1)
plus a structural negative control on an OFF clone that has no closure operator at all,
none of which reads z_goal; the frozen goal is arm-symmetric there exactly as the inert
goal term is arm-symmetric for V3-EXQ-615 in the dead-stream lint's own carrier table.

RETROFIT IS NOT FREE. Adding `update_z_goal` to one of these scripts is not a wiring
fix: the call is ALSO the SD-024 benefit-attractor producer (it calls
`ResidueField.accumulate_benefit` ahead of the `goal_state` guard), so it populates
`benefit_rbf_field` and un-zeroes the SD-025 curiosity bonus in
`HippocampalModule._curiosity_bonus`. For THIS family that specific path is gated off --
`residue.benefit_terrain_live_producer` defaults False and none of the 28 (nor the
scheduler) sets it -- but the retrofit still swaps a constant goal for 0.5%/step decay
plus contact reseeding, which moves the E3 goal term on every tick. Either way a patched
script is not comparable to the runs that came before it.
"""
import ast
import sys
from pathlib import Path

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCAFFOLD = EXPERIMENTS_DIR / "scaffolded_sd054_onboarding.py"


def _make_agent(**overrides):
    kwargs = dict(
        body_obs_dim=12,
        world_obs_dim=54,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        z_goal_enabled=True,
    )
    kwargs.update(overrides)
    return REEAgent(REEConfig.from_dims(**kwargs))


# ---- (1) the behavioural fact the whole triage rests on ----------------------------
# The sibling AST assertion in test_dead_z_goal_stream_lint.py pins that
# `REEAgent.reset()` contains no `goal_state.reset()` CALL. That is a source-shape
# check; this is the RUNTIME consequence, which is what the family actually depends on.

def test_agent_reset_leaves_a_seeded_z_goal_bit_identical():
    agent = _make_agent()
    assert agent.goal_state is not None

    z_world = torch.ones(1, agent.config.latent.world_dim)
    agent.goal_state.update(
        z_world_current=z_world, benefit_exposure=1.0, drive_level=1.0
    )
    seeded = agent.goal_state.z_goal.clone()
    assert agent.goal_state.is_active(), "seed did not fire -- test premise broken"

    for _ in range(5):
        agent.reset()

    assert torch.equal(agent.goal_state.z_goal, seeded), (
        "REEAgent.reset() now changes z_goal. The scaffold-warmed family's measurement "
        "phase depends on the goal surviving the per-episode reset unchanged; if this "
        "flipped, the family is no longer FROZEN and this whole contract file plus "
        "_uses_a_z_goal_driving_helper's discharge need re-reading.")
    assert agent.goal_state.is_active()


def test_z_goal_does_not_decay_without_update_z_goal():
    """Decay lives inside GoalState.update -- the second half of the freeze."""
    agent = _make_agent()
    z_world = torch.ones(1, agent.config.latent.world_dim)
    agent.goal_state.update(
        z_world_current=z_world, benefit_exposure=1.0, drive_level=1.0
    )
    seeded = agent.goal_state.z_goal.clone()

    # A measurement loop calls reset() per episode and never calls update_z_goal.
    for _ in range(50):
        agent.reset()
    assert torch.equal(agent.goal_state.z_goal, seeded)

    # ...whereas a single decay-only update DOES move it, which is the counterfactual
    # the family's measurement phase never applies.
    agent.goal_state.update(
        z_world_current=z_world, benefit_exposure=0.0, drive_level=0.0
    )
    assert not torch.equal(agent.goal_state.z_goal, seeded)


# ---- (2) the frozen goal is NOT inert ---------------------------------------------

def test_frozen_goal_still_drives_e3_for_the_family_config():
    """`goal_weight` resolves LIVE for this family, so a stale goal biases selection.

    None of the 28 sets `goal_weight`, and `E3Config.goal_weight`'s dataclass default is
    0.0 -- which reads as "the goal term is off, so who cares if z_goal is stale". That
    reading is wrong: `REEConfig.from_dims` carries its OWN default of 1.0 and assigns
    `config.e3.goal_weight`, so the E3 goal term (gated on `goal_weight > 0` AND
    `goal_state.is_active()`) fires on every E3 tick of the measurement phase.
    """
    agent = _make_agent()
    assert agent.e3.config.goal_weight > 0.0
    assert agent.config.goal.e1_goal_conditioned is True


# ---- (3) the scaffold's own freeze primitive is a PAIR ------------------------------

def _scaffold_tree():
    return ast.parse(SCAFFOLD.read_text(encoding="utf-8"))


def test_goal_pipeline_freeze_helper_silences_the_consumers():
    """`_set_goal_pipeline_frozen(frozen=True)` must also short-circuit MECH-295/307.

    This pairing is what stops a held goal from driving the MECH-295 liking bridge and
    the MECH-307 conjunction during the scaffold's own freeze windows. If a future edit
    drops this half, those stages lose even that much. It is NOT full goal isolation --
    see `test_goal_pipeline_freeze_does_not_touch_the_read_paths` immediately below,
    which pins the deliberate scope limit.
    """
    tree = _scaffold_tree()
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_set_goal_pipeline_frozen"),
        None)
    assert fn is not None, "_set_goal_pipeline_frozen not found in the scheduler"

    written = {
        t.attr
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Attribute)
    }
    assert "use_mech295_liking_bridge" in written
    assert "use_mech307_conjunction" in written


def _scaffold_module():
    """Import the scaffold module (not just its AST) for behavioural assertions."""
    import scaffolded_sd054_onboarding as S  # noqa: E402  (sys.path set above)
    return S


def test_goal_pipeline_freeze_does_not_touch_the_read_paths():
    """The freeze's DEFAULT scope limit, pinned as a deliberate decision.

    (2026-07-27 triage; AMENDED 2026-07-27 when the opt-in strict knob landed. The
    assertion used to be an AST equality on the helper's write set -- writes are
    EXACTLY the two MECH flags. That equality is now wrong by design: strict mode adds
    gated writes to the two read paths. What must still hold, and is what the triage
    actually cared about, is that the DEFAULT path -- `strict` unset, i.e. every one of
    the 78 landed scaffold importers -- leaves both read paths untouched. That is now
    asserted BEHAVIOURALLY, which is strictly stronger than the old source-shape check:
    an AST equality could be satisfied by a helper that reached the same flags through
    a helper call, whereas this fails unless the values are genuinely unchanged.)

    So: `_set_goal_pipeline_frozen(agent, frozen=True)` with no `strict=` argument does
    NOT zero `E3Config.goal_weight` and does NOT clear `GoalConfig.e1_goal_conditioned`.
    The E3 `goal_weight * goal_proximity` term and E1 goal-conditioning stay live inside
    every "frozen" stage once Stage-0 has seeded z_goal -- which is why
    `run_hazard_avoidance`'s docstring no longer claims survival is learned "without the
    goal pipeline".

    Widening the DEFAULT remains rejected: it changes E3 selection in three stages for
    all 78 scaffold importers and breaks comparability with every landed scaffold run,
    and no landed manifest's recorded conclusion rests on strict isolation. The strict
    form is opt-in per experiment (`scaffold_strict_goal_isolation`), never a default.
    """
    S = _scaffold_module()
    agent = _make_agent()

    # Premise: both read paths live before the freeze (the family-config fact).
    assert agent.e3.config.goal_weight > 0.0
    assert agent.config.goal.e1_goal_conditioned is True
    goal_weight_before = float(agent.e3.config.goal_weight)

    # The DEFAULT call -- exactly what every landed scaffold run executes.
    S._set_goal_pipeline_frozen(agent, frozen=True)

    assert agent.config.use_mech295_liking_bridge is False
    assert agent.config.use_mech307_conjunction is False
    assert agent.e3.config.goal_weight == goal_weight_before, (
        "the DEFAULT freeze path now zeroes e3.goal_weight. That is a behaviour change "
        "for all 78 scaffold importers and breaks comparability with every landed "
        "scaffold run -- the strict form must stay opt-in via "
        "scaffold_strict_goal_isolation. See this file's docstring.")
    assert agent.config.goal.e1_goal_conditioned is True, (
        "the DEFAULT freeze path now clears e1_goal_conditioned -- same objection as "
        "for goal_weight above.")
    assert not hasattr(agent, "_scaffold_strict_goal_isolation_saved"), (
        "the DEFAULT freeze path created strict-isolation save state; it must not "
        "enter strict mode at all.")

    # The write set is still confined to a known allowlist, so an unrelated new
    # mutation cannot ride in unremarked. (Subset, not equality: the strict-only
    # writes are legitimate members.)
    tree = _scaffold_tree()
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_set_goal_pipeline_frozen"),
        None)
    assert fn is not None
    written = {
        t.attr
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Attribute)
    }
    assert written == {"use_mech295_liking_bridge", "use_mech307_conjunction"}, (
        f"_set_goal_pipeline_frozen's own write set changed to {sorted(written)}. The "
        "strict-mode read-path writes live in _enter_strict_goal_isolation / "
        "_exit_strict_goal_isolation, NOT inline here -- keeping them out of this "
        "function is what makes the default path auditable at a glance.")


def test_default_freeze_path_is_equivalent_to_the_pre_knob_helper():
    """Bit-identity of the DEFAULT path, proven by equivalence rather than argued.

    The pre-knob helper body was exactly two assignments (below, verbatim). This runs
    the new helper on one agent and that replica on a seed-identical twin, then compares
    everything a curriculum could possibly read downstream: every goal-relevant config
    field, the full parameter state_dict bitwise, and the torch / numpy / stdlib-random
    RNG states. If the added code consumed a single RNG draw or touched one byte of
    state, the streams would diverge from the next tick onward and this fails.

    (Why not an end-to-end curriculum A/B instead: the 460c dry-run curriculum is NOT
    reproducible across processes -- two byte-identical checkouts diverge at Stage-0 even
    with torch, numpy AND stdlib random seeded -- so a run-vs-run diff cannot resolve a
    no-op change. Measured on ree-cloud-2, 2026-07-27.)
    """
    import random as _random

    import numpy as _np

    S = _scaffold_module()

    def _pre_knob_replica(agent, frozen):
        if frozen:
            agent.config.use_mech295_liking_bridge = False
            agent.config.use_mech307_conjunction = False
        else:
            agent.config.use_mech295_liking_bridge = True
            agent.config.use_mech307_conjunction = True

    def _fingerprint(agent):
        return {
            "goal_weight": float(agent.e3.config.goal_weight),
            "e1_goal_conditioned": bool(agent.config.goal.e1_goal_conditioned),
            "mech295": bool(agent.config.use_mech295_liking_bridge),
            "mech307": bool(agent.config.use_mech307_conjunction),
            "params": {
                k: v.detach().cpu().numpy().tobytes()
                for k, v in sorted(agent.state_dict().items())
                if hasattr(v, "detach")
            },
            "torch_rng": torch.get_rng_state().numpy().tobytes(),
            "np_rng": repr(_np.random.get_state()),
            "py_rng": repr(_random.getstate()),
        }

    prints = []
    for apply_freeze in (S._set_goal_pipeline_frozen, _pre_knob_replica):
        torch.manual_seed(1234)
        _np.random.seed(1234)
        _random.seed(1234)
        agent = _make_agent()
        # Both freeze and unfreeze, in the order a curriculum uses them.
        apply_freeze(agent, frozen=True)
        apply_freeze(agent, frozen=False)
        apply_freeze(agent, frozen=True)
        prints.append(_fingerprint(agent))

    new_fp, old_fp = prints
    assert new_fp["params"] == old_fp["params"], "a parameter tensor changed"
    for key in ("goal_weight", "e1_goal_conditioned", "mech295", "mech307",
                "torch_rng", "np_rng", "py_rng"):
        assert new_fp[key] == old_fp[key], (
            f"the DEFAULT freeze path diverged from the pre-knob helper on {key!r}: "
            f"{new_fp[key]!r} vs {old_fp[key]!r}. The knob must be bit-identical when "
            "scaffold_strict_goal_isolation is unset.")


# ---- (3b) the OPT-IN strict form (2026-07-27) ---------------------------------------
# The knob the triage chipped: a future experiment that genuinely needs a goal-free
# Stage-H can now get one, without moving the default for anybody else.

def test_strict_goal_isolation_defaults_off():
    S = _scaffold_module()
    cfg = S.ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_strict_goal_isolation is False, (
        "scaffold_strict_goal_isolation must default False. Flipping this default "
        "silently changes E3 selection in three stages for all 78 scaffold importers.")


def test_strict_goal_isolation_silences_both_read_paths():
    """strict=True must silence BOTH read paths, not just the E3 one."""
    S = _scaffold_module()
    agent = _make_agent()
    assert agent.e3.config.goal_weight > 0.0
    assert agent.config.goal.e1_goal_conditioned is True

    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)

    # E3: the gate is `goal_weight > 0.0`, so zero SKIPS the term rather than
    # scaling it -- compute_goal_score is not called at all.
    assert agent.e3.config.goal_weight == 0.0
    # E1: sense() then passes z_goal=None, the same path E1 takes when the goal
    # is inactive.
    assert agent.config.goal.e1_goal_conditioned is False
    # The write paths are still frozen -- strict is additive, not a replacement.
    assert agent.config.use_mech295_liking_bridge is False
    assert agent.config.use_mech307_conjunction is False


def test_strict_goal_isolation_restores_the_saved_prior_values():
    """Unfreeze restores what was there, NOT a hardcoded 1.0/True.

    An experiment may set a non-default goal_weight; restoring 1.0 would silently
    rewrite its config mid-curriculum.
    """
    S = _scaffold_module()
    agent = _make_agent()
    agent.e3.config.goal_weight = 0.37  # deliberately non-default
    agent.config.goal.e1_goal_conditioned = True

    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)
    assert agent.e3.config.goal_weight == 0.0

    S._set_goal_pipeline_frozen(agent, frozen=False, strict=True)
    assert agent.e3.config.goal_weight == 0.37, (
        "unfreeze did not restore the SAVED goal_weight -- a non-default value set by "
        "the experiment was overwritten.")
    assert agent.config.goal.e1_goal_conditioned is True
    assert not hasattr(agent, "_scaffold_strict_goal_isolation_saved")


def test_strict_goal_isolation_is_idempotent_and_unfreeze_needs_no_strict_flag():
    """Double-freeze must not save the zeroed value over the real one, and an
    unfreeze that forgets strict=True must still restore (the saved state, not the
    caller's flag, drives the restore)."""
    S = _scaffold_module()
    agent = _make_agent()
    original = float(agent.e3.config.goal_weight)

    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)
    S._set_goal_pipeline_frozen(agent, frozen=True, strict=True)  # second freeze
    assert agent.e3.config.goal_weight == 0.0

    S._set_goal_pipeline_frozen(agent, frozen=False)  # note: no strict=
    assert agent.e3.config.goal_weight == original
    assert agent.config.goal.e1_goal_conditioned is True

    # And an unfreeze with no prior strict freeze is a no-op, not an exception.
    S._set_goal_pipeline_frozen(agent, frozen=False)
    assert agent.e3.config.goal_weight == original


def test_every_freeze_call_site_threads_the_strict_knob():
    """The knob is useless if a stage forgets to pass it -- pin all call sites.

    Stage-0b / P0 / Stage-H are the frozen stages that must silence the read paths;
    run_stage0_nursery / run_p1 unfreeze and must restore them. All five read the
    same cfg field, so no stage can end up half-isolated.
    """
    tree = _scaffold_tree()
    seen = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef):
            continue
        for c in ast.walk(n):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "_set_goal_pipeline_frozen"):
                kws = {kw.arg: kw.value for kw in c.keywords}
                strict = kws.get("strict")
                seen[n.name] = (
                    ast.unparse(strict) if strict is not None else None)

    assert set(seen) == {
        "run_stage0_nursery", "run_stage0b_consolidation", "run_p0",
        "run_hazard_avoidance", "run_p1",
    }, f"freeze call sites moved: {sorted(seen)}"
    for name, expr in seen.items():
        assert expr == "self.cfg.scaffold_strict_goal_isolation", (
            f"{name} passes strict={expr!r}; every call site must thread the config "
            "knob, or a stage silently keeps the goal read paths live while its "
            "siblings silence them.")


def test_scaffold_hands_off_with_the_goal_consumers_unfrozen():
    """The half-inherited freeze, pinned as a fact rather than left to be rediscovered.

    `run_p1` is the last stage to set the freeze state and it UNFREEZES; `run_p2` does
    not touch it. So the measurement loop receives an agent whose z_goal is held still
    (nobody calls update_z_goal) while MECH-295/307 are live. If a future edit makes
    run_p2 re-freeze, or makes run_p1 leave it frozen, this triage's conclusion changes
    and the docstring above needs revisiting.
    """
    tree = _scaffold_tree()
    methods = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for c in ast.walk(n):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                        and c.func.id == "_set_goal_pipeline_frozen"):
                    for kw in c.keywords:
                        if kw.arg == "frozen":
                            methods.setdefault(n.name, []).append(
                                ast.literal_eval(kw.value))

    assert methods.get("run_stage0b_consolidation") == [True]
    assert methods.get("run_p0") == [True]
    assert methods.get("run_hazard_avoidance") == [True]
    assert methods.get("run_p1") == [False], (
        "run_p1 no longer unfreezes the goal pipeline -- re-read this file's docstring")
    assert "run_p2" not in methods, (
        "run_p2 now sets the freeze state; the hand-off condition to the measurement "
        "phase has changed")


# ---- (4) family membership ---------------------------------------------------------
# Re-derived 2026-07-27 rather than transcribed: a scaffold importer that enables a
# z_goal-dependent knob, never writes z_goal itself, and hand-rolls a measurement loop
# (its own `select_action` + env `step`, as opposed to delegating every step to the
# scheduler). The 22 other scaffold importers that enable z_goal do NOT hand-roll a loop
# -- the scheduler drives every step for them, so it keeps calling update_z_goal and
# they never freeze.
_FROZEN_FAMILY_SIZE = 28


def test_frozen_z_goal_family_size_is_pinned():
    fam = []
    for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "ScaffoldedSD054OnboardingScheduler" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if not V._sets_knob_truthy(tree, V._DEAD_ZGOAL_TRIGGER_KNOBS):
            continue
        if V._writes_z_goal_directly(tree):
            continue
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        if {"select_action", "step"} <= calls:
            fam.append(p.name)

    assert len(fam) == _FROZEN_FAMILY_SIZE, (
        f"frozen-z_goal family size moved: {len(fam)} vs pinned {_FROZEN_FAMILY_SIZE}. "
        f"A NEW member means a new script inherited the half-freeze -- have it choose "
        f"and DOCUMENT its measurement-phase goal state (drive the goal, re-freeze the "
        f"pipeline via _set_goal_pipeline_frozen, or state that a fixed goal is "
        f"intended). Note the SD-024 side effect before adding update_z_goal. "
        f"Members: {fam}")
