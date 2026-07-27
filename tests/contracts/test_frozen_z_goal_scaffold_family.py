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
the held goal cannot drive behaviour. `run_p1` sets `frozen=False` and nothing sets it
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

    This pairing is what makes the scaffold's in-curriculum freeze windows a genuine
    isolation rather than "act on a stale goal". If a future edit drops the consumer
    half, the scaffold's own frozen stages silently acquire the same defect the
    measurement phase has.
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
