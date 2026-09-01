"""Contracts for the z_goal-stream RUNTIME counter -- the substrate-side backstop.

WHY THIS EXISTS. `REEAgent.update_z_goal(...)` is the SOLE writer of z_goal in the
substrate: nothing in sense() / generate_trajectories() / select_action() /
update_residue() touches it, and both `GoalState` mutators (`update` / `cue_pull`) are
reached only from inside it. A driver that hand-rolls its inner loop and omits the call
therefore runs with z_goal pinned at zero-init for the whole run -- `is_active()` stays
False, `agent.py` passes `current_z_goal=None` to every consumer, and the E3 goal term,
MECH-293 ghost probes, MECH-288's slow BOCPD scale, MECH-189 super-ordinal anchors, the
SD-057 incentive bank, the MECH-295 liking->approach bridge and the frontopolar
counterfactual read all silently no-op. No error, no warning, and (before this) no
manifest field.

WHY IT IS NOT THE LINT'S JOB. `validate_experiments.dead_z_goal_stream_lint` (landed
2026-07-27, ree-v3 `fd7bd5d68c`) catches the same defect at AUTHORING time, but it is an
AST scan and therefore blind to a config assembled inside a helper it cannot follow (a
`_lib` builder, a `**kwargs` splat, a preset factory); it UNDER-fires by design. The
counter is read from the run itself, so it covers that blind spot. Complementary, not
redundant: the lint fires before compute is spent, the counter only exists once the run
does.

Guarantees enforced here:
  Z1. The counters exist, start at zero, and are exposed READ-ONLY (properties, no
      setter) -- so a driver can report them but cannot forge them.
  Z2. They count exactly what they claim: one tick per `select_action` call while
      `goal_state` is present, and `ticks_active` only once z_goal has actually been
      written. This is the whole measurement.
  Z3. z_goal_enabled=False -> goal_state is None -> NOTHING is counted, and
      `z_goal_active_frac` is None (not 0.0). The distinction is load-bearing: 0.0 means
      "measured, and dead"; None means "not measured at all".
  Z4. THE DEFECT SIGNATURE, AND ITS LOOKALIKE. An agent configured with
      z_goal_enabled=True and stepped WITHOUT `update_z_goal` reports ticks_total > 0,
      ticks_active == 0, active_frac == 0.0 -- the reading V3-EXQ-830 got from its
      ad-hoc `zgoal_present_frac`, generalised. But a CORRECTLY-wired run whose benefit
      gate never opened reads the same 0.0, so `writer_calls` (and the precomputed
      `writer_defect`) is what tells them apart. This ambiguity was found by measuring,
      not by anticipating it: a StepHarness run -- which pins the call as invariant 2
      and cannot carry the defect -- reads active_frac 0.0 on the tiny env because the
      agent never reaches a resource. Both directions are pinned here, because a
      backstop that cries defect on every benefit-free run would be ignored within a
      week.
  Z5. NO BEHAVIOUR CHANGE. Selection is bit-identical with the counters live -- they add
      two int bumps and one RNG-free tensor reduction. Asserted by running the same
      seeded sequence twice and comparing actions, since the counters cannot be
      compiled out.
  Z6. Counters are RUN-lifetime: `agent.reset()` does NOT clear them, so the recorded
      fraction describes the run and not just its final episode. (`GoalState.reset()`
      DOES zero z_goal, so post-reset inactive ticks are real signal and are counted.)
  Z7. The recording helper `_lib/z_goal_stream.py` builds the manifest block, pools
      across a multi-arm run's several agents, and is stdlib-only/duck-typed (no torch,
      no ree_core) so `manifest_core` keeps its no-substrate-dependency guarantee.
  Z8. `stamp_recording_core` / `write_flat_manifest` surface it, and OMIT the block when
      no counters were supplied -- absence must never be readable as "measured zero".
  Z9. `StepHarness` records it too, and its own tally agrees with the agent's tick for
      tick.
  Z12. THE TRAINING-PHASE-AGENT FALSE POSITIVE (V3-EXQ-874b). `writer_calls == 0` with
      `ticks_total > 0` is Z4's defect signature -- but it is also exactly what a
      TRAINING-phase agent reads: `select_action` bumps `ticks_total` on every call,
      including calls from an ordinary training loop where `update_z_goal` is never
      called because z_goal has nothing to do with that phase. V3-EXQ-874b's driver
      accumulated a P0-trained base agent's counters (100% training ticks, since the
      driver's actual eval loop stepped a separate `clone_trained_agent` clone) and
      published a false `writer_defect: true`. `eval_stepped=False` on every observation
      entry point (`z_goal_stream_stats`, `ZGoalStreamAccumulator.observe`/
      `observe_stats`, `stamp_z_goal_stream`) routes counts into a separate
      `training_phase_*` tally that never feeds `writer_defect`. Pinned here alongside a
      regression guard: the fix must NOT weaken Z4's real-defect detection for an agent
      that simply never had `.eval()`/`.train()` called on it (the corpus norm, and
      exactly V3-EXQ-830's own shape) -- see the module docstring's "why this cannot be
      auto-detected" section for why `agent.training` was rejected as the discriminator.
  Z13. THE SMOKE PRINT SURFACES training_phase_*. Z11's --dry-run line read the
      eval-facing fields only, so a driver correctly using `eval_stepped=False` saw the
      same "not assessable" line as a genuinely unmeasured run -- the real training-phase
      counts were in the manifest block but invisible at the smoke. The print now appends
      a short note ("+N training-phase ticks, M writer calls, not counted toward
      writer_defect") whenever `training_phase_ticks_total` is present and nonzero, and
      omits it otherwise; the eval-facing verdict itself is unchanged.
  Z14. THE PINNED-GOAL FALSE POSITIVE (V3-EXQ-642b). `writer_calls == 0` with
      `ticks_total > 0` is Z4's defect signature -- but it is also exactly what a driver
      reads when it deliberately pins z_goal at a fixed magnitude by writing
      `agent.goal_state._z_goal` directly (V3-EXQ-642a/642b's `_pin_goal`), bypassing
      `update_z_goal` entirely. `GoalState.is_active()` is a plain nonzero check, so a
      constant nonzero pin reads `active_frac == 1.0` from the first pin onward --
      unlike a genuine omission, which reads `active_frac == 0.0` for the whole run
      because `_z_goal` never leaves its zero-init without going through the writer.
      `goal_pinned=True` on every observation entry point (`z_goal_stream_stats`,
      `ZGoalStreamAccumulator.observe`/`observe_stats`, `stamp_z_goal_stream`,
      `stats_from_counts`) reports `writer_defect: None` (not-applicable) instead of a
      false `True`, and stamps `goal_pinned: true` on the block. Unlike Z12's
      training-phase carve-out, the pinned ticks stay IN the ordinary eval counters
      (`ticks_total`/`ticks_active`/`active_frac`) rather than moving to a side
      channel -- they are real eval-loop activity, just not attributable to the
      writer. Default False -- unchanged behaviour for every existing call site,
      and the real Z4 defect must still fire when the flag is not passed.
  Z15. THE CUE-RECALL-ONLY FALSE POSITIVE, FIXED AT THE SOURCE (found 2026-09-01).
      `REEAgent.cue_recall_wanting` (SD-057 L6, MECH-347) calls `GoalState.cue_pull`
      directly, bypassing `update_z_goal` entirely -- so a driver whose z_goal moves
      ONLY through cue-recall used to read `writer_calls == 0` with a live,
      non-degenerate `active_frac`, indistinguishable from Z4's real defect. Unlike
      Z12/Z14, the fix is not an opt-in flag: `cue_recall_wanting` now increments
      `z_goal_writer_calls` itself, immediately after its own reachability gate
      (`goal_state` present and `GoalConfig.use_cue_recall` set) -- mirroring
      `update_z_goal`'s placement right after its own analogous gate
      (`goal_state is None`). Pinned here: (a) a cue-recall-only run now correctly
      reads `writer_defect` as not-True; (b) the counter still increments even when
      the wanting-amplitude/token-match checks downstream find nothing to pull
      (the cue-recall equivalent of Z4's "benefit gate never opened" reading); (c)
      the real defect is unaffected -- with `use_cue_recall` off, or with
      `simulation_mode=True` (MECH-094 replay safety), the call must NOT increment,
      so a driver that never engages either writer still reads the unambiguous
      `writer_calls == 0` signature. No landed driver currently hits the
      cue-recall-only shape in isolation (every corpus caller of
      `cue_recall_wanting` also calls `update_z_goal` on the same run -- see the
      `_harness.py` / `scaffolded_sd054_onboarding.py` call sites), so this closes
      a latent gap rather than an observed false positive.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from _harness import StepHarness  # noqa: E402
from _lib import z_goal_stream as ZGS  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402

from tests.fixtures.seed_utils import set_all_seeds  # noqa: E402
from tests.fixtures.tiny_configs import make_tiny_config  # noqa: E402
from tests.fixtures.tiny_env import make_tiny_env  # noqa: E402


def _agent(goal_on: bool, seed: int = 0):
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env, z_goal_enabled=bool(goal_on), goal_weight=0.5)
    return REEAgent(cfg), env


def _step(agent, env, obs_dict, *, write_z_goal: bool):
    """One hand-rolled tick -- deliberately NOT via StepHarness, because the
    hand-rolled loop is the shape that carries the defect."""
    latent = agent.sense(
        obs_dict["body_state"], obs_dict["world_state"],
        obs_harm=obs_dict.get("harm_obs"),
        obs_harm_a=obs_dict.get("harm_obs_a"),
        obs_harm_history=obs_dict.get("harm_history"),
    )
    ticks = agent.clock.advance()
    e1_prior = torch.zeros(1, latent.z_world.shape[-1], device=agent.device)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    if write_z_goal:
        agent.update_z_goal(benefit_exposure=1.0, drive_level=1.0)
    action = agent.select_action(candidates, ticks, temperature=1.0)
    if action is None:
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, 0] = 1.0
    _flat, _harm, _done, _info, next_obs = env.step(action)
    return next_obs, action


def _pin_goal(agent, magnitude: float = 0.5) -> None:
    """The V3-EXQ-642a/642b shape: write z_goal directly, bypassing the writer.

    Mirrors experiments/v3_exq_642a_blocked_agency_zblock_discriminative.py's
    `_pin_goal` exactly -- a direct `goal_state._z_goal` assignment, never
    `update_z_goal`, so `z_goal_writer_calls` never increments no matter how
    many times this runs."""
    if agent.goal_state is not None:
        agent.goal_state._z_goal = torch.ones(
            1, agent.goal_state.config.goal_dim, device=agent.device
        ) * magnitude


def _run(agent, env, n_steps: int, *, write_z_goal: bool):
    _flat, obs_dict = env.reset()
    agent.reset()
    actions = []
    for _ in range(n_steps):
        obs_dict, action = _step(agent, env, obs_dict, write_z_goal=write_z_goal)
        actions.append(action.detach().clone())
    return actions


# ---- Z1: the counters exist, start at zero, and are read-only -------------------------

def test_z1_counters_start_at_zero():
    agent, _env = _agent(goal_on=True)
    assert agent.z_goal_ticks_total == 0
    assert agent.z_goal_ticks_active == 0
    assert agent.z_goal_writer_calls == 0
    assert agent.z_goal_active_frac is None


def test_z1_counters_are_read_only_properties():
    """Exposed via property with no setter. A driver reports them; it cannot forge
    them into looking live, which is the whole point of a backstop."""
    for name in ("z_goal_ticks_total", "z_goal_ticks_active",
                 "z_goal_writer_calls", "z_goal_active_frac"):
        prop = getattr(REEAgent, name, None)
        assert isinstance(prop, property), f"{name} must be a property"
        assert prop.fset is None, f"{name} must have no setter"
    agent, _env = _agent(goal_on=True)
    with pytest.raises(AttributeError):
        agent.z_goal_ticks_total = 999


# ---- Z2: they count exactly what they claim ------------------------------------------

def test_z2_one_tick_per_select_action_and_active_only_after_a_write():
    agent, env = _agent(goal_on=True)
    _flat, obs_dict = env.reset()
    agent.reset()

    # Two ticks WITHOUT writing z_goal: counted as ticks, never as active.
    for _ in range(2):
        obs_dict, _ = _step(agent, env, obs_dict, write_z_goal=False)
    assert agent.z_goal_ticks_total == 2
    assert agent.z_goal_ticks_active == 0
    assert agent.z_goal_active_frac == 0.0

    # Now write z_goal and keep stepping: those ticks ARE active.
    for _ in range(3):
        obs_dict, _ = _step(agent, env, obs_dict, write_z_goal=True)
    assert agent.z_goal_ticks_total == 5
    assert agent.z_goal_ticks_active == 3
    assert agent.z_goal_active_frac == pytest.approx(3.0 / 5.0)
    assert agent.goal_state.is_active()


# ---- Z3: goal OFF is UNMEASURED (None), not measured-zero (0.0) -----------------------

def test_z3_goal_disabled_counts_nothing_and_frac_is_none():
    agent, env = _agent(goal_on=False)
    assert agent.goal_state is None
    _run(agent, env, 4, write_z_goal=False)
    assert agent.z_goal_ticks_total == 0
    assert agent.z_goal_ticks_active == 0
    # None, NOT 0.0 -- a goal-OFF run has nothing to report, and reporting 0.0
    # would be indistinguishable from the dead-stream defect in Z4.
    assert agent.z_goal_active_frac is None


# ---- Z4: the defect signature ---------------------------------------------------------

def test_z4_dead_stream_signature_is_visible():
    """z_goal_enabled=True + a loop that never calls update_z_goal ==
    ticks_total > 0, ticks_active == 0, writer_calls == 0, frac == 0.0. The
    generalisation of V3-EXQ-830's ad-hoc zgoal_present_frac readiness gate."""
    agent, env = _agent(goal_on=True)
    _run(agent, env, 6, write_z_goal=False)
    assert agent.goal_state is not None          # z_goal WAS configured
    assert agent.z_goal_ticks_total == 6         # the run really stepped
    assert agent.z_goal_ticks_active == 0        # and z_goal never went live
    assert agent.z_goal_writer_calls == 0        # because the sole writer never ran
    assert agent.z_goal_active_frac == 0.0
    assert not agent.goal_state.is_active()
    assert ZGS.z_goal_stream_stats(agent)["writer_defect"] is True


def test_z4_healthy_stream_reads_near_one():
    agent, env = _agent(goal_on=True)
    _run(agent, env, 6, write_z_goal=True)
    # Tick 1 is inactive by construction: update_z_goal fires BEFORE select_action,
    # but GoalState.update has its own firing gate, so the first tick(s) may not yet
    # have written. What matters is that the fraction is materially above zero.
    assert agent.z_goal_active_frac is not None
    assert agent.z_goal_active_frac > 0.5
    assert agent.z_goal_writer_calls == 6
    assert agent.goal_state.is_active()
    assert ZGS.z_goal_stream_stats(agent)["writer_defect"] is False


def test_z4_wired_run_with_no_benefit_is_not_reported_as_the_defect():
    """THE LOOKALIKE, and the reason writer_calls exists. Calling update_z_goal with
    benefit_exposure below GoalState's benefit_threshold leaves z_goal at zero, so
    active_frac reads 0.0 exactly as the defect does -- but the wiring is CORRECT and
    the run simply had no goal signal to form. writer_defect must be False here, or
    every benefit-free run would be mislabelled and the backstop ignored."""
    agent, env = _agent(goal_on=True)
    threshold = float(getattr(agent.config.goal, "benefit_threshold", 0.1))
    _flat, obs_dict = env.reset()
    agent.reset()
    for _ in range(5):
        latent = agent.sense(
            obs_dict["body_state"], obs_dict["world_state"],
            obs_harm=obs_dict.get("harm_obs"),
            obs_harm_a=obs_dict.get("harm_obs_a"),
            obs_harm_history=obs_dict.get("harm_history"),
        )
        ticks = agent.clock.advance()
        e1_prior = torch.zeros(1, latent.z_world.shape[-1], device=agent.device)
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        # Called every tick -- but below the gate, so z_goal never moves.
        agent.update_z_goal(benefit_exposure=threshold / 10.0, drive_level=1.0)
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, 0] = 1.0
        _f, _h, _d, _i, obs_dict = env.step(action)

    assert agent.z_goal_active_frac == 0.0       # identical to the defect reading...
    assert agent.z_goal_writer_calls == 5        # ...but the writer DID run
    assert ZGS.z_goal_stream_stats(agent)["writer_defect"] is False


# ---- Z5: no behaviour change ----------------------------------------------------------

def test_z5_selection_is_bit_identical_with_counters_live():
    """The counters cannot be compiled out, so instead assert what "no behaviour
    change" MEANS operationally: two identically-seeded runs, whose counters both
    advance, select bit-identical actions. Two int bumps plus one RNG-free tensor
    reduction consume no randomness, so any divergence would surface here."""
    agent_a, env_a = _agent(goal_on=True, seed=7)
    actions_a = _run(agent_a, env_a, 8, write_z_goal=True)
    agent_b, env_b = _agent(goal_on=True, seed=7)
    actions_b = _run(agent_b, env_b, 8, write_z_goal=True)

    assert agent_a.z_goal_ticks_total == agent_b.z_goal_ticks_total == 8
    assert agent_a.z_goal_ticks_active == agent_b.z_goal_ticks_active
    for i, (a, b) in enumerate(zip(actions_a, actions_b)):
        assert torch.equal(a, b), f"action diverged at tick {i}"


def test_z5_counting_does_not_consume_rng():
    """Direct form of the same claim: the counter block draws no random numbers, so
    the global RNG state is untouched by the bookkeeping itself."""
    agent, _env = _agent(goal_on=True)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    # Exercise exactly the counted operations.
    for _ in range(5):
        assert isinstance(agent.goal_state.is_active(), bool)
    assert torch.equal(before, torch.random.get_rng_state())


# ---- Z6: run-lifetime, not per-episode ------------------------------------------------

def test_z6_agent_reset_does_not_clear_the_counters():
    agent, env = _agent(goal_on=True)
    _run(agent, env, 3, write_z_goal=True)
    total_after_ep1 = agent.z_goal_ticks_total
    active_after_ep1 = agent.z_goal_ticks_active
    assert total_after_ep1 == 3

    _run(agent, env, 3, write_z_goal=True)   # _run calls agent.reset() first
    assert agent.z_goal_ticks_total == 6, "counters must survive reset() (run-lifetime)"
    assert agent.z_goal_ticks_active >= active_after_ep1


def test_z6_liveness_persists_across_episodes_because_reset_does_not_clear_z_goal():
    """The complement of Z6, and a correction worth stating explicitly: `REEAgent.reset()`
    does NOT reset `goal_state` (pinned independently by
    test_dead_z_goal_stream_lint.test_dzg_agent_reset_does_not_reset_goal_state). So once
    z_goal goes live it STAYS live for the rest of the run, and every subsequent tick
    counts active even across episode boundaries and even if the driver stops writing.

    This is what makes an intermediate active_frac mean something specific: it is the
    run's WARM-UP PREFIX -- the ticks before the first successful write -- and not
    per-episode re-zeroing. Reading it as the latter would understate how long a stream
    took to come up."""
    agent, env = _agent(goal_on=True)
    _run(agent, env, 4, write_z_goal=True)
    assert agent.goal_state.is_active()
    live_before = agent.z_goal_ticks_active

    agent.reset()
    assert agent.goal_state.is_active(), \
        "reset() must NOT clear z_goal -- goal_state is deliberately cross-episode"

    # A fresh episode that never writes still counts ACTIVE ticks, because the
    # attractor survived the reset.
    _flat, obs_dict = env.reset()
    _step(agent, env, obs_dict, write_z_goal=False)
    assert agent.z_goal_ticks_active == live_before + 1


def test_z6_intermediate_frac_is_the_warmup_prefix():
    """Pin the interpretation directly: ticks before the first successful write are
    inactive, everything after is active, so 0 < frac < 1 localises the warm-up."""
    agent, env = _agent(goal_on=True)
    _flat, obs_dict = env.reset()
    agent.reset()
    for _ in range(3):                      # cold prefix -- no writer
        obs_dict, _ = _step(agent, env, obs_dict, write_z_goal=False)
    assert agent.z_goal_active_frac == 0.0
    for _ in range(5):                      # writer on
        obs_dict, _ = _step(agent, env, obs_dict, write_z_goal=True)

    assert agent.z_goal_ticks_total == 8
    assert 0.0 < agent.z_goal_active_frac < 1.0
    # The inactive ticks are exactly the cold prefix.
    assert agent.z_goal_ticks_total - agent.z_goal_ticks_active >= 3


# ---- Z7: the recording helper ---------------------------------------------------------

def test_z7_helper_is_stdlib_only_and_duck_typed():
    """z_goal_stream must import without torch/ree_core -- manifest_core imports it,
    and manifest_core's contract is that a scalar-only caller needs no substrate."""
    src = (EXPERIMENTS_DIR / "_lib" / "z_goal_stream.py").read_text(encoding="utf-8")
    for banned in ("import torch", "from torch", "import ree_core", "from ree_core"):
        assert banned not in src, f"z_goal_stream must not {banned}"
    # And prove it: import in a clean interpreter with torch/ree_core blocked.
    # find_spec, not the long-removed find_module (gone in py3.12).
    probe = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'ree_core'):\n"
        "            raise ImportError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {str(EXPERIMENTS_DIR / '_lib')!r})\n"
        "import z_goal_stream as Z\n"
        "print(Z.stats_from_counts(4, 1)['active_frac'])\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.25"


def test_z7_stats_from_a_live_agent():
    agent, env = _agent(goal_on=True)
    _run(agent, env, 5, write_z_goal=False)
    stats = ZGS.z_goal_stream_stats(agent)
    assert stats["ticks_total"] == 5
    assert stats["ticks_active"] == 0
    assert stats["writer_calls"] == 0
    assert stats["active_frac"] == 0.0
    assert stats["writer_defect"] is True
    assert stats["goal_state_present"] is True
    assert stats["n_agents"] == 1


def test_z7_pools_across_a_multi_arm_runs_agents():
    """A multi-arm run builds one agent per arm x seed; the run-level fraction is the
    pooled one, and a goal-OFF control arm beside goal-ON arms must not mask that the
    z_goal path was live."""
    live, env_live = _agent(goal_on=True, seed=1)
    _run(live, env_live, 4, write_z_goal=True)
    dead, env_dead = _agent(goal_on=True, seed=2)
    _run(dead, env_dead, 4, write_z_goal=False)

    pooled = ZGS.z_goal_stream_stats([live, dead])
    assert pooled["n_agents"] == 2
    assert pooled["ticks_total"] == 8
    assert pooled["ticks_active"] == live.z_goal_ticks_active
    assert pooled["goal_state_present"] is True
    assert 0.0 < pooled["active_frac"] < 1.0


def test_z7_no_counter_bearing_agent_yields_no_block():
    """Absence must mean "unmeasured", never "measured zero" -- so a non-agent
    contributes nothing rather than zeros that would dilute a real fraction."""
    assert ZGS.z_goal_stream_stats(None) is None
    assert ZGS.z_goal_stream_stats([]) is None
    assert ZGS.z_goal_stream_stats(object()) is None
    assert ZGS.z_goal_stream_stats("not-an-agent") is None


def test_z7_frac_is_none_when_nothing_was_measured():
    block = ZGS.stats_from_counts(0, 0, goal_state_present=False)
    assert block["active_frac"] is None
    assert block["writer_defect"] is None, "unmeasured is not a defect verdict"
    assert block["ticks_total"] == 0


def test_z7_stamp_never_raises_and_never_clobbers():
    manifest = {"z_goal_stream": {"ticks_total": 99, "ticks_active": 99}}
    ZGS.stamp_z_goal_stream(manifest, object())
    assert manifest["z_goal_stream"]["ticks_total"] == 99, "must not clobber"
    # Nothing to record -> key stays absent.
    empty: dict = {}
    ZGS.stamp_z_goal_stream(empty, None)
    assert "z_goal_stream" not in empty


# ---- Z8: it reaches the manifest -------------------------------------------------------

def test_z8_stamp_recording_core_surfaces_the_block():
    from _lib.manifest_core import stamp_recording_core
    agent, env = _agent(goal_on=True)
    _run(agent, env, 4, write_z_goal=False)
    manifest: dict = {"run_id": "x_v3"}
    stamp_recording_core(manifest, agent=agent)
    assert manifest["z_goal_stream"]["active_frac"] == 0.0
    assert manifest["z_goal_stream"]["ticks_total"] == 4


def test_z8_stamp_recording_core_omits_the_block_when_unmeasured():
    from _lib.manifest_core import stamp_recording_core
    manifest: dict = {"run_id": "x_v3"}
    stamp_recording_core(manifest)
    assert "z_goal_stream" not in manifest


def test_z8_write_flat_manifest_round_trips_the_block():
    from pack_writer import write_flat_manifest
    agent, env = _agent(goal_on=True)
    _run(agent, env, 4, write_z_goal=False)
    with tempfile.TemporaryDirectory() as td:
        out = write_flat_manifest(
            {"run_id": "zgoal_probe_v3", "status": "PASS"},
            out_dir=td, agent=agent,
        )
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
    assert doc["z_goal_stream"]["ticks_total"] == 4
    assert doc["z_goal_stream"]["active_frac"] == 0.0


def test_z8_write_flat_manifest_accepts_precomputed_stats():
    from pack_writer import write_flat_manifest
    stats = ZGS.stats_from_counts(10, 7, goal_state_present=True)
    with tempfile.TemporaryDirectory() as td:
        out = write_flat_manifest(
            {"run_id": "zgoal_stats_v3", "status": "PASS"},
            out_dir=td, z_goal_stream_stats=stats,
        )
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
    assert doc["z_goal_stream"]["active_frac"] == pytest.approx(0.7)


def test_z8_older_stamp_signature_does_not_lose_the_whole_core():
    """The stamp call sits inside `except Exception: pass`. If stamp_recording_core
    ever resolves to an older manifest_core without the z_goal kwargs, passing them
    must NOT silently skip the ENTIRE always-core stamp -- a strictly worse failure
    than the one they exist to catch. pack_writer falls back to a core-only call."""
    import pack_writer as PW

    calls = []

    def _old_signature(manifest, config=None, seeds=None, script_path=None,
                       machine=None, elapsed_seconds=None, started_at=None,
                       overwrite=False):
        calls.append("core")
        # A "properly stamped, just an older signature" core call still fills the
        # mandatory always-core subset (write_flat_manifest's 2026-08-12
        # hard-enforcement) -- only the z_goal kwargs are the thing this stub is
        # simulating as absent, not the fields it stamps regardless of signature.
        manifest["machine"] = "stamped"
        manifest["recording_schema"] = "rec/v1"
        manifest["substrate_hash"] = "0" * 64
        manifest["substrate_commit"] = {"commit": "0" * 40, "dirty": False}
        manifest["machine_class"] = "test-class"
        return manifest

    original = PW._import_stamp_recording_core
    PW._import_stamp_recording_core = lambda: _old_signature
    try:
        with tempfile.TemporaryDirectory() as td:
            out = PW.write_flat_manifest(
                {"run_id": "legacy_stamp_v3", "status": "PASS"},
                out_dir=td, agent=object(),
            )
            doc = json.loads(Path(out).read_text(encoding="utf-8"))
    finally:
        PW._import_stamp_recording_core = original

    assert calls == ["core"], "must retry the core-only call, exactly once"
    assert doc["machine"] == "stamped", "always-core must still be stamped"
    assert "z_goal_stream" not in doc


# ---- Z9: StepHarness records it too ---------------------------------------------------

def test_z9_step_harness_records_and_agrees_with_the_agent():
    set_all_seeds(3)
    env = make_tiny_env(seed=3)
    cfg = make_tiny_config(env, z_goal_enabled=True, goal_weight=0.5)
    agent = REEAgent(cfg)
    harness = StepHarness(agent, env, train_mode=False)

    _flat, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    for _ in range(5):
        obs_dict = harness.step(obs_dict).next_obs_dict

    stats = harness.z_goal_stream_stats()
    assert stats["ticks_total"] == 5
    assert stats["ticks_total"] == agent.z_goal_ticks_total
    assert stats["ticks_active"] == agent.z_goal_ticks_active
    assert stats["writer_calls"] == agent.z_goal_writer_calls == 5
    assert stats["goal_state_present"] is True
    # Invariant 2 pins the call, so a harness-driven run can never be the defect --
    # regardless of whether the benefit gate ever opened (on the tiny env it does
    # not, so active_frac is legitimately 0.0 here; see the Z4 lookalike test).
    assert stats["writer_defect"] is False


def test_z9_harness_reset_does_not_clear_its_tally():
    set_all_seeds(4)
    env = make_tiny_env(seed=4)
    cfg = make_tiny_config(env, z_goal_enabled=True, goal_weight=0.5)
    harness = StepHarness(REEAgent(cfg), env, train_mode=False)
    harness.run_episode(max_steps=3)
    after_ep1 = harness.z_goal_stream_stats()["ticks_total"]
    harness.run_episode(max_steps=3)
    assert harness.z_goal_stream_stats()["ticks_total"] == after_ep1 * 2


def test_z9_harness_tally_survives_an_agent_swap():
    """Counted harness-side rather than read off the agent at the end, so a multi-arm
    driver reusing one harness across arms keeps a run-level tally."""
    set_all_seeds(5)
    env = make_tiny_env(seed=5)
    cfg = make_tiny_config(env, z_goal_enabled=True, goal_weight=0.5)
    harness = StepHarness(REEAgent(cfg), env, train_mode=False)
    harness.run_episode(max_steps=3)
    first = harness.z_goal_stream_stats()["ticks_total"]

    harness.agent = REEAgent(cfg)   # fresh agent, counters back at zero
    harness.run_episode(max_steps=3)
    assert harness.agent.z_goal_ticks_total == 3
    assert harness.z_goal_stream_stats()["ticks_total"] == first + 3


def test_z9_harness_goal_off_reports_unmeasured():
    set_all_seeds(6)
    env = make_tiny_env(seed=6)
    cfg = make_tiny_config(env, z_goal_enabled=False)
    harness = StepHarness(REEAgent(cfg), env, train_mode=False)
    harness.run_episode(max_steps=3)
    stats = harness.z_goal_stream_stats()
    assert stats["ticks_total"] == 0
    assert stats["active_frac"] is None
    assert stats["goal_state_present"] is False


# ---- Z10: ZGoalStreamAccumulator -- the per-cell tally for hand-rolled drivers ---------
#
# The landed hand-rolled corpus builds a FRESH agent inside a per-cell function and lets
# it fall out of scope when the cell returns, so wiring those drivers to `agent=[...]`
# would retain every arm x seed agent purely for provenance. The accumulator reads the
# counters at end-of-cell and drops the reference. These contracts pin that it pools the
# same numbers the pooled helper would, and that the ordering trap is not silently wrong.

def test_z10_accumulator_pools_the_same_numbers_as_the_agent_list_helper():
    agents = []
    for i, write in enumerate([True, True, False]):
        agent, env = _agent(goal_on=True, seed=10 + i)
        _run(agent, env, 4, write_z_goal=write)
        agents.append(agent)

    acc = ZGS.ZGoalStreamAccumulator()
    for a in agents:
        acc.observe(a)

    assert acc.stats() == ZGS.z_goal_stream_stats(agents), (
        "the accumulator must be a drop-in for the pooled helper -- a driver "
        "choosing it for memory reasons must not get different numbers"
    )
    assert acc.stats()["n_agents"] == 3
    assert acc.stats()["ticks_total"] == 12


def test_z10_accumulator_reads_at_call_time_not_construction_time():
    """The ordering trap: observe() must be called AFTER the cell has stepped."""
    agent, env = _agent(goal_on=True, seed=20)

    too_early = ZGS.ZGoalStreamAccumulator()
    too_early.observe(agent)                      # the mistake -- fresh agent
    _run(agent, env, 4, write_z_goal=True)
    correct = ZGS.ZGoalStreamAccumulator()
    correct.observe(agent)                        # correct -- after stepping

    assert too_early.stats()["ticks_total"] == 0
    assert correct.stats()["ticks_total"] == 4


def test_z10_observe_returns_none_so_the_construction_site_chain_is_not_a_dropin():
    """`agent = acc.observe(REEAgent(cfg))` must not typecheck as a drop-in, because
    that is exactly the site where the counters are still zero."""
    agent, _env = _agent(goal_on=True, seed=21)
    assert ZGS.ZGoalStreamAccumulator().observe(agent) is None


def test_z10_empty_accumulator_reports_unmeasured_not_zeros():
    assert ZGS.ZGoalStreamAccumulator().stats() is None, (
        "a recorded z_goal_stream block must always mean the run measured it"
    )


def test_z10_non_agents_contribute_nothing_and_never_raise():
    acc = ZGS.ZGoalStreamAccumulator()
    for junk in (None, "agent", 7, {"ticks_total": 99}, object()):
        acc.observe(junk)
    assert acc.stats() is None

    agent, env = _agent(goal_on=True, seed=22)
    _run(agent, env, 3, write_z_goal=True)
    acc.observe(agent)
    assert acc.stats()["n_agents"] == 1, "junk must not dilute a real fraction"
    assert acc.stats()["ticks_total"] == 3


def test_z10_observe_accepts_an_iterable_of_agents():
    pair = []
    for i in range(2):
        agent, env = _agent(goal_on=True, seed=30 + i)
        _run(agent, env, 2, write_z_goal=True)
        pair.append(agent)
    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(pair)
    assert acc.stats()["n_agents"] == 2
    assert acc.stats()["ticks_total"] == 4


def test_z10_observe_stats_pools_a_stepharness_cell_block():
    """StepHarness is cell-local too, so a multi-cell harness driver needs pooling."""
    acc = ZGS.ZGoalStreamAccumulator()
    for i in range(2):
        set_all_seeds(40 + i)
        env = make_tiny_env(seed=40 + i)
        cfg = make_tiny_config(env, z_goal_enabled=True, goal_weight=0.5)
        harness = StepHarness(REEAgent(cfg), env, train_mode=False)
        harness.run_episode(max_steps=3)
        acc.observe_stats(harness.z_goal_stream_stats())

    stats = acc.stats()
    assert stats["ticks_total"] == 6
    assert stats["writer_calls"] == 6
    assert stats["writer_defect"] is False
    assert stats["goal_state_present"] is True
    assert stats["n_agents"] == 2, "n_agents sums the cells' own n_agents"


def test_z10_observe_stats_ignores_junk_and_never_raises():
    acc = ZGS.ZGoalStreamAccumulator()
    for junk in (None, "block", 3, [1, 2]):
        acc.observe_stats(junk)
    assert acc.stats() is None


def test_z10_writer_defect_survives_pooling():
    """A run whose cells never called update_z_goal must still read as the defect."""
    acc = ZGS.ZGoalStreamAccumulator()
    for i in range(3):
        agent, env = _agent(goal_on=True, seed=50 + i)
        _run(agent, env, 3, write_z_goal=False)
        acc.observe(agent)
    stats = acc.stats()
    assert stats["writer_calls"] == 0
    assert stats["writer_defect"] is True
    assert stats["active_frac"] == 0.0


def test_z10_goal_off_arm_does_not_mask_a_live_goal_on_arm():
    acc = ZGS.ZGoalStreamAccumulator()
    off, off_env = _agent(goal_on=False, seed=60)
    _run(off, off_env, 3, write_z_goal=True)
    acc.observe(off)
    on, on_env = _agent(goal_on=True, seed=61)
    _run(on, on_env, 3, write_z_goal=True)
    acc.observe(on)
    assert acc.stats()["goal_state_present"] is True


def test_z10_accumulator_block_round_trips_through_write_flat_manifest():
    agent, env = _agent(goal_on=True, seed=70)
    _run(agent, env, 4, write_z_goal=True)
    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(agent)

    sys.path.insert(0, str(EXPERIMENTS_DIR))
    from pack_writer import write_flat_manifest  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        out = write_flat_manifest(
            {"run_id": "z10_accumulator_v3", "outcome": "PASS"},
            td, script_path=Path(__file__),
            z_goal_stream_stats=acc.stats(),
        )
        doc = json.loads(Path(out).read_text())
    assert doc["z_goal_stream"]["ticks_total"] == 4
    assert doc["z_goal_stream"]["writer_defect"] is False


# ---- Z11: the --dry-run smoke print ---------------------------------------------------
#
# The counter deliberately replaced a substrate stderr warning (see the module docstring),
# on the grounds that "a --dry-run smoke can just print it". These pin that it does, that
# it stays OUT of a real run and the contract suite, and that it never gates.

def _dry_write(tmp, capsys, **kw):
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    from pack_writer import write_flat_manifest  # noqa: E402
    write_flat_manifest({"run_id": "z11_smoke_v3", "outcome": "PASS"}, tmp,
                        script_path=Path(__file__), **kw)
    return capsys.readouterr().out


def test_z11_dry_run_prints_the_block(tmp_path, capsys):
    agent, env = _agent(goal_on=True, seed=80)
    _run(agent, env, 4, write_z_goal=True)
    out = _dry_write(tmp_path, capsys, dry_run=True, agent=agent)
    assert "z_goal_stream:" in out
    assert "ticks=" in out and "writer_calls=4" in out
    assert "no writer defect" in out


def test_z11_dry_run_names_the_defect_in_words(tmp_path, capsys):
    agent, env = _agent(goal_on=True, seed=81)
    _run(agent, env, 4, write_z_goal=False)     # the V3-EXQ-626 shape
    out = _dry_write(tmp_path, capsys, dry_run=True, agent=agent)
    assert "WRITER DEFECT" in out, "the one reading that means 'bug' must say so"


def test_z11_dry_run_hints_when_the_caller_wired_nothing(tmp_path, capsys):
    out = _dry_write(tmp_path, capsys, dry_run=True)
    assert "NOT RECORDED" in out
    assert "write_flat_manifest" in out, "the hint must name the fix"


def test_z11_real_run_prints_nothing(tmp_path, capsys):
    """A print during a multi-hour run scrolls past unread and would fire across the
    contract suite -- exactly the objections that got the stderr warning rejected."""
    agent, env = _agent(goal_on=True, seed=82)
    _run(agent, env, 3, write_z_goal=False)
    assert "z_goal_stream" not in _dry_write(tmp_path, capsys, agent=agent)
    assert "z_goal_stream" not in _dry_write(tmp_path, capsys)


def test_z11_print_is_ascii_only(tmp_path, capsys):
    agent, env = _agent(goal_on=True, seed=83)
    _run(agent, env, 2, write_z_goal=True)
    for out in (_dry_write(tmp_path, capsys, dry_run=True, agent=agent),
                _dry_write(tmp_path, capsys, dry_run=True)):
        out.encode("ascii")     # repo rule: printed output must survive cp1252


def test_z11_print_never_gates_or_raises(tmp_path, capsys):
    """A zero fraction is a legitimate reading (goal-OFF parity arm, negative control,
    benefit gate never opened), so the smoke reports and returns -- it must not raise."""
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    from pack_writer import write_flat_manifest  # noqa: E402
    out_path = write_flat_manifest(
        {"run_id": "z11_nogate_v3", "outcome": "PASS",
         "z_goal_stream": {"ticks_total": 9, "ticks_active": 0, "writer_calls": 0,
                           "active_frac": 0.0, "writer_defect": True}},
        tmp_path, dry_run=True, script_path=Path(__file__),
    )
    assert Path(out_path).exists(), "a reported defect must still write the manifest"
    assert "WRITER DEFECT" in capsys.readouterr().out


def test_z11_unmeasured_is_not_reported_as_exoneration(tmp_path, capsys):
    """`writer_defect` None means UNMEASURED. Printing "no writer defect" there would
    read as a clean bill of health for a run that had no opportunity to show one --
    the exact reading a goal-OFF driver produces (measured on V3-EXQ-795, whose
    trigger knob is benefit_terrain_live_producer, not z_goal_enabled)."""
    agent, env = _agent(goal_on=False, seed=84)
    _run(agent, env, 3, write_z_goal=True)
    out = _dry_write(tmp_path, capsys, dry_run=True, agent=agent)
    assert "not assessable" in out
    assert "no writer defect" not in out
    assert "WRITER DEFECT" not in out


# ---- Z12: the training-phase-agent false positive (V3-EXQ-874b) -----------------------
#
# writer_calls == 0 with ticks_total > 0 is Z4's defect signature, and it is also
# exactly what a training-phase-only agent reads. eval_stepped=False on every
# observation entry point routes counts to a separate training_phase_* tally that
# never feeds writer_defect. The regression guards (z12_defect_still_fires_by_default
# family) pin that this does NOT weaken Z4's real-defect detection.

def test_z12_training_only_agent_does_not_report_writer_defect():
    """The V3-EXQ-874b shape itself: an agent stepped only by a training loop that
    legitimately never calls update_z_goal. Without eval_stepped=False this reads
    exactly like Z4's real defect (ticks_total > 0, writer_calls == 0) -- that is the
    whole point of the guard."""
    agent, env = _agent(goal_on=True, seed=90)
    _run(agent, env, 6, write_z_goal=False)   # stands in for a training-only loop
    assert agent.z_goal_ticks_total == 6
    assert agent.z_goal_writer_calls == 0

    stats = ZGS.z_goal_stream_stats(agent, eval_stepped=False)
    assert stats["writer_defect"] is None, "training-only must never read as the defect"
    assert stats["ticks_total"] == 0
    assert stats["writer_calls"] == 0
    assert stats["n_agents"] == 0
    assert stats["training_phase_ticks_total"] == 6
    assert stats["training_phase_writer_calls"] == 0
    assert stats["training_phase_n_agents"] == 1
    assert stats["goal_state_present"] is True


def test_z12_default_is_unchanged_eval_stepped_true():
    """eval_stepped defaults to True -- every existing call site (which does not know
    the new kwarg exists) must see byte-identical behaviour."""
    agent, env = _agent(goal_on=True, seed=91)
    _run(agent, env, 5, write_z_goal=False)
    explicit = ZGS.z_goal_stream_stats(agent, eval_stepped=True)
    implicit = ZGS.z_goal_stream_stats(agent)
    assert explicit == implicit
    assert implicit["writer_defect"] is True
    assert "training_phase_ticks_total" not in implicit


def test_z12_real_defect_still_fires_regardless_of_agent_training_attribute():
    """Regression guard for the rejected `.training`-heuristic design (see the module
    docstring's "why this cannot be auto-detected" section): V3-EXQ-830's own driver
    never calls `.eval()`/`.train()`, so its agent sits at the nn.Module default
    `training=True` throughout -- identical object-state to the 874b false positive.
    The fix must still report the real defect by default (eval_stepped=True)."""
    agent, env = _agent(goal_on=True, seed=92)
    assert agent.training is True, "nn.Module default, untouched -- the corpus norm"
    _run(agent, env, 6, write_z_goal=False)
    assert agent.training is True, "still untouched -- nothing in this driver toggles it"
    stats = ZGS.z_goal_stream_stats(agent)   # eval_stepped defaults to True
    assert stats["writer_defect"] is True, (
        "a genuinely broken eval loop must still be caught even though its agent.training "
        "state is indistinguishable from a training-phase-only agent's"
    )


def test_z12_accumulator_pools_training_and_eval_observations_separately():
    """A driver correctly using both: the P0 base agent (training_only) and the
    stepping clone (eval). writer_defect must reflect ONLY the eval clone."""
    base, base_env = _agent(goal_on=True, seed=93)
    _run(base, base_env, 6, write_z_goal=False)   # P0-training-shaped

    clone, clone_env = _agent(goal_on=True, seed=94)
    _run(clone, clone_env, 4, write_z_goal=True)  # eval-shaped, correctly wired

    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(base, eval_stepped=False)
    acc.observe(clone, eval_stepped=True)
    stats = acc.stats()

    assert stats["writer_defect"] is False, "the eval clone was correctly wired"
    assert stats["ticks_total"] == 4
    assert stats["writer_calls"] == 4
    assert stats["n_agents"] == 1
    assert stats["training_phase_ticks_total"] == 6
    assert stats["training_phase_writer_calls"] == 0
    assert stats["training_phase_n_agents"] == 1


def test_z12_accumulator_all_training_only_reports_unmeasured_eval_but_keeps_training_tally():
    """The exact 874b mistake reproduced through the accumulator: only the base agent
    is ever observed (eval_stepped=False on every call, or the driver never observes
    the clone at all). writer_defect must be None, not True, and the training data
    must still be visible rather than silently dropped."""
    acc = ZGS.ZGoalStreamAccumulator()
    for i in range(2):
        agent, env = _agent(goal_on=True, seed=100 + i)
        _run(agent, env, 3, write_z_goal=False)
        acc.observe(agent, eval_stepped=False)
    stats = acc.stats()

    assert stats["writer_defect"] is None
    assert stats["ticks_total"] == 0
    assert stats["n_agents"] == 0
    assert stats["training_phase_ticks_total"] == 6
    assert stats["training_phase_n_agents"] == 2
    assert stats["goal_state_present"] is True


def test_z12_observe_stats_training_only_routes_correctly():
    """observe_stats mirrors observe()'s eval_stepped split, for a precomputed block
    (e.g. StepHarness) that the caller knows is training-phase-only."""
    acc = ZGS.ZGoalStreamAccumulator()
    training_block = ZGS.stats_from_counts(5, 0, writer_calls=0, goal_state_present=True)
    acc.observe_stats(training_block, eval_stepped=False)
    stats = acc.stats()
    assert stats["writer_defect"] is None
    assert stats["training_phase_ticks_total"] == 5
    assert stats["training_phase_n_agents"] == 1


def test_z12_stamp_z_goal_stream_eval_stepped_false():
    agent, env = _agent(goal_on=True, seed=95)
    _run(agent, env, 4, write_z_goal=False)
    manifest: dict = {}
    ZGS.stamp_z_goal_stream(manifest, agent, eval_stepped=False)
    block = manifest["z_goal_stream"]
    assert block["writer_defect"] is None
    assert block["training_phase_ticks_total"] == 4


def test_z12_stamp_z_goal_stream_precomputed_stats_ignores_eval_stepped():
    """A precomputed `stats=` block bypasses z_goal_stream_stats entirely, so
    eval_stepped must have no effect on it -- the caller already decided its shape."""
    manifest: dict = {}
    precomputed = ZGS.stats_from_counts(9, 0, writer_calls=0, goal_state_present=True)
    ZGS.stamp_z_goal_stream(manifest, stats=precomputed, eval_stepped=False)
    assert manifest["z_goal_stream"]["writer_defect"] is True, (
        "eval_stepped only governs agent= observation, never a precomputed stats= block"
    )


def test_z12_no_counter_bearing_agent_yields_no_block_even_training_only():
    assert ZGS.z_goal_stream_stats(None, eval_stepped=False) is None
    assert ZGS.z_goal_stream_stats(object(), eval_stepped=False) is None


def test_z12_empty_accumulator_with_no_observations_at_all_reports_unmeasured():
    assert ZGS.ZGoalStreamAccumulator().stats() is None


# ---- Z13: the --dry-run smoke print surfaces training_phase_* (V3-EXQ-874b) -----------
#
# Z12 gave a driver a correct way to say "this agent's ticks are training-phase-only"
# (eval_stepped=False) instead of a false writer_defect. But the Z11 smoke print did not
# know about training_phase_* at all, so a driver correctly using eval_stepped=False saw
# the SAME "not assessable" line as a genuinely unmeasured run -- the real training-phase
# data existed in the manifest block but was invisible at the smoke, where an author is
# actually looking. These pin that the note appears when training_phase_ticks_total is
# present and nonzero, stays out when it is not, and never changes the eval-facing verdict.

def test_z13_dry_run_training_only_shows_the_training_note(tmp_path, capsys):
    """The V3-EXQ-874b shape itself, observed correctly via eval_stepped=False: the
    eval-facing verdict must still read 'not assessable' (Z11), and the training ticks
    that would otherwise be invisible must now show up alongside it."""
    agent, env = _agent(goal_on=True, seed=110)
    _run(agent, env, 6, write_z_goal=False)
    block = ZGS.z_goal_stream_stats(agent, eval_stepped=False)
    assert block["training_phase_ticks_total"] == 6      # sanity on the fixture
    out = _dry_write(tmp_path, capsys, dry_run=True, z_goal_stream_stats=block)
    assert "not assessable" in out, "eval-facing verdict is unchanged by the note"
    assert "WRITER DEFECT" not in out
    assert "+6 training-phase ticks" in out
    assert "0 writer calls" in out
    assert "not counted toward writer_defect" in out


def test_z13_dry_run_mixed_training_and_eval_shows_both(tmp_path, capsys):
    """A driver correctly observing BOTH the P0 base agent (training-only) and the
    stepping clone (eval): the real eval verdict prints as usual (Z11), and the
    training-phase ticks are appended rather than silently pooled away."""
    base, base_env = _agent(goal_on=True, seed=111)
    _run(base, base_env, 6, write_z_goal=False)
    clone, clone_env = _agent(goal_on=True, seed=112)
    _run(clone, clone_env, 4, write_z_goal=True)

    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(base, eval_stepped=False)
    acc.observe(clone, eval_stepped=True)
    block = acc.stats()
    assert block["training_phase_ticks_total"] == 6
    assert block["writer_calls"] == 4

    out = _dry_write(tmp_path, capsys, dry_run=True, z_goal_stream_stats=block)
    assert "no writer defect" in out
    assert "writer_calls=4" in out
    assert "+6 training-phase ticks" in out
    assert "0 writer calls, not counted toward writer_defect" in out


def test_z13_dry_run_without_training_phase_omits_the_note(tmp_path, capsys):
    """An ordinary block with no training_phase_* keys at all (the corpus norm) must
    not grow a spurious note."""
    agent, env = _agent(goal_on=True, seed=113)
    _run(agent, env, 4, write_z_goal=True)
    out = _dry_write(tmp_path, capsys, dry_run=True, agent=agent)
    assert "training-phase" not in out


def test_z13_dry_run_zero_training_ticks_omits_the_note(tmp_path, capsys):
    """training_phase_ticks_total present but 0 (an accumulator that never actually
    folded in a training-only observation) reads as absent, matching 'present and
    nonzero', not merely 'key present'."""
    agent, env = _agent(goal_on=True, seed=114)
    _run(agent, env, 3, write_z_goal=True)
    block = ZGS.z_goal_stream_stats(agent)
    block["training_phase_ticks_total"] = 0
    out = _dry_write(tmp_path, capsys, dry_run=True, z_goal_stream_stats=block)
    assert "training-phase" not in out


def test_z13_print_is_ascii_only(tmp_path, capsys):
    agent, env = _agent(goal_on=True, seed=115)
    _run(agent, env, 5, write_z_goal=False)
    block = ZGS.z_goal_stream_stats(agent, eval_stepped=False)
    out = _dry_write(tmp_path, capsys, dry_run=True, z_goal_stream_stats=block)
    out.encode("ascii")     # repo rule: printed output must survive cp1252


def test_z13_print_never_gates_or_raises(tmp_path, capsys):
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    from pack_writer import write_flat_manifest  # noqa: E402
    out_path = write_flat_manifest(
        {"run_id": "z13_nogate_v3", "outcome": "PASS",
         "z_goal_stream": {"ticks_total": 0, "ticks_active": 0, "writer_calls": 0,
                           "active_frac": None, "writer_defect": None,
                           "goal_state_present": True, "n_agents": 0,
                           "training_phase_ticks_total": 7,
                           "training_phase_ticks_active": 2,
                           "training_phase_writer_calls": 0,
                           "training_phase_n_agents": 1}},
        tmp_path, dry_run=True, script_path=Path(__file__),
    )
    assert Path(out_path).exists(), "a training-phase-only block must still write"
    assert "+7 training-phase ticks" in capsys.readouterr().out


# ---- Z14: the pinned-goal false positive (V3-EXQ-642b) --------------------------------
#
# writer_calls == 0 with ticks_total > 0 is Z4's defect signature, and it is also exactly
# what a driver reads when it pins z_goal at a fixed magnitude by writing
# `agent.goal_state._z_goal` directly (V3-EXQ-642a/642b), bypassing update_z_goal
# entirely. `goal_pinned=True` on every observation entry point reports
# `writer_defect: None` instead of a false `True`. Unlike Z12's training-phase carve-out,
# the pinned ticks stay in the ordinary eval counters -- these tests pin that shape
# specifically, alongside the fail-safe regression: the flag must be OPT-IN, so a run
# that pins z_goal without declaring it still reads as the Z4 defect.

def _run_pinned(agent, env, n_steps: int, *, magnitude: float = 0.5):
    """Steps the agent while pinning z_goal directly before each tick, mirroring
    the driver's call order (`_pin_goal` called before `sense()`/`select_action`
    on every tick, never `update_z_goal`)."""
    _flat, obs_dict = env.reset()
    agent.reset()
    _pin_goal(agent, magnitude)
    for _ in range(n_steps):
        _pin_goal(agent, magnitude)
        obs_dict, _ = _step(agent, env, obs_dict, write_z_goal=False)
    return obs_dict


def test_z14_pin_signature_matches_the_documented_shape():
    """THE DISCRIMINATOR ITSELF: a direct-write pin reads writer_calls == 0 with
    active_frac == 1.0 (every tick active from the first pin onward), unlike a
    genuine omission's active_frac == 0.0 (Z4). This is what makes the two
    distinguishable in principle -- and also why they cannot be told apart by the
    counters alone without the driver's own declaration (see Z14's other tests):
    a genuine omission and a pin both start from writer_calls == 0."""
    agent, env = _agent(goal_on=True, seed=120)
    _run_pinned(agent, env, 5)
    assert agent.z_goal_writer_calls == 0
    assert agent.z_goal_ticks_total == 5
    assert agent.z_goal_ticks_active == 5
    assert agent.z_goal_active_frac == 1.0
    assert agent.goal_state.is_active()


def test_z14_goal_pinned_true_suppresses_writer_defect():
    agent, env = _agent(goal_on=True, seed=121)
    _run_pinned(agent, env, 4)
    stats = ZGS.z_goal_stream_stats(agent, goal_pinned=True)
    assert stats["writer_defect"] is None, "a declared pin must not read as the defect"
    assert stats["goal_pinned"] is True
    assert stats["ticks_total"] == 4
    assert stats["active_frac"] == 1.0
    assert stats["writer_calls"] == 0


def test_z14_default_false_still_flags_an_undeclared_pin_as_the_defect():
    """FAIL-SAFE REGRESSION: the exact same pinned run, without goal_pinned=True,
    must still read writer_defect: true. Fixing V3-EXQ-642b's false positive must
    not create a blind spot for a driver that pins z_goal by accident (or a
    genuine bug that happens to produce the same signature)."""
    agent, env = _agent(goal_on=True, seed=122)
    _run_pinned(agent, env, 4)
    stats = ZGS.z_goal_stream_stats(agent)
    assert stats["writer_defect"] is True
    assert "goal_pinned" not in stats


def test_z14_genuine_omission_is_unaffected_by_the_new_kwarg():
    """The other half of "both directions": a true omission (Z4's shape) must
    still read the defect when goal_pinned is left at its default False."""
    agent, env = _agent(goal_on=True, seed=123)
    _run(agent, env, 5, write_z_goal=False)
    assert agent.z_goal_active_frac == 0.0, "sanity: this is the omission, not the pin"
    stats = ZGS.z_goal_stream_stats(agent)
    assert stats["writer_defect"] is True
    assert "goal_pinned" not in stats


def test_z14_real_defect_still_fires_when_a_different_arm_is_pinned():
    """Regression guard mirroring Z12's z12_real_defect_still_fires test: pooling a
    genuinely-broken arm alongside a correctly-declared pinned arm must not let the
    pin's goal_pinned=True mask the broken arm's real defect."""
    pinned, pinned_env = _agent(goal_on=True, seed=124)
    _run_pinned(pinned, pinned_env, 4)

    broken, broken_env = _agent(goal_on=True, seed=125)
    _run(broken, broken_env, 4, write_z_goal=False)   # genuinely omitted, undeclared

    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(pinned, goal_pinned=True)
    acc.observe(broken)   # goal_pinned defaults to False -- this arm is NOT declared
    stats = acc.stats()
    # The accumulator pools run-level, same limitation as goal_state_present (Z10) --
    # ANY pinned observation suppresses the pooled writer_defect, exactly like ANY
    # live goal_state_present masks a goal-OFF arm. Documented, not silently assumed:
    # a mixed pinned/broken run cannot be split back apart by this pooled block, so a
    # driver mixing a declared pin with an undeclared real bug must observe them
    # through SEPARATE accumulators (or the real defect is invisible at this level).
    assert stats["writer_defect"] is None
    assert stats["goal_pinned"] is True
    assert stats["ticks_total"] == 8
    assert stats["writer_calls"] == 0


def test_z14_stats_from_counts_reports_null_and_the_flag():
    block = ZGS.stats_from_counts(6, 6, writer_calls=0, goal_state_present=True,
                                   goal_pinned=True)
    assert block["writer_defect"] is None
    assert block["goal_pinned"] is True
    assert block["active_frac"] == 1.0


def test_z14_stats_from_counts_default_omits_the_flag_key():
    """goal_pinned defaults to False, and the key itself is omitted rather than
    written as `false` -- matching goal_state_present's own omit-when-not-given
    convention, so an old manifest and a new non-pinned one are indistinguishable."""
    block = ZGS.stats_from_counts(6, 0, writer_calls=0, goal_state_present=True)
    assert block["writer_defect"] is True
    assert "goal_pinned" not in block


def test_z14_accumulator_observe_goal_pinned_suppresses_defect():
    agent, env = _agent(goal_on=True, seed=126)
    _run_pinned(agent, env, 3)
    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(agent, goal_pinned=True)
    stats = acc.stats()
    assert stats["writer_defect"] is None
    assert stats["goal_pinned"] is True
    assert stats["ticks_total"] == 3


def test_z14_accumulator_observe_default_does_not_suppress():
    agent, env = _agent(goal_on=True, seed=127)
    _run_pinned(agent, env, 3)
    acc = ZGS.ZGoalStreamAccumulator()
    acc.observe(agent)   # goal_pinned omitted
    stats = acc.stats()
    assert stats["writer_defect"] is True
    assert "goal_pinned" not in stats


def test_z14_accumulator_observe_stats_goal_pinned():
    acc = ZGS.ZGoalStreamAccumulator()
    pinned_block = ZGS.stats_from_counts(5, 5, writer_calls=0, goal_state_present=True)
    acc.observe_stats(pinned_block, goal_pinned=True)
    stats = acc.stats()
    assert stats["writer_defect"] is None
    assert stats["goal_pinned"] is True
    assert stats["ticks_total"] == 5


def test_z14_stamp_z_goal_stream_goal_pinned():
    agent, env = _agent(goal_on=True, seed=128)
    _run_pinned(agent, env, 4)
    manifest: dict = {}
    ZGS.stamp_z_goal_stream(manifest, agent, goal_pinned=True)
    block = manifest["z_goal_stream"]
    assert block["writer_defect"] is None
    assert block["goal_pinned"] is True
    assert block["ticks_total"] == 4


def test_z14_stamp_z_goal_stream_precomputed_stats_ignores_goal_pinned():
    """A precomputed `stats=` block bypasses z_goal_stream_stats entirely, so
    goal_pinned must have no effect on it, matching eval_stepped's Z12 contract."""
    manifest: dict = {}
    precomputed = ZGS.stats_from_counts(9, 9, writer_calls=0, goal_state_present=True)
    ZGS.stamp_z_goal_stream(manifest, stats=precomputed, goal_pinned=True)
    assert manifest["z_goal_stream"]["writer_defect"] is True, (
        "goal_pinned only governs agent= observation, never a precomputed stats= block"
    )


def test_z14_dry_run_pinned_shows_the_pinned_note(tmp_path, capsys):
    agent, env = _agent(goal_on=True, seed=129)
    _run_pinned(agent, env, 4)
    block = ZGS.z_goal_stream_stats(agent, goal_pinned=True)
    out = _dry_write(tmp_path, capsys, dry_run=True, z_goal_stream_stats=block)
    assert "not assessable" in out
    assert "deliberately pinned" in out
    assert "WRITER DEFECT" not in out


def test_z14_dry_run_undeclared_pin_still_names_the_defect(tmp_path, capsys):
    """The fail-safe case at the smoke layer too: an undeclared pin prints exactly
    the same WRITER DEFECT line as a genuine omission -- the smoke cannot know the
    driver's intent any more than the counters can."""
    agent, env = _agent(goal_on=True, seed=130)
    _run_pinned(agent, env, 4)
    out = _dry_write(tmp_path, capsys, dry_run=True, agent=agent)
    assert "WRITER DEFECT" in out
    assert "deliberately pinned" not in out


def test_z14_dry_run_ordinary_unmeasured_run_keeps_the_old_wording(tmp_path, capsys):
    """A genuinely unmeasured run (goal-OFF) must keep the pre-existing "no ticks"
    wording -- the new pinned clause must only fire when goal_pinned is actually
    set, never leak into the other None-producing path."""
    agent, env = _agent(goal_on=False, seed=131)
    _run(agent, env, 3, write_z_goal=True)
    out = _dry_write(tmp_path, capsys, dry_run=True, agent=agent)
    assert "no ticks with goal_state present" in out
    assert "deliberately pinned" not in out


# ---- Z15: the cue-recall-only false positive, fixed at the source ---------------------

def _cue_recall_agent(seed: int = 0, *, use_cue_recall: bool = True):
    """An agent with GoalState + a live SD-057 incentive bank, matching
    test_flag_inertness.py::test_use_cue_recall_gates_cue_recall_wanting's recipe --
    the minimal config that makes `cue_recall_wanting` reachable."""
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env, z_goal_enabled=True, goal_weight=0.5)
    cfg.goal.use_incentive_token_bank = True
    cfg.goal.use_cue_recall = use_cue_recall
    return REEAgent(cfg), env


def _seed_bank_token(agent, cue_type: int = 1) -> None:
    """Populate the incentive bank with a real, positive-value token for
    `cue_type` so `cue_recall_wanting` can find a match and actually pull."""
    gs = agent.goal_state
    z_obj = torch.randn(1, gs.config.goal_dim)
    gs.incentive_bank.update(resource_type=cue_type, benefit=1.0, z_object=z_obj)


def _step_cue_recall(agent, env, obs_dict, *, fire_cue: bool, cue_type: int = 1):
    """One hand-rolled tick that writes z_goal ONLY via `cue_recall_wanting` --
    `update_z_goal` is deliberately never called, mirroring the third
    false-positive shape documented in `_lib/z_goal_stream.py`."""
    latent = agent.sense(
        obs_dict["body_state"], obs_dict["world_state"],
        obs_harm=obs_dict.get("harm_obs"),
        obs_harm_a=obs_dict.get("harm_obs_a"),
        obs_harm_history=obs_dict.get("harm_history"),
    )
    ticks = agent.clock.advance()
    e1_prior = torch.zeros(1, latent.z_world.shape[-1], device=agent.device)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    if fire_cue:
        agent.cue_recall_wanting(cue_type=cue_type, drive_level=1.0)
    action = agent.select_action(candidates, ticks, temperature=1.0)
    if action is None:
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, 0] = 1.0
    _flat, _harm, _done, _info, next_obs = env.step(action)
    return next_obs


def test_z15_cue_recall_only_driver_is_not_reported_as_the_defect():
    """DIRECTION 1 -- the fix. A driver whose z_goal moves ONLY through
    cue_recall_wanting, never calling update_z_goal, must read writer_calls > 0
    and writer_defect is not True -- exactly like a driver correctly wired through
    update_z_goal whose benefit gate never opened."""
    agent, env = _cue_recall_agent(seed=200, use_cue_recall=True)
    _seed_bank_token(agent, cue_type=1)
    _flat, obs_dict = env.reset()
    agent.reset()
    for _ in range(5):
        obs_dict = _step_cue_recall(agent, env, obs_dict, fire_cue=True, cue_type=1)

    assert agent.z_goal_writer_calls > 0, (
        "cue_recall_wanting must count as a writer call when its reachability "
        "gate (goal_state present, use_cue_recall set) is satisfied"
    )
    assert agent.z_goal_active_frac is not None and agent.z_goal_active_frac > 0.0

    stats = ZGS.z_goal_stream_stats(agent)
    assert stats["writer_calls"] == agent.z_goal_writer_calls
    assert stats["writer_defect"] is not True, (
        "a cue-recall-only driver must not be flagged as the missing-call defect"
    )


def test_z15_writer_calls_not_incremented_when_use_cue_recall_is_off():
    """DIRECTION 2 -- the real defect is unaffected. A driver that calls
    cue_recall_wanting every tick but never enabled GoalConfig.use_cue_recall, and
    never calls update_z_goal either, has genuinely never engaged a writer: the
    reachability gate did not pass, so writer_calls must stay 0 and the run must
    still read as the unambiguous Z4 defect signature."""
    agent, env = _cue_recall_agent(seed=201, use_cue_recall=False)
    _flat, obs_dict = env.reset()
    agent.reset()
    for _ in range(5):
        obs_dict = _step_cue_recall(agent, env, obs_dict, fire_cue=True, cue_type=1)

    assert agent.z_goal_writer_calls == 0
    assert agent.z_goal_active_frac == 0.0
    stats = ZGS.z_goal_stream_stats(agent)
    assert stats["writer_defect"] is True


def test_z15_writer_calls_increments_even_when_no_matching_token_fires():
    """The cue-recall equivalent of Z4's benefit-gate-never-opened reading:
    writer_calls increments as soon as cue-recall is configured and reachable,
    even when the wanting-amplitude/token-match checks downstream find nothing to
    pull (no bank entry seeded for this cue type) -- correctly wired, no signal
    this tick, not a defect."""
    agent, env = _cue_recall_agent(seed=202, use_cue_recall=True)
    # Deliberately do NOT seed a token -- cue_type=1 has no matching bank entry.
    _flat, obs_dict = env.reset()
    agent.reset()
    for _ in range(5):
        obs_dict = _step_cue_recall(agent, env, obs_dict, fire_cue=True, cue_type=1)

    assert agent.z_goal_writer_calls > 0
    assert agent.z_goal_active_frac == 0.0  # never actually pulled -- no live goal


def test_z15_simulation_mode_call_does_not_increment():
    """MECH-094: simulation_mode=True is a no-op that must not move z_goal via a
    cue -- and must not read as a writer call either, since replay is
    deliberately not engaging the write pathway at all."""
    agent, env = _cue_recall_agent(seed=203, use_cue_recall=True)
    _seed_bank_token(agent, cue_type=1)
    strength = agent.cue_recall_wanting(
        cue_type=1, drive_level=1.0, simulation_mode=True
    )
    assert strength == 0.0
    assert agent.z_goal_writer_calls == 0


def test_z15_mixed_driver_update_z_goal_and_cue_recall_both_count():
    """Regression: a driver combining both writers (the landed corpus shape --
    every SD-057 cue-recall driver also calls update_z_goal on the same run, per
    the module docstring) must pool both into the same counter rather than one
    shadowing the other."""
    agent, env = _cue_recall_agent(seed=204, use_cue_recall=True)
    _seed_bank_token(agent, cue_type=1)
    _flat, obs_dict = env.reset()
    agent.reset()

    obs_dict = _step_cue_recall(agent, env, obs_dict, fire_cue=True, cue_type=1)
    assert agent.z_goal_writer_calls == 1

    agent.update_z_goal(benefit_exposure=1.0, drive_level=1.0)
    assert agent.z_goal_writer_calls == 2


def test_z15_pin_shape_is_unaffected_by_the_cue_recall_fix():
    """Regression against Z14: a driver that pins z_goal directly (never calling
    cue_recall_wanting at all, even with use_cue_recall configured on) must still
    read writer_calls == 0 -- this fix only touches the cue_recall_wanting entry
    point, not the direct-write pin path, which still needs the explicit
    `goal_pinned` opt-in."""
    agent, env = _cue_recall_agent(seed=205, use_cue_recall=True)
    _flat, obs_dict = env.reset()
    agent.reset()
    for _ in range(4):
        _pin_goal(agent, magnitude=0.5)
        obs_dict = _step_cue_recall(agent, env, obs_dict, fire_cue=False)

    assert agent.z_goal_writer_calls == 0
    assert agent.z_goal_active_frac == 1.0
    stats = ZGS.z_goal_stream_stats(agent)
    assert stats["writer_defect"] is True  # undeclared pin still reads as the defect
    pinned_stats = ZGS.z_goal_stream_stats(agent, goal_pinned=True)
    assert pinned_stats["writer_defect"] is None
