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
        manifest["machine"] = "stamped"
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
