"""Contracts for the MODULE-owned env-seed knob on `goal_stream_stages_sd054`,
and for its CLI exposure in the three SD-054 goal-stream drivers.

WHAT THIS PINS, and why it is separate from its two siblings.
`test_scaffold_env_seed_determinism.py` covers
`ScaffoldedSD054OnboardingConfig.scaffold_env_seed` (streams 0/1, the envs the
SCHEDULER builds). `test_driver_closure_env_seed_determinism.py` covers the
DRIVER-owned stream 2 on V3-EXQ-460c. Neither covers
`GoalStreamStagesConfig.goal_stream_env_seed` (ree-v3 856b8219d5), which is a
THIRD owner on a stream of its own -- stream 3 -- and which, until the wiring
this file also pins, was reachable only by setting a config field in code.

Mechanism, restated because it is why a knob was needed at all:
`CausalGridWorldV2` is a factory forwarding to `CausalGridWorld.__init__`, whose
only RNG is `self._rng = np.random.default_rng(seed)`.
`np.random.default_rng(None)` takes OS entropy AT CONSTRUCTION and does NOT
consume the numpy global RNG, so a driver's `np.random.seed(...)` /
`torch.manual_seed(...)` never reached these envs. The defect is per
CONSTRUCTION, not per process -- hence the per-construction counter.

WHY STREAM 3 rather than reusing the driver-owned stream 2: a driver may import
this module AND the scaffold module, so a shared stream could put two different
curricula on one layout. The four streams are pinned pairwise disjoint below.

WHAT EACH DRIVER'S --env-seed ACTUALLY DRIVES -- they are NOT the same knob, and
conflating them is the trap this docstring exists to close:
  * V3-EXQ-622  -- imports this module, so its flag drives
    `goal_stream_env_seed` (stream 3). This was a genuine determinism fix: every
    env its staged curriculum builds was on OS entropy.
  * V3-EXQ-603e -- imports the SCAFFOLD module, NOT this one (verified by AST),
    so its flag drives `scaffold_env_seed` (streams 0/1) for the scheduler
    curriculum, plus stream 2 for its own two envs. Also a genuine fix: the
    scheduler's P0/P1 envs were on OS entropy.
  * V3-EXQ-626a -- imports NEITHER module. It builds its own envs and `_build_env`
    already takes `seed` as a REQUIRED positional that every call site fills, so
    it had NO unseeded env. Its flag is therefore a LAYOUT knob (stream 2), not a
    determinism fix, and its default path must keep handing over the bare run
    seed. Pinned explicitly below so a future edit cannot quietly redefine it.

BOTH directions are pinned, and both matter:
  * knob UNSET -> `seed=None` still reaches the factory, the counter does NOT
    advance, and every driver's default is bit-identical to its landed runs;
  * knob SET   -> deterministic, distinct, reproducible, collision-free seeds,
    and a same-seed env reproduces bitwise.

KNOWN DIVERGENCE, DELIBERATE -- 603e's two driver envs (documented 2026-07-29).
The batch's governing principle is that a PINNED run should differ from a landed
run only in seed VALUES, never in structural RELATIONSHIPS. 603e breaks that in
exactly one place, knowingly, and this is the record of why.

On the DEFAULT path both of 603e's own envs are built with the identical call --
`CausalGridWorldV2(seed=seed, **_target_env_kwargs(scaffold_cfg))` -- so they are
BIT-IDENTICAL layouts (measured, all three seeds). Under `--env-seed` they are
given distinct idx on stream 2 (0 = the dims probe, 1 = the bespoke P2
measurement env), so a pinned run has them DIFFERENT where a landed run has them
the same.

Why this is harmless, and why the divergence was kept rather than repaired:
  * `target_env` is a DIMS PROBE. It appears at exactly one line --
    `cfg = _make_config(target_env, arm)` -- to read body/world/action dims off
    the env. It is never `reset()`, never stepped, and discarded immediately.
    Its layout is therefore unobservable in any result, which is what makes the
    seed relationship between the two envs scientifically inert.
  * Distinct idx is the DEFENSIVE choice. If a future edit ever starts stepping
    `target_env`, distinct layouts prevent it from silently sharing a world with
    the measurement env; the faithful-to-landed choice (one shared idx) would
    couple them at exactly the moment that coupling started to matter.
  * A pinned run is ALREADY declared not comparable to a landed one -- every
    driver's `--env-seed` help text and its manifest `env_seed_base` comment say
    so -- so the divergence cannot mislead a reader who follows that contract.

TWO NEIGHBOURING BIT-IDENTITIES, pre-existing and NOT introduced by the knob,
recorded here because they look like the same finding and are not:
  * 626a rebuilds its P1 env every episode from the SAME run seed, so every P1
    episode shares one layout;
  * 626a's P1 and P2 envs share placement kwargs and the run seed, so the P2
    measurement phase runs on the exact world trained on in P1.
Neither is a scoring hazard: 626a carries `CLAIM_IDS = []` with
`experiment_purpose = "diagnostic"`, so it is excluded from confidence/conflict
scoring, and its C1-C5 criteria are all within-world ACROSS-ARM comparisons,
which a shared layout does not confound (it removes layout variance between
arms). The one reading it will NOT support is generalisation: P2 is not a
held-out world. These are observed properties of a landed driver, NOT ratified
design intent -- the 626a docstring says nothing about a fixed P1 world.

If you change the idx assignment below, update this block and say why. The test
is here to make the choice conscious, not to freeze it.

Env-level rather than whole-curriculum on purpose: the causal chain is
seed -> env layout -> observations, and a full dry-run of any of these drivers
costs minutes (a 626a/622-class full-curriculum dry-run was measured at >5m37s
unfinished on 2026-07-28).
"""
import inspect
import os
import sys
from dataclasses import fields

import numpy as np
import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO, os.path.join(_REPO, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# PACKAGE spelling for the two LIBRARY modules, bare-name for the three
# DRIVERS -- the asymmetry is deliberate. `d622` imports
# `experiments.goal_stream_stages_sd054`, so a bare `import
# goal_stream_stages_sd054` here produced a SECOND, distinct module object for
# the very module this file pins (measured 2026-07-29: the two spellings were
# not the same object), with separate module-level state -- so a monkeypatch on
# `gs` would silently miss the copy `d622` actually runs. The drivers stay
# bare-name because that is how experiment_runner.py loads them and nothing
# else in this process imports them by the package path.
import experiments.goal_stream_stages_sd054 as gs  # noqa: E402
import v3_exq_603e_q045_mech313_mech260_scaffolded_sd054 as d603  # noqa: E402
import v3_exq_622_goal_stream_staged_sd054 as d622  # noqa: E402
import v3_exq_626a_goal_pipeline_developmental_window_diagnostic as d626  # noqa: E402
from experiments.scaffolded_sd054_onboarding import _derive_env_seed  # noqa: E402

# Stream ids, restated as constants so a future edit that renumbers them fails
# here rather than silently colliding two callers onto one env layout.
CURRICULUM_STREAM = 0   # scaffolded_sd054_onboarding scheduler builds
PROBE_STREAM = 1        # scaffolded_sd054_onboarding harm-discriminativeness probe
DRIVER_STREAM = 2       # a driver's own envs
GOAL_STREAM_STREAM = 3  # goal_stream_stages_sd054's curriculum builds

STAGES = ("S0", "S1", "S2", "S3")


def _cfg(**kw):
    return gs.GoalStreamStagesConfig(steps_per_episode=5, **kw)


def _layout(env):
    """Layout identity of a constructed env -- what the seed actually controls.

    Deliberately NOT a rollout: these comparisons are about the world the seed
    lays out at CONSTRUCTION, before any policy touches it.
    """
    return (tuple(map(tuple, getattr(env, "resources", []) or [])),
            tuple(map(tuple, getattr(env, "hazards", []) or [])))


# --------------------------------------------------------------------------- #
# Default path: bit-identical to every landed run of every importing driver
# --------------------------------------------------------------------------- #

def test_config_knob_defaults_to_none():
    """The knob must default OFF. Pinning unconditionally would change every
    stage's env layout, and therefore the results, for runs already on record."""
    fld = {f.name: f for f in fields(gs.GoalStreamStagesConfig)}
    assert "goal_stream_env_seed" in fld
    assert fld["goal_stream_env_seed"].default is None


def test_next_env_seed_returns_none_when_unset():
    assert gs.GoalStreamStagesRunner(_cfg())._next_env_seed() is None


def test_counter_does_not_advance_when_unset():
    """The load-bearing half of the default guarantee. If the counter advanced
    on the None path, merely *reading* it would desynchronise the idx sequence
    against a pinned run, so the two would not be comparable."""
    r = gs.GoalStreamStagesRunner(_cfg())
    assert r._env_build_count == 0
    for _ in range(5):
        assert r._next_env_seed() is None
    assert r._env_build_count == 0


def test_build_env_hands_seed_none_to_the_factory_by_default(monkeypatch):
    """Not merely 'omits seed' -- the factory must still SEE None, which is what
    makes the explicit pass-through bit-identical to the pre-knob call."""
    for stage in STAGES:
        seen = {}
        monkeypatch.setattr(
            "ree_core.environment.causal_grid_world.CausalGridWorldV2",
            lambda **kw: seen.update(kw) or object(),
        )
        gs._build_env(_cfg(), stage)
        assert seen.get("seed", "<missing>") is None, stage


def test_build_env_seed_parameter_defaults_to_none():
    assert inspect.signature(gs._build_env).parameters["seed"].default is None


# --------------------------------------------------------------------------- #
# Knob SET: deterministic, distinct, collision-free
# --------------------------------------------------------------------------- #

def test_next_env_seed_uses_stream_3_and_advances():
    base = 4242
    r = gs.GoalStreamStagesRunner(_cfg(goal_stream_env_seed=base))
    got = [r._next_env_seed() for _ in range(4)]
    assert got == [_derive_env_seed(base, stream=GOAL_STREAM_STREAM, idx=i)
                   for i in range(4)]
    assert r._env_build_count == 4


def test_every_construction_gets_a_distinct_seed():
    """Keyed on construction ORDER, not stage: S0/S1/S2 build a FRESH env every
    episode, so a per-stage idx would collapse all of a stage's episodes onto one
    shared layout and silently remove the across-episode variation."""
    r = gs.GoalStreamStagesRunner(_cfg(goal_stream_env_seed=7))
    got = [r._next_env_seed() for _ in range(64)]
    assert len(set(got)) == 64


def test_stream_3_cannot_collide_with_the_other_three_streams():
    """The reason this module took a stream of its own: V3-EXQ-603e-class drivers
    import a scaffold curriculum too, and a shared stream would put two different
    curricula on one layout."""
    base = 99
    mine = {_derive_env_seed(base, stream=GOAL_STREAM_STREAM, idx=i)
            for i in range(2000)}
    others = {
        _derive_env_seed(base, stream=s, idx=i)
        for s in (CURRICULUM_STREAM, PROBE_STREAM, DRIVER_STREAM)
        for i in range(2000)
    }
    assert not (mine & others)


def test_all_four_streams_are_pairwise_disjoint_across_bases():
    """Stronger than the single-base check above: adjacent BASES must not bleed
    into each other either, which is what makes `base + run_seed` folding safe."""
    seen = {}
    for stream in (CURRICULUM_STREAM, PROBE_STREAM, DRIVER_STREAM, GOAL_STREAM_STREAM):
        for base in range(0, 40):
            for idx in range(500):
                key = _derive_env_seed(base, stream=stream, idx=idx)
                assert key not in seen or seen[key] == stream, (
                    f"seed {key} reachable from stream {seen.get(key)} and {stream}"
                )
                seen[key] = stream


def test_build_env_forwards_the_seed_on_every_stage_branch(monkeypatch):
    """The seed rides in `common`, which all four stage branches splat. Pinned
    per branch so a NEW stage branch that forgets the splat fails here rather
    than silently reverting that stage to OS entropy."""
    for stage in STAGES:
        seen = {}
        monkeypatch.setattr(
            "ree_core.environment.causal_grid_world.CausalGridWorldV2",
            lambda **kw: seen.update(kw) or object(),
        )
        gs._build_env(_cfg(goal_stream_env_seed=1), stage, seed=31337)
        assert seen.get("seed") == 31337, stage


# --------------------------------------------------------------------------- #
# CLI exposure -- the knob is reachable from the command line, not only in code
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", [
    "experiments/v3_exq_622_goal_stream_staged_sd054.py",
    "experiments/v3_exq_626a_goal_pipeline_developmental_window_diagnostic.py",
    "experiments/v3_exq_603e_q045_mech313_mech260_scaffolded_sd054.py",
])
def test_driver_exposes_env_seed_flag_defaulting_to_none(path):
    """Source-level rather than by importing __main__: these modules build a
    parser only under `if __name__ == "__main__"`, so there is nothing to
    introspect after import."""
    src = open(os.path.join(_REPO, path)).read()
    assert '"--env-seed"' in src, f"{path} exposes no --env-seed flag"
    i = src.index('"--env-seed"')
    window = src[i:i + 200]
    assert "type=int" in window and "default=None" in window, (
        f"{path}'s --env-seed must be an opt-in int defaulting to None -- a "
        "non-None default would change every landed run of this driver"
    )


def test_622_threads_the_module_knob_and_defaults_it_off():
    assert (inspect.signature(d622.make_stage_cfg)
            .parameters["env_seed"].default is None)
    assert d622.make_stage_cfg(1, 1, 1, 1, 5).goal_stream_env_seed is None
    assert d622.make_stage_cfg(1, 1, 1, 1, 5, env_seed=88).goal_stream_env_seed == 88
    assert (inspect.signature(d622.run_seed)
            .parameters["env_seed_base"].default is None)


def test_622_run_seeds_do_not_collapse_onto_one_layout():
    """`run_seed` folds the run seed into the base. Without that fold, one
    --env-seed would give all of SEEDS the same world -- a fresh
    GoalStreamStagesRunner (and so a fresh counter) is built per seed -- and the
    seed dimension would be silently destroyed."""
    base = 1000
    per_seed = {
        _derive_env_seed(base + s, stream=GOAL_STREAM_STREAM, idx=idx)
        for s in d622.SEEDS
        for idx in range(8)
    }
    assert len(per_seed) == 8 * len(d622.SEEDS)


def test_603e_threads_the_scaffold_knob_and_defaults_it_off():
    """603e imports the SCAFFOLD module, not this one, so its flag drives
    `scaffold_env_seed`. Pinned so a future edit does not point it at the wrong
    knob on the strength of the filename."""
    assert d603._make_scaffold_cfg().scaffold_env_seed is None
    assert d603._make_scaffold_cfg(env_seed=555).scaffold_env_seed == 555
    assert (inspect.signature(d603._run_arm_seed)
            .parameters["env_seed_base"].default is None)


def test_603e_does_not_import_the_goal_stream_module():
    """Guards the docstring's claim. If 603e ever DOES import this module, the
    two curricula share a run and the stream-disjointness argument above becomes
    load-bearing for it too -- so this should fail loudly, not drift."""
    import ast
    src = open(os.path.join(
        _REPO, "experiments/v3_exq_603e_q045_mech313_mech260_scaffolded_sd054.py")).read()
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
    assert not any("goal_stream_stages_sd054" in m for m in mods)


# --------------------------------------------------------------------------- #
# The KNOWN DIVERGENCE (see the module docstring). Executable documentation:
# these three pin BOTH sides of it, so it cannot drift silently in either
# direction and cannot be "tidied" without a conscious decision.
# --------------------------------------------------------------------------- #

def test_603e_default_gives_its_two_envs_the_SAME_layout():
    """The landed relationship. Both driver envs are built with the identical
    call, so on the default path they are bit-identical -- pinned here because
    it is the baseline the divergence below is measured against, and because a
    future edit that quietly de-duplicated them would change a landed run."""
    cfg = d603._make_scaffold_cfg()
    assert cfg.scaffold_env_seed is None
    kw = d603._target_env_kwargs(cfg)
    for seed in d603.SEEDS:
        a = _layout(d603.CausalGridWorldV2(seed=seed, **kw))
        b = _layout(d603.CausalGridWorldV2(seed=seed, **kw))
        assert a == b, f"seed {seed}: the landed default no longer shares a layout"


def _603e_driver_stream_idx():
    """The (stream, idx) pairs 603e ACTUALLY passes to `_derive_env_seed`.

    Read from source rather than re-derived: asserting that idx 0 and idx 1
    differ would pass even if the driver used idx 0 at both sites, which is
    precisely the change this is meant to catch.
    """
    import ast
    src = open(os.path.join(
        _REPO, "experiments/v3_exq_603e_q045_mech313_mech260_scaffolded_sd054.py")).read()
    pairs = []
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_derive_env_seed"):
            continue
        kw = {k.arg: k.value for k in n.keywords}
        if not {"stream", "idx"} <= set(kw):
            continue
        try:
            pairs.append((ast.literal_eval(kw["stream"]), ast.literal_eval(kw["idx"])))
        except ValueError:
            continue  # non-literal -- not one of the two fixed driver sites
    return pairs


def test_603e_pinned_gives_its_two_envs_DISTINCT_layouts():
    """The divergence itself. Deliberate and defensive -- `target_env` is a dims
    probe that is never stepped, so this is unobservable in any result; distinct
    idx keeps the two from silently sharing a world if one ever IS stepped.

    Pinned from 603e's OWN call sites, then carried through to real layouts, so
    the assertion fails if either the idx assignment or the derivation changes.
    """
    driver_sites = [p for p in _603e_driver_stream_idx() if p[0] == DRIVER_STREAM]
    assert len(driver_sites) == 2, (
        f"expected exactly 2 driver-stream sites in 603e, found {driver_sites}"
    )
    idxs = sorted(i for _s, i in driver_sites)
    assert idxs == [0, 1], (
        f"603e's driver-owned idx assignment changed to {idxs}. If the two envs "
        "were deliberately collapsed onto one idx (the faithful-to-landed "
        "choice), update the KNOWN DIVERGENCE block in this file's docstring and "
        "flip this test to assert equality."
    )

    sb = 7 + d603.SEEDS[0]
    kw = d603._target_env_kwargs(d603._make_scaffold_cfg(env_seed=sb))
    seeds = [_derive_env_seed(sb, stream=DRIVER_STREAM, idx=i) for i in idxs]
    assert seeds[0] != seeds[1]
    assert _layout(d603.CausalGridWorldV2(seed=seeds[0], **kw)) != \
           _layout(d603.CausalGridWorldV2(seed=seeds[1], **kw))


def test_603e_target_env_is_never_stepped():
    """THE LOAD-BEARING PREMISE of the divergence being harmless. If
    `target_env` ever gains a `.reset()` / `.step()` call, its layout becomes
    observable, the pinned-vs-default asymmetry stops being inert, and this
    decision must be revisited -- so assert the premise rather than trusting the
    prose."""
    import ast
    src = open(os.path.join(
        _REPO, "experiments/v3_exq_603e_q045_mech313_mech260_scaffolded_sd054.py")).read()
    uses = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "target_env"]
    assert not uses, (
        "target_env now has attribute access "
        f"({sorted({u.attr for u in uses})}) -- it was a write-only dims probe "
        "when the pinned/default idx divergence was accepted as harmless. "
        "Re-read the KNOWN DIVERGENCE block in this file's docstring."
    )


def test_626a_default_still_hands_over_the_bare_run_seed(monkeypatch):
    """626a had NO unseeded env, so its flag is a LAYOUT knob. Its default path
    must keep passing the bare run seed -- anything else would change a landed
    run. Pinned by driving `run_seed_arm` itself, not by re-deriving."""
    assert (inspect.signature(d626.run_seed_arm)
            .parameters["env_seed_base"].default is None)

    seeds_seen = []
    real_build = d626._build_env
    monkeypatch.setattr(d626, "_build_env",
                        lambda **kw: seeds_seen.append(kw.get("seed")) or real_build(**kw))
    monkeypatch.setattr(d626, "build_agent", lambda *a, **k: _StubAgent())
    monkeypatch.setattr(d626, "_run_episode", lambda *a, **k: _ZeroDict(episode_length=1))
    monkeypatch.setattr(d626, "_summarise_phase", lambda name, eps: _Zeros())

    try:
        d626.run_seed_arm(
            seed=42, arm=d626.arm_configs()[0], device=torch.device("cpu"),
            world_obs_dim=8, p0_eps=1, p1_eps=1, p2_eps=1, steps_per_episode=2,
            total_training_eps=2,
        )
    except Exception:  # noqa: BLE001 -- stubbed summaries trip downstream arithmetic
        pass           # every _build_env call site is upstream of that

    assert seeds_seen, "no env was built -- the stub short-circuited too early"
    assert all(s == 42 for s in seeds_seen), (
        f"626a's default path must hand over the bare run seed, got {seeds_seen}"
    )


class _ZeroDict(dict):
    def __missing__(self, k):
        return 0.0


class _Zeros:
    def __getattr__(self, k):
        return 0.0


class _StubAgent:
    """Only the attributes 626a touches before the (stubbed) episode loop."""

    def __init__(self):
        p = torch.nn.Parameter(torch.zeros(1))

        class _M:
            def parameters(self_inner):
                return [p]

        class _E2:
            def __init__(self_inner):
                self_inner.world_transition = _M()
                self_inner.world_action_encoder = _M()

        class _Cfg:
            class latent:
                world_dim = 4

            class goal:
                drive_floor = 0.0

            use_mech295_liking_bridge = True
            use_mech307_conjunction = True

        self.e1 = _M()
        self.e2 = _E2()
        self.config = _Cfg()
        self.goal_state = None

    def train(self):
        return self

    def eval(self):
        return self


# --------------------------------------------------------------------------- #
# The property that actually matters: same seed -> same env trajectory
# --------------------------------------------------------------------------- #

def _rollout_fingerprint(seed, stage="S0", n_steps=12):
    """Deterministic scripted rollout -- no policy, so only the env is tested."""
    env = gs._build_env(_cfg(), stage, seed=seed)
    obs, _info = env.reset()
    arr = obs.detach().cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs)
    parts = [arr.copy(), np.array([env.agent_x, env.agent_y], dtype=np.float64)]
    for t in range(n_steps):
        nxt, _reward, done, _info, _obs_dict = env.step(t % env.action_dim)
        parts.append(
            nxt.detach().cpu().numpy() if torch.is_tensor(nxt) else np.asarray(nxt)
        )
        parts.append(np.array([env.agent_x, env.agent_y], dtype=np.float64))
        if done:
            break
    return np.concatenate([p.ravel().astype(np.float64) for p in parts])


def test_same_seed_gives_bit_identical_stage_env_trajectory():
    """The causal link the knob buys: seed -> layout -> observations."""
    a = _rollout_fingerprint(20260729)
    b = _rollout_fingerprint(20260729)
    assert np.array_equal(a, b), (
        "two stage envs built with the SAME explicit seed diverged -- the env "
        "carries an entropy source beyond its own _rng, and seeding the "
        "curriculum cannot make these drivers reproducible"
    )


def test_global_reseeding_does_not_rescue_the_unseeded_stage_env():
    """Pins the DEFECT itself, deterministically, on this module's call path.

    `np.random.default_rng(None)` ignores the numpy global RNG, so reseeding the
    globals identically before each construction -- exactly what these drivers do
    per seed -- cannot make unseeded envs agree. Asserted as 'not all identical'
    over several builds rather than 'the first two differ', so a chance layout
    collision cannot flake it.
    """
    fps = []
    for _ in range(6):
        np.random.seed(42)
        torch.manual_seed(42)
        fps.append(_rollout_fingerprint(None, n_steps=4))
    assert not all(np.array_equal(fps[0], f) for f in fps[1:]), (
        "unseeded stage envs became reproducible under global reseeding -- if "
        "the env RNG was genuinely made seed-following, retire this test and the "
        "knob rather than weakening the assertion"
    )


def test_seeded_and_unseeded_stage_envs_coexist():
    """A seeded build must not perturb a later unseeded one into determinism, or
    vice versa -- the default path has to survive alongside a pinned one."""
    seeded_1 = _rollout_fingerprint(777, n_steps=4)
    _ = _rollout_fingerprint(None, n_steps=4)
    seeded_2 = _rollout_fingerprint(777, n_steps=4)
    assert np.array_equal(seeded_1, seeded_2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
