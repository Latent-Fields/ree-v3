"""Contract: `measure_action_mass` really is non-destructive (SD-074 / GFLAG-0036).

THE DEFECT THIS PINS. The function documented itself as "the caller gets back
bit-identically the agent it passed in" and restored `agent.state_dict()` plus a
HAND-LISTED tuple of four E3 attribute names. `nn.Module.state_dict()` carries
registered Parameters and buffers ONLY, so that left the whole plain-Python
attribute surface -- 586 attributes across 121 modules -- unprotected. Measured
(REE_assembly/evidence/planning/probe_warmup_nonbuffer_audit_2026-08-15.md): a
single 40-step read left 21 attributes changed, 17 of them surviving
`agent.reset()`.

The load-bearing ones were not in E3 at all. `_self_experience_buffer`,
`_world_experience_buffer` and `_e2_transition_buffer` are the pool
`agent.compute_prediction_loss()` (ree_core/agent.py:9970) samples a RANDOM WINDOW
from on every `warmup_train` tick, so a probe read fed EVAL-ROLLOUT DATA INTO THE
E1 TRAINING SET. They cap at 1000 (agent.py:5319) and V3-EXQ-784's realised reads
are 1081-1401 env steps, so one read fully DISPLACED the pool. Replicating 784's
ladder at its real scale, restoring the full surface changed the dependent variable
in 6 of 6 post-first-read cells and FLIPPED the saturation regime in 2 of 6.

WHY THE CENSUS IS NOT A NAME LIST HERE, and why that is the stronger form of the
S2 pattern in test_preservation_midlife_spike.py. S2 pins a per-store partition by
name because that store's restore IS a hand-list, so a new attribute has to be
classified by a human. `capture_agent_surface` inverts the default: it restores
EVERYTHING, so the thing that needs pinning is the two ways an attribute can escape
that default --

  * `_MODULE_INTERNALS`  -- torch's own bookkeeping keys, derived mechanically from
    a live `nn.Module` so a torch upgrade cannot silently add one (S2's failure mode
    in reverse);
  * `by_reference`       -- attributes the copier could NOT protect. PINNED EMPTY
    below. A new uncopyable attribute makes that test fail, and someone then has to
    classify it -- which is exactly S2's "a new attribute fails the contract until
    someone decides", applied at the only place a decision is still needed.

and then to assert the DEFAULT actually holds, behaviourally: `test_read_leaves_no_drift`
fingerprints the entire surface plus the entire state_dict either side of a REAL
read and requires an EMPTY diff. That is what makes this a contract about the
agent rather than about a list of names -- a new attribute added to any of the 121
modules is covered the day it lands, with nobody having to remember this file.

`test_read_without_restore_does_drift` is the non-vacuity control: it neuters the
restore and asserts the read demonstrably DOES contaminate the training buffers, so
a future refactor cannot make `test_read_leaves_no_drift` pass by making the read
do nothing.

ASCII-only. Run: pytest tests/contracts/test_probe_warmup_nondestructive.py -q
"""

from __future__ import annotations

import collections
import copy
import hashlib
import types

import pytest
import torch
import torch.nn as nn

from experiments._lib import probe_warmup
from experiments._lib.probe_warmup import (
    _BY_REFERENCE_TYPES,
    _MODULE_INTERNALS,
    capture_agent_surface,
    measure_action_mass,
    reapply_candidate_capture,
    restore_agent_surface,
)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _mk_env_agent():
    """V3-EXQ-784's operating config, on the smallest env that still exercises it.

    The config knobs (control-vector logging ON, action-class scaffold candidates
    ON, both regulators OFF) are 784's exactly -- they are what decide which of the
    ~10 optional stores get CONSTRUCTED, and therefore which attributes exist at
    all. The env is shrunk purely for wall clock; the attribute surface it produces
    is the same shape.
    """
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    from ree_core.utils.config import REEConfig

    env = CausalGridWorldV2(seed=11, size=5, num_hazards=1, num_resources=1)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    cfg.use_tonic_vigor = False
    cfg.use_noise_floor = False
    torch.manual_seed(0)
    return env, REEAgent(cfg)


def _read(agent, env, **kw):
    kw.setdefault("seed", 3)
    kw.setdefault("n_selections", 3)
    kw.setdefault("max_env_steps", 25)
    kw.setdefault("max_episodes", 25)
    kw.setdefault("steps_per_episode", 25)
    return measure_action_mass(agent, env, label="contract", **kw)


def _fingerprint(value, depth: int = 0) -> str:
    """Content hash of an arbitrary attribute value. Never uses id()."""
    if depth > 10:
        return "<deep>"
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return repr(value)
    if isinstance(value, float):
        return "f%.17g" % value
    if torch.is_tensor(value):
        flat = value.detach().cpu().flatten().contiguous()
        digest = (hashlib.sha1(flat.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                  if flat.numel() else "")
        return "T%s%s%s" % (tuple(value.shape), value.dtype, digest)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fingerprint(v, depth + 1) for v in value) + "]"
    if isinstance(value, collections.deque):
        return "D[" + ",".join(_fingerprint(v, depth + 1) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join("%r:%s" % (k, _fingerprint(value[k], depth + 1))
                              for k in sorted(value, key=repr)) + "}"
    if isinstance(value, (set, frozenset)):
        return "S{" + ",".join(sorted(_fingerprint(v, depth + 1) for v in value)) + "}"
    if isinstance(value, _BY_REFERENCE_TYPES):
        return "%s@%s" % (type(value).__name__, getattr(value, "__name__", "?"))
    if isinstance(value, nn.Module):
        return "Module<%s>" % type(value).__name__
    inner = getattr(value, "__dict__", None)
    if inner is not None:
        return "%s(%s)" % (type(value).__name__, _fingerprint(inner, depth + 1))
    return "%s@opaque" % type(value).__name__


def _attr_paths(agent):
    out = {}
    for mpath, module in agent.named_modules():
        for name, value in vars(module).items():
            if name in _MODULE_INTERNALS:
                continue
            out[("%s.%s" % (mpath, name)) if mpath else name] = value
    return out


def _whole_agent_fingerprint(agent):
    """Everything a resume could get wrong: the attribute surface AND state_dict."""
    out = {path: _fingerprint(v) for path, v in _attr_paths(agent).items()}
    for key, tensor in agent.state_dict().items():
        out["state_dict:" + key] = _fingerprint(tensor)
    return out


def _diff(before, after):
    return sorted(k for k in set(before) | set(after)
                  if before.get(k, "<absent>") != after.get(k, "<absent>"))


# --------------------------------------------------------------------------- #
# the census: the partition is exhaustive, disjoint, and mechanically derived   #
# --------------------------------------------------------------------------- #

def test_census_partition_is_exhaustive_and_disjoint():
    """Every live attribute is either torch bookkeeping or captured. No third bucket."""
    _env, agent = _mk_env_agent()
    surface = capture_agent_surface(agent)

    captured_by_module = {mpath: set(values) for _m, mpath, values, _n in surface.entries}
    for mpath, module in agent.named_modules():
        live = set(vars(module))
        captured = captured_by_module[mpath]
        internals = live & _MODULE_INTERNALS
        assert captured | internals == live, (
            "module %r grew attribute(s) that are neither captured nor torch "
            "bookkeeping: %s" % (mpath or "<agent>", sorted(live - captured - internals))
        )
        assert not (captured & _MODULE_INTERNALS), (
            "module %r: captured set overlaps torch bookkeeping: %s"
            % (mpath or "<agent>", sorted(captured & _MODULE_INTERNALS))
        )

    assert surface.n_modules > 100, "fixture no longer builds the full agent tree"
    assert surface.n_attrs > 400, "attribute surface collapsed -- capture is not walking"


def test_module_internals_are_exactly_torch_bookkeeping_minus_training():
    """`training` MUST be captured; every other nn.Module key MUST be skipped.

    Derived from a live probe instance rather than hardcoded, so a torch upgrade
    that adds a bookkeeping key is excluded automatically instead of being copied
    into the snapshot. `training` is the deliberate exception: the read calls
    agent.eval(), which flips it on all 121 submodules, and restoring it per-module
    is what lets the read drop the old agent-level train()/eval() dance -- which
    would have woken a submodule the caller had deliberately left in eval mode.
    """
    assert _MODULE_INTERNALS == frozenset(vars(nn.Module()).keys()) - {"training"}
    assert "training" not in _MODULE_INTERNALS
    for key in ("_parameters", "_buffers", "_modules", "_forward_hooks"):
        assert key in _MODULE_INTERNALS, "%s must stay excluded" % key


def test_by_reference_is_empty_under_the_operating_config():
    """PINNED EMPTY. A new entry here is a real unprotected leak, not a nit.

    `by_reference` collects attributes `_copy_state` could not snapshot: they are
    handed back to the agent unchanged, so a read CAN carry them forward. Under
    784's config the copier reaches everything, including the four trajectory-shaped
    attributes that plain `copy.deepcopy` chokes on. If this fires, classify the new
    attribute -- decide whether it is behaviour-bearing, and either teach
    `_copy_state` to handle its type or record here why leaving it live is safe.
    """
    _env, agent = _mk_env_agent()
    assert capture_agent_surface(agent).by_reference == []


def test_load_bearing_attributes_are_in_the_captured_surface():
    """Name the incident's own attributes, so a refactor that drops them fails LOUDLY.

    Deliberately redundant with the behavioural drift test: that one catches any
    regression, this one says WHICH attributes made 2 of 6 replicated V3-EXQ-784
    cells change their saturation verdict, so the failure message points a reader at
    the audit instead of at 586 anonymous attributes.
    """
    _env, agent = _mk_env_agent()
    captured = set()
    for _module, mpath, values, _names in capture_agent_surface(agent).entries:
        for name in values:
            captured.add(("%s.%s" % (mpath, name)) if mpath else name)

    # The training-pool contamination channel (ree_core/agent.py:9970 samples these).
    for path in ("_self_experience_buffer", "_world_experience_buffer",
                 "_e2_transition_buffer"):
        assert path in captured, "%s is the leak channel and MUST be captured" % path
    # Accumulators the audit measured surviving agent.reset(). Note the visitation
    # counter is a REGISTERED submodule, so the thing that drifts is its `_next_idx`
    # attribute and not an agent-level `visitation_counter` attribute -- the chip
    # brief's ~10-store list assumed the latter, and the census is what corrected it.
    for path in ("_z_goal_writer_calls", "_step_count",
                 "residue_field._harm_history",
                 "hippocampal.visitation_counter._next_idx"):
        assert path in captured, "%s drifts across a read and MUST be captured" % path
    # E3: the three real members of the old hand-list, plus the two it MISSED.
    for path in ("e3._running_variance", "e3._rv_history", "e3._novelty_ema",
                 "e3._volatility_estimate", "e3._last_instantaneous_pe"):
        assert path in captured, "%s MUST be captured" % path


def test_the_hand_listed_e3_tuple_is_gone_and_its_phantom_name_never_existed():
    """The old `_E3_NONBUFFER_STATE` named an attribute that does not exist.

    `_last_error_var` has zero occurrences in ree_core -- it silently captured
    nothing for the whole life of the module, while the two attributes it was
    presumably reaching for (`e3._last_instantaneous_pe`, the SD-069 instantaneous-PE
    source, and `e3._volatility_estimate`, the Q-007 LC-NE tonic volatility signal)
    went uncaptured. Both are inert under 784's config -- volatility_signal_dim
    defaults to 0 and phasic_burst_signal_source defaults to "running_variance" --
    so this was a LATENT defect that would have gone live for the first consumer to
    turn either knob on. Pinned as a name-level fact so the hand-list cannot come
    back.
    """
    assert not hasattr(probe_warmup, "_E3_NONBUFFER_STATE"), (
        "the hand-listed E3 tuple is back -- hand-listing attribute names is the "
        "defect (see this module's docstring), not the fix"
    )
    _env, agent = _mk_env_agent()
    assert not hasattr(agent.e3, "_last_error_var"), (
        "an attribute named _last_error_var now exists; the old hand-list may have "
        "been right after all -- re-read the 2026-08-15 audit before acting"
    )
    assert hasattr(agent.e3, "_last_instantaneous_pe")
    assert hasattr(agent.e3, "_volatility_estimate")


# --------------------------------------------------------------------------- #
# the behavioural contract: a read leaves NOTHING changed                       #
# --------------------------------------------------------------------------- #

def test_read_leaves_no_drift():
    """THE load-bearing assertion. Whole surface + whole state_dict, empty diff."""
    env, agent = _mk_env_agent()
    before = _whole_agent_fingerprint(agent)

    read = _read(agent, env)

    drift = _diff(before, _whole_agent_fingerprint(agent))
    assert drift == [], (
        "measure_action_mass mutated %d thing(s) it promised not to: %s -- this is "
        "the GFLAG-0036 defect class; do NOT fix it by adding names to a list"
        % (len(drift), drift[:12])
    )
    assert read["restore_report"]["n_by_reference"] == 0
    assert read["restore_report"]["n_attrs"] > 400


def test_read_without_restore_does_drift(monkeypatch):
    """Non-vacuity control for the test above: the read really does contaminate.

    Neuters ONLY the surface restore, so what is left is exactly the pre-2026-08-15
    behaviour (state_dict restored, attribute surface not). The experience buffers
    must grow -- that growth IS eval-rollout data entering the E1 training pool,
    which `agent.compute_prediction_loss()` samples on every subsequent
    `warmup_train` tick.
    """
    env, agent = _mk_env_agent()
    monkeypatch.setattr(probe_warmup, "restore_agent_surface", lambda a, s: {
        "n_by_reference": 0, "by_reference": [], "n_attrs": 0})

    n_self_before = len(agent._self_experience_buffer)
    n_world_before = len(agent._world_experience_buffer)
    _read(agent, env)

    assert len(agent._self_experience_buffer) > n_self_before, (
        "the read no longer grows _self_experience_buffer -- if this is a deliberate "
        "substrate change, test_read_leaves_no_drift has become vacuous and this "
        "whole contract needs re-deriving against the new channel"
    )
    assert len(agent._world_experience_buffer) > n_world_before


def test_read_restores_a_deliberate_mutation():
    """The restore mechanism itself, isolated from the rollout.

    Fast, direct proof that capture -> mutate -> restore round-trips the two shapes
    that matter: a mutable container (in-place restore) and a plain scalar (setattr).
    """
    _env, agent = _mk_env_agent()
    agent._self_experience_buffer.append(torch.zeros(1, 4))
    agent.e3._running_variance = 0.125
    n_before = len(agent._self_experience_buffer)

    surface = capture_agent_surface(agent)
    agent._self_experience_buffer.extend([torch.ones(1, 4)] * 7)
    agent.e3._running_variance = 99.0
    agent.e3._novelty_ema = 42.0

    report = restore_agent_surface(agent, surface)

    assert len(agent._self_experience_buffer) == n_before
    assert agent.e3._running_variance == 0.125
    assert agent.e3._novelty_ema != 42.0
    assert report["n_restored_setattr"] > 0


def test_restore_keeps_container_identity():
    """In-place restore, so an external holder of a buffer is not left holding the
    DRIFTED list while the agent points at a clean one -- a divergence with no error."""
    env, agent = _mk_env_agent()
    held = agent._self_experience_buffer
    n_before = len(held)

    _read(agent, env)

    assert agent._self_experience_buffer is held, "buffer identity was replaced"
    assert len(held) == n_before


def test_repeated_reads_do_not_nest_the_candidate_capture():
    """Hazard H3's monkeypatch is an ADDED attribute; the restore must delete it.

    Without the delete each read wraps the previous read's wrapper, so
    `generate_trajectories` gains a stack frame per read forever -- and the agent
    handed back is not the one passed in.
    """
    env, agent = _mk_env_agent()
    assert "generate_trajectories" not in vars(agent)

    for i in range(2):
        _read(agent, env, seed=5 + i)
        assert "generate_trajectories" not in vars(agent), (
            "read %d left its candidate-capture wrapper installed" % i
        )


def test_caller_supplied_candidate_capture_survives_the_read():
    """The mirror image: a patch the CALLER installed was there before the read and
    must still be there after it. V3-EXQ-777a installs it once per arm-agent."""
    env, agent = _mk_env_agent()
    captured = reapply_candidate_capture(agent)
    patched = agent.generate_trajectories

    _read(agent, env, captured=captured)

    assert agent.generate_trajectories is patched


# --------------------------------------------------------------------------- #
# why the copier is not copy.deepcopy                                          #
# --------------------------------------------------------------------------- #

def test_plain_deepcopy_fails_on_the_trajectory_shaped_attributes():
    """Pins WHY `_copy_state` exists, so nobody "simplifies" it to copy.deepcopy.

    `Trajectory.metadata` is an arbitrary dict and can hold a module object, which
    pickle -- and therefore deepcopy -- refuses. The subtle half is worse than the
    failure: `copy._reconstruct` registers the half-built, EMPTY object in the memo
    BEFORE deep-copying its state, so a fallback that consults that same memo gets
    an empty Trajectory back and the restore silently WIPES the attribute. That was
    observed as 4 attributes reading as "restored" while holding empty objects.
    """
    from ree_core.predictors.e3_selector import Trajectory

    traj = Trajectory(states=[torch.zeros(1, 4)], actions=torch.zeros(1, 1, 4))
    traj.metadata = {"source": "contract", "mod": types}

    with pytest.raises(Exception):
        copy.deepcopy(traj)

    memo = {}
    by_reference = []
    out = probe_warmup._copy_state(traj, memo, by_reference, "e3._test_trajectory")
    assert by_reference == [], "the module leaf must be carried by reference, not flagged"
    assert out is not traj
    assert out.metadata["mod"] is types, "a module leaf must NOT be copied"
    assert out.states[0] is not traj.states[0], "tensors must be cloned"
    assert torch.equal(out.states[0], traj.states[0])


def test_warm_start_blob_carries_the_surface_through_a_real_save_load():
    """Hazard H1 on the CACHE side: a HIT agent must not be younger than a MISS agent.

    The v1 blob carried four hand-listed E3 scalars, so a cache HIT restored an agent
    whose experience buffers, accumulators and episode state were all at
    fresh-construction values while its weights were warmed. No executed run ever
    took a HIT (the audit established that), which is exactly why this went unnoticed
    -- it is pinned here rather than left to the first consumer to enable the cache.

    Goes through a real torch.save/load rather than passing the dict along in memory,
    because the whole difficulty is that the surface contains values pickle refuses.
    """
    import io

    _env, agent = _mk_env_agent()
    agent._self_experience_buffer.extend([torch.ones(1, 4) * 3])
    agent._step_count = 77
    agent.e3._running_variance = 0.0625
    agent.e3._novelty_ema = 0.5

    cached = probe_warmup._capture_cached_surface(agent, lambda _m: None)

    # The unpicklable leaves must be NAMED and dropped, never allowed to take the
    # whole cache store down. hippocampal._rng is the stable ree_core-owned case (it
    # is the stdlib `random` module itself); the others are torch-version dependent.
    assert "hippocampal._rng" in cached["dropped"]
    assert len(cached["dropped"]) < 10, (
        "the cache is now dropping %d attributes: %s -- a HIT agent is missing that "
        "much state" % (len(cached["dropped"]), cached["dropped"])
    )

    buf = io.BytesIO()
    torch.save(cached, buf)
    buf.seek(0)
    reloaded = torch.load(buf, map_location="cpu", weights_only=False)

    _env2, fresh = _mk_env_agent()
    assert fresh._step_count != 77
    probe_warmup._restore_cached_surface(fresh, reloaded, lambda _m: None)

    assert fresh._step_count == 77
    assert fresh.e3._running_variance == 0.0625
    assert fresh.e3._novelty_ema == 0.5
    assert len(fresh._self_experience_buffer) == len(agent._self_experience_buffer)


def test_live_submodules_are_carried_by_reference_not_cloned():
    """An attribute holding a live submodule must keep pointing at the LIVE module.

    Cloning it would hand back an agent whose attribute references a detached copy
    with its own parameter tensors -- no optimiser owns them, no state_dict covers
    them, and nothing raises.
    """
    _env, agent = _mk_env_agent()
    memo = {id(m): m for _p, m in agent.named_modules()}
    out = probe_warmup._copy_state({"e3": agent.e3}, memo, [], "probe")
    assert out["e3"] is agent.e3
