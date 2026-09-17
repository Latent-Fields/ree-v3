"""Contract tests for MECH-042 sub-claim (1): telemetry READ-ONLY / NON-PARTICIPATION.

MECH-042 ("Telemetry exposure channels report internal control-plane state for
diagnostics.") has two CONFIRMING sub-claims, both required. Its own disposition
(claims.yaml, 2026-09-16) routes them differently:

    "Sub-claim (1) alone is a contract-test-shaped check and could land as a
     ree-v3 contract test rather than an EXQ; sub-claim (2) is the experiment."

THIS FILE IS SUB-CLAIM (1). Sub-claim (2) -- the developmental-safety lead-time
falsifier on an injected control-plane pathology -- is V3-EXQ-1053 and is NOT
covered here.

Sub-claim (1), verbatim from claims.yaml `what_would_answer`:

    READ-ONLY / NON-PARTICIPATION -- at a fixed seed, running with every
    telemetry channel read each tick (AgentState via get_state,
    e3.last_score_diagnostics, get_residue_statistics / get_coverage_telemetry,
    CommitReadiness.get_state, per_stream_vs) versus never reading any of them
    yields a BIT-IDENTICAL action, commit and weight trajectory (hash the
    surfaces every k ticks).

FALSIFYING (1): enabling or reading a diagnostic CHANGES selection -- the
channel is not read-only as implemented and the claim's safety premise (no new
decision pathway) is violated.

The claim adds one specific thing to check, and C5 is exactly it:

    Note agent.py already reads e3.last_score_diagnostics into
    _last_control_vector (the modulatory-authority context); sub-claim (1) must
    confirm that read is telemetry-only and not consumed by selection.

WHY A LATCH CLEAR IS *NOT* USED HERE. CLAUDE.md / queue-experiment Step 3.5
require a driver that reads `e3.last_*` in a per-env-step loop to clear the
latch first, because a latched re-read inflates the SAMPLE SIZE of a
statistical claim. This file makes no statistical claim and counts no
observations: the READER arm is deliberately the worst case the claim must
survive -- an unconditional every-tick read of the latched surface, which is
what a naive diagnostic consumer would actually do. Clearing the latch would
weaken the test rather than strengthen it. Nothing here is a denominator.

    C1  READER vs NON-READER emit a bit-identical ACTION stream.
    C2  READER vs NON-READER emit a bit-identical WEIGHT trajectory
        (sha256 over state_dict tensor bytes, every k ticks).
    C3  READER vs NON-READER emit a bit-identical COMMIT trajectory.
    C4  NON-DEGENERACY: the telemetry surfaces are NON-CONSTANT over the probe
        window, and every named channel is actually present. A constant or
        absent channel can detect nothing, so C1-C3 would pass vacuously.
    C5  The _last_control_vector read of e3.last_score_diagnostics is
        telemetry-only: toggling control-vector logging (which performs that
        read) does not move the action stream.
"""

import hashlib
import json
import random

import numpy as np
import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments._harness import StepHarness

# Hash the weight surface every k ticks, per the claim's "hash the surfaces
# every k ticks". k=1 here: the window is short and a per-tick hash is the
# strictest form of the same check.
HASH_EVERY_K_TICKS = 1
PROBE_STEPS = 24
SEED = 11


def _mk_env(seed=None):
    # A seed MUST be threaded through for any test asserting bit-identity:
    # CausalGridWorldV2 builds its RNG as np.random.default_rng(seed), so
    # seed=None draws a non-deterministic map and desynchronises the arms.
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _cfg(env, **overrides):
    """Config with every MECH-042 telemetry channel ENABLED.

    commit_readiness and the salience coordinator both default OFF, so without
    these two flags `agent.commit_readiness` is None and `agent.salience` is
    absent -- the reader arm would then silently read a two-channel surface and
    C1 would pass for the wrong reason. C4 asserts the channels are really here.
    """
    base = dict(
        use_commit_readiness=True,
        use_salience_coordinator=True,
        use_dacc=True,
    )
    base.update(overrides)
    return REEConfig.from_dims(**base, **_dims(env))


def _weight_hash(agent):
    """Stable sha256 over the agent's parameter bytes, sorted by key."""
    h = hashlib.sha256()
    for key, value in sorted(agent.state_dict().items()):
        h.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _jsonable(value):
    """Coerce a telemetry value to something json.dumps can canonicalise."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.flatten().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def _read_all_telemetry(agent):
    """Read EVERY MECH-042 channel, exactly as a diagnostic consumer would.

    This is the READER arm's per-tick action. It must not perturb anything.
    Returns a canonical-JSON-able snapshot dict so C4 can test non-constancy.
    """
    state = agent.get_state()
    snap = {
        "agent_state": {
            "precision": state.precision,
            "running_variance": state.running_variance,
            "step": state.step,
            "harm_accumulated": state.harm_accumulated,
            "is_committed": state.is_committed,
            "beta_elevated": state.beta_elevated,
            "e3_steps_per_tick": state.e3_steps_per_tick,
        },
        # Latched on non-E3 ticks BY DESIGN -- see module docstring.
        "e3_last_score_diagnostics": _jsonable(
            dict(getattr(agent.e3, "last_score_diagnostics", {}) or {})
        ),
        "residue_statistics": _jsonable(agent.get_residue_statistics()),
        "residue_coverage_telemetry": _jsonable(
            agent.residue_field.get_coverage_telemetry()
        ),
    }
    cr = getattr(agent, "commit_readiness", None)
    snap["commit_readiness"] = _jsonable(cr.get_state()) if cr is not None else None
    sal = getattr(agent, "salience", None)
    snap["salience_operating_mode"] = (
        _jsonable(dict(sal.operating_mode)) if sal is not None else None
    )
    return snap


def _run(cfg, *, read_telemetry, seed=SEED, steps=PROBE_STEPS):
    """Step one life, optionally reading every telemetry channel each tick.

    Returns (actions, weight_hashes, commits, snapshots).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = _mk_env(seed=seed)
    agent = REEAgent(cfg)

    weight_hashes = []
    commits = []
    snapshots = []
    tick = {"n": 0}

    def _on_step(_result):
        tick["n"] += 1
        if read_telemetry:
            snapshots.append(_read_all_telemetry(agent))
            commits.append(bool(agent.get_state().is_committed))
        else:
            # The NON-READER arm must not call get_state() either -- that call
            # is itself one of the channels under test. Read the underlying
            # attribute the dataclass would have copied.
            commits.append(
                agent.e3._committed_trajectory is not None
                or agent.e3._closure_committed_trajectory is not None
            )
        if tick["n"] % HASH_EVERY_K_TICKS == 0:
            weight_hashes.append(_weight_hash(agent))

    results = StepHarness(agent, env, train_mode=True, seed=seed).run_episode(
        max_steps=steps, on_step=_on_step
    )
    actions = [int(r.action.argmax().item()) for r in results]
    return actions, weight_hashes, commits, snapshots


@pytest.fixture(scope="module")
def arms():
    """Run the READER and NON-READER arms once, at a matched seed."""
    env = _mk_env(seed=SEED)
    cfg_reader = _cfg(env)
    cfg_quiet = _cfg(env)
    reader = _run(cfg_reader, read_telemetry=True)
    quiet = _run(cfg_quiet, read_telemetry=False)
    return reader, quiet


def test_c1_action_trajectory_bit_identical(arms):
    (a_actions, _, _, _), (q_actions, _, _, _) = arms
    assert a_actions, "probe produced no steps -- C1 would pass vacuously"
    assert a_actions == q_actions, (
        "MECH-042 sub-claim (1) FALSIFIED: reading telemetry changed the action "
        f"stream (reader={a_actions} non_reader={q_actions})"
    )


def test_c2_weight_trajectory_bit_identical(arms):
    (_, a_hashes, _, _), (_, q_hashes, _, _) = arms
    assert a_hashes, "no weight hashes captured -- C2 would pass vacuously"
    assert a_hashes == q_hashes, (
        "MECH-042 sub-claim (1) FALSIFIED: reading telemetry changed the weight "
        "trajectory (first divergence at hash index "
        f"{next((i for i, (x, y) in enumerate(zip(a_hashes, q_hashes)) if x != y), 'n/a')})"
    )


def test_c3_commit_trajectory_bit_identical(arms):
    (_, _, a_commits, _), (_, _, q_commits, _) = arms
    assert a_commits, "no commit samples captured -- C3 would pass vacuously"
    assert a_commits == q_commits, (
        "MECH-042 sub-claim (1) FALSIFIED: reading telemetry changed the commit "
        f"trajectory (reader={a_commits} non_reader={q_commits})"
    )


def test_c4_non_degeneracy_channels_present_and_non_constant(arms):
    """NON-DEGENERACY. C1-C3 compare two streams; if the telemetry surface were
    absent or frozen, they would agree for reasons that have nothing to do with
    the claim. This test is what stops that reading."""
    (_, _, _, snapshots), _ = arms
    assert len(snapshots) >= 2, "need >=2 snapshots to test non-constancy"

    # (a) every named channel is actually PRESENT (not None, not empty).
    first = snapshots[0]
    for channel in (
        "agent_state",
        "e3_last_score_diagnostics",
        "residue_statistics",
        "residue_coverage_telemetry",
        "commit_readiness",
        "salience_operating_mode",
    ):
        assert channel in first, f"telemetry channel absent: {channel}"
        assert first[channel] is not None, (
            f"telemetry channel {channel} is None -- it is not enabled in this "
            "config, so C1-C3 would pass vacuously for it"
        )
    assert first["e3_last_score_diagnostics"], (
        "e3.last_score_diagnostics is empty -- E3 never scored in the window"
    )

    # (b) the surface as a whole VARIES across the window.
    canon = [
        json.dumps(s, sort_keys=True, separators=(",", ":")) for s in snapshots
    ]
    assert len(set(canon)) > 1, (
        "telemetry surface is CONSTANT across the probe window -- a constant "
        "channel can detect nothing and C1-C3 are vacuous"
    )

    # (c) at least one scalar AgentState field moves, so the variation in (b)
    #     is not carried solely by the monotonic `step` counter.
    moving = [
        key
        for key in ("precision", "running_variance", "harm_accumulated",
                    "is_committed", "beta_elevated")
        if len({s["agent_state"][key] for s in snapshots}) > 1
    ]
    assert moving, (
        "no AgentState field except `step` varied over the window -- the "
        "control-plane surface is effectively frozen here"
    )


def test_c5_control_vector_read_of_e3_diagnostics_is_telemetry_only():
    """The claim names this specific risk: agent.py already reads
    e3.last_score_diagnostics into _last_control_vector. Confirm that read is
    telemetry-only and not consumed by selection."""
    env = _mk_env(seed=SEED)
    cfg_off = _cfg(env)
    cfg_on = _cfg(env, use_control_vector_logging=True)

    actions_off, _, _, _ = _run(cfg_off, read_telemetry=False)
    actions_on, _, _, _ = _run(cfg_on, read_telemetry=False)

    assert actions_off == actions_on, (
        "MECH-042 sub-claim (1) FALSIFIED: enabling control-vector logging "
        "(which reads e3.last_score_diagnostics) changed the action stream "
        f"(off={actions_off} on={actions_on})"
    )
