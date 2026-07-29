#!/opt/local/bin/python3
"""V3-EXQ-838 -- Q-081 prospective cross-stream recording run: does REE's
configured multi-rate execution (SD-006: E1/E2/E3 at 1/3/10) produce SHARED
cross-stream organisation (Outcome A), or only the configured rate separation
and wired gates it was built with (Outcome B)?

This is the PROSPECTIVE RECORDING RUN named as NOT STARTED in the two Q-081
design records (q081_surrogate_null_design.md sec.8, q081_landmark_removal_arm_
design.md sec.7). All three landed preconditions are used together here:
  * per-step multi-stream trace recorder     -- experiments/_lib/stream_recorder.py
  * structure-destroying landmark-removal arm -- experiments/_lib/q081_landmark_removal.py
  * constrained-realisation surrogate null    -- experiments/_lib/q081_surrogate.py

WHY A NEW NUMBER (838), AND WHAT IT ADDS OVER 824/824a
------------------------------------------------------
824 (FAIL, superseded) hit the inert-arm defect: use_invalidation_trigger alone
gave the landmark manipulation no reach. 824a added use_anchor_sets=True and the
reach-check then reported MET -- but rv_primary was STILL bit-identical between
the ON and LANDMARK_REMOVED arms at all 5 seeds, so 824a self-routed
substrate_not_ready_requeue. The 2026-07-28 failure autopsy
(failure_autopsy_2026-07-28-sweep.json, target V3-EXQ-824a; category
measurement_test_design_defect / non_contributory) diagnosed the cause exactly:
use_anchor_sets grants reach to z_world (MECH-269 write_anchor) but NOT to
operating_mode (the salience coordinator's mode read, driven via
apply_invalidation_broadcasts_to_regions -> vs_rollout_gate -> E3, which is gated
on use_per_region_vs). RV(z_world, operating_mode) therefore could not move under
the manipulation. The autopsy's PRIMARY recommended fix is a config change: "add
use_per_region_vs=True and re-verify empirically first" (before building the
heavier pair-specific reach assertion, sd_id_suggested Q081-REACH-CHECK-PAIR-
SPECIFIC, which remains follow-on library work).

838 is that empirical re-verification, expanded to the full control design the
Q-081 guard specifies:
  * THE FIX: make_agent() enables BOTH use_anchor_sets=True (z_world reach) AND
    use_per_region_vs=True (operating_mode reach). This is strictly more reach
    than 824a and directly addresses the confirmed 824a wall.
  * FOUR ARMS instead of two (see below): the two extra arms give the
    conservative-secondary dissociation and the lesion reference the guard names.

The arm_statistics_not_degenerately_bit_identical precondition (carried from
824a) remains the guard: if RV STILL does not move even with use_per_region_vs on,
838 self-routes substrate_not_ready_requeue cleanly (informative, not a false
verdict) and confirms the pair-specific reach check is genuinely required.

ARMS (per seed; shared P0 substrate deep-copied into each)
----------------------------------------------------------
  1. INTACT           (mode="off")            -- runs FIRST; intact landmark
                       structure; RECORDS the donor boundary train per
                       (seed, episode) for the yoked arms.
  2. IEI_PERMUTE      (mode="iei_permute")    -- PRE-REGISTERED PRIMARY. Yoked,
                       seed/episode-matched joint permutation of INTACT's own
                       boundary train. Count / IEI multiset / posterior multiset /
                       scale mix preserved EXACTLY (Level 1); env input statistics
                       MEASURED (Level 2). This is the A-vs-B discriminator arm.
  3. CIRCULAR_SHIFT   (mode="circular_shift") -- CONSERVATIVE SECONDARY. Rigid
                       circular shift of the donor train: destroys landmark-to-
                       stream ALIGNMENT while preserving the landmark train's OWN
                       internal structure (autocorrelation, burstiness, interval
                       sequence) up to one wrap. Dissociates loss-of-alignment
                       from loss-of-landmark-train-structure. A statistic that
                       dies under iei_permute but SURVIVES circular_shift was
                       reading landmark-train structure, not cross-stream
                       alignment. DIAGNOSTIC, not the primary discriminator.
  4. SUPPRESS         (mode="suppress")       -- LESION REFERENCE ONLY. Removes
                       the invalidation DRIVE as well as the alignment
                       (broadcast count -> 0), so it confounds "misaligned
                       landmarks" with "fewer landmarks". is_lesion=True. Reported
                       for context; NEVER the discriminator. No donor required.

PRIMARY STATISTIC -- RV coefficient (Robert & Escoufier 1976) between the FULL
z_world vector (E1-rate, fastest recorded stream) and the FULL operating_mode
vector (salience coordinator soft mode-probabilities + current-mode index +
switch trigger + salience aggregate; E3-rate). Both read on the coarser (E3)
stream's own tick grid per q081_surrogate's alignment rule. Deliberately NOT the
1-D reduction: reduce_stream's own docstring says a 1-D reduction "discards
exactly [the multi-dimensional configuration] Q-081 asks about" and is for
controls/diagnostics only. cross_stream_xcorr (1-D, reduce="norm") + its lag are
reported as SECONDARY/DIAGNOSTIC on the same pair.

MANDATORY GATES (all library-implemented; called, never reimplemented)
---------------------------------------------------------------------
  * assert_behavioural_reach(agent, strict=True) BEFORE any recording -- vacuous
    unless a REACH_CONSUMERS flag (use_anchor_sets / use_per_region_vs) is live
    alongside use_event_segmenter. Both are enabled here; strict raises otherwise.
  * screen_statistic() on EVERY candidate statistic (the RV primary AND the
    deliberately-artefactual rate statistic) before it adjudicates -- Q-081 rules
    out a priori any statistic that is a function of the configured update rates.
  * scrambler.preservation_report() per arm -> manifest; preserved_by_construction
    must be True for the yoked arms (iei_permute, circular_shift).
  * input_statistics_divergence(intact, arm) per scrambled arm -> manifest. On
    input_statistics_preserved=False the arm is CONFOUNDED and that seed is
    excluded (not reported as a null); if that leaves too few valid seeds the run
    self-routes substrate_not_ready_requeue.
  * lag_control_report() -- cross-stream LAG reported as a CONTROL quantity
    (role="control"), never the primary readout (a 1/3/10 lag is scheduler-
    guaranteed = Outcome B).

INTERPRETATION RULES (carried verbatim into the manifest + queue note)
---------------------------------------------------------------------
  * Clearing the surrogate null is NECESSARY and NOWHERE NEAR SUFFICIENT for
    Outcome A -- wired coordination clears it correctly.
  * A statistic that SURVIVES the landmark-removal arm was measuring the clock --
    Outcome B evidence, NOT support for the claim.
  * Neither control, nor both together, is the ABLATION SERIES, which remains the
    only full A-vs-B discriminator.

DV-SYMMETRY DECLARATION (mandatory, per arm). DV = RV(z_world, operating_mode), a
function of the REALISED joint trajectory of two streams across a run. For all
three non-intact arms the manipulation is a change to boundary-event TIMING
(iei_permute: yoked permutation; circular_shift: rigid shift; suppress: removal),
which alters WHEN write_anchor / broadcast events fire, hence anchor content, V_s,
vs_rollout_gate readings, E3 candidate scoring, the action selected at every
committed tick, and therefore the entire downstream trajectory of z_world (env
transitions) and operating_mode (salience read of the same downstream state). It
is NOT a broadcast additive constant on a fixed candidate set (RV centres each
block, so a constant is invisible either way), NOT a monotone rescaling (RV is
invariant to orthogonal per-block transforms; rescaling is not the manipulation),
and NOT a permutation of interchangeable aggregation units (seeds/candidates): it
is a permutation/removal of EVENT TIMING that materially alters the physical
trajectory each stream is sampled from. A null ON-vs-arm result is therefore a
MEASURED finding (Outcome B), not an arithmetic identity. (suppress additionally
removes the drive, so it is scoped as a lesion reference, never the discriminator.)

VALIDATE-THE-NULL-BEFORE-USE. (1) "spares an injected real relation" proven
generically at the library level 2026-07-22 by test_q081_surrogate_null.py::
test_null_spares_an_injected_real_relation (22-test suite). (2) "kills a
deliberately-artefactual statistic" re-checked HERE per run on THIS run's actual
tick grids via screen_statistic on artefactual_rate_statistic -- expected
"ruled_out".

No SLEEP DRIVER: use_sleep_loop deliberately left False (scope: waking-stream
organisation only). Sleep-phase-relative structure (Outcome E) is deferred to a
lettered follow-up once this primary A-vs-B result is banked.

GOV-REUSE-1: decisive readout is RV(z_world, operating_mode) INTACT-minus-
IEI_PERMUTE delta under a substrate with use_per_region_vs=True. No recorded run
carries this -- 824/824a ran with use_per_region_vs=False and produced bit-
identical arms. Not recoverable by reanalysis -> run.

Provenance: env/agent/P0/P1 construction and RV/analysis helpers adapted from
experiments/v3_exq_824a_q081_shared_organisation_landmark_removal.py (the 2-arm
predecessor); per-tick mechanics via experiments/_harness.StepHarness, which
discharges q081_profile's LOOP_DRIVEN_REQUIREMENTS (harm sense kwargs + z_goal).
ASCII-only output.
"""
from __future__ import annotations

import copy
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.stream_recorder import StreamTraceRecorder  # noqa: E402
from experiments._lib.q081_profile import (  # noqa: E402
    q081_profile_kwargs, q081_substrate_declaration,
)
from experiments._lib.q081_landmark_removal import (  # noqa: E402
    LandmarkRemovalConfig, LandmarkScrambler, BoundaryTrain,
    assert_behavioural_reach, input_statistics_divergence, PRIMARY_MODE,
)
from experiments._lib.q081_surrogate import (  # noqa: E402
    plan_blocks, screen_statistic, evaluate_against_null,
    lag_control_report, artefactual_rate_statistic, estimate_tick_period,
    SurrogateDesignError,
)
from _harness import StepHarness  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_838_q081_cross_stream_recording"
QUEUE_ID = "V3-EXQ-838"
CLAIM_IDS: List[str] = ["Q-081"]
EXPERIMENT_PURPOSE = "evidence"

# ------------------------------------------------------------------ config
WORLD_DIM = SELF_DIM = 32
ALPHA_WORLD = 0.9   # SD-008: z_world-fidelity-dependent experiments need >= 0.9
ENV_SIZE = 6
SEEDS = [0, 1, 2, 3, 4]

WARMUP_EPISODES = 20
REC_EPISODES = 20
STEPS_PER_EPISODE = 200
WARMUP_LR = 1e-3

# Arm labels. INTACT runs first and banks the donor trains the yoked arms replay.
INTACT_ARM = "INTACT"
IEI_ARM = "IEI_PERMUTE"        # PRE-REGISTERED PRIMARY discriminator arm
CIRC_ARM = "CIRCULAR_SHIFT"    # conservative secondary (dissociation)
SUPP_ARM = "SUPPRESS"          # lesion reference ONLY, never the discriminator

# (arm_label, LandmarkRemovalConfig.mode). Order matters: INTACT first.
ARM_MODES: List[Tuple[str, str]] = [
    (INTACT_ARM, "off"),
    (IEI_ARM, PRIMARY_MODE),      # "iei_permute"
    (CIRC_ARM, "circular_shift"),
    (SUPP_ARM, "suppress"),
]
# Arms whose Level-1 preservation must hold (donor-replay yoked arms). suppress is
# a lesion (preserved_by_construction=False by design) and INTACT is the reference.
PRESERVED_ARMS = (IEI_ARM, CIRC_ARM)
# Arms compared against INTACT for Level-2 input-statistics divergence.
SCRAMBLED_ARMS = (IEI_ARM, CIRC_ARM, SUPP_ARM)

PRIMARY_PAIR = ("z_world", "operating_mode")
SECONDARY_REDUCE = "norm"

# ---------------------------------------------------------- pre-registered
SCREEN_N = 64
SURR_N = 199
ADMISSIBLE_SPREAD_FLOOR = 1e-9     # RV stat: ensemble must show non-zero spread
ARTEFACT_SPREAD_CEIL = 1e-9        # artefactual stat: ensemble spread must be ~0

PRESERVATION_FRACTION_FLOOR = 0.6
INPUT_STATS_FRACTION_FLOOR = 0.5

# Effect-size PASS gate (project convention: scale noise on the SD of the DELTA
# between arms, plus an absolute floor -- never the SD of either arm alone).
EFFECT_SIZE_K = 1.5
EFFECT_SIZE_ABS_FLOOR = 0.03


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ============================================================================
# Substrate construction
# ============================================================================

def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=ENV_SIZE, seed=seed)


def make_agent(env: CausalGridWorldV2) -> REEAgent:
    flags = q081_profile_kwargs()
    flags["use_sleep_loop"] = False  # scope: waking streams only this iteration
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        world_dim=WORLD_DIM,
        self_dim=SELF_DIM,
        **flags,
    )
    # THE 838 FIX (2026-07-28 failure autopsy of V3-EXQ-824a). 824a set
    # use_anchor_sets=True (z_world reach via MECH-269 write_anchor) but left
    # use_per_region_vs=False, so the landmark manipulation never reached
    # operating_mode (apply_invalidation_broadcasts_to_regions -> vs_rollout_gate
    # -> E3 -> operating_mode is gated on use_per_region_vs; agent.py:4686-4692) and
    # RV(z_world, operating_mode) stayed bit-identical across all 5 seeds. Enable
    # BOTH reach paths -- the autopsy's own "add use_per_region_vs=True and
    # re-verify empirically first" recommendation. Both are REACH_CONSUMERS, so
    # assert_behavioural_reach(strict=True) passes and the manipulation can now
    # reach both members of the measured pair.
    cfg.hippocampal.use_anchor_sets = True
    cfg.hippocampal.use_per_region_vs = True
    return REEAgent(cfg)


# ============================================================================
# P0 -- shared substrate warmup (E1 + E2-self only; modelled on EXQ-824a run_p0).
# ============================================================================

def run_p0(agent: REEAgent, env: CausalGridWorldV2, seed: int) -> Dict[str, Any]:
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=WARMUP_LR)
    harness = StepHarness(agent, env, train_mode=True)
    harm_events = 0
    total_harm = 0.0

    for ep in range(WARMUP_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.harm_signal < 0:
                harm_events += 1
                total_harm += abs(result.harm_signal)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            loss = e1_loss + e2_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if result.done:
                break
        # deliberately "step", not "ep": P1's per-arm loop owns the canonical
        # "ep N/M" pattern (episodes_per_run == REC_EPISODES).
        if (ep + 1) % 5 == 0 or ep == WARMUP_EPISODES - 1:
            print(f"  [warmup] seed={seed} step {ep + 1}/{WARMUP_EPISODES} "
                  f"harm={total_harm:.2f}", flush=True)

    return {"warmup_harm_events": harm_events, "warmup_total_harm": total_harm}


# ============================================================================
# P1 -- recording rollout for one (seed, arm) cell.
# ============================================================================

def run_p1_recording(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    arm_label: str,
    scrambler: LandmarkScrambler,
    rec: StreamTraceRecorder,
    donor_lookup: Optional[Dict[Tuple[int, int], BoundaryTrain]] = None,
) -> Dict[str, Any]:
    agent.eval()
    harness = StepHarness(agent, env, train_mode=False)
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    state_visitation: Dict[str, int] = {}
    action_counts: Dict[int, int] = {}
    harm_events = 0
    reward_events = 0
    episode_lengths: List[int] = []
    obs_samples: List[np.ndarray] = []
    n_steps = 0

    for ep in range(REC_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        donor = donor_lookup.get((seed, ep)) if donor_lookup is not None else None
        scrambler.begin_episode(ep, n_steps=STEPS_PER_EPISODE, seed=seed, donor=donor)

        ep_len = 0
        for _ in range(STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            rec.on_step(extras={"reward": result.harm_signal})

            sx = int(round(float(getattr(env, "agent_x", 0.0))))
            sy = int(round(float(getattr(env, "agent_y", 0.0))))
            key = f"{sx}_{sy}"
            state_visitation[key] = state_visitation.get(key, 0) + 1
            a_idx = int(result.action.argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            if result.harm_signal < 0:
                harm_events += 1
            elif result.harm_signal > 0:
                reward_events += 1
            obs_vec = np.concatenate([
                np.asarray(obs_dict["body_state"], dtype=np.float64).ravel(),
                np.asarray(obs_dict["world_state"], dtype=np.float64).ravel(),
            ])
            obs_samples.append(obs_vec)

            obs_dict = result.next_obs_dict
            ep_len += 1
            n_steps += 1
            if result.done:
                break

        scrambler.end_episode()
        rec.on_episode_end()
        episode_lengths.append(ep_len)
        if (ep + 1) % 4 == 0 or ep == REC_EPISODES - 1:
            print(f"  [{arm_label}] seed={seed} ep {ep + 1}/{REC_EPISODES}", flush=True)

    obs_arr = np.stack(obs_samples) if obs_samples else np.zeros((0, 1))
    verdict = "PASS" if n_steps > 0 else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "state_visitation": state_visitation,
        "action_counts": {str(k): v for k, v in action_counts.items()},
        "harm_events": int(harm_events),
        "reward_events": int(reward_events),
        "n_steps": int(n_steps),
        "episode_lengths": episode_lengths,
        "obs_channel_mean": obs_arr.mean(axis=0).tolist() if obs_arr.size else [],
        "obs_channel_std": obs_arr.std(axis=0).tolist() if obs_arr.size else [],
    }


# ============================================================================
# Cross-stream statistics
# ============================================================================

def rv_coefficient(arrays: Dict[str, np.ndarray], name_a: str, name_b: str) -> float:
    """RV coefficient (Robert & Escoufier 1976) between two FULL-width vector
    streams, read on the coarser stream's own tick grid. Deliberately
    multivariate (a 1-D reduction is a control/diagnostic, not the Q-081 primary).
    """
    fa = np.asarray(arrays[f"{name_a}__fresh"], dtype=bool)
    fb = np.asarray(arrays[f"{name_b}__fresh"], dtype=bool)
    pa = estimate_tick_period(fa)
    pb = estimate_tick_period(fb)
    coarse_fresh = fb if pb >= pa else fa
    grid = np.flatnonzero(coarse_fresh)
    if grid.size < 8:
        return float("nan")

    X = np.atleast_2d(np.asarray(arrays[name_a], dtype=np.float64))
    Y = np.atleast_2d(np.asarray(arrays[name_b], dtype=np.float64))
    Xg, Yg = X[grid], Y[grid]
    Xc = Xg - Xg.mean(axis=0, keepdims=True)
    Yc = Yg - Yg.mean(axis=0, keepdims=True)
    Cxy = Xc.T @ Yc
    Cxx = Xc.T @ Xc
    Cyy = Yc.T @ Yc
    num = float(np.trace(Cxy @ Cxy.T))
    den = float(np.sqrt(np.trace(Cxx @ Cxx) * np.trace(Cyy @ Cyy)))
    return num / den if den > 1e-30 else float("nan")


def analyze_arm_trace(arrays: Dict[str, np.ndarray], seed: int, arm_label: str) -> Dict[str, Any]:
    """Primary RV statistic + its surrogate screen/p-value, plus the
    secondary/diagnostic lag control, for one (seed, arm) trace."""
    name_a, name_b = PRIMARY_PAIR
    out: Dict[str, Any] = {"seed": seed, "arm": arm_label}

    try:
        plan = plan_blocks(arrays, [name_a, name_b])
    except SurrogateDesignError as exc:
        out["surrogate_design_error"] = str(exc)
        out["rv_primary"] = float("nan")
        out["admissible"] = False
        out["artefact_ruled_out"] = False
        return out

    def _rv_stat(a: Dict[str, np.ndarray]) -> float:
        return rv_coefficient(a, name_a, name_b)

    screen = screen_statistic(arrays, [name_a, name_b], _rv_stat,
                              n_surrogates=SCREEN_N, seed=seed, plan=plan)
    artefact_screen = screen_statistic(
        arrays, [name_a, name_b],
        lambda a: artefactual_rate_statistic(a, name_a, name_b),
        n_surrogates=SCREEN_N, seed=seed, plan=plan,
    )
    null_eval = evaluate_against_null(arrays, [name_a, name_b], _rv_stat,
                                      n_surrogates=SURR_N, seed=seed, plan=plan)
    lag = lag_control_report(arrays, name_a, name_b, reduce=SECONDARY_REDUCE)

    out.update({
        "rv_primary": float(null_eval["observed"]),
        "rv_p_value_vs_surrogate": float(null_eval["p_value"]),
        "rv_surrogate_mean": float(null_eval["surrogate_mean"]),
        "rv_surrogate_std": float(null_eval["surrogate_std"]),
        "admissible": bool(screen["verdict"] == "admissible"
                           and screen["ensemble_spread"] > ADMISSIBLE_SPREAD_FLOOR),
        "screen_ensemble_spread": float(screen["ensemble_spread"]),
        "artefact_ruled_out": bool(artefact_screen["verdict"] == "ruled_out"
                                   and artefact_screen["ensemble_spread"] <= ARTEFACT_SPREAD_CEIL),
        "artefact_ensemble_spread": float(artefact_screen["ensemble_spread"]),
        "lag_control": lag,
        "block_plan": plan.as_dict(),
    })
    return out


# ============================================================================
# Per-(seed) cell -- runs all four arms.
# ============================================================================

def _run_arm_cell(
    arm_label: str,
    mode: str,
    seed: int,
    agent_p0: REEAgent,
    common_config_slice: Dict[str, Any],
    donor_index: Optional[Dict[Tuple[int, int], BoundaryTrain]],
    zg_accum: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Run one arm to completion; return stats + preservation + trace pointer.

    INTACT (mode="off") passes donor_index=None and its own donor bank is read by
    the caller afterwards. Yoked arms (iei_permute, circular_shift) pass the INTACT
    donor_index. suppress needs no donor.
    """
    ineligible = ["p0_substrate_trained_outside_fingerprinted_cell_scope"]
    if donor_index is not None:
        ineligible.append("yoked_to_INTACT_arm_donor_train_same_seed")
    with arm_cell(
        seed, config_slice={**common_config_slice, "arm": arm_label},
        script_path=Path(__file__), config_slice_declared=True,
        extra_ineligible_reasons=ineligible,
    ) as cell:
        agent = copy.deepcopy(agent_p0)
        env = make_env(seed)  # fresh env, byte-identical starting RNG per arm
        scr = LandmarkScrambler(LandmarkRemovalConfig(mode=mode, seed=seed))
        scr.attach(agent)
        rec = StreamTraceRecorder(
            agent, run_id=f"{EXPERIMENT_TYPE}_seed{seed}_{arm_label}",
            substrate_declaration=q081_substrate_declaration(agent.config),
        )
        stats = run_p1_recording(agent, env, seed, arm_label, scr, rec,
                                 donor_lookup=donor_index)
        preservation = scr.preservation_report()
        donor_out = scr.donor_index() if mode == "off" else None
        scr.detach()
        pointer = rec.finalize()
        zg_accum.observe(agent)
        row = {
            "arm_id": arm_label, "seed": seed, "mode": mode, "stats": stats,
            "preservation": preservation, "trace_pointer": pointer,
        }
        cell.stamp(row)
    return {
        "row": row, "stats": stats, "preservation": preservation,
        "pointer": pointer, "store": rec.store, "donor_index": donor_out,
    }


def run_seed(seed: int, zg_accum: ZGoalStreamAccumulator) -> Dict[str, Any]:
    env_p0 = make_env(seed)
    agent_p0 = make_agent(env_p0)
    p0_stats = run_p0(agent_p0, env_p0, seed)

    # Flush cached graph-attached tensors before deepcopy: P0 runs with grad
    # enabled, so several agent-internal caches hold non-leaf tensors that
    # torch.Tensor.__deepcopy__ refuses. One no-grad tick overwrites them.
    agent_p0.eval()
    with torch.no_grad():
        _, _flush_obs = env_p0.reset()
        agent_p0.reset()
        _flush_harness = StepHarness(agent_p0, env_p0, train_mode=False)
        _flush_harness.reset()
        _flush_harness.step(_flush_obs)

    # hippocampal._rng defaults to the `random` MODULE (ree_core/hippocampal/
    # module.py:156), which copy.deepcopy cannot pickle ("cannot pickle 'module'
    # object"). seed_replay_rng installs a picklable, per-seed random.Random
    # instance in its place -- it governs only diverse_replay(mode="auto")'s mode
    # roll and the exploration zero-weight fallback, both quiescent here
    # (use_sleep_loop=False), so this only makes the deepcopy below work and the
    # hippocampal replay RNG reproducible and identical across the four arms.
    agent_p0.hippocampal.seed_replay_rng(seed)

    # strict=True: use_anchor_sets AND use_per_region_vs give real reach to BOTH
    # z_world and operating_mode, so a failure here should stop the run loudly.
    reach = assert_behavioural_reach(agent_p0, strict=True)

    common_config_slice = {
        "env_size": ENV_SIZE, "alpha_world": ALPHA_WORLD, "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM, "warmup_episodes": WARMUP_EPISODES,
        "rec_episodes": REC_EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
        "q081_flags": {k: True for k in q081_profile_kwargs() if k != "use_sleep_loop"},
        "use_anchor_sets": True, "use_per_region_vs": True,
    }

    arm_out: Dict[str, Dict[str, Any]] = {}
    donor_index: Optional[Dict[Tuple[int, int], BoundaryTrain]] = None
    for arm_label, mode in ARM_MODES:
        # INTACT runs first (donor_index=None) and produces the donor bank the
        # yoked arms consume; suppress needs no donor.
        lookup = donor_index if mode in ("iei_permute", "circular_shift") else None
        res = _run_arm_cell(arm_label, mode, seed, agent_p0,
                            common_config_slice, lookup, zg_accum)
        arm_out[arm_label] = res
        if arm_label == INTACT_ARM:
            donor_index = res["donor_index"]

    # ---- Level 2: input-statistics divergence (INTACT = intact reference) ----
    def _l2_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "state_visitation": stats["state_visitation"],
            "action_counts": stats["action_counts"],
            "obs_channel_mean": stats["obs_channel_mean"],
            "obs_channel_std": stats["obs_channel_std"],
            "harm_events": stats["harm_events"],
            "reward_events": stats["reward_events"],
            "n_steps": stats["n_steps"],
            "episode_lengths": stats["episode_lengths"],
        }

    intact_l2 = _l2_summary(arm_out[INTACT_ARM]["stats"])
    divergence = {
        arm: input_statistics_divergence(intact_l2, _l2_summary(arm_out[arm]["stats"]))
        for arm in SCRAMBLED_ARMS
    }

    # ---- fetch recorded arrays back for the primary/secondary stats ----
    analysis: Dict[str, Any] = {}
    for arm_label, _mode in ARM_MODES:
        res = arm_out[arm_label]
        arrays = res["store"].get(res["pointer"])["arrays"]
        analysis[arm_label] = analyze_arm_trace(arrays, seed, arm_label)

    rv_intact = analysis[INTACT_ARM]["rv_primary"]
    rv_iei = analysis[IEI_ARM]["rv_primary"]
    rv_circ = analysis[CIRC_ARM]["rv_primary"]
    print(
        f"  [seed={seed}] RV_intact={rv_intact:.4f} RV_iei={rv_iei:.4f} "
        f"RV_circ={rv_circ:.4f} "
        f"preserved_iei={arm_out[IEI_ARM]['preservation']['preserved_by_construction']} "
        f"input_stats_iei={divergence[IEI_ARM]['input_statistics_preserved']}",
        flush=True,
    )

    return {
        "seed": seed,
        "p0_stats": p0_stats,
        "behavioural_reach": reach,
        "analysis": analysis,
        "preservation": {arm: arm_out[arm]["preservation"] for arm, _m in ARM_MODES},
        "input_statistics_divergence": divergence,
        # 824/824a defect signature: bit-identical INTACT vs IEI_PERMUTE rv_primary
        # is the tell that the manipulation never reached the measured statistic,
        # not evidence of true landmark-invariance. See adjudicate().
        "rv_bit_identical_intact_vs_iei": bool(rv_intact == rv_iei),
        "arm_fingerprints": {arm: arm_out[arm]["row"]["arm_fingerprint"] for arm, _m in ARM_MODES},
        "arm_rows": {arm: arm_out[arm]["row"] for arm, _m in ARM_MODES},
    }


# ============================================================================
# Adjudication.
# ============================================================================

def adjudicate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    reach_ok = all(bool(r["behavioural_reach"]["has_behavioural_reach"]) for r in per_seed)

    valid_seeds: List[int] = []
    deltas: List[float] = []                 # RV(INTACT) - RV(IEI_PERMUTE)
    circ_deltas: List[float] = []            # RV(INTACT) - RV(CIRCULAR_SHIFT)
    rv_intact_all: List[float] = []
    rv_iei_all: List[float] = []
    rv_circ_all: List[float] = []
    rv_supp_all: List[float] = []
    valid_bit_identical: List[bool] = []
    excluded: Dict[int, str] = {}

    for r in per_seed:
        seed = r["seed"]
        a = r["analysis"]
        rv_intact_all.append(a[INTACT_ARM]["rv_primary"])
        rv_iei_all.append(a[IEI_ARM]["rv_primary"])
        rv_circ_all.append(a[CIRC_ARM]["rv_primary"])
        rv_supp_all.append(a[SUPP_ARM]["rv_primary"])

        if not a[INTACT_ARM].get("admissible", False):
            excluded[seed] = "primary_statistic_not_admissible_on_INTACT_arm"
            continue
        if not a[INTACT_ARM].get("artefact_ruled_out", False):
            excluded[seed] = "artefactual_statistic_not_ruled_out_on_INTACT_arm"
            continue
        if not r["preservation"][IEI_ARM]["preserved_by_construction"]:
            excluded[seed] = "iei_permute_arm_not_preserved_by_construction"
            continue
        if not r["input_statistics_divergence"][IEI_ARM]["input_statistics_preserved"]:
            excluded[seed] = "iei_permute_arm_confounded_input_statistics_diverged"
            continue
        if not (np.isfinite(a[INTACT_ARM]["rv_primary"]) and np.isfinite(a[IEI_ARM]["rv_primary"])):
            excluded[seed] = "non_finite_rv_statistic"
            continue

        valid_seeds.append(seed)
        deltas.append(float(a[INTACT_ARM]["rv_primary"] - a[IEI_ARM]["rv_primary"]))
        if np.isfinite(a[CIRC_ARM]["rv_primary"]):
            circ_deltas.append(float(a[INTACT_ARM]["rv_primary"] - a[CIRC_ARM]["rv_primary"]))
        valid_bit_identical.append(bool(r["rv_bit_identical_intact_vs_iei"]))

    n_valid = len(valid_seeds)
    n_valid_bit_identical = sum(1 for b in valid_bit_identical if b)
    # ALL valid seeds bit-identical is the confirmed 824/824a signature of the
    # manipulation never reaching rv_primary, not a real Outcome B.
    all_valid_seeds_bit_identical = bool(n_valid >= 1 and n_valid_bit_identical == n_valid)

    # Preservation fraction over BOTH yoked arms (iei_permute + circular_shift);
    # suppress is a lesion (preserved_by_construction=False by design) and excluded.
    preserved_fraction = _fraction_over_arms(
        per_seed, PRESERVED_ARMS,
        lambda r, arm: r["preservation"][arm]["preserved_by_construction"],
    )
    input_stats_fraction = _fraction_over_arms(
        per_seed, PRESERVED_ARMS,
        lambda r, arm: r["input_statistics_divergence"][arm]["input_statistics_preserved"],
    )

    mean_delta = float(statistics.fmean(deltas)) if deltas else float("nan")
    sd_delta = float(statistics.stdev(deltas)) if len(deltas) >= 2 else 0.0
    effective_floor = max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR)
    delta_clears_floor = bool(n_valid >= 2 and mean_delta >= effective_floor)

    mean_circ_delta = float(statistics.fmean(circ_deltas)) if circ_deltas else float("nan")

    preconditions = [
        {
            "name": "landmark_arm_behavioural_reach",
            "description": "assert_behavioural_reach(strict): boundary stream must reach "
                           "(z_world, operating_mode) via REACH_CONSUMERS. 838 enables BOTH "
                           "use_anchor_sets (z_world, MECH-269) AND use_per_region_vs "
                           "(operating_mode via apply_invalidation_broadcasts_to_regions -> "
                           "vs_rollout_gate -> E3) -- the 2026-07-28 V3-EXQ-824a autopsy fix.",
            "measured": 1.0 if reach_ok else 0.0, "threshold": 1.0, "direction": "lower",
            "control": "make_agent() sets use_anchor_sets=True and use_per_region_vs=True",
            "met": reach_ok,
        },
        {
            "name": "arm_statistics_not_degenerately_bit_identical",
            "description": "824/824a defect guard: RV(INTACT) must NOT be bit-identical to "
                           "RV(IEI_PERMUTE) at every valid seed. Exact float64 equality across "
                           "independently-simulated seeds is the confirmed signature of the "
                           "manipulation never reaching the measured pair (824a's wall: it "
                           "reached z_world but not operating_mode), not landmark-invariance. "
                           "If this fails despite use_per_region_vs=True, the pair-specific "
                           "reach check (Q081-REACH-CHECK-PAIR-SPECIFIC) is genuinely required.",
            "measured": float(n_valid - n_valid_bit_identical), "threshold": 1.0,
            "direction": "lower",
            "control": "use_per_region_vs=True gives operating_mode reach (824a lacked it)",
            "met": not all_valid_seeds_bit_identical,
        },
        {
            "name": "yoked_arms_preserved_by_construction_fraction",
            "description": "Fraction of seeds where BOTH yoked arms (iei_permute, "
                           "circular_shift) held Level-1 preservation (count/IEI-multiset/"
                           "posterior-multiset/scale-mix) exactly, per preservation_report.",
            "measured": preserved_fraction, "threshold": PRESERVATION_FRACTION_FLOOR,
            "direction": "lower",
            "control": "yoked arms replay a rearrangement of INTACT's own multiset",
            "met": bool(preserved_fraction >= PRESERVATION_FRACTION_FLOOR),
        },
        {
            "name": "yoked_arms_input_statistics_preserved_fraction",
            "description": "Fraction of seeds where Level-2 input_statistics_divergence found "
                           "no breach for BOTH yoked arms (the closed-loop confound check). A "
                           "breaching seed is CONFOUNDED, not a null, and is excluded.",
            "measured": input_stats_fraction, "threshold": INPUT_STATS_FRACTION_FLOOR,
            "direction": "lower",
            "control": "pre-registered DEFAULT_INPUT_STAT_THRESHOLDS in q081_landmark_removal",
            "met": bool(input_stats_fraction >= INPUT_STATS_FRACTION_FLOOR),
        },
        {
            "name": "sufficient_valid_seeds_for_delta_sd",
            "description": "At least 2 seeds must clear every precondition above (admissible "
                           "primary, artefact ruled out, iei_permute preserved + unconfounded) "
                           "to compute an SD of the INTACT-minus-IEI_PERMUTE delta.",
            "measured": float(n_valid), "threshold": 2.0, "direction": "lower",
            "control": None, "met": bool(n_valid >= 2),
        },
    ]
    all_preconditions_met = all(p["met"] for p in preconditions)

    criteria = [
        {
            "name": "C_ablation_delta_clears_effect_size_floor",
            "load_bearing": True,
            "passed": delta_clears_floor,
            "measured": mean_delta,
            "threshold": effective_floor,
            "note": (
                f"PRIMARY discriminator: RV(INTACT) - RV(IEI_PERMUTE). effective_floor = "
                f"max({EFFECT_SIZE_K} * sd_delta={sd_delta:.4f}, {EFFECT_SIZE_ABS_FLOOR}) = "
                f"{effective_floor:.4f}; project convention scales noise on the SD of the "
                "DELTA between arms, never the SD of either arm alone."
            ),
        },
    ]
    criteria_non_degenerate = {
        "C_ablation_delta_clears_effect_size_floor": bool(all_preconditions_met and n_valid >= 2),
    }

    # Dissociation diagnostic (NON-load-bearing): a statistic that dies under
    # iei_permute but SURVIVES circular_shift was reading landmark-train internal
    # structure, not cross-stream alignment.
    dissociation = {
        "role": "diagnostic",
        "mean_intact_minus_iei_permute": mean_delta,
        "mean_intact_minus_circular_shift": mean_circ_delta,
        "interpretation": (
            "If the coupling drops under iei_permute (alignment destroyed) but is "
            "LARGELY PRESERVED under circular_shift (landmark-train internal structure "
            "kept, only rigid-shifted), the primary statistic was reading cross-stream "
            "ALIGNMENT. If it drops equally under both, the drop may reflect loss of the "
            "landmark train's own structure rather than its alignment. NOT load-bearing; "
            "circular_shift is the conservative secondary, not the discriminator."
        ),
    }

    if not all_preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            "one or more readiness preconditions failed: " +
            "; ".join(f"{p['name']}={p['met']}" for p in preconditions if not p["met"])
        )
        summary = (
            "Substrate/arm readiness gate did not clear -- see interpretation.preconditions. "
            f"Excluded seeds: {excluded}."
        )
        if all_valid_seeds_bit_identical:
            summary += (
                " NOTE: RV(INTACT) still bit-identical to RV(IEI_PERMUTE) at every valid seed "
                "DESPITE use_per_region_vs=True -- this confirms the pair-specific reach check "
                "(Q081-REACH-CHECK-PAIR-SPECIFIC) is genuinely required; the blanket "
                "use_anchor_sets/use_per_region_vs flag reach is insufficient for this pair."
            )
    elif delta_clears_floor:
        label = "shared_organisation_landmark_dependent"
        outcome = "PASS"
        evidence_direction = "supports"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"Outcome A supported: RV(z_world, operating_mode) drops from INTACT to "
            f"IEI_PERMUTE by mean delta={mean_delta:.4f} (sd={sd_delta:.4f}) across "
            f"{n_valid} valid seeds, clearing the effective floor {effective_floor:.4f}. The "
            "coupling depends on landmark alignment, not merely on the configured clock. "
            f"Dissociation: INTACT-minus-circular_shift mean={mean_circ_delta:.4f}. Clearing "
            "the surrogate null is necessary and nowhere near sufficient; the ablation series "
            "remains the only full A-vs-B discriminator."
        )
    else:
        label = "wired_gates_only_landmark_invariant"
        outcome = "FAIL"
        evidence_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"Outcome B: RV(z_world, operating_mode) delta (INTACT-minus-IEI_PERMUTE) mean="
            f"{mean_delta:.4f} did not clear the effective floor {effective_floor:.4f} across "
            f"{n_valid} valid seeds. Whatever cross-stream coupling exists SURVIVES landmark "
            "removal -- it was measuring the clock, not landmark-dependent organisation. "
            "Informative, not a null result: FAIL is the correct outcome for a weakening "
            "finding, per the project's non-standard-directions convention."
        )

    return {
        "label": label, "outcome": outcome, "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate, "degeneracy_reason": degeneracy_reason,
        "summary": summary, "preconditions": preconditions, "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "dissociation": dissociation,
        "valid_seeds": valid_seeds, "excluded_seeds": excluded,
        "mean_delta": mean_delta, "sd_delta": sd_delta, "effective_floor": effective_floor,
        "mean_circ_delta": mean_circ_delta,
        "rv_intact_all_seeds": rv_intact_all, "rv_iei_permute_all_seeds": rv_iei_all,
        "rv_circular_shift_all_seeds": rv_circ_all, "rv_suppress_all_seeds": rv_supp_all,
    }


def _fraction_over_arms(per_seed, arms, pred) -> float:
    """Fraction of seeds where `pred(r, arm)` holds for EVERY arm in `arms`."""
    if not per_seed:
        return 0.0
    ok = 0
    for r in per_seed:
        if all(bool(pred(r, arm)) for arm in arms):
            ok += 1
    return ok / len(per_seed)


# ============================================================================
# Main
# ============================================================================

def main(dry_run: bool = False) -> Any:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    global WARMUP_EPISODES, REC_EPISODES, STEPS_PER_EPISODE
    if dry_run:
        WARMUP_EPISODES, REC_EPISODES, STEPS_PER_EPISODE = 2, 2, 20

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run} seeds={seeds} "
          f"warmup_episodes={WARMUP_EPISODES} rec_episodes={REC_EPISODES} "
          f"steps_per_episode={STEPS_PER_EPISODE} arms={[a for a, _ in ARM_MODES]}", flush=True)
    t0 = time.time()

    zg_accum = ZGoalStreamAccumulator()
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(run_seed(seed, zg_accum))

    verdict = adjudicate(per_seed)
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] label={verdict['label']} outcome={verdict['outcome']} "
          f"mean_delta={verdict['mean_delta']:.4f} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    arm_results = []
    for arm_label, _mode in ARM_MODES:
        for r in per_seed:
            arm_results.append({
                "arm_id": arm_label, "seed": r["seed"],
                "arm_fingerprint": r["arm_fingerprints"][arm_label],
                "is_lesion": bool(arm_label == SUPP_ARM),
                "preservation": r["preservation"][arm_label],
                **r["analysis"][arm_label],
            })

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": verdict["outcome"],
        "evidence_direction": verdict["evidence_direction"],
        "non_degenerate": verdict["non_degenerate"],
        "degeneracy_reason": verdict["degeneracy_reason"],
        "interpretation": {
            "label": verdict["label"],
            "summary": verdict["summary"],
            "preconditions": verdict["preconditions"],
            "criteria_non_degenerate": verdict["criteria_non_degenerate"],
            "dissociation": verdict["dissociation"],
            "interpretation_rules": [
                "Clearing the surrogate null is NECESSARY and NOWHERE NEAR SUFFICIENT for "
                "Outcome A -- wired coordination clears it correctly.",
                "A statistic that SURVIVES the landmark-removal arm was measuring the clock -- "
                "Outcome B evidence, NOT support for the claim.",
                "Neither control, nor both together, is the ablation series, which remains the "
                "only full A-vs-B discriminator.",
                "The SUPPRESS arm is a LESION reference (removes the drive as well as the "
                "alignment); reported for context, never the discriminator.",
            ],
            "outcome_taxonomy_note": (
                "This design adjudicates ONLY Outcome A vs Outcome B. Outcome C (unrelated) is "
                "partially visible in per-seed rv_p_value_vs_surrogate on the INTACT arm. "
                "Outcomes D (too-much-sharing-collapses) and E (event-relative beats clock-"
                "relative, incl. sleep-phase structure) need different comparator arms and are "
                "deferred to a lettered follow-up."
            ),
        },
        "criteria": verdict["criteria"],
        "elapsed_seconds": elapsed,
        "per_seed_results": per_seed,
        "arm_results": arm_results,
        "delta_summary": {
            "valid_seeds": verdict["valid_seeds"],
            "excluded_seeds": verdict["excluded_seeds"],
            "mean_delta": verdict["mean_delta"],
            "sd_delta": verdict["sd_delta"],
            "effective_floor": verdict["effective_floor"],
            "mean_intact_minus_circular_shift": verdict["mean_circ_delta"],
            "rv_intact_all_seeds": verdict["rv_intact_all_seeds"],
            "rv_iei_permute_all_seeds": verdict["rv_iei_permute_all_seeds"],
            "rv_circular_shift_all_seeds": verdict["rv_circular_shift_all_seeds"],
            "rv_suppress_all_seeds": verdict["rv_suppress_all_seeds"],
        },
        "primary_pair": list(PRIMARY_PAIR),
        "secondary_reduce": SECONDARY_REDUCE,
        "arms": {a: m for a, m in ARM_MODES},
        "validate_the_null": {
            "spares_injected_real_relation": (
                "proven generically at the library level, 2026-07-22, by "
                "tests/contracts/test_q081_surrogate_null.py::"
                "test_null_spares_an_injected_real_relation (22-test suite)."
            ),
            "kills_artefactual_statistic_this_run": [
                {"seed": r["seed"],
                 "artefact_ruled_out": r["analysis"][INTACT_ARM]["artefact_ruled_out"],
                 "artefact_ensemble_spread": r["analysis"][INTACT_ARM]["artefact_ensemble_spread"]}
                for r in per_seed
            ],
        },
        "supersession_context": (
            "New number (not a lettered iteration of 824a): 838 expands the design to four "
            "arms and enables use_per_region_vs=True (the 2026-07-28 V3-EXQ-824a autopsy fix). "
            "824 FAIL (inert arm) and 824a substrate_not_ready_requeue (reached z_world not "
            "operating_mode) are NOT superseded evidence -- they remain the diagnostic trail."
        ),
    }

    out_path = write_flat_manifest(
        manifest, dry_run=False,
        config={
            "seeds": seeds, "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
            "alpha_world": ALPHA_WORLD, "env_size": ENV_SIZE,
            "warmup_episodes": WARMUP_EPISODES, "rec_episodes": REC_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE, "primary_pair": list(PRIMARY_PAIR),
            "arms": {a: m for a, m in ARM_MODES},
            "use_anchor_sets": True, "use_per_region_vs": True,
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=zg_accum.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)
    return verdict["outcome"], out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="V3-EXQ-838 Q-081 cross-stream recording: INTACT vs "
                    "iei_permute / circular_shift / suppress")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        emit_outcome(outcome="PASS", manifest_path=None, dry_run=True)
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path, dry_run=bool(args.dry_run))
    sys.exit(0)
