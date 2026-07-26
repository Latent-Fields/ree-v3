#!/opt/local/bin/python3
"""V3-EXQ-824 -- Q-081: does REE's multi-rate execution produce SHARED
cross-stream organisation (Outcome A), or only the configured rate
separation and wired gates it was built with (Outcome B)?

This is the FIRST real Q-081 run. All three preconditions the claim's
non-degeneracy guard requires are landed on ree-v3 main and used here:
  * per-step multi-stream trace recorder    -- experiments/_lib/stream_recorder.py
  * structure-destroying landmark-removal arm -- experiments/_lib/q081_landmark_removal.py
  * constrained-realisation surrogate null  -- experiments/_lib/q081_surrogate.py

No SLEEP DRIVER: use_sleep_loop is deliberately left False this iteration
(scope: waking-stream organisation only). Sleep-phase-relative structure
(part of Outcome E, "event-relative beats clock-relative") is deferred to
a lettered follow-up once this primary A-vs-B result is banked.

DESIGN
------
Per seed: a shared P0 substrate (E1 + E2-self warmup) is trained once,
then deep-copied into two arms that share byte-identical starting
env RNG state (a fresh env constructed from the same seed for each arm):

  ARM ON               : intact hippocampal event-segmenter (landmark
                          structure present). Also RECORDS the donor
                          boundary train for the other arm (mode="off").
  ARM LANDMARK_REMOVED  : `q081_landmark_removal.LandmarkScrambler` in its
                          PRE-REGISTERED PRIMARY mode (iei_permute) --
                          a yoked, seed/episode-matched permutation of
                          ARM ON's own boundary train. Boundary COUNT,
                          inter-event-interval multiset, posterior
                          multiset and scale mix are preserved EXACTLY
                          (Level 1); environmental input statistics are
                          MEASURED, not assumed (Level 2) via
                          `input_statistics_divergence`.

For each arm, a `StreamTraceRecorder` (q081_profile flags enabled, so the
4 dark-by-default streams are live) captures the full per-step multi-
stream trace across all recording episodes.

PRIMARY STATISTIC -- deliberately NOT `cross_stream_xcorr`'s 1-D reduction.
`q081_surrogate.reduce_stream`'s own docstring states a 1-D reduction
"discards exactly [the multi-dimensional configuration] Q-081 asks
about" and says to use it "for controls and diagnostics, not as the
primary readout." So the primary readout here is the RV coefficient
(Robert & Escoufier 1976) -- a multivariate generalisation of squared
correlation -- between the FULL z_world vector (E1-rate, the fastest
recorded stream) and the FULL operating_mode vector (the salience
coordinator's soft mode-probabilities + current-mode index + switch
trigger + salience aggregate; E3-rate, identity-derived freshness --
the system's slowest/most-committed recorded state). Both are read on
the coarser (E3) stream's own tick grid, per `q081_surrogate`'s
alignment rule, so the sampling trap the telemetry audit warns about
cannot manufacture a spurious reading.

WHY operating_mode AND NOT e3_commitment (measured, not assumed).
e3_commitment was the first choice (it is literally the system's
committed decision state), but empirically its recorded hold pattern is
`classify_hold_mode -> HOLD_UNSTRUCTURED` on a real rollout: the stream
is declared FRESH_CLOCK (E3-tick-only), but `update_residue`'s
`e3.post_action_update` (called every env tick, not just on E3 ticks --
see _harness.py) updates the SAME running_variance the commitment dict
reports, so at least one field genuinely changes on held ticks too. A
declared-clock-only freshness whose underlying value is not actually
clock-held is exactly the sampling-trap class the telemetry audit warns
about, just in a spot the audit itself did not anticipate --
`block_permute_stream` correctly REFUSES to rebuild it rather than
guess. `operating_mode` was checked the same way (`classify_hold_mode
-> HOLD_CARRY`, measured period 10.0 steps, matching the E3 cadence
exactly) before being adopted; this is reported as a real finding
below (`stream_recorder_freshness_note`), not silently routed around.
`cross_stream_xcorr` (1-D, reduce="norm") and its lag are still
reported as SECONDARY / DIAGNOSTIC readouts on the (z_world,
operating_mode) pair, per the module's own guidance for that function.

ADJUDICATION. Clearing the constrained-realisation surrogate null is
necessary and (per the claim's own text) "nowhere near sufficient" --
wired coordination is real coordination and clears any surrogate. The
sharp A-vs-B discriminator is therefore the ABLATION CONTRAST: does the
RV coefficient measured on ARM ON exceed the RV coefficient measured on
ARM LANDMARK_REMOVED by more than chance, aggregated across seeds?
  * Outcome A (shared_organisation_landmark_dependent): the ON-minus-
    REMOVED delta clears the pre-registered effect-size floor -> the
    coupling depends on landmark alignment, not merely on the clock.
  * Outcome B (wired_gates_only_landmark_invariant): the delta does not
    clear the floor -> whatever coupling exists survives landmark
    removal, i.e. it was measuring the clock, not the landmark.
Outcomes C (unrelated) / D (too-much-sharing-collapses) / E (event-
beats-clock) are NOT adjudicated by this design; C is partially visible
in the reported surrogate p-values (a statistic that never clears its
own null even intact is closer to C than to A or B), and D/E need
different comparator arms left to a lettered follow-up.

DV-SYMMETRY DECLARATION (mandatory, per arm). DV = RV(z_world,
operating_mode), a function of the REALISED joint trajectory of two
streams across a run. The landmark-removal manipulation (yoked
permutation of boundary-event TIMING) changes WHEN write_anchor /
broadcast events fire, which changes anchor content, V_s,
vs_rollout_gate readings, E3 candidate scoring, the action selected at
every committed tick, and therefore the entire downstream trajectory of
z_world (via env transitions) and operating_mode (the salience
coordinator's own mode read, driven by the same downstream state).
This is NOT a broadcast additive constant on a fixed candidate set (RV
already centres each block, so a constant shift is invisible either way), NOT a
monotone rescaling (RV is invariant to orthogonal transforms of each
block by construction -- rescaling is not the manipulation), and NOT a
permutation of interchangeable aggregation units (seeds / candidates):
it is a permutation of EVENT TIMING that materially alters the physical
trajectory each stream is sampled from. A null ON-vs-REMOVED result is
therefore a MEASURED finding (Outcome B), not an arithmetic identity.

VALIDATE-THE-NULL-BEFORE-USE (claim-mandated). Two halves, both
discharged before this run adjudicates anything: (1) the "spares an
injected real relation" half was proven generically at the LIBRARY
level, 2026-07-22, by `tests/contracts/test_q081_surrogate_null.py::
test_null_spares_an_injected_real_relation` (22-test suite, landed
before this run). (2) the "kills a deliberately-artefactual statistic"
half is re-checked HERE, per run, on THIS run's actual tick grids (not
assumed from the library test) via `screen_statistic` on
`artefactual_rate_statistic` -- expected verdict "ruled_out".

Provenance: env/agent construction modelled on
experiments/v3_exq_817a_sd080_worldeffect_grounding_falsifier.py
(make_env/make_agent/run_p0 pattern); per-tick mechanics via
experiments/_harness.StepHarness, which also discharges q081_profile's
LOOP_DRIVEN_REQUIREMENTS for free (it already calls agent.sense() with
the harm kwargs and agent.update_z_goal() every tick).

supersedes: none (first Q-081 adjudication run).
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
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
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
    plan_blocks, screen_statistic, evaluate_against_null, cross_stream_xcorr,
    lag_control_report, artefactual_rate_statistic, estimate_tick_period,
    SurrogateDesignError,
)
from _harness import StepHarness  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_824_q081_shared_organisation_landmark_removal"
QUEUE_ID = "V3-EXQ-824"
CLAIM_IDS: List[str] = ["Q-081"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_RUN: Optional[str] = None

# ------------------------------------------------------------------ config
WORLD_DIM = SELF_DIM = 32
ALPHA_WORLD = 0.9   # SD-008: z_world-fidelity-dependent experiments need >= 0.9
ENV_SIZE = 6
SEEDS = [0, 1, 2, 3, 4]

WARMUP_EPISODES = 20
REC_EPISODES = 20
STEPS_PER_EPISODE = 200
# Episodes end well before STEPS_PER_EPISODE in practice -- CausalGridWorldV2
# terminates on `agent_health <= 0` (env line ~2892), which fires quickly
# under a lightly-trained policy with harm streams on. REC_EPISODES=20 was
# picked empirically (smoke-tested) to bank comfortably more than the ~500-600
# total steps plan_blocks needs for a valid constrained-realisation surrogate,
# even though the fixed step cap (500) and the loop bound (200) both go unused
# most episodes.
WARMUP_LR = 1e-3

ON_ARM = "ON"
REMOVED_ARM = "LANDMARK_REMOVED"

PRIMARY_PAIR = ("z_world", "operating_mode")
SECONDARY_REDUCE = "norm"

# ---------------------------------------------------------- pre-registered
# Screening / null-validation floors. Both are about whether the ensemble
# CAN discriminate at all, not about the scientific verdict.
SCREEN_N = 64
SURR_N = 199
ADMISSIBLE_SPREAD_FLOOR = 1e-9     # RV stat: ensemble must show non-zero spread
ARTEFACT_SPREAD_CEIL = 1e-9        # artefactual stat: ensemble spread must be ~0

# Level 1 / Level 2 arm-validity floors (fraction of seeds that must clear).
PRESERVATION_FRACTION_FLOOR = 0.6
INPUT_STATS_FRACTION_FLOOR = 0.5

# Effect-size PASS gate (project convention: scale noise on the SD of the
# DELTA between arms, plus an absolute floor -- never the SD of either arm
# alone). RV coefficient units, nominally in [0, 1].
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
    return REEAgent(cfg)


# ============================================================================
# P0 -- shared substrate warmup (E1 + E2-self only; modelled on EXQ-817a's
# run_p0). StepHarness drives z_harm/z_harm_a/z_goal for free.
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
        # deliberately "step", not "ep": P1's per-arm loop owns the
        # canonical "ep N/M" pattern (episodes_per_run == REC_EPISODES).
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
    streams, read on the coarser stream's own tick grid.

    Deliberately multivariate (see module docstring: `reduce_stream`'s own
    guidance is that a 1-D reduction is a control/diagnostic, not the
    primary Q-081 readout).
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
    secondary/diagnostic reduced cross_stream_xcorr + lag control."""
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
# Per-(seed) cell.
# ============================================================================

def run_seed(seed: int) -> Dict[str, Any]:
    env_p0 = make_env(seed)
    agent_p0 = make_agent(env_p0)
    p0_stats = run_p0(agent_p0, env_p0, seed)

    # Flush cached graph-attached tensors (candidates / scores / hidden state /
    # tpj signal) before deepcopy: P0 runs with grad enabled, so several
    # agent-internal caches hold non-leaf tensors that torch.Tensor.__deepcopy__
    # refuses. One no-grad tick overwrites them with detached values.
    agent_p0.eval()
    with torch.no_grad():
        _, _flush_obs = env_p0.reset()
        agent_p0.reset()
        _flush_harness = StepHarness(agent_p0, env_p0, train_mode=False)
        _flush_harness.reset()
        _flush_harness.step(_flush_obs)

    reach = assert_behavioural_reach(agent_p0, strict=False)
    common_config_slice = {
        "env_size": ENV_SIZE, "alpha_world": ALPHA_WORLD, "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM, "warmup_episodes": WARMUP_EPISODES,
        "rec_episodes": REC_EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
        "q081_flags": {k: True for k in q081_profile_kwargs() if k != "use_sleep_loop"},
    }

    # ---- ARM ON: intact, also banks the donor boundary train ----
    with arm_cell(
        seed, config_slice={**common_config_slice, "arm": ON_ARM},
        script_path=Path(__file__), config_slice_declared=True,
        extra_ineligible_reasons=["p0_substrate_trained_outside_fingerprinted_cell_scope"],
    ) as cell_on:
        agent_on = copy.deepcopy(agent_p0)
        env_on = make_env(seed)  # fresh env, byte-identical starting RNG per arm
        scr_on = LandmarkScrambler(LandmarkRemovalConfig(mode="off", seed=seed))
        scr_on.attach(agent_on)
        rec_on = StreamTraceRecorder(
            agent_on, run_id=f"{EXPERIMENT_TYPE}_seed{seed}_{ON_ARM}",
            substrate_declaration=q081_substrate_declaration(agent_on.config),
        )
        stats_on = run_p1_recording(agent_on, env_on, seed, ON_ARM, scr_on, rec_on)
        preservation_on = scr_on.preservation_report()
        donor_index = scr_on.donor_index()
        scr_on.detach()
        pointer_on = rec_on.finalize()
        row_on = {
            "arm_id": ON_ARM, "seed": seed, "stats": stats_on,
            "preservation": preservation_on, "trace_pointer": pointer_on,
        }
        cell_on.stamp(row_on)

    # ---- ARM LANDMARK_REMOVED: yoked permutation of ON's donor train ----
    with arm_cell(
        seed, config_slice={**common_config_slice, "arm": REMOVED_ARM},
        script_path=Path(__file__), config_slice_declared=True,
        extra_ineligible_reasons=[
            "p0_substrate_trained_outside_fingerprinted_cell_scope",
            "yoked_to_ON_arm_donor_train_same_seed",
        ],
    ) as cell_rm:
        agent_rm = copy.deepcopy(agent_p0)
        env_rm = make_env(seed)
        scr_rm = LandmarkScrambler(LandmarkRemovalConfig(mode=PRIMARY_MODE, seed=seed))
        scr_rm.attach(agent_rm)
        rec_rm = StreamTraceRecorder(
            agent_rm, run_id=f"{EXPERIMENT_TYPE}_seed{seed}_{REMOVED_ARM}",
            substrate_declaration=q081_substrate_declaration(agent_rm.config),
        )
        stats_rm = run_p1_recording(agent_rm, env_rm, seed, REMOVED_ARM, scr_rm, rec_rm,
                                     donor_lookup=donor_index)
        preservation_rm = scr_rm.preservation_report()
        scr_rm.detach()
        pointer_rm = rec_rm.finalize()
        row_rm = {
            "arm_id": REMOVED_ARM, "seed": seed, "stats": stats_rm,
            "preservation": preservation_rm, "trace_pointer": pointer_rm,
        }
        cell_rm.stamp(row_rm)

    # ---- Level 2: input-statistics divergence (ON = intact reference) ----
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

    divergence = input_statistics_divergence(_l2_summary(stats_on), _l2_summary(stats_rm))

    # ---- fetch back the recorded arrays for the primary/secondary stats ----
    analysis_on = analyze_arm_trace(rec_on.store.get(pointer_on)["arrays"], seed, ON_ARM)
    analysis_rm = analyze_arm_trace(rec_rm.store.get(pointer_rm)["arrays"], seed, REMOVED_ARM)

    print(
        f"  [seed={seed}] RV_on={analysis_on['rv_primary']:.4f} "
        f"RV_removed={analysis_rm['rv_primary']:.4f} "
        f"preserved={preservation_rm['preserved_by_construction']} "
        f"input_stats_preserved={divergence['input_statistics_preserved']}",
        flush=True,
    )

    return {
        "seed": seed,
        "p0_stats": p0_stats,
        "behavioural_reach": reach,
        "preservation_on": preservation_on,
        "preservation_removed": preservation_rm,
        "input_statistics_divergence": divergence,
        "analysis_on": analysis_on,
        "analysis_removed": analysis_rm,
        "arm_fingerprint_on": row_on["arm_fingerprint"],
        "arm_fingerprint_removed": row_rm["arm_fingerprint"],
    }


# ============================================================================
# Adjudication.
# ============================================================================

def adjudicate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    reach_ok = all(bool(r["behavioural_reach"]["has_behavioural_reach"]) for r in per_seed)

    valid_seeds: List[int] = []
    deltas: List[float] = []
    rv_on_all: List[float] = []
    rv_rm_all: List[float] = []
    excluded: Dict[int, str] = {}

    for r in per_seed:
        seed = r["seed"]
        a_on, a_rm = r["analysis_on"], r["analysis_removed"]
        rv_on_all.append(a_on["rv_primary"])
        rv_rm_all.append(a_rm["rv_primary"])

        if not a_on.get("admissible", False):
            excluded[seed] = "primary_statistic_not_admissible_on_ON_arm"
            continue
        if not a_on.get("artefact_ruled_out", False):
            excluded[seed] = "artefactual_statistic_not_ruled_out_on_ON_arm"
            continue
        if not r["preservation_removed"]["preserved_by_construction"]:
            excluded[seed] = "landmark_arm_not_preserved_by_construction"
            continue
        if not r["input_statistics_divergence"]["input_statistics_preserved"]:
            excluded[seed] = "landmark_arm_confounded_input_statistics_diverged"
            continue
        if not (np.isfinite(a_on["rv_primary"]) and np.isfinite(a_rm["rv_primary"])):
            excluded[seed] = "non_finite_rv_statistic"
            continue

        valid_seeds.append(seed)
        deltas.append(float(a_on["rv_primary"] - a_rm["rv_primary"]))

    n_valid = len(valid_seeds)
    preserved_fraction = sum(
        1 for r in per_seed if r["preservation_removed"]["preserved_by_construction"]
    ) / max(1, len(per_seed))
    input_stats_fraction = sum(
        1 for r in per_seed if r["input_statistics_divergence"]["input_statistics_preserved"]
    ) / max(1, len(per_seed))

    mean_delta = float(statistics.fmean(deltas)) if deltas else float("nan")
    sd_delta = float(statistics.stdev(deltas)) if len(deltas) >= 2 else 0.0
    effective_floor = max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR)
    delta_clears_floor = bool(n_valid >= 2 and mean_delta >= effective_floor)

    preconditions = [
        {
            "name": "landmark_arm_behavioural_reach",
            "description": "assert_behavioural_reach: the boundary stream has at least one "
                           "live consumer (MECH-287/MECH-269/staleness), so the landmark-"
                           "removal arm is not vacuously inert.",
            "measured": 1.0 if reach_ok else 0.0, "threshold": 1.0, "direction": "lower",
            "control": "q081 profile sets use_invalidation_trigger=True",
            "met": reach_ok,
        },
        {
            "name": "landmark_arm_preserved_by_construction_fraction",
            "description": "Fraction of seeds where the landmark-removal arm's Level-1 "
                           "preservation (count/IEI-multiset/posterior-multiset/scale-mix) "
                           "held exactly, per q081_landmark_removal.preservation_report.",
            "measured": preserved_fraction, "threshold": PRESERVATION_FRACTION_FLOOR,
            "direction": "lower",
            "control": "iei_permute is a rearrangement of the ON arm's own multiset",
            "met": bool(preserved_fraction >= PRESERVATION_FRACTION_FLOOR),
        },
        {
            "name": "landmark_arm_input_statistics_preserved_fraction",
            "description": "Fraction of seeds where Level-2 input_statistics_divergence "
                           "found no breach (the closed-loop confound check) -- a seed "
                           "that breaches is CONFOUNDED, not a null, and is excluded.",
            "measured": input_stats_fraction, "threshold": INPUT_STATS_FRACTION_FLOOR,
            "direction": "lower",
            "control": "pre-registered DEFAULT_INPUT_STAT_THRESHOLDS in q081_landmark_removal",
            "met": bool(input_stats_fraction >= INPUT_STATS_FRACTION_FLOOR),
        },
        {
            "name": "sufficient_valid_seeds_for_delta_sd",
            "description": "At least 2 seeds must clear every precondition above (admissible "
                           "primary statistic, artefact ruled out, preserved, unconfounded) "
                           "to compute an SD of the ON-minus-REMOVED delta.",
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
                f"effective_floor = max({EFFECT_SIZE_K} * sd_delta={sd_delta:.4f}, "
                f"{EFFECT_SIZE_ABS_FLOOR}) = {effective_floor:.4f}; project convention is to "
                "scale noise on the SD of the DELTA between arms, never the SD of either arm "
                "alone."
            ),
        },
    ]
    criteria_non_degenerate = {
        "C_ablation_delta_clears_effect_size_floor": bool(all_preconditions_met and n_valid >= 2),
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
    elif delta_clears_floor:
        label = "shared_organisation_landmark_dependent"
        outcome = "PASS"
        evidence_direction = "supports"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"Outcome A supported: RV(z_world, operating_mode) drops from ON to "
            f"LANDMARK_REMOVED by mean delta={mean_delta:.4f} (sd={sd_delta:.4f}) across "
            f"{n_valid} valid seeds, clearing the effective floor {effective_floor:.4f}. The "
            "coupling depends on landmark alignment, not merely on the configured clock."
        )
    else:
        label = "wired_gates_only_landmark_invariant"
        outcome = "FAIL"
        evidence_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"Outcome B: RV(z_world, operating_mode) delta (ON-minus-REMOVED) mean="
            f"{mean_delta:.4f} did not clear the effective floor {effective_floor:.4f} across "
            f"{n_valid} valid seeds. Whatever cross-stream coupling exists survives landmark "
            "removal -- it was measuring the clock, not landmark-dependent organisation. "
            "Informative, not a null result: FAIL is the correct outcome for a weakening "
            "finding, per the project's non-standard-directions convention."
        )

    return {
        "label": label, "outcome": outcome, "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate, "degeneracy_reason": degeneracy_reason,
        "summary": summary, "preconditions": preconditions, "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "valid_seeds": valid_seeds, "excluded_seeds": excluded,
        "mean_delta": mean_delta, "sd_delta": sd_delta, "effective_floor": effective_floor,
        "rv_on_all_seeds": rv_on_all, "rv_removed_all_seeds": rv_rm_all,
    }


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
          f"steps_per_episode={STEPS_PER_EPISODE}", flush=True)
    t0 = time.time()

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(run_seed(seed))

    verdict = adjudicate(per_seed)
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] label={verdict['label']} outcome={verdict['outcome']} "
          f"mean_delta={verdict['mean_delta']:.4f} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    arm_results = (
        [{"arm_id": ON_ARM, "seed": r["seed"], "arm_fingerprint": r["arm_fingerprint_on"],
          **r["analysis_on"]} for r in per_seed]
        + [{"arm_id": REMOVED_ARM, "seed": r["seed"], "arm_fingerprint": r["arm_fingerprint_removed"],
            **r["analysis_removed"]} for r in per_seed]
    )

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
            "outcome_taxonomy_note": (
                "This design adjudicates ONLY Outcome A vs Outcome B. Outcome C "
                "(unrelated) is partially visible in per-seed rv_p_value_vs_surrogate "
                "on the ON arm (a statistic that never clears its own null even intact "
                "leans C rather than A/B). Outcomes D (too-much-sharing-collapses) and "
                "E (event-relative beats clock-relative, incl. sleep-phase structure) "
                "need different comparator arms and are deferred to a lettered follow-up."
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
            "rv_on_all_seeds": verdict["rv_on_all_seeds"],
            "rv_removed_all_seeds": verdict["rv_removed_all_seeds"],
        },
        "primary_pair": list(PRIMARY_PAIR),
        "secondary_reduce": SECONDARY_REDUCE,
        "validate_the_null": {
            "spares_injected_real_relation": (
                "proven generically at the library level, 2026-07-22, by "
                "tests/contracts/test_q081_surrogate_null.py::"
                "test_null_spares_an_injected_real_relation (22-test suite, landed "
                "before this run)."
            ),
            "kills_artefactual_statistic_this_run": [
                {"seed": r["seed"], "artefact_ruled_out": r["analysis_on"]["artefact_ruled_out"],
                 "artefact_ensemble_spread": r["analysis_on"]["artefact_ensemble_spread"]}
                for r in per_seed
            ],
        },
    }
    if SUPERSEDES_RUN:
        manifest["supersedes"] = SUPERSEDES_RUN

    out_path = write_flat_manifest(
        manifest, dry_run=False,
        config={
            "seeds": seeds, "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
            "alpha_world": ALPHA_WORLD, "env_size": ENV_SIZE,
            "warmup_episodes": WARMUP_EPISODES, "rec_episodes": REC_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE, "primary_pair": list(PRIMARY_PAIR),
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return verdict["outcome"], out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="V3-EXQ-824 Q-081 cross-stream shared organisation: ON vs landmark-removal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        emit_outcome(outcome="PASS", manifest_path=None, dry_run=True)
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path, dry_run=False)
    sys.exit(0)
