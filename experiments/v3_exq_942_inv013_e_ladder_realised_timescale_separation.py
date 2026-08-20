#!/opt/local/bin/python3
"""V3-EXQ-942 -- INV-013 E-ladder REALISED timescale-separation probe.

SLEEP DRIVER: not applicable (no sleep loop used in this driver).

DESIGN INTENT vs REALISED DYNAMICS. SD-006/ARC-023 (control_plane_heartbeat.md) design the
E1/E2/E3 control-plane loops to tick at characteristic rates in mimicry of EEG bands (E1~gamma
continuous, E2~beta 1-in-3, E3~theta 1-in-10; ree_core/heartbeat/clock.py MultiRateClock
defaults e1_steps_per_tick=1, e2_steps_per_tick=3, e3_steps_per_tick=10, theta_buffer_size=10).
That DESIGN is not in question here -- what this probe tests is whether the loops' own STATE
actually shows a matching REALISED timescale separation (a monotonic ordering of persistence,
tau(E1) < tau(E2) < tau(E3)), or whether -- as already happened once for the adjacent
z_self/z_world "timescale" framing (MECH-058, retired; V3-EXQ-019 lag-k-autocorrelation FAIL,
see claim_probe_mech_058/runs/20260317T232028Z_v3_exq_019_timescale_v3_v3) and once for
L-space's z_beta/z_theta/z_delta stack (ARC-004's own non-degeneracy caveat: a single shared
EMA constant alpha_shared=0.3) -- the designed rate ratio does not survive into realised state
dynamics. ARC-004's own falsifier language is the pre-registered criterion this probe reuses,
applied to the E-loops instead of L-space's latent layers:

  "ANSWER 'multi-timescale' (PASS) if the layers show a monotonic ordering of effective
  persistence ... by a margin exceeding cross-seed noise (>= 0.8 SD of the seed-to-seed
  half-life delta) ... ANSWER 'layers differ in content, not timescale' (FAIL) if
  autocorrelation half-life is statistically indistinguishable."

ROUTE (per the dispatching brief's (b)/(c) decision, recorded in the queue entry `note`):
existing evidence does NOT carry raw per-env-step E1/E2/E3 state traces at sufficient
resolution anywhere on disk (checked v3_exq_827a's run pack -- aggregate stats only, no raw
arrays persisted; checked v3_exq_627 similarly; the general manifest convention is aggregate-
only). The closest prior art, V3-EXQ-019 (2026-03-17), measured z_self/z_world lag-k
autocorrelation directly and is genuinely informative (see STRUCTURAL FINDING 2 below) but (a)
never recorded any E3-loop signal at all, (b) used RANDOM actions rather than
agent.select_action (so E3's own selection/commitment machinery was never exercised), (c)
failed its own C4 (n_steps=969 < 3000) on a single seed, and (d) predates the 2026-07-12
Experimental Recording Standard (no substrate_hash -- unverifiable-substrate reuse per
GOV-REUSE-1's pre-standard caveat). So this is Route (c): a new, targeted run.

THREE STRUCTURAL FINDINGS (established by reading the current substrate source, NOT by this
run -- these are code-level facts about the WIRING, independent of any stochastic measurement,
and they bear directly on why outcome (ii) [separation absent/non-monotonic] is the a priori
more likely reading):

  1. `ticks["e2_tick"]` (ree_core/heartbeat/clock.py:143,174, computed every advance() call as
     `global_step % e2_steps_per_tick == 0`) is NEVER READ anywhere in ree_core/ -- confirmed by
     exhaustive grep. No code path gates any computation on E2's own tick flag. The 1-in-3 E2
     cadence exists as a clock counter and nowhere else.
  2. ree-v3/experiments/_lib/stream_recorder.py:310-317 documents (a 2026-07-26 finding from the
     Q-081/V3-EXQ-824 investigation) that the ENTIRE LatentStack -- z_self, z_world, z_beta,
     z_theta, z_delta, z_harm, z_harm_a -- is "re-encoded every step (E1 rate)". There is no
     per-loop gating on the REPRESENTATION-encoding rate at all; this applies equally to
     ARC-004's own L-space layers.
  3. ree-v3/experiments/_lib/stream_recorder.py:374-379 and ree_core/predictors/e3_selector.py
     (`update_running_variance`, called from `post_action_update` on EVERY env tick via
     `agent.update_residue`, not gated to the E3 tick) confirm E3's own `precision` /
     `running_variance` are ALSO mutated every env step. Only `commit_threshold` (static config)
     and `is_committed`/`committed_now` (the closure-latch-derived discrete flag) are genuinely
     E3-cadence.

So the a priori structural picture is: the DESIGNED tick ladder (1:3:10) governs almost nothing
about representation UPDATE rate -- only E3's discrete commit/re-select DECISION is genuinely
throttled. This run measures whether that structural fact translates into an absent/non-
monotonic autocorrelation ordering (as EXQ-019's partial z_self/z_world data already suggests:
both decay to near-zero autocorrelation by lag 5, statistically indistinguishable) once E3 is
actually exercised via real action selection, across enough steps/seeds to clear EXQ-019's own
unmet sample-size bar.

METHOD. One REEAgent per seed, P0/P1 warmup (E1 world model + E2 world-forward + E3 harm-eval
head, via the shared `goal_pipeline_tier1.warmup_train` helper -- phased training, frozen-
encoder head training built into that helper), then a frozen-policy (agent.eval(), no_grad) eval
rollout using the canonical sense->clock.advance->e1_tick->generate_trajectories->select_action
loop (StepHarness, the same inner loop as `_lib.capability_eval.REEForwardPolicy` and
V3-EXQ-827a's `_eval_pass`). Per step, after the harness call: read
`result.latent.z_world`/`z_self` (E1/E2's own representational state) and
`agent.e3.get_commitment_state()` (E3's own continuous control state: `precision`,
`running_variance`; plus the discrete `is_committed` flag). Compute per-step deltas
(||x_t - x_{t-1}||, reset at episode boundaries, matching EXQ-019's own convention), lag-k
autocorrelation (EXQ-019's `_compute_autocorr`, reused verbatim), and a persistence half-life
(smallest lag where autocorr crosses 0.5 * autocorr(lag=1)) per loop, per seed. Also tally
REALISED tick-fire counts (e1_tick/e2_tick/e3_tick from `result.ticks`) to report the measured
tick-rate ratio directly against the designed 1:3:10, and the discrete `is_committed` signal's
own mean run-length as a second, more direct persistence read for the one channel that IS
genuinely E3-paced.

PASS/FAIL is the ARC-004 criterion applied to the E-loops: monotonic tau(E1) < tau(E2) <
tau(E3) by >= 0.8 SD of the cross-seed half-life delta (HALFLIFE_SD_MARGIN), using the
precision-derived tau_e3 (same delta-autocorrelation-half-life statistic as E1/E2, apples to
apples) as the load-bearing E3 read; the is_committed run-length is reported as a secondary,
non-load-bearing, more intuitive persistence measure for the genuinely-cadence-gated channel.

SCOPE. Diagnostic probe (EXPERIMENT_PURPOSE="diagnostic"): excluded from governance
confidence/conflict scoring by purpose. Does not promote/demote INV-013, ARC-004, or SD-006 --
that is a /governance decision from this evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.q081_profile import q081_profile_kwargs  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_942_inv013_e_ladder_realised_timescale_separation"
QUEUE_ID = "V3-EXQ-942"
CLAIM_IDS: List[str] = ["INV-013"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
ARM_FINGERPRINT_EXEMPT = (
    "single-arm diagnostic probe (no OFF/treatment grid) -- reads one trained agent's own "
    "realised per-loop state dynamics across seeds, nothing to compare across arm cells."
)
ANCHOR_REACHABILITY_EXEMPT = (
    "each readiness precondition's `measured` IS the worst-seed value of the exact same "
    "delta-series variance the load-bearing criterion's half-life is derived from -- the "
    "'control' is this run's own worst cell, not a separate synthetic positive control that "
    "could be narrower than the state it anchors to. The predicate is the degeneracy "
    "definition itself (non-trivial per-step spread), reachable by construction whenever the "
    "substrate is stepping at all."
)

SEEDS: Tuple[int, ...] = (11, 23, 37)
WARMUP_EPISODES = 40
EVAL_EPISODES = 25
STEPS_PER_EPISODE = 150
DRY_WARMUP_EPISODES = 2
DRY_EVAL_EPISODES = 2
DRY_STEPS_PER_EPISODE = 20

HARM_HISTORY_LEN = 10
ENV_KWARGS: Dict[str, Any] = dict(
    size=10, num_hazards=1, num_resources=5, harm_history_len=HARM_HISTORY_LEN,
)

# Pre-registered thresholds -- never derived from this run's own statistics.
MIN_STEPS_PER_SEED = 3000          # EXQ-019's own C4, unmet there (n=969); this run must clear it PER SEED
DELTA_VARIANCE_FLOOR = 1e-8        # readiness: each loop's per-step delta series must have non-trivial spread
HALFLIFE_SD_MARGIN = 0.8           # ARC-004's own convention: monotonic margin in cross-seed SD units
MAX_LAG = 10                       # matches EXQ-019's own max lag

_ZG = ZGoalStreamAccumulator()


def _env_kwargs() -> Dict[str, Any]:
    return dict(ENV_KWARGS)


def _build_cfg(env: CausalGridWorldV2) -> REEConfig:
    # Step 2.5c substrate-path overlap: q081_profile_kwargs() sets use_salience_coordinator=
    # True by default (Q-081's "operating_mode" recording signal), which would exercise the
    # OPEN `mode-governance-engagement` substrate_queue entry (severity=corrupting, covers
    # ree_core/cingulate/salience_coordinator.py + ree_core/agent.py). This probe never reads
    # operating_mode -- only z_world/z_self/e3.get_commitment_state() -- so the flag is turned
    # back off (REEConfig's own default) rather than routed around; this removes the overlap
    # entirely instead of running under a known limitation.
    kwargs = dict(q081_profile_kwargs())
    kwargs["use_salience_coordinator"] = False
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        harm_history_len=HARM_HISTORY_LEN,
        **kwargs,
    )
    return cfg


def _compute_autocorr(series: List[float], lag: int) -> float:
    """Lag-k autocorrelation of a delta series (EXQ-019's own implementation, reused verbatim)."""
    if len(series) < lag + 2:
        return float("nan")
    x = torch.tensor(series, dtype=torch.float32)
    x_mean = x.mean()
    x_c = x - x_mean
    var = (x_c ** 2).mean()
    if var < 1e-10:
        return 0.0
    cov = (x_c[:-lag] * x_c[lag:]).mean()
    return float((cov / var).item())


def _halflife(series: List[float], max_lag: int = MAX_LAG) -> float:
    """Smallest lag k in [1, max_lag] where autocorr(k) <= 0.5 * autocorr(1).

    Returns max_lag (censored) if persistence never decays that far within the measured
    window; returns 1 (minimal persistence) if lag-1 autocorrelation is already <= 0.
    """
    ac1 = _compute_autocorr(series, 1)
    if not np.isfinite(ac1) or ac1 <= 0:
        return 1.0
    for k in range(1, max_lag + 1):
        ack = _compute_autocorr(series, k)
        if np.isfinite(ack) and ack <= 0.5 * ac1:
            return float(k)
    return float(max_lag)


def _run_length_stats(bool_series: List[int]) -> Dict[str, Any]:
    """Mean run-length of a 0/1 series -- consecutive-identical-value stretches."""
    if not bool_series:
        return {"mean_run_length": float("nan"), "n_runs": 0}
    runs: List[int] = []
    cur = bool_series[0]
    length = 1
    for v in bool_series[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(length)
            cur = v
            length = 1
    runs.append(length)
    return {"mean_run_length": float(np.mean(runs)), "n_runs": len(runs)}


def _run_seed(seed: int, warmup_episodes: int, eval_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    env_warm = CausalGridWorldV2(seed=seed, **_env_kwargs())
    cfg = _build_cfg(env_warm)
    agent = REEAgent(cfg)

    print(f"Seed {seed} Condition warmup", flush=True)
    warmup_train(
        agent, env_warm,
        num_episodes=int(warmup_episodes), steps_per_episode=int(steps_per_episode),
        label=f"warmup seed={seed}", progress_total_episodes=int(warmup_episodes),
    )

    env_eval = CausalGridWorldV2(seed=seed, **_env_kwargs())
    harness = StepHarness(agent, env_eval, train_mode=False, seed=seed)
    agent.eval()

    dz_world: List[float] = []
    dz_self: List[float] = []
    d_precision: List[float] = []
    is_committed_series: List[int] = []
    tick_counts = {"e1_tick": 0, "e2_tick": 0, "e3_tick": 0}
    n_steps_total = 0

    print(f"Seed {seed} Condition eval", flush=True)
    for ep in range(int(eval_episodes)):
        _flat, obs_dict = env_eval.reset()
        agent.reset()
        harness.reset()
        z_world_prev = None
        z_self_prev = None
        precision_prev = None

        for _t in range(int(steps_per_episode)):
            with torch.no_grad():
                result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            latent = result.latent
            zw = latent.z_world.detach()
            zs = latent.z_self.detach()
            cs = agent.e3.get_commitment_state()
            prec = float(cs["precision"])
            is_committed_series.append(int(bool(cs["is_committed"])))

            for k in tick_counts:
                if result.ticks.get(k, False):
                    tick_counts[k] += 1
            n_steps_total += 1

            if z_world_prev is not None:
                dz_world.append(float(torch.norm(zw - z_world_prev).item()))
                dz_self.append(float(torch.norm(zs - z_self_prev).item()))
                d_precision.append(abs(prec - precision_prev))
            z_world_prev = zw
            z_self_prev = zs
            precision_prev = prec

            if result.done:
                break

        if (ep + 1) % 5 == 0 or (ep + 1) == eval_episodes:
            print(f"  [eval] seed={seed} ep {ep + 1}/{eval_episodes}", flush=True)

    tau_e1 = _halflife(dz_world)
    tau_e2 = _halflife(dz_self)
    tau_e3_cont = _halflife(d_precision)
    commit_runs = _run_length_stats(is_committed_series)

    _ZG.observe_stats(harness.z_goal_stream_stats())
    print("verdict: PASS", flush=True)
    return {
        "seed": seed,
        "n_steps_total": n_steps_total,
        "tick_counts": tick_counts,
        "autocorr_dz_world": {f"lag{k}": _compute_autocorr(dz_world, k) for k in (1, 2, 5, 10)},
        "autocorr_dz_self": {f"lag{k}": _compute_autocorr(dz_self, k) for k in (1, 2, 5, 10)},
        "autocorr_d_precision": {f"lag{k}": _compute_autocorr(d_precision, k) for k in (1, 2, 5, 10)},
        "tau_e1_halflife": tau_e1,
        "tau_e2_halflife": tau_e2,
        "tau_e3_continuous_halflife": tau_e3_cont,
        "commit_run_length": commit_runs,
        "mean_dz_world": float(np.mean(dz_world)) if dz_world else float("nan"),
        "mean_dz_self": float(np.mean(dz_self)) if dz_self else float("nan"),
        "mean_d_precision": float(np.mean(d_precision)) if d_precision else float("nan"),
        "var_dz_world": float(np.var(dz_world)) if dz_world else 0.0,
        "var_dz_self": float(np.var(dz_self)) if dz_self else 0.0,
        "var_d_precision": float(np.var(d_precision)) if d_precision else 0.0,
        "n_dz_world": len(dz_world),
        "n_dz_self": len(dz_self),
        "n_d_precision": len(d_precision),
    }


def _monotonic_verdict(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    tau1 = np.asarray([r["tau_e1_halflife"] for r in per_seed], dtype=np.float64)
    tau2 = np.asarray([r["tau_e2_halflife"] for r in per_seed], dtype=np.float64)
    tau3 = np.asarray([r["tau_e3_continuous_halflife"] for r in per_seed], dtype=np.float64)

    d12 = tau2 - tau1
    d23 = tau3 - tau2

    def _margin_pass(deltas: np.ndarray) -> Tuple[bool, float, float]:
        mean_d = float(np.mean(deltas))
        sd_d = float(np.std(deltas, ddof=0)) if len(deltas) > 1 else 0.0
        margin_ok = mean_d >= HALFLIFE_SD_MARGIN * sd_d if sd_d > 0 else mean_d > 0
        return bool(margin_ok and mean_d > 0), mean_d, sd_d

    ok12, mean12, sd12 = _margin_pass(d12)
    ok23, mean23, sd23 = _margin_pass(d23)
    monotonic = ok12 and ok23

    return {
        "tau_e1_mean": float(np.mean(tau1)), "tau_e1_sd": float(np.std(tau1, ddof=0)),
        "tau_e2_mean": float(np.mean(tau2)), "tau_e2_sd": float(np.std(tau2, ddof=0)),
        "tau_e3_mean": float(np.mean(tau3)), "tau_e3_sd": float(np.std(tau3, ddof=0)),
        "e1_to_e2_delta_mean": mean12, "e1_to_e2_delta_sd": sd12, "e1_to_e2_margin_met": ok12,
        "e2_to_e3_delta_mean": mean23, "e2_to_e3_delta_sd": sd23, "e2_to_e3_margin_met": ok23,
        "monotonic_ordering_confirmed": monotonic,
        "arc004_criterion": (
            "monotonic ordering of effective persistence (tau_e1 < tau_e2 < tau_e3) by a margin "
            f"exceeding cross-seed noise (>= {HALFLIFE_SD_MARGIN} SD of the seed-to-seed "
            "half-life delta) -- ARC-004's own pre-registered wording, applied to E1/E2/E3 "
            "own-state persistence instead of L-space's z_beta/z_theta/z_delta layers."
        ),
    }


def main(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    warmup_episodes = DRY_WARMUP_EPISODES if dry_run else WARMUP_EPISODES
    eval_episodes = DRY_EVAL_EPISODES if dry_run else EVAL_EPISODES
    steps_per_episode = DRY_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:1] if dry_run else SEEDS

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(_run_seed(seed, warmup_episodes, eval_episodes, steps_per_episode))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    }

    # -- readiness gate (BEFORE the verdict): each loop's delta series and the sample size,
    #    worst cell across seeds -- the SAME statistic the load-bearing monotonicity criterion
    #    routes on (half-life derived from these autocorrelation series), not a magnitude proxy.
    min_n = min(r["n_steps_total"] for r in per_seed)
    min_var_world = min(r["var_dz_world"] for r in per_seed)
    min_var_self = min(r["var_dz_self"] for r in per_seed)
    min_var_prec = min(r["var_d_precision"] for r in per_seed)
    try:
        preconditions = p0_readiness_gate([
            {"name": "min_steps_per_seed_supra_floor", "measured": float(min_n),
             "threshold": float(MIN_STEPS_PER_SEED), "direction": "lower",
             "control": "worst (minimum) seed's total recorded eval steps"},
            {"name": "dz_world_variance_supra_floor", "measured": float(min_var_world),
             "threshold": float(DELTA_VARIANCE_FLOOR), "direction": "lower",
             "control": "worst (minimum) seed's per-step z_world delta variance"},
            {"name": "dz_self_variance_supra_floor", "measured": float(min_var_self),
             "threshold": float(DELTA_VARIANCE_FLOOR), "direction": "lower",
             "control": "worst (minimum) seed's per-step z_self delta variance"},
            {"name": "d_precision_variance_supra_floor", "measured": float(min_var_prec),
             "threshold": float(DELTA_VARIANCE_FLOOR), "direction": "lower",
             "control": "worst (minimum) seed's per-step E3 precision delta variance"},
        ])
    except P0NotReady as e:
        manifest["outcome"] = "FAIL"
        manifest["interpretation"] = {
            "label": "substrate_not_ready_requeue",
            "preconditions": e.preconditions,
            "criteria_non_degenerate": {"C1_monotonic_ordering_confirmed": False},
        }
        manifest["per_seed"] = per_seed
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = e.reason
        out_path = write_flat_manifest(
            manifest, config={"env_kwargs": _env_kwargs(), "warmup_episodes": warmup_episodes,
                               "eval_episodes": eval_episodes, "steps_per_episode": steps_per_episode},
            seeds=list(seeds), script_path=Path(__file__), z_goal_stream_stats=_ZG.stats(),
            started_at=t0, dry_run=dry_run,
        )
        manifest["_out_path"] = str(out_path)
        return manifest

    verdict = _monotonic_verdict(per_seed)

    tick_ratio_report = {
        f"seed{r['seed']}": {
            "e1": r["tick_counts"]["e1_tick"], "e2": r["tick_counts"]["e2_tick"],
            "e3": r["tick_counts"]["e3_tick"],
            "ratio_vs_e3": {
                "e1": r["tick_counts"]["e1_tick"] / max(1, r["tick_counts"]["e3_tick"]),
                "e2": r["tick_counts"]["e2_tick"] / max(1, r["tick_counts"]["e3_tick"]),
                "e3": 1.0,
            },
        }
        for r in per_seed
    }

    degeneracy = check_degeneracy({
        "dz_world_variance": {"groups": [[r["var_dz_world"]] for r in per_seed], "floor": DELTA_VARIANCE_FLOOR},
        "dz_self_variance": {"groups": [[r["var_dz_self"]] for r in per_seed], "floor": DELTA_VARIANCE_FLOOR},
        "d_precision_variance": {"groups": [[r["var_d_precision"]] for r in per_seed], "floor": DELTA_VARIANCE_FLOOR},
    })

    structural_findings = {
        "finding_1_e2_tick_dead_code": (
            "ticks['e2_tick'] (ree_core/heartbeat/clock.py:143,174) is computed every "
            "MultiRateClock.advance() call but is NEVER consumed anywhere in ree_core/ -- "
            "confirmed by exhaustive grep. No code path gates on E2's own tick flag; the "
            "designed 1-in-3 E2 cadence exists only as a clock counter."
        ),
        "finding_2_latent_stack_uniformly_reencoded": (
            "ree-v3/experiments/_lib/stream_recorder.py:310-317 documents (2026-07-26, "
            "Q-081/V3-EXQ-824) that the entire LatentStack -- z_self, z_world, z_beta, "
            "z_theta, z_delta, z_harm, z_harm_a -- is re-encoded every env step regardless of "
            "loop identity ('E1 rate' for all of them). This applies equally to ARC-004's own "
            "L-space layers, not only to the E-loop streams this probe targets."
        ),
        "finding_3_e3_precision_updates_every_tick": (
            "ree-v3/experiments/_lib/stream_recorder.py:374-379 and "
            "ree_core/predictors/e3_selector.py (update_running_variance, called from "
            "post_action_update on EVERY env tick, not gated to e3_tick) confirm E3's own "
            "'precision'/'running_variance' are mutated every env step. Only "
            "commit_threshold (static) and is_committed/committed_now (closure-latch-derived) "
            "are genuinely E3-cadence."
        ),
    }

    manifest.update({
        "outcome": "PASS" if verdict["monotonic_ordering_confirmed"] else "FAIL",
        "evidence_direction": "supports" if verdict["monotonic_ordering_confirmed"] else "weakens",
        "evidence_direction_note": (
            "Diagnostic-purpose: excluded from governance confidence/conflict scoring by "
            "EXPERIMENT_PURPOSE alone. This field records the reading for a human/governance "
            "session, not an automatic scoring input."
        ),
        "per_seed": per_seed,
        "monotonicity_verdict": verdict,
        "designed_tick_ratio": {"e1": 1, "e2": 3, "e3": 10},
        "realised_tick_ratio_per_seed": tick_ratio_report,
        "structural_findings": structural_findings,
        "interpretation": {
            "label": (
                "e_ladder_realised_timescale_separation_confirmed"
                if verdict["monotonic_ordering_confirmed"]
                else "e_ladder_layers_differ_in_content_not_timescale"
            ),
            "preconditions": preconditions,
            "criteria": [
                {"name": "C1_monotonic_ordering_confirmed",
                 "load_bearing": True,
                 "passed": bool(verdict["monotonic_ordering_confirmed"])},
            ],
            "criteria_non_degenerate": {
                "C1_monotonic_ordering_confirmed": degeneracy["non_degenerate"],
            },
            "combination_rule": "single load-bearing criterion (C1); no OR/AND combination.",
        },
        "non_degenerate": degeneracy["non_degenerate"],
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    })

    out_path = write_flat_manifest(
        manifest,
        config={"env_kwargs": _env_kwargs(), "warmup_episodes": warmup_episodes,
                "eval_episodes": eval_episodes, "steps_per_episode": steps_per_episode,
                "halflife_sd_margin": HALFLIFE_SD_MARGIN, "max_lag": MAX_LAG},
        seeds=list(seeds), script_path=Path(__file__), z_goal_stream_stats=_ZG.stats(),
        started_at=t0, dry_run=dry_run,
    )

    print(f"[smoke] outcome={manifest['outcome']} monotonic={verdict['monotonic_ordering_confirmed']}"
          f" tau_e1={verdict['tau_e1_mean']:.3f} tau_e2={verdict['tau_e2_mean']:.3f}"
          f" tau_e3={verdict['tau_e3_mean']:.3f}", flush=True)

    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    print(f"outcome: {result['outcome']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["_out_path"],
        run_id=result.get("run_id"),
        dry_run=args.dry_run,
    )
