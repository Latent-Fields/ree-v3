"""
V3-EXQ-591d -- InfantCurriculumScheduler Phase 0->1 gate-robustness diagnostic
(ARC-046 / infant_substrate:GAP-14).

SUBSTRATE-READINESS / GATE-DISCRIMINATION DIAGNOSTIC. NOT the full curriculum-vs-flat
comparison (EXQ-ISEF-005); that successor stays blocked until the Phase 0->1 gate is
robust on every seed.

CONTEXT
-------
The user-adjudicated failure_autopsy_V3-EXQ-591c_2026-06-11 surfaced TWO orthogonal
Phase 0->1 defects under the landed single-episode-crossing gate
(H_POS_FRAC_OF_MAX=0.20, threshold 0.20*ln(144) ~= 0.994):
  (1) seed-46 NEVER escapes Phase 0 (h_pos_mean=0.0375) -- an exploration-STRENGTH
      gap owned by the Q-043/667->667a modulatory-bias-selection-authority thread.
      OUT OF SCOPE here; seed 46 is EXPECTED to stay in Phase 0 under every criterion.
  (2) seed-45 advanced to Phase 1 with a near-stationary policy (h_pos_mean=0.140 <
      0.20, only 2 eligible-episode crossings) because `_try_phase_0_to_1`
      (experiments/infant_curriculum.py) advances on a SINGLE episode crossing the
      threshold. THIS experiment tests defect (2): does a K-of-N or EMA crossing
      criterion REJECT the seed-45-like false-advancer while still ADMITTING the
      genuine explorers (seeds 42-44)?

SLEEP DRIVER: K=never (SleepLoopManager K > total episodes; never fires during this
readiness probe -- inherited from the 591/591b/591c reachability builder).

DESIGN (pure-analysis; the shared infant_curriculum.py scheduler is NOT mutated --
610e/669/586/667 and others depend on its default Phase 0->1 behaviour)
-----------------------------------------------------------------------------------
Re-run the SAME diversity-armed reachability probe as
v3_exq_591c_isef005_curriculum_phase_advance_readiness_diversity_v3.py (5 seeds 42-46,
160 ep, 200 steps/ep, grid 12, MECH-313 noise floor + MECH-314 curiosity ON at landed
defaults, SP-CEM main-path default-on, H_POS_FRAC_OF_MAX=0.20) and RECORD the full
per-episode h_pos (pos_entropy) sequence for every seed. The run is fully RNG-seeded
(torch.manual_seed(seed); env seed = seed*n_ep+ep), so it reproduces the 591c traces
deterministically -- it doubles as a reproducibility check of the seed-45/seed-46
profiles. Then replay each seed's h_pos sequence OFFLINE through three candidate
Phase 0->1 criteria, all gated by the same PHASE_EP_MIN[1]=100 episode minimum:
  (A) BASELINE single-episode crossing (current scheduler behaviour).
  (B) K-of-N: advance only when >= K of the last N episodes cleared the threshold
      (K=5, N=10 -- a sustained-exploration bar; a single fluke episode no longer
      advances a near-stationary policy).
  (C) EMA: advance only when an exponential-moving-average of h_pos (alpha=0.2,
      ~10-episode window) stays above the threshold.

INTERPRETATION GRID (also in experiment.md)
-------------------------------------------
- "genuine explorer" seed: h_pos_mean >= 0.20 AND cleared the threshold on >= 2
  post-ep_min episodes (the 591c seeds 42-44 profile).
- "false advancer" seed: advanced under the BASELINE single-episode gate but is NOT a
  genuine explorer (high h_pos_max, low h_pos_mean, < 2 crossings -- the seed-45
  profile).
- PRIMARY (load-bearing) C_gate_discriminates: at least one of {K-of-N, EMA} REJECTS
  every false-advancer AND ADMITS every genuine explorer -> that criterion is the
  recommended replacement; routes to an /implement-substrate change to
  infant_curriculum.py `_try_phase_0_to_1`. (Seed 46 staying in Phase 0 under all
  criteria does NOT count against any criterion -- that is the orthogonal
  exploration-strength defect, the 667a thread.)
- If NO candidate both rejects the false-advancer and admits the explorers ->
  needs_alternative_criterion (report which leg failed; do NOT force a gate change).

NON-VACUITY PRECONDITIONS (interpretation.preconditions[]; same shape as 591c)
-----------------------------------------------------------------------------
- early_policy_produces_nontrivial_h_pos: max per-episode h_pos across all seeds >= a
  movement floor (the agent-moves positive control; SAME statistic the gate routes
  on). Below floor -> degenerate non-moving artifact -> substrate_not_ready_requeue,
  NOT a discrimination verdict.
- genuine_explorers_present: >= 2 seeds are genuine explorers, so the "admits
  explorers" leg is non-vacuous. Below -> no explorer signal to discriminate against
  -> substrate_not_ready_requeue.
- false_advancer_present: >= 1 seed is a false advancer, so the "rejects
  false-advancer" leg is non-vacuous. Below -> the single-episode gate was NOT
  over-permissive on this seed draw (report; do NOT record a discrimination PASS).

ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from infant_curriculum import (  # noqa: E402
    InfantCurriculumScheduler,
    H_POS_FRAC_OF_MAX,
    PHASE_EP_MIN,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# Reuse the canonical 591 helpers + constants (DRY -- no copy-drift).
from v3_exq_591_isef005_curriculum_vs_flat_v3 import (  # noqa: E402
    _extract_obs,
    BODY_OBS_DIM,
    WORLD_OBS_DIM,
    GRID_SIZE,
    ACTION_DIM,
)

QUEUE_ID = "V3-EXQ-591d"
EXPERIMENT_TYPE = "v3_exq_591d_isef005_phase01_gate_robustness"
CLAIM_IDS: List[str] = []  # gate-robustness substrate diagnostic -- weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
N_EPISODES = 160          # faithful to V3-EXQ-591b/591c
STEPS_PER_EPISODE = 200   # faithful to V3-EXQ-591b/591c

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # 0.20 * ln(144) ~= 0.994
PHASE_01_EP_MIN = PHASE_EP_MIN[1]                # 100 (hard episode-count minimum)

# Readiness floor: the early-policy agent must actually MOVE (non-trivial per-episode
# H_pos) for the discrimination result to be a real verdict. SAME statistic the gate
# routes on (per-episode pos_entropy).
H_POS_MOVEMENT_FLOOR = 0.20
# Genuine-exploration floor: a seed explores genuinely iff its mean per-episode H_pos
# clears this AND it cleared the gate threshold on more than a single eligible episode
# (591c seeds 42-44 mean band 0.32-0.84). A near-stationary fluke (seed 46 h_pos_mean
# 0.0375) reads well below it.
GENUINE_EXPLORATION_H_POS_MEAN_FLOOR = 0.20
GENUINE_EXPLORATION_MIN_CROSSINGS = 2

# Candidate robust-criterion parameters (documented, pre-registered).
KOFN_K = 5    # >= 5 of the last N episodes must clear the threshold
KOFN_N = 10   # rolling window length
EMA_ALPHA = 0.2  # ~10-episode EMA half-life


def _build_diversity_agent() -> REEAgent:
    """Mirror of the 591c diversity-armed agent build: MECH-313 noise floor +
    MECH-314 structured curiosity ON at their landed default magnitudes (SP-CEM is
    already the main-path default). Bit-identical to the 591/591b/591c agent build
    otherwise."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        novelty_bonus_weight=0.5,
        use_sleep_loop=True,
        sleep_loop_episodes_K=N_EPISODES + 1,  # K=never (> total episodes)
        # --- exploration-diversity stack (matches 591c) ---
        use_noise_floor=True,                  # MECH-313 (defaults: alpha=0.1, min_T=1.0)
        use_structured_curiosity=True,         # MECH-314 (defaults: sub-flavours on, w=0.05)
        # use_support_preserving_cem defaults True (SP-CEM main-path; left implicit)
    )
    cfg.latent.alpha_world = 0.9
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _run_seed(*, seed: int, n_episodes: int) -> Dict[str, Any]:
    """Run the diversity-armed reachability probe for one seed and return the FULL
    per-episode h_pos sequence (plus the baseline-gate summary stats)."""
    torch.manual_seed(seed)
    agent = _build_diversity_agent()
    sched = InfantCurriculumScheduler(grid_size=GRID_SIZE)

    h_pos_window: deque = deque(maxlen=100)  # rolling (informational only)
    per_ep_h_pos: List[float] = []
    baseline_phase_01_at: Optional[int] = None

    for ep in range(n_episodes):
        env_kwargs = sched.env_kwargs()  # Phase 0 -> all infant features OFF
        agent.config.e3.novelty_bonus_weight = float(
            sched.config_overrides().get("novelty_bonus_weight", 0.5))

        env = CausalGridWorldV2(
            size=GRID_SIZE,
            seed=seed * n_episodes + ep,
            resource_respawn_on_consume=True,
            pos_telemetry_enabled=True,
            traj_telemetry_enabled=True,
        )
        _flat, obs_dict = env.reset()
        ob, ow = _extract_obs(obs_dict)

        ep_h_pos = -1.0
        ep_benefit_contacts = 0

        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(obs_body=ob, obs_world=ow)
            ai = int(action.argmax().item()) % ACTION_DIM
            _o, harm_signal, done, info, obs_dict = env.step(ai)
            agent.update_residue(float(harm_signal))
            ob, ow = _extract_obs(obs_dict)
            benefit = float(ob[11].item()) if ob.shape[0] > 11 else 0.0
            energy = float(ob[3].item()) if ob.shape[0] > 3 else 0.5
            drive = max(0.0, min(1.0, 1.0 - energy))
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
            ep_h_pos = float(info.get("pos_entropy", -1.0))
            ep_benefit_contacts += int(
                float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0)
            if done:
                _flat, obs_dict = env.reset()
                ob, ow = _extract_obs(obs_dict)

        z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
        cov = float(agent.residue_field.get_coverage_telemetry()["residue_coverage_pct"])

        per_ep_h_pos.append(ep_h_pos)
        h_pos_window.append(ep_h_pos)

        prev_phase = sched.current_phase
        sched.update(
            ep,
            h_pos=ep_h_pos if ep_h_pos >= 0.0 else None,
            z_goal_norm=z_norm,
            benefit_contacts=ep_benefit_contacts,
            residue_coverage_pct=cov,
        )
        if prev_phase == 0 and sched.current_phase >= 1 and baseline_phase_01_at is None:
            baseline_phase_01_at = ep

        if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] gate-robustness seed={seed} ep {ep + 1}/{n_episodes}"
                f" phase={sched.current_phase} h_pos={ep_h_pos:.3f}",
                flush=True,
            )

    # Baseline summary stats (clip invalid -1 readings out of the mean / band).
    valid = [h for h in per_ep_h_pos if h >= 0.0]
    eligible = [h for h in per_ep_h_pos[PHASE_01_EP_MIN:] if h >= 0.0]
    n_eligible_ge_threshold = sum(1 for h in eligible if h >= PHASE_01_THRESHOLD)
    h_pos_mean = (sum(valid) / len(valid)) if valid else -1.0
    h_pos_max = max(valid) if valid else -1.0
    reached_phase1_baseline = baseline_phase_01_at is not None

    print(f"verdict: {'PASS' if reached_phase1_baseline else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        # full sequence for offline criteria replay (negatives clipped to 0.0 so an
        # invalid reading never spuriously clears the positive threshold)
        "h_pos_sequence": [max(0.0, h) for h in per_ep_h_pos],
        "baseline_phase_01_advanced_at_episode": baseline_phase_01_at,
        "reached_phase1_baseline": reached_phase1_baseline,
        "h_pos_mean": round(h_pos_mean, 4) if valid else -1.0,
        "h_pos_max": round(h_pos_max, 4) if valid else -1.0,
        "n_eligible_episodes": len(eligible),
        "n_eligible_ge_threshold": n_eligible_ge_threshold,
    }


# ----------------------------------------------------------------------------------
# Offline Phase 0->1 criterion replay (pure functions over a per-episode h_pos seq).
# ----------------------------------------------------------------------------------

def _advance_single_episode(
    seq: List[float], threshold: float, ep_min: int
) -> Tuple[bool, Optional[int]]:
    for ep in range(ep_min, len(seq)):
        if seq[ep] >= threshold:
            return True, ep
    return False, None


def _advance_k_of_n(
    seq: List[float], threshold: float, ep_min: int, k: int, n: int
) -> Tuple[bool, Optional[int]]:
    for ep in range(ep_min, len(seq)):
        lo = max(0, ep - n + 1)
        n_cleared = sum(1 for h in seq[lo:ep + 1] if h >= threshold)
        if n_cleared >= k:
            return True, ep
    return False, None


def _advance_ema(
    seq: List[float], threshold: float, ep_min: int, alpha: float
) -> Tuple[bool, Optional[int]]:
    if not seq:
        return False, None
    ema = seq[0]
    for ep in range(1, len(seq)):
        ema = (1.0 - alpha) * ema + alpha * seq[ep]
        if ep >= ep_min and ema >= threshold:
            return True, ep
    return False, None


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    n_episodes = 4 if dry_run else N_EPISODES
    print(
        f"V3-EXQ-591d gate-robustness: seeds={seeds} n_episodes={n_episodes}"
        f" steps={STEPS_PER_EPISODE} threshold={PHASE_01_THRESHOLD:.4f}"
        f" ep_min={PHASE_01_EP_MIN} [K-of-N={KOFN_K}/{KOFN_N} EMA_alpha={EMA_ALPHA}]"
        f" [diversity stack ON]",
        flush=True,
    )
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition gate-robustness", flush=True)
        seed_results.append(_run_seed(seed=seed, n_episodes=n_episodes))

    # --- offline criterion replay + classification ---
    ep_min = PHASE_01_EP_MIN if not dry_run else 0  # dry-run uses ep_min 0 so it exercises
    per_seed_criteria: List[Dict[str, Any]] = []
    n_genuine = 0
    n_false = 0
    for r in seed_results:
        seq = r["h_pos_sequence"]
        adv_base, at_base = _advance_single_episode(seq, PHASE_01_THRESHOLD, ep_min)
        adv_kofn, at_kofn = _advance_k_of_n(seq, PHASE_01_THRESHOLD, ep_min, KOFN_K, KOFN_N)
        adv_ema, at_ema = _advance_ema(seq, PHASE_01_THRESHOLD, ep_min, EMA_ALPHA)

        genuine = bool(
            r["h_pos_mean"] >= GENUINE_EXPLORATION_H_POS_MEAN_FLOOR
            and r["n_eligible_ge_threshold"] >= GENUINE_EXPLORATION_MIN_CROSSINGS
        )
        # False advancer: the single-episode gate admitted it, but it does not explore
        # genuinely (the seed-45 profile).
        false_advancer = bool(adv_base and not genuine)
        if genuine:
            n_genuine += 1
        if false_advancer:
            n_false += 1

        per_seed_criteria.append({
            "seed": r["seed"],
            "h_pos_mean": r["h_pos_mean"],
            "h_pos_max": r["h_pos_max"],
            "n_eligible_ge_threshold": r["n_eligible_ge_threshold"],
            "genuine_explorer": genuine,
            "false_advancer": false_advancer,
            "baseline_single_episode": {"advanced": adv_base, "at_episode": at_base},
            "k_of_n": {"advanced": adv_kofn, "at_episode": at_kofn},
            "ema": {"advanced": adv_ema, "at_episode": at_ema},
        })

    genuine_seeds = [c["seed"] for c in per_seed_criteria if c["genuine_explorer"]]
    false_seeds = [c["seed"] for c in per_seed_criteria if c["false_advancer"]]

    def _discriminates(criterion_key: str) -> bool:
        # rejects EVERY false-advancer AND admits EVERY genuine explorer
        admits_genuine = all(
            c[criterion_key]["advanced"] for c in per_seed_criteria if c["genuine_explorer"]
        )
        rejects_false = all(
            not c[criterion_key]["advanced"] for c in per_seed_criteria if c["false_advancer"]
        )
        return bool(admits_genuine and rejects_false)

    kofn_discriminates = _discriminates("k_of_n")
    ema_discriminates = _discriminates("ema")
    discriminating = [
        name for name, ok in (("k_of_n", kofn_discriminates), ("ema", ema_discriminates)) if ok
    ]

    max_h_pos = max((r["h_pos_max"] for r in seed_results), default=-1.0)
    movement_ok = max_h_pos >= H_POS_MOVEMENT_FLOOR
    genuine_ok = n_genuine >= 2
    false_present = n_false >= 1

    # --- routing ---
    if not movement_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif not genuine_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"  # no genuine-explorer signal to admit
    elif not false_present:
        # The single-episode gate was NOT over-permissive on this seed draw -- there is
        # no false-advancer to reject. Not a substrate-readiness failure; surface it.
        outcome = "FAIL"
        label = "no_false_advancer_reproduced_gate_not_overpermissive_on_this_draw"
    elif discriminating:
        outcome = "PASS"
        label = f"robust_criterion_discriminates_{'_and_'.join(discriminating)}"
    else:
        outcome = "FAIL"
        label = "no_candidate_criterion_discriminates_needs_alternative"

    return {
        "outcome": outcome,
        "label": label,
        "seed_results": seed_results,
        "per_seed_criteria": per_seed_criteria,
        "n_genuine_explorers": n_genuine,
        "n_false_advancers": n_false,
        "genuine_explorer_seeds": genuine_seeds,
        "false_advancer_seeds": false_seeds,
        "kofn_discriminates": kofn_discriminates,
        "ema_discriminates": ema_discriminates,
        "discriminating_criteria": discriminating,
        "movement_ok": movement_ok,
        "max_h_pos": max_h_pos,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = out_dir / f"{run_id}.json"

    seeds_used = list(SEEDS if not dry_run else [SEEDS[0]])
    discriminates_any = bool(result["discriminating_criteria"])

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "sleep_driver_pattern": "K=never (SleepLoopManager K > total episodes; never fires)",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": seeds_used,
            "n_episodes": N_EPISODES if not dry_run else 4,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
            "phase_0to1_threshold": round(PHASE_01_THRESHOLD, 6),
            "phase_0to1_ep_min": PHASE_01_EP_MIN,
            "h_pos_movement_floor": H_POS_MOVEMENT_FLOOR,
            "genuine_exploration_h_pos_mean_floor": GENUINE_EXPLORATION_H_POS_MEAN_FLOOR,
            "genuine_exploration_min_crossings": GENUINE_EXPLORATION_MIN_CROSSINGS,
            "kofn_k": KOFN_K,
            "kofn_n": KOFN_N,
            "ema_alpha": EMA_ALPHA,
            "arm": "InfantCurriculumScheduler (experiments/infant_curriculum.py) -- OFFLINE criterion replay, scheduler NOT mutated",
            "diversity_stack": {
                "use_noise_floor": True,
                "use_structured_curiosity": True,
                "use_support_preserving_cem": "default-on (main-path since 2026-05-17)",
                "magnitudes": "landed defaults (noise_floor_alpha=0.1, curiosity_weight=0.05)",
            },
        },
        "acceptance_criteria": {
            "C_gate_discriminates_primary": (
                "at least one of {K-of-N, EMA} REJECTS every false-advancer (seed-45-like:"
                f" baseline-admitted but h_pos_mean < {GENUINE_EXPLORATION_H_POS_MEAN_FLOOR}"
                f" OR < {GENUINE_EXPLORATION_MIN_CROSSINGS} crossings) AND ADMITS every"
                " genuine explorer (seeds 42-44 profile). That criterion is the recommended"
                " replacement for the single-episode Phase 0->1 gate (routes to"
                " /implement-substrate on infant_curriculum.py _try_phase_0_to_1)."
            ),
            "C_seed46_out_of_scope": (
                "seed 46 staying in Phase 0 under ALL criteria does NOT count against any"
                " criterion -- that is the orthogonal exploration-strength defect owned by"
                " the Q-043/667->667a modulatory-bias-selection-authority thread, NOT a"
                " gate-criterion defect."
            ),
        },
        "interpretation": {
            "label": result["label"],
            "preconditions": [
                {
                    "name": "early_policy_produces_nontrivial_h_pos",
                    "description": (
                        "Early-policy agent must actually explore (max per-episode H_pos"
                        " clears a movement floor) for the discrimination result to be a"
                        " real verdict rather than a degenerate non-moving artifact. SAME"
                        " statistic the gate routes on (per-episode pos_entropy)."
                    ),
                    "measured": round(result["max_h_pos"], 4),
                    "threshold": H_POS_MOVEMENT_FLOOR,
                    "direction": "lower",
                    "control": "max per-episode H_pos across all seeds (the agent-moves positive control)",
                    "met": bool(result["movement_ok"]),
                },
                {
                    "name": "genuine_explorers_present",
                    "description": (
                        "At least 2 seeds explore genuinely (h_pos_mean >= floor AND >= 2"
                        " threshold crossings), so the 'admits explorers' discrimination"
                        " leg is non-vacuous. Below -> no explorer signal to discriminate"
                        " against -> substrate_not_ready_requeue. SAME genuine-explorer"
                        " statistic the C_gate_discriminates criterion routes on."
                    ),
                    "measured": result["n_genuine_explorers"],
                    "threshold": 2,
                    "direction": "lower",
                    "control": "count of genuine-explorer seeds (591c: seeds 42-44)",
                    "met": bool(result["n_genuine_explorers"] >= 2),
                },
                {
                    "name": "false_advancer_present",
                    "description": (
                        "At least 1 seed is a false advancer (baseline single-episode gate"
                        " admitted a non-explorer), so the 'rejects false-advancer' leg is"
                        " non-vacuous. Below -> the single-episode gate was NOT"
                        " over-permissive on this seed draw (report; do NOT record a"
                        " discrimination PASS). SAME false-advancer statistic the"
                        " C_gate_discriminates criterion routes on."
                    ),
                    "measured": result["n_false_advancers"],
                    "threshold": 1,
                    "direction": "lower",
                    "control": "count of false-advancer seeds (591c: seed 45)",
                    "met": bool(result["n_false_advancers"] >= 1),
                },
            ],
            "criteria_non_degenerate": {
                # Discrimination is meaningful only if the seed draw contains BOTH a
                # genuine explorer (to admit) AND a false advancer (to reject).
                "C_gate_discriminates": bool(
                    result["n_genuine_explorers"] >= 2 and result["n_false_advancers"] >= 1
                ),
            },
            "criteria": [
                {
                    "name": "C_gate_discriminates",
                    "load_bearing": True,
                    "passed": discriminates_any,
                },
            ],
        },
        "metrics": {
            "n_genuine_explorers": result["n_genuine_explorers"],
            "n_false_advancers": result["n_false_advancers"],
            "genuine_explorer_seeds": result["genuine_explorer_seeds"],
            "false_advancer_seeds": result["false_advancer_seeds"],
            "kofn_discriminates": result["kofn_discriminates"],
            "ema_discriminates": result["ema_discriminates"],
            "discriminating_criteria": result["discriminating_criteria"],
            "max_h_pos": round(result["max_h_pos"], 4),
            "n_seeds_total": len(seeds_used),
        },
        "per_seed_criteria": result["per_seed_criteria"],
        "notes": (
            "Gate-robustness diagnostic for ARC-046 / infant_substrate:GAP-14. Re-runs the"
            " V3-EXQ-591c diversity-armed Phase 0->1 reachability probe (deterministic; a"
            " reproducibility check of the 591c seed traces), records the full per-episode"
            " h_pos sequence per seed, and replays each through BASELINE single-episode /"
            " K-of-N / EMA Phase 0->1 criteria OFFLINE -- the shared infant_curriculum.py"
            " scheduler is NOT mutated (610e/669/586/667 depend on its default gate). Tests"
            " the user-adjudicated failure_autopsy_V3-EXQ-591c_2026-06-11 seed-45"
            " over-permissiveness defect ONLY: does a robust criterion reject the"
            " near-stationary false-advancer while admitting the genuine explorers? The"
            " orthogonal seed-46 exploration-STRENGTH collapse (Q-043/667->667a"
            " modulatory-bias-selection-authority thread) is OUT OF SCOPE -- seed 46 is"
            " expected to stay in Phase 0 under every criterion. A discriminating criterion"
            " (PASS) routes to /implement-substrate as a K-of-N/EMA replacement for"
            " _try_phase_0_to_1; only after BOTH this gate fix AND the exploration-strength"
            " fix land can the full curriculum-vs-flat EXQ-ISEF-005 (V3-EXQ-591 successor)"
            " be queued."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        summary = {k: v for k, v in manifest.items() if k not in ("per_seed_criteria",)}
        print(json.dumps(summary, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
