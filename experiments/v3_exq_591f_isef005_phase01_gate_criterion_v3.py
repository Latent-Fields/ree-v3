"""
V3-EXQ-591f -- InfantCurriculumScheduler Phase 0->1 gate-CRITERION diagnostic
(ARC-046 / infant_substrate:GAP-14). Supersedes V3-EXQ-591e (which superseded 591d).

SUBSTRATE-READINESS / GATE-DISCRIMINATION DIAGNOSTIC. NOT the full curriculum-vs-flat
comparison (EXQ-ISEF-005); that successor stays blocked until BOTH GAP-14 legs resolve:
(c-2) gate-criterion [this 591f] AND (c-1) seed-46 exploration-strength [Q-043/667 thread].

WHY 591f SUPERSEDES 591e
------------------------
591e replayed a SINGLE corrected candidate -- EMA(alpha=0.2) of the per-episode h_pos
LEVEL vs the ~0.20 genuine-explorer floor -- on the autopsy's reasoning that "the EMA
converges toward the per-episode mean (seed45 0.14 < 0.20), so seed 45 is rejected." That
reasoning is NUMERICALLY UNSAFE for alpha=0.2: an EMA(alpha=0.2) has a ~5-episode effective
window and is spike-responsive, so a SINGLE per-episode h_pos of 1.453 (which seed 45 HAS,
and the baseline single-episode gate advanced 45 at episode 142 = inside the decision
window) drives the EMA from a ~0.1 baseline to 0.8*0.1 + 0.2*1.453 ~= 0.37, ABOVE the 0.20
floor -- so plain EMA-of-level@0.20 likely RE-ADMITS seed 45 on its spike. (Confirmed in an
offline unit-check on synthetic 591d-shaped traces: EMA-of-level advances a seed-45-like
sequence; window-mean / EMA-with-hold / crossing-count reject it.) 591f therefore replays a
SWEEP of sustained-level candidates and PASSes if ANY cleanly discriminates -- the
robust-form sweep the user chose via AskUserQuestion 2026-06-15, overriding the prior
EMA-of-level-only route.

CONTEXT (the 591d failure this corrects)
----------------------------------------
The user-adjudicated failure_autopsy_V3-EXQ-591d_2026-06-15 (Section 8) established: 591d
FAILed because both candidate robust criteria (K-of-N 5/10, EMA-of-level vs 0.994) reused
the single-episode SPIKE threshold (0.994 = H_POS_FRAC_OF_MAX * ln(grid^2) = 0.20 * ln(144))
as the bar for a SUSTAINED statistic. They correctly REJECT the seed-45 false-advancer (the
591c over-permissiveness IS fixed) but ALSO reject genuine explorers 42/43, whose mean h_pos
LEVEL (seed42 0.562, seed43 0.323, seed44 0.842) sits BELOW the spike bar -- they spike
supra-0.994 only 6-7x of 160 episodes; only seed44's 36 dense crossings clears K-of-N/EMA-
vs-0.994. The false-advancer seed45 has mean 0.140 but a transient h_pos_max of 1.453 (it
crosses the 0.994 spike bar exactly twice). The discriminating signal is the SUSTAINED
LEVEL -- exactly the genuine-explorer ORACLE's own statistic
(GENUINE_EXPLORATION_H_POS_MEAN_FLOOR ~= 0.20), NOT threshold-crossing density.

THE FIX (robust sustained-level SWEEP; user-confirmed AskUserQuestion 2026-06-15)
--------------------------------------------------------------------------------
Replay the SAME 5-seed (42-46), 160-ep diversity-armed reachability traces through a SWEEP
of sustained-level candidates (all gated at the genuine-explorer floor ~0.20, NOT the 0.994
spike bar) and PASS if ANY candidate cleanly admits 42/43/44 AND rejects 45 at the real
decision episodes. Plain EMA-of-level@floor is retained as the spike-vulnerable baseline
candidate; window-mean, EMA-with-hold, and crossing-count>=3 are the spike-robust forms.
Everything else is held identical to 591d/591e: same seeds, same ep_min=100, same diversity
stack (MECH-313 noise floor + MECH-314 curiosity at landed defaults, SP-CEM main-path), same
non-vacuity preconditions, same two-leg discrimination definition (rejects every false-
advancer AND admits every genuine explorer). The single-episode crossing of the 0.994 SPIKE
bar is RETAINED only as the informational BASELINE leg (the current scheduler behaviour +
the classifier that flags a false-advancer).

NOTE on offline validation from 591d: the 591d manifest persisted only per-seed SUMMARIES
(h_pos_mean, h_pos_max, n_eligible_ge_threshold), NOT the per-episode arrays -- so the exact
online behaviour of any sustained-level criterion at the decision episode (which depends on
the temporal ordering of episodes 100-160) could not be validated from 591d alone. 591f
PERSISTS the full per-episode h_pos arrays and replays the sweep over them, which is the
follow-on diagnostic the 591d data could not supply.

SLEEP DRIVER: K=never (SleepLoopManager K > total episodes; never fires during this
readiness probe -- inherited from the 591/591b/591c/591d/591e reachability builder).

DESIGN (pure-analysis; the shared infant_curriculum.py scheduler is NOT mutated --
610e/669/586/667 and others depend on its default Phase 0->1 behaviour)
-----------------------------------------------------------------------------------
Re-run the SAME diversity-armed reachability probe as 591d/591e (5 seeds 42-46, 160 ep,
200 steps/ep, grid 12, MECH-313 noise floor + MECH-314 curiosity ON at landed defaults,
SP-CEM main-path default-on, H_POS_FRAC_OF_MAX=0.20) and RECORD the full per-episode
h_pos (pos_entropy) sequence for every seed. The run is fully RNG-seeded
(torch.manual_seed(seed); env seed = seed*n_ep+ep), so it reproduces the 591c/591d traces
deterministically -- it doubles as a reproducibility check of the seed-45/seed-46 profiles.
Then replay each seed's h_pos sequence OFFLINE through the baseline + four sustained-level
candidates, all gated by PHASE_EP_MIN[1]=100:
  (A) BASELINE single-episode SPIKE crossing of PHASE_01_THRESHOLD (0.994) -- current
      scheduler behaviour; INFORMATIONAL ONLY (classifies false-advancers; NOT a candidate
      replacement).
  (B1) EMA-of-LEVEL: advance when EMA(alpha=0.2) of per-episode h_pos LEVEL >= the floor
       (~0.20). The 591e pick; expected spike-vulnerable on seed 45.
  (B2) WINDOW-MEAN: advance when the trailing-mean over the last W=20 episodes >= floor.
  (B3) EMA-of-LEVEL with HOLD: advance when the EMA(alpha=0.2) stays >= floor for H=5
       consecutive episodes (hysteresis against a single transient spike).
  (B4) CROSSING-COUNT: advance when the cumulative count of post-ep_min episodes crossing
       the 0.994 spike bar reaches CROSSING_COUNT_MIN=3 (seed 45 has only 2).

INTERPRETATION GRID (Phase 0->1; also in experiment.md)
-------------------------------------------------------
- "genuine explorer" seed: h_pos_mean >= GENUINE_EXPLORATION_H_POS_MEAN_FLOOR (0.20) AND
  cleared the 0.994 spike bar on >= GENUINE_EXPLORATION_MIN_CROSSINGS (2) post-ep_min
  episodes (the 591c/591d seeds 42-44 profile). The ORACLE; unchanged from 591d.
- "false advancer" seed: advanced under the BASELINE single-episode SPIKE gate but is NOT
  a genuine explorer (high h_pos_max, low h_pos_mean, < 2 crossings -- the seed-45 profile).
- PRIMARY (load-bearing) C_gate_discriminates: AT LEAST ONE swept candidate (B1-B4) REJECTS
  every false-advancer AND ADMITS every genuine explorer.
    PASS  -> the discriminating candidate(s) are recorded; recommended_criterion (the
             preferred discriminating form) routes to /implement-substrate on
             infant_curriculum.py `_try_phase_0_to_1` (replacing the single-episode SPIKE
             crossing with the sustained-level criterion).
    FAIL (movement floor met, both classes present, NO candidate discriminates)
          -> needs_alternative_criterion (report which leg each candidate failed; do NOT
             force a gate change). Expectation (from 591d per-seed levels + the spike
             analysis): the robust candidates (window-mean / EMA-with-hold / crossing-count)
             discriminate even if plain EMA-of-level re-admits seed 45.
  (Seed 46 staying in Phase 0 under all criteria does NOT count against any candidate --
   that is the orthogonal exploration-STRENGTH defect, the Q-043/667->667a thread.)

NON-VACUITY PRECONDITIONS (interpretation.preconditions[]; same shape + statistic as 591d)
-----------------------------------------------------------------------------------------
- early_policy_produces_nontrivial_h_pos: max per-episode h_pos across all seeds >= a
  movement floor (the agent-moves positive control; SAME LEVEL statistic the gate routes
  on). Below floor -> degenerate non-moving artifact -> substrate_not_ready_requeue, NOT a
  discrimination verdict.
- genuine_explorers_present: >= 2 seeds are genuine explorers (h_pos_mean >= floor AND
  >= 2 crossings), so the "admits explorers" leg is non-vacuous. Below -> no explorer
  signal to discriminate against -> substrate_not_ready_requeue. SAME genuine-explorer
  LEVEL statistic the C_gate_discriminates candidates route on.
- false_advancer_present: >= 1 seed is a false advancer, so the "rejects false-advancer"
  leg is non-vacuous. Below -> the single-episode gate was NOT over-permissive on this
  seed draw (report; do NOT record a discrimination PASS).

ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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

QUEUE_ID = "V3-EXQ-591f"
EXPERIMENT_TYPE = "v3_exq_591f_isef005_phase01_gate_criterion"
SUPERSEDES = "V3-EXQ-591e"
CLAIM_IDS: List[str] = []  # gate-criterion substrate diagnostic -- weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
N_EPISODES = 160          # faithful to V3-EXQ-591b/591c/591d/591e
STEPS_PER_EPISODE = 200   # faithful to V3-EXQ-591b/591c/591d/591e

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # 0.20 * ln(144) ~= 0.994 (SPIKE bar)
PHASE_01_EP_MIN = PHASE_EP_MIN[1]                # 100 (hard episode-count minimum)

# Readiness floor: the early-policy agent must actually MOVE (non-trivial per-episode
# H_pos) for the discrimination result to be a real verdict. SAME LEVEL statistic the
# gate routes on (per-episode pos_entropy).
H_POS_MOVEMENT_FLOOR = 0.20
# Genuine-exploration floor (the ORACLE; unchanged from 591d): a seed explores genuinely
# iff its mean per-episode H_pos clears this AND it cleared the 0.994 spike bar on more
# than a single eligible episode (591c/591d seeds 42-44 mean band 0.32-0.84). A
# near-stationary fluke (seed 46 h_pos_mean 0.0375) reads well below it.
GENUINE_EXPLORATION_H_POS_MEAN_FLOOR = 0.20
GENUINE_EXPLORATION_MIN_CROSSINGS = 2

# --- Sustained-level candidate-criterion parameters (the 591e -> 591f robust sweep) ---
# All candidates compare a SUSTAINED-LEVEL statistic against the genuine-explorer LEVEL
# floor (~0.20), NOT the 0.994 single-episode SPIKE threshold the 591d candidates wrongly
# reused. The sweep protects against the EMA(alpha=0.2)-of-level spike-sensitivity 591e's
# steady-state reasoning missed (a single 1.453 spike pushes EMA(0.2) to ~0.37).
PHASE_01_LEVEL_FLOOR = GENUINE_EXPLORATION_H_POS_MEAN_FLOOR  # ~= 0.20 (shared by B1/B2/B3)
EMA_ALPHA = 0.2          # B1 / B3: ~5-episode effective window (unchanged from 591d/591e)
WINDOW_W = 20            # B2: trailing-window length (episodes)
HOLD_H = 5               # B3: consecutive episodes EMA must stay supra-floor (hysteresis)
CROSSING_COUNT_MIN = 3   # B4: cumulative post-ep_min crossings of the 0.994 spike bar

# Candidate keys (the swept sustained-level criteria) and the recommendation preference
# order. recommended_criterion is the FIRST candidate in this list that discriminates.
# Order rationale: EMA-with-hold and window-mean are biologically-faithful "integrate over
# a window" reads that are spike-robust (preferred); crossing-count is robust but discrete;
# plain EMA-of-level is the spike-vulnerable baseline candidate (last). The downstream
# /implement-substrate session makes the final pick among the discriminating set.
CANDIDATE_KEYS = ["ema_hold", "window_mean", "crossing_count", "ema_level"]


def _build_diversity_agent() -> REEAgent:
    """Mirror of the 591c/591d/591e diversity-armed agent build: MECH-313 noise floor +
    MECH-314 structured curiosity ON at their landed default magnitudes (SP-CEM is
    already the main-path default). Bit-identical to the 591/591b/591c/591d/591e agent
    build otherwise."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        novelty_bonus_weight=0.5,
        use_sleep_loop=True,
        sleep_loop_episodes_K=N_EPISODES + 1,  # K=never (> total episodes)
        # --- exploration-diversity stack (matches 591c/591d/591e) ---
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
    per-episode h_pos sequence (plus the baseline-gate summary stats). Identical to the
    591d/591e seed runner -- only the OFFLINE criterion replay differs."""
    torch.manual_seed(seed)
    agent = _build_diversity_agent()
    sched = InfantCurriculumScheduler(grid_size=GRID_SIZE)

    per_ep_h_pos: List[float] = []
    baseline_phase_01_at: Optional[int] = None

    for ep in range(n_episodes):
        sched.env_kwargs()  # Phase 0 -> all infant features OFF (informational call)
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
                f"  [train] gate-criterion seed={seed} ep {ep + 1}/{n_episodes}"
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
        # full sequence for offline criterion replay (negatives clipped to 0.0 so an
        # invalid reading never spuriously clears the positive floor)
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
# Each returns (advanced, at_episode). All candidates are gated by ep_min; the BASELINE
# uses the 0.994 SPIKE bar (informational), every candidate uses the ~0.20 LEVEL floor.
# ----------------------------------------------------------------------------------

def _advance_single_episode(
    seq: List[float], threshold: float, ep_min: int
) -> Tuple[bool, Optional[int]]:
    """BASELINE (informational): single-episode SPIKE crossing of `threshold` (0.994)."""
    for ep in range(ep_min, len(seq)):
        if seq[ep] >= threshold:
            return True, ep
    return False, None


def _advance_ema_level(
    seq: List[float], level_floor: float, ep_min: int, alpha: float
) -> Tuple[bool, Optional[int]]:
    """B1 (spike-vulnerable baseline candidate): advance when an EMA(alpha) of the
    per-episode h_pos LEVEL is at/above `level_floor` at any ep >= ep_min. A single
    transient spike can push EMA(0.2) above the floor for ~1-2 episodes, so this is
    expected to RE-ADMIT the seed-45 false-advancer on its episode-142 spike (the 591e
    failure mode this sweep guards against)."""
    if not seq:
        return False, None
    ema = seq[0]
    for ep in range(1, len(seq)):
        ema = (1.0 - alpha) * ema + alpha * seq[ep]
        if ep >= ep_min and ema >= level_floor:
            return True, ep
    return False, None


def _advance_window_mean(
    seq: List[float], level_floor: float, ep_min: int, window: int
) -> Tuple[bool, Optional[int]]:
    """B2 (robust): advance when the trailing-mean over the last `window` episodes is
    at/above `level_floor` at any ep >= ep_min. A transient spike is diluted across the
    window (two 1.45 spikes over 20 episodes contribute ~0.145), so a near-stationary
    false-advancer (mean 0.14) stays below the floor while a genuine explorer (mean
    0.32-0.84) clears it. Uses the available trailing episodes when fewer than `window`
    have accumulated."""
    for ep in range(ep_min, len(seq)):
        lo = max(0, ep - window + 1)
        win = seq[lo:ep + 1]
        if win and (sum(win) / len(win)) >= level_floor:
            return True, ep
    return False, None


def _advance_ema_level_with_hold(
    seq: List[float], level_floor: float, ep_min: int, alpha: float, hold: int
) -> Tuple[bool, Optional[int]]:
    """B3 (robust): advance only when the EMA(alpha) of per-episode h_pos LEVEL stays
    at/above `level_floor` for `hold` CONSECUTIVE episodes (all at ep >= ep_min). An
    isolated spike pushes the EMA supra-floor for only ~1-2 episodes before it decays
    back below, so it cannot satisfy a consecutive-hold of 5 -- the hysteresis that
    rejects the seed-45 transient while admitting a sustained explorer. Advances at the
    episode the hold streak completes."""
    if not seq:
        return False, None
    ema = seq[0]
    streak = 0
    for ep in range(1, len(seq)):
        ema = (1.0 - alpha) * ema + alpha * seq[ep]
        if ep >= ep_min:
            if ema >= level_floor:
                streak += 1
                if streak >= hold:
                    return True, ep
            else:
                streak = 0
    return False, None


def _advance_crossing_count(
    seq: List[float], spike_threshold: float, ep_min: int, min_count: int
) -> Tuple[bool, Optional[int]]:
    """B4 (robust conjunction): advance when the cumulative count of post-ep_min episodes
    crossing the 0.994 SPIKE bar reaches `min_count`. Seed 45 crosses the spike bar only
    twice (n_eligible_ge_threshold=2), so >= 3 rejects it while genuine explorers (6/7/36
    crossings) clear it. Robust because it requires repeated crossings, not a single
    spike. Advances at the episode the count reaches `min_count`."""
    count = 0
    for ep in range(ep_min, len(seq)):
        if seq[ep] >= spike_threshold:
            count += 1
            if count >= min_count:
                return True, ep
    return False, None


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    n_episodes = 4 if dry_run else N_EPISODES
    print(
        f"V3-EXQ-591f gate-criterion: seeds={seeds} n_episodes={n_episodes}"
        f" steps={STEPS_PER_EPISODE} spike_threshold={PHASE_01_THRESHOLD:.4f}"
        f" level_floor={PHASE_01_LEVEL_FLOOR:.4f} ep_min={PHASE_01_EP_MIN}"
        f" [sweep: ema_level(a={EMA_ALPHA}) window_mean(W={WINDOW_W})"
        f" ema_hold(H={HOLD_H}) crossing_count(>={CROSSING_COUNT_MIN})]"
        f" [diversity stack ON] [supersedes {SUPERSEDES}]",
        flush=True,
    )
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition gate-criterion", flush=True)
        seed_results.append(_run_seed(seed=seed, n_episodes=n_episodes))

    # --- offline criterion replay + classification ---
    ep_min = PHASE_01_EP_MIN if not dry_run else 0  # dry-run uses ep_min 0 so it exercises
    per_seed_criteria: List[Dict[str, Any]] = []
    n_genuine = 0
    n_false = 0
    for r in seed_results:
        seq = r["h_pos_sequence"]
        adv_base, at_base = _advance_single_episode(seq, PHASE_01_THRESHOLD, ep_min)
        adv_ema, at_ema = _advance_ema_level(
            seq, PHASE_01_LEVEL_FLOOR, ep_min, EMA_ALPHA)
        adv_win, at_win = _advance_window_mean(
            seq, PHASE_01_LEVEL_FLOOR, ep_min, WINDOW_W)
        adv_hold, at_hold = _advance_ema_level_with_hold(
            seq, PHASE_01_LEVEL_FLOOR, ep_min, EMA_ALPHA, HOLD_H)
        adv_cross, at_cross = _advance_crossing_count(
            seq, PHASE_01_THRESHOLD, ep_min, CROSSING_COUNT_MIN)

        genuine = bool(
            r["h_pos_mean"] >= GENUINE_EXPLORATION_H_POS_MEAN_FLOOR
            and r["n_eligible_ge_threshold"] >= GENUINE_EXPLORATION_MIN_CROSSINGS
        )
        # False advancer: the single-episode SPIKE gate admitted it, but it does not
        # explore genuinely (the seed-45 profile).
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
            "ema_level": {"advanced": adv_ema, "at_episode": at_ema},
            "window_mean": {"advanced": adv_win, "at_episode": at_win},
            "ema_hold": {"advanced": adv_hold, "at_episode": at_hold},
            "crossing_count": {"advanced": adv_cross, "at_episode": at_cross},
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

    per_criterion_discriminates = {k: _discriminates(k) for k in CANDIDATE_KEYS}
    discriminating = [k for k in CANDIDATE_KEYS if per_criterion_discriminates[k]]
    any_discriminates = bool(discriminating)
    # recommended_criterion = first discriminating candidate in the preference order.
    recommended_criterion = discriminating[0] if discriminating else None

    max_h_pos = max((r["h_pos_max"] for r in seed_results), default=-1.0)
    movement_ok = max_h_pos >= H_POS_MOVEMENT_FLOOR
    genuine_ok = n_genuine >= 2
    false_present = n_false >= 1

    # --- routing ---
    if not movement_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"  # degenerate non-moving artifact
    elif not genuine_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"  # no genuine-explorer signal to admit
    elif not false_present:
        # The single-episode SPIKE gate was NOT over-permissive on this seed draw -- there
        # is no false-advancer to reject. Not a substrate-readiness failure; surface it.
        outcome = "FAIL"
        label = "no_false_advancer_reproduced_gate_not_overpermissive_on_this_draw"
    elif any_discriminates:
        outcome = "PASS"
        label = f"sustained_level_criterion_discriminates_{recommended_criterion}"
    else:
        outcome = "FAIL"
        label = "no_sustained_level_candidate_discriminates_needs_alternative"

    return {
        "outcome": outcome,
        "label": label,
        "seed_results": seed_results,
        "per_seed_criteria": per_seed_criteria,
        "n_genuine_explorers": n_genuine,
        "n_false_advancers": n_false,
        "genuine_explorer_seeds": genuine_seeds,
        "false_advancer_seeds": false_seeds,
        "per_criterion_discriminates": per_criterion_discriminates,
        "discriminating_criteria": discriminating,
        "any_discriminates": any_discriminates,
        "recommended_criterion": recommended_criterion,
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
    any_discriminates = bool(result["any_discriminates"])

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": SUPERSEDES,
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
            "phase_0to1_spike_threshold": round(PHASE_01_THRESHOLD, 6),
            "phase_0to1_level_floor": round(PHASE_01_LEVEL_FLOOR, 6),
            "phase_0to1_ep_min": PHASE_01_EP_MIN,
            "h_pos_movement_floor": H_POS_MOVEMENT_FLOOR,
            "genuine_exploration_h_pos_mean_floor": GENUINE_EXPLORATION_H_POS_MEAN_FLOOR,
            "genuine_exploration_min_crossings": GENUINE_EXPLORATION_MIN_CROSSINGS,
            "ema_alpha": EMA_ALPHA,
            "window_w": WINDOW_W,
            "hold_h": HOLD_H,
            "crossing_count_min": CROSSING_COUNT_MIN,
            "candidate_preference_order": CANDIDATE_KEYS,
            "corrected_parameter_note": (
                "591e -> 591f robust sustained-level SWEEP (user override of the EMA-only"
                " route via AskUserQuestion 2026-06-15). All four candidates compare a"
                " sustained-level statistic against the genuine-explorer LEVEL floor"
                f" ({PHASE_01_LEVEL_FLOOR:.4f}), NOT the single-episode SPIKE threshold"
                f" ({PHASE_01_THRESHOLD:.4f}) the 591d candidates reused. Plain"
                " EMA-of-level@floor (B1) is retained as the spike-vulnerable baseline"
                " candidate (591e's pick); B2 window-mean, B3 EMA-with-hold, and B4"
                " crossing-count>=3 are the spike-robust forms (a single 1.453 spike pushes"
                " EMA(0.2) to ~0.37 > floor, re-admitting seed 45 -- confirmed in an offline"
                " unit-check on synthetic 591d-shaped traces)."
            ),
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
                "AT LEAST ONE swept sustained-level candidate (B1 EMA-of-level@floor, B2"
                f" trailing-window-mean(W={WINDOW_W})@floor, B3 EMA-of-level@floor-with-hold(H={HOLD_H}),"
                f" B4 crossing-count>={CROSSING_COUNT_MIN} of the spike bar) -- all gated at the"
                f" genuine-explorer floor {PHASE_01_LEVEL_FLOOR:.4f}, NOT the 0.994 spike bar --"
                " REJECTS every false-advancer (seed-45-like: baseline-admitted but"
                f" h_pos_mean < {GENUINE_EXPLORATION_H_POS_MEAN_FLOOR} OR <"
                f" {GENUINE_EXPLORATION_MIN_CROSSINGS} crossings) AND ADMITS every genuine"
                " explorer (seeds 42-44 profile). PASS => recommended_criterion (the first"
                " discriminating candidate in the preference order) is the recommended"
                " replacement for the single-episode Phase 0->1 gate (routes to"
                " /implement-substrate on infant_curriculum.py _try_phase_0_to_1)."
            ),
            "C_seed46_out_of_scope": (
                "seed 46 staying in Phase 0 under ALL criteria does NOT count against any"
                " candidate -- that is the orthogonal exploration-strength defect owned by"
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
                        " LEVEL statistic the gate routes on (per-episode pos_entropy)."
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
                        " LEVEL statistic the C_gate_discriminates candidates route on."
                    ),
                    "measured": result["n_genuine_explorers"],
                    "threshold": 2,
                    "direction": "lower",
                    "control": "count of genuine-explorer seeds (591c/591d: seeds 42-44)",
                    "met": bool(result["n_genuine_explorers"] >= 2),
                },
                {
                    "name": "false_advancer_present",
                    "description": (
                        "At least 1 seed is a false advancer (baseline single-episode SPIKE"
                        " gate admitted a non-explorer), so the 'rejects false-advancer' leg"
                        " is non-vacuous. Below -> the single-episode gate was NOT"
                        " over-permissive on this seed draw (report; do NOT record a"
                        " discrimination PASS). SAME false-advancer statistic the"
                        " C_gate_discriminates candidates route on."
                    ),
                    "measured": result["n_false_advancers"],
                    "threshold": 1,
                    "direction": "lower",
                    "control": "count of false-advancer seeds (591c/591d: seed 45)",
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
                    "passed": any_discriminates,
                },
            ],
        },
        "metrics": {
            "n_genuine_explorers": result["n_genuine_explorers"],
            "n_false_advancers": result["n_false_advancers"],
            "genuine_explorer_seeds": result["genuine_explorer_seeds"],
            "false_advancer_seeds": result["false_advancer_seeds"],
            "per_criterion_discriminates": result["per_criterion_discriminates"],
            "discriminating_criteria": result["discriminating_criteria"],
            "any_discriminates": result["any_discriminates"],
            "recommended_criterion": result["recommended_criterion"],
            "max_h_pos": round(result["max_h_pos"], 4),
            "n_seeds_total": len(seeds_used),
        },
        "per_seed_criteria": result["per_seed_criteria"],
        "notes": (
            "Gate-CRITERION diagnostic for ARC-046 / infant_substrate:GAP-14; supersedes"
            " V3-EXQ-591e (robust-sweep correction of the spike-vulnerable EMA-of-level-only"
            " 591e; 591e itself superseded 591d). Re-runs the V3-EXQ-591c/591d diversity-armed"
            " Phase 0->1 reachability probe (deterministic; a reproducibility check of the"
            " 591c/591d seed traces), PERSISTS the full per-episode h_pos sequence per seed"
            " (which the 591d manifest did NOT -- it kept only per-seed summaries, so the"
            " online behaviour of a sustained-level criterion at the decision episode could"
            " not be validated from 591d), and replays each through BASELINE single-episode"
            " SPIKE (informational) + a SWEEP of four sustained-level Phase 0->1 candidates"
            " OFFLINE. The shared infant_curriculum.py scheduler is NOT mutated"
            " (610e/669/586/667 depend on its default gate). Robust-sweep design (user override"
            " of the prior EMA-of-level-only route via AskUserQuestion 2026-06-15): plain"
            " EMA-of-level@0.20 is spike-vulnerable -- a single per-episode h_pos of 1.453"
            " (seed 45 has one inside the decision window; baseline advanced 45 at ep 142)"
            " pushes EMA(alpha=0.2) to ~0.37 > the 0.20 floor, re-admitting the false-advancer"
            " (confirmed in an offline unit-check). The sweep adds trailing-window-mean"
            f" (W={WINDOW_W}), EMA-of-level-with-hold(H={HOLD_H}), and crossing-count>="
            f"{CROSSING_COUNT_MIN} (seed 45 crosses the spike bar only twice), which survive"
            " the transient. PASS if ANY candidate admits 42/43/44 and rejects 45 at the real"
            " decision episodes; recommended_criterion is the first discriminating candidate"
            " in the documented preference order. The orthogonal seed-46 exploration-STRENGTH"
            " collapse (Q-043/667->667a thread) is OUT OF SCOPE -- seed 46 is expected to stay"
            " in Phase 0 under every criterion. A discriminating criterion (PASS) routes to"
            " /implement-substrate as the sustained-level replacement for _try_phase_0_to_1;"
            " only after BOTH this gate-criterion fix AND the exploration-strength fix land"
            " can the full curriculum-vs-flat EXQ-ISEF-005 (V3-EXQ-591 successor) be queued."
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
