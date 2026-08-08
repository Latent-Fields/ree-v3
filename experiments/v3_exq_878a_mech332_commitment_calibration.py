"""V3-EXQ-878a: MECH-332 -- E3 commitment-gate calibration pilot (Pathway-2 readiness).

EXPERIMENT_PURPOSE: diagnostic (calibration probe -- deliverable is a SCHEDULE, not a
claim verdict). Work-graph class: complex (probe-gated) / puzzle (known rules).

=== WHY THIS EXISTS (governance-ratified routing, 2026-08-03) ===
The confirmed failure autopsy
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-878_2026-08-03.md`
(routing user-confirmed 2026-08-03, applied to MECH-332's evidence_quality_note in
claims.yaml) adjudicated run
`v3_exq_878_mech332_efference_aic_dissociation_20260803T023041Z_v3`.

Its Pathway 2 (commitment-gated PAG/RVM descending suppression -- SD-021, now subsumed
by SD-032c; substrate `ree_core/cingulate/aic_analog.py` AICAnalog, gated on
`use_aic_analog` + `harm_descending_mod_enabled`) NEVER got a valid test:
`n_committed_steps = 0` vs `N_COMMITTED_FLOOR = 8` across ALL 3 seeds, in BOTH
AIC-active arms (ARM_AIC_ONLY, ARM_BOTH). E3's commitment gate (MECH-090 BetaGate)
never elevated during eval, so Pathway 2's claim-intrinsic trigger (sustained committed
threat-approach) never fired. Pathway 2 is therefore UNTESTED, not falsified. Dominant
diagnosis: environment/schedule calibration gap (test-design), explicitly NOT a
substrate gap and NOT substrate_ceiling; MECH-090's commitment gate is independently
well-supported elsewhere.

Autopsy section 6, repair spec (this script implements it; deviations stated below):
  1. CALIBRATION PILOT FIRST -- a smaller sweep over training budget that watches
     n_committed_steps AS ITS OWN PASS/FAIL GATE, to find a schedule reliably clearing
     N_COMMITTED_FLOOR=8 in the AIC-active arms, BEFORE repeating the full dissociation.
  2. HARDEN THE MEASUREMENT WRINKLE -- z_harm_s_ratio's empty-committed-sample fallback
     goes from 1.0 to None (NaN), so a zero-commitment run fails loudly instead of
     silently reporting a spurious "no attenuation".
  3. DO NOT re-test Pathway 1 -- the efference-copy comparator (MECH-256/SD-029) D2
     result is already valid; this probe does not measure it at all.

=== WHAT MAKES COMMITMENT FIRE (re-verified fresh by code inspection this session) ===
`agent.e3._committed_trajectory` is written non-None only inside
`E3TrajectorySelector.select()` (e3_selector.py:3559), and only when the commit
predicate holds (e3_selector.py:3181-3185, default branch):
    committed = running_variance < effective_threshold        # effective_threshold
                                                              # starts as commit_threshold
`running_variance` (e3._running_variance) inits at `precision_init = 0.5` and is
EMA-updated (`precision_ema_alpha = 0.05`) toward observed world-prediction error;
`commit_threshold = commitment_threshold = 0.40` (E3Config). So an UNTRAINED world model
sits at rv ~= 0.5 > 0.40 and NEVER commits; rv only converges toward ~0.33 (below the
bar) once the world model is competent (config.py:584-588 says exactly this).

THE DIAGNOSIS IS AIRTIGHT, not a guess: in the 878 manifest the AIC-active arms recorded
`n_fresh_select_ticks` of 76-331 per cell -- plenty of genuine E3 selection ticks where
commitment COULD have been read -- yet `n_committed_steps = 0` in every one. So the
observability cadence (e3_steps_per_tick default 10) was NOT the limiter; the variance
gate simply never opened because the world model was undertrained at P0+P1 = 180
episodes (consistent with the MECH-457 cold-start / competence-floor precedent the
autopsy cites).

=== DESIGN: single-axis TRAINING-BUDGET sweep x 2 AIC-active arms x 3 seeds ===
DELIBERATE NARROWING (deviation from the autopsy's "training budget / eval episode count
/ arena richness", with stated reason): the sweep varies TRAINING BUDGET ONLY; eval
length and arena are held fixed. Justification is empirical, not convenience: the 878
manifest's own `n_fresh_select_ticks` (76-331) proves the eval already exercised the E3
selection path far more than the floor of 8 needs, so eval exposure was demonstrably NOT
the bottleneck -- the variance gate was. Sweeping eval length or arena richness would
spend compute on an axis the data already rules out. If (and only if) even the largest
budget fails to open the gate, THAT is the informative negative that reopens the other
axes (and points at world-model convergence / the gate itself), and this probe self-
routes to say so.

  Schedules (training budget = P0 encoder warmup + P1 E2_harm_s phased training):
    S0_baseline : P0=80,  P1=100  (180 eps)  -- reproduces 878; negative reference
    S1_2x       : P0=160, P1=200  (360 eps)
    S2_3x       : P0=240, P1=300  (540 eps)  -- MECH-457 cold-start range
  Arms (both AIC-active -- the two the autopsy names; commitment is world-model-driven
  and ~flag-independent, so both are expected to track together -- running both directly
  confirms the floor is cleared in ARM_AIC_ONLY and ARM_BOTH as the autopsy requires):
    ARM_AIC_ONLY : use_e2=False, use_aic=True,  harm_descending=True
    ARM_BOTH     : use_e2=True,  use_aic=True,  harm_descending=True
  Seeds: 42, 7, 13 (the 878 seeds; reliability-across-seeds -- seed 13 was the 878
  long-episode outlier, kept so the pilot sees the same spread).

Arena/agent config, LR, and the SD-022+SD-029 (EXQ-479-calibrated) curriculum are
imported verbatim from the lineage baseline module
`experiments/_lib/baselines/exq878_mech332_efference_aic_baseline.py`, so only the two
swept axes (budget, arm flags) differ from 878.

=== PRE-REGISTERED GATE (n_committed_steps as its own pass/fail -- autopsy spec #1) ===
Per cell (schedule x arm x seed): CELL CLEARS iff n_committed_steps >= N_COMMITTED_FLOOR.
Per (schedule x arm): CLEARS iff it clears on >= ceil(2/3 * n_seeds) seeds.
Winning schedule = the SMALLEST-budget schedule that clears in BOTH arms.
  found          -> outcome PASS, evidence_direction non_contributory (diagnostic),
                    label "commitment_calibration_schedule_found"; the winning budget is
                    recorded for the 878b full-dissociation re-run.
  none found     -> outcome FAIL, evidence_direction non_contributory,
                    label "commitment_gate_not_budget_limited": budget is NOT the lever
                    (the gate stayed shut even at 3x) -> reopens the deeper diagnosis
                    (world-model convergence / the commit_threshold itself), NOT another
                    budget bump. Informative negative.
  measurement    -> if any cell has n_fresh_select_ticks < FRESH_FLOOR the commitment
     starved        instrument never fired there: substrate_not_ready_requeue.

=== DIAGNOSTIC ADJUDICATION (self-route must be falsifiable by the pipeline) ===
The verdict routes on n_committed_steps, a measured quantity, so per the /queue-experiment
Diagnostic adjudication + P0 readiness-assert rules:
  - readiness precondition (SAME statistic family as the load-bearing count): the eval
    actually exercised genuine E3 selection ticks -- min n_fresh_select_ticks across cells
    >= FRESH_FLOOR. Positive control: the 878 run recorded 76-331. A below-floor reading
    means the measurement was STARVED, so this self-routes below-floor to
    `substrate_not_ready_requeue`, never to a substrate-verdict label.
  - criteria_non_degenerate: the commitment measurement was live (fresh ticks > 0) and
    the budget sweep spanned genuinely different budgets.
  - load_bearing criterion: a winning schedule was found. A PASS carried by anything else
    is flagged vacuous by the indexer.

=== EXPLANATORY PAYLOAD (record generously) ===
Per cell we record the direct predictor of the gate, not just its outcome:
  running_variance_mean/min over fresh E3 ticks, commit_threshold, and
  frac_fresh_below_threshold (fraction of fresh E3 ticks with rv < commit_threshold --
  this IS the commitment rate). This is what lets a reader see whether budget is moving
  rv toward the bar (calibratable) or leaving it pinned at ~0.5 (a deeper blocker).

=== MEASUREMENT-WRINKLE HARDENING (autopsy spec #2) ===
z_harm_s_ratio = mean(z_committed)/mean(z_uncommitted) is recorded as JSON null (None)
when either sample is empty -- NOT the legacy 1.0 fallback. SD-021's own record carries
two historical FAILs read as bit-identical s_ratio_D=1.0000 / s_ratio_C=1.0000
(V3-EXQ-325b, V3-EXQ-325f) that are INDISTINGUISHABLE from the exact zero-commitment
artifact this probe exists to characterise -- see custom_information.measurement_wrinkle_note.

=== ETHICS PREFLIGHT (descriptive, non-enforced) ===
involves_negative_valence: true (z_harm_s drive, as in every V3 harm-stream experiment) --
all other involvement flags false -- decision: allow. Pre-ethical instrumentation per
SENT-0; no V4-boundary concern. No sleep phases (SLEEP DRIVER: not applicable).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig, HeartbeatConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.baselines import (  # noqa: E402
    exq878_mech332_efference_aic_baseline as BASE,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                            #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_878a_mech332_commitment_calibration"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = ["MECH-332", "SD-021", "SD-032c"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = "EVB-0268"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# The one readiness precondition (`commitment_instrument_live`) is a pure
# measurement-LIVENESS count -- min n_fresh_select_ticks >= FRESH_FLOOR (8) -- not a
# narrow substrate-signature predicate that could be unmeetable by construction. It is
# reachable by construction: V3-EXQ-878 recorded n_fresh_select_ticks of 76-331 per cell
# (~10-40x this floor) on the identical arena/agent, so the gate cannot report met=false
# forever and cannot launder an instrument-specification gap into a substrate verdict.
# The predicate IS the degeneracy definition ("did the E3 selection path fire at all"),
# which is the sanctioned exemption case.
ANCHOR_REACHABILITY_EXEMPT = (
    "commitment_instrument_live is a measurement-liveness count "
    "(n_fresh_select_ticks >= 8), reachable by construction -- V3-EXQ-878 recorded "
    "76-331 fresh ticks/cell on this exact substrate; it is the degeneracy definition, "
    "not a substrate-signature predicate."
)

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [42, 7, 13]

# The two AIC-active arms the autopsy names. Both have the descending-modulation
# pathway (Pathway 2) live; ARM_BOTH additionally trains the efference-copy forward
# model (Pathway 1) so it is the strictly-more-loaded config.
ARMS: Dict[str, Dict[str, bool]] = {
    "ARM_AIC_ONLY": {
        "use_e2_harm_s_forward": False,
        "use_aic_analog": True,
        "harm_descending_mod_enabled": True,
    },
    "ARM_BOTH": {
        "use_e2_harm_s_forward": True,
        "use_aic_analog": True,
        "harm_descending_mod_enabled": True,
    },
}

# Training-budget sweep. steps_per_ep + eval length held fixed (see docstring narrowing).
SCHEDULES: Dict[str, Dict[str, int]] = {
    "S0_baseline": {"p0_eps": 80, "p1_eps": 100},   # 180 -- reproduces 878
    "S1_2x": {"p0_eps": 160, "p1_eps": 200},        # 360
    "S2_3x": {"p0_eps": 240, "p1_eps": 300},        # 540 -- MECH-457 cold-start range
}
STEPS_PER_EP = BASE.STEPS_PER_EP          # 150, identical across the sweep
EVAL_MAX_EPS = BASE.EVAL_MAX_EPS          # 50, identical across the sweep
EVAL_FRESH_TARGET = 150                   # early-exit eval once this many fresh E3 ticks seen

N_COMMITTED_FLOOR = 8                     # autopsy's floor -- the gate this pilot chases
FRESH_FLOOR = N_COMMITTED_FLOOR           # need >= floor genuine selection opportunities
PASS_FRACTION_REQUIRED = 2.0 / 3.0        # a (schedule x arm) clears on >= ceil(2/3) seeds

# Dry-run overrides (smoke only -- never used for the pre-registered gate).
DRY_SEEDS = [SEEDS[0]]
DRY_SCHEDULES = {"S0_baseline": {"p0_eps": 2, "p1_eps": 2}}
DRY_STEPS_PER_EP = 25
DRY_EVAL_MAX_EPS = 3
DRY_EVAL_FRESH_TARGET = 4


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def _onehot(idx: int, n: int) -> torch.Tensor:
    v = torch.zeros(1, n)
    v[0, idx] = 1.0
    return v


def make_config(env: CausalGridWorldV2, flags: Dict[str, bool]) -> REEConfig:
    hb = HeartbeatConfig(beta_gate_bistable=True)
    kwargs = BASE.agent_kwargs(flags)
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        heartbeat=hb,
        **kwargs,
    )


# ------------------------------------------------------------------ #
# Train phase -- P0 (encoder warmup) + P1 (E2_harm_s phased training)  #
# Mirrors v3_exq_878 train_arm exactly, only the budget varies.        #
# ------------------------------------------------------------------ #
def train_arm(
    agent: REEAgent,
    env: CausalGridWorldV2,
    gen: torch.Generator,
    p0_eps: int,
    p1_eps: int,
    steps_per_ep: int,
    use_e2: bool,
    label: str,
    seed: int,
) -> None:
    action_dim = env.action_dim
    optimizer = torch.optim.Adam(agent.parameters(), lr=BASE.LR)
    harm_opt = None
    if use_e2 and agent.e2_harm_s is not None:
        harm_opt = torch.optim.Adam(agent.e2_harm_s.parameters(), lr=BASE.HARM_FWD_LR)

    agent.train()
    total_eps = p0_eps + p1_eps

    for ep in range(total_eps):
        train_e2_this_ep = use_e2 and harm_opt is not None and ep >= p0_eps
        _, obs_dict = env.reset()
        agent.reset()
        last_z_harm: Optional[torch.Tensor] = None
        last_action: Optional[torch.Tensor] = None

        for _ in range(steps_per_ep):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_history = obs_dict.get("harm_history")

            latent = agent.sense(
                obs_body, obs_world, obs_harm=obs_harm,
                obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
            )
            agent.clock.advance()

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss

            if (
                train_e2_this_ep
                and last_z_harm is not None
                and last_action is not None
                and latent.z_harm is not None
            ):
                z_pred = agent.e2_harm_s.forward(last_z_harm.detach(), last_action.detach())
                target = latent.z_harm.detach()
                h_loss = agent.e2_harm_s.compute_loss(z_pred, target)
                h_total = h_loss
                if torch.rand(1, generator=gen).item() < BASE.INTERVENTIONAL_FRACTION:
                    actual_idx = last_action.argmax(dim=-1)
                    cf_idx = (
                        actual_idx + 1
                        + torch.randint(0, action_dim - 1, actual_idx.shape, generator=gen)
                    ) % action_dim
                    a_cf = F.one_hot(cf_idx, num_classes=action_dim).float()
                    h_int = agent.e2_harm_s.compute_interventional_loss(
                        last_z_harm.detach(), last_action.detach(), a_cf
                    )
                    h_total = h_total + h_int
                harm_opt.zero_grad()
                h_total.backward()
                harm_opt.step()

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            action_idx = int(torch.randint(0, action_dim, (1,), generator=gen).item())
            action_oh = _onehot(action_idx, action_dim)
            agent._last_action = action_oh
            if latent.z_harm is not None:
                last_z_harm = latent.z_harm.detach()
                last_action = action_oh.detach()
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == total_eps - 1:
            # Progress line: M is the loop-bound variable (total_eps), never hardcoded.
            print(f"  [train] {label} seed={seed} ep {ep + 1}/{total_eps}", flush=True)


# ------------------------------------------------------------------ #
# Eval phase -- commitment-readiness probe (no efference machinery)    #
# ------------------------------------------------------------------ #
def eval_probe(
    agent: REEAgent,
    env: CausalGridWorldV2,
    gen: torch.Generator,
    eval_max_eps: int,
    steps_per_ep: int,
    fresh_target: int,
) -> Dict[str, Any]:
    action_dim = env.action_dim
    agent.eval()

    z_s_committed: List[float] = []
    z_s_uncommitted: List[float] = []
    aic_saliences: List[float] = []
    rv_fresh: List[float] = []          # running_variance sampled on genuine E3 ticks
    n_committed = 0
    n_uncommitted = 0
    n_fresh_select_ticks = 0
    n_latched_ticks = 0

    commit_threshold = float(getattr(agent.e3, "commit_threshold", float("nan")))

    for _ in range(eval_max_eps):
        if n_fresh_select_ticks >= fresh_target:
            break
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_ep):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_history = obs_dict.get("harm_history")

            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world, obs_harm=obs_harm,
                    obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
                )
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            # A genuine (fresh) E3 selection only happens on an e3_tick -- on any other
            # tick select_action returns the held/stepped action and e3.select() did not
            # run (agent.py early-return), so _committed_trajectory / _running_variance
            # were not re-evaluated this tick. Only fresh ticks are real observations.
            is_fresh_select = bool(ticks.get("e3_tick", False))
            if is_fresh_select:
                n_fresh_select_ticks += 1
                rv_fresh.append(float(getattr(agent.e3, "_running_variance", float("nan"))))
            else:
                n_latched_ticks += 1

            if agent.aic is not None:
                aic_saliences.append(float(agent.aic.aic_salience))

            is_committed = agent.e3._committed_trajectory is not None
            z_harm_curr = latent.z_harm.detach() if latent.z_harm is not None else None
            z_s_post = float(z_harm_curr.norm().item()) if z_harm_curr is not None else 0.0
            if is_committed:
                z_s_committed.append(z_s_post)
                n_committed += 1
            else:
                z_s_uncommitted.append(z_s_post)
                n_uncommitted += 1

            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    # MEASUREMENT-WRINKLE HARDENING (autopsy spec #2): None, NOT 1.0, on empty samples.
    if z_s_committed and z_s_uncommitted:
        z_ratio: Optional[float] = float(_mean(z_s_committed) / max(1e-8, _mean(z_s_uncommitted)))
    else:
        z_ratio = None

    rv_valid = [v for v in rv_fresh if v == v]  # drop any NaN
    frac_fresh_below = (
        float(sum(1 for v in rv_valid if v < commit_threshold) / len(rv_valid))
        if rv_valid else None
    )

    return {
        "n_committed_steps": n_committed,
        "n_uncommitted_steps": n_uncommitted,
        "n_fresh_select_ticks": n_fresh_select_ticks,
        "n_latched_ticks": n_latched_ticks,
        "commit_threshold": commit_threshold,
        "running_variance_mean_fresh": _mean(rv_valid),
        "running_variance_min_fresh": (min(rv_valid) if rv_valid else None),
        "frac_fresh_below_threshold": frac_fresh_below,
        "aic_salience_mean": _mean(aic_saliences),
        "z_harm_s_ratio": z_ratio,   # None on empty committed/uncommitted (hardened)
        "z_harm_s_mean_committed": _mean(z_s_committed),
        "z_harm_s_mean_uncommitted": _mean(z_s_uncommitted),
    }


# ------------------------------------------------------------------ #
# Cell driver + config-slice declaration (arm_fingerprint)             #
# ------------------------------------------------------------------ #
def _cell_config_slice(
    arm_id: str, seed: int, schedule: Dict[str, int], steps_per_ep: int, eval_max_eps: int
) -> Dict[str, Any]:
    flags = ARMS[arm_id]
    return {
        "env_kwargs": BASE.env_kwargs(seed),
        "agent_kwargs": BASE.agent_kwargs(flags),
        "schedule": {
            "p0_eps": schedule["p0_eps"],
            "p1_eps": schedule["p1_eps"],
            "steps_per_ep": steps_per_ep,
            "eval_max_eps": eval_max_eps,
        },
        "lr": BASE.LR,
        "harm_fwd_lr": BASE.HARM_FWD_LR,
        "interventional_fraction": BASE.INTERVENTIONAL_FRACTION,
        "interventional_margin": BASE.INTERVENTIONAL_MARGIN,
    }


def _run_cell(
    schedule_id: str, arm_id: str, seed: int, schedule: Dict[str, int],
    steps_per_ep: int, eval_max_eps: int, fresh_target: int,
    zg_accumulator: Optional[ZGoalStreamAccumulator] = None,
) -> Dict[str, Any]:
    flags = ARMS[arm_id]
    env = CausalGridWorldV2(**BASE.env_kwargs(seed))
    cfg = make_config(env, flags)
    agent = REEAgent(cfg)
    gen = torch.Generator()
    gen.manual_seed(seed)

    label = f"{schedule_id}/{arm_id}"
    train_arm(
        agent, env, gen, schedule["p0_eps"], schedule["p1_eps"],
        steps_per_ep, flags["use_e2_harm_s_forward"], label, seed,
    )
    result = eval_probe(agent, env, gen, eval_max_eps, steps_per_ep, fresh_target)
    row: Dict[str, Any] = {
        "arm_id": f"{schedule_id}__{arm_id}",
        "schedule_id": schedule_id,
        "pathway_arm": arm_id,
        "seed": seed,
        "p0_eps": schedule["p0_eps"],
        "p1_eps": schedule["p1_eps"],
        "train_eps": schedule["p0_eps"] + schedule["p1_eps"],
        **flags,
        **result,
    }
    row["cell_clears_floor"] = bool(result["n_committed_steps"] >= N_COMMITTED_FLOOR)
    if zg_accumulator is not None:
        zg_accumulator.observe(agent)
    return row


# ------------------------------------------------------------------ #
# Analysis                                                            #
# ------------------------------------------------------------------ #
def _analyse(rows: List[Dict[str, Any]], schedule_ids: List[str]) -> Dict[str, Any]:
    by_key = {(r["schedule_id"], r["pathway_arm"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    arms = list(ARMS.keys())
    seed_clear_floor = math.ceil(len(seeds) * PASS_FRACTION_REQUIRED)

    per_schedule: List[Dict[str, Any]] = []
    winning_schedule: Optional[str] = None
    for sid in schedule_ids:
        arm_blocks = {}
        both_arms_clear = True
        for arm in arms:
            cells = [by_key[(sid, arm, s)] for s in seeds]
            n_clear = sum(1 for c in cells if c["cell_clears_floor"])
            arm_clears = n_clear >= seed_clear_floor
            both_arms_clear = both_arms_clear and arm_clears
            arm_blocks[arm] = {
                "seeds_clearing_floor": n_clear,
                "n_seeds": len(seeds),
                "arm_clears": arm_clears,
                "n_committed_per_seed": {str(c["seed"]): c["n_committed_steps"] for c in cells},
                "frac_fresh_below_threshold_per_seed": {
                    str(c["seed"]): c["frac_fresh_below_threshold"] for c in cells
                },
                "running_variance_mean_fresh_per_seed": {
                    str(c["seed"]): c["running_variance_mean_fresh"] for c in cells
                },
            }
        per_schedule.append({
            "schedule_id": sid,
            "train_eps": by_key[(sid, arms[0], seeds[0])]["train_eps"],
            "both_arms_clear": both_arms_clear,
            "arms": arm_blocks,
        })
        if both_arms_clear and winning_schedule is None:
            winning_schedule = sid  # schedule_ids is in ascending-budget order

    # Measurement-adequacy (non-degeneracy) gate: was the instrument live everywhere?
    min_fresh = min(r["n_fresh_select_ticks"] for r in rows)
    measurement_live = bool(min_fresh >= FRESH_FLOOR)

    return {
        "per_schedule": per_schedule,
        "seeds": seeds,
        "seed_clear_floor_required": seed_clear_floor,
        "winning_schedule": winning_schedule,
        "min_n_fresh_select_ticks": min_fresh,
        "measurement_live": measurement_live,
    }


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Dict[str, Any]:
    seeds = DRY_SEEDS if dry_run else SEEDS
    schedules = DRY_SCHEDULES if dry_run else SCHEDULES
    steps_per_ep = DRY_STEPS_PER_EP if dry_run else STEPS_PER_EP
    eval_max_eps = DRY_EVAL_MAX_EPS if dry_run else EVAL_MAX_EPS
    fresh_target = DRY_EVAL_FRESH_TARGET if dry_run else EVAL_FRESH_TARGET
    schedule_ids = list(schedules.keys())

    rows: List[Dict[str, Any]] = []
    zg_accumulator = ZGoalStreamAccumulator()
    for sid in schedule_ids:
        for arm_id in ARMS:
            for seed in seeds:
                cond = f"{sid}/{arm_id}"
                print(f"Seed {seed} Condition {cond}", flush=True)
                slice_ = _cell_config_slice(
                    arm_id, seed, schedules[sid], steps_per_ep, eval_max_eps
                )
                with arm_cell(
                    seed,
                    config_slice=slice_,
                    script_path=Path(__file__),
                    config_slice_declared=True,
                ) as cell:
                    row = _run_cell(
                        sid, arm_id, seed, schedules[sid], steps_per_ep,
                        eval_max_eps, fresh_target, zg_accumulator=zg_accumulator,
                    )
                    cell.stamp(row)
                rows.append(row)
                print(f"verdict: {'PASS' if row['cell_clears_floor'] else 'FAIL'}", flush=True)

    analysis = _analyse(rows, schedule_ids)

    # --- self-route ------------------------------------------------------------ #
    if not analysis["measurement_live"]:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route_note = (
            f"Measurement starved: min n_fresh_select_ticks="
            f"{analysis['min_n_fresh_select_ticks']} < FRESH_FLOOR={FRESH_FLOOR}. "
            "The commitment instrument did not fire in at least one cell; n_committed "
            "readings are uninformative. Re-queue with a longer eval, not a verdict."
        )
    elif analysis["winning_schedule"] is not None:
        outcome = "PASS"
        label = "commitment_calibration_schedule_found"
        wsid = analysis["winning_schedule"]
        wblock = next(b for b in analysis["per_schedule"] if b["schedule_id"] == wsid)
        route_note = (
            f"Winning schedule {wsid} ({wblock['train_eps']} train eps) clears "
            f"N_COMMITTED_FLOOR={N_COMMITTED_FLOOR} in BOTH ARM_AIC_ONLY and ARM_BOTH on "
            f">= {analysis['seed_clear_floor_required']}/{len(analysis['seeds'])} seeds. "
            "Apply this budget to the 878b full-dissociation re-run; Pathway 2 now has a "
            "schedule that produces sustained committed threat-approach."
        )
    else:
        outcome = "FAIL"
        label = "commitment_gate_not_budget_limited"
        route_note = (
            "No schedule (up to 3x the 878 budget) opened the E3 commitment gate in both "
            "AIC-active arms. Budget is NOT the lever: inspect the per-cell "
            "running_variance_mean_fresh vs commit_threshold trend -- if rv stays pinned "
            "near precision_init (~0.5) rather than trending toward the 0.40 bar, the "
            "blocker is world-model convergence or the commit_threshold itself, and the "
            "correct next step is a deeper diagnostic, NOT another budget bump."
        )

    # Diagnostic is excluded from confidence scoring; direction is non_contributory.
    evidence_direction = "non_contributory"
    evidence_direction_per_claim = {c: "unknown" for c in CLAIM_IDS}

    # --- adjudication structures (self-route falsifiable by the pipeline) ------- #
    min_fresh = analysis["min_n_fresh_select_ticks"]
    winning_found = analysis["winning_schedule"] is not None
    preconditions = [
        {
            "name": "commitment_instrument_live",
            "kind": "readiness",
            "description": (
                "Eval exercised genuine E3 selection ticks (the denominator of "
                "n_committed_steps). Below FRESH_FLOOR means the measurement was starved, "
                "not that commitment is absent."
            ),
            "control": "V3-EXQ-878 recorded n_fresh_select_ticks 76-331 per cell",
            "measured": min_fresh,
            "threshold": FRESH_FLOOR,
            "direction": "lower",
            "met": bool(min_fresh >= FRESH_FLOOR),
        },
    ]
    criteria_non_degenerate = {
        "commitment_measurement_live": bool(min_fresh >= FRESH_FLOOR),
        "budget_sweep_non_trivial": len({SCHEDULES[s]["p0_eps"] + SCHEDULES[s]["p1_eps"]
                                         for s in SCHEDULES}) >= 2,
    }
    criteria = [
        {
            "name": "winning_commitment_schedule_found",
            "load_bearing": True,
            "passed": bool(winning_found),
        },
    ]

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "non_degenerate": bool(analysis["measurement_live"]),
        "degeneracy_reason": (
            None if analysis["measurement_live"]
            else f"min n_fresh_select_ticks {min_fresh} < FRESH_FLOOR {FRESH_FLOOR}"
        ),
        "arm_results": rows,
        "analysis": analysis,
        "interpretation": {
            "label": label,
            "route_note": route_note,
            "winning_schedule": analysis["winning_schedule"],
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
        },
        "pre_registered_thresholds": {
            "N_COMMITTED_FLOOR": N_COMMITTED_FLOOR,
            "FRESH_FLOOR": FRESH_FLOOR,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "custom_information": {
            "routing_provenance": (
                "Implements the governance-ratified (2026-08-03) repair routing from "
                "failure_autopsy_V3-EXQ-878_2026-08-03.md section 6: calibration pilot "
                "before repeating the full MECH-332 dissociation."
            ),
            "sweep_narrowing_note": (
                "Autopsy named training budget / eval episode count / arena richness. "
                "This pilot sweeps TRAINING BUDGET ONLY (eval + arena fixed) because the "
                "878 manifest's own n_fresh_select_ticks (76-331) proves eval exposure "
                "was NOT the limiter -- the variance gate was. If even 3x budget fails, "
                "the informative negative reopens the other axes and points at world-"
                "model convergence / the gate itself."
            ),
            "measurement_wrinkle_note": (
                "z_harm_s_ratio is recorded as null (None) on empty committed/uncommitted "
                "samples, NOT the legacy 1.0 fallback (v3_exq_878 lines 559-562). SD-021's "
                "record carries two historical FAILs read as bit-identical "
                "s_ratio_D=1.0000 / s_ratio_C=1.0000 (V3-EXQ-325b, V3-EXQ-325f) that are "
                "indistinguishable from a pure zero-commitment artifact -- those readings "
                "are now SUSPECT and should be re-checked against their n_committed."
            ),
            "pathway1_untouched_note": (
                "This probe does NOT measure Pathway 1 (efference-copy comparator, "
                "MECH-256/SD-029). The 878 D2 result (valid, clean, 3/3 seeds) stands; "
                "autopsy spec #3."
            ),
            "commitment_mechanism_note": (
                "commitment = running_variance < commit_threshold (default 0.40). "
                "running_variance inits at precision_init=0.5 and converges below the bar "
                "only when the world model is trained (config.py:584-588). frac_fresh_"
                "below_threshold per cell is the direct commitment rate; watch it trend "
                "with train_eps."
            ),
            "counting_comparability_note": (
                "n_committed_steps is counted on EVERY eval tick (fresh + latched), "
                "identical to the v3_exq_878 instrument, so the N_COMMITTED_FLOOR=8 gate "
                "is directly comparable to the 878 finding of 0 -- the meaningful signal "
                "is the 0 -> nonzero transition that a clearing budget produces. The "
                "clean per-decision commitment rate, unaffected by cross-tick latch "
                "persistence, is frac_fresh_below_threshold (sampled ONLY on fresh E3 "
                "ticks where running_variance was freshly re-evaluated). Deviation from "
                "878: the eval uses the agent's own policy throughout (no scripted "
                "hazard-forcing), since Pathway 1's efference trials are not measured "
                "here -- a cleaner probe of whether the agent's OWN trajectory sustains "
                "commitment."
            ),
            "ethics_preflight": {
                "involves_negative_valence": True,
                "involves_suffering_like_state": False,
                "involves_self_model": False,
                "involves_inescapability_or_helplessness": False,
                "involves_offline_replay_over_harm": False,
                "involves_social_mind_or_language": False,
                "involves_human_data_or_clinical_context": False,
                "decision": "allow",
            },
        },
        "_zg_stats_tmp": zg_accumulator.stats(),
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)
    zg_stats = manifest.pop("_zg_stats_tmp", None)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["timestamp_utc"] = ts
    manifest["dry_run"] = bool(args.dry_run)

    seeds_used = DRY_SEEDS if args.dry_run else SEEDS
    schedules_used = DRY_SCHEDULES if args.dry_run else SCHEDULES
    full_config = {
        "arms": ARMS,
        "schedules": schedules_used,
        "steps_per_ep": DRY_STEPS_PER_EP if args.dry_run else STEPS_PER_EP,
        "eval_max_eps": DRY_EVAL_MAX_EPS if args.dry_run else EVAL_MAX_EPS,
        "eval_fresh_target": DRY_EVAL_FRESH_TARGET if args.dry_run else EVAL_FRESH_TARGET,
        "arena": BASE.ARENA,
        "agent_common_kwargs": BASE.COMMON_AGENT_KWARGS,
        "lr": BASE.LR, "harm_fwd_lr": BASE.HARM_FWD_LR,
        "interventional_fraction": BASE.INTERVENTIONAL_FRACTION,
        "interventional_margin": BASE.INTERVENTIONAL_MARGIN,
        "pre_registered_thresholds": manifest["pre_registered_thresholds"],
    }

    # AFTER arm_results is assembled, so substrate_hash HOISTS from the per-cell
    # fingerprints rather than being recomputed (and mismatching).
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg_stats,
    )

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run, stamp=False,
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "label": manifest["interpretation"]["label"],
        "winning_schedule": manifest["analysis"]["winning_schedule"],
        "measurement_live": manifest["analysis"]["measurement_live"],
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
