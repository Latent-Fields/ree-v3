"""V3-EXQ-813 -- Re-run 737b's PPO-on-latent design under a survival-zeroed objective
(GOV-FANOUT-1 policy-axis probe for H-policy-learning; conversion_ceiling_root DIAGNOSTIC).

GOV-FANOUT-1 discriminating probe for the leg `H-policy-learning` (axis `policy`) of the
frozen-ledger question `conversion_ceiling_root`, pre-registered from the confirmed cluster
autopsy `failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22` (amended
2026-07-24). This is the POLICY-AXIS sibling of V3-EXQ-808 (REWARD axis: an empirical
consumption:survival weighting sweep on a single latent-only learner). The autopsy's own
`fanout_recommendation.suggested_probes` names both legs explicitly and states this one as:
"Re-run 737b's PPO-on-latent arm under a re-weighted objective in which the consumption term
provably dominates. Declared null: with the objective corrected the learner still fails to
clear the 1.0 floor, which would isolate policy-learning after all."

THE FINDING THIS PROBES. At hazard-free D3, V3-EXQ-737b's ppo_ree_latent (a real PPO actor
on a genuinely prediction-trained z_world, guard GREEN) scored 0.233 res/ep and its matched
ppo_raw_obs control scored 0.567 -- both BELOW the run's own random-walk anchor of 0.933,
while a local-view greedy reading the SAME 5x5 field scored 48.05 and the oracle 57.2. The
autopsy's reading: the training return is dominated by episode survival, survival is
maximised by NOT foraging, so 737b's null does not discriminate whether the policy-learning
stage itself is inadequate -- it could equally be a learner correctly optimising the wrong
objective. V3-EXQ-808 tests this on the REWARD axis (does re-weighting move competence at
all, empirically). This script tests it on the POLICY axis (does the SAME two-arm design
recover competence once the objective is corrected so consumption is the ONLY term).

=== HYPOTHESES UNDER TEST ===

H-policy-learning (this probe's leg, axis `policy`):
    The policy-learning stage genuinely cannot convert representation into competent
    foraging action, independent of the objective. If true, correcting the objective so
    consumption structurally dominates should NOT rescue either PPO arm at D3.

H-objective-misspecification (INCUMBENT; V3-EXQ-808 is its own discriminating probe on the
reward axis; stays alive regardless of this script's outcome -- see below):
    The training return was dominated by episode survival; correcting the objective so
    consumption is the only term should recover foraging IF the policy-learning stage is
    itself adequate.

DECLARED NULL (pre-registered; the result that ISOLATES policy-learning, per the autopsy):
    Neither ppo_ree_latent NOR ppo_raw_obs clears the 1.0 competence floor (strict majority
    of seeds) at hazard-free D3 under the W3_survival_zeroed (consumption-only) objective.
    The failure can no longer be attributed to a survival-dominated objective -- that
    confound is deleted BY CONSTRUCTION, not merely by empirical measurement -- so it
    isolates policy-learning inadequacy (or a deeper obstruction 737b's own grid already
    flags: PPO under-powered, weighed against V3-EXQ-738's local_view_greedy clearing the
    same env) as the surviving candidate explanation. This does NOT prove H-policy-learning;
    it closes the objective-misspecification confound on the policy axis specifically, the
    same way V3-EXQ-808 closes it (or does not) on the reward axis. The two probes are
    independent legs on different design axes and neither one's outcome pre-determines the
    other's (autopsy section 6 routing table; module docstring of V3-EXQ-808 states the
    symmetric fact from its own side).

ALTERNATIVE OUTCOME (floor cleared): if ppo_ree_latent (or ppo_raw_obs) clears the floor once
the objective is corrected, that WEAKENS H-policy-learning and further corroborates
H-objective-misspecification from the policy side -- a real actor on the representation CAN
forage; 737b's original null was an artifact of the mis-set reward, not of the learner or the
representation.

THE FIXED WEIGHTING. W3_survival_zeroed (w_consume=1.0, w_survival=0.0) is imported verbatim
from V3-EXQ-808 (`x808.WEIGHTINGS`), NOT redefined here, so this probe cannot drift from its
reward-axis sibling's definition of "provably dominates". It is the ONE level in 808's sweep
that deletes the survival-linked terms (proximity/novelty/other) STRUCTURALLY rather than
merely rescaling them, so consumption dominance here is a design-time proof, not an empirical
claim contingent on 808's own (not-yet-run) sweep results -- this script does not need 808 to
have resolved before it can run or be read. `harm` is never reweighted (808's own rule: it is
a penalty, not a survival incentive) and remains fully in force, which is immaterial at
hazard-free D3 where harm ticks are rare to absent.

TWO ARMS, REUSING V3-EXQ-808's REWARD-DECOMPOSITION TRAINER VERBATIM. 737b's original
two-arm design (ppo_ree_latent + matched ppo_raw_obs control) is preserved so an
encoder-specific finding remains separable from a general policy-learning finding, exactly as
in 737b's own four-way grid. Both arms train via `x808._train_ppo_decomposed` (identical GAE
/ running-std normalisation / PPO update to the family) at the SAME fixed W3 weighting -- the
only change from 808 is that `state_fn` is swapped (z_world latent vs raw observation vector)
instead of sweeping `w_consume`/`w_survival`. Reusing 808's trainer rather than reimplementing
reward-shaping code means both probes are reading the identical mechanism.

=== INTERPRETATION GRID ===

Pre-registered self-route (a HYPOTHESIS, not a verdict -- adjudicate via /failure-autopsy
before any governance use). This experiment PROMOTES AND DEMOTES NOTHING (claim_ids=[]).

  READINESS gate red on BOTH arms (neither scorable)
      -> `substrate_not_ready_requeue`. NEVER a policy-learning verdict.

  ppo_ree_latent scorable AND clears the floor (majority of seeds)
      -> `objective_correction_recovers_latent_policy` [PASS]. H-policy-learning WEAKENED;
         H-objective-misspecification corroborated on the policy axis.

  ppo_ree_latent scorable, does NOT clear, but ppo_raw_obs scorable AND clears
      -> `objective_correction_recovers_raw_obs_only_latent_lossy` [FAIL]. A real actor CAN
         forage under the corrected objective on the raw interface but not on z_world --
         implicates the ENCODER specifically (737's original `latent_lossy_raw_obs_recovers`
         reading, replicated under the corrected objective), not policy-learning generally.

  ppo_ree_latent scorable, does NOT clear, AND ppo_raw_obs does not clear (or is unscored)
      -> `policy_learning_isolated_after_objective_correction` [FAIL]. THE DECLARED NULL.

  ppo_ree_latent NOT scorable (z_world guard red) but ppo_raw_obs scorable AND clears
      -> `objective_correction_recovers_raw_obs_encoder_not_ready` [FAIL]. Informative on the
         raw-obs/policy question; the latent arm needs a re-run once the encoder trains.

  ppo_ree_latent NOT scorable, ppo_raw_obs scorable but does not clear
      -> `policy_learning_isolated_raw_obs_only_latent_encoder_not_ready` [FAIL]. Partial
         isolation via raw_obs only.

UNTRAINED-WORLD-ENCODER GUARD: GREEN-GATING for the ppo_ree_latent arm only (scoped OUT of
ppo_raw_obs via `applies_to`, since that arm never reads z_world). This mirrors V3-EXQ-808's
choice (both are z_world-dependent discriminating probes for conversion_ceiling_root) rather
than 737b's own record-not-refuse policy: 737b's non-strict stance was licensed because its
ppo_raw_obs control was reported as the definitive discriminator regardless of the latent
arm's guard state. Here the ppo_ree_latent verdict specifically feeds "isolates
policy-learning" -- a claim this script does not want to license on a frozen random
projection -- so it is gated, exactly as 808 gates the same guard for the same reason. 734
and 742a (the autopsy's OTHER two targets) needed gating for a different reason (a
REPRESENTATION question); this is a POLICY question, but the frozen-encoder confound applies
identically to any conclusion drawn from the ppo_ree_latent arm's number, so it is gated here
too, following 808's precedent rather than 737b's.

GOV-REUSE-1 (Step 2.4): the decisive readout -- "does a PPO actor on z_world / raw obs clear
the 1.0 floor at D3 under a survival-zeroed (consumption-only) objective" -- was searched via
`reanalysis_query.py query --readout ppo_ree_latent_survival_zeroed_foraging_competence`
against all 667 flat manifests; zero carry it (no manifest in the 724/734/737/742/808 family
computes a decomposed-reward PPO run under this specific weighting on the two-arm
latent+raw_obs design; 808 computes only the latent arm across a 4-level sweep, not this
fixed weighting on both arms). Not recoverable -> run.

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    assert_world_encoder_trained,
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_813_survival_zeroed_ppo_latent_policy_probe"
QUEUE_ID = "V3-EXQ-813"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Two-arm grid (arm x seed) but every cell trains a PPO actor from scratch on a per-seed
# re-warmed REE stack -- nothing is a pure function of a shared baseline config a later run
# could reuse. Matches 737b's and 808's own exemption for the same structural reason.
ARM_FINGERPRINT_EXEMPT = (
    "per-seed REE warmup + per-arm PPO training from scratch under a fixed reward level; "
    "no reusable baseline arm to bank"
)

DEVICE = torch.device("cpu")

SEEDS: List[int] = [42, 43, 44]

# Budget sourced from x734 so this driver cannot drift from the family it compares against.
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES        # 60; SD-070 z_world encoder warmup (P0a)
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES        # 200
P1_REINFORCE_EPISODES = x734.P1_REINFORCE_EPISODES  # 90
P1_PPO_EPISODES = x734.P1_PPO_EPISODES              # 1000
EVAL_EPISODES = x734.EVAL_EPISODES                  # 20
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE          # 200
PPO_ROLLOUT_EPISODES = x734.PPO_ROLLOUT_EPISODES    # 8

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x734.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_PPO = 6
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 20
DRY_RUN_ROLLOUT = 3

# The decisive rung: hazard-free, oracle-achievable, hazard confound removed. Imported from
# 734 so it is byte-identical to the sibling family (737b, 808) this probe reads against.
RUNG = x734.DIFFICULTY_RUNGS[-1]
RUNG_ID = RUNG["rung_id"]

# The fixed weighting -- imported verbatim from V3-EXQ-808, never redefined here, so this
# probe cannot drift from its reward-axis sibling's definition of "provably dominates".
_W3 = next(w for w in x808.WEIGHTINGS if w["id"] == "W3_survival_zeroed")
LEVEL_ID = str(_W3["id"])
W_CONSUME = float(_W3["w_consume"])
W_SURVIVAL = float(_W3["w_survival"])

ARM_IDS = ["ppo_ree_latent", "ppo_raw_obs"]
ANCHOR_IDS = ["random_walk", "local_view_greedy", "greedy_oracle"]

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own statistics)
# --------------------------------------------------------------------------------------
ZWORLD_DELTA_FLOOR = x808.ZWORLD_DELTA_FLOOR         # 1e-6; same guard floor as the family
# Sanity check that the objective correction genuinely landed: at hazard-free D3 with
# survival-linked terms deleted by construction, the realised train-phase return should be
# overwhelmingly consumption (harm ticks are rare-to-absent at D3). A worst-cell reading
# below this floor signals an implementation bug (e.g. a leaked survival-linked term), not a
# scientific finding -- this is a READINESS check, not a load-bearing criterion.
CONSUMPTION_SHARE_FLOOR = 0.90


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _arm_contexts() -> List[Dict[str, Any]]:
    return [{"id": aid} for aid in ARM_IDS]


# --------------------------------------------------------------------------------------
# Preconditions. Regime-conditioned via `applies_to` (the 785 rule): the z_world guard is
# scoped OUT of ppo_raw_obs (which never reads z_world) so an unmoved encoder red-flags only
# the ppo_ree_latent arm, never vacating the raw_obs arm's result.
# --------------------------------------------------------------------------------------
PRECONDITIONS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="zworld_encoder_trained_in_p0",
        description=(
            "The z_world world_encoder must have moved during P0 (SD-070 warmup path). An "
            "unmoved encoder means ppo_ree_latent is PPO on a frozen random projection, "
            "which cannot license an 'isolates policy-learning' reading."
        ),
        control="worst-cell world_encoder_max_abs_delta over all seed warmups vs 1e-6",
        threshold=ZWORLD_DELTA_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: ctx["id"] == "ppo_ree_latent",
        applies_note=(
            "ppo_raw_obs never reads z_world, so an untrained encoder is not meaningful for "
            "that arm's readiness"
        ),
    ),
    PreconditionSpec(
        name="d3_oracle_clears_floor",
        description="The hazard-free D3 env must be floor-achievable with global information.",
        control="greedy_oracle worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="d3_local_view_greedy_clears_floor",
        description=(
            "The env must be floor-achievable under the LEARNER'S OWN observability (the "
            "same 5x5 local field), not only under a privileged global oracle (the 732a "
            "confound). 738 measured 48.05 here."
        ),
        control="local_view_greedy worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="consumption_share_dominates_under_correction",
        description=(
            "At the W3_survival_zeroed weighting, the train-phase realised return must be "
            "overwhelmingly consumption (survival-linked terms are deleted by construction; "
            "harm is rare-to-absent at hazard-free D3). A below-floor reading signals an "
            "implementation defect, not a finding -- the objective correction did not "
            "actually land."
        ),
        control="worst-cell train-phase consumption_share for this arm at W3_survival_zeroed",
        threshold=CONSUMPTION_SHARE_FLOOR,
        direction="lower",
        kind="readiness",
    ),
]


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min") -> Tuple[float, str]:
    return x808._worst_cell(rows, key, mode)


# --------------------------------------------------------------------------------------
def _run_seed(
    seed: int,
    zworld_p0: int,
    p0: int,
    p1: int,
    ppo_eps: int,
    eval_eps: int,
    steps: int,
    rollout: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """One seed: warm ONE all-ON stack, then train + eval BOTH arms at the fixed W3 level."""
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    total_denom = p0 + p1

    torch.manual_seed(seed)
    np.random.seed(seed)
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x734._make_all_on_agent(warm_env)
    print(f"Seed {seed} Condition {RUNG_ID}:warmup_all_on", flush=True)
    before = latent_stack_snapshot(agent)
    x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=total_denom,
        zworld_p0_episodes=zworld_p0,
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
        zworld_p0_dry_run=dry_run,
    )
    guard = latent_stack_weight_delta(agent, before)
    guard["seed"] = int(seed)
    guard["rung_id"] = str(RUNG_ID)
    guard["p0_episodes"] = int(p0)
    guard["zworld_p0_episodes"] = int(zworld_p0)
    # Loud, unmissable warning. NOT strict here (the run must land an interpretable
    # manifest and route to /failure-autopsy); the ppo_ree_latent arm's verdict is instead
    # gated via precondition zworld_encoder_trained_in_p0 (scoped OUT of ppo_raw_obs).
    assert_world_encoder_trained(
        agent, before, p0=p0, strict=False,
        context=f"{QUEUE_ID} rung={RUNG_ID} seed={seed}",
        escape_hint=(
            "this driver gates the ppo_ree_latent arm via precondition "
            "zworld_encoder_trained_in_p0 (scoped OUT of ppo_raw_obs, which does not read "
            "z_world): an unmoved encoder red-flags only the latent arm's gate, not the "
            "whole run"
        ),
    )

    _flat, probe_obs = x734._make_env(seed, env_kwargs).reset()
    z_dim = int(x737._agent_zworld(agent, probe_obs).shape[-1])
    action_dim = int(warm_env.action_dim)
    _flat2, probe2 = x734._make_env(seed, env_kwargs).reset()
    obs_keys = x734._raw_obs_keys_present(probe2)
    raw_dim = int(x734._raw_obs_vector(probe2, obs_keys, DEVICE).shape[-1])

    arm_rows: Dict[str, Dict[str, Any]] = {}

    # --- arm: ppo_ree_latent -----------------------------------------------------------
    print(f"Seed {seed} Condition {RUNG_ID}:ppo_ree_latent", flush=True)
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    latent_net = x734.PPOPolicyNet(in_dim=z_dim, action_dim=action_dim).to(DEVICE)
    latent_opt = torch.optim.Adam(latent_net.parameters(), lr=x734.PPO_LR)
    train_env = x734._make_env(seed, env_kwargs)
    latent_train_decomp = x808._train_ppo_decomposed(
        train_env, latent_net, latent_opt,
        state_fn=lambda od: x737._agent_zworld(agent, od),
        on_reset=agent.reset,
        n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
        level_id=LEVEL_ID, w_consume=W_CONSUME, w_survival=W_SURVIVAL, seed=seed,
        total_denominator=ppo_eps,
    )
    latent_row = evaluate_seed(
        x737.LatentPPOEvalPolicy(latent_net, agent), x734._make_env(seed, env_kwargs),
        eval_eps, steps,
    )
    latent_eval_decomp = x808._decompose_eval(
        x737.LatentPPOEvalPolicy(latent_net, agent), x734._make_env(seed, env_kwargs),
        eval_eps, steps, W_CONSUME, W_SURVIVAL,
    )
    arm_rows["ppo_ree_latent"] = {
        "cell_id": f"ppo_ree_latent|seed{seed}",
        "arm_id": "ppo_ree_latent",
        "seed": int(seed),
        "foraging_competence": float(latent_row["foraging_competence"]),
        "survival_horizon": float(latent_row["survival_horizon"]),
        "death_rate": float(latent_row["death_rate"]),
        "mean_episode_reward": float(latent_row["mean_episode_reward"]),
        "competence_supra_floor": bool(latent_row["competence_supra_floor"]),
        "train_decomposition": latent_train_decomp,
        "eval_decomposition": latent_eval_decomp,
        "train_consumption_share": float(latent_train_decomp["consumption_share"]),
    }
    print(f"verdict: {'PASS' if latent_row['competence_supra_floor'] else 'FAIL'}", flush=True)

    # --- arm: ppo_raw_obs (matched control; encoder-loss reference) ---------------------
    print(f"Seed {seed} Condition {RUNG_ID}:ppo_raw_obs", flush=True)
    torch.manual_seed(seed + 2000)
    np.random.seed(seed + 2000)
    raw_net = x734.PPOPolicyNet(in_dim=raw_dim, action_dim=action_dim).to(DEVICE)
    raw_opt = torch.optim.Adam(raw_net.parameters(), lr=x734.PPO_LR)
    train_env2 = x734._make_env(seed, env_kwargs)
    raw_train_decomp = x808._train_ppo_decomposed(
        train_env2, raw_net, raw_opt,
        state_fn=lambda od: x734._raw_obs_vector(od, obs_keys, DEVICE),
        on_reset=None,
        n_episodes=ppo_eps, rollout_episodes=rollout, steps_per_episode=steps,
        level_id=LEVEL_ID, w_consume=W_CONSUME, w_survival=W_SURVIVAL, seed=seed,
        total_denominator=ppo_eps,
    )
    raw_row = evaluate_seed(
        x734.PPOEvalPolicy(raw_net, obs_keys, DEVICE), x734._make_env(seed, env_kwargs),
        eval_eps, steps,
    )
    raw_eval_decomp = x808._decompose_eval(
        x734.PPOEvalPolicy(raw_net, obs_keys, DEVICE), x734._make_env(seed, env_kwargs),
        eval_eps, steps, W_CONSUME, W_SURVIVAL,
    )
    arm_rows["ppo_raw_obs"] = {
        "cell_id": f"ppo_raw_obs|seed{seed}",
        "arm_id": "ppo_raw_obs",
        "seed": int(seed),
        "foraging_competence": float(raw_row["foraging_competence"]),
        "survival_horizon": float(raw_row["survival_horizon"]),
        "death_rate": float(raw_row["death_rate"]),
        "mean_episode_reward": float(raw_row["mean_episode_reward"]),
        "competence_supra_floor": bool(raw_row["competence_supra_floor"]),
        "train_decomposition": raw_train_decomp,
        "eval_decomposition": raw_eval_decomp,
        "train_consumption_share": float(raw_train_decomp["consumption_share"]),
    }
    print(f"verdict: {'PASS' if raw_row['competence_supra_floor'] else 'FAIL'}", flush=True)

    # --- anchors: random_walk (floor) / local_view_greedy (own-observability ceiling) /
    #     greedy_oracle (global-info ceiling) --------------------------------------------
    anchor_rows: List[Dict[str, Any]] = []
    anchors = {
        "random_walk": RandomPolicy(seed),
        "local_view_greedy": LocalViewGreedyPolicy(seed),
        "greedy_oracle": OraclePolicy(),
    }
    for aid in ANCHOR_IDS:
        print(f"Seed {seed} Condition {RUNG_ID}:{aid}", flush=True)
        row = evaluate_seed(anchors[aid], x734._make_env(seed, env_kwargs), eval_eps, steps)
        anchor_rows.append({
            "cell_id": f"{aid}|seed{seed}",
            "anchor_id": aid,
            "seed": int(seed),
            "foraging_competence": float(row["foraging_competence"]),
            "survival_horizon": float(row["survival_horizon"]),
            "mean_episode_reward": float(row["mean_episode_reward"]),
            "competence_supra_floor": bool(row["competence_supra_floor"]),
        })
        print(f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}", flush=True)

    return {"seed": int(seed), "guard": guard, "arms": arm_rows, "anchors": anchor_rows}


# --------------------------------------------------------------------------------------
def run_experiment(
    seeds: List[int],
    zworld_p0: int,
    p0: int,
    p1: int,
    ppo_eps: int,
    eval_eps: int,
    steps: int,
    rollout: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    contexts = _arm_contexts()
    # Design-time refusal BEFORE compute: catches a gate no arm could pass from its own
    # pre-registered config, for free at queue time (the V3-EXQ-785 arithmetic).
    audited = assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, contexts, arm_id_key="id")
    print(
        f"structural-vacuity audit: {len(audited)} (spec, arm) pairs checked, "
        f"no unsatisfiable gate", flush=True,
    )

    seed_rows = [
        _run_seed(s, zworld_p0, p0, p1, ppo_eps, eval_eps, steps, rollout, dry_run)
        for s in seeds
    ]

    guards = [r["guard"] for r in seed_rows]
    for g in guards:
        g["cell_id"] = f"warmup|seed{g['seed']}"

    all_arm_rows: Dict[str, List[Dict[str, Any]]] = {
        aid: [r["arms"][aid] for r in seed_rows] for aid in ARM_IDS
    }
    all_anchor_rows = [row for r in seed_rows for row in r["anchors"]]

    per_arm: Dict[str, Dict[str, Any]] = {}
    for aid in ARM_IDS:
        rows = all_arm_rows[aid]
        n_supra = int(sum(1 for r in rows if r["competence_supra_floor"]))
        per_arm[aid] = {
            "arm_id": aid,
            "n_seeds": len(rows),
            "foraging_competence_mean": round(_mean([r["foraging_competence"] for r in rows]), 6),
            "foraging_competence_per_seed": [round(r["foraging_competence"], 6) for r in rows],
            "survival_horizon_mean": round(_mean([r["survival_horizon"] for r in rows]), 6),
            "n_seeds_supra_floor": n_supra,
            "majority_supra_floor": bool(n_supra >= (len(rows) + 1) // 2) if rows else False,
            "train_consumption_share_mean": round(
                _mean([r["train_consumption_share"] for r in rows]), 6),
            "train_consumption_share_per_seed": [
                round(r["train_consumption_share"], 6) for r in rows],
            "per_seed_cells": rows,
        }

    per_anchor: Dict[str, Dict[str, Any]] = {}
    for aid in ANCHOR_IDS:
        rows = [r for r in all_anchor_rows if r["anchor_id"] == aid]
        per_anchor[aid] = {
            "anchor_id": aid,
            "n_seeds": len(rows),
            "foraging_competence_mean": round(_mean([r["foraging_competence"] for r in rows]), 6),
            "foraging_competence_per_seed": [round(r["foraging_competence"], 6) for r in rows],
            "survival_horizon_mean": round(_mean([r["survival_horizon"] for r in rows]), 6),
            "per_seed_cells": rows,
        }

    guard_measured, guard_cell = _worst_cell(guards, "world_encoder_max_abs_delta", "min")
    oracle_measured, oracle_cell = _worst_cell(
        [r for r in all_anchor_rows if r["anchor_id"] == "greedy_oracle"],
        "foraging_competence", "min")
    lvg_measured, lvg_cell = _worst_cell(
        [r for r in all_anchor_rows if r["anchor_id"] == "local_view_greedy"],
        "foraging_competence", "min")

    arm_gates: List[Dict[str, Any]] = []
    for ctx in contexts:
        aid = str(ctx["id"])
        rows = all_arm_rows[aid]
        cshare_measured, cshare_cell = _worst_cell(rows, "train_consumption_share", "min")
        measured = {
            "zworld_encoder_trained_in_p0": guard_measured,
            "d3_oracle_clears_floor": oracle_measured,
            "d3_local_view_greedy_clears_floor": lvg_measured,
            "consumption_share_dominates_under_correction": cshare_measured,
        }
        gate = evaluate_arm_gate(aid, ctx, PRECONDITIONS, measured)
        gate["offending_cells"] = {
            "zworld_encoder_trained_in_p0": guard_cell,
            "d3_oracle_clears_floor": oracle_cell,
            "d3_local_view_greedy_clears_floor": lvg_cell,
            "consumption_share_dominates_under_correction": cshare_cell,
        }
        arm_gates.append(gate)

    agg = aggregate_arm_gates(arm_gates)

    latent_green = "ppo_ree_latent" in agg["green_arms"]
    raw_green = "ppo_raw_obs" in agg["green_arms"]
    latent_clears = bool(per_arm["ppo_ree_latent"]["majority_supra_floor"])
    raw_clears = bool(per_arm["ppo_raw_obs"]["majority_supra_floor"])

    readiness_met = bool(agg["any_green"])
    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif latent_green and latent_clears:
        outcome, label = "PASS", "objective_correction_recovers_latent_policy"
    elif latent_green and (not latent_clears) and raw_green and raw_clears:
        outcome, label = "FAIL", "objective_correction_recovers_raw_obs_only_latent_lossy"
    elif latent_green and not latent_clears:
        outcome, label = "FAIL", "policy_learning_isolated_after_objective_correction"
    elif (not latent_green) and raw_green and raw_clears:
        outcome, label = "FAIL", "objective_correction_recovers_raw_obs_encoder_not_ready"
    else:
        outcome, label = "FAIL", "policy_learning_isolated_raw_obs_only_latent_encoder_not_ready"

    criteria_nd = arm_criteria_non_degenerate(
        {
            "ppo_ree_latent": ["C_ppo_ree_latent_clears_floor_at_D3_corrected_objective"],
            "ppo_raw_obs": ["C_ppo_raw_obs_clears_floor_at_D3_corrected_objective"],
        },
        agg,
    )

    interpretation = {
        "label": label,
        "declared_null": (
            "Neither ppo_ree_latent nor ppo_raw_obs clears the 1.0 competence floor (strict "
            "majority of seeds) at hazard-free D3 under the W3_survival_zeroed "
            "(consumption-only, imported verbatim from V3-EXQ-808) objective -> "
            "policy_learning_isolated_after_objective_correction. The failure can no longer "
            "be attributed to a survival-dominated objective -- deleted BY CONSTRUCTION, not "
            "merely by measurement -- so it isolates policy-learning inadequacy (or a deeper "
            "obstruction) as the surviving candidate explanation. This does NOT prove "
            "H-policy-learning outright; it closes the objective-misspecification confound "
            "on the POLICY axis specifically. H-objective-misspecification itself is "
            "adjudicated on the reward axis by its own dedicated probe, V3-EXQ-808, "
            "independently of this script's outcome."
        ),
        "alternative_outcome_note": (
            "If ppo_ree_latent (or ppo_raw_obs) DOES clear the floor once the objective is "
            "corrected, that WEAKENS H-policy-learning and further corroborates "
            "H-objective-misspecification from the policy side: a real actor on the "
            "representation CAN forage; 737b's original null was an artifact of the "
            "mis-set reward, not of the learner or the representation."
        ),
        "sibling_probe_independence_note": (
            "This probe and V3-EXQ-808 are independent legs on different design axes "
            "(policy vs reward) of the same GOV-FANOUT-1 portfolio. Neither one's outcome "
            "is a precondition for the other; this script does not read or depend on "
            "V3-EXQ-808's manifest, and vice versa."
        ),
        "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
        "preconditions": agg["adjudication_preconditions"],
        "criteria_non_degenerate": criteria_nd,
        "criteria": [
            {
                "name": "C_ppo_ree_latent_clears_floor_at_D3_corrected_objective",
                "load_bearing": True,
                "passed": bool(latent_clears),
                "measured": per_arm["ppo_ree_latent"]["foraging_competence_mean"],
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
            },
            {
                "name": "C_ppo_raw_obs_clears_floor_at_D3_corrected_objective",
                "load_bearing": True,
                "passed": bool(raw_clears),
                "measured": per_arm["ppo_raw_obs"]["foraging_competence_mean"],
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
            },
        ],
    }

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "non_degenerate": bool(agg["non_degenerate"]),
        "degeneracy_reason": agg["degeneracy_reason"],
        "per_arm_gate": agg["per_arm_gate"],
        "per_arm": per_arm,
        "per_anchor": per_anchor,
        "per_seed_arms": {aid: all_arm_rows[aid] for aid in ARM_IDS},
        "per_seed_anchors": all_anchor_rows,
        "diagnostics": {
            "zworld_encoder_guard": {
                "policy": "green_gating_latent_arm_only",
                "policy_reason": (
                    "The ppo_ree_latent arm's verdict feeds an 'isolates policy-learning' "
                    "reading this script does not want to license on a frozen random "
                    "projection, so the guard gates that arm specifically (scoped OUT of "
                    "ppo_raw_obs via applies_to). Mirrors V3-EXQ-808's choice rather than "
                    "737b's own record-not-refuse stance."
                ),
                "n_cells": len(guards),
                "worst_cell_max_abs_delta": round(guard_measured, 9),
                "worst_cell": guard_cell,
                "all_trained": bool(all(
                    float(g.get("world_encoder_max_abs_delta", 0.0)) > ZWORLD_DELTA_FLOOR
                    for g in guards)),
                "per_cell": guards,
            },
        },
        "headline": {
            "ppo_ree_latent_forage": per_arm["ppo_ree_latent"]["foraging_competence_mean"],
            "ppo_ree_latent_clears_majority": latent_clears,
            "ppo_raw_obs_forage": per_arm["ppo_raw_obs"]["foraging_competence_mean"],
            "ppo_raw_obs_clears_majority": raw_clears,
            "anchor_competence": {
                aid: per_anchor[aid]["foraging_competence_mean"] for aid in ANCHOR_IDS},
            "readiness_met": readiness_met,
            "latent_arm_green": latent_green,
            "raw_obs_arm_green": raw_green,
        },
    }


# --------------------------------------------------------------------------------------
def _build_manifest(result: Dict[str, Any], timestamp_utc: str, cfg: Dict[str, Any],
                    dry_run: bool) -> Dict[str, Any]:
    return {
        "run_id": f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "brake_exempt": True,
        "brake_exempt_reason": (
            "GOV-FANOUT-1 discrimination probe; claim_ids=[]; promotes/demotes nothing and "
            "adds no ceiling reading to MECH-457 (the source autopsy records this explicitly "
            "for both suggested probes)"
        ),
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation"]["label"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "diagnostics": result["diagnostics"],
        "headline": result["headline"],
        "per_arm": result["per_arm"],
        "per_anchor": result["per_anchor"],
        "per_seed_arms": result["per_seed_arms"],
        "per_seed_anchors": result["per_seed_anchors"],
        "config": cfg,
        "hypothesis_space": {
            "question_id": "conversion_ceiling_root",
            "leg": "H-policy-learning",
            "axis": "policy",
            "role": "discriminating probe (GOV-FANOUT-1); policy-axis sibling of V3-EXQ-808 (reward axis)",
            "source_autopsy": (
                "failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22"
            ),
            "sibling_probe": "V3-EXQ-808",
            "reading": (
                "PASS (ppo_ree_latent clears the floor) WEAKENS H-policy-learning and "
                "further corroborates H-objective-misspecification from the policy side. "
                "FAIL/declared-null (neither arm clears) ISOLATES policy-learning as the "
                "surviving candidate explanation but does not itself PROVE it -- a deeper "
                "obstruction (PPO under-powered; a representation ceiling this two-arm "
                "design cannot rule out) remains possible per 737b's own grid."
            ),
        },
        "load_bearing_dv": (
            "C_ppo_ree_latent_clears_floor_at_D3_corrected_objective and "
            "C_ppo_raw_obs_clears_floor_at_D3_corrected_objective: D3 foraging_competence "
            "mean (strict majority of seeds) vs the 1.0 competence floor, under the fixed "
            "W3_survival_zeroed objective, for each of the two arms."
        ),
        "notes": (
            "Policy-axis GOV-FANOUT-1 probe for H-policy-learning (conversion_ceiling_root), "
            "sibling of V3-EXQ-808 (reward axis). Re-runs V3-EXQ-737b's two-arm design "
            "(ppo_ree_latent PPO actor on the frozen REE z_world latent + matched "
            "ppo_raw_obs control) at the SAME hazard-free D3 rung and 724-A0 recipe, but "
            "trains BOTH arms under V3-EXQ-808's W3_survival_zeroed (consumption-only) "
            "reward instead of 737b's flat-sum reward, reusing 808's reward-decomposition "
            "trainer verbatim rather than reimplementing reward-shaping code. DIAGNOSTIC: "
            "promotes and demotes nothing; route to /failure-autopsy before any governance "
            "use. Does NOT eliminate H-policy-learning on a null; does NOT depend on "
            "V3-EXQ-808's own (independent) outcome."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-813 policy-axis GOV-FANOUT-1 probe for H-policy-learning "
            "(conversion_ceiling_root; diagnostic; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        zworld_p0, p0, p1, ppo = DRY_RUN_ZWORLD_P0, DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_PPO
        eval_eps, steps, rollout = DRY_RUN_EVAL, DRY_RUN_STEPS, DRY_RUN_ROLLOUT
    else:
        seeds = list(SEEDS)
        zworld_p0, p0, p1, ppo = (
            ZWORLD_P0_EPISODES, P0_WARMUP_EPISODES, P1_REINFORCE_EPISODES, P1_PPO_EPISODES)
        eval_eps, steps, rollout = EVAL_EPISODES, STEPS_PER_EPISODE, PPO_ROLLOUT_EPISODES

    cfg: Dict[str, Any] = {
        "seeds": seeds,
        "rung": RUNG_ID,
        "rung_overrides": RUNG["overrides"],
        "env_kwargs": {k: v for k, v in x734._env_kwargs_for_rung(RUNG).items()
                       if isinstance(v, (int, float, bool, str)) or v is None},
        "arms": ARM_IDS,
        "anchors": ANCHOR_IDS,
        "weighting_level_id": LEVEL_ID,
        "w_consume": W_CONSUME,
        "w_survival": W_SURVIVAL,
        "zworld_p0_episodes": zworld_p0,
        "p0_warmup_episodes": p0,
        "p1_reinforce_episodes": p1,
        "p1_ppo_episodes": ppo,
        "eval_episodes": eval_eps,
        "steps_per_episode": steps,
        "ppo_rollout_episodes": rollout,
        "ppo_lr": float(x734.PPO_LR),
        "forage_bonus": float(x734.FORAGE_BONUS),
        "novelty_coef": float(x734.NOVELTY_COEF),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "zworld_delta_floor": float(ZWORLD_DELTA_FLOOR),
        "consumption_share_floor": float(CONSUMPTION_SHARE_FLOOR),
        "term_families": list(x808.TERM_FAMILIES),
        "survival_linked_families": list(x808.SURVIVAL_LINKED_FAMILIES),
    }

    result = run_experiment(
        seeds=seeds, zworld_p0=zworld_p0, p0=p0, p1=p1, ppo_eps=ppo,
        eval_eps=eval_eps, steps=steps, rollout=rollout, dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, cfg, dry_run=bool(args.dry_run))

    out_dir = (Path(args.out_dir) if args.out_dir is not None
               else REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments")
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=cfg, seeds=seeds, script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _started).total_seconds(),
    )

    hl = result["headline"]
    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"readiness_met={hl['readiness_met']} "
        f"ppo_ree_latent/ep={hl['ppo_ree_latent_forage']} "
        f"(clears={hl['ppo_ree_latent_clears_majority']}, green={hl['latent_arm_green']}) "
        f"ppo_raw_obs/ep={hl['ppo_raw_obs_forage']} "
        f"(clears={hl['ppo_raw_obs_clears_majority']}, green={hl['raw_obs_arm_green']})",
        flush=True,
    )
    for aid in ANCHOR_IDS:
        pa = result["per_anchor"][aid]
        print(
            f"  ANCHOR {aid}: forage/ep={pa['foraging_competence_mean']} "
            f"survival={pa['survival_horizon_mean']}", flush=True,
        )
    pag = result["per_arm_gate"]
    print(
        f"  gate: green={pag['green_arms']} red={pag['red_arms']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, (str(out_path) if not args.dry_run else None), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
