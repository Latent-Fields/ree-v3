#!/opt/local/bin/python3
"""
V3-EXQ-734 -- ENV-DIFFICULTY COMPETENCE-RECOVERY SWEEP (Track-1a, post-724 campaign).

WHY THIS EXISTS (a DIAGNOSTIC, not an evidence falsifier; the self-routed label below is a
HYPOTHESIS the adjudication pipeline can falsify, NOT a governance verdict; claim_ids=[],
experiment_purpose=diagnostic, non_contributory -- PROMOTES / DEMOTES NOTHING).

V3-EXQ-724 (competence-localization OFAT) held ENV DIFFICULTY, DRIVE/reward balance, and the
LEARNER/RL-regime FIXED and varied only training DURATION, encoder-thaw, and mechanism-count:
NO single-factor arm -- and not even the A4 recovery ceiling (minimal stack + P1-long +
e2-unfrozen) -- recovered foraging above the 1.0 res/ep floor (competence_deficit_diffuse,
recovery_ceiling_supra=false). V3-EXQ-728 landed the trained all-ON capability point: it
SURVIVES well (survival_horizon normalized +6.07) but does NOT forage (foraging_competence
0.167 res/ep, BELOW the random-walk floor 0.267) -- a forage-vs-survive INVERSION: with
hazard_food_attraction=0.7 + reef_bipartite, food sits in danger and every arm learned pure
avoidance.

724 varied everything EXCEPT the env. This experiment holds the 724-A0 all-ON recipe FIXED
and sweeps ENV DIFFICULTY DOWN from the 724 config toward a de-risked env, asking the PIVOTAL
FORK: is the foraging deficit an ENV-DIFFICULTY ceiling (buildable via curriculum / reward
shaping) or a substrate/learner ceiling?

  FORK. foraging recovers above the 1.0 floor at SOME difficulty => env/curriculum-buildable
  (route to /implement-substrate on a curriculum); never recovers even in an easy/hazard-free
  env => a deeper substrate/learner problem.

BRAKE-EXEMPT. This asks a DIFFERENT question ("is the foraging deficit env-difficulty-driven
or substrate/learner-driven") than any conversion-ceiling claim, and tags NO claim -- so the
/failure-autopsy re-derive brake does not apply. It is a COMPETENCE/ENV diagnostic, NOT a
conversion or de-commit falsifier -- the conversion_ceiling_campaign_plan.md re-derive brakes
do NOT apply. Maps to node conversion_ceiling_campaign:CAMPAIGN (competence-localization gate).

------------------------------------------------------------------------------------------
DESIGN -- monotone difficulty STAIRCASE x 4 policy arms x SEEDS, capability_eval yardstick DV.

DIFFICULTY STAIRCASE (each rung strictly <= the previous in every risk axis; sweeps the four
task axes hazard_food_attraction 0.7->0.0, proximity_harm_scale 0.1->0.0, reef_bipartite_layout
True->False, num_hazards 4->0, plus reef_enabled at the hazard-free rung so "hazard-free" is
genuine -- otherwise reef contact stays lethal):

  D0_baseline_724     hazard_food_attraction=0.7 proximity_harm_scale=0.1 reef_bipartite=True
                      reef_enabled=True num_hazards=4  -- IDENTICAL to the 724/728 env; the
                      control rung that must REPRODUCE the incompetence (all-ON below floor).
  D1_food_decoupled   hazard_food_attraction=0.0 (only) -- food no longer biased into danger;
                      the single axis 728 most implicated in the forage-vs-survive inversion.
  D2_proximity_layout hazard_food_attraction=0.0 proximity_harm_scale=0.0 reef_bipartite=False
                      -- + no proximity-harm gradient + food no longer spatially segregated.
  D3_hazard_free      hazard_food_attraction=0.0 proximity_harm_scale=0.0 reef_bipartite=False
                      reef_enabled=False n_reef_patches=0 num_hazards=0 -- maximally de-risked
                      pure foraging. The DECISIVE rung: if foraging does not recover even HERE,
                      the deficit is not env-difficulty.

FOUR POLICY ARMS per rung (all measured on the identical capability_eval yardstick, mirroring
V3-EXQ-728; a uniform 4-metric readout: foraging_competence / survival_horizon / goal_reach_rate
/ planning_depth):

  random_walk         FLOOR anchor (uniform-random actions; capability_eval.RandomPolicy).
  ree_trained_allon   the SUBSTRATE arm. All-ON REE stack (714 ARM_ON) TRAINED ON THIS RUNG'S
                      ENV with the 724-A0 recipe (P0=200 world-model warmup THEN P1=90 two-head
                      REINFORCE, SD-056 e2 encoder FROZEN through P1), then evaluated via the
                      yardstick's mechanism-agnostic REEForwardPolicy. Trained+evaluated MATCHED
                      per rung (this tests whether training on an EASIER env lets it LEARN to
                      forage, NOT transfer).
  vanilla_ppo         LEARNER-ADEQUACY control (folds in the V3-EXQ-732a "learner inadequate"
                      finding). A NON-REE matched-capacity PPO actor-critic reading the RAW
                      observation vector (body_state (+) world_state (+) harm channels) -- the
                      SAME observation interface the REE encoder senses, so ree-vs-ppo is a fair
                      same-observability comparison. Trained on THIS RUNG'S env with clipped-
                      surrogate PPO + GAE + count-based novelty + running-std reward scaling
                      (vendored from 732a's power-fixed learner); greedy-argmax eval, DV counts
                      REAL env resource transitions (unshaped). This is deliberately NOT the
                      privileged-global oracle -- it addresses the 732a observability confound
                      (the oracle reads all resource coords; the learner sees only the env obs).
  greedy_oracle       CEILING / achievability anchor (nearest-resource greedy forager, no agent;
                      capability_eval.OraclePolicy). READINESS gate PER RUNG.

DV (load-bearing): foraging_competence = capability_eval mean resources/episode (env.step info
transition_type=='resource'). Reported alongside: survival_horizon (confirm the 728 forage-vs-
survive inversion RELAXES as food de-risks) + goal_reach_rate + planning_depth.

READINESS (self-routing gate; match 724/728 convention) -- PER ENV CELL: the greedy oracle
must clear the 1.0 floor at each rung (proving the floor is ACHIEVABLE in that env). Below-floor
at a rung => that cell is substrate_not_ready for the recovery read. Global readiness also
requires D0 to REPRODUCE the incompetence (all-ON below floor on the hardest rung -- else the
premise is not reproduced) and enough eval episodes per cell.

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, not a verdict; refines the task fork with the PPO
learner-adequacy contrast so the "deeper problem" branch is itself decomposed):
  * READINESS fails (D0 oracle below floor, OR all-ON already forages >= floor at D0, OR too
    few eval episodes) -> `substrate_not_ready_requeue`. Draw NO conclusion.
  * RECOVERS: readiness holds AND ree_trained_allon clears the floor on a strict majority of
    seeds at SOME de-risked rung (oracle-achievable) -> `env_difficulty_recoverable` (PASS).
    The deficit is an env-difficulty ceiling. Route to /implement-substrate on a CURRICULUM
    (report the hardest recovering rung).
  * REE-CEILING: readiness holds, ree_trained_allon NEVER clears the floor, BUT the matched
    vanilla PPO DOES clear it at some de-risked rung -> `ree_substrate_ceiling` (FAIL). A
    standard learner forages the de-risked env on the SAME observation interface but the REE
    stack cannot -> a REE-specific substrate/representation ceiling, not env-buildable for REE.
  * LEARNER/OBS-CEILING: readiness holds (incl. the hazard-free D3 oracle clearing the floor),
    and NEITHER ree_trained_allon NOR the matched vanilla PPO clears the floor even in the
    maximally de-risked D3 env -> `learner_or_observability_ceiling` (FAIL). No powered learner
    forages even a hazard-free env on this observation interface -> a deeper learner /
    observation-encoding problem (echoes the 732a observation-interface reading). Route to
    /implement-substrate on the observation encoding, NOT a curriculum.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement):
  * EVIDENCE the deficit is ENV-DIFFICULTY (env/curriculum-buildable): ree_trained_allon clears
    the floor at some de-risked rung while failing at D0.
  * EVIDENCE the deficit is a REE-SPECIFIC substrate ceiling: vanilla PPO clears the floor at a
    de-risked rung where ree_trained_allon never does.
  * EVIDENCE the deficit is a deeper LEARNER / OBSERVATION problem: neither learner clears the
    floor even in the hazard-free D3 env (oracle-achievable).
  * EVIDENCE AGAINST any conclusion (substrate_not_ready_requeue): D0 oracle cannot clear the
    floor, OR all-ON already forages >= floor at D0 (premise not reproduced), OR insufficient
    eval episodes.
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING and tags NO claim.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

UNTRAINED-WORLD-ENCODER GUARD (wired 2026-07-19; DETECTION ONLY).
  experiments/_lib/zworld_encoder_guard.py is wired into the `ree_trained_allon` arm only.
  NONE of the optimizer groups inside `_train_all_on_agent` (e2, lateral-PFC bias head, OFC
  devaluation head) covers ANY of the 61 latent_stack parameters, so
  split_encoder.world_encoder receives no gradient in P0 or P1 and z_world stays a FROZEN
  RANDOM PROJECTION at initialisation (measured: 0/61 latent_stack tensors changed,
  0/4 world_encoder tensors changed, max|delta| = 0.000e+00).

  POLICY = STRICT for this script. Its P0 phase is documented as a world-model warmup
  mirroring 728, and the `ree_trained_allon` arm is presented as a TRAINED REE agent compared
  against PPO / oracle / random. That arm's premise REQUIRES a learned world representation,
  so a frozen random projection VOIDS the premise rather than merely weakening it: the arm is
  REFUSED, not annotated.

  THE GUARD SITS AT THIS SCRIPT'S CALL SITE, NOT INSIDE `_train_all_on_agent`.
  v3_exq_737 and v3_exq_742 import this module's `_train_all_on_agent`; putting the guard
  inside that function would force one shared strictness on three drivers whose premises
  differ. Each driver therefore gates at its own call site with its own policy.

  ARM-SCOPED AND RUNG-SCOPED, NEVER RUN-SCOPED. The guard refuses the ARM on the RUNG where
  it fired -- never the RUN. `vanilla_ppo`, `greedy_oracle` and `random_walk` run no all-ON
  P0 warmup and their premise does not involve z_world, so the guard precondition is
  explicitly scoped OUT of them and their yardstick anchors stay valid and scored. Per the
  multi-arm regime-conditioning rule (.claude/skills/queue-experiment/SKILL.md, the
  V3-EXQ-785 defect), no arm's gate result may vacate another arm's, and no rung's may vacate
  another rung's: `non_degenerate` is ANY-(rung,arm)-green, not all-green.

  SCOPE: DETECTION ONLY. Nothing here attempts to make the encoder train; that fix is
  downstream of the V3-EXQ-783 adjudication and belongs to governance.
  Diagnosis: REE_assembly/evidence/planning/
             zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Sourced config (all-ON matched stack + A0 training harness + oracle/random/forward yardstick +
matched PPO learner):
  experiments/v3_exq_724_competence_localization_diagnostic.py  (A0 all-ON config builders +
    SD-056 e2 warmup + P1 two-head REINFORCE loss helpers + obs helpers; imported as x724),
  experiments/v3_exq_728_trained_allon_capability_point.py       (train-then-yardstick pattern),
  experiments/v3_exq_732a_policy_learning_discriminator.py       (matched PPO learner design),
  experiments/_lib/capability_eval.py                            (4-metric yardstick + anchors),
  ree_core/environment/causal_grid_world.py, ree_core/agent.py, ree_core/utils/config.py.
See REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md,
    REE_assembly/evidence/planning/ree_ai_design_critique_plan.md (WS-1/WS-3),
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-732a_2026-07-10.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    METRIC_KEYS,
    OraclePolicy,
    Policy,
    RandomPolicy,
    REEForwardPolicy,
    build_report,
    evaluate_seed,
    summarize_arm,
)
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.zworld_encoder_guard import (
    ZWorldEncoderUntrainedError,
    latent_stack_snapshot,
    latent_stack_weight_delta,
    assert_world_encoder_trained,
    zworld_precondition,
)
import experiments.v3_exq_724_competence_localization_diagnostic as x724
from ree_core.environment.causal_grid_world import CausalGridWorldV2


EXPERIMENT_TYPE = "v3_exq_734_env_difficulty_competence_recovery_sweep"
QUEUE_ID = "V3-EXQ-734"
CLAIM_IDS: List[str] = []                 # tags NO claim -- pure diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45]           # 42/43/44 base + 45 for power (potential-null fork)
ZWORLD_P0_EPISODES = 60            # P0a SD-070 z_world ENCODER warmup, run AHEAD of the e2
                                   # warmup below. 60 is SD-070's validated operating point
                                   # (exq783_zworld_granularity.OFF_P0_ENCODER_EPISODES); this
                                   # script's 200 steps/ep buffers ~12k observations against
                                   # the 9k that point was measured at, i.e. strictly more.
                                   # Set 0 to restore the pre-fix (frozen-encoder) behaviour.
P0_WARMUP_EPISODES = 200           # all-ON SD-056 e2 forward-model warmup (724 A0 recipe).
                                   # This warmup trains e2 ONLY -- no optimizer group here
                                   # covers a latent_stack parameter. It is P0a above, NOT
                                   # this stage, that trains split_encoder.world_encoder.
                                   # With ZWORLD_P0_EPISODES=0 z_world stays a frozen random
                                   # projection (the V3-EXQ-780 defect), which is what
                                   # zworld_encoder_guard detects -- see docstring block.
P1_REINFORCE_EPISODES = 90         # all-ON two-head REINFORCE (724 A0 recipe; e2 frozen in P1)
E2_TRAIN_IN_P1 = False             # A0 recipe: SD-056 e2 encoder FROZEN through P1
P1_PPO_EPISODES = 1000             # matched vanilla-PPO learner-adequacy budget per cell
EVAL_EPISODES = 20                 # capability-eval episodes per (rung, arm, seed) cell
STEPS_PER_EPISODE = 200

# Pre-registered floor (shared with 724 / capability_eval): a decisive forager clears
# >= 1.0 resource/episode comfortably; achievability validated per rung by the greedy oracle.
# (COMPETENCE_RESOURCE_FLOOR imported from capability_eval == 1.0.)
MIN_EVAL_EPISODES = 5              # per cell: below this the DV is not estimable

# ---------------------------------------------------------------------------
# Matched vanilla-PPO learner (raw-obs; vendored from V3-EXQ-732a's power-fixed learner).
# ---------------------------------------------------------------------------
PPO_TRUNK_HIDDEN = 128
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_ENTROPY_BETA = 0.03
PPO_VALUE_COEF = 0.5
PPO_GRAD_CLIP = 0.5
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
PPO_ROLLOUT_EPISODES = 8
FORAGE_BONUS = 1.0                 # dense training-reward per resource (DV eval is unshaped)
NOVELTY_COEF = 0.1                 # count-based exploration bonus for the sparse phase
REWARD_STD_EPS = 1e-6

# ---------------------------------------------------------------------------
# Dry-run budget (tiny; exercises the full staircase + self-route code path fast)
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = 2               # P0a: enough to exercise the real SD-070 training code
                                    # (the trainer's batch is scaled down for the smoke path)
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_PPO = 6                     # >= 2 rollouts at DRY rollout size so PPO update path runs
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 15
DRY_RUN_ROLLOUT_EPISODES = 3

# ---------------------------------------------------------------------------
# Difficulty staircase. Each rung = the 724 env with a monotone-de-risking override set.
# ---------------------------------------------------------------------------
DIFFICULTY_RUNGS: List[Dict[str, Any]] = [
    {"rung_id": "D0_baseline_724", "role": "control_reproduce", "overrides": {}},
    {"rung_id": "D1_food_decoupled", "role": "derisk",
     "overrides": {"hazard_food_attraction": 0.0}},
    {"rung_id": "D2_proximity_layout_derisked", "role": "derisk",
     "overrides": {"hazard_food_attraction": 0.0, "proximity_harm_scale": 0.0,
                   "reef_bipartite_layout": False}},
    {"rung_id": "D3_hazard_free", "role": "derisk_decisive",
     "overrides": {"hazard_food_attraction": 0.0, "proximity_harm_scale": 0.0,
                   "reef_bipartite_layout": False, "reef_enabled": False,
                   "n_reef_patches": 0, "num_hazards": 0}},
]
D0_RUNG_ID = DIFFICULTY_RUNGS[0]["rung_id"]
D3_RUNG_ID = DIFFICULTY_RUNGS[-1]["rung_id"]

ARMS = ("random_walk", "ree_trained_allon", "vanilla_ppo", "greedy_oracle")


def _env_kwargs_for_rung(rung: Dict[str, Any]) -> Dict[str, Any]:
    kw = dict(x724.ENV_KWARGS)
    kw.update(rung["overrides"])
    return kw


def _make_env(seed: int, env_kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **env_kwargs)


def _strict_majority(n: int) -> int:
    """Strict majority: floor(n/2)+1 (n=4->3, n=3->2, n=2->2). Conservative recovery bar."""
    return n // 2 + 1


# ---------------------------------------------------------------------------
# All-ON agent (714 ARM_ON via x724 config builders).
# ---------------------------------------------------------------------------
def _make_all_on_agent(env: CausalGridWorldV2):
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    cfg = x724.REEConfig.from_dims(**kwargs)
    return x724.REEAgent(cfg)


# ---------------------------------------------------------------------------
# Train the all-ON agent on a PROVIDED env with the 724-A0 recipe:
# P0a SD-070 z_world encoder warmup (OPT-IN) THEN P0b e2 forward-model warmup THEN P1 two-head
# REINFORCE (lateral-PFC bias + OFC devaluation).
#
# THE V3-EXQ-780 DEFECT AND ITS REMEDY. The three optimizer groups below (e2, lPFC bias, OFC
# devaluation) cover NO latent_stack parameter, so on this path split_encoder.world_encoder is
# never stepped and z_world stays a frozen random projection -- measured 0 of 61 latent_stack
# tensors changed at p0_episodes=200 on two independent drivers (V3-EXQ-737a, V3-EXQ-728).
# `zworld_p0_episodes > 0` adds the SD-070 recipe as a P0a stage AHEAD of the e2 warmup, which
# is the remedy adjudicated by V3-EXQ-783. See experiments/_lib/zworld_p0_warmup.py.
#
# ORDERING IS NOT ARBITRARY: e2 regresses on z_world, so the encoder must be trained BEFORE the
# e2 warmup. Training it afterwards would leave e2 fitted to the random projection -- the same
# defect one phase later.
#
# DEFAULT zworld_p0_episodes=0 IS EXACTLY THE PRIOR BEHAVIOUR, bit-identical: no extra tensor,
# no extra optimizer group, and no RNG draw (the warmup is RNG-neutral by construction and runs
# on its own env instance). Every existing caller is unaffected until it opts in.
#
# NOTE 737 and 742 import this function; the guard lives at each driver's own call site, not
# here, so each can set its own strictness.
# SD-056 e2 encoder FROZEN through P1. Mirrors V3-EXQ-728._train_all_on_agent, but the env is
# passed in (difficulty-parameterised). No P2 phase -- competence is measured downstream by the
# capability yardstick's REEForwardPolicy eval.
# ---------------------------------------------------------------------------
def _train_all_on_agent(
    agent,
    train_env: CausalGridWorldV2,
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    rung_id: str,
    total_denominator: int,
    zworld_p0_episodes: int = 0,
    zworld_p0_env: Optional[CausalGridWorldV2] = None,
    zworld_p0_dry_run: bool = False,
) -> Dict[str, Any]:
    env = train_env

    # -- P0a: SD-070 z_world encoder warmup (opt-in; see the header note) -------------------
    zworld_p0_stats: Dict[str, Any] = {"p0a_recipe": "sd070", "p0a_ran": False}
    if zworld_p0_episodes > 0:
        if zworld_p0_env is None:
            raise ValueError(
                "zworld_p0_episodes=%d requires zworld_p0_env: the warmup rollout consumes "
                "env RNG, so reusing train_env would shift the layout sequence P0b/P1 then "
                "see. Build a dedicated env with the same seed and kwargs."
                % (zworld_p0_episodes,)
            )
        zworld_p0_stats = run_zworld_p0(
            agent, zworld_p0_env, seed, zworld_p0_episodes, steps_per_episode,
            policy=RandomPolicy(seed), label=f"ree_allon rung={rung_id}",
            dry_run=zworld_p0_dry_run,
        )
    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=x724.E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=x724.LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(list(agent.ofc.devaluation_bias_head_parameters()), lr=x724.LR_OFC_DEVAL)
        if has_ofc else None
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=x724.TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_e2_train_steps = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    for ep in range(total_train_eps):
        is_p1 = (ep >= p1_start)
        is_p0 = not is_p1
        phase_label = "P1" if is_p1 else "P0"

        _flat, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=x724._obs_harm(obs_dict),
                obs_harm_a=x724._obs_harm_a(obs_dict),
                obs_harm_history=x724._obs_harm_history(obs_dict),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and has_lpfc and candidates and len(candidates) >= 2:
                cs = x724._consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 always; P1 only if the recipe unfreezes (A0: frozen).
            train_e2_now = is_p0 or (is_p1 and E2_TRAIN_IN_P1)
            if train_e2_now and (tick_in_ep % x724.E2_TRAIN_EVERY_K_TICKS == 0):
                if x724._e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng) is not None:
                    n_e2_train_steps += 1

            _flat, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=harm_signal, world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=max(0.0, 1.0 - energy),
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        # P1 end-of-episode: TWO-head REINFORCE (lateral-PFC bias + OFC devaluation).
        if is_p1 and (has_lpfc or has_ofc):
            reinforce_baseline = (
                x724.EMA_DECAY * reinforce_baseline + (1.0 - x724.EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > x724.OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-x724.OUTCOME_BUF_MAX:]
            if has_lpfc and bias_opt is not None:
                l_loss = x724._lpfc_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()
            if has_ofc and ofc_deval_opt is not None:
                ofc_loss = x724._ofc_deval_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
                if ofc_loss.requires_grad:
                    ofc_deval_opt.zero_grad()
                    ofc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.ofc.devaluation_bias_head_parameters(), 1.0)
                    ofc_deval_opt.step()

        cur = ep + 1
        if cur % 50 == 0 or cur == total_train_eps or phase_label == "P1":
            print(
                f"  [train] ree_allon rung={rung_id} seed={seed} phase={phase_label} "
                f"ep {cur}/{total_denominator}",
                flush=True,
            )

    return {
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_e2_train_steps": int(n_e2_train_steps),
        "zworld_p0": zworld_p0_stats,
    }


# ---------------------------------------------------------------------------
# Matched vanilla-PPO learner (raw-obs). Vendored from V3-EXQ-732a (power-fixed learner).
# ---------------------------------------------------------------------------
class PPOPolicyNet(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, hidden: int = PPO_TRUNK_HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class _RunningStd:
    """Welford running variance (no mean subtraction on apply -- preserves reward sign)."""

    def __init__(self) -> None:
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2.0:
            return 1.0
        return float(math.sqrt(self.m2 / (self.count - 1.0)))


def _novelty_bonus(counter: Dict[Tuple[int, int], int], pos: Tuple[int, int]) -> float:
    counter[pos] = counter.get(pos, 0) + 1
    return float(NOVELTY_COEF / math.sqrt(counter[pos]))


# Raw observation vector -- the SAME channels the REE encoder senses. Keys present are derived
# once per cell from a probe obs (tolerant to a channel being absent for a given env config).
_RAW_OBS_CANDIDATE_KEYS = ("body_state", "world_state", "harm_obs", "harm_obs_a", "harm_history")


def _raw_obs_keys_present(obs_dict: Dict[str, Any]) -> Tuple[str, ...]:
    keys = tuple(k for k in _RAW_OBS_CANDIDATE_KEYS if obs_dict.get(k) is not None)
    if "body_state" not in keys or "world_state" not in keys:
        raise KeyError("raw obs missing body_state/world_state")
    return keys


def _raw_obs_vector(obs_dict: Dict[str, Any], keys: Tuple[str, ...], device: torch.device) -> torch.Tensor:
    parts = [obs_dict[k].float().reshape(-1) for k in keys]
    return torch.cat(parts, dim=0).to(device).unsqueeze(0)


def _ppo_update(
    policy: PPOPolicyNet,
    optimiser: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    device: torch.device,
) -> None:
    n = states.shape[0]
    if n == 0:
        return
    adv = advantages
    if n > 1:
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, PPO_MINIBATCH_SIZE):
            idx = perm[start:start + PPO_MINIBATCH_SIZE]
            mb_states = states[idx]
            mb_actions = actions[idx]
            mb_old_logp = old_log_probs[idx]
            mb_returns = returns[idx]
            mb_adv = adv[idx]
            logits, values = policy(mb_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (mb_returns - values).pow(2).mean()
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_BETA * entropy
            if not torch.isfinite(loss):
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), PPO_GRAD_CLIP)
            optimiser.step()


def _compute_gae(
    rewards: List[float],
    values: List[float],
    bootstrap_value: float,
    terminal: bool,
) -> Tuple[List[float], List[float]]:
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0 if terminal else bootstrap_value
            next_nonterminal = 0.0 if terminal else 1.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0
        delta = rewards[t] + PPO_GAMMA * next_value * next_nonterminal - values[t]
        last_gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


def _train_ppo(
    env: CausalGridWorldV2,
    obs_keys: Tuple[str, ...],
    policy: PPOPolicyNet,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    n_episodes: int,
    rollout_episodes: int,
    steps_per_episode: int,
    rung_id: str,
    seed: int,
    total_denominator: int,
) -> None:
    reward_std = _RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    episodes_done = 0

    def state_fn(od: Dict[str, Any]) -> torch.Tensor:
        return _raw_obs_vector(od, obs_keys, device)

    while episodes_done < n_episodes:
        batch_states: List[torch.Tensor] = []
        batch_actions: List[int] = []
        batch_old_logp: List[float] = []
        batch_returns: List[float] = []
        batch_advantages: List[float] = []

        eps_this_batch = min(rollout_episodes, n_episodes - episodes_done)
        for _b in range(eps_this_batch):
            _, obs_dict = env.reset()
            ep_states: List[torch.Tensor] = []
            ep_actions: List[int] = []
            ep_logp: List[float] = []
            ep_values: List[float] = []
            ep_rewards: List[float] = []
            terminal = False
            bootstrap_value = 0.0
            for _step in range(steps_per_episode):
                state = state_fn(obs_dict)
                with torch.no_grad():
                    logits, value = policy(state)
                    dist = torch.distributions.Categorical(logits=logits.reshape(1, -1))
                    a = dist.sample()
                    logp = dist.log_prob(a)
                a_idx = int(a.item())
                _, harm_signal, done, info, obs_dict = env.step(a_idx)
                ttype = str(info.get("transition_type", "none"))
                pos = (int(env.agent_x), int(env.agent_y))
                shaped = (
                    float(harm_signal)
                    + (FORAGE_BONUS if ttype == "resource" else 0.0)
                    + _novelty_bonus(novelty_counter, pos)
                )
                reward_std.update(shaped)
                ep_states.append(state.reshape(-1).detach())
                ep_actions.append(a_idx)
                ep_logp.append(float(logp.item()))
                ep_values.append(float(value.item()))
                ep_rewards.append(shaped)
                if done:
                    terminal = True
                    break
            if not terminal:
                with torch.no_grad():
                    _, bv = policy(state_fn(obs_dict))
                bootstrap_value = float(bv.item())
            scale = reward_std.std + REWARD_STD_EPS
            scaled_rewards = [r / scale for r in ep_rewards]
            advs, rets = _compute_gae(scaled_rewards, ep_values, bootstrap_value, terminal)
            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_old_logp.extend(ep_logp)
            batch_returns.extend(rets)
            batch_advantages.extend(advs)
            episodes_done += 1
            cur = episodes_done
            if episodes_done % 200 == 0 or episodes_done == n_episodes:
                print(
                    f"  [train] vanilla_ppo rung={rung_id} seed={seed} phase=P1 "
                    f"ep {cur}/{total_denominator}", flush=True,
                )

        if not batch_states:
            continue
        states_t = torch.stack(batch_states).to(device)
        actions_t = torch.tensor(batch_actions, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(batch_old_logp, dtype=torch.float32, device=device)
        returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
        _ppo_update(policy, optimiser, states_t, actions_t, old_logp_t, returns_t, adv_t, device)


class PPOEvalPolicy(Policy):
    """Greedy (argmax) eval wrapper for a trained PPO net, over the raw obs interface.

    Wraps a trained PPOPolicyNet as a capability_eval.Policy so evaluate_seed measures the SAME
    four claim-agnostic metrics on it as on every other arm. Eval is greedy and unshaped -- the
    DV counts REAL env resource transitions inside capability_eval.rollout_episode.
    """

    name = "vanilla_ppo"

    def __init__(self, policy_net: PPOPolicyNet, obs_keys: Tuple[str, ...], device: torch.device) -> None:
        self.policy_net = policy_net
        self.obs_keys = obs_keys
        self.device = device

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        state = _raw_obs_vector(obs_dict, self.obs_keys, self.device)
        with torch.no_grad():
            logits, _v = self.policy_net(state)
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits.reshape(-1), dim=-1).item())


# ---------------------------------------------------------------------------
# Per-cell (rung x arm x seed) run: build the policy (training the learner arms on THIS rung's
# env), then measure the four capability metrics via the shared yardstick.
# ---------------------------------------------------------------------------
def _run_cell(
    rung: Dict[str, Any],
    arm_id: str,
    seed: int,
    env_kwargs: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p1_ppo_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    rollout_episodes: int,
    zworld_p0_episodes: int = 0,
    zworld_p0_dry_run: bool = False,
) -> Dict[str, Any]:
    device = torch.device("cpu")
    rung_id = rung["rung_id"]
    train_stats: Dict[str, Any] = {}
    guard_report: Optional[Dict[str, Any]] = None
    guard_ok: Optional[bool] = None
    guard_message: Optional[str] = None

    if arm_id == "random_walk":
        policy: Policy = RandomPolicy(seed)
    elif arm_id == "greedy_oracle":
        policy = OraclePolicy()
    elif arm_id == "ree_trained_allon":
        train_env = _make_env(seed, env_kwargs)
        agent = _make_all_on_agent(train_env)
        # z_world untrained-encoder guard -- DETECTION ONLY, scoped to THIS arm on THIS rung.
        # The guard lives here rather than inside _train_all_on_agent because 737/742 import
        # that function and carry different premises (see module docstring).
        before = latent_stack_snapshot(agent)
        train_stats = _train_all_on_agent(
            agent, train_env, seed, p0_episodes, p1_episodes, steps_per_episode,
            rung_id, p0_episodes + p1_episodes,
            zworld_p0_episodes=zworld_p0_episodes,
            # A DEDICATED env instance, same seed and kwargs: the P0a rollout consumes env RNG,
            # so running it on train_env would shift the layout sequence P0b/P1 then see.
            zworld_p0_env=(
                _make_env(seed, env_kwargs) if zworld_p0_episodes > 0 else None
            ),
            zworld_p0_dry_run=zworld_p0_dry_run,
        )
        guard_report = latent_stack_weight_delta(agent, before)
        guard_report["p0_episodes"] = int(p0_episodes)
        guard_report["guard_checked"] = bool(p0_episodes > 0 and before)
        # Exercise the guard's real raising path, but catch it here: a bare raise would abort
        # the process, yielding a runner ERROR with NO manifest and destroying the valid
        # vanilla_ppo / greedy_oracle / random_walk arms. Refuse the ARM, never the RUN.
        try:
            assert_world_encoder_trained(
                agent, before, p0=p0_episodes, strict=True,
                context=f"V3-EXQ-734a ree_trained_allon rung={rung_id}",
            )
            guard_ok = True
        except ZWorldEncoderUntrainedError as exc:
            guard_ok = False
            guard_message = str(exc)
            print(
                f"[GUARD-REFUSAL] arm=ree_trained_allon rung={rung_id} seed={seed}: "
                f"{guard_message}",
                file=sys.stderr, flush=True,
            )
            print(
                f"[GUARD-REFUSAL] arm=ree_trained_allon rung={rung_id} seed={seed}: "
                f"{guard_message}",
                flush=True,
            )
        policy = REEForwardPolicy(agent, name="ree_trained_allon")
    elif arm_id == "vanilla_ppo":
        train_env = _make_env(seed, env_kwargs)
        _, probe_obs = train_env.reset()
        obs_keys = _raw_obs_keys_present(probe_obs)
        in_dim = int(sum(int(probe_obs[k].reshape(-1).shape[0]) for k in obs_keys))
        policy_net = PPOPolicyNet(in_dim, int(train_env.action_dim)).to(device)
        ppo_opt = torch.optim.Adam(policy_net.parameters(), lr=PPO_LR)
        _train_ppo(
            train_env, obs_keys, policy_net, ppo_opt, device,
            p1_ppo_episodes, rollout_episodes, steps_per_episode,
            rung_id, seed, p1_ppo_episodes,
        )
        policy_net.eval()
        policy = PPOEvalPolicy(policy_net, obs_keys, device)
        train_stats = {"n_ppo_train_episodes": int(p1_ppo_episodes)}
    else:
        raise ValueError(f"unknown arm {arm_id!r}")

    eval_env = _make_env(seed, env_kwargs)
    row = evaluate_seed(policy, eval_env, eval_episodes, steps_per_episode)
    row["rung_id"] = rung_id
    row["rung_role"] = rung["role"]
    row["arm_id"] = arm_id
    row["seed"] = int(seed)
    row["train_stats"] = train_stats
    # Other arms run no all-ON warmup: record None rather than fabricating a report.
    row["zworld_guard"] = guard_report
    row["zworld_guard_ok"] = guard_ok
    if guard_message is not None:
        row["zworld_guard_message"] = guard_message
    return row


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p1_ppo_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    rollout_episodes: int,
    dry_run: bool,
    zworld_p0_episodes: int = 0,
) -> Dict[str, Any]:
    print(
        f"Env-difficulty competence-recovery sweep ({len(DIFFICULTY_RUNGS)} rungs x "
        f"{len(ARMS)} arms x {len(seeds)} seeds; P0a={zworld_p0_episodes}, "
        f"P0={p0_episodes}, P1={p1_episodes}, "
        f"PPO={p1_ppo_episodes}, eval={eval_episodes}, steps={steps_per_episode}, "
        f"dry_run={dry_run})",
        flush=True,
    )

    # ----- rung x arm x seed cells -----
    per_rung_arm_summ: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_rung_report: Dict[str, Dict[str, Any]] = {}
    all_cells: List[Dict[str, Any]] = []
    min_eval_eps = None

    for rung in DIFFICULTY_RUNGS:
        rung_id = rung["rung_id"]
        env_kwargs = _env_kwargs_for_rung(rung)
        rung_cells: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
        for arm_id in ARMS:
            for s in seeds:
                print(f"Seed {s} Condition {rung_id}:{arm_id}", flush=True)
                is_learner = arm_id in ("ree_trained_allon", "vanilla_ppo")
                slice_cfg = {
                    "rung_id": rung_id,
                    "arm_id": arm_id,
                    "env_kwargs": dict(env_kwargs),
                    # P0a MUST appear in the fingerprinted slice: an arm trained with the
                    # SD-070 encoder warmup is a DIFFERENT arm from the frozen-random-
                    # projection arm of every prior run. Omitting it would let a pre-fix
                    # cached arm falsely satisfy a post-fix cell.
                    "zworld_p0_episodes": (
                        int(zworld_p0_episodes) if arm_id == "ree_trained_allon" else 0
                    ),
                    "p0_episodes": int(p0_episodes) if arm_id == "ree_trained_allon" else 0,
                    "p1_episodes": int(p1_episodes) if arm_id == "ree_trained_allon" else 0,
                    "e2_train_in_p1": bool(E2_TRAIN_IN_P1) if arm_id == "ree_trained_allon" else False,
                    "p1_ppo_episodes": int(p1_ppo_episodes) if arm_id == "vanilla_ppo" else 0,
                    "eval_episodes": int(eval_episodes),
                    "steps_per_episode": int(steps_per_episode),
                }
                with arm_cell(
                    s,
                    config_slice=slice_cfg,
                    script_path=Path(__file__),
                    config_slice_declared=True,
                ) as cell:
                    row = _run_cell(
                        rung, arm_id, s, env_kwargs,
                        p0_episodes, p1_episodes, p1_ppo_episodes,
                        eval_episodes, steps_per_episode, rollout_episodes,
                        zworld_p0_episodes=zworld_p0_episodes,
                        zworld_p0_dry_run=dry_run,
                    )
                    cell.stamp(row)
                rung_cells[arm_id].append(row)
                all_cells.append(row)
                n_eps = int(row.get("n_episodes", 0))
                min_eval_eps = n_eps if min_eval_eps is None else min(min_eval_eps, n_eps)
                print(
                    f"verdict: PASS (rung={rung_id} arm={arm_id} seed={s} "
                    f"forage/ep={row['foraging_competence']} "
                    f"survival={row['survival_horizon']} "
                    f"goal_reach={row['goal_reach_rate']} "
                    f"supra_floor={row['competence_supra_floor']})",
                    flush=True,
                )

        arm_summaries = {a: summarize_arm(rung_cells[a]) for a in ARMS}
        per_rung_arm_summ[rung_id] = arm_summaries
        per_rung_report[rung_id] = build_report(
            arm_summaries, floor="random_walk", ceiling="greedy_oracle"
        )

    if min_eval_eps is None:
        min_eval_eps = 0

    n_seeds = len(seeds)
    maj = _strict_majority(n_seeds)

    def _arm_supra_seeds(rung_id: str, arm_id: str) -> int:
        return int(per_rung_arm_summ[rung_id][arm_id].get("n_seeds_supra_floor", 0))

    def _arm_recovers(rung_id: str, arm_id: str) -> bool:
        return bool(_arm_supra_seeds(rung_id, arm_id) >= maj)

    def _oracle_clears(rung_id: str) -> bool:
        return bool(per_rung_report[rung_id]["readiness"]["oracle_clears_floor"])

    def _oracle_forage(rung_id: str) -> float:
        return float(per_rung_report[rung_id]["readiness"]["oracle_foraging_competence"])

    def _arm_forage(rung_id: str, arm_id: str) -> float:
        return float(per_rung_arm_summ[rung_id][arm_id].get("foraging_competence_mean", 0.0))

    # ----- readiness -----
    d0_oracle_ok = _oracle_clears(D0_RUNG_ID)
    d0_baseline_reproduces = bool(not _arm_recovers(D0_RUNG_ID, "ree_trained_allon"))
    d3_oracle_ok = _oracle_clears(D3_RUNG_ID)
    sufficient_eval = bool(min_eval_eps >= MIN_EVAL_EPISODES)
    all_oracle_clear = all(_oracle_clears(r["rung_id"]) for r in DIFFICULTY_RUNGS)
    readiness_met = bool(d0_oracle_ok and d0_baseline_reproduces and sufficient_eval)

    # ----- recovery read (de-risked rungs where the oracle clears the floor) -----
    derisked_ids = [r["rung_id"] for r in DIFFICULTY_RUNGS[1:] if _oracle_clears(r["rung_id"])]
    ree_recovery_rungs = [r for r in derisked_ids if _arm_recovers(r, "ree_trained_allon")]
    ppo_recovery_rungs = [r for r in derisked_ids if _arm_recovers(r, "vanilla_ppo")]
    ree_recovers = bool(ree_recovery_rungs)
    ppo_recovers = bool(ppo_recovery_rungs)
    # rungs are ordered hardest -> easiest; the HARDEST recovering rung needs least de-risking.
    hardest_ree_recovery_rung = ree_recovery_rungs[0] if ree_recovery_rungs else None
    ppo_beats_random_at_d3 = bool(
        _arm_forage(D3_RUNG_ID, "vanilla_ppo") > _arm_forage(D3_RUNG_ID, "random_walk")
    )
    ppo_control_informative = bool(ppo_recovers or ppo_beats_random_at_d3)

    # ----- z_world untrained-encoder gate: PER (rung, arm), never whole-run -----
    # The guard applies ONLY to ree_trained_allon (the sole arm running an all-ON P0 warmup
    # whose premise requires a learned world representation). Every other arm is scoped OUT
    # with an explicit reason, and a red ree arm on one rung vacates neither the other arms
    # on that rung nor the ree arm on any other rung.
    GUARDED_ARM = "ree_trained_allon"
    SCOPE_OUT_REASON = (
        "arm runs no all-ON P0 warmup and its premise does not involve z_world"
    )
    per_arm_gate: Dict[str, Any] = {}
    failed_preconditions_by_arm: Dict[str, List[str]] = {}
    guard_diagnostics: Dict[str, Any] = {}
    ree_guard_green_rungs: List[str] = []
    ree_guard_red_rungs: List[str] = []
    green_cell_keys: List[str] = []
    red_cell_keys: List[str] = []
    ree_guard_precondition_by_rung: Dict[str, Dict[str, Any]] = {}

    for rung in DIFFICULTY_RUNGS:
        rid = rung["rung_id"]
        for arm_id in ARMS:
            key = f"{rid}:{arm_id}"
            rows = [c for c in all_cells
                    if c.get("rung_id") == rid and c.get("arm_id") == arm_id]
            if arm_id != GUARDED_ARM:
                per_arm_gate[key] = {
                    "rung_id": rid,
                    "arm_id": arm_id,
                    "green": True,
                    "guard_applies": False,
                    "scope_reason": SCOPE_OUT_REASON,
                    "failed_preconditions": [],
                }
                green_cell_keys.append(key)
                continue
            reports = [c.get("zworld_guard") for c in rows if c.get("zworld_guard")]
            oks = [c.get("zworld_guard_ok") for c in rows]
            checked = [r for r in reports if r.get("guard_checked")]
            refused = [c for c in rows if c.get("zworld_guard_ok") is False]
            green = bool(oks) and all(o is not False for o in oks)
            per_arm_gate[key] = {
                "rung_id": rid,
                "arm_id": arm_id,
                "green": bool(green),
                "guard_applies": True,
                "n_cells": len(rows),
                "n_guard_checked": len(checked),
                "n_refused": len(refused),
                "refused_seeds": sorted(int(c.get("seed", -1)) for c in refused),
                "failed_preconditions": (
                    [] if green else [f"{key}:zworld_world_encoder_trained"]
                ),
                "guard_message": (
                    refused[0].get("zworld_guard_message") if refused else None
                ),
            }
            if not green:
                failed_preconditions_by_arm[key] = per_arm_gate[key]["failed_preconditions"]
                ree_guard_red_rungs.append(rid)
                red_cell_keys.append(key)
            else:
                ree_guard_green_rungs.append(rid)
                green_cell_keys.append(key)
            if reports:
                # Represent the rung with a REFUSED cell's report when any seed was refused:
                # a green seed's report would recompute `met` as true and false-clear the
                # rung in the all-red flat-list case.
                rep_for_precondition = (
                    refused[0].get("zworld_guard") if refused else reports[0]
                )
                ree_guard_precondition_by_rung[rid] = zworld_precondition(
                    rep_for_precondition,
                    arm=GUARDED_ARM,
                    context=(
                        f"V3-EXQ-734a ree_trained_allon rung={rid} "
                        f"(refused_seeds={sorted(int(c.get('seed', -1)) for c in refused)})"
                        if refused else
                        f"V3-EXQ-734a ree_trained_allon rung={rid}"
                    ),
                )
            guard_diagnostics[rid] = {
                str(int(c.get("seed", -1))): c.get("zworld_guard") for c in rows
            }

    any_ree_guard_red = bool(ree_guard_red_rungs)
    non_degenerate = bool(green_cell_keys)
    degeneracy_reason = None
    if not non_degenerate:
        degeneracy_reason = (
            "every (rung, arm) cell is RED -- refused: "
            + (", ".join(red_cell_keys) or "(none enumerated)")
            + "; still scored: (none). No arm survives, so the whole-run verdict is "
            "substrate_not_ready_requeue rather than any substrate finding."
        )

    # A refused ree arm cannot carry a recovery read on its rung.
    ree_recovery_rungs = [r for r in ree_recovery_rungs if r not in ree_guard_red_rungs]
    ree_recovers = bool(ree_recovery_rungs)
    hardest_ree_recovery_rung = ree_recovery_rungs[0] if ree_recovery_rungs else None

    # ----- self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif ree_recovers:
        outcome = "PASS"
        label = "env_difficulty_recoverable"
    elif ppo_recovers:
        outcome = "FAIL"
        label = "ree_substrate_ceiling"
    elif d3_oracle_ok:
        outcome = "FAIL"
        label = "learner_or_observability_ceiling"
    else:
        # neither learner recovered but the hazard-free rung was not oracle-achievable ->
        # cannot license the deepest branch; treat as not-ready.
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    # The refused-arm override: the load-bearing DV belongs to ree_trained_allon, so a run in
    # which ANY ree cell ran on a frozen random projection must NOT report a PASS carried by
    # that arm, and must NEVER emit a substrate-verdict label.
    if any_ree_guard_red:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    direction = "non_contributory"

    load_bearing_passed = bool(ree_recovers and not any_ree_guard_red)

    # Flat `interpretation.preconditions` is read ARM-BLIND and WHOLE-RUN by the REE_assembly
    # indexer (first unmet entry => whole-run precondition_unmet). On a PARTIAL run only the
    # GREEN arms' guard preconditions go in the flat list; the red ones are carried under
    # per_arm_gate. If EVERY cell is red the whole-run verdict IS correct, so they all go in.
    if non_degenerate:
        flat_guard_preconditions = [
            ree_guard_precondition_by_rung[r]
            for r in [x["rung_id"] for x in DIFFICULTY_RUNGS]
            if r in ree_guard_precondition_by_rung and r not in ree_guard_red_rungs
        ]
        excluded_rungs = [r for r in ree_guard_red_rungs
                          if r in ree_guard_precondition_by_rung]
    else:
        flat_guard_preconditions = [
            ree_guard_precondition_by_rung[r]
            for r in [x["rung_id"] for x in DIFFICULTY_RUNGS]
            if r in ree_guard_precondition_by_rung
        ]
        excluded_rungs = []

    preconditions_scope_note = (
        "The zworld_world_encoder_trained precondition applies ONLY to the "
        "ree_trained_allon arm; vanilla_ppo / greedy_oracle / random_walk are scoped out "
        f"({SCOPE_OUT_REASON}). Guard entries for rungs "
        + (", ".join(excluded_rungs) if excluded_rungs else "(none)")
        + " are EXCLUDED from this flat list because their ree_trained_allon arm is RED and "
        "the indexer reads this list arm-blind and whole-run; those entries are carried "
        "under interpretation.per_arm_gate.failed_preconditions_by_arm instead, so a refused "
        "arm cannot vacate the arms and rungs that remain valid and scored."
    )

    interpretation = {
        "label": label,
        "ree_recovery_rungs": ree_recovery_rungs,
        "ppo_recovery_rungs": ppo_recovery_rungs,
        "hardest_ree_recovery_rung": hardest_ree_recovery_rung,
        "preconditions": [
            {
                "name": "oracle_clears_floor_all_rungs",
                "kind": "readiness",
                "description": (
                    "The greedy nearest-resource ORACLE must clear COMPETENCE_RESOURCE_FLOOR "
                    "resources/episode at EACH difficulty rung, proving the floor is ACHIEVABLE "
                    "in that env. Same statistic as the load-bearing recovery criterion "
                    "(mean resources/episode). Below-floor at a rung => that cell is not ready; "
                    "below-floor at D0 => substrate_not_ready_requeue, NEVER a ceiling verdict."
                ),
                "control": "greedy nearest-resource oracle forager per rung, same env/seed, no agent",
                "measured": float(round(min(_oracle_forage(r["rung_id"]) for r in DIFFICULTY_RUNGS), 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(all_oracle_clear),
            },
            {
                "name": "baseline_reproduces_incompetence_at_D0",
                "kind": "readiness",
                "description": (
                    "The all-ON stack trained on the D0 (=724) env must forage BELOW the floor on "
                    "a strict majority of seeds -- i.e. the 724/728 incompetence must reproduce -- "
                    "for the sweep premise to hold. If it already clears the floor at D0 the "
                    "premise is not reproduced => substrate_not_ready_requeue."
                ),
                "control": "ree_trained_allon D0 mean resources/ep vs floor (strict majority of seeds)",
                "measured": float(round(_arm_forage(D0_RUNG_ID, "ree_trained_allon"), 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "direction": "upper",
                "met": bool(d0_baseline_reproduces),
            },
            {
                "name": "hazard_free_rung_achievable",
                "kind": "readiness",
                "description": (
                    "The maximally de-risked D3 (hazard-free) env must be oracle-achievable "
                    "(oracle clears the floor) so that 'neither learner recovers even hazard-free' "
                    "can license the deepest learner/observability branch rather than an "
                    "env-too-hard artifact."
                ),
                "control": "greedy oracle on D3 hazard-free env vs floor",
                "measured": float(round(_oracle_forage(D3_RUNG_ID), 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(d3_oracle_ok),
            },
            {
                "name": "sufficient_eval_episodes_all_cells",
                "kind": "readiness",
                "description": (
                    "Every completed cell must log >= MIN_EVAL_EPISODES eval episodes so "
                    "foraging_competence is estimable. Below => substrate_not_ready_requeue."
                ),
                "control": "min completed eval episodes across all cells",
                "measured": float(min_eval_eps),
                "threshold": float(MIN_EVAL_EPISODES),
                "met": bool(sufficient_eval),
            },
        ] + flat_guard_preconditions,
        "preconditions_scope_note": preconditions_scope_note,
        "per_arm_gate": {
            "gate_by_rung_arm": per_arm_gate,
            "failed_preconditions_by_arm": failed_preconditions_by_arm,
            "guarded_arm": GUARDED_ARM,
            "scope_out_reason": SCOPE_OUT_REASON,
            "green_cells": green_cell_keys,
            "red_cells": red_cell_keys,
            # Attribution even on a PARTIAL run: which cells a refusal removed from scoring,
            # and which cells remain fully valid and scored despite it.
            "still_scored_cells": green_cell_keys,
            "refused_cells": red_cell_keys,
            "ree_guard_green_rungs": ree_guard_green_rungs,
            "ree_guard_red_rungs": ree_guard_red_rungs,
            "any_ree_guard_red": bool(any_ree_guard_red),
        },
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": degeneracy_reason,
        "criteria": [
            {
                "name": "ree_allon_recovers_above_floor_at_some_difficulty",
                "load_bearing": True,
                "passed": load_bearing_passed,
            },
        ],
        "criteria_non_degenerate": {
            "oracle_clears_floor_all_rungs": bool(all_oracle_clear),
            "baseline_reproduces_incompetence_at_D0": bool(d0_baseline_reproduces),
            "hazard_free_rung_achievable": bool(d3_oracle_ok),
            "sufficient_eval_episodes": bool(sufficient_eval),
            "ppo_control_informative": bool(ppo_control_informative),
            # Keyed to the OWNING arm's gate: this criterion belongs to ree_trained_allon and
            # is scoped out of the other three arms, which remain non-degenerate regardless.
            "zworld_world_encoder_trained_ree_trained_allon": bool(not any_ree_guard_red),
            "any_arm_green": bool(non_degenerate),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": list(seeds),
        "strict_majority_seeds": int(maj),
        "zworld_p0_episodes": int(zworld_p0_episodes),
        "p0_warmup_episodes": int(p0_episodes),
        "p1_reinforce_episodes": int(p1_episodes),
        "p1_ppo_episodes": int(p1_ppo_episodes),
        "e2_train_in_p1": bool(E2_TRAIN_IN_P1),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps_per_episode),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "difficulty_rungs": [
            {"rung_id": r["rung_id"], "role": r["role"],
             "env_kwargs": _env_kwargs_for_rung(r)}
            for r in DIFFICULTY_RUNGS
        ],
        "readiness_gates": {
            "d0_oracle_clears_floor": d0_oracle_ok,
            "d0_baseline_reproduces_incompetence": d0_baseline_reproduces,
            "d3_hazard_free_oracle_clears_floor": d3_oracle_ok,
            "all_rungs_oracle_clear": all_oracle_clear,
            "sufficient_eval_episodes": sufficient_eval,
            "min_eval_episodes": int(min_eval_eps),
            "readiness_met": readiness_met,
        },
        "recovery_gates": {
            "derisked_oracle_achievable_rungs": derisked_ids,
            "ree_recovery_rungs": ree_recovery_rungs,
            "ppo_recovery_rungs": ppo_recovery_rungs,
            "ree_recovers": ree_recovers,
            "ppo_recovers": ppo_recovers,
            "hardest_ree_recovery_rung": hardest_ree_recovery_rung,
            "ppo_beats_random_at_d3": ppo_beats_random_at_d3,
        },
        "zworld_guard_gates": {
            "guarded_arm": GUARDED_ARM,
            "any_ree_guard_red": bool(any_ree_guard_red),
            "ree_guard_green_rungs": ree_guard_green_rungs,
            "ree_guard_red_rungs": ree_guard_red_rungs,
            "non_degenerate": bool(non_degenerate),
            "degeneracy_reason": degeneracy_reason,
        },
        "zworld_guard_reports_by_rung_seed": guard_diagnostics,
        "per_rung_report": per_rung_report,
        "per_rung_arm_summaries": per_rung_arm_summ,
        "arm_results": all_cells,
        "interpretation_grid": {
            "env_difficulty_recoverable": (
                "readiness holds AND ree_trained_allon clears the floor on a strict majority of "
                "seeds at some de-risked (oracle-achievable) rung. The deficit is an "
                "env-difficulty ceiling. HYPOTHESIS (not a verdict): route to /implement-substrate "
                "on a CURRICULUM (report hardest_ree_recovery_rung)."
            ),
            "ree_substrate_ceiling": (
                "readiness holds, ree_trained_allon never clears the floor, but the matched "
                "vanilla PPO does at some de-risked rung. A standard learner forages the de-risked "
                "env on the SAME observation interface but the REE stack cannot -> a REE-specific "
                "substrate/representation ceiling, NOT env-buildable for REE."
            ),
            "learner_or_observability_ceiling": (
                "readiness holds (incl. the hazard-free D3 oracle clearing the floor) and NEITHER "
                "ree_trained_allon NOR the matched vanilla PPO clears the floor even in the "
                "maximally de-risked D3 env. No powered learner forages even a hazard-free env on "
                "this observation interface -> a deeper learner / observation-encoding problem "
                "(echoes 732a). Route to /implement-substrate on the observation encoding."
            ),
            "substrate_not_ready_requeue": (
                "D0 oracle cannot clear the floor, OR the all-ON stack already forages >= floor at "
                "D0 (724/728 premise not reproduced), OR a cell logged fewer than "
                "MIN_EVAL_EPISODES eval episodes, OR neither learner recovered but the hazard-free "
                "rung was not oracle-achievable. NOT a verdict -- re-examine env/floor/budget and "
                "re-queue. Draw NO conclusion about the competence root."
            ),
        },
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    rg = result["readiness_gates"]
    rec = result["recovery_gates"]
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-734 ENV-DIFFICULTY COMPETENCE-RECOVERY SWEEP (Track-1a; "
            f"experiment_purpose=diagnostic, claim_ids=[], non_contributory -- EXCLUDED from "
            f"governance scoring; PROMOTES / DEMOTES NOTHING). Holds the 724-A0 all-ON recipe "
            f"FIXED and sweeps ENV DIFFICULTY DOWN a monotone staircase (D0=724 config -> D3 "
            f"hazard-free) to answer the pivotal fork: is the foraging deficit an ENV-DIFFICULTY "
            f"ceiling (curriculum-buildable) or a substrate/learner ceiling? 724 varied training "
            f"duration/encoder-thaw/mechanism-count and found competence_deficit_diffuse; 728 "
            f"landed the trained all-ON point (survives, does not forage: 0.167 res/ep below the "
            f"0.267 random floor -- a forage-vs-survive inversion). Four arms per rung on the "
            f"capability_eval yardstick: random_walk floor / ree_trained_allon (724-A0 all-ON, "
            f"trained+evaluated MATCHED per rung) / vanilla_ppo (matched raw-obs PPO learner-"
            f"adequacy control, folds in 732a -- SAME observation interface, NOT the privileged "
            f"oracle) / greedy_oracle ceiling+readiness. Load-bearing DV: foraging_competence "
            f"(mean resources/ep). Readiness per rung: oracle clears the "
            f"{COMPETENCE_RESOURCE_FLOOR} floor (all_rungs_oracle_clear="
            f"{rg['all_rungs_oracle_clear']}, D0={rg['d0_oracle_clears_floor']}, "
            f"D3_hazard_free={rg['d3_hazard_free_oracle_clears_floor']}); D0 reproduces "
            f"incompetence={rg['d0_baseline_reproduces_incompetence']}. Self-route (HYPOTHESIS, "
            f"not a verdict): readiness_met={rg['readiness_met']} -> if ree recovers at some "
            f"de-risked rung => env_difficulty_recoverable (route curriculum, hardest rung="
            f"{rec['hardest_ree_recovery_rung']}); elif ppo recovers where ree does not => "
            f"ree_substrate_ceiling; elif neither recovers even hazard-free => "
            f"learner_or_observability_ceiling (route observation-encoding); else "
            f"substrate_not_ready_requeue. label={result['interpretation_label']} "
            f"(ree_recovery_rungs={rec['ree_recovery_rungs']}, ppo_recovery_rungs="
            f"{rec['ppo_recovery_rungs']}). BRAKE-EXEMPT (competence/env diagnostic, not a "
            f"conversion/de-commit falsifier; tags no claim). Maps to node "
            f"conversion_ceiling_campaign:CAMPAIGN. Route to /failure-autopsy for adjudication "
            f"before any governance action."
        ),
        "dry_run": bool(dry_run),
        "diagnostics": {
            "zworld_encoder_guard": {
                "policy": "strict",
                "scope": "detection_only",
                "guarded_arm": "ree_trained_allon",
                "guard_site": (
                    "this script's _run_cell call site, NOT inside _train_all_on_agent "
                    "(v3_exq_737 and v3_exq_742 import that function)"
                ),
                "diagnosis_doc": (
                    "REE_assembly/evidence/planning/"
                    "zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md"
                ),
                "gates": result["zworld_guard_gates"],
                "reports_by_rung_seed": result["zworld_guard_reports_by_rung_seed"],
            },
        },
        "env_kwargs": dict(x724.ENV_KWARGS),
        "config_summary": {
            "design": (
                f"env-difficulty competence-recovery sweep; "
                f"{len(DIFFICULTY_RUNGS)} difficulty rungs x {len(ARMS)} policy arms x "
                f"{len(result['seeds'])} seeds; capability_eval 4-metric yardstick"
            ),
            "difficulty_rungs": {
                r["rung_id"]: r["role"] for r in DIFFICULTY_RUNGS
            },
            "difficulty_sweep_axes": (
                "hazard_food_attraction 0.7->0.0; proximity_harm_scale 0.1->0.0; "
                "reef_bipartite_layout True->False; num_hazards 4->0; reef_enabled True->False "
                "(+ n_reef_patches 3->0) at the hazard-free D3 rung so 'hazard-free' is genuine"
            ),
            "arms": {
                "random_walk": "uniform-random -- FLOOR anchor",
                "ree_trained_allon": (
                    "all-ON REE stack (714 ARM_ON) trained on THIS rung's env with the 724-A0 "
                    "recipe (P0 world-model warmup + P1 two-head REINFORCE, e2 frozen in P1), "
                    "evaluated via the mechanism-agnostic yardstick REEForwardPolicy"
                ),
                "vanilla_ppo": (
                    "matched-capacity NON-REE PPO actor-critic over the RAW observation vector "
                    "(body_state (+) world_state (+) harm channels) -- learner-adequacy control "
                    "on the SAME observation interface (folds in 732a); clipped-surrogate PPO + "
                    "GAE + count-based novelty + running-std reward scaling; greedy unshaped eval"
                ),
                "greedy_oracle": "nearest-resource greedy forager -- CEILING / per-rung readiness anchor",
            },
            "load_bearing_dv": "foraging_competence = capability_eval mean resources/episode",
            "reported_metrics": list(METRIC_KEYS),
            "reusable_block": "experiments/_lib/capability_eval.py",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-724",
            "training_recipe_sourced_from": "V3-EXQ-724 A0_baseline_allon_p1short_frozen",
            "ppo_learner_sourced_from": "V3-EXQ-732a (power-fixed matched PPO)",
            "zworld_p0_episodes": int(result["zworld_p0_episodes"]),
            "p0_warmup_episodes": int(result["p0_warmup_episodes"]),
            "p1_reinforce_episodes": int(result["p1_reinforce_episodes"]),
            "p1_ppo_episodes": int(result["p1_ppo_episodes"]),
            "e2_train_in_p1": bool(result["e2_train_in_p1"]),
            "strict_majority_seeds": int(result["strict_majority_seeds"]),
            "alpha_world": 0.9,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-734 env-difficulty competence-recovery sweep DIAGNOSTIC "
            "(is the foraging deficit env-difficulty or substrate/learner; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        zp0 = DRY_RUN_ZWORLD_P0
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        ppo = DRY_RUN_PPO
        eval_eps = DRY_RUN_EVAL
        steps = DRY_RUN_STEPS
        rollout = DRY_RUN_ROLLOUT_EPISODES
    else:
        seeds = list(SEEDS)
        zp0 = ZWORLD_P0_EPISODES
        p0 = P0_WARMUP_EPISODES
        p1 = P1_REINFORCE_EPISODES
        ppo = P1_PPO_EPISODES
        eval_eps = EVAL_EPISODES
        steps = STEPS_PER_EPISODE
        rollout = PPO_ROLLOUT_EPISODES

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        p1_ppo_episodes=ppo,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        rollout_episodes=rollout,
        dry_run=bool(args.dry_run),
        zworld_p0_episodes=zp0,
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=bool(args.dry_run),
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    rg = result["readiness_gates"]
    rec = result["recovery_gates"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness_met={rg['readiness_met']} "
        f"all_oracle_clear={rg['all_rungs_oracle_clear']} "
        f"d0_reproduces={rg['d0_baseline_reproduces_incompetence']} "
        f"ree_recovers={rec['ree_recovers']} ppo_recovers={rec['ppo_recovers']} "
        f"hardest_ree_recovery={rec['hardest_ree_recovery_rung']}",
        flush=True,
    )
    zg = result["zworld_guard_gates"]
    print(
        f"zworld_guard: policy=strict arm={zg['guarded_arm']} "
        f"any_ree_guard_red={zg['any_ree_guard_red']} "
        f"red_rungs={zg['ree_guard_red_rungs']} green_rungs={zg['ree_guard_green_rungs']} "
        f"non_degenerate={zg['non_degenerate']} "
        f"degeneracy_reason={zg['degeneracy_reason']}",
        flush=True,
    )
    for rung in DIFFICULTY_RUNGS:
        rid = rung["rung_id"]
        summ = result["per_rung_arm_summaries"][rid]
        rep = result["per_rung_report"][rid]["readiness"]
        print(
            f"  RUNG {rid}: oracle_forage/ep={rep['oracle_foraging_competence']} "
            f"ree_forage/ep={summ['ree_trained_allon']['foraging_competence_mean']} "
            f"(supra {summ['ree_trained_allon']['n_seeds_supra_floor']}/"
            f"{summ['ree_trained_allon']['n_seeds']}) "
            f"ppo_forage/ep={summ['vanilla_ppo']['foraging_competence_mean']} "
            f"(supra {summ['vanilla_ppo']['n_seeds_supra_floor']}/{summ['vanilla_ppo']['n_seeds']}) "
            f"ree_survival={summ['ree_trained_allon']['survival_horizon_mean']}",
            flush=True,
        )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
