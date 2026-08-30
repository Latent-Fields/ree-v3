"""
V3-EXQ-965 -- SD-e1-rollout-consistency-training ITEM 1 validation: does E1's
action-conditioned transition produce measurably different predicted transitions
per action?

Claims: [] (diagnostic; this chip validates the SUBSTRATE, not MECH-135/INV-088.
Both keep pending_retest_after_substrate: true -- their retest is a SEPARATE,
governance-routed successor once this substrate validation lands.)
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
ree-v3 26557a3758 (2026-08-29) built ITEM 1 of SD-e1-rollout-consistency-training:
E1DeepPredictor.forward()/predict_long_horizon() now accept an `actions` argument
and, under E1Config.action_conditioned_transition, feed the LSTM a DEDICATED
action channel (cat([state_i, action_encoder(a_i)])) instead of squeezing the
action signal through the single world_dim-wide prior_generator projection that
V3-EXQ-954 (2026-08-29, PASS, confirmed autopsy failure_autopsy_V3-EXQ-954)
showed floors cr_ratio at h=1 (4.76e-07 / 5.39e-07 against a 0.1 bar) with the
horizon-compounding signature ABSENT -- the action-blindness signature. The
red-team pass behind that autopsy localised a ~5,000x per-action divergence
attenuation inside E1 itself (trained E2 per-action z_self divergence 2.8e-2 vs
5.6e-6 at the E1 output).

THIS SCRIPT is the owed validation: does conditioning E1's transition on the
action ACTUALLY produce measurably different predicted transitions per action,
on a TRAINED model (not just at untrained init, where the design doc's own
smoke measurement -- mean per-action pairwise L2 EXACTLY 0 off / 2.6e-04 on --
bounds nothing about a trained model)?

THE SINGLE MOST IMPORTANT INSTRUCTION IN THIS SCRIPT: DO NOT read this as a
retest of the C3_VAR_THRESHOLD (e1coe_score_var vs 0.002) or of cr_ratio against
the 0.1 CR_ROLLOUT_COLLAPSE_RATIO bar. Those are ITEM 2's target (the multi-step
/ rollout-consistency training objective, still pending_implementation) --
ITEM 1 is an INTERFACE fix only; the dominant ~675x crush the red-team
localised sits at the LSTM + output_proj stage, untouched here. An experiment
that FAILed item 1 for not clearing C3_VAR_THRESHOLD would be adjudicating
item 2's target on item 1's build. cr_ratio(h) and e1coe_score_var are still
computed and RECORDED below (so item 2 has its baseline, unchanged from
V3-EXQ-954's own methodology) but are explicitly NON-GATING -- see
`interpretation.non_gating_reference` in the output.

ITEM 1's OWN acceptance criterion is a MATERIAL, THRESHOLD-DERIVED LIFT in
ONE-STEP PER-ACTION DIVERGENCE at the E1 OUTPUT: from ~5.6e-6 (the red-team's
trained-E1 number; this run's own OFF arm independently re-measures the same
quantity as a positive control) toward E2's trained 2.8e-2 reference. See
Phase 4c below.

METHODOLOGY -- REUSE, NOT RE-DESIGN
---------------------------------------------------------------------------
Phases 0a/0b/1/2/3/4/4b are the SAME SD-070 z_world encoder warmup, the SAME
bespoke E1/E2 single-step training, the SAME goal template, the SAME warmup
state, the SAME 40 random 30-step candidate action sequences, and the SAME
multi-horizon cr_ratio(h)/CR_real(h) sweep V3-EXQ-954 used -- with exactly two
categories of change: (a) every direct `agent.e1(...)` call now threads an
explicit `actions=` argument (a no-op when the arm's E1Config has
action_conditioned_transition=False -- E1.forward ignores an actions argument
on that path), and (b) three arms instead of one, per the SD doc's own
sub-flag ablation:
  A_off      action_conditioned_transition=False                    (954 baseline)
  B_action   action_conditioned_transition=True,  action_cond_unzero_self_slot=False
  C_both     action_conditioned_transition=True,  action_cond_unzero_self_slot=True
Same seeds as 954 ([42, 123]) so A_off is checkable against V3-EXQ-954's own
recorded numbers -- if A_off does not reproduce them (within the readiness
band below), the run self-routes substrate_not_ready_requeue rather than
reading as a scientific result. Phase 4 (rollout scoring) is retained
UNCHANGED beyond threading `actions=` at each rollout step, so cr_ratio(h) and
e1coe_score_var(h) are recorded across the SAME checkpoints V3-EXQ-954 used --
this is what "item 2 has its baseline" means in practice.

PHASE 4c -- REDESIGNED to test ITEM 1 directly, not through E2
---------------------------------------------------------------------------
V3-EXQ-954's own Phase 4c (`_one_step_action_divergence`) predates ITEM 1: the
ONLY channel by which an action could reach E1 at the time was indirect
(action -> agent.e2.predict_next_self -> z_self -> E1's prior_generator), so it
routed each action through E2 before calling E1. ITEM 1 adds a DIRECT channel
(E1's own `actions=` argument), and testing THAT channel means holding
everything else fixed and varying ONLY the action fed to E1 -- exactly the SD
doc's own measurement protocol ("Reference measurement... From the SAME
Phase-2 warmup state (z_self_0, z_world_0), for EACH of the K grid-world
actions... a SINGLE E1 forward call (horizon=1) is made"). This script's Phase
4c therefore holds (z_self_0, z_world_0) FIXED across all K actions and varies
only the `actions=` one-hot passed to `agent.e1(...)` -- the DIRECT test of
"does E1's transition respond to the action", independent of any E2-mediated
indirect signal. This is a stricter, cleaner isolation of ITEM 1's own
contribution than V3-EXQ-954's E2-routed version, and (for the OFF arm) exactly
reproduces the design doc's own "EXACTLY 0 with the flag off" signature by
construction, since (z_self_0, z_world_0) is bitwise identical across all K
calls when the action channel is inert.

TWO VACUITY TRAPS FROM THE 108/108a/954 HISTORY -- both closed by DESIGN here,
not merely checked after the fact, and both documented per chip instruction:
  (1) E1DeepPredictor._action_cond_missing_calls increments whenever an
      action-conditioned E1's predict_long_horizon() receives actions=None
      (silent zero-action fallback). Because this script passes an EXPLICIT,
      freshly-constructed one-hot `actions=` argument at every single
      `agent.e1(...)` call site in every phase, this counter is 0 by
      construction for the whole run on the ON arms -- verified, not assumed,
      via `agent.e1_action_buffer_stats()["missing_action_calls"]` after each
      seed, recorded as a readiness precondition.
  (2) The 954-lineage failure mode: a driver that steps the env directly with
      random.randint actions and never calls select_action() leaves
      REEAgent._last_action == None, so the internal
      _action_experience_buffer (populated only inside REEAgent._e1_tick, which
      this script -- like 954 before it -- never calls, following the SAME
      bespoke direct-E1-call methodology) fills with zero actions; a driver
      relying on that buffer for its action-conditioning signal would be
      silently testing an OFF arm under an ON label. THIS SCRIPT NEVER RELIES
      ON THAT BUFFER OR ON `_last_action` FOR ANY `actions=` ARGUMENT IT PASSES
      TO agent.e1(...) -- every `actions=` value is a freshly-constructed,
      locally-scoped one-hot tensor, verified non-None and exactly-one-hot at
      construction. So trap (2)'s specific failure mode (a real action silently
      degrading to a zero vector between "chosen" and "consumed") cannot occur
      here by construction; there is no buffer indirection between the two.
      `agent.record_executed_action(...)` is STILL called immediately after
      every action choice that is actually EXECUTED against the real env
      (Phase 0b training, Phase 1 goal template, Phase 2 warmup state, Phase 4b
      real-state sampling) in BOTH arms, per the chip's explicit instruction
      and for consistency with every other REEAgent subsystem that reads
      _last_action -- but it is NOT the source of the `actions=` tensors this
      script passes to agent.e1(), and NOT called in Phase 4/4c, which score
      IMAGINED (not executed) candidate sequences and a synthetic per-action
      probe respectively; recording those as "executed" would be exactly the
      kind of MECH-094 hygiene violation this substrate's other subsystems are
      built to avoid. Because `_e1_tick()` is never invoked, the internal
      `_action_experience_buffer` stays empty for the whole run regardless of
      this script's design -- `agent.e1_action_buffer_stats()["nonzero_fraction"]`
      is recorded for cross-reference but is EXPECTED to read
      n=0/nonzero_fraction=0.0 here and is NOT the load-bearing non-vacuity
      signal for this script (that signal is `missing_action_calls == 0`,
      trap (1) above, plus this script's own direct_action_supply_fraction,
      computed straight off the tensors actually passed to agent.e1() -- see
      Phase 4c).

DECISION RULE
  ITEM 1 PASSes on an (arm, seed) cell iff BOTH tails clear:
    (a) RELATIVE lift: ratio_action_vs_real_h1[ON] / ratio_action_vs_real_h1[OFF]
        (same seed) >= LIFT_FACTOR, where LIFT_FACTOR is derived from THIS
        RUN's own measured OFF-arm cross-seed noise ratio (never an inherited
        historical constant) -- see LIFT_FACTOR computation in run().
    (b) ABSOLUTE floor: the ON arm's raw one-step contrast ratio clears both
        an absolute floor (ABS_ACTION_CR_FLOOR) and a minimum multiple of the
        OFF arm's own absolute value on the same seed -- guards against a
        "lift" manufactured by an OFF arm sitting at numerical noise (the
        V3-EXQ-936a both-tails-need-floors lesson).
  Both tails route on the SAME statistic (contrast_ratio.contrast_ratio /
  ratio_action_vs_real_h1) the criterion itself reads, per the
  V3-EXQ-643/936a same-statistic discipline.

READINESS (P0, self-route to substrate_not_ready_requeue on any miss -- never a
substrate-verdict label)
  P_ENCODER_TRAINED: at least one split_encoder.world_encoder tensor moved
    during Phase 0a, every seed (inherited unchanged from V3-EXQ-954/108b).
  P_REAL_ZWORLD_NONDEGENERATE_H1: CR_real(h=1) finite, positive, backed by
    >= MIN_REAL_SAMPLES_PER_HORIZON samples, every seed (the same statistic
    the ratio_action_vs_real_h1 criterion's denominator uses).
  P_NO_MISSING_ACTION_CALLS: agent.e1_action_buffer_stats()["missing_action_calls"]
    == 0 on every ON-arm (B_action, C_both) cell -- vacuity trap (1).
  P_DIRECT_ACTION_SUPPLY: this script's own count of agent.e1(...) calls that
    received a genuine non-None, exactly-one-hot `actions=` argument, divided
    by the total agent.e1(...) call count, == 1.0 on every ON-arm cell --
    defense-in-depth analogue of vacuity trap (2)/(3) for this script's
    buffer-free design (see above).
  P_OFF_ARM_REPRODUCES_954: A_off's ratio_action_vs_real_h1 at h=1 is below
    OFF_ARM_COLLAPSE_CEILING on every seed -- confirms this run's OFF arm
    faithfully reproduces the known action-blindness signature before any
    ON-vs-OFF comparison is trusted. If this fails, something about THIS run
    differs from V3-EXQ-954's own substrate/config, and the comparison is not
    yet trustworthy -- self-route, do not read as a scientific result.

DECLARED NULL. A FAIL/PARTIAL_LIFT/NO_LIFT label here does not reopen
V3-EXQ-108/108a/954's own C1/C2/C3 findings -- those stand regardless. This
run's job is to confirm (or refute) that ITEM 1's interface change produces a
measurable, non-vacuous per-action signal at the E1 output on a trained model,
which is the precondition ITEM 2 (the multi-step objective) needs before it can
be meaningfully built on top.

Re-derive brake (Step 2.5b): not applicable -- this validates a substrate that
did not exist until ree-v3 26557a3758 (2026-08-29); there is no prior autopsy
to brake against at this granularity (SD-e1-rollout-consistency-training ITEM 1
action-conditioning-carries-a-per-action-signal-on-a-trained-model has never
been measured before).

GOV-REUSE-1 (Step 2.4): the decisive readout (per-action one-hot-conditioned
E1 one-step divergence on a TRAINED, action-conditioned E1) does not exist in
any prior manifest -- V3-EXQ-954's own manifest predates ITEM 1's build by
construction (it is the failure record ITEM 1 was built to address) and its
one-step probe routes actions through E2, never through E1's own `actions=`
channel. Not recoverable from existing data -> proceed to author (this script).

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: real (GoalState with a genuine collected template, unchanged from
V3-EXQ-954/108a/108b) -- recorded via z_goal_stream_stats at manifest-write
time. The goal-proximity score is bonus/cross-reference data only in this
script; the decision rule above routes entirely on the one-step per-action
divergence probe (Phase 4c).
"""

import itertools
import sys
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_encoder_guard import (
    latent_stack_snapshot,
    assert_world_encoder_trained,
)
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import p0_readiness_gate, P0NotReady


EXPERIMENT_TYPE = "v3_exq_965_sd_e1_item1_action_conditioning_validation"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint exemption -- same shape and same
# justification as V3-EXQ-954/108b/108a: every scored quantity in this script
# is read off the LITERAL SAME agent object within a (seed, arm) cell (never
# two independently-constructed agents), so unseeded weight init cannot
# confound any within-cell comparison this script makes (action-vs-action,
# horizon-vs-horizon). Cross-arm comparisons ARE across independently-
# constructed agents by design (that is the whole point of an ablation) and
# are seeded identically per arm via arm_cell()'s RNG reset.
AGENT_SEED_ORDER_EXEMPT = (
    "Every within-cell comparison (action-vs-action, horizon-vs-horizon) is "
    "scored off the literal same agent object; cross-arm comparisons are the "
    "ablation itself and are seed-matched via arm_cell()."
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4               # unchanged from V3-EXQ-954/108b -- P_REAL_ZWORLD floor
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # NON-GATING here -- ITEM 2's target, recorded only
C3_VAR_THRESHOLD = 0.002           # NON-GATING here -- ITEM 2's target, recorded only
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches V3-EXQ-954
N_REAL_SAMPLES = 40                # per-checkpoint target sample count for CR_real(h)
MIN_REAL_SAMPLES_PER_HORIZON = 10  # readiness floor: surviving real samples per checkpoint
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]

# ITEM 1's OWN load-bearing thresholds (Phase 4c). LIFT_FACTOR is NOT a fixed
# constant -- it is derived per-run from the OFF arm's own measured cross-seed
# noise ratio in run(), per the "derive from a measured baseline" instruction.
LIFT_FACTOR_ABS_FLOOR = 3.0        # minimum relative-lift bar regardless of measured noise
LIFT_FACTOR_NOISE_MULTIPLE = 2.0   # required margin over the measured OFF-arm noise ratio
ABS_ACTION_CR_FLOOR = 1e-7         # absolute floor on the ON arm's one-step contrast ratio
ABS_LIFT_MULTIPLE = 2.0            # ON arm's absolute contrast ratio must clear this x OFF
OFF_ARM_COLLAPSE_CEILING = 1e-3    # P_OFF_ARM_REPRODUCES_954 -- comfortably above 954's ~1e-6
                                    # measured value, comfortably below any plausible lift bar

ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A_off": {"action_conditioned_transition": False, "action_cond_unzero_self_slot": False},
    "B_action": {"action_conditioned_transition": True, "action_cond_unzero_self_slot": False},
    "C_both": {"action_conditioned_transition": True, "action_cond_unzero_self_slot": True},
}
ARM_ORDER = ["A_off", "B_action", "C_both"]
SEEDS_DEFAULT = [42, 123]


# ---------------------------------------------------------------------------
# Helpers (unchanged from V3-EXQ-954 unless noted)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-954/108b."""
    return dict(
        size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )


def _build_agent(
    seed: int, world_dim: int, self_dim: int, arm: str,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    arm_cfg = ARM_CONFIGS[arm]
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        action_conditioned_transition=arm_cfg["action_conditioned_transition"],
        action_cond_unzero_self_slot=arm_cfg["action_cond_unzero_self_slot"],
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from 954)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_965 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_965_sd_e1_item1_action_conditioning_validation",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 single-step training. MODIFIED from V3-EXQ-954:
# threads actions=action_prev through the E1 call (a no-op on A_off, since
# E1.forward ignores an actions argument when action_conditioned_transition
# is False) and calls agent.record_executed_action(action_curr) after every
# chosen action, in every arm (trap 3).
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    e1_call_counter: Dict[str, int],
) -> None:
    """Train agent with random policy (E1 + E2 only). Byte-identical to
    V3-EXQ-954's _train_agent's call structure except the E1 call now threads
    the executed action. Calls agent.sense() exactly ONCE per env step
    (StepHarness invariant #1). sense() runs under torch.no_grad() throughout,
    so Phase 0a's now-trained encoder is never further disturbed."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        n_steps = 0

        latent_prev: Optional[object] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent_curr = agent.sense(obs_body, obs_world)

            action_idx = random.randint(0, env.action_dim - 1)
            action_curr = _action_to_onehot(action_idx, env.action_dim, agent.device)
            # trap (3): record the executed action in EVERY arm, not only the
            # action-conditioned ones -- keeps _last_action correct for every
            # other REEAgent subsystem that reads it.
            agent.record_executed_action(action_curr)

            if latent_prev is not None:
                opt_e1.zero_grad()
                total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
                total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1)
                # action_prev is the action that carries total_prev -> total_curr
                # (it was chosen and executed on the PRIOR loop iteration, before
                # env.step produced the observation now sensed as latent_curr).
                # Same convention as agent.py's compute_prediction_loss()
                # [start_idx+1:end_idx] slice -- see the SD doc's "Buffer
                # alignment" note.
                e1_pred, _ = agent.e1(total_prev, horizon=1, actions=action_prev)
                e1_call_counter["n_e1_calls"] += 1
                e1_call_counter["n_e1_calls_nonzero_action"] += (
                    1 if action_prev is not None and float(action_prev.abs().sum()) > 0.0 else 0
                )
                e1_loss = F.mse_loss(e1_pred[:, 0, :], total_curr.detach())
                e1_loss.backward()
                opt_e1.step()
                ep_loss_e1 += e1_loss.item()

                opt_e2.zero_grad()
                z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action_prev)
                e2_loss = F.mse_loss(z_self_pred, latent_curr.z_self.detach())
                e2_loss.backward()
                opt_e2.step()
                ep_loss_e2 += e2_loss.item()
                n_steps += 1

            _, _, done, _, obs_dict = env.step(action_curr)

            latent_prev = latent_curr
            action_prev = action_curr

            if done:
                break

        if (ep + 1) % 20 == 0:
            print(
                f"  [Train] ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    agent.eval()
    print(f"  [Train] Done. {n_episodes} episodes.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: goal template (unchanged from 954 except record_executed_action)
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, max_steps: int,
) -> Tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent.record_executed_action(action)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach(), "resource_contact"
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using fallback unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal, "fallback_unit_vector"


# ---------------------------------------------------------------------------
# Phase 2: warmup state (unchanged from 954 except record_executed_action)
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    warmup_actions: List[int] = []

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        warmup_actions.append(action_idx)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent.record_executed_action(action)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None
            warmup_actions = []

    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    return latent.z_self.detach(), latent.z_world.detach(), warmup_actions


# ---------------------------------------------------------------------------
# Phase 3: generate candidate sequences (unchanged from 954)
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int, horizon: int, n_actions: int, seed: int,
) -> List[List[int]]:
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 4 (NON-GATING; recorded so ITEM 2 has its baseline): rollout scoring
# at every horizon checkpoint. MODIFIED from 954: threads actions= at every
# rollout step (a no-op on A_off). NOT part of the ITEM 1 decision rule.
# ---------------------------------------------------------------------------

def _score_sequence_e1coe_multi_horizon(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
    checkpoints: List[int],
    e1_call_counter: Dict[str, int],
) -> Dict[int, Tuple[float, torch.Tensor]]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    checkpoint_set = set(checkpoints)

    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()
    out: Dict[int, Tuple[float, torch.Tensor]] = {}

    for step_idx, a_idx in enumerate(action_sequence, start=1):
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1, actions=action)
        e1_call_counter["n_e1_calls"] += 1
        e1_call_counter["n_e1_calls_nonzero_action"] += (
            1 if float(action.abs().sum()) > 0.0 else 0
        )
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

        if step_idx in checkpoint_set:
            score = float(goal_state.goal_proximity(z_world_curr).item())
            out[step_idx] = (score, z_world_curr.detach().clone())

    return out


# ---------------------------------------------------------------------------
# Phase 4b (NON-GATING beyond the h=1 readiness floor; recorded so ITEM 2 has
# its baseline): real z_world sample at every horizon checkpoint. MODIFIED
# from 954: calls record_executed_action per step.
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    """n_samples independent random-policy rollouts from reset. Uses its own
    seed offset (+3000), mirroring V3-EXQ-954's Phase 4b, so it does not
    disturb the deterministic warmup-state RNG stream."""
    torch.manual_seed(seed + 3000)
    random.seed(seed + 3000)
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)
    samples_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for _ in range(n_samples):
        _, obs_dict = env.reset()
        agent.reset()
        for step_idx in range(1, max_h + 1):
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent.record_executed_action(action)
            _, _, done, _, obs_dict = env.step(action)
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            if step_idx in checkpoint_set:
                samples_by_h[step_idx].append(latent.z_world.detach())
            if done:
                break

    return samples_by_h


def _contrast_ratio(vectors: List[torch.Tensor]) -> Dict[str, float]:
    """CR = spread / ||centroid||, per zworld_near_static_characterisation_2026-07-18
    sec 2's offset-invariant statistic. Unchanged from V3-EXQ-954."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, dim]
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


# ---------------------------------------------------------------------------
# Phase 4c (LOAD-BEARING -- the ITEM 1 decision rule): one-step per-action
# divergence probe, DIRECT E1 channel. REDESIGNED from V3-EXQ-954: holds
# (z_self_0, z_world_0) FIXED across all K actions (no E2-mediated indirect
# routing) and varies ONLY the actions= one-hot passed to agent.e1(...) --
# see the "PHASE 4c -- REDESIGNED" module-docstring section for why.
# ---------------------------------------------------------------------------

def _one_step_action_divergence(
    agent: REEAgent,
    z_self_0: torch.Tensor,
    z_world_0: torch.Tensor,
    self_dim: int,
    e1_call_counter: Dict[str, int],
) -> Dict[str, Any]:
    """From the SAME Phase-2 warmup state, one deterministic single-step E1
    forward call per action (every action tested exactly once, agent.e1's
    hidden state reset before each so the K calls are independent). (z_self_0,
    z_world_0) is held BITWISE FIXED across all K calls; the ONLY varying
    input is the actions= one-hot -- the direct, isolated test of ITEM 1's own
    channel. On A_off (action_conditioned_transition=False) this reproduces
    the design doc's own "EXACTLY 0" signature by construction, since the
    total_curr fed to agent.e1 is bitwise identical across all K actions when
    the action channel is inert."""
    device = agent.device
    n_actions = agent.config.e2.action_dim
    total_curr = torch.cat([z_self_0, z_world_0], dim=-1)

    predictions: List[torch.Tensor] = []
    n_direct_supply_ok = 0
    for a_idx in range(n_actions):
        agent.e1.reset_hidden_state()
        action = _action_to_onehot(a_idx, n_actions, device)
        # Vacuity guard (2)/(3) analogue for this buffer-free design: verify,
        # AT THE CALL SITE, that the tensor about to be passed is genuinely
        # non-None and exactly one-hot before it is consumed.
        is_direct_supply_ok = (
            action is not None
            and float(action.abs().sum().item()) == 1.0
            and int((action != 0).sum().item()) == 1
        )
        n_direct_supply_ok += 1 if is_direct_supply_ok else 0
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1, actions=action)
        e1_call_counter["n_e1_calls"] += 1
        e1_call_counter["n_e1_calls_nonzero_action"] += 1 if is_direct_supply_ok else 0
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        predictions.append(z_world_next.detach())

    pairwise_dists = [
        float((predictions[i] - predictions[j]).norm().item())
        for i, j in itertools.combinations(range(len(predictions)), 2)
    ]
    cr = _contrast_ratio(predictions)

    return {
        "n_actions": n_actions,
        "pairwise_dists": pairwise_dists,
        "pairwise_dist_mean": float(sum(pairwise_dists) / len(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_min": float(min(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_max": float(max(pairwise_dists)) if pairwise_dists else 0.0,
        "contrast_ratio": cr,
        "n_direct_supply_ok": n_direct_supply_ok,
        "direct_action_supply_fraction": (n_direct_supply_ok / n_actions) if n_actions else 0.0,
    }


# ---------------------------------------------------------------------------
# Single-cell (seed, arm) runner
# ---------------------------------------------------------------------------

def run_cell(
    seed: int,
    arm: str,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    checkpoints: List[int],
    dry_run: bool = False,
) -> Dict[str, Any]:
    print(f"\n[EXQ-965] seed={seed} arm={arm}", flush=True)
    print(f"Seed {seed} Condition {arm}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim, arm)
    e1_call_counter = {"n_e1_calls": 0, "n_e1_calls_nonzero_action": 0}

    # Phase 0a: SD-070 sanctioned encoder warmup (unchanged from 954)
    print(f"[EXQ-965] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    # Phase 0b: bespoke E1/E2 single-step training, action-conditioned
    print(f"[EXQ-965] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode, e1_call_counter)

    # Phase 1: goal template (unchanged from 954)
    print("[EXQ-965] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state (unchanged from 954)
    print("[EXQ-965] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences (unchanged from 954)
    print(f"[EXQ-965] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4 (NON-GATING): score sequences at every horizon checkpoint
    print(f"[EXQ-965] Phase 4: scoring sequences at horizons {checkpoints} (non-gating)...", flush=True)
    scores_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for i, seq in enumerate(seqs):
        per_h = _score_sequence_e1coe_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
            e1_call_counter,
        )
        for h, (score, endpoint) in per_h.items():
            scores_by_h[h].append(score)
            endpoints_by_h[h].append(endpoint)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_score_var_by_h: Dict[int, float] = {}
    cr_rollout_by_h: Dict[int, Dict[str, float]] = {}
    for h in checkpoints:
        scores_t = torch.tensor(scores_by_h[h])
        e1coe_score_var_by_h[h] = float(scores_t.var().item()) if len(scores_by_h[h]) > 1 else 0.0
        cr_rollout_by_h[h] = _contrast_ratio(endpoints_by_h[h])
        print(
            f"  h={h:>2d}: e1coe_score_var={e1coe_score_var_by_h[h]:.6e} "
            f"CR_rollout={cr_rollout_by_h[h]['contrast_ratio']:.6e}",
            flush=True,
        )

    # Phase 4b: real z_world sample at every horizon checkpoint (NON-GATING
    # beyond the h=1 readiness floor)
    print(f"[EXQ-965] Phase 4b: sampling {n_real_samples} real trajectories at horizons {checkpoints}...", flush=True)
    real_samples_by_h = _collect_real_zworld_sample_multi_horizon(
        agent, env, seed, n_real_samples, checkpoints,
    )
    cr_real_by_h: Dict[int, Dict[str, float]] = {}
    cr_ratio_by_h: Dict[int, float] = {}
    for h in checkpoints:
        samples = real_samples_by_h[h]
        if len(samples) >= 2:
            cr_real_by_h[h] = _contrast_ratio(samples)
        else:
            cr_real_by_h[h] = {"spread": 0.0, "centroid_norm": 0.0, "contrast_ratio": float("nan"), "n": len(samples)}
        cr_real = cr_real_by_h[h]["contrast_ratio"]
        cr_roll = cr_rollout_by_h[h]["contrast_ratio"]
        cr_ratio_by_h[h] = (cr_roll / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        print(
            f"  h={h:>2d}: CR_real={cr_real:.6e} (n={cr_real_by_h[h]['n']}) "
            f"ratio={cr_ratio_by_h[h]:.6e}",
            flush=True,
        )

    # Phase 4c (LOAD-BEARING): direct-channel one-step per-action divergence
    print("[EXQ-965] Phase 4c: direct-channel one-step per-action divergence probe...", flush=True)
    action_probe = _one_step_action_divergence(agent, z_self_0, z_world_0, self_dim, e1_call_counter)
    cr_real_h1 = cr_real_by_h.get(1, {}).get("contrast_ratio", float("nan"))
    action_cr = action_probe["contrast_ratio"]["contrast_ratio"]
    ratio_action_vs_real_h1 = (
        (action_cr / cr_real_h1) if (cr_real_h1 == cr_real_h1 and cr_real_h1 > 0) else float("nan")
    )
    print(
        f"  K={action_probe['n_actions']} pairwise_dist mean={action_probe['pairwise_dist_mean']:.6e} "
        f"min={action_probe['pairwise_dist_min']:.6e} max={action_probe['pairwise_dist_max']:.6e} "
        f"cr_action_h1={action_cr:.6e} ratio_vs_CR_real(h=1)={ratio_action_vs_real_h1:.6e}",
        flush=True,
    )

    missing_action_calls = float(
        getattr(agent.e1, "_action_cond_missing_calls", 0)
    )
    buffer_stats = agent.e1_action_buffer_stats()
    direct_supply_fraction = (
        (e1_call_counter["n_e1_calls_nonzero_action"] / e1_call_counter["n_e1_calls"])
        if e1_call_counter["n_e1_calls"] else 0.0
    )
    print(
        f"  [vacuity] missing_action_calls={missing_action_calls:.0f} "
        f"direct_action_supply_fraction={direct_supply_fraction:.4f} "
        f"(internal_buffer_nonzero_fraction={buffer_stats.get('nonzero_fraction', 0.0):.4f}, "
        "recorded for cross-reference only -- see module docstring trap (2))",
        flush=True,
    )

    verdict = "PASS" if readiness_report.get("zworld_encoder_trained") else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "arm": arm,
        "readiness": readiness_report,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "checkpoints": checkpoints,
        "e1coe_score_var_by_h": e1coe_score_var_by_h,
        "cr_rollout_by_h": cr_rollout_by_h,
        "cr_real_by_h": cr_real_by_h,
        "cr_ratio_by_h": cr_ratio_by_h,
        "action_probe": action_probe,
        "ratio_action_vs_real_h1": ratio_action_vs_real_h1,
        "action_cr": action_cr,
        "cr_real_h1": cr_real_h1,
        "missing_action_calls": missing_action_calls,
        "e1_action_buffer_stats": buffer_stats,
        "direct_action_supply_fraction": direct_supply_fraction,
        "n_e1_calls_total": e1_call_counter["n_e1_calls"],
    }, agent


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    seeds: List[int],
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    checkpoints = sorted(set(
        [h for h in HORIZON_CHECKPOINTS_FULL if h <= rollout_horizon] + [rollout_horizon]
    ))

    cell_config_slice = {
        "world_dim": world_dim, "self_dim": self_dim,
        "n_train_episodes": n_train_episodes, "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences, "rollout_horizon": rollout_horizon,
        "n_warmup_steps": n_warmup_steps, "goal_max_steps": goal_max_steps,
        "zworld_p0_episodes": zworld_p0_episodes, "n_real_samples": n_real_samples,
        "env_kwargs": _env_kwargs(),
    }

    arm_results: List[Dict[str, Any]] = []
    agents_for_manifest = []
    for arm in ARM_ORDER:
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice={**cell_config_slice, "arm": arm, **ARM_CONFIGS[arm]},
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row, agent = run_cell(
                    seed=seed, arm=arm,
                    world_dim=world_dim, self_dim=self_dim,
                    n_train_episodes=n_train_episodes, steps_per_episode=steps_per_episode,
                    n_sequences=n_sequences, rollout_horizon=rollout_horizon,
                    n_warmup_steps=n_warmup_steps, goal_max_steps=goal_max_steps,
                    zworld_p0_episodes=zworld_p0_episodes, n_real_samples=n_real_samples,
                    checkpoints=checkpoints, dry_run=dry_run,
                )
                cell.stamp(row)
            arm_results.append(row)
            agents_for_manifest.append(agent)

    by_arm_seed: Dict[Tuple[str, int], Dict[str, Any]] = {
        (r["arm"], r["seed"]): r for r in arm_results
    }

    # ---- Readiness (P0) ----
    encoder_trained_per_cell = [bool(r["readiness"].get("zworld_encoder_trained")) for r in arm_results]
    p_encoder_trained_met = all(encoder_trained_per_cell)
    min_encoder_delta = min(
        r["readiness"].get("world_encoder_max_abs_delta", 0.0) for r in arm_results
    )

    p_real_nondegenerate_met = True
    min_real_samples_h1 = min(
        r["cr_real_by_h"].get(1, {}).get("n", 0) for r in arm_results
    )
    for r in arm_results:
        cr1 = r["cr_real_by_h"].get(1, {})
        ok = (
            cr1.get("contrast_ratio", float("nan")) == cr1.get("contrast_ratio", float("nan"))
            and cr1.get("contrast_ratio", 0.0) > CR_REAL_FLOOR
            and cr1.get("n", 0) >= MIN_REAL_SAMPLES_PER_HORIZON
        )
        p_real_nondegenerate_met = p_real_nondegenerate_met and ok

    on_arm_rows = [r for r in arm_results if r["arm"] != "A_off"]
    off_arm_rows = [r for r in arm_results if r["arm"] == "A_off"]

    max_missing_action_calls = max((r["missing_action_calls"] for r in on_arm_rows), default=0.0)
    p_no_missing_action_calls = max_missing_action_calls == 0.0

    min_direct_supply_fraction = min(
        (r["direct_action_supply_fraction"] for r in on_arm_rows), default=0.0
    )
    p_direct_action_supply = min_direct_supply_fraction >= 0.999

    off_ratios_h1 = [r["ratio_action_vs_real_h1"] for r in off_arm_rows]
    max_off_ratio_h1 = max(
        (v for v in off_ratios_h1 if v == v), default=float("nan")
    )
    p_off_reproduces_954 = (
        len(off_ratios_h1) == len(seeds)
        and all(v == v for v in off_ratios_h1)
        and max_off_ratio_h1 < OFF_ARM_COLLAPSE_CEILING
    )

    preconditions = [
        {
            "name": "encoder_trained",
            "kind": "readiness",
            "description": (
                "At least one split_encoder.world_encoder tensor moved during "
                "the Phase 0a SD-070 warmup, per every (seed, arm) cell."
            ),
            "measured": min_encoder_delta,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_encoder_trained_met,
        },
        {
            "name": "real_zworld_nondegenerate_h1",
            "kind": "readiness",
            "description": (
                "CR_real(h=1) is finite, positive, and backed by at least "
                f"{MIN_REAL_SAMPLES_PER_HORIZON} surviving real samples, every "
                "cell -- the same statistic the ratio_action_vs_real_h1 "
                "criterion's denominator uses (same-statistic discipline)."
            ),
            "measured": float(min_real_samples_h1),
            "threshold": float(MIN_REAL_SAMPLES_PER_HORIZON),
            "direction": "lower",
            "met": p_real_nondegenerate_met,
        },
        {
            "name": "no_missing_action_calls",
            "kind": "readiness",
            "description": (
                "Vacuity trap (1): agent.e1_action_buffer_stats()['missing_action_calls'] "
                "(E1DeepPredictor._action_cond_missing_calls) is 0 on every "
                "ON-arm cell -- confirms every actions= call this script made "
                "on an action-conditioned E1 supplied a real action, never a "
                "silent zero-fallback."
            ),
            "measured": max_missing_action_calls,
            "threshold": 0.0,
            "direction": "upper",
            "comparator": "<=",
            "met": p_no_missing_action_calls,
        },
        {
            "name": "direct_action_supply_fraction",
            "kind": "readiness",
            "description": (
                "Vacuity trap (2)/(3) analogue for this script's buffer-free "
                "design: the fraction of agent.e1(...) calls on ON-arm cells "
                "that received a genuine, verified non-None exactly-one-hot "
                "actions= argument, minimum across cells."
            ),
            "measured": min_direct_supply_fraction,
            "threshold": 0.999,
            "direction": "lower",
            "met": p_direct_action_supply,
        },
        {
            "name": "off_arm_reproduces_954",
            "kind": "readiness",
            "control": "V3-EXQ-954's own A_off-equivalent measurement (cr_ratio(h=1) 4.76e-07/5.39e-07)",
            "description": (
                "A_off's ratio_action_vs_real_h1 at h=1 stays below "
                f"{OFF_ARM_COLLAPSE_CEILING:.0e} on every seed -- confirms "
                "this run's OFF arm reproduces the known action-blindness "
                "signature before any ON-vs-OFF comparison is trusted."
            ),
            "measured": max_off_ratio_h1,
            "threshold": OFF_ARM_COLLAPSE_CEILING,
            "direction": "upper",
            "comparator": "<",
            "met": p_off_reproduces_954,
        },
    ]

    non_degenerate = bool(
        p_encoder_trained_met and p_real_nondegenerate_met
        and p_no_missing_action_calls and p_direct_action_supply
        and p_off_reproduces_954
    )

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        unmet_names = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = "P0 readiness unmet: " + ", ".join(unmet_names)
        status = "FAIL"
        evidence_direction = "non_contributory"
        per_cell_lift = {}
        lift_factor_used = None
        noise_ratio_measured = None
        criteria = []
        criteria_non_degenerate = {"C1_material_lift": False}
    else:
        degeneracy_reason = None

        # LIFT_FACTOR derived from THIS run's own OFF-arm cross-seed noise
        # ratio -- never an inherited historical constant.
        off_vals_sorted = sorted(v for v in off_ratios_h1 if v == v and v > 0)
        noise_ratio_measured = (
            (off_vals_sorted[-1] / off_vals_sorted[0]) if len(off_vals_sorted) >= 2 else 1.0
        )
        lift_factor_used = max(LIFT_FACTOR_ABS_FLOOR, LIFT_FACTOR_NOISE_MULTIPLE * noise_ratio_measured)
        print(
            f"\n[EXQ-965] Measured OFF-arm cross-seed noise ratio = {noise_ratio_measured:.4f}; "
            f"LIFT_FACTOR = max({LIFT_FACTOR_ABS_FLOOR}, {LIFT_FACTOR_NOISE_MULTIPLE}*noise) "
            f"= {lift_factor_used:.4f}",
            flush=True,
        )

        per_cell_lift: Dict[str, Dict[str, Any]] = {}
        for arm in ("B_action", "C_both"):
            for seed in seeds:
                on_row = by_arm_seed[(arm, seed)]
                off_row = by_arm_seed[("A_off", seed)]
                on_ratio = on_row["ratio_action_vs_real_h1"]
                off_ratio = off_row["ratio_action_vs_real_h1"]
                on_abs = on_row["action_cr"]
                off_abs = off_row["action_cr"]
                rel_lift = (on_ratio / off_ratio) if (off_ratio == off_ratio and off_ratio > 0) else float("nan")
                rel_ok = (rel_lift == rel_lift) and (rel_lift >= lift_factor_used)
                abs_floor_ok = (on_abs == on_abs) and (on_abs >= ABS_ACTION_CR_FLOOR)
                abs_lift_ok = (
                    (on_abs == on_abs) and (off_abs == off_abs)
                    and (on_abs >= ABS_LIFT_MULTIPLE * max(off_abs, 0.0))
                )
                cell_pass = bool(rel_ok and abs_floor_ok and abs_lift_ok)
                per_cell_lift[f"{arm}__seed{seed}"] = {
                    "arm": arm, "seed": seed,
                    "on_ratio_action_vs_real_h1": on_ratio,
                    "off_ratio_action_vs_real_h1": off_ratio,
                    "relative_lift": rel_lift,
                    "on_action_cr": on_abs,
                    "off_action_cr": off_abs,
                    "relative_lift_met": rel_ok,
                    "absolute_floor_met": abs_floor_ok,
                    "absolute_lift_met": abs_lift_ok,
                    "cell_pass": cell_pass,
                }

        n_cells_pass_by_arm: Dict[str, int] = {"B_action": 0, "C_both": 0}
        for key, v in per_cell_lift.items():
            if v["cell_pass"]:
                n_cells_pass_by_arm[v["arm"]] += 1

        arms_full_pass = [a for a, n in n_cells_pass_by_arm.items() if n == len(seeds)]
        arms_partial_pass = [
            a for a, n in n_cells_pass_by_arm.items() if 0 < n < len(seeds)
        ]

        if len(arms_full_pass) == 2:
            label = "action_conditioning_converts_both_arms"
            status = "PASS"
        elif len(arms_full_pass) == 1:
            label = "action_conditioning_converts_one_arm"
            status = "PASS"
        elif arms_partial_pass:
            label = "action_conditioning_converts_partial"
            status = "FAIL"
        else:
            label = "action_conditioning_no_lift"
            status = "FAIL"

        evidence_direction = "non_contributory"  # diagnostic, claim-free -- see docstring
        criteria = [
            {
                "name": "C1_material_lift",
                "load_bearing": True,
                "passed": bool(len(arms_full_pass) >= 1),
                "measured": max(
                    (v["relative_lift"] for v in per_cell_lift.values() if v["relative_lift"] == v["relative_lift"]),
                    default=float("nan"),
                ),
                "threshold": lift_factor_used,
                "statement": (
                    "At least one action-conditioned arm (B_action or C_both) "
                    "shows a relative lift in one-step per-action divergence "
                    ">= LIFT_FACTOR over its seed-matched A_off cell on every "
                    "seed, AND clears the absolute floor/lift-multiple tails, "
                    "per SD-e1-rollout-consistency-training ITEM 1's own "
                    "acceptance criterion (material lift toward E2's trained "
                    "per-action reference, NOT the C3/cr_ratio bars)."
                ),
            },
        ]
        criteria_non_degenerate = {"C1_material_lift": non_degenerate}

    print(f"\n[EXQ-965] Label: {label}", flush=True)
    print(f"[EXQ-965] Status: {status}", flush=True)

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_class": "diagnostic_disambiguation",
        "evidence_direction": evidence_direction,
        "seeds": seeds,
        "arms": ARM_ORDER,
        "arm_configs": ARM_CONFIGS,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences,
        "rollout_horizon": rollout_horizon,
        "horizon_checkpoints": checkpoints,
        "n_warmup_steps": n_warmup_steps,
        "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_lift_factor_abs_floor": LIFT_FACTOR_ABS_FLOOR,
        "registered_lift_factor_noise_multiple": LIFT_FACTOR_NOISE_MULTIPLE,
        "measured_lift_factor_used": lift_factor_used,
        "measured_off_arm_noise_ratio": noise_ratio_measured,
        "registered_abs_action_cr_floor": ABS_ACTION_CR_FLOOR,
        "registered_abs_lift_multiple": ABS_LIFT_MULTIPLE,
        "registered_off_arm_collapse_ceiling": OFF_ARM_COLLAPSE_CEILING,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_cell_lift": per_cell_lift,
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "non_gating_reference": {
                "note": (
                    "cr_ratio(h) / e1coe_score_var(h) / C3_VAR_THRESHOLD are "
                    "recorded per (seed, arm) below for ITEM 2's future baseline "
                    "ONLY -- they do NOT gate this run's PASS/FAIL. See the "
                    "module docstring's 'THE SINGLE MOST IMPORTANT INSTRUCTION' "
                    "section."
                ),
                "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
                "registered_c3_var_threshold": C3_VAR_THRESHOLD,
            },
        },
        "source_autopsy": "failure_autopsy_V3-EXQ-954_2026-08-29",
        "source_substrate_commit": "ree-v3 26557a3758",
        "source_design_doc": "sd_e1_rollout_consistency_training.md",
        "hypothesis_space_qid": "inv088_evaluator_degeneracy_cause",
    }

    # Flatten per-cell metrics (dict-of-dicts kept as-is -- JSON-serialisable)
    for r in arm_results:
        key = f"{r['arm']}_seed{r['seed']}"
        for k, v in r.items():
            if k not in ("seed", "arm"):
                result[f"cell_{key}_{k}"] = v

    result["arm_results"] = arm_results
    result["_agents_for_manifest"] = agents_for_manifest
    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-965: SD-e1-rollout-consistency-training ITEM 1 validation "
            "-- E1 action-conditioned transition per-action divergence (diagnostic)"
        )
    )
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--zworld-p0-episodes", type=int, default=ZWORLD_P0_EPISODES)
    parser.add_argument("--n-real-samples", type=int, default=N_REAL_SAMPLES)
    parser.add_argument("--dry-run", "--smoke-test", dest="dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dry_run:
        n_train = 2
        steps_ep = 50
        n_sequences = 5
        horizon = 10
        warmup = 5
        goal_max = 300
        zworld_p0 = 3
        n_real = 15  # > MIN_REAL_SAMPLES_PER_HORIZON so the smoke exercises the label branch too
        seeds = seeds[:1]
        print("[V3-EXQ-965] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        zworld_p0 = args.zworld_p0_episodes
        n_real = args.n_real_samples

    t0 = time.perf_counter()
    result = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        n_warmup_steps=warmup,
        goal_max_steps=goal_max,
        zworld_p0_episodes=zworld_p0,
        n_real_samples=n_real,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    agents_for_manifest = result.pop("_agents_for_manifest", [])

    full_config = {
        "seeds": seeds,
        "arms": ARM_ORDER,
        "arm_configs": ARM_CONFIGS,
        "world_dim": args.world_dim,
        "self_dim": args.self_dim,
        "n_train_episodes": n_train,
        "steps_per_episode": steps_ep,
        "n_sequences": n_sequences,
        "rollout_horizon": horizon,
        "n_warmup_steps": warmup,
        "goal_max_steps": goal_max,
        "zworld_p0_episodes": zworld_p0,
        "n_real_samples": n_real,
        "cr_real_floor": CR_REAL_FLOOR,
        "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD,
        "lift_factor_abs_floor": LIFT_FACTOR_ABS_FLOOR,
        "lift_factor_noise_multiple": LIFT_FACTOR_NOISE_MULTIPLE,
        "abs_action_cr_floor": ABS_ACTION_CR_FLOOR,
        "abs_lift_multiple": ABS_LIFT_MULTIPLE,
        "off_arm_collapse_ceiling": OFF_ARM_COLLAPSE_CEILING,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "unified_latent_mode": False,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=agents_for_manifest,
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Label: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("[V3-EXQ-965] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
