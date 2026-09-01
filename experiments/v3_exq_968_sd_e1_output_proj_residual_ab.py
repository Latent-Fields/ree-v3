"""
V3-EXQ-968 -- SD-e1-rollout-consistency-training "absolute-vs-residual branch":
does switching E1's output_proj readout from absolute-next-state to E2's
residual `z + delta(z, a)` form materially change per-action divergence at
the E1 output, on the trained, action-conditioned (ITEM 1 ON) substrate?

Claims: [] (diagnostic; discriminates a substrate-design question, not a
MECH-135/INV-088 retest -- neither claim is tagged or moved by this run;
both keep pending_retest_after_substrate: true regardless of outcome here).
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
V3-EXQ-965 (2026-08-30, confirmed autopsy failure_autopsy_V3-EXQ-965) validated
SD-e1-rollout-consistency-training ITEM 1: E1's action-conditioned transition
produces genuine per-action structure at the E1 output on a trained model
(ON arms 1.46e-03..2.19e-03 mean pairwise L2 vs an A_off arm at exactly 0.0),
and cr_ratio(h=1) rises 6455x-9775x to 2.67e-03..3.96e-03 -- but the ON arm is
STILL 25-37x short of the 0.1 evaluator bar (e1coe_score_var 5-7 orders below
0.002). So the ITEM-1-ON arm still shows crushed per-action divergence
relative to what the evaluator consumes, which is exactly the precondition
the design doc's own pre-registered branch fires on (see
REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md, "The
absolute-vs-residual branch -- FIRED"): "output_proj predicts the ABSOLUTE
next state, where E2 uses a residual z + delta(z, a) parameterisation. If the
item-1 ON arm still shows crushed per-action divergence at the E1 output,
that parameterisation is the next thing to test."

ree-v3 (2026-09-01, chip-20260901-sde1-outputproj-residual-knob) built the
knob: E1Config.output_proj_residual (default False) switches EVERY rollout
step in predict_long_horizon (both the ITEM-1 action-conditioned branch and
the legacy branch -- forward() delegates to predict_long_horizon(), so one
change covers both substrate_paths) from
    predicted = self.output_proj(output.squeeze(1))            # absolute
to
    predicted = state_i + self.output_proj(output.squeeze(1))  # residual
copied from e2_fast.py's self_forward/world_forward form. Contract:
tests/contracts/test_e1_output_proj_residual.py (16/16 passing, re-confirmed
here 2026-09-01) pins OFF bit-identical to the legacy absolute form, parameter
identity across the flag (no module/parameter added), the h=1 algebraic
identity ON == seed + OFF, non-vacuity in both rollout branches, gradient
still reaching output_proj, and from_dims reachability. It deliberately does
NOT assert that residual beats absolute -- that is this experiment, not the
contract.

THIS SCRIPT is the owed A/B: on the SAME ITEM-1-ON configuration (only
output_proj_residual toggled), does the residual form materially change the
per-action divergence signal at the E1 output and cr_ratio(h=1)?

RESIDUAL BASE CLARIFICATION (added after Step 4.5 red-team review, Fable,
finding 1). "state_i" / "seed" above and throughout this script is NOT the
raw (z_self, z_world) latent -- at rollout step 1 it is `prior_full`
(e1_deep.py:911-929: `context = context_memory.read(current_state)`,
`prior = prior_generator(cat([current_state, context]))`, then
`prior_full = cat([prior_self, prior])`), a LEARNED PROJECTION of the input
state, not the state itself. The design doc's and the substrate's own source
comment's "E2's z + delta(z, a) form" phrasing (e1_deep.py's own inline
comment: "SD-e1: residual parameterisation, E2's z + delta(z, a) form") is an
analogy to E2's residual structure, not a literal claim that E1's residual
base equals raw z -- this script inherits that analogy from the substrate's
own description rather than asserting it independently, and this note exists
so a reader does not over-read it. Consequence for interpretation: in the
residual arm (output_proj_residual=True), Phase 0b's training gradient gives
prior_generator a DIRECT additive path to the next-step world-latent
prediction (since `predicted = prior_full + output_proj(...)`), which is a
different training signal than a residual literally anchored on raw z_world
would produce. This script measures what the LANDED knob actually does
(agent.e1(...) is called with the real flag in every phase), which is
correct regardless of this naming question -- but a reader citing this run
against the design doc's literal "z + delta(z, a)" framing should read
"z" there as "prior_full's world slice", not raw z_world.

DV-SYMMETRY INVARIANCE -- MANDATORY DECLARATION (CLAUDE.md /queue-experiment
Step 3, "DV-SYMMETRY INVARIANCE"), INCLUDING A CORRECTION MADE DURING
AUTHORING (recorded here rather than silently fixed, per the skill's own
"verify before acting" discipline -- a design assumption was checked against
a smoke run and found wrong before this script was queued).

The contract's own h=1 algebraic identity
(`test_on_h1_is_exactly_seed_plus_absolute_readout`) shows that for a FIXED,
UNTRAINED-BETWEEN-COMPARISON model (both configs built from the SAME seed,
no gradient steps in between), residual predicted_k = seed + absolute
predicted_k at h=1, where seed is a PER-CELL, ACTION-INDEPENDENT constant --
so pairwise_dist / spread (which cancel any additive constant shared across
the K action-branches) would be analytically invariant to output_proj_residual
in that frozen-model setting, while contrast_ratio (which divides by
||centroid||) would not be. AN EARLIER DRAFT of this script mis-applied that
frozen-model identity to THIS experiment's actual design, in which
ARM_absolute and ARM_residual are INDEPENDENTLY TRAINED (Phase 0b) rather than
sharing frozen weights -- and drew a "sanity precondition" from it
(P_H1_IDENTITY_HOLDS: spread ratio ~1.0) that a --dry-run smoke immediately
falsified (spread differed by ~44% after just 2 training episodes, seed 42:
2.371e-04 absolute vs 3.414e-04 residual). The reason the frozen-model
identity does NOT transfer to a trained-per-arm design: Phase 0b's E1 loss is
`mse(e1_pred, total_curr.detach())`, and under output_proj_residual=True,
`e1_pred = prior_full(state_prev) + output_proj(LSTM_out)` (see the RESIDUAL
BASE CLARIFICATION above -- prior_full, not state_prev itself), so gradient
descent implicitly trains output_proj AND prior_generator (both now have a
direct additive path to the target, via backprop through the whole graph) to
predict the DELTA `total_curr - prior_full(state_prev)` rather than the
ABSOLUTE `total_curr` the ARM_absolute arm's output_proj is trained to
predict. The two arms therefore learn GENUINELY DIFFERENT E1 weights, not a
shared network with a flag toggled at read-out time -- the frozen-model
identity's premise (same weights, same input, same hidden-state init) does
not hold here at all. The flawed precondition was REMOVED, not patched with a
looser tolerance -- there is no principled tolerance for an assumption whose
premise does not apply.

CONSEQUENCE FOR THIS SCRIPT'S DESIGN: with independently-trained arms, NEITHER
pairwise_dist/spread (Phase 4c) NOR contrast_ratio/cr_ratio(h) (Phase 4/4b) is
analytically confounded by an additive-constant symmetry -- both are valid,
non-degenerate discrimination statistics, and neither is presumptively
"safer" than the other. The load-bearing criterion is cr_ratio(h=1) relative
lift (matching the design doc's own pre-registered readout and the
originating chip's explicit ask), with Phase 4c's pairwise_dist_mean /
contrast_ratio recorded alongside as a genuine (not merely sanity-check)
secondary cross-reference. The full multi-horizon cr_ratio(h) sweep (Phase
4/4b, unchanged from the 954/965 methodology, checkpoints up to
rollout_horizon) is recorded for free as NON-GATING context for a future
ITEM 2 build, per the Experimental Recording Standard's bias to over-record.

METHODOLOGY -- REUSE, NOT RE-DESIGN (from V3-EXQ-954/965)
---------------------------------------------------------------------------
Same env, same SD-070 z_world encoder warmup, same bespoke E1/E2 single-step
training, same goal template, same warmup state, same random candidate action
sequences, same multi-horizon cr_ratio(h)/CR_real(h) sweep, same direct-channel
Phase 4c one-step per-action divergence probe -- unchanged from V3-EXQ-965
except: (a) TWO arms instead of three, BOTH with
action_conditioned_transition=True, action_cond_unzero_self_slot=True (the
ITEM-1-ON configuration V3-EXQ-965 validated shows a real, still-crushed,
per-action signal), differing ONLY in output_proj_residual -- no A_off arm
needed here, since ITEM 1's own action-blindness-vs-signal question was
already answered by V3-EXQ-965; and (b) the decision rule below replaces
965's ON-vs-OFF lift comparison with a residual-vs-absolute comparison.
  ARM_absolute  action_conditioned_transition=True, action_cond_unzero_self_slot=True, output_proj_residual=False
  ARM_residual  action_conditioned_transition=True, action_cond_unzero_self_slot=True, output_proj_residual=True
action_cond_unzero_self_slot is fixed EXPLICITLY to True (matching
E1Config's shipped default) in BOTH arms so it cannot silently confound this
A/B (per substrate_queue.json's own open_decision note: V3-EXQ-965 returned a
null on this sub-flag -- C_both <= B_action on 3 of 4 h=1 comparisons -- so it
is left at its production default here rather than re-ablated; ITEM 2 does
not need to re-ablate it either).
Same seeds as 954/965 ([42, 123]).

READINESS (P0, self-route to substrate_not_ready_requeue on any miss -- never
a substrate-verdict label)
  P_ENCODER_TRAINED, P_REAL_ZWORLD_NONDEGENERATE_H1, P_NO_MISSING_ACTION_CALLS,
  P_DIRECT_ACTION_SUPPLY -- unchanged from V3-EXQ-965 (both arms here are
  "ON-arm" in 965's sense, so all four apply to both). Substrate wiring
  correctness (the knob reads the right residual base) is already covered by
  the contract (test_e1_output_proj_residual.py, 16/16 passing) at the
  frozen-model level where that identity actually applies -- this experiment
  does not re-derive it (see the DV-SYMMETRY INVARIANCE correction above for
  why a trained-per-arm re-derivation was attempted and withdrawn).

DECISION RULE (informational three-way classification -- NOT a pass/fail on
direction; per the contract's own stance, this experiment does not assert
residual beats absolute)
  Per (arm-pair, seed) at h=1: relative_lift = cr_ratio_h1[ARM_residual] /
  cr_ratio_h1[ARM_absolute]. LIFT_FACTOR is derived from THIS run's own
  ARM_absolute cross-seed noise ratio (never an inherited historical
  constant, per the V3-EXQ-936a both-tails-need-floors lesson -- same
  convention V3-EXQ-965 used, with ARM_absolute playing the reference-arm
  role A_off played there).
    residual_materially_exceeds_absolute: relative_lift >= LIFT_FACTOR on
      every seed.
    residual_materially_below_absolute: 1/relative_lift >= LIFT_FACTOR on
      every seed (the MECHANICALLY EXPECTED direction per the DV-symmetry
      note above -- NOT itself evidence the residual form is worse for
      per-action divergence, only that cr_ratio(h=1)'s centroid-norm
      denominator inflated; see interpretation.dv_symmetry_note in the
      output for the required caveat).
    residual_no_material_difference: neither bound clears on both seeds.
  STATUS is PASS whenever P0 readiness is met, in ALL THREE cases -- this is
  a pure discrimination, not a claim-hypothesis test with a directional bar;
  a "no material difference" or "below absolute" outcome is exactly as
  informative (and exactly as valid a PASS) as "exceeds absolute" would be.
  STATUS is FAIL only for substrate_not_ready_requeue (P0 unmet).

Re-derive brake (Step 2.5b): not applicable -- validates a substrate knob
(output_proj_residual) that did not exist before 2026-09-01
(chip-20260901-sde1-outputproj-residual-knob); no prior autopsy exists to
brake against at this granularity.

GOV-REUSE-1 (Step 2.4): checked cr_ratio_h1 / per-action-divergence readouts
across manifests tagged MECH-135/INV-088 or matching this experiment family
via REE_assembly/scripts/reanalysis_query.py (2026-09-01) -- 3 manifests
scanned (V3-EXQ-954, V3-EXQ-108a, V3-EXQ-108b), none carries the readout on a
substrate with output_proj_residual (the knob landed 2026-09-01, after every
existing manifest in the corpus, including V3-EXQ-965 itself). Not
recoverable -> proceed to author.

Step 2.5c (substrate-path overlap gate): SD-e1-rollout-consistency-training
itself (severity corrupting, substrate_paths e1_deep.py::forward and
::predict_long_horizon) IS this build's own target entry -- the sanctioned
exception per its own substrate_gate_note (this build TARGETS that entry, is
not routing around it). Three OTHER open corrupting entries were checked for
file-level overlap with this driver's imports (ree_core/agent.py,
ree_core/utils/config.py, ree_core/predictors/e1_deep.py) and VERIFIED
UNREACHED, not merely dismissed:
  * mode-governance-engagement (ree_core/agent.py -- SalienceCoordinator.tick()
    clamp discontinuity + the _et_commit boolean commitment latch at
    agent.py:6944): SalienceCoordinator.tick() is called ONLY from within
    REEAgent.select_action() (grep-verified: the sole call site sits inside
    that method's body). This driver family (954/965/968) never calls
    select_action() -- it steps the env directly with random actions and
    calls agent.sense()/agent.e1(...)/agent.e2.predict_next_self(...)/
    agent.record_executed_action(...) only. Unreached.
  * SD-082 (ree_core/pfc/lateral_pfc_analog.py::compute_bias,
    ree_core/agent.py -- trained bias-head consumer): lateral_pfc is
    constructed only under use_lateral_pfc_analog=True, not set by this
    script's REEConfig.from_dims(...) call (defaults False), and its
    consumer is likewise reached only via select_action(). Unreached on both
    counts.
  * contextmemory-write-path-addressing-degeneracy
    (e1_deep.py::ContextMemory.write -- hard-argmin write-address
    degeneracy): ContextMemory.write() is called only from
    E1DeepPredictor.update_from_observation() (grep-verified single call
    site), which this driver never calls -- every state transition here goes
    through predict_long_horizon()/forward() (context_memory.read() only).
    Unreached.

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: real (GoalState with a genuine collected template, unchanged from
V3-EXQ-954/965) -- recorded via z_goal_stream_stats at manifest-write time.
Bonus/cross-reference data only; the decision rule above routes entirely on
the Phase 4/4b/4c per-action-divergence and cr_ratio(h=1) readouts.

red-team (fable): see queue entry note for verdict + model.
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


EXPERIMENT_TYPE = "v3_exq_968_sd_e1_output_proj_residual_ab"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint exemption -- same shape and same
# justification as V3-EXQ-954/965: every scored quantity in this script is
# read off the LITERAL SAME agent object within a (seed, arm) cell. Cross-arm
# comparisons ARE across independently-constructed agents by design (that is
# the whole point of the A/B) and are seed-matched via arm_cell()'s RNG reset.
AGENT_SEED_ORDER_EXEMPT = (
    "Every within-cell comparison (action-vs-action, horizon-vs-horizon) is "
    "scored off the literal same agent object; cross-arm comparisons are the "
    "A/B itself and are seed-matched via arm_cell()."
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4               # unchanged from V3-EXQ-954/965 -- P_REAL_ZWORLD floor
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # NON-GATING here -- ITEM 2's target, recorded only
C3_VAR_THRESHOLD = 0.002           # NON-GATING here -- ITEM 2's target, recorded only
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches V3-EXQ-954/965
N_REAL_SAMPLES = 40                # per-checkpoint target sample count for CR_real(h)
MIN_REAL_SAMPLES_PER_HORIZON = 10  # readiness floor: surviving real samples per checkpoint
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]

# This A/B's OWN load-bearing thresholds (Phase 4/4b cr_ratio(h=1)).
# LIFT_FACTOR is derived per-run from ARM_absolute's own measured cross-seed
# noise ratio in run() -- never a fixed inherited constant.
LIFT_FACTOR_ABS_FLOOR = 3.0        # minimum relative-lift bar regardless of measured noise
LIFT_FACTOR_NOISE_MULTIPLE = 2.0   # required margin over the measured ARM_absolute noise ratio

ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARM_absolute": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
    },
    "ARM_residual": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": True,
    },
}
ARM_ORDER = ["ARM_absolute", "ARM_residual"]
SEEDS_DEFAULT = [42, 123]


# ---------------------------------------------------------------------------
# Helpers (unchanged from V3-EXQ-954/965 unless noted)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-954/965."""
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
        output_proj_residual=arm_cfg["output_proj_residual"],
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from 954/965)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_968 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_968_sd_e1_output_proj_residual_ab",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 single-step training. Unchanged from V3-EXQ-965
# (threads actions=action_prev through the E1 call; calls
# agent.record_executed_action(action_curr) after every chosen action).
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    e1_call_counter: Dict[str, int],
) -> None:
    """Train agent with random policy (E1 + E2 only). Calls agent.sense()
    exactly ONCE per env step (StepHarness invariant #1). sense() runs under
    torch.no_grad() throughout, so Phase 0a's now-trained encoder is never
    further disturbed."""
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
            agent.record_executed_action(action_curr)

            if latent_prev is not None:
                opt_e1.zero_grad()
                total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
                total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1)
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
# Phase 1: goal template (unchanged from 954/965)
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
# Phase 2: warmup state (unchanged from 954/965)
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
# Phase 3: generate candidate sequences (unchanged from 954/965)
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
# at every horizon checkpoint. Unchanged from V3-EXQ-965.
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
# its baseline): real z_world sample at every horizon checkpoint. Unchanged
# from V3-EXQ-965.
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    """n_samples independent random-policy rollouts from reset. Uses its own
    seed offset (+3000), mirroring V3-EXQ-954/965's Phase 4b, so it does not
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
    sec 2's offset-invariant statistic. Unchanged from V3-EXQ-954/965."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, dim]
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


# ---------------------------------------------------------------------------
# Phase 4c (secondary discrimination cross-reference -- see DV-SYMMETRY note
# in the module docstring for why this is a genuine, non-confounded readout
# for this trained-per-arm design, not merely a sanity check): one-step
# per-action divergence probe, DIRECT E1 channel. Unchanged from V3-EXQ-965's
# redesign.
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
    input is the actions= one-hot."""
    device = agent.device
    n_actions = agent.config.e2.action_dim
    total_curr = torch.cat([z_self_0, z_world_0], dim=-1)

    predictions: List[torch.Tensor] = []
    n_direct_supply_ok = 0
    for a_idx in range(n_actions):
        agent.e1.reset_hidden_state()
        action = _action_to_onehot(a_idx, n_actions, device)
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
    print(f"\n[EXQ-968] seed={seed} arm={arm}", flush=True)
    print(f"Seed {seed} Condition {arm}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim, arm)
    e1_call_counter = {"n_e1_calls": 0, "n_e1_calls_nonzero_action": 0}

    # Phase 0a: SD-070 sanctioned encoder warmup (unchanged from 954/965)
    print(f"[EXQ-968] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    # Phase 0b: bespoke E1/E2 single-step training, action-conditioned
    print(f"[EXQ-968] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode, e1_call_counter)

    # Phase 1: goal template (unchanged from 954/965)
    print("[EXQ-968] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state (unchanged from 954/965)
    print("[EXQ-968] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences (unchanged from 954/965)
    print(f"[EXQ-968] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4 (NON-GATING): score sequences at every horizon checkpoint
    print(f"[EXQ-968] Phase 4: scoring sequences at horizons {checkpoints} (non-gating)...", flush=True)
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
    print(f"[EXQ-968] Phase 4b: sampling {n_real_samples} real trajectories at horizons {checkpoints}...", flush=True)
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

    # Phase 4c (sanity/positive-control -- see DV-SYMMETRY note in module docstring)
    print("[EXQ-968] Phase 4c: direct-channel one-step per-action divergence probe...", flush=True)
    action_probe = _one_step_action_divergence(agent, z_self_0, z_world_0, self_dim, e1_call_counter)
    cr_real_h1 = cr_real_by_h.get(1, {}).get("contrast_ratio", float("nan"))
    action_cr = action_probe["contrast_ratio"]["contrast_ratio"]
    ratio_action_vs_real_h1 = (
        (action_cr / cr_real_h1) if (cr_real_h1 == cr_real_h1 and cr_real_h1 > 0) else float("nan")
    )
    print(
        f"  K={action_probe['n_actions']} pairwise_dist mean={action_probe['pairwise_dist_mean']:.6e} "
        f"min={action_probe['pairwise_dist_min']:.6e} max={action_probe['pairwise_dist_max']:.6e} "
        f"spread={action_probe['contrast_ratio']['spread']:.6e} "
        f"centroid_norm={action_probe['contrast_ratio']['centroid_norm']:.6e} "
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
        "recorded for cross-reference only, expected 0.0 -- _e1_tick() is never invoked)",
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

    # Both arms here are ITEM-1-ON ("on_arm" in V3-EXQ-965's sense) -- both
    # rows carry the missing-action-calls / direct-supply vacuity checks.
    max_missing_action_calls = max((r["missing_action_calls"] for r in arm_results), default=0.0)
    p_no_missing_action_calls = max_missing_action_calls == 0.0

    min_direct_supply_fraction = min(
        (r["direct_action_supply_fraction"] for r in arm_results), default=0.0
    )
    p_direct_action_supply = min_direct_supply_fraction >= 0.999

    # Cross-arm spread comparison at h=1 is recorded (per-seed ratio) as a
    # genuine discrimination cross-reference -- see the DV-SYMMETRY note in
    # the module docstring for why this is NOT expected to be ~1.0 in this
    # trained-per-arm design (that expectation was checked and withdrawn).
    h1_spread_ratios: Dict[int, float] = {}
    for seed in seeds:
        spread_abs = by_arm_seed[("ARM_absolute", seed)]["action_probe"]["contrast_ratio"]["spread"]
        spread_res = by_arm_seed[("ARM_residual", seed)]["action_probe"]["contrast_ratio"]["spread"]
        h1_spread_ratios[seed] = (spread_res / spread_abs) if spread_abs > 1e-12 else float("nan")

    # ---- P_CR_RATIO_H1_FINITE (added after Step 4.5 red-team review, Fable,
    # finding 2): cr_ratio(h=1) = CR_rollout(h=1)/CR_real(h=1) is the load-
    # bearing statistic the decision rule routes on. CR_real's finiteness is
    # already guarded by real_zworld_nondegenerate_h1 above, but CR_rollout's
    # centroid_norm (the Phase 4/4b rollout-endpoint contrast ratio, NOT the
    # Phase 4c action probe) can independently collapse toward 0 -- exactly
    # the collapse regime this SD is about -- making cr_ratio(h=1) NaN.
    # Without this guard a NaN cr_ratio on EITHER arm silently satisfies
    # neither the "exceeds" nor the "below" bound and falls through to
    # "residual_no_material_difference" with criteria_non_degenerate=True --
    # a label asserting a comparison that never actually happened. Self-route
    # instead.
    cr_ratio_h1_values = [
        by_arm_seed[(arm, seed)]["cr_ratio_by_h"].get(1, float("nan"))
        for arm in ARM_ORDER for seed in seeds
    ]
    p_cr_ratio_h1_finite = all(v == v and v > 0 for v in cr_ratio_h1_values)
    min_cr_ratio_h1 = min((v for v in cr_ratio_h1_values if v == v), default=float("nan"))

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
                "cell -- the same statistic cr_ratio(h=1)'s denominator uses "
                "(same-statistic discipline)."
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
                "agent.e1_action_buffer_stats()['missing_action_calls'] "
                "(E1DeepPredictor._action_cond_missing_calls) is 0 on every "
                "cell (both arms are ITEM-1-ON) -- confirms every actions= "
                "call this script made supplied a real action, never a "
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
                "The fraction of agent.e1(...) calls that received a "
                "genuine, verified non-None exactly-one-hot actions= "
                "argument, minimum across cells."
            ),
            "measured": min_direct_supply_fraction,
            "threshold": 0.999,
            "direction": "lower",
            "met": p_direct_action_supply,
        },
        {
            "name": "cr_ratio_h1_finite",
            "kind": "readiness",
            "description": (
                "cr_ratio(h=1) = CR_rollout(h=1)/CR_real(h=1) is finite and "
                "positive on every (arm, seed) cell. CR_real's finiteness is "
                "covered by real_zworld_nondegenerate_h1; this covers "
                "CR_rollout's centroid_norm independently collapsing toward "
                "0 -- the collapse regime this SD is about -- which would "
                "otherwise make cr_ratio(h=1) NaN and silently satisfy "
                "neither the 'exceeds' nor 'below' bound below, reading as "
                "residual_no_material_difference for a comparison that never "
                "happened."
            ),
            "measured": min_cr_ratio_h1,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_cr_ratio_h1_finite,
        },
    ]

    non_degenerate = bool(
        p_encoder_trained_met and p_real_nondegenerate_met
        and p_no_missing_action_calls and p_direct_action_supply
        and p_cr_ratio_h1_finite
    )

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        unmet_names = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = "P0 readiness unmet: " + ", ".join(unmet_names)
        status = "FAIL"
        evidence_direction = "non_contributory"
        per_seed_lift = {}
        lift_factor_used = None
        noise_ratio_measured = None
        criteria = []
        criteria_non_degenerate = {"C1_cr_ratio_h1_relative_lift": False}
    else:
        degeneracy_reason = None

        # LIFT_FACTOR derived from THIS run's own ARM_absolute cross-seed
        # noise ratio -- never an inherited historical constant.
        abs_h1_vals = sorted(
            by_arm_seed[("ARM_absolute", seed)]["cr_ratio_by_h"].get(1, float("nan"))
            for seed in seeds
        )
        abs_h1_vals = [v for v in abs_h1_vals if v == v and v > 0]
        noise_ratio_measured = (
            (abs_h1_vals[-1] / abs_h1_vals[0]) if len(abs_h1_vals) >= 2 else 1.0
        )
        lift_factor_used = max(LIFT_FACTOR_ABS_FLOOR, LIFT_FACTOR_NOISE_MULTIPLE * noise_ratio_measured)
        print(
            f"\n[EXQ-968] Measured ARM_absolute cross-seed noise ratio = {noise_ratio_measured:.4f}; "
            f"LIFT_FACTOR = max({LIFT_FACTOR_ABS_FLOOR}, {LIFT_FACTOR_NOISE_MULTIPLE}*noise) "
            f"= {lift_factor_used:.4f}",
            flush=True,
        )

        per_seed_lift: Dict[str, Dict[str, Any]] = {}
        for seed in seeds:
            abs_row = by_arm_seed[("ARM_absolute", seed)]
            res_row = by_arm_seed[("ARM_residual", seed)]
            abs_cr1 = abs_row["cr_ratio_by_h"].get(1, float("nan"))
            res_cr1 = res_row["cr_ratio_by_h"].get(1, float("nan"))
            rel_lift = (res_cr1 / abs_cr1) if (abs_cr1 == abs_cr1 and abs_cr1 > 0) else float("nan")
            exceeds = (rel_lift == rel_lift) and (rel_lift >= lift_factor_used)
            below = (rel_lift == rel_lift) and (rel_lift > 0) and ((1.0 / rel_lift) >= lift_factor_used)
            # Decomposition fields for cr_ratio_h1_* MUST be sourced from
            # cr_rollout_by_h[1] -- the statistic that actually composes
            # cr_ratio(h=1) = cr_rollout_by_h[1]/cr_real_by_h[1] -- not from
            # the Phase 4c action_probe, which is a DIFFERENT measurement (K
            # fixed actions from one held state, vs Phase 4's 40 random
            # candidate sequences). An earlier draft sourced these fields
            # from action_probe, which decomposes the wrong statistic (Step
            # 4.5 red-team review, Fable, finding 3). The action_probe
            # fields are kept below under their own action_probe_* names as
            # a genuine secondary cross-reference, not a decomposition.
            cr_rollout_abs = abs_row["cr_rollout_by_h"].get(1, {})
            cr_rollout_res = res_row["cr_rollout_by_h"].get(1, {})
            per_seed_lift[f"seed{seed}"] = {
                "seed": seed,
                "cr_ratio_h1_absolute": abs_cr1,
                "cr_ratio_h1_residual": res_cr1,
                "relative_lift_residual_over_absolute": rel_lift,
                "cr_rollout_spread_h1_absolute": cr_rollout_abs.get("spread"),
                "cr_rollout_spread_h1_residual": cr_rollout_res.get("spread"),
                "cr_rollout_centroid_norm_h1_absolute": cr_rollout_abs.get("centroid_norm"),
                "cr_rollout_centroid_norm_h1_residual": cr_rollout_res.get("centroid_norm"),
                "action_probe_spread_h1_absolute": abs_row["action_probe"]["contrast_ratio"]["spread"],
                "action_probe_spread_h1_residual": res_row["action_probe"]["contrast_ratio"]["spread"],
                "action_probe_centroid_norm_h1_absolute": abs_row["action_probe"]["contrast_ratio"]["centroid_norm"],
                "action_probe_centroid_norm_h1_residual": res_row["action_probe"]["contrast_ratio"]["centroid_norm"],
                "residual_materially_exceeds": exceeds,
                "residual_materially_below": below,
            }

        n_exceeds = sum(1 for v in per_seed_lift.values() if v["residual_materially_exceeds"])
        n_below = sum(1 for v in per_seed_lift.values() if v["residual_materially_below"])

        if n_exceeds == len(seeds):
            label = "residual_materially_exceeds_absolute"
        elif n_below == len(seeds):
            label = "residual_materially_below_absolute"
        elif n_exceeds > 0 or n_below > 0:
            label = "residual_vs_absolute_mixed_across_seeds"
        else:
            label = "residual_no_material_difference"

        status = "PASS"  # diagnostic discrimination -- informative in every direction
        evidence_direction = "non_contributory"  # diagnostic, claim-free -- see docstring
        criteria = [
            {
                "name": "C1_cr_ratio_h1_relative_lift",
                "load_bearing": True,
                "passed": True,  # this criterion CLASSIFIES, it does not gate PASS/FAIL
                "measured": max(
                    (v["relative_lift_residual_over_absolute"] for v in per_seed_lift.values()
                     if v["relative_lift_residual_over_absolute"] == v["relative_lift_residual_over_absolute"]),
                    default=float("nan"),
                ),
                "threshold": lift_factor_used,
                "statement": (
                    "cr_ratio(h=1) relative lift of ARM_residual over ARM_absolute, "
                    "per seed, against LIFT_FACTOR derived from ARM_absolute's own "
                    "cross-seed noise. See interpretation.dv_symmetry_note: a "
                    "'below' result is the mechanically expected direction "
                    "(centroid-norm inflation), not evidence the residual form "
                    "harms per-action divergence."
                ),
            },
        ]
        criteria_non_degenerate = {"C1_cr_ratio_h1_relative_lift": non_degenerate}

    print(f"\n[EXQ-968] Label: {label}", flush=True)
    print(f"[EXQ-968] Status: {status}", flush=True)

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
        "measured_absolute_arm_noise_ratio": noise_ratio_measured,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_seed_lift": per_seed_lift,
        "h1_spread_ratios_residual_over_absolute": h1_spread_ratios,
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "dv_symmetry_note": (
                "ARM_absolute and ARM_residual are INDEPENDENTLY TRAINED "
                "(Phase 0b), not a shared frozen E1 with a flag toggled at "
                "read-out time -- so the h=1 frozen-model algebraic identity "
                "the contract test pins (residual = seed + absolute, an "
                "additive constant that would cancel in pairwise_dist/spread) "
                "does NOT apply here: Phase 0b's loss implicitly trains "
                "output_proj (and, via backprop, the whole E1 network) toward "
                "a DIFFERENT target under each flag (the full next-state under "
                "absolute; the state-to-state DELTA under residual), so the "
                "two arms' learned weights genuinely differ. An earlier draft "
                "of this script assumed the frozen-model invariance would "
                "still hold and added a precondition on it; a --dry-run smoke "
                "falsified that (spread differed ~44% after 2 training "
                "episodes), and the precondition was removed rather than "
                "loosened. Consequently pairwise_dist/spread (Phase 4c) and "
                "contrast_ratio/cr_ratio(h) (Phase 4/4b) are BOTH genuine, "
                "non-confounded discrimination statistics for this design -- "
                "see h1_spread_ratios_residual_over_absolute above and the "
                "per-seed cr_ratio comparison in per_seed_lift. CAVEAT ADDED "
                "AFTER STEP 4.5 RED-TEAM REVIEW (Fable, finding 3): because "
                "cr_ratio(h=1) = cr_rollout(h=1)/cr_real(h=1) and cr_real(h=1) "
                "is arm-invariant per seed (measured directly from the "
                "environment, unaffected by output_proj_residual), the label "
                "reduces to cr_rollout(h=1) = spread/centroid_norm, whose "
                "centroid_norm denominator the manipulation can shift "
                "mechanically (the residual arm's predictions include an "
                "additive state-derived term absent from the absolute arm's). "
                "This can push cr_ratio(h=1) in EITHER direction independent "
                "of genuine per-action-divergence movement -- a "
                "'residual_materially_exceeds_absolute' OR a "
                "'residual_materially_below_absolute' label can each be a "
                "centroid-norm artifact rather than a real change in per-"
                "action divergence quality. Check "
                "cr_rollout_spread_h1_absolute vs cr_rollout_spread_h1_residual "
                "in per_seed_lift (the numerator alone, unaffected by the "
                "centroid-norm effect) before crediting either direction to "
                "genuine divergence improvement. See the module docstring's "
                "DV-SYMMETRY INVARIANCE section for the full derivation and "
                "the correction it records."
            ),
            "non_gating_reference": {
                "note": (
                    "cr_ratio(h) / e1coe_score_var(h) at h>1 are recorded per "
                    "(seed, arm) below for ITEM 2's future baseline ONLY -- they "
                    "do NOT gate this run's label. C3_VAR_THRESHOLD and "
                    "CR_ROLLOUT_COLLAPSE_RATIO are ITEM 2's bars, not this A/B's."
                ),
                "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
                "registered_c3_var_threshold": C3_VAR_THRESHOLD,
            },
        },
        "source_autopsy": "failure_autopsy_V3-EXQ-965_2026-08-30",
        "source_substrate_commit_item1": "ree-v3 26557a3758",
        "source_substrate_chip_residual_knob": "chip-20260901-sde1-outputproj-residual-knob",
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
            "V3-EXQ-968: SD-e1-rollout-consistency-training absolute-vs-residual "
            "output_proj A/B on the ITEM 1 ON arm (diagnostic)"
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
        seeds = seeds[:1] if len(seeds) > 1 else seeds
        print("[V3-EXQ-968] SMOKE TEST MODE", flush=True)
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
        print("[V3-EXQ-968] SMOKE TEST COMPLETE", flush=True)
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
