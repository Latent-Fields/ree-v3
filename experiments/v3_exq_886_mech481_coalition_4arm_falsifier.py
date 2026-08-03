"""V3-EXQ-886: MECH-481 / SD-091 typed-coalition 4-arm falsifier.

=================================================================================
STATUS: NOT QUEUED -- blocked on a competence/adaptation harness prerequisite
        (empirical premise probe, /queue-experiment Step 2.5a, 2026-08-03).
=================================================================================
This script is a COMPLETE, strict-validated, smoke-passing 4-arm falsifier INSTRUMENT
(correct recording core, per-cell arm_fingerprint, non-degeneracy gate, DV-symmetry
declaration, progress instrumentation, and a self-route to substrate_not_ready_requeue
when its own readiness gates fail). It is deliberately LEFT UNQUEUED because the
mandated empirical premise probe (Step 2.5a) showed the falsifier cannot yet yield a
clean, non-degenerate, interpretable verdict on the naive CausalGridWorldV2 harness:

  1. THE COALITION SUBSTRATE IS READY. It is wired + ablatable: SENSORY_RESAMPLE vs
     PROVENANCE_CHECK produce disjoint G_t (P1 templates_distinct = 7 sites), the
     coalition engages fully in ARM4 (P4 coalition_engages = 1.0), and the cost DV
     moves exactly as predicted (untyped over-recruits: ARM3 cost 75.0 vs ARM4 37.5).
  2. THE DOUBT TYPES ARE DISTINGUISHABLE. The non-degeneracy signal works strongly:
     ARM1 baseline percept_volatility dissociates the two doubt types ~95x
     (perceptual 13.29 vs provenance 0.14). P2 (non-degeneracy) is satisfiable.
  3. THE PERFORMANCE-RECOVERY DV HAS NO CLEAN SIGNAL ON THIS HARNESS (the blocker):
     (a) The untrained agent has near-ceiling HARM-AVOIDANCE competence but no
         goal-directed competence (~0 net reward; it mainly avoids harm).
     (b) The PERCEPTUAL (obs-noise) deficit stays ~0 even at sigma=4.0 (harm-avoidance
         is robust to percept noise), so SENSORY_RESAMPLE has no deficit to recover ->
         the perceptual arm is inert and the type x template interaction is
         uninterpretable.
     (c) The PROVENANCE (action-map-shift) deficit IS measurable at high swap depth,
         but the matched PROVENANCE_CHECK coalition does NOT recover it (it made it
         worse). Mechanistic reason (general, not a tuning artefact): the coalition's
         templates modulate WHICH subsystems are engaged / HOW readily the agent
         commits, but with NO ONLINE ADAPTATION during the trial, engaging a frozen e2
         more or delaying commitment cannot REPAIR a corrupted percept or a scrambled
         action-model. The claim's "type-selective performance recovery" presupposes a
         goal-directed, ONLINE-ADAPTING agent, which this harness does not provide.

PREREQUISITE / ROUTING (for a governance session to disposition): a runnable MECH-481
falsifier needs, before it is queued, a harness that supplies (i) goal-directed agent
competence (MECH-457 actor-critic + a training/curriculum warmup, or an equivalent
online-adapting policy) so there is task performance to protect, and (ii) a perturbation
battery whose deficit the coalition's recruit/suppress/gain mechanism can ACTUALLY
recover (i.e. one where more sensory engagement / delayed commitment genuinely resolves
the type-specific uncertainty online). That is /implement-substrate-scale harness work +
a falsifier REDESIGN, not another lettered iteration of this script. This module is kept
as the validated instrument scaffold that redesign should build on. Do NOT queue this
script as-is: it would self-route substrate_not_ready (P3) or emit a confounded verdict.

Step 7 (the one remaining step) of REE_assembly/docs/architecture/
sd_091_coalition_topology_control.md's "Minimum-viable V3 implementation path".
Steps 1-6 (the claustrum coalition-control substrate + its live wiring into
REEAgent.select_action and the 8 named consumer sites) landed 2026-08-02/03
(ree-v3 87a7e2115c). This experiment is the first ablation of that mechanism.

EXPERIMENT_PURPOSE = diagnostic. This is substrate/mechanism validation, not yet
governance evidence -- it discriminates the design doc's own three Falsification-
signature categories: "mechanism does real, type-selective work" vs "mechanism
present but typing does no work" vs "primitive present but miswired / substrate
not ready". Diagnostic runs are excluded from governance confidence/conflict
scoring; the interpretation block below is falsifiable by the pipeline
(preconditions[] + criteria_non_degenerate{} + a load-bearing readiness gate).

SLEEP: not used (no use_sleep_loop / sws_enabled / rem_enabled).

=================================================================================
THE MECHANISM UNDER TEST (from claims.yaml MECH-481 what_would_answer + doc Sec 1-2)
=================================================================================
The claustrum CoalitionController exposes a graph-valued control output G_t: a
per-tick, per-subsystem recruit/suppress/gain vector (star topology). Two typed
templates are built:
  SENSORY_RESAMPLE : recruits e1_sensory_encoder + e2_fast_forward_model
                     (write_gate 0.9), widens e3_candidate_count (channel_gain 1.5).
  PROVENANCE_CHECK : recruits hippocampal_anchor_set / persistence_appraisal /
                     e3_commitment_monitor (write_gate 0.9), suppresses
                     motor_commitment (0.4) + hippocampal_write_consolidation (0.5)
                     -> readiness_margin x0.36, i.e. delays commitment / re-appraises.
write_gate() is attenuation-only (<=1.0); channel_gain() is the parametric axis.
The driver plays the Monitor+Classify role (doc Sec 1 "Injection, not derivation"):
it detects a typed discrepancy at a scripted trial onset and injects the matching
(or, per arm, a wrong / undifferentiated / parametric-only / no) coalition.

=================================================================================
TASK BATTERY -- two doubt types at MATCHED base config (doc Sec "Falsification")
=================================================================================
CausalGridWorldV2 (size 5, 1 hazard, 1 resource, use_proxy_fields=True -- the exact
config the wiring contracts validated on). Each trial: SETTLE_STEPS clean steps to
establish baseline competence + settle E3 running variance, then a persistent
perturbation from the doubt onset to episode end:
  PERCEPTUAL doubt : additive Gaussian noise on obs_world (exteroceptive percept
                     unreliable; env dynamics unchanged). The right corrective is to
                     resample sensory input + broaden the candidate search ->
                     SENSORY_RESAMPLE is the matched template.
  PROVENANCE doubt : a transposition of the env action->movement map (env._action_map)
                     at the onset (the agent's LEARNED forward model / committed
                     policy is now stale; the percept stays clean). The right
                     corrective is to delay commitment + re-appraise provenance ->
                     PROVENANCE_CHECK is the matched template.
Both perturbations are driver-controlled and persist through the post-onset window;
matched difficulty is MEASURED (Arm1 degradation per type is reported, and a large
mismatch is surfaced as a caveat, not silently assumed).

=================================================================================
THE FOUR ARMS (+ one selectivity control) -- coalition POLICIES over both types
=================================================================================
  ARM1_NONE     (monitoring-only): inject nothing. use_coalition_controller=True but
                no coalition requested -> bit-identical to OFF (wiring contract W2).
                The driver still LOGS the typed discrepancy signals (the "monitoring").
                Predicted deficit: accurate detection, no behavioural improvement.
  ARM2_PARAM    (parametric-only): inject a coalition with EMPTY participating/
                suppressed and channel_gain={e3_candidate_count:1.5} only -- the
                parametric axis, no subsystem recruitment. Same for both doubt types.
                Predicted deficit: generic caution/search, no type-specific resolution.
  ARM3_UNTYPED  (undifferentiated coalition): inject the UNION of both templates'
                recruit/suppress/gain -- one broadcast "REFLECT harder" set applied
                regardless of doubt type. Predicted deficit: costly over-recruitment /
                interference (recruits provenance subsystems on perceptual trials and
                vice versa).
  ARM4_TYPED    (typed coalition, THE MECHANISM): SENSORY_RESAMPLE on perceptual
                trials, PROVENANCE_CHECK on provenance trials -- discrepancy type
                MATCHED to template.
  CTRL_MISMATCH (selectivity control, not one of the 4 claim arms): SENSORY_RESAMPLE
                on provenance trials and PROVENANCE_CHECK on perceptual trials --
                deliberately WRONG template. Isolates whether ARM4's benefit is
                type-SELECTIVE (matched > mismatched) rather than "any coalition helps".
Plus ARM1_CLEAN: ARM1_NONE with NO perturbation, the no-doubt performance reference
used to measure how much each doubt type degrades baseline competence.

=================================================================================
DEPENDENT VARIABLES
=================================================================================
  performance : cumulative env reward (harm_signal; +benefit / -harm) over the
                post-onset window [doubt_onset, episode_len). Higher = better coping.
  cost        : sum over post-onset steps of the ACTIVE coalition's recruitment
                footprint |recruited union suppressed| + max(0, cc_gain - 1). This is
                the "computational cost" the claim's Arm3-vs-Arm4 contrast is about
                (over-recruitment). ARM1 cost = 0; ARM2 cost = candidate widening only.
  discrepancy : percept_volatility = ||delta raw obs_world|| ; model_pe =
                ||z_world_cur - e2.world_forward(z_world_prev, action_prev)||.
                Logged in every arm; ARM1's baseline profile drives NON-DEGENERACY.

=================================================================================
PRECONDITIONS (interpretation.preconditions[], recomputed by the indexer)
=================================================================================
  P1 templates_distinct              (readiness) : the two templates produce disjoint
        recruited target sets with write_gate < 1.0 at >= 4 sites. Rules out the
        doc's "primitive present but miswired" signature (#3). Measured at setup.
  P2 nondegeneracy_signals_dissociate(readiness, LOAD-BEARING gate) : ARM1's baseline
        discrepancy signals SEPARATE perceptual vs provenance trials (standardized
        cluster separation > SEP_FLOOR). If the agent cannot distinguish doubt types
        at all, the Arm3-vs-Arm4 typing manipulation is vacuous -> the run is
        SUBSTRATE_NOT_READY, per the claim's own NON-DEGENERACY clause -- NOT a verdict.
  P3 perturbation_degrades_baseline  (readiness) : ARM1_CLEAN performance exceeds
        ARM1_NONE (perturbed) by > DEGRADE_FLOOR for BOTH doubt types. If a doubt type
        does not degrade competence there is no headroom for a coalition to recover;
        below-floor -> substrate_not_ready (nothing to measure).
  P4 coalition_engages               (readiness) : in ARM4, the fraction of post-onset
        E3 ticks with an ACTIVE coalition (write_gate != 1.0 somewhere) > ENGAGE_FLOOR.
        Rules out the "decisive readout never fired" class (V3-EXQ-475b).
Any P2/P3/P4 unmet -> outcome FAIL, interpretation.label = substrate_not_ready_requeue
(NOT a substrate-verdict label). P1 unmet -> primitive_miswired_requeue.

=================================================================================
CRITERIA (PASS) -- combination_rule recorded explicitly
=================================================================================
  C1 (MECH-481 typing does discriminative work, LOAD-BEARING):
        matched_improvement > mismatched_improvement + SEL_MARGIN
        AND matched_improvement > untyped_improvement  + SEL_MARGIN
        (targeting the right subsystems beats both the wrong template and
         undifferentiated recruitment).
  C2 (SD-091 recruitment adds beyond parametric, LOAD-BEARING):
        matched_improvement > param_improvement + PARAM_MARGIN.
  C3 (cost, corroborating, NOT load-bearing):
        cost[ARM4_TYPED] < cost[ARM3_UNTYPED] (typed is cheaper than over-recruitment).
  combination_rule: PASS iff (C1 AND C2). FAIL if C1 fails (typing does no work:
    Arm4 ~ Arm3) OR C2 fails (parametric suffices: Arm2 ~ Arm4). C3 corroborates the
    cost half of the PASS narrative but does not gate.

Self-route labels:
  precondition-fail  -> substrate_not_ready_requeue / primitive_miswired_requeue
  C1 & C2 pass       -> typed_coalition_selective_supports   (PASS)
  C1 fails           -> typing_does_no_discriminative_work    (FAIL, weakens MECH-481)
  C2 fails           -> parametric_suffices_g_t_superfluous   (FAIL, weakens SD-091)

=================================================================================
DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per arm)
=================================================================================
The performance DV is cumulative env reward, produced by the realised action
sequence; the cost DV is a direct count of the active coalition's recruited targets.
Each arm's manipulation acts on candidate_count (ARM2/ARM3/ARM4-perceptual),
readiness_margin / commitment (ARM3/ARM4-provenance), and e1/e2 precision --
quantities that CHANGE the argmax action selection and thus the reward trajectory.
No arm's manipulation is a uniform additive constant on a selection score, a monotone
rescaling of a rank/argmax DV, or a permutation of interchangeable units invisible to
a set-aggregate DV: candidate_count widening changes the SIZE of the candidate set
(not a rescaling), commitment attenuation changes WHETHER an action is committed (a
threshold effect, not an order-preserving map), and the reward DV integrates the
resulting distinct trajectories. The cost DV is the manipulation's own footprint,
not a quantity invariant under it. Hence every arm's measured delta is a genuine
measurement, not an arithmetic identity fixed before the run.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
_REE_V3_ROOT = _THIS.parents[1]
if str(_REE_V3_ROOT) not in sys.path:
    sys.path.insert(0, str(_REE_V3_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.claustrum.control_demand import ControlDemandType
from ree_core.claustrum.coalition_controller import CoalitionState
from ree_core.claustrum.coalition_templates import COALITION_TEMPLATES

from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome


EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_886_mech481_coalition_4arm_falsifier"
CLAIM_IDS = ["MECH-481", "SD-091"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# -- pre-registered constants (thresholds defined here, never derived post-hoc) --
SEEDS = [0, 1, 2, 3, 4]           # seed 44 deliberately avoided (reef-config instability)
N_TRIALS = 6                       # trial episodes per (policy, type, seed) cell
SETTLE_STEPS = 50                  # clean settle prefix per trial (baseline competence)
POST_STEPS = 50                    # post-onset window (perturbation persists here)
EPISODE_STEPS = SETTLE_STEPS + POST_STEPS

ENV_KW = dict(size=5, num_hazards=1, num_resources=1, use_proxy_fields=True)
SELF_DIM = 16
WORLD_DIM = 16
# The agent's action space (matches the wiring contracts' from_dims(action_dim=4)):
# the 4 movement actions. The env may expose a larger action_dim (e.g. a CONSUME
# action at index 4); the agent never selects it, and env.step accepts a movement
# index unchanged. e2.world_forward expects an AGENT_ACTION_DIM-wide one-hot.
AGENT_ACTION_DIM = 4

PERCEPTUAL_NOISE_SIGMA = 0.6       # obs_world Gaussian noise magnitude (perceptual doubt)
PROVENANCE_SWAP_PAIRS = [(0, 3), (1, 2)]  # action_map transpositions (provenance doubt)

# pre-registered gate thresholds
SEP_FLOOR = 0.5                    # P2 non-degeneracy: >= 0.5 SD cluster separation
DEGRADE_FLOOR = 0.02               # P3: min per-type baseline degradation from clean
ENGAGE_FLOOR = 0.5                 # P4: >= 50% of post-onset E3 ticks coalition-active
DISTINCT_SITES_FLOOR = 4           # P1: >= 4 sites where the two templates differ

# pre-registered effect margins (on the reward-improvement scale)
SEL_MARGIN = 0.02                  # C1 selectivity margin
PARAM_MARGIN = 0.02                # C2 recruitment-beyond-parametric margin

TRIAL_TYPES = ["perceptual", "provenance"]
# scored policy arms + the selectivity control
POLICIES = ["ARM1_NONE", "ARM2_PARAM", "ARM3_UNTYPED", "ARM4_TYPED", "CTRL_MISMATCH"]

_SENSORY = ControlDemandType.SENSORY_RESAMPLE
_PROVENANCE = ControlDemandType.PROVENANCE_CHECK


# ----------------------------------------------------------------------
# Coalition-policy -> CoalitionState builder (the driver's Classify step)
# ----------------------------------------------------------------------
def _union_template() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Undifferentiated (Arm3) coalition = union of both MVP templates."""
    part: Dict[str, float] = {}
    supp: Dict[str, float] = {}
    gain: Dict[str, float] = {}
    for dt in (_SENSORY, _PROVENANCE):
        t = COALITION_TEMPLATES[dt]
        part.update(dict(t.participating))
        supp.update(dict(t.suppressed))
        gain.update(dict(t.channel_gain))
    return part, supp, gain


def _coalition_state_for(
    policy: str, trial_type: str, opened_tick: int, duration: int
) -> Optional[CoalitionState]:
    """Return the CoalitionState this (policy, trial_type) injects at the doubt
    onset, or None for the monitoring-only arm. Mirrors the driver playing the
    Monitor+Classify role -- doc Section 1 'Injection, not derivation'.
    """
    if policy == "ARM1_NONE":
        return None
    if policy == "ARM2_PARAM":
        # Parametric axis only: candidate widening, NO subsystem recruitment.
        return CoalitionState(
            demand_type=_SENSORY,  # nominal tag; participating/suppressed empty
            participating={},
            suppressed={},
            channel_gain={"e3_candidate_count": 1.5},
            opened_tick=opened_tick,
            max_duration_ticks=duration,
        )
    if policy == "ARM3_UNTYPED":
        part, supp, gain = _union_template()
        return CoalitionState(
            demand_type=_SENSORY,  # nominal tag; set is undifferentiated by design
            participating=part,
            suppressed=supp,
            channel_gain=gain,
            opened_tick=opened_tick,
            max_duration_ticks=duration,
        )
    # ARM4_TYPED: matched template; CTRL_MISMATCH: the wrong one.
    if policy == "ARM4_TYPED":
        dt = _SENSORY if trial_type == "perceptual" else _PROVENANCE
    elif policy == "CTRL_MISMATCH":
        dt = _PROVENANCE if trial_type == "perceptual" else _SENSORY
    else:
        raise ValueError(f"unknown policy {policy!r}")
    tmpl = COALITION_TEMPLATES[dt]
    return CoalitionState(
        demand_type=dt,
        participating=dict(tmpl.participating),
        suppressed=dict(tmpl.suppressed),
        channel_gain=dict(tmpl.channel_gain),
        opened_tick=opened_tick,
        max_duration_ticks=duration,
    )


def _inject(agent: REEAgent, state: Optional[CoalitionState]) -> None:
    """Inject a driver-defined coalition into the live controller. Direct _active
    append is the codebase's own test/experiment injection idiom (see
    tests/contracts/test_sd091_coalition_controller_wiring.py W7, which sets
    controller._active directly): request_coalition() only serves the 2 templated
    types, so custom Arm2/Arm3/CTRL coalitions have no template path.
    """
    if state is None or agent.coalition is None:
        return
    agent.coalition._active.append(state)


def _coalition_footprint(agent: REEAgent) -> float:
    """Recruitment footprint of the currently-active coalition(s): number of
    distinct recruited/suppressed targets + candidate widening above 1.0. This is
    the 'computational cost' the Arm3-vs-Arm4 contrast is about (over-recruitment).
    """
    if agent.coalition is None:
        return 0.0
    active = agent.coalition.active_coalitions
    if not active:
        return 0.0
    targets = set()
    for st in active:
        targets |= set(st.participating.keys())
        targets |= set(st.suppressed.keys())
    cc_gain = float(agent.coalition.channel_gain("e3_candidate_count"))
    return float(len(targets)) + max(0.0, cc_gain - 1.0)


def _coalition_active(agent: REEAgent) -> bool:
    if agent.coalition is None:
        return False
    return len(agent.coalition.active_coalitions) > 0


# ----------------------------------------------------------------------
# One trial episode
# ----------------------------------------------------------------------
def _run_trial(
    agent: REEAgent,
    env: Any,
    policy: str,
    trial_type: str,
    perturbed: bool,
    settle_steps: int,
    post_steps: int,
) -> Dict[str, Any]:
    """Run one trial episode. Returns per-trial readouts.

    Timeline: [0, settle) clean -> at `settle` apply the persistent perturbation and
    inject the arm's coalition -> [settle, settle+post) measure. When perturbed is
    False (ARM1_CLEAN reference) no perturbation is applied and no coalition injected.
    """
    agent.reset()
    _flat, od = env.reset()
    # env.reset() deliberately does NOT reset env._action_map (nor _maybe_shift_world_rule
    # state), so a provenance transposition would otherwise persist into the next trial in
    # this cell. Snapshot the canonical map and restore it at trial end.
    _orig_action_map = dict(env._action_map)

    total_steps = settle_steps + post_steps
    post_reward = 0.0
    cost_accum = 0.0
    percept_vols: List[float] = []
    model_pes: List[float] = []
    n_post_e3_active = 0
    n_post_e3 = 0

    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None
    obs_world_prev: Optional[torch.Tensor] = None
    injected = False

    for step in range(total_steps):
        in_window = step >= settle_steps

        body = od["body_state"]
        world = od["world_state"]
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)

        # --- perturbation onset (once, at the window boundary) ---
        if in_window and perturbed and not injected:
            if trial_type == "provenance":
                # Transpose the env action->movement map: the learned model is stale.
                for a, b in PROVENANCE_SWAP_PAIRS:
                    env._action_map[a], env._action_map[b] = (
                        env._action_map[b],
                        env._action_map[a],
                    )
            # Inject this arm's coalition, sized to persist through the window.
            state = _coalition_state_for(
                policy, trial_type,
                opened_tick=int(agent._step_count),
                duration=post_steps + 5,
            )
            _inject(agent, state)
            injected = True

        # --- perceptual doubt: additive Gaussian obs_world noise while in window ---
        world_in = world
        if in_window and perturbed and trial_type == "perceptual":
            world_in = world + torch.randn_like(world) * PERCEPTUAL_NOISE_SIGMA

        # --- manual act loop (so E3 running variance / discrepancy signals are live) ---
        with torch.no_grad():
            latent = agent.sense(body, world_in)
            z_world_cur = latent.z_world.detach()
            if z_world_prev is not None and action_prev is not None:
                pred = agent.e2.world_forward(z_world_prev, action_prev)
                resid = z_world_cur - pred.detach()
                agent.e3.update_running_variance(resid)
                model_pe = float(torch.linalg.vector_norm(resid).item())
            else:
                model_pe = 0.0

            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        act_idx = (
            int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        )
        act_idx = act_idx % AGENT_ACTION_DIM
        action_prev = torch.zeros(1, AGENT_ACTION_DIM)
        action_prev[0, act_idx] = 1.0

        # percept volatility on the (possibly noised) exteroceptive input the agent saw
        if obs_world_prev is not None:
            percept_vol = float(
                torch.linalg.vector_norm(world_in.detach() - obs_world_prev).item()
            )
        else:
            percept_vol = 0.0
        obs_world_prev = world_in.detach()
        z_world_prev = z_world_cur

        _obs, reward, done, _info, od = env.step(act_idx)

        if in_window:
            post_reward += float(reward)
            if ticks_d.get("e3_tick", False):
                n_post_e3 += 1
                if _coalition_active(agent):
                    n_post_e3_active += 1
            cost_accum += _coalition_footprint(agent)
            percept_vols.append(percept_vol)
            model_pes.append(model_pe)

        if done:
            _flat, od = env.reset()

    # restore the canonical action map so the next trial starts unperturbed
    env._action_map = dict(_orig_action_map)

    return {
        "policy": policy,
        "trial_type": trial_type,
        "perturbed": perturbed,
        "performance": post_reward,
        "cost": cost_accum,
        "percept_volatility_mean": statistics.fmean(percept_vols) if percept_vols else 0.0,
        "model_pe_mean": statistics.fmean(model_pes) if model_pes else 0.0,
        "n_post_e3": n_post_e3,
        "n_post_e3_coalition_active": n_post_e3_active,
    }


# ----------------------------------------------------------------------
# One (policy, trial_type, seed) cell
# ----------------------------------------------------------------------
def _build_agent(seed: int) -> REEAgent:
    torch.manual_seed(123 + seed)
    env0 = CausalGridWorldV2(seed=seed, **ENV_KW)
    _f, od = env0.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=od["body_state"].shape[-1],
        world_obs_dim=od["world_state"].shape[-1],
        action_dim=AGENT_ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        use_coalition_controller=True,  # ARM1 injects nothing -> bit-identical off (W2)
    )
    agent = REEAgent(cfg)
    agent.eval()
    return agent


def _run_cell(
    policy: str,
    trial_type: str,
    seed: int,
    perturbed: bool,
    n_trials: int,
    settle_steps: int,
    post_steps: int,
    script_path: Path,
) -> Dict[str, Any]:
    cond = f"{policy}:{trial_type}{'' if perturbed else ':clean'}"
    print(f"Seed {seed} Condition {cond}", flush=True)

    config_slice = {
        "policy": policy,
        "trial_type": trial_type,
        "perturbed": perturbed,
        "env_kw": ENV_KW,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "settle_steps": settle_steps,
        "post_steps": post_steps,
        "n_trials": n_trials,
        "perceptual_noise_sigma": PERCEPTUAL_NOISE_SIGMA,
        "provenance_swap_pairs": PROVENANCE_SWAP_PAIRS,
    }

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=script_path,
        config_slice_declared=True,
    ) as cell:
        env = CausalGridWorldV2(seed=seed, **ENV_KW)
        agent = _build_agent(seed)
        trials = []
        for i in range(n_trials):
            trials.append(
                _run_trial(agent, env, policy, trial_type, perturbed, settle_steps, post_steps)
            )
            print(f"  [train] {cond} seed={seed} ep {i + 1}/{n_trials}", flush=True)
        perf = [t["performance"] for t in trials]
        cost = [t["cost"] for t in trials]
        pvol = [t["percept_volatility_mean"] for t in trials]
        mpe = [t["model_pe_mean"] for t in trials]
        e3_tot = sum(t["n_post_e3"] for t in trials)
        e3_active = sum(t["n_post_e3_coalition_active"] for t in trials)
        row = {
            "arm_id": f"{policy}__{trial_type}{'' if perturbed else '__clean'}",
            "policy": policy,
            "trial_type": trial_type,
            "perturbed": perturbed,
            "seed": seed,
            "n_trials": n_trials,
            "performance_mean": statistics.fmean(perf),
            "performance_per_trial": perf,
            "cost_mean": statistics.fmean(cost),
            "percept_volatility_mean": statistics.fmean(pvol),
            "model_pe_mean": statistics.fmean(mpe),
            "n_post_e3": e3_tot,
            "n_post_e3_coalition_active": e3_active,
            "coalition_active_frac": (e3_active / e3_tot) if e3_tot else 0.0,
        }
        cell.stamp(row)
    # Per-cell completion marker for the runner's progress bar (one per seed x
    # condition). NOT the experiment verdict -- the scientific outcome is in the
    # manifest interpretation. A cell that ran to completion prints PASS.
    print("verdict: PASS", flush=True)
    return row


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------
def _templates_distinct_count() -> int:
    """P1: number of consumer sites where SENSORY vs PROVENANCE templates give a
    different (write_gate-relevant) recruit/suppress state. Disjoint by construction.
    """
    s = COALITION_TEMPLATES[_SENSORY]
    p = COALITION_TEMPLATES[_PROVENANCE]
    s_sites = set(s.participating) | set(s.suppressed)
    p_sites = set(p.participating) | set(p.suppressed)
    return len(s_sites ^ p_sites)  # symmetric difference: sites one names and the other does not


def _standardized_separation(a: List[float], b: List[float]) -> float:
    """|mean(a) - mean(b)| / pooled_sd. 0 when indistinguishable."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa = statistics.pstdev(a)
    sb = statistics.pstdev(b)
    pooled = math.sqrt((sa * sa + sb * sb) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return abs(statistics.fmean(a) - statistics.fmean(b)) / pooled


def _mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute preconditions, criteria, interpretation from the cell grid."""
    def sel(policy: str, ttype: str, perturbed: bool = True) -> List[Dict[str, Any]]:
        return [
            r for r in rows
            if r["policy"] == policy and r["trial_type"] == ttype
            and r["perturbed"] == perturbed
        ]

    # per (policy, type) mean performance / cost across seeds (perturbed cells only)
    perf: Dict[str, Dict[str, float]] = {}
    cost: Dict[str, Dict[str, float]] = {}
    for policy in POLICIES:
        perf[policy] = {}
        cost[policy] = {}
        for ttype in TRIAL_TYPES:
            cells = sel(policy, ttype, True)
            perf[policy][ttype] = _mean([c["performance_mean"] for c in cells])
            cost[policy][ttype] = _mean([c["cost_mean"] for c in cells])

    # clean reference (type-independent env; ARM1_NONE, perturbed=False)
    clean_cells = [r for r in rows if r["policy"] == "ARM1_NONE" and r["perturbed"] is False]
    clean_perf = _mean([c["performance_mean"] for c in clean_cells])

    baseline = perf["ARM1_NONE"]  # perturbed no-coalition per type
    def improvement(policy: str) -> float:
        return _mean([perf[policy][t] - baseline[t] for t in TRIAL_TYPES])

    matched_improvement = improvement("ARM4_TYPED")
    mismatched_improvement = improvement("CTRL_MISMATCH")
    untyped_improvement = improvement("ARM3_UNTYPED")
    param_improvement = improvement("ARM2_PARAM")

    # ---- P1 templates_distinct ----
    distinct_sites = _templates_distinct_count()
    p1 = {
        "name": "templates_distinct",
        "description": "SENSORY vs PROVENANCE templates recruit/suppress disjoint sites (write_gate < 1.0)",
        "control": "COALITION_TEMPLATES symmetric-difference of named targets",
        "measured": float(distinct_sites),
        "threshold": float(DISTINCT_SITES_FLOOR),
        "direction": "lower",
        "kind": "readiness",
        "met": distinct_sites >= DISTINCT_SITES_FLOOR,
    }

    # ---- P2 non-degeneracy: ARM1 baseline signals dissociate the two doubt types ----
    a1_perc = sel("ARM1_NONE", "perceptual", True)
    a1_prov = sel("ARM1_NONE", "provenance", True)
    sep_pvol = _standardized_separation(
        [c["percept_volatility_mean"] for c in a1_perc],
        [c["percept_volatility_mean"] for c in a1_prov],
    )
    sep_mpe = _standardized_separation(
        [c["model_pe_mean"] for c in a1_perc],
        [c["model_pe_mean"] for c in a1_prov],
    )
    nondegen_sep = max(sep_pvol, sep_mpe)
    p2 = {
        "name": "nondegeneracy_signals_dissociate",
        "description": "ARM1 baseline discrepancy signals separate perceptual vs provenance trials (max standardized separation over percept_volatility, model_pe)",
        "control": "ARM1_NONE perturbed cells, perceptual vs provenance clusters",
        "measured": float(nondegen_sep),
        "threshold": float(SEP_FLOOR),
        "direction": "lower",
        "kind": "readiness",
        "load_bearing": True,
        "met": nondegen_sep > SEP_FLOOR,
        "sep_percept_volatility": float(sep_pvol),
        "sep_model_pe": float(sep_mpe),
    }

    # ---- P3 perturbation degrades baseline (per type) ----
    degrade = {t: clean_perf - baseline[t] for t in TRIAL_TYPES}
    min_degrade = min(degrade.values()) if degrade else 0.0
    p3 = {
        "name": "perturbation_degrades_baseline",
        "description": "clean (no-doubt) ARM1 performance exceeds perturbed ARM1 for BOTH doubt types",
        "control": "ARM1_CLEAN vs ARM1_NONE perturbed, worst type",
        "measured": float(min_degrade),
        "threshold": float(DEGRADE_FLOOR),
        "direction": "lower",
        "kind": "readiness",
        "met": min_degrade > DEGRADE_FLOOR,
        "degrade_per_type": {t: float(degrade[t]) for t in TRIAL_TYPES},
        "clean_performance": float(clean_perf),
    }

    # ---- P4 coalition engages in ARM4 ----
    a4_cells = [r for r in rows if r["policy"] == "ARM4_TYPED" and r["perturbed"] is True]
    a4_e3 = sum(c["n_post_e3"] for c in a4_cells)
    a4_active = sum(c["n_post_e3_coalition_active"] for c in a4_cells)
    engage_frac = (a4_active / a4_e3) if a4_e3 else 0.0
    p4 = {
        "name": "coalition_engages",
        "description": "fraction of ARM4 post-onset E3 ticks with an active coalition",
        "control": "ARM4_TYPED perturbed cells",
        "measured": float(engage_frac),
        "threshold": float(ENGAGE_FLOOR),
        "direction": "lower",
        "kind": "readiness",
        "met": engage_frac > ENGAGE_FLOOR,
    }

    preconditions = [p1, p2, p3, p4]

    # matched-difficulty diagnostic (non-gating): |degrade_perc - degrade_prov|
    matched_difficulty_gap = abs(degrade.get("perceptual", 0.0) - degrade.get("provenance", 0.0))

    # ---- Criteria (only meaningful if the readiness gates pass) ----
    c1 = (matched_improvement > mismatched_improvement + SEL_MARGIN) and (
        matched_improvement > untyped_improvement + SEL_MARGIN
    )
    c2 = matched_improvement > param_improvement + PARAM_MARGIN
    c3 = cost["ARM4_TYPED"] and cost["ARM3_UNTYPED"] and (
        _mean(list(cost["ARM4_TYPED"].values())) < _mean(list(cost["ARM3_UNTYPED"].values()))
    )
    c3 = bool(c3)

    criteria = [
        {"name": "C1_typing_discriminative", "load_bearing": True, "passed": bool(c1)},
        {"name": "C2_recruitment_beyond_parametric", "load_bearing": True, "passed": bool(c2)},
        {"name": "C3_typed_cheaper_than_untyped", "load_bearing": False, "passed": bool(c3)},
    ]
    combination_rule = (
        "PASS iff (C1 AND C2). C1: matched_improvement beats BOTH mismatched and "
        "untyped by SEL_MARGIN (typing does discriminative work). C2: matched_improvement "
        "beats parametric by PARAM_MARGIN (recruitment adds beyond parametric). C3 "
        "(cost) corroborates but does not gate. FAIL if C1 fails (Arm4~Arm3) or C2 "
        "fails (Arm2~Arm4)."
    )

    # ---- self-route ----
    readiness_ok = p2["met"] and p3["met"] and p4["met"]
    if not p1["met"]:
        label = "primitive_miswired_requeue"
        outcome = "FAIL"
    elif not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1 and c2:
        label = "typed_coalition_selective_supports"
        outcome = "PASS"
    elif not c1:
        label = "typing_does_no_discriminative_work"
        outcome = "FAIL"
    else:  # c1 true, c2 false
        label = "parametric_suffices_g_t_superfluous"
        outcome = "FAIL"

    # ---- criteria_non_degenerate: did each criterion discriminate, or pass/fail trivially? ----
    # A criterion is non-degenerate iff the readiness gates held AND its inputs varied.
    improvements_vary = len({round(matched_improvement, 9), round(mismatched_improvement, 9),
                             round(untyped_improvement, 9), round(param_improvement, 9)}) > 1
    criteria_non_degenerate = {
        "C1_typing_discriminative": bool(readiness_ok and improvements_vary),
        "C2_recruitment_beyond_parametric": bool(readiness_ok and improvements_vary),
        "C3_typed_cheaper_than_untyped": bool(
            readiness_ok
            and _mean(list(cost["ARM3_UNTYPED"].values())) != _mean(list(cost["ARM4_TYPED"].values()))
        ),
    }

    # ---- per-claim direction ----
    if not (p1["met"] and readiness_ok):
        dir_mech481 = "unknown"
        dir_sd091 = "unknown"
    else:
        dir_mech481 = "supports" if c1 else "weakens"
        dir_sd091 = "supports" if c2 else "weakens"
    if outcome == "PASS":
        overall_dir = "supports"
    elif label in ("substrate_not_ready_requeue", "primitive_miswired_requeue"):
        overall_dir = "unknown"
    else:
        overall_dir = "weakens"

    interpretation = {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "matched_difficulty_gap": float(matched_difficulty_gap),
        "matched_difficulty_note": (
            "non-gating diagnostic: |degrade_perceptual - degrade_provenance|; a large "
            "gap means the two doubt types are not at matched base difficulty and the "
            "cross-type improvement comparison should be read with that caveat."
        ),
    }

    summary = {
        "clean_performance": float(clean_perf),
        "performance_by_policy_type": {p: {t: float(perf[p][t]) for t in TRIAL_TYPES} for p in POLICIES},
        "cost_by_policy_type": {p: {t: float(cost[p][t]) for t in TRIAL_TYPES} for p in POLICIES},
        "baseline_degrade_per_type": {t: float(degrade[t]) for t in TRIAL_TYPES},
        "matched_improvement": float(matched_improvement),
        "mismatched_improvement": float(mismatched_improvement),
        "untyped_improvement": float(untyped_improvement),
        "param_improvement": float(param_improvement),
        "cost_arm4_typed": float(_mean(list(cost["ARM4_TYPED"].values()))),
        "cost_arm3_untyped": float(_mean(list(cost["ARM3_UNTYPED"].values()))),
        "nondegeneracy_separation": float(nondegen_sep),
        "coalition_engage_frac": float(engage_frac),
        "templates_distinct_sites": int(distinct_sites),
    }

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "summary": summary,
        "evidence_direction": overall_dir,
        "evidence_direction_per_claim": {"MECH-481": dir_mech481, "SD-091": dir_sd091},
    }


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = [0] if dry_run else SEEDS
    n_trials = 1 if dry_run else N_TRIALS
    settle = 10 if dry_run else SETTLE_STEPS
    post = 10 if dry_run else POST_STEPS
    script_path = _THIS

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        # scored + control policies, both doubt types, perturbed
        for policy in POLICIES:
            for ttype in TRIAL_TYPES:
                rows.append(
                    _run_cell(policy, ttype, seed, True, n_trials, settle, post, script_path)
                )
        # clean reference (ARM1_NONE, no perturbation; type label 'perceptual' as a
        # nominal tag -- env is identical, perturbed=False disables both perturbations)
        rows.append(
            _run_cell("ARM1_NONE", "perceptual", seed, False, n_trials, settle, post, script_path)
        )

    analysis = analyze(rows)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "env_kw": ENV_KW,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "seeds": seeds,
        "n_trials": n_trials,
        "settle_steps": settle,
        "post_steps": post,
        "episode_steps": settle + post,
        "perceptual_noise_sigma": PERCEPTUAL_NOISE_SIGMA,
        "provenance_swap_pairs": PROVENANCE_SWAP_PAIRS,
        "policies": POLICIES,
        "trial_types": TRIAL_TYPES,
        "thresholds": {
            "SEP_FLOOR": SEP_FLOOR,
            "DEGRADE_FLOOR": DEGRADE_FLOOR,
            "ENGAGE_FLOOR": ENGAGE_FLOOR,
            "DISTINCT_SITES_FLOOR": DISTINCT_SITES_FLOOR,
            "SEL_MARGIN": SEL_MARGIN,
            "PARAM_MARGIN": PARAM_MARGIN,
        },
        "use_coalition_controller": True,
        "dry_run": dry_run,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": analysis["outcome"],
        "timestamp_utc": ts,
        "evidence_direction": analysis["evidence_direction"],
        "evidence_direction_per_claim": analysis["evidence_direction_per_claim"],
        "interpretation": analysis["interpretation"],
        "summary": analysis["summary"],
        "arm_results": rows,
        "dry_run": dry_run,
    }

    return {
        "manifest": manifest,
        "config": full_config,
        "seeds": seeds,
        "elapsed": time.perf_counter() - t0,
        "started_at": t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="V3-EXQ-886 MECH-481/SD-091 4-arm coalition falsifier"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="tiny smoke config, manifest relocated out of evidence/",
    )
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    manifest = result["manifest"]

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=result["config"],
        seeds=result["seeds"],
        script_path=_THIS,
        started_at=result["started_at"],
    )

    print(
        f"[{'dry' if args.dry_run else 'run'}] outcome={manifest['outcome']} "
        f"label={manifest['interpretation']['label']} -> {out_path}",
        flush=True,
    )

    _out = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_out if _out in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        dry_run=args.dry_run,
    )
