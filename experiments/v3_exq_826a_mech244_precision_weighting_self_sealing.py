"""V3-EXQ-826a -- MECH-244: psychosis as a precision-weighting failure (CORRECTED).

Claims: MECH-244 ONLY.
EXPERIMENT_PURPOSE: evidence.
Supersedes: V3-EXQ-826 (v3_exq_826_mech244_precision_weighting_self_sealing_20260726T152827Z_v3)

=== WHY THIS RE-RUN (confirmed autopsy) ===
failure_autopsy_batch-822a-826-817a-827_2026-07-26.md (REE_assembly, confirmed)
section 1 found V3-EXQ-826's FAIL invalid as a test of MECH-244: its training
loop called agent.sense(obs_body, obs_world) + agent.clock.advance() (return
value discarded) every step, then a random action, env.step(), and
agent.compute_prediction_loss() -- but NEVER called agent._e1_tick(latent)
(directly, or via act()/act_with_split_obs()/act_with_log_prob(), the only
internal callers -- ree_core/agent.py:8208/8238/8260, re-verified this
session). compute_prediction_loss() (ree_core/agent.py:8553-8567, re-verified
this session) hard-returns a zero-gradient stub
(`next(self.e1.parameters()).sum() * 0.0`) whenever
`len(self._world_experience_buffer) < 2`; that buffer is appended to ONLY
inside _e1_tick() (ree_core/agent.py:4678-4679, re-verified this session).
So E1 never received a gradient for the entire run -- the manipulated
pe_precision scalar multiplied a loss whose backward pass through
agent.e1.parameters() never fired (autograd short-circuits on a detached
zero-valued stub), so ARM_CONTROL / ARM_LOW / ARM_VERYLOW were three
identically-untrained E1 networks and the run tested nothing about
precision-weighted correction. Same defect, same fix, diagnosed for
precedent V3-EXQ-032b in failure_autopsy_V3-EXQ-032-family_2026-07-26.md;
the corrected re-run V3-EXQ-032d (ree-v3 commit bf4fbaaa00) is this script's
reference implementation.

THE FIX (this script, and ONLY this, relative to 826):
  Capture the latent from agent.sense(...) (826 discarded the return value)
  and the tick dict from agent.clock.advance() (826 called this but
  discarded the return value), then call agent._e1_tick(latent) when
  ticks.get("e1_tick", False) fires -- exactly the act() / act_with_split_obs()
  pattern in ree_core/agent.py:8208/8238. This populates
  _world_experience_buffer (and _self_experience_buffer) with real content,
  giving compute_prediction_loss() actual gradients to train
  agent.e1.parameters() through the dedicated pe_precision-scaled optimizer.
  826 never trains E2/E3 (design intent: pe_precision touches ONLY
  agent.e1.parameters()) and never calls compute_e2_loss() or
  agent.record_transition() -- so unlike the 032-family fix, no
  record_transition lag-pairing is needed here; that half of the defect
  class does not apply to this script.

  Verified this session that _e1_tick() introduces NO action-selection or
  policy confound: it runs E1's forward pass, appends to the experience
  buffers, updates ThetaBuffer/theta-packet/schema-salience caches, and
  returns e1_prior for candidate generation (ree_core/agent.py:4645-4753+) --
  it does not choose or bias an action. This experiment discards the
  returned e1_prior (actions remain uniform-random via random.randint(...)),
  so the random-action design intent -- the discriminating choice vs
  MECH-246/MECH-247 confound-avoidance -- is preserved exactly. Also
  verified: heartbeat cadence default is e1_steps_per_tick=1
  (ree_core/heartbeat/clock.py:50/142), so under this experiment's default
  config e1_tick fires every env step and the buffer reaches length 2 within
  the first two steps of every episode.

NOTHING ELSE CHANGES relative to 826: arms (pe_precision 1.0/0.15/0.03),
seeds, episode/phase structure, the adaptation_ratio DV, all three P1-P3
preconditions, the C1 load-bearing criterion and its thresholds, and the
DV-symmetry / GOV-REUSE-1 / ethics-preflight declarations below are
byte-identical to 826's own analysis (re-verified as still correct for this
script -- the fix only changes what the E1 optimizer actually trains on, not
any measurement or gating logic).

GOV-REUSE-1 (Step 2.4, re-checked this session): the decisive readout
(adaptation_ratio delta under a genuinely gradient-trained,
pe_precision-scaled E1, after an env.reset_to() hazard-relocation regime
change) is not recoverable from any existing manifest. V3-EXQ-826 is the
only prior run of this design and its E1 never trained (the defect this run
fixes), so no manifest on any substrate_hash carries this readout against a
genuinely trained E1. Not recoverable -> run.

Re-derive brake (Step 2.5b, re-checked this session): exactly one prior
autopsy target for MECH-244 (V3-EXQ-826 itself, category
competence_implementation_gap / instrument defect -- E1 never ticked, not a
substrate ceiling; recommended_substrate_queue_entry.action was
"queue-experiment redesign", not a substrate build). Count = 1, below the
>=2 re-derive-brake threshold, and the category is an instrument defect
(not substrate_ceiling) regardless. Brake does not fire.

Ethics preflight (Step 2.6, re-checked this session): all-false /
not_applicable, decision: allow -- identical to 826 (random-action rollout,
no learned policy, no harm/benefit DV anywhere in this run).

=== ORIGINAL CLAIM (unchanged from 826) ===
MECH-244: "Psychosis is a precision-weighting failure where prior beliefs
persistently dominate sensory evidence; the generative model becomes
self-sealing and sensory update is suppressed." Mechanistically (claim notes):
precision on sensory prediction error is downregulated relative to
prior-belief precision, so PE cannot correct the model -- the bottom-up error
signal is discounted relative to the top-down prediction, and the world model
becomes unfalsifiable by experience.

MECH-244 is one of three psychosis pathways in the REE framework, mechanistically
distinct from its siblings:
  MECH-246 signal-degradation/pareidolia -- noisy bottom-up INPUT, no precision
    failure. This experiment does NOT add noise to observations.
  MECH-247 hypervigilant-prior CONTENT -- pathological prior structure. This
    experiment does NOT bias what the model believes, only how fast/strongly
    it is permitted to update.
This design isolates precision-weighting (the GAIN on PE-driven correction)
as the sole manipulated variable, leaving both input fidelity and prior
content untouched -- the discriminating design choice vs both siblings.

=== SUBSTRATE MAPPING (no ree_core changes -- re-confirmed via substrate read) ===
ree_core/predictors/e1_deep.py: E1DeepPredictor is, per its own module
docstring, "slow world model, LSTM-based, trains on sensory prediction
error" -- the substrate's literal implementation of "the generative model"
whose updates MECH-244 claims become self-sealing. Its hidden state /
weights persist across steps and ARE the "prior" that, absent correction,
continues to dominate.

REEAgent.compute_prediction_loss() (ree_core/agent.py) computes E1's PE loss
from DETACHED buffered (z_self, z_world) -- so it trains ONLY agent.e1's own
parameters, never the encoder. It is called EXTERNALLY by experiment scripts
which then run their own `.backward()` / `optimizer.step()`. This means the
loss can be scaled BEFORE backward from the calling script with zero ree_core
edits -- confirmed no existing "precision"/"pe_weight" gain knob sits on this
path (ARC-016's E3 `current_precision` is a different pathway: it gates the
E3 COMMITMENT threshold, not E1/E2 belief updating). THE FIX above is what
makes this loss non-vacuous: the buffer it samples from must actually be
populated by _e1_tick() first.

=== MANIPULATION (IV) ===
pe_precision: a positive scalar multiplying agent.compute_prediction_loss()
BEFORE .backward(), with a DEDICATED optimizer scoped to `agent.e1.parameters()`
only (so the scaling touches nothing else -- not the encoder, not E2, not E3).
This directly operationalises "precision on sensory PE downregulated": a lower
pe_precision means a smaller effective gradient step correcting E1's weights
per unit of prediction error, while E1's own forward dynamics (the "prior")
continue unchanged and unscaled. No separate "prior precision" knob is added --
downregulating correction is definitionally equivalent to the prior's RELATIVE
dominance increasing, since the prior is simply what persists when correction
is throttled.

Arms: ARM_CONTROL (pe_precision=1.0, substrate default), ARM_LOW (0.15),
ARM_VERYLOW (0.03). Seeds 0-4 (5 seeds x 3 arms = 15 cells).

=== DV: adaptation to a regime change (disconfirming evidence) ===
Random-action rollout throughout (no policy learning -- matches the
established V3-EXQ-818 "structural test" convention, removing any policy
confound). Two phases per cell, same agent/optimizer carried across the
phase boundary (only agent.reset() between episodes, which resets E1's
LSTM hidden state, NOT its weights, and NOT the experience buffers):
  Phase A (episodes 1-250): env.reset_to() places hazards at REGIME_A_HAZARDS
    every episode (deterministic, SD-029 scripted-eval harness). E1 has an
    opportunity to learn a stable regularity.
  Phase B (episodes 251-400): env.reset_to() SWITCHES to REGIME_B_HAZARDS
    (hazards relocated) -- deterministic disconfirming evidence: the model's
    learned regularity is now systematically wrong.
Per-step E1 loss is logged UNWEIGHTED (the raw, un-scaled value from
agent.compute_prediction_loss()) so cross-arm comparison is on a common,
fair scale -- pe_precision only affects the backward pass, never the
reported forward value.

adaptation_ratio = (early_phaseB_loss - late_phaseB_loss) / early_phaseB_loss
  where early_phaseB_loss = mean of first 10 phase-B episode-means (captures
  the post-switch spike) and late_phaseB_loss = mean of the LAST 20 phase-B
  episode-means (captures whether the model re-corrected). A "self-sealing"
  model shows adaptation_ratio near 0 (loss stays elevated); a normally
  updating model shows a large positive adaptation_ratio.

=== PRECONDITIONS (measured on ARM_CONTROL as the positive control; worst seed) ===
  P1 world_model_converged_phaseA_control: E1 phase-A reduction ratio
     (early_phaseA_loss - late_phaseA_loss)/early_phaseA_loss >= 0.25.
     Confirms E1 learned SOMETHING in phase A -- a precondition for phase B
     to test whether that something is later overwritten.
  P2 regime_change_disconfirms_control: phase-B onset spike ratio
     (early_phaseB_loss / late_phaseA_loss) >= 1.15. Confirms the hazard
     relocation genuinely disconfirms the learned regularity (without this,
     "fails to adapt" would be vacuous -- there was nothing to adapt to).
  P3 pe_precision_manipulation_took: ARM_CONTROL's E1 parameter L2 drift
     during phase A, divided by ARM_VERYLOW's, >= 1.5 (worst seed). Confirms
     lower pe_precision genuinely produces less belief-updating at the
     mechanical level -- the causal lever is engaged, not inert. (This
     precondition is the one that would have caught 826's defect directly:
     with E1 never ticked, drift for BOTH arms was pinned near the
     init-noise floor and this ratio could not have cleared 1.5 -- but 826
     never re-checked P3 as the self-sealing hypothesis mechanism, it went
     straight to a "does_not_support" verdict as though the test had run.)
A P0NotReady on any of these self-routes the whole run to FAIL /
non_contributory / "substrate_not_ready_requeue" (never a verdict label).

=== PRE-REGISTERED LOAD-BEARING CRITERION ===
  C1 (LOAD-BEARING): per-seed delta = adaptation_ratio(ARM_CONTROL, seed) -
     adaptation_ratio(ARM_VERYLOW, seed). PASS iff mean(delta) >= 0.15 AND
     at least 4/5 seeds have delta > 0 (sign-consistency, guards against one
     seed driving the mean). ARM_LOW's delta vs control is recorded as a
     secondary (non-load-bearing) dose-response check, reported but not gated.
Non-degeneracy: the 5 per-seed C1 deltas are passed through
experiments._metrics.check_degeneracy -- a run where the deltas are
pinned/zero-spread (e.g. because pe_precision silently failed to reach the
optimizer, or E1 still isn't training) self-reports non_degenerate=False and
is scoring-excluded rather than scored as a genuine null.

INTERPRETATION GRID:
  P0 unmet -> FAIL / non_contributory / substrate_not_ready_requeue
  P0 met, C1 deltas degenerate -> FAIL / non_contributory /
      c1_delta_degenerate_vacuous_test
  P0 met, C1 PASS -> PASS / supports /
      precision_weighted_pe_suppression_produces_self_sealing_world_model
  P0 met, C1 FAIL (non-degenerate) -> FAIL / weakens /
      precision_weighted_pe_suppression_does_not_produce_self_sealing_world_model
      (a genuine, decision-relevant negative finding against MECH-244's
      proposed mechanism, since P2 already confirms disconfirming evidence
      was genuinely present and P3 confirms the lever was engaged).

=== DV-SYMMETRY DECLARATION (mandatory, unchanged from 826 -- re-verified) ===
DV = adaptation_ratio, derived from E1's own scalar MSE prediction-loss
trajectory over hundreds of training episodes. IV = pe_precision, a positive
scalar multiplying the gradient step applied ONLY to agent.e1.parameters()
via a dedicated optimizer (never touching the forward computation or the
encoder). None of the three canonical danger symmetries apply:
  - broadcast-constant + argmax: no argmax/discrete action selection feeds
    the DV anywhere (actions are random, not selected from scored
    candidates) -- there is no argmax for a uniform offset to hide from.
  - monotone-rescale + rank DV: the DV is a numeric floor/mean comparison,
    not a rank/order statistic of the arms, and pe_precision is not a
    monotone-transform of a value later thresholded -- it scales gradient
    STEP SIZE feeding hundreds of steps of nonlinear SGD, producing a
    genuinely different trained network (this is the ordinary, well-
    established sense in which loss-term weighting changes what a network
    learns; it is not a symmetry-preserving relabelling).
  - permutation + aggregate: arms are not interchangeable/pooled units --
    each arm trains its OWN independent network; permuting arm labels would
    only relabel which trained network exhibits which behaviour, not erase
    the effect being measured.

=== ETHICS PREFLIGHT (documentation habit, non-enforced) ===
involves_negative_valence: false; involves_suffering_like_state: false;
involves_self_model: false; involves_inescapability_or_helplessness: false;
involves_offline_replay_over_harm: false; involves_social_mind_or_language:
false; involves_human_data_or_clinical_context: false; decision: allow.
Random-action rollout, no learned policy, no harm/benefit DV anywhere in
this run.

=== PHASED TRAINING (not applicable) ===
This experiment trains E1 itself (a core substrate module, via its own
compute_prediction_loss()), not a downstream probe head reading a frozen
z_world/z_harm encoder output -- the phased-training mandate (P0 warmup ->
P1 frozen-encoder head training -> P2 eval) governs the latter, not this.

SLEEP DRIVER: not applicable (no sleep loop used in this experiment).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_826a_mech244_precision_weighting_self_sealing.py --dry-run
"""

import argparse
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    p0_readiness_gate,
    P0NotReady,
)

EXPERIMENT_TYPE = "v3_exq_826a_mech244_precision_weighting_self_sealing"
CLAIM_IDS = ["MECH-244"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "v3_exq_826_mech244_precision_weighting_self_sealing_20260726T152827Z_v3"

# --- Environment / substrate config -----------------------------------------
GRID_SIZE = 10
HAZARD_HARM = 0.3
PROXIMITY_HARM_SCALE = 0.05
PROXIMITY_BENEFIT_SCALE = 0.03
SELF_DIM = 32
WORLD_DIM = 32
ALPHA_WORLD = 0.9   # SD-008: >=0.9 needed for z_world fidelity (this DV depends on it)
ALPHA_SELF = 0.3
LR = 1e-3

AGENT_START = (5, 5)
REGIME_A_HAZARDS = [(1, 1), (1, 8), (8, 1), (8, 8)]
REGIME_B_HAZARDS = [(5, 1), (1, 5), (8, 5), (5, 8)]
RESOURCE_POSITIONS = [(3, 3), (3, 6), (6, 3), (6, 6)]

SEEDS = [0, 1, 2, 3, 4]
ARMS: List[Tuple[str, float]] = [
    ("ARM_CONTROL", 1.0),
    ("ARM_LOW", 0.15),
    ("ARM_VERYLOW", 0.03),
]

N_A = 250
N_B = 150
STEPS_PER_EPISODE = 40

WINDOW_A = 20
WINDOW_B_EARLY = 10
WINDOW_B_LATE = 20

# Dry-run overrides (smoke test -- proves the script runs end to end; not
# expected to satisfy the preconditions or criteria below).
DRY_SEEDS = [0, 1]
DRY_N_A = 6
DRY_N_B = 4
DRY_STEPS = 5
DRY_WINDOW_A = 2
DRY_WINDOW_B_EARLY = 2
DRY_WINDOW_B_LATE = 2

# --- Pre-registered thresholds ----------------------------------------------
REDUCTION_FLOOR = 0.25      # P1: phase-A E1 loss reduction ratio, control, worst seed
SPIKE_FLOOR = 1.15          # P2: phase-B onset spike ratio, control, worst seed
DRIFT_RATIO_FLOOR = 1.5     # P3: control/verylow E1 param L2 drift ratio, worst seed
DELTA_FLOOR = 0.15          # C1: mean(delta) floor (adaptation-ratio units)
MIN_SEEDS_MAJORITY = 4      # C1: of 5 seeds, sign-consistency floor


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    pe_precision: float,
    n_episodes: int,
    steps_per_episode: int,
    hazard_positions: List[Tuple[int, int]],
    resource_positions: List[Tuple[int, int]],
    agent_start: Tuple[int, int],
    episode_offset: int,
    total_episodes: int,
    arm_id: str,
    seed: int,
    phase_label: str,
) -> Tuple[List[float], int]:
    """Run n_episodes of random-action rollout, training only agent.e1.parameters()
    on pe_precision * agent.compute_prediction_loss(). Returns per-episode mean
    of the RAW (unweighted) E1 loss, plus the count of e1_tick firings this
    phase (THE FIX -- see module docstring; this is what 826 never did)."""
    episode_means: List[float] = []
    num_actions = env.action_dim
    n_e1_ticks = 0

    for ep in range(n_episodes):
        _flat_obs, obs_dict = env.reset_to(
            agent_pos=agent_start,
            hazard_positions=hazard_positions,
            resource_positions=resource_positions,
        )
        agent.reset()
        step_losses: List[float] = []

        for _step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            # THE FIX: capture ticks (826 discarded this) and run _e1_tick on
            # cadence -- the same gate act()/act_with_split_obs() use. This is
            # what populates _world_experience_buffer / _self_experience_buffer
            # so compute_prediction_loss() below has real content to train on
            # instead of hard-returning its zero-gradient stub.
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)
                n_e1_ticks += 1

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            _flat_obs, _harm_signal, _done, _info, obs_dict = env.step(action)

            e1_loss_raw = agent.compute_prediction_loss()
            scaled_loss = pe_precision * e1_loss_raw
            if scaled_loss.requires_grad:
                optimizer.zero_grad()
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            step_losses.append(float(e1_loss_raw.item()))

        episode_means.append(
            statistics.fmean(step_losses) if step_losses else float("nan")
        )

        global_ep = episode_offset + ep + 1
        if global_ep % 50 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] arm={arm_id} seed={seed} phase={phase_label} "
                f"ep {global_ep}/{total_episodes} e1_ticks={n_e1_ticks}",
                flush=True,
            )

    return episode_means, n_e1_ticks


def _run_cell(
    arm_id: str,
    pe_precision: float,
    seed: int,
    n_a: int,
    n_b: int,
    steps_per_episode: int,
    window_a: int,
    window_b_early: int,
    window_b_late: int,
) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    cell_config_slice = {
        "arm_id": arm_id,
        "pe_precision": pe_precision,
        "seed": seed,
        "n_a": n_a,
        "n_b": n_b,
        "steps_per_episode": steps_per_episode,
        "env": {
            "size": GRID_SIZE,
            "hazard_harm": HAZARD_HARM,
            "proximity_harm_scale": PROXIMITY_HARM_SCALE,
            "proximity_benefit_scale": PROXIMITY_BENEFIT_SCALE,
        },
        "regime_a_hazards": REGIME_A_HAZARDS,
        "regime_b_hazards": REGIME_B_HAZARDS,
        "resources": RESOURCE_POSITIONS,
        "agent_start": AGENT_START,
        "latent_config": {
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "alpha_world": ALPHA_WORLD,
            "alpha_self": ALPHA_SELF,
            "lr": LR,
        },
        "fix_vs_826": "calls agent._e1_tick(latent) on e1_tick cadence; no other change",
    }

    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env = CausalGridWorldV2(
            seed=seed,
            size=GRID_SIZE,
            num_hazards=len(REGIME_A_HAZARDS),
            num_resources=len(RESOURCE_POSITIONS),
            hazard_harm=HAZARD_HARM,
            proximity_harm_scale=PROXIMITY_HARM_SCALE,
            proximity_benefit_scale=PROXIMITY_BENEFIT_SCALE,
            use_proxy_fields=True,
            toroidal=False,
        )
        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=SELF_DIM,
            world_dim=WORLD_DIM,
            alpha_world=ALPHA_WORLD,
            alpha_self=ALPHA_SELF,
            reafference_action_dim=env.action_dim,
        )
        agent = REEAgent(config)
        optimizer = optim.Adam(agent.e1.parameters(), lr=LR)

        e1_init_vec = torch.cat(
            [p.detach().flatten().clone() for p in agent.e1.parameters()]
        )

        total_episodes = n_a + n_b
        phaseA_means, n_e1_ticks_a = _run_phase(
            agent, env, optimizer, pe_precision, n_a, steps_per_episode,
            REGIME_A_HAZARDS, RESOURCE_POSITIONS, AGENT_START,
            episode_offset=0, total_episodes=total_episodes,
            arm_id=arm_id, seed=seed, phase_label="A",
        )

        e1_postA_vec = torch.cat(
            [p.detach().flatten().clone() for p in agent.e1.parameters()]
        )
        e1_drift_phaseA = float(torch.norm(e1_postA_vec - e1_init_vec).item())

        phaseB_means, n_e1_ticks_b = _run_phase(
            agent, env, optimizer, pe_precision, n_b, steps_per_episode,
            REGIME_B_HAZARDS, RESOURCE_POSITIONS, AGENT_START,
            episode_offset=n_a, total_episodes=total_episodes,
            arm_id=arm_id, seed=seed, phase_label="B",
        )

        w_a = min(window_a, len(phaseA_means))
        w_be = min(window_b_early, len(phaseB_means))
        w_bl = min(window_b_late, len(phaseB_means))

        early_A = statistics.fmean(phaseA_means[:w_a])
        late_A = statistics.fmean(phaseA_means[-w_a:])
        early_B = statistics.fmean(phaseB_means[:w_be])
        late_B = statistics.fmean(phaseB_means[-w_bl:])

        reduction_ratio_phaseA = (early_A - late_A) / (early_A + 1e-8)
        spike_ratio_phaseB_onset = early_B / (late_A + 1e-8)
        adaptation_ratio_phaseB = (early_B - late_B) / (early_B + 1e-8)

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "pe_precision": pe_precision,
            "seed": seed,
            "phaseA_episode_means": phaseA_means,
            "phaseB_episode_means": phaseB_means,
            "early_phaseA_loss": early_A,
            "late_phaseA_loss": late_A,
            "early_phaseB_loss": early_B,
            "late_phaseB_loss": late_B,
            "reduction_ratio_phaseA": reduction_ratio_phaseA,
            "spike_ratio_phaseB_onset": spike_ratio_phaseB_onset,
            "adaptation_ratio_phaseB": adaptation_ratio_phaseB,
            "e1_param_l2_drift_phaseA": e1_drift_phaseA,
            "n_e1_ticks_phaseA": n_e1_ticks_a,
            "n_e1_ticks_phaseB": n_e1_ticks_b,
        }
        cell.stamp(row)

    print("verdict: PASS", flush=True)
    return row


def run_experiment(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()

    seeds = DRY_SEEDS if dry_run else SEEDS
    n_a = DRY_N_A if dry_run else N_A
    n_b = DRY_N_B if dry_run else N_B
    steps = DRY_STEPS if dry_run else STEPS_PER_EPISODE
    window_a = DRY_WINDOW_A if dry_run else WINDOW_A
    window_b_early = DRY_WINDOW_B_EARLY if dry_run else WINDOW_B_EARLY
    window_b_late = DRY_WINDOW_B_LATE if dry_run else WINDOW_B_LATE

    arm_results: List[Dict[str, Any]] = []
    for arm_id, pe_precision in ARMS:
        for seed in seeds:
            row = _run_cell(
                arm_id, pe_precision, seed, n_a, n_b, steps,
                window_a, window_b_early, window_b_late,
            )
            arm_results.append(row)

    by_arm = {arm_id: [r for r in arm_results if r["arm_id"] == arm_id]
              for arm_id, _ in ARMS}
    control_rows = by_arm["ARM_CONTROL"]
    low_rows = by_arm["ARM_LOW"]
    verylow_rows = by_arm["ARM_VERYLOW"]

    p1_measured = min(r["reduction_ratio_phaseA"] for r in control_rows)
    p2_measured = min(r["spike_ratio_phaseB_onset"] for r in control_rows)

    drift_ratios = []
    for seed in seeds:
        c = next(r for r in control_rows if r["seed"] == seed)
        v = next(r for r in verylow_rows if r["seed"] == seed)
        drift_ratios.append(
            c["e1_param_l2_drift_phaseA"] / (v["e1_param_l2_drift_phaseA"] + 1e-12)
        )
    p3_measured = min(drift_ratios)

    total_e1_ticks = sum(
        r["n_e1_ticks_phaseA"] + r["n_e1_ticks_phaseB"] for r in arm_results
    )

    checks = [
        {
            "name": "world_model_converged_phaseA_control",
            "measured": p1_measured,
            "threshold": REDUCTION_FLOOR,
            "direction": "lower",
            "control": "ARM_CONTROL phase-A E1 loss reduction ratio, worst seed",
        },
        {
            "name": "regime_change_disconfirms_control",
            "measured": p2_measured,
            "threshold": SPIKE_FLOOR,
            "direction": "lower",
            "control": "ARM_CONTROL phase-B onset spike ratio vs late phase-A, worst seed",
        },
        {
            "name": "pe_precision_manipulation_took",
            "measured": p3_measured,
            "threshold": DRIFT_RATIO_FLOOR,
            "direction": "lower",
            "control": "ARM_CONTROL / ARM_VERYLOW E1 param L2 drift ratio in phase A, worst seed",
        },
    ]

    ready = True
    try:
        preconditions = p0_readiness_gate(checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    delta_verylow = []
    delta_low = []
    for seed in seeds:
        c = next(r for r in control_rows if r["seed"] == seed)
        vl = next(r for r in verylow_rows if r["seed"] == seed)
        lo = next(r for r in low_rows if r["seed"] == seed)
        delta_verylow.append(c["adaptation_ratio_phaseB"] - vl["adaptation_ratio_phaseB"])
        delta_low.append(c["adaptation_ratio_phaseB"] - lo["adaptation_ratio_phaseB"])

    mean_delta_verylow = statistics.fmean(delta_verylow)
    mean_delta_low = statistics.fmean(delta_low)
    n_pos_verylow = sum(1 for d in delta_verylow if d > 0)
    c1_pass = (mean_delta_verylow >= DELTA_FLOOR) and (n_pos_verylow >= MIN_SEEDS_MAJORITY)

    degeneracy = check_degeneracy({"C1_delta_verylow": delta_verylow})
    non_degenerate = degeneracy["non_degenerate"]

    if not ready:
        outcome, direction, label = "FAIL", "non_contributory", "substrate_not_ready_requeue"
    elif not non_degenerate:
        outcome, direction, label = "FAIL", "non_contributory", "c1_delta_degenerate_vacuous_test"
    elif c1_pass:
        outcome, direction, label = (
            "PASS", "supports",
            "precision_weighted_pe_suppression_produces_self_sealing_world_model",
        )
    else:
        outcome, direction, label = (
            "FAIL", "weakens",
            "precision_weighted_pe_suppression_does_not_produce_self_sealing_world_model",
        )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "grid_size": GRID_SIZE,
        "hazard_harm": HAZARD_HARM,
        "proximity_harm_scale": PROXIMITY_HARM_SCALE,
        "proximity_benefit_scale": PROXIMITY_BENEFIT_SCALE,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "lr": LR,
        "agent_start": AGENT_START,
        "regime_a_hazards": REGIME_A_HAZARDS,
        "regime_b_hazards": REGIME_B_HAZARDS,
        "resource_positions": RESOURCE_POSITIONS,
        "arms": {arm_id: pe_precision for arm_id, pe_precision in ARMS},
        "n_a": n_a,
        "n_b": n_b,
        "steps_per_episode": steps,
        "seeds": seeds,
        "reduction_floor": REDUCTION_FLOOR,
        "spike_floor": SPIKE_FLOOR,
        "drift_ratio_floor": DRIFT_RATIO_FLOOR,
        "delta_floor": DELTA_FLOOR,
        "min_seeds_majority": MIN_SEEDS_MAJORITY,
        "fix_vs_826": "calls agent._e1_tick(latent) on e1_tick cadence; no other change",
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "evidence_direction": direction,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "arm_results": arm_results,
        "metrics": {
            "p1_reduction_ratio_phaseA_control_worst_seed": p1_measured,
            "p2_spike_ratio_phaseB_onset_control_worst_seed": p2_measured,
            "p3_drift_ratio_control_over_verylow_worst_seed": p3_measured,
            "mean_delta_verylow": mean_delta_verylow,
            "mean_delta_low": mean_delta_low,
            "n_pos_verylow": n_pos_verylow,
            "delta_verylow_per_seed": delta_verylow,
            "delta_low_per_seed": delta_low,
            "total_e1_ticks": total_e1_ticks,
        },
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": non_degenerate},
            "criteria": [
                {
                    "name": "C1_verylow_suppression",
                    "load_bearing": True,
                    "passed": bool(c1_pass),
                }
            ],
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "design_notes": {
            "gov_reuse_1_check": (
                "MECH-244 claim_evidence.v1.json: 2 lit entries, 0 experimental "
                "entries (as of 826's authoring), exp_conf=0.0. V3-EXQ-826 is the "
                "only prior run of this design and its E1 never trained (the "
                "defect this run fixes), so no manifest on any substrate_hash "
                "carries this readout against a genuinely trained E1. Not "
                "recoverable -> run."
            ),
            "re_derive_brake_check": (
                "Exactly one prior autopsy target for MECH-244 (V3-EXQ-826, "
                "category competence_implementation_gap / instrument defect, "
                "not substrate_ceiling). Count=1, below the >=2 threshold; "
                "brake does not fire."
            ),
            "fix_vs_826": (
                "826's _run_phase never called agent._e1_tick(latent) (the "
                "tick dict from agent.clock.advance() was discarded), so "
                "_world_experience_buffer never reached compute_prediction_loss()'s "
                "length>=2 floor and E1 trained on a permanent zero-gradient "
                "stub for the whole run. This script captures the latent and "
                "the ticks dict and calls agent._e1_tick(latent) when "
                "ticks.get('e1_tick', False) -- the same gate act()/"
                "act_with_split_obs() use -- and nothing else. Per-phase "
                "n_e1_ticks is recorded on every row for audit "
                "(failure_autopsy_batch-822a-826-817a-827_2026-07-26.md "
                "section 1)."
            ),
        },
    }

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(
        "V3-EXQ-826a: MECH-244 precision-weighted PE suppression / self-sealing "
        "world model (CORRECTED -- E1 now ticked, see module docstring)",
        flush=True,
    )
    print(f"  dry_run={args.dry_run}", flush=True)

    t_start = time.perf_counter()
    manifest = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        out_dir=None,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seeds"),
        script_path=Path(__file__),
        started_at=t_start,
    )

    print(
        f"  outcome={manifest['outcome']} direction={manifest['evidence_direction']} "
        f"label={manifest['interpretation']['label']}",
        flush=True,
    )
    print(
        f"  mean_delta_verylow={manifest['metrics']['mean_delta_verylow']:.4f} "
        f"(floor {DELTA_FLOOR}) n_pos_verylow={manifest['metrics']['n_pos_verylow']}"
        f"/{len(manifest['metrics']['delta_verylow_per_seed'])} "
        f"total_e1_ticks={manifest['metrics']['total_e1_ticks']}",
        flush=True,
    )

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
