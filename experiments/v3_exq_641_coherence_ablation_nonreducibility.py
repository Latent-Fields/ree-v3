#!/opt/local/bin/python3
"""
V3-EXQ-641 -- Coherence-ablation: is C(tau) non-reducible to E(tau)?

Claims: [] (experiment_purpose=diagnostic; no claim weighting -- runs are
        excluded from confidence/conflict scoring per REE_assembly Phase-3
        governance rules). Bears on (cited here only, NOT tagged): INV-002
        (coherence includes temporal/phase binding), ARC-018 (rollout viability
        mapping), MECH-061 (commitment boundary), MECH-269 (per-stream
        verisimilitude), MECH-270 (ephaptic coupling).

Settles two intake docs at once (they share one discriminator):
  - REE_assembly/evidence/planning/thought_intake_2026-04-23_binding.md
  - REE_assembly/evidence/planning/thought_intake_2026-04-23_path_integral_constraints_search.md

THE QUESTION (verbatim from both intakes)
-----------------------------------------
Is the coherence term C(tau) non-reducible to the integrated prediction error
E(tau)? I.e. does a multiplicative coherence term in the selection rule
P(tau) ~ exp(-beta E(tau)) * C(tau) change WHICH trajectory/binding is
selected, INDEPENDENTLY of prediction-error magnitude?

WHY ONE EXPERIMENT SETTLES BOTH
-------------------------------
binding-coherence and trajectory-coherence are the same C(tau). If a coherence
term changes the committed trajectory and that propagates to BEHAVIOUR -- driven
by information orthogonal to E and not reproducible by a range-matched random
tie-breaker -- both intakes' candidate Q (entities/selection.coherence_nonreducibility)
converts to a registerable claim. Otherwise both close as "structural analogy,
no mechanism" (the philosophy-right / mechanism-wrong risk class the path-integral
import sits in; memory feedback_biology_before_formal_definitions -- register a
claim only on BEHAVIOURAL divergence, never on formal similarity).

SUBSTRATE (no ree_core change -- harness-level ablation)
--------------------------------------------------------
- Per-candidate E(tau) is the substrate's own per-candidate cost:
  agent.e3.score_trajectory(cand, goal_state=...) (e3_selector.py). The baseline
  selector A IS the existing rule probs = softmax(-scores / T) (e3_selector.py:831)
  -- i.e. exp(-beta E) with beta = 1/T.
- Per-candidate C(tau) is NOT wired into selection: hippocampal.per_stream_vs
  (MECH-269) runs on the WAKING stream once per tick, not rolled forward per
  candidate. So we compute C(tau) IN THE HARNESS from already-exposed
  per-candidate Trajectory streams (.world_states = z_world rollout, .states =
  z_self rollout, .action_objects = o_t rollout). Wiring C into the substrate's
  own E3 selector is a downstream step ONLY if this run supports non-reducibility.

THE SCALE FIX (gap-relative authority -- mandatory)
---------------------------------------------------
A pre-flight probe found E spreads ~O(50) across candidates while the coherence
term -log C spreads ~O(0.004): a fixed-magnitude multiplicative C never changes
the argmin -- the "drowning" failure the substrate's own modulatory-bias-
selection-authority substrate documents and fixes (CLAUDE.md 2026-06-03). beta in
exp(-beta E) is a FREE temperature; the question is only meaningful at a beta
where exp(-beta E) does not already saturate to argmin. So arm B uses GAP-RELATIVE
authority: scale the coherence log-term so its range == COHERENCE_AUTHORITY_GAIN *
range(E). With gain < 1 the coherence term is competitive in near-ties but
subdominant when the prediction-error gap exceeds gain*range(E) (a clearly-lower-E
trajectory still wins). This is the substrate's modulatory_authority_gain pattern,
applied in the harness.

DESIGN (paired A/B rollouts; three conditions)
----------------------------------------------
Two agents (agent_A, agent_B) built bit-identically and stepped through their own
env copies seeded identically. Selection is harness-controlled and deterministic
given (E, C):
  - E_i  = e3.score_trajectory(cand_i, goal_state, z_harm_a).mean()  (identical
           call for both arms -> identical E by construction).
  - C_i  = cross-system temporal/phase consistency of cand_i over the rollout
           window (independent of E / outcome; _candidate_coherence). For the
           random control condition, C_i is replaced by a range-matched,
           rollout-uncorrelated random value.
  - Arm A picks argmin_i E_i.                                  (pure error-minimiser)
  - Arm B picks argmin_i ( E_i + lam_i * (-log C_i) ), lam gap-relative so the
           coherence-term range == gain * range(E)            (coherence-weighted;
           == argmax exp(-beta E)*C in the soft-beta regime; higher C favoured).
The chosen candidate's first action (cand[idx].actions[:, 0, :]) is executed
directly in env. We DO NOT route through select_action's stateful commitment
logic, so A and B are pure functions of (E, C) with no hidden-state drift.

Lockstep guarantee: at every step both agents reseed torch with a per-step
deterministic seed immediately before generate_trajectories, so while in lockstep
their candidate POOLS are identical and the ONLY difference is the selection rule.
The first time C flips B's pick they diverge in state; thereafter behavioural
divergence accumulates.

P0 warmup runs BOTH agents under the SAME selector (pure-E argmin) with identical
per-step reseeding, so they enter the P1 measurement window bit-identical; the
arms split only at P1 (A keeps pure-E, B switches to coherence-weighted). No new
encoder head is trained (agent.eval(); online updates only via record_transition
+ update_z_goal), so there is no joint-training / moving-target collapse risk --
the P0->P1 structure is the phased-training discipline for this measurement.

CONDITIONS (3 x SEEDS)
  real_C_clean   : B = gap-relative REAL coherence, perturb 0.0      (PRIMARY)
  rand_C_control : B = gap-relative RANDOM coherence, perturb 0.0    (SPECIFICITY)
  real_C_perturb : B = gap-relative REAL coherence, perturb 0.20     (REBINDING)
The perturbation condition injects controlled temporal misalignment / additive
noise onto z_world fed to generate (E1 z_world vs E2 z_self desync); the binding
intake predicts REBINDING (a competing config overtakes the current one in
exp(-beta E)*C), measured as B's committed pick changing vs its clean-rank pick.

BEHAVIOURAL DIVERGENCE METRIC
  frac_state_div = fraction of P1 steps where agent_A and agent_B world_state
  observations differ at that step (instantaneous; robust to episode resets
  re-syncing the paired agents). frac_flip = fraction of P1 steps
  where the committed first-action class differs (immediate). These are the
  BEHAVIOURAL signals; "selected a different index" alone is not counted.

ACCEPTANCE (pre-registered; PASS = a registerable, coherence-specific divergence)
On real_C_clean, per seed:
  D1 (behavioural divergence): frac_state_div >= D1_MIN_FRAC_STATE_DIV.
  D3 (C non-reducible to E): |Spearman corr(E, C)| < D3_MAX_ABS_CORR (C carries
     information orthogonal to E).
Coherence-specificity (per seed, matched across conditions):
  SPEC: real_C_clean.frac_state_div >= rand_C_control.frac_state_div +
        COH_SPEC_MARGIN (real coherence drives MORE behavioural divergence than a
        range-matched random tie-breaker -- the divergence is coherence-specific,
        not generic-tie-break).
Run PASS = D1 AND D3 AND SPEC on >= MIN_SEEDS_FOR_PASS of SEEDS.

FALSIFICATION (maps the intakes' explicit falsifiers)
  F1 (D1 fails): removing C produces no behavioural difference -> close both intakes.
  F2 (D3 fails): |corr(E,C)| ~ 1 -> C collapses to a reparameterisation of E ->
                 close both intakes.
  F3 (SPEC fails, D1+D3 hold): C changes selection but no more than a random
                 tie-breaker -> the divergence is "any orthogonal bias breaks
                 ties", not coherence-specific -> do NOT register; route a
                 follow-up before any claim ("selection explained by error
                 magnitude + generic tie-break, not by coherence" -- the intakes'
                 falsifier 3).

INTERPRETATION GRID (one row per plausible outcome -> next action)
  C_nonreducible_coherence_specific_register  (D1+D3+SPEC)
      -> Coherence IS a non-reducible, coherence-specific selection factor.
         Convert entities/selection.coherence_nonreducibility to a registerable
         claim (governance). Next: /implement-substrate wiring C into E3.select
         under gap-relative authority (the modulatory-bias pattern).
  C_changes_selection_specificity_unproven_route_followup  (D1+D3, NOT SPEC)
      -> Coherence changes selection + behaviour but not beyond a random
         tie-break. Do NOT register. /failure-autopsy or a sharper coherence read
         less correlated with the score's reality-cost term, OR a coherence-tracked
         rebinding test (real_C_perturb signature vs random).
  F1_no_behavioural_difference_close_intakes  (NOT D1)
      -> No behavioural effect even with authority. Both intakes CLOSE. No claim.
  F2_c_reduces_to_e_close_intakes  (NOT D3)
      -> C reduces to E. Both intakes CLOSE. No claim.

Notes
-----
- experiment_purpose=diagnostic; claim_ids=[] -> not weighted in governance
  confidence / conflict. The result drives registration vs closure of the two
  intakes, NOT a confidence update on any existing claim.
- No agent module is monkey-patched; no ree_core modification. The substrate's
  generate_trajectories + score_trajectory + Trajectory streams are read; the
  selection is re-implemented in the harness purely to ablate the C term.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_641_coherence_ablation_nonreducibility"
QUEUE_ID = "V3-EXQ-641"
CLAIM_IDS: List[str] = []  # diagnostic -- bears on INV-002/ARC-018/MECH-061/MECH-269/MECH-270, tags NONE
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 40
P1_MEASUREMENT_EPISODES = 25
TOTAL_TRAIN_EPISODES = P0_WARMUP_EPISODES + P1_MEASUREMENT_EPISODES
STEPS_PER_EPISODE = 120

# Conditions: (name, coherence_source, perturb).  PRIMARY + control + rebinding.
COND_PRIMARY = "real_C_clean"
COND_CONTROL = "rand_C_control"
COND_PERTURB = "real_C_perturb"
PERTURB_MAGNITUDE = 0.20
CONDITIONS: List[Dict[str, Any]] = [
    {"name": COND_PRIMARY, "coherence_source": "real", "perturb": 0.0},
    {"name": COND_CONTROL, "coherence_source": "random", "perturb": 0.0},
    {"name": COND_PERTURB, "coherence_source": "real", "perturb": PERTURB_MAGNITUDE},
]

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Gap-relative coherence authority (modulatory-bias-selection-authority pattern):
# arm-B coherence-term range == COHERENCE_AUTHORITY_GAIN * range(E). gain < 1 ->
# subdominant when the E-gap exceeds gain*range(E); decisive in near-ties.
COHERENCE_AUTHORITY_GAIN = 0.5
AUTHORITY_GAIN_SWEEP: List[float] = [0.25, 0.5, 1.0]  # reported only, not a gate.
COH_EPS = 1e-6

# Coherence read weights (pre-registered).
COH_STREAMS = ("world_states", "states", "action_objects")
COH_CROSS_STREAM_WEIGHT = 0.5  # weight of the z_world<->z_self delta-alignment term.

# Acceptance thresholds (pre-registered, never derived from this run).
D1_MIN_FRAC_STATE_DIV = 0.05    # >=5% of P1 steps with divergent world_state.
D3_MAX_ABS_CORR = 0.90          # pooled |Spearman corr(E, C)| < 0.90.
COH_SPEC_MARGIN = 0.05          # real divergence exceeds random by >= this.
# Descriptive (reported, not gated): matched-E tie epsilon.
TIE_EPSILON_RANGE_FRAC = 0.10

MIN_SEEDS_FOR_PASS = 2     # of 3 seeds must satisfy D1+D3+SPEC (primary cond).
MIN_SEEDS_COMPLETED = 2    # of 3 primary-condition runs must reach P1.

# Mirror V3-EXQ-543k / V3-EXQ-605 / V3-EXQ-608 reef-bipartite SD-054 wiring
# (categorically-different reef-vs-forage first actions -> >=2 first-action
# classes per candidate pool, so matched-E-different-C ties can occur).
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Main-path SP-CEM + MECH-313 + V_s + SD-054 stack. No substrate change."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # ARC-065 SP-CEM child (Layer A; landed as default 2026-05-17) -- gives
        # >= 2 first-action classes in the candidate pool (needed for ties).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # MECH-313 noise floor (Layer C).
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # MECH-269 V_s regional verisimilitude (Layer D) -- the waking-stream
        # coherence machinery this experiment rolls forward per candidate.
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# obs helpers (mirror V3-EXQ-608)
# ---------------------------------------------------------------------------


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _world_state_tensor(obs_dict) -> torch.Tensor:
    w = obs_dict["world_state"].float()
    return w.reshape(-1)


# ---------------------------------------------------------------------------
# Per-candidate E and C
# ---------------------------------------------------------------------------


def _trajectory_first_action_class(traj) -> int:
    """First-action class id (mirror hippocampal.module static helper)."""
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _candidate_energy(
    agent: REEAgent, cand, z_harm_a: Optional[torch.Tensor]
) -> float:
    """E(tau): the substrate's own per-candidate cost J(zeta). Identical call
    for both arms -> E identical between arms. Lower is better (REE convention).
    """
    score = agent.e3.score_trajectory(
        cand, goal_state=agent.goal_state, z_harm_a=z_harm_a,
    )
    return float(score.detach().reshape(-1).float().mean().item())


def _stream_seq(cand, name: str) -> Optional[List[torch.Tensor]]:
    seq = getattr(cand, name, None)
    if seq is None or len(seq) < 2:
        return None
    return list(seq)


def _temporal_consistency(seq: List[torch.Tensor]) -> List[float]:
    """Per-step V_s proxy 1 - ||z_{t+1}-z_t|| / (||z_t|| + eps), clipped [0,1]
    (the MECH-269 verisimilitude form) -- temporal smoothness of one stream,
    independent of outcome.
    """
    out: List[float] = []
    for t in range(len(seq) - 1):
        z_t = seq[t].detach().reshape(-1).float()
        z_n = seq[t + 1].detach().reshape(-1).float()
        denom = float(z_t.norm().item()) + COH_EPS
        err = float((z_n - z_t).norm().item()) / denom
        out.append(max(0.0, min(1.0, 1.0 - err)))
    return out


def _cross_stream_alignment(
    seq_a: List[torch.Tensor], seq_b: List[torch.Tensor]
) -> List[float]:
    """Per-step cosine alignment of the step-to-step DELTAS of two streams,
    mapped to [0,1] = 0.5*(1+cos). The phase/binding term: are z_world and
    z_self changing coherently TOGETHER over the rollout window?
    """
    n = min(len(seq_a), len(seq_b))
    out: List[float] = []
    for t in range(n - 1):
        da = (seq_a[t + 1] - seq_a[t]).detach().reshape(-1).float()
        db = (seq_b[t + 1] - seq_b[t]).detach().reshape(-1).float()
        d = min(da.numel(), db.numel())
        if d == 0:
            continue
        da = da[:d]
        db = db[:d]
        na = float(da.norm().item())
        nb = float(db.norm().item())
        if na < COH_EPS or nb < COH_EPS:
            out.append(1.0)  # no change in either -> trivially aligned
            continue
        cos = float(torch.dot(da, db).item()) / (na * nb)
        cos = max(-1.0, min(1.0, cos))
        out.append(0.5 * (1.0 + cos))
    return out


def _candidate_coherence(cand) -> float:
    """C(tau) in (0,1]: cross-system temporal/phase consistency across the
    E1-prediction (z_world) / E2-transition (z_self) / action streams over the
    rollout window. Geometric mean of (a) per-stream temporal-consistency scores
    and (b) the z_world<->z_self cross-stream delta-alignment term. Defined
    INDEPENDENTLY of E / outcome.
    """
    logs: List[float] = []
    weights: List[float] = []
    for name in COH_STREAMS:
        seq = _stream_seq(cand, name)
        if seq is None:
            continue
        for v in _temporal_consistency(seq):
            logs.append(math.log(max(v, COH_EPS)))
            weights.append(1.0)
    seq_w = _stream_seq(cand, "world_states")
    seq_s = _stream_seq(cand, "states")
    if seq_w is not None and seq_s is not None:
        for v in _cross_stream_alignment(seq_w, seq_s):
            logs.append(math.log(max(v, COH_EPS)) * COH_CROSS_STREAM_WEIGHT)
            weights.append(COH_CROSS_STREAM_WEIGHT)
    if not logs:
        return 1.0  # no usable streams -> coherence-neutral
    c = math.exp(sum(logs) / max(sum(weights), COH_EPS))
    return max(COH_EPS, min(1.0, c))


def _spearman_abs(xs: List[float], ys: List[float]) -> Optional[float]:
    """|Spearman rank correlation| over paired samples; None if degenerate."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def _rank(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx < COH_EPS or vy < COH_EPS:
        return None
    return abs(cov / math.sqrt(vx * vy))


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return s[m]
    return 0.5 * (s[m - 1] + s[m])


# ---------------------------------------------------------------------------
# Selection over a candidate pool (harness-controlled; ablates C)
# ---------------------------------------------------------------------------


def _select_pure_e(energies: List[float]) -> int:
    """Arm A: argmin E (pure error-minimiser)."""
    return min(range(len(energies)), key=lambda i: energies[i])


def _select_coherence_weighted(
    energies: List[float], coherences: List[float], gain: float
) -> int:
    """Arm B: gap-relative coherence authority. Scale the coherence log-term so
    its range == gain * range(E), then argmin( E + lam*(-log C) ). Higher C ->
    smaller -log C -> lower score -> favoured (REE lower-is-better). When the
    pool has no E spread or no C spread, falls back to pure-E (no authority to
    exert). This is the substrate's modulatory_authority_gain pattern.
    """
    neg = [-math.log(max(c, COH_EPS)) for c in coherences]
    e_range = max(energies) - min(energies)
    c_range = max(neg) - min(neg)
    if e_range < COH_EPS or c_range < COH_EPS:
        return _select_pure_e(energies)
    lam = gain * e_range / c_range
    scores = [energies[i] + lam * neg[i] for i in range(len(energies))]
    return min(range(len(scores)), key=lambda i: scores[i])


def _random_coherences(n: int, gen: torch.Generator) -> List[float]:
    """Range-matched, rollout-UNCORRELATED coherence control. Uniform in
    (0.5, 1.0]; the gap-relative rule normalises the range, so what matters is
    that these carry NO temporal-coherence structure.
    """
    r = torch.rand(n, generator=gen)
    return [float(0.5 + 0.5 * v.item()) for v in r]


def _perturb_latent(latent, magnitude: float, gen: torch.Generator):
    """Inject controlled temporal misalignment / additive noise onto z_world
    only (desynchronising the E1 z_world rollout from the E2 z_self rollout).
    magnitude == 0.0 -> returns latent unchanged.
    """
    if magnitude <= 0.0:
        return latent
    lp = latent.detach()
    zw = lp.z_world
    rms = float(zw.detach().norm().item()) / max(zw.numel(), 1) ** 0.5
    noise = torch.randn(zw.shape, generator=gen, device=zw.device) * (magnitude * rms)
    lp.z_world = zw + noise
    return lp


# ---------------------------------------------------------------------------
# One harness-controlled decision for one arm
# ---------------------------------------------------------------------------


def _decision(
    agent: REEAgent,
    obs_dict,
    global_step: int,
    arm: str,
    is_p1: bool,
    perturb: float,
    coherence_source: str,
    perturb_gen: torch.Generator,
    rand_gen: torch.Generator,
    zself_prev: Optional[torch.Tensor],
    act_prev: Optional[torch.Tensor],
):
    """Returns (action[1,A], committed_first_action_class, diag, z_self_now).

    diag (arm B at P1 only) carries energies / coherences / picks. z_self_now
    threads the next-step online E2 update. Lockstep: reseed torch
    deterministically immediately before generate_trajectories so A and B see
    identical candidate pools while in lockstep. The clean-rank reference pool
    (rebinding) is drawn from the SAME per-step seed as the perturbed pool, so
    the two pools differ ONLY by the z_world perturbation.
    """
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)

    latent = agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs_harm(obs_dict),
        obs_harm_a=_obs_harm_a(obs_dict),
        obs_harm_history=_obs_harm_history(obs_dict),
    )
    z_self_now = latent.z_self.detach()
    if zself_prev is not None and act_prev is not None:
        agent.record_transition(zself_prev, act_prev, z_self_now)

    z_harm_a = getattr(latent, "z_harm_a", None)
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick", False)
        else torch.zeros(1, wdim, device=agent.device)
    )
    step_seed = (global_step * 2654435761) % (2 ** 31)

    torch.manual_seed(step_seed)
    clean_candidates = agent.generate_trajectories(latent, e1_prior, ticks)

    if perturb > 0.0:
        gen_latent = _perturb_latent(latent, perturb, perturb_gen)
        torch.manual_seed(step_seed)
        exec_candidates = agent.generate_trajectories(gen_latent, e1_prior, ticks)
    else:
        exec_candidates = clean_candidates

    if not exec_candidates:
        action = agent.select_action(exec_candidates, ticks)
        cls = int(action[0].argmax().item())
        return action, cls, None, z_self_now

    energies = [_candidate_energy(agent, c, z_harm_a) for c in exec_candidates]
    if coherence_source == "random":
        coherences = _random_coherences(len(exec_candidates), rand_gen)
    else:
        coherences = [_candidate_coherence(c) for c in exec_candidates]
    pick_a = _select_pure_e(energies)

    if arm == "A":
        pick = pick_a
    elif is_p1:
        pick = _select_coherence_weighted(
            energies, coherences, COHERENCE_AUTHORITY_GAIN
        )
    else:
        pick = pick_a  # P0: B mirrors A so arms enter P1 identical.

    chosen = exec_candidates[pick]
    action = chosen.actions[:, 0, :].detach()
    cls = _trajectory_first_action_class(chosen)

    diag = None
    if arm == "B" and is_p1:
        clean_pick_b = None
        if perturb > 0.0 and clean_candidates:
            ce = [_candidate_energy(agent, c, z_harm_a) for c in clean_candidates]
            if coherence_source == "random":
                cc = _random_coherences(len(clean_candidates), rand_gen)
            else:
                cc = [_candidate_coherence(c) for c in clean_candidates]
            clean_pick_b = _select_coherence_weighted(
                ce, cc, COHERENCE_AUTHORITY_GAIN
            )
        diag = {
            "energies": energies,
            "coherences": coherences,
            "pick_a": pick_a,
            "pick_b": pick,
            "clean_pick_b": clean_pick_b,
        }
    return action, cls, diag, z_self_now


# ---------------------------------------------------------------------------
# Paired A/B rollout for one (seed, condition)
# ---------------------------------------------------------------------------


def _run_pair(
    seed: int,
    condition: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    cond_name = condition["name"]
    coherence_source = condition["coherence_source"]
    perturb = float(condition["perturb"])
    total_train_eps = p0_episodes + p1_episodes
    error_note: Optional[str] = None

    torch.manual_seed(seed)
    env_a = _make_env(seed)
    agent_a = _make_agent(env_a)
    agent_a.eval()
    torch.manual_seed(seed)
    env_b = _make_env(seed)
    agent_b = _make_agent(env_b)
    agent_b.eval()

    perturb_gen = torch.Generator()
    perturb_gen.manual_seed(seed * 7919 + int(perturb * 1000) + 1)
    rand_gen = torch.Generator()
    rand_gen.manual_seed(seed * 6271 + 13)

    n_p1_steps = 0
    n_flip_steps = 0           # immediate committed first-action class differs
    n_state_div_steps = 0      # world_state of agent_A != agent_B (behavioural)
    pooled_e: List[float] = []
    pooled_c: List[float] = []
    e_gap_at_flip: List[float] = []
    n_tie_ticks = 0
    n_tie_b_higher_c = 0
    n_rebind = 0
    contacts_a = 0
    contacts_b = 0
    gain_flip: Dict[str, int] = {f"{g:g}": 0 for g in AUTHORITY_GAIN_SWEEP}

    global_step = 0

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        _, obs_a = env_a.reset()
        _, obs_b = env_b.reset()
        agent_a.reset()
        agent_b.reset()
        zp_a = ap_a = zp_b = ap_b = None

        for _step in range(steps_per_episode):
            act_a, cls_a, _diag_a, zself_a = _decision(
                agent_a, obs_a, global_step, arm="A", is_p1=is_p1,
                perturb=0.0, coherence_source=coherence_source,
                perturb_gen=perturb_gen, rand_gen=rand_gen,
                zself_prev=zp_a, act_prev=ap_a,
            )
            act_b, cls_b, diag_b, zself_b = _decision(
                agent_b, obs_b, global_step, arm="B", is_p1=is_p1,
                perturb=(perturb if is_p1 else 0.0),
                coherence_source=coherence_source,
                perturb_gen=perturb_gen, rand_gen=rand_gen,
                zself_prev=zp_b, act_prev=ap_b,
            )

            if not (torch.isfinite(act_a).all() and torch.isfinite(act_b).all()):
                error_note = (
                    f"non-finite action seed={seed} cond={cond_name} "
                    f"ep={ep} step={_step}"
                )
                break

            if is_p1:
                n_p1_steps += 1
                if cls_a != cls_b:
                    n_flip_steps += 1
                if diag_b is not None:
                    energies = diag_b["energies"]
                    coherences = diag_b["coherences"]
                    pick_a = diag_b["pick_a"]
                    pick_b = diag_b["pick_b"]
                    pooled_e.extend(energies)
                    pooled_c.extend(coherences)
                    e_range = (
                        (max(energies) - min(energies))
                        if len(energies) >= 2 else 0.0
                    )
                    if cls_a != cls_b and e_range > COH_EPS:
                        e_gap_at_flip.append(
                            abs(energies[pick_b] - energies[pick_a]) / e_range
                        )
                    if len(energies) >= 2 and e_range > COH_EPS:
                        se = sorted(energies)
                        if (se[1] - se[0]) / e_range < TIE_EPSILON_RANGE_FRAC:
                            n_tie_ticks += 1
                            if coherences[pick_b] > coherences[pick_a]:
                                n_tie_b_higher_c += 1
                    for g in AUTHORITY_GAIN_SWEEP:
                        if _select_coherence_weighted(energies, coherences, g) != pick_a:
                            gain_flip[f"{g:g}"] += 1
                    if perturb > 0.0 and diag_b.get("clean_pick_b") is not None:
                        if diag_b["clean_pick_b"] != pick_b:
                            n_rebind += 1

            _, _hs_a, done_a, info_a, obs_a = env_a.step(act_a)
            _, _hs_b, done_b, info_b, obs_b = env_b.step(act_b)

            if is_p1:
                # Instantaneous behavioural divergence: are the two paired
                # agents in different world-states right now? (Robust to
                # episode resets re-syncing them; measures the fraction of P1
                # the agents are actually on different trajectories.)
                if not torch.equal(
                    _world_state_tensor(obs_a), _world_state_tensor(obs_b)
                ):
                    n_state_div_steps += 1
                if float(info_a.get("benefit_exposure", 0.0)) > 0.0:
                    contacts_a += 1
                if float(info_b.get("benefit_exposure", 0.0)) > 0.0:
                    contacts_b += 1

            for ag, ob, info in ((agent_a, obs_a, info_a), (agent_b, obs_b, info_b)):
                if ag.goal_state is not None:
                    be = float(info.get("benefit_exposure", 0.0))
                    en = float(ob["body_state"].float().reshape(-1)[3].item())
                    ag.update_z_goal(
                        benefit_exposure=be, drive_level=max(0.0, 1.0 - en)
                    )

            zp_a, ap_a = zself_a, act_a.detach()
            zp_b, ap_b = zself_b, act_b.detach()
            global_step += 1
            if done_a or done_b:
                break

        if error_note is not None:
            break
        if (ep + 1) % 10 == 0 or ep == total_train_eps - 1:
            print(
                f"  [train] seed={seed} cond={cond_name} "
                f"ep {ep + 1}/{total_train_eps} "
                f"phase={'P1' if is_p1 else 'P0'} "
                f"p1_steps={n_p1_steps} flips={n_flip_steps} "
                f"state_div={n_state_div_steps}",
                flush=True,
            )

    frac_flip = (n_flip_steps / n_p1_steps) if n_p1_steps else 0.0
    frac_state_div = (n_state_div_steps / n_p1_steps) if n_p1_steps else 0.0
    abs_corr = _spearman_abs(pooled_e, pooled_c)
    tie_higher_c_frac = (n_tie_b_higher_c / n_tie_ticks) if n_tie_ticks else None
    median_e_gap_at_flip = _median(e_gap_at_flip)

    d1 = bool(frac_state_div >= D1_MIN_FRAC_STATE_DIV)
    d3 = bool(abs_corr is not None and abs_corr < D3_MAX_ABS_CORR)

    return {
        "seed": int(seed),
        "condition": cond_name,
        "coherence_source": coherence_source,
        "perturb": perturb,
        "error_note": error_note,
        "n_p1_steps": int(n_p1_steps),
        "n_flip_steps": int(n_flip_steps),
        "frac_flip": float(frac_flip),
        "n_state_div_steps": int(n_state_div_steps),
        "frac_state_div": float(frac_state_div),
        "abs_spearman_corr_e_c": abs_corr,
        "n_tie_ticks": int(n_tie_ticks),
        "tie_higher_c_frac": tie_higher_c_frac,
        "median_e_gap_at_flip_norm": median_e_gap_at_flip,
        "n_rebind_under_perturb": int(n_rebind),
        "contacts_a": int(contacts_a),
        "contacts_b": int(contacts_b),
        "gain_flip_counts": gain_flip,
        "D1_behavioural_divergence": d1,
        "D3_c_nonreducible": d3,
    }


# ---------------------------------------------------------------------------
# Interpret / run / manifest
# ---------------------------------------------------------------------------


def _interpret(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PASS gate on the PRIMARY condition, with the random-control specificity
    check matched per seed.
    """
    by_seed_primary = {
        r["seed"]: r for r in rows
        if r["condition"] == COND_PRIMARY and r["error_note"] is None
    }
    by_seed_control = {
        r["seed"]: r for r in rows
        if r["condition"] == COND_CONTROL and r["error_note"] is None
    }

    n_d1 = sum(1 for r in by_seed_primary.values() if r["D1_behavioural_divergence"])
    n_d3 = sum(1 for r in by_seed_primary.values() if r["D3_c_nonreducible"])
    n_spec = 0
    spec_detail: List[Dict[str, Any]] = []
    for s, pr in by_seed_primary.items():
        cr = by_seed_control.get(s)
        if cr is None:
            continue
        real_div = pr["frac_state_div"]
        rand_div = cr["frac_state_div"]
        is_spec = bool(real_div >= rand_div + COH_SPEC_MARGIN)
        if is_spec:
            n_spec += 1
        spec_detail.append({
            "seed": s,
            "real_frac_state_div": real_div,
            "rand_frac_state_div": rand_div,
            "coherence_specific": is_spec,
        })

    n_seed_pass = 0
    for s, pr in by_seed_primary.items():
        cr = by_seed_control.get(s)
        spec = bool(
            cr is not None
            and pr["frac_state_div"] >= cr["frac_state_div"] + COH_SPEC_MARGIN
        )
        if pr["D1_behavioural_divergence"] and pr["D3_c_nonreducible"] and spec:
            n_seed_pass += 1

    if not by_seed_primary:
        label = "no_clean_completion"
    elif n_seed_pass >= MIN_SEEDS_FOR_PASS:
        label = "C_nonreducible_coherence_specific_register"
    elif n_d1 < MIN_SEEDS_FOR_PASS:
        label = "F1_no_behavioural_difference_close_intakes"
    elif n_d3 < MIN_SEEDS_FOR_PASS:
        label = "F2_c_reduces_to_e_close_intakes"
    else:
        label = "C_changes_selection_specificity_unproven_route_followup"

    return {
        "primary_condition": COND_PRIMARY,
        "control_condition": COND_CONTROL,
        "perturb_condition": COND_PERTURB,
        "n_primary_seeds": len(by_seed_primary),
        "n_D1": int(n_d1),
        "n_D3": int(n_d3),
        "n_coherence_specific": int(n_spec),
        "n_seed_pass": int(n_seed_pass),
        "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
        "specificity_detail": spec_detail,
        "majority_label": label,
    }


def run_experiment(
    seeds: List[int],
    conditions: List[Dict[str, Any]],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    n_runs_completed = 0
    first = True
    for cond in conditions:
        for s in seeds:
            print(f"Seed {s} Condition {cond['name']}", flush=True)
            if first:
                print(
                    f"  (P0={p0_episodes} ep, P1={p1_episodes} ep, "
                    f"steps_per_episode={steps_per_episode}, "
                    f"authority_gain={COHERENCE_AUTHORITY_GAIN}, dry_run={dry_run})",
                    flush=True,
                )
                first = False
            row = _run_pair(s, cond, p0_episodes, p1_episodes, steps_per_episode)
            rows.append(row)
            if row["error_note"] is None:
                n_runs_completed += 1
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    interp = _interpret(rows)
    n_primary_completed = sum(
        1 for r in rows
        if r["condition"] == COND_PRIMARY and r["error_note"] is None
    )
    passed = bool(
        n_primary_completed >= MIN_SEEDS_COMPLETED
        and interp["majority_label"] == "C_nonreducible_coherence_specific_register"
    )

    return {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": seeds,
        "conditions": [c["name"] for c in conditions],
        "n_runs_completed": int(n_runs_completed),
        "n_total_runs": int(len(seeds) * len(conditions)),
        "min_seeds_completed": int(MIN_SEEDS_COMPLETED),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "selection": {
            "coherence_authority_gain": float(COHERENCE_AUTHORITY_GAIN),
            "authority_gain_sweep": list(AUTHORITY_GAIN_SWEEP),
            "coh_streams": list(COH_STREAMS),
            "coh_cross_stream_weight": float(COH_CROSS_STREAM_WEIGHT),
            "rule": "gap_relative_authority (range(coh-term) == gain*range(E))",
        },
        "acceptance_thresholds": {
            "D1_min_frac_state_div": float(D1_MIN_FRAC_STATE_DIV),
            "D3_max_abs_corr": float(D3_MAX_ABS_CORR),
            "coh_spec_margin": float(COH_SPEC_MARGIN),
            "tie_epsilon_range_frac": float(TIE_EPSILON_RANGE_FRAC),
        },
        "per_run_results": rows,
        "interpretation": interp,
    }


def _build_manifest(
    result: Dict[str, Any], timestamp_utc: str, dry_run: bool
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",  # diagnostic
        "evidence_direction_note": (
            "experiment_purpose=diagnostic; claim_ids=[]; not weighted in "
            "confidence or conflict scoring. Settles thought_intake_2026-04-23"
            "_binding.md + _path_integral_constraints_search.md: "
            "C_nonreducible_coherence_specific_register converts the candidate Q "
            "entities/selection.coherence_nonreducibility to a registerable claim; "
            "F1/F2 close both intakes as structural-analogy-no-mechanism; "
            "specificity_unproven routes a follow-up before any registration. "
            "majority_label="
            f"{result.get('interpretation', {}).get('majority_label')}."
        ),
        "bears_on_not_tagged": [
            "INV-002", "ARC-018", "MECH-061", "MECH-269", "MECH-270",
        ],
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "use_support_preserving_cem": True,
            "use_noise_floor": True,
            "use_per_stream_vs": True,
            "use_per_region_vs": True,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "harness_level_ablation": True,
            "ree_core_modified": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short smoke-test run (1 seed, 2+2 ep, 30 steps, all conditions).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output dir (default: REE_assembly evidence/experiments).",
    )
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        conditions=list(CONDITIONS),
        p0_episodes=p0,
        p1_episodes=p1,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"interp(majority)={result['interpretation']['majority_label']} "
        f"runs_completed={result['n_runs_completed']}/{result['n_total_runs']}",
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
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
