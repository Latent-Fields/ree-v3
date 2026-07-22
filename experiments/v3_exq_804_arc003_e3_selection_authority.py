"""V3-EXQ-804: ARC-003 LEG A -- does E3 SELECT? A commitment-free, score-level test.

Authored from EXP-0394 (REE_assembly evidence/planning/manual_proposals.v1.json,
minted by /thought-digestion 2026-07-21). claim_ids = [ARC-003] ONLY.

=== WHY THIS EXPERIMENT EXISTS ===
ARC-003 ("E3 selects and commits trajectories") carries 45 reverse dependencies on
exp_conf 0.0 -- ZERO experiments have ever been run against it; its 0.783 lit_conf
rests on 3 literature supports. It is the load-bearing premise of MECH-060/061/062
and the tri-loop gating claims, and nothing has ever tested it.

The claim is CONJUNCTIVE and its two legs have OPPOSITE readiness (claims.yaml
what_would_answer, digested 2026-07-21):

  LEG A -- "E3 SELECTS".   Measurable at the candidate-SCORING level. No sustained
                           execution required. THIS RUN.
  LEG B -- "E3 COMMITS".   Requires SUSTAINED MULTI-STEP ACTION COMMITMENT and is
                           BLOCKED on the basal-ganglia / action-commitment layer
                           (ARC-107 BG-selector constitution; MECH-448 provisional;
                           MECH-449 substrate_conditional; commit/release-duration
                           falsifiers 460j/485i/654h pending). Registered separately
                           as EXP-0395 with status blocked_substrate.

=== THE SCOPE GUARD (do NOT "strengthen" this run by adding a behavioural DV) ===
This run is DELIBERATELY COMMITMENT-FREE, per the standing rule from the
IGW-20260620-194 incident. A behavioural DV requiring the agent to hold and execute
a multi-step plan would, on the current substrate, RE-DERIVE THE F-DOMINANCE
CONVERSION CEILING ("a navigable landscape does not convert to committed action
because the selector drowns it") and return a confounded FAIL -- a re-derivation,
not an adjudication of ARC-003. Every DV here is read at the moment of scoring.

Concretely, that is why this driver calls `agent.e3.select(...)` DIRECTLY rather
than going through `agent.select_action(...)`: select_action interposes the E3
cadence (heartbeat.e3_steps_per_tick, default 10), the commitment latch, and the
BetaGate -- i.e. exactly the commitment machinery LEG B is blocked on. Calling
select() directly means every tick is one genuine, independent selection, which
also makes the ~9x E3-diagnostics pseudo-replication defect (V3-EXQ-785, measured
600 rows behind 67 real select() calls) STRUCTURALLY IMPOSSIBLE here rather than
merely guarded against. n_latched_ticks is 0 by construction and is emitted as 0.

SCOPE OF "THE E3 SCORE" TESTED. select() is called on its core scoring path with
the modulatory bias channels NOT injected (score_bias / score_bias_channels /
channel_route_bias all default None, as agent.select_action would otherwise
assemble them). So this tests E3's OWN scoring authority (F / M / Phi_R), NOT the
modulatory channels' authority, which is separately claimed under MECH-314 /
MECH-341 / MECH-320 and separately tested. A PASS here licenses "E3 selects"; it
licenses nothing about the modulatory channels.

=== THE DV, AND WHY IT IS THE RIGHT ONE ===
"E3 selects" is the assertion that E3's score DETERMINES which candidate is taken,
as against E3 being a pass-through that ratifies whatever the proposer produced.
The signature of that is a systematic departure of the SELECTED-candidate
distribution from the proposer's own GENERATION PRIOR:

  per tick:  p_prior(a) = fraction of the K candidates whose FIRST ACTION is class a
             (the proposer's own composition -- the null a pass-through would give)
  per cell:  q(a)       = empirical distribution of the EXECUTED first-action class
  DV:        KL(q || p_bar)   with p_bar the mean per-tick prior

A selector that ignores candidate content -- any content-free rule -- draws
uniformly over candidates and therefore reproduces p_bar, giving KL ~ 0 up to
sampling noise. KL > 0 is exactly the signature of "the selector used information
about the candidates". That is the claim.

=== ARMS (all three EXECUTE; the executed action is what the env sees) ===
  A0_INTACT          take the candidate E3 selected (result.selected_index).
  A1_E3_LESIONED     E3's contribution to the choice is removed: select() is still
                     called (so E3 state, running variance and the score record
                     evolve identically) but the EXECUTED candidate is drawn
                     UNIFORMLY over the untouched pool. Candidate GENERATION is
                     untouched in every arm.
  ARM_NOISE          the same content-destroying null by a DIFFERENT route: E3's
                     REAL score vector is randomly PERMUTED across candidates and
                     the same argmin rule applied. The score DISTRIBUTION (spread,
                     scale, its relation to the selection temperature) is held
                     EXACTLY fixed; only the score<->candidate pairing is destroyed.

WHY TWO NULL ARMS, STATED HONESTLY. Both are estimates of the SAME floor, and both
are ~0 BY CONSTRUCTION, not by measurement (see the DV-symmetry declaration below).
That is not a defect and it is not a hidden vacuity -- it is the point: any
content-free selector reproduces the prior. Their distinct value is WHAT EACH HOLDS
FIXED. A1 holds the selection RULE content-free while letting the score
distribution be whatever it is; ARM_NOISE holds the score DISTRIBUTION exactly
fixed while destroying its content. If A0's KL were an artifact of the
score-MAGNITUDE regime rather than of score CONTENT, ARM_NOISE would reproduce it
and A1 might not. Agreement between the two strengthens the floor; disagreement is
diagnostic. Neither is cited as a measured effect.

VERIFIED-LIFTING (the V3-EXQ-689d defect, explicitly guarded). 689d's matched-noise
control was BIT-IDENTICAL to its other control on every metric, so a "strict above
BOTH" criterion degraded SILENTLY to "strict above one" and the pre-registered
guard fired without blocking the PASS. Here P5 (lesion_took) is a GATING
precondition on each null arm: the executed index must differ from E3's pick at
close to the (1 - 1/K) rate a genuinely content-free draw implies. A null arm that
did not actually lift is RED, and its contrast is not scored.

=== DV-SYMMETRY DECLARATION (mandatory, per arm) ===
The DV is KL(executed-first-action-class distribution || per-tick pool prior). Its
symmetry group: (i) any permutation of candidates occupying the SAME first-action
class, (ii) any relabelling of candidate SLOTS that preserves the per-tick class
counts, and (iii) any transform of the score vector preserving the within-tick
argmin (in particular any monotone rescaling, and any uniform additive constant --
the V3-EXQ-604c broadcast-scalar trap).

  A0_INTACT -- NOT INVARIANT. E3's score is a state-dependent function of each
    candidate's predicted outcome, so it selects class-differentially and
    tick-differentially. It is not a broadcast constant (the per-candidate score
    RANGE is measured and gated by C2/P3, not merely its magnitude -- the
    V3-EXQ-643 magnitude-vs-range GAP), and it is not a monotone relabelling of
    slot order (a slot-order preference would show as a flat class profile, which
    the pool-composition record would expose).
  A1_E3_LESIONED -- INVARIANT BY CONSTRUCTION, DECLARED. A uniform draw over
    candidates is exactly symmetric under (i) and (ii), so its DV is the sampling
    floor at this n, fixed before the run. It is a REFERENCE arm. Its value is
    never reported as a measured effect and never routes a verdict on its own.
  ARM_NOISE -- INVARIANT BY CONSTRUCTION, DECLARED. argmin over a randomly
    permuted score vector is uniform over candidates. Same status as A1.

=== NON-DEGENERACY (the GAP-A guard, and the one nobody remembers) ===
  P1 POOL DIVERGENCE. cand_world_pairwise_dist (SD-056, e2_fast.py:224) above a
     floor. Scoring a pool of near-identical candidates tests nothing: the score
     gradient is trivially flat and a FAIL would be MANUFACTURED. This is the GAP-A
     guard the proposal names, and V3-EXQ-571 measured exactly this collapse
     (cand_world_pairwise_dist = 0.0000 across K=8 candidates differing only in
     first action).
  P2 POOL CLASS SUPPORT -- the one that is easy to miss. If the proposer emits a
     pool whose candidates all share ONE first-action class, p_prior is a point
     mass and KL is IDENTICALLY 0 for EVERY arm including A0. The contrast would
     read as a clean FAIL while measuring nothing at all. Gated at > 1.5 mean
     distinct classes per pool; support_preserving_min_first_action_classes=2 is
     set on the proposer to make it structurally likely, and P2 verifies it held.
  P3 SCORE FINITE AND BOUNDED (upper). The 643 float32-cancellation regime
     (~1e32 raw scores) makes every downstream statistic meaningless.
  P4 SELECTION COUNT floor.
  P5 LESION TOOK -- null arms only; SCOPED OUT of A0 (disposition (a): it is not
     meaningful there, since A0's executed index IS E3's pick by definition, so
     asserting it would make A0 structurally un-passable -- the V3-EXQ-785 defect).

Preconditions are evaluated PER ARM via experiments/_lib/precondition_gate.py and
aggregated with aggregate_arm_gates, so a red arm cannot vacate a green arm's
result. NOTE the one place this run legitimately differs from 785a: C1 is a
CROSS-ARM CONTRAST, so a red NULL arm removes the contrast it participates in --
that is not the 785 vacating defect (which destroyed an arm's own self-contained
result) but the arithmetic of a difference. C2 and C3 are A0-INTERNAL and still
score when a null arm is red; arm_criteria_non_degenerate carries that per
criterion.

=== PRE-REGISTERED CRITERIA ===
  C1 SELECTION AUTHORITY (LOAD-BEARING): KL_A0 exceeds BOTH null arms' KL by more
     than max(KL_ABS_FLOOR, 2 x SD of the per-seed KL delta) -- an effect-size
     floor scaled on the SD of the DELTA plus an absolute floor.
  C2 SCORE GRADIENT NON-FLAT (LOAD-BEARING): the per-tick cross-candidate score
     RANGE, normalised by score scale, above a floor in A0. Deliberately a RANGE
     and never a mean-abs / magnitude proxy: a uniform per-candidate offset has
     large magnitude and ~0 range, and range is the statistic the argmin routes on
     (the V3-EXQ-643 same-statistic rule).
  C3 TAKEN-CANDIDATE PREDICTABILITY (NOT load-bearing): on UNCOMMITTED ticks the
     taken candidate is E3's argmin above the 1/K chance rate. Declared
     non-load-bearing and its degeneracy declared: on COMMITTED ticks selection IS
     a deterministic argmin (e3_selector.py:1891, "argmin when committed;
     gap-agnostic softmax sample otherwise"), so this statistic is an arithmetic
     IDENTITY there and is scored on the uncommitted subset only, with a minimum-n
     floor. Its null: a flat rate means the score gradient is small relative to the
     selection temperature -- effective, not nominal, authority is absent.

INTERPRETATION GRID:
  C1 pass + C2 pass -> PASS  / supports        e3_selects_beyond_generation_prior
  C1 FAIL + C2 pass -> FAIL  / weakens         e3_score_gradient_without_selection_authority
                                               (the 643-shaped signature: a score
                                                exists but has no selection
                                                consequence)
  C1 FAIL + C2 FAIL -> FAIL  / weakens         e3_pass_through_not_selector
                                               (the decision-flipping negative:
                                                MECH-060/061/062 and the tri-loop
                                                gating claims lose their premise)
  C1 pass + C2 FAIL -> FAIL  / mixed           selection_departs_prior_without_score_gradient
                                               (anomalous -- flag for autopsy)
  gate RED          -> FAIL  / non_contributory  substrate_not_ready_requeue
                                               NEVER a substrate verdict.

=== TRAINING PROTOCOL ===
P0 substrate warmup (E1 + E2.self + world-decoder reconstruction + E2.world_forward
one-step MSE + the SD-056 InfoNCE action-contrastive auxiliary), then P2 frozen
eval under agent.eval() / no_grad. NO downstream probe head is trained on a latent,
so the P1 freeze-and-detach phase of the phased-training protocol does not apply --
there is no head to chase a moving target. The contrastive auxiliary is included
because without it E2.world_forward is known to fit the action contribution to zero
under reconstruction-shaped training (V3-EXQ-571 / SD-056), which would collapse
the candidate pool and trip P1 -- i.e. it protects against a
substrate_not_ready_requeue that would waste the whole run.

GOV-REUSE-1: decisive readout is KL(executed-class || per-tick pool prior) under an
E3-lesion contrast. No recorded manifest carries it -- ARC-003 has ZERO experiments
of any kind (exp_conf 0.0, genuine_exp_count 0), so there is nothing to reanalyse.
Not recoverable -> must run.

MINT-AS-YOU-GO: every cell is stamped with include_driver_script_in_hash=False, so
the A1_E3_LESIONED baseline cells are cross-driver reusable by any successor.

Run:
  /opt/local/bin/python3 experiments/v3_exq_804_arc003_e3_selection_authority.py --dry-run
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_804_arc003_e3_selection_authority"
QUEUE_ID = "V3-EXQ-804"
EXPERIMENT_PURPOSE = "evidence"
# ONLY the claim this implementation directly tests. ARC-003's LEG B (commitment)
# is NOT tested here and is registered blocked_substrate as EXP-0395. The
# proposal's related_claims (ARC-005 / ARC-002 / ARC-001 / INV-012 / MECH-060 /
# MECH-061 / MECH-062 / Q-015 / Q-016) are DELIBERATELY NOT TAGGED: this run does
# not exercise their mechanisms, and tagging a claim a run did not test corrupts
# governance confidence scores.
CLAIM_IDS: List[str] = ["ARC-003"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2]
TRAIN_EPISODES = 200
STEPS_PER_EPISODE = 100
EVAL_TICKS = 1000

DRY_TRAIN_EPISODES = 4
DRY_STEPS_PER_EPISODE = 20
DRY_EVAL_TICKS = 60

ARM_INTACT = "A0_INTACT"
ARM_LESIONED = "A1_E3_LESIONED"
ARM_NOISE = "ARM_NOISE"
ARMS: List[str] = [ARM_INTACT, ARM_LESIONED, ARM_NOISE]
NULL_ARMS = (ARM_LESIONED, ARM_NOISE)

# Substrate config.
ALPHA_WORLD = 0.9          # z_world fidelity; the 0.3 default is a known SD-008 root cause
SELECT_TEMPERATURE = 1.0
E2_CONTRASTIVE_WEIGHT = 0.1
E2W_BUF_MAX = 500
E2W_BATCH = 32
LR = 1e-3

# Gate floors (FLOORS unless marked upper).
CAND_DIST_FLOOR = 1e-3        # P1: GAP-A pool divergence (V3-EXQ-571 measured 0.0000)
POOL_CLASS_FLOOR = 1.5        # P2: mean distinct first-action classes per pool
SCORE_ABS_CEILING = 1e6       # P3: UPPER -- the 643 float32-cancellation regime
MIN_SELECTIONS = 300.0        # P4
# P5: a content-free draw over K candidates re-picks E3's own index at rate 1/K, so
# the override rate is (1 - 1/K) in expectation. The floor is set well below that
# with slack for repeated-class pools; it exists to catch a lesion that did not
# take AT ALL, not to certify its exact rate.
LESION_OVERRIDE_FLOOR = 0.35

# PASS criteria.
KL_ABS_FLOOR = 0.02           # nats; absolute floor on the A0-minus-null KL delta
KL_GAP_SD_MULTIPLIER = 2.0    # effect-size floor scaled on the SD of the DELTA
SCORE_RANGE_FLOOR = 1e-4      # C2: normalised cross-candidate score RANGE (never mean-abs)
C3_MIN_UNCOMMITTED = 100      # C3 degeneracy floor: uncommitted selections needed
C3_MARGIN = 0.05              # C3: argmin-hit rate must clear 1/K by this margin


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _kl(q: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    """KL(q || p) in nats, with additive smoothing on both sides."""
    q = np.asarray(q, dtype=float) + eps
    p = np.asarray(p, dtype=float) + eps
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * np.log(q / p)))


def _first_action_classes(candidates) -> List[int]:
    """First-action class per candidate, in the order select() indexes them."""
    out: List[int] = []
    for traj in candidates:
        out.append(int(traj.actions[:, 0, :].detach().reshape(-1).argmax().item()))
    return out


def _first_actions_K(candidates) -> torch.Tensor:
    return torch.stack(
        [t.actions[:, 0, :].detach().reshape(-1) for t in candidates], dim=0)


def _make_world_decoder(world_dim: int, world_obs_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, 64), nn.ReLU(), nn.Linear(64, world_obs_dim))


def _config_slice(env: CausalGridWorldV2, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """The config every arm shares. Declared for the arm fingerprint."""
    return dict(
        body_obs_dim=int(obs_dict["body_state"].shape[-1]),
        world_obs_dim=int(obs_dict["world_state"].shape[-1]),
        action_dim=int(env.action_dim),
        alpha_world=ALPHA_WORLD,
        # SD-056: keep candidates action-divergent, or the GAP-A guard (P1) trips
        # and the whole run self-routes substrate_not_ready_requeue.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
        # Proposer support preservation: makes a multi-class pool structurally
        # likely, which P2 then VERIFIES held (it is not assumed).
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
    )


def _build(seed: int) -> Tuple[REEAgent, CausalGridWorldV2, Dict[str, Any],
                               nn.Module, Dict[str, Any]]:
    # CausalGridWorld carries its OWN np.random.default_rng(seed); omitting seed=
    # here makes the run non-reproducible and any bit-identity check meaningless.
    env = CausalGridWorldV2(use_proxy_fields=True, seed=seed)
    _obs, obs_dict = env.reset()
    slice_ = _config_slice(env, obs_dict)
    cfg = REEConfig.from_dims(**slice_)
    agent = REEAgent(cfg)
    decoder = _make_world_decoder(cfg.latent.world_dim, int(slice_["world_obs_dim"]))
    return agent, env, obs_dict, decoder, slice_


def _train_p0(agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
              decoder: nn.Module, n_episodes: int, steps: int,
              rng: np.random.Generator, arm_id: str, seed: int) -> Dict[str, Any]:
    """P0 substrate warmup. Random actions; no head is trained on a latent."""
    agent.train()
    decoder.train()
    opt = optim.Adam(list(agent.parameters()) + list(decoder.parameters()), lr=LR)

    buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    contrast_terms: List[float] = []
    e2w_terms: List[float] = []

    for ep in range(n_episodes):
        _obs, obs_dict_ep = env.reset()
        agent.reset()
        zw_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None

        for _step in range(steps):
            obs_b = obs_dict_ep["body_state"].unsqueeze(0)
            obs_w = obs_dict_ep["world_state"].unsqueeze(0)
            latent = agent.sense(obs_b, obs_w)
            agent.clock.advance()
            zw_cur = latent.z_world.detach()

            act_idx = int(rng.integers(0, env.action_dim))
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, act_idx] = 1.0
            agent._last_action = action

            if zw_prev is not None and a_prev is not None:
                buf.append((zw_prev, a_prev, zw_cur))
                if len(buf) > E2W_BUF_MAX:
                    buf = buf[-E2W_BUF_MAX:]

            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon_loss = F.mse_loss(decoder(z_w), obs_w)

            e2w_loss = torch.zeros((), device=agent.device)
            contrast = torch.zeros((), device=agent.device)
            if len(buf) >= 16:
                k = min(E2W_BATCH, len(buf))
                idxs = torch.randperm(len(buf))[:k].tolist()
                zw_t = torch.cat([buf[i][0] for i in idxs], dim=0)
                a_t = torch.cat([buf[i][1] for i in idxs], dim=0)
                zw_t1 = torch.cat([buf[i][2] for i in idxs], dim=0)
                e2w_loss = F.mse_loss(agent.e2.world_forward(zw_t, a_t), zw_t1)
                # SD-056 auxiliary: returns the UNWEIGHTED CE (e2_fast.py:279);
                # the caller composes the weight. Returns exactly 0 when the batch
                # carries fewer than min_batch_classes distinct first-action
                # classes, so it is self-guarding.
                contrast = agent.e2.world_forward_contrastive_loss(zw_t, a_t, zw_t1)
                e2w_terms.append(float(e2w_loss.item()))
                contrast_terms.append(float(contrast.item()))

            total = e1_loss + e2_self_loss + recon_loss + e2w_loss \
                + E2_CONTRASTIVE_WEIGHT * contrast
            if total.requires_grad:
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            zw_prev = zw_cur
            a_prev = action.detach()
            _obs, _r, done, _info, obs_dict_ep = env.step(act_idx)
            if done:
                _obs, obs_dict_ep = env.reset()

        if (ep + 1) % 25 == 0 or ep == n_episodes - 1:
            _e2w = e2w_terms[-1] if e2w_terms else 0.0
            _con = contrast_terms[-1] if contrast_terms else 0.0
            print(f"  [train] arc003 {arm_id} seed={seed} ep {ep + 1}/{n_episodes} "
                  f"e2w={_e2w:.6f} contrast={_con:.4f}", flush=True)

    return {
        "e2w_loss_final_mean": float(np.mean(e2w_terms[-50:])) if e2w_terms else 0.0,
        "contrastive_loss_final_mean": (
            float(np.mean(contrast_terms[-50:])) if contrast_terms else 0.0),
    }


def _collect_cell(arm_id: str, seed: int, n_train: int, steps: int, n_ticks: int,
                  rng: np.random.Generator) -> Dict[str, Any]:
    """One (arm, seed) cell: P0 warmup then P2 frozen selection-level eval."""
    agent, env, obs_dict, decoder, _slice = _build(seed)
    print(f"Seed {seed} Condition {arm_id}", flush=True)
    train_stats = _train_p0(agent, env, obs_dict, decoder, n_train, steps, rng,
                            arm_id, seed)

    agent.eval()
    decoder.eval()
    _obs, obs_dict = env.reset()
    agent.reset()

    n_actions = int(env.action_dim)
    sel_counts = np.zeros(n_actions, dtype=float)     # executed classes
    prior_accum = np.zeros(n_actions, dtype=float)    # mean per-tick pool prior
    # Within-tick matched counterfactuals, recorded in EVERY arm from the SAME
    # ticks. Zero trajectory-divergence between the contrasted quantities, so they
    # are strictly better powered than the across-arm contrast; reported as
    # corroboration, never as the pre-registered criterion.
    ctf_prior_counts = np.zeros(n_actions, dtype=float)
    ctf_perm_counts = np.zeros(n_actions, dtype=float)

    pairwise: List[float] = []
    pool_classes: List[float] = []
    score_ranges: List[float] = []
    score_ranges_norm: List[float] = []
    score_abs_max = 0.0
    k_sizes: List[float] = []
    n_override = 0
    n_committed = 0
    n_uncommitted = 0
    n_argmin_hit_uncommitted = 0
    taken_ranks: List[float] = []
    per_tick: List[Dict[str, Any]] = []

    zw_prev: Optional[torch.Tensor] = None
    a_prev: Optional[torch.Tensor] = None

    for tick in range(n_ticks):
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"].unsqueeze(0),
                                 obs_dict["world_state"].unsqueeze(0))
            zw_cur = latent.z_world.detach()
            # V3-EXQ-396a: update_running_variance() has NO CALLER in ree_core, so
            # the driver must drive the EMA or _running_variance stays pinned at
            # precision_init and the commit branch never varies. Commitment is not
            # a DV here, but it selects argmin-vs-softmax inside select(), so the
            # regime must be genuine rather than frozen at init.
            if zw_prev is not None and a_prev is not None:
                pred = agent.e2.world_forward(zw_prev, a_prev)
                agent.e3.update_running_variance(zw_cur - pred.detach())

            ticks_d = agent.clock.advance()
            e1_prior = (agent._e1_tick(latent) if ticks_d["e1_tick"]
                        else torch.zeros(1, agent.config.latent.world_dim,
                                         device=agent.device))
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            if not candidates or len(candidates) < 2:
                continue

            classes = _first_action_classes(candidates)
            k = len(candidates)
            k_sizes.append(float(k))
            pool_classes.append(float(len(set(classes))))

            actions_K = _first_actions_K(candidates).to(agent.device)
            pairwise.append(float(
                agent.e2.cand_world_pairwise_dist(zw_cur, actions_K).item()))

            # DIRECT select() call: one genuine selection per tick, no cadence
            # gating, no commitment latch -> n_latched_ticks is 0 by construction.
            result = agent.e3.select(candidates, temperature=SELECT_TEMPERATURE)
            scores = result.scores.detach().cpu().numpy().astype(float).reshape(-1)
            e3_idx = int(result.selected_index)
            committed = bool(result.committed)

            s_rng = float(scores.max() - scores.min())
            s_scale = float(np.abs(scores).mean()) + 1e-12
            score_ranges.append(s_rng)
            score_ranges_norm.append(s_rng / s_scale)
            score_abs_max = max(score_abs_max, float(np.abs(scores).max()))

            # --- the three executed-selection rules -------------------------- #
            perm = rng.permutation(k)
            perm_idx = int(np.argmin(scores[perm]))     # score<->candidate broken
            unif_idx = int(rng.integers(0, k))
            if arm_id == ARM_INTACT:
                exec_idx = e3_idx
            elif arm_id == ARM_LESIONED:
                exec_idx = unif_idx
            else:
                exec_idx = perm_idx
            if exec_idx != e3_idx:
                n_override += 1

            sel_counts[classes[exec_idx]] += 1.0
            ctf_prior_counts[classes[unif_idx]] += 1.0
            ctf_perm_counts[classes[perm_idx]] += 1.0
            for c in classes:
                prior_accum[c] += 1.0 / k

            if committed:
                n_committed += 1
            else:
                n_uncommitted += 1
                # On the COMMITTED path selection is a deterministic argmin
                # (e3_selector.py:1891), so this statistic is an identity there and
                # is accumulated on the uncommitted subset ONLY.
                if e3_idx == int(np.argmin(scores)):
                    n_argmin_hit_uncommitted += 1
                taken_ranks.append(
                    float(int(np.argsort(scores).tolist().index(e3_idx))) / max(1, k - 1))

            if len(per_tick) < 200:
                per_tick.append({
                    "tick": tick, "k": k, "n_classes": len(set(classes)),
                    "e3_idx": e3_idx, "exec_idx": exec_idx,
                    "exec_class": classes[exec_idx],
                    "committed": committed,
                    "score_range": round(s_rng, 9),
                    "score_range_norm": round(s_rng / s_scale, 9),
                    "cand_world_pairwise_dist": round(pairwise[-1], 6),
                })

        act_idx = classes[exec_idx]
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, act_idx] = 1.0
        agent._last_action = action
        a_prev = action.detach()
        zw_prev = zw_cur
        _obs, _r, done, _info, obs_dict = env.step(act_idx)
        if done:
            _obs, obs_dict = env.reset()

        if (tick + 1) % 250 == 0 or tick == n_ticks - 1:
            print(f"  [eval] arc003 {arm_id} seed={seed} tick {tick + 1}/{n_ticks} "
                  f"sel={int(sel_counts.sum())}", flush=True)

    n_sel = float(sel_counts.sum())
    prior = prior_accum / max(1.0, prior_accum.sum())
    k_mean = float(np.mean(k_sizes)) if k_sizes else 0.0

    row: Dict[str, Any] = {
        "arm_id": arm_id,
        "seed": seed,
        "n_ticks": n_ticks,
        # Structurally 0: select() is called directly, so no tick can re-read a
        # previous selection's latched diagnostics (the V3-EXQ-785 ~9x defect).
        "n_latched_ticks": 0,
        "n_selections": n_sel,
        "k_mean": k_mean,
        "kl_selected_vs_prior": _kl(sel_counts, prior),
        "kl_ctf_uniform_vs_prior": _kl(ctf_prior_counts, prior),
        "kl_ctf_permuted_vs_prior": _kl(ctf_perm_counts, prior),
        "selected_class_counts": sel_counts.tolist(),
        "pool_prior": prior.tolist(),
        "pool_classes_mean": float(np.mean(pool_classes)) if pool_classes else 0.0,
        "cand_world_pairwise_dist_mean": float(np.mean(pairwise)) if pairwise else 0.0,
        "cand_world_pairwise_dist_min": float(np.min(pairwise)) if pairwise else 0.0,
        "score_range_mean": float(np.mean(score_ranges)) if score_ranges else 0.0,
        "score_range_norm_mean": (
            float(np.mean(score_ranges_norm)) if score_ranges_norm else 0.0),
        "score_abs_max": score_abs_max,
        "lesion_override_frac": (n_override / n_sel) if n_sel else 0.0,
        "committed_frac": (n_committed / n_sel) if n_sel else 0.0,
        "n_uncommitted": float(n_uncommitted),
        "argmin_hit_frac_uncommitted": (
            (n_argmin_hit_uncommitted / n_uncommitted) if n_uncommitted else 0.0),
        "chance_argmin_rate": (1.0 / k_mean) if k_mean else 0.0,
        "taken_rank_frac_mean": float(np.mean(taken_ranks)) if taken_ranks else 0.0,
        "per_tick_sample": per_tick,
    }
    row.update(train_stats)
    return row


# ------------------------------------------------------------------ #
# Gate                                                               #
# ------------------------------------------------------------------ #
def _specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="cand_pool_divergent",
            description=(
                "mean cand_world_pairwise_dist across the candidate pool (SD-056, "
                "e2_fast.py). The GAP-A guard: scoring a pool of near-identical "
                "candidates makes the score gradient trivially flat and would "
                "MANUFACTURE a FAIL. V3-EXQ-571 measured exactly this collapse at "
                "0.0000 across K=8 candidates differing only in first action"),
            control="SD-056 action-contrastive P0 warmup + support-preserving CEM",
            threshold=CAND_DIST_FLOOR),
        PreconditionSpec(
            name="pool_first_action_class_support",
            description=(
                "mean distinct FIRST-ACTION classes per candidate pool. If the pool "
                "is single-class the generation prior is a point mass and the KL DV "
                "is IDENTICALLY 0 for every arm INCLUDING A0 -- the contrast would "
                "read as a clean FAIL while measuring nothing"),
            control="support_preserving_min_first_action_classes=2 on the proposer",
            threshold=POOL_CLASS_FLOOR),
        PreconditionSpec(
            name="score_magnitude_bounded",
            description=(
                "max |E3 per-candidate score|. CEILING. The V3-EXQ-643 float32 "
                "cancellation regime (~1e32 raw scores) makes every downstream "
                "statistic meaningless"),
            control="e2_rollout_output_norm_clamp at ratio 4.0",
            threshold=SCORE_ABS_CEILING, direction="upper"),
        PreconditionSpec(
            name="selection_count",
            description="genuine E3 selections banked in this cell",
            control="direct e3.select() per tick -- every tick is one selection",
            threshold=MIN_SELECTIONS),
        PreconditionSpec(
            name="lesion_took",
            description=(
                "fraction of selections where the EXECUTED candidate differs from "
                "E3's own pick. Verifies the lesion was APPLIED and not merely "
                "requested -- the V3-EXQ-689d defect, where a matched-noise control "
                "was bit-identical to its sibling control and 'strict above BOTH' "
                "degraded silently to 'strict above one'"),
            control="a content-free draw over K candidates re-picks E3 at rate 1/K",
            threshold=LESION_OVERRIDE_FLOOR,
            applies_to=lambda ctx: ctx["arm_id"] in NULL_ARMS,
            applies_note=(
                "not meaningful for A0_INTACT: its executed index IS E3's pick by "
                "definition, so the override rate is 0 BY CONSTRUCTION and "
                "asserting it would make A0 structurally un-passable (the "
                "V3-EXQ-785 whole-run gate defect)"),
            # Design-time proof: A0 can never exceed 0, so the applies_to scoping
            # above is REQUIRED for the run to be launchable at all.
            structural_max=lambda ctx: 0.0 if ctx["arm_id"] == ARM_INTACT else None),
    ]


def _measured(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "cand_pool_divergent": float(np.mean(
            [r["cand_world_pairwise_dist_mean"] for r in rows])),
        "pool_first_action_class_support": float(np.mean(
            [r["pool_classes_mean"] for r in rows])),
        # WORST cell, not the mean: `met` is a claim about every cell, and a mean
        # can hide one out-of-band cell that recomputes as MET.
        "score_magnitude_bounded": float(np.max([r["score_abs_max"] for r in rows])),
        "selection_count": float(np.min([r["n_selections"] for r in rows])),
        "lesion_took": float(np.min([r["lesion_override_frac"] for r in rows])),
    }


def run_experiment(dry_run: bool):
    t0 = time.perf_counter()
    n_train = DRY_TRAIN_EPISODES if dry_run else TRAIN_EPISODES
    steps = DRY_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE
    n_ticks = DRY_EVAL_TICKS if dry_run else EVAL_TICKS
    seeds = SEEDS[:2] if dry_run else SEEDS

    specs = _specs()
    arm_ctxs = [{"arm_id": a} for a in ARMS]
    # Refuses the run BEFORE compute is spent if any gate is structurally
    # unsatisfiable for an arm from its pre-registered config (the V3-EXQ-785
    # check). Here it proves that `lesion_took` MUST be scoped out of A0.
    assert_no_structurally_unsatisfiable_gate(specs, arm_ctxs)

    arm_results: List[Dict[str, Any]] = []
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}

    for arm_id in ARMS:
        for seed in seeds:
            probe_slice = _build(seed)[4]
            probe_slice = dict(probe_slice, arm_id=arm_id,
                               train_episodes=n_train, steps=steps,
                               eval_ticks=n_ticks,
                               select_temperature=SELECT_TEMPERATURE)
            with arm_cell(
                seed,
                config_slice=probe_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                # MINT-AS-YOU-GO: excludes the driver from the hash so a future,
                # different-driver consumer can reuse these cells.
                include_driver_script_in_hash=False,
            ) as cell:
                rng = np.random.default_rng(20_000 + 97 * seed)
                row = _collect_cell(arm_id, seed, n_train, steps, n_ticks, rng)
                cell.stamp(row)
            by_arm[arm_id].append(row)
            arm_results.append(row)
            print(f"verdict: {'PASS' if row['n_selections'] > 0 else 'FAIL'}",
                  flush=True)

    # ---------------- per-arm gate ---------------- #
    arm_gates = [
        evaluate_arm_gate(arm_id, {"arm_id": arm_id}, specs, _measured(by_arm[arm_id]))
        for arm_id in ARMS
    ]
    agg = aggregate_arm_gates(arm_gates)
    green = {g["arm"] for g in arm_gates if g["gate_green"]}

    # ---------------- criteria ---------------- #
    a0_rows = by_arm[ARM_INTACT]
    kl_a0 = float(np.mean([r["kl_selected_vs_prior"] for r in a0_rows]))
    kl_null = {a: float(np.mean([r["kl_selected_vs_prior"] for r in by_arm[a]]))
               for a in NULL_ARMS}
    worst_null = max(kl_null.values())

    # Effect-size floor: scaled on the SD of the per-seed DELTA, plus an absolute
    # floor (per feedback_effect_size_pass_gate_margin).
    per_seed_deltas: List[float] = []
    for i, _seed in enumerate(seeds):
        null_i = max(by_arm[a][i]["kl_selected_vs_prior"] for a in NULL_ARMS)
        per_seed_deltas.append(a0_rows[i]["kl_selected_vs_prior"] - null_i)
    delta_sd = float(np.std(per_seed_deltas)) if len(per_seed_deltas) >= 2 else 0.0
    kl_margin = max(KL_ABS_FLOOR, KL_GAP_SD_MULTIPLIER * delta_sd)

    c1_scorable = ({ARM_INTACT, *NULL_ARMS} <= green)
    c1_pass = bool(c1_scorable and (kl_a0 - worst_null) > kl_margin)

    score_range_norm = float(np.mean([r["score_range_norm_mean"] for r in a0_rows]))
    c2_scorable = ARM_INTACT in green
    c2_pass = bool(c2_scorable and score_range_norm > SCORE_RANGE_FLOOR)

    n_uncommitted = float(np.sum([r["n_uncommitted"] for r in a0_rows]))
    argmin_hit = float(np.mean([r["argmin_hit_frac_uncommitted"] for r in a0_rows]))
    chance = float(np.mean([r["chance_argmin_rate"] for r in a0_rows]))
    c3_scorable = bool(c2_scorable and n_uncommitted >= C3_MIN_UNCOMMITTED)
    c3_pass = bool(c3_scorable and argmin_hit > chance + C3_MARGIN)

    # Within-tick matched counterfactual (corroboration only, never a criterion).
    ctf_delta = float(np.mean([
        r["kl_selected_vs_prior"] - max(r["kl_ctf_uniform_vs_prior"],
                                        r["kl_ctf_permuted_vs_prior"])
        for r in a0_rows]))

    # ---------------- routing ---------------- #
    if not agg["any_green"] or not c2_scorable:
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "ARC-003 leg-A gate RED: " + agg["degeneracy_reason"]
            + ". A collapsed candidate pool, a single-class pool, or an unbounded "
              "score makes the KL contrast uninterpretable, so this run is NOT "
              "scored and is NOT evidence against ARC-003.")
    else:
        non_degenerate, degeneracy_reason = True, ""
        if c1_pass and c2_pass:
            outcome, evidence_direction = "PASS", "supports"
            label = "e3_selects_beyond_generation_prior"
        elif c2_pass and not c1_pass:
            outcome, evidence_direction = "FAIL", "weakens"
            label = "e3_score_gradient_without_selection_authority"
        elif c1_pass and not c2_pass:
            outcome, evidence_direction = "FAIL", "mixed"
            label = "selection_departs_prior_without_score_gradient"
        else:
            outcome, evidence_direction = "FAIL", "weakens"
            label = "e3_pass_through_not_selector"
        if not c1_scorable:
            # The contrast could not be formed. Do NOT let that read as a
            # refutation: an unusable NULL arm removes the contrast, it does not
            # supply evidence against the claim.
            outcome, evidence_direction = "FAIL", "non_contributory"
            label = "selection_contrast_unscorable_null_arm_red"

    criteria = [
        {"name": "C1_selection_departs_generation_prior", "load_bearing": True,
         "passed": c1_pass, "scorable": c1_scorable,
         "measured_kl_a0": kl_a0, "measured_kl_worst_null": worst_null,
         "measured_delta": kl_a0 - worst_null, "threshold_delta": kl_margin,
         "kl_by_null_arm": kl_null, "per_seed_deltas": per_seed_deltas,
         "null_note": (
             "a null here means the selected-candidate distribution is "
             "statistically indistinguishable from the proposer's own generation "
             "prior -- E3 is a PASS-THROUGH, not a selector. That is the "
             "decision-flipping negative: MECH-060/061/062 and the tri-loop gating "
             "claims lose their premise. It is NOT a statement about ARC-003 leg B "
             "(commitment), which is untested here")},
        {"name": "C2_e3_score_gradient_non_flat", "load_bearing": True,
         "passed": c2_pass, "scorable": c2_scorable,
         "measured": score_range_norm, "threshold": SCORE_RANGE_FLOOR,
         "null_note": (
             "measured as the cross-candidate score RANGE normalised by score "
             "scale, never a mean-abs/magnitude proxy: a uniform per-candidate "
             "offset has large magnitude and ~0 range, and RANGE is the statistic "
             "the argmin routes on (V3-EXQ-643). A flat gradient means E3 assigns "
             "no differential value to the candidates it is given")},
        {"name": "C3_taken_candidate_predictable_above_chance", "load_bearing": False,
         "passed": c3_pass, "scorable": c3_scorable,
         "measured": argmin_hit, "threshold": chance + C3_MARGIN,
         "chance_rate": chance, "n_uncommitted": n_uncommitted,
         "null_note": (
             "NOT load-bearing, and DEGENERATE ON THE COMMITTED PATH: when "
             "result.committed is true, selection IS a deterministic argmin over "
             "the score (e3_selector.py:1891), so the statistic is an arithmetic "
             "IDENTITY and carries no information. Scored on the UNCOMMITTED "
             "(softmax-sampled) subset only, with a minimum-n floor. Its null -- a "
             "chance-level rate -- means the score gradient is small relative to "
             "the selection temperature, i.e. EFFECTIVE rather than nominal "
             "authority is absent")},
    ]
    # Every criterion is READ OFF A0 (C1 is a contrast whose reference is A0), so
    # A0 is the owning arm for all three. The null arms' gate status enters through
    # `extra` via c1_scorable, which requires ALL THREE arms green -- a contrast
    # cannot be formed from a red null arm.
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {ARM_INTACT: ["C1_selection_departs_generation_prior",
                      "C2_e3_score_gradient_non_flat",
                      "C3_taken_candidate_predictable_above_chance"]},
        agg,
        extra={"C1_selection_departs_generation_prior": c1_scorable,
               "C3_taken_candidate_predictable_above_chance": c3_scorable})

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "arms": ARMS,
        "train_episodes": n_train,
        "steps_per_episode": steps,
        "eval_ticks": n_ticks,
        "alpha_world": ALPHA_WORLD,
        "select_temperature": SELECT_TEMPERATURE,
        "e2_action_contrastive_weight": E2_CONTRASTIVE_WEIGHT,
        "lr": LR,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True},
        "thresholds": {
            "CAND_DIST_FLOOR": CAND_DIST_FLOOR,
            "POOL_CLASS_FLOOR": POOL_CLASS_FLOOR,
            "SCORE_ABS_CEILING": SCORE_ABS_CEILING,
            "MIN_SELECTIONS": MIN_SELECTIONS,
            "LESION_OVERRIDE_FLOOR": LESION_OVERRIDE_FLOOR,
            "KL_ABS_FLOOR": KL_ABS_FLOOR,
            "KL_GAP_SD_MULTIPLIER": KL_GAP_SD_MULTIPLIER,
            "SCORE_RANGE_FLOOR": SCORE_RANGE_FLOOR,
            "C3_MIN_UNCOMMITTED": C3_MIN_UNCOMMITTED,
            "C3_MARGIN": C3_MARGIN,
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "config": full_config,
        "seeds": seeds,
        "arm_results": arm_results,
        "per_arm_gate": agg["per_arm_gate"],
        "metrics": {
            "kl_a0_selected_vs_prior": kl_a0,
            "kl_worst_null": worst_null,
            "kl_delta_a0_minus_worst_null": kl_a0 - worst_null,
            "kl_margin_applied": kl_margin,
            "within_tick_ctf_delta_a0": ctf_delta,
            "score_range_norm_mean_a0": score_range_norm,
            "argmin_hit_frac_uncommitted_a0": argmin_hit,
            "chance_argmin_rate_a0": chance,
            "committed_frac_a0": float(np.mean([r["committed_frac"] for r in a0_rows])),
            "cand_world_pairwise_dist_mean_a0": float(np.mean(
                [r["cand_world_pairwise_dist_mean"] for r in a0_rows])),
        },
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": agg["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
            "leg_scope_note": (
                "ARC-003 is CONJUNCTIVE. This run tests LEG A (E3 SELECTS) ONLY, "
                "at the candidate-scoring level, with NO sustained multi-step "
                "action commitment anywhere in the design. LEG B (E3 COMMITS) is "
                "BLOCKED on the BG / action-commitment layer and is registered "
                "separately as EXP-0395 (blocked_substrate). NO result here "
                "licenses any reading of leg B, in either direction."),
            "commitment_free_note": (
                "The driver calls agent.e3.select() DIRECTLY rather than "
                "agent.select_action(), because select_action interposes the E3 "
                "cadence (heartbeat.e3_steps_per_tick default 10), the commitment "
                "latch and the BetaGate -- the very machinery leg B is blocked on. "
                "Consequence: every tick is one genuine independent selection, so "
                "the V3-EXQ-785 ~9x diagnostics pseudo-replication defect is "
                "structurally impossible here (n_latched_ticks = 0 by "
                "construction, emitted as 0, not merely guarded)."),
            "score_scope_note": (
                "select() is called on its CORE scoring path: score_bias, "
                "score_bias_channels and channel_route_bias are NOT injected (they "
                "are what agent.select_action would otherwise assemble). So this "
                "tests E3's OWN scoring authority (F / M / Phi_R). It tests NOTHING "
                "about the modulatory channels' authority, which is claimed under "
                "MECH-314 / MECH-341 / MECH-320 and tested separately."),
            "dv_symmetry_note": (
                "DV = KL(executed first-action-class distribution || per-tick pool "
                "prior). Symmetry group: permutations of candidates within a class, "
                "relabellings of candidate slots preserving per-tick class counts, "
                "and any score transform preserving the within-tick argmin "
                "(including a uniform additive constant -- the V3-EXQ-604c "
                "broadcast-scalar trap). A0_INTACT's manipulation is NOT invariant: "
                "E3's score is a state-dependent function of predicted candidate "
                "outcomes and its cross-candidate RANGE is measured and gated. "
                "A1_E3_LESIONED and ARM_NOISE ARE invariant BY CONSTRUCTION and are "
                "DECLARED REFERENCE ARMS: a content-free selector draws uniformly "
                "over candidates and so reproduces the prior. Their KL is the "
                "sampling floor at this n, fixed before the run; it is never "
                "reported as a measured effect and never routes a verdict alone."),
            "two_null_arms_note": (
                "Both null arms estimate the SAME floor and both are ~0 by "
                "construction. Their distinct value is WHAT EACH HOLDS FIXED: A1 "
                "holds the selection rule content-free while the score distribution "
                "is whatever it is; ARM_NOISE permutes E3's REAL scores across "
                "candidates, holding the score DISTRIBUTION exactly fixed while "
                "destroying its content. If A0's KL were an artifact of the "
                "score-magnitude regime rather than of score content, ARM_NOISE "
                "would reproduce it and A1 might not. Agreement strengthens the "
                "floor; disagreement is diagnostic."),
            "within_tick_counterfactual_note": (
                "kl_ctf_uniform_vs_prior and kl_ctf_permuted_vs_prior are the same "
                "two nulls computed WITHIN each tick from the SAME candidate pool "
                "the executed selection saw, in every arm. They carry zero "
                "trajectory-divergence between the contrasted quantities and are "
                "therefore strictly better powered than the across-arm contrast. "
                "They are reported as CORROBORATION and are deliberately NOT the "
                "pre-registered criterion, which stays the across-arm contrast the "
                "proposal specified."),
            "cross_arm_contrast_note": (
                "C1 is a CROSS-ARM CONTRAST, so a red NULL arm removes the contrast "
                "it participates in and C1 becomes unscorable "
                "(selection_contrast_unscorable_null_arm_red, non_contributory). "
                "That is the arithmetic of a difference, NOT the V3-EXQ-785 "
                "vacating defect, which destroyed an arm's own self-contained "
                "result. C2 and C3 are A0-INTERNAL and still score when a null arm "
                "is red."),
            "gap_a_note": (
                "The GAP-A guard is P1 (cand_world_pairwise_dist above floor). Its "
                "companion P2 -- mean distinct first-action classes per pool -- is "
                "the one that is easy to miss: on a single-class pool the "
                "generation prior is a point mass and the KL DV is IDENTICALLY 0 "
                "for EVERY arm including A0, so the contrast would read as a clean "
                "FAIL while measuring nothing at all."),
        },
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
        "custom_information": {
            "proposal_id": "EXP-0394",
            "gov_reuse_1_check": (
                "Decisive readout: KL(executed first-action-class distribution || "
                "per-tick candidate-pool generation prior) under an E3-selection "
                "lesion contrast. ARC-003 has ZERO recorded experiments of any kind "
                "(exp_conf 0.0, genuine_exp_count 0), so no manifest carries this "
                "readout or its inputs and there is nothing to reanalyse. Not "
                "recoverable -> must run."),
            "blocked_sibling": (
                "EXP-0395 (ARC-003 leg B, commitment) is registered "
                "blocked_substrate; release condition is ARC-107 BG-selector "
                "constitution landed AND the commit/release-duration levers "
                "(V3-EXQ-460j / 485i / 654h) returned with a non-degenerate "
                "commitment signal."),
            "per_arm_kl": {a: kl_null.get(a, kl_a0) for a in ARMS},
        },
    }

    stamp_recording_core(manifest, config=full_config, seeds=seeds,
                         script_path=Path(__file__), started_at=t0)
    return manifest


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("V3-EXQ-804: ARC-003 leg A -- does E3 select? (commitment-free)",
          flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    t_start = time.perf_counter()
    manifest = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run,
        config=manifest.get("config"), seeds=manifest.get("seeds"),
        script_path=Path(__file__), started_at=t_start,
    )

    m = manifest["metrics"]
    print(f"  outcome={manifest['outcome']} "
          f"direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    print(f"  KL_A0={m['kl_a0_selected_vs_prior']:.5f} "
          f"KL_worst_null={m['kl_worst_null']:.5f} "
          f"delta={m['kl_delta_a0_minus_worst_null']:.5f} "
          f"margin={m['kl_margin_applied']:.5f}", flush=True)
    print(f"  score_range_norm={m['score_range_norm_mean_a0']:.6g} "
          f"argmin_hit={m['argmin_hit_frac_uncommitted_a0']:.4f} "
          f"(chance {m['chance_argmin_rate_a0']:.4f}) "
          f"committed_frac={m['committed_frac_a0']:.3f}", flush=True)
    print(f"  pool_dist={m['cand_world_pairwise_dist_mean_a0']:.6g} "
          f"within_tick_ctf_delta={m['within_tick_ctf_delta_a0']:.5f}", flush=True)
    _pag = manifest["per_arm_gate"]
    for g in list(_pag.get("green", [])) + list(_pag.get("red", [])):
        print(f"  [{g.get('arm')}] gate="
              f"{'GREEN' if g.get('gate_green') else 'RED'} "
              f"failed={g.get('failed_preconditions')}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
