#!/opt/local/bin/python3
"""V3-EXQ-485m: SD-033b trained-OFC-head BEHAVIOURAL DEVALUATION arm (MECH-263
sig-a) on the DECOUPLED OFC devaluation head -- ON the MECH-448 (ARC-107)
rank-preserving F->eligibility DEMOTION-ENABLED E3 selector WITH the MECH-449
Go/No-Go ELIGIBILITY CONSTITUTION ENGAGED.

================ 485m DECOUPLE (the structural fix; supersedes 485l) ================
SUPERSEDES V3-EXQ-485l (run_id v3_exq_485l_sd033b_devaluation_nogo_behavioural_
20260622T063547Z_v3), confirmed FAIL/non_contributory/substrate_not_ready_requeue
(failure_autopsy_V3-EXQ-485l_2026-06-22, re-derive brake FIRED -> implement-substrate).
485e->485l established that the SINGLE shared OFCAnalog.state_bias_head under the
+/-ofc_bias_scale clamp has NO feasible gain band: 485k gain 4.0 SATURATED the
devalued range to 0.0; 485l gain 1.5 UNDERSHOT the 0.05 readout floor (0.031) --
because the SAME head + clamp must also carry the C2 high-threat discrimination
range, so devalued magnitude is traded against C2. The bias VECTOR inverted cleanly
(l2 1.83, cosine -0.716, C1b 2/3) -- DIRECTION right, MAGNITUDE clamp-starved -> the
MECH-449 No-Go viability trigger engaged only 1/3 (< 2/3 gate). A plain 485m
gain-tweak on the shared clamped head was REFUSED (no feasible band).

THE BUILD this retest runs on (ree-v3 main 758956f, 2026-06-22): the devaluation
re-ranking is DECOUPLED into a SECOND OFC output head. OFCConfig.use_devaluation_head
builds devaluation_bias_head, sharing the [state_code, candidate] input with
state_bias_head but clamped to +/-ofc_devaluation_bias_scale (default 2.0,
INDEPENDENT of ofc_bias_scale=0.5). This experiment:
  * drives + reads the DEVALUED re-ranking (C1/C1b + the No-Go viability) from
    agent.ofc.compute_devaluation_bias() / agent.ofc.devaluation_bias_head_parameters()
    (the new head, larger clamp);
  * keeps the C2 high-threat DISCRIMINATION on agent.ofc.compute_bias() /
    state_bias_head (the +/-0.5 clamp UNTOUCHED -- magnitude no longer traded);
  * re-states the per-seed devalued-range readiness gate (b) on
    compute_devaluation_bias (the SAME statistic the C1/C1b DVs route on), so the
    larger-clamp head clears the 0.05 floor without saturating;
  * clears the re-derive brake that 485l fired: the named upstream substrate (the
    OFC devaluation-head clamp decouple/rescale) is now BUILT, so this re-test is
    meaningful (it is NOT another letter circling the shared-clamp ceiling -- it is
    the decoupled-head structural redesign on built substrate).
ofc_devaluation_bias_scale is calibratable (default 2.0); a saturate (range -> 0) or
undershoot (range < 0.05) at the chosen scale self-routes substrate_not_ready_requeue
(NEVER a weakens), exactly as the readiness floor does.

NOTE on the 485l framing below: the "IN-BAND re-rank gain to defeat the +/-0.5 clamp"
discussion (FIX 2) was the LAST shared-clamp workaround; on the decoupled head the
devalued re-ranking has its OWN +/-2.0 clamp, so the in-band-gain pressure is gone --
the lowered DEVALUED_RERANK_GAIN/WEIGHT are kept only as conservative driver defaults,
not as a clamp-saturation workaround. Everything else (MECH-448 demotion, MECH-449
No-Go viability mapping, C2 protection, per-seed readiness, the 3-arm dissociation) is
inherited verbatim from 485l.
====================================================================================

Inherited 485l framing (the MECH-448 demotion + MECH-449 constitution machinery):
ON the MECH-448 (ARC-107) rank-preserving F->eligibility DEMOTION-ENABLED E3
selector, WITH THE MECH-449 Go/No-Go ELIGIBILITY CONSTITUTION ENGAGED so the active
No-Go WITHDRAWAL that behavioural outcome-devaluation requires can express.

commitment_closure:GAP-8 evidence-grade follow-on. SUPERSEDES V3-EXQ-485k (run_id
v3_exq_485k_sd033b_demotion_devalued_rerank_behavioural_20260621T192541Z_v3), a
confirmed FAIL/non_contributory (failure_autopsy_V3-EXQ-485k_2026-06-21): both DVs
vacuous. 485k's FIX-1 re-ranking devaluation driver (DEVALUED_RERANK_GAIN=4.0,
weight 0.5) OVERSHOT the +/-0.5 OFC bias clamp -> uniform saturation -> readable
devalued range = exactly 0.0 on 2/3 seeds (the bias vector still INVERTED, cosine
-0.71/-0.57, but the committed-argmax TV was blind to clamp-saturated re-ranking);
and it REGRESSED 485j's C2 discrimination conversion (1.0/0.0/1.0 -> 0/0/0) by
swamping the shared OFC head. Root cause per the autopsy: demotion-alone (MECH-448
eligibility ACCESS) CANNOT express the active No-Go WITHDRAWAL that behavioural
outcome-devaluation requires -- the rank-preserving F->eligibility envelope provides
access only, no oppositely-signed No-Go suppression. The discrimination signature
(sig-b) is demotion-tractable (it CONVERTED in 485j) but the devaluation signature
(sig-a) is blocked on the MECH-449 Go/No-Go constitution, which was BUILT 2026-06-21
(689g PASSED, MECH-449 promoted candidate->provisional 2026-06-22).

THE FIVE 485l FIXES (off the 485k harness; the demotion lever is KEPT):

  (1) MECH-449 Go/No-Go CONSTITUTION ENGAGED (use_go_nogo_constitution=True, armed as
      a matched-stack CONSTANT on both demotion-on arms). The OFC outcome-devaluation
      is rendered as the biologically-faithful BG active No-Go WITHDRAWAL: at the
      devalued state the trained head's per-candidate devalued valuation (bias_low) is
      mapped to a per-candidate VIABILITY No-Go signal -- the candidate(s) the devalued
      head most DISFAVOURS (the previously-preferred, now-devalued action) drop below
      gng_viability_floor and are actively WITHDRAWN from the eligible set (injected via
      agent.set_injected_go_nogo_signals / passed as go_nogo_signals to e3.select). This
      is what rank-preserving demotion structurally cannot do (Rudebeck & Murray 2014;
      Mink 1996 focal-go + surround-no-go; Maia & Frank 2011). NON-VACUOUS: the No-Go is
      DERIVED from the trained valuation -- a frozen/untrained head (ARM_0) yields a flat
      bias_low -> flat viability -> NO candidate below floor -> NO withdrawal -> silence;
      only a genuinely-trained devaluation valuation withdraws. The withdrawal is
      injected ONLY at the devaluation comparison (P2-a), NOT at the high-threat baseline
      or the C2 discrimination drives (so C2 stays demotion-alone, as it converted in
      485j -- FIX 3).

  (2) IN-BAND re-rank devaluation gain (DEVALUED_RERANK_GAIN 4.0 -> 1.5,
      DEVALUED_RERANK_WEIGHT 0.5 -> 0.3). 485k's overstrong driver pushed the head into
      uniform +/-0.5 clamp saturation (readable range exactly 0.0). The lower gain keeps
      the devalued bias DIFFERENTIATED WITHIN the band so it carries genuine
      cross-candidate range (the readiness floor catches a re-saturated run as
      substrate_not_ready, NEVER a weakens, because uniform saturation -> range 0).

  (3) C2 DISCRIMINATION PROTECTED. The C2 task-role discrimination drives (P2-b) run with
      go_nogo_signals=None (the constitution gate is config-on but signal-inert there),
      so C2 reverts to the demotion-alone path that converted in 485j. The re-rank
      driver's in-band gain (FIX 2) also reduces the driver-coupling that destabilised
      the shared OFC head's high-threat spread (the 485k C2 regression cause).

  (4) BIAS-VECTOR l2/cosine INVERSION promoted to a SCORED DV (C1b). The high-vs-devalued
      OFC bias-VECTOR change (||bias_high - bias_low|| L2 shift + cosine) was a secondary
      readout in 485k; here it is a LOAD-BEARING acceptance criterion (l2_shift >
      BIAS_L2_FLOOR AND cosine < BIAS_COSINE_CEIL on >= MIN_PASS_SEEDS), so the valuation
      re-ranking is scored directly even where the committed-argmax TV is thin.

  (5) use_f_eligibility_adaptive_floor ADOPTED on the OFC channel (mean-relative,
      scale-invariant), REPLACING the 485i/485j bespoke per-(arm,seed)
      _calibrate_envelope_floor dance. 485e->485k showed the fixed-0.30 floor all-admits
      on the OFC bank (485i excluded_count==0) and needed a bespoke per-seed
      recalibration (485j) to engage. The 689e channel-adaptive floor collapses that
      ~5-channel hand-floor dance into ONE global knob (excludes a non-empty tail by
      construction at mean_factor>=1.0 on a non-flat F field), so the demotion envelope
      genuinely EXCLUDES (excluded_count > 0 = the MECH-448 non-degeneracy gate) with no
      per-seed calibration.

PER-SEED READINESS PRECONDITION FIX (the 485k aggregate-max bug; the meta-version of
the V3-EXQ-642 / V3-EXQ-643 same-statistic miss): 485k's interpretation.preconditions
PANEL reported max-over-seeds (max_test_bias_range_devalued=0.107 from seed 0) and read
met=True, MASKING a per-seed 2-of-3 collapse (only seed 0 cleared the devalued floor;
ready_seeds=1). The indexer recomputes met from measured vs threshold, so a MAX-over-
seeds measured spuriously reads met. 485l's preconditions report the per-seed PASS COUNT
(measured = number of seeds clearing the floor; threshold = MIN_PASS_SEEDS), so the
indexer recompute requires >= 2/3 seeds, NOT aggregate-max.

KEPT FROM 485j/485k (do NOT change): ofc_bias_scale=0.5; SD-056 e2_action_contrastive
  armed in P0; phased training (P0 encoder warmup with SD-056 contrastive -> P1 head
  training on the frozen-encoder state_code -> P2 BEHAVIOURAL eval); the OFC-isolated
  SD-054 bipartite reef/forage env (ofc_harm_dim>0); LR_OFC_BIAS 2e-3 / P1 120 ep; the
  MECH-448 demotion-on config (use_f_eligibility_demotion / dn_sigma=0.0); the
  excluded_count>0 MECH-448 non-degeneracy HARD gate; and the 3-arm dissociation.

THREE ARMS (clean dissociation; conversion attributable to the CONJUNCTION
trained-head AND demotion AND active-No-Go -- not any alone):
  ARM_0_frozen_demotion_on   : frozen zeroed OFC head + demotion ON + constitution ON.
                               Silence control: the frozen head yields a flat devalued
                               bias -> flat viability -> NO No-Go withdrawal -> silent.
  ARM_1_trained_demotion_off : trained OFC head + demotion OFF (constitution config-on
                               but the f_demotion / gate block is skipped without
                               demotion). F-dominance ceiling control: F dominates the
                               argmin -> ~0 conversion.
  ARM_2_trained_demotion_on  : trained OFC head + demotion ON + constitution ON. The
                               TEST -- the trained devaluation valuation drives the active
                               No-Go WITHDRAWAL of the devalued action within the
                               F-eligible set, F removed from the final argmin.
  All three pass the OFC bias as score_bias and use the SAME settle/eval pipeline; the
  ONLY varied factors are (head trained?) x (demotion on?). The constitution flag is a
  matched constant.

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (non-vacuity precondition, on the ARM_2 test arm; reported as PER-SEED PASS
    COUNTS, NOT aggregate-max -- the 485k fix; SAME statistics the load-bearing DVs route
    on):
      (a) trained-head OFC bias cross-candidate RANGE at the HIGH-THREAT positive control
          > BIAS_RANGE_FLOOR (0.05) on >= MIN_PASS_SEEDS -- the C2 discrimination control.
      (b) trained-head OFC bias cross-candidate RANGE at the DEVALUED state >
          BIAS_RANGE_FLOOR (0.05) on >= MIN_PASS_SEEDS -- the SAME statistic the
          load-bearing C1 / C1b devaluation DVs route on; the in-band re-ranking driver
          (FIX 2) must produce a differentiated (non-flat, non-saturated) devalued bias.
      (c) BOTH OFC heads trained (state_bias_head AND the decoupled devaluation_bias_head
          weight-delta > HEAD_DELTA_MIN) on >= MIN_PASS_SEEDS -- the C1 devaluation DVs +
          the No-Go viability route on the devaluation_bias_head (485m DECOUPLE).
      (d) MECH-448 NON-DEGENERACY: the adaptive envelope actually EXCLUDED a candidate
          (f_eligibility_excluded_count > 0 AND f_eligibility_demotion_active) on
          >= MIN_PASS_SEEDS, not an all-admit no-op.
      (e) MECH-449 NON-VACUITY: the active No-Go WITHDRAWAL genuinely engaged at the
          devalued state (go_nogo_n_soft_applied > 0) on >= MIN_PASS_SEEDS -- the
          constitution is not a config-on no-op.
    Any below floor / unmet -> substrate_not_ready_requeue (non_contributory), NEVER a
    weakens.
  C1_devaluation_behavioural_shift (load-bearing): ARM_2 devaluation_selection_shift
    > DEVAL_SHIFT_MARGIN AND > ARM_1 (demotion-off ceiling) shift + DEVAL_SHIFT_MARGIN.
  C1b_devaluation_bias_vector_inversion (load-bearing, FIX 4): ARM_2
    devaluation_bias_l2_shift > BIAS_L2_FLOOR AND devaluation_bias_cosine <
    BIAS_COSINE_CEIL -- the valuation genuinely re-ranks (anti-correlated) high->devalued.
  C2_discrimination_behavioural_separation (load-bearing): ARM_2 separation_ratio
    >= SEPARATION_RATIO_MIN AND between_context_selection_tv >= BETWEEN_TV_FLOOR AND
    > ARM_1 between_tv + BETWEEN_TV_FLOOR.
  C3_silence_control (load-bearing): ARM_0 (frozen head, demotion+constitution on) shift
    < 1e-9 AND between-context TV < 1e-9.
  PASS (supports SD-033b + MECH-263) = READINESS AND C1 AND C1b AND C2 AND C3 on a majority.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports: the trained OFC outcome-value channel, routed through the
    demotion-enabled selector WITH the MECH-449 active No-Go constitution, produces
    devaluation-sensitive (active withdrawal) AND task-role-discriminative committed
    selection. SD-033b candidate -> provisional behavioural evidence (the devaluation
    half, completing the 485j discrimination conversion); contributes to closing
    behavioral_diversity_isolation:GAP-I.
  FAIL / weakens (readiness MET, DVs fail): the trained OFC channel has genuine
    cross-candidate range at BOTH the high-threat and devalued states, the demotion
    envelope genuinely excluded F-dominant candidates, AND the active No-Go genuinely
    withdrew the devalued action, yet committed selection still does not convert. Route
    /failure-autopsy (conversion-ceiling-persists-despite-go-nogo -> V4) BEFORE stamping.
    An honest result.
  FAIL / substrate_not_ready_requeue (readiness UNMET): below the 0.05 bias-range floor
    at the high-threat OR DEVALUED state on >= 2/3 seeds, head untrained, the adaptive
    envelope still admitted all (excluded_count == 0), OR the No-Go never engaged
    (go_nogo_n_soft_applied == 0) -> the behavioural test never ran. Re-queue. NOT a weakens.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_485m_sd033b_devaluation_decoupled_head_behavioural"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-485m"
SUPERSEDES = "v3_exq_485l_sd033b_devaluation_nogo_behavioural"
CLAIM_IDS: List[str] = ["SD-033b", "MECH-263"]

SELF_DIM = 8
WORLD_DIM = 32
HARM_DIM = 4
HARM_A_DIM = 4
HARM_HISTORY_LEN = 10

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

P0_EPISODES = 30
P1_EPISODES = 120                      # kept from 485h/485k: give the driver budget
STEPS_PER_EPISODE = 100
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 2e-3                     # kept from 485h/485k
BATCH_SIZE = 32
WF_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
EMA_DECAY = 0.9

# -- 485h paired threat-conditioned driver (high-threat spread kept verbatim) --
OUTCOME_COUPLE_GAIN = 4.0
ADV_MIN_THRESHOLD = 1e-4
# -- 485l FIX 2: IN-BAND re-ranking devalued driver. The SAME REINFORCE-over-candidates
#    form as 485k (inverted high-threat outcome-coupling -> favour high predicted harm at
#    the devalued state so the head DOWN-biases the previously-preferred candidate /
#    UP-biases an alternative; Rudebeck & Murray 2014; Dickinson & Balleine) but with the
#    gain + weight LOWERED so the head stays DIFFERENTIATED WITHIN the +/-0.5 OFC clamp
#    instead of overshooting into uniform saturation (485k range exactly 0.0 on 2/3
#    seeds). A re-saturated run still self-routes substrate_not_ready (range -> 0 fails
#    the floor), NEVER a weakens.
DEVALUED_RERANK_WEIGHT = 0.3   # 485k 0.5 -> 0.3 (in-band)
DEVALUED_RERANK_GAIN = 1.5     # 485k 4.0 -> 1.5 (in-band; defeat the +/-0.5 clamp overshoot)
DRIVER_SNAP_BATCH = 16

# -- 485f Fix 1 (kept): C2 discrimination head clamp (state_bias_head, +/-0.5) --
OFC_BIAS_SCALE = 0.5
# -- 485m DECOUPLE: the SEPARATE devaluation head's own clamp (independent of
#    OFC_BIAS_SCALE). Larger by intent so the in-band re-ranking gain produces a
#    supra-floor differentiated devalued range without saturating; the C2 head's
#    +/-0.5 clamp is untouched (magnitude no longer traded). Calibratable: a
#    saturate/undershoot at this scale self-routes substrate_not_ready (NEVER weakens).
OFC_DEVAL_BIAS_SCALE = 2.0

# -- 485f Fix 2 (kept): SD-056 e2_action_contrastive in P0 -> action-divergent bank --
E2_CONTRASTIVE_WEIGHT = 0.05
E2_CONTRASTIVE_TEMP = 0.1

# -- P2 behavioural-eval drive lengths (parallel to 485b/485c/485f/485g/485h/485k) --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- MECH-448 (ARC-107) demotion lever constants (matched on the demotion-on arms) --
# 485l FIX 5: use_f_eligibility_adaptive_floor (mean-relative, scale-invariant) REPLACES
# the 485i/485j bespoke per-(arm,seed) _calibrate_envelope_floor dance. The fixed floor
# below is the substrate-default fallback only -- the adaptive floor overrides it.
F_ELIG_ENVELOPE_FLOOR = 0.30   # absolute DN-share floor (substrate default / fallback)
F_ELIG_DN_SIGMA = 0.0          # DN semi-saturation / global tightness (KEEP 0.0)
F_ELIG_ADAPTIVE_MEAN_FACTOR = 1.0  # "above-average share" threshold multiple (689e knob)

# -- MECH-449 (ARC-107) Go/No-Go constitution constants (matched on both arms) --
# The devalued OFC valuation is mapped to a per-candidate VIABILITY No-Go: the
# most-devalued (highest devalued bias) candidate -> viability ~0 -> No-Go (withdrawn)
# when viability < GNG_VIABILITY_FLOOR. Fail-open keeps >= GNG_PROTECT_MIN_ELIGIBLE.
GNG_VIABILITY_FLOOR = 0.1
GNG_PROTECT_MIN_ELIGIBLE = 1

# -- Deterministic committed readout: force the argmin-within-eligible path so each
#    DV is the genuine committed selection of the demotion-enabled selector (not a
#    multinomial sample). Eval-time only; training keeps commitment_threshold=0.5. --
EVAL_COMMIT_THRESHOLD = 1.0e6

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3
BIAS_RANGE_FLOOR = 0.05      # == DEVAL_SHIFT_MARGIN (same-statistic non-vacuity gate)
DEVAL_SHIFT_MARGIN = 0.05
BIAS_L2_FLOOR = 0.1          # C1b: minimum high-vs-devalued OFC bias-vector L2 shift
BIAS_COSINE_CEIL = 0.0       # C1b: devalued bias must be anti-correlated (genuine inversion)
SEPARATION_RATIO_MIN = 3.0
BETWEEN_TV_FLOOR = 0.05
FROZEN_SILENCE_EPS = 1e-9
MIN_PASS_SEEDS = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _onehot_vec(idx: int, k: int) -> torch.Tensor:
    """[K] one-hot of the committed candidate index (the deterministic committed
    selection distribution under the demotion+constitution selector)."""
    v = torch.zeros(k)
    if 0 <= idx < k:
        v[idx] = 1.0
    return v


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two selection distributions."""
    return float(0.5 * (p - q).abs().sum().item())


def _bias_range(agent: REEAgent, bank: torch.Tensor) -> float:
    """Cross-candidate range (max - min) of the C2 OFC bias (state_bias_head) over
    the bank -- the high-threat / discrimination readiness statistic."""
    bias = agent.ofc.compute_bias(bank).detach()
    if bias.numel() < 2:
        return 0.0
    return float((bias.max() - bias.min()).item())


def _deval_bias_range(agent: REEAgent, bank: torch.Tensor) -> float:
    """485m DECOUPLE: cross-candidate range of the DEVALUATION bias
    (devaluation_bias_head, +/-OFC_DEVAL_BIAS_SCALE) over the bank -- the SAME
    statistic the load-bearing C1/C1b devaluation DVs + the No-Go viability route on.
    The decoupled head's larger clamp lets this clear the 0.05 floor without
    saturating; a saturate (-> 0) or undershoot (< floor) self-routes
    substrate_not_ready, NEVER a weakens."""
    bias = agent.ofc.compute_devaluation_bias(bank).detach()
    if bias.numel() < 2:
        return 0.0
    return float((bias.max() - bias.min()).item())


def _bank_zworld_spread(bank: torch.Tensor) -> float:
    """Mean pairwise L2 across the K candidate-bank z_world summaries (diagnostic)."""
    if bank is None or bank.shape[0] < 2:
        return 0.0
    d = torch.cdist(bank, bank)
    K = bank.shape[0]
    eye = torch.eye(K, dtype=torch.bool, device=bank.device)
    return float(d[~eye].mean().item())


def _build_viability_nogo(bias_low: torch.Tensor) -> Optional[torch.Tensor]:
    """485l FIX 1: map the trained OFC DEVALUED valuation to a per-candidate VIABILITY
    No-Go signal for the MECH-449 Go/No-Go constitution.

    Under REE lower-is-better-favoured (the driver uses log_softmax(-bias)), a HIGHER
    devalued bias = MORE disfavoured = the previously-preferred, now-devalued action. We
    min-max normalise the devalued bias to [0, 1] (the most-devalued candidate -> 1.0)
    and return viability = 1 - normalised, so the most-devalued candidate gets viability
    ~0 < gng_viability_floor and is actively WITHDRAWN (the BG indirect / No-Go pathway).

    NON-VACUOUS: the signal is DERIVED from the trained valuation. A flat bias_low
    (frozen / untrained head) has ~0 cross-candidate range -> returns None -> the gate is
    passed no viability axis -> no No-Go -> the frozen arm stays silent. Only a genuinely
    differentiated devalued valuation withdraws.
    """
    bl = bias_low.detach().reshape(-1)
    if bl.numel() < 2:
        return None
    rng = float((bl.max() - bl.min()).item())
    if rng < 1e-6:
        return None  # flat -- no differentiated devaluation -> no withdrawal
    bln = (bl - bl.min()) / (bl.max() - bl.min())
    return (1.0 - bln).detach()


def _settle_state_code(
    agent: REEAgent,
    z_world_rep: torch.Tensor,
    z_harm: Optional[torch.Tensor],
    ticks: int,
    reset: bool,
) -> None:
    """Drive the OFC state_code toward a regime via gated EMA updates. SHARED by the
    P1 driver and the P2 eval so the trained and tested state_codes are constructed
    identically (kept from 485h/485k)."""
    if reset:
        agent.ofc.reset()
    for _ in range(ticks):
        agent.ofc.update(z_world=z_world_rep, z_harm=z_harm, gate=1.0)


def _make_agent(seed: int, train_head: bool, demotion: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # OFC isolated: no gated_policy / dACC / lateral_pfc bias channels.
        use_gated_policy=False,
        use_dacc=False,
        dacc_weight=0.0,
        use_lateral_pfc_analog=False,
        # SP-CEM main-path for first-action candidate diversity.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # 485f Fix 2 (kept): SD-056 e2 action-conditional contrastive.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_temperature=E2_CONTRASTIVE_TEMP,
        e2_action_contrastive_min_batch_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters state_code.
        use_ofc_analog=True,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,
        ofc_train_state_bias_head=train_head,
        # 485m DECOUPLE: a SEPARATE devaluation_bias_head with its OWN clamp
        # (+/-OFC_DEVAL_BIAS_SCALE, independent of the +/-0.5 C2 head). The devalued
        # re-ranking driver + readout + No-Go viability route through this head
        # (compute_devaluation_bias); the C2 discrimination stays on state_bias_head.
        # train flag mirrors the state head (frozen for ARM_0 silence control).
        use_ofc_devaluation_head=True,
        ofc_devaluation_bias_scale=OFC_DEVAL_BIAS_SCALE,
        ofc_train_devaluation_head=train_head,
        # MECH-448 (ARC-107) rank-preserving F->eligibility demotion (the ARM_ON
        # variable). The OFC bias is passed as score_bias at eval -> _modulatory_accum
        # is non-None -> the demotion eligibility branch arbitrates the committed
        # action within the F-eligible set with F removed from the final argmin.
        use_f_eligibility_demotion=demotion,
        f_eligibility_envelope_floor=F_ELIG_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIG_DN_SIGMA,
        # 485l FIX 5: channel-adaptive (mean-relative) envelope floor -- collapses the
        # 485i/485j bespoke per-seed floor dance into one global knob; excludes a
        # non-empty tail by construction at mean_factor>=1.0 on a non-flat F field.
        use_f_eligibility_adaptive_floor=True,
        f_eligibility_adaptive_mean_factor=F_ELIG_ADAPTIVE_MEAN_FACTOR,
        # 485l FIX 1: MECH-449 (ARC-107) Go/No-Go eligibility constitution (matched
        # CONSTANT on both arms). The active No-Go WITHDRAWAL is driven by the injected
        # per-candidate VIABILITY signal at the devaluation comparison only; with
        # use_dacc=False there is no MECH-260 perseveration source, so the gate engages
        # solely on the injected devaluation viability (silent without it -> C2 / ARM_0
        # protected). The gate runs only inside the f_demotion block (demotion-on arms).
        use_go_nogo_constitution=True,
        gng_viability_floor=GNG_VIABILITY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    frozen, _ = _make_agent(1, train_head=False, demotion=True)
    trainable, _ = _make_agent(2, train_head=True, demotion=True)
    ceiling, _ = _make_agent(3, train_head=True, demotion=False)
    assert frozen.ofc is not None and trainable.ofc is not None
    assert frozen.ofc.config.train_state_bias_head is False
    assert trainable.ofc.config.train_state_bias_head is True
    assert frozen.ofc.config.harm_dim == HARM_DIM
    assert abs(float(trainable.ofc.config.bias_scale) - OFC_BIAS_SCALE) < 1e-9
    assert hasattr(trainable.e2, "world_forward_contrastive_loss")
    assert bool(getattr(trainable.config.e2, "e2_action_contrastive_enabled", False))
    # readiness floor re-aligned to the DV floor (same-statistic gate).
    assert abs(BIAS_RANGE_FLOOR - DEVAL_SHIFT_MARGIN) < 1e-9
    fz = frozen.ofc.state_bias_head[-1]
    tz = trainable.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trainable.ofc.bias_head_parameters())) == 4
    # 485m DECOUPLE: the SEPARATE devaluation head is built, with its own clamp,
    # zeroed-when-frozen / random-when-trained, and its params are reachable.
    assert bool(getattr(trainable.ofc.config, "use_devaluation_head", False))
    assert trainable.ofc.devaluation_bias_head is not None
    assert frozen.ofc.devaluation_bias_head is not None
    assert trainable.ofc.config.train_devaluation_head is True
    assert frozen.ofc.config.train_devaluation_head is False
    assert abs(float(trainable.ofc.config.devaluation_bias_scale)
               - OFC_DEVAL_BIAS_SCALE) < 1e-9
    assert trainable.ofc.state_bias_head is not trainable.ofc.devaluation_bias_head
    fdz = frozen.ofc.devaluation_bias_head[-1]
    tdz = trainable.ofc.devaluation_bias_head[-1]
    assert bool(torch.all(fdz.weight == 0)) and bool(torch.all(fdz.bias == 0))
    assert not (bool(torch.all(tdz.weight == 0)) and bool(torch.all(tdz.bias == 0)))
    assert len(list(trainable.ofc.devaluation_bias_head_parameters())) == 4
    # frozen dev head -> compute_devaluation_bias is exactly zero (silence control).
    _bnk = torch.randn(N_PROBE_CANDIDATES, WORLD_DIM)
    assert bool(torch.all(frozen.ofc.compute_devaluation_bias(_bnk) == 0.0))
    # MECH-448 contract: E3Config carries the demotion lever + adaptive floor.
    assert hasattr(trainable.e3.config, "use_f_eligibility_demotion")
    assert trainable.e3.config.use_f_eligibility_demotion is True
    assert ceiling.e3.config.use_f_eligibility_demotion is False
    assert bool(getattr(trainable.e3.config, "use_f_eligibility_adaptive_floor", False))
    assert (
        abs(float(trainable.e3.config.f_eligibility_adaptive_mean_factor)
            - F_ELIG_ADAPTIVE_MEAN_FACTOR) < 1e-9
    )
    # MECH-449 contract: the Go/No-Go constitution is a matched CONSTANT on all arms.
    assert bool(getattr(trainable.e3.config, "use_go_nogo_constitution", False))
    assert bool(getattr(ceiling.e3.config, "use_go_nogo_constitution", False))
    assert bool(getattr(frozen.e3.config, "use_go_nogo_constitution", False))
    assert hasattr(trainable, "set_injected_go_nogo_signals")
    del frozen, trainable, ceiling
    print(
        "Preflight PASS: OFC analog + harm_dim + trainable/frozen head + "
        "bias_head_parameters + ofc_bias_scale=0.5 + SD-056 contrastive + readiness "
        "floor 0.05 == DV floor + MECH-448 demotion + adaptive envelope floor + "
        "MECH-449 Go/No-Go constitution (matched constant; injection seam present)",
        flush=True,
    )


def _build_snap(agent: REEAgent, candidates: List) -> Optional[torch.Tensor]:
    """First-step z_world summaries [K, world_dim] from the proposer candidates."""
    if not isinstance(candidates, list) or len(candidates) < N_PROBE_CANDIDATES:
        return None
    if getattr(candidates[0], "world_states", None) is None:
        return None
    if len(candidates[0].world_states) < 2:
        return None
    return torch.cat(
        [c.world_states[1].detach().clone() for c in candidates[:N_PROBE_CANDIDATES]],
        dim=0,
    )  # [K, world_dim]


def _head_weight_vector(agent: REEAgent) -> torch.Tensor:
    """C2 discrimination head (state_bias_head) flat weight vector."""
    return torch.cat(
        [p.detach().reshape(-1).cpu() for p in agent.ofc.bias_head_parameters()]
    )


def _devaluation_head_weight_vector(agent: REEAgent) -> torch.Tensor:
    """485m DECOUPLE: devaluation_bias_head flat weight vector -- the C1-relevant
    head the devaluation DVs route on; readiness (c) gates on its training delta."""
    return torch.cat(
        [p.detach().reshape(-1).cpu()
         for p in agent.ofc.devaluation_bias_head_parameters()]
    )


def _all_trainable_ofc_params(agent: REEAgent) -> List[torch.nn.Parameter]:
    """485m DECOUPLE: BOTH OFC output heads -- the C2 state_bias_head (trained by the
    high-threat spread regime) AND the devaluation_bias_head (trained by the devalued
    re-rank regime). One optimizer over both steps each from its own loss term."""
    return list(agent.ofc.bias_head_parameters()) + list(
        agent.ofc.devaluation_bias_head_parameters()
    )


# --------------------------------------------------------------------------- #
# training (P0 encoder warmup + P1 paired threat-conditioned head driver)
# kept from 485h/485k -- the demotion + constitution flags do NOT affect training,
# only the eval committed selection through E3.select().
# --------------------------------------------------------------------------- #
def _encoder_step(agent: REEAgent, env, steps: int, arm: str, ep_global: int) -> float:
    device = agent.device
    e1_opt = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    he_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    he_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ep_reward = 0.0
    z_wp = z_sp = act_p = None
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(
            obs_body,
            obs_world,
            obs_harm=obs_dict.get("harm_obs"),
            obs_harm_a=obs_dict.get("harm_obs_a"),
            obs_harm_history=obs_dict.get("harm_history"),
        )
        z_w = latent.z_world.detach()
        if z_wp is not None and act_p is not None:
            agent.record_transition(z_sp, act_p, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, WORLD_DIM, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        drive = REEAgent.compute_drive_level(obs_body)
        agent.update_z_goal(
            benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
            drive_level=drive,
        )
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = _action_to_onehot(
                random.randint(0, env.action_dim - 1), env.action_dim, device
            )
            agent._last_action = action
        _, harm, done, _, obs_dict = env.step(action)
        ep_reward += float(harm)
        if z_wp is not None and act_p is not None:
            wf_buf.append((z_wp.cpu(), act_p.cpu(), z_w.cpu()))
            if len(wf_buf) > WF_BUF_MAX:
                wf_buf = wf_buf[-WF_BUF_MAX:]
        he_buf.append(
            (z_w.cpu(), torch.tensor([abs(float(harm)) if float(harm) < 0 else 0.0]))
        )
        if len(wf_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([wf_buf[i][0] for i in idx]).to(device)
            a = torch.cat([wf_buf[i][1] for i in idx]).to(device)
            zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(device)
            pred = agent.e2.world_forward(zw, a)
            loss = F.mse_loss(pred, zw1)
            # SD-056 action-contrastive auxiliary loss (kept from 485f/485g/485h/485k).
            contrast = agent.e2.world_forward_contrastive_loss(
                zw, a, zw1,
                temperature=E2_CONTRASTIVE_TEMP,
                min_batch_classes=2,
            )
            loss = loss + E2_CONTRASTIVE_WEIGHT * contrast
            if loss.requires_grad:
                e2_opt.zero_grad()
                loss.backward()
                e2_opt.step()
            with torch.no_grad():
                agent.e3.update_running_variance((pred.detach() - zw1).detach())
        if len(he_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(he_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([he_buf[i][0] for i in idx]).to(device)
            ht = torch.cat([he_buf[i][1] for i in idx]).to(device)
            loss = F.mse_loss(agent.e3.harm_eval(zw).squeeze(), ht.squeeze())
            if loss.requires_grad:
                he_opt.zero_grad()
                loss.backward()
                he_opt.step()
        if len(agent._world_experience_buffer) >= 2:
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()
        z_wp, z_sp, act_p = z_w, latent.z_self.detach(), action.detach()
        if done:
            break
    if ep_global == 1 or ep_global % 10 == 0:
        print(
            f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P0  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _ofc_threat_conditioned_driver_loss(
    agent: REEAgent,
    snap_buf: List[Tuple[torch.Tensor, torch.Tensor]],
    z_harm_high: torch.Tensor,
    z_harm_low: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, int, int]:
    """485l paired threat-conditioned, outcome-coupled, per-candidate driver. Trains the
    trained-OFC head on BOTH regimes using a SHARED state_code construction
    (_settle_state_code) identical to the eval:
      - high-threat regime (KEPT verbatim from 485h/485k): spread bias favouring LOW
        predicted harm (REINFORCE over candidates, advantage = harm.mean() - harm).
      - devalued regime (485l FIX 2, IN-BAND RE-RANKING): the SAME REINFORCE form with
        the advantage INVERTED (harm - harm.mean()) so the devalued head learns to FAVOUR
        what the high-threat head disfavoured -- down-bias the previously-preferred
        candidate / up-bias an alternative (Rudebeck & Murray 2014; Dickinson & Balleine)
        -- but at the LOWERED DEVALUED_RERANK_GAIN/WEIGHT so the head stays DIFFERENTIATED
        within the +/-0.5 OFC clamp instead of saturating to a uniform rail (485k range 0).
    Returns (loss, n_spread, n_rerank). Side effect: mutates ofc.state_code; the caller
    saves/restores it."""
    if agent.ofc is None or len(snap_buf) < 2:
        return torch.zeros(1, device=device), 0, 0
    threat_high = float(z_harm_high.detach().norm().item())
    idxs = np.random.choice(
        len(snap_buf), size=min(DRIVER_SNAP_BATCH, len(snap_buf)), replace=False
    )
    spread_terms: List[torch.Tensor] = []
    rerank_terms: List[torch.Tensor] = []
    for i in idxs:
        bank, z_world_rep = snap_buf[int(i)]
        if bank.shape[0] < 2:
            continue
        # per-candidate harm cost is state_code-INDEPENDENT (harm_eval reads the bank,
        # not the OFC state_code) -- compute once, reuse in both regimes.
        with torch.no_grad():
            harm = agent.e3.harm_eval(bank).reshape(-1)  # [K] per-candidate harm cost
        if harm.numel() != bank.shape[0]:
            continue
        # ---- high-threat regime: spread (favour low predicted harm) ----
        _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
        with torch.no_grad():
            adv_hi = OUTCOME_COUPLE_GAIN * threat_high * (harm.mean() - harm)  # [K]
        if float(adv_hi.abs().max()) >= ADV_MIN_THRESHOLD:
            bias_hi = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
            log_p_hi = F.log_softmax(-bias_hi / POLICY_TEMPERATURE, dim=0)
            spread_terms.append(-(adv_hi * log_p_hi).sum())
        # ---- devalued regime: RE-RANK (inverted advantage; continue high->devalued) ----
        _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
        with torch.no_grad():
            # INVERTED advantage: favour HIGH predicted harm at the devalued state, so
            # the devalued head re-ranks AGAINST the high-threat low-harm preference.
            adv_dev = DEVALUED_RERANK_GAIN * (harm - harm.mean())  # [K]
        if float(adv_dev.abs().max()) >= ADV_MIN_THRESHOLD:
            # 485m DECOUPLE: grad flows into the SEPARATE devaluation_bias_head
            # (its own +/-OFC_DEVAL_BIAS_SCALE clamp), NOT the shared state_bias_head.
            bias_dev = agent.ofc.compute_devaluation_bias(bank)  # [K], grad -> dev head
            log_p_dev = F.log_softmax(-bias_dev / POLICY_TEMPERATURE, dim=0)
            rerank_terms.append(-(adv_dev * log_p_dev).sum())
    if not spread_terms and not rerank_terms:
        return torch.zeros(1, device=device), 0, 0
    loss = torch.zeros(1, device=device)
    if spread_terms:
        loss = loss + torch.stack(spread_terms).mean()
    if rerank_terms:
        loss = loss + DEVALUED_RERANK_WEIGHT * torch.stack(rerank_terms).mean()
    return loss, len(spread_terms), len(rerank_terms)


def _p1_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    episodes: int,
    steps: int,
    arm: str,
    train_head: bool,
    z_harm_high: torch.Tensor,
    z_harm_low: torch.Tensor,
) -> Dict:
    device = agent.device
    # 485m DECOUPLE: one optimizer over BOTH heads -- the high-threat spread regime's
    # loss term grads flow to state_bias_head, the devalued re-rank regime's to
    # devaluation_bias_head; a single Adam over both param sets steps each.
    bias_opt = (
        optim.Adam(_all_trainable_ofc_params(agent), lr=LR_OFC_BIAS)
        if train_head
        else None
    )
    head_init = _head_weight_vector(agent)
    deval_head_init = _devaluation_head_weight_vector(agent)
    grad_nonzero_updates = 0
    n_spread_terms = 0
    n_rerank_terms = 0
    snap_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    bias_samples: List[float] = []
    agent.train()
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        step = 0
        z_wp = z_sp = act_p = None
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(
                obs_body,
                obs_world,
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            if z_wp is not None and act_p is not None:
                agent.record_transition(z_sp, act_p, latent.z_self.detach())
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(obs_body),
            )
            if step % RECORD_EVERY_N_STEPS == 0:
                snap = _build_snap(agent, candidates)
                if snap is not None:
                    snap_buf.append((snap, latent.z_world.detach().clone()))
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, device
                )
                agent._last_action = action
            if agent.ofc is not None:
                bias_samples.append(float(agent.ofc._last_bias_abs_mean))
            _, harm, done, _, obs_dict = env.step(action)
            z_wp, z_sp, act_p = (
                latent.z_world.detach(),
                latent.z_self.detach(),
                action.detach(),
            )
            step += 1
            if done:
                break
        if len(snap_buf) > OUTCOME_BUF_MAX:
            snap_buf = snap_buf[-OUTCOME_BUF_MAX:]
        if bias_opt is not None and len(snap_buf) >= 2:
            saved_state = agent.ofc.state_code.detach().clone()
            l_loss, n_sp, n_rr = _ofc_threat_conditioned_driver_loss(
                agent, snap_buf, z_harm_high, z_harm_low, device
            )
            with torch.no_grad():
                agent.ofc.state_code.copy_(saved_state)
            n_spread_terms += n_sp
            n_rerank_terms += n_rr
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                # 485m DECOUPLE: detect + clip grads across BOTH heads.
                _params = _all_trainable_ofc_params(agent)
                gsum = sum(
                    float(p.grad.abs().sum())
                    for p in _params
                    if p.grad is not None
                )
                if gsum > 0:
                    grad_nonzero_updates += 1
                torch.nn.utils.clip_grad_norm_(_params, 1.0)
                bias_opt.step()
        ep_global = P0_EPISODES + ep + 1
        if ep_global % 10 == 0:
            print(
                f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P1"
                f"  bias_abs={bias_samples[-1] if bias_samples else 0:.5f}"
                f"  grad_updates={grad_nonzero_updates}",
                flush=True,
            )
    head_final = _head_weight_vector(agent)
    head_delta = float(torch.norm(head_final - head_init).item())
    deval_head_final = _devaluation_head_weight_vector(agent)
    deval_head_delta = float(torch.norm(deval_head_final - deval_head_init).item())
    return {
        "p1_mean_abs_ofc_bias": float(np.mean(bias_samples)) if bias_samples else 0.0,
        "head_weight_delta_norm": head_delta,
        "devaluation_head_weight_delta_norm": deval_head_delta,
        "n_grad_nonzero_updates": grad_nonzero_updates,
        "n_spread_loss_terms": n_spread_terms,
        "n_rerank_loss_terms": n_rerank_terms,
    }


# --------------------------------------------------------------------------- #
# P2 behavioural eval on the demotion+constitution-enabled E3 selector
# --------------------------------------------------------------------------- #
def _collect_eval_candidates(
    agent: REEAgent, env: CausalGridWorldV2, steps: int
) -> Tuple[Optional[List], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the trained agent briefly and return (candidates[:K], candidate_bank
    [K, world_dim], z_world_rep [1, world_dim]) -- a real candidate list aligned with
    its first-step z_world bank for the controlled devaluation / discrimination drives
    (kept from 485h/485k)."""
    device = agent.device
    cands: Optional[List] = None
    bank: Optional[torch.Tensor] = None
    z_world_rep: Optional[torch.Tensor] = None
    agent.eval()
    with torch.no_grad():
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps):
            latent = agent.sense(
                obs_dict["body_state"],
                obs_dict["world_state"],
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            snap = _build_snap(agent, candidates)
            if snap is not None and bank is None:
                bank = snap
                z_world_rep = latent.z_world.detach().clone()
                cands = list(candidates[:N_PROBE_CANDIDATES])
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(0, env.action_dim, device)
            _, _, done, _, obs_dict = env.step(action)
            if bank is not None:
                break
            if done:
                _, obs_dict = env.reset()
                agent.reset()
    return cands, bank, z_world_rep


def _committed_select(
    agent: REEAgent, cands: List, score_bias: torch.Tensor, temperature: float,
    go_nogo_signals: Optional[Dict[str, Any]] = None,
) -> Tuple[int, bool, Dict, float]:
    """Route the OFC bias through the REAL E3 selector as score_bias and return the
    COMMITTED candidate index + commit flag + diagnostics + raw (F-score) range.

    With use_f_eligibility_demotion on the agent's E3Config, the OFC bias (passed as
    score_bias -> _modulatory_accum) arbitrates the committed action WITHIN the
    F-eligible envelope, F removed from the final argmin. When go_nogo_signals is
    supplied AND use_go_nogo_constitution is on, the MECH-449 gate further withdraws
    No-Go'd candidates from the eligible set BEFORE the within-eligible arbitration
    (the active devaluation withdrawal). EVAL_COMMIT_THRESHOLD has forced the committed
    (argmin-within-eligible) path so selected_index is deterministic."""
    kwargs: Dict[str, Any] = dict(
        temperature=temperature,
        goal_state=getattr(agent, "goal_state", None),
        score_bias=score_bias,
    )
    if go_nogo_signals is not None:
        kwargs["go_nogo_signals"] = go_nogo_signals
    res = agent.e3.select(cands, **kwargs)
    diag = dict(agent.e3.last_score_diagnostics)
    raw = agent.e3.last_raw_scores
    raw_range = (
        float((raw.max() - raw.min()).item())
        if raw is not None and raw.numel() > 1
        else 0.0
    )
    return int(res.selected_index), bool(res.committed), diag, raw_range


def _demotion_committed_eval(
    agent: REEAgent, cands: List, bank: torch.Tensor, z_world_rep: torch.Tensor,
    z_harm_high: torch.Tensor, z_harm_low: torch.Tensor, seed: int,
) -> Dict:
    """PRIMARY behavioural eval: devaluation-selection shift (active No-Go withdrawal) +
    task-role discrimination measured on the COMMITTED selection through the
    demotion+constitution-enabled E3 selector. Records the readiness positive control
    (OFC bias range at the high-threat state), the devalued-state range (the C1/C1b
    statistic), the high-vs-devalued bias-VECTOR inversion (C1b), the MECH-448
    non-degeneracy signals (f_eligibility_excluded_count / demotion_active), and the
    MECH-449 non-vacuity signal (go_nogo_n_soft_applied at the devalued state)."""
    K = bank.shape[0]
    agent.eval()
    prev_thr = float(agent.e3.config.commitment_threshold)
    agent.e3.config.commitment_threshold = EVAL_COMMIT_THRESHOLD
    excluded_count = -1
    envelope_size = -1
    demotion_active = False
    winner_neq_f = False
    raw_range_hi = 0.0
    deval_bias_l2_shift = 0.0      # C1b primary readout: ||bias_high - bias_low||
    deval_bias_cosine = 1.0        # C1b primary readout: cos(bias_high, bias_low)
    go_nogo_active = False
    go_nogo_n_soft_applied = 0
    go_nogo_envelope_size = -1
    try:
        with torch.no_grad():
            # ---- P2-a devaluation sensitivity (active No-Go withdrawal) ----
            _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
            bias_high = agent.ofc.compute_bias(bank).detach()
            bias_range_high = _bias_range(agent, bank)  # readiness positive control

            # High-threat baseline select: constitution config-on but signal-INERT
            # (go_nogo_signals=None) so the high-threat reference is NOT withdrawn --
            # the devaluation shift is measured against an unbiased high-threat commit.
            idx_high, comm_high, diag_hi, raw_range_hi = _committed_select(
                agent, cands, bias_high, POLICY_TEMPERATURE, go_nogo_signals=None
            )
            excluded_count = int(diag_hi.get("f_eligibility_excluded_count", -1))
            envelope_size = int(diag_hi.get("f_eligibility_envelope_size", -1))
            demotion_active = bool(diag_hi.get("f_eligibility_demotion_active", False))
            winner_neq_f = bool(diag_hi.get("f_eligibility_winner_neq_f_argmin", False))

            _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
            # 485m DECOUPLE: the DEVALUED valuation reads the SEPARATE devaluation
            # head (its own +/-OFC_DEVAL_BIAS_SCALE clamp), so the cross-candidate
            # range can clear the 0.05 floor without saturating against C2.
            bias_low = agent.ofc.compute_devaluation_bias(bank).detach()
            bias_range_devalued = _deval_bias_range(agent, bank)  # the C1/C1b DVs route on this
            # C1b: the high-vs-devalued OFC bias-VECTOR change. A re-ranking devalued
            # driver INVERTS the bias vector (large L2 shift, negative cosine).
            _bh = bias_high.reshape(-1)
            _bl = bias_low.reshape(-1)
            deval_bias_l2_shift = float((_bh - _bl).norm().item())
            deval_bias_cosine = (
                float(F.cosine_similarity(_bh.unsqueeze(0), _bl.unsqueeze(0), dim=1).item())
                if _bh.numel() > 1
                else 1.0
            )
            # 485l FIX 1: map the trained DEVALUED valuation to a per-candidate VIABILITY
            # No-Go so the MECH-449 constitution actively WITHDRAWS the previously-preferred
            # (now-devalued) candidate from the eligible set. None when flat (frozen head)
            # -> gate inert -> silent (ARM_0 control).
            viability_nogo = _build_viability_nogo(bias_low)
            gng_signals = (
                {"viability": viability_nogo} if viability_nogo is not None else None
            )
            idx_low, comm_low, diag_lo, _ = _committed_select(
                agent, cands, bias_low, POLICY_TEMPERATURE, go_nogo_signals=gng_signals
            )
            go_nogo_active = bool(diag_lo.get("go_nogo_constitution_active", False))
            go_nogo_n_soft_applied = int(diag_lo.get("go_nogo_n_soft_applied", 0))
            go_nogo_envelope_size = int(diag_lo.get("go_nogo_envelope_size", -1))
            deval_shift = _tv(_onehot_vec(idx_high, K), _onehot_vec(idx_low, K))

            # ---- P2-b task-role discrimination (C2 PROTECTED: go_nogo_signals=None) ----
            ctx = CONTEXT_TICKS
            g = torch.Generator().manual_seed(20000 + seed)
            hist_a = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
            hist_b = z_world_rep - 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
            hist_a2 = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)

            def _drive_then_select(hist: torch.Tensor) -> Tuple[int, float]:
                agent.ofc.reset()
                for t in range(hist.shape[0]):
                    agent.ofc.update(z_world=hist[t:t + 1], z_harm=None, gate=1.0)
                agent.ofc.update(z_world=z_world_rep, z_harm=None, gate=1.0)
                b = agent.ofc.compute_bias(bank).detach()
                # C2 protected: no devaluation No-Go injected -> demotion-alone path
                # (which converted in 485j).
                idx, _, _, _ = _committed_select(
                    agent, cands, b, POLICY_TEMPERATURE, go_nogo_signals=None
                )
                rng = _bias_range(agent, bank)
                return idx, rng

            idx_a, bias_range_context_a = _drive_then_select(hist_a)
            idx_b, bias_range_context_b = _drive_then_select(hist_b)
            idx_a2, _ = _drive_then_select(hist_a2)
            between = _tv(_onehot_vec(idx_a, K), _onehot_vec(idx_b, K))
            within = _tv(_onehot_vec(idx_a, K), _onehot_vec(idx_a2, K))
            separation_ratio = between / max(within, 1e-6)
    finally:
        agent.e3.config.commitment_threshold = prev_thr

    collapse_ratio = (
        bias_range_devalued / bias_range_high if bias_range_high > 1e-9 else 0.0
    )
    return {
        "devaluation_selection_shift": deval_shift,
        "bias_range_high_threat": bias_range_high,
        "bias_range_devalued": bias_range_devalued,
        "devaluation_bias_l2_shift": deval_bias_l2_shift,
        "devaluation_bias_cosine": deval_bias_cosine,
        "devaluation_range_collapse_ratio": collapse_ratio,
        "bias_range_context_a": bias_range_context_a,
        "bias_range_context_b": bias_range_context_b,
        "bank_zworld_spread": _bank_zworld_spread(bank),
        "between_context_selection_tv": between,
        "within_context_selection_jitter": within,
        "discrimination_separation_ratio": separation_ratio,
        "committed_high": comm_high,
        "committed_low": comm_low,
        "f_eligibility_demotion_active": demotion_active,
        "f_eligibility_excluded_count": excluded_count,
        "f_eligibility_envelope_size": envelope_size,
        "f_eligibility_winner_neq_f_argmin": winner_neq_f,
        "raw_score_range_high": raw_range_hi,
        "go_nogo_constitution_active": go_nogo_active,
        "go_nogo_n_soft_applied": go_nogo_n_soft_applied,
        "go_nogo_envelope_size_devalued": go_nogo_envelope_size,
    }


def _empty_eval() -> Dict:
    return {
        "devaluation_selection_shift": 0.0,
        "bias_range_high_threat": 0.0,
        "bias_range_devalued": 0.0,
        "devaluation_bias_l2_shift": 0.0,
        "devaluation_bias_cosine": 1.0,
        "devaluation_range_collapse_ratio": 0.0,
        "bias_range_context_a": 0.0,
        "bias_range_context_b": 0.0,
        "bank_zworld_spread": 0.0,
        "between_context_selection_tv": 0.0,
        "within_context_selection_jitter": 0.0,
        "discrimination_separation_ratio": 0.0,
        "committed_high": False,
        "committed_low": False,
        "f_eligibility_demotion_active": False,
        "f_eligibility_excluded_count": -1,
        "f_eligibility_envelope_size": -1,
        "f_eligibility_winner_neq_f_argmin": False,
        "raw_score_range_high": 0.0,
        "go_nogo_constitution_active": False,
        "go_nogo_n_soft_applied": 0,
        "go_nogo_envelope_size_devalued": -1,
        "eval_context_built": False,
    }


def run_arm_seed(arm: str, train_head: bool, demotion: bool, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    print(
        f"\nSeed {seed} Condition {arm} train_head={train_head} demotion={demotion}",
        flush=True,
    )
    full_config = {
        "arm": arm,
        "train_head": train_head,
        "demotion": demotion,
        "p0": p0,
        "p1": p1,
        "steps": steps,
        "ofc_bias_scale": OFC_BIAS_SCALE,
        "e2_contrastive_weight": E2_CONTRASTIVE_WEIGHT,
        "lr_ofc_bias": LR_OFC_BIAS,
        "outcome_couple_gain": OUTCOME_COUPLE_GAIN,
        "devalued_rerank_weight": DEVALUED_RERANK_WEIGHT,
        "devalued_rerank_gain": DEVALUED_RERANK_GAIN,
        "bias_range_floor": BIAS_RANGE_FLOOR,
        "bias_l2_floor": BIAS_L2_FLOOR,
        "bias_cosine_ceil": BIAS_COSINE_CEIL,
        "f_eligibility_envelope_floor_default": F_ELIG_ENVELOPE_FLOOR,
        "f_eligibility_dn_sigma": F_ELIG_DN_SIGMA,
        "f_eligibility_adaptive_mean_factor": F_ELIG_ADAPTIVE_MEAN_FACTOR,
        "gng_viability_floor": GNG_VIABILITY_FLOOR,
        "gng_protect_min_eligible": GNG_PROTECT_MIN_ELIGIBLE,
        "eval_commit_threshold": EVAL_COMMIT_THRESHOLD,
        "env_kwargs": ENV_KWARGS,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        agent, env = _make_agent(seed, train_head=train_head, demotion=demotion)
        # Canonical high-threat / devalued z_harm, built ONCE per (arm, seed) and
        # shared by the P1 driver and the P2 eval so trained regimes == tested.
        g = torch.Generator().manual_seed(20000 + seed)
        z_harm_high = torch.randn(1, HARM_DIM, generator=g) * 1.0
        z_harm_low = torch.zeros(1, HARM_DIM)
        for ep in range(p0):
            _encoder_step(agent, env, steps, arm, ep + 1)
        p1m = _p1_train(agent, env, p1, steps, arm, train_head, z_harm_high, z_harm_low)
        cands, bank, z_world_rep = _collect_eval_candidates(agent, env, steps)
        if bank is None or z_world_rep is None or cands is None:
            beh = _empty_eval()
        else:
            beh = _demotion_committed_eval(
                agent, cands, bank, z_world_rep, z_harm_high, z_harm_low, seed
            )
            beh["eval_context_built"] = True
        row = {
            "arm": arm,
            "seed": seed,
            "ofc_train_state_bias_head": train_head,
            "use_f_eligibility_demotion": demotion,
            "head_weight_delta_norm": p1m["head_weight_delta_norm"],
            "devaluation_head_weight_delta_norm": p1m["devaluation_head_weight_delta_norm"],
            "n_grad_nonzero_updates": p1m["n_grad_nonzero_updates"],
            "n_spread_loss_terms": p1m["n_spread_loss_terms"],
            "n_rerank_loss_terms": p1m["n_rerank_loss_terms"],
            "p1_mean_abs_ofc_bias": p1m["p1_mean_abs_ofc_bias"],
            **beh,
        }
        cell.stamp(row)
    print(
        f"verdict: {'PASS' if beh.get('eval_context_built') else 'FAIL'}  "
        f"head_delta={p1m['head_weight_delta_norm']:.6f}  "
        f"bias_range_hi={beh['bias_range_high_threat']:.6f}  "
        f"bias_range_dev={beh['bias_range_devalued']:.6f}  "
        f"deval_shift={beh['devaluation_selection_shift']:.4f}  "
        f"deval_bias_l2={beh['devaluation_bias_l2_shift']:.4f}  "
        f"deval_bias_cos={beh['devaluation_bias_cosine']:.4f}  "
        f"between_tv={beh['between_context_selection_tv']:.4f}  "
        f"excluded={beh['f_eligibility_excluded_count']}  "
        f"nogo_applied={beh['go_nogo_n_soft_applied']}",
        flush=True,
    )
    return row


# --------------------------------------------------------------------------- #
# aggregation + interpretation
# --------------------------------------------------------------------------- #
ARM_FROZEN = "ARM_0_frozen_demotion_on"
ARM_CEILING = "ARM_1_trained_demotion_off"
ARM_TEST = "ARM_2_trained_demotion_on"


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    arms = [
        (ARM_FROZEN, False, True),
        (ARM_CEILING, True, False),
        (ARM_TEST, True, True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, train_head, demotion in arms:
            by_arm[arm_label].append(
                run_arm_seed(arm_label, train_head, demotion, seed, dry_run)
            )

    frozen = by_arm[ARM_FROZEN]
    ceiling = by_arm[ARM_CEILING]
    test = by_arm[ARM_TEST]
    n = len(seeds)

    # READINESS (non-vacuity, on the ARM_2 test arm) -- reported as PER-SEED PASS COUNTS,
    # NOT aggregate-max (the 485k fix; the indexer recomputes met from measured vs
    # threshold, so a MAX-over-seeds measured spuriously reads met=True even at 1/3 seeds).
    # 485m DECOUPLE: readiness (c) requires BOTH heads trained -- state_bias_head
    # (the C2 discrimination valuation) AND devaluation_bias_head (the C1 devaluation
    # valuation the load-bearing devaluation DVs + the No-Go viability route on).
    head_trained_seeds = sum(
        1 for r in test
        if r["head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["devaluation_head_weight_delta_norm"] > HEAD_DELTA_MIN
    )
    deval_head_trained_seeds = sum(
        1 for r in test if r["devaluation_head_weight_delta_norm"] > HEAD_DELTA_MIN
    )
    ready_high_seeds = sum(1 for r in test if r["bias_range_high_threat"] > BIAS_RANGE_FLOOR)
    ready_devalued_seeds = sum(1 for r in test if r["bias_range_devalued"] > BIAS_RANGE_FLOOR)
    eval_built_seeds = sum(1 for r in test if r.get("eval_context_built", False))
    ready_seeds = sum(
        1
        for r in test
        if r["head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["devaluation_head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["bias_range_high_threat"] > BIAS_RANGE_FLOOR
        and r["bias_range_devalued"] > BIAS_RANGE_FLOOR
        and r.get("eval_context_built", False)
    )
    nondegen_seeds = sum(
        1
        for r in test
        if int(r.get("f_eligibility_excluded_count", -1)) > 0
        and bool(r.get("f_eligibility_demotion_active", False))
    )
    nogo_engaged_seeds = sum(
        1 for r in test if int(r.get("go_nogo_n_soft_applied", 0)) > 0
    )
    readiness_met = (
        ready_seeds >= MIN_PASS_SEEDS
        and nondegen_seeds >= MIN_PASS_SEEDS
        and nogo_engaged_seeds >= MIN_PASS_SEEDS
    )

    c1_seeds = 0
    c1b_seeds = 0
    c2_seeds = 0
    for ce, te in zip(ceiling, test):
        if (
            te["devaluation_selection_shift"] > DEVAL_SHIFT_MARGIN
            and te["devaluation_selection_shift"]
            > ce["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
        ):
            c1_seeds += 1
        if (
            te["devaluation_bias_l2_shift"] > BIAS_L2_FLOOR
            and te["devaluation_bias_cosine"] < BIAS_COSINE_CEIL
        ):
            c1b_seeds += 1
        if (
            te["discrimination_separation_ratio"] >= SEPARATION_RATIO_MIN
            and te["between_context_selection_tv"] >= BETWEEN_TV_FLOOR
            and te["between_context_selection_tv"]
            > ce["between_context_selection_tv"] + BETWEEN_TV_FLOOR
        ):
            c2_seeds += 1
    c3_seeds = sum(
        1
        for r in frozen
        if r["devaluation_selection_shift"] < FROZEN_SILENCE_EPS
        and r["between_context_selection_tv"] < FROZEN_SILENCE_EPS
    )

    c1 = c1_seeds >= MIN_PASS_SEEDS
    c1b = c1b_seeds >= MIN_PASS_SEEDS
    c2 = c2_seeds >= MIN_PASS_SEEDS
    c3 = c3_seeds >= MIN_PASS_SEEDS
    overall_pass = readiness_met and c1 and c1b and c2 and c3

    return {
        "by_arm": by_arm,
        "n_seeds": n,
        "head_trained_seeds": head_trained_seeds,
        "deval_head_trained_seeds": deval_head_trained_seeds,
        "ready_high_seeds": ready_high_seeds,
        "ready_devalued_seeds": ready_devalued_seeds,
        "eval_built_seeds": eval_built_seeds,
        "ready_seeds": ready_seeds,
        "nondegen_seeds": nondegen_seeds,
        "nogo_engaged_seeds": nogo_engaged_seeds,
        "readiness_met": readiness_met,
        "max_test_head_delta": max((r["head_weight_delta_norm"] for r in test), default=0.0),
        "max_test_deval_head_delta": max((r["devaluation_head_weight_delta_norm"] for r in test), default=0.0),
        "max_test_bias_range": max((r["bias_range_high_threat"] for r in test), default=0.0),
        "max_test_bias_range_devalued": max((r["bias_range_devalued"] for r in test), default=0.0),
        "max_test_deval_bias_l2_shift": max((r["devaluation_bias_l2_shift"] for r in test), default=0.0),
        "min_test_deval_bias_cosine": min((r["devaluation_bias_cosine"] for r in test), default=1.0),
        "max_test_excluded_count": max((int(r.get("f_eligibility_excluded_count", -1)) for r in test), default=-1),
        "max_test_nogo_applied": max((int(r.get("go_nogo_n_soft_applied", 0)) for r in test), default=0),
        "max_test_between_tv": max((r["between_context_selection_tv"] for r in test), default=0.0),
        "max_test_deval_shift": max((r["devaluation_selection_shift"] for r in test), default=0.0),
        "max_ceiling_deval_shift": max((r["devaluation_selection_shift"] for r in ceiling), default=0.0),
        "max_ceiling_between_tv": max((r["between_context_selection_tv"] for r in ceiling), default=0.0),
        "acceptance": {
            "C1_devaluation_behavioural_shift": c1,
            "C1b_devaluation_bias_vector_inversion": c1b,
            "C2_discrimination_behavioural_separation": c2,
            "C3_silence_control": c3,
            "n_c1_seeds": c1_seeds,
            "n_c1b_seeds": c1b_seeds,
            "n_c2_seeds": c2_seeds,
            "n_c3_seeds": c3_seeds,
            "pass": overall_pass,
        },
    }


def _interpretation(result: Dict) -> Dict:
    acc = result["acceptance"]
    ready = bool(result["readiness_met"])
    if not ready:
        label = "substrate_not_ready_requeue"
    elif acc["pass"]:
        label = "sd033b_go_nogo_devaluation_conversion_supported"
    else:
        label = "conversion_ceiling_persists_despite_go_nogo"

    # PER-SEED PASS COUNTS (the 485k aggregate-max fix). Each precondition's `measured`
    # is the number of seeds clearing the underlying per-seed floor; `threshold` is
    # MIN_PASS_SEEDS, so the indexer recompute requires >= 2/3 seeds, NOT max-over-seeds.
    preconditions = [
        {
            "name": "ofc_bias_range_supra_floor_seed_count_HIGH_THREAT",
            "description": (
                "Number of ARM_2 seeds whose trained-head OFC bias cross-candidate RANGE "
                "at the HIGH-THREAT positive control clears BIAS_RANGE_FLOOR (0.05) -- the "
                "C2 discrimination positive control. Reported as a PER-SEED PASS COUNT (>= "
                "MIN_PASS_SEEDS), NOT aggregate-max (the 485k bug: max-over-seeds read "
                "met=True at 1/3). Below MIN_PASS_SEEDS -> substrate_not_ready_requeue."
            ),
            "measured": float(result["ready_high_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "control": "ARM_2 trained-arm bias over a real candidate bank at the high-threat state",
            "met": result["ready_high_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "ofc_bias_range_supra_floor_seed_count_DEVALUED_STATE",
            "description": (
                "Number of ARM_2 seeds whose trained-head OFC bias cross-candidate RANGE "
                "at the DEVALUED state clears BIAS_RANGE_FLOOR (0.05) -- the SAME statistic "
                "the load-bearing C1 / C1b devaluation DVs route on. The in-band re-ranking "
                "driver (FIX 2) must produce a DIFFERENTIATED (non-flat, non-saturated) "
                "devalued bias. PER-SEED PASS COUNT (>= MIN_PASS_SEEDS), the 485k "
                "aggregate-max fix. Below MIN_PASS_SEEDS -> substrate_not_ready_requeue."
            ),
            "measured": float(result["ready_devalued_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "control": "ARM_2 trained-arm bias over a real candidate bank at the DEVALUED state",
            "met": result["ready_devalued_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "ofc_head_trained_seed_count",
            "description": (
                "485m DECOUPLE: number of ARM_2 seeds where BOTH OFC heads trained -- "
                "state_bias_head (C2 discrimination) AND devaluation_bias_head (C1 "
                "devaluation, the head the load-bearing devaluation DVs + the No-Go "
                "viability route on) -- weight-delta-from-init L2 norm clears "
                "HEAD_DELTA_MIN. PER-SEED PASS COUNT (>= MIN_PASS_SEEDS)."
            ),
            "measured": float(result["head_trained_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "control": "trainable-arm BOTH heads trained on frozen-encoder state_code in P1",
            "met": result["head_trained_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "ofc_devaluation_head_trained_seed_count",
            "description": (
                "485m DECOUPLE: number of ARM_2 seeds whose SEPARATE devaluation_bias_head "
                "weight-delta-from-init L2 norm clears HEAD_DELTA_MIN -- the devaluation "
                "re-ranking head (clamp +/-OFC_DEVAL_BIAS_SCALE) genuinely trained under "
                "the inverted-advantage re-rank driver. The C1/C1b DVs + the No-Go "
                "viability route on THIS head. PER-SEED PASS COUNT (>= MIN_PASS_SEEDS)."
            ),
            "measured": float(result["deval_head_trained_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "control": "trainable-arm devaluation_bias_head trained in P1",
            "met": result["deval_head_trained_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "mech448_f_eligibility_excluded_seed_count",
            "description": (
                "MECH-448 NON-DEGENERACY: number of ARM_2 seeds where the channel-adaptive "
                "F->eligibility envelope actually EXCLUDED at least one candidate "
                "(f_eligibility_excluded_count > 0) AND the demotion path was active -- F "
                "genuinely demoted on a divergent F pool, not an all-admit no-op. PER-SEED "
                "PASS COUNT (>= MIN_PASS_SEEDS). excluded_count == 0 across seeds -> "
                "substrate_not_ready_requeue, NEVER a weakens."
            ),
            "measured": float(result["nondegen_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "control": "ARM_2 demotion-on E3.select diagnostics on the trained-agent eval bank",
            "met": result["nondegen_seeds"] >= MIN_PASS_SEEDS,
        },
        {
            "name": "mech449_go_nogo_withdrawal_engaged_seed_count",
            "description": (
                "MECH-449 NON-VACUITY: number of ARM_2 seeds where the active No-Go "
                "WITHDRAWAL genuinely engaged at the devalued state "
                "(go_nogo_n_soft_applied > 0) -- the Go/No-Go constitution is not a "
                "config-on no-op; the trained devaluation valuation actually withdrew the "
                "previously-preferred candidate. PER-SEED PASS COUNT (>= MIN_PASS_SEEDS). "
                "Zero withdrawals across seeds -> substrate_not_ready_requeue, NEVER a weakens."
            ),
            "measured": float(result["nogo_engaged_seeds"]),
            "threshold": float(MIN_PASS_SEEDS),
            "control": "ARM_2 devalued-state E3.select go_nogo diagnostics (injected viability No-Go)",
            "met": result["nogo_engaged_seeds"] >= MIN_PASS_SEEDS,
        },
    ]
    criteria_non_degenerate = {
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and ready,
        "C1b": bool(acc["C1b_devaluation_bias_vector_inversion"]) and ready,
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and ready,
        "C3": bool(acc["C3_silence_control"]),
    }
    criteria = [
        {"name": "C1_devaluation_behavioural_shift", "load_bearing": True, "passed": bool(acc["C1_devaluation_behavioural_shift"])},
        {"name": "C1b_devaluation_bias_vector_inversion", "load_bearing": True, "passed": bool(acc["C1b_devaluation_bias_vector_inversion"])},
        {"name": "C2_discrimination_behavioural_separation", "load_bearing": True, "passed": bool(acc["C2_discrimination_behavioural_separation"])},
        {"name": "C3_silence_control", "load_bearing": True, "passed": bool(acc["C3_silence_control"])},
    ]
    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "note": (
            "485l corrects the 485k double-vacuous FAIL by ENGAGING the MECH-449 Go/No-Go "
            "constitution (built 2026-06-21, 689g PASSED). FIX 1: the trained OFC devalued "
            "valuation is mapped to a per-candidate VIABILITY No-Go so the constitution "
            "actively WITHDRAWS the previously-preferred (now-devalued) candidate from the "
            "eligible set (the active No-Go withdrawal demotion-alone structurally cannot "
            "express) -- injected ONLY at the devaluation comparison. FIX 2: in-band "
            "re-rank gain (1.5/0.3) keeps the devalued bias differentiated WITHIN the +/-0.5 "
            "OFC clamp (485k overshot -> uniform saturation -> range 0). FIX 3: C2 "
            "discrimination runs with go_nogo_signals=None (demotion-alone, as it converted "
            "in 485j). FIX 4: the high-vs-devalued bias-VECTOR inversion (C1b) is a SCORED "
            "load-bearing DV (l2_shift > 0.1 AND cosine < 0.0). FIX 5: "
            "use_f_eligibility_adaptive_floor replaces the 485i/485j bespoke per-seed "
            "envelope-floor calibration. PRIMARY DVs measured on the COMMITTED selection "
            "through the real E3.select() (SelectionResult.selected_index) forced to the "
            "deterministic argmin-within-eligible path. Conversion is attributable to the "
            "trained-head x demotion x active-No-Go conjunction via the 3-arm dissociation: "
            "ARM_2 must convert where ARM_1 (trained head, demotion off = F-dominance "
            "ceiling) and ARM_0 (frozen head = flat devalued bias -> flat viability -> no "
            "withdrawal -> silence) do not. Readiness preconditions are PER-SEED PASS "
            "COUNTS (>= 2/3), NOT aggregate-max (the 485k bug; the meta-version of the "
            "V3-EXQ-642/643 same-statistic miss). Below the high-threat OR devalued "
            "bias-range floor OR excluded_count == 0 OR go_nogo_n_soft_applied == 0 on "
            ">= 2/3 seeds -> substrate_not_ready_requeue (non_contributory), NEVER a false "
            "SD-033b/MECH-263 weakens. Readiness-met + DVs-fail is an honest result "
            "(conversion_ceiling_persists_despite_go_nogo -> /failure-autopsy before stamping)."
        ),
    }


def _evidence_direction(result: Dict) -> str:
    if not result["readiness_met"]:
        return "non_contributory"          # substrate_not_ready_requeue, NOT a weakens
    return "supports" if result["acceptance"]["pass"] else "weakens"


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    interpretation = _interpretation(result)
    direction = _evidence_direction(result)
    outcome = "PASS" if (acc["pass"] and result["readiness_met"]) else "FAIL"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "SD-033b": direction,
            "MECH-263": direction,
        },
        "non_degenerate": bool(result["readiness_met"]),
        "degeneracy_reason": (
            None if result["readiness_met"]
            else "substrate_not_ready_requeue: ARM_2 trained-head OFC bias range below "
                 "floor at the HIGH-THREAT (C2 control) OR DEVALUED (the C1/C1b statistic) "
                 "state on >= 2/3 seeds, head untrained, the channel-adaptive MECH-448 "
                 "envelope still admitted all candidates (excluded_count == 0), OR the "
                 "MECH-449 active No-Go never engaged (go_nogo_n_soft_applied == 0) -> the "
                 "behavioural test never ran"
        ),
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "interpretation": interpretation,
        "readiness_met": result["readiness_met"],
        "head_trained_seeds": result["head_trained_seeds"],
        "deval_head_trained_seeds": result["deval_head_trained_seeds"],
        "max_test_deval_head_delta": result["max_test_deval_head_delta"],
        "ready_high_seeds": result["ready_high_seeds"],
        "ready_devalued_seeds": result["ready_devalued_seeds"],
        "eval_built_seeds": result["eval_built_seeds"],
        "ready_seeds": result["ready_seeds"],
        "nondegen_seeds": result["nondegen_seeds"],
        "nogo_engaged_seeds": result["nogo_engaged_seeds"],
        "n_seeds": result["n_seeds"],
        "max_test_head_delta": result["max_test_head_delta"],
        "max_test_bias_range": result["max_test_bias_range"],
        "max_test_bias_range_devalued": result["max_test_bias_range_devalued"],
        "max_test_deval_bias_l2_shift": result["max_test_deval_bias_l2_shift"],
        "min_test_deval_bias_cosine": result["min_test_deval_bias_cosine"],
        "max_test_excluded_count": result["max_test_excluded_count"],
        "max_test_nogo_applied": result["max_test_nogo_applied"],
        "max_test_between_context_tv": result["max_test_between_tv"],
        "max_test_deval_shift": result["max_test_deval_shift"],
        "max_ceiling_deval_shift": result["max_ceiling_deval_shift"],
        "max_ceiling_between_tv": result["max_ceiling_between_tv"],
        "arm_results": [r for arm in result["by_arm"].values() for r in arm],
        "substrate_under_test": (
            "SD-033b train_state_bias_head + ofc_bias_scale=0.5 + SD-056 "
            "e2_action_contrastive P0 + paired threat-conditioned driver with an IN-BAND "
            "485l RE-RANKING devalued-state term (inverted outcome-coupling, gain 1.5 / "
            "weight 0.3 -- defeats the 485k +/-0.5-clamp saturation), ROUTED through the "
            "MECH-448 (ARC-107) rank-preserving F->eligibility demotion-enabled E3 selector "
            "(use_f_eligibility_demotion / dn_sigma=0.0) with the channel-adaptive "
            "mean-relative envelope floor (use_f_eligibility_adaptive_floor, mean_factor=1.0 "
            "-- REPLACES the 485i/485j bespoke per-seed calibration), AND the MECH-449 "
            "(ARC-107) Go/No-Go eligibility constitution (use_go_nogo_constitution; the "
            "trained devalued valuation -> per-candidate VIABILITY No-Go -> active "
            "WITHDRAWAL of the previously-preferred candidate at the devaluation comparison)"
        ),
        "unblocks": (
            "commitment_closure:GAP-8 (SD-033b candidate -> provisional behavioural "
            "evidence -- the devaluation half via the active No-Go withdrawal, completing "
            "the 485j discrimination conversion); contributes to "
            "behavioral_diversity_isolation:GAP-I"
        ),
        "supersedes": "V3-EXQ-485l",
        "re_derive_brake_cleared": (
            "Clears the re-derive brake fired by failure_autopsy_V3-EXQ-485l (10th "
            "non_contributory autopsy on SD-033b/MECH-263): the named upstream substrate -- "
            "the OFC devaluation-head clamp DECOUPLE -- is now BUILT (ree-v3 main 758956f, "
            "ree-v3/CLAUDE.md 'SD-033b GAP-8 DECOUPLE'), so this re-test is meaningful (the "
            "decoupled-head structural redesign on built substrate, NOT another letter "
            "circling the shared-clamp ceiling -- and NOT the refused plain 485m gain-tweak)."
        ),
        "predecessor": (
            "V3-EXQ-485l (run_id v3_exq_485l_sd033b_devaluation_nogo_behavioural_"
            "20260622T063547Z_v3; confirmed FAIL/non_contributory/substrate_not_ready, "
            "failure_autopsy_V3-EXQ-485l_2026-06-22, re-derive brake FIRED). With MECH-449 "
            "built + engaged, the residual re-localised to STRUCTURAL SCALE: the single "
            "shared state_bias_head under the +/-0.5 clamp has no feasible gain band -- "
            "485k gain 4.0 SATURATED the devalued range to 0.0; 485l gain 1.5 UNDERSHOT the "
            "0.05 readout floor (0.031) -- because the same head + clamp must also carry the "
            "C2 high-threat discrimination range. The bias VECTOR inverted cleanly (l2 1.83, "
            "cosine -0.716, C1b 2/3) so DIRECTION was right, MAGNITUDE clamp-starved -> the "
            "MECH-449 No-Go viability engaged only 1/3 (< 2/3 gate). A plain 485m gain-tweak "
            "on the shared clamped head was REFUSED. Fix = DECOUPLE the devaluation head "
            "(this experiment runs on it)"
        ),
        "predecessors": [
            "V3-EXQ-485b (devaluation, representation-level)",
            "V3-EXQ-485c (task-role discrimination, representation-level)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e (trained-OFC-head behavioural; non_contributory range-starved)",
            "V3-EXQ-485f (trained-OFC-head behavioural; non_contributory readiness-gate miscalibration)",
            "V3-EXQ-485g (trained-OFC-head behavioural; substrate_ceiling, first non-vacuous test, zero conversion)",
            "V3-EXQ-485h (disambiguating redesign; FAIL/non_contributory, F-dominance conversion ceiling)",
            "V3-EXQ-485i (demotion-enabled; FAIL/substrate_not_ready_requeue, MECH-448 envelope all-admit no-op excluded_count==0)",
            "V3-EXQ-485j (demotion-envelope-calibrated; SPLIT -- C2 discrimination CONVERTED, C1 devaluation FAIL on the flat anti-range devalued head)",
            "V3-EXQ-485k (demotion + re-ranking devalued driver; FAIL/non_contributory -- clamp saturation + C2 regression; routed to MECH-449 build)",
            "V3-EXQ-485l (demotion + MECH-449 engaged on the shared head; FAIL/non_contributory/substrate_not_ready -- devalued range 0.031 < 0.05 floor, No-Go 1/3; re-derive brake FIRED -> OFC devaluation-head DECOUPLE built)",
        ],
        "notes": (
            "485m DECOUPLE: OFC outcome-devaluation arm on the SEPARATE devaluation_bias_head "
            "(clamp +/-OFC_DEVAL_BIAS_SCALE), MECH-449-ENGAGING. SUPERSEDES V3-EXQ-485l "
            "(FAIL/non_contributory/substrate_not_ready, failure_autopsy_V3-EXQ-485l_2026-06-22, "
            "re-derive brake FIRED). 485l's residual was STRUCTURAL SCALE: the single shared "
            "state_bias_head under the +/-0.5 clamp has no feasible gain band (485k saturated, "
            "485l undershot) because it also carries C2; the decoupled head gives the devalued "
            "re-ranking its OWN +/-OFC_DEVAL_BIAS_SCALE clamp so it clears the 0.05 floor "
            "without saturating, C2 untraded. The brake CLEARS because the named upstream "
            "substrate (the decouple) is now BUILT (ree-v3 main 758956f). MECH-449 framing "
            "(inherited from 485l): demotion-alone "
            "(MECH-448) converts the passive OFC discrimination signature (sig-b, 485j) but "
            "CANNOT express the active No-Go WITHDRAWAL that behavioural outcome-devaluation "
            "(sig-a) requires; the 485e->485k six-autopsy lineage cleared the ARC-107 "
            "demotion-alone-insufficient build gate, and MECH-449 (Go/No-Go eligibility "
            "constitution) was BUILT 2026-06-21 (selection-face falsifier 689g PASSED 3/3; "
            "MECH-449 promoted candidate->provisional 2026-06-22). 485l engages it: (1) the "
            "trained OFC devalued valuation -> per-candidate VIABILITY No-Go -> the "
            "constitution actively WITHDRAWS the previously-preferred candidate from the "
            "eligible set (the BG indirect/No-Go pathway; Rudebeck & Murray 2014; Mink 1996; "
            "Maia & Frank 2011), injected ONLY at the devaluation comparison; (2) the re-rank "
            "gain is lowered to 1.5/0.3 so the devalued bias stays differentiated WITHIN the "
            "+/-0.5 OFC clamp (no 485k saturation); (3) C2 discrimination runs "
            "go_nogo_signals=None (demotion-alone, as it converted in 485j) + the in-band "
            "gain reduces the driver-coupling that regressed 485k C2; (4) the high-vs-devalued "
            "bias-VECTOR inversion (devaluation_bias_l2_shift / _cosine) is promoted to a "
            "SCORED load-bearing DV (C1b); (5) use_f_eligibility_adaptive_floor "
            "(mean-relative, 689e) REPLACES the 485i/485j bespoke per-seed envelope-floor "
            "calibration. Readiness preconditions report PER-SEED PASS COUNTS (>= 2/3), NOT "
            "aggregate-max (the 485k interpretation.preconditions panel reported "
            "max-over-seeds and masked a per-seed 2-of-3 devalued-range collapse -- the "
            "meta-version of the V3-EXQ-642/643 same-statistic miss; the indexer recomputes "
            "met from measured vs threshold). PRIMARY C1 devaluation_selection_shift + C1b "
            "bias-vector inversion + C2 between-context TV measured on the COMMITTED "
            "selection through the real E3.select() (SelectionResult.selected_index). "
            "Readiness gates: OFC bias range > 0.05 at BOTH the high-threat (C2 control) AND "
            "devalued (C1) states + head trained + MECH-448 non-degeneracy (excluded_count > "
            "0) + MECH-449 non-vacuity (go_nogo_n_soft_applied > 0), each on >= 2/3 seeds, "
            "self-route substrate_not_ready_requeue below floor, NEVER a false SD-033b/"
            "MECH-263 weakens. PROMOTES NOTHING by itself; a PASS contributes provisional "
            "behavioural evidence (the devaluation half) and helps close "
            "behavioral_diversity_isolation:GAP-I. Clears a re-derive brake on SD-033b/"
            "MECH-263 (>=2 substrate_ceiling/non_contributory autopsies): the named upstream "
            "substrate MECH-449 is now BUILT/VALIDATED (689g PASS)."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    t0 = time.time()
    _preflight()
    seeds = args.seeds if args.seeds is not None else ([0] if args.dry_run else [0, 1, 2])
    result = run(seeds=seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0
    out_path, outcome = write_manifest(result, args.dry_run, elapsed)
    acc = result["acceptance"]
    interp = _interpretation(result)
    print("\n=== V3-EXQ-485m SUMMARY ===", flush=True)
    print(f"  readiness_met={result['readiness_met']} (ready_seeds={result['ready_seeds']}/{result['n_seeds']}, ready_high={result['ready_high_seeds']}/{result['n_seeds']}, ready_devalued={result['ready_devalued_seeds']}/{result['n_seeds']}, head_trained={result['head_trained_seeds']}/{result['n_seeds']}, nondegen={result['nondegen_seeds']}/{result['n_seeds']}, nogo_engaged={result['nogo_engaged_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  max_test_bias_range_high_threat={result['max_test_bias_range']:.6f} (floor {BIAS_RANGE_FLOOR})", flush=True)
    print(f"  max_test_bias_range_devalued={result['max_test_bias_range_devalued']:.6f} (floor {BIAS_RANGE_FLOOR}; the C1/C1b-routed statistic)", flush=True)
    print(f"  max_test_deval_bias_l2_shift={result['max_test_deval_bias_l2_shift']:.4f} (floor {BIAS_L2_FLOOR})  min_test_deval_bias_cosine={result['min_test_deval_bias_cosine']:.4f} (ceil {BIAS_COSINE_CEIL})", flush=True)
    print(f"  max_test_excluded_count={result['max_test_excluded_count']}  max_test_nogo_applied={result['max_test_nogo_applied']}", flush=True)
    print(f"  ARM_2 deval_shift={result['max_test_deval_shift']:.4f} vs ARM_1 ceiling={result['max_ceiling_deval_shift']:.4f}", flush=True)
    print(f"  ARM_2 between_tv={result['max_test_between_tv']:.4f} vs ARM_1 ceiling={result['max_ceiling_between_tv']:.4f}", flush=True)
    print(f"  C1_devaluation_behavioural_shift={acc['C1_devaluation_behavioural_shift']} ({acc['n_c1_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C1b_devaluation_bias_vector_inversion={acc['C1b_devaluation_bias_vector_inversion']} ({acc['n_c1b_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C2_discrimination_behavioural_separation={acc['C2_discrimination_behavioural_separation']} ({acc['n_c2_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C3_silence_control={acc['C3_silence_control']} ({acc['n_c3_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  interpretation: {interp['label']}", flush=True)
    print(f"  evidence_direction: {_evidence_direction(result)}", flush=True)
    print(f"  outcome: {outcome}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run="--dry-run" in sys.argv,
    )
