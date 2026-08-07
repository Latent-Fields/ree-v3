"""
V3-EXQ-887a: SD-014 hippocampal node-valence vector -- POST-FIX RETEST of REPRESENTATIONAL
separability and FUNCTIONAL drive-gated replay prioritisation, with the incentive-
sensitization write-path decouple ENABLED.

SUPERSEDES v3_exq_887 (2026-08-04), which FAILED C2 (|Spearman(wanting, liking)| =
0.93-0.97 vs RHO_MAX 0.90) because VALENCE_WANTING was written as
serotonin.benefit_salience(benefit_exposure) = tonic_5ht * benefit_exposure while
VALENCE_LIKING is written as raw benefit_exposure -- both monotone transforms of ONE
shared input. SD-014 (sd014-decouple-wanting-liking-writepath, IMPLEMENTED ree-v3
2026-08-07) adds a per-node, drive-coupled, saturating incentive-sensitization gain on the
WANTING write (Smith/Berridge/Aldridge 2011: DA sensitization raises wanting without
raising liking), so wanting can diverge from raw hedonic magnitude over repeated exposure.

This driver is byte-for-byte the V3-EXQ-887 instrument (its P0 synthetic-control gate, R1-R4
readiness floors, and C1/C2/C3 acceptance are UNCHANGED and remain the pre-registered
criteria) with exactly two substrate changes:
  (1) make_config() sets incentive_sensitization_enabled=True (+ sensitization_rate/max/
      coupling), routing the WANTING write through
      ResidueField.update_wanting_sensitized().
  (2) the rollout threads the homeostatic drive_level into
      agent.update_benefit_salience(benefit_exposure, drive_level) -- WITHOUT a real
      drive_level the sensitization gain stays inert, so this is load-bearing.
Everything about the DVs, the wall-independence, and the adjudication is identical to 887,
so a PASS here is a clean before/after of the decouple fix on the same validated instrument.

Claim: SD-014 (hippocampus.valence_vector_node_recording)
Backlog: EVB-0245
Design docs:
  REE_assembly/docs/claims/claims.yaml (SD-014)
  ree-v3/CLAUDE.md "SD-014 ... IMPLEMENTED 2026-04-04" write-path block

WHY THIS EXPERIMENT EXISTS
--------------------------
SD-014 has a BUILT substrate and lit_confidence 0.813 but ZERO valid experimental
evidence. Its two prior touchpoints are both non-contributory:
  - V3-EXQ-432 (the SD-014 replay-gate prioritization test): n_surprise_writes_d2=0
    across all seeds -- the surprise signal never reached the replay gate, so
    priority(node)=dot(V,d) was never exercised.
  - The EXP-0098 approach-behaviour family (V3-EXQ-514k/l/n/p/q/r, MECH-229 /
    MECH-295 / SD-049) is a wall of non_contributory reclassifications; 514q's
    drive-coupled dissociation criterion returned EXACTLY 0.0 (WL_drive ==
    WL_nodrive) on every guard-passing seed, autopsied as missing object-bound
    incentive-salience substrate plus an SP-CEM (GAP-2) confound.

This experiment is deliberately WALL-INDEPENDENT: it does NOT measure approach
rate, approach latency, or any other behavioural DV, so it cannot re-derive the
514-family ceiling. Its DVs are read off the SUBSTRATE'S OWN STATE -- the
per-node valence store, and the prioritisation function SD-014 specifies -- which
is what "representational / functional signature" means here.

WHAT IS AND IS NOT AVAILABLE (empirical probe, 2026-08-03; Step 2.5a)
--------------------------------------------------------------------
The EVB-0245 proposal is marked blocked_substrate on a 2026-07-30 reading whose
first two blocks DO NOT HOLD as of 2026-08-03. Verified by direct probe, not by
reading docs:

  (1) "NODE-VALENCE-VECTOR ABSENT" -- FALSE. RBFLayer.valence_vecs is a
      [num_centers, VALENCE_DIM] buffer (ree_core/residue/field.py:106) written
      incrementally at the nearest ACTIVE center by ResidueField.update_valence().
      The 07-30 grep was scoped to ree_core/hippocampal/ and missed
      ree_core/residue/, where the store actually lives. It has been there since
      commit 9dba0d6d ("Implement SD-014, ARC-028, MECH-105 hippocampal claims").

  (2) "REPLAY-GATE HALF BLOCKED (n_surprise_writes=0)" -- NO LONGER HOLDS. Probe
      over 12 episodes: surprise_write_count=616, 26 nodes with non-zero
      VALENCE_SURPRISE. The write fires once update_residue() is called on EVERY
      step (not only harm steps) with surprise_gated_replay=True; the V3-EXQ-432
      zero is consistent with a driver that called update_residue() on harm steps
      only. All FOUR channels acquire non-zero per-node entries under the config
      below (probe, num_resources=5, 12 eps: WANTING 21 nodes / LIKING 18 /
      HARM_DISC 24 / SURPRISE 26).

  (3) "APPROACH-BEHAVIOUR HALF BLOCKED" -- STILL TRUE, and this design does not
      go near it. No behavioural DV is measured.

SUBSTRATE DEFECT FOUND BY THIS EXPERIMENT'S PROBE -- FIXED BEFORE IT RAN
-----------------------------------------------------------------------
The 2026-08-03 Step 2.5a probe for this experiment found that
ree_core/agent.py::_do_replay built a FOUR-element drive_state
    [t5ht, 0.5, 1.0 - t5ht, surprise_weight]
and passed it to HippocampalModule._select_valence_weighted_start ->
ResidueField.get_valence_priority. Since MECH-307 (2026-05-11) widened the
valence vector to VALENCE_DIM=6, that call raised
    RuntimeError: The size of tensor a (6) must match the size of tensor b (4)
with no try/except anywhere on the path, so the AGENT-LEVEL valence-weighted
replay entry point could not run at all whenever serotonin or
surprise_gated_replay was enabled.

FIXED in ree-v3 32edd553 (2026-08-04), before this experiment ran:
  * _do_replay now builds the vector BY INDEX at VALENCE_DIM, with the MECH-307
    split-surprise slots (4, 5) deliberately zero-weighted in BOTH flag states
    (weighting them would double-count surprise -- the write site already
    mirrors the magnitude into the legacy VALENCE_SURPRISE slot);
  * ResidueField.get_valence_priority zero-PADS a genuinely legacy narrow
    drive_state and RECORDS having done so (valence_priority_pad_count /
    last_valence_priority_pad_width + a one-shot RuntimeWarning), while still
    refusing longer-than-VALENCE_DIM and non-1-D vectors.
  * Pinned by tests/contracts/test_do_replay_drive_state_width.py.

This changes nothing about the design below. SD-014's claim is about the
per-node vector and priority(node) = dot(V_node, d_current) -- not about
_do_replay's argument width -- so this experiment exercises the function with a
correctly sized 6-element drive vector whose MECH-307 slots (4,5) are zero, i.e.
exactly SD-014's 4-component prioritisation embedded in the widened field, as it
always did. P4 below is retained as a non-gating REGRESSION WITNESS: it now
MEASURES what _do_replay hands to replay (spying on the replay entry points and
on get_valence_priority) instead of restating the old 4-element literal, and it
records the pad counter -- which matters precisely because the pad means a
future re-narrowing would no longer raise.

DESIGN (single-arm, measurement-only, commitment-free)
------------------------------------------------------
Per seed:
  P0  INSTRUMENT CONTROL (run ONCE, before any rollout compute). A throwaway
      ResidueField is given a synthetic orthogonal-peak valence matrix (node i
      peaks in component i mod 4) and read back through the REAL
      get_valence_priority. The mean pairwise Kendall tau across the four
      canonical drive vectors must be LOW. This asserts the SAME STATISTIC the
      load-bearing C1 routes on, on a known-positive control, so a low measured
      tau later cannot be confused with a broken tau computation.
  P1  WAKING ROLLOUT. All four SD-014 write paths are driven every step:
        WANTING  <- agent.update_benefit_salience(benefit_exposure)   [MECH-203 SR-2]
        LIKING   <- agent.update_liking(benefit_exposure)             [SD-014 l]
        HARM     <- agent.update_harm_salience(harm_exposure)         [MECH-203 SR-2 harm-symmetric,
                    plus the automatic valence_harm_enabled write in sense()]
        SURPRISE <- agent.update_residue(harm_signal) EVERY step      [MECH-205]
  P2  FREEZE the field. Read the per-node store V_store = valence_vecs[active]
      (REPRESENTATIONAL) and the consumer-visible priorities through
      get_valence_priority at those node locations (FUNCTIONAL).
  P3  READOUT ABLATION over the SAME frozen state -- no extra compute, no
      separate training arm:
        DISSOCIATED : priority_i = dot(V_i, d)          (SD-014 as specified)
        COMPOSITE   : priority_i = c_i * sum(d), c_i = sum_j V_ij
                      (the "single composite value signal" SD-014 says is
                      insufficient)
  P4  Non-gating diagnostics, incl. the _do_replay drive-state width witness
      described above (measured off the live path, never restated).

DVs (all read off substrate state; none behavioural)
----------------------------------------------------
  C1 FUNCTIONAL drive-gating (load-bearing): mean pairwise Kendall tau between
     the node rankings induced by the four canonical drive vectors
     d_w=[1,0,0,0,0,0], d_l, d_h, d_s, computed through the REAL
     get_valence_priority. Low tau = drive state genuinely re-ranks replay
     candidates.
  C2 REPRESENTATIONAL separability (load-bearing): max |Spearman| over the six
     channel pairs of the STORED node matrix, and specifically |Spearman(w, l)|.
     SD-014: "w and l MUST be stored as separate scalars"; a rank-1 collinear
     store would mean the vector carries no more information than a scalar.
  C3 REPLAY-SET drive selectivity (load-bearing): mean pairwise Jaccard of the
     top-K(K=5) replay-priority node SETS across the four drive vectors. This is
     the operational form of "replay prioritisation is weighted by drive-state-
     gated valence relevance".

DV-SYMMETRY DECLARATION (604c check) -- ONE ARM, stated explicitly
------------------------------------------------------------------
Arm: the single frozen-field measurement arm.
  Manipulation = the drive vector d: a NON-UNIFORM per-component reweighting of
    the stored 6-vector.
  DV symmetry group = that of a rank / set-membership statistic: monotone
    transforms of the priority scores, and permutations of interchangeable nodes.
  Is the manipulation invariant under it? NO -- a per-component reweighting is a
    monotone transform of the priority scores ONLY IF the node matrix is rank-1
    collinear across components. That collinearity is not assumed: it is exactly
    what C2 MEASURES, and R1/R2 guarantee the sample is non-empty in every
    channel. So C1/C3 are measured substrate properties, not arithmetic
    identities -- conditional on a condition this run itself reports.
  The COMPOSITE readout IS invariant by arithmetic: rank(c_i * sum(d)) =
    rank(c_i) for any d with sum(d) > 0, so its mean pairwise tau is exactly 1.0
    independent of anything the substrate does. It is therefore a PRE-REGISTERED
    NULL-REFERENCE and a design self-check ONLY. It is recorded, it is asserted,
    and it MUST NOT be cited as evidence for or against SD-014.

SAMPLE-SIZE INTEGRITY
---------------------
No E3 `last_*` latched diagnostic is read in the step loop, so the
e3_diagnostics_staleness pseudo-replication hazard does not apply here. The
per-drive-pair count of NON-TIED comparable node pairs is recorded and gated
(R3), because many nodes hold exact zeros in a channel and a tau computed over
mostly-tied pairs would report a denominator it does not have.

PRE-REGISTERED ACCEPTANCE
-------------------------
Readiness gate (R1-R4). If ANY is unmet the run self-routes to
`substrate_not_ready_requeue` (experiment_purpose flips to diagnostic,
evidence_direction non_contributory, non_degenerate false). It NEVER self-routes
to a substrate-verdict label from a readiness failure.
  R1 min over the four channels of (nodes with a non-zero entry) >= 4   [floor]
  R2 nodes with any non-zero valence                          >= 8      [floor]
  R3 min over the six drive-pairs of comparable (non-tied) node pairs >= 10 [floor]
  R4 synthetic-control mean pairwise tau                       <= 0.50  [UPPER]

Given a green gate, PASS iff on EVERY seed:
  C1 mean_pairwise_tau_dissociated <= TAU_MAX (0.85)
  C2 max_channel_abs_spearman <= RHO_MAX (0.90) AND abs_spearman_w_l <= RHO_MAX
  C3 mean_pairwise_topk_jaccard <= JACCARD_MAX (0.80)
A FAIL with a green gate is a REAL refutation: the node valence vector is stored
but is collinear / drive-insensitive, i.e. a composite scalar would lose nothing.
"""

import sys
import time
import random
import argparse
import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.residue.field import (
    ResidueField,
    VALENCE_WANTING,
    VALENCE_LIKING,
    VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_SURPRISE,
    VALENCE_DIM,
)
from ree_core.utils.config import REEConfig

from experiments.pack_writer import write_flat_manifest
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_887a_sd014_node_valence_repfunc_sensitized"
SUPERSEDES_RUN_ID = "v3_exq_887_sd014_node_valence_representational_functional_20260804T022547Z_v3"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-014"]

# -- Pre-registered acceptance thresholds (constants; never derived from the run) --
TAU_MAX = 0.85          # C1 ceiling on mean pairwise Kendall tau across drive vectors
RHO_MAX = 0.90          # C2 ceiling on |Spearman| between stored channels
JACCARD_MAX = 0.80      # C3 ceiling on mean pairwise top-K Jaccard
TOP_K = 5               # replay-set size for C3

# -- Pre-registered readiness floors / ceilings (R1-R4) --
# R1 floor is 8, not a token 1-4: with ~25 valence nodes, a channel populated at
# only a handful of them leaves the C2 rank correlation dominated by ties at
# zero, so a low |Spearman| would report dissociation it did not measure. The
# 2026-08-03 probe measured the WEAKEST channel at 18 nodes after only 12
# episodes, so 8 is ~2x-margin attainable at ROLLOUT_EPISODES=40. Raising a
# READINESS floor makes the gate stricter (more likely to refuse), so it cannot
# manufacture a PASS.
CHANNEL_NODES_FLOOR = 8     # R1  per-channel non-zero node count (worst channel)
NODE_FLOOR = 8              # R2  nodes with any non-zero valence
COMPARABLE_PAIRS_FLOOR = 10  # R3  non-tied node pairs (worst drive-pair)
TAU_CONTROL_CEIL = 0.50     # R4  synthetic positive control (UPPER bound)
CONTROL_NODES = 16          # synthetic control node count

# -- Architecture dims (standard V3 CausalGridWorldV2 proxy mode) --
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

# body_obs layout (proxy mode): index 10 = harm_exposure, 11 = benefit_exposure
IDX_HARM_EXPOSURE = 10
IDX_BENEFIT_EXPOSURE = 11

# -- Rollout params --
ROLLOUT_EPISODES = 40
STEPS_PER_EPISODE = 120
SEEDS = [42, 137, 2026]     # seed 44 deliberately excluded (reef-config instability)

# -- Env params --
# These are the CausalGridWorldV2 defaults (size=10, num_hazards=3,
# hazard_harm=0.5, resource_benefit=0.3) with num_resources raised 5 -> 6. That
# is deliberately the exact configuration the 2026-08-03 readiness probe
# validated, and it is NOT the weak-signal config (hazard_harm=0.05,
# resource_benefit=0.05) some sibling drivers use: at resource_benefit=0.05 the
# EMA'd benefit_exposure peaks around ~0.01, which is below any workable
# consummatory-contact gate and would empty the LIKING channel outright -- the
# V3-EXQ-432 vacuous-zero failure mode arriving through the env config instead
# of the write path.
ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=6,
    hazard_harm=0.5,
    resource_benefit=0.3,
    use_proxy_fields=True,
)

# LIKING_THRESHOLD calibration (Step 3 "Proxy label calibration"). This is a
# SUBSTRATE gate (what counts as consummatory contact), NOT an acceptance
# threshold. benefit_exposure is an EMA of resource_benefit, and a 2026-08-03
# probe on this exact env config measured max benefit_exposure ~0.059 -- so the
# substrate default liking_threshold=0.1 is STRUCTURALLY UNREACHABLE here and
# would zero the LIKING channel outright (the V3-EXQ-432 failure mode, in a
# different channel). 0.02 is set to the measured distribution, matching the
# EMA-scale-calibration convention MECH-203's harm_salience write path already
# uses. R1 independently verifies the channel actually populated.
LIKING_THRESHOLD = 0.02

# SD-014 incentive-sensitization (V3-EXQ-887 decouple fix) -- the feature UNDER TEST.
# These are the ree_core substrate DEFAULTS (config.py), stated explicitly so the manifest
# records the operating point. drive_level is threaded into update_benefit_salience below;
# without it the gain stays inert, so this is the load-bearing difference from V3-EXQ-887.
SENSITIZATION_RATE = 0.05
SENSITIZATION_MAX = 4.0
SENSITIZATION_COUPLING = 1.0

# The four canonical drive vectors of SD-014, embedded in the MECH-307-widened
# VALENCE_DIM=6 field with the split-surprise slots (4, 5) held at zero.
DRIVE_NAMES = ["d_wanting", "d_liking", "d_harm", "d_surprise"]
DRIVE_COMPONENTS = [
    VALENCE_WANTING,
    VALENCE_LIKING,
    VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_SURPRISE,
]
CHANNEL_NAMES = ["wanting", "liking", "harm_discriminative", "surprise"]

_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------- #
# statistics helpers                                                          #
# --------------------------------------------------------------------------- #

def _drive_vector(component: int) -> torch.Tensor:
    """Canonical unit drive vector for one valence component, width VALENCE_DIM."""
    d = torch.zeros(VALENCE_DIM, dtype=torch.float32)
    d[component] = 1.0
    return d


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Kendall tau over non-tied pairs, plus the comparable-pair count.

    Computed on the RAW score vectors, not on argsort ranks: argsort would break
    exact ties by array index and spuriously inflate agreement between two
    rankings that are in truth undetermined on those pairs. Pairs tied in either
    vector are excluded from both numerator and denominator, and the denominator
    is returned so the true sample size is auditable (R3 gates on it).
    """
    n = len(a)
    if n < 2:
        return float("nan"), 0
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0.0 or db == 0.0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    if tot == 0:
        return float("nan"), 0
    return (conc - disc) / tot, tot


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dep). 0.0 on degenerate input."""
    n = len(x)
    if n < 2:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(list(x)), ranks(list(y))
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n))
    dy = sum((ry[i] - my) ** 2 for i in range(n))
    if dx <= 0 or dy <= 0:
        return 0.0
    return num / (dx * dy) ** 0.5


def _top_k_set(scores: np.ndarray, k: int) -> set:
    """Tie-safe top-K: every index whose score >= the k-th largest score.

    Avoids argsort's index-order tie-break, which would make the set depend on
    node numbering rather than on valence.
    """
    if len(scores) == 0:
        return set()
    kk = min(k, len(scores))
    cutoff = float(np.sort(scores)[::-1][kk - 1])
    return {int(i) for i in np.nonzero(scores >= cutoff)[0]}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _pairwise_tau_stats(priorities: Dict[str, np.ndarray]) -> Dict:
    """Mean pairwise Kendall tau + worst-pair comparable count over drive vectors."""
    taus = {}
    pairs_counts = {}
    for ka, kb in combinations(DRIVE_NAMES, 2):
        t, npairs = _kendall_tau(priorities[ka], priorities[kb])
        taus[f"{ka}|{kb}"] = float(t)
        pairs_counts[f"{ka}|{kb}"] = int(npairs)
    finite = [v for v in taus.values() if np.isfinite(v)]
    return {
        "pairwise_tau": taus,
        "pairwise_comparable_counts": pairs_counts,
        "mean_pairwise_tau": float(np.mean(finite)) if finite else float("nan"),
        "min_comparable_pairs": int(min(pairs_counts.values())) if pairs_counts else 0,
        "worst_pair": (min(pairs_counts, key=pairs_counts.get) if pairs_counts else ""),
    }


# --------------------------------------------------------------------------- #
# P0 instrument control                                                       #
# --------------------------------------------------------------------------- #

def _instrument_control(cfg: REEConfig) -> Dict:
    """R4: prove the tau statistic CAN move, on a known-positive control.

    Builds a throwaway ResidueField, activates CONTROL_NODES centers, writes a
    synthetic ORTHOGONAL-peak valence matrix (node i peaks in component i mod 4),
    and reads the priorities back through the REAL
    ResidueField.get_valence_priority. If the four drive vectors do NOT re-rank
    this deliberately-orthogonal field, the measurement instrument is broken and
    no low-tau reading later would be interpretable.

    Asserts the SAME statistic C1 routes on (mean pairwise Kendall tau), so this
    is a same-statistic readiness check, not a magnitude proxy for it.
    """
    field = ResidueField(cfg.residue)
    with torch.no_grad():
        field.rbf_field.valence_vecs.zero_()
        n = min(CONTROL_NODES, field.rbf_field.centers.shape[0])
        field.rbf_field.active_mask[:] = False
        field.rbf_field.active_mask[:n] = True
        # Orthogonal peaks: node i carries all of its mass in component i mod 4.
        for i in range(n):
            field.rbf_field.valence_vecs[i, DRIVE_COMPONENTS[i % 4]] = 1.0 + 0.05 * i
        centers = field.rbf_field.centers[:n].detach().clone()

    priorities = {}
    for name, comp in zip(DRIVE_NAMES, DRIVE_COMPONENTS):
        with torch.no_grad():
            p = field.get_valence_priority(centers, _drive_vector(comp))
        priorities[name] = p.detach().cpu().numpy().astype(float)

    stats = _pairwise_tau_stats(priorities)
    stats["n_control_nodes"] = int(n)
    return stats


# --------------------------------------------------------------------------- #
# rollout                                                                     #
# --------------------------------------------------------------------------- #

def make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,            # z_world fidelity (SD-008); valence localisation
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        goal_weight=1.0,
        tonic_5ht_enabled=True,     # WANTING + harm-symmetric HARM write paths
        valence_harm_enabled=True,  # automatic HARM write in sense()
        valence_liking_enabled=True,
        liking_threshold=LIKING_THRESHOLD,
        surprise_gated_replay=True,  # SURPRISE write in update_residue()
        pe_surprise_threshold=0.001,
        # SD-014 decouple fix: per-node drive-coupled incentive-sensitization on WANTING
        incentive_sensitization_enabled=True,
        sensitization_rate=SENSITIZATION_RATE,
        sensitization_max=SENSITIZATION_MAX,
        sensitization_coupling=SENSITIZATION_COUPLING,
    )


def _run_waking_rollout(agent, env, n_episodes, n_steps, seed) -> Dict:
    """Drive all four SD-014 valence write paths; return write-side counters."""
    n_steps_total = 0
    n_liking_gate_open = 0
    n_resource_contacts = 0
    max_benefit_exposure = 0.0
    max_harm_exposure = 0.0

    for ep in range(n_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [train] rollout seed={seed} ep {ep+1}/{n_episodes}", flush=True)
        obs, info = env.reset()
        agent.reset()
        body_obs = torch.as_tensor(
            obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
        world_obs = torch.as_tensor(
            obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM],
            dtype=torch.float32).unsqueeze(0)

        for _step in range(n_steps):
            latent = agent.sense(body_obs, world_obs)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            benefit_exposure = float(body_obs[0, IDX_BENEFIT_EXPOSURE])
            harm_exposure = float(body_obs[0, IDX_HARM_EXPOSURE])
            drive_level = agent.compute_drive_level(body_obs)
            max_benefit_exposure = max(max_benefit_exposure, benefit_exposure)
            max_harm_exposure = max(max_harm_exposure, harm_exposure)
            if benefit_exposure >= LIKING_THRESHOLD:
                n_liking_gate_open += 1

            # SD-014 write paths, all four channels.
            agent.serotonin_step(benefit_exposure)
            agent.update_z_goal(benefit_exposure, drive_level)
            agent.update_benefit_salience(benefit_exposure, drive_level)   # -> VALENCE_WANTING (drive-coupled sensitization)
            agent.update_harm_salience(harm_exposure)         # -> VALENCE_HARM_DISCRIMINATIVE
            agent.update_liking(benefit_exposure)             # -> VALENCE_LIKING

            action_idx = int(action.argmax(dim=-1).item())
            flat_obs, harm_signal, done, info_next, obs_dict = env.step(action_idx)
            if harm_signal is not None and harm_signal > 0:
                n_resource_contacts += 1

            # EVERY step, not only harm steps: this is the VALENCE_SURPRISE write
            # path (MECH-205), and calling it on harm steps only is the most
            # likely cause of V3-EXQ-432's n_surprise_writes=0.
            agent.update_residue(float(harm_signal) if harm_signal is not None else 0.0)

            body_obs = torch.as_tensor(
                flat_obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
            world_obs = torch.as_tensor(
                flat_obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM],
                dtype=torch.float32).unsqueeze(0)
            n_steps_total += 1
            if done:
                break

    return {
        "n_steps_total": n_steps_total,
        "n_liking_gate_open": n_liking_gate_open,
        "n_resource_contacts": n_resource_contacts,
        "max_benefit_exposure": max_benefit_exposure,
        "max_harm_exposure": max_harm_exposure,
        "surprise_write_count": int(getattr(agent, "_surprise_write_count", 0)),
    }


# --------------------------------------------------------------------------- #
# readouts                                                                    #
# --------------------------------------------------------------------------- #

def _do_replay_width_diagnostic(agent) -> Dict:
    """P4 (NON-GATING): MEASURE the drive_state the substrate itself builds.

    This probe does NOT restate _do_replay's construction. It spies on both
    hippocampal replay entry points (plain and the MECH-165 diverse one) and on
    ResidueField.get_valence_priority, then calls the agent's OWN _do_replay --
    so what is recorded is whatever the live substrate actually hands over, and
    the record cannot drift out of step with the substrate the way a restated
    literal can. Same capture technique as
    tests/contracts/test_do_replay_drive_state_width.py.

    HISTORY (why this probe exists, and why it was rewritten). Until ree-v3
    32edd553 (2026-08-04), _do_replay built a FOUR-element drive_state against a
    VALENCE_DIM=6 valence store -- MECH-307 widened the store 4 -> 6 on
    2026-05-11 and this construction was not widened with it -- so
    ResidueField.get_valence_priority raised RuntimeError, uncaught, on every
    agent-level valence-weighted replay. That defect is FIXED: the vector is now
    built BY INDEX at VALENCE_DIM with the MECH-307 split-surprise slots (4, 5)
    deliberately zero-weighted in both flag states, and get_valence_priority
    zero-PADS a genuinely legacy narrow vector (recording the pad) rather than
    raising. The first version of this probe rebuilt the 4-element literal
    itself, so post-fix it would have reported a defect that no longer exists
    while its own ok/error flags flipped for the wrong reason.

    It is kept, rather than deleted, as a live regression witness: precisely
    BECAUSE get_valence_priority now pads, a future re-narrowing of _do_replay
    would no longer raise, and the pad counter is the only thing that would
    notice. Non-gating either way -- SD-014's claim is about the per-node vector
    and priority(node) = dot(V_node, d), not about _do_replay's argument width.
    """
    captured: Dict = {}

    def _spy_replay(name):
        orig = getattr(agent.hippocampal, name)

        def wrapper(*args, **kwargs):
            captured.setdefault("replay_entry_point", name)
            captured.setdefault("drive_state", kwargs.get("drive_state"))
            return orig(*args, **kwargs)

        return wrapper

    orig_priority = agent.residue_field.get_valence_priority
    priority_widths: List[int] = []

    def _spy_priority(z_world, drive_state, *a, **kw):
        priority_widths.append(int(drive_state.numel()))
        return orig_priority(z_world, drive_state, *a, **kw)

    recent = agent.theta_buffer.recent
    pad_before = int(getattr(agent.residue_field, "valence_priority_pad_count", 0))
    out = {
        "valence_dim": int(VALENCE_DIM),
        "probe_measures_substrate": True,
        "theta_buffer_rows": int(recent.shape[0]) if recent is not None else 0,
        "pad_count_before": pad_before,
        "defect_fixed_in_commit": "ree-v3 32edd553",
        "defect_contract": "tests/contracts/test_do_replay_drive_state_width.py",
    }

    agent.hippocampal.replay = _spy_replay("replay")
    agent.hippocampal.diverse_replay = _spy_replay("diverse_replay")
    agent.residue_field.get_valence_priority = _spy_priority
    try:
        agent._do_replay(agent._current_latent)
        out["do_replay_ok"] = True
        out["error"] = ""
    except Exception as exc:  # recorded, never swallowed silently
        out["do_replay_ok"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        # Drop the instance attributes shadowing the bound methods, so the agent
        # is left exactly as it was found.
        agent.hippocampal.__dict__.pop("replay", None)
        agent.hippocampal.__dict__.pop("diverse_replay", None)
        agent.residue_field.__dict__.pop("get_valence_priority", None)

    ds = captured.get("drive_state")
    out["replay_entry_point"] = str(captured.get("replay_entry_point", ""))
    out["reached_replay_entry_point"] = bool("replay_entry_point" in captured)
    if ds is None:
        # Either _do_replay returned early (no theta-buffer window) or neither
        # switch that builds a drive_state was on. Nothing was measured -- say so
        # rather than reporting a width of 0 as if it were a reading.
        out["drive_state_built"] = False
        out["substrate_drive_state_width"] = None
        out["substrate_drive_state"] = []
        out["widths_match"] = None
    else:
        width = int(ds.numel())
        out["drive_state_built"] = True
        out["substrate_drive_state_width"] = width
        out["substrate_drive_state"] = [float(x) for x in ds.flatten().tolist()]
        out["widths_match"] = bool(width == VALENCE_DIM)

    pad_after = int(getattr(agent.residue_field, "valence_priority_pad_count", 0))
    out["n_get_valence_priority_calls"] = len(priority_widths)
    out["get_valence_priority_drive_state_widths"] = sorted(set(priority_widths))
    out["pad_count_after"] = pad_after
    out["pad_count_delta"] = pad_after - pad_before
    out["last_pad_width"] = getattr(
        agent.residue_field, "last_valence_priority_pad_width", None)
    # The consumer-side reading: the valence-weighted start selection actually
    # ran AND reached get_valence_priority at full width without needing the
    # legacy pad. A pad_count_delta > 0 here means _do_replay has re-narrowed.
    out["valence_weighted_start_reached_priority"] = bool(priority_widths)
    out["reached_priority_at_full_width"] = bool(
        priority_widths
        and all(w == VALENCE_DIM for w in priority_widths)
        and pad_after == pad_before
    )
    return out


def _read_node_state(agent) -> Dict:
    """P2/P3: representational store + functional priorities over the frozen field."""
    rbf = agent.residue_field.rbf_field
    active_idx = rbf.active_mask.nonzero(as_tuple=True)[0]
    n_active = int(active_idx.numel())
    if n_active == 0:
        return {
            "n_active_centers": 0, "n_valence_nodes": 0,
            "channel_nonzero_nodes": {c: 0 for c in CHANNEL_NAMES},
            "channel_sums": {c: 0.0 for c in CHANNEL_NAMES},
            "node_valence_matrix": [], "dissociated": {}, "composite": {},
            "topk_sets": {}, "channel_spearman": {},
        }

    with torch.no_grad():
        v_all = rbf.valence_vecs[active_idx].detach().clone()      # [n_active, 6]
        centers = rbf.centers[active_idx].detach().clone()          # [n_active, world_dim]

    v_np = v_all.cpu().numpy().astype(float)
    keep = np.nonzero(np.abs(v_np).sum(axis=1) > 0)[0]              # nodes with any valence
    v_keep = v_np[keep]
    centers_keep = centers[torch.as_tensor(keep, dtype=torch.long)]

    channel_nonzero = {
        name: int((np.abs(v_np[:, comp]) > 0).sum())
        for name, comp in zip(CHANNEL_NAMES, DRIVE_COMPONENTS)
    }
    channel_sums = {
        name: float(v_np[:, comp].sum())
        for name, comp in zip(CHANNEL_NAMES, DRIVE_COMPONENTS)
    }

    # -- C2 representational: |Spearman| across STORED channel pairs -----------
    channel_spearman = {}
    for (na, ca), (nb, cb) in combinations(
            list(zip(CHANNEL_NAMES, DRIVE_COMPONENTS)), 2):
        channel_spearman[f"{na}|{nb}"] = float(
            _spearman(v_keep[:, ca], v_keep[:, cb])) if len(v_keep) >= 2 else 0.0

    # -- C1/C3 functional: priorities through the REAL substrate read path ------
    dissociated = {}
    for name, comp in zip(DRIVE_NAMES, DRIVE_COMPONENTS):
        with torch.no_grad():
            p = agent.residue_field.get_valence_priority(
                centers_keep, _drive_vector(comp))
        dissociated[name] = p.detach().cpu().numpy().astype(float)

    # -- COMPOSITE null-reference: collapse the vector to one scalar ------------
    # rank(c_i * sum(d)) == rank(c_i) for any d with sum(d) > 0, so every drive
    # vector induces an IDENTICAL ranking. This invariance is an arithmetic
    # identity and is a design self-check ONLY -- never evidence.
    with torch.no_grad():
        val_read = agent.residue_field.evaluate_valence(centers_keep)
    composite_scalar = val_read.detach().cpu().numpy().astype(float)[:, :4].sum(axis=1)
    composite = {name: composite_scalar * 1.0 for name in DRIVE_NAMES}

    topk = {name: _top_k_set(dissociated[name], TOP_K) for name in DRIVE_NAMES}

    return {
        "n_active_centers": n_active,
        "n_valence_nodes": int(len(keep)),
        "channel_nonzero_nodes": channel_nonzero,
        "channel_sums": channel_sums,
        "node_valence_matrix": [[float(x) for x in row] for row in v_keep],
        "dissociated": dissociated,
        "composite": composite,
        "topk_sets": {k: sorted(v) for k, v in topk.items()},
        "channel_spearman": channel_spearman,
        "_topk_raw": topk,
    }


def _w_over_l_diagnostic(v_keep: List[List[float]]) -> Dict:
    """NON-GATING: incentive-trap readout (SD-014 w/h and w/l early-warning).

    Recorded generously so a successor testing the devaluation / tolerance
    signature does not have to re-run this rollout to get the distribution.
    """
    if not v_keep:
        return {"n_nodes": 0}
    arr = np.asarray(v_keep, dtype=float)
    w = arr[:, VALENCE_WANTING]
    l = arr[:, VALENCE_LIKING]
    h = arr[:, VALENCE_HARM_DISCRIMINATIVE]

    # Ratios are computed ONLY where the denominator is non-zero. An
    # epsilon-regularised ratio over l==0 nodes produces meaningless 1e7-scale
    # values that swamp the summary; the qualitative incentive-trap signature of
    # those nodes (wanting accumulated where liking never did) is reported as a
    # COUNT instead, which is what it actually is.
    def _ratio_stats(num, den, label):
        mask = den != 0
        if not np.any(mask):
            return {f"{label}_n_defined": 0}
        r = num[mask] / np.abs(den[mask])
        return {
            f"{label}_n_defined": int(mask.sum()),
            f"{label}_mean": float(np.mean(r)),
            f"{label}_max": float(np.max(r)),
            f"{label}_p90": float(np.percentile(r, 90)),
        }

    out = {
        "n_nodes": int(arr.shape[0]),
        "n_nodes_w_gt_l": int(np.sum(w > l)),
        "n_nodes_l_gt_w": int(np.sum(l > w)),
        "n_nodes_w_and_l_both_nonzero": int(np.sum((w != 0) & (l != 0))),
        # The two structural-dissociation populations: approach-only nodes
        # (wanting written, liking never) and contact-only nodes.
        "n_nodes_w_nonzero_l_zero": int(np.sum((w != 0) & (l == 0))),
        "n_nodes_l_nonzero_w_zero": int(np.sum((l != 0) & (w == 0))),
    }
    out.update(_ratio_stats(w, l, "w_over_l"))
    out.update(_ratio_stats(w, h, "w_over_h"))
    return out


# --------------------------------------------------------------------------- #
# per-seed driver                                                             #
# --------------------------------------------------------------------------- #

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    print(f"Seed {seed} Condition node_valence_repfunc")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = make_config()
    agent = REEAgent(cfg)
    agent.to(torch.device("cpu"))
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)

    n_episodes = 2 if dry_run else ROLLOUT_EPISODES
    n_steps = 25 if dry_run else STEPS_PER_EPISODE

    write_stats = _run_waking_rollout(agent, env, n_episodes, n_steps, seed)
    _ZG.observe(agent)

    state = _read_node_state(agent)
    replay_width = _do_replay_width_diagnostic(agent)

    tau_diss = _pairwise_tau_stats(state["dissociated"]) if state["dissociated"] else {
        "pairwise_tau": {}, "pairwise_comparable_counts": {},
        "mean_pairwise_tau": float("nan"), "min_comparable_pairs": 0, "worst_pair": "",
    }
    tau_comp = _pairwise_tau_stats(state["composite"]) if state["composite"] else {
        "mean_pairwise_tau": float("nan"),
    }

    # C2: worst (largest) |Spearman| over stored channel pairs, plus the w|l pair.
    sp = state["channel_spearman"]
    if sp:
        worst_pair_name = max(sp, key=lambda k: abs(sp[k]))
        max_abs_rho = float(abs(sp[worst_pair_name]))
    else:
        worst_pair_name, max_abs_rho = "", 0.0
    abs_rho_wl = float(abs(sp.get("wanting|liking", 0.0)))

    # C3: mean pairwise Jaccard of the top-K replay sets.
    topk_raw = state.get("_topk_raw", {})
    if topk_raw:
        jaccs = {f"{a}|{b}": _jaccard(topk_raw[a], topk_raw[b])
                 for a, b in combinations(DRIVE_NAMES, 2)}
        mean_jacc = float(np.mean(list(jaccs.values())))
        max_jacc = float(max(jaccs.values()))
        worst_jacc_pair = max(jaccs, key=jaccs.get)
    else:
        jaccs, mean_jacc, max_jacc, worst_jacc_pair = {}, 1.0, 1.0, ""

    min_channel_nodes = min(state["channel_nonzero_nodes"].values()) \
        if state["channel_nonzero_nodes"] else 0
    offending_channel = (
        min(state["channel_nonzero_nodes"], key=state["channel_nonzero_nodes"].get)
        if state["channel_nonzero_nodes"] else ""
    )

    c1 = bool(np.isfinite(tau_diss["mean_pairwise_tau"])
              and tau_diss["mean_pairwise_tau"] <= TAU_MAX)
    c2 = bool(max_abs_rho <= RHO_MAX and abs_rho_wl <= RHO_MAX and len(state["node_valence_matrix"]) >= 2)
    c3 = bool(mean_jacc <= JACCARD_MAX)
    seed_pass = bool(c1 and c2 and c3)
    print(f"  seed={seed} tau_diss={tau_diss['mean_pairwise_tau']:.4f} "
          f"tau_comp={tau_comp['mean_pairwise_tau']:.4f} "
          f"max_rho={max_abs_rho:.4f} rho_wl={abs_rho_wl:.4f} "
          f"mean_jaccard={mean_jacc:.4f} nodes={state['n_valence_nodes']}", flush=True)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}")

    return {
        "seed": seed,
        "write_stats": write_stats,
        "n_active_centers": state["n_active_centers"],
        "n_valence_nodes": state["n_valence_nodes"],
        "channel_nonzero_nodes": state["channel_nonzero_nodes"],
        "channel_sums": state["channel_sums"],
        "min_channel_nonzero_nodes": int(min_channel_nodes),
        "offending_channel": offending_channel,
        "node_valence_matrix": state["node_valence_matrix"],
        "channel_spearman": state["channel_spearman"],
        "max_abs_channel_spearman": max_abs_rho,
        "max_abs_channel_spearman_pair": worst_pair_name,
        "abs_spearman_wanting_liking": abs_rho_wl,
        "dissociated_pairwise_tau": tau_diss["pairwise_tau"],
        "dissociated_pairwise_comparable_counts": tau_diss["pairwise_comparable_counts"],
        "mean_pairwise_tau_dissociated": float(tau_diss["mean_pairwise_tau"]),
        "min_comparable_pairs": int(tau_diss["min_comparable_pairs"]),
        "worst_comparable_pair": tau_diss["worst_pair"],
        "mean_pairwise_tau_composite_control": float(tau_comp["mean_pairwise_tau"]),
        "topk_sets": state["topk_sets"],
        "topk_pairwise_jaccard": jaccs,
        "mean_pairwise_topk_jaccard": mean_jacc,
        "max_pairwise_topk_jaccard": max_jacc,
        "worst_jaccard_pair": worst_jacc_pair,
        "w_over_l_diagnostic": _w_over_l_diagnostic(state["node_valence_matrix"]),
        "do_replay_drive_state_width_probe": replay_width,
        "c1_functional_drive_gating": c1,
        "c2_representational_separability": c2,
        "c3_replay_set_selectivity": c3,
        "seed_pass": seed_pass,
    }


# --------------------------------------------------------------------------- #
# run                                                                         #
# --------------------------------------------------------------------------- #

def _run(dry_run: bool):
    t0 = time.perf_counter()
    print(f"{EXPERIMENT_TYPE}  dry_run={dry_run}")

    cfg_probe = make_config()

    # ---- P0: instrument control FIRST, before any rollout compute is spent ----
    control = _instrument_control(cfg_probe)
    print(f"  [P0] instrument control: mean_pairwise_tau="
          f"{control['mean_pairwise_tau']:.4f} (ceiling {TAU_CONTROL_CEIL}) "
          f"nodes={control['n_control_nodes']}", flush=True)

    preconditions = []
    gate_unmet = None
    try:
        preconditions += p0_readiness_gate([
            {
                "name": "instrument_tau_can_move_on_positive_control",
                "measured": float(control["mean_pairwise_tau"]),
                "threshold": TAU_CONTROL_CEIL,
                "direction": "upper",
                "comparator": "<=",
                "control": ("synthetic orthogonal-peak valence matrix (node i peaks in "
                            "component i mod 4) read back through the REAL "
                            "ResidueField.get_valence_priority; asserts the SAME "
                            "statistic C1 routes on (mean pairwise Kendall tau)"),
                "n_control_nodes": control["n_control_nodes"],
                "certifies": "the C1/C3 tau + ranking instrument only; says nothing about channel population",
            },
        ])
    except P0NotReady as exc:
        preconditions = list(exc.preconditions)
        gate_unmet = "instrument control could not detect re-ranking on an orthogonal-peak field"

    per_seed: List[Dict] = []
    if gate_unmet is None:
        per_seed = [run_seed(seed, dry_run=dry_run) for seed in SEEDS]

        # ---- R1-R3: worst-cell readiness over the seeds actually run ----------
        worst_channel_seed = min(per_seed, key=lambda r: r["min_channel_nonzero_nodes"])
        worst_nodes_seed = min(per_seed, key=lambda r: r["n_valence_nodes"])
        worst_pairs_seed = min(per_seed, key=lambda r: r["min_comparable_pairs"])
        try:
            preconditions += p0_readiness_gate([
                {
                    "name": "all_four_valence_channels_populated",
                    "measured": float(worst_channel_seed["min_channel_nonzero_nodes"]),
                    "threshold": float(CHANNEL_NODES_FLOOR),
                    "direction": "lower",
                    "control": ("worst channel of the worst seed; this is the direct "
                                "V3-EXQ-432 guard (n_surprise_writes=0 made that run "
                                "vacuous). A tau computed over an empty channel measures "
                                "absence, not drive-gating."),
                    "offending_cell": (f"seed={worst_channel_seed['seed']} "
                                       f"channel={worst_channel_seed['offending_channel']}"),
                    "certifies": "channel population only; does not certify the tau instrument (see R4)",
                },
                {
                    "name": "n_nodes_with_valence",
                    "measured": float(worst_nodes_seed["n_valence_nodes"]),
                    "threshold": float(NODE_FLOOR),
                    "direction": "lower",
                    "control": "worst seed; sample floor for the rank statistics",
                    "offending_cell": f"seed={worst_nodes_seed['seed']}",
                },
                {
                    "name": "min_comparable_node_pairs_across_drive_pairs",
                    "measured": float(worst_pairs_seed["min_comparable_pairs"]),
                    "threshold": float(COMPARABLE_PAIRS_FLOOR),
                    "direction": "lower",
                    "control": ("worst drive-pair of the worst seed; ties are excluded "
                                "from tau, so this is the TRUE denominator rather than "
                                "the node count"),
                    "offending_cell": (f"seed={worst_pairs_seed['seed']} "
                                       f"pair={worst_pairs_seed['worst_comparable_pair']}"),
                },
            ])
        except P0NotReady as exc:
            preconditions += list(exc.preconditions)
            gate_unmet = "; ".join(
                f"{p['name']} measured={p.get('measured')} threshold={p.get('threshold')}"
                for p in exc.preconditions if not p.get("met", True)
            ) or "readiness floors unmet"

    # ---- adjudication ------------------------------------------------------- #
    all_c1 = bool(per_seed) and all(r["c1_functional_drive_gating"] for r in per_seed)
    all_c2 = bool(per_seed) and all(r["c2_representational_separability"] for r in per_seed)
    all_c3 = bool(per_seed) and all(r["c3_replay_set_selectivity"] for r in per_seed)

    # Design self-check (NOT evidence): the composite null-reference must be an
    # exact arithmetic identity. If it is not 1.0 the readout ablation is
    # mis-implemented and the whole comparison is meaningless.
    composite_identity_ok = bool(per_seed) and all(
        (not np.isfinite(r["mean_pairwise_tau_composite_control"]))
        or abs(r["mean_pairwise_tau_composite_control"] - 1.0) < 1e-9
        for r in per_seed
    )

    degeneracy = check_degeneracy({
        "mean_pairwise_tau_dissociated": [
            r["mean_pairwise_tau_dissociated"] for r in per_seed
            if np.isfinite(r["mean_pairwise_tau_dissociated"])
        ] or [0.0, 0.0],
        "mean_pairwise_topk_jaccard": [
            r["mean_pairwise_topk_jaccard"] for r in per_seed] or [1.0, 1.0],
    }) if per_seed else {"non_degenerate": False, "degeneracy_reason": "no seeds run",
                         "degenerate_metrics": {}}

    manifest_purpose = EXPERIMENT_PURPOSE
    if gate_unmet is not None:
        # Readiness failure NEVER self-routes to a substrate-verdict label.
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        manifest_purpose = "diagnostic"
        interp_label = "substrate_not_ready_requeue"
        interp = (f"SD-014 node-valence measurement preconditions unmet ({gate_unmet}); "
                  f"re-queue under a NEW letter at an adequate P0. This is NOT a "
                  f"substrate ceiling and NOT a refutation of SD-014.")
        degeneracy["non_degenerate"] = False
        degeneracy["degeneracy_reason"] = (
            (degeneracy.get("degeneracy_reason") or "") + "; readiness gate unmet: " + gate_unmet
        ).strip("; ")
    elif not composite_identity_ok:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        manifest_purpose = "diagnostic"
        interp_label = "readout_ablation_misimplemented"
        interp = ("the COMPOSITE null-reference did not return the arithmetic identity "
                  "tau=1.0; the DISSOCIATED-vs-COMPOSITE comparison is not interpretable")
        degeneracy["non_degenerate"] = False
        degeneracy["degeneracy_reason"] = "composite null-reference identity violated"
    elif all_c1 and all_c2 and all_c3:
        outcome = "PASS"
        evidence_direction = "supports"
        interp_label = "node_valence_separable_and_drive_gated"
        interp = ("the per-node valence vector is stored non-collinearly (w and l "
                  "separately recoverable) AND drive state genuinely re-ranks replay "
                  "priority and the top-K replay set -- information a single composite "
                  "value scalar provably cannot carry")
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        interp_label = "node_valence_collapsible_to_composite"
        failed = [n for n, ok in
                  (("C1 functional drive-gating", all_c1),
                   ("C2 representational separability", all_c2),
                   ("C3 replay-set selectivity", all_c3)) if not ok]
        interp = ("with all four channels populated and the tau instrument verified, "
                  + ", ".join(failed) + " failed: the node valence vector carries no "
                  "more usable information than a composite scalar on this substrate")

    print(f"\n  gate_green={gate_unmet is None}  composite_identity_ok={composite_identity_ok}")
    print(f"  C1 functional drive-gating (all seeds)     = {all_c1}")
    print(f"  C2 representational separability (all seeds)= {all_c2}")
    print(f"  C3 replay-set selectivity (all seeds)      = {all_c3}")
    print(f"  non_degenerate={degeneracy.get('non_degenerate')}")
    print(f"  VERDICT: {outcome}  direction={evidence_direction}  label={interp_label}")

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "tau_max": TAU_MAX,
        "rho_max": RHO_MAX,
        "jaccard_max": JACCARD_MAX,
        "top_k": TOP_K,
        "channel_nodes_floor": CHANNEL_NODES_FLOOR,
        "node_floor": NODE_FLOOR,
        "comparable_pairs_floor": COMPARABLE_PAIRS_FLOOR,
        "tau_control_ceil": TAU_CONTROL_CEIL,
        "control_nodes": CONTROL_NODES,
        "rollout_episodes": ROLLOUT_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "seeds": SEEDS,
        "env": dict(ENV_KWARGS),
        "alpha_world": 0.9,
        "tonic_5ht_enabled": True,
        "valence_harm_enabled": True,
        "valence_liking_enabled": True,
        "liking_threshold": LIKING_THRESHOLD,
        "incentive_sensitization_enabled": True,
        "sensitization_rate": SENSITIZATION_RATE,
        "sensitization_max": SENSITIZATION_MAX,
        "sensitization_coupling": SENSITIZATION_COUPLING,
        "surprise_gated_replay": True,
        "pe_surprise_threshold": 0.001,
        "drive_vector_width": int(VALENCE_DIM),
        "drive_vector_note": (
            "drive vectors are width VALENCE_DIM=6 with the MECH-307 split-surprise "
            "slots (4,5) held at zero -- i.e. SD-014's 4-component prioritisation "
            "embedded in the widened field. A 4-element vector no longer raises "
            "against the 6-wide store: as of ree-v3 32edd553 (2026-08-04) "
            "ResidueField.get_valence_priority zero-PADS a legacy narrow vector and "
            "records having done so (valence_priority_pad_count / "
            "last_valence_priority_pad_width + a one-shot RuntimeWarning), while "
            "still refusing longer-than-VALENCE_DIM and non-1-D vectors. The vectors "
            "built here are full width, so they never take that path; "
            "do_replay_drive_state_width_probe records the pad counter per seed as "
            "the regression witness."
        ),
    }

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "backlog_id": "EVB-0245",
        "supersedes": SUPERSEDES_RUN_ID,
        "experiment_purpose": manifest_purpose,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "mechanism_validation",
        "timestamp_utc": ts,
        "interpretation_note": interp,
        "interpretation": {
            "label": interp_label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1": bool(gate_unmet is None and per_seed),
                "C2": bool(gate_unmet is None and per_seed),
                "C3": bool(gate_unmet is None and per_seed),
            },
        },
        "criteria": [
            {"name": "C1_functional_drive_gating", "load_bearing": True, "passed": all_c1},
            {"name": "C2_representational_separability", "load_bearing": True, "passed": all_c2},
            {"name": "C3_replay_set_selectivity", "load_bearing": True, "passed": all_c3},
            {"name": "composite_null_reference_identity", "load_bearing": False,
             "passed": composite_identity_ok,
             "note": "design self-check; an arithmetic identity, NEVER evidence"},
        ],
        "combination_rule": "PASS iff C1 AND C2 AND C3 hold on EVERY seed, given a green readiness gate",
        "acceptance": {
            "c1_functional_drive_gating_all_seeds": all_c1,
            "c2_representational_separability_all_seeds": all_c2,
            "c3_replay_set_selectivity_all_seeds": all_c3,
            "composite_null_reference_identity_ok": composite_identity_ok,
            "readiness_gate_green": gate_unmet is None,
        },
        "instrument_control": control,
        "per_seed_results": per_seed,
        "dv_symmetry_note": (
            "ONE arm (the frozen-field measurement arm). Manipulation = the drive "
            "vector d, a NON-UNIFORM per-component reweighting -- not a broadcast "
            "additive constant, not a monotone rescaling, not a permutation. The DV "
            "(node rank / top-K set membership) is therefore NOT invariant under it, "
            "EXCEPT in the degenerate case where the stored node matrix is rank-1 "
            "collinear across components -- which is not assumed but is precisely what "
            "C2 measures and R1/R2 guard the sample for. The COMPOSITE readout IS "
            "invariant by arithmetic (rank(c_i*sum(d)) = rank(c_i) for sum(d)>0); that "
            "invariance is the PRE-REGISTERED NULL-REFERENCE and a design self-check "
            "ONLY, and must not be cited as evidence."
        ),
        "wall_independence_note": (
            "No behavioural DV is measured (no approach rate, latency, or choice). "
            "The 514-family / EXP-0098 substrate ceiling therefore cannot be "
            "re-derived by this design; its DVs read the substrate's own node store "
            "and prioritisation function."
        ),
        "known_substrate_defect_note": (
            "FIXED before this run -- recorded so the defect's history is legible, NOT "
            "as a live finding. ree_core/agent.py::_do_replay used to build a 4-element "
            "drive_state against a VALENCE_DIM=6 field (widened by MECH-307, "
            "2026-05-11), so the agent-level valence-weighted replay entry point raised "
            "RuntimeError with no try/except; the defect was found by this "
            "experiment's own Step 2.5a substrate probe on 2026-08-03. As of ree-v3 "
            "32edd553 (2026-08-04) _do_replay builds the vector BY INDEX at "
            "VALENCE_DIM with the MECH-307 split-surprise slots (4,5) deliberately "
            "zero-weighted in both flag states, and ResidueField.get_valence_priority "
            "zero-pads-and-records a legacy narrow vector instead of raising. Pinned by "
            "tests/contracts/test_do_replay_drive_state_width.py. "
            "do_replay_drive_state_width_probe now MEASURES the live path per seed "
            "(spying on the hippocampal replay entry points and on "
            "get_valence_priority) rather than restating the old construction, so it "
            "stands as a regression witness: because the pad means a re-narrowing "
            "would no longer raise, pad_count_delta is the only thing that would "
            "notice. SD-014's own claim (the node vector and priority(node)=dot(V,d)) "
            "is exercised here with a correctly sized vector either way."
        ),
        "custom_information": {
            "probe_2026_08_03": (
                "Step 2.5a probe confirmed all four SD-014 channels populate "
                "(WANTING 21 / LIKING 18 / HARM_DISC 24 / SURPRISE 26 nodes over 12 "
                "episodes at num_resources=5) and that surprise_write_count=616, "
                "lifting the EVB-0245 blocked_substrate blocks (1) and (2)."
            ),
            "gov_reuse_1_check": (
                "reanalysis_query over readouts valence_wl_node_corr / valence_vecs / "
                "replay_priority / node_valence / wanting_liking_corr matched 4 "
                "SD-014-tagged manifests (v3_exq_432, v3_exq_493 x3); ALL carry no "
                "substrate_hash (pre-standard -> UNVERIFIABLE) and 0 carry the decisive "
                "readout. Not recoverable -> run."
            ),
        },
    }
    manifest.update(degeneracy)

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"  manifest -> {out_path}")
    return outcome, out_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path, _run_id = _run(args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        dry_run=args.dry_run,
    )
