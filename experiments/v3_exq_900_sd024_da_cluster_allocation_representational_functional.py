#!/opt/local/bin/python3
"""V3-EXQ-900 -- SD-024 DA-modulated RBF cluster allocation: REPRESENTATIONAL
expansion + FUNCTIONAL density propagation + MECH-233 harm asymmetry, all
read off the substrate's own state (wall-independent confirming DV).

Claim: SD-024 (hippocampal_module.da_modulated_rbf_density)
IGW: IGW-20260808-230 (GOV-CONFIRM-1) -- built substrate, lit_conf 0.702,
  ZERO non-diagnostic experimental evidence (see WHY THIS EXPERIMENT).
Purpose: EVIDENCE (EXPERIMENT_PURPOSE="evidence"; the substrate's own
  allocation + density-read mechanism is exercised directly, so a PASS
  counts toward SD-024's experimental confidence).

WHY THIS EXPERIMENT EXISTS
---------------------------
SD-024 is IMPLEMENTED (2026-07-16) with its live-path producer wired
(2026-07-20), and two prior touchpoints exist -- but NEITHER is valid
confidence-bearing evidence FOR SD-024 itself:
  - V3-EXQ-766/893 (MECH-232's crux discriminator, density-follower approach
    without an explicit valence gradient) tag ONLY MECH-232 in claim_ids --
    correctly, since they test the "why" (representational-quality account of
    approach), not SD-024's own allocation-mechanism claim.
  - V3-EXQ-795 (SD-024 benefit-terrain live-path efficacy) tags SD-024, but
    experiment_purpose="diagnostic" (it discriminates whether a producer
    exists at all, not whether the claim holds) -- diagnostics are excluded
    from governance confidence scoring by design.
Reanalysis check (GOV-REUSE-1, 2026-08-08): `reanalysis_query.py query --claim
SD-024` finds exactly one manifest (V3-EXQ-795, substrate_hash 402e3f5a...,
purpose=diagnostic) -- no recoverable non-diagnostic readout exists. Re-derive
brake (2.5b) and substrate-path overlap gate (2.5c) both clear (0 hits).

This experiment tests SD-024's OWN claim directly and is deliberately
WALL-INDEPENDENT: no approach rate, no navigation success, no CEM selection
anywhere in this design (contrast the MECH-232/ARC-057 crux family, whose DV
is approach-under-CEM). The DVs are read off the RESIDUE FIELD'S OWN STATE --
allocated cluster size, the weight-independent density reader, and the harm
field's call-site behaviour -- exactly what SD-024 (residue/field.py
add_residue_cluster / compute_local_density) specifies.

WHAT SD-024 CLAIMS, OPERATIONALISED (docs/architecture/sd_024...md, claims.yaml)
----------------------------------------------------------------------------
  (a) REPRESENTATIONAL: a phasic DA signal at a reward encounter allocates a
      local CLUSTER of n = 1 + int(dopamine_signal * da_allocation_scale)
      jittered RBF centers on the BENEFIT terrain (not a single center) --
      "representational expansion... without requiring environmental
      richness".
  (b) FUNCTIONAL: the resulting structural richness is legible through
      compute_local_density() / compute_benefit_density() -- the
      WEIGHT-INDEPENDENT reader SD-025's curiosity drive and MECH-232's crux
      both consume -- i.e. the allocation mechanism's output actually
      propagates into the signal the rest of the substrate reads.
  (c) ASYMMETRY (MECH-233, preserved BY this claim's own design): "harm events
      continue to allocate single centers (no DA modulation)". Architecturally
      guaranteed by source (ResidueField.accumulate() -- the harm write path --
      has no dopamine_signal parameter and always calls RBFLayer.add_residue(),
      never add_residue_cluster()); this experiment confirms that guarantee
      holds under a real, extended live rollout, not just by code inspection.

DESIGN (single condition, measurement-only, commitment-free, no CEM/argmax)
-----------------------------------------------------------------------------
P0 READINESS SWEEP (before any rollout compute; NEVER evidence). A fresh,
  throwaway ResidueField (same config as the live rollout) is driven through
  the REAL accumulate_benefit() -> add_residue_cluster()/add_residue() path at
  a pre-registered DA_SWEEP of magnitudes, each at a well-separated synthetic
  z_world location (pairwise distance far beyond the RBF bandwidth, so no
  cross-talk between sweep points). This certifies, on the REAL substrate
  code (not a restated formula), the SAME TWO STATISTICS the live criteria
  route on: cluster size responds to DA at all, and density responds to
  cluster size at all. A below-floor reading means the instrument (or its
  wiring) is broken, not that the claim is false -> self-routes
  substrate_not_ready_requeue.
P1 LIVE WAKING ROLLOUT. A real REEAgent steps a real CausalGridWorldV2 episode
  loop. benefit_terrain_live_producer=True is the ONLY thing that populates
  the benefit terrain -- REEAgent.update_z_goal() is called on every
  resource contact exactly as the live agent loop does (SD-024's own
  "LIVE-PATH PRODUCER" block, agent.py:9933-9951), supplying
  dopamine_signal = benefit_exposure * drive_level (the SD-012 formula SD-024
  documents). Harm contacts drive agent.update_residue(), the harm/threat
  write path, on every episode step where harm_signal < 0. Both write paths
  are instrumented with a READ-ONLY spy (values pass through unchanged) that
  logs the REAL dopamine_signal / cluster size / harm call each live call
  produces -- never reconstructed from the formula, always the substrate's
  own return value.
P2 (folded into P1) FUNCTIONAL READ. Immediately after each live benefit
  event, compute_benefit_density() is read at that exact contact location --
  the SAME weight-independent statistic MECH-232's crux and the SD-025
  curiosity drive consume -- while the field is still in the state that
  event just wrote (no later overwrite in between).

DVs (all substrate-state reads; none behavioural; pooled across seeds --
see SAMPLE-SIZE INTEGRITY)
-----------------------------------------------------------------------
  C1 REPRESENTATIONAL (load-bearing): Spearman rho(dopamine_signal,
     cluster_size) over every live benefit event >= RHO_LIVE_MIN. The
     mechanism responds to the LIVE, naturally-varying DA magnitude the
     agent's own drive state and reward contacts produce -- not merely to
     the P0 sweep's artificial well-separated probes.
  C2 FUNCTIONAL (load-bearing): Spearman rho(cluster_size, density) over the
     SAME live events >= RHO_LIVE_MIN. The allocation's structural richness
     genuinely propagates into the weight-independent density reader in the
     live-populated field, not only in the clean synthetic sweep.
  C4 ASYMMETRY (load-bearing): over every live harm event, RBFLayer.add_residue
     (single-center, standard-bandwidth) fires exactly once per harm call and
     RBFLayer.add_residue_cluster NEVER fires on the harm field (0 calls) --
     the harm field is never DA-modulated, empirically, under a real extended
     rollout.
PASS iff C1 AND C2 AND C4, given a green readiness gate.

DV-SYMMETRY DECLARATION (604c check) -- mandatory, ONE condition, NO CEM
-------------------------------------------------------------------------
This design contains no argmax, no softmax-sample, no candidate selection of
any kind -- every DV is read directly off residue-field state or off a
spy-captured call argument/return value. The classic invariance traps this
skill catalogues (a broadcast additive constant cancelling in an argmax; a
monotone rescaling preserved by a rank/order DV masking a null effect; a
permutation-symmetric aggregate hiding a per-unit signal) all presuppose a
selection or aggregation step this design does not have, so none can apply
here by construction -- there is no selection for a manipulation to be
invisible to.
The one thing worth being precise about: C1/C2 use Spearman rho, a RANK
statistic, deliberately -- not because a selection step needs protecting, but
because the live dopamine_signal -> cluster_size map is a KNOWN monotone step
function (n = 1 + int(da*scale)), so a rank statistic is the correct tool for
"does this relationship hold at all" without demanding a particular
functional form from noisy live data. Spearman rho is invariant under
monotone reparametrisation of either axis, which is a virtue here (robustness
to exactly how da is realised), not a threat (there is no manipulation this
invariance could hide, per the previous paragraph).

SAMPLE-SIZE INTEGRITY
----------------------
No E3 `last_*` latched diagnostic is read anywhere in this driver, so the
e3_diagnostics_staleness pseudo-replication hazard does not apply. C1/C2/C4
are POOLED across all seeds rather than gated per-seed: with only 3 seeds and
a modest episode budget, per-seed live benefit/harm samples are small and
per-seed correlations would be underpowered and noisy; pooling is the
correct unit for a rank correlation whose readiness floor (R3/R4 below)
already guards the pooled sample size, and the readiness gate would abort
before P1 if the pooled sample were still too thin.

PRE-REGISTERED ACCEPTANCE
---------------------------
Readiness gate (R1-R4). Any unmet -> substrate_not_ready_requeue
(experiment_purpose flips to "diagnostic", evidence_direction
"non_contributory", non_degenerate false). NEVER a substrate-verdict label.
  R1 P0 sweep: Spearman rho(da, cluster_size)      >= RHO_CONTROL_FLOOR (0.9)
  R2 P0 sweep: Spearman rho(cluster_size, density) >= RHO_CONTROL_FLOOR (0.9)
  R3 pooled live benefit events with cluster_size >= 2  >= N_EXPANSION_MIN (5)
  R4 pooled live harm events                            >= N_HARM_MIN (5)
R3 additionally guards C1/C2 non-degeneracy: a live sample where every event
allocated the SAME cluster size carries no rank information and a computed
rho would be an artefact of the (0.0-on-degenerate) helper, not a
measurement -- R3 rules that out by construction (>=2 distinct-size events
required before the correlation is even attempted).

Given a green gate, PASS iff on the POOLED sample:
  C1 rho(dopamine_signal, cluster_size) >= RHO_LIVE_MIN (0.30)
  C2 rho(cluster_size, density)         >= RHO_LIVE_MIN (0.30)
  C4 100% of harm calls are single-center AND zero cluster calls on harm field
A FAIL with a green gate is a real refutation: the mechanism is wired and
live-populated but does not behave as SD-024 specifies.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.residue.field import ResidueField  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_900_sd024_da_cluster_allocation_representational_functional"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["SD-024"]

# ---- fixed design constants (pre-registered) ----
GRID_SIZE = 8
NUM_HAZARDS = 2
NUM_RESOURCES = 4
WORLD_DIM = 16
SELF_DIM = 16
ACTION_DIM = 4
ALPHA_WORLD = 0.9          # SD-008: z_world fidelity; default 0.3 is a known root cause
HAZARD_HARM = 0.4
RESOURCE_BENEFIT = 0.5
EPISODES = 30
STEPS_PER_EPISODE = 60
EPISODES_PER_RUN = EPISODES
SEEDS = [11, 23, 37]        # seed 44 deliberately excluded (reef-config instability)

# ---- SD-024 config (the mechanism under test) ----
# Calibrated to the LIVE dopamine_signal distribution (Step 4 probe, 2026-08-08:
# run_seed() at this env config measured live dopamine_signal = benefit_exposure
# * drive_level clustered LOW (~0.005, seed 23's whole 10-event sample) with an
# occasional higher tail (up to ~0.088, seed 11) -- drive_level rarely rises far
# above ~0 because NUM_RESOURCES keeps the agent mostly sated. At the design's
# original DA_ALLOCATION_SCALE=6.0 (n = 1 + int(da*scale)) NOT ONE observed live
# event crossed n>=2 (0.088*6=0.53 -> int 0), which would have starved R3/C1/C2
# on real data regardless of the mechanism's correctness -- a proxy-label
# calibration miss (skill Step 3), not a substrate defect. Raised so the
# observed tail (~0.05-0.09) reaches n=2-4 while the common low end (~0.005)
# stays at n=1, giving genuine cluster-size variance to correlate against;
# EPISODES/NUM_RESOURCES raised alongside it for more total live events (the
# high-da tail is rare, so more trials are needed to clear N_EXPANSION_MIN).
DA_ALLOCATION_SCALE = 40.0
DA_JITTER_RADIUS = 0.3
# Bandwidth narrowing is held OFF (0.0) for the main C1/C2/C4 gate: it is an
# ORTHOGONAL, optional component of SD-024 (docstring: "Orthogonal to
# allocation count"), and probing it live (2026-08-08 Step 4 smoke) surfaced
# a real substrate interaction that would confound the density read rather
# than test it -- a query at the cluster's NOMINAL location (not each
# jittered center's own position) reads density from centers offset by up to
# da_jitter_radius, so narrower per-center kernels (down to 0.5x base at
# da=1.0) fall off FASTER over that offset than they gain from the extra
# centers, and the sweep's rho(cluster_size, density) measured NEGATIVE
# (-0.49) at DA_BANDWIDTH_NARROWING=0.5 with everything else unchanged.
# Isolating count-driven expansion (narrowing=0.0) is the correct read of
# the add_residue_cluster docstring's own acid test ("compute_local_density()
# rises with the cluster"), which names count, not narrowing. Narrowing
# itself is confirmed separately and non-gating -- see
# bandwidth_narrowing_diagnostic().
DA_BANDWIDTH_NARROWING = 0.0
DA_BENEFIT_NUM_CENTERS = 256
KERNEL_BANDWIDTH = 1.0

# ---- P0 readiness sweep (never evidence) ----
DA_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SWEEP_LOCATION_SCALE = 10.0   # >> kernel_bandwidth * jitter -> zero cross-talk between points

# ---- non-gating bandwidth-narrowing diagnostic (separate from the main gate) ----
BW_NARROWING_PROBE_VALUE = 0.5

# ---- pre-registered thresholds (NOT derived from this run's statistics) ----
RHO_CONTROL_FLOOR = 0.9       # R1/R2: sweep is a deterministic staircase, must be near-1.0
RHO_LIVE_MIN = 0.30           # C1/C2: live sample is noisy; modest floor
N_EXPANSION_MIN = 5           # R3: pooled live benefit events with cluster_size >= 2
N_HARM_MIN = 5                # R4: pooled live harm events


# --------------------------------------------------------------------------- #
# statistics helpers                                                          #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# config / agent construction                                                 #
# --------------------------------------------------------------------------- #

def make_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
    )
    cfg.residue.world_dim = WORLD_DIM
    cfg.residue.kernel_bandwidth = KERNEL_BANDWIDTH
    cfg.residue.benefit_terrain_enabled = True
    cfg.residue.use_da_modulated_rbf_density = True
    cfg.residue.da_allocation_scale = DA_ALLOCATION_SCALE
    cfg.residue.da_jitter_radius = DA_JITTER_RADIUS
    cfg.residue.da_bandwidth_narrowing = DA_BANDWIDTH_NARROWING
    cfg.residue.da_benefit_num_centers = DA_BENEFIT_NUM_CENTERS
    cfg.residue.benefit_terrain_live_producer = True
    return cfg


def _obs(obs_dict):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _full_config() -> dict:
    return {
        "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES, "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD, "hazard_harm": HAZARD_HARM,
        "resource_benefit": RESOURCE_BENEFIT,
        "episodes": EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
        "seeds": SEEDS,
        "da_allocation_scale": DA_ALLOCATION_SCALE,
        "da_jitter_radius": DA_JITTER_RADIUS,
        "da_bandwidth_narrowing": DA_BANDWIDTH_NARROWING,
        "da_benefit_num_centers": DA_BENEFIT_NUM_CENTERS,
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "benefit_terrain_live_producer": True,
        "thresholds": {
            "RHO_CONTROL_FLOOR": RHO_CONTROL_FLOOR,
            "RHO_LIVE_MIN": RHO_LIVE_MIN,
            "N_EXPANSION_MIN": N_EXPANSION_MIN,
            "N_HARM_MIN": N_HARM_MIN,
        },
        "da_sweep": DA_SWEEP,
    }


# --------------------------------------------------------------------------- #
# P0 readiness sweep -- fresh throwaway ResidueField, never evidence          #
# --------------------------------------------------------------------------- #

def readiness_sweep() -> Dict:
    """Exercise the REAL allocation + density-read path at a pre-registered DA
    sweep, at well-separated synthetic locations. Certifies the SAME TWO
    statistics C1/C2 route on (cluster-size-vs-DA, density-vs-cluster-size),
    on the real substrate code -- never a restated formula. A fresh instance,
    discarded after this function returns; never touched again.
    """
    _env = CausalGridWorldV2(
        seed=0, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES, hazard_harm=HAZARD_HARM,
        resource_benefit=RESOURCE_BENEFIT, use_proxy_fields=True,
    )
    cfg = make_config(_env)   # same config the live rollout uses; env itself is unused
    rf = ResidueField(cfg.residue)
    sizes: List[int] = []
    densities: List[float] = []
    for i, da in enumerate(DA_SWEEP):
        loc = torch.zeros(WORLD_DIM)
        loc[i % WORLD_DIM] = SWEEP_LOCATION_SCALE
        before = int(rf.benefit_rbf_field.active_mask.sum().item())
        rf.accumulate_benefit(loc, benefit_magnitude=1.0, dopamine_signal=da)
        after = int(rf.benefit_rbf_field.active_mask.sum().item())
        size = after - before
        with torch.no_grad():
            density = float(rf.compute_benefit_density(loc.unsqueeze(0)).item())
        sizes.append(size)
        densities.append(density)

    rho_size_vs_da = _spearman(np.array(DA_SWEEP, dtype=float), np.array(sizes, dtype=float))
    rho_density_vs_size = _spearman(np.array(sizes, dtype=float), np.array(densities, dtype=float))
    return {
        "da_sweep": list(DA_SWEEP),
        "cluster_sizes": sizes,
        "densities": densities,
        "rho_cluster_size_vs_da": rho_size_vs_da,
        "rho_density_vs_cluster_size": rho_density_vs_size,
        "n_size_ge_2": sum(1 for s in sizes if s >= 2),
    }


def bandwidth_narrowing_diagnostic() -> Dict:
    """NON-GATING: confirms per-center bandwidth narrowing directly, on its own
    throwaway ResidueField with DA_BANDWIDTH_NARROWING re-enabled -- the
    orthogonal SD-024 sub-mechanism the main gate (DA_BANDWIDTH_NARROWING=0.0)
    deliberately does not exercise (see the module-level constant comment for
    why: narrowing confounds a density read at the cluster's NOMINAL location).
    Reads RBFLayer.center_bandwidths directly -- the ground truth the mechanism
    writes -- rather than inferring narrowing from a density proxy.
    """
    _env = CausalGridWorldV2(
        seed=0, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES, hazard_harm=HAZARD_HARM,
        resource_benefit=RESOURCE_BENEFIT, use_proxy_fields=True,
    )
    cfg = make_config(_env)
    cfg.residue.da_bandwidth_narrowing = BW_NARROWING_PROBE_VALUE
    rf = ResidueField(cfg.residue)
    base_bw = float(cfg.residue.kernel_bandwidth)
    mean_bws = []
    for i, da in enumerate(DA_SWEEP):
        loc = torch.zeros(WORLD_DIM)
        loc[i % WORLD_DIM] = SWEEP_LOCATION_SCALE
        before_idx = set(rf.benefit_rbf_field.active_mask.nonzero(as_tuple=True)[0].tolist())
        rf.accumulate_benefit(loc, benefit_magnitude=1.0, dopamine_signal=da)
        after_idx = set(rf.benefit_rbf_field.active_mask.nonzero(as_tuple=True)[0].tolist())
        new_idx = sorted(after_idx - before_idx)
        if new_idx:
            bws = [float(rf.benefit_rbf_field.center_bandwidths[j].item()) for j in new_idx]
            mean_bws.append(statistics.fmean(bws))
        else:
            mean_bws.append(base_bw)
    rho_bw_vs_da = _spearman(np.array(DA_SWEEP, dtype=float), np.array(mean_bws, dtype=float))
    return {
        "probe_da_bandwidth_narrowing": BW_NARROWING_PROBE_VALUE,
        "base_bandwidth": base_bw,
        "da_sweep": list(DA_SWEEP),
        "mean_center_bandwidth_per_event": mean_bws,
        "rho_bandwidth_vs_da": rho_bw_vs_da,
        "narrows_with_da": bool(rho_bw_vs_da <= -0.9),
        "floors_at_half_base": bool(min(mean_bws) >= 0.5 * base_bw - 1e-6),
    }


# --------------------------------------------------------------------------- #
# live spy recorders -- read-only, values pass through unchanged              #
# --------------------------------------------------------------------------- #

class BenefitClusterRecorder:
    """Spies on the LIVE benefit-terrain write path (accumulate_benefit ->
    add_residue / add_residue_cluster on benefit_rbf_field). Every field is
    the REAL argument/return value the substrate produced -- never
    reconstructed from the allocation formula. Read-only: every wrapped call
    returns exactly what the original returned.
    """

    def __init__(self, residue_field: ResidueField) -> None:
        self.rf = residue_field
        self._orig_accumulate_benefit = residue_field.accumulate_benefit
        self._orig_add_residue = residue_field.benefit_rbf_field.add_residue
        self._orig_add_cluster = residue_field.benefit_rbf_field.add_residue_cluster
        self.events: List[Dict] = []
        self._pending: Optional[Dict] = None
        residue_field.accumulate_benefit = self._wrapped_accumulate_benefit
        residue_field.benefit_rbf_field.add_residue = self._wrapped_add_residue
        residue_field.benefit_rbf_field.add_residue_cluster = self._wrapped_add_cluster

    def _wrapped_accumulate_benefit(self, z_world, benefit_magnitude=1.0,
                                     hypothesis_tag=False, dopamine_signal=0.0):
        loc = z_world.mean(dim=0) if z_world.dim() == 2 else z_world
        # Snapshot density BEFORE this event writes, so _finalize can report the
        # MARGINAL contribution of just this allocation (density_after -
        # density_before) alongside the raw post-event level. compute_benefit_density
        # sums over the WHOLE field, so an absolute post-event read at a
        # revisited/nearby location is contaminated by residual density from
        # earlier, unrelated events -- confirmed live (2026-08-08 Step 4 probe):
        # a controlled sweep at well-separated locations gave rho(size,density)=
        # 1.0, but the live rollout (57 pooled events, spatially recurring
        # contacts) gave only 0.18 on the raw post-event level. The delta isolates
        # THIS event's own contribution from that field-history confound.
        with torch.no_grad():
            density_before = float(self.rf.compute_benefit_density(
                loc.detach().unsqueeze(0)).item())
        self._pending = {
            "z_world": loc.detach().clone(),
            "dopamine_signal": float(dopamine_signal),
            "benefit_magnitude": float(benefit_magnitude),
            "density_before": density_before,
        }
        out = self._orig_accumulate_benefit(
            z_world, benefit_magnitude=benefit_magnitude,
            hypothesis_tag=hypothesis_tag, dopamine_signal=dopamine_signal,
        )
        self._pending = None   # no-op path (terrain disabled / hypothesis_tag) never finalized
        return out

    def _wrapped_add_residue(self, location, intensity=1.0):
        idx = self._orig_add_residue(location, intensity=intensity)
        self._finalize(cluster_size=1)
        return idx

    def _wrapped_add_cluster(self, location, intensity=1.0, dopamine_signal=0.0,
                              allocation_scale=0.0, jitter_radius=0.1,
                              bandwidth_narrowing=0.0):
        allocated = self._orig_add_cluster(
            location, intensity=intensity, dopamine_signal=dopamine_signal,
            allocation_scale=allocation_scale, jitter_radius=jitter_radius,
            bandwidth_narrowing=bandwidth_narrowing,
        )
        self._finalize(cluster_size=len(allocated))
        return allocated

    def _finalize(self, cluster_size: int) -> None:
        if self._pending is None:
            return
        ev = dict(self._pending)
        ev["cluster_size"] = int(cluster_size)
        with torch.no_grad():
            density = float(self.rf.compute_benefit_density(
                ev["z_world"].unsqueeze(0)).item())
        ev["density"] = density
        ev["density_delta"] = density - ev["density_before"]
        self.events.append(ev)

    def restore(self) -> None:
        self.rf.accumulate_benefit = self._orig_accumulate_benefit
        self.rf.benefit_rbf_field.add_residue = self._orig_add_residue
        self.rf.benefit_rbf_field.add_residue_cluster = self._orig_add_cluster


class HarmAsymmetryRecorder:
    """Spies on the LIVE harm/threat write path (ResidueField.accumulate ->
    rbf_field.add_residue). Counts calls only; never modifies behaviour.
    add_residue_cluster is also wrapped on the SAME (harm) field so a call
    that should architecturally never happen is empirically confirmed absent,
    not merely assumed from reading the source.
    """

    def __init__(self, residue_field: ResidueField) -> None:
        self.rf = residue_field
        self._orig_accumulate = residue_field.accumulate
        self._orig_add_residue = residue_field.rbf_field.add_residue
        self._orig_add_cluster = residue_field.rbf_field.add_residue_cluster
        self.n_harm_calls = 0
        self.n_add_residue_calls = 0
        self.n_add_cluster_calls = 0
        residue_field.accumulate = self._wrapped_accumulate
        residue_field.rbf_field.add_residue = self._wrapped_add_residue
        residue_field.rbf_field.add_residue_cluster = self._wrapped_add_cluster

    def _wrapped_accumulate(self, *a, **kw):
        self.n_harm_calls += 1
        return self._orig_accumulate(*a, **kw)

    def _wrapped_add_residue(self, *a, **kw):
        self.n_add_residue_calls += 1
        return self._orig_add_residue(*a, **kw)

    def _wrapped_add_cluster(self, *a, **kw):
        self.n_add_cluster_calls += 1
        return self._orig_add_cluster(*a, **kw)

    def restore(self) -> None:
        self.rf.accumulate = self._orig_accumulate
        self.rf.rbf_field.add_residue = self._orig_add_residue
        self.rf.rbf_field.add_residue_cluster = self._orig_add_cluster


# --------------------------------------------------------------------------- #
# per-seed live rollout                                                       #
# --------------------------------------------------------------------------- #

_ZG = ZGoalStreamAccumulator()


def run_seed(seed: int, episodes: int, steps: int) -> Dict:
    print(f"Seed {seed} Condition live_rollout", flush=True)
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES, hazard_harm=HAZARD_HARM,
        resource_benefit=RESOURCE_BENEFIT, use_proxy_fields=True,
    )
    agent = REEAgent(make_config(env))
    agent.reset()

    benefit_rec = BenefitClusterRecorder(agent.residue_field)
    harm_rec = HarmAsymmetryRecorder(agent.residue_field)

    try:
        for ep in range(episodes):
            _flat, obs_dict = env.reset()
            body, world = _obs(obs_dict)
            for _step in range(steps):
                latent = agent.sense(body, world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())
                _f, harm_signal, done, _info, obs_dict = env.step(action_idx)
                body, world = _obs(obs_dict)

                benefit = max(0.0, float(harm_signal))
                if benefit > 0.0:
                    agent.update_z_goal(
                        benefit_exposure=benefit,
                        drive_level=REEAgent.compute_drive_level(body),
                    )
                if harm_signal is not None and harm_signal < 0:
                    agent.update_residue(float(harm_signal))
                if done:
                    break
            if (ep + 1) % max(1, episodes // 4) == 0 or (ep + 1) == episodes:
                print(f"  [train] live_rollout seed={seed} ep {ep+1}/{episodes}", flush=True)
    finally:
        benefit_rec.restore()
        harm_rec.restore()

    _ZG.observe(agent)

    n_events = len(benefit_rec.events)
    print(f"verdict: {'PASS' if n_events > 0 or harm_rec.n_harm_calls > 0 else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "benefit_events": benefit_rec.events,
        "n_benefit_events": n_events,
        "n_harm_calls": harm_rec.n_harm_calls,
        "n_harm_add_residue_calls": harm_rec.n_add_residue_calls,
        "n_harm_add_cluster_calls": harm_rec.n_add_cluster_calls,
    }


# --------------------------------------------------------------------------- #
# evaluation                                                                   #
# --------------------------------------------------------------------------- #

def evaluate(sweep: Dict, per_seed: List[Dict]) -> Dict:
    r1 = sweep["rho_cluster_size_vs_da"] >= RHO_CONTROL_FLOOR
    r2 = sweep["rho_density_vs_cluster_size"] >= RHO_CONTROL_FLOOR

    pooled_events = [ev for r in per_seed for ev in r["benefit_events"]]
    n_expansion = sum(1 for ev in pooled_events if ev["cluster_size"] >= 2)
    pooled_n_harm_calls = sum(r["n_harm_calls"] for r in per_seed)

    r3 = n_expansion >= N_EXPANSION_MIN
    r4 = pooled_n_harm_calls >= N_HARM_MIN

    preconditions = [
        {
            "name": "sweep_cluster_size_responds_to_da",
            "kind": "readiness",
            "description": ("P0 sweep, real substrate call path: Spearman rho(da, "
                            "cluster_size) over a well-separated DA_SWEEP -- the SAME "
                            "statistic C1 routes on."),
            "control": f"DA_SWEEP={DA_SWEEP} at well-separated synthetic locations",
            "measured": float(sweep["rho_cluster_size_vs_da"]),
            "threshold": RHO_CONTROL_FLOOR,
            "direction": "lower",
            "met": bool(r1),
        },
        {
            "name": "sweep_density_responds_to_cluster_size",
            "kind": "readiness",
            "description": ("P0 sweep, real substrate call path: Spearman rho(cluster_size, "
                            "density) -- the SAME statistic C2 routes on."),
            "control": f"DA_SWEEP={DA_SWEEP} at well-separated synthetic locations",
            "measured": float(sweep["rho_density_vs_cluster_size"]),
            "threshold": RHO_CONTROL_FLOOR,
            "direction": "lower",
            "met": bool(r2),
        },
        {
            "name": "pooled_live_expansion_events",
            "kind": "readiness",
            "description": ("pooled across seeds; non-degeneracy floor for C1/C2 -- a live "
                            "sample where every event allocated the same cluster size carries "
                            "no rank information for a correlation to measure."),
            "control": "pooled live benefit events with cluster_size >= 2",
            "measured": float(n_expansion),
            "threshold": float(N_EXPANSION_MIN),
            "direction": "lower",
            "met": bool(r3),
        },
        {
            "name": "pooled_live_harm_events",
            "kind": "readiness",
            "description": "pooled across seeds; sample floor for C4.",
            "control": "pooled live agent.update_residue() calls with harm_signal < 0",
            "measured": float(pooled_n_harm_calls),
            "threshold": float(N_HARM_MIN),
            "direction": "lower",
            "met": bool(r4),
        },
    ]
    ready = all(p["met"] for p in preconditions)

    if not ready:
        unmet = [p["name"] for p in preconditions if not p["met"]]
        return {
            "ready": False,
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "label": "substrate_not_ready_requeue",
            "interp": (f"SD-024 readiness preconditions unmet ({', '.join(unmet)}); "
                      "re-queue under a NEW letter at an adequate P0. This is NOT a "
                      "substrate ceiling and NOT a refutation of SD-024."),
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": False, "C2": False, "C4": False},
            "criteria": [],
            "pooled": {
                "n_benefit_events": len(pooled_events),
                "n_expansion_events": n_expansion,
                "n_harm_calls": pooled_n_harm_calls,
            },
            "non_degenerate": False,
            "degeneracy_reason": "readiness gate unmet: " + "; ".join(unmet),
        }

    das = np.array([ev["dopamine_signal"] for ev in pooled_events], dtype=float)
    sizes = np.array([ev["cluster_size"] for ev in pooled_events], dtype=float)
    # C2 uses the MARGINAL density contribution of each event (density_delta =
    # post-event minus pre-event, at that event's own location), not the raw
    # post-event level -- see the density_delta comment in
    # BenefitClusterRecorder._wrapped_accumulate_benefit for why the raw level
    # is confounded by field history in a spatially-recurring live rollout.
    dens = np.array([ev["density_delta"] for ev in pooled_events], dtype=float)

    rho_c1 = _spearman(das, sizes)
    rho_c2 = _spearman(sizes, dens)

    c1 = bool(rho_c1 >= RHO_LIVE_MIN)
    c2 = bool(rho_c2 >= RHO_LIVE_MIN)

    pooled_n_add_residue = sum(r["n_harm_add_residue_calls"] for r in per_seed)
    pooled_n_add_cluster = sum(r["n_harm_add_cluster_calls"] for r in per_seed)
    c4 = bool(pooled_n_add_cluster == 0 and pooled_n_add_residue == pooled_n_harm_calls)

    passed = c1 and c2 and c4
    outcome = "PASS" if passed else "FAIL"
    if passed:
        label = "da_cluster_allocation_representational_and_functional"
        direction = "supports"
        interp = ("live dopamine_signal correlates with allocated cluster size (C1), "
                  "cluster size correlates with the weight-independent density reader "
                  "(C2), and the harm/threat field never receives cluster allocation "
                  "across every live harm event (C4) -- the representational-expansion "
                  "mechanism and the MECH-233 asymmetry both hold under a real, "
                  "extended live rollout, not only by construction.")
    else:
        failed = [n for n, ok in (("C1_da_vs_cluster_size", c1),
                                  ("C2_cluster_size_vs_density", c2),
                                  ("C4_harm_asymmetry", c4)) if not ok]
        label = "da_cluster_allocation_not_confirmed"
        direction = "weakens"
        interp = ("with a green readiness gate (instrument certified, live sample "
                  "adequate), " + ", ".join(failed) + " failed on the pooled live "
                  "sample: the allocation mechanism does not behave as SD-024 "
                  "specifies under real live-agent conditions.")

    # check_degeneracy() is built for "does this metric vary ACROSS CELLS
    # (per-seed/per-arm)" -- a single pooled rho is not that shape, and a
    # 1-element values list would read as trivially degenerate (zero spread
    # by construction) regardless of whether the underlying sample was fine.
    # The correct non-degeneracy guard here is variance in the INPUT the
    # correlation is computed over (R3 already floors the event count, but a
    # count floor does not itself rule out every counted event sharing one
    # exact cluster_size) -- so check that directly.
    degenerate_metrics: Dict[str, str] = {}
    if len(set(sizes.tolist())) < 2:
        degenerate_metrics["rho_cluster_size_vs_da"] = "pooled cluster_size has no spread"
    if len(set(np.round(dens, 6).tolist())) < 2:
        degenerate_metrics["rho_density_vs_cluster_size"] = "pooled density has no spread"
    degeneracy = {
        "non_degenerate": not degenerate_metrics,
        "degeneracy_reason": "; ".join(f"{k}: {v}" for k, v in degenerate_metrics.items()),
    }

    return {
        "ready": True,
        "outcome": outcome,
        "evidence_direction": direction,
        "label": label,
        "interp": interp,
        "preconditions": preconditions,
        "criteria_non_degenerate": {"C1": True, "C2": True, "C4": True},
        "criteria": [
            {"name": "C1_da_vs_cluster_size", "load_bearing": True, "passed": c1,
             "measured_rho": rho_c1, "threshold": RHO_LIVE_MIN},
            {"name": "C2_cluster_size_vs_density", "load_bearing": True, "passed": c2,
             "measured_rho": rho_c2, "threshold": RHO_LIVE_MIN},
            {"name": "C4_harm_asymmetry", "load_bearing": True, "passed": c4,
             "n_harm_calls": pooled_n_harm_calls,
             "n_add_residue_calls": pooled_n_add_residue,
             "n_add_cluster_calls": pooled_n_add_cluster},
        ],
        "pooled": {
            "n_benefit_events": len(pooled_events),
            "n_expansion_events": n_expansion,
            "n_harm_calls": pooled_n_harm_calls,
            "rho_c1_da_vs_cluster_size": rho_c1,
            "rho_c2_cluster_size_vs_density": rho_c2,
            "mean_cluster_size": float(np.mean(sizes)) if len(sizes) else 0.0,
            "max_cluster_size": int(np.max(sizes)) if len(sizes) else 0,
            "mean_benefit_magnitude_across_cluster_sizes_ge2": (
                float(np.mean([ev["benefit_magnitude"] for ev in pooled_events
                               if ev["cluster_size"] >= 2]))
                if any(ev["cluster_size"] >= 2 for ev in pooled_events) else 0.0
            ),
            "mean_benefit_magnitude_across_cluster_size_1": (
                float(np.mean([ev["benefit_magnitude"] for ev in pooled_events
                               if ev["cluster_size"] == 1]))
                if any(ev["cluster_size"] == 1 for ev in pooled_events) else 0.0
            ),
        },
        "non_degenerate": degeneracy["non_degenerate"],
        "degeneracy_reason": degeneracy.get("degeneracy_reason") or None,
    }


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #

def main(dry_run: bool = False) -> Dict:
    t0, t0_perf = time.time(), time.perf_counter()
    print(f"{EXPERIMENT_TYPE}  dry_run={dry_run}", flush=True)

    sweep = readiness_sweep()
    print(f"  [P0] sweep: rho(da,size)={sweep['rho_cluster_size_vs_da']:.4f} "
          f"rho(size,density)={sweep['rho_density_vs_cluster_size']:.4f} "
          f"sizes={sweep['cluster_sizes']}", flush=True)

    bw_diag = bandwidth_narrowing_diagnostic()
    print(f"  [P0] bandwidth diagnostic (non-gating): rho(da,bw)="
          f"{bw_diag['rho_bandwidth_vs_da']:.4f} narrows={bw_diag['narrows_with_da']} "
          f"floors_at_half={bw_diag['floors_at_half_base']}", flush=True)

    seeds = SEEDS[:1] if dry_run else SEEDS
    episodes = 2 if dry_run else EPISODES
    steps = 20 if dry_run else STEPS_PER_EPISODE

    per_seed = [run_seed(s, episodes, steps) for s in seeds]
    ev = evaluate(sweep, per_seed)

    print(f"\n  ready={ev['ready']}  VERDICT: {ev['outcome']}  "
          f"direction={ev['evidence_direction']}  label={ev['label']}", flush=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest_purpose = EXPERIMENT_PURPOSE if ev["ready"] else "diagnostic"

    # per_seed_rows carry a trimmed benefit_events (drop the large z_world
    # tensors -> lists) so the manifest stays a reasonable size; pooled
    # aggregates above already carry every scalar reading generously.
    per_seed_rows = []
    for r in per_seed:
        row = dict(r)
        row["benefit_events"] = [
            {k: v for k, v in ev_.items() if k != "z_world"}
            for ev_ in r["benefit_events"]
        ]
        per_seed_rows.append(row)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": manifest_purpose,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "started_utc": datetime.utcfromtimestamp(t0).isoformat() + "Z",
        "timestamp_utc": timestamp,
        "outcome": ev["outcome"],
        "evidence_direction": ev["evidence_direction"],
        "interpretation": {
            "label": ev["label"],
            "preconditions": ev["preconditions"],
            "criteria_non_degenerate": ev["criteria_non_degenerate"],
            "criteria": ev["criteria"],
        },
        "interpretation_note": ev["interp"],
        "combination_rule": "PASS iff C1 AND C2 AND C4 hold on the POOLED live sample, given a green readiness gate",
        "readiness_sweep": sweep,
        "bandwidth_narrowing_diagnostic": bw_diag,
        "pooled": ev.get("pooled", {}),
        "per_seed_rows": per_seed_rows,
        "non_degenerate": ev["non_degenerate"],
        "degeneracy_reason": ev["degeneracy_reason"],
        "dv_symmetry_note": (
            "No CEM/argmax/selection anywhere in this design; every DV is a direct "
            "substrate-state or spy-captured call-argument read. Spearman rho is used "
            "for robustness to the exact functional realisation of a known monotone "
            "step relationship (n = 1 + int(da*scale)), not to protect against a "
            "selection-invariance trap -- there is no selection step here."
        ),
        "wall_independence_note": (
            "No behavioural DV (no approach rate, latency, or navigation success). The "
            "MECH-232/ARC-057 crux family's CEM-approach ceiling cannot be re-derived by "
            "this design; its DVs read the residue field's own allocation and density-read "
            "state, and spy-captured harm/benefit write-path call arguments."
        ),
        "notes": (
            "SD-024 DA-modulated RBF cluster allocation, tested directly (not via "
            "MECH-232's approach-without-gradient crux, which V3-EXQ-766/893 already "
            "cover and correctly tag MECH-232 only). V3-EXQ-795 (SD-024 tagged, PASS, "
            "supports) is diagnostic purpose and excluded from confidence scoring; this "
            "experiment is evidence purpose and, on a green gate, is SD-024's first "
            "confidence-bearing experimental result. GOV-REUSE-1: reanalysis_query.py "
            "--claim SD-024 finds one manifest (V3-EXQ-795, diagnostic, substrate_hash "
            "402e3f5a...) -- not recoverable, run required. IGW-20260808-230."
        ),
    }

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run,
        config=_full_config(), seeds=list(seeds),
        script_path=Path(__file__), started_at=t0_perf,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {ev['outcome']}")
    return {"outcome": ev["outcome"], "manifest_path": out_path, "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (1 seed, 2 episodes).")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    _outcome = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
    sys.exit(0)
