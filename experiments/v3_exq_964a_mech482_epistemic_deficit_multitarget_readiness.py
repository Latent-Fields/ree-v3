"""V3-EXQ-964a -- SD-102 (MECH-482) epistemic-deficit accumulator: MULTI-TARGET
READINESS validation. Supersedes V3-EXQ-964, whose C2 was uninterpretable.

EXPERIMENT_PURPOSE = "diagnostic" -- this validates the SUBSTRATE readiness
entry sd_epistemic_deficit_multitarget_readiness (does the readiness geometry
make a second target reachable, and does the per-candidate readout then
actually differentiate), NOT MECH-482's own claim hypothesis.

SLEEP: none. No sleep flag is set; no SLEEP DRIVER line required.

red-team (fable): see queue entry note.

WHY 964 WAS UNINTERPRETABLE, AND WHAT CHANGED. V3-EXQ-964 armed
curiosity_learning_progress_source="epistemic_deficit" on ONE arm only
(ARM_LP_BROADCAST vs ARM_LP_EPISTEMIC_DEFICIT), so the accumulator existed on
a single arm and there was no contrast for its OWN behaviour. It reached
n_targets == 1 on all 3 seeds with all 32 candidates matching that one target,
so readout() returned a CONSTANT vector; StructuredCuriosity applies
    total = total - weight * lp_vec
and a candidate-uniform constant provably cannot move an argmax. Its C2
(yoked_divergence_frac > 0) was therefore structurally unsatisfiable -- it
measured 0.0 by arithmetic, not by observation -- while the run self-routed
"accumulator_live_but_never_changes_committed_action", which reads as a
substrate verdict about a test that never actually ran. Ratified in
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-964_2026-08-30.json,
whose binding target is: "a validation run must reach n_targets>=2 AND
candidates matching >=2 distinct targets (equivalently _last_lp_dev_range > 0)
before C2 is interpretable".

THIS DRIVER ARMS THE ACCUMULATOR ON BOTH ARMS. Both arms set
curiosity_learning_progress_source="epistemic_deficit", so the accumulator is
instantiated, stepped and MEASURED on both. The manipulation is the readiness
GEOMETRY alone:

  ARM_PREREADINESS (reference, drives the env): the four readiness knobs at
    their pre-readiness defaults -- match_radius_mode="absolute" (literal 1.0),
    center_update="replace", readout_mode="hard_match",
    persist_targets_across_episodes=False. This REPRODUCES the 964 collapse as
    a measured negative control rather than an assumed one.
  ARM_READINESS: match_radius_mode="relative" (frac 0.5), center_update="ema"
    (beta 0.1), readout_mode="rbf_weighted",
    persist_targets_across_episodes=True. Stepped on ARM_PREREADINESS's
    identical observation sequence (yoked), so every comparison is a paired
    argmax flip, non-compounding.

DV-SYMMETRY INVARIANCE -- DECLARED PER ARM (mandatory; this is the exact class
of defect that made 964 uninterpretable).
  The DV is the COMMITTED ACTION, an argmax over per-candidate scores. Its
  symmetry group is (i) addition of a candidate-UNIFORM constant and (ii) any
  monotone rescaling: neither can move an argmax.
  * ARM_PREREADINESS: its manipulation IS invariant under (i). hard_match with
    a single enclosing target returns the same deficit for every candidate, so
    -w * lp_vec is a uniform shift. This arm is therefore DISPOSITION (b) --
    structurally vacuous for the argmax DV. It is scoped OUT of the C3 argmax
    scoring and used only as the yoked reference and the collapse control; its
    own readout statistics are still recorded (C5). This is DECLARED here at
    design time, not discovered after the run.
  * ARM_READINESS: its manipulation is NOT invariant under (i). rbf_weighted
    is a distance-weighted sum over ALL targets, continuous in candidate
    position, so lp_vec varies whenever the candidates differ at all -- it does
    not depend on a knife-edge radius. C2 MEASURES that differentiation rather
    than assuming it, and C3 is gated behind it.

WHY THE MATCHED-TARGET COUNT IS RECORDED BUT NOT GATED ON. The ratified
autopsy phrases its target as "candidates matching >=2 distinct targets
(equivalently _last_lp_dev_range > 0)". Those two ARE equivalent under
hard_match, which is what V3-EXQ-964 ran: there, out = deficits[nearest] where
matched, else 0.0, so nothing but a match can produce variation. They are NOT
equivalent under rbf_weighted. Read epistemic_deficit.py readout(): in the
rbf branch `out` is a distance-weighted sum over ALL targets and the `matched`
mask is never consulted -- it feeds only the diagnostics
_last_n_targets_matched_at_readout / _last_readout_n_distinct_targets. So a
readout can be genuinely, continuously differentiated across candidates while
matching ZERO targets, which is precisely the mode's stated purpose ("continuous
in candidate position ... without depending on a knife-edge radius"). Gating on
the matched count here would therefore manufacture a substrate_not_ready
verdict out of a readout that is in fact differentiating -- so the binding gate
is the RANGE of the vector actually returned (the source's own words: "deficit_
range is the max-min of the vector actually returned, which is what
StructuredCuriosity turns into lp_contrib"), and the matched counts are
recorded as diagnostics for the adjudicator instead.

YOKING CAVEAT, STATED HONESTLY. The pair is non-compounding only while the arms
agree: the env is stepped with the REFERENCE arm's action, so once a tick
diverges, the readiness arm's own accumulator UPDATE pairs its chosen action
with an outcome produced by a different action. Every divergence figure here is
therefore a count of FIRST-divergence-eligible argmax flips under a shared
observation stream, not a claim about a freely-running readiness agent. At the
observed divergence of 0.0 the distinction is currently moot; it is recorded so
a non-zero result is not over-read.

SAME-STATISTIC READINESS (the V3-EXQ-643 rule). C3 routes on argmax flips,
which require a non-zero cross-candidate RANGE. The readiness precondition
therefore asserts a RANGE (_last_lp_dev_range > 0), not a magnitude or mean --
a magnitude can be large while the range is ~0, which is precisely the 964
state. Range-gated criterion -> range readiness.

require_differentiated_readout is left FALSE on both arms ON PURPOSE. Setting
it True makes readout() REFUSE when constant, which routes those ticks back to
the Phase-1 broadcast fallback and would CONFOUND the divergence DV with a
changed selection path. Differentiation is measured here
(n_undifferentiated_readouts, _last_lp_dev_range), not suppressed.

GOV-REUSE-1. The broadcast-vs-epistemic_deficit contrast is already recorded in
v3_exq_964_mech482_epistemic_deficit_validation_20260829T215030Z_v3 and is NOT
re-run here; that manifest's decisive readouts (n_targets_min 1,
yoked_divergence_frac_max 0.0) are the pre-readiness reference. The readiness
knobs did not exist at that substrate_hash, so the readiness question itself is
not recoverable from it -> this run is required.

LATCHED-DIAGNOSTIC DISCIPLINE. StructuredCuriosity._last_lp_dev_range is
written only inside compute_score_bias(), which runs only on an E3 tick
(heartbeat.e3_steps_per_tick), so a per-env-step read would re-record one
selection as many observations. This driver CLEARS the latch to None before
every select_action and records nothing on a tick where it stayed None,
counting those as n_latched_ticks so the true denominator is auditable.

OWNS ITS OWN RNG STREAM. Each _Runner snapshots and restores its own generator
state around every tick, episode reset and residue update; the self-yoked
paired_control_divergence instrument control runs for both arm identities
before any scored compute.

REPRODUCIBILITY LIMIT, MEASURED AND DECLARED. Per-arm RNG ownership makes the
WITHIN-run yoking exact -- paired_control_divergence is 0.000000 on every
observed run, so every paired argmax comparison is trustworthy. It does NOT
make the run bit-reproducible ACROSS invocations: repeated dry-runs at the same
seed gave max_n_targets 6 vs 7 and n_lp_reads 11 vs 18, i.e. the absolute
trajectory and the E3 tick cadence both vary. Consequences, stated rather than
left implicit:
  * Every cell is stamped reuse-INELIGIBLE (extra_ineligible_reasons below). A
    cell that is not a pure function of (substrate, config, seed) can never be
    safely reused as a baseline mint, and silently minting one would corrupt a
    future consumer's OFF arm.
  * No load-bearing criterion depends on cross-run reproducibility. C3 is a
    PAIRED within-run comparison whose instrument control is exactly 0. C1's
    observed values (6-7) sit far above its floor of 2, and C2's range is
    non-zero on every observed run -- neither is a knife-edge call against the
    variation measured here.
  * The per-tick magnitude diagnostics (max_pert_over_margin, the reachability
    decomposition) DO vary run to run and are reported as per-run observations,
    never as stable constants of the substrate.

PRE-REGISTERED CRITERIA (diagnostic-scope):
  C1 (load-bearing): ARM_READINESS reaches max_n_targets >= 2 on every seed --
     the multi-target regime is reachable at all.
  C2 (load-bearing): ARM_READINESS's per-candidate readout DIFFERENTIATES --
     max observed _last_lp_dev_range > 0 on every seed.
  C3 (load-bearing): the downstream consumer can change the committed action --
     yoked divergence > 0 on at least one seed. INTERPRETABLE ONLY IF C1, C2 AND
     the MAGNITUDE gate hold; all three are declared as binding readiness
     preconditions on ARM_READINESS, so a C3 failure under an unmet precondition
     self-routes substrate_not_ready_requeue, never a substrate verdict.
     MAGNITUDE GATE (lp_perturbation_can_reach_argmax_margin): C1+C2 establish
     that the readout VARIES across candidates, which is necessary but NOT
     sufficient -- a varying perturbation of
     curiosity_learning_progress_weight (0.05) x range still cannot flip an
     argmax whose committed-selection margin is orders of magnitude larger.
     That is the F-dominated-argmax washout (V3-EXQ-569g/684/700), and left
     ungated it would reproduce 964's defect one layer down: C3 reading 0 by
     arithmetic while presenting as a measurement. The gate is measured POST-F
     at the committed-selection layer via E3.decisiveness_margin
     (arbitration_aware=True), on exactly the ticks the DV is read. Its remedy
     is CONFIGURATION (raise the weight / bias scale and re-queue), not a
     substrate rebuild -- stated so an unmet gate is not misread as a ceiling.
     REACHABILITY IS DECOMPOSED, because the two routes mean different things.
     n_flip_reachable_strict counts ticks where pert >= margin > 0 (the
     mechanism genuinely out-competed F); n_zero_margin_ticks counts TIES,
     where the margin is 0 and any non-zero differential perturbation decides
     the winner. Both are real reachability and both count toward the gate, but
     a C3 > 0 carried ENTIRELY by ties is a tie-breaking effect, not evidence
     that the accumulator competes with F -- the manifest records both counts
     and max_pert_over_margin so adjudication cannot conflate them.
     NO VERIFY-LIFT ARM IS INCLUDED, deliberately. The smoke measured
     max_pert_over_margin ~ 3e-5, so reaching parity would need
     curiosity_learning_progress_weight ~ 1.7e3 (vs the 0.05 default) -- a
     weight at which the curiosity term is the ONLY term, so the arm would
     prove nothing except that a dominating term dominates. The margin
     instrumentation above is the honest substitute: it MEASURES reachability
     per tick instead of manufacturing it.
  C4 (supporting): self-yoked instrument control is bit-identical (== 0) for
     both arm identities.
  C5 (supporting): ARM_PREREADINESS reproduces the collapse (max_n_targets == 1)
     -- confirms the geometry knobs are what moved C1, not episode length.
Combination: overall_pass = C1 AND C2 AND C3 AND (ARM_READINESS gate green).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"

EXPERIMENT_TYPE = "v3_exq_964a_mech482_epistemic_deficit_multitarget_readiness"
QUEUE_ID = "V3-EXQ-964a"
CLAIM_IDS = ["MECH-482"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SUPERSEDES = "V3-EXQ-964"

SEEDS = [71, 101, 202]
EPISODES = 3
STEPS_PER_EPISODE = 60
WORLD_DIM = 16
SELF_DIM = 32

E2U_WARMUP_STEPS = 60
E2U_BATCH_SIZE = 16

_ACTIVE_WARMUP = E2U_WARMUP_STEPS

# --- pre-registered thresholds (constants, never derived from the run) ------
VACUOUS_READOUT_RATE_CEILING = 0.5
MULTITARGET_FLOOR = 2.0          # C1: max_n_targets >= 2
LP_DEV_RANGE_FLOOR = 1e-12       # C2: range strictly > 0. PreconditionSpec
                                 # has no comparator and a "lower" bound is
                                 # INCLUSIVE, so an epsilon floor is how
                                 # "non-constant" is encoded honestly --
                                 # threshold 0.0 would recompute range==0
                                 # as MET, passing the exact degeneracy
                                 # this gate exists to catch.
DIVERGENCE_FLOOR = 1e-9          # C3: same, for the argmax-flip fraction
                                 # (smallest attainable non-zero is 1/n_cmp)
CONTROL_DIVERGENCE_CEILING = 1e-9

ARM_PRE = "ARM_PREREADINESS"
ARM_RDY = "ARM_READINESS"

# Readiness geometry, exactly the sd_epistemic_deficit_multitarget_readiness
# levers. Values are the module docstring's own measured recommendations.
READINESS_KNOBS: Dict[str, Any] = {
    "epistemic_deficit_match_radius_mode": "relative",
    "epistemic_deficit_match_radius_relative_frac": 0.5,
    "epistemic_deficit_center_update": "ema",
    "epistemic_deficit_center_ema_beta": 0.1,
    "epistemic_deficit_readout_mode": "rbf_weighted",
    "epistemic_deficit_persist_targets_across_episodes": True,
    # THE FIFTH LEVER. UPDATE keys targets on the encoder's zw_prev under
    # "realized"; READOUT matches candidates in the e2.world_forward frame.
    # config.py's own measured note: centroid offset between the two frames is
    # 0.286 against an internal spread of ~0.05, so under "realized" NO radius
    # can both separate targets and match candidates -- the matched mask is
    # all-False and any range is RBF tail structure, not differentiation.
    # tests/contracts/test_sd_epistemic_deficit_multitarget_readiness.py
    # defines the readiness arm with this knob set.
    "epistemic_deficit_target_frame": "predicted",
}
PREREADINESS_KNOBS: Dict[str, Any] = {
    "epistemic_deficit_match_radius_mode": "absolute",
    "epistemic_deficit_match_radius_relative_frac": 0.5,
    "epistemic_deficit_center_update": "replace",
    "epistemic_deficit_center_ema_beta": 0.1,
    "epistemic_deficit_readout_mode": "hard_match",
    "epistemic_deficit_persist_targets_across_episodes": False,
    "epistemic_deficit_target_frame": "realized",
}


def _knobs(readiness: bool) -> Dict[str, Any]:
    return dict(READINESS_KNOBS if readiness else PREREADINESS_KNOBS)


def build_config(readiness: bool) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
    )
    cfg.use_structured_curiosity = True
    # Isolate the learning-progress (314c) sub-flavour: 314a/314b OFF on both
    # arms, so a yoked divergence is attributable to the readiness geometry
    # alone, not a novelty/uncertainty confound.
    cfg.use_curiosity_novelty = False
    cfg.use_curiosity_uncertainty = False
    cfg.use_curiosity_learning_progress = True
    # ARMED ON BOTH ARMS -- the 964 fix.
    cfg.curiosity_learning_progress_source = "epistemic_deficit"
    # Measured, never suppressed -- see module docstring.
    cfg.epistemic_deficit_require_differentiated_readout = False
    for k, v in _knobs(readiness).items():
        setattr(cfg, k, v)
    # Shared DEPENDENCY, trained identically on both arms so it is not itself
    # a confound.
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = True
    cfg.latent.e2_world_uncertainty_warmup_steps = _ACTIVE_WARMUP
    cfg.latent.e2_world_uncertainty_batch_size = E2U_BATCH_SIZE
    return cfg


def config_slice(readiness: bool) -> Dict[str, Any]:
    """Exactly what this cell's computation reads -- no acceptance thresholds."""
    base = {
        "env": "CausalGridWorldV2",
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "episodes": EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_structured_curiosity": True,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": True,
        "curiosity_learning_progress_source": "epistemic_deficit",
        "epistemic_deficit_require_differentiated_readout": False,
        "use_e2_world_uncertainty": True,
        "use_e2_world_uncertainty_online_training": True,
        "e2_world_uncertainty_warmup_steps": _ACTIVE_WARMUP,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
    }
    base.update(_knobs(readiness))
    return base


class _Runner:
    """One agent, driven tick by tick. OWNS ITS OWN RNG STREAM."""

    def __init__(self, cfg: REEConfig) -> None:
        self.agent = REEAgent(cfg)
        self._rng_state = torch.get_rng_state()
        self.world_dim = cfg.latent.world_dim
        self.n_ticks = 0
        # Latched-diagnostic accounting -- see module docstring.
        self.n_latched_ticks = 0
        self.n_lp_reads = 0
        self.max_lp_dev_range = 0.0
        self.n_positive_lp_dev_range = 0
        # C3 MAGNITUDE gate -- can the lp perturbation reach the argmax margin?
        self.lp_weight = float(
            getattr(cfg, "curiosity_learning_progress_weight", 0.05))
        # get_state() counters are cleared by reset() on EVERY episode, even
        # under persist_targets_across_episodes -- so the final snapshot is
        # episode-N-only. These accumulate the run-lifetime truth.
        self.max_distinct_matched = 0
        self.run_max_n_targets = 0
        self.run_n_updates = 0
        self.run_n_readouts = 0
        self.run_n_vacuous = 0
        self.run_n_undiff = 0
        self.n_margin_reads = 0
        self.n_flip_reachable_ticks = 0
        self.n_zero_margin_ticks = 0
        self.n_flip_reachable_strict = 0
        self.min_argmax_margin = float("inf")
        self.max_pert_over_margin = 0.0

    def reset_episode(self) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.reset()
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def choose(self, obs: Dict[str, Any]) -> int:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            return self._choose_inner(obs)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def _choose_inner(self, obs: Dict[str, Any]) -> int:
        agent = self.agent
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick")
            else torch.zeros(1, self.world_dim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if candidates:
            self.n_ticks += 1
        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(obs["body_state"]),
        )
        # CLEAR THE LATCH immediately before the call. compute_score_bias
        # assigns _last_lp_dev_range on every path it runs, so a value still
        # None afterwards proves it did NOT run this tick (no E3 tick).
        cur = getattr(agent, "curiosity", None)
        if cur is not None:
            cur._last_lp_dev_range = None
        action = agent.select_action(candidates, ticks)
        if cur is not None:
            val = cur._last_lp_dev_range
            if val is None:
                self.n_latched_ticks += 1
            else:
                self.n_lp_reads += 1
                fval = float(val)
                if math.isfinite(fval):
                    if fval > self.max_lp_dev_range:
                        self.max_lp_dev_range = fval
                    if fval > 0.0:
                        self.n_positive_lp_dev_range += 1
                # lp fresh <=> compute_score_bias ran <=> select() ran, so
                # e3.last_scores is fresh on exactly these ticks. Nothing is
                # cleared on e3 -- clearing last_scores could change behaviour.
                acc_now = getattr(agent, "epistemic_deficit", None)
                if acc_now is not None:
                    nd = int(getattr(
                        acc_now, "_last_readout_n_distinct_targets", 0) or 0)
                    if nd > self.max_distinct_matched:
                        self.max_distinct_matched = nd
                e3 = getattr(agent, "e3", None)
                margin = None
                if e3 is not None:
                    try:
                        margin = e3.decisiveness_margin(arbitration_aware=True)
                    except Exception:
                        margin = None
                if margin is not None and math.isfinite(float(margin)):
                    fmargin = float(margin)
                    self.n_margin_reads += 1
                    if fmargin < self.min_argmax_margin:
                        self.min_argmax_margin = fmargin
                    # NOT self.lp_weight * fval: structured_curiosity.py:548
                    # computes _last_lp_dev_range from lp_contrib, which is
                    # ALREADY curiosity_learning_progress_weight * lp_vec.
                    # Multiplying again understates reachability by 1/weight.
                    pert = fval
                    if fmargin <= 0.0:
                        # A TIE: any non-zero differential perturbation decides
                        # the winner, so a flip is genuinely reachable -- but
                        # it is reachable via tie-breaking, NOT by the
                        # mechanism out-competing F. Counted, and recorded
                        # SEPARATELY so adjudication can tell the two apart.
                        self.n_zero_margin_ticks += 1
                        self.n_flip_reachable_ticks += 1
                    else:
                        ratio = pert / fmargin
                        if ratio > self.max_pert_over_margin:
                            self.max_pert_over_margin = ratio
                        if pert >= fmargin:
                            self.n_flip_reachable_strict += 1
                            self.n_flip_reachable_ticks += 1
                # Restore a real float so nothing downstream sees the sentinel.
                cur._last_lp_dev_range = fval
            if cur._last_lp_dev_range is None:
                cur._last_lp_dev_range = 0.0
        return int(action.argmax(dim=-1).item())

    def snapshot_episode(self) -> None:
        """Fold this episode's counters into run-lifetime totals BEFORE the
        next reset() clears them (see __init__)."""
        acc = getattr(self.agent, "epistemic_deficit", None)
        if acc is None:
            return
        st = acc.get_state()
        self.run_max_n_targets = max(
            self.run_max_n_targets,
            int(st.get("max_n_targets", 0) or 0),
            int(st.get("n_targets", 0) or 0),
        )
        self.run_n_updates += int(st.get("n_updates", 0) or 0)
        self.run_n_readouts += int(st.get("n_readouts", 0) or 0)
        self.run_n_vacuous += int(st.get("n_vacuous_readouts", 0) or 0)
        self.run_n_undiff += int(st.get("n_undifferentiated_readouts", 0) or 0)

    def observe(self, harm: Any) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.update_residue(harm)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def accumulator_summary(self) -> Dict[str, Any]:
        acc = getattr(self.agent, "epistemic_deficit", None)
        if acc is None:
            return {}
        state = acc.get_state()
        # DENOMINATOR: mark_vacuous_readout increments _n_vacuous_readouts but
        # NOT _n_readouts, so n_vacuous/n_readouts is denominated on
        # non-refused reads only and a total-refusal run would report 0.0.
        # The honest denominator is every ATTEMPTED read.
        attempts = self.run_n_readouts + self.run_n_vacuous
        vacuous_rate = (self.run_n_vacuous / attempts) if attempts > 0 else 1.0
        undiff_rate = (
            (self.run_n_undiff / self.run_n_readouts)
            if self.run_n_readouts > 0 else 1.0
        )
        return {
            **state,
            # RUN-LIFETIME (state[...] above is episode-N-only; see __init__).
            "max_n_targets": self.run_max_n_targets,
            "n_targets_final_episode": state.get("n_targets", 0),
            "n_updates": self.run_n_updates,
            "n_readouts": self.run_n_readouts,
            "n_vacuous_readouts": self.run_n_vacuous,
            "n_undifferentiated_readouts": self.run_n_undiff,
            "n_readout_attempts": attempts,
            "max_distinct_targets_matched": self.max_distinct_matched,
            "n_candidate_ticks": self.n_ticks,
            "vacuous_readout_rate": vacuous_rate,
            "undifferentiated_readout_rate": undiff_rate,
            # Latched-diagnostic denominators -- the auditable sample size.
            "n_latched_ticks": self.n_latched_ticks,
            "n_lp_reads": self.n_lp_reads,
            "max_lp_dev_range": self.max_lp_dev_range,
            "n_positive_lp_dev_range": self.n_positive_lp_dev_range,
            "n_margin_reads": self.n_margin_reads,
            "n_flip_reachable_ticks": self.n_flip_reachable_ticks,
            "n_zero_margin_ticks": self.n_zero_margin_ticks,
            "n_flip_reachable_strict": self.n_flip_reachable_strict,
            "min_argmax_margin": (
                None if self.min_argmax_margin == float("inf")
                else self.min_argmax_margin),
            "max_pert_over_margin": self.max_pert_over_margin,
            "lp_weight": self.lp_weight,
        }


def run_yoked_pair(seed: int, episodes: int, steps: int,
                   zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    reset_all_rng(seed)
    ref = _Runner(build_config(readiness=False))
    reset_all_rng(seed)
    rdy = _Runner(build_config(readiness=True))
    reset_all_rng(seed)

    env = CausalGridWorldV2()
    n_cmp = 0
    n_diff = 0
    per_episode: List[Dict[str, Any]] = []
    for ep in range(episodes):
        _, obs = env.reset()
        ref.reset_episode()
        rdy.reset_episode()
        ep_cmp = ep_diff = 0
        for _ in range(steps):
            a_ref = ref.choose(obs)
            a_rdy = rdy.choose(obs)
            n_cmp += 1
            ep_cmp += 1
            if a_rdy != a_ref:
                n_diff += 1
                ep_diff += 1
            _f, harm, _d, _i, obs = env.step(a_ref)
            ref.observe(harm)
            rdy.observe(harm)
        ref.snapshot_episode()
        rdy.snapshot_episode()
        per_episode.append({
            "episode": ep, "n_compared": ep_cmp, "n_diverged": ep_diff,
            "divergence_frac": ep_diff / ep_cmp if ep_cmp else 0.0,
        })
        print(f"  [train] yoked seed={seed} ep {ep + 1}/{episodes} "
              f"diverged={ep_diff}/{ep_cmp}", flush=True)
    zg.observe(ref.agent)
    zg.observe(rdy.agent)

    return {
        "ref_n_candidate_ticks": ref.n_ticks,
        ARM_PRE: {"accumulator": ref.accumulator_summary()},
        ARM_RDY: {
            "accumulator": rdy.accumulator_summary(),
            "yoked_n_compared": n_cmp,
            "yoked_n_diverged": n_diff,
            "yoked_divergence_frac": n_diff / n_cmp if n_cmp else 0.0,
            "yoked_per_episode": per_episode,
        },
    }


def paired_control_divergence(seed: int, readiness: bool, episodes: int = 1,
                              steps: int = 15) -> float:
    """INSTRUMENT CONTROL: yoke an arm against ITSELF. MUST be 0.0."""
    reset_all_rng(seed)
    a = _Runner(build_config(readiness))
    reset_all_rng(seed)
    b = _Runner(build_config(readiness))
    reset_all_rng(seed)
    env = CausalGridWorldV2()
    n = d = 0
    for _ in range(episodes):
        _, obs = env.reset()
        a.reset_episode()
        b.reset_episode()
        for _ in range(steps):
            aa = a.choose(obs)
            bb = b.choose(obs)
            n += 1
            if aa != bb:
                d += 1
            _f, harm, _dn, _i, obs = env.step(aa)
            a.observe(harm)
            b.observe(harm)
    return (d / n) if n else 0.0


PRECONDITIONS = [
    PreconditionSpec(
        name="paired_control_is_bit_identical",
        description=(
            "INSTRUMENT CONTROL: an arm yoked against ITSELF diverges on 0 "
            "ticks. Non-zero means the two yoked agents differ in something "
            "other than the manipulation and the whole DV is void"
        ),
        control="same arm vs same arm, identical seed and config",
        threshold=CONTROL_DIVERGENCE_CEILING,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="vacuous_readout_rate_bounded",
        description=(
            "the readiness-gate refusal rate (self-reported vacuous READOUTs) "
            "stays below a generous ceiling across the rollout -- a near-100% "
            "rate would mean the accumulator never got a chance to matter"
        ),
        control="live rollout with the SD-063 head trained identically on both arms",
        threshold=VACUOUS_READOUT_RATE_CEILING,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="multitarget_regime_reached",
        description=(
            "the readiness geometry makes a SECOND persistent target reachable "
            "(max_n_targets >= 2). This is the autopsy's binding gate: below "
            "it, a single enclosing target makes the readout constant and C3 "
            "cannot discriminate by arithmetic"
        ),
        control="live rollout, readiness geometry enabled",
        threshold=MULTITARGET_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["readiness"]),
        applies_note=(
            "the pre-readiness arm is the NEGATIVE CONTROL for exactly this "
            "collapse -- failing it is that arm's purpose, not a readiness "
            "failure of the run (disposition (a): scoped out, arm stays scorable)"
        ),
    ),
    PreconditionSpec(
        name="readout_differentiates_across_candidates",
        description=(
            "the per-candidate readout has NON-ZERO cross-candidate RANGE "
            "(max observed StructuredCuriosity._last_lp_dev_range > 0). Same "
            "statistic C3 routes on: an argmax can only move if the vector it "
            "is subtracting varies across candidates"
        ),
        control="live rollout, readiness geometry enabled, rbf_weighted readout",
        threshold=LP_DEV_RANGE_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["readiness"]),
        applies_note=(
            "the pre-readiness arm's readout is constant BY CONSTRUCTION "
            "(hard_match, single enclosing target) -- that is the 964 finding "
            "this arm reproduces as a control"
        ),
    ),
]


PRECONDITIONS.append(
    PreconditionSpec(
        name="lp_perturbation_can_reach_argmax_margin",
        description=(
            "on at least one tick, the actual score perturbation "
            "(curiosity_learning_progress_weight * cross-candidate lp range) "
            "is at least as large as the committed-selection margin "
            "(E3.decisiveness_margin, arbitration-aware). BELOW THIS FLOOR AN "
            "ARGMAX FLIP IS ARITHMETICALLY UNREACHABLE, so C3 would measure 0 "
            "by magnitude rather than by observation -- the SAME class of "
            "defect that made V3-EXQ-964's C2 uninterpretable, one layer down. "
            "REMEDY IS CONFIGURATION, NOT A REBUILD: raise "
            "curiosity_learning_progress_weight (or curiosity_bias_scale) and "
            "re-queue; the substrate itself is not implicated"
        ),
        control=(
            "measured post-F at the committed-selection layer, on the same "
            "ticks the DV is read (never at the proposer, which an F-dominated "
            "argmax washes out -- V3-EXQ-569g/684/700)"
        ),
        threshold=1.0,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["readiness"]),
        applies_note=(
            "the pre-readiness arm's readout is constant, so its perturbation "
            "is a uniform shift that cannot move an argmax at ANY magnitude"
        ),
    )
)


def _arm_ctx(readiness: bool) -> Dict[str, Any]:
    return {"arm_id": (ARM_RDY if readiness else ARM_PRE), "readiness": readiness}


def _flat_scalar(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flat numeric projection for the runpack converter. Booleans -> 0/1;
    non-finite / None DROPPED (an absent key correctly reads as unmeasured)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            f = float(v)
            if math.isfinite(f):
                out[k] = v
    return out


def run_experiment(episodes: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    zg = ZGoalStreamAccumulator()

    all_ctxs = [_arm_ctx(False), _arm_ctx(True)]
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, all_ctxs)

    control_div: Dict[str, float] = {}
    for readiness, aid in ((False, ARM_PRE), (True, ARM_RDY)):
        control_div[aid] = paired_control_divergence(
            seeds[0], readiness, episodes=1, steps=(6 if dry_run else 15))
        print(f"[control] {aid}: self-yoked divergence = "
              f"{control_div[aid]:.6f} (must be 0)", flush=True)

    rows: List[Dict[str, Any]] = []
    pre_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition yoked_pair", flush=True)
        with arm_cell(seed, config_slice=config_slice(readiness=True),
                      script_path=Path(__file__),
                      config_slice_declared=True,
                      extra_ineligible_reasons=[
                          "measured_non_reproducible_at_fixed_seed: repeated dry-runs at the same seed differ (n_targets 6 vs 7, n_lp_reads 11 vs 18), so a cell is NOT a pure function of (substrate, config, seed) and must never be reused as a baseline mint"],
                      include_driver_script_in_hash=False) as cell:
            pair = run_yoked_pair(seed, episodes, steps, zg)
            row = {
                "arm_id": ARM_RDY, "seed": seed,
                "ref_n_candidate_ticks": pair["ref_n_candidate_ticks"],
                **pair[ARM_RDY],
            }
            cell.stamp(row)
        rows.append(row)
        # Reference-arm cell, minted reuse-eligible as its own row.
        with arm_cell(seed, config_slice=config_slice(readiness=False),
                      script_path=Path(__file__),
                      config_slice_declared=True,
                      extra_ineligible_reasons=[
                          "measured_non_reproducible_at_fixed_seed: repeated dry-runs at the same seed differ (n_targets 6 vs 7, n_lp_reads 11 vs 18), so a cell is NOT a pure function of (substrate, config, seed) and must never be reused as a baseline mint"],
                      include_driver_script_in_hash=False) as pcell:
            prow = {"arm_id": ARM_PRE, "seed": seed, **pair[ARM_PRE]}
            pcell.stamp(prow)
        pre_rows.append(prow)
        print(f"verdict: {'PASS' if row['ref_n_candidate_ticks'] > 0 else 'FAIL'}",
              flush=True)

    def _acc(rs: List[Dict[str, Any]], key: str, default: float = 0.0):
        return [r["accumulator"].get(key, default) for r in rs]

    rdy_max_targets_min = min(_acc(rows, "max_n_targets"))
    rdy_lp_range_min = min(_acc(rows, "max_lp_dev_range"))
    rdy_vacuous_max = max(_acc(rows, "vacuous_readout_rate", 1.0))
    rdy_flip_reachable_min = min(_acc(rows, "n_flip_reachable_ticks"))
    rdy_distinct_matched_min = min(_acc(rows, "max_distinct_targets_matched"))
    rdy_zero_margin_total = sum(_acc(rows, "n_zero_margin_ticks"))
    rdy_flip_strict_total = sum(_acc(rows, "n_flip_reachable_strict"))
    rdy_pert_over_margin_max = max(_acc(rows, "max_pert_over_margin"))
    pre_max_targets_max = max(_acc(pre_rows, "max_n_targets"))
    pre_vacuous_max = max(_acc(pre_rows, "vacuous_readout_rate", 1.0))
    max_div = max(r["yoked_divergence_frac"] for r in rows)
    mean_div = sum(r["yoked_divergence_frac"] for r in rows) / len(rows)

    # ---- per-arm readiness gates -------------------------------------------
    arm_gates = []
    for readiness, aid in ((False, ARM_PRE), (True, ARM_RDY)):
        ctx = _arm_ctx(readiness)
        if readiness:
            measured = {
                "paired_control_is_bit_identical": control_div.get(aid, 1.0),
                "vacuous_readout_rate_bounded": rdy_vacuous_max,
                "multitarget_regime_reached": float(rdy_max_targets_min),
                "readout_differentiates_across_candidates": float(rdy_lp_range_min),
                "lp_perturbation_can_reach_argmax_margin": float(
                    rdy_flip_reachable_min),
            }
        else:
            measured = {
                "paired_control_is_bit_identical": control_div.get(aid, 1.0),
                "vacuous_readout_rate_bounded": pre_vacuous_max,
            }
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITIONS, measured))
    aggregate = aggregate_arm_gates(arm_gates)

    # ---- pre-registered criteria (diagnostic-scope) ------------------------
    c1 = bool(rdy_max_targets_min >= MULTITARGET_FLOOR)
    c2 = bool(rdy_lp_range_min >= LP_DEV_RANGE_FLOOR)
    c3 = bool(max_div >= DIVERGENCE_FLOOR)
    c4 = bool(max(control_div.values()) < CONTROL_DIVERGENCE_CEILING) if control_div else False
    c5 = bool(pre_max_targets_max <= 1)

    green = set(aggregate["green_arms"])
    rdy_green = ARM_RDY in green
    # c4 GATES: the paired control runs on BOTH arm identities, and a
    # non-zero self-yoked divergence on EITHER arm voids the whole paired DV
    # -- ARM_READINESS being green does not cover the reference arm.
    overall_pass = bool(c1 and c2 and c3 and c4 and rdy_green)

    criteria = [
        {"name": "C1_multitarget_regime_reached", "load_bearing": True,
         "passed": c1, "measured": float(rdy_max_targets_min),
         "threshold": MULTITARGET_FLOOR},
        {"name": "C2_readout_differentiates", "load_bearing": True,
         "passed": c2, "measured": float(rdy_lp_range_min),
         "threshold": LP_DEV_RANGE_FLOOR},
        {"name": "C3_downstream_consumer_can_diverge", "load_bearing": True,
         "passed": c3, "measured": float(max_div),
         "threshold": DIVERGENCE_FLOOR},
        {"name": "C4_instrument_control_bit_identical", "load_bearing": True,
         "passed": c4,
         "measured": float(max(control_div.values())) if control_div else 1.0,
         "threshold": CONTROL_DIVERGENCE_CEILING},
        {"name": "C5_prereadiness_reproduces_collapse", "load_bearing": False,
         "passed": c5, "measured": float(pre_max_targets_max),
         "threshold": 1.0, "comparator": "<="},
    ]
    combination_rule = (
        "overall_pass = C1 AND C2 AND C3 AND (ARM_READINESS readiness gate "
        "green) AND C4. C4 gates because the paired control covers BOTH arm "
        "identities and a non-zero self-yoked divergence on EITHER voids the "
        "paired DV. C1, C2 and the MAGNITUDE gate "
        "(lp_perturbation_can_reach_argmax_margin) are ALSO binding "
        "readiness preconditions on ARM_READINESS, so a C3 failure while "
        "any of them is unmet self-routes "
        "substrate_not_ready_requeue rather than any substrate verdict -- this "
        "is the specific defect that made V3-EXQ-964 uninterpretable. C4 is an "
        "negative control and does not gate. DIAGNOSTIC substrate-readiness check, NOT a "
        "governance-evidence verdict on MECH-482's own claim hypothesis."
    )

    if not rdy_green:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        direction = "supports"
        label = "multitarget_readiness_achieved_and_selection_relevant"
    elif c1 and c2 and not c3:
        # Only reachable with a DIFFERENTIATED readout -- so this is a genuine
        # measurement, unlike 964's arithmetic zero.
        direction = "mixed"
        label = "readout_differentiates_but_never_changes_committed_action"
    elif c1 and not c2:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    else:
        direction = "unknown"
        label = "substrate_not_ready_requeue"

    criteria_nd = arm_criteria_non_degenerate(
        {
            ARM_RDY: [
                "C1_multitarget_regime_reached",
                "C2_readout_differentiates",
                "C3_downstream_consumer_can_diverge",
            ],
        },
        aggregate,
        extra={"C5_prereadiness_reproduces_collapse": True},
    )

    readout = _flat_scalar({
        "rdy_max_n_targets_min": rdy_max_targets_min,
        "rdy_max_lp_dev_range_min": rdy_lp_range_min,
        "rdy_vacuous_readout_rate_max": rdy_vacuous_max,
        "rdy_undifferentiated_rate_max": max(
            _acc(rows, "undifferentiated_readout_rate", 1.0)),
        "pre_max_n_targets_max": pre_max_targets_max,
        "yoked_divergence_frac_max": max_div,
        "yoked_divergence_frac_mean": mean_div,
        "paired_control_divergence_max": max(control_div.values())
        if control_div else 1.0,
        "rdy_max_distinct_targets_matched_min": rdy_distinct_matched_min,
        "rdy_n_flip_reachable_ticks_min": rdy_flip_reachable_min,
        "rdy_max_pert_over_margin": rdy_pert_over_margin_max,
        "rdy_n_zero_margin_ticks_total": rdy_zero_margin_total,
        "rdy_n_flip_reachable_strict_total": rdy_flip_strict_total,
        "n_margin_reads_total": sum(_acc(rows, "n_margin_reads")),
        "n_latched_ticks_total": sum(_acc(rows, "n_latched_ticks")),
        "n_lp_reads_total": sum(_acc(rows, "n_lp_reads")),
        "c1_multitarget_regime_reached": c1,
        "c2_readout_differentiates": c2,
        "c3_downstream_can_diverge": c3,
        "c4_instrument_control_clean": c4,
        "c5_prereadiness_reproduces_collapse": c5,
        "overall_pass": overall_pass,
    })

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "dry_run": dry_run,
        "readout": readout,
        "metrics": dict(readout),
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": rows + pre_rows,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": aggregate["non_degenerate"],
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate.get(
                "preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
        },
        "custom_information": {
            "supersession_note": (
                "Supersedes V3-EXQ-964 "
                "(v3_exq_964_mech482_epistemic_deficit_validation_"
                "20260829T215030Z_v3), whose C2 was structurally unsatisfiable: "
                "n_targets==1 made the per-candidate readout constant, and a "
                "candidate-uniform constant cannot move an argmax. That run's "
                "yoked_divergence_frac_max of 0.0 was an arithmetic identity, "
                "not a measurement."
            ),
            "dv_symmetry_declaration": (
                "DV = committed action (argmax over candidate scores). Symmetry "
                "group: addition of a candidate-uniform constant; monotone "
                "rescaling. ARM_PREREADINESS's manipulation IS invariant under "
                "it (hard_match + single enclosing target -> constant lp_vec), "
                "so that arm is scoped out of C3 scoring and serves as the "
                "yoked reference and collapse control. ARM_READINESS's is NOT "
                "invariant (rbf_weighted is continuous in candidate position); "
                "C2 measures that rather than assuming it."
            ),
            "latched_diagnostic_note": (
                "StructuredCuriosity._last_lp_dev_range is written only inside "
                "compute_score_bias, which runs only on an E3 tick. The latch "
                "is cleared to None before every select_action; ticks where it "
                "stayed None are counted as n_latched_ticks and contribute NO "
                "observation, so n_lp_reads is the true denominator."
            ),
            "isolation_note": (
                "314a novelty and 314b uncertainty are OFF on both arms, and "
                "curiosity_learning_progress_source is 'epistemic_deficit' on "
                "BOTH, so the yoked divergence is attributable to the readiness "
                "GEOMETRY alone."
            ),
            "reuse_check_note": (
                "GOV-REUSE-1: the broadcast-vs-epistemic_deficit contrast is "
                "already recorded in V3-EXQ-964 and is not re-run. The "
                "readiness knobs did not exist at that substrate_hash, so the "
                "readiness question is not recoverable from it."
            ),
            "known_open_substrate_note": (
                "Runs under open substrate_queue entry "
                "sd_epistemic_deficit_multitarget_readiness (severity "
                "degrading, implemented_pending_validation) -- which is the "
                "very entry this run validates."
            ),
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
    }

    full_config = {
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "env": "CausalGridWorldV2",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": True,
        "curiosity_learning_progress_source": "epistemic_deficit",
        "epistemic_deficit_require_differentiated_readout": False,
        "e2_world_uncertainty_warmup_steps": _ACTIVE_WARMUP,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
        "arm_knobs": {ARM_PRE: PREREADINESS_KNOBS, ARM_RDY: READINESS_KNOBS},
        "thresholds": {
            "VACUOUS_READOUT_RATE_CEILING": VACUOUS_READOUT_RATE_CEILING,
            "MULTITARGET_FLOOR": MULTITARGET_FLOOR,
            "LP_DEV_RANGE_FLOOR": LP_DEV_RANGE_FLOOR,
            "DIVERGENCE_FLOOR": DIVERGENCE_FLOOR,
            "CONTROL_DIVERGENCE_CEILING": CONTROL_DIVERGENCE_CEILING,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return {"outcome": manifest["outcome"], "manifest": manifest,
            "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    args = ap.parse_args()

    global _ACTIVE_WARMUP
    if args.dry_run:
        episodes, steps, seeds = 2, 30, [71]
        _ACTIVE_WARMUP = 12
    else:
        episodes, steps, seeds = args.episodes, args.steps, SEEDS

    result = run_experiment(episodes, steps, seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"manifest: {out_path}", flush=True)
    print(json.dumps(result["manifest"]["readout"], indent=2), flush=True)

    if args.dry_run:
        m = result["manifest"]["readout"]
        checks = {
            "INSTRUMENT CONTROL: self-yoked arms bit-identical (==0)":
                m["paired_control_divergence_max"] == 0.0,
            "accumulator armed on BOTH arms (pre-readiness arm has targets)":
                m["pre_max_n_targets_max"] > 0,
            "latched-tick accounting is live (n_lp_reads>0)":
                m["n_lp_reads_total"] > 0,
            "DECISIVE READOUT ENGAGED: lp_dev_range observed at all":
                m["n_lp_reads_total"] > 0,
        }
        print("[smoke] multi-target readiness checks:", flush=True)
        for label, ok in checks.items():
            print(f"  [{'OK' if ok else 'XX'}] {label}", flush=True)
        print(f"  [info] C1 max_n_targets(min over seeds) = "
              f"{m['rdy_max_n_targets_min']} (floor {MULTITARGET_FLOOR})",
              flush=True)
        print(f"  [info] distinct targets MATCHED at a readout(min) = "
              f"{m['rdy_max_distinct_targets_matched_min']} (floor 2)",
              flush=True)
        print(f"  [info] C2 max_lp_dev_range(min over seeds) = "
              f"{m['rdy_max_lp_dev_range_min']} (floor {LP_DEV_RANGE_FLOOR})",
              flush=True)
        print(f"  [info] C3 magnitude gate: flip-reachable ticks(min) = "
              f"{m['rdy_n_flip_reachable_ticks_min']} (floor 1), "
              f"max pert/margin = {m['rdy_max_pert_over_margin']:.3e}",
              flush=True)
        print(f"  [info]    decomposition: strict(pert>=margin>0) = "
              f"{m['rdy_n_flip_reachable_strict_total']}, "
              f"zero-margin ties = {m['rdy_n_zero_margin_ticks_total']}",
              flush=True)
        print(f"  [info] n_latched_ticks total = {m['n_latched_ticks_total']} "
              f"(recorded NO observation, by design)", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(
        outcome=_outcome_raw,
        manifest_path=_out_path,
        dry_run=_dry,
    )
