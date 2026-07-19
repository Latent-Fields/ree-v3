"""MECH-457 competence bootstrap explorer -- the composed floor->competent build.

WHY THIS EXISTS. The GOV-FANOUT-1 discrimination is CLOSED (five autopsies, 751-756):
no SINGLE mechanism and no PAIRWISE combination clears the 1.0 foraging floor toward the
48.05 local-view competence ceiling. The wall is ONE structural property with two joined
halves, not an unfound mechanism (failure_autopsy_MECH-457-fanout-755_2026-07-15 cluster read):

  HALF 1 -- cold-start / success-dependence. Prioritized backward credit-replay (752),
    Go-Explore archive+return (753), AMIGo goal-frontier (754), and the credit x return
    PAIR (756) all amplify signal DERIVED FROM PRIOR TASK SUCCESS, so they collapse sub-floor
    from ~0 competence. Only the two SUCCESS-INDEPENDENT classes in the arc break the floor:
    RND novelty (751 -> 5.22) and BC imitation (748 -> 32.72, needs an expert).

  HALF 2 -- capacity to convert. Even a competent explore/exploit arbitration GATE (755)
    adds nothing over fixed-coefficient RND. RND reaches only ~11% of the 48.05 ceiling
    because the explorer lacks the CAPACITY / BUDGET / CREDIT to convert coverage into
    competence -- not because exploration is scheduled wrong.

THE MISSING PRIMITIVE (this module). Every fanout CONVERTER (752 credit, 753 return, 756
pair) ran on the SPARSE base -- no dense drive -> nothing to convert. The genuinely-untested,
highest-composition build is the RND success-independent drive COMPOSED WITH a converter,
plus adequate budget and a DEVELOPMENTAL explore->exploit anneal that lets the extrinsic
forage reward take over as competence rises. This module composes ONLY landed pieces
(honours the lit-pull duplication objection -- SYNTHESIS.md rejected building another novelty
module):

  * RND success-independent dense drive     -- experiments/_lib/mech457_explorer_classes.RNDModule
    (Burda et al. 2018; == ARC-065/MECH-314 novelty class REE already owns). HALF 1.
  * first-class RPE actor-critic            -- ree_core/action_learning/actor_critic.ActorCriticPolicy
    via the RepAgent adapters (z_world cotrain AND raw 5x5, per the reuse directive).
  * prioritized backward credit-replay      -- mech457_explorer_classes._prioritized_credit_replay
    (Mattar & Daw 2018 priority + Foster & Wilson 2006 reverse replay), NOW fed by the
    RND-generated successes 752 lacked. HALF 2 (converter).
  * training-progress intrinsic-coef anneal -- THE NEW COUPLING (linear_anneal): a
    DEVELOPMENTAL schedule coef_start->coef_end over anneal_fraction of training, so coverage
    consolidates into exploitation. Deliberately NOT the critic-utility ModeGate 755 refuted
    (that gate is downstream of competence -- it modulates an already-competent policy and
    cannot manufacture the competence a low-capacity learner never had). LC-NE explore/exploit
    consolidation (Aston-Jones & Cohen 2005; Daw 2006) instantiated as an ontogenetic schedule.
  * increased training budget                -- n_episodes (HALF 2, capacity) above the 1000
    that plateaus RND at 5.22.

NO-OP DEFAULT / OFF == 751 RND PLATEAU. BootstrapExplorerConfig defaults reproduce the 751
RND-explorer arm (constant coef 1.0, no anneal, no credit-replay, budget 1000). make_off_config
/ make_on_config give the validation ablation pair. The training UPDATE lives here in _lib (as
actor_critic.py mandates and as the fanout mechanisms do) so every (config x representation x
seed) cell folds into the arm_fingerprint substrate_hash via the experiments/_lib/** glob and
can be emitted reuse-eligible.

MECH-094: N/A -- no memory writes on simulated / non-waking ticks; RND trains on detached
features and the actor emits an action. Phased training: N/A -- no new encoder head (the RND
predictor and the actor are trained by their own losses; z_world cotrain follows the existing
742/751 single-optimizer path).

REFERENCE BAND (foraging_competence @D3, resource/ep; cited constants, same substrate): floor
1.0; RND novelty plateau (751) 5.22; BC expert (748) 32.72; local-view ceiling (738) 48.05;
greedy oracle (742) 57.2; lift_above_plateau_threshold ~7.83 (a mechanism "lifts" only if its
mean forage exceeds the 5.22 plateau by >=50%, i.e. >=~13.05 res/ep).

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from experiments._lib.capability_eval import COMPETENCE_RESOURCE_FLOOR
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_fanout as fan

# Re-exported reference constants (single source of truth = mech457_explorer_classes).
RND_PLATEAU_5_22 = mech.RND_PLATEAU_5_22
BC_REFERENCE_32_72 = mech.BC_REFERENCE_32_72
CEILING_48_05 = mech.CEILING_48_05
ORACLE_57_2 = mech.ORACLE_57_2
LIFT_ABOVE_PLATEAU = mech.LIFT_ABOVE_PLATEAU        # ~7.83 (the absolute lift margin)
LIFT_COMPETENCE_TARGET = round(RND_PLATEAU_5_22 + LIFT_ABOVE_PLATEAU, 6)  # ~13.05 res/ep

# ON-arm defaults (the composed bootstrap; each is a design choice, not a tuned magic number).
ON_INTRINSIC_COEF_START = 1.0        # start fully explore (== the 751 plateau coefficient)
ON_INTRINSIC_COEF_END = 0.05         # anneal toward exploit (small residual coverage floor)
ON_ANNEAL_FRACTION = 0.6             # anneal over 60% of training (AFTER the warm-start), then hold
ON_ENTROPY_BETA_START = mech.EXPLORE_ENTROPY_BETA    # 0.10 (raised-explore start)
ON_ENTROPY_BETA_END = mech.AC_ENTROPY_BETA           # 0.03 (sparse-baseline exploit level)

# ------------------------------------------------------------------------------------------
# MECH-457 capacity-side amend (2026-07-16, routed from failure_autopsy_V3-EXQ-765). The 765
# retest showed the composed DRIVE half works on raw (ON 6.48 vs OFF 0.62, +5.87 converter-
# driven lift) but the actor-critic PLATEAUS at ~13% of the 48.05 achievable ceiling and clears
# neither representation's 13.05 lift-competence target, with large raw seed variance
# (15.9/3.05/0.5), while z_world cotrain is DESTRUCTIVE (ON 0.35 < OFF 5.22). Three knobs on ONE
# build (NOT a discrimination fanout -- capacity + reliability + z_world-detach are joint):
#   (a) CAPACITY   -- raise the actor-critic policy/critic width + the training budget.
#   (b) RELIABILITY -- a full-explore WARM-START before the anneal + more credit-replay passes /
#                      wider top-|TD| selection, to cut the seed variance (unreliable convert).
#   (c) INTEGRATION -- default the z_world path to DETACHED (train the policy on the frozen
#                      prediction-trained encoder, Stooke 2021), since cotrain corrupts z_world.
# All three are OFF-preserving no-op defaults on BootstrapExplorerConfig (bit-identical OFF).
ON_BUDGET_MULTIPLIER = 5             # 5x the 1000-ep plateau budget (was 3x in 765; capacity)
ON_ACTOR_CRITIC_HIDDEN = 256         # 2x the 742/765 trunk width of 128 (policy capacity)
ON_WARM_START_FRACTION = 0.2         # hold full-explore coef/entropy for the first 20% (warm-start)
ON_CREDIT_REPLAY_PASSES = 6          # 2x the 752-756 CREDIT_REPLAY_PASSES=3 (credit reliability)
ON_CREDIT_TOPK = 64                  # 2x the 752-756 CREDIT_TOPK=32 (wider credit selection)
ON_COTRAIN_ENCODER = False           # z_world DETACHED (frozen encoder; cotrain was destructive)


@dataclass
class BootstrapExplorerConfig:
    """Composed-explorer specification. Defaults == the 751 RND plateau (no-op OFF)."""

    # HALF 1 -- success-independent dense drive.
    use_rnd: bool = True                 # RND novelty bonus (the off-floor generator)

    # THE NEW PRIMITIVE -- developmental intrinsic-coefficient anneal.
    # coef_start == coef_end (or anneal_fraction 0) => constant coefficient (751 plateau OFF).
    intrinsic_coef_start: float = mech.INTRINSIC_COEF   # 1.0
    intrinsic_coef_end: float = mech.INTRINSIC_COEF     # 1.0 -- OFF: no anneal
    anneal_fraction: float = 0.0                        # OFF: 0 -> constant coefficient

    # Optional entropy anneal alongside the coefficient (OFF: constant explore beta).
    entropy_beta_start: float = mech.EXPLORE_ENTROPY_BETA   # 0.10
    entropy_beta_end: float = mech.EXPLORE_ENTROPY_BETA     # 0.10 -- OFF: constant

    # Warm-start (MECH-457 capacity-amend): hold coef/entropy at their START values for the first
    # `warm_start_fraction` of episodes (a guaranteed full-explore coverage phase) BEFORE the
    # anneal begins -- cuts the seed variance from premature exploitation. OFF: 0.0 (no warm-start).
    warm_start_fraction: float = 0.0

    # HALF 2 -- converter + capacity.
    credit_replay: bool = False          # prioritized backward credit-replay converter (OFF)
    credit_replay_passes: int = mech.CREDIT_REPLAY_PASSES  # 3 -- OFF: the 752-756 value
    credit_topk: int = mech.CREDIT_TOPK                    # 32 -- OFF: the 752-756 value
    n_episodes: int = fan.RL_EPISODES    # 1000 -- OFF: the plateau budget

    # Policy capacity + z_world integration mode (MECH-457 capacity-amend). Defaults reproduce
    # the 742/751/765 arms byte-identical: 128-wide trunk, z_world co-shaped (cotrain).
    actor_critic_hidden: int = fan.ACTOR_CRITIC_HIDDEN    # 128 -- OFF: the plateau width
    cotrain_encoder: bool = True         # z_world co-shape (OFF); ON detaches (frozen encoder)

    # COMPETENCE-DIRECTED BOOTSTRAP hooks (GOV-FANOUT-1 H-bc-prior / H-approach-primitive,
    # 2026-07-18; routed by failure_autopsy_MECH-457-fanout-770-771-772). All no-op-default OFF.
    #   * bc_aux_coef (H-bc-prior, learning-signal): weight of the persistent imitation CE
    #     auxiliary (bc_demo passed to train_bootstrap_explorer). OFF: 0.0 -> no BC term.
    #   * use_approach_primitive / approach_coef (H-approach-primitive, intrinsic-architecture):
    #     enable a non-extinguishing appetitive resource-proximity intrinsic drive of weight
    #     approach_coef. OFF: False/0.0 -> no approach term.
    #   * bc_aux_coef_end / bc_aux_anneal_fraction (mech457_bc_aux_schedule,
    #     H-retention-auxiliary-decay, 2026-07-18): make the imitation auxiliary's PERSISTENCE
    #     sweepable -- constant (end=None), annealed (end<start over the first
    #     bc_aux_anneal_fraction of episodes), or off (bc_aux_coef=0.0). OFF: None/0.0 ->
    #     constant bc_aux_coef, byte-identical to the 780/781 callers.
    #     Uses linear_anneal, NOT warm_then_anneal: the latter is parameterised by the SHARED
    #     warm_start_fraction, which would couple BC persistence to the exploration anneal and
    #     confound the leg's single intervention.
    bc_aux_coef: float = 0.0
    bc_aux_coef_end: Optional[float] = None
    bc_aux_anneal_fraction: float = 0.0
    use_approach_primitive: bool = False
    approach_coef: float = 0.0

    # DISTRIBUTIONAL CRITIC (GOV-FANOUT-1 H-retention-critic, 2026-07-18; routed by
    # failure_autopsy_MECH-457-gov-fanout-1-cluster-780-781-782). Swaps the VALUE ESTIMATOR
    # only -- the scalar value head becomes a categorical head over a symlog bin support,
    # trained by cross-entropy against an HL-Gauss projection of the return and decoded to a
    # scalar by expectation. Motivated by the measured V3-EXQ-782 R-(b) reading: the shared
    # CTRL critic is flat and uninformed (std(V)/std(G)=0.041 vs a 0.25 collapse threshold).
    # Applied at REP CONSTRUCTION (make_rep), like actor_critic_hidden/cotrain_encoder.
    # ANTI-ALIAS: leaves the policy update untouched (that locus is mech457_policy_kl_anchor
    # / H-retention-consolidation). OFF: False -> the scalar-MSE critic, byte-identical.
    use_distributional_critic: bool = False

    # RETENTION TRAJECTORY PROBE (mech457_retention_trajectory_probe, 2026-07-19). Episode
    # cadence for the non-perturbing mid-training competence probe. The competence_floor
    # retention legs must read a post-installation competence TRAJECTORY rather than terminal
    # competence (portfolio 2026-07-18 sec 53); terminal-only measurement is what kept the
    # deficit invisible for ten legs. INSTRUMENTATION ONLY -- no update rule, loss term,
    # schedule or value estimator changes, so it is orthogonal to all three manipulation knobs
    # above. OFF: None -> no probe, empty trajectory, byte-identical.
    # Declared in as_slice() for the same reason bc_aux_coef_end is: a varyable knob absent from
    # the config_slice would let two materially different arms share an arm fingerprint. It is
    # declared even though the probe cannot change the learned result, because a probed and an
    # unprobed cell are not interchangeable ARTIFACTS -- only one carries the trajectory a
    # consumer may later read.
    retention_probe_every: Optional[int] = None

    def as_slice(self) -> Dict[str, Any]:
        """Config fields for the arm_fingerprint config_slice / manifest (declared)."""
        return {
            "use_rnd": bool(self.use_rnd),
            "intrinsic_coef_start": float(self.intrinsic_coef_start),
            "intrinsic_coef_end": float(self.intrinsic_coef_end),
            "anneal_fraction": float(self.anneal_fraction),
            "warm_start_fraction": float(self.warm_start_fraction),
            "entropy_beta_start": float(self.entropy_beta_start),
            "entropy_beta_end": float(self.entropy_beta_end),
            "credit_replay": bool(self.credit_replay),
            "credit_replay_passes": int(self.credit_replay_passes),
            "credit_topk": int(self.credit_topk),
            "n_episodes": int(self.n_episodes),
            "actor_critic_hidden": int(self.actor_critic_hidden),
            "cotrain_encoder": bool(self.cotrain_encoder),
            "bc_aux_coef": float(self.bc_aux_coef),
            "bc_aux_coef_end": (
                None if self.bc_aux_coef_end is None else float(self.bc_aux_coef_end)
            ),
            "bc_aux_anneal_fraction": float(self.bc_aux_anneal_fraction),
            "use_approach_primitive": bool(self.use_approach_primitive),
            "approach_coef": float(self.approach_coef),
            "use_distributional_critic": bool(self.use_distributional_critic),
            "retention_probe_every": (
                None if self.retention_probe_every is None else int(self.retention_probe_every)
            ),
        }


def make_off_config(n_episodes: Optional[int] = None) -> BootstrapExplorerConfig:
    """The OFF / RND-plateau reproduction arm: RND drive, constant coef 1.0, no anneal, no
    credit-replay, plateau budget. Reproduces the 751 ~5.22 band (drift guard + reuse mint)."""
    cfg = BootstrapExplorerConfig()
    if n_episodes is not None:
        cfg.n_episodes = int(n_episodes)
    return cfg


def make_on_config(budget_multiplier: int = ON_BUDGET_MULTIPLIER) -> BootstrapExplorerConfig:
    """The ON / composed-bootstrap arm (MECH-457 capacity-side amend): RND drive + a full-explore
    WARM-START + developmental coef+entropy anneal + prioritized credit-replay (reliability-
    raised passes/topk) + increased policy capacity + 5x budget, with the z_world path DETACHED
    (train the policy on the frozen prediction-trained encoder). Targets floor->competent."""
    return BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=ON_INTRINSIC_COEF_START,
        intrinsic_coef_end=ON_INTRINSIC_COEF_END,
        anneal_fraction=ON_ANNEAL_FRACTION,
        warm_start_fraction=ON_WARM_START_FRACTION,
        entropy_beta_start=ON_ENTROPY_BETA_START,
        entropy_beta_end=ON_ENTROPY_BETA_END,
        credit_replay=True,
        credit_replay_passes=ON_CREDIT_REPLAY_PASSES,
        credit_topk=ON_CREDIT_TOPK,
        n_episodes=int(fan.RL_EPISODES * int(budget_multiplier)),
        actor_critic_hidden=ON_ACTOR_CRITIC_HIDDEN,
        cotrain_encoder=ON_COTRAIN_ENCODER,
    )


def linear_anneal(v_start: float, v_end: float, frac: float, ep: int, n_episodes: int) -> float:
    """Linear schedule from v_start to v_end over the first `frac` of n_episodes, then hold at
    v_end. frac <= 0 (or v_start == v_end) -> constant v_start (the no-op OFF path)."""
    if frac <= 0.0:
        return float(v_start)
    cutoff = max(1.0, float(frac) * float(n_episodes))
    t = min(1.0, float(ep) / cutoff)
    return float(v_start + (v_end - v_start) * t)


def warm_then_anneal(
    v_start: float, v_end: float, warm_frac: float, anneal_frac: float, ep: int, n_episodes: int
) -> float:
    """Warm-start + linear anneal (MECH-457 capacity-amend). Hold v_start for the first
    `warm_frac` of n_episodes (a guaranteed full-explore coverage phase), then linearly anneal to
    v_end over the next `anneal_frac` of n_episodes, then hold v_end. warm_frac <= 0 reduces
    EXACTLY to linear_anneal(v_start, v_end, anneal_frac, ...); anneal_frac <= 0 or v_start ==
    v_end -> constant v_start (the no-op OFF path)."""
    if anneal_frac <= 0.0 or v_start == v_end:
        return float(v_start)
    warm_cut = max(0.0, float(warm_frac)) * float(n_episodes)
    if float(ep) < warm_cut:
        return float(v_start)
    span = max(1.0, float(anneal_frac) * float(n_episodes))
    t = min(1.0, (float(ep) - warm_cut) / span)
    return float(v_start + (v_end - v_start) * t)


def train_bootstrap_explorer(
    rep: mech.RepAgent, env: Any, seed: int, steps: int, arm_label: str,
    cfg: BootstrapExplorerConfig, denom: Optional[int] = None,
    bc_demo: Optional[Any] = None,
    probe_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Train the composed bootstrap explorer on `rep` (z_world cotrain or raw 5x5) for
    cfg.n_episodes episodes, returning the mech457_explorer_classes.train_a2c guard dict.

    Composition: RND drive (cfg.use_rnd) + developmental warm-start+coef/entropy anneal
    (warm_then_anneal over cfg.warm_start_fraction/cfg.anneal_fraction) + prioritized credit-
    replay (cfg.credit_replay, cfg.credit_replay_passes/credit_topk). The policy capacity
    (cfg.actor_critic_hidden), the z_world co-shape-vs-frozen mode (cfg.cotrain_encoder) and the
    distributional-critic swap (cfg.use_distributional_critic) are applied at REP CONSTRUCTION
    (make_rep), not here. With default (OFF) cfg this is the 751
    RND-plateau arm (constant coef, no warm-start, no anneal, no credit, plateau budget/width).

    bc_demo (H-bc-prior, V3-EXQ-780): OPTIONAL demonstrator Policy for the persistent imitation
    AUXILIARY. When set with cfg.bc_aux_coef>0, a CE(actor_logits, demo_action)*coef term is added
    to each episode loss (the BC warm-start is a separate, prior driver-side step via
    mech.warmstart_bc_rep). cfg.use_approach_primitive/cfg.approach_coef (H-approach-primitive,
    V3-EXQ-781): OPTIONAL non-extinguishing appetitive resource-proximity intrinsic drive. Both
    default OFF -> byte-identical to the pre-existing 765/769/770/771/772 callers."""
    n_episodes = int(cfg.n_episodes)
    denom = int(denom) if denom is not None else n_episodes
    # Config preconditions FIRST, before any module is constructed -- a misconfigured arm should
    # fail on its config, not partway through allocation. Reads the MAX of start and end: a
    # ramp-UP cell (start 0.0 -> end >0) still needs a demonstrator and would slip past a
    # start-only check.
    bc_aux_peak = max(
        float(cfg.bc_aux_coef),
        float(cfg.bc_aux_coef_end) if cfg.bc_aux_coef_end is not None else 0.0,
    )
    if bc_aux_peak > 0.0 and bc_demo is None:
        raise ValueError(
            "train_bootstrap_explorer: a nonzero BC auxiliary weight (cfg.bc_aux_coef or "
            "cfg.bc_aux_coef_end) requires a bc_demo demonstrator."
        )
    # Retention trajectory probe: the cadence lives on the config (so it is fingerprint-declared)
    # while the probe CLOSURE is passed by the driver (it needs a fresh env + the demonstrator-
    # free eval policy, neither of which this module owns). Both-or-neither is enforced by
    # train_a2c, which raises on a half-wired probe rather than silently returning an empty
    # trajectory.
    if (cfg.retention_probe_every is None) != (probe_fn is None):
        raise ValueError(
            "train_bootstrap_explorer: cfg.retention_probe_every and probe_fn must be supplied "
            "together (got retention_probe_every=%r, probe_fn=%r)."
            % (cfg.retention_probe_every, "set" if probe_fn is not None else None)
        )
    intrinsic = mech.RNDModule(rep.feature_dim) if cfg.use_rnd else None
    approach_drive = mech.resource_proximity if bool(cfg.use_approach_primitive) else None

    coef_schedule = (
        lambda ep, n: warm_then_anneal(
            cfg.intrinsic_coef_start, cfg.intrinsic_coef_end,
            cfg.warm_start_fraction, cfg.anneal_fraction, ep, n
        )
    )
    entropy_schedule = (
        lambda ep, n: warm_then_anneal(
            cfg.entropy_beta_start, cfg.entropy_beta_end,
            cfg.warm_start_fraction, cfg.anneal_fraction, ep, n
        )
    )
    # None when no end value is declared -> train_a2c holds the constant bc_aux_coef (OFF path).
    bc_aux_schedule = (
        None if cfg.bc_aux_coef_end is None else
        (lambda ep, n: linear_anneal(
            cfg.bc_aux_coef, float(cfg.bc_aux_coef_end), cfg.bc_aux_anneal_fraction, ep, n
        ))
    )

    return mech.train_a2c(
        rep, env, seed=seed, n_episodes=n_episodes, steps=steps,
        arm_label=arm_label, denom=denom,
        intrinsic=intrinsic,
        entropy_beta=cfg.entropy_beta_start,
        intrinsic_coef=cfg.intrinsic_coef_start,
        credit_replay=bool(cfg.credit_replay),
        credit_replay_passes=int(cfg.credit_replay_passes),
        credit_topk=int(cfg.credit_topk),
        coef_schedule=coef_schedule,
        entropy_schedule=entropy_schedule,
        bc_demo=bc_demo,
        bc_aux_coef=float(cfg.bc_aux_coef),
        bc_aux_schedule=bc_aux_schedule,
        approach_drive=approach_drive,
        approach_coef=float(cfg.approach_coef),
        probe_every=cfg.retention_probe_every,
        probe_fn=probe_fn,
    )


def reference_band() -> Dict[str, Any]:
    return {
        "floor": float(COMPETENCE_RESOURCE_FLOOR),
        "rnd_novelty_plateau_751": RND_PLATEAU_5_22,
        "bc_expert_748": BC_REFERENCE_32_72,
        "local_view_ceiling_738": CEILING_48_05,
        "greedy_oracle_742": ORACLE_57_2,
        "lift_above_plateau_threshold": round(LIFT_ABOVE_PLATEAU, 6),
        "lift_competence_target": LIFT_COMPETENCE_TARGET,
    }
