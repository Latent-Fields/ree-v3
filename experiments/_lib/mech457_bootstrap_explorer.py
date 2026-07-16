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
from typing import Any, Dict, List, Optional

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
ON_ANNEAL_FRACTION = 0.6             # anneal over the first 60% of training, then hold
ON_ENTROPY_BETA_START = mech.EXPLORE_ENTROPY_BETA    # 0.10 (raised-explore start)
ON_ENTROPY_BETA_END = mech.AC_ENTROPY_BETA           # 0.03 (sparse-baseline exploit level)
ON_BUDGET_MULTIPLIER = 3             # 3x the 1000-episode budget that plateaus RND (capacity)


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

    # HALF 2 -- converter + capacity.
    credit_replay: bool = False          # prioritized backward credit-replay converter (OFF)
    n_episodes: int = fan.RL_EPISODES    # 1000 -- OFF: the plateau budget

    def as_slice(self) -> Dict[str, Any]:
        """Config fields for the arm_fingerprint config_slice / manifest (declared)."""
        return {
            "use_rnd": bool(self.use_rnd),
            "intrinsic_coef_start": float(self.intrinsic_coef_start),
            "intrinsic_coef_end": float(self.intrinsic_coef_end),
            "anneal_fraction": float(self.anneal_fraction),
            "entropy_beta_start": float(self.entropy_beta_start),
            "entropy_beta_end": float(self.entropy_beta_end),
            "credit_replay": bool(self.credit_replay),
            "n_episodes": int(self.n_episodes),
        }


def make_off_config(n_episodes: Optional[int] = None) -> BootstrapExplorerConfig:
    """The OFF / RND-plateau reproduction arm: RND drive, constant coef 1.0, no anneal, no
    credit-replay, plateau budget. Reproduces the 751 ~5.22 band (drift guard + reuse mint)."""
    cfg = BootstrapExplorerConfig()
    if n_episodes is not None:
        cfg.n_episodes = int(n_episodes)
    return cfg


def make_on_config(budget_multiplier: int = ON_BUDGET_MULTIPLIER) -> BootstrapExplorerConfig:
    """The ON / composed-bootstrap arm: RND drive + developmental coef+entropy anneal +
    prioritized credit-replay converter + increased budget. Targets floor->competent."""
    return BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=ON_INTRINSIC_COEF_START,
        intrinsic_coef_end=ON_INTRINSIC_COEF_END,
        anneal_fraction=ON_ANNEAL_FRACTION,
        entropy_beta_start=ON_ENTROPY_BETA_START,
        entropy_beta_end=ON_ENTROPY_BETA_END,
        credit_replay=True,
        n_episodes=int(fan.RL_EPISODES * int(budget_multiplier)),
    )


def linear_anneal(v_start: float, v_end: float, frac: float, ep: int, n_episodes: int) -> float:
    """Linear schedule from v_start to v_end over the first `frac` of n_episodes, then hold at
    v_end. frac <= 0 (or v_start == v_end) -> constant v_start (the no-op OFF path)."""
    if frac <= 0.0:
        return float(v_start)
    cutoff = max(1.0, float(frac) * float(n_episodes))
    t = min(1.0, float(ep) / cutoff)
    return float(v_start + (v_end - v_start) * t)


def train_bootstrap_explorer(
    rep: mech.RepAgent, env: Any, seed: int, steps: int, arm_label: str,
    cfg: BootstrapExplorerConfig, denom: Optional[int] = None,
) -> Dict[str, Any]:
    """Train the composed bootstrap explorer on `rep` (z_world cotrain or raw 5x5) for
    cfg.n_episodes episodes, returning the mech457_explorer_classes.train_a2c guard dict.

    Composition: RND drive (cfg.use_rnd) + developmental coef/entropy anneal
    (linear_anneal over cfg.anneal_fraction) + prioritized credit-replay (cfg.credit_replay).
    With default (OFF) cfg this is the 751 RND-plateau arm (constant coef, no anneal, no
    credit, plateau budget)."""
    n_episodes = int(cfg.n_episodes)
    denom = int(denom) if denom is not None else n_episodes
    intrinsic = mech.RNDModule(rep.feature_dim) if cfg.use_rnd else None

    coef_schedule = (
        lambda ep, n: linear_anneal(
            cfg.intrinsic_coef_start, cfg.intrinsic_coef_end, cfg.anneal_fraction, ep, n
        )
    )
    entropy_schedule = (
        lambda ep, n: linear_anneal(
            cfg.entropy_beta_start, cfg.entropy_beta_end, cfg.anneal_fraction, ep, n
        )
    )

    return mech.train_a2c(
        rep, env, seed=seed, n_episodes=n_episodes, steps=steps,
        arm_label=arm_label, denom=denom,
        intrinsic=intrinsic,
        entropy_beta=cfg.entropy_beta_start,
        intrinsic_coef=cfg.intrinsic_coef_start,
        credit_replay=bool(cfg.credit_replay),
        coef_schedule=coef_schedule,
        entropy_schedule=entropy_schedule,
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
