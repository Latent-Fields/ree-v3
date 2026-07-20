"""Canonical OFF/baseline arm for the MECH-457 competence_floor RETENTION lineage.

Shared by the GOV-FANOUT-1 retention legs queued as a pair on 2026-07-19:

  * V3-EXQ-788  H-retention-critic          (value estimator: scalar vs distributional)
  * V3-EXQ-789  H-retention-auxiliary-decay (auxiliary persistence: constant vs annealed vs off)

A THIRD leg, H-retention-consolidation (update constraint: unconstrained vs KL-anchored to the
installed policy), became buildable 2026-07-19 when mech457_policy_kl_anchor landed, and its
knobs are threaded through reference_config() below (use_policy_kl_anchor / kl_anchor_coef). It
is NOT YET QUEUED -- queueing is governed by GOV-FANOUT-1 and routed through /queue-experiment.
Its OFF cell is this same shared control, so it reuses the banked baseline rather than minting.

Both legs share ONE control: BC-install the raw_view policy to the ~20.9 competence point, then
run the REFERENCE (non-regressed) RL refinement with no manipulation -- scalar critic, constant
bc_aux_coef. That shared control is what this module builds, so the two legs' OFF cells agree by
CONSTRUCTION rather than by two drivers happening to spell the same config.

WHY A MODULE RATHER THAN A COPIED BLOCK. The arm fingerprint is computed over a declared
config_slice; two drivers that spell "the same" OFF arm slightly differently produce different
fingerprints and never share a cell, while two that spell it the same by accident can share one
they did not mean to. Anchoring both on this module makes the match structural. Living under
experiments/_lib/** also auto-binds it into substrate_hash, so any edit here correctly refuses a
stale reuse rather than silently serving one.

MINT: emit the OFF cell with include_driver_script_in_hash=False (see off_path_config_slice
below). The two legs have DIFFERENT driver scripts, so folding the driver into the hash would
make every cross-driver reuse a guaranteed miss -- the exact defect fixed 2026-06-09
(arm_reuse_fingerprint_plan.md sec 9.7). The first of the two legs to run mints; the second
reuses. There is deliberately NO separate baseline-only mint job: neither sanctioned exception
applies (both legs run cloud-class, and no third consumer is planned ahead of them).

REFERENCE BUILD (load-bearing -- NOT the boot.make_on_config() defaults). The registry pins the
NON-REGRESSED build: 128-wide / 3x budget / z_world detached / credit-replay 3 / topk 32. The
module-level ON_* constants in mech457_bootstrap_explorer are the 769-FALSIFIED capacity
regression (ON_ACTOR_CRITIC_HIDDEN=256, ON_BUDGET_MULTIPLIER=5) and must NOT be used here. This
mirrors v3_exq_780's REF_* block, which is the worked precedent for the same distinction.

REPRESENTATION: raw_view ONLY. V3-EXQ-780 measured post-BC competence 20.933 on raw_view with
3/3 seeds taking the install, against 0.583 with 0/3 taking on z_world. A retention question is
unanswerable on a representation where the install does not take in the first place -- that is
an install failure, not a retention finding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_fanout as fan
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    evaluate_seed,
)
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734

# --- Reference (non-regressed) composed-bootstrap capacity -------------------------------
# Identical to v3_exq_780's REF_* block. Held FIXED across every arm of both legs; the ONLY
# thing a leg varies is its single declared manipulation.
REF_ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN   # 128 (NOT boot.ON_ACTOR_CRITIC_HIDDEN=256)
REF_BUDGET_MULTIPLIER = 3                           # 3x (NOT boot.ON_BUDGET_MULTIPLIER=5)
REF_CREDIT_PASSES = mech.CREDIT_REPLAY_PASSES       # 3
REF_CREDIT_TOPK = mech.CREDIT_TOPK                  # 32
REF_COTRAIN_ENCODER = False                         # z_world detached
REF_WARM_START_FRACTION = 0.0                       # no warm-start (770/780 reference ctrl)

# The install. 780 measured raw_view post-BC 20.933 at these settings, 3/3 seeds taking.
REF_REPRESENTATION = "raw_view"
BC_WARMSTART_EPISODES = fan.BC_EPISODES             # 300
BC_AUX_COEF_BASELINE = 0.5                          # the persistent auxiliary 780 ran

# Competence bands (pre-registered; declared, never derived from a run).
RND_PLATEAU = boot.RND_PLATEAU_5_22                 # 5.22
LIFT_COMPETENCE_TARGET = boot.LIFT_COMPETENCE_TARGET  # ~13.05
POST_BC_INSTALL_FLOOR = COMPETENCE_RESOURCE_FLOOR   # an install below this did NOT take


def reference_config(
    on_budget: Optional[int] = None,
    *,
    bc_aux_coef: float = BC_AUX_COEF_BASELINE,
    bc_aux_coef_end: Optional[float] = None,
    bc_aux_anneal_fraction: float = 0.0,
    use_distributional_critic: bool = False,
    use_policy_kl_anchor: bool = False,
    kl_anchor_coef: float = 0.0,
    retention_probe_every: Optional[int] = None,
) -> boot.BootstrapExplorerConfig:
    """The shared reference RL-refinement config.

    Every keyword is a leg's declared manipulation; the defaults are the OFF/baseline arm. The
    three legs vary DISJOINT knobs -- 788 varies use_distributional_critic (the value
    estimator), 789 varies the bc_aux_* triple (auxiliary persistence), and the third retention
    leg varies use_policy_kl_anchor/kl_anchor_coef (the update constraint) -- which is the
    anti-alias the portfolio depends on: a leg that moved two of them would read as neither.
    """
    budget = int(on_budget) if on_budget is not None else int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)
    return boot.BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=1.0,
        intrinsic_coef_end=boot.ON_INTRINSIC_COEF_END,
        anneal_fraction=boot.ON_ANNEAL_FRACTION,
        warm_start_fraction=REF_WARM_START_FRACTION,
        entropy_beta_start=boot.ON_ENTROPY_BETA_START,
        entropy_beta_end=boot.ON_ENTROPY_BETA_END,
        credit_replay=True,
        credit_replay_passes=REF_CREDIT_PASSES,
        credit_topk=REF_CREDIT_TOPK,
        n_episodes=budget,
        actor_critic_hidden=REF_ACTOR_CRITIC_HIDDEN,
        cotrain_encoder=REF_COTRAIN_ENCODER,
        bc_aux_coef=float(bc_aux_coef),
        bc_aux_coef_end=bc_aux_coef_end,
        bc_aux_anneal_fraction=float(bc_aux_anneal_fraction),
        use_distributional_critic=bool(use_distributional_critic),
        use_policy_kl_anchor=bool(use_policy_kl_anchor),
        kl_anchor_coef=float(kl_anchor_coef),
        retention_probe_every=retention_probe_every,
    )


def off_path_config_slice(
    env_kwargs: Dict[str, Any],
    *,
    eval_eps: int,
    steps: int,
    on_budget: Optional[int] = None,
    retention_probe_every: Optional[int] = None,
) -> Dict[str, Any]:
    """The declared config_slice for the shared OFF cell.

    Declares ONLY what the OFF computation reads: env, schedule, eval geometry, the install, and
    the reference build. It must NOT carry a leg's ON-arm gains, acceptance thresholds, or
    labels -- an under-declared slice yields a false HIT (two materially different arms sharing a
    fingerprint), and an over-declared one yields a needless MISS. retention_probe_every IS
    declared: a probed and an unprobed cell are bit-identical as COMPUTATIONS but are not
    interchangeable ARTIFACTS, since only one carries the trajectory a consumer reads.

    Emit with include_driver_script_in_hash=False so the two legs' distinct drivers can match.
    """
    cfg = reference_config(on_budget, retention_probe_every=retention_probe_every)
    base: Dict[str, Any] = {
        "arm_id": "retention_off",
        "rung_id": fan.RUNG_ID,
        "kind": "mech457_retention_baseline",
        "env_kwargs": dict(env_kwargs),
        "representation": REF_REPRESENTATION,
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
        "bc_warmstart_episodes": int(BC_WARMSTART_EPISODES),
        "bc_demonstrator": "local_view_greedy",
        "p0_warmup_episodes": 0,          # raw_view needs no encoder warmup
    }
    base.update(cfg.as_slice())
    return base


def build_off_arm(seed: int, env_kwargs: Dict[str, Any], *, steps: int,
                  cfg: Optional[boot.BootstrapExplorerConfig] = None) -> Any:
    """Construct the raw_view RepAgent at reference capacity (pre-install, pre-RL).

    Builds from cfg, so passing a treatment cfg yields the treatment arm -- both legs construct
    every arm through this one function rather than only their OFF cell.

    use_distributional_critic MUST be forwarded here: the critic swap is applied at REP
    CONSTRUCTION (like actor_critic_hidden/cotrain_encoder), not inside the training call. A
    version of this helper that dropped the kwarg would build the H-retention-critic treatment
    arm with the SCALAR critic -- silently collapsing it into the control, so the two arms
    return identical results and the leg reads as a clean null it never actually tested. That is
    invisible in the manifest (both arms look well-formed), which is exactly the class of
    degenerate-arm-read-as-a-verdict failure the substrate's raise-on-scalar-path guard exists
    to make loud. Caught in review of V3-EXQ-788, 2026-07-19.

    The KL-anchor knobs are deliberately ABSENT here, and that is not the same oversight: the
    anchor is an UPDATE-rule knob applied inside train_bootstrap_explorer, not a rep-construction
    one. Adding use_policy_kl_anchor to this call would be a no-op at best (make_rep does not
    accept it) -- the constraint must reach train_off_arm via cfg, which it does. Do not
    "symmetrise" this function with the critic swap.
    """
    cfg = cfg if cfg is not None else reference_config()
    warm_env = x734._make_env(seed, env_kwargs)
    return mech.make_rep(
        REF_REPRESENTATION, warm_env, seed=seed, p0=0, steps=int(steps),
        actor_critic_hidden=int(cfg.actor_critic_hidden),
        cotrain_encoder=bool(cfg.cotrain_encoder),
        use_distributional_critic=bool(cfg.use_distributional_critic),
    )


def install_bc_prior(rep_agent: Any, seed: int, env_kwargs: Dict[str, Any], *,
                     steps: int, eval_eps: int, arm_label: str) -> Dict[str, Any]:
    """BC warm-start to the ~20.9 competence point, then measure whether the install TOOK.

    Returns post_bc_foraging_competence and install_took. This is the covariate V3-EXQ-780
    declared but never consumed -- its grid enumerated only a ~0 null, so a manipulation that
    succeeded ABOVE target was scored a null and self-routed bc_prior_not_the_axis (rejected on
    autopsy). Both retention legs MUST route on it: an install that did not take is
    UNINFORMATIVE about retention and self-routes substrate_not_ready_requeue, never a
    retention verdict.
    """
    demo = LocalViewGreedyPolicy(seed=seed)
    bc_env = x734._make_env(seed, env_kwargs)
    wguard = mech.warmstart_bc_rep(
        rep_agent, bc_env, seed=seed, n_bc=int(BC_WARMSTART_EPISODES), steps=int(steps),
        demo=demo, arm_label=arm_label, denom=int(BC_WARMSTART_EPISODES),
    )
    postbc_env = x734._make_env(seed, env_kwargs)
    postbc_row = evaluate_seed(
        rep_agent.eval_policy(arm_label + "_postbc"), postbc_env, int(eval_eps), int(steps)
    )
    post_bc = float(postbc_row["foraging_competence"])
    return {
        "demo": demo,
        "post_bc_foraging_competence": round(post_bc, 6),
        "install_took": bool(post_bc >= POST_BC_INSTALL_FLOOR),
        "bc_warmstart_action_match_recent": round(
            float(wguard.get("bc_warmstart_action_match_accuracy_recent", 0.0)), 6
        ),
    }


def make_probe_fn(rep_agent: Any, seed: int, env_kwargs: Dict[str, Any], *,
                  steps: int, eval_eps: int, arm_label: str):
    """A non-perturbing mid-training competence probe for train_a2c's probe_fn hook.

    Builds a FRESH env per reading so the probe never touches the training env (contract T5),
    and evaluates through the deterministic-argmax eval policy. train_a2c snapshots and restores
    the torch/numpy/random streams around every call, so measurement neutrality is guaranteed by
    the substrate (contract T2) rather than by this function's good behaviour.
    """
    def _probe(ep: int) -> Dict[str, Any]:
        probe_env = x734._make_env(seed, env_kwargs)
        row = evaluate_seed(
            rep_agent.eval_policy(f"{arm_label}_probe_ep{int(ep)}"),
            probe_env, int(eval_eps), int(steps),
        )
        return {"foraging_competence": round(float(row["foraging_competence"]), 6)}

    return _probe


def train_off_arm(rep_agent: Any, seed: int, env_kwargs: Dict[str, Any], *,
                  steps: int, arm_label: str, cfg: boot.BootstrapExplorerConfig,
                  demo: Optional[Any] = None, probe_fn: Optional[Any] = None) -> Dict[str, Any]:
    """Run the reference RL refinement on an installed policy; returns the train_a2c guard dict
    (which carries competence_trajectory, bc_aux_coef_first/_last)."""
    train_env = x734._make_env(seed, env_kwargs)
    return boot.train_bootstrap_explorer(
        rep_agent, train_env, seed=seed, steps=int(steps), arm_label=arm_label, cfg=cfg,
        denom=int(cfg.n_episodes), bc_demo=demo, probe_fn=probe_fn,
    )


# --- Trajectory statistics (shared DV machinery) -------------------------------------------

def retained_fraction(trajectory: List[Dict[str, Any]], post_bc: float) -> Optional[float]:
    """Terminal trajectory competence as a fraction of the installed competence.

    1.0 = fully retained, 0.0 = eroded to nothing. None when the trajectory is empty or the
    install was ~0 (the ratio is undefined, NOT zero -- reporting 0.0 there would manufacture a
    maximal-erosion reading out of an install that never happened).
    """
    if not trajectory or post_bc <= 0.0:
        return None
    final = float(trajectory[-1].get("foraging_competence", 0.0))
    return round(final / float(post_bc), 6)


def competence_half_life(trajectory: List[Dict[str, Any]], post_bc: float) -> Optional[float]:
    """Episodes until competence first falls below half the installed value.

    The DV H-retention-auxiliary-decay is defined on -- a trajectory statistic by construction.
    Returns None when the install was ~0 (undefined) or when competence NEVER halves. None means
    "did not decay", which is a substantively different reading from a large number, so callers
    must branch on it rather than coercing it to a sentinel.
    """
    if not trajectory or post_bc <= 0.0:
        return None
    half = 0.5 * float(post_bc)
    for row in trajectory:
        if float(row.get("foraging_competence", 0.0)) < half:
            return float(row.get("episode", 0))
    return None
