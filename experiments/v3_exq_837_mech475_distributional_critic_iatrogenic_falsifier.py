#!/opt/local/bin/python3
"""V3-EXQ-837 -- MECH-475 DECISIVE FALSIFIER: is the destructiveness of added optimisation
pressure the UNINFORMATIVE VALUE BASELINE's doing? Re-run the THREE destructive ACQUISITION-side
treatments with the V3-EXQ-788 DISTRIBUTIONAL (informative) CRITIC in place and ask whether the
treatment-below-control inversion REVERSES OR FLATTENS.

EVIDENCE experiment (experiment_purpose=evidence; claim_ids=["MECH-475"]). This is the claim's
own what_would_answer (a) -- the decisive test. MECH-475 is a candidate/v3_pending mechanism
hypothesis (split_from MECH-457, registered 2026-07-22): the value baseline is UNINFORMATIVE on
the policy's own state distribution, so the advantage carries variance rather than signal and
ADDED OPTIMISATION PRESSURE IS IATROGENIC -- capacity/budget, an earned approach drive, and
metabolic reward-coupling each drove competence BELOW THEIR OWN CONTROLS (raw ON 6.48->0.12;
2.983->0.200; 3.47->1.12/death 1.0), rather than plateauing beneath a ceiling. A ceiling
plateaus; this does not. The positive control already exists: V3-EXQ-788 (clean PASS) showed a
distributional critic RETAINS installed competence where the scalar critic does not, i.e. making
the baseline informative changes the SIGN of the outcome on at least one instance. This run tests
whether that generalises.

VERDICT (2-of-3, read ACROSS the three heterogeneous families -- NEVER all(...)'d whole-run):
  SUPPORTED (PASS) -- the treatment-below-control inversion REVERSES OR FLATTENS with the
      informative baseline in >= 2 of the 3 families -> the destructiveness WAS the baseline's
      doing. label = baseline_informativeness_reverses_iatrogenesis; evidence_direction=supports.
  WEAKENED / WITHDRAWN (FAIL) -- treatments still land below their own controls with an
      informative baseline in a majority -> the destructiveness is NOT the baseline's doing and
      MECH-475 is WITHDRAWN (its whole content is the causal attribution; it does not degrade to
      a weaker version). label = iatrogenesis_persists_under_informative_baseline;
      evidence_direction=weakens.
  substrate_not_ready_requeue (FAIL, non_degenerate=false, evidence unknown) -- fewer than 2
      families are SCORABLE (their scalar CONTROL did not clear the readiness anchors, or the
      scalar inversion did not reproduce on this build). A family that cannot exhibit a
      treatment-below-control inversion CANNOT exhibit a reversal, so it is excluded, never
      scored as a verdict (the claim's NON-DEGENERACY GUARD).

DESIGN. Three treatment families, each scored INDEPENDENTLY as a precondition-gate ARM
(experiments/_lib/precondition_gate.py; aggregate_arm_gates -> non_degenerate = ANY family green;
a red family NEVER vacates a green one -- the V3-EXQ-785 lesson, which this run's whole point
depends on since the three families are heterogeneous). Each family runs its CONTROL and its
TREATMENT under BOTH value estimators (scalar = the original arm; distributional = the informative
baseline), on raw_view, on the NON-REGRESSED reference build (128-wide / 3x budget / z_world
detached / credit-replay 3 / topk 32), everything but the family's one manipulation held fixed.

  FAMILY capacity   (V3-EXQ-769): control = reference composed bootstrap (128-wide, 3x budget);
      treatment = the capacity-amended build (make_on_config: 256-wide, 5x budget, warm-start 0.2,
      credit-replay 6 / topk 64) -- the 769 ON arm, byte-identical. The +capacity/+budget IS the
      manipulation (a departure from the reference build); the reference build is the control.
  FAMILY approach   (V3-EXQ-781): control = reference composed bootstrap, NO approach drive
      (== the 781 ctrl, == the capacity control -- SHARED, run once); treatment = ctrl + the
      non-extinguishing appetitive-approach drive (use_approach_primitive=True, approach_coef=1.0,
      the 781 treat arm, byte-identical).
  FAMILY metabolic  (V3-EXQ-771): control = the 771 DECOUPLE arm (contamination-off env, survival
      trivial and uncoupled from foraging -- the arm the metabolic coupling is read against, 3.47);
      treatment = the METABOLIC env (MetabolicForageWrapper: energy->health starvation coupling, so
      survival REQUIRES foraging, the 771 treat arm, 1.12 / death 1.0). The env is the manipulation.

The capacity and approach families share ONE reference control (ref_ctrl_scalar / ref_ctrl_dist):
their control is the same reference composed bootstrap on the standard D3 env, byte-identical, so
it is run once and used as the control for both.

MANIPULATION IS THE VALUE ESTIMATOR + ONE FAMILY KNOB ONLY. The distributional critic is a clean
value-estimator swap applied at REP CONSTRUCTION (mech.make_rep(use_distributional_critic=True));
critic_loss() raises if used while disabled (fails loud). The update rule is otherwise untouched.
Confirmed the flag threads BootstrapExplorerConfig -> make_rep -> train_bootstrap_explorer for the
ACQUISITION path exactly as it does for the 788 retention path.

DV = the competence TRAJECTORY, NOT terminal competence (MANDATORY, MECH-475 what_would_answer).
retention_probe_every wires the substrate's non-perturbing mid-training competence probe
(train_a2c snapshots/restores the torch/numpy/random streams around every reading), 250-episode
cadence -> 12 readings over the 3000-ep reference budget (20 over the 5000-ep capacity treat).
Every reading is recorded per cell. The verdict routes on late_competence = the mean of the last
three trajectory readings (a stable, trajectory-DERIVED late-competence estimate -- not a single
terminal eval), the statistic the reversal/flatten test is defined on. Terminal-only measurement
is precisely what hid this signature across nineteen autopsy targets.

NON-DEGENERACY GUARD (the claim's, enforced per family). A family is SCORABLE only if, on this
build: (1) its env's LocalViewGreedy anchor clears the 1.0 foraging floor (env solvable from the
5x5 view); (2) its SCALAR control is competent (late_competence mean >= the 1.0 floor -- there is
a control LEVEL to be below); (3) the SCALAR inversion REPRODUCES (scalar control beats scalar
treatment by >= INVERSION_MARGIN -- the destructive inversion exists here to potentially reverse);
(4) the trajectory carries >= 2 readings (a shape). A family failing any of these is excluded from
scoring (never a reversal verdict). Fewer than 2 scorable families -> substrate_not_ready_requeue.

DV-SYMMETRY INVARIANCE (mandatory declaration; failure_autopsy_V3-EXQ-604c). The DV is
foraging_competence -- a resource-collection RATE (a magnitude), not an argmax/rank/set-aggregate.
Each manipulation is NOT invariant under any symmetry of this DV: +capacity/+budget, an intrinsic
approach STATE reward, an env starvation coupling, and the value-estimator swap each change the
LEARNED POLICY (or the env dynamics) and therefore the actual foraging rate. None is a broadcast
scalar, a monotone rescaling, or a permutation of interchangeable units. The measured
control->treatment delta is a genuine measurement, not an arithmetic identity fixed before the run.

MINT (mint-as-you-go). The scalar/distributional reference controls and the metabolic-decouple
controls are emitted reuse-ELIGIBLE (rng_fully_reset via arm_cell + config_slice_declared +
include_driver_script_in_hash=False); the mechanism logic lives in experiments/_lib/** (in the
substrate hash), so the fingerprint refuses on substrate drift. Treatment arms are leg-specific
and keep the driver in their hash. No separate baseline-only mint job (neither sanctioned
exception applies: all arms run cloud-class and no distinct consumer is planned ahead).

RE-DERIVE BRAKE: DOES NOT FIRE. MECH-475 is a NEWLY-registered claim (2026-07-22, split_from
MECH-457) testing a DIFFERENT mechanism (baseline informativeness) via a NEW manipulation (the
distributional critic on the acquisition path). This is the sanctioned "substrate now built ->
brake released" case: the distributional critic became a substrate feature and 788 is its positive
control. Not a lettered re-pose of an eliminated axis.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Shared machinery: experiments/_lib/mech457_bootstrap_explorer.py + mech457_explorer_classes.py +
mech457_fanout.py + mech457_probe_envs.py (metabolic env) + precondition_gate.py. ASCII-only in
all runtime strings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    evaluate_seed,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_bootstrap_explorer as boot  # noqa: E402
import experiments._lib.mech457_explorer_classes as mech  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments._lib.mech457_probe_envs as probe  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_837_mech475_distributional_critic_iatrogenic_falsifier"
QUEUE_ID = "V3-EXQ-837"
CLAIM_IDS: List[str] = ["MECH-475"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE
REPRESENTATION = "raw_view"   # raw_view ONLY: the reference-build regime where the install/bootstrap
                              # takes and where the claim's inversions were all measured (raw). z_world
                              # carries encoder/install confounds orthogonal to baseline informativeness.

# --- Budgets (reference vs capacity-amended) -----------------------------------------------
REF_BUDGET_MULTIPLIER = 3      # 3x the 1000-ep plateau budget == the non-regressed reference build
CAP_BUDGET_MULTIPLIER = boot.ON_BUDGET_MULTIPLIER   # 5x -- part of the capacity/budget manipulation

# --- Reference build (non-regressed) -- held FIXED across every arm but the family manipulation --
REF_ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN    # 128
REF_CREDIT_PASSES = mech.CREDIT_REPLAY_PASSES        # 3
REF_CREDIT_TOPK = mech.CREDIT_TOPK                   # 32
REF_COTRAIN_ENCODER = False                          # z_world detached (no-op for raw_view)
APPROACH_COEF = 1.0          # 781's exact non-extinguishing appetitive-approach coefficient

# --- Probe cadence (the trajectory the whole run exists to record) --------------------------
RETENTION_PROBE_EVERY = 250          # 3000/250 = 12 readings; 5000/250 = 20 for capacity treat
DRY_RETENTION_PROBE_EVERY = 2        # DRY_RL == 6 -> 3 readings under --dry-run
LATE_READINGS = 3                    # late_competence = mean of the last min(3, n) trajectory readings

# --- Pre-registered verdict thresholds (declared; never derived from the run) ---------------
COMPETENCE_FLOOR = float(COMPETENCE_RESOURCE_FLOOR)   # 1.0 -- foraging competence floor
INVERSION_MARGIN = 1.0        # scalar control must beat scalar treatment by >= this for the
                              # destructive inversion to count as REPRODUCED on this build
                              # (the three inversions are 6.36 / 2.78 / 2.35 -- safely above 1.0)
FLATTEN_MARGIN = 1.0          # distributional treatment within this of its control counts as FLAT
FLATTEN_FRACTION = 0.5        # OR: distributional inversion <= half the scalar inversion == FLAT
MIN_TRAJECTORY_READINGS = 1.5  # need >= 2 readings for a trajectory SHAPE (floor, strict >)

FAMILIES: Tuple[str, ...] = ("capacity", "approach", "metabolic")
CRITICS: Tuple[str, ...] = ("scalar", "dist")


# ---------------------------------------------------------------------------------------
# Arm specs. Each boot arm is one (role x critic) cell family; env/cfg/budget declared here.
# ---------------------------------------------------------------------------------------
# cfg builders: "ref" = reference composed bootstrap; "capacity" = make_on_config (256/5x);
#               "approach" = reference bootstrap + non-extinguishing approach drive.
# env builders: "standard" = D3 (contamination ON); "decouple" = contamination-OFF (bare);
#               "metabolic" = contamination-OFF + MetabolicForageWrapper (starvation coupling).
BOOT_ARM_SPECS: Tuple[Dict[str, Any], ...] = (
    {"arm_id": "ref_ctrl_scalar",       "critic": "scalar", "cfg": "ref",      "env": "standard",  "budget": "ref", "role": "control",   "mint": True,  "families": ["capacity", "approach"]},
    {"arm_id": "ref_ctrl_dist",         "critic": "dist",   "cfg": "ref",      "env": "standard",  "budget": "ref", "role": "control",   "mint": True,  "families": ["capacity", "approach"]},
    {"arm_id": "cap_treat_scalar",      "critic": "scalar", "cfg": "capacity", "env": "standard",  "budget": "cap", "role": "treatment", "mint": False, "families": ["capacity"]},
    {"arm_id": "cap_treat_dist",        "critic": "dist",   "cfg": "capacity", "env": "standard",  "budget": "cap", "role": "treatment", "mint": False, "families": ["capacity"]},
    {"arm_id": "appr_treat_scalar",     "critic": "scalar", "cfg": "approach", "env": "standard",  "budget": "ref", "role": "treatment", "mint": False, "families": ["approach"]},
    {"arm_id": "appr_treat_dist",       "critic": "dist",   "cfg": "approach", "env": "standard",  "budget": "ref", "role": "treatment", "mint": False, "families": ["approach"]},
    {"arm_id": "metab_decouple_scalar", "critic": "scalar", "cfg": "ref",      "env": "decouple",  "budget": "ref", "role": "control",   "mint": True,  "families": ["metabolic"]},
    {"arm_id": "metab_decouple_dist",   "critic": "dist",   "cfg": "ref",      "env": "decouple",  "budget": "ref", "role": "control",   "mint": True,  "families": ["metabolic"]},
    {"arm_id": "metab_treat_scalar",    "critic": "scalar", "cfg": "ref",      "env": "metabolic", "budget": "ref", "role": "treatment", "mint": False, "families": ["metabolic"]},
    {"arm_id": "metab_treat_dist",      "critic": "dist",   "cfg": "ref",      "env": "metabolic", "budget": "ref", "role": "treatment", "mint": False, "families": ["metabolic"]},
)
BOOT_ARMS: Tuple[str, ...] = tuple(s["arm_id"] for s in BOOT_ARM_SPECS)
SPEC_BY_ARM: Dict[str, Dict[str, Any]] = {s["arm_id"]: s for s in BOOT_ARM_SPECS}

# Anchors: standard-env (capacity + approach readiness) and metabolic-env (metabolic readiness).
STD_ANCHORS: Tuple[str, ...] = fan.ANCHOR_ARMS                              # solvability on the standard env
METAB_ANCHORS: Tuple[str, ...] = tuple(a + "_metab" for a in fan.ANCHOR_ARMS)  # ... on the metabolic env
_METAB_ANCHOR_BASE: Dict[str, str] = {a + "_metab": a for a in fan.ANCHOR_ARMS}
ANCHOR_ARMS: Tuple[str, ...] = STD_ANCHORS + METAB_ANCHORS
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + ANCHOR_ARMS

# Family -> (control arm per critic, treatment arm per critic, readiness env).
FAMILY_MAP: Dict[str, Dict[str, Any]] = {
    "capacity": {
        "control": {"scalar": "ref_ctrl_scalar", "dist": "ref_ctrl_dist"},
        "treat": {"scalar": "cap_treat_scalar", "dist": "cap_treat_dist"},
        "env_anchor": "standard",
        "manipulation": "actor_critic_hidden 128->256 + budget 3x->5x (make_on_config; 769 ON arm)",
    },
    "approach": {
        "control": {"scalar": "ref_ctrl_scalar", "dist": "ref_ctrl_dist"},   # SHARED with capacity
        "treat": {"scalar": "appr_treat_scalar", "dist": "appr_treat_dist"},
        "env_anchor": "standard",
        "manipulation": "non-extinguishing appetitive-approach drive (approach_coef=1.0; 781 treat)",
    },
    "metabolic": {
        "control": {"scalar": "metab_decouple_scalar", "dist": "metab_decouple_dist"},
        "treat": {"scalar": "metab_treat_scalar", "dist": "metab_treat_dist"},
        "env_anchor": "metabolic",
        "manipulation": "MetabolicForageWrapper energy->health starvation coupling (771 treat)",
    },
}


# ---------------------------------------------------------------------------------------
# Per-family precondition specs (each FAMILY is one gate arm). All four apply to every
# family, so no applies_to scoping is needed -- but they are still evaluated PER FAMILY and
# aggregated so a red family never vacates a green one (the 785 lesson).
# ---------------------------------------------------------------------------------------
FAMILY_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="env_local_view_clears_floor",
        description=(
            "LocalViewGreedyPolicy reading the same 5x5 resource_field_view forages above the 1.0 "
            "competence floor on THIS family's env -- the positive control that the env is solvable "
            "from the local view (same statistic the verdict routes on). Below-floor means the "
            "substrate/env is not ready, NOT that the baseline mechanism failed."
        ),
        control="local_view_greedy foraging_competence on the family's control env vs the 1.0 floor",
        threshold=COMPETENCE_FLOOR,
        kind="readiness",
    ),
    PreconditionSpec(
        name="scalar_control_competent",
        description=(
            "THE LOAD-BEARING NON-DEGENERACY GUARD. The family's SCALAR control must be competent -- "
            "mean late_competence across seeds clears the 1.0 floor -- so there is a control LEVEL "
            "for the treatment to be below. A control at floor cannot exhibit a treatment-below-"
            "control inversion, so the family self-routes substrate_not_ready_requeue, never a "
            "reversal verdict."
        ),
        control="scalar control arm mean late_competence vs the 1.0 floor",
        threshold=COMPETENCE_FLOOR,
        kind="readiness",
    ),
    PreconditionSpec(
        name="scalar_inversion_reproduces",
        description=(
            "The destructive treatment-below-control inversion must REPRODUCE on this build under "
            "the scalar critic: scalar control mean late_competence beats scalar treatment by >= "
            "INVERSION_MARGIN. Without a reproduced inversion there is nothing for the informative "
            "baseline to reverse, so the family is excluded (guards against substrate drift having "
            "already erased the phenomenon)."
        ),
        control="scalar (control - treatment) mean late_competence vs INVERSION_MARGIN",
        threshold=INVERSION_MARGIN,
        kind="readiness",
    ),
    PreconditionSpec(
        name="trajectory_readings",
        description=(
            "The DV is a TRAJECTORY, so the family's worst cell must carry at least two probe "
            "readings -- one reading has no shape and cannot support a late-competence mean."
        ),
        control="worst-cell number of mid-training competence probe readings in the family",
        threshold=MIN_TRAJECTORY_READINGS,
        kind="measurability",
        # Design-time proof: cadence and budget are both pre-registered, so an unsatisfiable probe
        # schedule is caught before compute is spent.
        structural_max=lambda ctx: float(
            int(ctx["n_episodes"]) // max(1, int(ctx["probe_every"]))
        ),
    ),
)


# ---------------------------------------------------------------------------------------
# Config + env construction
# ---------------------------------------------------------------------------------------
def _ref_cfg(budget: int, use_dist: bool, use_approach: bool, probe_every: int
             ) -> boot.BootstrapExplorerConfig:
    """The reference composed bootstrap (annealed drive 1.0->0.05, warm-start 0, credit-replay
    3/topk 32, 128-wide, z_world detached) -- identical to the 770/781/771 ctrl. use_approach adds
    the non-extinguishing appetitive-approach drive; use_dist swaps in the distributional critic."""
    return boot.BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=1.0, intrinsic_coef_end=boot.ON_INTRINSIC_COEF_END,
        anneal_fraction=boot.ON_ANNEAL_FRACTION,
        warm_start_fraction=0.0,
        entropy_beta_start=boot.ON_ENTROPY_BETA_START, entropy_beta_end=boot.ON_ENTROPY_BETA_END,
        credit_replay=True, credit_replay_passes=REF_CREDIT_PASSES, credit_topk=REF_CREDIT_TOPK,
        n_episodes=int(budget),
        actor_critic_hidden=REF_ACTOR_CRITIC_HIDDEN, cotrain_encoder=REF_COTRAIN_ENCODER,
        use_approach_primitive=bool(use_approach),
        approach_coef=(APPROACH_COEF if use_approach else 0.0),
        use_distributional_critic=bool(use_dist),
        retention_probe_every=int(probe_every),
    )


def _capacity_cfg(budget: int, use_dist: bool, probe_every: int) -> boot.BootstrapExplorerConfig:
    """The 769 capacity-amended build (make_on_config: 256-wide, 5x budget, warm-start 0.2,
    credit-replay 6 / topk 64, z_world detached), byte-identical to the 769 ON arm, plus the
    critic swap and the trajectory probe."""
    cfg = boot.make_on_config()
    cfg.n_episodes = int(budget)
    cfg.use_distributional_critic = bool(use_dist)
    cfg.retention_probe_every = int(probe_every)
    return cfg


def _make_cfg(spec: Dict[str, Any], budget: int, probe_every: int) -> boot.BootstrapExplorerConfig:
    use_dist = (spec["critic"] == "dist")
    if spec["cfg"] == "ref":
        return _ref_cfg(budget, use_dist=use_dist, use_approach=False, probe_every=probe_every)
    if spec["cfg"] == "approach":
        return _ref_cfg(budget, use_dist=use_dist, use_approach=True, probe_every=probe_every)
    if spec["cfg"] == "capacity":
        return _capacity_cfg(budget, use_dist=use_dist, probe_every=probe_every)
    raise ValueError(f"unknown cfg kind {spec['cfg']!r}")


def _env_kwargs(env_kind: str, std_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The env_kwargs for one env kind (contamination knob differs; the wrapper is applied
    separately in _make_env so a bare kwargs slice is fingerprint-declarable)."""
    if env_kind == "standard":
        return dict(std_kwargs)
    return probe.metabolic_env_kwargs(std_kwargs)   # contaminated_harm = 0.0 (decouple + metabolic)


def _make_env(env_kind: str, seed: int, std_kwargs: Dict[str, Any]) -> Any:
    env = x734._make_env(seed, _env_kwargs(env_kind, std_kwargs))
    if env_kind == "metabolic":
        return probe.MetabolicForageWrapper(env)   # energy->health starvation coupling
    return env


def _config_slice(spec: Dict[str, Any], cfg: boot.BootstrapExplorerConfig,
                  std_kwargs: Dict[str, Any], eval_eps: int, steps: int) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "arm_id": spec["arm_id"], "rung_id": fan.RUNG_ID,
        "env_kwargs": _env_kwargs(spec["env"], std_kwargs),
        "env_kind": spec["env"],
        "metabolic_wrapper": bool(spec["env"] == "metabolic"),
        "representation": REPRESENTATION,
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
        "kind": "mech475_baseline_informativeness_falsifier",
        "p0_warmup_episodes": 0,   # raw_view needs no encoder warmup
    }
    base.update(cfg.as_slice())
    if spec["env"] == "metabolic":
        base.update(probe.config_slice_extra())
    return base


def _make_probe_fn(rep_agent: Any, env_kind: str, seed: int, std_kwargs: Dict[str, Any],
                   eval_eps: int, steps: int, arm_label: str) -> Callable[[int], Dict[str, Any]]:
    """Non-perturbing mid-training competence probe. A FRESH env per reading (never the training
    env); train_a2c snapshots/restores the RNG streams around every call, so neutrality is a
    substrate guarantee, not this closure's good behaviour."""
    def _probe(ep: int) -> Dict[str, Any]:
        probe_env = _make_env(env_kind, seed, std_kwargs)
        row = evaluate_seed(
            rep_agent.eval_policy(f"{arm_label}_probe_ep{int(ep)}"), probe_env, int(eval_eps), int(steps)
        )
        return {"foraging_competence": round(float(row["foraging_competence"]), 6)}
    return _probe


def _late_competence(traj_vals: List[float], terminal: float) -> float:
    """The verdict DV: mean of the last min(LATE_READINGS, n) trajectory readings -- a stable,
    trajectory-DERIVED late-competence estimate. Falls back to the terminal eval only when the
    trajectory is empty (should not happen with the probe wired)."""
    if traj_vals:
        tail = traj_vals[-min(LATE_READINGS, len(traj_vals)):]
        return round(float(sum(tail) / len(tail)), 6)
    return round(float(terminal), 6)


# ---------------------------------------------------------------------------------------
def _run_boot_cell(spec: Dict[str, Any], seed: int, std_kwargs: Dict[str, Any],
                   budget: int, eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    arm_id = spec["arm_id"]
    cfg = _make_cfg(spec, budget, probe_every)

    warm_env = _make_env(spec["env"], seed, std_kwargs)
    rep_agent = mech.make_rep(
        REPRESENTATION, warm_env, seed=seed, p0=0, steps=int(steps),
        actor_critic_hidden=int(cfg.actor_critic_hidden),
        cotrain_encoder=bool(cfg.cotrain_encoder),
        use_distributional_critic=bool(cfg.use_distributional_critic),
    )

    probe_fn = _make_probe_fn(rep_agent, spec["env"], seed, std_kwargs, eval_eps, steps, arm_id)
    train_env = _make_env(spec["env"], seed, std_kwargs)
    guard = boot.train_bootstrap_explorer(
        rep_agent, train_env, seed=seed, steps=int(steps), arm_label=arm_id, cfg=cfg,
        denom=int(cfg.n_episodes), probe_fn=probe_fn,
    )
    trajectory: List[Dict[str, Any]] = list(guard.get("competence_trajectory", []))

    eval_env = _make_env(spec["env"], seed, std_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, int(eval_eps), int(steps))

    traj_vals = [float(r.get("foraging_competence", 0.0)) for r in trajectory]
    terminal_eval = float(row["foraging_competence"])
    late = _late_competence(traj_vals, terminal_eval)

    row["families"] = list(spec.get("families", []))
    row["role"] = spec["role"]
    row["critic"] = spec["critic"]
    row["use_distributional_critic"] = bool(cfg.use_distributional_critic)
    row["n_episodes"] = int(cfg.n_episodes)
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["late_competence"] = late
    row["trajectory_peak_competence"] = round(max(traj_vals), 6) if traj_vals else terminal_eval
    row["trajectory_terminal_competence"] = round(traj_vals[-1], 6) if traj_vals else terminal_eval
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["mean_approach_reward_recent"] = float(guard.get("mean_approach_reward_recent", 0.0))
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    return row


def _family_arm_contexts(budget_ref: int, probe_every: int) -> List[Dict[str, Any]]:
    # Worst-case (smallest) budget cell drives the trajectory structural_max: the reference-budget
    # control gives the fewest readings even in the capacity family (whose treat runs 5x).
    return [
        {"id": fam, "n_episodes": int(budget_ref), "probe_every": int(probe_every)}
        for fam in FAMILIES
    ]


def run_experiment(seeds: List[int], budget_ref: int, budget_cap: int, eval_eps: int,
                   steps: int, probe_every: int) -> Dict[str, Any]:
    print(
        f"MECH-475 baseline-informativeness DECISIVE FALSIFIER "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; rep={REPRESENTATION}, "
        f"budget_ref={budget_ref}, budget_cap={budget_cap}, eval={eval_eps}, steps={steps}, "
        f"probe_every={probe_every}; manipulation = distributional critic + one family knob; "
        f"verdict = 2-of-3 reversal/flatten)",
        flush=True,
    )
    fam_ctxs = _family_arm_contexts(budget_ref, probe_every)
    # Design-audit BEFORE any compute: refuse a run carrying a structurally unsatisfiable gate.
    assert_no_structurally_unsatisfiable_gate(FAMILY_SPECS, fam_ctxs)

    std_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []
    cell_by_arm_seed: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _budget_for(spec: Dict[str, Any]) -> int:
        return int(budget_cap) if spec["budget"] == "cap" else int(budget_ref)

    def _run_boot(spec: Dict[str, Any], seed: int) -> Dict[str, Any]:
        arm_id = spec["arm_id"]
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _config_slice(spec, _make_cfg(spec, _budget_for(spec), probe_every),
                                  std_kwargs, eval_eps, steps)
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=not spec["mint"]) as cell:
            row = _run_boot_cell(spec, seed, std_kwargs, _budget_for(spec), eval_eps, steps, probe_every)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        per_arm_forage[arm_id].append(float(row["foraging_competence"]))
        all_cells.append(row)
        cell_by_arm_seed[(arm_id, int(seed))] = row
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={row['foraging_competence']} "
            f"late={row['late_competence']} readings={row['n_trajectory_readings']})", flush=True,
        )
        return row

    def _run_anchor(anchor_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{anchor_id}", flush=True)
        is_metab = anchor_id in METAB_ANCHORS
        base_name = _METAB_ANCHOR_BASE.get(anchor_id, anchor_id)
        env_kind = "metabolic" if is_metab else "standard"
        slice_cfg = {"arm_id": anchor_id, "rung_id": fan.RUNG_ID,
                     "env_kwargs": _env_kwargs(env_kind, std_kwargs),
                     "env_kind": env_kind, "metabolic_wrapper": bool(is_metab),
                     "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                     "kind": "anchor"}
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            anchor_env = _make_env(env_kind, seed, std_kwargs)
            row = fan.run_anchor_cell(base_name, anchor_env, seed, int(eval_eps), int(steps))
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = anchor_id
            row["seed"] = int(seed)
            cell.stamp(row)
        per_arm_forage[anchor_id].append(float(row["foraging_competence"]))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={anchor_id} seed={seed} forage/ep={row['foraging_competence']})", flush=True,
        )
        return row

    # Anchors first (readiness gate + denominators on both envs), then the boot arms.
    for anchor_id in ANCHOR_ARMS:
        for seed in seeds:
            _run_anchor(anchor_id, seed)
    for spec in BOOT_ARM_SPECS:
        for seed in seeds:
            _run_boot(spec, seed)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _late_of(arm_id: str, seed: int) -> Optional[float]:
        cell = cell_by_arm_seed.get((arm_id, int(seed)))
        return None if cell is None else float(cell.get("late_competence", 0.0))

    std_local_view = _mean("local_view_greedy")
    std_oracle = _mean("greedy_oracle")
    metab_local_view = _mean("local_view_greedy_metab")
    metab_oracle = _mean("greedy_oracle_metab")
    env_anchor_local_view = {"standard": std_local_view, "metabolic": metab_local_view}

    # ---- per-family reversal analysis (paired within seed) --------------------------------
    per_family: Dict[str, Any] = {}
    for fam in FAMILIES:
        m = FAMILY_MAP[fam]
        ctrl_arm, treat_arm = m["control"], m["treat"]
        env_kind = m["env_anchor"]

        # scalar control competence (late_competence per seed)
        ctrl_scalar_late = [_late_of(ctrl_arm["scalar"], s) for s in seeds]
        ctrl_scalar_late = [v for v in ctrl_scalar_late if v is not None]
        ctrl_scalar_mean = round(statistics.fmean(ctrl_scalar_late), 6) if ctrl_scalar_late else 0.0

        # per-seed paired inversions
        scalar_inv, dist_inv, per_seed = [], [], []
        reversed_flags: List[bool] = []
        for s in seeds:
            cs = _late_of(ctrl_arm["scalar"], s)
            ts = _late_of(treat_arm["scalar"], s)
            cd = _late_of(ctrl_arm["dist"], s)
            td = _late_of(treat_arm["dist"], s)
            if None in (cs, ts, cd, td):
                continue
            inv_s = round(cs - ts, 6)      # scalar inversion (expected > 0 = destructive)
            inv_d = round(cd - td, 6)      # distributional inversion
            flat = bool(inv_d <= FLATTEN_MARGIN or (inv_s > 0.0 and inv_d <= FLATTEN_FRACTION * inv_s))
            scalar_inv.append(inv_s)
            dist_inv.append(inv_d)
            reversed_flags.append(flat)
            per_seed.append({
                "seed": int(s),
                "control_scalar_late": round(cs, 6), "treat_scalar_late": round(ts, 6),
                "control_dist_late": round(cd, 6), "treat_dist_late": round(td, 6),
                "scalar_inversion": inv_s, "dist_inversion": inv_d,
                "reverses_or_flattens": flat,
            })
        n_seed = len(per_seed)
        n_reversed_seeds = int(sum(1 for f in reversed_flags if f))
        scalar_inv_mean = round(statistics.fmean(scalar_inv), 6) if scalar_inv else 0.0
        dist_inv_mean = round(statistics.fmean(dist_inv), 6) if dist_inv else 0.0
        n_ctrl_competent = int(sum(1 for v in ctrl_scalar_late if v >= COMPETENCE_FLOOR))

        # worst-cell trajectory readings across the family's four arms
        fam_arms = [ctrl_arm["scalar"], ctrl_arm["dist"], treat_arm["scalar"], treat_arm["dist"]]
        readings = [
            int(cell_by_arm_seed[(a, int(s))]["n_trajectory_readings"])
            for a in fam_arms for s in seeds if (a, int(s)) in cell_by_arm_seed
        ]
        worst_readings = float(min(readings)) if readings else 0.0

        family_reverses = bool(n_seed and n_reversed_seeds > (n_seed / 2.0))
        per_family[fam] = {
            "family": fam,
            "manipulation": m["manipulation"],
            "env_kind": env_kind,
            "control_arms": ctrl_arm, "treat_arms": treat_arm,
            "n_seeds_paired": n_seed,
            "scalar_control_late_per_seed": [round(v, 6) for v in ctrl_scalar_late],
            "scalar_control_late_mean": ctrl_scalar_mean,
            "n_seeds_scalar_control_competent": n_ctrl_competent,
            "scalar_inversion_mean": scalar_inv_mean,
            "dist_inversion_mean": dist_inv_mean,
            "n_seeds_reverses_or_flattens": n_reversed_seeds,
            "reverses_or_flattens_strict_majority": family_reverses,
            "worst_trajectory_readings": worst_readings,
            "env_local_view_greedy": round(env_anchor_local_view[env_kind], 6),
            "per_seed": per_seed,
        }

    # ---- per-family gate (a red family NEVER vacates a green one; 785 lesson) --------------
    arm_gates = []
    for ctx in fam_ctxs:
        fam = ctx["id"]
        r = per_family[fam]
        measured = {
            "env_local_view_clears_floor": float(r["env_local_view_greedy"]),
            "scalar_control_competent": float(r["scalar_control_late_mean"]),
            "scalar_inversion_reproduces": float(r["scalar_inversion_mean"]),
            "trajectory_readings": float(r["worst_trajectory_readings"]),
        }
        arm_gates.append(evaluate_arm_gate(fam, ctx, FAMILY_SPECS, measured=measured))
    gate = aggregate_arm_gates(arm_gates)
    green_families = list(gate["green_arms"])
    n_green = len(green_families)
    reversed_green = [f for f in green_families if per_family[f]["reverses_or_flattens_strict_majority"]]
    n_reversed = len(reversed_green)

    # ---- run-level verdict -----------------------------------------------------------------
    if n_green < 2:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        evidence_direction = "unknown"
        non_degenerate = False
    elif n_reversed >= 2:
        outcome, label = "PASS", "baseline_informativeness_reverses_iatrogenesis"
        evidence_direction = "supports"
        non_degenerate = True
    else:
        outcome, label = "FAIL", "iatrogenesis_persists_under_informative_baseline"
        evidence_direction = "weakens"
        non_degenerate = True

    # Substrate-degeneracy net: does the substrate produce ANY variation across arms + anchors?
    # (A fully-pinned substrate -- every arm and anchor identical -- has no signal to read.)
    # Deliberately NOT the cross-family reversal spread: a genuine SUPPORTED result (all families
    # flatten to ~0 inversion) has LOW cross-family spread, so keying degeneracy on it would flag
    # exactly the outcome the claim predicts.
    degeneracy = check_degeneracy({
        "boot_arm_and_anchor_foraging_spread": {
            "values": [_mean(a) for a in BOOT_ARMS]
                      + [std_local_view, metab_local_view, _mean("random_walk")]
        },
    })
    if not non_degenerate:
        degeneracy_reason = (
            f"substrate_not_ready_requeue: only {n_green}/3 families are scorable "
            f"(green={green_families}); a 2-of-3 reversal verdict is unreachable. "
            + (gate["degeneracy_reason"] or "")
        )
    else:
        degeneracy_reason = gate["degeneracy_reason"] or (
            "" if degeneracy["non_degenerate"] else degeneracy["degeneracy_reason"]
        )

    criteria_by_fam = {f: [f"C_{f}_reverses_or_flattens"] for f in FAMILIES}
    criteria_nd = arm_criteria_non_degenerate(criteria_by_fam, gate)
    criteria_nd["two_of_three_families_reverse_or_flatten"] = bool(n_green >= 2 and degeneracy["non_degenerate"])

    interpretation = {
        "label": label,
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "criteria": [
            {"name": "C_two_of_three_families_reverse_or_flatten",
             "load_bearing": True,
             "description": (
                 "In >= 2 of the 3 SCORABLE families, the treatment-below-control inversion "
                 "REVERSES OR FLATTENS under the distributional (informative) critic on a strict "
                 "majority of seeds -- late_competence, paired within seed."
             ),
             "passed": bool(n_reversed >= 2)},
        ] + [
            {"name": f"C_{f}_reverses_or_flattens", "load_bearing": False,
             "description": f"family '{f}' ({FAMILY_MAP[f]['manipulation']}) reverses/flattens on a strict majority of seeds",
             "passed": bool(per_family[f]["reverses_or_flattens_strict_majority"])}
            for f in FAMILIES
        ],
        "criteria_non_degenerate": criteria_nd,
        "interpretation_grid": [
            {"label": "baseline_informativeness_reverses_iatrogenesis", "outcome": "PASS",
             "condition": ">= 2 of 3 scorable families reverse or flatten under the distributional critic",
             "reading": (
                 "The uninformative value baseline WAS the source of the destructiveness: making "
                 "the baseline informative changes the sign/flattens the treatment-below-control "
                 "inversion across a majority of independent interventions. Supports MECH-475."
             )},
            {"label": "iatrogenesis_persists_under_informative_baseline", "outcome": "FAIL",
             "condition": ">= 2 families scorable but < 2 reverse/flatten",
             "reading": (
                 "Treatments still land below their own controls with an informative baseline. The "
                 "destructiveness is NOT the baseline's doing -> MECH-475 is WITHDRAWN (the whole "
                 "content is the causal attribution; it does not degrade to a weaker version). "
                 "Weakens MECH-475."
             )},
            {"label": "substrate_not_ready_requeue", "outcome": "FAIL",
             "condition": (
                 "fewer than 2 families are scorable: an env anchor is sub-floor, the scalar "
                 "control is not competent, the scalar inversion did not reproduce, or the "
                 "trajectory has no shape"
             ),
             "reading": (
                 "A family whose control did not clear the readiness anchors (or whose destructive "
                 "inversion did not reproduce on this build) CANNOT exhibit a reversal. "
                 "UNINFORMATIVE about the claim; requeue. NEVER a reversal or a weakens verdict."
             )},
        ],
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": evidence_direction,
        "per_arm_gate": gate["per_arm_gate"],
        "readiness": {
            "n_scorable_families": n_green,
            "scorable_families": green_families,
            "n_families_reverse_or_flatten": n_reversed,
            "families_reverse_or_flatten": reversed_green,
            "std_local_view_greedy": round(std_local_view, 6),
            "std_greedy_oracle": round(std_oracle, 6),
            "metab_local_view_greedy": round(metab_local_view, 6),
            "metab_greedy_oracle": round(metab_oracle, 6),
            # P0 readiness-assert, restated flat + numeric: worst family env anchor vs the 1.0 floor.
            "readiness_assert": {
                "name": "env_local_view_clears_floor_worst_family",
                "kind": "readiness",
                "description": (
                    "Every scored family's env must be solvable from the 5x5 local view: "
                    "LocalViewGreedy clears the 1.0 foraging floor. Below it the family self-routes "
                    "substrate_not_ready_requeue rather than any reversal verdict."
                ),
                "control": "min over families of local_view_greedy foraging_competence on the family env",
                "direction": "lower",
                "measured": round(min(std_local_view, metab_local_view), 6),
                "threshold": COMPETENCE_FLOOR,
                "met": bool(std_local_view >= COMPETENCE_FLOOR and metab_local_view >= COMPETENCE_FLOOR),
            },
        },
        "headline": {
            "verdict": label,
            "n_scorable_families": n_green,
            "n_families_reverse_or_flatten": n_reversed,
            "per_family_reverses_or_flattens": {
                f: per_family[f]["reverses_or_flattens_strict_majority"] for f in FAMILIES
            },
            "per_family_scalar_inversion_mean": {
                f: per_family[f]["scalar_inversion_mean"] for f in FAMILIES
            },
            "per_family_dist_inversion_mean": {
                f: per_family[f]["dist_inversion_mean"] for f in FAMILIES
            },
            "flatten_margin": FLATTEN_MARGIN,
            "flatten_fraction": FLATTEN_FRACTION,
            "inversion_margin": INVERSION_MARGIN,
            "competence_floor": COMPETENCE_FLOOR,
        },
        "per_family": per_family,
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": COMPETENCE_FLOOR,
            "std_local_view_greedy_d3_live": round(std_local_view, 6),
            "metab_local_view_greedy_live": round(metab_local_view, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
        },
        "arm_results": all_cells,
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "discrimination_verdict": result["discrimination_verdict"],
        "per_arm_gate": result["per_arm_gate"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_family": result["per_family"],
        "reference_band": result["reference_band"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "reuse_mint": {
            "reusable_arms": [s["arm_id"] for s in BOOT_ARM_SPECS if s["mint"]],
            "reuse_eligible": True,
            "note": (
                "The scalar/distributional reference controls and the metabolic-decouple controls "
                "are emitted reuse-ELIGIBLE (rng_fully_reset via arm_cell + config_slice_declared + "
                "include_driver_script_in_hash=False); the mechanism logic lives in "
                "experiments/_lib/** (in the substrate hash). Treatment arms keep the driver in "
                "their hash (leg-specific). No separate baseline-only mint job."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The competence TRAJECTORY (late_competence = mean of the last three probe readings; "
            "NOT terminal competence) of the composed bootstrap explorer across three destructive "
            "ACQUISITION-side treatment families (capacity/budget, earned approach drive, metabolic "
            "coupling), each run under a SCALAR and a DISTRIBUTIONAL critic on the non-regressed "
            "reference build, raw_view. Statistic: the treatment-below-control inversion "
            "(control_late - treat_late), paired within seed, scalar vs distributional. PASS: in "
            ">= 2 of the 3 scorable families the distributional inversion REVERSES OR FLATTENS "
            f"(dist_inv <= {FLATTEN_MARGIN} OR <= {FLATTEN_FRACTION} x scalar_inv) on a strict "
            "majority of seeds. NON-DEGENERACY: a family whose scalar control is not competent, or "
            "whose scalar inversion did not reproduce, is excluded and self-routes "
            "substrate_not_ready_requeue."
        ),
        "notes": (
            "MECH-475 DECISIVE FALSIFIER (what_would_answer (a)). EVIDENCE experiment tagging "
            "MECH-475 only. Tests whether the destructiveness of added optimisation pressure "
            "(769 capacity/budget, 781 approach drive, 771 metabolic coupling) is the UNINFORMATIVE "
            "value baseline's doing, by re-running each treatment's control and treatment under a "
            "SCALAR and a DISTRIBUTIONAL (informative) critic and asking whether the "
            "treatment-below-control inversion reverses or flattens (2-of-3). SUPPORTED -> the "
            "baseline was the cause; WEAKENED -> MECH-475 WITHDRAWN (whole content is the causal "
            "attribution). RE-DERIVE BRAKE DOES NOT FIRE: MECH-475 is newly registered (2026-07-22, "
            "split_from MECH-457) testing a DIFFERENT mechanism (baseline informativeness) via a NEW "
            "manipulation (the distributional critic, now a built substrate feature with 788 as its "
            "positive control) -- the sanctioned substrate-now-built brake-release case. Three "
            "heterogeneous families are scored INDEPENDENTLY via precondition_gate.aggregate_arm_gates "
            "(a red family never vacates a green one -- the V3-EXQ-785 lesson). DV-SYMMETRY: "
            "foraging_competence is a magnitude, and none of the manipulations is invariant under a "
            "symmetry of it (each changes the learned policy or env dynamics). Reference build "
            "128-wide / 3x budget / z_world detached / credit-replay 3 / topk 32; capacity treat is "
            "the 769 256/5x arm. raw_view ONLY (the regime where the bootstrap takes and where the "
            "inversions were measured). MANDATORY trajectory recorded per cell (250-ep cadence)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-837 MECH-475 distributional-critic iatrogenic-optimisation falsifier"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--budget-ref", type=int, default=None,
                        help="reference-build RL budget (episodes) per control/approach/metabolic arm")
    parser.add_argument("--budget-cap", type=int, default=None,
                        help="capacity-amended RL budget (episodes) for the 769 treat arm")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        budget_ref = fan.DRY_RL
        budget_cap = fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
        probe_every = DRY_RETENTION_PROBE_EVERY
    else:
        seeds = list(fan.SEEDS)
        budget_ref = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)   # 3000
        budget_cap = int(fan.RL_EPISODES * CAP_BUDGET_MULTIPLIER)   # 5000
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE
        probe_every = RETENTION_PROBE_EVERY

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    if args.budget_ref is not None:
        budget_ref = int(args.budget_ref)
    if args.budget_cap is not None:
        budget_cap = int(args.budget_cap)
    if args.steps is not None:
        steps = int(args.steps)
    if args.eval_episodes is not None:
        eval_eps = int(args.eval_episodes)
    if args.probe_every is not None:
        probe_every = int(args.probe_every)

    result = run_experiment(seeds=seeds, budget_ref=budget_ref, budget_cap=budget_cap,
                            eval_eps=eval_eps, steps=steps, probe_every=probe_every)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representation": REPRESENTATION,
        "budget_ref_episodes": budget_ref,
        "budget_cap_episodes": budget_cap,
        "ref_budget_multiplier": REF_BUDGET_MULTIPLIER,
        "cap_budget_multiplier": CAP_BUDGET_MULTIPLIER,
        "retention_probe_every": probe_every,
        "late_readings": LATE_READINGS,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "cap_actor_critic_hidden": boot.ON_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": REF_CREDIT_PASSES, "ref_credit_topk": REF_CREDIT_TOPK,
        "ref_cotrain_encoder": REF_COTRAIN_ENCODER,
        "approach_coef": APPROACH_COEF,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "dist_critic_n_bins": fan.DIST_CRITIC_N_BINS,
        "dist_critic_limit": fan.DIST_CRITIC_LIMIT,
        "dist_critic_sigma": fan.DIST_CRITIC_SIGMA,
        "competence_floor": COMPETENCE_FLOOR,
        "inversion_margin": INVERSION_MARGIN,
        "flatten_margin": FLATTEN_MARGIN, "flatten_fraction": FLATTEN_FRACTION,
        "min_trajectory_readings": MIN_TRAJECTORY_READINGS,
        "families": list(FAMILIES),
        "family_map": {f: {"manipulation": FAMILY_MAP[f]["manipulation"],
                           "env_anchor": FAMILY_MAP[f]["env_anchor"],
                           "control": FAMILY_MAP[f]["control"], "treat": FAMILY_MAP[f]["treat"]}
                       for f in FAMILIES},
        "on_config": boot.make_on_config().as_slice(),
        "metabolic_env_extra": probe.config_slice_extra(),
        "supersedes": None,
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
    # AFTER arm_results is assembled, so substrate_hash hoists from the per-cell fingerprints.
    stamp_recording_core(manifest, config=cfg, seeds=seeds, script_path=Path(__file__),
                         started_at=t0)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"evidence={result['evidence_direction']} "
        f"scorable={result['readiness']['n_scorable_families']} "
        f"reversed={result['readiness']['n_families_reverse_or_flatten']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    for fam in FAMILIES:
        r = result["per_family"][fam]
        print(
            f"  {fam}: scalar_inv={r['scalar_inversion_mean']} dist_inv={r['dist_inversion_mean']} "
            f"ctrl_competent={r['n_seeds_scalar_control_competent']}/{r['n_seeds_paired']} "
            f"reverses={r['reverses_or_flattens_strict_majority']} "
            f"(seeds {r['n_seeds_reverses_or_flattens']}/{r['n_seeds_paired']}) "
            f"env_lvg={r['env_local_view_greedy']}", flush=True,
        )
    print(
        f"  green_families={result['per_arm_gate']['green_arms']} "
        f"red_families={result['per_arm_gate']['red_arms']}", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
