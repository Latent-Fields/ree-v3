"""
infant_substrate:GAP-14 -- the 603n-style harm/survival warm-up, ported so it can
precede an InfantCurriculumScheduler run.

WHY THIS MODULE EXISTS (the GAP-14 section-3 fork, resolved side (i) PORT by user
decision 2026-09-16; see
REE_assembly/evidence/planning/infant_gap14_redesign_staged_20260827.md).

GAP-14's prerequisite (b) -- "the goal-pipeline training regime produces non-trivial
z_goal in default config" -- was cleared 2026-06-10 by V3-EXQ-603n, but 603n runs
`scaffolded_sd054_onboarding`, NOT `InfantCurriculumScheduler`. The two harnesses are
disjoint: 603n carries a Stage-0 forced-benefit nursery plus a Stage-H harm/survival
co-training stage, while InfantCurriculumScheduler's Phase 0 ("babbling") is the agent's
own native E3 selection (act_with_split_obs, executed class = argmax % 4, so class 4 is
never emitted -- NOT random-policy stepping; corrected 2026-09-25, coupled campaign plan
F1) with no training of any kind. So a naive "flip the crossing-count
flag and re-run 591" inherits none of 603n's z_goal-forming scaffold and produces a
SECOND vacuous z_goal ~ 0 null -- the Phase 1 exit gate (z_goal.norm() >= 0.30) can never
clear, for a reason that has nothing to do with the curriculum under test.

WHAT THE PORT IS, AND WHAT IT IS NOT
------------------------------------
It is NOT a knob on InfantCurriculumScheduler and NOT a REEConfig field. That shape was
proposed and is not implementable: `experiments/infant_curriculum.py` imports only `math`
and `typing` -- it holds no agent, no optimizer and no trainable state, so training cannot
be hosted there; and `scaffold_train_harm_pathway` is a field of
`ScaffoldedSD054OnboardingConfig` (experiments/scaffolded_sd054_onboarding.py:584) backed
by driver-level training code (`_harm_pathway_params`:1311, `_harm_pathway_step`:1396,
`_measure_harm_discriminativeness`:1493, `_make_harm_pathway`:2413), with ZERO occurrences
anywhere under `ree_core/`. There is correspondingly no `ree_core/curriculum/` package.
Verified against ree-v3 HEAD, 2026-09-16.

It IS a harness-layer composition: run the scaffold's own warm-up stages on an agent, then
hand that SAME agent to InfantCurriculumScheduler. Nothing in `ree_core/` changes, nothing
in `scaffolded_sd054_onboarding.py` changes, and no existing driver's behaviour changes --
this module is new and is imported by nothing that predates it.

THE DIMENSION CONSTRAINT (the non-obvious part -- do not "simplify" it away)
---------------------------------------------------------------------------
The scaffold's envs and the 591-lineage infant env have DIFFERENT observation widths:

    scaffold `_build_env(...)`         -> body 17, world 350, action 5
    591-lineage CausalGridWorldV2      -> body 12, world 250, action 5

because the scaffold enables limb damage (+5 body), the reef scent-gradient view (+25
world) and SD-049 per-type resource views (+75 world at n_resource_types=3). An agent is
built at fixed encoder widths, so a warmed agent CANNOT be stepped on a 12/250 env.

Zero-padding a 250-wide observation into a 350-wide slot is REFUSED here: it feeds the
world encoder 100 constant-zero dims where it was trained to read per-type resource
proximity and the reef gradient, i.e. it silently moves the agent off its training
distribution -- the same class of defect as running a probe on a channel that cannot fire.

The resolution is `INFANT_STRUCT_ENV_KWARGS`: build the INFANT curriculum's envs with the
same structural features the warm-up trained under. Measured 2026-09-16 -- all four infant
phases then match the scaffold reference exactly at body 17 / world 350 / action 5. These
structural kwargs are ORTHOGONAL to what the curriculum manipulates
(`InfantCurriculumScheduler.env_kwargs()` varies harm_gradient / transient_benefit /
microhabitat and, in Phase 3, the SD-047/SD-048 destabilizers), so the curriculum-vs-flat
contrast is preserved intact; every arm simply shares a richer common env.

Consequence to state plainly in any consuming experiment: the env is no longer
bit-identical to the 2026-05 V3-EXQ-591 env. Cross-ARM comparability (the thing the design
tests) is preserved because all arms share it; comparability to 591's own recorded numbers
is NOT, and must not be claimed.

COST (measured 2026-09-16 on DLAPTOP, darwin-arm64; inference only, per env step, at
body 17 / world 350)
    603n-style warmed build      0.1168 s/step
    591c diversity-armed build   0.0909 s/step
The 591-lineage 2000-episode shape is therefore ~13.0 h per cell for the warmed build
(~195 h over 3 arms x 5 seeds) BEFORE warm-up. Any consumer must size its episode budget
against that and declare which curriculum phases its budget can actually reach
(PHASE_EP_MIN = [0, 100, 500, 2000]).

KNOWN-INERT CHANNEL -- do not narrate it as the curriculum's mechanism. These structural
kwargs leave `use_proxy_fields` at the CausalGridWorldV2 factory default (True), under which
`harm_gradient_enabled` is STRUCTURALLY INERT: the `hazard_approach` branch pre-empts the
`transition_type == "none"` gate the gradient reward sits behind (confirmed
failure_autopsy_V3-EXQ-996_2026-09-04 red-team F1/F3; pinned since 2026-05-16 by
tests/contracts/test_harm_gradient_gap1.py::test_c3_suppressed_by_proxy_approach; reachable only
above proximity_approach_threshold ~0.33, or with use_proxy_fields=False). The infant
curriculum's Phase 1/2/3 env_kwargs all set `harm_gradient_enabled=True`, and V3-EXQ-996
measured `transient_benefit_enabled` ALONE reproducing its entire phase divergence while
harm_gradient fired 0/600. So a consuming experiment must record a per-cell harm-gradient fire
count and read a zero as a READINESS result, not as a measurement -- and must not attribute a
phase effect to the ARC-046 harm-gradient -> residue -> E3 story without that count being
non-zero.

REUSE: the warm-up is a pure function of (substrate, warm-up config, seed) and is IDENTICAL
across the arms of a curriculum-vs-flat comparison, which differ only in what happens after
it. Warm once per seed and share the result across arms rather than paying it per cell.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _set_goal_pipeline_frozen,
)

# --------------------------------------------------------------------------
# Structural env kwargs -- what makes an INFANT env dimension-match the warm-up.
# --------------------------------------------------------------------------
# limb_damage_enabled  -> body 12 -> 17 (damage[4] + residual_pain)
# reef_*               -> world +25 (reef scent-gradient view); also the refuge the
#                         Stage-0 nursery and P0 spawn the agent into
# multi_resource_*     -> world +75 at n_resource_types=3 (SD-049 per-type views),
#                         which the SD-057 cue-recall bridge binds its tokens against
INFANT_STRUCT_ENV_KWARGS: Dict[str, Any] = {
    "limb_damage_enabled": True,
    "reef_enabled": True,
    "reef_bipartite_layout": True,
    "reef_bipartite_axis": "horizontal",
    "reef_bipartite_agent_band_radius": 1,
    "multi_resource_heterogeneity_enabled": True,
    "n_resource_types": 3,
}

# 603n's own goal-pipeline / encoder dims and calibration, mirrored so the ported
# warm-up reproduces the regime that cleared prerequisite (b).
WORLD_DIM = 32
SELF_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3

# 603n full-scale stage budgets.
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
STEPS_PER_EPISODE = 200

# Non-vacuity floors, mirrored from 603n so a consumer reads the same bars.
HARM_EVAL_RANGE_FLOOR = 0.005
HARM_TRAIN_STEPS_FLOOR = 1
# The gate this warm-up exists to make clearable: InfantCurriculumScheduler's
# Phase 1 -> 2 exit condition (infant_curriculum.Z_GOAL_THRESHOLD).
Z_GOAL_HANDOFF_FLOOR = 0.30


@dataclass
class WarmupResult:
    """What the warm-up achieved, in the terms GAP-14 prerequisite (b) is stated in."""

    z_goal_norm_at_exit: float
    stage0_z_goal_norm_peak: float
    stage0_z_goal_formed: bool
    stage0b_retention_ratio: float
    stage0b_retention_gate_passed: bool
    p0_mean_episode_length: float
    hazard_median_last_window: float
    hazard_survival_gate_passed: bool
    harm_eval_range: float
    harm_pathway_n_train_steps: int
    stages_run: List[str] = field(default_factory=list)
    aborted_at: Optional[str] = None
    abort_reason: str = ""

    # -- the two readiness questions a consumer must gate on ----------------
    @property
    def z_goal_cleared(self) -> bool:
        """Prereq (b): does the agent hand off a z_goal the Phase 1->2 gate admits?"""
        return bool(self.z_goal_norm_at_exit >= Z_GOAL_HANDOFF_FLOOR)

    @property
    def harm_pathway_discriminative(self) -> bool:
        """603n's non-vacuity guard: did the harm landscape actually become trained?"""
        return bool(
            self.harm_pathway_n_train_steps >= HARM_TRAIN_STEPS_FLOOR
            and self.harm_eval_range >= HARM_EVAL_RANGE_FLOOR
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "z_goal_norm_at_exit": float(self.z_goal_norm_at_exit),
            "z_goal_cleared": self.z_goal_cleared,
            "z_goal_handoff_floor": Z_GOAL_HANDOFF_FLOOR,
            "stage0_z_goal_norm_peak": float(self.stage0_z_goal_norm_peak),
            "stage0_z_goal_formed": bool(self.stage0_z_goal_formed),
            "stage0b_retention_ratio": float(self.stage0b_retention_ratio),
            "stage0b_retention_gate_passed": bool(self.stage0b_retention_gate_passed),
            "p0_mean_episode_length": float(self.p0_mean_episode_length),
            "hazard_median_last_window": float(self.hazard_median_last_window),
            "hazard_survival_gate_passed": bool(self.hazard_survival_gate_passed),
            "harm_eval_range": float(self.harm_eval_range),
            "harm_pathway_n_train_steps": int(self.harm_pathway_n_train_steps),
            "harm_pathway_discriminative": self.harm_pathway_discriminative,
            "harm_eval_range_floor": HARM_EVAL_RANGE_FLOOR,
            "stages_run": list(self.stages_run),
            "aborted_at": self.aborted_at,
            "abort_reason": self.abort_reason,
        }


def warmup_scaffold_config(
    *,
    stage0_budget: int = STAGE0_BUDGET,
    stage0b_budget: int = STAGE0B_BUDGET,
    p0_budget: int = P0_BUDGET,
    hazard_budget: int = HAZARD_STAGE_BUDGET,
    steps_per_episode: int = STEPS_PER_EPISODE,
    env_seed: Optional[int] = None,
) -> ScaffoldedSD054OnboardingConfig:
    """The 603n scaffold configuration, restricted to the warm-up stages.

    P1 / P2 are deliberately NOT configured here: the whole point of the port is that the
    InfantCurriculumScheduler takes over after Stage-H, so the scaffold's own foraging and
    measurement stages are replaced rather than run.
    """
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_steps_per_episode=steps_per_episode,
        scaffold_env_seed=env_seed,
        # -- Stage-0 forced-benefit nursery (the z_goal former) --
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0_budget,
        # -- Stage-0b protected consolidation --
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b_budget,
        scaffold_contact_gated_goal_updates=True,
        # -- P0 guided low-conflict warm-up --
        scaffold_p0_episode_budget=p0_budget,
        scaffold_p0_num_hazards=1,
        # -- 634c seeding calibration --
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        # -- SD-057 cue-recall bridge (also what puts SD-049 in the env) --
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # -- Stage-H isolated hazard avoidance, harm pathway TRAINED --
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard_budget,
        scaffold_hazard_stage_num_hazards=4,
        scaffold_hazard_stage_num_resources=2,
        scaffold_hazard_stage_hazard_food_attraction=0.0,
        scaffold_hazard_stage_proximity_harm_scale=0.1,
        scaffold_hazard_stage_spawn_in_reef_half=True,
        scaffold_hazard_stage_survival_gate_steps=75,
        scaffold_hazard_stage_stability_window=10,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    # Short-episode runs (smoke / contract) cannot clear a 75-step survival gate.
    if steps_per_episode < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps_per_episode // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps_per_episode // 4)
    return cfg


def warmup_agent_config(env, *, sleep_loop_episodes_K: int) -> REEConfig:
    """The 603n agent build, at the env's dims, plus the 591c diversity arming.

    The diversity flags (MECH-313 noise floor + MECH-314 structured curiosity) are the
    591b-591h lineage's build; the rest is 603n's. Both are needed: the warm-up trains the
    harm pathway that 603n's flags expose, and the curriculum run that follows is a
    diversity-armed infant-stage run.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # sensory + affective harm streams (SD-010 / SD-011) and the E2 harm forward
        # model (ARC-033), so all four harm-pathway training terms engage
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # 591c diversity arming
        novelty_bonus_weight=0.5,
        use_noise_floor=True,
        use_structured_curiosity=True,
        use_sleep_loop=True,
        sleep_loop_episodes_K=sleep_loop_episodes_K,
    )
    cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return cfg


def infant_env(
    sched,
    *,
    phase: Optional[int] = None,
    seed: Optional[int] = None,
    grid_size: int = 12,
    **extra,
) -> CausalGridWorldV2:
    """An InfantCurriculumScheduler-driven env at WARM-UP-COMPATIBLE dimensions.

    `sched.env_kwargs(phase)` supplies the curriculum manipulation;
    INFANT_STRUCT_ENV_KWARGS supplies the structural features the warmed agent's encoders
    were built for. Caller kwargs win over both, so a flat-baseline arm can pin its own
    env_kwargs.
    """
    kwargs: Dict[str, Any] = {
        "size": grid_size,
        "seed": seed,
        "resource_respawn_on_consume": True,
        "pos_telemetry_enabled": True,
        "traj_telemetry_enabled": True,
    }
    kwargs.update(INFANT_STRUCT_ENV_KWARGS)
    kwargs.update(sched.env_kwargs() if phase is None else sched.env_kwargs(phase=phase))
    kwargs.update(extra)
    return CausalGridWorldV2(**kwargs)


def build_warmed_agent(
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    *,
    sleep_loop_episodes_K: int,
    device: Optional[torch.device] = None,
) -> REEAgent:
    """Build an agent at the warm-up envs' dims (which infant_env() also matches)."""
    ref_env = _build_env(scaffold_cfg, "p0", seed=0)
    ref_env.reset()
    agent = REEAgent(
        warmup_agent_config(ref_env, sleep_loop_episodes_K=sleep_loop_episodes_K))
    return agent.to(device) if device is not None else agent


def run_warmup(
    agent: REEAgent,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    *,
    device: Optional[torch.device] = None,
    include_hazard_stage: bool = True,
    verbose: bool = False,
) -> WarmupResult:
    """Run Stage-0 -> Stage-0b -> P0 -> Stage-H on `agent`, then UNFREEZE the goal pipeline.

    Mutates `agent` in place and returns what it achieved. The goal pipeline is left
    UNFROZEN on exit -- the scaffold's own `run_p1` is what normally does that, and it is
    exactly the stage the InfantCurriculumScheduler replaces, so the unfreeze is performed
    here instead. A caller that skipped it would hand the curriculum an agent whose
    MECH-295 / MECH-307 goal writes are still short-circuited.

    `include_hazard_stage=False` runs only the z_goal-forming stages (Stage-0, Stage-0b).
    That is the cheap path for a contract test of prerequisite (b): Stage-H trains
    harm/survival and does not write z_goal (it runs with seed_goal=False).
    """
    device = device or torch.device("cpu")
    sched = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    stages: List[str] = []

    def _zg() -> float:
        gs = getattr(agent, "goal_state", None)
        return float(gs.goal_norm()) if gs is not None else 0.0

    def _abort(stage: str, reason: str, **kw) -> WarmupResult:
        _set_goal_pipeline_frozen(
            agent, frozen=False, strict=scaffold_cfg.scaffold_strict_goal_isolation)
        base: Dict[str, Any] = {
            "z_goal_norm_at_exit": _zg(), "stage0_z_goal_norm_peak": 0.0,
            "stage0_z_goal_formed": False, "stage0b_retention_ratio": 0.0,
            "stage0b_retention_gate_passed": False, "p0_mean_episode_length": 0.0,
            "hazard_median_last_window": 0.0, "hazard_survival_gate_passed": False,
            "harm_eval_range": 0.0, "harm_pathway_n_train_steps": 0,
        }
        base.update(kw)
        return WarmupResult(
            stages_run=stages, aborted_at=stage, abort_reason=reason, **base)

    # -- Stage 0: forced-benefit nursery. This is the z_goal FORMER. --------
    s0 = sched.run_stage0_nursery(agent, device)
    stages.append("stage0")
    if verbose:
        print("  [warmup] stage0 z_goal_peak=%.4f formed=%s eps=%d"
              % (s0.z_goal_norm_peak, s0.z_goal_formed, s0.n_episodes), flush=True)
    if s0.aborted:
        return _abort("stage0", s0.abort_reason,
                      stage0_z_goal_norm_peak=float(s0.z_goal_norm_peak))

    # -- Stage 0b: PROTECTED consolidation (update_z_goal is not called). ---
    s0b = sched.run_stage0b_consolidation(
        agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    stages.append("stage0b")
    if verbose:
        print("  [warmup] stage0b retention=%.3f gate=%s"
              % (s0b.retention_ratio, s0b.retention_gate_passed), flush=True)
    if s0b.aborted:
        return _abort("stage0b", s0b.abort_reason,
                      stage0_z_goal_norm_peak=float(s0.z_goal_norm_peak),
                      stage0_z_goal_formed=bool(s0.z_goal_formed),
                      stage0b_retention_ratio=float(s0b.retention_ratio))

    p0_mean = 0.0
    hz_median = 0.0
    hz_pass = False
    harm_range = 0.0
    harm_steps = 0

    if include_hazard_stage:
        # -- P0: guided low-conflict warm-up, goal frozen, harm pathway co-trains. --
        p0 = sched.run_p0(agent, device)
        stages.append("p0")
        p0_mean = float(p0.mean_episode_length)
        if verbose:
            print("  [warmup] p0 mean_len=%.1f" % p0_mean, flush=True)
        if p0.aborted:
            return _abort("p0", p0.abort_reason,
                          stage0_z_goal_norm_peak=float(s0.z_goal_norm_peak),
                          stage0_z_goal_formed=bool(s0.z_goal_formed),
                          stage0b_retention_ratio=float(s0b.retention_ratio),
                          stage0b_retention_gate_passed=bool(s0b.retention_gate_passed),
                          p0_mean_episode_length=p0_mean)

        # -- Stage-H: isolated hazard avoidance with the harm pathway TRAINED. --
        hz = sched.run_hazard_avoidance(agent, device)
        stages.append("hazard")
        harm_diag = dict(getattr(hz, "harm_discriminativeness", {}) or {})
        harm_pathway_diag = dict(getattr(hz, "harm_pathway_diag", {}) or {})
        hz_median = float(hz.median_last_window_episode_length)
        hz_pass = bool(hz.survival_gate_passed)
        harm_range = float(harm_diag.get("harm_eval_range", 0.0))
        harm_steps = int(harm_pathway_diag.get("n_train_steps", 0))
        if verbose:
            print("  [warmup] hazard median_last=%.1f survival=%s harm_eval_range=%.4f"
                  " steps=%d" % (hz_median, hz_pass, harm_range, harm_steps), flush=True)
        if hz.aborted:
            return _abort("hazard", hz.abort_reason,
                          stage0_z_goal_norm_peak=float(s0.z_goal_norm_peak),
                          stage0_z_goal_formed=bool(s0.z_goal_formed),
                          stage0b_retention_ratio=float(s0b.retention_ratio),
                          stage0b_retention_gate_passed=bool(s0b.retention_gate_passed),
                          p0_mean_episode_length=p0_mean,
                          harm_eval_range=harm_range,
                          harm_pathway_n_train_steps=harm_steps)

    # -- Hand-off: the scaffold's run_p1 normally unfreezes the goal pipeline; the
    # InfantCurriculumScheduler replaces run_p1, so do it here. -------------
    _set_goal_pipeline_frozen(
        agent, frozen=False, strict=scaffold_cfg.scaffold_strict_goal_isolation)

    result = WarmupResult(
        z_goal_norm_at_exit=_zg(),
        stage0_z_goal_norm_peak=float(s0.z_goal_norm_peak),
        stage0_z_goal_formed=bool(s0.z_goal_formed),
        stage0b_retention_ratio=float(s0b.retention_ratio),
        stage0b_retention_gate_passed=bool(s0b.retention_gate_passed),
        p0_mean_episode_length=p0_mean,
        hazard_median_last_window=hz_median,
        hazard_survival_gate_passed=hz_pass,
        harm_eval_range=harm_range,
        harm_pathway_n_train_steps=harm_steps,
        stages_run=stages,
    )
    if verbose:
        print("  [warmup] EXIT z_goal=%.4f cleared=%s (floor %.2f)"
              % (result.z_goal_norm_at_exit, result.z_goal_cleared,
                 Z_GOAL_HANDOFF_FLOOR), flush=True)
    return result


def warmup_config_slice(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> Dict[str, Any]:
    """Declared config slice for arm-fingerprint reuse of a warmed cell.

    Only what the warm-up computation READS -- never acceptance thresholds or arm labels.
    """
    return {
        "warmup": "infant_warmup/v1",
        "stage0_budget": scaffold_cfg.scaffold_stage0_episode_budget,
        "stage0b_budget": scaffold_cfg.scaffold_stage0b_episode_budget,
        "p0_budget": scaffold_cfg.scaffold_p0_episode_budget,
        "hazard_budget": scaffold_cfg.scaffold_hazard_stage_episode_budget,
        "steps_per_episode": scaffold_cfg.scaffold_steps_per_episode,
        "harm_pathway_lr": scaffold_cfg.scaffold_harm_pathway_lr,
        "seeding": {
            "gain": scaffold_cfg.scaffold_z_goal_seeding_gain,
            "benefit_threshold": scaffold_cfg.scaffold_benefit_threshold,
            "drive_floor": scaffold_cfg.scaffold_drive_floor,
        },
        "agent": {
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "drive_weight": DRIVE_WEIGHT,
            "z_harm_a_dim": HARM_A_DIM, "cue_recall_gain": CUE_RECALL_GAIN,
            "novelty_bonus_weight": 0.5, "use_noise_floor": True,
            "use_structured_curiosity": True,
        },
        "env_struct": dict(INFANT_STRUCT_ENV_KWARGS),
    }
