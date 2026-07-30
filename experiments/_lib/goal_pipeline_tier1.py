"""
Shared helpers for goal_pipeline:GAP-4 Tier-1 StepHarness retest cohort.

Operating substrate (post GAP-3 / V3-EXQ-582a):
  drive_floor=0.9, drive_ema_alpha=1.0 (Option 1 OFF),
  REEConfig.goal_stream() bundle (MECH-307 + MECH-295 + schema wanting),
  post-540e relaxed MECH-295 activation floors, use_dacc=True.

2026-05-29 rebuild (post V3-EXQ-490g-cohort autopsy Fork A):
  - cfg.use_dacc=True is now UNCONDITIONAL in build_config (was nested in the
    gap4_operating=True branch only). Closes the 483c primary diagnosis:
    agent.dacc=None across all 12 runs because callers used gap4_operating
    arms but downstream config wiring left use_dacc unset under the
    REEConfig.goal_stream + arm.extra_config composition path. Every cohort
    experiment now gets dACC instantiation without per-script opt-in.
  - evaluate_tier1_cohort C3_lift_vs_baseline metric DEFAULT switched from
    approach_commit_rate (saturates at 1.0 in OFF_OFF baseline under
    drive_floor=0.9 + goal_stream + reef -- no headroom for lift per 483c
    OFF_OFF=ON_OFF=1.0 byte-identical observation) to goal_norm_peak delta
    (substrate-side, cross-claim-comparable; 483c/524a manifests show
    range 0.09-0.36 under realistic substrate firing).
  - SD-037-specific scripts can override via c3_lift_metric=
    "override_signal_nonzero_steps" at the evaluate_tier1_cohort call site
    -- measures the primary PAG override pathway directly. eval_tier1 now
    populates this metric automatically (broadcast_override.override_signal
    > 1e-3 step count).

2026-07-30 correction (THIRD defeat-by-construction in this cohort).
The 2026-05-29 bullet above is WRONG in both directions, and the C3 lift
criterion it installed could not pass on the pairing it was applied to.
Both errors are recorded here rather than deleted, because the shape of the
mistake is the reason for the capability guard added below.

  (a) The SATURATION evidence was measured on the wrong contrast.
      483c's four arms -- OFF_OFF / ON_OFF / OFF_ON / ON_ON -- are ALL
      gap4_operating=True; they factorialise use_pag_freeze_gate x
      use_broadcast_override WITHIN the GAP-4 stack. So "saturates at 1.0 in
      OFF_OFF baseline under drive_floor=0.9 + goal_stream" is a statement
      about a gap4 arm, and 524a has no baseline arm at all (single-arm).
      approach_commit_rate does saturate on a WITHIN-gap4 sub-feature
      contrast; it is perfectly discriminative on the BETWEEN-path contrast
      the default was then applied to -- 490i measured base 0.0 vs gap4 1.0
      on 3/3 seeds. The deprecation generalised one contrast type to the
      other.

  (b) The REPLACEMENT range was also measured on the gap4 arm only.
      "483c/524a manifests show range 0.09-0.36" describes ARM_1/ON_ON
      values. Nobody measured goal_norm_peak on a non-gap4 arm. When 490i
      finally did, the legacy arm sat ABOVE the gap4 arm on every seed
      (base 0.792/12.489/0.468 vs gap4 0.226/0.092/0.296), so the
      `gap4 > base + 0.01` predicate returned False 3/3 -- inverted, not
      merely short of threshold.

Why goal_norm_peak cannot carry a cross-arm lift criterion at all:
  * It is a LIFETIME RUNNING MAXIMUM. GoalState.reset() zeroes it, but
    REEAgent.reset() never calls GoalState.reset(), so the value reported by
    eval_tier1 is the max over every warmup AND eval step of the agent's
    life (~12k steps), not a per-episode or per-eval statistic. An
    extreme-value statistic over 12k samples has no central tendency and is
    dominated by a single excursion -- which is what the ARM_0 seed-7 spike
    (12.489, ~26x its own seed-42 value) is.
  * It is a norm in each arm's OWN FREE-SCALE latent space. z_goal is an EMA
    toward z_world (goal.py:825-830), and nothing in the latent stack L2- or
    layer-normalises z_world. The two arms are built by DIFFERENT REEConfig
    constructors (goal_stream(...) vs from_dims(...)) with different encoders
    and different auxiliary losses, so their z_world scales are not
    commensurable. Comparing them with a fixed ADDITIVE threshold (0.01)
    compares two unrelated units.
  * ARM_0_legacy_collapsed is not a goal-severed control: from_dims(...) sets
    z_goal_enabled=True. "Collapsed" names the legacy config path, not an
    absent goal stream. The 490i autopsy already flagged this as
    "C3_lift_vs_baseline metric-design contamination"; the metric was left
    in place as the default.

Fixes installed 2026-07-30:
  * C3_METRICS registry -- every C3 lift metric now DECLARES its direction,
    its ceiling, and whether it is valid across a gap4/non-gap4 boundary.
    _c3_lift_compare reads the registry instead of hardcoding `>`.
  * c3_lift_capability() -- a PRE-REGISTRATION CAPABILITY GUARD. Before a
    lift result may be read as a substrate finding, it asks whether the
    metric was able to move in the claimed direction on this pairing at all,
    and returns invalid_cross_arm / degenerate / saturated / inverted / ok.
  * evaluate_tier1_cohort now returns C3_lift_status + criterion_valid +
    recommended_evidence_direction, and tier1_evidence_direction() maps a
    defeated-by-construction criterion to "non_contributory" instead of
    "weakens". This is the direct fix for the algorithm-generated-weakens
    pattern the 2026-05-29 cluster autopsy had to correct by hand.
  * DEFAULT_C3_LIFT_METRIC reverted to approach_commit_rate, which is the
    correct default for this harness's canonical BETWEEN-path contrast. On a
    within-gap4 contrast the guard now reports `saturated` loudly rather than
    silently returning a wrong answer.
  * harm_norm_sustain_ratio added -- a scale-free, decay-shaped metric for
    SD-036 / Q-040. See its registry entry for why no goal-side metric can
    serve those claims (the SD-036 regulator never touches z_goal).
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Path shim -- the _lib idiom (see probe_warmup.py:143-153 and
# baselines/maturation_curriculum.py:117-123). Required, not cosmetic: BOTH
# load orders were broken before this existed.
#   * `import experiments._lib.goal_pipeline_tier1` from the repo root raised
#     ModuleNotFoundError: _harness. A bare-name import resolves only when
#     experiments/ is itself on sys.path -- true under direct script execution
#     (sys.path[0] = the script's own dir, which is how experiment_runner.py
#     invokes every driver, and why production never sees this) but not under
#     the package spelling used by contract tests and tooling.
#   * `cd experiments && import _lib.goal_pipeline_tier1` raised
#     ModuleNotFoundError: ree_core -- the repo root is not on the path there.
# Inserting both directories makes this module import cleanly either way.
#
# The harness import below stays BARE deliberately. Rewriting it to
# `experiments._harness` would also clear the first error, but it would split
# _harness into two module objects -- and therefore two copies of the
# module-level `_action_random` (_harness.py:90) -- in any process that imports
# the harness bare and this module package-spelled. Measured across this
# module's 30 importers, that rewrite CREATES the split in 11 of them (the ten
# bare/bare drivers plus probe_warmup.py, whose own shim comment at :144 notes
# that this module imports its harness flat) and removes it in only 3
# (827/827a/828). Same reasoning as ree-v3 73407e22e1, which left
# test_driver_closure_env_seed_determinism.py:48 bare because switching it
# would have created a second identity rather than removed one.
_LIB_DIR = Path(__file__).resolve().parent          # experiments/_lib
_EXP_DIR = _LIB_DIR.parent                          # experiments
_REPO_ROOT = _EXP_DIR.parent                        # ree-v3
for _p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

DRIVE_FLOOR_OPERATING = 0.9
DRIVE_EMA_ALPHA_OFF = 1.0
SEEDS_DEFAULT = [42, 7, 19]
WARMUP_EPISODES_DEFAULT = 50
EVAL_EPISODES_DEFAULT = 10
STEPS_PER_EPISODE_DEFAULT = 200

WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

APPROACH_WANTING_THRESH = 0.05
TIER1_CUE_FIRES_MIN = 1
TIER1_DACC_BIAS_MIN = 1
TIER1_APPROACH_COMMIT_MIN = 1
TIER1_GOAL_ACTIVE_FRAC_MIN = 0.05
TIER1_SEEDS_PASS_MIN = 2
TIER1_GOAL_NORM_PEAK_DELTA = 0.01
TIER1_OVERRIDE_SIGNAL_DELTA = 1
TIER1_HARM_SUSTAIN_DELTA = 0.01
# Tolerance below which two arms' metric values count as the SAME value, i.e.
# the toggled feature did not perturb this metric at all. Deliberately loose
# relative to float noise: 483c seed 7 produced goal_norm_peak values equal to
# 8 decimal places across OFF/ON arms, which is a degenerate criterion, not a
# near-miss.
TIER1_C3_DEGENERATE_TOL = 1e-6

# ---------------------------------------------------------------------------
# C3 lift metric registry.
#
# Every C3_lift_vs_baseline metric DECLARES its properties so the comparison
# and the capability guard can both be derived rather than hardcoded. Adding a
# metric means adding an entry here; _c3_lift_compare and c3_lift_capability
# pick it up automatically.
#
#   key             row field read from the per-(arm, seed) metrics dict
#   direction       "higher" -> gap4 must EXCEED base to count as lift
#                   "lower"  -> gap4 must FALL BELOW base to count as lift
#                   (a decay/suppression mechanism is expected to REDUCE its
#                   readout; hardcoding `>` silently mis-signs those claims)
#   delta           required absolute margin beyond the baseline
#   ceiling/floor   saturation bound of the metric, or None. A baseline
#                   already at the bound cannot be beaten in that direction.
#   cross_arm_valid False -> the metric is NOT comparable between a
#                   gap4_operating=True and a gap4_operating=False arm.
#   scale_free      informational: True when the metric is a ratio of an
#                   arm's own quantities and so carries no latent-scale unit.
# ---------------------------------------------------------------------------
C3_METRICS: Dict[str, Dict[str, Any]] = {
    "approach_commit_rate": dict(
        key="approach_commit_rate",
        direction="higher",
        delta=0.0,
        ceiling=1.0,
        floor=0.0,
        cross_arm_valid=True,
        scale_free=True,
        note=(
            "Canonical BETWEEN-path lift: a severed/legacy arm cannot commit "
            "approach, an operating GAP-4 arm can. 490i measured base 0.0 vs "
            "gap4 1.0 on 3/3 seeds. SATURATES at 1.0 on a WITHIN-gap4 "
            "sub-feature contrast (483c: all 12 rows = 1.0). Replaying 483c "
            "through the guard reports `degenerate` rather than `saturated` -- "
            "both hold, and the degeneracy check runs first because 'the "
            "toggle does not move this metric' is the more specific finding. "
            "Either way criterion_valid is False; pick a mechanism-specific "
            "metric for a within-gap4 contrast."
        ),
    ),
    "goal_norm_peak_delta": dict(
        key="goal_norm_peak",
        direction="higher",
        delta=TIER1_GOAL_NORM_PEAK_DELTA,
        ceiling=None,
        floor=0.0,
        cross_arm_valid=False,
        scale_free=False,
        note=(
            "RETAINED FOR BACK-COMPAT ONLY -- do not select for a new "
            "experiment. goal_norm_peak is a LIFETIME RUNNING MAXIMUM of an "
            "UNNORMALISED latent norm (see module docstring, 2026-07-30). It "
            "is not comparable across the gap4 boundary (different REEConfig "
            "constructors -> different z_world scales), and an additive "
            "threshold on an extreme-value statistic has no stable meaning "
            "even within one arm (ARM_0 spread 0.47-12.49 across 3 seeds)."
        ),
    ),
    "override_signal_nonzero_steps": dict(
        key="override_signal_nonzero_steps",
        direction="higher",
        delta=TIER1_OVERRIDE_SIGNAL_DELTA,
        ceiling=None,
        floor=0.0,
        cross_arm_valid=True,
        scale_free=False,
        note=(
            "SD-037-specific. OFF arms have broadcast_override=None so the "
            "signal is 0 by construction; cleanly discriminative for ON arms."
        ),
    ),
    "harm_norm_sustain_ratio": dict(
        key="harm_norm_sustain_ratio",
        direction="lower",
        delta=TIER1_HARM_SUSTAIN_DELTA,
        ceiling=1.0,
        floor=0.0,
        cross_arm_valid=True,
        scale_free=True,
        note=(
            "SD-036 / Q-040 decay-shaped readout: mean ||z_harm|| over eval "
            "steps divided by that arm's own peak ||z_harm||. Scale-free (a "
            "ratio of one arm's own quantities), and DECAY-shaped -- a "
            "working GABAergic decay regulator should LOWER it, hence "
            "direction='lower'. This is the metric a goal-side readout cannot "
            "replace: the SD-036 regulator ticks z_harm / z_harm_a / z_beta "
            "(regulators/gabaergic_decay.py) and NEVER touches z_goal, so no "
            "goal_* metric is causally reachable by the toggle. The substrate "
            "comment at agent.py:4163 names exactly this quantity as the "
            "V3-EXQ-471 catatonic-lock signature: 'a single hazard contact "
            "pinned z_harm_norm at ~0.7 for 199 steps'."
        ),
    ),
}

# Canonical contrast for this harness is BETWEEN-path (gap4_operating False vs
# True), for which approach_commit_rate is the discriminative metric. See the
# module docstring for why the 2026-05-29 switch away from it was made on
# within-gap4 evidence and did not hold here.
DEFAULT_C3_LIFT_METRIC = "approach_commit_rate"

ENV_FISHTANK_KWARGS: Dict[str, Any] = dict(
    size=10,
    num_hazards=3,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=10,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    n_landmarks_b=2,
)

ENV_REEF_KWARGS: Dict[str, Any] = dict(
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
)

WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4


@dataclass
class ArmSpec:
    arm_id: str
    gap4_operating: bool = False
    use_gabaergic_decay: bool = False
    use_pag_freeze_gate: bool = False
    use_broadcast_override: bool = False
    extra_config: Dict[str, Any] = field(default_factory=dict)


def make_env(seed: int, env_kwargs: Optional[Dict[str, Any]] = None) -> CausalGridWorldV2:
    kw = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    return CausalGridWorldV2(seed=seed, **kw)


def build_config(
    env: CausalGridWorldV2,
    arm: ArmSpec,
    *,
    enable_affective_harm_stream: bool = False,
) -> REEConfig:
    """Build the cohort REEConfig for an arm.

    enable_affective_harm_stream: SD-011 affective harm stream toggle for the
        gap4_operating=True path. The non-gap4 (from_dims) branch ALREADY
        enables the stream unconditionally, but the gap4 (goal_stream) branch
        historically did NOT forward the SD-011 flags, so z_harm_a stayed None
        on every latent (every consumer-input quantity read exactly 0.0 -- the
        V3-EXQ-620 / V3-EXQ-625 measurement artifact diagnosed 2026-06-01).
        Default False keeps every existing gap4 caller bit-identical; SD-037
        consumer-input-distribution scripts (V3-EXQ-620b / V3-EXQ-625b) opt in
        with True. Flipping this default would alter agent behaviour for all
        gap4 callers whose harm consumers are on (483d/483e/490g-j/524a/...) and
        must be a separate, re-validated decision -- do NOT change the default
        here.
    """
    if arm.gap4_operating:
        gs_kwargs: Dict[str, Any] = dict(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
            drive_weight=2.0,
            goal_weight=0.5,
            benefit_threshold=0.1,
            use_mech307=True,
            use_consumer_conjunction_read=True,
            use_resource_encoder=True,
            drive_floor=DRIVE_FLOOR_OPERATING,
            drive_ema_alpha=DRIVE_EMA_ALPHA_OFF,
        )
        if enable_affective_harm_stream:
            # Mirror the non-gap4 branch's SD-011 enablement so the
            # AffectiveHarmEncoder is instantiated and the harness-threaded
            # env harm_obs_a populates z_harm_a. Forwarded through
            # goal_stream(**kwargs) -> from_dims(...). limb_damage_enabled is
            # passed through to from_dims so it auto-sizes harm_obs_a_dim to the
            # env's actual harm_obs_a width (7 under limb damage, else 50);
            # without it the encoder is built for the 50-dim legacy width and
            # crashes on the 7-dim body-damage harm_obs_a (shape mismatch).
            _env_kw = env_kwargs_or_default(env)
            gs_kwargs.update(
                use_harm_stream=True,
                z_harm_dim=HARM_DIM,
                use_affective_harm_stream=True,
                z_harm_a_dim=HARM_A_DIM,
                harm_history_len=HARM_HISTORY_LEN,
                limb_damage_enabled=bool(_env_kw.get("limb_damage_enabled", False)),
            )
        cfg = REEConfig.goal_stream(**gs_kwargs)
        cfg.mech295_min_drive_to_fire = 0.01
        cfg.mech295_min_z_goal_norm_to_fire = 0.005
        cfg.mech295_drive_to_liking_gain = 2.0
        cfg.mech295_liking_to_approach_cue_gain = 0.5
        cfg.use_e2_harm_a = True
        cfg.residue.valence_enabled = True
    else:
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
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
            drive_floor=0.0,
            drive_ema_alpha=DRIVE_EMA_ALPHA_OFF,
            limb_damage_enabled=bool(
                env_kwargs_or_default(env).get("limb_damage_enabled", False)
            ),
            damage_increment=float(env_kwargs_or_default(env).get("damage_increment", 0.15)),
            failure_prob_scale=float(
                env_kwargs_or_default(env).get("failure_prob_scale", 0.3)
            ),
            heal_rate=float(env_kwargs_or_default(env).get("heal_rate", 0.002)),
        )
        cfg.e3.goal_weight = float(cfg.goal.goal_weight)
        cfg.residue.valence_enabled = True

    cfg.e3.commitment_threshold = 0.5
    cfg.heartbeat.beta_gate_bistable = True
    cfg.harm_descending_mod_enabled = True
    cfg.descending_attenuation_factor = 0.5
    # use_dacc is the GAP-4 cohort default per the 2026-05-29 V3-EXQ-490g-cohort
    # autopsy Fork A library rebuild. Closes the 483c primary diagnosis (agent.dacc
    # is None -> C2_dacc_bias=0 unconditionally). Applies to both gap4_operating
    # branches so every cohort experiment gets dACC instantiation without per-script
    # opt-in. arm.extra_config can still override (e.g. {"use_dacc": False} for an
    # ablation arm).
    cfg.use_dacc = True
    cfg.use_gabaergic_decay = bool(arm.use_gabaergic_decay)
    cfg.use_pag_freeze_gate = bool(arm.use_pag_freeze_gate)
    cfg.use_broadcast_override = bool(arm.use_broadcast_override)

    for key, val in arm.extra_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    return cfg


def env_kwargs_or_default(env: CausalGridWorldV2) -> Dict[str, Any]:
    return getattr(env, "_exq_env_kwargs", ENV_FISHTANK_KWARGS)


def _approach_commit(agent: REEAgent) -> bool:
    if not bool(getattr(agent.beta_gate, "is_elevated", False)):
        return False
    if agent._current_latent is None:
        return False
    z = agent._current_latent.z_world
    with torch.no_grad():
        v = agent.residue_field.evaluate_valence(z)
    wanting_amp = float(v[0, VALENCE_WANTING].item())
    return wanting_amp > APPROACH_WANTING_THRESH


def _dacc_bias_norm(agent: REEAgent) -> float:
    if agent.dacc is None:
        return 0.0
    bundle = getattr(agent.dacc, "_last_bundle", None)
    if bundle is None:
        return 0.0
    sb = bundle.get("mode_ev")
    if sb is None:
        sb = bundle.get("harm_interaction")
    if sb is None:
        return 0.0
    try:
        return float(torch.as_tensor(sb).norm().item())
    except Exception:
        return 0.0


def _override_signal_value(agent: REEAgent) -> float:
    """SD-037 BroadcastOverrideRegulator override_signal readout; 0.0 in OFF arms."""
    bo = getattr(agent, "broadcast_override", None)
    if bo is None:
        return 0.0
    return float(getattr(bo, "override_signal", 0.0))


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    label: str,
    progress_total_episodes: Optional[int] = None,
) -> Dict[str, float]:
    progress_denom = progress_total_episodes or num_episodes
    device = agent.device
    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    harness = StepHarness(agent, env, train_mode=True)

    agent.train()
    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        z_world_prev = None
        action_prev = None

        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            latent = result.latent
            z_world_curr = latent.z_world.detach()

            rv = result.next_obs_dict.get("resource_field_view")
            if isinstance(rv, torch.Tensor):
                prox_t = float(rv.max().item())
            else:
                prox_t = float(np.max(rv)) if rv is not None else 0.0

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(result.harm_signal)) if float(result.harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            action_prev = result.action.detach()
            obs_dict = result.next_obs_dict
            if result.done:
                break

        if (ep + 1) % 10 == 0 or ep + 1 == num_episodes:
            print(
                f"  [train] {label} ep {ep + 1}/{progress_denom}",
                flush=True,
            )

    return {"warmup_episodes": float(num_episodes)}


def eval_tier1(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "approach_commit_steps": 0,
        "total_eval_steps": 0,
        "dacc_bias_nonzero_steps": 0,
        "override_signal_nonzero_steps": 0,
        "bridge_cue_fires": 0,
        "bridge_write_fires": 0,
        "goal_active_steps": 0,
        "resource_contacts": 0,
        "action_counts": {},
        # SD-036 / Q-040 decay readout accumulators (see harm_norm_sustain_ratio
        # in C3_METRICS). Tracked over EVAL steps only.
        "harm_norm_sum": 0.0,
        "harm_norm_peak": 0.0,
        "harm_norm_steps": 0,
    }

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        metrics["total_eval_steps"] += 1
        zh = getattr(latent, "z_harm", None)
        if zh is not None:
            with torch.no_grad():
                hn = float(torch.as_tensor(zh).norm().item())
            metrics["harm_norm_sum"] += hn
            metrics["harm_norm_steps"] += 1
            if hn > metrics["harm_norm_peak"]:
                metrics["harm_norm_peak"] = hn
        if _approach_commit(agent):
            metrics["approach_commit_steps"] += 1
        if _dacc_bias_norm(agent) > 1e-6:
            metrics["dacc_bias_nonzero_steps"] += 1
        if _override_signal_value(agent) > 1e-3:
            metrics["override_signal_nonzero_steps"] += 1
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
        br = getattr(agent, "mech295_bridge", None)
        if br is not None:
            metrics["bridge_cue_fires"] = int(getattr(br, "_n_cue_fires", 0))
            metrics["bridge_write_fires"] = int(getattr(br, "_n_write_fires", 0))

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        if getattr(agent, "mech295_bridge", None) is not None:
            agent.mech295_bridge._n_cue_fires = 0
            agent.mech295_bridge._n_write_fires = 0

        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                metrics["resource_contacts"] += 1
            aidx = int(result.action.argmax(dim=-1).item())
            ac = metrics["action_counts"]
            ac[aidx] = ac.get(aidx, 0) + 1
            obs_dict = result.next_obs_dict
            if result.done:
                break

    total = max(1, int(metrics["total_eval_steps"]))
    metrics["approach_commit_rate"] = float(metrics["approach_commit_steps"]) / total
    metrics["goal_active_fraction"] = float(metrics["goal_active_steps"]) / total
    metrics["action_entropy"] = _entropy(metrics["action_counts"])
    metrics["action_counts"] = {str(k): int(v) for k, v in metrics["action_counts"].items()}
    if agent.goal_state is not None:
        metrics["goal_norm_peak"] = float(getattr(agent.goal_state, "_goal_norm_peak", 0.0))
    else:
        metrics["goal_norm_peak"] = 0.0
    # Provenance, because the field name reads like an eval-scoped statistic and
    # is not one: GoalState.reset() zeroes _goal_norm_peak but REEAgent.reset()
    # never calls it, so this is a running maximum over the agent's whole life
    # (warmup + eval, ~12k steps). Recorded rather than "fixed" so historical
    # manifests stay comparable; see C3_METRICS["goal_norm_peak_delta"].
    metrics["goal_norm_peak_scope"] = "agent_lifetime_running_max"

    # SD-036 / Q-040: scale-free decay readout. Mean ||z_harm|| over eval steps
    # as a fraction of this arm's OWN peak ||z_harm||. A working decay regulator
    # lowers it (harm returns toward baseline between contacts instead of
    # staying pinned). 0.0 when the arm has no harm stream or never fired.
    _hpeak = float(metrics["harm_norm_peak"])
    _hsteps = int(metrics["harm_norm_steps"])
    if _hpeak > 1e-12 and _hsteps > 0:
        metrics["harm_norm_mean"] = float(metrics["harm_norm_sum"]) / _hsteps
        metrics["harm_norm_sustain_ratio"] = metrics["harm_norm_mean"] / _hpeak
    else:
        metrics["harm_norm_mean"] = 0.0
        metrics["harm_norm_sustain_ratio"] = 0.0
    return metrics


def tier1_seed_pass(metrics: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "C1_cue_fires": int(metrics.get("bridge_cue_fires", 0)) >= TIER1_CUE_FIRES_MIN,
        "C2_dacc_bias": int(metrics.get("dacc_bias_nonzero_steps", 0)) >= TIER1_DACC_BIAS_MIN,
        "C3_approach_commit": int(metrics.get("approach_commit_steps", 0)) >= TIER1_APPROACH_COMMIT_MIN,
        "C4_goal_active": float(metrics.get("goal_active_fraction", 0.0)) >= TIER1_GOAL_ACTIVE_FRAC_MIN,
    }


def _c3_metric_spec(metric: str) -> Dict[str, Any]:
    try:
        return C3_METRICS[metric]
    except KeyError:
        raise ValueError(
            "Unknown c3_lift_metric '{}'. Supported: {}.".format(
                metric, ", ".join(sorted(C3_METRICS))
            )
        )


def _c3_lift_compare(
    gap4_row: Dict[str, Any],
    base_row: Dict[str, Any],
    metric: str,
) -> bool:
    """Per-seed C3 lift predicate, direction-aware via the C3_METRICS registry.

    "Lift" means the gap4 arm moved AWAY from the baseline in the direction the
    metric declares, by at least its declared delta. For direction="lower" (a
    decay / suppression readout) that means gap4 < base - delta; hardcoding `>`
    for every metric is what mis-signed SD-036. See C3_METRICS for the per-
    metric rationale and the module docstring for the 2026-07-30 correction.
    """
    spec = _c3_metric_spec(metric)
    key = spec["key"]
    delta = float(spec["delta"])
    g = float(gap4_row.get(key, 0.0))
    b = float(base_row.get(key, 0.0))
    if spec["direction"] == "lower":
        return g < (b - delta)
    return g > (b + delta)


def _rows_cross_gap4_boundary(
    gap4_rows: List[Dict[str, Any]], base_rows: List[Dict[str, Any]]
) -> Optional[bool]:
    """True when the two arms sit on opposite sides of the gap4_operating split.

    Returns None when the rows do not carry gap4_operating provenance (runs
    produced before run_seed_arm started emitting it), in which case the
    cross-arm validity check is skipped rather than guessed.
    """
    g_flags = {r.get("gap4_operating") for r in gap4_rows}
    b_flags = {r.get("gap4_operating") for r in base_rows}
    if None in g_flags or None in b_flags or not g_flags or not b_flags:
        return None
    return bool(g_flags != b_flags)


def c3_lift_capability(
    gap4_rows: List[Dict[str, Any]],
    base_rows: List[Dict[str, Any]],
    metric: str,
) -> Dict[str, Any]:
    """PRE-REGISTRATION CAPABILITY GUARD for the C3 lift criterion.

    Asks, against the rows actually produced, whether this metric was CAPABLE
    of moving in its declared direction on this arm pairing -- before a failed
    lift is allowed to be read as a substrate finding.

    Motivation: goal_pipeline:GAP-4 Tier-1 has now been defeated by
    construction three times, each time discovered only after a cohort had
    run and stamped claim directions off the result.
      1. approach_commit_rate saturated at 1.0 in the 483c/524a baseline arm
         (within-gap4 contrast) -- no headroom.
      2. mech295_bias_range_mean = 0.0, making an argmin-flip impossible
         (V3-EXQ-490k).
      3. goal_norm_peak_delta inverted on the between-path pairing, introduced
         by the rebuild that was fixing (1) -- see the module docstring.
    Each was individually invisible and collectively a pattern, so the check
    belongs in the harness rather than in each author's head.

    status values, most severe first:
      "invalid_cross_arm" -- the metric declares cross_arm_valid=False and the
          two arms straddle the gap4_operating boundary. The comparison has no
          meaning; the result must not be read at all.
      "degenerate" -- base and gap4 values are equal within
          TIER1_C3_DEGENERATE_TOL on >= TIER1_SEEDS_PASS_MIN paired seeds. The
          toggled feature does not perturb this metric, so the criterion cannot
          fire regardless of substrate behaviour.
      "saturated" -- the baseline already sits at the metric's bound in the
          direction of the claim on >= TIER1_SEEDS_PASS_MIN paired seeds. It
          cannot be beaten in that direction.
      "inverted" -- the baseline beats the gap4 arm in the claimed direction on
          EVERY paired seed. Either the pairing or the metric's declared
          direction is wrong; this is a criterion defect, not a null result.
      "no_pairs" -- no seed pairs to compare.
      "ok" -- the metric could move in the claimed direction here. A False lift
          under "ok" IS a substrate finding.

    Only "ok" (and "no_pairs", which simply disables the criterion) leaves
    criterion_valid True.
    """
    spec = _c3_metric_spec(metric)
    key = spec["key"]
    lower = spec["direction"] == "lower"
    bound = spec["floor"] if lower else spec["ceiling"]

    pairs: List[Tuple[float, float]] = []
    for g in gap4_rows:
        b = next((x for x in base_rows if x.get("seed") == g.get("seed")), None)
        if b is None:
            continue
        pairs.append((float(g.get(key, 0.0)), float(b.get(key, 0.0))))

    detail: Dict[str, Any] = {
        "metric": metric,
        "metric_key": key,
        "direction": spec["direction"],
        "delta": spec["delta"],
        "n_pairs": len(pairs),
        "pairs": [{"gap4": g, "base": b} for g, b in pairs],
    }

    if not pairs:
        return dict(detail, status="no_pairs", criterion_valid=True, reason="no paired seeds")

    crosses = _rows_cross_gap4_boundary(gap4_rows, base_rows)
    detail["crosses_gap4_boundary"] = crosses
    if crosses and not spec["cross_arm_valid"]:
        return dict(
            detail,
            status="invalid_cross_arm",
            criterion_valid=False,
            reason=(
                "metric '{}' is not comparable between a gap4_operating arm and a "
                "non-gap4 arm (different REEConfig constructors -> different latent "
                "scales); pick a scale-free metric".format(metric)
            ),
        )

    n_degenerate = sum(1 for g, b in pairs if abs(g - b) <= TIER1_C3_DEGENERATE_TOL)
    if n_degenerate >= TIER1_SEEDS_PASS_MIN:
        return dict(
            detail,
            status="degenerate",
            criterion_valid=False,
            reason=(
                "baseline and gap4 '{}' are identical within {} on {}/{} paired seeds "
                "-- the toggled feature does not perturb this metric".format(
                    key, TIER1_C3_DEGENERATE_TOL, n_degenerate, len(pairs)
                )
            ),
        )

    if bound is not None:
        n_sat = sum(1 for _g, b in pairs if abs(b - float(bound)) <= TIER1_C3_DEGENERATE_TOL)
        if n_sat >= TIER1_SEEDS_PASS_MIN:
            return dict(
                detail,
                status="saturated",
                criterion_valid=False,
                reason=(
                    "baseline '{}' already sits at the {} bound {} on {}/{} paired seeds "
                    "-- no headroom in the '{}' direction".format(
                        key,
                        "floor" if lower else "ceiling",
                        bound,
                        n_sat,
                        len(pairs),
                        spec["direction"],
                    )
                ),
            )

    # "Baseline beats gap4" = the baseline sits FURTHER in the claimed direction
    # than the arm that is supposed to be moving there.
    if lower:
        beaten_by_base = sum(1 for g, b in pairs if b < g)
    else:
        beaten_by_base = sum(1 for g, b in pairs if b > g)
    if beaten_by_base == len(pairs):
        return dict(
            detail,
            status="inverted",
            criterion_valid=False,
            reason=(
                "baseline beats gap4 on '{}' in the claimed ('{}') direction on all "
                "{} paired seeds -- the pairing or the declared direction is wrong, "
                "this is a criterion defect not a null result".format(
                    key, spec["direction"], len(pairs)
                )
            ),
        )

    return dict(detail, status="ok", criterion_valid=True, reason="metric can move in the claimed direction")


def evaluate_tier1_cohort(
    rows: List[Dict[str, Any]],
    *,
    gap4_arm_id: str,
    baseline_arm_id: Optional[str] = None,
    c3_lift_metric: str = DEFAULT_C3_LIFT_METRIC,
) -> Dict[str, Any]:
    """PASS when gap4 arm clears C1-C4 in >= TIER1_SEEDS_PASS_MIN seeds and beats baseline on C3 if set.

    c3_lift_metric default is "goal_norm_peak_delta" (substrate-side,
    cross-claim-comparable; chosen post-2026-05-29 V3-EXQ-490g-cohort autopsy
    after approach_commit_rate was shown to ceiling-saturate). SD-037-specific
    scripts should pass c3_lift_metric="override_signal_nonzero_steps" to
    measure the primary PAG override pathway directly. See _c3_lift_compare.
    """
    gap4_rows = [r for r in rows if r.get("arm") == gap4_arm_id]
    base_rows = [r for r in rows if r.get("arm") == baseline_arm_id] if baseline_arm_id else []

    per_seed = [tier1_seed_pass(r) for r in gap4_rows]
    c1 = sum(1 for p in per_seed if p["C1_cue_fires"]) >= TIER1_SEEDS_PASS_MIN
    c2 = sum(1 for p in per_seed if p["C2_dacc_bias"]) >= TIER1_SEEDS_PASS_MIN
    c3_direct = sum(1 for p in per_seed if p["C3_approach_commit"]) >= TIER1_SEEDS_PASS_MIN
    c4 = sum(1 for p in per_seed if p["C4_goal_active"]) >= TIER1_SEEDS_PASS_MIN

    c3_lift = True
    lifts = 0
    capability: Dict[str, Any] = {
        "status": "no_pairs",
        "criterion_valid": True,
        "reason": "no baseline arm configured",
    }
    if baseline_arm_id and base_rows:
        capability = c3_lift_capability(gap4_rows, base_rows, c3_lift_metric)
        for g in gap4_rows:
            seed = g.get("seed")
            b = next((x for x in base_rows if x.get("seed") == seed), None)
            if b is None:
                continue
            if _c3_lift_compare(g, b, c3_lift_metric):
                lifts += 1
        c3_lift = lifts >= TIER1_SEEDS_PASS_MIN

    passed = bool(c1 and c2 and c3_direct and c4 and c3_lift)
    criterion_valid = bool(capability.get("criterion_valid", True))
    acceptance: Dict[str, Any] = {
        "pass": passed,
        "C1_cue_fires": c1,
        "C2_dacc_bias": c2,
        "C3_approach_commit": c3_direct,
        "C3_lift_vs_baseline": c3_lift,
        "C3_lift_count": lifts,
        "C3_lift_metric": c3_lift_metric,
        "C3_lift_status": capability.get("status"),
        "C3_lift_capability": capability,
        "C4_goal_active": c4,
        "criterion_valid": criterion_valid,
        "gap4_arm_id": gap4_arm_id,
        "baseline_arm_id": baseline_arm_id,
    }
    acceptance["recommended_evidence_direction"] = tier1_evidence_direction(acceptance)
    return acceptance


def tier1_evidence_direction(acceptance: Dict[str, Any]) -> str:
    """Map a Tier-1 acceptance block to a claim evidence_direction.

    Use this INSTEAD of the `"supports" if outcome == "PASS" else "weakens"`
    idiom. That idiom is what produced the algorithm-generated `weakens`
    stamps the 2026-05-29 V3-EXQ-490g-cohort autopsy had to correct by hand:
    it cannot distinguish "the substrate did not do the thing" from "the
    acceptance criterion was incapable of registering the thing".

      PASS                      -> "supports"
      FAIL, criterion invalid   -> "non_contributory"   (criterion defect)
      FAIL, criterion valid     -> "weakens"            (substrate finding)

    A criterion-capability failure is a fact about the harness, not about the
    claim, and must never move claim confidence. See c3_lift_capability.
    """
    if acceptance.get("pass"):
        return "supports"
    if not acceptance.get("criterion_valid", True):
        return "non_contributory"
    return "weakens"


def run_seed_arm(
    seed: int,
    arm: ArmSpec,
    *,
    env_kwargs: Optional[Dict[str, Any]] = None,
    warmup_episodes: int = WARMUP_EPISODES_DEFAULT,
    eval_episodes: int = EVAL_EPISODES_DEFAULT,
    steps_per_episode: int = STEPS_PER_EPISODE_DEFAULT,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(seed, env_kwargs)
    env._exq_env_kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    cfg = build_config(env, arm)
    agent = REEAgent(cfg)
    label = f"seed={seed} arm={arm.arm_id}"
    print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
    total_episodes = warmup_episodes + eval_episodes
    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=total_episodes,
    )
    for ep in range(eval_episodes):
        if (ep + 1) == eval_episodes:
            print(
                f"  [train] {label} ep {warmup_episodes + ep + 1}/{total_episodes}",
                flush=True,
            )
    metrics = eval_tier1(
        agent,
        env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        seed=seed,
        arm_label=arm.arm_id,
    )
    # Arm provenance -- c3_lift_capability needs it to detect a comparison that
    # straddles the gap4_operating boundary (different REEConfig constructors,
    # so unnormalised latent quantities are not commensurable). Rows produced
    # before this field existed simply skip that check rather than guess.
    metrics["gap4_operating"] = bool(arm.gap4_operating)
    checks = tier1_seed_pass(metrics)
    passed = all(checks.values())
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    metrics["tier1_checks"] = checks
    metrics["seed_pass"] = passed
    return metrics
