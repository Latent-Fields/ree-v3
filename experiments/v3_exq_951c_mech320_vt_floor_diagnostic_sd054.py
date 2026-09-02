"""
V3-EXQ-951c -- MECH-320 (tonic_vigor_coupling_score_bias) v_t-floor diagnostic:
does the gated vigor scalar sit at its configured v_t_floor because the
internal-state GATES suppress it, or because the ungated EWMA (v_raw) itself
never rises above the floor?

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler, same as 951/812/603n).

RED-TEAM (fable, 2026-09-02): CONTESTED, 4 findings (F1-F4), all fixed prior
to queueing. F1 (agent.py:7138's non-E3-tick select_action short-circuit
meant a select_action-wrapping probe re-read the SAME latched TonicVigor
state every non-E3 tick, pseudo-replicating one observation into many) and
F2 (a one-tick v_raw/last_v_t pairing skew from reading state post-
select_action, after update_score_receipt had already advanced v_raw) are
both eliminated by wrapping tv.compute_score_bias directly instead (see
_VtFloorProbe docstring) -- it fires exactly once per genuine vigor
computation, before update_score_receipt runs. F3 (a pooled gate-caused
fraction can mask cross-seed/cross-arm heterogeneity) is now surfaced via
`heterogeneous_across_cells` + a `_heterogeneous_across_cells` label suffix
when any individual cell disagrees with the pooled classification. F4
(P1 wasn't checked for .aborted like every earlier phase; the pool included
seeds that failed the P1 survival gate) is fixed: an aborted P1 now routes
to _fail like every other phase, and _pool_p2_floor_stats excludes
unsurvived seeds (n_seeds_excluded_unsurvived reported).

BACKGROUND: chip-20260901-mech320-dv-headroom-and-vt-floor (2026-09-01)
found that in every V3-EXQ-951/951a arm/seed/bed/pilot-length tried, the
reported `v_t_mean` came back at EXACTLY 0.0500 -- the driver's explicitly-
configured `v_t_floor` (a V3-EXQ-563-era diagnostic hard floor; NOT the
architecture's normal behaviour). The committed V3-EXQ-951 manifest
(v3_exq_951_mech320_tonic_vigor_authority_sd054_20260830T125526Z_v3.json)
partially reproduces this: seed 44's ARM_0/1/2 all read v_t_mean=0.0500
with gate_product_mean=1.0000 (suggestive of v_raw itself, not the gates,
being the limiting factor for that seed) -- but seeds 42/43 in that SAME
run read v_t_mean in the 1.3-2.6 range, well above floor. That recorded
evidence is (i) on the UNCALMED P2 bed (not the calmed bed
chip-20260901-mech320-noop-margin-dv-substrate's V3-EXQ-951b will use),
(ii) includes seed 44 (the documented reef-config per-seed instability
seed, EXQ-539-540/V3-EXQ-538a -- substituted for seed 45 in that sibling
chip's design), and (iii) only records `gate_product_mean` (an aggregate
mean over the whole run), not a per-tick JOINT test of whether the floor
engaged because of the gates or despite them. GOV-REUSE-1 check: the
decisive per-tick joint classification (see below) is NOT recoverable from
that manifest for the calmed-bed/seed-45 regime this chip needs to
characterise -- this experiment runs fresh.

DECISIVE PER-TICK TEST: TonicVigor.compute_score_bias (ree_core/policy/
tonic_vigor.py:366-369) computes
    v_t = max(v_t_floor, max(0, v_raw) * gate_energy * gate_drive * gate_pe)
On any tick where v_t == v_t_floor (the floor engaged), the floor did so
EITHER because max(0, v_raw) already sat at or below v_t_floor BEFORE any
gating was applied (gates are then provably irrelevant to that tick -- the
floor would have engaged even with all three gates at 1.0: "raw-caused"),
OR because max(0, v_raw) was ABOVE v_t_floor but the gate product pulled
the product back down below it ("gate-caused"). This is a clean, per-tick,
decisive classification requiring only TonicVigor.get_state()'s already-
exposed `v_raw` / `last_v_t` / `last_gate_energy` / `last_gate_drive` /
`last_gate_pe` fields (verified present by direct read of tonic_vigor.py
2026-09-01/02) -- no ree_core change needed. `_VtFloorProbe` below computes
this classification live, per tick, with O(1) memory (no full per-tick
series retained), across both P1 (frozen agent NOT yet -- P1 is live
training, TonicVigor is LIVE throughout per the 951 module docstring's own
confound-avoidance reasoning) and all three frozen-policy P2 arms.

DESIGN: identical curriculum, base REEConfig, and 3-arm frozen-policy P2
toggle to V3-EXQ-951 (Stage0 nursery -> Stage0b consolidation -> P0 ->
Stage-H hazard-avoidance -> P1 foraging -> P2 x3 arms on the same trained
weights), with two changes:
  (1) CALMED P2 measurement bed (per chip-20260901-mech320-noop-margin-dv-
      substrate's design, reproduced here so this script does not depend
      on that chip's own driver landing first): scaffold_p2_num_hazards=0
      (951: 4), scaffold_p2_proximity_harm_scale=0.0 (951: 0.1),
      scaffold_p2_hazard_food_attraction_guard=0.0 (951: 0.3),
      scaffold_p2_num_resources=5 (unchanged). Pure scaffold-config change;
      the TRAINING curriculum is untouched.
  (2) SEEDS = [42, 43, 45], not [42, 43, 44] -- seed 44 is the documented
      recurring per-seed instability seed on reef-config envs (EXQ-539-540,
      V3-EXQ-538a); the curriculum keeps scaffold_p1_reef_spawn_hold_fraction
      > 0 so the reef config is live regardless of the P2 calming above.
This experiment does NOT change MECH-320's config, does NOT propose a
substrate change, and does NOT re-test the C1 selection-authority lift --
it is read-only instrumentation on top of already-exposed TonicVigor state,
run purely to characterise WHY v_t saturates at its floor in this regime.

claim_ids = ["MECH-320"] (the mechanism being characterised).
experiment_purpose = "diagnostic" -- excluded from governance confidence/
conflict scoring; this run does not move MECH-320's evidence direction.

NOT A SUPERSESSION of V3-EXQ-951/951a/951b -- different question (why v_t
sits at floor, not whether MECH-320 has selection authority).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_951c_mech320_vt_floor_diagnostic_sd054.py
or:
  /opt/local/bin/python3 experiments/v3_exq_951c_mech320_vt_floor_diagnostic_sd054.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_951c_mech320_vt_floor_diagnostic_sd054"
QUEUE_ID = "V3-EXQ-951c"
CLAIM_IDS: List[str] = ["MECH-320"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 45]  # seed 44 excluded: documented reef-config instability (EXQ-539-540)
MIN_FRACTION = 2.0 / 3.0  # matches 951/603n/812's own validated seed-majority convention

# ---- Curriculum config: mirror V3-EXQ-951 exactly (the landed readiness config) ----
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

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

# ---- 951c-specific: calmed P2 measurement bed (per chip-20260901-mech320-
# noop-margin-dv-substrate's design; reproduced here, not depended on) ----
P2_CALM_NUM_HAZARDS = 0                    # 951: 4 (scaffold_p2_num_hazards default)
P2_CALM_PROXIMITY_HARM_SCALE = 0.0         # 951: 0.1 (scaffold_p2_proximity_harm_scale default)
P2_CALM_HFA_GUARD = 0.0                    # 951: 0.3 (P2_HFA_GUARD)
P2_CALM_NUM_RESOURCES = 5                  # unchanged from 951 (scaffold_p2_num_resources default)

# ---- 951-specific: MECH-320 vigor + selection-authority config (unchanged from 951) ----
ACTION_DIM = 5  # CausalGridWorldV2.ACTIONS: 0=up,1=down,2=left,3=right,4=noop(stay)
NOOP_CLASS = 4  # see 951 module docstring: NOT the TonicVigorConfig library default (0)
V_T_FLOOR = 0.05  # forced-vigor probe (V3-EXQ-549 fix), matches 951/624-series
VIGOR_W_ACTION_TRAINED = 0.1
VIGOR_W_PASSIVE_TRAINED = 0.1
VIGOR_HALF_LIFE = 100.0
USE_MODULATORY_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

ANCHOR_REACHABILITY_EXEMPT = (
    "reached_p2_alive mirrors 951/812/603n's already-validated P1-survival precondition "
    "(repeatedly demonstrated reachable on this exact curriculum); "
    "sufficient_floor_hit_sample was directly demonstrated reachable, POST the red-team "
    "fable F1 fix (probe wraps tv.compute_score_bias, counting only genuine E3-tick vigor "
    "computations -- see _VtFloorProbe docstring), in this script's own --dry-run smoke "
    "test (1 seed: sufficient_floor_hit_sample met, label=vt_floor_driven_by_low_v_raw, "
    "not the insufficient-sample route) -- both are reachable by construction, not a "
    "hand-tuned degeneracy definition."
)

# ---- 951c-specific diagnostic thresholds ----
MIN_FLOOR_HIT_TICKS = 30  # minimum floor-engaged ticks (pooled) before trusting a classification
GATE_CAUSED_LOW = 0.1     # pooled floor_hit_gate_caused_frac <= this -> "raw_caused" (finding b)
GATE_CAUSED_HIGH = 0.9    # pooled floor_hit_gate_caused_frac >= this -> "gate_caused" (finding a)

ARM_LABELS = [
    "ARM_0_baseline",
    "ARM_1_vigor_additive",
    "ARM_2_vigor_multiplicative",
]
PHASE_LABELS = ["P1_training"] + ARM_LABELS

CONFIG_SLICE = {
    "scaffold_cfg": "see _make_scaffold_cfg (curriculum budgets + landed levers, mirrors 951/603n/812)",
    "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
    "seed_gain": SEED_GAIN, "seed_benefit_threshold": SEED_BENEFIT_THRESHOLD,
    "seed_drive_floor": SEED_DRIVE_FLOOR, "cue_recall_gain": CUE_RECALL_GAIN,
    "harm_pathway_lr": HARM_PATHWAY_LR,
    "v_t_floor": V_T_FLOOR,
    "vigor_w_action_trained": VIGOR_W_ACTION_TRAINED,
    "vigor_w_passive_trained": VIGOR_W_PASSIVE_TRAINED,
    "use_modulatory_selection_authority": USE_MODULATORY_AUTHORITY,
    "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
    "p2_calm_num_hazards": P2_CALM_NUM_HAZARDS,
    "p2_calm_proximity_harm_scale": P2_CALM_PROXIMITY_HARM_SCALE,
    "p2_calm_hfa_guard": P2_CALM_HFA_GUARD,
    "p2_calm_num_resources": P2_CALM_NUM_RESOURCES,
}


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        # 951c calmed P2 measurement bed (training curriculum untouched):
        scaffold_p2_num_hazards=P2_CALM_NUM_HAZARDS,
        scaffold_p2_proximity_harm_scale=P2_CALM_PROXIMITY_HARM_SCALE,
        scaffold_p2_hazard_food_attraction_guard=P2_CALM_HFA_GUARD,
        scaffold_p2_num_resources=P2_CALM_NUM_RESOURCES,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """Base REEConfig mirroring V3-EXQ-951's landed-substrate config exactly.
    See 951's own module docstring for the full rationale -- unchanged here.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
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
        mech295_liking_to_approach_cue_gain=0.5,
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
        use_tonic_vigor=True,
        tonic_vigor_v_t_floor=V_T_FLOOR,
        tonic_vigor_noop_class=NOOP_CLASS,
        use_modulatory_selection_authority=USE_MODULATORY_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


FLOOR_EPS = 1e-9


class _VtFloorProbe:
    """Wraps agent.tonic_vigor.compute_score_bias (NOT agent.select_action)
    to record TonicVigor.get_state() diagnostics on every GENUINE call, with
    O(1) memory per phase (running sum/min/max, no full per-tick series
    retained).

    WHY compute_score_bias, not select_action (red-team fable, 2026-09-02,
    findings F1+F2 -- see queue entry note): agent.py:7138
    (`if not ticks["e3_tick"] and self._last_action is not None: ... return
    action`) short-circuits select_action on every NON-E3 tick, well before
    compute_score_bias (agent.py:8249) or update_score_receipt
    (agent.py:9390) execute. A probe wrapping select_action therefore
    re-reads the SAME latched get_state() on every non-E3 tick between real
    E3 deliberations -- pseudo-replicating one observation into many and
    silently inflating n_floor_hit past MIN_FLOOR_HIT_TICKS without any
    independent evidence (the exact E3-latch hazard the /queue-experiment
    skill's "Sample-size integrity" section warns about for `last_*`
    diagnostics). Wrapping compute_score_bias directly means this probe
    fires exactly once per genuine vigor computation (never on the
    short-circuited path, which never reaches agent.py:8220 at all) and
    reads state BEFORE update_score_receipt (a separate, later call in the
    same select_action invocation) has advanced v_raw -- eliminating both
    the pseudo-replication (F1) and a one-tick v_raw/last_v_t pairing skew
    that reading post-select_action would otherwise introduce (F2).

    Decisive classification (see module docstring): on any tick where
    last_v_t == v_t_floor (the floor engaged), test whether
    max(0, v_raw) already sat at or below v_t_floor BEFORE gating
    ("raw_caused" -- gates provably irrelevant to that tick) or whether it
    was ABOVE v_t_floor and the gate product pulled it back down
    ("gate_caused"). Computed live per tick; no full series needed.
    """

    def __init__(self, agent: REEAgent, v_t_floor: float) -> None:
        self.agent = agent
        self.v_t_floor = float(v_t_floor)
        self._installed = False
        self._active_phase: Optional[str] = None
        self._stats: Dict[str, Dict[str, Any]] = {}

    def install(self) -> None:
        if self._installed:
            return
        tv = getattr(self.agent, "tonic_vigor", None)
        if tv is None:
            self._installed = True  # MECH-320 not active on this agent; nothing to wrap.
            return
        orig_compute_score_bias = tv.compute_score_bias

        def wrapped_compute_score_bias(*args, **kwargs):
            bias = orig_compute_score_bias(*args, **kwargs)
            phase = self._active_phase
            # agent.py:8255 always passes simulation_mode=False at its sole call
            # site; still honour the flag defensively (MECH-094 pattern).
            sim = bool(kwargs.get("simulation_mode", False))
            if not sim and len(args) >= 6:
                sim = bool(args[5])
            if phase is not None and not sim:
                try:
                    st = tv.get_state()
                    self._record(phase, st)
                except Exception:
                    pass
            return bias

        tv.compute_score_bias = wrapped_compute_score_bias
        self._installed = True

    def set_phase(self, phase_label: Optional[str]) -> None:
        self._active_phase = phase_label
        if phase_label is not None:
            self._stats.setdefault(phase_label, self._new_bucket())

    @staticmethod
    def _new_bucket() -> Dict[str, Any]:
        def _series() -> Dict[str, float]:
            return {"sum": 0.0, "min": float("inf"), "max": float("-inf")}

        return {
            "n_ticks": 0,
            "n_floor_hit": 0,
            "n_floor_hit_gate_caused": 0,
            "n_floor_hit_raw_caused": 0,
            "v_raw": _series(),
            "gate_energy": _series(),
            "gate_drive": _series(),
            "gate_pe": _series(),
            "v_t": _series(),
        }

    def _record(self, phase: str, st: Dict[str, Any]) -> None:
        b = self._stats[phase]
        v_raw = float(st["v_raw"])
        gate_e = float(st["last_gate_energy"])
        gate_d = float(st["last_gate_drive"])
        gate_p = float(st["last_gate_pe"])
        v_t = float(st["last_v_t"])
        b["n_ticks"] += 1
        for key, val in (
            ("v_raw", v_raw), ("gate_energy", gate_e), ("gate_drive", gate_d),
            ("gate_pe", gate_p), ("v_t", v_t),
        ):
            s = b[key]
            s["sum"] += val
            if val < s["min"]:
                s["min"] = val
            if val > s["max"]:
                s["max"] = val
        if abs(v_t - self.v_t_floor) < FLOOR_EPS:
            b["n_floor_hit"] += 1
            raw_component = max(0.0, v_raw)
            if raw_component > self.v_t_floor:
                b["n_floor_hit_gate_caused"] += 1
            else:
                b["n_floor_hit_raw_caused"] += 1

    def summary(self, phase: str) -> Dict[str, Any]:
        b = self._stats.get(phase)
        if b is None or b["n_ticks"] == 0:
            return {"n_ticks": 0}
        n = b["n_ticks"]
        out: Dict[str, Any] = {
            "n_ticks": n,
            "n_floor_hit": b["n_floor_hit"],
            "floor_hit_frac": b["n_floor_hit"] / n,
            "n_floor_hit_gate_caused": b["n_floor_hit_gate_caused"],
            "n_floor_hit_raw_caused": b["n_floor_hit_raw_caused"],
            "floor_hit_gate_caused_frac": (
                b["n_floor_hit_gate_caused"] / b["n_floor_hit"] if b["n_floor_hit"] > 0 else None
            ),
        }
        for key in ("v_raw", "gate_energy", "gate_drive", "gate_pe", "v_t"):
            s = b[key]
            out[key] = {"mean": s["sum"] / n, "min": s["min"], "max": s["max"]}
        return out


class _ActionDensityProbe:
    """Wraps agent.select_action to count P2 ticks where the committed action
    is non-noop (action_density). Mirrors V3-EXQ-951's own probe -- kept as
    secondary/context evidence (a saturated action_density is expected here,
    per the sibling noop-margin-dv-substrate chip; not this script's DV).
    """

    def __init__(self, agent: REEAgent) -> None:
        self.agent = agent
        self.n_ticks = 0
        self.n_nonnoop_ticks = 0
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        orig_select_action = self.agent.select_action

        def wrapped_select_action(*args, **kwargs):
            action = orig_select_action(*args, **kwargs)
            try:
                action_idx = int(action.argmax(dim=-1).item())
                self.n_ticks += 1
                if action_idx != NOOP_CLASS:
                    self.n_nonnoop_ticks += 1
            except Exception:
                pass
            return action

        self.agent.select_action = wrapped_select_action
        self._installed = True

    def summary(self) -> Dict[str, Any]:
        n = self.n_ticks
        return {
            "n_ticks": n,
            "action_density": (self.n_nonnoop_ticks / n) if n > 0 else 0.0,
        }


def _run_p2_arm(
    scheduler: ScaffoldedSD054OnboardingScheduler,
    agent: REEAgent,
    device: torch.device,
    seed: int,
    arm_label: str,
    w_action: float,
    w_passive: float,
    form: str,
    vt_probe: _VtFloorProbe,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run one frozen-policy P2 pass at a fixed vigor-weight/form setting,
    instrumented for action_density AND the vt-floor probe. reset_all_rng(seed)
    is called first so every arm of this seed sees an identical env
    realisation sequence (matches V3-EXQ-951/812's arm-toggle idiom).
    """
    reset_all_rng(seed)
    tv = getattr(agent, "tonic_vigor", None)
    if tv is not None:
        tv.config.w_action = float(w_action)
        tv.config.w_passive = float(w_passive)
        tv.config.form = str(form)

    density_probe = _ActionDensityProbe(agent)
    density_probe.install()
    vt_probe.set_phase(arm_label)
    agent.eval()
    scheduler.run_p2(agent, device)
    vt_probe.set_phase(None)
    density_summary = density_probe.summary()
    vt_summary = vt_probe.summary(arm_label)

    row: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "w_action": float(w_action),
        "w_passive": float(w_passive),
        "form": str(form),
        **density_summary,
        "vt_floor_diagnostics": vt_summary,
    }
    gc_frac = vt_summary.get("floor_hit_gate_caused_frac")
    print(
        f"  [p2_arm] seed={seed} arm={arm_label} w_action={w_action:.2f}"
        f" w_passive={w_passive:.2f} form={form}"
        f" action_density={density_summary['action_density']:.4f}"
        f" v_t_mean={vt_summary.get('v_t', {}).get('mean', 0.0):.4f}"
        f" floor_hit_frac={vt_summary.get('floor_hit_frac', 0.0):.4f}"
        f" gate_caused_frac={gc_frac if gc_frac is not None else 'n/a'}",
        flush=True,
    )

    with arm_cell(
        seed,
        config_slice=CONFIG_SLICE,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=[
            "shared_trained_agent_eval_time_toggle_not_independently_trained",
        ],
        do_reset=False,  # RNG already reset above (same seed, matched-comparison purpose)
    ) as cell:
        cell.stamp(row)
    return row


_ZG = ZGoalStreamAccumulator()


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    from scaffolded_sd054_onboarding import _build_env
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    vt_probe = _VtFloorProbe(agent, V_T_FLOOR)
    vt_probe.install()

    print(f"Seed {seed} Condition MECH320_VT_FLOOR_DIAGNOSTIC_SD054", flush=True)

    def _fail(stage: str, reason: str) -> Dict[str, Any]:
        print(f"verdict: FAIL seed={seed} aborted_at={stage} reason={reason}", flush=True)
        _ZG.observe(agent)
        return {
            "seed": seed, "aborted_at": stage, "abort_reason": reason,
            "arms": [], "g1_p1_survival": False, "seed_pass": False,
            "p1_vt_diagnostics": {"n_ticks": 0},
        }

    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}", flush=True)
    if s0.aborted:
        return _fail("stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}", flush=True)
    if s0b.aborted:
        return _fail("stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        return _fail("p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        return _fail("hazard", hz.abort_reason)

    # ---- P1 foraging: TonicVigor is LIVE throughout training (per 951's own
    # confound-avoidance reasoning) -- instrument this phase too. ----
    vt_probe.set_phase("P1_training")
    p1 = scheduler.run_p1(agent, device)
    vt_probe.set_phase(None)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)
    # Defensive: mirrors the .aborted check on every earlier phase (red-team
    # fable F4a). P1OnboardingResult.aborted is structurally always False in
    # the enabled path for this curriculum (only master_switch_off sets it),
    # but an aborted-P1 seed feeding degenerate P2 vigor data into the pooled
    # classification is exactly the failure mode this diagnostic must not
    # silently absorb.
    if p1.aborted:
        return _fail("p1", p1.abort_reason)
    p1_vt_diagnostics = vt_probe.summary("P1_training")

    # ---- Frozen-policy P2: run ALL THREE arms on the SAME trained agent ----
    arm0 = _run_p2_arm(scheduler, agent, device, seed, "ARM_0_baseline",
                        0.0, 0.0, "additive", vt_probe, dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_arm0 seed={seed} ep {done}/{total_eps}", flush=True)
    arm1 = _run_p2_arm(scheduler, agent, device, seed, "ARM_1_vigor_additive",
                        VIGOR_W_ACTION_TRAINED, VIGOR_W_PASSIVE_TRAINED, "additive",
                        vt_probe, dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_arm1 seed={seed} ep {done}/{total_eps}", flush=True)
    arm2 = _run_p2_arm(scheduler, agent, device, seed, "ARM_2_vigor_multiplicative",
                        VIGOR_W_ACTION_TRAINED, VIGOR_W_PASSIVE_TRAINED, "multiplicative",
                        vt_probe, dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_arm2 seed={seed} ep {done}/{total_eps}", flush=True)

    g1 = bool(p1.survival_gate_passed)
    seed_pass = bool(g1)  # harness completed end-to-end; this is a diagnostic, no claim verdict
    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} g1={g1}",
        flush=True,
    )

    # Restore trained-with vigor weights on the live agent before moving on
    # (hygiene only -- the agent is not reused past this seed).
    tv = getattr(agent, "tonic_vigor", None)
    if tv is not None:
        tv.config.w_action = VIGOR_W_ACTION_TRAINED
        tv.config.w_passive = VIGOR_W_PASSIVE_TRAINED
        tv.config.form = "additive"

    _ZG.observe(agent)
    return {
        "seed": seed, "aborted_at": None, "abort_reason": "",
        "g1_p1_survival": g1,
        "arms": [arm0, arm1, arm2],
        "seed_pass": seed_pass,
        "p1_vt_diagnostics": p1_vt_diagnostics,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _pool_p2_floor_stats(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool floor-hit classification counts across P2 arms of SURVIVED seeds
    only (g1_p1_survival=True -- red-team fable F4b). A seed that failed the
    P1 survival gate still runs its P2 arms (harness completes end-to-end,
    matching V3-EXQ-951's own convention), but its vigor dynamics reflect a
    moribund/depleted trajectory (energy-gate suppression is expected there
    by construction, tonic_vigor.py:442-455) rather than the "reached P2
    alive" regime `reached_p2_alive` certifies -- pooling it in would let a
    dying seed's gate-caused hits leak into a pool presented as
    survivors-only. Also reports per-cell (seed, arm) heterogeneity (F3):
    the pooled fraction can mask a mix of seeds/arms individually landing
    on OPPOSITE classifications.
    """
    n_ticks = 0
    n_floor_hit = 0
    n_gate_caused = 0
    n_raw_caused = 0
    per_cell_fracs: List[Dict[str, Any]] = []
    n_seeds_excluded_unsurvived = 0
    for r in per_seed:
        if not r.get("g1_p1_survival", False):
            if r.get("arms"):
                n_seeds_excluded_unsurvived += 1
            continue
        for a in r.get("arms", []):
            vt = a.get("vt_floor_diagnostics", {})
            n_ticks += int(vt.get("n_ticks", 0))
            n_floor_hit += int(vt.get("n_floor_hit", 0))
            n_gate_caused += int(vt.get("n_floor_hit_gate_caused", 0))
            n_raw_caused += int(vt.get("n_floor_hit_raw_caused", 0))
            cell_hit = int(vt.get("n_floor_hit", 0))
            if cell_hit > 0:
                per_cell_fracs.append({
                    "seed": r.get("seed"), "arm": a.get("arm"),
                    "n_floor_hit": cell_hit,
                    "gate_caused_frac": vt.get("floor_hit_gate_caused_frac"),
                })
    gate_caused_frac = (n_gate_caused / n_floor_hit) if n_floor_hit > 0 else None
    # Heterogeneity: does every individual cell land in the SAME band as the
    # pooled value, or does the pool average over opposite-classification
    # cells (which would make a "vt_floor_driven_by_X" label misleading)?
    heterogeneous = False
    if gate_caused_frac is not None and per_cell_fracs:
        if gate_caused_frac <= GATE_CAUSED_LOW:
            heterogeneous = any(
                c["gate_caused_frac"] is not None and c["gate_caused_frac"] > GATE_CAUSED_LOW
                for c in per_cell_fracs
            )
        elif gate_caused_frac >= GATE_CAUSED_HIGH:
            heterogeneous = any(
                c["gate_caused_frac"] is not None and c["gate_caused_frac"] < GATE_CAUSED_HIGH
                for c in per_cell_fracs
            )
        # else: pooled value is already in the mixed band -- not "masking" anything.
    return {
        "n_ticks": n_ticks,
        "n_floor_hit": n_floor_hit,
        "floor_hit_frac": (n_floor_hit / n_ticks) if n_ticks > 0 else 0.0,
        "n_floor_hit_gate_caused": n_gate_caused,
        "n_floor_hit_raw_caused": n_raw_caused,
        "floor_hit_gate_caused_frac": gate_caused_frac,
        "n_seeds_excluded_unsurvived": n_seeds_excluded_unsurvived,
        "per_cell_gate_caused_fracs": per_cell_fracs,
        "heterogeneous_across_cells": heterogeneous,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 3 * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + 3 * P2_BUDGET
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n_seeds = len(per_seed)
    g1_frac = _frac([r.get("g1_p1_survival", False) for r in per_seed])
    reached_p2_alive = bool(g1_frac >= MIN_FRACTION)

    pooled_p2 = _pool_p2_floor_stats(per_seed)
    sufficient_sample = bool(pooled_p2["n_floor_hit"] >= MIN_FLOOR_HIT_TICKS)

    gc_frac = pooled_p2["floor_hit_gate_caused_frac"]
    if not reached_p2_alive:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif not sufficient_sample:
        # The floor rarely (or never) engaged in this regime/seed set -- itself
        # an informative negative finding (contradicts the DV-headroom chip's
        # premise for THIS calmed bed + seed set), but too few floor-hit ticks
        # to trust a gate-vs-raw classification.
        outcome = "PASS"
        label = "floor_rarely_engaged_insufficient_sample_for_classification"
    elif gc_frac is not None and gc_frac <= GATE_CAUSED_LOW:
        outcome = "PASS"
        label = "vt_floor_driven_by_low_v_raw"  # finding (b)
    elif gc_frac is not None and gc_frac >= GATE_CAUSED_HIGH:
        outcome = "PASS"
        label = "vt_floor_driven_by_gate_suppression"  # finding (a)
    else:
        outcome = "PASS"
        label = "vt_floor_mixed_causes"

    # Red-team fable F3: a pooled label can mask cross-seed/cross-arm
    # heterogeneity (e.g. one always-floored raw_caused seed swamping a
    # rarely-floored gate_caused seed into a pooled "low_v_raw" reading).
    # Downgrade a decisive-looking label to the explicit heterogeneous
    # variant when any individual (seed, arm) cell disagrees with the
    # pooled classification -- never silently drop the disagreement.
    if pooled_p2.get("heterogeneous_across_cells") and label in (
        "vt_floor_driven_by_low_v_raw", "vt_floor_driven_by_gate_suppression",
    ):
        label = f"{label}_heterogeneous_across_cells"

    print(
        f"[{EXPERIMENT_TYPE}] g1_frac={g1_frac:.2f} pooled_floor_hit_frac="
        f"{pooled_p2['floor_hit_frac']:.4f} pooled_gate_caused_frac="
        f"{gc_frac if gc_frac is not None else 'n/a'} "
        f"heterogeneous={pooled_p2.get('heterogeneous_across_cells')} "
        f"-> outcome={outcome} label={label}",
        flush=True,
    )

    preconditions = [
        {
            "name": "reached_p2_alive", "kind": "readiness",
            "description": "P1 survival >= 2/3 seeds so the agent reaches P2 alive "
                            "(same precondition shape 951/603n/812 already validate on "
                            "this exact curriculum).",
            "control": "P1 survival gate (median episode length last window >= "
                       "scaffold_p1_survival_gate_steps).",
            "measured": float(g1_frac), "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(reached_p2_alive),
        },
        {
            "name": "sufficient_floor_hit_sample", "kind": "readiness",
            "description": "Pooled P2 floor-hit tick count (across all arms/seeds) is "
                            "large enough to trust a gate-caused-vs-raw-caused "
                            "classification, rather than a handful of ticks.",
            "control": "n_floor_hit ticks pooled across ARM_0/1/2 x all valid seeds.",
            "measured": float(pooled_p2["n_floor_hit"]), "threshold": float(MIN_FLOOR_HIT_TICKS),
            "direction": "lower",
            "met": bool(sufficient_sample),
        },
    ]
    criteria_non_degenerate = {
        "gate_caused_classification": bool(sufficient_sample and gc_frac is not None),
    }
    criteria = [
        {
            "name": "gate_caused_classification", "load_bearing": True,
            "passed": bool(sufficient_sample),
        },
    ]

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",  # diagnostic: does not move MECH-320's evidence
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "pooled_p2_floor_diagnostics": pooled_p2,
        },
        "gate_summary": {
            "g1_p1_survival_frac": g1_frac,
            "reached_p2_alive": reached_p2_alive,
            "sufficient_floor_hit_sample": sufficient_sample,
            "min_floor_hit_ticks": MIN_FLOOR_HIT_TICKS,
            "gate_caused_low_threshold": GATE_CAUSED_LOW,
            "gate_caused_high_threshold": GATE_CAUSED_HIGH,
        },
        "per_seed": per_seed,
        "arm_results": [a for r in per_seed for a in r.get("arms", [])],
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1, MECH-320 vigor ON throughout training; frozen-policy P2 run "
                     "THREE times per seed toggling only the live TonicVigor.config object, "
                     "identical to V3-EXQ-951 EXCEPT: calmed P2 measurement bed and seeds "
                     "[42,43,45] instead of [42,43,44]) -- per-tick v_raw/gate diagnostics added "
                     "via _VtFloorProbe, no ree_core change.",
        "predecessor": "V3-EXQ-951 (evidence run; not superseded -- different question, this "
                       "run does not re-test C1 selection-authority). Spawned by "
                       "chip-20260901-mech320-dv-headroom-and-vt-floor's v_t-floor finding.",
        "design_note": "Identical to V3-EXQ-951's 3-arm frozen-policy P2 toggle on the same "
                       "per-seed trained agent, plus: (1) calmed P2 bed "
                       "(scaffold_p2_num_hazards=0, scaffold_p2_proximity_harm_scale=0.0, "
                       "scaffold_p2_hazard_food_attraction_guard=0.0), matching the design "
                       "chip-20260901-mech320-noop-margin-dv-substrate specified for V3-EXQ-951b "
                       "(reproduced here independently); (2) seeds [42,43,45] not [42,43,44]; "
                       "(3) _VtFloorProbe recording v_raw/gate_energy/gate_drive/gate_pe/v_t "
                       "per tick across P1 training and all 3 P2 arms, with a live per-tick "
                       "gate-caused-vs-raw-caused classification on every floor-engaged tick.",
        "red_team_review": "red-team (fable): CONTESTED, 4 findings (F1-F4), all fixed "
                           "prior to queueing -- see module docstring RED-TEAM section "
                           "and queue entry note for detail.",
        "gov_reuse_1_check": "decisive readout (per-tick joint v_raw-vs-gate floor-engagement "
                             "classification, calmed-bed/seed-45 regime) checked against "
                             "v3_exq_951_mech320_tonic_vigor_authority_sd054_20260830T125526Z_v3"
                             ".json (uncalmed bed, seeds 42/43/44): that manifest's seed-44 "
                             "arms show v_t_mean=0.0500 with gate_product_mean=1.0000 "
                             "(suggestive but not a per-tick joint test), while seeds 42/43 "
                             "show v_t_mean 1.3-2.6 (floor NOT engaged) -- not recoverable for "
                             "the calmed-bed/seed-45 regime this chip needs; not derivable "
                             "post-hoc since the source manifest records no per-tick series. Run.",
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget_per_arm": P2_BUDGET, "n_p2_arms": 3,
            "train_steps": TRAIN_STEPS,
            "v_t_floor": V_T_FLOOR,
            "vigor_w_action_trained": VIGOR_W_ACTION_TRAINED,
            "vigor_w_passive_trained": VIGOR_W_PASSIVE_TRAINED,
            "use_modulatory_selection_authority": USE_MODULATORY_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "noop_class": NOOP_CLASS,
            "p2_calm_num_hazards": P2_CALM_NUM_HAZARDS,
            "p2_calm_proximity_harm_scale": P2_CALM_PROXIMITY_HARM_SCALE,
            "p2_calm_hfa_guard": P2_CALM_HFA_GUARD,
            "p2_calm_num_resources": P2_CALM_NUM_RESOURCES,
        },
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=CONFIG_SLICE,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
            dry_run=args.dry_run,
        )
