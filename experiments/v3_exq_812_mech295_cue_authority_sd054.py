"""
V3-EXQ-812 -- MECH-295 Layer-C cue-authority diagnostic, redesigned onto the
NOW-READY scaffolded_sd054_onboarding substrate (successor to the stale
2026-06-03 "V3-EXQ-631" proposal, which targeted the gap4 regime).

BACKGROUND (why this exists, and why on THIS substrate):
The 2026-06-03 audit (evidence/planning/cue_system_goal_stream_audit_2026-06-03.md)
found that across the MECH-295 cohort (490i/490j/493/490k), the cue's *firing*
(bridge_cue_fires>0, score-bias reaching dacc_score_bias) was proven, but its
*behavioural authority* -- does the cue bias actually change the SELECTED
candidate's proximity, not just fire and get added into a scoring blend that
approach_commit_rate (a saturated/degenerate metric per 490j) cannot discriminate
-- was never measured. On 2026-06-09 governance closed goal_pipeline:GAP-4 by
re-scoping MECH-295 from a necessity to a modulation claim and explicitly
declined a V3-EXQ-490L-style successor, noting the authority retest is "OPTIONAL,
non-GAP-blocking... gated on the scaffolded_sd054_onboarding substrate (owned
elsewhere, in flight via the 603-line Stage-H programme)" (claims.yaml MECH-295
evidence_quality_note). That programme landed: V3-EXQ-603n (2026-06-10) cleared
the full four-leg readiness gate (G0 Stage-0 positive control, G1 P1 survival,
G2 P2 contact, G3 P2 consumption-gated z_goal>0.4) on >=2/3 seeds, and
substrate_queue scaffolded_sd054_onboarding has read ready:true since 2026-06-11
(reaffirmed 2026-07-10, WORKSPACE_STATE.md). No run since 2026-06-09 has
re-exercised the MECH-295 bridge on this now-ready substrate (last bridge-signal
read was 490k, 2026-06-04, pre-603n).

WHAT CHANGED VS THE ORIGINAL 631 PROPOSAL (gap4 regime): that design forced
z_goal via drive_floor=0.9 because gap4 was the only regime where z_goal formed
reliably at the time, and included a THIRD arm (forced supra-threshold z_goal
positive control) to guarantee a non-degenerate candidate-proximity surface on
seeds where natural formation was weak. On scaffolded_sd054_onboarding, z_goal
now forms reliably via the full curriculum (G3 >=2/3 seeds, 603n), so the forced
positive-control arm is dropped: 2 arms suffice (severed vs intact), each
seed's OWN P2 stage supplies the non-degenerate surface, and C0 below explicitly
checks this rather than assuming it.

DESIGN: train ONE agent per seed through the full landed curriculum (Stage0 ->
Stage0b -> P0 -> Stage-H -> P1, bridge ON throughout -- the landed default
trajectory; severing the bridge DURING TRAINING would confound "does authority
exist in the trained policy" with "what did a differently-trained policy learn
to compensate"). At P2 (frozen-policy measurement, agent.eval(), no training),
run TWO eval passes on the SAME trained weights: cue_recall_gain unrelated;
here we toggle mech295_bridge.config.liking_to_approach_cue_gain between its
landed default (ARM_CUE_ON, matches training) and 0.0 (ARM_CUE_OFF, the
"severed bridge" arm per mech295_liking_bridge.py's own falsifiability-test
docstring). RNG is fully reset (same seed) before EACH P2 pass so both arms see
the identical env realisation sequence -- the controlled-comparison design this
diagnostic needs, and incidentally the multi-arm skill's mandatory RNG-reset
obligation (arm_cell). The two arms structurally share the trained agent (mutable
state), so per-cell fingerprints are emitted (required) but marked
reuse_eligible=False via extra_ineligible_reasons (not independently retrainable).

NEW TELEMETRY (the 490j/631 gap this closes): existing substrate instrumentation
already covers PART of this (agent._last_score_bias_decomp, landed via MECH-451,
gives per-channel [dacc/lateral_pfc/ofc/mech295_liking/curiosity/tonic_vigor]
mean/std/RANGE -- this closes the V3-EXQ-643-class magnitude-vs-range trap for
free). What is still missing and this script adds via a two-method monkeypatch
(mech295_bridge.compute_approach_cue_score_bias to capture per-tick
candidate_proximities + the returned bias; agent.select_action, wrapping the
whole call, to read agent._last_e3_selection_result.selected_index for the SAME
tick immediately after orig returns) is the Layer-C3/L4 readout 490j lacked:
SELECTED-candidate proximity vs pool-mean proximity, per cue-firing tick.

MANDATORY INVALID_HARNESS / non_contributory branch: if goal_state is inactive,
or candidate_proximity has zero variance, in > 1/3 seeds, route to
non_contributory (INVALID_HARNESS) -- never a FAIL that could misread as
MECH-295 falsification (the 603e / 643 lesson).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler, same as its parent 603n).

experiment_purpose: diagnostic
claim_ids: []  (isolates cue authority; does NOT adjudicate MECH-295 -- per the
  490k/631-proposal precedent, results route to /governance for a claim-bearing
  follow-up, not self-weighted here)
predecessor (not supersedes): V3-EXQ-490j (weakens/evidence on the necessity
  claim, gap4 regime) and V3-EXQ-490k (diagnostic, pre-603n substrate, could not
  test authority: mech295_bias_range_mean=0.0). 812 is the Layer-C authority
  measurement neither could complete, run on the substrate governance actually
  gated it to.
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
    _build_env,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_812_mech295_cue_authority_sd054"
QUEUE_ID = "V3-EXQ-812"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "harm_pathway_discriminative / reached_p2_alive reuse V3-EXQ-603n's already-"
    "landed, already-validated precondition shape verbatim (same curriculum, same "
    "floors) -- reachability was established by 603n's own PASS. "
    "candidate_proximity_evaluable is a NEW anchor but its control is the wrapped "
    "mech295_bridge call itself on the ARM_CUE_ON positive-gain arm, not a "
    "hand-tuned narrower proxy -- it IS the degeneracy definition C3 depends on."
)

SEEDS = [42, 43, 44]

# ---- Curriculum config: mirror 603n exactly (the landed readiness config) ----
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
P2_HFA_GUARD = 0.3
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

STAGE0_POSITIVE_CONTROL_FLOOR = 0.3
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

HARM_EVAL_RANGE_FLOOR = 0.005
HARM_TRAIN_STEPS_FLOOR = 1.0

# ---- 812-specific gates (Layer-C authority) ----
# Landed default cue-side gain (MECH295LikingBridgeConfig.liking_to_approach_cue_gain).
CUE_GAIN_NATURAL = 0.5
CUE_GAIN_SEVERED = 0.0
# C0: candidate_proximity must show real cross-candidate spread on the positive
# (ON) arm for the comparison to be meaningful at all (same-statistic-as-criterion
# rule: C3 below gates on a PROXIMITY DELTA, so C0 must assert proximity RANGE,
# not a magnitude proxy -- the V3-EXQ-643 lesson).
PROXIMITY_RANGE_FLOOR = 1e-4
MIN_CUE_FIRE_TICKS = 3  # per seed, in ARM_CUE_ON, to trust the pooled read
# C3: selected-candidate proximity must exceed the pool mean by this margin,
# averaged over cue-firing ticks in ARM_CUE_ON, AND exceed the SAME statistic
# measured in ARM_CUE_OFF by this margin (pre-registered; not derived post hoc).
SELECTED_PROXIMITY_LIFT_MARGIN = 0.01


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
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
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
        mech295_liking_to_approach_cue_gain=CUE_GAIN_NATURAL,
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
    )
    cfg.latent.use_resource_encoder = True
    return cfg


CONFIG_SLICE = {
    "scaffold_cfg": "see _make_scaffold_cfg (curriculum budgets + landed levers, mirrors 603n)",
    "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
    "seed_gain": SEED_GAIN, "seed_benefit_threshold": SEED_BENEFIT_THRESHOLD,
    "seed_drive_floor": SEED_DRIVE_FLOOR, "cue_recall_gain": CUE_RECALL_GAIN,
    "harm_pathway_lr": HARM_PATHWAY_LR,
    "cue_gain_natural": CUE_GAIN_NATURAL, "cue_gain_severed": CUE_GAIN_SEVERED,
}


class _CueAuthorityProbe:
    """Wraps agent.select_action + mech295_bridge.compute_approach_cue_score_bias
    to record, on EVERY tick where a candidate_proximity surface exists (goal
    active), the selected candidate's proximity vs the pool mean -- the Layer-
    C3/L4 readout 490j lacked -- plus whether the MECH-295 cue itself fired that
    tick (a separate boolean, NOT a gate on recording).

    CRITICAL: proximity/selection is recorded regardless of whether the bias
    this tick is nonzero. agent.py calls compute_approach_cue_score_bias
    whenever goal_state.is_active() -- independent of liking_to_approach_cue_gain
    -- so in ARM_CUE_OFF the call still happens and candidate_proximities is
    still real; only the RETURNED bias is forced to exact zero (severed cue).
    Gating the recording on "bias nonzero" would make ARM_CUE_OFF's selection-
    proximity statistic trivially unmeasured (always n=0), collapsing the C3
    diff-in-differences test (ARM_CUE_ON lift vs ARM_CUE_OFF lift) into a
    single-arm check against an unearned zero baseline -- caught in code review
    before the first smoke test.

    Pattern: mech295_bridge's method is called INSIDE select_action, strictly
    BEFORE agent._last_e3_selection_result is set (agent.py composes all score-
    bias channels, THEN calls e3.select(), THEN stashes the result). So the
    bridge wrapper stashes this tick's (candidate_proximities, bias) into a
    scratch slot; the select_action wrapper clears the scratch before calling
    orig (so a tick where the bridge is not called -- goal_state inactive --
    never carries over a stale reading, the same discipline the skill's
    e3.last_score_diagnostics latch rule requires) and, immediately after orig
    returns, pairs the scratch (if populated this tick) with
    agent._last_e3_selection_result.selected_index.
    """

    def __init__(self, agent: REEAgent) -> None:
        self.agent = agent
        self._scratch: Optional[Dict[str, Any]] = None
        self.ticks: List[Dict[str, Any]] = []
        self.n_select_action_calls = 0
        self.n_bridge_call_silent_ticks = 0  # goal inactive / bridge not called this tick

    def install(self) -> None:
        if self._installed:
            return
        bridge = getattr(self.agent, "mech295_bridge", None)
        if bridge is None:
            self._installed = True
            return
        orig_cue = bridge.compute_approach_cue_score_bias
        orig_select_action = self.agent.select_action

        def wrapped_cue(*args, **kwargs):
            out = orig_cue(*args, **kwargs)
            try:
                prox = kwargs.get("candidate_proximities")
                if prox is None and len(args) >= 2:
                    prox = args[1]
                if prox is not None and prox.numel() >= 2:
                    self._scratch = {
                        "proximities": prox.detach().clone(),
                        "bias": out.detach().clone(),
                    }
            except Exception:
                pass
            return out

        def wrapped_select_action(*args, **kwargs):
            self._scratch = None
            self.n_select_action_calls += 1
            action = orig_select_action(*args, **kwargs)
            scratch = self._scratch
            if scratch is not None:
                result = getattr(self.agent, "_last_e3_selection_result", None)
                sel_idx = getattr(result, "selected_index", None)
                prox = scratch["proximities"]
                bias = scratch["bias"]
                if sel_idx is not None and 0 <= int(sel_idx) < prox.numel():
                    decomp = dict(getattr(self.agent, "_last_score_bias_decomp", {}) or {})
                    bias_fired = bool((bias.abs() > 0).any().item())
                    self.ticks.append({
                        "selected_proximity": float(prox[int(sel_idx)].item()),
                        "pool_mean_proximity": float(prox.mean().item()),
                        "proximity_range": float((prox.max() - prox.min()).item()),
                        "bias_fired": bias_fired,
                        "bias_range": float((bias.max() - bias.min()).item()),
                        "mech295_liking_bias_range_mean": float(
                            decomp.get("mech295_liking_bias_range_mean", 0.0)
                        ),
                        "dacc_bias_range_mean": float(decomp.get("dacc_bias_range_mean", 0.0)),
                        "lateral_pfc_bias_range_mean": float(
                            decomp.get("lateral_pfc_bias_range_mean", 0.0)
                        ),
                        "ofc_bias_range_mean": float(decomp.get("ofc_bias_range_mean", 0.0)),
                        "total_bias_bias_range_mean": float(
                            decomp.get("total_bias_bias_range_mean", 0.0)
                        ),
                    })
                else:
                    self.n_bridge_call_silent_ticks += 1
            else:
                self.n_bridge_call_silent_ticks += 1
            self._scratch = None
            return action

        bridge.compute_approach_cue_score_bias = wrapped_cue
        self.agent.select_action = wrapped_select_action
        self._installed = True

    _installed = False

    def summary(self) -> Dict[str, Any]:
        n = len(self.ticks)
        n_fired = sum(1 for t in self.ticks if t["bias_fired"])
        if n == 0:
            return {
                "n_select_action_calls": self.n_select_action_calls,
                "n_proximity_evaluable_ticks": 0,
                "n_cue_fire_ticks": 0,
                "n_bridge_call_silent_ticks": self.n_bridge_call_silent_ticks,
                "mean_selected_minus_pool_mean_proximity": 0.0,
                "mean_proximity_range": 0.0,
                "mean_mech295_liking_bias_range_mean": 0.0,
                "mean_dacc_bias_range_mean": 0.0,
            }
        deltas = [t["selected_proximity"] - t["pool_mean_proximity"] for t in self.ticks]
        return {
            "n_select_action_calls": self.n_select_action_calls,
            "n_proximity_evaluable_ticks": n,
            "n_cue_fire_ticks": n_fired,
            "n_bridge_call_silent_ticks": self.n_bridge_call_silent_ticks,
            # Lift is computed over EVERY proximity-evaluable tick (not just
            # cue-fire ticks) so ARM_CUE_OFF yields a genuine, non-trivial
            # baseline: does the SELECTED action tend toward high-proximity
            # candidates via pathways other than MECH-295 when the cue is
            # severed? See class docstring.
            "mean_selected_minus_pool_mean_proximity": float(sum(deltas) / n),
            "mean_proximity_range": float(sum(t["proximity_range"] for t in self.ticks) / n),
            "mean_mech295_liking_bias_range_mean": float(
                sum(t["mech295_liking_bias_range_mean"] for t in self.ticks) / n
            ),
            "mean_dacc_bias_range_mean": float(
                sum(t["dacc_bias_range_mean"] for t in self.ticks) / n
            ),
        }


def _run_p2_arm(
    scheduler: ScaffoldedSD054OnboardingScheduler,
    agent: REEAgent,
    device: torch.device,
    seed: int,
    arm_label: str,
    cue_gain: float,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run one frozen-policy P2 pass at a fixed mech295 cue gain, instrumented."""
    reset_all_rng(seed)  # matched env realisation across arms of this seed
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is not None:
        bridge.config.liking_to_approach_cue_gain = float(cue_gain)
        cue_fires_before = int(getattr(bridge, "_n_cue_fires", 0))
        write_fires_before = int(getattr(bridge, "_n_write_fires", 0))
    else:
        cue_fires_before = write_fires_before = 0

    probe = _CueAuthorityProbe(agent)
    probe.install()
    agent.eval()
    p2 = scheduler.run_p2(agent, device)

    cue_fires_after = int(getattr(bridge, "_n_cue_fires", 0)) if bridge is not None else 0
    write_fires_after = int(getattr(bridge, "_n_write_fires", 0)) if bridge is not None else 0
    probe_summary = probe.summary()

    row: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "cue_gain": float(cue_gain),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_z_goal_norm_peak": float(p2.z_goal_norm_peak_max),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_approach_commit_rate": float(p2.approach_commit_rate),
        "p2_bridge_cue_fires_delta": cue_fires_after - cue_fires_before,
        "p2_bridge_write_fires_delta": write_fires_after - write_fires_before,
        **probe_summary,
    }
    print(
        f"  [p2_arm] seed={seed} arm={arm_label} gain={cue_gain:.2f}"
        f" cue_fire_ticks={probe_summary['n_cue_fire_ticks']}"
        f" selected_minus_pool={probe_summary['mean_selected_minus_pool_mean_proximity']:.4f}"
        f" proximity_range={probe_summary['mean_proximity_range']:.4f}"
        f" m295_bias_range={probe_summary['mean_mech295_liking_bias_range_mean']:.4f}",
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


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition MECH295_CUE_AUTHORITY_SD054", flush=True)

    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    print(
        f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
        f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}",
        flush=True,
    )
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return {"seed": seed, "aborted_at": "stage0", "abort_reason": s0.abort_reason,
                "arms": [], "g0_stage0_zgoal": False, "g1_p1_survival": False,
                "harm_eval_range": 0.0, "harm_train_steps": 0, "seed_pass": False}

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(
        f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
        f" retention={s0b.retention_ratio:.3f}",
        flush=True,
    )
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return {"seed": seed, "aborted_at": "stage0b", "abort_reason": s0b.abort_reason,
                "arms": [], "g0_stage0_zgoal": bool(s0.z_goal_norm_peak > STAGE0_POSITIVE_CONTROL_FLOOR),
                "g1_p1_survival": False, "harm_eval_range": 0.0, "harm_train_steps": 0,
                "seed_pass": False}

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return {"seed": seed, "aborted_at": "p0", "abort_reason": p0.abort_reason,
                "arms": [], "g0_stage0_zgoal": bool(s0.z_goal_norm_peak > STAGE0_POSITIVE_CONTROL_FLOOR),
                "g1_p1_survival": False, "harm_eval_range": 0.0, "harm_train_steps": 0,
                "seed_pass": False}

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    harm_diag = dict(getattr(hz, "harm_discriminativeness", {}) or {})
    harm_pathway_diag = dict(getattr(hz, "harm_pathway_diag", {}) or {})
    harm_eval_range = float(harm_diag.get("harm_eval_range", 0.0))
    harm_train_steps = int(harm_pathway_diag.get("n_train_steps", 0))
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
          f" harm_eval_range={harm_eval_range:.4f}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return {"seed": seed, "aborted_at": "hazard", "abort_reason": hz.abort_reason,
                "arms": [], "g0_stage0_zgoal": bool(s0.z_goal_norm_peak > STAGE0_POSITIVE_CONTROL_FLOOR),
                "g1_p1_survival": False, "harm_eval_range": harm_eval_range,
                "harm_train_steps": harm_train_steps, "seed_pass": False}

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    # ---- Frozen-policy P2: run BOTH arms on the SAME trained agent ----
    arm_on = _run_p2_arm(scheduler, agent, device, seed, "ARM_CUE_ON", CUE_GAIN_NATURAL, dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_cue_on seed={seed} ep {done}/{total_eps}", flush=True)
    arm_off = _run_p2_arm(scheduler, agent, device, seed, "ARM_CUE_OFF", CUE_GAIN_SEVERED, dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_cue_off seed={seed} ep {done}/{total_eps}", flush=True)

    g0 = bool(s0.z_goal_norm_peak > STAGE0_POSITIVE_CONTROL_FLOOR)
    g1 = bool(p1.survival_gate_passed)
    seed_pass = bool(g0 and g1)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} g0={g0} g1={g1}", flush=True)

    return {
        "seed": seed, "aborted_at": None, "abort_reason": "",
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "g0_stage0_zgoal": g0, "g1_p1_survival": g1,
        "harm_eval_range": harm_eval_range, "harm_train_steps": harm_train_steps,
        "arms": [arm_on, arm_off],
        "seed_pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + 2 * P2_BUDGET
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    reached_p2 = [r for r in per_seed if r.get("arms")]
    n = len(per_seed)

    # ---- Non-vacuity readiness preconditions (mirror 603n) ----
    harm_train_steps_max = max((r.get("harm_train_steps", 0) for r in per_seed), default=0)
    harm_eval_range_max = max((r.get("harm_eval_range", 0.0) for r in per_seed), default=0.0)
    harm_pathway_discriminative = bool(
        harm_train_steps_max >= HARM_TRAIN_STEPS_FLOOR and harm_eval_range_max >= HARM_EVAL_RANGE_FLOOR
    )
    g1_frac = _frac([r.get("g1_p1_survival", False) for r in per_seed])
    reached_p2_alive = bool(g1_frac >= MIN_FRACTION)
    preconditions_met = bool(harm_pathway_discriminative and reached_p2_alive)

    # ---- C0 evaluability: candidate_proximity range non-degenerate + >=MIN_CUE_FIRE_TICKS
    # cue-fire ticks in ARM_CUE_ON, on >=2/3 seeds that reached P2. ----
    c0_per_seed: List[bool] = []
    for r in reached_p2:
        on = next((a for a in r["arms"] if a["arm"] == "ARM_CUE_ON"), None)
        if on is None:
            c0_per_seed.append(False)
            continue
        c0_per_seed.append(
            bool(
                on["mean_proximity_range"] > PROXIMITY_RANGE_FLOOR
                and on["n_cue_fire_ticks"] >= MIN_CUE_FIRE_TICKS
            )
        )
    c0_frac = _frac(c0_per_seed) if c0_per_seed else 0.0
    c0_pass = bool(c0_frac >= MIN_FRACTION)

    # ---- C1 cue fires (trivial by construction in ARM_CUE_OFF; sanity-checked) ----
    c1_per_seed = [
        bool(
            next((a for a in r["arms"] if a["arm"] == "ARM_CUE_ON"), {}).get("n_cue_fire_ticks", 0) > 0
            and next((a for a in r["arms"] if a["arm"] == "ARM_CUE_OFF"), {}).get("n_cue_fire_ticks", 0) == 0
        )
        for r in reached_p2
    ]
    c1_pass = bool(_frac(c1_per_seed) >= MIN_FRACTION) if c1_per_seed else False

    # ---- C2 cue reaches scoring: mech295_liking_bias_range_mean > 0 in ARM_CUE_ON ----
    c2_per_seed = [
        bool(
            next((a for a in r["arms"] if a["arm"] == "ARM_CUE_ON"), {}).get(
                "mean_mech295_liking_bias_range_mean", 0.0
            ) > 0.0
        )
        for r in reached_p2
    ]
    c2_pass = bool(_frac(c2_per_seed) >= MIN_FRACTION) if c2_per_seed else False

    # ---- C3 selection-level effect (the load-bearing new test): selected-candidate
    # proximity exceeds pool mean by >= SELECTED_PROXIMITY_LIFT_MARGIN in ARM_CUE_ON,
    # AND that lift itself exceeds the SAME statistic in ARM_CUE_OFF by the same
    # margin (differences-in-differences vs the severed-bridge reference; ARM_CUE_OFF's
    # lift should be ~0 / noise since no cue is present to bias selection toward
    # proximity there). ----
    c3_per_seed: List[bool] = []
    c3_deltas: List[Dict[str, float]] = []
    for r in reached_p2:
        on = next((a for a in r["arms"] if a["arm"] == "ARM_CUE_ON"), None)
        off = next((a for a in r["arms"] if a["arm"] == "ARM_CUE_OFF"), None)
        if on is None or off is None:
            c3_per_seed.append(False)
            continue
        lift_on = on["mean_selected_minus_pool_mean_proximity"]
        lift_off = off["mean_selected_minus_pool_mean_proximity"]
        c3_deltas.append({"seed": r["seed"], "lift_on": lift_on, "lift_off": lift_off,
                           "diff_in_diff": lift_on - lift_off})
        c3_per_seed.append(
            bool(
                lift_on >= SELECTED_PROXIMITY_LIFT_MARGIN
                and (lift_on - lift_off) >= SELECTED_PROXIMITY_LIFT_MARGIN
            )
        )
    c3_pass = bool(_frac(c3_per_seed) >= MIN_FRACTION) if c3_per_seed else False

    # ---- C4 behavioural sanity (informative, NOT decisive per 490j precedent) ----
    c4_per_seed = [
        bool(
            next((a for a in r["arms"] if a["arm"] == "ARM_CUE_ON"), {}).get(
                "p2_approach_commit_rate", 0.0
            ) > next((a for a in r["arms"] if a["arm"] == "ARM_CUE_OFF"), {}).get(
                "p2_approach_commit_rate", 0.0
            )
        )
        for r in reached_p2
    ]
    c4_frac = _frac(c4_per_seed) if c4_per_seed else 0.0

    # ---- Routing ----
    if not preconditions_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif not c0_pass:
        outcome = "FAIL"
        label = "INVALID_HARNESS"
    elif not (c1_pass and c2_pass) and c3_pass:
        # Should not happen (C3 presupposes C1/C2 by construction) -- kept as an
        # explicit branch so an unexpected combination is visible, not silently routed.
        outcome = "FAIL"
        label = "internal_inconsistency_c3_without_c1c2"
    elif c1_pass and c2_pass and not c3_pass:
        outcome = "FAIL"
        label = "cue_fires_reaches_scoring_no_selection_authority"
    elif c1_pass and c2_pass and c3_pass:
        outcome = "PASS"
        label = "cue_authority_established_at_selection_layer"
    else:
        outcome = "FAIL"
        label = "cue_not_evaluable_or_does_not_fire"

    print(
        f"[{EXPERIMENT_TYPE}] C0={c0_pass} C1={c1_pass} C2={c2_pass} C3={c3_pass}"
        f" C4_informative_frac={c4_frac:.2f} -> outcome={outcome} label={label}",
        flush=True,
    )

    preconditions = [
        {
            "name": "harm_pathway_discriminative", "kind": "readiness",
            "description": "Harm pathway trained + harm_eval range lifts above 603i noise floor "
                            "(same non-vacuity precondition as 603n; this experiment reuses the "
                            "same landed curriculum).",
            "control": "scaffold_train_harm_pathway=True; measured on P0+Stage-H.",
            "measured": float(harm_eval_range_max), "threshold": float(HARM_EVAL_RANGE_FLOOR),
            "direction": "lower",
            "met": bool(harm_pathway_discriminative),
        },
        {
            "name": "reached_p2_alive", "kind": "readiness",
            "description": "P1 survival >=2/3 seeds so the agent reaches P2 alive.",
            "control": "P1 survival gate (median episode length last 10 P1 eps >= 75).",
            "measured": float(g1_frac), "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(reached_p2_alive),
        },
        {
            "name": "candidate_proximity_evaluable", "kind": "readiness",
            "description": "candidate_proximity range non-degenerate (same statistic C3 routes "
                            "on -- NOT a magnitude proxy, the V3-EXQ-643 lesson) AND >= "
                            f"{MIN_CUE_FIRE_TICKS} cue-fire ticks recorded, in ARM_CUE_ON, on "
                            ">=2/3 seeds that reached P2.",
            "control": "measured directly from the wrapped mech295_bridge calls in ARM_CUE_ON "
                       "(the positive-gain arm; candidates genuinely differ in world-state, so "
                       "this is a non-degenerate positive control for the range statistic).",
            "measured": float(c0_frac), "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(c0_pass),
        },
    ]
    criteria_non_degenerate = {
        "C0_evaluability": bool(len(reached_p2) >= 2),
        "C1_cue_fires": bool(c1_per_seed),
        "C2_reaches_scoring": bool(c2_per_seed),
        "C3_selection_authority": bool(c3_per_seed),
    }
    criteria = [
        {"name": "C0_evaluability", "load_bearing": True, "passed": bool(c0_pass)},
        {"name": "C1_cue_fires", "load_bearing": True, "passed": bool(c1_pass)},
        {"name": "C2_reaches_scoring", "load_bearing": True, "passed": bool(c2_pass)},
        {"name": "C3_selection_authority", "load_bearing": True, "passed": bool(c3_pass)},
    ]

    return {
        "outcome": outcome,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "c3_diff_in_diff_per_seed": c3_deltas,
            "c4_informative_approach_commit_lift_fraction": c4_frac,
        },
        "gate_summary": {
            "c0_evaluability_fraction": c0_frac, "c0_pass": c0_pass,
            "c1_cue_fires_pass": c1_pass, "c2_reaches_scoring_pass": c2_pass,
            "c3_selection_authority_pass": c3_pass,
            "min_fraction": MIN_FRACTION,
            "proximity_range_floor": PROXIMITY_RANGE_FLOOR,
            "min_cue_fire_ticks": MIN_CUE_FIRE_TICKS,
            "selected_proximity_lift_margin": SELECTED_PROXIMITY_LIFT_MARGIN,
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
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
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
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1, bridge ON throughout training; frozen-policy P2 run TWICE per "
                     "seed at cue_gain=natural vs cue_gain=0.0 -- the eval-time severed-bridge "
                     "authority test)",
        "predecessor": "V3-EXQ-490j (weakens/evidence, gap4, necessity layer); V3-EXQ-490k "
                       "(diagnostic, pre-603n substrate, could not test authority: "
                       "mech295_bias_range_mean=0.0). Neither measured selected-candidate "
                       "proximity vs pool mean; this experiment closes that gap on the "
                       "now-ready scaffolded_sd054_onboarding substrate (ready since 2026-06-11, "
                       "603n).",
        "design_note": "2 arms (ARM_CUE_ON natural gain, ARM_CUE_OFF severed) on the SAME "
                       "per-seed trained agent (bridge ON during training throughout, matching "
                       "the landed default trajectory -- severing during training would confound "
                       "authority with a differently-adapted policy). RNG reset to the seed value "
                       "before EACH P2 arm pass so both arms see an identical env realisation "
                       "sequence. Dropped the original 2026-06-03 proposal's third forced-z_goal "
                       "arm: this substrate now forms z_goal reliably (G3 gate, 603n), so C0 "
                       "checks natural non-degeneracy directly instead of assuming it.",
        "pre_registered_gates": {
            "c0_candidate_proximity_evaluable": f"range > {PROXIMITY_RANGE_FLOOR} AND "
                                                f">= {MIN_CUE_FIRE_TICKS} cue-fire ticks, "
                                                f"ARM_CUE_ON, >= {MIN_FRACTION:.3f} seeds",
            "c1_cue_fires": "n_cue_fire_ticks>0 in ARM_CUE_ON AND ==0 in ARM_CUE_OFF (sanity, "
                            "trivial by construction -- liking_to_approach_cue_gain=0 zeroes the "
                            "bias tensor exactly per mech295_liking_bridge.py)",
            "c2_reaches_scoring": "mean mech295_liking_bias_range_mean > 0 in ARM_CUE_ON (the "
                                  "V3-EXQ-643-class range statistic, not a magnitude proxy)",
            "c3_selection_authority": f"selected-minus-pool-mean proximity >= "
                                      f"{SELECTED_PROXIMITY_LIFT_MARGIN} in ARM_CUE_ON AND that "
                                      f"lift exceeds the same statistic in ARM_CUE_OFF by >= "
                                      f"{SELECTED_PROXIMITY_LIFT_MARGIN} (diff-in-diff vs the "
                                      f"severed-bridge reference) -- the LOAD-BEARING new test "
                                      f"490j/490k could not run",
            "c4_behavioural_sanity_informative_only": "approach_commit_rate ON > OFF; per 490j, "
                                                       "failure here alone does NOT falsify "
                                                       "authority if C1-C3 pass (known saturated "
                                                       "metric)",
            "min_fraction": MIN_FRACTION,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget_per_arm": P2_BUDGET, "n_p2_arms": 2,
            "train_steps": TRAIN_STEPS,
            "cue_gain_natural": CUE_GAIN_NATURAL, "cue_gain_severed": CUE_GAIN_SEVERED,
            "seeding_gain": SEED_GAIN, "seeding_benefit_threshold": SEED_BENEFIT_THRESHOLD,
            "seeding_drive_floor": SEED_DRIVE_FLOOR,
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
