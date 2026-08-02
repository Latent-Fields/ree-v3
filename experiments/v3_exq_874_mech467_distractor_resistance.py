"""V3-EXQ-874 (EXP-0398): MECH-467 distractor battery -- three dissociable
failure-mode legs, instrumented SEPARATELY on existing substrate.

EXPERIMENT_PURPOSE: see EXPERIMENT_PURPOSE constant below.

SCOPE (per the EXP-0398 proposal note -- read this before touching the design):
this is a DIAGNOSTIC OVER EXISTING SUBSTRATE, NOT a new attention module. REE-v3
attention is distributed precision-selection control (2026-06-04 standing
containment rule); a new attention substrate would only be justified if THIS
run exposes a specific failure the existing mechanisms cannot explain, which is
exactly what Q-082 (v4, unbuilt) is gated on. Q-082 is DOWNSTREAM of this run,
not a precondition for it.

GOV-REUSE-1 scoping pass (done before any code was written, per the proposal's
own instruction to scope the battery against existing telemetry first):
  - `grep -rl "distract" / "capture" REE_assembly/evidence/experiments/` (excluding
    runs/ packs) turns up exactly ONE prior distractor manifest family:
    V3-EXQ-484 (v3_exq_484_sd033a_distractor_resistance), which tests ONLY
    rule-state drift (MECH-262 leg (ii)) via a synthetic tensor harness with NO
    env and NO action selection -- it cannot speak to sensory capture or
    wrong-target selection at all. No "capture" manifest exists anywhere in the
    corpus. So the decisive readout for THIS run (co-occurrence, within one
    condition, of floor rule-drift + elevated conditioned wrong-target-selection
    rate) is NOT recorded and NOT derivable from any existing manifest -- this
    run is necessary. Checked: v3_exq_484_sd033a_distractor_resistance_*.json
    (4 runs), no substrate_hash recorded (pre-2026-07-12 manifests), so no
    reanalysis-eligible baseline exists regardless.
  - The scoping pass also identified which EXISTING telemetry to reuse instead
    of inventing new instrumentation (all three legs are read off fields the
    substrate already emits for other purposes):
      (a) sensory capture   -> obs_dict["resource_field_view_<distractor_name>"]
          (SD-049 per-type proximity field view; already fed into world_state,
          i.e. it is already part of the "active representation" E1/z_world
          reads -- causal_grid_world.py line ~3503 appends it to world_parts).
      (b) rule-state drift  -> agent.lateral_pfc.rule_state.norm() (SD-033a;
          the exact quantity V3-EXQ-484 validated the freeze/drift signature
          on, and the same read-out agent.py:7802 already uses).
      (c) wrong-target select -> info["sd049_consumed_type_tag_this_tick"]
          (SD-049 Phase 2 per-tick consumed-type tag; already emitted by
          causal_grid_world.py step() for the V3-EXQ-514 identity-classifier
          line).
  The DISTRACTOR itself is instantiated via `resource_type_benefit_amplitudes`
  set to 0.0 on the "distractor" type (SD-049 per-type amplitude, already-
  built): a resource that is visually/proximally identical to the goal type at
  the ENTITY_TYPES/grid level, spawned SIMULTANEOUSLY with the goal via the
  existing GAP-3 dual_cue primitive, but confers ZERO benefit on contact. That
  is simultaneously a "physically salient irrelevant cue", a "false affordance"
  and (via dual_cue) a "competing orienting target" from the proposal's
  distractor-class list. Other classes in that list (misleading reward-
  proximal cues, repeated interruption cues) are NOT covered by this first pass
  -- if this run is informative, those are natural V3-EXQ-874b/c follow-ons,
  not something this run needs to exhaust.

TIMING ARMS (per the proposal: "pre-commitment and during-replay-or-planning
ONLY -- EXCLUDE the during-commitment arm, it depends on stable multi-step
commitment, which the substrate cannot currently sustain"):
  ARM_PRECOMMIT -- operating_mode pinned to internal_planning (deliberation /
                   candidate-evaluation, before the agent commits to a course
                   of action -- V3-EXQ-484's ARM-C reading: "an active
                   rule-update regime, not a protected one").
  ARM_REPLAY    -- operating_mode pinned to internal_replay (V3-EXQ-484's
                   protected ARM-B regime, gate ~ 0.05).
  Sustained external_task / committed multi-step pursuit is deliberately NOT
  an arm here (that is the excluded during-commitment condition; also see
  memory note "no commitment-dependent behavioural experiments" -- the
  substrate cannot sustain multi-step commitment yet).
  `beta_gate_bistable` is left at its default False for the same reason.

SUBSTRATE-API FINDING (read before touching the mode-pin code -- this is the
reason the V3-EXQ-484 `_force_mode` idiom does NOT transfer to a live loop):
`SalienceCoordinator.tick()` unconditionally recomputes and OVERWRITES
`self._operating_mode` from softmax(logits) on every call
(salience_coordinator.py ~L434-465), and `agent.py` calls `self.salience.tick()`
inside `select_action()` on every env step. V3-EXQ-484 never calls `.tick()` at
all (it drives `LateralPFCAnalog.update()` directly against a hand-set
`_operating_mode`), so a one-shot attribute assignment before the loop is
invisible there. Verified this WOULD silently no-op here: pinning
`_operating_mode` before `agent.select_action()` gets clobbered by the very
`tick()` call inside that same `select_action()`, before `write_gate()` is ever
read for the `lateral_pfc.update()` gate. The fix used below
(`_pin_operating_mode_across_ticks`) wraps `coord.tick` so the REAL tick still
runs (preserving `_n_ticks` / salience_aggregate / hysteresis bookkeeping) and
then forces `_operating_mode` / `_current_mode` back to the pinned one-hot
immediately after, so every downstream `write_gate()` read within that same
tick (including the one gating `lateral_pfc.update`) sees the pin. This is the
correct live-loop adaptation of the V3-EXQ-484 technique, not a substitute for
it.

RULE-STATE-RESET FINDING: `agent.reset()` resets `lateral_pfc.rule_state` to
zero at episode boundaries (SD-033a spec). A per-episode-reset drift metric
would therefore mostly measure "does the rule re-establish from zero", not
distractor resistance. So each (seed, arm) cell runs ONE continuous episode
(no mid-run `agent.reset()`), long enough that the environment's own step
ceiling (`done = agent_health<=0 or steps>=500`) never fires within budget:
`num_hazards=0` removes the health-death path entirely, and the step budget
below is well under 500.

ARC-062 GAP-B CAVEAT (proposal non-degeneracy check (c)): the upstream
rule-creator producing rule_state is still reported MONOMODAL as of the most
recent claims.yaml note on MECH-262 (arc_062_rule_apprehension:GAP-B /
behavioral_diversity_isolation:GAP-A monostrategy collapse; latest falsifier
V3-EXQ-654f FAIL 2026-06-18, 654g in flight). This experiment does NOT require
rule_state to be DIFFERENTIATED/multimodal -- leg (c)'s "rule verified intact"
conditioning here is operationalised identically to V3-EXQ-484's validated
norm-drift-at-floor methodology (is the established rule frozen or not), never
content differentiation between distinct rules. That is a materially weaker
and GAP-B-independent claim than MECH-262's own blocked retest (which needs
differentiated rule CONTENT). The one place GAP-B monomodality COULD still
make the conditioning vacuous is if rule_state drift is degenerate in one of
two trivial ways -- literally never drifting, or literally always drifting, in
every arm -- and that is exactly the non-degeneracy check (c) implemented
below (self-routes substrate_not_ready_requeue rather than reading either
degenerate case as a verdict).

Run:
  /opt/local/bin/python3 experiments/v3_exq_874_mech467_distractor_resistance.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_874_mech467_distractor_resistance.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.committed_mode_curriculum import (  # noqa: E402
    clone_trained_agent,
    run_p0_warmup,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_874_mech467_distractor_resistance"
QUEUE_ID = "V3-EXQ-874"
CLAIM_IDS = ["MECH-467"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]  # not a reef-config env; seed 44 instability does not apply

# SD-049 resource types: index 0 = goal (benefit-bearing), index 1 = distractor
# (zero benefit -- a false affordance, physically identical at the grid level).
RESOURCE_TYPE_NAMES = ("goal", "distractor")
GOAL_TAG = 1        # type_idx 0 -> tag 1 (SD-049 tag = type_idx + 1)
DISTRACTOR_TAG = 2  # type_idx 1 -> tag 2

ARMS = ("ARM_PRECOMMIT", "ARM_REPLAY")
ARM_MODE = {
    "ARM_PRECOMMIT": "internal_planning",
    "ARM_REPLAY": "internal_replay",
}

# Pre-registered thresholds (constants; NOT derived from the run's own stats).
SENSORY_CAPTURE_THRESHOLD = 0.05   # resource_field_view is a [0,1]-normalised
                                    # proximity gradient (proxy fields); this
                                    # floor is well below the ~0.22 minimum the
                                    # skill's own hazard_field_view precedent
                                    # cites for a single-source proxy field, so
                                    # any in-range contact registers.
RULE_DRIFT_FLOOR = 0.05            # "near-frozen" reading, matching the order
                                    # of magnitude V3-EXQ-484 used (its
                                    # replay/task drift ratio floor); rule_dim
                                    # default is small so norm-scale is modest.
WRONG_TARGET_ELEVATED_MIN = 0.10   # conditioned wrong-target rate must clear
                                    # this to count as "elevated" (leg c).
MIN_INTACT_TICKS = 10              # sample floor for the conditioned leg (c)
                                    # rate to be meaningful, pooled across
                                    # seeds within an arm.

N_WARMUP_STEPS = 60   # in-episode rule-establishment window, mode unpinned
N_EVAL_STEPS = 150     # distractor-exposure window, mode pinned per arm
PASS_FRACTION_REQUIRED = 1  # dissociability need only be shown in ONE arm
                              # (per the proposal's own PASS wording: "at least
                              # one condition")


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_easy_env(size: int, smoke: bool) -> CausalGridWorldV2:
    """P0 warmup env: goal type only (distractor density 0), no dual_cue.

    Establishes ordinary goal-seeking behaviour before the distractor battery
    is ever introduced. num_hazards=0 so the P0 phase cannot terminate the
    agent early either (kept symmetric with the target env).
    """
    return CausalGridWorldV2(
        seed=None,
        size=size,
        num_hazards=0,
        num_resources=4,
        num_waypoints=2,
        resource_respawn_on_consume=True,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=2,
        resource_type_names=RESOURCE_TYPE_NAMES,
        resource_type_drive_axes=("goal_need", "distractor_need"),
        resource_type_benefit_curves=("sigmoidal_saturating", "sigmoidal_saturating"),
        resource_type_distribution=(1.0, 0.0),         # distractor type absent
        resource_type_benefit_amplitudes=(1.0, 0.0),
        per_axis_drive_enabled=False,
        dual_cue_enabled=False,
    )


def _build_target_env(size: int) -> CausalGridWorldV2:
    """Distractor-battery env: both types spawn, distractor gives ZERO benefit,
    dual_cue guarantees both are simultaneously live (GAP-3 primitive)."""
    return CausalGridWorldV2(
        seed=None,
        size=size,
        num_hazards=0,
        num_resources=5,
        num_waypoints=2,
        resource_respawn_on_consume=True,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=2,
        resource_type_names=RESOURCE_TYPE_NAMES,
        resource_type_drive_axes=("goal_need", "distractor_need"),
        resource_type_benefit_curves=("sigmoidal_saturating", "sigmoidal_saturating"),
        resource_type_distribution=(1.0, 1.0),
        resource_type_benefit_amplitudes=(1.0, 0.0),   # distractor: zero benefit
        per_axis_drive_enabled=False,
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(GOAL_TAG, DISTRACTOR_TAG),
    )


def _build_agent(world_obs_dim: int, body_obs_dim: int) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
    )
    return REEAgent(cfg)


def _pin_operating_mode_across_ticks(agent: REEAgent, mode: str) -> None:
    """Wrap agent.salience.tick so the FORCED mode survives every subsequent
    call within a live env loop.

    See the module docstring "SUBSTRATE-API FINDING" for why a plain one-shot
    `_operating_mode` assignment (the V3-EXQ-484 idiom) does not survive here:
    `tick()` unconditionally recomputes `_operating_mode` from live signals on
    every call. This wrapper still calls the REAL tick() first (so `_n_ticks`,
    `salience_aggregate`, and the enter/exit hysteresis bookkeeping all run
    exactly as they would unpinned), then forces `_operating_mode` /
    `_current_mode` back to a one-hot pin on `mode` and patches the returned
    dict to match, so every downstream `write_gate()` read in the SAME
    select_action() call (including the one gating `lateral_pfc.update`) sees
    the pin.
    """
    coord = agent.salience
    real_tick = coord.tick

    def _pinned_tick(*args, **kwargs):
        result = real_tick(*args, **kwargs)
        coord._operating_mode = {
            m: (1.0 if m == mode else 0.0) for m in coord.mode_names
        }
        coord._current_mode = mode
        result["operating_mode"] = dict(coord._operating_mode)
        result["current_mode"] = mode
        return result

    coord.tick = _pinned_tick


def _env_step(agent: REEAgent, env: CausalGridWorldV2, obs_dict: dict, device) -> tuple:
    """One live env step via the standard REEAgent pipeline (matches the
    generate_trajectories/select_action pattern used by V3-EXQ-464b's
    _eval_competing_goals). Returns (obs_dict, info, done)."""
    world_dim = agent.config.latent.world_dim
    obs_body = obs_dict["body_state"].to(device)
    obs_world = obs_dict["world_state"].to(device)
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick")
        else torch.zeros(1, world_dim, device=device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    action_idx = int(action.argmax(dim=-1).item())
    _, _, done, info, next_obs_dict = env.step(action_idx)
    return next_obs_dict, info, done


def _run_arm(
    agent_base: REEAgent,
    arm: str,
    env: CausalGridWorldV2,
    device: torch.device,
    smoke: bool,
) -> dict:
    """One (seed, arm) cell: clone -> in-episode warmup (unpinned) -> pin ->
    distractor-exposure window (pinned). ONE continuous episode; no mid-run
    agent.reset() (see module docstring "RULE-STATE-RESET FINDING")."""
    agent = clone_trained_agent(agent_base, bistable=False, device=device)
    agent.eval()

    n_warmup = 5 if smoke else N_WARMUP_STEPS
    n_eval = 8 if smoke else N_EVAL_STEPS

    agent.reset()
    _, obs_dict = env.reset()

    with torch.no_grad():
        for _ in range(n_warmup):
            obs_dict, _info, done = _env_step(agent, env, obs_dict, device)
            if done:
                break

    rule_pre = agent.lateral_pfc.rule_state.clone()

    _pin_operating_mode_across_ticks(agent, ARM_MODE[arm])

    sensory_capture_hits = 0
    n_ticks = 0
    n_intact = 0
    n_wrong_target_intact = 0
    n_wrong_target_total = 0
    n_correct_target_total = 0
    final_drift = 0.0
    write_gate_samples = []

    distractor_field_key = f"resource_field_view_{RESOURCE_TYPE_NAMES[1]}"

    with torch.no_grad():
        for _ in range(n_eval):
            distractor_val = float(obs_dict.get(distractor_field_key, torch.zeros(1)).max().item())
            sensory_capture_hits += int(distractor_val > SENSORY_CAPTURE_THRESHOLD)

            obs_dict, info, done = _env_step(agent, env, obs_dict, device)
            n_ticks += 1

            drift_now = float((agent.lateral_pfc.rule_state - rule_pre).norm().item())
            final_drift = drift_now
            rule_intact = drift_now < RULE_DRIFT_FLOOR
            n_intact += int(rule_intact)

            consumed_tag = int(info.get("sd049_consumed_type_tag_this_tick", 0))
            wrong_target = consumed_tag == DISTRACTOR_TAG
            correct_target = consumed_tag == GOAL_TAG
            n_wrong_target_total += int(wrong_target)
            n_correct_target_total += int(correct_target)
            if rule_intact and wrong_target:
                n_wrong_target_intact += 1

            write_gate_samples.append(float(agent.salience.write_gate("sd_033a")))

            if done:
                break

    sensory_capture_rate = sensory_capture_hits / max(1, n_ticks)
    wrong_target_rate_conditioned = (
        n_wrong_target_intact / n_intact if n_intact > 0 else 0.0
    )
    wrong_target_rate_unconditioned = n_wrong_target_total / max(1, n_ticks)

    return {
        "arm": arm,
        "operating_mode": ARM_MODE[arm],
        "n_ticks": n_ticks,
        "sensory_capture_rate": round(sensory_capture_rate, 4),
        "final_rule_drift": round(final_drift, 6),
        "n_intact_ticks": n_intact,
        "wrong_target_rate_conditioned": round(wrong_target_rate_conditioned, 4),
        "wrong_target_rate_unconditioned": round(wrong_target_rate_unconditioned, 4),
        "n_wrong_target_total": n_wrong_target_total,
        "n_correct_target_total": n_correct_target_total,
        "mean_write_gate_sd033a": round(
            sum(write_gate_samples) / max(1, len(write_gate_samples)), 4
        ),
    }


def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    torch.manual_seed(seed)

    size = 8 if smoke else 10
    p0_budget = 3 if smoke else 100
    p0_steps_per_ep = 15 if smoke else 100

    easy_env = _build_easy_env(size, smoke)
    target_env = _build_target_env(size)

    world_obs_dim = target_env.world_obs_dim
    body_obs_dim = 12

    agent = _build_agent(world_obs_dim, body_obs_dim).to(device)

    print(f"Seed {seed} Condition warmup P0", flush=True)
    p0 = run_p0_warmup(
        agent, easy_env, device,
        budget=p0_budget, steps_per_episode=p0_steps_per_ep,
    )
    print(
        f"  [train] ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
        flush=True,
    )
    if p0.aborted:
        print("verdict: FAIL", flush=True)
        return {
            "seed": seed,
            "outcome": "commitment_not_elicited",
            "p0_aborted": True,
            "p0_abort_reason": p0.abort_reason,
            "arms": {},
            "pass": False,
        }

    arm_rows = {}
    for arm in ARMS:
        print(f"Seed {seed} Condition {arm}", flush=True)
        with arm_cell(
            seed,
            config_slice={
                "arm": arm,
                "operating_mode": ARM_MODE[arm],
                "n_warmup_steps": N_WARMUP_STEPS,
                "n_eval_steps": N_EVAL_STEPS,
                "resource_type_benefit_amplitudes": (1.0, 0.0),
                "dual_cue_type_tags": (GOAL_TAG, DISTRACTOR_TAG),
            },
            script_path=Path(__file__),
            config_slice_declared=True,
            # The P0 warmup training is SHARED across both arms for a given
            # seed (both arms clone the same p0-trained agent), so neither
            # arm's cell is a pure function of (seed, arm config) alone from a
            # fresh RNG reset -- honestly ineligible for future reuse rather
            # than falsely marked eligible. See module docstring GOV-REUSE-1
            # section / queue note.
            extra_ineligible_reasons=["shared_p0_warmup_across_seed_arms"],
        ) as cell:
            row = _run_arm(agent, arm, target_env, device, smoke)
            cell.stamp(row)
        seed_pass = (
            row["final_rule_drift"] < RULE_DRIFT_FLOOR
            and row["wrong_target_rate_conditioned"] >= WRONG_TARGET_ELEVATED_MIN
            and row["n_intact_ticks"] >= (2 if smoke else MIN_INTACT_TICKS)
        )
        row["dissociation_pass"] = seed_pass
        arm_rows[arm] = row
        print(
            f"verdict: {'PASS' if seed_pass else 'FAIL'}"
            f" drift={row['final_rule_drift']:.4f}"
            f" wrong_target_cond={row['wrong_target_rate_conditioned']:.4f}"
            f" sensory_capture={row['sensory_capture_rate']:.4f}",
            flush=True,
        )

    return {
        "seed": seed,
        "arms": arm_rows,
        "p0_final_rv": p0.final_rv,
        "pass": any(r["dissociation_pass"] for r in arm_rows.values()),
    }


def _pool_across_seeds(seed_results: list) -> dict:
    """Pool per-arm metrics across seeds for the run-level non-degeneracy /
    acceptance checks."""
    pooled = {}
    for arm in ARMS:
        rows = [
            r["arms"][arm] for r in seed_results
            if arm in r.get("arms", {})
        ]
        if not rows:
            pooled[arm] = None
            continue
        n_ticks_total = sum(r["n_ticks"] for r in rows)
        n_intact_total = sum(r["n_intact_ticks"] for r in rows)
        n_wrong_intact_total = sum(
            r["wrong_target_rate_conditioned"] * r["n_intact_ticks"] for r in rows
        )
        sensory_capture_any = any(r["sensory_capture_rate"] > 0 for r in rows)
        mean_final_drift = sum(r["final_rule_drift"] for r in rows) / len(rows)
        pooled[arm] = {
            "n_seeds": len(rows),
            "n_ticks_total": n_ticks_total,
            "n_intact_ticks_total": n_intact_total,
            "wrong_target_rate_conditioned_pooled": round(
                n_wrong_intact_total / n_intact_total, 4
            ) if n_intact_total > 0 else 0.0,
            "sensory_capture_rate_mean": round(
                sum(r["sensory_capture_rate"] for r in rows) / len(rows), 4
            ),
            "sensory_capture_any_seed": sensory_capture_any,
            "mean_final_rule_drift": round(mean_final_drift, 6),
            "rule_at_floor": mean_final_drift < RULE_DRIFT_FLOOR,
        }
    return pooled


def build_manifest(seed_results: list, smoke: bool, started_at: float) -> dict:
    pooled = _pool_across_seeds(seed_results)

    # -- Non-degeneracy checks (proposal's own NON-DEGENERACY block) --
    any_sensory_capture = any(
        (p is not None and p["sensory_capture_any_seed"]) for p in pooled.values()
    )
    any_leg_moves = any(
        p is not None and (
            p["sensory_capture_rate_mean"] > 0
            or p["mean_final_rule_drift"] > 1e-6
            or p["wrong_target_rate_conditioned_pooled"] > 0
        )
        for p in pooled.values()
    )
    # conditioning-vacuous: EVERY arm has either 0 intact ticks (rule never
    # stays intact) or ALL ticks intact (rule never drifts at all) -- either
    # way the (b)-at-floor conditioning in leg (c) does no work.
    conditioning_vacuous = all(
        p is None or p["n_intact_ticks_total"] == 0 or p["n_intact_ticks_total"] == p["n_ticks_total"]
        for p in pooled.values()
    )

    non_degenerate = any_sensory_capture and any_leg_moves and not conditioning_vacuous
    degeneracy_reasons = []
    if not any_sensory_capture:
        degeneracy_reasons.append(
            "distractor never registered in the active representation in any arm "
            "(sensory_capture_rate == 0 everywhere) -- non-degeneracy check (a) failed"
        )
    if not any_leg_moves:
        degeneracy_reasons.append(
            "ALL-FLOOR: no leg (sensory capture / rule drift / wrong-target rate) "
            "moved in any arm -- non-degeneracy check (b) failed"
        )
    if conditioning_vacuous:
        degeneracy_reasons.append(
            "rule-intact conditioning is vacuous in every arm (either never intact "
            "or always intact) -- non-degeneracy check (c) failed; see ARC-062 GAP-B "
            "caveat in module docstring"
        )
    degeneracy_reason = "; ".join(degeneracy_reasons)

    substrate_not_ready = not non_degenerate

    if substrate_not_ready:
        outcome = "FAIL"
        direction = "non_contributory"
        interpretation_label = "substrate_not_ready_requeue"
    else:
        # PASS = dissociability established in >= 1 arm: rule drift at floor
        # WHILE conditioned wrong-target rate is elevated.
        dissociation_found = any(
            p is not None
            and p["rule_at_floor"]
            and p["wrong_target_rate_conditioned_pooled"] >= WRONG_TARGET_ELEVATED_MIN
            and p["n_intact_ticks_total"] >= (2 if smoke else MIN_INTACT_TICKS)
            for p in pooled.values()
        )
        outcome = "PASS" if dissociation_found else "FAIL"
        direction = "supports" if dissociation_found else "does_not_support"
        interpretation_label = (
            "behavioural_capture_dissociated" if dissociation_found
            else "legs_covary_no_dissociation"
        )

    n_seeds = len(seed_results)
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"
    elapsed = time.perf_counter() - started_at

    arm_results = []
    for r in seed_results:
        for arm, row in r.get("arms", {}).items():
            row_copy = dict(row)
            row_copy["seed"] = r["seed"]
            arm_results.append(row_copy)

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-467": direction},
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": {
            "label": interpretation_label,
            "preconditions": [
                {
                    "name": "distractor_registered_in_active_representation",
                    "description": (
                        "resource_field_view_distractor non-zero in at least "
                        "one arm -- distractor must be encoded, not just present"
                    ),
                    "measured": max(
                        (p["sensory_capture_rate_mean"] for p in pooled.values() if p),
                        default=0.0,
                    ),
                    "threshold": 0.0,
                    "comparator": ">",
                    "met": any_sensory_capture,
                },
                {
                    "name": "rule_intact_conditioning_non_vacuous",
                    "description": (
                        "at least one arm has SOME intact ticks and SOME "
                        "non-intact ticks -- the (b)-floor conditioning must "
                        "actually discriminate ticks, not trivially include/"
                        "exclude everything"
                    ),
                    "measured": 0.0 if conditioning_vacuous else 1.0,
                    "threshold": 1.0,
                    "direction": "lower",
                    "met": not conditioning_vacuous,
                },
            ],
            "criteria_non_degenerate": {
                "sensory_capture_registered": any_sensory_capture,
                "any_leg_moves": any_leg_moves,
                "conditioning_non_vacuous": not conditioning_vacuous,
            },
        },
        "thresholds": {
            "sensory_capture_threshold": SENSORY_CAPTURE_THRESHOLD,
            "rule_drift_floor": RULE_DRIFT_FLOOR,
            "wrong_target_elevated_min": WRONG_TARGET_ELEVATED_MIN,
            "min_intact_ticks": MIN_INTACT_TICKS,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": sum(1 for r in seed_results if r.get("pass")),
        "smoke": smoke,
        "arm_results": arm_results,
        "pooled_by_arm": pooled,
        "metrics": {"per_seed": seed_results},
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-467 three-leg distractor battery: sensory capture "
            "(resource_field_view_distractor), rule-state drift "
            "(lateral_pfc.rule_state norm, SD-033a), and wrong-target "
            "selection (sd049_consumed_type_tag_this_tick == distractor tag) "
            "conditioned on rule verified intact. Diagnostic over existing "
            "substrate per the EXP-0398 proposal note -- no new attention "
            "module. Arms: ARM_PRECOMMIT (internal_planning, pinned) and "
            "ARM_REPLAY (internal_replay, pinned); the during-commitment arm "
            "is deliberately excluded (substrate cannot sustain multi-step "
            "commitment yet). PASS requires dissociability in >= 1 arm: rule "
            "drift at floor while conditioned wrong-target rate is elevated -- "
            "that is behavioural capture with the rule intact, invisible to "
            "every existing rule-drift metric. Q-082 (v4 pre-selection "
            "substrate) is downstream of this result, not a precondition."
        ),
    }
    return manifest


def _full_config() -> dict:
    """Full env + agent + schedule config snapshot for the recording standard
    always-core (`config` field). Env kwargs mirror _build_easy_env /
    _build_target_env; agent kwargs mirror _build_agent."""
    return {
        "env_easy_kwargs": {
            "num_hazards": 0,
            "num_resources": 4,
            "num_waypoints": 2,
            "resource_respawn_on_consume": True,
            "multi_resource_heterogeneity_enabled": True,
            "n_resource_types": 2,
            "resource_type_names": RESOURCE_TYPE_NAMES,
            "resource_type_distribution": (1.0, 0.0),
            "resource_type_benefit_amplitudes": (1.0, 0.0),
            "dual_cue_enabled": False,
        },
        "env_target_kwargs": {
            "num_hazards": 0,
            "num_resources": 5,
            "num_waypoints": 2,
            "resource_respawn_on_consume": True,
            "multi_resource_heterogeneity_enabled": True,
            "n_resource_types": 2,
            "resource_type_names": RESOURCE_TYPE_NAMES,
            "resource_type_distribution": (1.0, 1.0),
            "resource_type_benefit_amplitudes": (1.0, 0.0),
            "dual_cue_enabled": True,
            "dual_cue_min_active_ticks": 10,
            "dual_cue_replace_on_early_consume": False,
            "dual_cue_type_tags": (GOAL_TAG, DISTRACTOR_TAG),
        },
        "agent_kwargs": {
            "action_dim": 4,
            "use_dacc": True,
            "use_salience_coordinator": True,
            "use_lateral_pfc_analog": True,
            "beta_gate_bistable": False,
        },
        "arms": {arm: ARM_MODE[arm] for arm in ARMS},
        "schedule": {
            "p0_budget": 100,
            "p0_steps_per_episode": 100,
            "n_warmup_steps": N_WARMUP_STEPS,
            "n_eval_steps": N_EVAL_STEPS,
        },
        "thresholds": {
            "sensory_capture_threshold": SENSORY_CAPTURE_THRESHOLD,
            "rule_drift_floor": RULE_DRIFT_FLOOR,
            "wrong_target_elevated_min": WRONG_TARGET_ELEVATED_MIN,
            "min_intact_ticks": MIN_INTACT_TICKS,
        },
    }


def main(smoke: bool):
    device = torch.device("cpu")
    seeds = SEEDS[:1] if smoke else SEEDS
    started_at = time.perf_counter()
    seed_results = [run_seed(s, device, smoke) for s in seeds]
    manifest = build_manifest(seed_results, smoke, started_at)

    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    print(
        f"outcome: {manifest['outcome']}"
        f" ({manifest['n_seeds_pass']}/{manifest['n_seeds']} seeds pass)"
        f" non_degenerate={manifest['non_degenerate']}",
        flush=True,
    )

    if smoke:
        return None

    out_path = write_flat_manifest(
        manifest,
        out_dir=None,
        dry_run=False,
        config=_full_config(),
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=started_at,
    )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Smoke run (tiny budgets, no manifest)."
    )
    args = parser.parse_args()
    result = main(smoke=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        exit_reason="ok" if _outcome == "PASS" else "fail",
        dry_run=args.dry_run,
    )
    sys.exit(0)
