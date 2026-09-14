"""V3-EXQ-1038 -- ARC-131 endogenous coalition-recruitment RATE probe (diagnostic).

red-team (fable): CONTESTED, 4 findings (F1a, F1b, F2, F3b), all dispositioned by
fix -- see "TICKS_PER_EPISODE", "E3-MARGIN CONTEXT" and "SAMPLE UNIT" notes below
and the queue entry note.

LINEAGE / ROUTING
-----------------
EVB-1242 / EXP-0279 (claim ARC-131). ARC-131 asserts "installability" is dissociable
from isolated component-level validation: a mechanism that passes construction/wiring
tests in isolation can still stay dormant once composed into the full agent. Its own
blocked_note names coalition control (SD-091/MECH-481) as a concrete illustrative case,
but until chip-20260902-arc131-coalition-endogenous-recruitment-driver (ree-v3 main
88c7c3332c) there was no live caller of request_coalition() at all, so no isolated-vs-
composed contrast could even be instantiated. That chip landed REEAgent's endogenous
trigger (ree_core/agent.py select_action, gated by use_endogenous_coalition_trigger,
default off) and its own contract tests (tests/contracts/test_sd091_coalition_controller_
wiring.py W9-W13) prove the driver CAN fire (margin_threshold=1e6, guaranteed) and CAN be
suppressed (margin_threshold=-1.0, unreachable) -- but neither tells us whether it fires
at the real DEFAULT threshold (0.05) during an actual live run. That is exactly the
"existence + a passing construction gate is not evidence of runtime exercise" gap
REE_assembly/evidence/planning/claim_synthesis_ARC-120_2026-09-07.md found for ARC-120's
four other default-off gates -- this run is the same check applied to ARC-131's own
illustrative case.

This is DELIBERATELY NOT V3-EXQ-886 (the coalition 4-arm performance-recovery falsifier,
left unqueued -- see that script's own STATUS block). 886 needs a goal-directed, online-
adapting agent competence the naive CausalGridWorldV2 harness does not supply; recruitment
RATE needs none of that -- it is a direct read of REEAgent's own diagnostic counter
(agent._endogenous_coalition_request_count), which increments regardless of whether the
agent is competent at the task. No performance claim is made or needed here.

WHAT THIS RUN MEASURES
-----------------------
Per seed, per episode, at the config DEFAULT margin_threshold=0.05 (never overridden --
overriding it would just re-derive W11/W13's already-proven CAN-fire/CAN-be-suppressed
boundary, not answer whether the driver engages under its shipped default): does
CoalitionController actually get recruited, and how often? Primary readout:
  (a) request rate per episode (mean agent._endogenous_coalition_request_count per ep)
  (b) fraction of episodes with count > 0 ("gets recruited at all this episode")

TICKS_PER_EPISODE (red-team fable, F1a): E3 only re-evaluates every
e3_steps_per_tick=10 raw env steps (ree_core/heartbeat/clock.py), and the driver
additionally needs a PRIOR E3 result to compare against (it reads the previous
tick's margin, per the "MUST read the PREVIOUS tick's result" comment at the call
site) -- so a 20-tick episode gives at most one eligible evaluation and collapses
readout (a) onto readout (b) by construction. TICKS_PER_EPISODE=100 gives ~9
eligible evaluations/episode (ticks 20,30,...,100; tick 10 has no prior yet), so
(a) is a genuinely distinct statistic from (b), not a restatement of it.

E3-MARGIN CONTEXT (red-team fable, F1b/F2): the agent is untrained, so whether the
observed E3 candidate-score margin clears 0.05 is partly a property of this
harness's random-init score SCALE, not solely of the architecture's disposition to
recruit control. The 1e6 positive control proves the request_coalition() call path
and the comparison machinery both work (ruling out "comparison always False" /
"call never reached" instrument defects) but, being unreachably permissive by
design, it cannot certify that 0.05 is well *calibrated* against the margins this
harness actually produces -- a defect that would look identical to genuine
architectural dormancy. So this run also records the observed E3-margin
distribution per seed (sampled once per FRESH E3 result, de-duplicated by object
identity so a latched read is never counted as a new observation -- the same
latch hazard CLAUDE.md's "Sample-size integrity" section documents for
`last_score_diagnostics`) as diagnostic context: a "dormant" reading alongside
margins that sit far above 0.05 supports a scale/calibration reading; a "dormant"
reading with margins hovering near 0.05 supports a genuine near-tie-rate reading.
Neither changes the outcome/PASS criterion (readiness only), but both are recorded
so a governance reader is not left inferring "installability failure" from a
number that may instead reflect an untrained score scale.

SAMPLE UNIT (red-team fable, F3b): with a static (untrained, no online learning)
network per seed, episodes within one seed are correlated draws, not independent
samples -- the true independent unit is the SEED (7), not the pooled episode count
(210). The interpretation label is therefore derived from the fraction of SEEDS
ever recruited (n_seeds_ever_recruited / n_seeds), not from the pooled per-episode
fraction; the pooled figure is still recorded (as `overall_frac_episodes_recruited`)
for descriptive context but is not what the label routes on.

MACHINE-CLASS NOTE: E3 selection uses torch.multinomial, so per-seed values are not
bit-identical across darwin-arm64 / linux-x86_64 (machine_affinity: any is still
appropriate here -- this is a rate diagnostic, not an exact-action-sequence
assertion).

READINESS CONTROL (closes the "measurement mechanism itself broken" confound)
-------------------------------------------------------------------------------
A structural zero at the default threshold is ambiguous by itself: it could mean either
(i) the mechanism engages but rarely at this threshold/harness (the ARC-131-relevant
reading) or (ii) the counting/wiring is broken in a live full-loop run even though it
works in the contract tests' narrower synthetic harness (an instrument defect). To
distinguish them, EVERY seed also runs a short POSITIVE-CONTROL burst at
margin_threshold=1e6 (the exact value W11's contract test uses to guarantee firing) BEFORE
the main measurement. If the control fails to fire on some seed, that seed's main-loop
zero is uninterpretable and the run self-routes substrate_not_ready_requeue rather than a
false "dormant" verdict. This is the P0 readiness-assert pattern (queue-experiment SKILL.md
Step 3) applied to a diagnostic COUNTER rather than a continuous score.

EXPERIMENT_PURPOSE = diagnostic: this reports a measurement, not a claim-confidence-bearing
hypothesis test. It is excluded from claim confidence/conflict scoring by convention; its
value is informational for governance's ARC-131 disposition.

No sleep, no phased training (no head is trained on any encoder output -- the agent takes
random/untrained forward passes purely to exercise the endogenous-trigger code path).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

_ZG = ZGoalStreamAccumulator()

ANCHOR_REACHABILITY_EXEMPT = (
    "endogenous_trigger_mechanism_live's control (margin_threshold=1e6) IS the "
    "degeneracy definition, not a hand-tuned approximation of it: it is the exact "
    "value contract test W11 (test_w11_endogenous_trigger_fires_on_high_threshold, "
    "tests/contracts/test_sd091_coalition_controller_wiring.py) uses to prove the "
    "driver fires on essentially every eligible tick once a prior-tick E3 result "
    "exists -- reachable by construction, not by a predicate that could be "
    "narrower than the state it anchors to."
)

EXPERIMENT_TYPE = "v3_exq_1038_arc131_coalition_endogenous_recruitment_rate_probe"
QUEUE_ID = "V3-EXQ-1038"
BACKLOG_ID = "EVB-1242"
CLAIM_IDS: List[str] = ["ARC-131"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [0, 1, 2, 3, 4, 5, 6]
DRY_RUN_SEEDS = [0, 1]

# Main measurement, at the config-shipped DEFAULT threshold. Never overridden.
DEFAULT_MARGIN_THRESHOLD = 0.05
N_EPISODES = 30
DRY_RUN_N_EPISODES = 3
# red-team fable F1a: 20 ticks gives at most 1 eligible E3 evaluation/episode
# (e3_steps_per_tick=10, plus the first E3 tick of each episode has no PRIOR
# result to compare -- see agent.reset() clearing _last_e3_selection_result).
# 100 ticks gives ~9 eligible evaluations/episode, making the per-episode count
# a real (non-binary) statistic.
TICKS_PER_EPISODE = 100

# Readiness positive control, matching contract test W11 exactly.
CONTROL_MARGIN_THRESHOLD = 1e6
CONTROL_TICKS = 40

# Env / agent construction (matches the W9-W13 contract-test harness).
GRID_SIZE = 5
NUM_HAZARDS = 1
NUM_RESOURCES = 1
ACTION_DIM = 4
SELF_DIM = 16
WORLD_DIM = 16

# Pre-registered readiness bar: EVERY seed's control burst must fire at least once.
G0_CONTROL_ALL_SEEDS_FIRE_THRESHOLD = 1.0


def _build(seed: int, margin_threshold: float) -> Any:
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        use_coalition_controller=True,
        use_endogenous_coalition_trigger=True,
        endogenous_coalition_margin_threshold=margin_threshold,
        endogenous_coalition_demand_type="sensory_resample",
    )
    torch.manual_seed(1000 + seed)
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env


def _obs(env: Any, od: Dict[str, Any]) -> Any:
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return b, w


def _run_control_burst(seed: int) -> Dict[str, Any]:
    """Positive-control burst: margin_threshold=1e6 guarantees firing (matches W11)."""
    agent, env = _build(seed, CONTROL_MARGIN_THRESHOLD)
    _flat, od = env.reset()
    b, w = _obs(env, od)
    for _ in range(CONTROL_TICKS):
        with torch.no_grad():
            action = agent.act_with_split_obs(b, w)
        _flat, _harm, done, _info, od = env.step(action)
        b, w = _obs(env, od)
        if done:
            _flat, od = env.reset()
            b, w = _obs(env, od)
    result = {
        "seed": seed,
        "control_request_count": int(agent._endogenous_coalition_request_count),
        "control_fired": agent._endogenous_coalition_request_count > 0,
    }
    _ZG.observe(agent)
    return result


def _run_main_measurement(seed: int, n_episodes: int) -> Dict[str, Any]:
    """Main measurement: N episodes at the shipped DEFAULT margin_threshold."""
    agent, env = _build(seed, DEFAULT_MARGIN_THRESHOLD)
    per_episode_counts: List[int] = []
    margin_samples: List[float] = []
    last_seen_result_id = None
    for ep in range(n_episodes):
        agent.reset()  # per-episode: also zeroes _endogenous_coalition_request_count
        last_seen_result_id = None  # _last_e3_selection_result is also cleared by reset()
        _flat, od = env.reset()
        b, w = _obs(env, od)
        for _ in range(TICKS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(b, w)
            # Diagnostic-only margin sample: de-duplicate by object identity so a
            # tick that did NOT run a fresh E3 selection (the attribute latches
            # the previous tick's result) is never counted as a new observation.
            result = agent._last_e3_selection_result
            if result is not None and id(result) != last_seen_result_id:
                last_seen_result_id = id(result)
                try:
                    scores = result.scores.detach()
                    if int(scores.numel()) >= 2:
                        sorted_scores, _ = torch.sort(scores)
                        margin_samples.append(
                            float(sorted_scores[1].item() - sorted_scores[0].item())
                        )
                except (AttributeError, RuntimeError, TypeError):
                    pass
            _flat, _harm, done, _info, od = env.step(action)
            b, w = _obs(env, od)
            if done:
                _flat, od = env.reset()
                b, w = _obs(env, od)
        per_episode_counts.append(int(agent._endogenous_coalition_request_count))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [train] seed={seed} ep {ep + 1}/{n_episodes}", flush=True)

    n_recruited = sum(1 for c in per_episode_counts if c > 0)
    frac_recruited = n_recruited / len(per_episode_counts)
    mean_rate = sum(per_episode_counts) / len(per_episode_counts)
    sorted_margins = sorted(margin_samples)
    median_margin = (
        sorted_margins[len(sorted_margins) // 2] if sorted_margins else None
    )
    _ZG.observe(agent)
    return {
        "seed": seed,
        "n_episodes": n_episodes,
        "per_episode_request_counts": per_episode_counts,
        "episodes_recruited": n_recruited,
        "frac_episodes_recruited": frac_recruited,
        "mean_request_rate_per_episode": mean_rate,
        "total_requests": sum(per_episode_counts),
        "e3_margin_samples": margin_samples,
        "e3_margin_n_samples": len(margin_samples),
        "e3_margin_median": median_margin,
        "e3_margin_mean": (
            sum(margin_samples) / len(margin_samples) if margin_samples else None
        ),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    n_episodes = DRY_RUN_N_EPISODES if dry_run else N_EPISODES

    print(f"Seed sweep Condition main ({len(seeds)} seeds)", flush=True)

    control_rows: List[Dict[str, Any]] = []
    main_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition control", flush=True)
        control_rows.append(_run_control_burst(seed))
        print(f"Seed {seed} Condition main", flush=True)
        row = _run_main_measurement(seed, n_episodes)
        main_rows.append(row)
        per_seed_pass = control_rows[-1]["control_fired"]
        print(
            f"  -> frac_episodes_recruited={row['frac_episodes_recruited']:.3f} "
            f"mean_rate={row['mean_request_rate_per_episode']:.3f} "
            f"control_fired={per_seed_pass}"
        )
        print(f"verdict: {'PASS' if per_seed_pass else 'FAIL'}")

    n_seeds_control_fired = sum(1 for r in control_rows if r["control_fired"])
    control_fire_fraction = n_seeds_control_fired / len(control_rows)

    try:
        preconditions = p0_readiness_gate(
            [
                {
                    "name": "endogenous_trigger_mechanism_live",
                    "measured": control_fire_fraction,
                    "threshold": G0_CONTROL_ALL_SEEDS_FIRE_THRESHOLD,
                    "control": (
                        "margin_threshold=1e6 guaranteed-fire positive control "
                        f"({CONTROL_TICKS} ticks/seed), matching contract test W11"
                    ),
                    "direction": "lower",
                    "met": control_fire_fraction >= G0_CONTROL_ALL_SEEDS_FIRE_THRESHOLD,
                }
            ]
        )
    except P0NotReady as e:
        manifest = {
            "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
            "queue_id": QUEUE_ID,
            "backlog_id": BACKLOG_ID,
            "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
                "criteria_non_degenerate": {"C0_readiness_control_fires": False},
            },
            "criteria": [
                {
                    "name": "C0_readiness_control_fires",
                    "load_bearing": True,
                    "passed": False,
                    "measured": control_fire_fraction,
                    "threshold": G0_CONTROL_ALL_SEEDS_FIRE_THRESHOLD,
                }
            ],
            "control_results": control_rows,
            "main_results": main_rows,
        }
        print("verdict: FAIL")
        return manifest

    # Readiness confirmed live -- report the measurement. This diagnostic makes no
    # pass/fail claim about the FINDING itself (dormant vs engaged is the reported
    # content, not a criterion) -- PASS here means "a valid, instrument-verified
    # measurement was obtained", regardless of which direction the finding points.
    pooled_episodes = sum(r["n_episodes"] for r in main_rows)
    pooled_recruited = sum(r["episodes_recruited"] for r in main_rows)
    overall_frac_recruited = pooled_recruited / pooled_episodes
    overall_mean_rate = sum(r["total_requests"] for r in main_rows) / pooled_episodes
    n_seeds_ever_recruited = sum(1 for r in main_rows if r["total_requests"] > 0)
    # red-team fable F3b: the SEED is the independent sampling unit (a static,
    # untrained network makes episodes within one seed correlated draws, not
    # independent ones), so the label routes on the per-seed fraction, not the
    # pooled per-episode fraction (kept below as descriptive context only).
    frac_seeds_recruited = n_seeds_ever_recruited / len(main_rows)
    all_margin_samples = [m for r in main_rows for m in r["e3_margin_samples"]]
    pooled_median_margin = (
        sorted(all_margin_samples)[len(all_margin_samples) // 2]
        if all_margin_samples
        else None
    )

    if frac_seeds_recruited < 0.05:
        label = "endogenous_recruitment_dormant_at_default_threshold"
    elif frac_seeds_recruited > 0.5:
        label = "endogenous_recruitment_engaged_at_default_threshold"
    else:
        label = "endogenous_recruitment_partial_at_default_threshold"

    outcome = "PASS"
    print(
        f"[measurement] {label}: frac_seeds_recruited={frac_seeds_recruited:.4f} "
        f"(overall_frac_episodes_recruited={overall_frac_recruited:.4f}, "
        f"pooled_median_e3_margin={pooled_median_margin})"
    )
    print(f"verdict: {outcome}")

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C0_readiness_control_fires": True},
        },
        "criteria": [
            {
                "name": "C0_readiness_control_fires",
                "load_bearing": True,
                "passed": True,
                "measured": control_fire_fraction,
                "threshold": G0_CONTROL_ALL_SEEDS_FIRE_THRESHOLD,
            }
        ],
        "control_results": control_rows,
        "main_results": main_rows,
        "readout": {
            "frac_seeds_recruited": float(frac_seeds_recruited),
            "overall_frac_episodes_recruited": float(overall_frac_recruited),
            "overall_mean_request_rate_per_episode": float(overall_mean_rate),
            "n_seeds_ever_recruited": int(n_seeds_ever_recruited),
            "n_seeds": len(seeds),
            "n_episodes_per_seed": n_episodes,
            "ticks_per_episode": TICKS_PER_EPISODE,
            "control_fire_fraction": float(control_fire_fraction),
            "default_margin_threshold": DEFAULT_MARGIN_THRESHOLD,
            "pooled_median_e3_margin": pooled_median_margin,
            "pooled_e3_margin_n_samples": len(all_margin_samples),
        },
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== {EXPERIMENT_TYPE} ===")
    print(f"Queue ID: {QUEUE_ID}")
    print(f"Claim: {CLAIM_IDS}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL RUN'}")
    print()

    result = run_experiment(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        result,
        dry_run=bool(args.dry_run),
        seeds=DRY_RUN_SEEDS if args.dry_run else SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nWrote manifest to: {out_path}")
    print(f"Outcome: {result['outcome']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=str(out_path),
        dry_run=bool(args.dry_run),
    )
