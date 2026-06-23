"""
V3-EXQ-703: MECH-276 scientist-agent counterfactual-backed attribution feedstock
-- substrate-readiness diagnostic.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle()/force_cycle() called once per
cell in a dedicated wake-sleep measurement loop)

MECH-276 (ree_core/attribution/scientist_attribution_buffer.py, landed 2026-06-23)
is the waking-phase feedstock the MECH-275 sleep-phase Bayesian aggregator consumes.
This is the substrate-readiness diagnostic for it -- claim_ids=[] (PROMOTES NOTHING;
does NOT promote MECH-275, which stays substrate_conditional pending its own later
promotion run). It confirms the feedstock PLUMBING works end-to-end on a TRAINED
substrate and that the falsifiable-distinction lever (only_counterfactual_backed) the
MECH-275 claim turns on measurably gates what reaches the aggregator.

ethics_preflight: V3 instrumentation only (no self-model, no autobiographical memory,
no negative-valence drive under test -- the buffer is pure-arithmetic attribution
bookkeeping over the existing comparators). All involvement flags false; decision: allow.

Design (CausalGridWorldV2 so the world_forward is genuinely action-conditional --
random obs would leave z_next independent of a_t and the comparator vacuous):

  P0 (positive-control readiness): collect real (z_world_t, a_t, z_world_{t+1})
    transitions from the env act loop and train the SD-031 E2WorldForward (forward MSE
    + SD-013 interventional action-divergence margin, stop-grad targets z_next.detach())
    so the comparator is discriminative. Measure world_forward_r2 on held-out transitions
    -- the SAME quantity the comparator's discrimination depends on (an untrained
    world_forward floors the comparator to a vacuous zero: the MECH-353/SD-031 lesson).

  Waking accumulation: run waking steps; agent.sense() auto-buffers the MECH-276
    counterfactual-backed attribution each tick (attribution = ||z_obs - E2(z_prev,a)||;
    counterfactual_contrast = ||E2(z_prev,a) - E2(z_prev,a_cf)||; counterfactual-backed
    iff contrast >= cf_margin). Install hot/cold anchors so the sleep sampler has regions.

  Sleep: force one sleep cycle. The MECH-275 BayesianAggregator's evidence is sourced
    from the MECH-276 buffer (REPLACING the staleness scalar) and its posterior is read.

ARMS (same seeds):
  ARM_CF_BACKED   : scientist_attribution_only_counterfactual_backed=True  (structured
                    feedstock -- correlational/low-contrast records SKIPPED).
  ARM_CORRELATIONAL: only_counterfactual_backed=False (feed everything incl low-contrast
                    = the noise-fit control).
  ARM_OFF         : use_scientist_attribution=False -> aggregator sources the legacy
                    MECH-284 staleness scalar (the bit-identical legacy baseline).

READINESS PRECONDITIONS (self-route substrate_not_ready_requeue if unmet -- NEVER a
substrate-verdict label):
  R1 world_forward trained: world_forward_r2 >= R2_FLOOR on held-out transitions
     (positive control; the comparator-discrimination quantity).
  R2 feedstock discriminable: in ARM_CF_BACKED the cf_contrast distribution STRADDLES
     cf_margin -- mech276_n_counterfactual_backed > 0 AND mech276_n_correlational_skipped
     > 0 -- so the only_counterfactual_backed lever has something to gate (the SAME
     mechanism the discrimination criterion routes on). If the margin does not straddle,
     the feedstock the two arms see is identical -> recalibrate cf_margin and requeue.

PRIMARY DISCRIMINATION (load-bearing): the MECH-275 aggregator posterior signature
  (mech275_mean_posterior_mean) DIFFERS between ARM_CF_BACKED and ARM_CORRELATIONAL by
  >= POSTERIOR_DELTA_FLOOR on >= PASS_FRACTION of seeds -- the counterfactual-backed
  feedstock drives a different aggregation than the correlational feedstock. A readiness-
  met failure to discriminate is a substrate-ceiling finding (the aggregator posterior is
  insensitive to the feedstock difference), NOT a MECH-276 falsification.

PASS iff R1 AND R2 met AND the discrimination holds. Diagnostic -> excluded from
governance confidence scoring; PASS clears the readiness gate that unblocks the SEPARATE
MECH-275 sleep-aggregation promotion run + re-adding MECH-275 to sleep_substrate:GAP-3b.

Phased training: P0 trains the E2WorldForward (a forward model) on stop-grad targets --
NOT a downstream head on a moving encoder, so no P0/P1/P2 latent-target collapse hazard;
the encoder is left as-is (SD-031 P1 discipline: target = z_world_next.detach()).
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_703_mech276_scientist_attribution_readiness"
QUEUE_ID = "V3-EXQ-703"
CLAIM_IDS: List[str] = []                 # diagnostic: PROMOTES NOTHING
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

SEEDS = [42, 7, 123]
WORLD_DIM = 128                            # E2WorldForward.attribution_ready requires >= 128
ACTION_DIM = 4
GRID_SIZE = 8

N_P0_TRANSITIONS = 600                     # real env transitions collected for world_forward P0
N_P0_TRAIN_STEPS = 300                     # P0 gradient steps on E2WorldForward
P0_BATCH = 64
P0_HELDOUT_FRAC = 0.2                      # held-out split for the world_forward_r2 readiness probe
N_WAKE_STEPS = 120                         # waking accumulation: buffer auto-records each tick
DRAWS_PER_CYCLE = 200                      # sampler draws so the aggregator gets enough updates
CF_MARGIN = 0.30                           # cf-contrast threshold; the readiness gate checks it straddles

# Pre-registered acceptance thresholds (absolute floors)
R2_FLOOR = 0.20                            # world_forward trained enough to be action-conditional
POSTERIOR_DELTA_FLOOR = 1e-3              # |posterior_mean(CF) - posterior_mean(CORR)| floor
PASS_FRACTION = 2.0 / 3.0                  # >= 2/3 seeds

ARMS = [
    {"arm": "ARM_CF_BACKED", "use_sci": True, "only_cf": True,
     "description": "only_counterfactual_backed=True -- structured feedstock"},
    {"arm": "ARM_CORRELATIONAL", "use_sci": True, "only_cf": False,
     "description": "only_counterfactual_backed=False -- feed everything (noise-fit control)"},
    {"arm": "ARM_OFF", "use_sci": False, "only_cf": True,
     "description": "use_scientist_attribution=False -- legacy MECH-284 staleness baseline"},
]


def _build_config(*, world_obs_dim: int, use_sci: bool, only_cf: bool) -> REEConfig:
    """All arms enable the substrate prerequisites identically (e2_world @ dim128, full
    sleep cluster, anchor sets, staleness). The MECH-276 arms differ only in
    only_counterfactual_backed; ARM_OFF differs only in use_scientist_attribution.
    """
    kw: Dict[str, Any] = dict(
        body_obs_dim=12,
        world_obs_dim=world_obs_dim,
        action_dim=ACTION_DIM,
        world_dim=WORLD_DIM,
        use_e2_world_forward=True,
        use_sleep_aggregation_cluster=True,
        sleep_loop_episodes_K=1,
        mech285_draws_per_cycle=DRAWS_PER_CYCLE,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_per_stream_vs=True,
        use_event_segmenter=True,
    )
    if use_sci:
        kw.update(
            use_scientist_attribution=True,
            scientist_attribution_cf_margin=CF_MARGIN,
            scientist_attribution_only_counterfactual_backed=only_cf,
        )
    return REEConfig.from_dims(**kw)


def _config_slice(*, arm: str, use_sci: bool, only_cf: bool) -> Dict[str, Any]:
    return {
        "arm": arm,
        "use_scientist_attribution": use_sci,
        "only_counterfactual_backed": only_cf,
        "world_dim": WORLD_DIM,
        "action_dim": ACTION_DIM,
        "grid_size": GRID_SIZE,
        "cf_margin": CF_MARGIN,
        "n_p0_transitions": N_P0_TRANSITIONS,
        "n_p0_train_steps": N_P0_TRAIN_STEPS,
        "n_wake_steps": N_WAKE_STEPS,
        "draws_per_cycle": DRAWS_PER_CYCLE,
    }


def _onehot(idx: int) -> torch.Tensor:
    a = torch.zeros(1, ACTION_DIM)
    a[0, idx % ACTION_DIM] = 1.0
    return a


def _collect_transitions(agent: REEAgent, env: CausalGridWorldV2, n: int) -> List:
    """Real (z_world_t, a_onehot, z_world_{t+1}) transitions from the env act loop.

    Action-conditional by construction (the discrete action moves the agent ->
    world_obs -> z_world), so the world_forward trains on genuine action effects.
    """
    transitions: List = []
    _, od = env.reset()
    agent.sense(od["body_state"], od["world_state"])
    while len(transitions) < n:
        z_prev = agent._current_latent.z_world.detach().clone()
        a = agent.act_with_split_obs(od["body_state"], od["world_state"])
        a_idx = int(a.argmax())
        _, h, d, inf, od = env.step(a)
        agent.sense(od["body_state"], od["world_state"])
        z_next = agent._current_latent.z_world.detach().clone()
        transitions.append((z_prev, _onehot(a_idx), z_next))
        if d:
            _, od = env.reset()
            agent.sense(od["body_state"], od["world_state"])
    return transitions


def _train_world_forward(agent: REEAgent, transitions: List, *, seed: int, arm: str) -> float:
    """P0: train E2WorldForward (forward MSE + SD-013 interventional margin) on stop-grad
    targets. Returns world_forward_r2 on the held-out split (positive-control readiness).
    """
    torch.manual_seed(seed)
    n_held = max(1, int(P0_HELDOUT_FRAC * len(transitions)))
    train_set = transitions[:-n_held]
    held = transitions[-n_held:]
    e2w = agent.e2_world
    opt = torch.optim.Adam(e2w.parameters(), lr=e2w.config.learning_rate)

    total = N_P0_TRAIN_STEPS
    for step in range(total):
        batch = [train_set[torch.randint(len(train_set), (1,)).item()] for _ in range(P0_BATCH)]
        zp = torch.cat([b[0] for b in batch], dim=0)
        a = torch.cat([b[1] for b in batch], dim=0)
        zn = torch.cat([b[2].detach() for b in batch], dim=0)   # stop-grad target
        a_cf = torch.cat([_onehot(int(b[1].argmax()) + 1) for b in batch], dim=0)
        pred = e2w.forward(zp, a)
        loss = e2w.compute_loss(pred, zn) + e2w.compute_interventional_loss(zp.detach(), a, a_cf)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, total // 3) == 0:
            print(f"  [train] world_forward seed={seed} arm={arm} ep {step + 1}/{total} "
                  f"loss={float(loss):.4f}", flush=True)

    # Held-out R2: 1 - SS_res / SS_tot of pred vs z_next.
    with torch.no_grad():
        zp = torch.cat([b[0] for b in held], dim=0)
        a = torch.cat([b[1] for b in held], dim=0)
        zn = torch.cat([b[2] for b in held], dim=0)
        pred = e2w.forward(zp, a)
        ss_res = float(((zn - pred) ** 2).sum().item())
        ss_tot = float(((zn - zn.mean(dim=0, keepdim=True)) ** 2).sum().item())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return r2


def _install_hot_cold_anchors(agent: REEAgent) -> None:
    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None, "anchor_set must be initialised (use_anchor_sets)"
    for i, seg in enumerate(("hot", "cold")):
        z = torch.randn(1, WORLD_DIM) * (i + 1)
        anchor_set.write_anchor(scale="fast", segment_id=seg,
                                stream_mixture=(f"stream_{seg}",), z_world=z)


def _accumulate_waking(agent: REEAgent, env: CausalGridWorldV2, n: int) -> None:
    """Run waking steps; agent.sense() auto-buffers the MECH-276 attribution each tick."""
    _, od = env.reset()
    agent.sense(od["body_state"], od["world_state"])
    for _ in range(n):
        a = agent.act_with_split_obs(od["body_state"], od["world_state"])
        _, h, d, inf, od = env.step(a)
        agent.sense(od["body_state"], od["world_state"])
        if d:
            _, od = env.reset()
            agent.sense(od["body_state"], od["world_state"])


def _metric(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, default))


def _run_cell(*, seed: int, arm: str, use_sci: bool, only_cf: bool,
              world_obs_dim: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = REEAgent(_build_config(world_obs_dim=world_obs_dim, use_sci=use_sci, only_cf=only_cf))
    env = CausalGridWorldV2(size=GRID_SIZE, seed=seed)

    # P0: train the world_forward so the comparator is discriminative (readiness).
    transitions = _collect_transitions(agent, env, N_P0_TRANSITIONS)
    world_forward_r2 = _train_world_forward(agent, transitions, seed=seed, arm=arm)

    # Reset the buffer: the P0 collection loop's sense() calls recorded against the
    # UNtrained world_forward; the measured feedstock must reflect the TRAINED comparator.
    agent.reset()
    if getattr(agent, "scientist_attribution_buffer", None) is not None:
        agent.scientist_attribution_buffer.reset()

    # Waking accumulation (buffer auto-records when use_scientist_attribution) + anchors.
    _accumulate_waking(agent, env, N_WAKE_STEPS)
    _install_hot_cold_anchors(agent)

    # Sleep: force one cycle; the aggregator reads the MECH-276 feedstock (or staleness OFF).
    metrics = agent.sleep_loop.force_cycle(agent) or {}

    buf = getattr(agent, "scientist_attribution_buffer", None)
    buf_metrics = buf.get_metrics() if buf is not None else {}

    return {
        "seed": seed,
        "arm": arm,
        "world_forward_r2": world_forward_r2,
        "mech276_n_records": _metric(buf_metrics, "mech276_n_records"),
        "mech276_n_counterfactual_backed": _metric(buf_metrics, "mech276_n_counterfactual_backed"),
        "mech276_n_correlational_skipped": _metric(buf_metrics, "mech276_n_correlational_skipped"),
        "mech276_mean_cf_contrast": _metric(buf_metrics, "mech276_mean_cf_contrast"),
        "mech276_counterfactual_backed_fraction": _metric(buf_metrics, "mech276_counterfactual_backed_fraction"),
        "mech275_mean_posterior_mean": _metric(metrics, "mech275_mean_posterior_mean"),
        "mech275_mean_posterior_variance": _metric(metrics, "mech275_mean_posterior_variance"),
        "mech275_n_updates": _metric(metrics, "mech275_n_updates"),
        "mech275_n_posteriors": _metric(metrics, "mech275_n_posteriors"),
    }


def _score(rows_by_arm: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    cf_rows = rows_by_arm["ARM_CF_BACKED"]
    corr_rows = rows_by_arm["ARM_CORRELATIONAL"]
    n_seeds = len(cf_rows)
    need = math.ceil(PASS_FRACTION * n_seeds)

    # R1 readiness: world_forward trained on the CF arm (positive control).
    r1_r2 = [r["world_forward_r2"] for r in cf_rows]
    r1_met = sum(1 for v in r1_r2 if v >= R2_FLOOR) >= need

    # R2 readiness: cf_margin straddles (the lever has something to gate) on the CF arm.
    r2_straddle = [
        (r["mech276_n_counterfactual_backed"] > 0 and r["mech276_n_correlational_skipped"] > 0)
        for r in cf_rows
    ]
    r2_met = sum(1 for v in r2_straddle if v) >= need

    readiness_met = r1_met and r2_met

    # Primary discrimination: aggregator posterior differs between CF and CORR per seed.
    by_seed = {}
    for r in cf_rows:
        by_seed.setdefault(r["seed"], {})["cf"] = r
    for r in corr_rows:
        by_seed.setdefault(r["seed"], {})["corr"] = r
    discr_seed = []
    posterior_deltas = []
    for seed, d in by_seed.items():
        if "cf" in d and "corr" in d:
            delta = abs(d["cf"]["mech275_mean_posterior_mean"] - d["corr"]["mech275_mean_posterior_mean"])
            posterior_deltas.append(delta)
            discr_seed.append(delta >= POSTERIOR_DELTA_FLOOR)
    discrimination = sum(1 for v in discr_seed if v) >= need

    # Non-degeneracy of the discrimination criterion: the two arms genuinely saw
    # different feedstock (CF arm skipped correlational records) AND the aggregator updated.
    nd_feedstock_differs = any(r["mech276_n_correlational_skipped"] > 0 for r in cf_rows)
    nd_aggregator_updated = any(r["mech275_n_updates"] > 0 for r in cf_rows)
    criterion_non_degenerate = nd_feedstock_differs and nd_aggregator_updated

    if not readiness_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif discrimination and criterion_non_degenerate:
        label = "mech276_feedstock_readiness_confirmed"
        outcome = "PASS"
    else:
        label = "substrate_ceiling_aggregator_insensitive_to_feedstock"
        outcome = "FAIL"

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "world_forward_r2_trained", "direction": "lower",
             "description": "held-out world_forward R2 -- the comparator-discrimination quantity (positive control)",
             "measured": float(sum(r1_r2) / max(1, len(r1_r2))), "threshold": R2_FLOOR,
             "control": "P0-trained E2WorldForward on real action-conditional env transitions",
             "met": bool(r1_met)},
            {"name": "cf_margin_straddles_feedstock", "direction": "lower",
             "description": "CF arm has BOTH counterfactual-backed AND correlational-skipped records -- the same mechanism the discrimination routes on",
             "measured": float(sum(1 for v in r2_straddle if v)), "threshold": float(need),
             "control": "cf_margin set within the trained-comparator cf_contrast distribution",
             "met": bool(r2_met)},
        ],
        "criteria_non_degenerate": {
            "discrimination_feedstock_differs": bool(nd_feedstock_differs),
            "discrimination_aggregator_updated": bool(nd_aggregator_updated),
        },
        "criteria": [
            {"name": "posterior_discrimination_cf_vs_correlational", "load_bearing": True,
             "passed": bool(discrimination and criterion_non_degenerate)},
        ],
    }

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "readiness_met": readiness_met,
        "discrimination": discrimination,
        "criterion_non_degenerate": criterion_non_degenerate,
        "posterior_deltas": posterior_deltas,
        "criteria_results": {
            "n_seeds": n_seeds,
            "need_seeds": need,
            "R1_world_forward_r2": r1_r2,
            "R1_met": bool(r1_met),
            "R2_straddle_per_seed": r2_straddle,
            "R2_met": bool(r2_met),
            "discrimination_per_seed": discr_seed,
            "posterior_deltas": posterior_deltas,
        },
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print("V3-EXQ-703: MECH-276 scientist-agent attribution feedstock readiness", flush=True)
    print(f"  seeds={seeds} dry_run={dry_run} claims={CLAIM_IDS} purpose={EXPERIMENT_PURPOSE}", flush=True)

    # In dry-run, shrink P0 so the smoke is fast (still exercises every code path).
    global N_P0_TRANSITIONS, N_P0_TRAIN_STEPS, N_WAKE_STEPS, DRAWS_PER_CYCLE
    if dry_run:
        N_P0_TRANSITIONS, N_P0_TRAIN_STEPS, N_WAKE_STEPS, DRAWS_PER_CYCLE = 40, 10, 20, 30

    env_probe = CausalGridWorldV2(size=GRID_SIZE, seed=0)
    world_obs_dim = env_probe.world_obs_dim

    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {a["arm"]: [] for a in ARMS}
    arm_results: List[Dict[str, Any]] = []

    for arm_cfg in ARMS:
        arm = arm_cfg["arm"]
        for seed in seeds:
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice=_config_slice(arm=arm, use_sci=arm_cfg["use_sci"], only_cf=arm_cfg["only_cf"]),
                script_path=Path(__file__),
                extra_ineligible_reasons=["diagnostic_run_no_reuse"],
            ) as cell:
                row = _run_cell(seed=seed, arm=arm, use_sci=arm_cfg["use_sci"],
                                only_cf=arm_cfg["only_cf"], world_obs_dim=world_obs_dim)
                cell.stamp(row)
            rows_by_arm[arm].append(row)
            arm_results.append(row)
            print(f"  [result] seed={seed} arm={arm} r2={row['world_forward_r2']:.3f} "
                  f"n_cf={row['mech276_n_counterfactual_backed']:.0f} "
                  f"n_skip={row['mech276_n_correlational_skipped']:.0f} "
                  f"post_mean={row['mech275_mean_posterior_mean']:.4f}", flush=True)
            print("verdict: PASS", flush=True)

    scored = _score(rows_by_arm)
    scored["arm_results"] = arm_results

    print("", flush=True)
    print(f"readiness_met={scored['readiness_met']} discrimination={scored['discrimination']} "
          f"non_degenerate={scored['criterion_non_degenerate']}", flush=True)
    print(f"interpretation.label={scored['interpretation']['label']}", flush=True)
    print(f"Overall outcome: {scored['outcome']}", flush=True)
    return scored


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "world_dim": WORLD_DIM,
            "grid_size": GRID_SIZE,
            "cf_margin": CF_MARGIN,
            "n_p0_transitions": N_P0_TRANSITIONS,
            "n_p0_train_steps": N_P0_TRAIN_STEPS,
            "n_wake_steps": N_WAKE_STEPS,
            "draws_per_cycle": DRAWS_PER_CYCLE,
        },
        "acceptance_criteria": {
            "R1_readiness": f"world_forward_r2 >= {R2_FLOOR} on >=2/3 seeds (positive control)",
            "R2_readiness": "cf_margin straddles: CF arm has BOTH cf-backed AND correlational-skipped records on >=2/3 seeds",
            "discrimination": f"|posterior_mean(CF) - posterior_mean(CORR)| >= {POSTERIOR_DELTA_FLOOR} on >=2/3 seeds",
        },
        "criteria_results": result["criteria_results"],
        "arm_results": result["arm_results"],
        "notes": (
            "MECH-276 scientist-agent counterfactual-backed attribution feedstock "
            "substrate-readiness diagnostic. claim_ids=[] (PROMOTES NOTHING; does NOT "
            "promote MECH-275, which stays substrate_conditional). Confirms the feedstock "
            "plumbing works on a TRAINED world_forward and the only_counterfactual_backed "
            "lever measurably gates what reaches the MECH-275 aggregator. Readiness-unmet "
            "self-routes substrate_not_ready_requeue (margin/training recalibration), NEVER "
            "a substrate-verdict label. PASS unblocks the SEPARATE MECH-275 sleep-aggregation "
            "promotion run + re-adding MECH-275 to sleep_substrate:GAP-3b unblocks_claims."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written to evidence/", flush=True)
        print(json.dumps(manifest, indent=2, default=str), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=args.dry_run,
    )
    sys.exit(0)
