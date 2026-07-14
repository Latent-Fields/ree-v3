"""V3-EXQ-761 -- MECH-092 quiescent-E3 SWR-equivalent replay selectivity (wall-independent functional signature).

WHAT / WHY
==========
MECH-092: "Quiescent E3 heartbeat cycles trigger hippocampal SWR-equivalent
replay." lit_conf 0.868, exp_conf 0.0, genuine_exp_count 0. Surfaced by the
GOV-CONFIRM-1 evidence-confirmer detector (2026-07-14) as a confirmable-but-
unconfirmed candidate with BUILT substrate. This run empirically CHECKS the
detector's precision: if the confirming DV turns out non-viable in the built
substrate, it SELF-ROUTES substrate_not_ready_requeue (a valid detector-
validation outcome, NOT a false weakens).

WALL-INDEPENDENT: the DV is a read-only FUNCTIONAL-SIGNATURE (replay events
firing selectively on quiescent E3 cycles). It does NOT read behavioural
competence / reward, so it passes independent of the V3 competence wall
(the V3-EXQ-752..756 attack that). The agent is UNTRAINED -- the MECH-092
wiring is present at init, so no training is required (reinforcing wall-
independence; precedent V3-EXQ-455/447/448 functional DVs).

SUBSTRATE PATH (verified in ree-v3, 2026-07-14)
-----------------------------------------------
- clock (ree_core/heartbeat/clock.py:advance): e3_quiescent = E3 ticked AND no
  salient event this cycle (MECH-092). Salient events are marked ONLY via
  clock.phase_reset(), fired from agent.update_residue when harm_signal < 0
  (agent.py:8020, MECH-091). So harm events drive the SALIENT (non-quiescent)
  population; harm-free cycles are QUIESCENT.
- agent.act_with_split_obs (agent.py:7779): the production loop -- advances the
  clock and, iff ticks["e3_quiescent"], calls agent._do_replay(latent) which
  calls hippocampal.replay(theta_buffer.recent) (agent.py:7811). NOTE the
  StepHarness driver does NOT include this branch, so this run drives the agent
  via act_with_split_obs directly (the real MECH-092 path).
- hippocampal.replay (module.py:1402): returns num_replay_steps (=5) rollout
  Trajectory objects when theta_buffer.recent is non-empty, else []. The theta
  buffer is filled every step in _e1_tick (agent.py:4504), so replay produces
  real (non-no-op) SWR-equivalent events from the first quiescent tick onward.

DESIGN (read-only, action-free DV)
----------------------------------
Wrap agent.clock.advance (capture every per-step tick dict) and
hippocampal.replay / diverse_replay (record fire + n_trajectories per call,
keyed by the clock global_step). Drive the untrained agent through a hazard
grid; feed the env harm_signal to update_residue each step so harm marks
salient cycles. Then cross-tabulate replay firing against the E3 tick class.

The replay-fires-iff-quiescent ALIGNMENT is code-structural (the `if
e3_quiescent` gate), so it is NOT the load-bearing content. The genuinely
falsifiable, non-structural facts MECH-092 asserts -- and that this run confirms
-- are:
  C1  quiescent E3 cycles genuinely OCCUR in an integrated run (non-vacuity).
  C2  salient (non-quiescent) E3 cycles ALSO occur, so the selectivity contrast
      is non-degenerate (a real OFF population).
  C3  replay produces REAL SWR-equivalent events (non-empty trajectories) when
      it fires -- the hippocampal module is functionally live, not a silent
      no-op (the case an untrained/misconfigured substrate could produce).
  C4  selectivity: replay fires on ~all quiescent E3 ticks and ~no salient /
      non-E3 steps (confirms the wiring is live end-to-end).

ROUTING (non-vacuity gate -- NEVER a false weakens)
---------------------------------------------------
- C1 & C2 & C3 & C4 all pass -> PASS / supports MECH-092.
- C1 or C2 or C3 fail (no quiescent cycles / no salient cycles / replay a
  no-op) -> FAIL, evidence_direction=non_contributory, non_degenerate=False,
  interpretation.label=substrate_not_ready_requeue. This is the valid detector-
  validation "confirming DV non-viable" outcome; it does NOT weaken MECH-092.
- C1 & C2 & C3 pass but C4 fails (replay misaligned with quiescent cycles) ->
  FAIL / weakens (a genuine refutation of the triggering wiring).

GOV-REUSE-1: the only prior MECH-092 manifest (v3_exq_136_..., 2026-03-29)
measured a DIFFERENT readout -- a downstream consolidation BENEFIT
(gap_replay_on vs gap_replay_ablated, the wall-BOUND consequence reading, which
FAILED/weakened). It did NOT record the triggering-signature selectivity, and
its substrate_hash is None (pre-standard, unverifiable). Decisive readout
absent -> run. This run tests the mechanism/TRIGGERING reading, distinct from
136's consequence reading; supersedes NOTHING.

Re-derive brake: 0 substrate_ceiling/non_contributory autopsies tag MECH-092.
No substrate build owed (hippocampal module + replay path exist and are armed by
default; this run explicitly asserts replay_diversity_enabled/serotonin/
surprise_gated_replay are OFF so the plain replay() path fires).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_761_mech092_quiescent_replay_selectivity"
QUEUE_ID = "V3-EXQ-761"
SUPERSEDES = None
CLAIM_IDS = ["MECH-092"]
EXPERIMENT_PURPOSE = "evidence"

# --- design constants (untrained agent -> fast, deterministic) ---
SEEDS = [0, 1, 2, 3, 4]
EPISODES_PER_SEED = 10
STEPS_PER_EP = 80
WORLD_DIM = 32
SELF_DIM = 32
ALPHA_WORLD = 0.9

# proximity_harm_scale=0.0 makes harm CONTACT-only (intermittent, ~7% of steps),
# so both quiescent (harm-free) and salient (harm) E3 cycles are well-represented.
# A dense proximity-harm regime would mark ~every cycle salient (no quiescent
# population) and correctly self-route substrate_not_ready_requeue.
ENV_KWARGS = {
    "size": 10,
    "num_hazards": 3,
    "num_resources": 2,
    "proximity_harm_scale": 0.0,
}

# --- pre-registered thresholds (defined here, not derived from run stats) ---
Q_MIN_PER_SEED = 3        # C1: min quiescent E3 ticks per seed (non-vacuity)
S_MIN_PER_SEED = 1        # C2: min salient E3 ticks per seed (non-degenerate contrast)
REPLAY_N_MIN = 1.0        # C3: min mean trajectories produced per quiescent replay call
FIRE_HI = 0.99            # C4: min replay-fire rate on quiescent E3 ticks
FIRE_LO = 0.01            # C4: max replay-fire rate on salient E3 ticks / non-E3 steps


def _build_agent(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Untrained agent + hazard env. All neuromodulatory replay branches OFF so
    the plain hippocampal.replay() path fires with drive_state=None (MECH-092
    baseline wiring)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=0.3,
        use_harm_stream=False,          # replay needs z_world (theta buffer), not z_harm
        replay_diversity_enabled=False,  # -> plain replay() path (not diverse_replay)
        surprise_gated_replay=False,     # -> drive_state stays None
        tonic_5ht_enabled=False,         # serotonin OFF -> drive_state stays None
    )
    agent = REEAgent(cfg)
    return agent, env


def _instrument(agent: REEAgent) -> Dict[str, Any]:
    """Wrap clock.advance + hippocampal.replay/diverse_replay to record, per
    step, the E3 tick class and any replay firing (keyed by global_step). This
    is a read-only observation of the real production path -- it does not alter
    control flow."""
    state: Dict[str, Any] = {"ticks": [], "replays": {}}

    orig_advance = agent.clock.advance

    def advance_wrap():
        tick = orig_advance()
        state["ticks"].append(dict(tick))
        return tick

    agent.clock.advance = advance_wrap  # type: ignore[assignment]

    def _record_replay(out):
        # Key by the MONOTONIC per-advance tick index, NOT clock.global_step:
        # the clock resets to global_step=0 on each agent.reset()/episode, so
        # global_step is not unique across episodes and would alias a quiescent
        # tick's replay onto a same-global_step non-E3 step in a later episode.
        idx = len(state["ticks"]) - 1 if state["ticks"] else -1
        n = len(out) if out is not None else 0
        # A step calls replay at most once; keep the max seen for that tick.
        prev = state["replays"].get(idx, 0)
        state["replays"][idx] = max(prev, n)
        return out

    orig_replay = agent.hippocampal.replay

    def replay_wrap(*a, **kw):
        return _record_replay(orig_replay(*a, **kw))

    agent.hippocampal.replay = replay_wrap  # type: ignore[assignment]

    orig_dreplay = agent.hippocampal.diverse_replay

    def dreplay_wrap(*a, **kw):
        return _record_replay(orig_dreplay(*a, **kw))

    agent.hippocampal.diverse_replay = dreplay_wrap  # type: ignore[assignment]

    return state


def _run_seed(seed: int, episodes: int, steps_per_ep: int) -> Dict[str, Any]:
    agent, env = _build_agent(seed)
    state = _instrument(agent)

    print(f"Seed {seed} Condition integrated", flush=True)
    for ep in range(episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_ep):
            with torch.no_grad():
                action = agent.act_with_split_obs(
                    obs_dict["body_state"], obs_dict["world_state"]
                )
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            # canonical post-action path: harm_signal < 0 -> clock.phase_reset
            # (MECH-091) -> next E3 cycle is salient (non-quiescent).
            agent.update_residue(
                harm_signal=float(harm_signal),
                world_delta=None,
                hypothesis_tag=False,
                owned=True,
            )
            if done:
                flat_obs, obs_dict = env.reset()
                agent.reset()
        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"  [rollout] seed={seed} ep {ep + 1}/{episodes}",
                flush=True,
            )

    # --- cross-tabulate replay firing against E3 tick class ---
    replays = state["replays"]
    n_steps = len(state["ticks"])
    n_quiescent = n_quiescent_fired = 0
    n_salient = n_salient_fired = 0
    n_nonE3 = n_nonE3_fired = 0
    quiescent_replay_ns: List[int] = []
    for idx, tick in enumerate(state["ticks"]):
        n = replays.get(idx, 0)
        fired = n > 0
        if tick.get("e3_tick", False) and tick.get("e3_quiescent", False):
            n_quiescent += 1
            if fired:
                n_quiescent_fired += 1
                quiescent_replay_ns.append(n)
        elif tick.get("e3_tick", False):  # E3 tick but not quiescent = salient
            n_salient += 1
            if fired:
                n_salient_fired += 1
        else:  # non-E3 step
            n_nonE3 += 1
            if fired:
                n_nonE3_fired += 1

    mean_replay_n_quiescent = (
        float(np.mean(quiescent_replay_ns)) if quiescent_replay_ns else 0.0
    )
    return {
        "seed": seed,
        "n_steps": n_steps,
        "n_quiescent": n_quiescent,
        "n_quiescent_fired": n_quiescent_fired,
        "n_salient": n_salient,
        "n_salient_fired": n_salient_fired,
        "n_nonE3": n_nonE3,
        "n_nonE3_fired": n_nonE3_fired,
        "mean_replay_n_quiescent": mean_replay_n_quiescent,
    }


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    episodes = 2 if dry_run else EPISODES_PER_SEED
    steps_per_ep = 40 if dry_run else STEPS_PER_EP

    per_seed = [_run_seed(s, episodes, steps_per_ep) for s in seeds]

    # --- aggregate ---
    min_quiescent = min(r["n_quiescent"] for r in per_seed)
    min_salient = min(r["n_salient"] for r in per_seed)
    min_mean_replay_n = min(r["mean_replay_n_quiescent"] for r in per_seed)
    tot_q = sum(r["n_quiescent"] for r in per_seed)
    tot_q_fired = sum(r["n_quiescent_fired"] for r in per_seed)
    tot_s = sum(r["n_salient"] for r in per_seed)
    tot_s_fired = sum(r["n_salient_fired"] for r in per_seed)
    tot_ne = sum(r["n_nonE3"] for r in per_seed)
    tot_ne_fired = sum(r["n_nonE3_fired"] for r in per_seed)

    quiescent_fire_rate = _rate(tot_q_fired, tot_q)
    salient_fire_rate = _rate(tot_s_fired, tot_s)
    nonE3_fire_rate = _rate(tot_ne_fired, tot_ne)

    # --- pre-registered criteria ---
    c1_quiescent_occur = min_quiescent >= Q_MIN_PER_SEED
    c2_salient_occur = min_salient >= S_MIN_PER_SEED
    c3_replay_produces_events = min_mean_replay_n >= REPLAY_N_MIN
    c4_selective = (
        quiescent_fire_rate >= FIRE_HI
        and salient_fire_rate <= FIRE_LO
        and nonE3_fire_rate <= FIRE_LO
    )

    preconditions_met = c1_quiescent_occur and c2_salient_occur and c3_replay_produces_events

    if not preconditions_met:
        # non-vacuity gate: substrate did not produce the needed populations /
        # replay was a no-op. NEVER a weakens.
        reasons = []
        if not c1_quiescent_occur:
            reasons.append(f"no_quiescent_cycles(min={min_quiescent}<{Q_MIN_PER_SEED})")
        if not c2_salient_occur:
            reasons.append(f"no_salient_cycles(min={min_salient}<{S_MIN_PER_SEED})")
        if not c3_replay_produces_events:
            reasons.append(f"replay_noop(min_mean_n={min_mean_replay_n:.2f}<{REPLAY_N_MIN})")
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = "; ".join(reasons)
        label = "substrate_not_ready_requeue"
    elif c4_selective:
        outcome = "PASS"
        evidence_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        label = "quiescent_replay_selectivity_confirmed"
    else:
        # valid, non-degenerate test but replay firing misaligned with quiescent
        # cycles -> genuine refutation of the MECH-092 triggering wiring.
        outcome = "FAIL"
        evidence_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""
        label = "selectivity_broken"

    criteria = {
        "C1_quiescent_cycles_occur": bool(c1_quiescent_occur),
        "C2_salient_cycles_occur": bool(c2_salient_occur),
        "C3_replay_produces_events": bool(c3_replay_produces_events),
        "C4_selective": bool(c4_selective),
        "min_quiescent_per_seed": int(min_quiescent),
        "min_salient_per_seed": int(min_salient),
        "min_mean_replay_n_quiescent": float(min_mean_replay_n),
        "quiescent_fire_rate": round(quiescent_fire_rate, 6),
        "salient_fire_rate": round(salient_fire_rate, 6),
        "nonE3_fire_rate": round(nonE3_fire_rate, 6),
    }

    # criteria list with load-bearing tags (auditability)
    criteria_list = [
        {"name": "C1_quiescent_cycles_occur", "load_bearing": True, "passed": bool(c1_quiescent_occur)},
        {"name": "C2_salient_cycles_occur", "load_bearing": True, "passed": bool(c2_salient_occur)},
        {"name": "C3_replay_produces_events", "load_bearing": True, "passed": bool(c3_replay_produces_events)},
        {"name": "C4_selective", "load_bearing": True, "passed": bool(c4_selective)},
    ]
    criteria_non_degenerate = {
        "C1_quiescent_cycles_occur": bool(min_quiescent > 0),
        "C2_salient_cycles_occur": bool(min_salient > 0),
        "C3_replay_produces_events": bool(min_mean_replay_n > 0),
        "C4_selective": bool(tot_q > 0 and (tot_s + tot_ne) > 0),
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "label": label,
        "preconditions_met": bool(preconditions_met),
        "criteria": criteria,
        "criteria_list": criteria_list,
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_seed": per_seed,
        "n_cells": len(per_seed),
    }


def _write_manifest(result: Dict[str, Any], started_at: float) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    full_config = {
        "seeds": SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "steps_per_ep": STEPS_PER_EP,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "alpha_world": ALPHA_WORLD,
        "env_kwargs": ENV_KWARGS,
        "replay_diversity_enabled": False,
        "surprise_gated_replay": False,
        "tonic_5ht_enabled": False,
        "use_harm_stream": False,
        "thresholds": {
            "Q_MIN_PER_SEED": Q_MIN_PER_SEED,
            "S_MIN_PER_SEED": S_MIN_PER_SEED,
            "REPLAY_N_MIN": REPLAY_N_MIN,
            "FIRE_HI": FIRE_HI,
            "FIRE_LO": FIRE_LO,
        },
    }
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "exp:simulation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "thresholds": full_config["thresholds"],
        "criteria": result["criteria"],
        "per_seed": result["per_seed"],
        "interpretation": {
            "label": result["label"],
            "preconditions": [
                {
                    "name": "quiescent_cycles_occur",
                    "description": "quiescent E3 ticks occur (non-vacuity gate 1)",
                    "measured": result["criteria"]["min_quiescent_per_seed"],
                    "threshold": Q_MIN_PER_SEED,
                    "met": result["criteria"]["C1_quiescent_cycles_occur"],
                    "control": "min over seeds of quiescent E3 tick count",
                },
                {
                    "name": "salient_cycles_occur",
                    "description": "salient (non-quiescent) E3 ticks occur so the selectivity contrast has an OFF population",
                    "measured": result["criteria"]["min_salient_per_seed"],
                    "threshold": S_MIN_PER_SEED,
                    "met": result["criteria"]["C2_salient_cycles_occur"],
                    "control": "min over seeds of salient E3 tick count (harm-driven)",
                },
                {
                    "name": "replay_produces_events",
                    "description": "replay() returns real SWR-equivalent trajectories on quiescent ticks (not a no-op)",
                    "measured": result["criteria"]["min_mean_replay_n_quiescent"],
                    "threshold": REPLAY_N_MIN,
                    "met": result["criteria"]["C3_replay_produces_events"],
                    "control": "min over seeds of mean trajectories per quiescent replay call",
                },
            ],
            "criteria_non_degenerate": result["criteria_non_degenerate"],
            "criteria": result["criteria_list"],
        },
        "deliverable_note": (
            "Wall-independent CONFIRMING test of MECH-092's TRIGGERING signature: "
            "hippocampal SWR-equivalent replay fires selectively on quiescent E3 "
            "heartbeat cycles. Read-only functional-signature DV (no behavioural / "
            "reward read) driven through the real production path "
            "(act_with_split_obs -> _do_replay -> hippocampal.replay) on an "
            "UNTRAINED agent, so it passes independent of the V3 competence wall. "
            "The replay-fires-iff-quiescent alignment (C4) is code-structural; the "
            "load-bearing, falsifiable content is C1 (quiescent cycles occur), C2 "
            "(salient cycles also occur -> non-degenerate contrast), C3 (replay "
            "produces real events, not a no-op). A degenerate substrate (no "
            "quiescent cycles / no salient cycles / replay no-op) self-routes "
            "substrate_not_ready_requeue (non_contributory), NEVER a false weakens. "
            "Spawned to empirically check GOV-CONFIRM-1 detector precision; a "
            "non-viable confirming DV here is a valid detector-validation outcome."
        ),
        "gov_reuse_1_note": (
            "Decisive readout (quiescent-vs-salient replay-event selectivity) "
            "absent from the only prior MECH-092 manifest (v3_exq_136_..., "
            "2026-03-29), which measured a downstream consolidation BENEFIT "
            "(gap_replay_on/ablated; the wall-BOUND consequence reading, "
            "FAILED/weakened) and carries no substrate_hash (unverifiable). New "
            "readout on the triggering reading -> run. Supersedes nothing."
        ),
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest, out_dir, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=started_at,
    )
    print(f"Wrote manifest: {out_path}", flush=True)
    return out_path


def main(dry_run: bool = False) -> Tuple[str, Any]:
    started_at = time.perf_counter()
    result = run_experiment(dry_run=dry_run)

    c = result["criteria"]
    print("=== MECH-092 quiescent-replay selectivity (761) ===", flush=True)
    print(
        f"  C1 quiescent_occur:  {c['C1_quiescent_cycles_occur']} "
        f"(min_quiescent/seed={c['min_quiescent_per_seed']} >= {Q_MIN_PER_SEED})",
        flush=True,
    )
    print(
        f"  C2 salient_occur:    {c['C2_salient_cycles_occur']} "
        f"(min_salient/seed={c['min_salient_per_seed']} >= {S_MIN_PER_SEED})",
        flush=True,
    )
    print(
        f"  C3 replay_events:    {c['C3_replay_produces_events']} "
        f"(min_mean_n={c['min_mean_replay_n_quiescent']:.2f} >= {REPLAY_N_MIN})",
        flush=True,
    )
    print(
        f"  C4 selective:        {c['C4_selective']} "
        f"(quiescent_fire={c['quiescent_fire_rate']:.4f}>= {FIRE_HI}, "
        f"salient_fire={c['salient_fire_rate']:.4f}<= {FIRE_LO}, "
        f"nonE3_fire={c['nonE3_fire_rate']:.4f}<= {FIRE_LO})",
        flush=True,
    )
    print(
        f"  OUTCOME: {result['outcome']} (direction={result['evidence_direction']}, "
        f"label={result['label']})",
        flush=True,
    )
    # one verdict line per seed x condition (progress instrumentation contract)
    passed = result["outcome"] == "PASS"
    for _ in result["per_seed"]:
        print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    if dry_run:
        print(f"DRY_RUN complete: {result['n_cells']} cells, pipeline OK", flush=True)
        return "PASS", None

    out_path = _write_manifest(result, started_at)
    _outcome_raw = str(result["outcome"]).upper()
    outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
        dry_run=args.dry_run,
    )
