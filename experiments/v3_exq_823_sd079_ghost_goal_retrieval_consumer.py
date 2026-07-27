"""V3-EXQ-823 -- SD-079 DOWNSTREAM CONSUMER: does centering the z_goal cue let the
ghost-goal bank DISCRIMINATE anchors well enough to make a reliable goal-directed
retrieval DECISION and to run the MECH-340 rumination guard?

PROMOTION GATE (experiment_purpose = evidence). SD-079 is candidate_substrate_landed:
its INTERNAL cue geometry was validated in isolation by V3-EXQ-807 (goal_match spread
0.009 raw -> 0.999 centered; the MECH-292 floor and MECH-339 outshining gate unpin).
What that run did NOT establish is that fixing the cue changes an AGENT-LEVEL
decision. This run supplies exactly that missing link and is the gate to promoting
SD-079 candidate_substrate_landed -> provisional. It does NOT edit claims.yaml; the
run + review does.

WHAT IS NEW OVER V3-EXQ-807 (why this is not recoverable from it). 807 recorded the
INTERNAL readouts goal_match_spread + n_below_floor + gate-nonzero count. It did NOT
record the DECISION the bank exists to make: given a current goal drive, WHICH stored
goal-anchor does the agent identify as the one to pursue, and how confidently (the
retrieval MARGIN that a downstream approach/disengage commitment would rest on)? That
margin, and the rumination-guard DISENGAGEMENT efficacy (does the goal_match_floor
correctly exclude the NON-matching anchors), are the agent-level consequences and are
absent from every 807 manifest. GOV-REUSE-1 (Step 2.4): the decisive readouts are
retrieval-margin reliability and per-query rumination-guard efficacy under a CENTERED
z_goal cue; goal_cue_centering did not exist before 2026-07-22 and 807 recorded neither
readout, so no manifest on any substrate_hash can carry them. Not recoverable -> run.

THE BEHAVIOURAL BOTTLENECK. z_goal is an EMA attractor pulled toward z_world, so under
SD-008 it inherits the common-mode offset MORE strongly than its source (807: pairwise
cosine min 0.9878). Every stored goal-anchor then looks the same to a raw cosine, so
the bank cannot tell which goal the current drive corresponds to -- goal-directed
behaviour is UNDIRECTED (the agent cannot commit to the right goal) and the rumination
guard cannot disengage from stale ghost goals (n_below_floor = 0). Centering subtracts
the common mode so the residuals separate; the retrieval decision becomes reliable and
the guard can exclude mismatches.

DESIGN. Two arms, same seeds; single swept variable = goal_cue_centering (ARM_OFF
False = the pinned SD-079 defect, ARM_ON True). The whole agent is driven (forced
benefit + drive, as 807) to produce a z_goal stream; K goal-anchors are enrolled from
that stream. The enrollment writes advance the AnchorSet common-mode baseline; that
baseline is then FROZEN for scoring (faithful to intended operation -- the SD-079
baseline is a slow population common mode, NOT a per-query attractor; scoring queries in
a tight loop pulls it toward each query and zeros every residual, the SD-079 C6
write-path hazard). Then, for EACH enrolled anchor k, the whole pool is scored with
anchor k's own z_goal snapshot through Anchor.goal_match (the exact centered-cosine
primitive the GhostGoalBank and the MECH-340 rumination guard consume), and the
retrieval decision is scored:
  * discrimination_frac_k = fraction of the OTHER anchors the queried goal clearly
    outranks (goal_match(self=k) - goal_match(j) > MARGIN). How much of the pool the
    current goal stands apart from -- the discriminability an approach/disengage commit
    rests on. PRIMARY behavioural-decision DV. Robust to a few near-duplicate enrolled
    snapshots (a twin stays near self, but the BULK of the pool separates), unlike a
    margin-to-single-best-competitor which one near-twin can pin to ~0.
  * guard_efficacy_k = fraction of the OTHER anchors that fall below goal_match_floor
    when querying with k (correct rumination-guard DISENGAGEMENT), scored only when the
    self anchor itself stays above the floor. MECH-340 efficacy DV.
The real GhostGoalBank is also exercised (bank.rank) for a recorded n_below_floor
diagnostic. Self-match is ~1.0 for a non-central anchor in BOTH arms (SD-079 contract
C5: raw-snapshot self-match), so a broken retrieval mechanism is NOT the confound; the
ONLY thing that differs between arms is whether the NON-self anchors are separable,
which is exactly the discrimination centering restores.

READINESS (self-route substrate_not_ready_requeue, never a false does_not_support). The
z_goal common-mode cone SD-079 exists to repair must actually be present in this run:
min pairwise z_goal cosine >= CONE_MIN_COSINE_FLOOR (the SAME statistic the hazard is
DEFINED by, measured on the run's own z_goal stream -- not a magnitude proxy). Below it
there is no common mode to subtract and a null says nothing about centering. A minimum
enrolled-pool size is also required (a retrieval decision over < MIN_POOL anchors is
degenerate).

DV-SYMMETRY (Step 3 mandatory declaration, per arm). ARM_OFF / ARM_ON DV =
discrimination_frac, guard_efficacy (and margin as a diagnostic). Symmetry group:
permutation of the anchor pool (all are symmetric functions over the pool) and, for the
threshold counts against the FIXED constants MARGIN / goal_match_floor, any monotone
reparameterisation of goal_match that preserves each anchor's position relative to those
fixed thresholds. The manipulation is NOT invariant under either: subtracting a
common-mode baseline from both operands of the cosine is not a monotone map on goal_match
(it rotates the residuals, so the anchor ORDER -- hence which anchors clear the gap, hence
discrimination_frac -- can change) and it moves absolute goal_match values across the
fixed thresholds, which is precisely what the two fractions count. The thresholds are
absolute constants, not run-derived, so the counts cannot be satisfied by rescaling.

EVIDENCE DIRECTIONS (both declared):
  PASS  (readiness met + C1 paired discrimination lift + C2 OFF pinned + C3 guard live) ->
        SUPPORTS SD-079: centering makes the ghost-goal retrieval decision discriminate
        and the rumination guard functional -- a real downstream behavioural consequence.
  FAIL, readiness MET (C1/C3 fail) -> DOES_NOT_SUPPORT: the cue geometry is fixed (807)
        but the retrieval DECISION does not become reliable / the guard stays dead --
        the downstream benefit is absent and the promotion is not warranted on this DV.
  FAIL, readiness UNMET (cone absent / pool too small) -> substrate_not_ready_requeue,
        non_contributory (non_degenerate=false); re-queue at an adequate regime. Never
        a weakens.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig, AnchorSetConfig, GhostGoalBankConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.hippocampal.anchor_set import AnchorSet, AnchorGoalPayload  # noqa: E402
from ree_core.hippocampal.ghost_goal_bank import GhostGoalBank  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_823_sd079_ghost_goal_retrieval_consumer"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-079"]

SEEDS = [101, 202, 303]
ARMS = ["ARM_OFF", "ARM_ON"]
N_EPISODES = 8
STEPS_PER_EPISODE = 32
ANCHOR_EVERY = 4          # enroll a goal-anchor every N steps
FORCED_BENEFIT = 0.5
FORCED_DRIVE = 0.9

# Bank defaults this run exercises (absolute constants, NOT run-derived).
GOAL_MATCH_FLOOR = 0.05
OUTSHINE_PIVOT = 0.5

# Pre-registered thresholds (defined HERE, never inferred post-hoc).
# discrimination = fraction of OTHER anchors the queried goal clearly outranks
# (self_goal_match - other_goal_match > MARGIN). Robust to a few near-duplicate
# enrolled snapshots (a twin stays near self, but the BULK of the pool separates) --
# unlike a margin-to-single-best-competitor, which one near-twin can pin to ~0.
MARGIN = 0.10                # a goal_match gap above this = the goal clearly outranks j
# C1 is a paired-by-seed EFFECT SIZE (centering's improvement) plus an absolute floor,
# per the effect-size gate convention (scale on the delta, not a bare absolute cut that
# a borderline level could straddle). OFF is reliably ~0 discrimination, so the delta is
# the causal quantity; the absolute floor guards against a vacuously tiny ON level.
DISCR_DELTA_MARGIN = 0.15    # C1: ON - OFF discrimination_frac must exceed this (paired)
DISCR_FLOOR = 0.20           # C1: AND ON discrimination_frac must clear this absolute floor
OFF_DISCR_CEIL = 0.10        # C2: OFF must reproduce the pinned indiscriminate retrieval
GUARD_FLOOR = 0.30           # C3: ON rumination guard excludes >= this frac of mismatches
MIN_POOL = 6                 # readiness: retrieval over fewer anchors is degenerate
SEED_PASS_FRACTION = 2.0 / 3.0

# READINESS: the z_goal common-mode cone must be present for SD-079 to have anything to
# repair. SAME statistic the hazard is defined by (min pairwise cosine).
CONE_MIN_COSINE_FLOOR = 0.90


# z_goal-stream liveness, pooled across arm x seed cells for the manifest block.
_ZG = ZGoalStreamAccumulator()


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE,
                     "anchor_every": ANCHOR_EVERY},
        "alpha_world": 0.9,
        "z_goal_enabled": True,
        "drive_weight": 2.0,
        "benefit_eval_enabled": True,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
        "goal_cue_centering": centering,
        "goal_cue_baseline_alpha": 0.05,
        "use_composite_cue_outshining": True,
        "outshine_pivot": OUTSHINE_PIVOT,
        "goal_match_floor": GOAL_MATCH_FLOOR,
        # DV thresholds folded in so the config slice fully describes the run.
        "margin": MARGIN,
    }


def _cone_min_cosine(vecs: List[torch.Tensor]) -> float:
    if len(vecs) < 2:
        return 1.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].min())


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    centering = (arm == "ARM_ON")
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(centering),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
            action_dim=4, alpha_world=0.9, z_goal_enabled=True, drive_weight=2.0,
            benefit_eval_enabled=True, benefit_weight=1.0,
            use_sd039_anchor_payload=True,
            goal_cue_centering=centering, goal_cue_baseline_alpha=0.05,
        )
        agent = REEAgent(cfg)
        wd = agent.config.latent.world_dim

        aset = AnchorSet(AnchorSetConfig(
            use_sd039_anchor_payload=True,
            goal_cue_centering=centering,
            goal_cue_baseline_alpha=0.05,
        ))
        bank = GhostGoalBank(
            GhostGoalBankConfig(
                goal_match_floor=GOAL_MATCH_FLOOR,
                use_composite_cue_outshining=True,
                context_weight=1.0,
                outshine_pivot=OUTSHINE_PIVOT,
                include_inactive=True,
                include_active=True,
            ),
            aset,
        )

        # ENROLL: drive the agent, write a goal-anchor every ANCHOR_EVERY steps, and
        # remember (anchor object, its z_goal snapshot) so we know the ground-truth
        # target for each retrieval query.
        zgoals: List[torch.Tensor] = []
        enrolled: List[Dict[str, Any]] = []   # {"anchor": Anchor, "snap": tensor}
        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            for t in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                cands = agent.generate_trajectories(latent, e1, ticks)
                action = agent.select_action(cands, ticks)
                agent.update_z_goal(benefit_exposure=FORCED_BENEFIT,
                                    drive_level=FORCED_DRIVE)
                zg = agent.goal_state.z_goal if agent.goal_state is not None else None
                if zg is not None:
                    zg = zg.detach().reshape(-1).clone()
                    zgoals.append(zg)
                    if t % ANCHOR_EVERY == 0:
                        anchor = aset.write_anchor(
                            "fast", f"e{ep}s{t}", ("world",),
                            z_world=latent.z_world.detach().reshape(-1),
                            goal_payload=AnchorGoalPayload(
                                z_goal_snapshot=zg.clone(), wanting_strength=0.5),
                        )
                        enrolled.append({"anchor": anchor, "snap": zg.clone()})
                _, _h, done, _, obs = env.step(int(action.argmax(dim=-1).item()))
                if done:
                    break
            if (ep + 1) % 4 == 0:
                print(f"  [train] goal seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"enrolled={len(enrolled)}", flush=True)

        # Establish the common-mode baseline from ENROLLMENT (write_anchor advanced it
        # on each waking write cue), then FREEZE it for scoring. This is faithful to
        # intended operation -- the SD-079 baseline is a slow EMA population common mode,
        # not a per-query attractor -- and it AVOIDS the write-path hazard the SD-079
        # entry warns about: calling rank/query in a tight loop pulls the baseline toward
        # each query, driving every residual (and every goal_match) to 0 (SD-079 C6).
        # None on the OFF arm -> Anchor.goal_match is the raw cosine (identity).
        frozen_baseline = (aset.goal_cue_baseline.detach().clone()
                           if aset.goal_cue_baseline is not None else None)
        pool = aset.all_anchors()
        n_pool = len(pool)

        # RETRIEVAL DECISION: for each enrolled goal-anchor k, query with its own z_goal
        # snapshot and score the whole pool through Anchor.goal_match (the exact centered-
        # cosine primitive the GhostGoalBank and the MECH-340 rumination guard consume),
        # against the frozen common-mode baseline.
        discr_fracs: List[float] = []
        guard_effs: List[float] = []
        margins: List[float] = []
        self_scores: List[float] = []
        n_reliable = 0
        for item in enrolled:
            target = item["anchor"]
            query = item["snap"]
            self_score = None
            other_scores: List[float] = []
            for anchor in pool:
                s = anchor.goal_match(query, baseline=frozen_baseline)
                if anchor is target:
                    self_score = float(s)
                else:
                    other_scores.append(float(s))
            if self_score is None or not other_scores:
                continue
            self_scores.append(self_score)
            margins.append(self_score - max(other_scores))
            # PRIMARY: fraction of the pool the queried goal clearly outranks.
            discr = sum(1 for s in other_scores if (self_score - s) > MARGIN) / len(other_scores)
            discr_fracs.append(discr)
            if discr >= 0.5:
                n_reliable += 1
            # MECH-340 rumination-guard DISENGAGEMENT efficacy: with the self anchor above
            # the floor (the goal is RETAINED), what fraction of the OTHERS is correctly
            # excluded below the goal_match_floor (disengaged).
            if self_score >= GOAL_MATCH_FLOOR:
                n_excl = sum(1 for s in other_scores if s < GOAL_MATCH_FLOOR)
                guard_effs.append(n_excl / len(other_scores))

        n_queries = len(discr_fracs)
        discrimination_frac = statistics.fmean(discr_fracs) if discr_fracs else 0.0
        reliable_frac = (n_reliable / n_queries) if n_queries else 0.0
        guard_efficacy = statistics.fmean(guard_effs) if guard_effs else 0.0

        # Exercise the REAL GhostGoalBank once for a recorded diagnostic (its n_below_floor
        # over the pool), on a representative query, without perturbing the scoring above.
        bank_n_below_floor = -1
        if enrolled:
            bank.rank(enrolled[-1]["snap"], simulation_mode=False)
            bank_n_below_floor = int((bank._last_diagnostics or {}).get("n_below_floor", -1))

        row = {
            "arm": arm,
            "seed": seed,
            "n_pool": int(n_pool),
            "n_queries": int(n_queries),
            "discrimination_frac": float(discrimination_frac),
            "reliable_frac": float(reliable_frac),
            "guard_efficacy": float(guard_efficacy),
            "margin_mean": float(statistics.fmean(margins)) if margins else 0.0,
            "self_score_mean": float(statistics.fmean(self_scores)) if self_scores else 0.0,
            "bank_n_below_floor": bank_n_below_floor,
            "goal_baseline_allocated": aset.goal_cue_baseline is not None,
            "zgoal_cone_min_cosine": _cone_min_cosine(zgoals[:200]),
            "n_zgoal_sampled": min(len(zgoals), 200),
        }
        cell.stamp(row)
    # z_goal liveness -- read AFTER the cell has stepped, then the agent is dropped.
    # z_goal liveness -- read AFTER this cell stepped; the agent is not retained.
    _ZG.observe(agent)
    passed = (row["discrimination_frac"] > DISCR_FLOOR) if centering else \
             (row["discrimination_frac"] <= OFF_DISCR_CEIL)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    off = [r for r in rows if r["arm"] == "ARM_OFF"]
    on = [r for r in rows if r["arm"] == "ARM_ON"]

    # READINESS 1: the common-mode cone SD-079 repairs must be present (worst cell).
    cone_worst, cone_cell = _worst_cell(rows, "zgoal_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR
    # READINESS 2: the enrolled pool must be large enough for a non-degenerate decision.
    pool_worst, pool_cell = _worst_cell(rows, "n_queries", mode="min")
    pool_ok = pool_worst >= MIN_POOL
    ready = cone_present and pool_ok

    # C1 is paired BY SEED: centering's discrimination lift over its own OFF control at
    # the same seed must clear DISCR_DELTA_MARGIN AND ON must clear the absolute floor.
    off_by_seed = {r["seed"]: r for r in off}
    c1_seed_pass = 0
    for r in on:
        o = off_by_seed.get(r["seed"])
        if o is None:
            continue
        delta = r["discrimination_frac"] - o["discrimination_frac"]
        if delta > DISCR_DELTA_MARGIN and r["discrimination_frac"] > DISCR_FLOOR:
            c1_seed_pass += 1
    frac_c1 = c1_seed_pass / len(on)
    frac_c2 = sum(1 for r in off if r["discrimination_frac"] <= OFF_DISCR_CEIL) / len(off)
    frac_c3 = sum(1 for r in on if r["guard_efficacy"] > GUARD_FLOOR) / len(on)
    c1 = frac_c1 >= SEED_PASS_FRACTION
    c2 = frac_c2 >= SEED_PASS_FRACTION
    c3 = frac_c3 >= SEED_PASS_FRACTION

    # Non-degeneracy: ON discrimination is attributable to centering only if the retrieval
    # mechanism itself works in BOTH arms (self-match ~1.0 -- so the difference is
    # discrimination of OTHERS, not a broken self-match), the OFF arm is genuinely
    # pinned + ran the OFF path (no baseline allocated), and the OFF guard is dead.
    self_match_ok = all(r["self_score_mean"] >= 0.5 for r in rows)
    off_pinned = all(r["discrimination_frac"] <= OFF_DISCR_CEIL for r in off)
    off_no_baseline = all(not r["goal_baseline_allocated"] for r in off)
    on_baseline = all(r["goal_baseline_allocated"] for r in on)
    off_guard_dead = all(r["guard_efficacy"] <= GUARD_FLOOR for r in off)

    overall = ready and c1 and c2 and c3
    if not ready:
        label = "substrate_not_ready_requeue"
        direction = "unknown"
    elif overall:
        label = "sd079_centering_enables_reliable_goal_retrieval_and_rumination_guard"
        direction = "supports"
    else:
        label = "sd079_centering_did_not_make_goal_retrieval_reliable"
        direction = "does_not_support"

    metrics = {
        "on_discrimination_frac_mean": statistics.fmean(r["discrimination_frac"] for r in on),
        "off_discrimination_frac_mean": statistics.fmean(r["discrimination_frac"] for r in off),
        "on_reliable_frac_mean": statistics.fmean(r["reliable_frac"] for r in on),
        "off_reliable_frac_mean": statistics.fmean(r["reliable_frac"] for r in off),
        "on_margin_mean": statistics.fmean(r["margin_mean"] for r in on),
        "off_margin_mean": statistics.fmean(r["margin_mean"] for r in off),
        "on_guard_efficacy_mean": statistics.fmean(r["guard_efficacy"] for r in on),
        "off_guard_efficacy_mean": statistics.fmean(r["guard_efficacy"] for r in off),
        "on_self_score_mean": statistics.fmean(r["self_score_mean"] for r in on),
        "off_self_score_mean": statistics.fmean(r["self_score_mean"] for r in off),
        "frac_c1_on_discrimination": frac_c1,
        "frac_c2_off_reproduces_pin": frac_c2,
        "frac_c3_on_guard_live": frac_c3,
        "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        "zgoal_cone_min_cosine_worst": cone_worst,
        "zgoal_cone_worst_cell": cone_cell,
        "n_queries_worst": pool_worst,
        "n_queries_worst_cell": pool_cell,
    }
    result: Dict[str, Any] = {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": direction,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zgoal_common_mode_cone_present",
                 "description": ("min pairwise z_goal cosine clears the cone floor -- the "
                                 "geometry SD-079 exists to repair. Below it there is no "
                                 "common mode to subtract and a null says nothing."),
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower",
                 "control": ("worst cell across all arms/seeds ({}), on the run's own "
                             "z_goal stream".format(cone_cell)),
                 "offending_cell": cone_cell,
                 "met": cone_present},
                {"name": "enrolled_pool_non_degenerate",
                 "description": ("a retrieval decision must range over at least MIN_POOL "
                                 "distinct goal-anchors to be meaningful."),
                 "measured": float(pool_worst), "threshold": float(MIN_POOL),
                 "direction": "lower",
                 "control": "worst cell across all arms/seeds ({})".format(pool_cell),
                 "offending_cell": pool_cell,
                 "met": pool_ok},
            ],
            "criteria": [
                {"name": "C1_on_retrieval_discriminates", "load_bearing": True, "passed": c1},
                {"name": "C2_off_reproduces_pin", "load_bearing": False, "passed": c2},
                {"name": "C3_on_rumination_guard_live", "load_bearing": True, "passed": c3},
            ],
            "criteria_non_degenerate": {
                "C1_on_retrieval_discriminates": bool(self_match_ok and off_pinned),
                "C2_off_reproduces_pin": bool(off_no_baseline and on_baseline),
                "C3_on_rumination_guard_live": bool(off_guard_dead and self_match_ok),
            },
        },
    }
    # Non-degeneracy net (applies to evidence runs): a readiness-unmet run measured
    # nothing usable about centering -> exclude from confidence scoring.
    if not ready:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = (
            "readiness unmet ({}): the z_goal common-mode cone SD-079 repairs is absent "
            "or the enrolled pool is too small, so the retrieval-decision DV says nothing "
            "about centering; substrate_not_ready_requeue".format(
                "cone" if not cone_present else "pool")
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, N_EPISODES
    if args.dry_run:
        SEEDS = [101]
        N_EPISODES = 3

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS, "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE, "anchor_every": ANCHOR_EVERY,
        "margin": MARGIN, "discr_delta_margin": DISCR_DELTA_MARGIN,
        "discr_floor": DISCR_FLOOR,
        "off_discr_ceil": OFF_DISCR_CEIL, "guard_floor": GUARD_FLOOR,
        "min_pool": MIN_POOL,
        "goal_match_floor": GOAL_MATCH_FLOOR, "outshine_pivot": OUTSHINE_PIVOT,
        "cone_min_cosine_floor": CONE_MIN_COSINE_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    if "non_degenerate" in result:
        manifest["non_degenerate"] = result["non_degenerate"]
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"ON discr={m['on_discrimination_frac_mean']:.3f} guard={m['on_guard_efficacy_mean']:.3f} "
          f"margin={m['on_margin_mean']:.3f} | "
          f"OFF discr={m['off_discrimination_frac_mean']:.3f} guard={m['off_guard_efficacy_mean']:.3f} "
          f"margin={m['off_margin_mean']:.3f}", flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} "
          f"cone_worst={m['zgoal_cone_min_cosine_worst']:.4f} ({m['zgoal_cone_worst_cell']})",
          flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
