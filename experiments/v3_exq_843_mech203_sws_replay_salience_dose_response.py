"""
V3-EXQ-843: MECH-203 serotonergic replay-salience tagging -- SWS replay
start-point selection as a graded function of tonic 5-HT (dose-response).

SLEEP DRIVER: manual-cycle-loop (replay start-point selection probed directly
via HippocampalModule._select_valence_weighted_start over a frozen post-waking
valence field; SWS phase entered via agent.enter_sws_mode()).

Claim: MECH-203 (serotonergic_replay_salience_tagging)
Design docs:
  REE_assembly/docs/architecture/sleep/serotonergic_cross_state_substrate.md
  REE_assembly/evidence/planning/sleep_substrate_plan.md

MECHANISM UNDER TEST
--------------------
During SWS the agent selects replay start points by valence priority
(ree_core/hippocampal/module.py::_select_valence_weighted_start ->
ResidueField.get_valence_priority), scoring each buffered z_world state by
    priority(z) = dot( evaluate_valence(z), drive_state )
where the serotonergic system supplies (ree_core/agent.py::_do_replay, the
serotonin-enabled branch, lines ~8596-8604):
    drive_state = [ t5ht, 0.5, 1.0 - t5ht, surprise_weight ]   # surprise_weight=0.3
and evaluate_valence(z) = [wanting, liking, harm_discriminative, surprise].

So tonic 5-HT (t5ht) trades off the WANTING (benefit) weight against the HARM
weight in exactly the balance MECH-203 predicts: high tonic 5-HT tags
benefit-salient experiences for preferential replay; low tonic 5-HT (the
depressive attractor, INV-053) shifts replay toward harm-salient experiences,
under-tagging benefit. This experiment measures whether the SWS replay
start-point selection is graded by t5ht as claimed.

WHY THIS IS NOT THE PRIOR (INVALID) TEST
----------------------------------------
v3_exq_255 / v3_exq_256 (the only prior MECH-203 runs) are BOTH
non_contributory / INVALID: an env.step() return-order bug
(`obs,reward,done,truncated,info = env.step()` against the real 5-tuple
`flat_obs, harm_signal, done, info, obs_dict`) broke every episode after one
step, so n_benefit_samples=0 and all metrics were 0. MECH-203 has therefore
NEVER had a valid test; sleep_substrate_plan.md's "EXQ-255/256 PASS; adequate"
note is stale. This is a redesign (new EXQ NUMBER), not a bug-fix letter: it
tests the REPLAY-SELECTION half of MECH-203 (the drive_state -> priority path),
not the static VALENCE_WANTING-field tagging 255 measured.

DESIGN (commitment-free, measurement-only)
------------------------------------------
Per seed:
  1. WAKING ROLLOUT (serotonin enabled) through a benefit+harm grid world so the
     ResidueField accumulates VALENCE_WANTING at benefit-contact z_worlds and
     VALENCE_HARM_DISCRIMINATIVE at harm-contact z_worlds, and a pool of visited
     z_world states is recorded. The valence field is FROZEN after the rollout.
  2. Enter SWS (agent.enter_sws_mode()).
  3. DOSE SWEEP over tonic 5-HT in {0.1, 0.3, 0.5, 0.7, 0.9}. For each dose the
     drive_state is built by the EXACT substrate formula above (read through the
     real SerotoninModule, whose tonic is held during SWS). The FIELD IS NOT
     REWRITTEN -- only the drive weights change, isolating the selection effect
     from any experience-density confound. For each dose we draw R replay windows
     (contiguous slices of the visited pool, the theta-buffer analogue of a
     replay opportunity) and call the REAL _select_valence_weighted_start; the
     selected start's valence is classified benefit-dominant (wanting > harm) or
     harm-dominant.

DV: benefit_replay_fraction(dose) over DISCRIMINATING windows (windows that
contain BOTH a benefit-dominant and a harm-dominant candidate, so the selection
CAN shift). A non-discriminating window is dose-invariant by construction and is
excluded from the load-bearing DV.

DV-SYMMETRY (604c check): the manipulation (t5ht changing the drive-weight
VECTOR [t5ht,0.5,1-t5ht,surprise]) is a non-uniform reweighting, NOT a broadcast
additive constant, monotone rescaling, or permutation of candidates. The argmax
selection DV is therefore NOT invariant under it -- changing t5ht genuinely
re-ranks candidates. The measured dose-response is an empirical property of the
substrate's valence field, not an arithmetic identity.

PRE-REGISTERED ACCEPTANCE (evidence)
------------------------------------
Admissibility (C3, non-degeneracy gate -- if it fails the run is
non_degenerate=False / scoring_excluded, NOT a FAIL):
  - active RBF centers > 0 after the rollout, AND the visited pool contains BOTH
    >= 1 benefit-dominant and >= 1 harm-dominant state (else no shift is possible)
  - mean discriminating-window fraction across seeds >= MIN_DISCRIM (0.10)
  - benefit_replay_fraction takes >= 3 distinct values across the 5-dose grid
    (not saturated at 0 or 1)

Given C3 admissible:
  C1 (benefit dose-response): per seed, Spearman rho(dose, benefit_fraction)
     >= RHO_MIN (0.9) AND margin = benefit_fraction(0.9) - benefit_fraction(0.1)
     >= MARGIN (0.15); sign-consistent (all seeds positive).
  C2 (harm complement): per seed, Spearman rho(dose, harm_fraction) <= -RHO_MIN.

PASS iff C1 and C2 hold for all seeds (given C3). A FAIL here means the wired
MECH-203 mechanism does NOT actually grade replay selection by tonic 5-HT
(drive_state not consumed / selection insensitive) -- a real refutation of the
replay-salience prediction, not a substrate-readiness artefact.
"""

import sys
import time
import random
import argparse
import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.residue.field import (
    VALENCE_WANTING,
    VALENCE_HARM_DISCRIMINATIVE,
)

from experiments.pack_writer import write_flat_manifest
from experiments._metrics import check_degeneracy
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_843_mech203_sws_replay_salience_dose_response"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-203"]

# -- Pre-registered constants --
DOSE_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]     # tonic 5-HT levels swept at replay time
SURPRISE_WEIGHT = 0.3                       # substrate default (serotonin path, surprise_gated off)
RHO_MIN = 0.9                               # Spearman monotonicity floor
MARGIN = 0.15                               # benefit_fraction(hi) - benefit_fraction(lo) floor
MIN_DISCRIM = 0.10                          # mean discriminating-window fraction floor (admissibility)
WINDOW_LEN = 12                             # replay window length (theta-buffer analogue)
N_WINDOWS = 200                             # replay windows per dose

# -- Architecture dims (standard V3 CausalGridWorldV2 proxy mode) --
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

# -- Rollout params --
ROLLOUT_EPISODES = 40
STEPS_PER_EPISODE = 120
SEEDS = [42, 137, 2026]

_ZG = ZGoalStreamAccumulator()


def make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,               # z_world fidelity (SD-008) needed for valence localisation
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        goal_weight=1.0,
        tonic_5ht_enabled=True,        # MECH-203 substrate on
    )


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation (no scipy dep). Returns 0.0 for degenerate input."""
    n = len(x)
    if n < 2:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n))
    dy = sum((ry[i] - my) ** 2 for i in range(n))
    if dx <= 0 or dy <= 0:
        return 0.0
    return num / (dx * dy) ** 0.5


def _run_waking_rollout(agent, env, n_episodes, n_steps, seed):
    """Step the agent through the env; return the pool of visited z_world states.

    Populates the ResidueField with VALENCE_WANTING (via update_benefit_salience)
    at benefit-contact states and VALENCE_HARM_DISCRIMINATIVE (via update_residue)
    at harm-contact states.
    """
    visited_z: List[torch.Tensor] = []
    for ep in range(n_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [train] rollout seed={seed} ep {ep+1}/{n_episodes}", flush=True)
        obs, info = env.reset()
        agent.reset()
        body_obs = torch.tensor(obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
        world_obs = torch.tensor(
            obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM], dtype=torch.float32
        ).unsqueeze(0)

        for _step in range(n_steps):
            latent = agent.sense(body_obs, world_obs)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            benefit_exposure = float(body_obs[0, 11]) if body_obs.shape[-1] > 11 else 0.0
            drive_level = agent.compute_drive_level(body_obs)

            # MECH-203 SR-1/SR-2: tonic 5-HT step + benefit-salience (VALENCE_WANTING) write
            agent.serotonin_step(benefit_exposure)
            agent.update_z_goal(benefit_exposure, drive_level)
            agent.update_benefit_salience(benefit_exposure)

            # Record the visited z_world (frozen post-rollout replay pool)
            if agent._current_latent is not None:
                visited_z.append(agent._current_latent.z_world.detach().clone())

            # Environment step -- CORRECT 5-tuple (the 255/256 bug was here)
            action_idx = int(action.argmax(dim=-1).item())
            flat_obs, harm_signal, done, info_next, obs_dict = env.step(action_idx)

            # Harm valence write (VALENCE_HARM_DISCRIMINATIVE) + center allocation
            if harm_signal is not None and harm_signal < 0:
                agent.update_residue(float(harm_signal))

            body_obs = torch.tensor(flat_obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
            world_obs = torch.tensor(
                flat_obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM], dtype=torch.float32
            ).unsqueeze(0)

            if done:
                break
    return visited_z


def _classify_pool(agent, visited_z):
    """Return per-state (wanting, harm) valence and dominance labels for the pool."""
    labels = []  # 'benefit' | 'harm' | 'neutral'
    wanting_vals, harm_vals = [], []
    for z in visited_z:
        val = agent.residue_field.evaluate_valence(z)
        w = float(val[0, VALENCE_WANTING].item())
        h = float(val[0, VALENCE_HARM_DISCRIMINATIVE].item())
        wanting_vals.append(w)
        harm_vals.append(h)
        if w > h and w > 0:
            labels.append("benefit")
        elif h > w and h > 0:
            labels.append("harm")
        else:
            labels.append("neutral")
    return labels, wanting_vals, harm_vals


def _drive_state_for(agent, dose):
    """Build the SWS drive_state exactly as agent._do_replay does (serotonin branch).

    Reads t5ht through the real SerotoninModule (tonic held during SWS) after
    setting it to the swept dose, mirroring ree_core/agent.py::_do_replay.
    """
    agent.serotonin._tonic_5ht = float(dose)
    t5ht = agent.serotonin.tonic_5ht if agent.serotonin.enabled else 0.0
    return torch.tensor([t5ht, 0.5, 1.0 - t5ht, SURPRISE_WEIGHT]), t5ht


def _sweep_doses(agent, visited_z, labels, rng):
    """For each dose, measure benefit/harm replay-selection fractions over windows.

    Uses the REAL substrate selection path
    (HippocampalModule._select_valence_weighted_start -> get_valence_priority).
    """
    n = len(visited_z)
    per_dose = []
    for dose in DOSE_GRID:
        drive_state, t5ht = _drive_state_for(agent, dose)
        n_benefit_sel = 0
        n_harm_sel = 0
        n_discrim = 0
        n_total = 0
        for _ in range(N_WINDOWS):
            start = int(rng.integers(0, max(1, n - WINDOW_LEN)))
            idxs = list(range(start, min(n, start + WINDOW_LEN)))
            if len(idxs) < 2:
                continue
            window_labels = [labels[i] for i in idxs]
            has_benefit = any(l == "benefit" for l in window_labels)
            has_harm = any(l == "harm" for l in window_labels)
            n_total += 1
            if not (has_benefit and has_harm):
                continue  # non-discriminating: dose cannot change the outcome
            n_discrim += 1
            window = torch.stack([visited_z[i] for i in idxs], dim=0)  # [W, 1, world_dim]
            selected = agent.hippocampal._select_valence_weighted_start(window, drive_state)
            sval = agent.residue_field.evaluate_valence(selected)
            sw = float(sval[0, VALENCE_WANTING].item())
            sh = float(sval[0, VALENCE_HARM_DISCRIMINATIVE].item())
            if sw > sh:
                n_benefit_sel += 1
            elif sh > sw:
                n_harm_sel += 1
        benefit_frac = n_benefit_sel / n_discrim if n_discrim else 0.0
        harm_frac = n_harm_sel / n_discrim if n_discrim else 0.0
        discrim_frac = n_discrim / n_total if n_total else 0.0
        per_dose.append({
            "dose_t5ht": dose,
            "benefit_replay_fraction": benefit_frac,
            "harm_replay_fraction": harm_frac,
            "n_discriminating_windows": n_discrim,
            "n_total_windows": n_total,
            "discriminating_window_fraction": discrim_frac,
        })
    return per_dose


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    print(f"Seed {seed} Condition sws_replay_dose_response")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cfg = make_config()
    agent = REEAgent(cfg)
    agent.to(torch.device("cpu"))
    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.05,
        resource_benefit=0.05,
        use_proxy_fields=True,
    )

    n_episodes = 2 if dry_run else ROLLOUT_EPISODES
    n_steps = 12 if dry_run else STEPS_PER_EPISODE

    visited_z = _run_waking_rollout(agent, env, n_episodes, n_steps, seed)
    _ZG.observe(agent)   # record z_goal stream liveness (benefit pathway)

    # Enter SWS (5-HT held at waking level; MECH-203 replay salience lives here)
    agent.enter_sws_mode()

    labels, wanting_vals, harm_vals = _classify_pool(agent, visited_z)
    n_benefit = labels.count("benefit")
    n_harm = labels.count("harm")
    active_centers = int(agent.residue_field.rbf_field.active_mask.sum().item())

    pool_admissible = active_centers > 0 and n_benefit >= 1 and n_harm >= 1 and len(visited_z) >= WINDOW_LEN

    if pool_admissible:
        per_dose = _sweep_doses(agent, visited_z, labels, rng)
    else:
        per_dose = []

    benefit_series = [d["benefit_replay_fraction"] for d in per_dose]
    harm_series = [d["harm_replay_fraction"] for d in per_dose]
    rho_benefit = _spearman(DOSE_GRID, benefit_series) if per_dose else 0.0
    rho_harm = _spearman(DOSE_GRID, harm_series) if per_dose else 0.0
    margin = (benefit_series[-1] - benefit_series[0]) if per_dose else 0.0
    mean_discrim = float(np.mean([d["discriminating_window_fraction"] for d in per_dose])) if per_dose else 0.0
    distinct_benefit = len({round(b, 6) for b in benefit_series})

    c1 = pool_admissible and rho_benefit >= RHO_MIN and margin >= MARGIN
    c2 = pool_admissible and rho_harm <= -RHO_MIN
    seed_pass = bool(c1 and c2)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}")

    return {
        "seed": seed,
        "pool_admissible": pool_admissible,
        "active_centers": active_centers,
        "n_visited": len(visited_z),
        "n_benefit_states": n_benefit,
        "n_harm_states": n_harm,
        "per_dose": per_dose,
        "benefit_series": benefit_series,
        "harm_series": harm_series,
        "rho_benefit": rho_benefit,
        "rho_harm": rho_harm,
        "margin": margin,
        "mean_discriminating_window_fraction": mean_discrim,
        "distinct_benefit_values": distinct_benefit,
        "c1_benefit_dose_response": bool(c1),
        "c2_harm_complement": bool(c2),
        "seed_pass": seed_pass,
    }


def _run(dry_run: bool):
    t0 = time.perf_counter()

    print(f"{EXPERIMENT_TYPE}  dry_run={dry_run}")
    per_seed = [run_seed(seed, dry_run=dry_run) for seed in SEEDS]

    admissible_seeds = [r for r in per_seed if r["pool_admissible"]]

    # C3 admissibility / non-degeneracy gate
    mean_discrim_all = (
        float(np.mean([r["mean_discriminating_window_fraction"] for r in admissible_seeds]))
        if admissible_seeds else 0.0
    )
    max_distinct = max((r["distinct_benefit_values"] for r in admissible_seeds), default=0)
    c3_admissible = (
        len(admissible_seeds) == len(SEEDS)
        and mean_discrim_all >= MIN_DISCRIM
        and max_distinct >= 3
    )

    # Non-degeneracy self-report (evidence-run scoring net): the load-bearing DV is
    # per-seed benefit_replay_fraction spread across doses. Degenerate if EVERY seed's
    # series is pinned (no spread).
    degeneracy = check_degeneracy({
        "benefit_replay_fraction": {"groups": [r["benefit_series"] for r in per_seed if r["benefit_series"]]},
    })
    if not c3_admissible:
        degeneracy["non_degenerate"] = False
        reason = []
        if len(admissible_seeds) != len(SEEDS):
            reason.append("pool inadmissible on >=1 seed (missing benefit/harm states or centers)")
        if mean_discrim_all < MIN_DISCRIM:
            reason.append(f"mean discriminating-window fraction {mean_discrim_all:.3f} < {MIN_DISCRIM}")
        if max_distinct < 3:
            reason.append(f"benefit fraction saturated (<3 distinct values across doses)")
        degeneracy["degeneracy_reason"] = "; ".join(reason) or degeneracy.get("degeneracy_reason", "")

    all_c1 = all(r["c1_benefit_dose_response"] for r in per_seed)
    all_c2 = all(r["c2_harm_complement"] for r in per_seed)
    sign_consistent = all(r["margin"] > 0 for r in admissible_seeds) if admissible_seeds else False

    non_degenerate = degeneracy.get("non_degenerate", True)
    if not c3_admissible or not non_degenerate:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        interp = "inadmissible: valence pool degenerate or non-discriminating; MECH-203 selection not scoreable"
    elif all_c1 and all_c2 and sign_consistent:
        outcome = "PASS"
        evidence_direction = "supports"
        interp = "SWS replay start-point selection is graded by tonic 5-HT as MECH-203 predicts (benefit up, harm down, monotone, sign-consistent)"
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        interp = "wired MECH-203 mechanism does not grade replay selection by tonic 5-HT (selection insensitive to drive_state)"

    print(f"\n  C3 admissible={c3_admissible} (mean_discrim={mean_discrim_all:.3f}, max_distinct={max_distinct})")
    print(f"  C1 benefit dose-response (all seeds)={all_c1}")
    print(f"  C2 harm complement (all seeds)={all_c2}")
    print(f"  sign_consistent={sign_consistent}  non_degenerate={non_degenerate}")
    print(f"  VERDICT: {outcome}  direction={evidence_direction}")

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "dose_grid": DOSE_GRID,
        "surprise_weight": SURPRISE_WEIGHT,
        "rho_min": RHO_MIN,
        "margin": MARGIN,
        "min_discrim": MIN_DISCRIM,
        "window_len": WINDOW_LEN,
        "n_windows": N_WINDOWS,
        "rollout_episodes": ROLLOUT_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "seeds": SEEDS,
        "env": {"size": 10, "num_hazards": 3, "num_resources": 3,
                "hazard_harm": 0.05, "resource_benefit": 0.05, "use_proxy_fields": True},
        "alpha_world": 0.9,
        "tonic_5ht_enabled": True,
    }

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "mechanism_validation",
        "timestamp_utc": ts,
        "sleep_driver_pattern": "manual-cycle-loop (replay selection probed directly over frozen post-waking valence field)",
        "interpretation_note": interp,
        "acceptance": {
            "c3_admissible": c3_admissible,
            "c1_benefit_dose_response_all_seeds": all_c1,
            "c2_harm_complement_all_seeds": all_c2,
            "sign_consistent": sign_consistent,
            "mean_discriminating_window_fraction": mean_discrim_all,
            "max_distinct_benefit_values": max_distinct,
        },
        "per_seed_results": per_seed,
        "dv_symmetry_note": (
            "manipulation = drive-weight vector [t5ht,0.5,1-t5ht,surprise], a non-uniform "
            "reweighting; the argmax start-selection DV is NOT invariant under it (not a "
            "broadcast constant / monotone rescale / permutation), so the dose-response is "
            "a measured substrate property, not an arithmetic identity"
        ),
    }
    manifest.update(degeneracy)

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"  manifest -> {out_path}")

    return outcome, out_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path, _run_id = _run(args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        dry_run=args.dry_run,
    )
