#!/opt/local/bin/python3
"""
V3-EXQ-731 -- Q-080.b: does a least-effort PRIOR break a tied-benefit choice?

Sub-question (Q-080.b)
----------------------
Under a TIED benefit (both corridors reach the same resource with identical
cumulative benefit), does a FIXED least-effort PRIOR default the agent to the
LOW-effort corridor from step 0 -- where a value-subtraction-only arm, whose
per-action effort estimate starts uninformative and only learns online, leaves
the tie unbroken early? And is the value-subtraction term itself NON-VACUOUS
(a benefit-asymmetry control where the value term SHOULD prefer HIGH)?

Design (selection modelled at the EXPERIMENT level)
---------------------------------------------------
The two corridors are EQUIDISTANT, so dACC's length-based candidate_effort is
identical for both -- only the ENV per-step energy cost differs. We therefore
model the effort-sensitive selection as an additive bias over the agent's own
per-first-action score (agent.e3.last_scores; lower is better in REE):

  base[a]   = min score over candidates whose first action == a  (np.inf if none)
  bias_value[a] = VALUE_W * ec_hat[a]      # value-subtraction, BOTH arms, LEARNED
                                           # ec_hat = online EMA of realised
                                           # effort_cost_this_step (init 0 =
                                           # uninformative -> tie unbroken early)
  bias_prior[a] = PRIOR_W * eff[a]         # least-effort PRIOR, PRIOR_ON only,
                                           # FIXED from step 0 (eff =
                                           # env effort_cost_by_action)
  final[a]  = base[a] + bias_value[a] + (bias_prior[a] if arm==PRIOR_ON else 0)
  chosen    = argmin_a final[a] over finite base[a]

We STILL call agent.select_action first (so agent state advances and
e3.last_scores populates), then OVERRIDE the executed action with our
effort-biased one-hot. The agent starts at the bottom junction (NO teleport).

Arms
----
  PRIOR_OFF   : value-subtraction term only,  asym=0.0 (TIED probe)
  PRIOR_ON    : value term + least-effort prior, asym=0.0 (TIED probe)
  CONTROL_ASYM: value term only, asym=0.5 (HIGH corridor pays MORE benefit ->
                the value term SHOULD prefer HIGH; proves it is non-vacuous)

Evidence directions (BOTH stated per repo policy)
-------------------------------------------------
SUPPORTS a least-effort prior (Q-080.b): PRIOR_ON enters the LOW corridor early
  (before ec_hat has learned) markedly more than PRIOR_OFF -- the fixed prior
  breaks the tie the value-only arm cannot yet break -- AND CONTROL_ASYM shows
  the value term correctly prefers HIGH when benefit justifies it.
REFUTES / weakens: PRIOR_OFF already low-defaults early (the value-subtraction
  term alone suffices), so a separate fixed prior is redundant.

Lit anchors: Selinger 2015 (humans continuously optimise for metabolic
least-effort, re-learning within minutes); Charnov 1976 Marginal Value Theorem
(effort/cost trades against benefit in foraging). Env spec:
REE_assembly/docs/architecture/effort_dissociation_env.md.

experiment_purpose: "diagnostic".
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_731_q080b_least_effort_prior"
CLAIM_IDS = ["Q-080"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
EPISODES = 6

GRID_SIZE = 11
VALUE_W = 5.0          # value-subtraction weight over the learned ec_hat EMA
PRIOR_W = 5.0          # least-effort prior weight over env effort_cost_by_action
EC_EMA_ALPHA = 0.05    # ec_hat online EMA rate
ASYM_CONTROL = 0.5     # benefit asymmetry for the non-vacuity control

# Windows (episode indices).
EARLY_EPS = (0, 1)
LATE_EPS = (4, 5)

# Criterion thresholds.
C1_EARLY_LOW_MARGIN = 0.25   # PRIOR_ON high_entry_frac_early < PRIOR_OFF by this
C1_SEED_MAJORITY = 2
OFF_LOW_DEFAULT_CEIL = 0.34  # PRIOR_OFF already low-defaults if <= this (mean)

# Readiness (non-degeneracy) floors.
READY_BENEFIT_TIED = 1e-3        # |benefit_high - benefit_low| must be below this
READY_COST_DIFFERENTIAL = 0.005  # mean(high effort cost) - mean(low) must exceed


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, harm_hist


def _make_env(seed: int, asym: float) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=1,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        energy_decay=0.005,
        effort_dissociation_enabled=True,
        effort_benefit_asymmetry=asym,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        z_harm_a_dim=16,
        use_dacc=True,
    )
    return REEAgent(cfg)


def _base_scores_by_action(candidates, action_dim: int, agent) -> np.ndarray:
    """base[a] = min agent score over candidates whose first action == a.

    Uses agent.e3.last_scores (lower = better). Falls back to a uniform (all
    zero) base when scores are unavailable, so the effort biases still decide.
    """
    base = np.full(action_dim, np.inf, dtype=float)
    scores = getattr(agent.e3, "last_scores", None)
    if scores is not None:
        scores = scores.detach().cpu().flatten()
    n = len(candidates)
    for i in range(n):
        a = int(candidates[i].actions[:, 0, :].argmax().item())
        if a < 0 or a >= action_dim:
            continue
        s = float(scores[i].item()) if (scores is not None and i < scores.numel()) else 0.0
        if s < base[a]:
            base[a] = s
    return base


def _run_cell(seed: int, arm: str, asym: float, prior_on: bool, episodes: int,
              steps_per_ep: int) -> Dict[str, Any]:
    """One (seed x arm) cell. arm in {PRIOR_OFF, PRIOR_ON, CONTROL_ASYM}."""
    print(f"Seed {seed} Condition {arm}", flush=True)

    config_slice = {
        "arm": arm,
        "effort_benefit_asymmetry": asym,
        "prior_on": prior_on,
        "size": GRID_SIZE,
        "num_hazards": 0,
        "num_resources": 1,
        "energy_decay": 0.005,
        "effort_dissociation_enabled": True,
        "use_dacc": True,
        "value_w": VALUE_W,
        "prior_w": PRIOR_W,
        "ec_ema_alpha": EC_EMA_ALPHA,
        "steps_per_ep": steps_per_ep,
        "episodes": episodes,
    }

    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env = _make_env(seed, asym)
        agent = _make_agent(env)
        action_dim = env.action_dim

        # Learned online effort estimate (value-subtraction term). Shared across
        # episodes WITHIN a cell (the agent keeps learning); init uninformative.
        ec_hat = np.zeros(action_dim, dtype=float)

        # first corridor entered per episode: 0 none, 1 low, 2 high.
        first_entry_per_ep: List[int] = []

        for ep in range(episodes):
            agent.reset()
            _obs, obs_dict = env.reset()

            if ep % 2 == 0 or ep == episodes - 1:
                print(f"  [train] seed={seed} arm={arm} ep {ep+1}/{episodes}",
                      flush=True)

            first_entry = 0
            for step in range(steps_per_ep):
                body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
                latent = agent.sense(
                    obs_body=body,
                    obs_world=world,
                    obs_harm=harm,
                    obs_harm_a=harm_a,
                    obs_harm_history=harm_hist,
                )
                ticks = agent.clock.advance()
                world_dim_local = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim_local, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                # Advance agent state + populate e3.last_scores, then override.
                _committed = agent.select_action(candidates, ticks)

                base = _base_scores_by_action(candidates, action_dim, agent)
                eff = obs_dict["effort_cost_by_action"]
                if hasattr(eff, "detach"):
                    eff = eff.detach().cpu().numpy()
                eff = np.asarray(eff, dtype=float).flatten()

                final = base.copy()
                for a in range(action_dim):
                    if not np.isfinite(final[a]):
                        continue
                    final[a] = final[a] + VALUE_W * ec_hat[a]
                    if prior_on:
                        final[a] = final[a] + PRIOR_W * eff[a]

                if np.all(~np.isfinite(final)):
                    chosen = 4  # stay, degenerate fallback
                else:
                    chosen = int(np.nanargmin(np.where(np.isfinite(final), final, np.inf)))

                act = torch.zeros(1, action_dim, dtype=torch.float32)
                act[0, chosen] = 1.0

                _obs, harm_signal, done, info, obs_dict = env.step(act)

                # Update the learned value-subtraction EMA on the realised cost.
                ec_hat[chosen] = (1.0 - EC_EMA_ALPHA) * ec_hat[chosen] \
                    + EC_EMA_ALPHA * float(info["effort_cost_this_step"])

                if first_entry == 0:
                    corr = int(info["effort_corridor"])
                    if corr in (1, 2):
                        first_entry = corr

                if done:
                    break

            first_entry_per_ep.append(first_entry)

        def _high_frac(window):
            idxs = [i for i in window if i < len(first_entry_per_ep)]
            entered = [first_entry_per_ep[i] for i in idxs if first_entry_per_ep[i] in (1, 2)]
            if not entered:
                return 0.0
            return float(sum(1 for c in entered if c == 2) / len(entered))

        high_entry_frac_early = _high_frac(EARLY_EPS)
        high_entry_frac_late = _high_frac(LATE_EPS)

        row = {
            "seed": seed,
            "arm": arm,
            "effort_benefit_asymmetry": asym,
            "prior_on": prior_on,
            "high_entry_frac_early": float(high_entry_frac_early),
            "high_entry_frac_late": float(high_entry_frac_late),
            "low_default_early": float(1.0 - high_entry_frac_early),
            "first_entry_per_ep": first_entry_per_ep,
            "ec_hat_final": [float(x) for x in ec_hat],
        }
        cell.stamp(row)

    verdict = "PASS" if len(first_entry_per_ep) > 0 else "FAIL"
    print(
        f"  [seed={seed} {arm}] high_early={high_entry_frac_early:.3f} "
        f"high_late={high_entry_frac_late:.3f} "
        f"entries={first_entry_per_ep} verdict: {verdict}",
        flush=True,
    )
    return row


def _corridor_benefit_probe(steps: int = 6) -> Dict[str, float]:
    """Cumulative benefit (positive harm_signal) along each corridor at asym=0,
    plus the per-step energy-cost differential. Mirrors the effort-env smoke.
    """
    def walk(col_kind: str, asym: float) -> (float, List[float]):
        env = _make_env(seed=1, asym=asym)
        env.reset()
        col = env._effort_low_col if col_kind == "low" else env._effort_high_col
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["empty"]
        env.agent_x, env.agent_y = 2, col
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["agent"]
        env._reset_effort_state()
        tot = 0.0
        costs: List[float] = []
        for _ in range(steps):
            _obs, hs, _done, info, _od = env.step(torch.tensor(1))
            tot += float(hs)
            costs.append(float(info["effort_cost_this_step"]))
        return tot, costs

    b_low, low_costs = walk("low", 0.0)
    b_high, high_costs = walk("high", 0.0)
    mean_high = float(np.mean(high_costs)) if high_costs else 0.0
    mean_low = float(np.mean(low_costs)) if low_costs else 0.0
    return {
        "benefit_low": b_low,
        "benefit_high": b_high,
        "benefit_tied_abs_diff": abs(b_high - b_low),
        "energy_cost_differential": mean_high - mean_low,
    }


def run_experiment(dry_run: bool) -> Dict[str, Any]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    seeds = [42] if dry_run else SEEDS
    episodes = 2 if dry_run else EPISODES
    steps_per_ep = 20 if dry_run else STEPS_PER_EP
    # arm: (name, asym, prior_on)
    if dry_run:
        arms = [("PRIOR_ON", 0.0, True)]
    else:
        arms = [("PRIOR_OFF", 0.0, False),
                ("PRIOR_ON", 0.0, True),
                ("CONTROL_ASYM", ASYM_CONTROL, False)]

    # -- P0 readiness (non-degeneracy) --------------------------------------
    probe = _corridor_benefit_probe(steps=(3 if dry_run else 6))
    checks = [
        {"name": "benefit_tied",
         "measured": probe["benefit_tied_abs_diff"],
         "threshold": READY_BENEFIT_TIED, "direction": "upper"},
        {"name": "energy_cost_differential",
         "measured": probe["energy_cost_differential"],
         "threshold": READY_COST_DIFFERENTIAL, "direction": "lower"},
    ]
    ready = True
    try:
        preconditions = p0_readiness_gate(checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    if not ready and not dry_run:
        return {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": ts,
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": preconditions,
                "criteria_non_degenerate": {
                    "C1_arms_differ": False,
                    "benefit_tied_at_probe": probe["benefit_tied_abs_diff"] < READY_BENEFIT_TIED,
                },
            },
            "criteria": [
                {"name": "C1_least_effort_prior_warranted",
                 "load_bearing": True, "passed": False},
                {"name": "C2_value_term_non_vacuous", "load_bearing": False,
                 "passed": False},
            ],
            "readiness_probe": probe,
            "arm_results": [],
        }

    # -- Arms ---------------------------------------------------------------
    arm_results = []
    for arm, asym, prior_on in arms:
        for seed in seeds:
            row = _run_cell(seed, arm, asym, prior_on, episodes, steps_per_ep)
            arm_results.append(row)

    if dry_run:
        print("Smoke test PASSED", flush=True)
        return {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": ts,
            "outcome": "FAIL",
            "evidence_direction": "unknown",
            "interpretation": {
                "label": "inconclusive",
                "preconditions": preconditions,
                "criteria_non_degenerate": {
                    "C1_arms_differ": False,
                    "benefit_tied_at_probe": probe["benefit_tied_abs_diff"] < READY_BENEFIT_TIED,
                },
            },
            "criteria": [
                {"name": "C1_least_effort_prior_warranted",
                 "load_bearing": True, "passed": False},
                {"name": "C2_value_term_non_vacuous", "load_bearing": False,
                 "passed": False},
            ],
            "readiness_probe": probe,
            "arm_results": arm_results,
            "dry_run": True,
        }

    # -- Adjudication (real run) --------------------------------------------
    off = {r["seed"]: r for r in arm_results if r["arm"] == "PRIOR_OFF"}
    on = {r["seed"]: r for r in arm_results if r["arm"] == "PRIOR_ON"}
    ctrl = {r["seed"]: r for r in arm_results if r["arm"] == "CONTROL_ASYM"}

    # C1: PRIOR_ON high_entry_frac_early < PRIOR_OFF by >=0.25, seed majority.
    c1_seed_hits = 0
    for s in seeds:
        o, n = off.get(s), on.get(s)
        if o is None or n is None:
            continue
        if (o["high_entry_frac_early"] - n["high_entry_frac_early"]) >= C1_EARLY_LOW_MARGIN:
            c1_seed_hits += 1
    c1 = c1_seed_hits >= C1_SEED_MAJORITY

    # C2: CONTROL_ASYM high_entry_frac > tied PRIOR_OFF high_entry_frac (value
    #     correctly prefers HIGH when benefit justifies it), seed majority.
    c2_seed_hits = 0
    for s in seeds:
        o, c = off.get(s), ctrl.get(s)
        if o is None or c is None:
            continue
        # compare pooled (early+late) high-entry fraction
        c_frac = 0.5 * (c["high_entry_frac_early"] + c["high_entry_frac_late"])
        o_frac = 0.5 * (o["high_entry_frac_early"] + o["high_entry_frac_late"])
        if c_frac > o_frac:
            c2_seed_hits += 1
    c2 = c2_seed_hits >= C1_SEED_MAJORITY

    mean_off_early = float(np.mean([off[s]["high_entry_frac_early"] for s in seeds if s in off])) if off else 0.0
    mean_on_early = float(np.mean([on[s]["high_entry_frac_early"] for s in seeds if s in on])) if on else 0.0
    off_already_low_defaults = mean_off_early <= OFF_LOW_DEFAULT_CEIL

    # Navigation non-degeneracy (CRITICAL). high_entry_frac only counts episodes
    # that entered SOME corridor, so an agent that never reaches a corridor gives
    # high_entry_frac_early == 0 -- indistinguishable from "always entered LOW".
    # Without this guard a non-navigating substrate would spuriously self-route
    # "value_subtraction_term_suffices" (a vacuous PASS). Require an adequate
    # corridor-entry rate in the early window, else substrate_not_ready_requeue.
    def _entry_rate(cell_row, window):
        ep = cell_row["first_entry_per_ep"]
        w = [ep[i] for i in window if i < len(ep)]
        return (sum(1 for c in w if c in (1, 2)) / len(w)) if w else 0.0
    off_entry_rate = float(np.mean([_entry_rate(off[s], EARLY_EPS) for s in seeds if s in off])) if off else 0.0
    on_entry_rate = float(np.mean([_entry_rate(on[s], EARLY_EPS) for s in seeds if s in on])) if on else 0.0
    substrate_can_navigate = (off_entry_rate >= 0.5) or (on_entry_rate >= 0.5)

    # Non-degeneracy.
    c1_arms_differ = abs(mean_off_early - mean_on_early) > 1e-9
    benefit_tied_at_probe = probe["benefit_tied_abs_diff"] < READY_BENEFIT_TIED

    if not substrate_can_navigate:
        # Agent cannot reliably reach either corridor -> the corridor-choice
        # measurement is undefined; requeue at a navigation-competent substrate.
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        outcome = "FAIL"
    elif c1:
        label = "least_effort_prior_warranted"
        evidence_direction = "supports"
        outcome = "PASS"
    elif off_already_low_defaults:
        label = "value_subtraction_term_suffices_prior_redundant"
        evidence_direction = "weakens"
        outcome = "PASS"
    else:
        label = "inconclusive"
        evidence_direction = "unknown"
        outcome = "FAIL"

    print(
        f"\nFINAL: {outcome}  label={label} "
        f"C1={c1} (hits={c1_seed_hits}) C2={c2} (hits={c2_seed_hits}) "
        f"off_early={mean_off_early:.3f} on_early={mean_on_early:.3f}",
        flush=True,
    )

    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"Q-080": evidence_direction},
        "interpretation": {
            "label": label,
            "preconditions": preconditions + [
                {"name": "agent_reaches_corridor_early",
                 "description": "fraction of early-window episodes entering a corridor",
                 "measured": max(off_entry_rate, on_entry_rate),
                 "threshold": 0.5, "direction": "lower",
                 "met": bool(substrate_can_navigate), "kind": "readiness",
                 "control": "PRIOR_OFF/PRIOR_ON early episodes"},
            ],
            "criteria_non_degenerate": {
                "C1_arms_differ": bool(c1_arms_differ),
                "benefit_tied_at_probe": bool(benefit_tied_at_probe),
                "agent_navigates": bool(substrate_can_navigate),
            },
        },
        "criteria": [
            {"name": "C1_least_effort_prior_warranted", "load_bearing": True,
             "passed": bool(c1), "seed_hits": c1_seed_hits,
             "desc": "PRIOR_ON high_entry_frac_early < PRIOR_OFF by >=0.25 in "
                     ">=2/3 seeds (prior breaks the tie the value-only arm cannot)"},
            {"name": "C2_value_term_non_vacuous", "load_bearing": False,
             "passed": bool(c2), "seed_hits": c2_seed_hits,
             "desc": "CONTROL_ASYM high-entry fraction > tied PRIOR_OFF in "
                     ">=2/3 seeds (value term prefers HIGH when benefit justifies)"},
        ],
        "metrics": {
            "mean_prior_off_high_entry_early": mean_off_early,
            "mean_prior_on_high_entry_early": mean_on_early,
            "off_already_low_defaults": bool(off_already_low_defaults),
            "off_corridor_entry_rate_early": off_entry_rate,
            "on_corridor_entry_rate_early": on_entry_rate,
            "substrate_can_navigate": bool(substrate_can_navigate),
        },
        "readiness_probe": probe,
        "config": {
            "seeds": SEEDS, "episodes": EPISODES, "steps_per_ep": STEPS_PER_EP,
            "grid_size": GRID_SIZE, "value_w": VALUE_W, "prior_w": PRIOR_W,
            "ec_ema_alpha": EC_EMA_ALPHA, "asym_control": ASYM_CONTROL,
            "c1_early_low_margin": C1_EARLY_LOW_MARGIN,
        },
        "arm_results": arm_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== {EXPERIMENT_TYPE} ===", flush=True)
    print(f"Claim: {CLAIM_IDS}  purpose: {EXPERIMENT_PURPOSE}", flush=True)
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL RUN'}", flush=True)

    result = run_experiment(dry_run=args.dry_run)

    script_dir = Path(__file__).resolve().parents[1]
    out_dir = (script_dir.parent / "REE_assembly" / "evidence"
               / "experiments" / EXPERIMENT_TYPE)
    out_file = write_flat_manifest(
        result,
        out_dir,
        dry_run=False,
        config=result.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Output written to: {out_file}", flush=True)
    print(f"Outcome: {result['outcome']}  "
          f"evidence_direction: {result['evidence_direction']}", flush=True)

    return out_file, result["outcome"], args.dry_run


if __name__ == "__main__":
    _out_file, _outcome, _dry = main()
    # Reached on every manifest-writing path (readiness-abort, dry-run, and full
    # adjudication all return through main()).
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_file,
        dry_run=_dry,
    )
